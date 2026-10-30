"""获批 tokenizer 的独立准备、固定来源校验与离线缓存复制；不包含模型权重。"""

import argparse
import hashlib
import json
import os
import re
import stat
import urllib.parse
import urllib.request
from pathlib import Path

LOCK = Path(".github/ppu-functional-tokenizers.json")
CACHE_RELATIVE = Path("tokenizers/huggingface")
APPROVED_DESTINATION = Path(
    "/wl_nas/anyunge.ayg/megatron-functional-ci/artifacts-base/tokenizers/huggingface"
)
MAX_FILE_BYTES = 20 * 1024 * 1024
REPOSITORIES = {
    "gpt2": "openai-community/gpt2",
    "bert-large-uncased": "google-bert/bert-large-uncased",
    "bert-large-cased": "google-bert/bert-large-cased",
}
COMMON_FILES = {"config.json", "tokenizer_config.json", "tokenizer.json"}
OPTIONAL_FILES = {"special_tokens_map.json", "added_tokens.json"}


def required_files(repository: str) -> set:
    if repository not in REPOSITORIES:
        raise ValueError(f"Unapproved tokenizer repository: {repository}")
    return COMMON_FILES | ({"vocab.json", "merges.txt"} if repository == "gpt2" else {"vocab.txt"})


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def unique_json(content: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate tokenizer metadata key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite tokenizer metadata value: {value}")

    value = json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Tokenizer metadata must be a JSON object")
    return value


def validate_source(repository: str, source: dict) -> None:
    required = required_files(repository)
    if not isinstance(source, dict) or set(source) != {
        "revision",
        "canonical_repository",
        "api_url",
        "files",
    }:
        raise ValueError("Invalid tokenizer source fields")
    revision = source["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Tokenizer revision must be an immutable commit SHA")
    if source["canonical_repository"] != REPOSITORIES[repository]:
        raise ValueError("Tokenizer canonical repository mismatch")
    if source["api_url"] != f"https://huggingface.co/api/models/{repository}/revision/{revision}":
        raise ValueError("Tokenizer revision provenance mismatch")
    files = source["files"]
    if not isinstance(files, dict) or not required <= files.keys() <= required | OPTIONAL_FILES:
        raise ValueError("Tokenizer files must match the approved allowlist")
    for name, row in files.items():
        if not isinstance(row, dict) or set(row) != {"url", "size", "sha256"}:
            raise ValueError("Invalid tokenizer file provenance")
        expected_url = (
            f"https://huggingface.co/{REPOSITORIES[repository]}/resolve/{revision}/{name}"
        )
        if row["url"] != expected_url:
            raise ValueError("Tokenizer file URL is not pinned to its official revision")
        if type(row["size"]) is not int or not 0 < row["size"] <= MAX_FILE_BYTES:
            raise ValueError("Invalid tokenizer file size")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid tokenizer file SHA256")


def read_file(root: Path, relative: str | Path) -> bytes:
    """仅读取预期的小型普通文件，拒绝缓存内的符号链接和路径越界。"""
    relative = Path(relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in (".", "..") for part in relative.parts)
    ):
        raise ValueError("Unsafe tokenizer cache path")
    path = root
    for part in ("", *relative.parts):
        path /= part
        if path.is_symlink():
            raise ValueError("Tokenizer cache symlinks are not allowed")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_FILE_BYTES:
        raise ValueError("Tokenizer cache requires a nonempty bounded regular file")
    with path.open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) != info.st_size:
        raise ValueError("Tokenizer cache file changed or exceeded its size limit")
    return content


def load_sources(control: Path) -> dict:
    path = control / LOCK
    if not path.exists() and not path.is_symlink():
        return {}
    return validate_lock(unique_json(read_file(control, LOCK)))


def validate_lock(lock: dict) -> dict:
    if (
        not isinstance(lock, dict)
        or set(lock) != {"schema_version", "repositories"}
        or type(lock["schema_version"]) is not int
        or lock["schema_version"] != 1
    ):
        raise ValueError("Invalid tokenizer source lock")
    sources = lock["repositories"]
    if not isinstance(sources, dict) or not sources or sources.keys() - REPOSITORIES.keys():
        raise ValueError("Invalid locked tokenizer repositories")
    for repository, source in sources.items():
        validate_source(repository, source)
    return sources


def sources_for(dependencies: list) -> dict:
    sources = {}
    for dependency in dependencies:
        repository = dependency["repository"]
        if dependency.get("status") != "fixed_source" or not dependency.get("source"):
            raise ValueError("Unresolved tokenizer source lock")
        source = dependency["source"]
        validate_source(repository, source)
        if repository in sources and sources[repository] != source:
            raise ValueError("Conflicting tokenizer revisions across training phases")
        sources[repository] = source
    return sources


def cache_files(sources: dict) -> dict:
    """HF 的原仓库别名和 refs/main 均保留，原训练参数无须添加 revision。"""
    files = {}
    for repository, source in sorted(sources.items()):
        validate_source(repository, source)
        base = f"hub/models--{repository}"
        ref = source["revision"].encode()
        files[f"{base}/refs/main"] = {"size": len(ref), "sha256": digest(ref)}
        for name, row in sorted(source["files"].items()):
            files[f"{base}/snapshots/{source['revision']}/{name}"] = {
                "size": row["size"],
                "sha256": row["sha256"],
            }
    return files


def asset_requirements(dependencies: list) -> list:
    sources = sources_for(dependencies)
    purposes = sorted({row["purpose"] for row in dependencies})
    return [
        {
            "relative_path": str(CACHE_RELATIVE / name),
            "kind": "file",
            "purposes": purposes,
            "tokenizer_cache": True,
            "expected_size": row["size"],
            "expected_sha256": row["sha256"],
        }
        for name, row in cache_files(sources).items()
    ]


def checked_bytes(root: Path, name: str | Path, expected: dict) -> bytes:
    content = read_file(root, name)
    if len(content) != expected["size"] or digest(content) != expected["sha256"]:
        raise ValueError(f"Tokenizer cache digest or size mismatch: {name}")
    return content


def verify_cache(root: Path, sources: dict) -> dict:
    files = cache_files(sources)
    for name, row in files.items():
        checked_bytes(root, name, row)
    return {"schema_version": 1, "repositories": sources, "files": files}


def inspect_candidate(data_root: Path) -> dict:
    """只读回传获准路径内的候选清单及缓存一致性，不提升其信任级别。"""
    relative = CACHE_RELATIVE / "source-lock.json"
    report = {
        "checked_path": str(data_root / relative),
        "trusted": False,
        "status": "source_lock_invalid",
        "source_lock": None,
        "source_lock_sha256": None,
        "cache_matches_candidate": False,
        "file_count": 0,
        "reason": "",
    }
    try:
        content = read_file(data_root, relative)
        if len(content) > 128 * 1024:
            raise ValueError("Candidate source lock exceeds its metadata size limit")
        lock = unique_json(content)
        sources = validate_lock(lock)
    except FileNotFoundError:
        report.update(status="source_lock_missing", reason="Candidate source lock is missing")
        return report
    except (OSError, ValueError, RecursionError) as exc:
        report["reason"] = f"Candidate source lock is unreadable or invalid ({type(exc).__name__})"
        return report
    report.update(source_lock=lock, source_lock_sha256=digest(content), status="cache_mismatch")
    try:
        verified = verify_cache(data_root / CACHE_RELATIVE, sources)
        if read_file(data_root, relative) != content:
            raise ValueError("Candidate source lock changed during verification")
    except (OSError, ValueError) as exc:
        report["reason"] = f"Cache does not match a stable candidate ({type(exc).__name__})"
        return report
    report.update(
        status="candidate_matches_cache",
        cache_matches_candidate=True,
        file_count=len(verified["files"]),
    )
    return report


def inspect_receipts(directory: Path) -> dict:
    """仅回传准备凭据的摘要、已知成功输出及退出码，不公开原始日志或提升信任。"""
    report = {"checked_path": str(directory), "status": "unreadable", "files": [], "reason": ""}
    expected = {"repositories": sorted(REPOSITORIES), "numerical_validation": False}
    try:
        if directory.is_symlink() or directory.resolve() != directory.absolute():
            raise ValueError("Preparation receipt directory must not contain symlinks")
        entries = []
        for path in directory.iterdir():
            if len(entries) >= 64:
                raise ValueError("Too many preparation receipt entries")
            entries.append(path)
        for path in sorted(entries):
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", path.name):
                continue
            row = {"name": path.name, "status": "metadata_only"}
            report["files"].append(row)
            info = path.lstat()
            row["size"] = info.st_size
            if not stat.S_ISREG(info.st_mode) or info.st_size > 256 * 1024:
                row["status"] = "not_bounded_regular_file"
                continue
            phase = re.search(r"(?:^|[._-])(prepare|verify)(?:[._-]|$)", path.name)
            if not phase and path.name != "ppu_functional_hf.py":
                continue
            content = read_file(directory, path.name) if info.st_size else b""
            row.update(status="inspected", sha256=digest(content))
            if not phase:
                continue
            row.update(command=phase[1], reported_exit_code=None, expected_summary_count=0)
            is_exit = re.search(r"(?:^|[._-])(?:exit(?:_?code)?|rc|status)(?:[._-]|$)", path.name)
            if is_exit and re.fullmatch(rb"\s*(0|[1-9][0-9]{0,2})\s*", content):
                code = int(content)
                if code <= 255:
                    row["reported_exit_code"] = code
                continue
            unexpected = 0
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    value = unique_json(line)
                except (ValueError, RecursionError):
                    value = None
                if (
                    value == expected
                    and value["numerical_validation"] is False
                    and type(value["numerical_validation"]) is bool
                ):
                    row["expected_summary_count"] += 1
                else:
                    unexpected += 1
            row["unrecognized_line_count"] = unexpected
        report["status"] = "inspected"
    except (OSError, ValueError, RecursionError) as exc:
        report["reason"] = f"Preparation receipts unavailable ({type(exc).__name__})"
    return report


def populate(data_root: Path, destination: Path, dependencies: list) -> dict:
    """逐重复复制并复算；不向输入 NAS 写回，也不携带额外缓存文件。"""
    input_root, output_root = data_root.resolve(), destination.resolve()
    if output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root):
        raise ValueError("Tokenizer output must not overlap the input asset root")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing repeat cache")
    sources = sources_for(dependencies)
    files = cache_files(sources)
    contents = {
        name: checked_bytes(data_root, CACHE_RELATIVE / name, row) for name, row in files.items()
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    return verify_cache(destination, sources)


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    """仅接受官方 HTTPS 重定向；不把授权范围扩展到任意镜像或下载站。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_redirect(req.full_url, newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("Tokenizer downloads require official HTTPS HuggingFace URLs")


def validate_redirect(origin: str, target: str) -> None:
    """重定向只能规范化批准仓库名或进入同一 commit 的官方小文件缓存。"""
    validate_url(origin)
    validate_url(target)
    old, new = urllib.parse.urlsplit(origin).path, urllib.parse.urlsplit(target).path
    for alias, canonical in REPOSITORIES.items():
        for repository in (alias, canonical):
            prefix = f"/api/models/{repository}"
            if old == prefix or re.fullmatch(re.escape(prefix) + r"/revision/[0-9a-f]{40}", old):
                suffix = old[len(prefix) :]
                if new in {f"/api/models/{name}{suffix}" for name in (alias, canonical)}:
                    return
        for name in required_files(alias) | OPTIONAL_FILES:
            paths = (f"/{canonical}/resolve/", f"/api/resolve-cache/models/{canonical}/")
            for prefix in paths:
                match = re.fullmatch(re.escape(prefix) + r"([0-9a-f]{40})/" + re.escape(name), old)
                if match and new in {f"{base}{match[1]}/{name}" for base in paths}:
                    return
    raise ValueError("Tokenizer redirect changed the approved repository, revision or file")


def fetch_official(url: str) -> bytes:
    validate_url(url)
    opener = urllib.request.build_opener(OfficialRedirect())
    with opener.open(url, timeout=60) as response:
        validate_redirect(url, response.geturl())
        content = response.read(MAX_FILE_BYTES + 1)
    if not 0 < len(content) <= MAX_FILE_BYTES:
        raise ValueError("Official tokenizer response is empty or exceeds the size limit")
    return content


def validate_content(name: str, content: bytes) -> None:
    if not 0 < len(content) <= MAX_FILE_BYTES:
        raise ValueError("Invalid tokenizer download size")
    if name.endswith(".json"):
        value = unique_json(content)
        if name in ("config.json", "tokenizer_config.json") and "auto_map" in value:
            raise ValueError("Remote tokenizer code is not authorized")
    else:
        content.decode("utf-8")


def prepare(destination: Path, *, fetch=None) -> dict:
    """仅显式命令调用：先解析官方 commit，再按该 commit 下载 allowlist。"""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing tokenizer cache")
    fetch = fetch or fetch_official
    sources, contents = {}, {}
    for repository, canonical in REPOSITORIES.items():
        latest = unique_json(fetch(f"https://huggingface.co/api/models/{repository}"))
        revision = latest.get("sha", "")
        if latest.get("id") != canonical or not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError("Unexpected official tokenizer repository or revision")
        api_url = f"https://huggingface.co/api/models/{repository}/revision/{revision}"
        metadata = unique_json(fetch(api_url))
        if metadata.get("id") != canonical or metadata.get("sha") != revision:
            raise ValueError("Pinned tokenizer metadata changed identity")
        siblings = metadata.get("siblings")
        if not isinstance(siblings, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("rfilename"), str)
            for row in siblings
        ):
            raise ValueError("Invalid official tokenizer file listing")
        names = {row["rfilename"] for row in siblings}
        if len(names) != len(siblings):
            raise ValueError("Duplicate official tokenizer file listing")
        required = required_files(repository)
        if not required <= names:
            raise ValueError(f"Missing required official tokenizer files: {repository}")
        files = {}
        for name in sorted(names & (required | OPTIONAL_FILES)):
            url = f"https://huggingface.co/{canonical}/resolve/{revision}/{name}"
            content = fetch(url)
            validate_content(name, content)
            files[name] = {"url": url, "size": len(content), "sha256": digest(content)}
            contents[f"hub/models--{repository}/snapshots/{revision}/{name}"] = content
        contents[f"hub/models--{repository}/refs/main"] = revision.encode()
        sources[repository] = {
            "revision": revision,
            "canonical_repository": canonical,
            "api_url": api_url,
            "files": files,
        }
        validate_source(repository, sources[repository])
    # 所有网络及内容校验成功后才创建新缓存；已有 vocab/merges 或缓存一律不覆盖。
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    verify_cache(destination, sources)
    lock = {"schema_version": 1, "repositories": sources}
    with (destination / "source-lock.json").open("x") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--destination", type=Path, default=APPROVED_DESTINATION)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if os.environ.get("GITHUB_ACTIONS") == "true":
            raise ValueError("Tokenizer preparation is separate from CI asset preflight")
        if (
            args.destination != APPROVED_DESTINATION
            or args.destination.resolve() != APPROVED_DESTINATION
        ):
            raise ValueError(
                "Tokenizer preparation requires the explicitly approved NAS destination"
            )
        if args.lock:
            raise ValueError(
                "Prepare writes a candidate source-lock.json; it cannot replace a trusted lock"
            )
        lock = prepare(args.destination)
    else:
        if not args.lock:
            raise ValueError("Read-only verification requires an explicitly supplied trusted lock")
        lock = unique_json(read_file(args.lock.parent, args.lock.name))
        verify_cache(args.destination, validate_lock(lock))
    print(json.dumps({"repositories": sorted(lock["repositories"]), "numerical_validation": False}))


if __name__ == "__main__":
    main()
