"""HF 来源、只读资产门禁、逐重复隔离及原 tokenizer 离线行为的 CPU 回归。"""

import copy
import io
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from test_ppu_functional import ROOT, SCRIPTS, functional
from test_ppu_functional_results import envelope, evidence, results
from test_ppu_functional_runtime import FakeRunner, prepare_case, runtime
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

hf = functional.hf
REVISION = "a" * 40


def encoded(value):
    return json.dumps(value).encode()


@pytest.fixture
def official():
    """合成官方响应结构，不代表已下载真实资产或获得真实官方 SHA。"""
    payloads = {}
    for repository, canonical in hf.REPOSITORIES.items():
        if repository == "gpt2":
            vocab = {token: i for i, token in enumerate(pre_tokenizers.ByteLevel.alphabet())}
            vocab["<|endoftext|>"] = len(vocab)
            tokenizer = Tokenizer(models.BPE(vocab, []))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            config = {"model_type": "gpt2"}
            tokenizer_config = {"tokenizer_class": "GPT2Tokenizer", "model_max_length": 1024}
            files = {"vocab.json": encoded(vocab), "merges.txt": b"#version: 0.2\n"}
        else:
            tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
            tokenizer = Tokenizer(
                models.WordPiece(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]")
            )
            tokenizer.normalizer = normalizers.BertNormalizer(
                lowercase=repository.endswith("uncased")
            )
            tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            config = {"model_type": "bert"}
            tokenizer_config = {"tokenizer_class": "BertTokenizer", "model_max_length": 512}
            files = {"vocab.txt": ("\n".join(tokens) + "\n").encode()}
        files.update(
            {
                "config.json": encoded(config),
                "tokenizer_config.json": encoded(tokenizer_config),
                "tokenizer.json": tokenizer.to_str().encode(),
                "special_tokens_map.json": b"{}",
            }
        )
        metadata = {
            "id": canonical,
            "sha": REVISION,
            "siblings": [
                {"rfilename": name}
                for name in [
                    *files,
                    "model.safetensors",
                    "pytorch_model.bin",
                    "tokenizer.py",
                    "teacher/data.bin",
                ]
            ],
        }
        api = f"https://huggingface.co/api/models/{repository}"
        payloads[api] = encoded(metadata)
        payloads[f"{api}/revision/{REVISION}"] = encoded(metadata)
        for name, content in files.items():
            payloads[f"https://huggingface.co/{canonical}/resolve/{REVISION}/{name}"] = content
    return payloads


@pytest.fixture
def provisioned(tmp_path, official):
    data = tmp_path / "nas-input"
    destination = data / hf.CACHE_RELATIVE
    fetch = Mock(side_effect=lambda url: official[url])
    lock = hf.prepare(destination, fetch=fetch)
    return data, destination, lock, fetch


def dependencies(lock, repositories=None):
    return [
        {
            "repository": repository,
            "purpose": "MODEL_ARGS.tokenizer",
            "status": "fixed_source",
            "source": source,
        }
        for repository, source in lock["repositories"].items()
        if repositories is None or repository in repositories
    ]


def snapshot(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_prepare_pins_only_approved_tokenizer_files(provisioned, official, tmp_path):
    _, destination, lock, fetch = provisioned
    assert {call.args[0] for call in fetch.call_args_list} == set(official)
    assert len(fetch.call_args_list) == len(official)
    assert hf.validate_lock(lock) == lock["repositories"]
    assert hf.unique_json((destination / "source-lock.json").read_bytes()) == lock
    assert not any(
        path.suffix in (".bin", ".safetensors", ".py") for path in destination.rglob("*")
    )
    for repository, source in lock["repositories"].items():
        assert source["revision"] == REVISION
        assert (
            destination / f"hub/models--{repository}/refs/main"
        ).read_bytes() == REVISION.encode()
    control = tmp_path / "control"
    path = control / hf.LOCK
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded(lock))
    assert hf.load_sources(control) == lock["repositories"]


@pytest.mark.parametrize("mutation", ["empty", "boolean", "version", "unknown", "extra", "null"])
def test_invalid_lock_fails_library_and_cli(provisioned, tmp_path, mutation):
    _, destination, original, _ = provisioned
    lock = copy.deepcopy(original)
    if mutation == "empty":
        lock["repositories"] = {}
    elif mutation == "boolean":
        lock["schema_version"] = True
    elif mutation == "version":
        lock["schema_version"] = 2
    elif mutation == "unknown":
        lock["repositories"]["unapproved"] = next(iter(lock["repositories"].values()))
    elif mutation == "extra":
        lock["extra"] = 1
    else:
        lock = None
    path = tmp_path / hf.LOCK
    path.parent.mkdir()
    path.write_bytes(encoded(lock))
    with pytest.raises(ValueError):
        hf.load_sources(tmp_path)
    before = snapshot(destination)
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "ppu_functional_hf.py"),
            "verify",
            "--destination",
            str(destination),
            "--lock",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode != 0
    assert "numerical_validation" not in process.stdout
    assert snapshot(destination) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "canonical",
        "api",
        "missing",
        "weight",
        "path",
        "url",
        "size",
        "boolean",
        "digest",
        "extra",
    ],
)
def test_source_lock_rejects_unpinned_or_unauthorized_files(provisioned, mutation):
    source = copy.deepcopy(provisioned[2]["repositories"]["gpt2"])
    row = source["files"]["config.json"]
    if mutation == "revision":
        source["revision"] = "main"
    elif mutation == "canonical":
        source["canonical_repository"] = "someone/gpt2"
    elif mutation == "api":
        source["api_url"] = "https://huggingface.co/api/models/gpt2"
    elif mutation == "missing":
        source["files"].pop("merges.txt")
    elif mutation in ("weight", "path"):
        source["files"]["model.safetensors" if mutation == "weight" else "../vocab.json"] = row
    elif mutation == "url":
        row["url"] = row["url"].replace(REVISION, "main")
    elif mutation in ("size", "boolean"):
        row["size"] = 0 if mutation == "size" else True
    elif mutation == "digest":
        row["sha256"] = "bad"
    else:
        row["extra"] = "value"
    with pytest.raises(ValueError):
        hf.validate_source("gpt2", source)


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "revision",
        "pinned",
        "missing",
        "siblings",
        "duplicate",
        "remote_code",
        "empty",
        "malformed",
    ],
)
def test_prepare_failure_never_creates_partial_ready_cache(tmp_path, official, mutation):
    api = "https://huggingface.co/api/models/gpt2"
    pinned = f"{api}/revision/{REVISION}"
    file_url = f"https://huggingface.co/openai-community/gpt2/resolve/{REVISION}/config.json"
    if mutation in ("identity", "revision"):
        metadata = json.loads(official[api])
        metadata["id" if mutation == "identity" else "sha"] = "wrong"
        official[api] = encoded(metadata)
    elif mutation in ("pinned", "missing", "siblings", "duplicate"):
        metadata = json.loads(official[pinned])
        if mutation == "pinned":
            metadata["sha"] = "b" * 40
        elif mutation == "missing":
            metadata["siblings"] = []
        elif mutation == "siblings":
            metadata["siblings"] = [None]
        else:
            metadata["siblings"] *= 2
        official[pinned] = encoded(metadata)
    else:
        official[file_url] = {
            "remote_code": b'{"auto_map": {}}',
            "empty": b"",
            "malformed": b"not json",
        }[mutation]
    destination = tmp_path / "new-cache"
    with pytest.raises(ValueError):
        hf.prepare(destination, fetch=lambda url: official[url])
    assert not destination.exists()


def test_prepare_never_overwrites_existing_or_linked_directory(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "vocab.json").write_bytes(b"original")
    linked = tmp_path / "linked"
    linked.symlink_to(original, target_is_directory=True)
    fetch = Mock(side_effect=AssertionError("下载不应发生"))
    for path in (original, linked):
        with pytest.raises(FileExistsError):
            hf.prepare(path, fetch=fetch)
    fetch.assert_not_called()
    assert (original / "vocab.json").read_bytes() == b"original"


@pytest.mark.parametrize(
    "content",
    [b'{"a":1,"a":2}', b'{"nested":{"a":1,"a":2}}', b'{"a":NaN}', b'{"a":Infinity}', b"[]"],
)
def test_json_metadata_is_strict(content):
    with pytest.raises(ValueError):
        hf.unique_json(content)


@pytest.mark.parametrize(
    "name,content",
    [
        ("config.json", b'{"auto_map":{}}'),
        ("tokenizer_config.json", b'{"auto_map":null}'),
        ("vocab.txt", b"\xff"),
        ("vocab.txt", b""),
    ],
)
def test_content_validation_rejects_code_and_invalid_text(name, content):
    with pytest.raises(ValueError):
        hf.validate_content(name, content)


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/x",
        "https://evil.example/x",
        "https://huggingface.co.evil.example/x",
        "https://user@huggingface.co/x",
        "https://huggingface.co:443/x",
        "https://huggingface.co/x#fragment",
    ],
)
def test_download_host_is_strict(url):
    with pytest.raises(ValueError):
        hf.validate_url(url)


@pytest.mark.parametrize(
    "target",
    [
        f"https://huggingface.co/openai-community/gpt2/resolve/{'b' * 40}/config.json",
        "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
        f"https://huggingface.co/openai-community/gpt2/resolve/{REVISION}/pytorch_model.bin",
        "https://example.com/file",
        "http://huggingface.co/file",
    ],
)
def test_redirect_cannot_change_commit_file_or_host(target):
    origin = f"https://huggingface.co/openai-community/gpt2/resolve/{REVISION}/config.json"
    handler = hf.OfficialRedirect()
    with pytest.raises(ValueError):
        handler.redirect_request(urllib.request.Request(origin), None, 302, "redirect", {}, target)


def test_official_canonical_and_immutable_cache_redirects():
    hf.validate_redirect(
        f"https://huggingface.co/api/models/gpt2/revision/{REVISION}",
        f"https://huggingface.co/api/models/openai-community/gpt2/revision/{REVISION}",
    )
    hf.validate_redirect(
        f"https://huggingface.co/openai-community/gpt2/resolve/{REVISION}/config.json",
        f"https://huggingface.co/api/resolve-cache/models/openai-community/gpt2/{REVISION}/config.json?etag=test",
    )
    with pytest.raises(ValueError):
        hf.validate_redirect(
            f"https://huggingface.co/api/models/gpt2/revision/{REVISION}",
            "https://huggingface.co/api/models/gpt2",
        )


@pytest.mark.parametrize("content", [b"", b"0123456789abcdefX"])
def test_network_response_size_is_bounded(monkeypatch, content):
    url = "https://huggingface.co/api/models/gpt2"
    response = io.BytesIO(content)
    response.geturl = lambda: url
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(hf.urllib.request, "build_opener", lambda *args: opener)
    monkeypatch.setattr(hf, "MAX_FILE_BYTES", 16)
    with pytest.raises(ValueError, match="size limit"):
        hf.fetch_official(url)
    opener.open.assert_called_once_with(url, timeout=60)


@pytest.mark.parametrize("relative", ["../outside", "/absolute", ".", "a/../../outside"])
def test_cache_paths_cannot_escape(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        hf.read_file(tmp_path, relative)


@pytest.mark.parametrize("component", ["leaf", "parent", "root"])
def test_cache_rejects_symlinks(tmp_path, component):
    root = tmp_path / "root"
    (root / "parent").mkdir(parents=True)
    (root / "parent/file").write_bytes(b"value")
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    (root / "alias").symlink_to(root / "parent", target_is_directory=True)
    (root / "parent/alias").symlink_to(root / "parent/file")
    base, relative = {
        "root": (link, "parent/file"),
        "parent": (root, "alias/file"),
        "leaf": (root, "parent/alias"),
    }[component]
    with pytest.raises(ValueError, match="symlinks"):
        hf.read_file(base, relative)


def test_empty_oversized_directory_cache_entries_fail(tmp_path, monkeypatch):
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "oversized").write_bytes(b"12345")
    (tmp_path / "directory").mkdir()
    monkeypatch.setattr(hf, "MAX_FILE_BYTES", 4)
    for name in ("empty", "oversized", "directory"):
        with pytest.raises(ValueError, match="regular file"):
            hf.read_file(tmp_path, name)


def test_repeats_copy_only_manifest_files_and_never_write_input(provisioned, tmp_path):
    data, destination, lock, _ = provisioned
    (destination / "unapproved-extra").write_bytes(b"must not propagate")
    before = snapshot(data)
    deps = dependencies(lock)
    outputs = [tmp_path / f"repeat-{index}/cache/huggingface" for index in (1, 2)]
    for output in outputs:
        report = hf.populate(data, output, deps)
        assert report == hf.verify_cache(output, lock["repositories"])
        assert set(snapshot(output)) == set(hf.cache_files(lock["repositories"]))
    assert snapshot(data) == before
    name = next(iter(hf.cache_files(lock["repositories"])))
    (outputs[0] / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest or size"):
        hf.verify_cache(outputs[0], lock["repositories"])
    hf.verify_cache(outputs[1], lock["repositories"])
    assert snapshot(data) == before
    with pytest.raises(FileExistsError):
        hf.populate(data, outputs[1], deps)


def test_populate_rejects_overlapping_inputs_even_through_alias(provisioned, tmp_path):
    data, _, lock, _ = provisioned
    link = tmp_path / "nas-alias"
    link.symlink_to(data, target_is_directory=True)
    before = snapshot(data)
    for destination in (data, data / "new-output", data.parent, link / "new-output"):
        with pytest.raises(ValueError, match="overlap"):
            hf.populate(data, destination, dependencies(lock))
    assert snapshot(data) == before


def test_unmapped_or_conflicting_phase_sources_fail(provisioned):
    deps = dependencies(provisioned[2], ["gpt2"])
    assert hf.sources_for(deps * 2) == {"gpt2": deps[0]["source"]}
    conflict = copy.deepcopy(deps[0])
    conflict["source"]["files"]["config.json"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Conflicting"):
        hf.sources_for([*deps, conflict])
    deps[0]["status"] = "source_mapping_required"
    with pytest.raises(ValueError, match="Unresolved"):
        hf.sources_for(deps)


def bind_case(case, lock):
    case["tokenizer_dependencies"] = functional.tokenizer_dependencies(
        ROOT, yaml.safe_load((ROOT / case["config_path"]).read_text()), case["training_script"]
    )
    for dependency in case["tokenizer_dependencies"]:
        dependency.update(
            status="fixed_source", source=lock["repositories"][dependency["repository"]]
        )
    case["assets"].extend(hf.asset_requirements(case["tokenizer_dependencies"]))


def test_catalogue_binds_trusted_lock_for_all_training_phases(provisioned, monkeypatch):
    lock = provisioned[2]
    monkeypatch.setattr(hf, "load_sources", lambda control: hf.validate_lock(lock))
    original_sha = functional.sha256
    monkeypatch.setattr(
        functional,
        "sha256",
        lambda path: hf.digest(encoded(lock)) if path == ROOT / hf.LOCK else original_sha(path),
    )
    catalogue = functional.catalogue(ROOT, ROOT, functional.select_config({}))
    assert catalogue["tokenizer_lock_sha256"] == hf.digest(encoded(lock))
    count = 0
    purposes = set()
    for case in catalogue["functional"].values():
        if case["tokenizer_dependencies"]:
            count += 1
            sources = hf.sources_for(case["tokenizer_dependencies"])
            assert sources
            assert functional.validate_tokenizer_assets(case)
            purposes.update(row["purpose"] for row in case["tokenizer_dependencies"])
    assert count == 84
    assert purposes


def test_fixed_sources_cover_replacement_args_without_inheriting_first_phase(provisioned):
    lock = provisioned[2]
    config = {
        "TEST_TYPE": "ckpt-resume",
        "MODEL_ARGS": {"--tokenizer-type": "BertWordPieceCase"},
        "MODEL_ARGS_2": {},
    }
    deps = functional.tokenizer_dependencies(ROOT, config, "pretrain_gpt.py")
    assert [(row["section"], row["repository"]) for row in deps] == [
        ("MODEL_ARGS", "bert-large-cased"),
        ("MODEL_ARGS_2", "gpt2"),
    ]
    for row in deps:
        row.update(status="fixed_source", source=lock["repositories"][row["repository"]])
    assert set(hf.sources_for(deps)) == {"bert-large-cased", "gpt2"}
    assert any("models--gpt2" in row["relative_path"] for row in hf.asset_requirements(deps))


@pytest.mark.parametrize("location", ["cpu", "pod"])
def test_asset_preflight_reads_fixed_digests_without_downloading(
    evidence, provisioned, location, monkeypatch
):
    manifest, case, _ = evidence
    data, _, lock, _ = provisioned
    for asset in case["assets"]:
        path = data / asset["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"explicit input fixture")
    bind_case(case, lock)
    monkeypatch.setattr(hf, "fetch_official", Mock(side_effect=AssertionError("资产预检禁止下载")))
    before = snapshot(data)
    report = functional.asset_report(manifest, data, location=location)
    assert report["ok"] and report["result"] == "assets_ready"
    assert report["numerical_validation"] is False
    assert snapshot(data) == before
    requirement = hf.asset_requirements(case["tokenizer_dependencies"])[0]
    (data / requirement["relative_path"]).write_bytes(b"tampered")
    before = snapshot(data)
    report = functional.asset_report(manifest, data, location=location)
    assert not report["ok"] and report["result"] == "asset_layout_error"
    assert any("digest or size" in row["reason"] for row in report["cases"][0]["assets"])
    assert snapshot(data) == before
    hf.fetch_official.assert_not_called()


def test_candidate_readback_is_read_only_and_does_not_download(provisioned, monkeypatch):
    data, destination, lock, _ = provisioned
    (destination / "unapproved-extra").write_bytes(b"private_marker")
    before = snapshot(data)
    trusted_before = hf.load_sources(ROOT)
    monkeypatch.setattr(hf, "fetch_official", Mock(side_effect=AssertionError("禁止隐式下载")))
    reader = Mock(wraps=hf.read_file)
    monkeypatch.setattr(hf, "read_file", reader)
    report = hf.inspect_candidate(data)
    assert report["status"] == "candidate_matches_cache"
    assert report["cache_matches_candidate"] and not report["trusted"]
    assert report["source_lock"] == lock
    assert report["source_lock_sha256"] == hf.digest(
        (destination / "source-lock.json").read_bytes()
    )
    assert report["file_count"] == len(hf.cache_files(lock["repositories"]))
    assert "private_marker" not in json.dumps(report)
    assert all("unapproved-extra" not in str(call.args) for call in reader.call_args_list)
    assert snapshot(data) == before and hf.load_sources(ROOT) == trusted_before
    hf.fetch_official.assert_not_called()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "empty",
        "malformed",
        "duplicate",
        "oversized",
        "nested",
        "weight",
        "symlink",
        "directory",
        "permission",
    ],
)
def test_candidate_invalid_metadata_is_not_exported(provisioned, monkeypatch, mutation):
    data, destination, lock, _ = provisioned
    path = destination / "source-lock.json"
    payloads = {
        "empty": b"",
        "malformed": b"private_marker",
        "duplicate": b'{"private_marker":1,"private_marker":2}',
        "oversized": b" " * (128 * 1024 + 1),
        "nested": b'{"a":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
    }
    if mutation in payloads:
        path.write_bytes(payloads[mutation])
    elif mutation == "weight":
        lock["repositories"]["gpt2"]["files"]["private_marker.bin"] = {}
        path.write_bytes(encoded(lock))
    elif mutation == "permission":
        monkeypatch.setattr(hf, "read_file", Mock(side_effect=PermissionError("private_marker")))
    else:
        saved = path.with_name("saved-candidate")
        path.rename(saved)
        if mutation == "symlink":
            path.symlink_to(saved)
        elif mutation == "directory":
            path.mkdir()
    before = snapshot(data)
    report = hf.inspect_candidate(data)
    assert report["status"] == (
        "source_lock_missing" if mutation == "missing" else "source_lock_invalid"
    )
    assert report["source_lock"] is None and report["source_lock_sha256"] is None
    assert not report["cache_matches_candidate"] and not report["trusted"]
    assert "private_marker" not in json.dumps(report)
    assert snapshot(data) == before


@pytest.mark.parametrize("mutation", ["missing", "digest", "symlink", "ref", "changed_lock"])
def test_candidate_cache_mismatch_is_not_accepted(provisioned, monkeypatch, mutation):
    data, destination, lock, _ = provisioned
    name = f"hub/models--gpt2/snapshots/{REVISION}/config.json"
    path = destination / name
    if mutation in ("missing", "symlink"):
        saved = path.with_name("saved-config")
        path.rename(saved)
        if mutation == "symlink":
            path.symlink_to(saved)
    elif mutation == "digest":
        path.write_bytes(b"tampered")
    elif mutation == "ref":
        (destination / "hub/models--gpt2/refs/main").write_bytes(b"b" * 40)
    else:
        original_read = hf.read_file
        count = 0

        def changed_read(root, relative):
            nonlocal count
            value = original_read(root, relative)
            if str(relative).endswith("source-lock.json"):
                count += 1
                if count == 2:
                    return value + b"\n"
            return value

        monkeypatch.setattr(hf, "read_file", changed_read)
    before = snapshot(data)
    report = hf.inspect_candidate(data)
    assert report["status"] == "cache_mismatch"
    assert report["source_lock"] == lock
    assert not report["cache_matches_candidate"] and not report["trusted"]
    assert report["file_count"] == 0 and report["reason"]
    assert snapshot(data) == before


def test_preparation_receipts_export_only_expected_fields(tmp_path, monkeypatch):
    directory = tmp_path.resolve()
    expected = {"repositories": sorted(hf.REPOSITORIES), "numerical_validation": False}
    for command in ("prepare", "verify"):
        (directory / f"{command}.log").write_bytes(encoded(expected) + b"\n")
        (directory / f"{command}.exitcode").write_text("0\n")
    (directory / "verify.stderr").write_bytes(b"private_marker\n\xff\n")
    (directory / "ppu_functional_hf.py").write_bytes(b"private_script_marker")
    (directory / "unrelated.txt").write_bytes(b"private_unrelated_marker")
    before = snapshot(directory)
    reader = Mock(wraps=hf.read_file)
    monkeypatch.setattr(hf, "read_file", reader)
    monkeypatch.setattr(hf, "fetch_official", Mock(side_effect=AssertionError("禁止下载")))
    report = hf.inspect_receipts(directory)
    assert report["status"] == "inspected" and not report["reason"]
    rows = {row["name"]: row for row in report["files"]}
    for command in ("prepare", "verify"):
        assert rows[f"{command}.log"]["expected_summary_count"] == 1
        assert rows[f"{command}.log"]["unrecognized_line_count"] == 0
        assert rows[f"{command}.exitcode"]["reported_exit_code"] == 0
    assert rows["verify.stderr"]["unrecognized_line_count"] == 2
    assert rows["ppu_functional_hf.py"]["sha256"] == hf.digest(b"private_script_marker")
    assert rows["unrelated.txt"]["status"] == "metadata_only"
    assert all(call.args[1] != "unrelated.txt" for call in reader.call_args_list)
    assert "private_" not in json.dumps(report)
    assert snapshot(directory) == before
    hf.fetch_official.assert_not_called()


@pytest.mark.parametrize(
    "content,code",
    [(b"1\n", 1), (b"255", 255), (b"256", None), (b"-1", None), (b"true", None), (b"", None)],
)
def test_receipt_exit_code_is_not_inferred_from_success_output(tmp_path, content, code):
    directory = tmp_path.resolve()
    (directory / "prepare.exit_code.txt").write_bytes(content)
    row = hf.inspect_receipts(directory)["files"][0]
    assert row["reported_exit_code"] == code
    assert row["expected_summary_count"] == 0


@pytest.mark.parametrize("mutation", ["numeric_false", "extra", "duplicate", "nested"])
def test_receipt_summary_must_match_strict_success_schema(tmp_path, mutation):
    expected = {"repositories": sorted(hf.REPOSITORIES), "numerical_validation": False}
    if mutation == "numeric_false":
        expected["numerical_validation"] = 0
    elif mutation == "extra":
        expected["secret"] = "private_marker"
    content = encoded(expected)
    if mutation == "duplicate":
        content = content[:-1] + b', "numerical_validation": false}'
    elif mutation == "nested":
        content = b'{"a":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"
    directory = tmp_path.resolve()
    (directory / "prepare.log").write_bytes(content)
    report = hf.inspect_receipts(directory)
    assert report["files"][0]["expected_summary_count"] == 0
    assert report["files"][0]["unrecognized_line_count"] == 1
    assert "private_marker" not in json.dumps(report)


@pytest.mark.parametrize("kind", ["missing", "symlink", "large", "directory", "fifo", "many"])
def test_receipts_reject_unbounded_or_linked_inputs(tmp_path, kind):
    directory = tmp_path.resolve() / "receipts"
    directory.mkdir()
    path = directory / "prepare.log"
    if kind == "missing":
        directory = directory / "missing"
    elif kind == "symlink":
        alias = directory.with_name("alias")
        alias.symlink_to(directory, target_is_directory=True)
        directory = alias
    elif kind == "large":
        path.write_bytes(b"x" * (256 * 1024 + 1))
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        for index in range(65):
            (directory / f"prepare-{index}.log").touch()
    report = hf.inspect_receipts(directory)
    if kind in ("missing", "symlink", "many"):
        assert report["status"] == "unreadable" and report["reason"]
    else:
        assert report["files"][0]["status"] == "not_bounded_regular_file"


@pytest.mark.parametrize("include_receipts", [False, True])
@pytest.mark.parametrize(
    "mode,location,reachable,fixed",
    [
        ("assets", "cpu", True, False),
        ("assets", "cpu", True, True),
        ("assets", "cpu", False, False),
        ("assets", "pod", True, False),
        ("regression", "cpu", True, False),
        ("regression", "pod", True, True),
    ],
)
def test_candidate_cli_never_promotes_trust_or_changes_gate(
    evidence, provisioned, tmp_path, monkeypatch, mode, location, reachable, fixed, include_receipts
):
    manifest, case, _ = evidence
    data, _, lock, _ = provisioned
    manifest["config"]["functional_mode"] = mode
    if reachable:
        for asset in case["assets"]:
            path = data / asset["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"explicit input fixture")
    else:
        data = tmp_path / "unreachable-nas"
    if fixed:
        bind_case(case, lock)
    else:
        case["tokenizer_dependencies"] = [
            {
                "repository": "gpt2",
                "status": "source_mapping_required",
                "purpose": "MODEL_ARGS.tokenizer",
            }
        ]
    output = tmp_path / "cpu-plan"
    output.mkdir()
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(encoded(manifest))
    baseline = functional.asset_report(manifest, data, location=location)
    before = snapshot(data)
    inspect = Mock(wraps=hf.inspect_candidate)
    monkeypatch.setattr(hf, "inspect_candidate", inspect)
    receipts = Mock(return_value={"status": "inspected", "files": []})
    monkeypatch.setattr(hf, "inspect_receipts", receipts)
    monkeypatch.setattr(hf, "fetch_official", Mock(side_effect=AssertionError("禁止隐式下载")))
    summary = tmp_path / "step-summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ppu_functional.py",
            "assets",
            "--output",
            str(output),
            "--assets-root",
            str(data),
            "--location",
            location,
            *(["--tokenizer-receipts", str(tmp_path)] if include_receipts else []),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        functional.main()
    assert stopped.value.code == (0 if baseline["ok"] else 1)
    report = json.loads((output / "assets-report.json").read_text())
    candidate = report.pop("tokenizer_source_candidate", None)
    receipt_report = report.pop("tokenizer_preparation_receipts", None)
    if include_receipts and mode == "assets" and location == "cpu" and reachable:
        receipts.assert_called_once_with(tmp_path)
        assert receipt_report == receipts.return_value
    else:
        receipts.assert_not_called()
        assert receipt_report is None
    if mode == "assets" and location == "cpu" and reachable:
        inspect.assert_called_once_with(data)
        assert candidate["cache_matches_candidate"] and candidate["trusted"] is False
        assert "not trusted" in summary.read_text()
    else:
        inspect.assert_not_called()
        assert candidate is None
    assert report == baseline and report["numerical_validation"] is False
    assert manifest_path.read_bytes() == encoded(manifest)
    assert snapshot(data) == before
    hf.fetch_official.assert_not_called()


def runtime_case(evidence, provisioned, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    source = provisioned[1]
    target = Path(manifest["config"]["data_path"]) / hf.CACHE_RELATIVE
    for name in hf.cache_files(provisioned[2]["repositories"]):
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    bind_case(case, provisioned[2])
    return manifest, case, directory


def test_runtime_caches_and_evidence_are_independently_verified(evidence, provisioned, tmp_path):
    manifest, case, directory = runtime_case(evidence, provisioned, tmp_path)
    before = snapshot(Path(manifest["config"]["data_path"]))
    runner = FakeRunner(manifest, case)
    assert runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)["success"]
    report = (directory / "assets-report.json").read_bytes()
    envelope(manifest, case, directory)
    (directory / "assets-report.json").write_bytes(report)
    assert results.verify_case(manifest, ROOT, case, directory)["success"]
    assert len(runner.calls) == 2
    for _, env, _ in runner.calls:
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert Path(env["HF_HOME"]).is_relative_to(env["OUTPUT_PATH"])
        assert (Path(env["OUTPUT_PATH"]) / "tokenizer-cache.json").is_file()
    assert snapshot(Path(manifest["config"]["data_path"])) == before


@pytest.mark.parametrize("mutation", ["missing", "digest", "report", "preflight", "manifest"])
def test_forged_cache_or_preflight_evidence_cannot_pass(evidence, provisioned, tmp_path, mutation):
    manifest, case, directory = runtime_case(evidence, provisioned, tmp_path)
    runner = FakeRunner(manifest, case)
    assert runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)["success"]
    preflight = json.loads((directory / "assets-report.json").read_text())
    envelope(manifest, case, directory)
    path = directory / "attempt-1/repeat-1"
    name = next(iter(hf.cache_files(hf.sources_for(case["tokenizer_dependencies"]))))
    cached = path / "cache/huggingface" / name
    if mutation == "missing":
        cached.rename(cached.with_name("missing"))
    elif mutation == "digest":
        cached.write_bytes(b"tampered")
    elif mutation == "report":
        (path / "tokenizer-cache.json").write_text("{}")
    elif mutation == "preflight":
        next(row for row in preflight["cases"][0]["assets"] if row.get("tokenizer_cache"))[
            "sha256"
        ] = ("0" * 64)
    else:
        case["assets"] = [row for row in case["assets"] if not row.get("tokenizer_cache")]
        with pytest.raises(ValueError, match="asset requirements"):
            functional.validate_tokenizer_assets(case)
    (directory / "assets-report.json").write_bytes(encoded(preflight))
    assert not results.verify_case(manifest, ROOT, case, directory)["success"]


def test_copy_failure_after_preflight_never_trains_or_retries(
    evidence, provisioned, tmp_path, monkeypatch
):
    manifest, case, directory = runtime_case(evidence, provisioned, tmp_path)

    def corrupt_after_preflight(*args, **kwargs):
        row = hf.asset_requirements(case["tokenizer_dependencies"])[0]
        (Path(manifest["config"]["data_path"]) / row["relative_path"]).write_bytes(b"tampered")
        return hf.populate(*args, **kwargs)

    monkeypatch.setattr(runtime.control.hf, "populate", corrupt_after_preflight)
    runner = FakeRunner(manifest, case)
    record = runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)
    assert not record["success"] and record["attempts"] == 1
    assert not runner.calls
    path = directory / "attempt-1/repeat-1"
    assert (
        json.loads((path / "verification.json").read_text())["classification"] == "integrity_error"
    )
    execution = json.loads((path / "execution.json").read_text())
    assert execution["shell_exit_code"] is None and "digest or size" in execution["error"]
    assert (
        results.verify_repeat(manifest, ROOT, case, path, 1, 1)["classification"]
        == "integrity_error"
    )


def offline_environment(cache):
    return {
        **os.environ,
        "HF_HOME": str(cache),
        "HF_HUB_CACHE": str(cache / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "hub"),
        "TRANSFORMERS_CACHE": str(cache / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }


@pytest.mark.parametrize("repository", list(hf.REPOSITORIES))
@pytest.mark.parametrize("explicit_vocab", [False, True])
def test_original_auto_tokenizer_call_in_fresh_offline_process(
    provisioned, tmp_path, repository, explicit_vocab
):
    data, _, lock, _ = provisioned
    cache = tmp_path / "repeat/cache/huggingface"
    deps = dependencies(lock, [repository])
    hf.populate(data, cache, deps)
    source = ROOT / "megatron/core/tokenizers/text/libraries/huggingface_tokenizer.py"
    # 只执行原构造函数的 AutoTokenizer 调用段；不是完整 Megatron/PPU 训练验收。
    program = '''
import ast, json, socket, sys
from pathlib import Path
from types import SimpleNamespace

def no_network(*args, **kwargs):
    raise AssertionError("offline tokenizer attempted network access")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network
from transformers import AutoTokenizer
source, repository, cache, revision, explicit = sys.argv[1:]
module = ast.parse(Path(source).read_text())
cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "HuggingFaceTokenizer")
init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
call = next(node for node in init.body if isinstance(node, ast.Try))
snapshot = Path(cache) / "hub" / ("models--" + repository) / "snapshots" / revision
vocab = snapshot / ("vocab.json" if repository == "gpt2" else "vocab.txt")
namespace = dict(AutoTokenizer=AutoTokenizer, self=SimpleNamespace(), tokenizer_path=repository,
                 vocab_file=str(vocab) if explicit == "true" else None,
                 merges_file=str(snapshot / "merges.txt") if explicit == "true" and repository == "gpt2" else None,
                 use_fast=True, trust_remote_code=False)
exec(compile(ast.Module(body=[call], type_ignores=[]), source, "exec"), namespace)
tokenizer = namespace["self"].tokenizer
ids = tokenizer.encode("hello world")
assert ids and tokenizer.is_fast
print(json.dumps(dict(repository=repository, tokens=len(ids), offline=True)))
'''
    env = offline_environment(cache)
    command = [
        sys.executable,
        "-c",
        program,
        str(source),
        repository,
        str(cache),
        REVISION,
        str(explicit_vocab).lower(),
    ]
    process = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["offline"] is True
    hf.verify_cache(cache, hf.sources_for(deps))
    ref = cache / f"hub/models--{repository}/refs/main"
    ref.rename(ref.with_name("missing"))
    failed = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert failed.returncode != 0
    assert "Unable to instantiate HuggingFace AutoTokenizer" in failed.stderr
    assert "offline tokenizer attempted network access" not in failed.stderr


@pytest.mark.parametrize("repository", list(hf.REPOSITORIES))
@pytest.mark.parametrize("include_special_tokens", [False, True])
def test_complete_original_tokenizer_classes_offline(
    provisioned, tmp_path, repository, include_special_tokens
):
    """执行完整原类与映射，隔离训练栈导入；不替代真实官方资产或 PPU 验收。"""
    data, _, lock, _ = provisioned
    cache = tmp_path / "repeat/cache/huggingface"
    deps = dependencies(lock, [repository])
    hf.populate(data, cache, deps)
    before = snapshot(data)
    cache_before = snapshot(cache)
    source = ROOT / "megatron/core/tokenizers/text/libraries"
    program = '''
import ast, json, logging, socket, sys
from pathlib import Path
from typing import List, Optional

network_calls = []
def no_network(*args, **kwargs):
    network_calls.append(True)
    raise AssertionError("offline tokenizer attempted network access")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network
from transformers import AutoTokenizer
source, repository, cache, revision, include_special = sys.argv[1:]
source, cache = Path(source), Path(cache)
namespace = dict(AutoTokenizer=AutoTokenizer, List=List, Optional=Optional,
                 logging=logging, logger=logging.getLogger("original-tokenizer"),
                 log_single_rank=lambda logger, level, message: logger.log(level, message))
path = source / "abstract_tokenizer.py"
exec(compile(path.read_text(), str(path), "exec"), namespace)
for filename, class_name in [
    ("huggingface_tokenizer.py", "HuggingFaceTokenizer"),
    ("megatron_hf_tokenizer.py", "MegatronHFTokenizer"),
]:
    path = source / filename
    module = ast.parse(path.read_text())
    nodes = [node for node in module.body if
             (isinstance(node, ast.ClassDef) and node.name == class_name) or
             (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
              target.id == "MEGATRON_CONFIG_MAP" for target in node.targets))]
    assert sum(isinstance(node, ast.ClassDef) for node in nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

def reject_download(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "_download" and frame.f_code.co_filename == str(source / "megatron_hf_tokenizer.py"):
        raise AssertionError("original tokenizer entered download path")
sys.setprofile(reject_download)
name = next(name for name, row in namespace["MEGATRON_CONFIG_MAP"].items()
            if row["tokenizer_name"] == repository)
files = cache / "hub" / ("models--" + repository) / "snapshots" / revision
sentinels = ["<extra_id_0>", "<extra_id_1>"]
kwargs = dict(vocab_file=str(files / ("vocab.json" if repository == "gpt2" else "vocab.txt")),
              merges_file=str(files / "merges.txt") if repository == "gpt2" else None,
              include_special_tokens=include_special == "true", additional_special_tokens=sentinels)
tokenizer = namespace["MegatronHFTokenizer"](name, **kwargs)
sys.setprofile(None)
assert tokenizer.tokenizer.is_fast
assert tokenizer._hf_tokenizer is tokenizer.tokenizer
assert not tokenizer.use_gigatoken
assert tokenizer.vocab_size == len(tokenizer.vocab) == len(tokenizer.inv_vocab)
assert tokenizer.vocab_size == tokenizer.original_vocab_size + len(sentinels)
ids = tokenizer.text_to_ids("hello world")
assert ids and tokenizer.ids_to_text(ids, remove_special_tokens=True).strip() == "hello world"
assert tokenizer.ids_to_text(ids) == tokenizer.ids_to_text(ids, remove_special_tokens=include_special != "true")
if repository != "gpt2" and include_special == "true":
    assert ids[0] == tokenizer.cls_id and ids[-1] == tokenizer.sep_id
    assert "[CLS]" in tokenizer.ids_to_text(ids) and "[SEP]" in tokenizer.ids_to_text(ids)
tokens = tokenizer.text_to_tokens("hello world")
assert tokenizer.ids_to_tokens(tokenizer.tokens_to_ids(tokens)) == tokens
assert tokenizer.tokens_to_text(tokens).strip() == "hello world"
assert tokenizer.token_to_id(tokens[0]) == tokenizer.tokens_to_ids(tokens)[0]
expected_ids = tokenizer.tokenizer("hello world").input_ids if include_special == "true" else tokenizer.tokens_to_ids(tokens)
assert ids == expected_ids
assert tokenizer.space_sensitive == (tokenizer.text_to_tokens("x y") != tokenizer.text_to_tokens("x") + tokenizer.text_to_tokens("y"))
assert tokenizer.additional_special_tokens_ids == tokenizer.tokens_to_ids(sentinels)
assert len(set(tokenizer.additional_special_tokens_ids)) == len(sentinels)
assert all(index >= tokenizer.original_vocab_size for index in tokenizer.additional_special_tokens_ids)
assert tokenizer.eod == tokenizer.eos_id == tokenizer.sep_id
assert tokenizer.bos_id == tokenizer.cls_id
assert tokenizer.eod is not None and tokenizer.bos_id is not None
if repository != "gpt2":
    assert tokenizer.pad_id == tokenizer.token_to_id("[PAD]")
    assert tokenizer.unk_id == tokenizer.token_to_id("[UNK]")
    assert tokenizer.mask_id == tokenizer.token_to_id("[MASK]")
assert tokenizer.ids_to_text(tokenizer.additional_special_tokens_ids, remove_special_tokens=True) == ""
assert "<extra_id_0>" in tokenizer.ids_to_text(tokenizer.additional_special_tokens_ids, remove_special_tokens=False)
assert not network_calls
print(json.dumps(dict(repository=repository, offline=True, full_classes=True)))
'''
    command = [
        sys.executable,
        "-c",
        program,
        str(source),
        repository,
        str(cache),
        REVISION,
        str(include_special_tokens).lower(),
    ]
    process = subprocess.run(
        command,
        cwd=tmp_path,
        env=offline_environment(cache),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {
        "repository": repository,
        "offline": True,
        "full_classes": True,
    }
    assert snapshot(cache) == cache_before
    assert snapshot(data) == before
    hf.verify_cache(cache, hf.sources_for(deps))
    ref = cache / f"hub/models--{repository}/refs/main"
    ref.rename(ref.with_name("missing"))
    failed = subprocess.run(
        command,
        cwd=tmp_path,
        env=offline_environment(cache),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert failed.returncode != 0
    assert "Unable to instantiate HuggingFace AutoTokenizer" in failed.stderr
    assert "offline tokenizer attempted network access" not in failed.stderr
    assert "original tokenizer entered download path" not in failed.stderr
    assert snapshot(data) == before


@pytest.mark.parametrize(
    "in_ci,arguments",
    [
        (True, ["prepare"]),
        (False, ["prepare", "--destination", "/tmp/unapproved"]),
        (False, ["prepare", "--lock", "/tmp/untrusted-lock"]),
        (False, ["verify"]),
    ],
)
def test_cli_cannot_download_from_ci_or_without_approved_destination(in_ci, arguments, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", str(in_ci).lower())
    monkeypatch.setattr(sys, "argv", ["ppu_functional_hf.py", *arguments])
    fetch = Mock(side_effect=AssertionError("下载不应发生"))
    monkeypatch.setattr(hf, "fetch_official", fetch)
    with pytest.raises(ValueError):
        hf.main()
    fetch.assert_not_called()


def test_cli_verify_only_reports_asset_readiness(provisioned, monkeypatch, capsys):
    data, destination, _, _ = provisioned
    before = snapshot(data)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ppu_functional_hf.py",
            "verify",
            "--destination",
            str(destination),
            "--lock",
            str(destination / "source-lock.json"),
        ],
    )
    monkeypatch.setattr(hf, "fetch_official", Mock(side_effect=AssertionError("只读校验禁止下载")))
    hf.main()
    report = json.loads(capsys.readouterr().out)
    assert report == {"repositories": sorted(hf.REPOSITORIES), "numerical_validation": False}
    assert snapshot(data) == before
    hf.fetch_official.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "same_size", "symlink"])
def test_populate_revalidates_every_source_before_creating_output(provisioned, tmp_path, damage):
    data, destination, lock, _ = provisioned
    files = hf.cache_files(lock["repositories"])
    name = next(iter(files))
    path = destination / name
    if damage == "same_size":
        path.write_bytes(b"b" * files[name]["size"])
    else:
        saved = path.with_name("saved")
        path.rename(saved)
        if damage == "symlink":
            path.symlink_to(saved)
    output = tmp_path / "output-cache"
    with pytest.raises((OSError, ValueError)):
        hf.populate(data, output, dependencies(lock))
    assert not output.exists()


def test_download_transport_failure_does_not_create_cache(tmp_path, official):
    def failed_fetch(url):
        if "/resolve/" in url:
            raise ConnectionResetError("connection reset by peer")
        return official[url]

    path = tmp_path / "new-cache"
    with pytest.raises(ConnectionResetError):
        hf.prepare(path, fetch=failed_fetch)
    assert not path.exists()


def test_source_lock_cannot_be_a_symlink(provisioned, tmp_path):
    destination = provisioned[1]
    path = tmp_path / "control" / hf.LOCK
    path.parent.mkdir(parents=True)
    path.symlink_to(destination / "source-lock.json")
    with pytest.raises(ValueError, match="symlinks"):
        hf.load_sources(tmp_path / "control")
