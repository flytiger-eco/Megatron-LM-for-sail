"""HF source pinning, read-only asset gates, per-repeat isolation and original tokenizer
offline behaviour (CPU regression).
"""

import copy
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from test_ppu_functional import ROOT, functional
from test_ppu_functional_results import envelope, evidence, results
from test_ppu_functional_runtime import FakeRunner, prepare_case, runtime
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

hf = functional.hf
REVISION = "a" * 40


def encoded(value):
    return json.dumps(value).encode()


@pytest.fixture
def official():
    """Synthetic official response structure; does not mean real assets were downloaded
    or real official SHAs obtained.
    """
    payloads = {}
    for repository, canonical in hf.REPOSITORIES.items():
        if repository == "gpt2":
            vocab = {token: i for i, token in enumerate(pre_tokenizers.ByteLevel.alphabet())}
            vocab["<|endoftext|>"] = len(vocab)
            tokenizer = Tokenizer(models.BPE(vocab, []))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
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
            tokenizer.decoder = decoders.WordPiece(prefix="##")
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
            )
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


def test_registered_lock_matches_accepted_nas_digest_and_binds_all_dependencies():
    content = hf.read_file(ROOT, hf.LOCK)
    assert hf.digest(content) == "f6ed92a6a685939dee223c2735cf60d4e0b73e7aea6347060db65c79bb97511b"
    sources = hf.load_sources(ROOT)
    assert set(sources) == set(hf.REPOSITORIES)
    assert len(hf.cache_files(sources)) == 16
    catalogue = functional.catalogue(ROOT, ROOT, functional.select_config({}))
    assert catalogue["tokenizer_lock_sha256"] == hf.digest(content)
    cases = list(catalogue["functional"].values())
    assert cases and any(case["tokenizer_dependencies"] for case in cases)
    for case in cases:
        assert all(row["status"] == "fixed_source" for row in case["tokenizer_dependencies"])
        assert functional.validate_tokenizer_assets(case) == hf.asset_requirements(
            case["tokenizer_dependencies"]
        )


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
    fetch = Mock(side_effect=AssertionError("Download must not happen"))
    for path in (original, linked):
        with pytest.raises(FileExistsError):
            hf.prepare(path, fetch=fetch)
    fetch.assert_not_called()
    assert (original / "vocab.json").read_bytes() == b"original"


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


def bind_case(case, lock):
    case["tokenizer_dependencies"] = functional.tokenizer_dependencies(
        ROOT, yaml.safe_load((ROOT / case["config_path"]).read_text()), case["training_script"]
    )
    for dependency in case["tokenizer_dependencies"]:
        dependency.update(
            status="fixed_source", source=lock["repositories"][dependency["repository"]]
        )
    case["assets"].extend(hf.asset_requirements(case["tokenizer_dependencies"]))


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
    monkeypatch.setattr(
        hf, "fetch_official", Mock(side_effect=AssertionError("Asset preflight must not download"))
    )
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
@pytest.mark.parametrize("include_special_tokens", [False, True])
def test_complete_original_tokenizer_classes_offline(
    provisioned, tmp_path, repository, include_special_tokens
):
    """Run the full original class and mapping with training-stack imports isolated; no substitute
    for real assets or PPU acceptance.
    """
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
    fetch = Mock(side_effect=AssertionError("Download must not happen"))
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
    monkeypatch.setattr(
        hf,
        "fetch_official",
        Mock(side_effect=AssertionError("Read-only verification must not download")),
    )
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
