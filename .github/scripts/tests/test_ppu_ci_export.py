"""Public evidence export preserves numeric evidence and blocks unsafe path disclosures."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from string import Template

import pytest
from test_ppu_ci_paths import path_env
from test_ppu_functional import ROOT, SCRIPTS, functional
from test_ppu_functional_results import completion, evidence, results, successful_repeat

exporter = functional.module_from_path("public_export_tests", SCRIPTS / "ppu_ci_export.py")


def snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("encoding", ["plain", "escaped-slashes", "unicode-escapes"])
def test_json_escapes_are_redacted_structurally(tmp_path, path_env, encoding):
    source = tmp_path / "private"
    source.mkdir()
    value = path_env["PPU_CPU_RUN_ROOT"] + "/run.log"
    raw = json.dumps({"path": value, "actual": [1.25, 0, -4.0]})
    if encoding == "escaped-slashes":
        raw = raw.replace("/", "\\/")
    elif encoding == "unicode-escapes":
        raw = raw.replace("/", "\\u002f")
    (source / "report.json").write_text(raw)
    before = snapshot(source)
    destination = tmp_path / "public"
    exporter.export(source, destination, env=path_env)
    report = json.loads((destination / "report.json").read_text())
    assert report == {"path": "ppu://runs/cpu/run.log", "actual": [1.25, 0, -4.0]}
    assert snapshot(source) == before
    exporter.verify_public(destination)


def test_text_yaml_jsonl_and_failure_stack_are_redacted(tmp_path, path_env):
    source = tmp_path / "private"
    source.mkdir()
    private = path_env["PPU_POD_ASSETS_ROOT"]
    (source / "stack.log").write_text(f"FileNotFoundError: {private}/missing\nexit=23\n")
    (source / "report.yaml").write_text(
        'path: "' + private.replace("/", "\\u002f") + '"\nactual: 1.25\n'
    )
    (source / "report.jsonl").write_text(json.dumps({"path": private}) + "\n")
    destination = tmp_path / "public"
    exporter.export(source, destination, env=path_env)
    assert "ppu://assets/pod/missing" in (destination / "stack.log").read_text()
    for content in snapshot(destination).values():
        assert private.encode() not in content


def test_upstream_expanded_model_config_is_public_text_not_unknown_binary(tmp_path, path_env):
    source = tmp_path / "private/attempt-1/repeat-1"
    source.mkdir(parents=True)
    recipe = ROOT / (
        "tests/functional_tests/test_cases/gpt/gpt3_mcore_te_tp2_pp2_cp2/model_config.yaml"
    )
    original = recipe.read_bytes()
    (source / "model_config.yaml").write_bytes(original)
    run_root = path_env["PPU_POD_RUN_ROOT"] + "/single-case/attempt-1/repeat-1"
    env = {
        "DATA_PATH": path_env["PPU_POD_ASSETS_ROOT"],
        "DATA_CACHE_PATH": run_root + "/cache",
        "TENSORBOARD_PATH": run_root + "/tensorboard",
        "CHECKPOINT_SAVE_PATH": run_root + "/checkpoints",
        "CHECKPOINT_LOAD_PATH": run_root + "/initial-checkpoints",
    }
    expanded = Template(original.decode()).substitute(env).encode()
    temporary = source / "model_config.yaml.tmp"
    temporary.write_bytes(expanded)
    before = snapshot(source)
    destination = tmp_path / "public"
    report = exporter.export(source, destination, env=path_env)
    public = (destination / temporary.name).read_bytes()
    assert public == exporter.Redactor(path_env).text(expanded.decode()).encode()
    assert b"--train-iters: 50" in public and b"--lr: 0.00015" in public
    assert (destination / "model_config.yaml").read_bytes() == original
    assert report["files"][temporary.name]["sha256"] == hashlib.sha256(public).hexdigest()
    assert snapshot(source) == before
    exporter.verify_public(destination)
    upstream = (ROOT / "tests/functional_tests/shell_test_utils/_run_training.sh").read_text()
    assert '>$TRAINING_PARAMS_PATH.tmp' in upstream
    assert 'TRAINING_PARAMS_PATH="$TRAINING_PARAMS_PATH.tmp"' in upstream


@pytest.mark.parametrize(
    "filename", ["model_config.yaml", "golden_values_fixture.json", "observation.json"]
)
def test_immutable_config_and_actual_bytes_never_change(tmp_path, path_env, filename):
    source = tmp_path / "private"
    path = source / filename
    path.parent.mkdir(parents=True)
    content = b'{"metric":{"values":{"1":1.25,"2":0}},"revision":"fixed"}\n'
    path.write_bytes(content)
    report = exporter.export(source, tmp_path / "public", env=path_env)
    assert (tmp_path / "public" / filename).read_bytes() == content
    assert report["files"][filename]["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("filename", ["model_config.yaml", "observation.json"])
def test_private_path_in_immutable_configuration_fails_instead_of_changing_hash(
    tmp_path, path_env, filename
):
    source = tmp_path / "private"
    path = source / filename
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"path": path_env["PPU_CPU_ASSETS_ROOT"]}).replace("/", "\\u002f"))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Immutable"):
        exporter.export(source, tmp_path / "public", env=path_env)
    assert path.read_bytes() == before


def test_private_archives_and_checkpoint_cache_payloads_are_not_public(tmp_path, path_env):
    source = tmp_path / "private"
    source.mkdir()
    for name in (
        "source.tar.gz",
        "environment.tar.gz",
        "compiled-source.tar.gz",
        "control-runtime.tar.gz",
        "wheels/megatron.whl",
        "functional-tools/yq",
        "checkpoints/model.bin",
        "cache/compiler/kernel.so",
        "data/private.bin",
        "control/ppu_ci_paths.py",
        "manifest.json",
        "build-timing.json",
        "runtime-image.json",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if path.suffix == ".json" else b"retained privately\n")
    record = {
        "schema_version": 1,
        "runtime_image_digest": "ghcr.io/flytiger-eco/megatron-lm-for-sail/ppu-ci@sha256:"
        + "e" * 64,
        "base_image_digest": functional.IMAGE.rsplit(":", 1)[0] + "@sha256:" + "f" * 64,
        "environment_digest": "a" * 64,
        "control_sha": "b" * 40,
        "target_sha": "c" * 40,
    }
    (source / "runtime-image.json").write_text(json.dumps(record, indent=2) + "\n")
    before = snapshot(source)
    report = exporter.export(source, tmp_path / "public", kind="bundle", env=path_env)
    assert set(report["files"]) == {
        "control/ppu_ci_paths.py",
        "manifest.json",
        "build-timing.json",
        "runtime-image.json",
    }
    assert (tmp_path / "public/runtime-image.json").read_bytes() == (
        source / "runtime-image.json"
    ).read_bytes()
    assert {
        "source.tar.gz",
        "environment.tar.gz",
        "compiled-source.tar.gz",
        "control-runtime.tar.gz",
        "wheels/megatron.whl",
        "functional-tools/yq",
        "data/private.bin",
    } <= set(report["omitted"])
    assert report["original_digest_applies_to"] == "private NAS originals only"
    assert snapshot(source) == before


@pytest.mark.parametrize("relative", ["escape.txt", "logs/escape.txt"])
@pytest.mark.parametrize("link_kind", ["file", "directory", "source", "destination-parent"])
def test_symlink_escapes_are_rejected(tmp_path, path_env, link_kind, relative):
    source = tmp_path / "private"
    source.mkdir()
    (source / "safe.txt").write_text("Evidence")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("Not public")
    destination = tmp_path / "public"
    if link_kind == "source":
        link = tmp_path / "source-link"
        link.symlink_to(source, target_is_directory=True)
        source = link
    elif link_kind == "destination-parent":
        link = tmp_path / "target-link"
        link.symlink_to(outside, target_is_directory=True)
        destination = link / "public"
    else:
        link = source / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside if link_kind == "directory" else outside / "secret")
    with pytest.raises(ValueError):
        exporter.export(source, destination, env=path_env)


def test_local_overlay_replaces_only_public_placeholder_and_retains_partial_failures(
    tmp_path, path_env
):
    source = tmp_path / "private"
    source.mkdir()
    (source / "action-timing.json").write_bytes(b"")
    (source / "driver.log").write_text(path_env["PPU_POD_RUN_ROOT"] + "/failed: exit 23")
    overlay = tmp_path / "local"
    overlay.mkdir()
    (overlay / "action-timing.json").write_text(
        '{"action_outcome":"failure","action_status":"failed","outputs":{}}'
    )
    before = snapshot(source)
    exporter.export(source, tmp_path / "public", overlay=overlay, env=path_env)
    assert snapshot(source) == before
    assert (
        json.loads((tmp_path / "public/action-timing.json").read_text())["action_outcome"]
        == "failure"
    )
    exporter.export(
        tmp_path / "missing-source", tmp_path / "partial", overlay=overlay, env=path_env
    )
    assert (tmp_path / "partial/action-timing.json").exists()
    (overlay / "model_config.yaml").write_text("Forbidden replacement")
    with pytest.raises(ValueError, match="Unapproved"):
        exporter.export(source, tmp_path / "unapproved", overlay=overlay, env=path_env)


@pytest.mark.parametrize("damage", ["modified", "removed", "extra", "symlink", "inventory"])
def test_independent_public_inventory_verification_rejects_tampering(tmp_path, path_env, damage):
    source = tmp_path / "private"
    source.mkdir()
    (source / "evidence.txt").write_text("Original")
    destination = tmp_path / "public"
    exporter.export(source, destination, env=path_env)
    if damage == "modified":
        (destination / "evidence.txt").write_text("Modified")
    elif damage == "removed":
        (destination / "evidence.txt").rename(tmp_path / "removed")
    elif damage == "extra":
        (destination / "extra.txt").write_text("Uninventoried")
    elif damage == "symlink":
        (destination / "link").symlink_to(source / "evidence.txt")
    else:
        report = json.loads((destination / "public-export.json").read_text())
        report["files"]["../private/evidence.txt"] = report["files"].pop("evidence.txt")
        (destination / "public-export.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        exporter.verify_public(destination)


@pytest.mark.parametrize("damage", [None, "actual", "model", "identity", "repeat", "asset-root"])
def test_public_evidence_preserves_upstream_results_and_rejects_tampering(
    evidence, tmp_path, damage
):
    manifest, case, directory = evidence
    manifest["schema_version"] = 2
    manifest["config"].update(data_path="ppu://assets/pod", cpu_data_path="ppu://assets/cpu")
    for repeat in (1, 2):
        successful_repeat(manifest, case, directory, 1, repeat)
    completion(manifest, case, directory)
    assets = tmp_path / "configured-assets"
    assets.mkdir()
    for requirement in case["assets"]:
        path = assets / requirement["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Explicit fixture input")
    functional.write_asset_report(
        functional.asset_report(manifest, assets, location="pod", case_id=case["id"]), directory
    )
    (directory / "driver.log").write_text(f"Read assets from {assets}\nTraining complete\n")
    for path in directory.rglob("model_config.yaml"):
        path.with_name("model_config.yaml.tmp").write_text(
            f"--data-path: {assets}\n--train-iters: 50\n"
        )
    before = snapshot(directory)
    env = {"PPU_POD_ASSETS_ROOT": str(assets)}
    public = tmp_path / "public"
    exporter.export(directory, public, env=env)
    assert snapshot(directory) == before
    if damage == "actual":
        actual = public / "attempt-1/repeat-1" / Path(case["golden_path"]).name
        data = json.loads(actual.read_text())
        key = next(iter(data))
        step = next(iter(data[key]["values"]))
        data[key]["values"][step] += 1000
        actual.write_text(json.dumps(data))
    elif damage == "model":
        (public / "attempt-1/repeat-1/model_config.yaml").write_text("MODEL_ARGS: {}\n")
    elif damage == "identity":
        report = json.loads((public / "case-run.json").read_text())
        report["identity"]["target_sha"] = "c" * 40
        (public / "case-run.json").write_text(json.dumps(report))
    elif damage == "repeat":
        (public / "attempt-1/repeat-2/execution.json").rename(tmp_path / "removed-repeat")
    elif damage == "asset-root":
        report = json.loads((public / "assets-report.json").read_text())
        report["root"]["checked_path"] = "ppu://unapproved/assets"
        (public / "assets-report.json").write_text(json.dumps(report))
    if damage:
        # Rehash the public view to prove the original numeric/identity/configuration gates still fail.
        checked = tmp_path / "rehash"
        exporter.export(public, checked, env=env)
        public = checked
    report = results.verify_case(manifest, case, public)
    assert report["success"] is (damage is None), report
    if damage != "model":
        for path in directory.rglob("model_config.yaml"):
            relative = path.relative_to(directory)
            assert (tmp_path / "public" / relative).read_bytes() == before[relative.as_posix()]


@pytest.mark.parametrize("target", ["inside", "existing"])
def test_export_never_writes_over_private_or_existing_evidence(tmp_path, path_env, target):
    source = tmp_path / "private"
    source.mkdir()
    (source / "file.txt").write_text("Original")
    destination = source / "nested" if target == "inside" else tmp_path / "existing"
    if target == "existing":
        destination.mkdir()
        (destination / "old.txt").write_text("Keep existing evidence")
    with pytest.raises(ValueError, match="new and outside"):
        exporter.export(source, destination, env=path_env)
    assert (source / "file.txt").read_text() == "Original"


@pytest.mark.parametrize("github_actions", ["true", "false"])
def test_required_export_failure_does_not_print_private_exception_paths(
    tmp_path, path_env, github_actions
):
    source = tmp_path / "private"
    source.mkdir()
    (source / "unknown.bin").write_bytes(path_env["PPU_CPU_RUN_ROOT"].encode())
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPTS / "ppu_ci_export.py"),
            "--source",
            str(source),
            "--destination",
            str(tmp_path / "public"),
        ],
        env={**os.environ, **path_env, "GITHUB_ACTIONS": github_actions},
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0 and "Public evidence export failed" in process.stderr
    # Captured subprocess output contains the masking protocol before Actions processes it.
    # Allow only exact registration commands; errors and all ordinary output must be redacted.
    expected_masks = (
        {
            f"::add-mask::{value}"
            for value in (path_env["PPU_HOST_VOLUMES"], *exporter.paths.replacements(path_env))
        }
        if github_actions == "true"
        else set()
    )
    ordinary_output = "\n".join(
        line for line in process.stdout.splitlines() if line not in expected_masks
    )
    assert not any(value in ordinary_output + process.stderr for value in path_env.values())


def test_public_export_omits_raw_binary_and_cache_without_modifying_them(tmp_path, path_env):
    source = tmp_path / "private"
    source.mkdir()
    (source / "summary.json").write_text('{"success": true}')
    for name in (
        "events.out.tfevents.raw",
        "tensorboard/events.raw",
        "cache/huggingface/config.json",
        "unknown.bin",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private binary:" + path_env["PPU_POD_RUN_ROOT"].encode())
    before = snapshot(source)
    report = exporter.export(source, tmp_path / "public", env=path_env)
    assert set(report["files"]) == {"summary.json"}
    assert len(report["omitted"]) == 4
    assert snapshot(source) == before
