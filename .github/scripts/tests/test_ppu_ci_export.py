"""Public evidence export preserves numeric evidence and blocks unsafe path disclosures."""

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from string import Template

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.summary.writer.event_file_writer import EventFileWriter
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


def write_events(directory, private):
    writer = EventFileWriter(str(directory))
    for step in range(1, 4):
        writer.add_event(
            Event(
                step=step,
                wall_time=100.0 + step,
                summary=Summary(
                    value=[
                        Summary.Value(
                            tag="lm loss",
                            simple_value=1.25,
                            metadata=SummaryMetadata(
                                display_name="loss", summary_description=private
                            ),
                        ),
                        Summary.Value(
                            tag="note", tensor=TensorProto(dtype=7, string_val=[private.encode()])
                        ),
                    ]
                ),
            )
        )
    writer.close()
    return next(directory.glob("events.out.tfevents.*"))


def scalar_events(directory):
    reader = EventAccumulator(str(directory)).Reload()
    return [(row.step, row.wall_time, row.value) for row in reader.Scalars("lm loss")]


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


@pytest.mark.parametrize("kind", ["set", "binary-key", "binary-value"])
def test_unsupported_yaml_collections_and_binary_paths_fail_closed(tmp_path, path_env, kind):
    import base64

    source = tmp_path / "private"
    source.mkdir()
    private = path_env["PPU_CPU_RUN_ROOT"]
    if kind == "set":
        raw = 'paths: !!set {"' + private.replace("/", "\\u002f") + '": null}\n'
    else:
        encoded = base64.b64encode(private.encode()).decode()
        raw = (
            f"!!binary {encoded}: safe\n" if kind == "binary-key" else f"path: !!binary {encoded}\n"
        )
    (source / "report.yaml").write_text(raw)
    with pytest.raises(ValueError, match="Unsupported"):
        exporter.export(source, tmp_path / "public", env=path_env)
    assert not (tmp_path / "public/public-export.json").exists()


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_unknown_binary_with_private_path_blocks_export(tmp_path, path_env, encoding):
    source = tmp_path / "private"
    source.mkdir()
    (source / "unknown.bin").write_bytes(
        b"\x00\x01" + path_env["PPU_CPU_RUN_ROOT"].encode(encoding)
    )
    before = snapshot(source)
    with pytest.raises(ValueError, match="Private directory"):
        exporter.export(source, tmp_path / "public", env=path_env)
    assert snapshot(source) == before
    assert not (tmp_path / "public/public-export.json").exists()


def test_upstream_expanded_model_config_is_public_text_not_unknown_binary(tmp_path, path_env):
    source = tmp_path / "private/attempt-1/repeat-1"
    source.mkdir(parents=True)
    recipe = ROOT / (
        "tests/functional_tests/zhenwu_test_cases/gpt/gpt3_mcore_te_tp2_pp2_cp2/model_config.yaml"
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


@pytest.mark.parametrize("filename", ["unknown.tmp", "model_config.yaml.tmp.bin"])
def test_expanded_config_exception_does_not_allow_arbitrary_tmp_files(tmp_path, path_env, filename):
    source = tmp_path / "private"
    source.mkdir()
    (source / filename).write_text("path: " + path_env["PPU_POD_RUN_ROOT"])
    with pytest.raises(ValueError, match="Private directory"):
        exporter.export(source, tmp_path / "public", env=path_env)
    assert not (tmp_path / "public/public-export.json").exists()


@pytest.mark.parametrize("encoding", ["invalid-utf8", "utf-16-le", "utf-16-be"])
def test_expanded_config_must_be_utf8_and_never_rewrites_originals(tmp_path, path_env, encoding):
    source = tmp_path / "private"
    source.mkdir()
    private = path_env["PPU_POD_RUN_ROOT"]
    content = b"\xff" if encoding == "invalid-utf8" else private.encode(encoding)
    path = source / "model_config.yaml.tmp"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        exporter.export(source, tmp_path / "public", env=path_env)
    assert path.read_bytes() == content
    assert not (tmp_path / "public/public-export.json").exists()


@pytest.mark.parametrize("escaping", ["unicode", "hex", "yaml-set", "json-slashes"])
def test_expanded_config_cannot_hide_encoded_paths_or_unsafe_structures(
    tmp_path, path_env, escaping
):
    source = tmp_path / "private"
    source.mkdir()
    private = path_env["PPU_POD_RUN_ROOT"]
    if escaping == "yaml-set":
        raw = 'paths: !!set {"' + private + '": null}\n'
    elif escaping == "json-slashes":
        raw = json.dumps({"path": private}).replace("/", "\\/")
    else:
        escape = "\\u002f" if escaping == "unicode" else "\\x2f"
        raw = 'path: "' + private.replace("/", escape) + '"\n'
    path = source / "model_config.yaml.tmp"
    path.write_text(raw)
    with pytest.raises(ValueError, match="Unsupported"):
        exporter.export(source, tmp_path / "public", env=path_env)
    assert path.read_text() == raw
    assert not (tmp_path / "public/public-export.json").exists()


def test_unknown_safe_binary_is_copied_without_byte_replacement(tmp_path, path_env):
    source = tmp_path / "private"
    source.mkdir()
    content = bytes(range(256))
    (source / "safe.bin").write_bytes(content)
    exporter.export(source, tmp_path / "public", env=path_env)
    assert (tmp_path / "public/safe.bin").read_bytes() == content


@pytest.mark.parametrize(
    "filename",
    [
        "model_config.yaml",
        "golden_values_fixture.json",
        "observation.json",
        "cache/huggingface/hub/tokenizer_config.json",
    ],
)
def test_immutable_config_actual_and_hf_bytes_never_change(tmp_path, path_env, filename):
    source = tmp_path / "private"
    path = source / filename
    path.parent.mkdir(parents=True)
    content = b'{"metric":{"values":{"1":1.25,"2":0}},"revision":"fixed"}\n'
    path.write_bytes(content)
    report = exporter.export(source, tmp_path / "public", env=path_env)
    assert (tmp_path / "public" / filename).read_bytes() == content
    assert report["files"][filename]["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    "filename", ["model_config.yaml", "observation.json", "cache/huggingface/config.json"]
)
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


def test_real_tensorboard_text_redaction_preserves_every_scalar_and_crc(tmp_path, path_env):
    source = tmp_path / "private"
    events = source / "tensorboard"
    original = write_events(events, path_env["PPU_POD_RUN_ROOT"] + "/attempt-1")
    before = scalar_events(events)
    private_bytes = original.read_bytes()
    exporter.export(source, tmp_path / "public", env=path_env)
    assert (
        scalar_events(tmp_path / "public/tensorboard")
        == before
        == [(1, 101.0, 1.25), (2, 102.0, 1.25), (3, 103.0, 1.25)]
    )
    assert original.read_bytes() == private_bytes
    public_file = tmp_path / "public/tensorboard" / original.name
    assert path_env["PPU_POD_RUN_ROOT"].encode() not in public_file.read_bytes()
    assert b"ppu://runs/pod/attempt-1" in public_file.read_bytes()
    # A second pass validates rewritten length and payload checksums with no further text changes.
    exporter.tensorboard(public_file, tmp_path / "second-pass", exporter.Redactor(path_env))
    assert public_file.read_bytes() == (tmp_path / "second-pass").read_bytes()


@pytest.mark.parametrize(
    "damage", ["header-crc", "payload-crc", "truncated", "unknown-field", "tag"]
)
def test_tensorboard_corruption_or_unmodifiable_fields_fail_closed(tmp_path, path_env, damage):
    source = tmp_path / "private"
    path = write_events(source, path_env["PPU_CPU_RUN_ROOT"])
    raw = bytearray(path.read_bytes())
    if damage == "header-crc":
        raw[8] ^= 1
    elif damage == "payload-crc":
        raw[-1] ^= 1
    elif damage == "truncated":
        raw = raw[:-1]
    else:
        event = Event(step=1, wall_time=123)
        if damage == "unknown-field":
            event.graph_def = path_env["PPU_CPU_RUN_ROOT"].encode()
        else:
            event.summary.value.add(tag=path_env["PPU_CPU_RUN_ROOT"], simple_value=1.25)
        data = event.SerializeToString()
        length = struct.pack("<Q", len(data))
        raw = (
            length
            + struct.pack("<I", exporter.crc32c(length))
            + data
            + struct.pack("<I", exporter.crc32c(data))
        )
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        exporter.export(source, tmp_path / "public", env=path_env)


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


@pytest.mark.parametrize(
    "relative", ["attempt-1/checkpoints/link", "initial-checkpoints/link", "cache/compiler/link"]
)
@pytest.mark.parametrize("link_kind", ["file", "directory", "dangling"])
def test_excluded_links_never_export_or_follow_private_targets(
    tmp_path, path_env, monkeypatch, relative, link_kind
):
    source = tmp_path / "private"
    source.mkdir()
    (source / "driver.log").write_text("Training complete\n")
    hf = source / "cache/huggingface/config.json"
    hf.parent.mkdir(parents=True)
    hf.write_bytes(b'{"revision":"fixed"}\n')
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "private.bin"
    target.write_bytes(path_env["PPU_POD_RUN_ROOT"].encode())
    if link_kind == "directory":
        target = outside
    elif link_kind == "dangling":
        target = outside / "missing"
    link = source / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=link_kind == "directory")
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path, *args, **kwargs):
        assert path != link and not path.is_relative_to(outside), "Excluded target must not be read"
        return original_open(path, *args, **kwargs)

    def guarded_stat(path, *args, **kwargs):
        if kwargs.get("follow_symlinks", True):
            assert path != link, "Excluded link must not be followed"
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    destination = tmp_path / "public"
    report = exporter.export(source, destination, env=path_env)
    assert set(report["files"]) == {"driver.log", "cache/huggingface/config.json"}
    assert (destination / "cache/huggingface/config.json").read_bytes() == hf.read_bytes()
    assert relative not in report["omitted"]
    exporter.verify_public(destination)


@pytest.mark.parametrize("name", ["cache", "checkpoints", "initial-checkpoints"])
def test_excluded_directory_links_do_not_traverse_external_trees(tmp_path, path_env, name):
    source = tmp_path / "private"
    source.mkdir()
    (source / "driver.log").write_text("Training complete\n")
    outside = tmp_path / "outside/huggingface"
    outside.mkdir(parents=True)
    (outside / "config.json").write_text(json.dumps({"path": path_env["PPU_POD_RUN_ROOT"]}))
    (source / name).symlink_to(outside.parent, target_is_directory=True)
    report = exporter.export(source, tmp_path / "public", env=path_env)
    assert set(report["files"]) == {"driver.log"}
    assert report["omitted"] == []


@pytest.mark.parametrize("relative", ["escape", "cache/huggingface/escape"])
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


def test_local_overlay_does_not_bypass_retained_source_symlink_rejection(tmp_path, path_env):
    source = tmp_path / "private"
    source.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (source / "action-timing.json").symlink_to(target)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "action-timing.json").write_text("{}")
    with pytest.raises(ValueError, match="Symlinks"):
        exporter.export(source, tmp_path / "public", overlay=overlay, env=path_env)
    assert target.read_text() == "{}"
    assert not (tmp_path / "public/public-export.json").exists()


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


def test_action_outputs_are_allowlisted_without_polluting_captured_json(tmp_path, path_env):
    env = {
        **os.environ,
        **path_env,
        "GITHUB_ACTIONS": "true",
        "ACTION_OUTPUTS": json.dumps(
            {
                "job_status": "failed",
                "duration_seconds": "12",
                "queue_duration_seconds": "1",
                "occupied_ppu_card_hours": "0.01",
                "execution_ppu_card_hours": "0.01",
                "artifact_dir": path_env["PPU_CPU_RUN_ROOT"],
                "task_pods": "private",
            }
        ),
    }
    process = subprocess.run(
        [sys.executable, "-I", str(SCRIPTS / "ppu_ci_export.py"), "--action-outputs"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert set(json.loads(process.stdout)) == exporter.ACTION_FIELDS
    assert "add-mask" not in process.stdout and "fictional-host" not in process.stdout


@pytest.mark.parametrize("damage", [None, "actual", "model", "identity", "repeat", "asset-root"])
def test_schema_two_public_evidence_recomputes_original_gates(evidence, tmp_path, damage):
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
    for index, path in enumerate(directory.rglob("events.out.tfevents.fixture")):
        real = write_events(path.parent, str(assets))
        path.write_bytes(real.read_bytes())
        real.rename(tmp_path / f"removed-event-{index}")
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
    report = results.verify_case(manifest, ROOT, case, public)
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


def test_checkpoint_audit_is_advisory_and_refuses_nas_mutation(tmp_path, monkeypatch):
    source = tmp_path / "private"
    checkpoint = source / 'attempt-1/checkpoints/quoted"file.bin'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"123")
    report = tmp_path / "local/checkpoint-omissions.json"
    exporter.checkpoint_audit(source, report)
    assert json.loads(report.read_text())["omitted"] == [
        {"path": "attempt-1/checkpoints", "files": 1, "bytes": 3}
    ]
    with pytest.raises(ValueError, match="must not write"):
        exporter.checkpoint_audit(source, source / "checkpoint-omissions.json")
    monkeypatch.setattr(
        Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(PermissionError())
    )
    exporter.checkpoint_audit(source, tmp_path / "failed/audit.json")
    assert checkpoint.read_bytes() == b"123"


@pytest.mark.parametrize("nested", [False, True])
def test_private_directory_in_public_filename_is_rejected(tmp_path, path_env, nested):
    source = tmp_path / "private"
    relative = Path(path_env["PPU_CPU_RUN_ROOT"].lstrip("/")) / "evidence.txt"
    if nested:
        relative = Path("nested") / relative
    path = source / relative
    path.parent.mkdir(parents=True)
    path.write_text("Safe content with an unsafe filename")
    with pytest.raises(ValueError, match="filename"):
        exporter.export(source, tmp_path / "public", env=path_env)
    assert not (tmp_path / "public/public-export.json").exists()


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {"size": True, "sha256": "x"},
        {"size": -1, "sha256": "x"},
        {"size": 8, "sha256": 1},
        {"size": 8, "sha256": "x", "extra": 1},
    ],
)
def test_public_inventory_rejects_malformed_file_metadata(tmp_path, path_env, metadata):
    source = tmp_path / "private"
    source.mkdir()
    (source / "evidence.txt").write_text("Original")
    destination = tmp_path / "public"
    exporter.export(source, destination, env=path_env)
    report_path = destination / "public-export.json"
    report = json.loads(report_path.read_text())
    report["files"]["evidence.txt"] = metadata
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metadata"):
        exporter.verify_public(destination)


@pytest.mark.parametrize("target", ["root", "inventory"])
def test_public_verifier_rejects_links_before_reading_inventory(tmp_path, path_env, target):
    source = tmp_path / "private"
    source.mkdir()
    (source / "evidence.txt").write_text("Original")
    destination = tmp_path / "public"
    exporter.export(source, destination, env=path_env)
    if target == "root":
        link = tmp_path / "linked-public"
        link.symlink_to(destination, target_is_directory=True)
        destination = link
    else:
        report = destination / "public-export.json"
        moved = tmp_path / "original-inventory.json"
        report.rename(moved)
        report.symlink_to(moved)
    with pytest.raises(ValueError, match="unlinked"):
        exporter.verify_public(destination)


@pytest.mark.parametrize(
    "filename,content,code,category",
    [
        ("untrusted-name.bin", b"PRIVATE", "private-path-bytes", "unknown-binary"),
        ("untrusted-name.json", b"{PRIVATE", "invalid-json", "structured-evidence"),
        ("model_config.yaml.tmp", b"\xff", "invalid-utf8", "expanded-model-config"),
        ("untrusted-name.log", b"\xff", "invalid-utf8", "text-evidence"),
    ],
)
def test_public_failure_diagnostics_use_fixed_codes_not_filenames(
    tmp_path, path_env, filename, content, code, category
):
    source = tmp_path / "private"
    source.mkdir()
    (source / filename).write_bytes(
        content.replace(b"PRIVATE", path_env["PPU_CPU_RUN_ROOT"].encode())
    )
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
        env={**os.environ, **path_env, "GITHUB_ACTIONS": "false"},
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert f"code={code}; evidence={category}" in process.stderr
    assert filename not in process.stdout + process.stderr
    assert str(source) not in process.stdout + process.stderr
    assert not any(value in process.stdout + process.stderr for value in path_env.values())
    assert "Traceback" not in process.stderr
    assert not (tmp_path / "public/public-export.json").exists()


def test_arbitrary_exception_messages_are_never_public_diagnostics(path_env):
    private = path_env["PPU_POD_RUN_ROOT"] + "/arbitrary-secret-filename"
    error = exporter.ExportFileError("structured-evidence", ValueError(private))
    summary = exporter.failure_summary(error)
    assert summary == (
        "Public evidence export failed (ValueError; code=export-failed; evidence=structured-evidence)"
    )
    assert private not in summary and "arbitrary-secret-filename" not in summary


def test_required_export_failure_does_not_print_private_exception_paths(tmp_path, path_env):
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
        env={**os.environ, **path_env},
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0 and "Public evidence export failed" in process.stderr
    assert not any(value in process.stdout + process.stderr for value in path_env.values())
