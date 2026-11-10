"""CPU-only checks for trusted path routing, masks and logical manifest compatibility."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from test_ppu_functional import ROOT, SCRIPTS, functional
from test_ppu_functional_hf import hf
from test_ppu_functional_results import evidence
from test_ppu_functional_runtime import runtime
from test_ppu_upstream import ci, sealed_bundle

paths = functional.paths
NAMES = ("PPU_HOST_VOLUMES", *paths.ROOT_NAMES)


@pytest.fixture
def path_env():
    return {
        "PPU_HOST_VOLUMES": "/fictional-host:/fictional-pod,/other-host:/other-pod",
        "PPU_CPU_RUN_ROOT": "/fictional-host/runs",
        "PPU_POD_RUN_ROOT": "/fictional-pod/runs",
        "PPU_CPU_ASSETS_ROOT": "/other-host/assets",
        "PPU_POD_ASSETS_ROOT": "/other-pod/assets",
        "PPU_TOKENIZER_RECEIPTS_ROOT": "/fictional-host/receipts",
    }


@pytest.mark.parametrize("route", ["run", "assets", "functional"])
def test_independent_cpu_and_pod_roots_share_their_own_mount(route, path_env):
    paths.validate(route, path_env)
    assert path_env["PPU_POD_RUN_ROOT"] != "/mnt" + path_env["PPU_CPU_RUN_ROOT"]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/",
        "relative",
        "/a/",
        "/a//b",
        "/a/../b",
        "/a/./b",
        "/a\nb",
        "/a\rb",
        "/a\x00b",
        "/a,b",
        "/a:b",
        "/a b",
        "/a;id",
        "/a$(id)",
        "/a`id`",
        "/a\\b",
        "/a\tb",
    ],
)
def test_untrusted_directory_format_never_echoes_the_value(value):
    with pytest.raises(ValueError) as error:
        paths.absolute(value, "TEST_ROOT")
    assert str(error.value) == "Missing or invalid trusted directory: TEST_ROOT"


@pytest.mark.parametrize("name", NAMES)
def test_missing_route_secret_is_fail_closed(name, path_env):
    del path_env[name]
    route = "assets" if "ASSETS" in name or "RECEIPTS" in name else "run"
    with pytest.raises(ValueError):
        paths.validate(route, path_env)


@pytest.mark.parametrize("route,unused", [("run", "ASSETS"), ("assets", "RUN")])
def test_unused_route_configuration_can_be_empty(route, unused, path_env):
    for name in paths.ROOT_NAMES:
        if unused in name or (route == "run" and "RECEIPTS" in name):
            path_env.pop(name)
    paths.validate(route, path_env)


@pytest.mark.parametrize(
    "mounts", ["", "/a", "/a:/b:/c", "/a:/b,/a:/c", "/a:/c,/b:/c", "/a:/b,", "/a:/b,$(id):/c"]
)
def test_invalid_or_ambiguous_mount_mapping(mounts, path_env):
    path_env["PPU_HOST_VOLUMES"] = mounts
    with pytest.raises(ValueError):
        paths.validate("run", path_env)


@pytest.mark.parametrize("side", ["CPU", "POD"])
@pytest.mark.parametrize("kind", ["RUN", "ASSETS"])
def test_mismatched_root_mapping_is_rejected(side, kind, path_env):
    path_env[f"PPU_{side}_{kind}_ROOT"] = "/unmapped/root"
    with pytest.raises(ValueError):
        paths.validate("functional", path_env)


def test_nested_pod_mount_cannot_shadow_the_expected_host_root(path_env):
    path_env["PPU_HOST_VOLUMES"] += ",/shadow-host:/fictional-pod/runs"
    with pytest.raises(ValueError, match="share a mount"):
        paths.validate("run", path_env)
    path_env["PPU_CPU_RUN_ROOT"] = "/shadow-host"
    paths.validate("run", path_env)


def logical_manifest():
    return {
        "schema_version": 2,
        "test_kind": "functional",
        "config": {"data_path": "ppu://assets/pod", "cpu_data_path": "ppu://assets/cpu"},
    }


@pytest.mark.parametrize("location", ["cpu", "pod"])
def test_logical_asset_root_uses_only_trusted_environment(location, path_env):
    manifest = logical_manifest()
    paths.require_current(manifest)
    assert paths.asset_root(manifest, location, path_env) == Path(
        path_env[f"PPU_{location.upper()}_ASSETS_ROOT"]
    )
    key = "cpu_data_path" if location == "cpu" else "data_path"
    manifest["config"][key] = "/fictional-unapproved/root"
    with pytest.raises(ValueError):
        paths.asset_root(manifest, location, path_env)


@pytest.mark.parametrize("schema", [0, 1, 3, 4, "2", None])
def test_old_or_unknown_schema_cannot_start_new_training(schema):
    manifest = logical_manifest()
    manifest["schema_version"] = schema
    with pytest.raises(ValueError):
        paths.require_current(manifest)


def test_old_schema_is_available_only_for_offline_physical_resolution(path_env):
    manifest = {"schema_version": 1, "config": {"data_path": "/historical-fixture/assets"}}
    assert paths.asset_root(manifest, "pod", path_env) == Path("/historical-fixture/assets")
    manifest["schema_version"] = 7
    with pytest.raises(ValueError):
        paths.asset_root(manifest, "pod", path_env)


def test_masks_cover_all_configured_roots_and_mount_components(path_env, capsys):
    paths.register_masks({**path_env, "GITHUB_ACTIONS": "true"})
    output = capsys.readouterr().out
    for value in (*path_env.values(), *paths.replacements(path_env)):
        assert f"::add-mask::{value}\n" in output
    text = json.dumps(path_env)
    public = paths.redact(text, path_env)
    assert "fictional-host" not in public and "other-host" not in public
    assert "ppu://assets/cpu" in public and "ppu://runs/pod" in public


@pytest.mark.parametrize(
    "filename", ["ppu-build", "ppu-execute", "ppu-ut", "ppu-functional", "ppu-compiler-diagnostic"]
)
def test_all_parent_child_secret_boundaries_are_explicit(filename):
    workflow = yaml.safe_load((ROOT / f".github/workflows/{filename}.yml").read_text())
    event = workflow.get("on", workflow.get(True))
    if "workflow_call" in event:
        assert set(event["workflow_call"]["secrets"]) == set(NAMES)
        assert all(
            value["required"] is False for value in event["workflow_call"]["secrets"].values()
        )
    for job in workflow["jobs"].values():
        if job.get("uses", "").startswith("./.github/workflows/ppu-"):
            assert job["secrets"] == {name: "${{ secrets." + name + " }}" for name in NAMES}
        for index, step in enumerate(job.get("steps", [])):
            if "shared-workflows/cluster-task" in step.get("uses", ""):
                assert step["with"]["host_volumes"] == "${{ secrets.PPU_HOST_VOLUMES }}"
                assert "${{ secrets.PPU_POD_RUN_ROOT }}" in step["with"]["extra_env"]
                assert "PPU_CPU_RUN_ROOT" not in step["with"]["extra_env"]
                assert "GH_TOKEN" not in step["with"]["extra_env"]
                assert "github.token" not in json.dumps(step["with"])
                assert "ppu_ci_paths.py" in job["steps"][index - 1]["run"]
            if step.get("uses") == "actions/upload-artifact@v4":
                if step["with"].get("name") == "ppu-control-environment":
                    assert filename == "ppu-build"
                    assert job is workflow["jobs"]["control-environment"]
                    assert step["with"]["path"] == ".ppu-control-export/"
                    assert "outputs.allowed == 'true'" in job["if"]
                    assert "secrets." not in json.dumps(job)
                    continue
                assert step["with"]["path"].startswith(".ppu-public/")
                assert "outcome == 'success'" in step["if"]
    assert "secrets: inherit" not in json.dumps(workflow)


def test_workflow_steps_receive_only_their_runtime_route_secrets(path_env):
    for filename in (
        "ppu-build",
        "ppu-execute",
        "ppu-ut",
        "ppu-functional",
        "ppu-compiler-diagnostic",
    ):
        workflow = yaml.safe_load((ROOT / f".github/workflows/{filename}.yml").read_text())
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                env = step.get("env", {})
                configured = set(env) & set(NAMES)
                run = step.get("run", "")
                if "ppu_ci_paths.py run" in run:
                    assert configured == {
                        "PPU_HOST_VOLUMES",
                        "PPU_CPU_RUN_ROOT",
                        "PPU_POD_RUN_ROOT",
                    }
                elif "ppu_ci_paths.py assets" in run:
                    assert configured == {
                        "PPU_HOST_VOLUMES",
                        "PPU_CPU_ASSETS_ROOT",
                        "PPU_POD_ASSETS_ROOT",
                        "PPU_TOKENIZER_RECEIPTS_ROOT",
                    }
                elif "ppu_ci_paths.py functional" in run:
                    assert configured == set(NAMES) - {"PPU_TOKENIZER_RECEIPTS_ROOT"}
                if step.get("id") in {"public-authorize-140", "public-configure-131"}:
                    assert not configured
                if "ppu_ci_paths.py" in run:
                    actual = {name: path_env[name] for name in configured}
                    actual["GITHUB_ACTIONS"] = "true"
                    paths.replacements(actual)


def test_hf_preparation_root_cannot_be_changed_by_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("PPU_CPU_ASSETS_ROOT", str(tmp_path / "authorized-assets"))
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    monkeypatch.setattr(sys, "argv", ["hf", "prepare", "--destination", str(tmp_path / "other")])
    prepare = Mock(side_effect=AssertionError("Unapproved preparation must not start"))
    monkeypatch.setattr(hf, "prepare", prepare)
    with pytest.raises(ValueError):
        hf.main()
    prepare.assert_not_called()
    assert hf.approved_destination() == tmp_path / "authorized-assets/tokenizers/huggingface"


@pytest.mark.parametrize("name", ["ppu_ci_paths.py", "ppu_ci_export.py"])
def test_new_bundle_requires_path_helpers_without_changing_old_bundle(sealed_bundle, name):
    plan, bundle = sealed_bundle
    original = (bundle / "bundle-integrity.json").read_bytes()
    ci.verify_bundle(bundle, plan, hashlib.sha256(original).hexdigest())
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["runtime_path_schema"] = 1
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    for helper in ("ppu_ci_paths.py", "ppu_ci_export.py"):
        (bundle / "control" / helper).write_bytes((SCRIPTS / helper).read_bytes())
    ci.seal_bundle(bundle)
    (bundle / "control" / name).rename(bundle / "missing-helper")
    with pytest.raises(ValueError, match="path control helper"):
        ci.seal_bundle(bundle)


def test_normal_runtime_rejects_old_manifest_before_process_start(tmp_path):
    manifest = {
        "schema_version": 1,
        "test_kind": "functional",
        "config": {
            "data_path": "/historical-fixture/assets",
            "cpu_data_path": "/historical-fixture/assets",
        },
    }
    manifest["identity"] = {"target_sha": "a" * 40, "control_sha": "b" * 40}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPTS / "ppu_functional_runtime.py"),
            "--manifest",
            str(path),
            "--root",
            str(ROOT),
            "--case-id",
            "unknown",
            "--output",
            str(tmp_path / "result"),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PPU_POD_ASSETS_ROOT": "/fictional-pod/assets",
            "PPU_TARGET_SHA": "a" * 40,
            "PPU_CONTROL_SHA": "b" * 40,
        },
    )
    assert process.returncode != 0 and "logical-path functional manifest" in process.stderr
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("schema", [1, 2])
def test_combined_sidecar_keeps_its_digest_binding_and_offline_compatibility(tmp_path, schema):
    identity = {"target_sha": "a" * 40, "control_sha": "b" * 40, "run_id": "123", "run_attempt": 1}
    sidecar = logical_manifest()
    sidecar.update(schema_version=schema, identity=identity)
    if schema == 1:
        sidecar["config"].update(data_path="/fictional-old/pod", cpu_data_path="/fictional-old/cpu")
    content = json.dumps(sidecar).encode()
    (tmp_path / "functional-manifest.json").write_bytes(content)
    top = {
        "schema_version": 4,
        "test_kind": "combined",
        "identity": identity,
        "functional_plan": {
            "path": "functional-manifest.json",
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(top))
    assert functional.load_manifest(manifest) == sidecar
    (tmp_path / "functional-manifest.json").write_bytes(content + b" ")
    with pytest.raises(ValueError, match="digest"):
        functional.load_manifest(manifest)


def test_schema_two_runtime_resolves_asset_env_without_putting_roots_into_manifest(
    evidence, tmp_path, monkeypatch
):
    manifest, case, directory = evidence
    manifest["schema_version"] = 2
    manifest["config"].update(data_path="ppu://assets/pod", cpu_data_path="ppu://assets/cpu")
    root = tmp_path / "trusted-assets"
    monkeypatch.setenv("PPU_POD_ASSETS_ROOT", str(root))
    before = json.dumps(manifest)
    with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
        assert env["DATA_PATH"] == str(root)
    assert json.dumps(manifest) == before and str(root) not in before


def test_asset_cli_cannot_override_the_trusted_root(evidence, tmp_path, monkeypatch):
    manifest, _, directory = evidence
    manifest["schema_version"] = 2
    manifest["config"].update(data_path="ppu://assets/pod", cpu_data_path="ppu://assets/cpu")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("PPU_CPU_ASSETS_ROOT", str(tmp_path / "trusted"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assets",
            "assets",
            "--output",
            str(directory),
            "--assets-root",
            str(tmp_path / "unapproved"),
        ],
    )
    with pytest.raises(ValueError, match="differs from trusted"):
        functional.main()
