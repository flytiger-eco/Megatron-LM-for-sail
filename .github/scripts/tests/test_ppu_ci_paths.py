"""CPU-only checks for trusted path routing, masks and logical manifest compatibility."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_ppu_functional import ROOT, SCRIPTS, functional

paths = functional.paths
hf = functional.hf
NAMES = ("PPU_HOST_VOLUMES", *paths.ROOT_NAMES)


@pytest.fixture
def path_env():
    return {
        "PPU_HOST_VOLUMES": "/fictional-host:/fictional-pod,/other-host:/other-pod",
        "PPU_CPU_RUN_ROOT": "/fictional-host/runs",
        "PPU_POD_RUN_ROOT": "/fictional-pod/runs",
        "PPU_CPU_ASSETS_ROOT": "/other-host/assets",
        "PPU_POD_ASSETS_ROOT": "/other-pod/assets",
    }


@pytest.mark.parametrize("route", ["run", "assets", "functional"])
def test_independent_cpu_and_pod_roots_share_their_own_mount(route, path_env):
    paths.validate(route, path_env)
    assert path_env["PPU_POD_RUN_ROOT"] != "/mnt" + path_env["PPU_CPU_RUN_ROOT"]


@pytest.mark.parametrize("name", NAMES)
def test_missing_route_secret_is_fail_closed(name, path_env):
    del path_env[name]
    route = "assets" if "ASSETS" in name or "RECEIPTS" in name else "run"
    with pytest.raises(ValueError):
        paths.validate(route, path_env)


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


def test_masks_cover_all_configured_roots_and_mount_components(path_env, capsys):
    paths.register_masks({**path_env, "GITHUB_ACTIONS": "true"})
    output = capsys.readouterr().out
    for value in (*path_env.values(), *paths.replacements(path_env)):
        assert f"::add-mask::{value}\n" in output
    text = json.dumps(path_env)
    public = paths.redact(text, path_env)
    assert "fictional-host" not in public and "other-host" not in public
    assert "ppu://assets/cpu" in public and "ppu://runs/pod" in public


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
