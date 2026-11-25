"""CPU-only coverage for functional selection, assets and upstream numeric parity."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".github/scripts"
spec = importlib.util.spec_from_file_location("ppu_functional", SCRIPTS / "ppu_functional.py")
functional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functional)


def metric(values=None, start=1, end=4, interval=1, precision=None):
    result = {
        "start_step": start,
        "end_step": end,
        "step_interval": interval,
        "values": (
            values if values is not None else {str(step): 1.0 for step in range(start, end + 1)}
        ),
    }
    if precision:
        result["value_precision"] = precision
    return result


def read_metric(tmp_path, values=None, **kwargs):
    path = tmp_path / "metric.json"
    path.write_text(json.dumps({"lm loss": metric(values, **kwargs)}))
    return functional.read_metrics(path)


@pytest.fixture(scope="module")
def catalogue():
    return functional.catalogue(ROOT, ROOT, functional.select_config({}))


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return tuple(
        importlib.import_module(f"tests.functional_tests.python_test_utils.{name}")
        for name in (
            "common",
            "test_pretraining_regular_pipeline",
            "test_pretraining_resume_checkpoint_pipeline",
        )
    )


@pytest.mark.parametrize(
    "event,ref",
    [
        ("push", functional.BRANCH),
        ("pull_request_target", functional.BRANCH),
        ("workflow_dispatch", "refs/heads/ci/ppu-validation"),
        ("merge_group", functional.BRANCH),
    ],
)
def test_functional_only_manual_validation(event, ref):
    with pytest.raises(ValueError, match="manual dispatch"):
        functional.validate_route(event, ref, {"test_kind": "functional"})


@pytest.mark.parametrize("key", ["pr_number", "pr_sha", "pr_base_sha"])
def test_functional_rejects_any_pr_input(key):
    with pytest.raises(ValueError, match="PR status"):
        functional.validate_route(
            "workflow_dispatch", functional.BRANCH, {"test_kind": "functional", key: "1"}
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("scope", "full"),
        ("unit_bucket", "tests/unit_tests/a.py"),
        ("repeat", 0),
        ("build", False),
        ("unit_test", "no"),
    ],
)
def test_functional_rejects_unit_control(key, value):
    with pytest.raises(ValueError):
        functional.validate_route(
            "workflow_dispatch", functional.BRANCH, {"test_kind": "functional", key: value}
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("functional_mode", "full"),
        ("functional_scope", "L4"),
        ("functional_repeat", 0),
        ("functional_repeat", 11),
        ("functional_timeout", 1381),
        ("functional_repeat", True),
        ("functional_timeout", "1.5"),
        ("functional_cases", ""),
        ("functional_cases", "gpt/a,gpt/a"),
        ("functional_cases", "../bad"),
        ("functional_cases", "gpt/a;true"),
        ("functional_cases", "all,gpt/a"),
    ],
)
def test_invalid_config(key, value):
    with pytest.raises(ValueError):
        functional.select_config({key: value})


def test_m890_adapter_changes_only_launcher_and_preserves_explicit_ft_cases():
    for name in {str(Path(path).parent) for path in functional.baseline_inventory(ROOT)}:
        path = ROOT / functional.model_config_path(name)
        original = yaml.safe_load(path.read_text())
        effective = functional.m890_model_config(path, name)
        assert original.get("LAUNCHER", "torchrun") in ("torchrun", "ft_launcher")
        expected_launcher = "ft_launcher" if name in functional.FT_LAUNCHER_CASES else "torchrun"
        assert effective.get("LAUNCHER", "torchrun") == expected_launcher
        assert {key: value for key, value in effective.items() if key != "LAUNCHER"} == {
            key: value for key, value in original.items() if key != "LAUNCHER"
        }
    assert len(functional.FT_LAUNCHER_CASES) == 5


@pytest.mark.parametrize("damage", ["missing_primary", "symlink", "invalid_json"])
def test_m890_inventory_requires_a_valid_primary_reference(tmp_path, damage):
    name = "gpt/example"
    path = tmp_path / functional.BASELINES / name
    path.mkdir(parents=True)
    (path / "model_config.yaml").write_text("MODEL_ARGS: {}\n")
    primary = path / "golden_values_dev_m890.json"
    secondary = path / "golden_values_dev_m890_2nd.json"
    secondary.write_text(json.dumps({"lm loss": metric()}))
    if damage == "symlink":
        primary.symlink_to(secondary)
    elif damage == "invalid_json":
        primary.write_text("{}")
    with pytest.raises(ValueError):
        functional.baseline_inventory(tmp_path)


def test_catalogue_real_upstream_intersection(catalogue):
    cases = list(catalogue["functional"].values())
    selected = {row["name"] for row in cases}
    baseline_names = {str(Path(path).parent) for path in functional.baseline_inventory(ROOT)}
    omitted = {row["name"] for row in catalogue["coverage"]["baseline_not_selected"]}
    assert cases and len(selected) == len(cases)
    assert selected.isdisjoint(omitted) and selected | omitted == baseline_names
    assert all(row["environment"] == "dev" and row["cards"] == 8 for row in cases)
    assert any(not row["observation"] for row in cases)
    assert {row["observation"] for row in cases if row["observation"]} == {"lightweight"}
    assert all(row["repeat"] == 1 for row in cases)
    assert not (set(functional.SKIPPED_CASES) & {row["name"] for row in cases})
    assert all(row["second_golden_role"] == "provenance_only_not_resume_reference" for row in cases)
    assert any(row["test_type"] == "regular" and "resume" in row["name"] for row in cases)
    assert any(row["launcher"] == "ft_launcher" for row in cases)
    assert any(row["training_script"].endswith("pretrain_hybrid_flex.py") for row in cases)


def test_skip_policy_only_removes_isolated_cases_without_changing_other_cases(monkeypatch):
    policy = {
        "gpt/gpt3_mcore_te_tp2_pp1_modelopt_distill_resume": "teacher_checkpoint_unavailable",
        "gpt/gpt3_weekly_mcore_tp4_cp2_current_scaling_native_fp8_tp_sp_cp_tp_overlap": (
            "weekly_fp8_numerical_instability_temporarily_skipped"
        ),
    }
    config = functional.select_config({})
    current = functional.catalogue(ROOT, ROOT, config)
    assert functional.SKIPPED_CASES == policy
    skipped = current["coverage"]["skipped_by_policy"]
    assert {row["name"]: row["reason"] for row in skipped} == policy
    assert {
        row["name"]: row["reason"]
        for row in current["coverage"]["baseline_not_selected"]
        if row["name"] in policy
    } == policy
    monkeypatch.setattr(functional, "SKIPPED_CASES", {})
    original = functional.catalogue(ROOT, ROOT, config)
    expected = {
        row["name"]: {key: value for key, value in row.items() if key != "id"}
        for row in original["functional"].values()
        if row["name"] not in policy
    }
    actual = {
        row["name"]: {key: value for key, value in row.items() if key != "id"}
        for row in current["functional"].values()
    }
    assert actual == expected
    assert len(original["functional"]) == len(current["functional"]) + len(policy)
    assert current["tokenizer_lock_sha256"] == original["tokenizer_lock_sha256"]


@pytest.mark.parametrize("mode", ["assets", "regression"])
@pytest.mark.parametrize("include_runnable", [True, False])
@pytest.mark.parametrize("skipped_name,reason", list(functional.SKIPPED_CASES.items()))
def test_explicit_policy_selection_is_skipped_not_silently_green(
    mode, include_runnable, skipped_name, reason
):
    runnable = "gpt/gpt3_mcore_te_tp2_pp2_cp2"
    config = functional.select_config(
        {
            "functional_mode": mode,
            "functional_cases": f"{skipped_name},{runnable}" if include_runnable else skipped_name,
        }
    )
    if not include_runnable:
        with pytest.raises(ValueError, match=f"explicitly skipped.*{reason}"):
            functional.catalogue(ROOT, ROOT, config)
        return
    result = functional.catalogue(ROOT, ROOT, config)
    assert [row["name"] for row in result["functional"].values()] == [runnable]
    assert [row["name"] for row in result["coverage"]["skipped_by_policy"]] == [skipped_name]


@pytest.mark.parametrize("tier", functional.TIERS)
def test_tiers_are_exact_not_cumulative(tier, catalogue):
    expected = {row["name"] for row in catalogue["functional"].values() if tier in row["scopes"]}
    if not expected:
        skipped = [
            row for row in catalogue["coverage"]["skipped_by_policy"] if tier in row["scopes"]
        ]
        with pytest.raises(ValueError, match="explicitly skipped" if skipped else "1..256"):
            functional.catalogue(ROOT, ROOT, functional.select_config({"functional_scope": tier}))
    else:
        result = functional.catalogue(
            ROOT, ROOT, functional.select_config({"functional_scope": tier})
        )
        assert {row["name"] for row in result["functional"].values()} == expected


@pytest.mark.parametrize(
    "requested,match",
    [
        ("gpt/does_not_exist", "Unknown"),
        ("multimodal-llava/multimodal_llava_mcore_te_tp1_pp1", "outside active"),
        (
            "moe/gpt3_mcore_te_tp2_zp_z3_resume_torch_dist_te_8experts2parallel_top2router",
            "outside active",
        ),
    ],
)
def test_request_unknown_or_disabled_fails(requested, match):
    with pytest.raises(ValueError, match=match):
        functional.catalogue(ROOT, ROOT, functional.select_config({"functional_cases": requested}))


def test_request_without_golden_is_not_silently_green():
    with pytest.raises(ValueError, match="1..256"):
        functional.catalogue(
            ROOT,
            ROOT,
            functional.select_config({"functional_cases": "gpt/gpt3_mcore_te_mfsdp_v2_cp2"}),
        )


def test_conflicting_recipe_fails(monkeypatch):
    rows = functional.load_recipes(ROOT)
    row = next(row for row in rows if row.get("model") == "bert" and row["scope"] == "L2")
    conflicting = {**row, "nodes": 2}
    monkeypatch.setattr(functional, "load_recipes", lambda _: [*rows, conflicting])
    with pytest.raises(ValueError, match="Conflicting recipes"):
        functional.catalogue(ROOT, ROOT, functional.select_config({}))


@pytest.mark.parametrize("damage", ["none", "changed_after_plan", "missing", "symlink"])
@pytest.mark.parametrize("source_kind", ["recipe", "config", "golden"])
def test_target_revision_is_planned_then_frozen_in_manifest(tmp_path, damage, source_kind):
    config = functional.select_config(
        {
            "functional_mode": "regression",
            "functional_cases": (
                "bert/bert_mcore_tp1_pp2,"
                "gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files"
            ),
        }
    )
    identity = {"target_sha": "a" * 40, "control_sha": "b" * 40, "run_id": "123", "run_attempt": 1}
    original = functional.build_manifest(ROOT, ROOT, config, identity)
    original_case = next(iter(original["functional"].values()))
    source = tmp_path / "source"
    files = set(original["source_files"]) | {
        case["training_script"] for case in original["functional"].values()
    }
    files.update(
        (functional.BASELINES / path).as_posix() for path in functional.baseline_inventory(ROOT)
    )
    for name in files:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    changed_name = original_case[f"{source_kind}_path"]
    changed = source / changed_name
    if source_kind == "golden":
        golden = json.loads(changed.read_text())
        metric_name = original_case["metrics"][0]
        step = next(iter(golden[metric_name]["values"]))
        golden[metric_name]["values"][step] += 0.01
        changed.write_text(json.dumps(golden))
    else:
        changed.write_text("# Target revision differs from control.\n" + changed.read_text())
    expected_digest = functional.sha256(changed)
    if damage in ("missing", "symlink"):
        changed.unlink()
        if damage == "symlink":
            changed.symlink_to(ROOT / changed_name)
        with pytest.raises(ValueError):
            functional.build_manifest(source, ROOT, config, identity)
        return
    manifest = functional.build_manifest(source, ROOT, config, identity)
    case = next(iter(manifest["functional"].values()))
    assert case == {**original_case, f"{source_kind}_sha256": expected_digest}
    if source_kind == "recipe":
        assert manifest["source_files"][changed_name] == expected_digest
        assert manifest["control_files"][changed_name] == original_case["recipe_sha256"]
    assert expected_digest != original_case[f"{source_kind}_sha256"]
    results = functional.module_from_path(
        "ppu_recipe_results", SCRIPTS / "ppu_functional_results.py"
    )
    results.verify_source(manifest, source, case)
    if damage == "changed_after_plan":
        changed.write_text(changed.read_text() + "\n")
        with pytest.raises(ValueError, match="Functional source digest mismatch"):
            results.verify_source(manifest, source, case)


def test_assets_prefixes_and_checkpoint(catalogue):
    assets = {
        asset["relative_path"]
        for row in catalogue["functional"].values()
        for asset in row["assets"]
        if not asset.get("tokenizer_cache")
    }
    assert len(assets) == 10
    assert "text/common_pile/v01_filtered_data/my-gpt3_00_text_document.bin" in assets
    assert "text/common_pile/v01_filtered_data/my-gpt3_00_text_document.idx" in assets
    assert "model/gpt_dummy_pyt/ckpt/24.10.0_bf16_teacher" not in assets
    assert not any("distill_config" in path for path in assets)
    teacher_config = yaml.safe_load(
        (
            ROOT
            / functional.BASELINES
            / "gpt/gpt3_mcore_te_tp2_pp1_modelopt_distill_resume/model_config.yaml"
        ).read_text()
    )
    assert "model/gpt_dummy_pyt/ckpt/24.10.0_bf16_teacher" in {
        row["relative_path"] for row in functional.input_assets(teacher_config)
    }


def test_tokenizer_source_is_parsed_without_execution(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text(
        "raise RuntimeError('must not execute')\nrun(args_defaults={'tokenizer_type': 'Example'})\n"
    )
    source = tmp_path / "megatron/core/tokenizers/text/libraries/megatron_hf_tokenizer.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "raise RuntimeError('must not execute')\nMEGATRON_CONFIG_MAP = {'Example': {'tokenizer_name': 'original-repo'}}\n"
    )
    config = {"TEST_TYPE": "regular", "MODEL_ARGS": {}}
    assert (
        functional.tokenizer_dependencies(tmp_path, config, "entry.py")[0]["repository"]
        == "original-repo"
    )
    entry.write_text(
        "run(args_defaults={'tokenizer_type': 'Example'})\nrun(args_defaults={'tokenizer_type': 'Conflicting'})\n"
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        functional.tokenizer_dependencies(tmp_path, config, "entry.py")


def test_assets_mock_data_does_not_require_data():
    config = {
        "MODEL_ARGS": {
            "--mock-data": True,
            "--data-path": "${DATA_PATH}/unused",
            "--vocab-file": "${DATA_PATH}/vocab.txt",
        }
    }
    assert [row["relative_path"] for row in functional.input_assets(config)] == ["vocab.txt"]


@pytest.mark.parametrize(
    "argument,value",
    [
        ("--vocab-file", "/somewhere/vocab"),
        ("--vocab-file", "${UNKNOWN}/vocab"),
        ("--vocab-file", "${DATA_PATH}/../escape"),
        ("--vocab-file", "$(touch /tmp/file)"),
        ("--unknown-file", "${DATA_PATH}/something"),
    ],
)
def test_ambiguous_or_unsafe_assets_fail(argument, value):
    with pytest.raises(ValueError):
        functional.input_assets({"MODEL_ARGS": {argument: value}})


def test_asset_check_is_read_only_and_distinguishes_layout(tmp_path):
    (tmp_path / "dataset.bin").write_bytes(b"data")
    requirement = {"relative_path": "dataset.idx", "kind": "file", "purposes": ["index"]}
    result = functional.check_asset(tmp_path, requirement)
    assert not result["ok"] and not result["exists"]
    assert not (tmp_path / "dataset.idx").exists()
    (tmp_path / "dataset.idx").write_bytes(b"index")
    assert functional.check_asset(tmp_path, requirement)["ok"]
    (tmp_path / "dataset.idx").write_bytes(b"")
    assert not functional.check_asset(tmp_path, requirement)["ok"]


def test_asset_symlink_escape_fails(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").write_text("private")
    (root / "vocab").symlink_to(tmp_path / "outside")
    assert not functional.check_asset(root, {"relative_path": "vocab", "kind": "file"})["ok"]


def test_asset_report_cpu_unreachable_does_not_assert_ppu_unreachable(tmp_path, catalogue):
    manifest = {**catalogue, "identity": {}, "config": functional.select_config({})}
    report = functional.asset_report(manifest, tmp_path / "not-mounted")
    assert report["result"] == "cpu_preflight_unreachable"
    assert report["pod_data_path"] == functional.POD_ASSETS
    assert report["numerical_validation"] is False


@pytest.mark.parametrize("location", ["cpu", "pod"])
def test_explicit_assets_cannot_hide_unmapped_tokenizer_cache(tmp_path, location, monkeypatch):
    monkeypatch.setattr(functional.hf, "load_sources", lambda control: {})
    catalogue = functional.catalogue(ROOT, ROOT, functional.select_config({}))
    manifest = {**catalogue, "identity": {}, "config": functional.select_config({})}
    monkeypatch.setenv("HF_HOME", str(tmp_path / "unapproved-inherited-cache"))
    for row in catalogue["functional"].values():
        for requirement in row["assets"]:
            path = tmp_path / requirement["relative_path"]
            if requirement["kind"] == "directory":
                path.mkdir(parents=True, exist_ok=True)
                (path / "checkpoint").write_bytes(b"checkpoint")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
    before = {str(path.relative_to(tmp_path)): path.stat().st_mode for path in tmp_path.rglob("*")}
    report = functional.asset_report(manifest, tmp_path, location=location)
    assert not report["ok"] and report["result"] == "asset_source_mapping_required"
    assert all(asset["ok"] for row in report["cases"] for asset in row["assets"])
    unresolved = [asset for row in report["cases"] for asset in row["unresolved_dependencies"]]
    assert {row["repository"] for row in unresolved} == {
        "gpt2",
        "bert-large-uncased",
        "bert-large-cased",
    }
    assert all(row["checked_path"] is None and row["exists"] is None for row in unresolved)
    assert before == {
        str(path.relative_to(tmp_path)): path.stat().st_mode for path in tmp_path.rglob("*")
    }
    assert report["numerical_validation"] is False


@pytest.mark.parametrize("missing_other_asset", [False, True])
def test_skipped_teacher_is_reported_without_weakening_asset_gate(
    tmp_path, catalogue, monkeypatch, missing_other_asset
):
    checked = []
    missing = "text/common_pile/v01_filtered_data/my-gpt3_00_text_document.bin"

    def check(root, requirement):
        relative = requirement["relative_path"]
        checked.append(relative)
        ok = not (missing_other_asset and relative == missing)
        return {
            **requirement,
            "checked_path": str(root / relative),
            "exists": ok,
            "readable": ok,
            "size": 1 if ok else None,
            "ok": ok,
            "reason": "" if ok else "missing",
        }

    monkeypatch.setattr(functional, "check_asset", check)
    summary = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    manifest = {**catalogue, "identity": {}, "config": functional.select_config({})}
    report = functional.asset_report(manifest, tmp_path)
    assert len(report["cases"]) == len(catalogue["functional"])
    assert report["ok"] is not missing_other_asset
    assert report["result"] == ("asset_layout_error" if missing_other_asset else "assets_ready")
    assert not report["numerical_validation"]
    assert missing in checked
    assert "model/gpt_dummy_pyt/ckpt/24.10.0_bf16_teacher" not in checked
    functional.write_asset_report(report, tmp_path)
    saved = json.loads((tmp_path / "assets-report.json").read_text())
    assert saved["coverage"]["skipped_by_policy"] == catalogue["coverage"]["skipped_by_policy"]
    assert "Skipped by policy:" in summary.read_text()
    assert "teacher_checkpoint_unavailable" in summary.read_text()
    assert "weekly_fp8_numerical_instability_temporarily_skipped" in summary.read_text()
    assert "not executed or counted as passed" in summary.read_text()


def test_asset_report_success_is_not_numeric_success(tmp_path):
    config = functional.select_config({"functional_cases": "gpt/gpt3_mcore_te_tp2_pp2_cp2"})
    manifest = {**functional.catalogue(ROOT, ROOT, config), "identity": {}, "config": config}
    for row in manifest["functional"].values():
        assert not row["tokenizer_dependencies"]
        for asset in row["assets"]:
            path = tmp_path / asset["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
    report = functional.asset_report(manifest, tmp_path)
    assert report["ok"] and report["result"] == "assets_ready"
    assert report["numerical_validation"] is False


@pytest.mark.parametrize("text", ['{}', '{"lm loss": {}}', '{"lm loss": 1}', '{"a": 1, "a": 2}'])
def test_corrupt_golden_is_a_hard_error(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        functional.read_metrics(path)


def test_missing_golden_sampling_point_fails(tmp_path):
    with pytest.raises(ValueError, match="sampling grid"):
        read_metric(tmp_path, {"1": 1.0, "3": 1.0, "4": 1.0})


@pytest.mark.parametrize("bad", ["nan", "inf", float("nan"), float("inf"), None])
def test_missing_and_nonfinite_actual_are_hard_errors(tmp_path, bad):
    reference = read_metric(tmp_path)
    actual = copy.deepcopy(reference)
    actual["lm loss"]["values"][3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        functional.validate_region(reference, actual, ["lm loss"])


def test_route_import_does_not_require_yaml_on_hosted_authorization_runner():
    import subprocess
    import sys

    code = "import runpy; route = runpy.run_path(__import__('sys').argv[1]); assert route['validate_route']('workflow_dispatch', route['BRANCH'], {'test_kind':'functional'}) == 'functional'"
    run = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(SCRIPTS / "ppu_functional.py")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 0, run.stderr
