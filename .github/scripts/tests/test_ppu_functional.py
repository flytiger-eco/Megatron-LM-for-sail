"""CPU-only coverage for functional selection, assets and upstream numeric parity."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".github/scripts"
spec = importlib.util.spec_from_file_location("ppu_functional", SCRIPTS / "ppu_functional.py")
functional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functional)
snapshot = functional.module_from_path("snapshot", SCRIPTS / "ppu_functional_snapshot.py")


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


@pytest.mark.parametrize("kind", ["unit", "functional"])
def test_route_defaults(kind):
    assert (
        functional.validate_route("workflow_dispatch", functional.BRANCH, {"test_kind": kind})
        == kind
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("functional_mode", "regression"),
        ("functional_scope", "L0"),
        ("functional_cases", "gpt/example"),
        ("functional_repeat", 2),
        ("functional_timeout", 20),
    ],
)
def test_unit_route_rejects_nondefault_functional_input(key, value):
    with pytest.raises(ValueError, match="require test_kind"):
        functional.validate_route("workflow_dispatch", functional.BRANCH, {key: value})


@pytest.mark.parametrize(
    "event,ref",
    [
        ("push", functional.BRANCH),
        ("pull_request_target", functional.BRANCH),
        ("workflow_dispatch", "refs/heads/main"),
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


@pytest.mark.parametrize("mode", ["assets", "regression"])
def test_functional_config_does_not_inherit_unit_gate(mode):
    config = functional.select_config({"functional_mode": mode})
    assert config["build"] is (mode == "regression")
    assert config["unit_test"] is False
    assert config["repeat"] == 5
    assert config["timeout"] == 30
    assert config["threshold_percent"] == 100
    assert config["data_path"] == ("/mnt/wl_nas/anyunge.ayg/megatron-functional-ci/artifacts-base")
    assert config["cpu_data_path"] == ("/wl_nas/anyunge.ayg/megatron-functional-ci/artifacts-base")
    assert config["data_path"] == "/mnt" + config["cpu_data_path"]


def test_imported_snapshot_matches_source_bytes():
    files = snapshot.verify(ROOT, ROOT)
    assert len(files) == 224
    assert sum(Path(name).name == "model_config.yaml" for name in files) == 93
    assert sum(Path(name).name == "golden_values_dev_ppu.json" for name in files) == 93
    assert sum(Path(name).name == "golden_values_dev_ppu_2nd.json" for name in files) == 38
    assert not any(".DS_Store" in name for name in files)


def test_catalogue_real_upstream_intersection(catalogue):
    cases = list(catalogue["functional"].values())
    assert len(cases) == 85
    assert len({row["name"] for row in cases}) == 85
    assert all(row["environment"] == "dev" and row["cards"] == 8 for row in cases)
    assert len(catalogue["coverage"]["baseline_not_selected"]) == 8
    assert {row["observation"] for row in cases if row["observation"]} == {
        "lightweight",
        "SKIP_PYTEST",
    }
    assert all(row["repeat"] == (1 if row["observation"] else 5) for row in cases)
    assert max(row["train_iters"] for row in cases) == 2000
    assert all(row["second_golden_role"] == "provenance_only_not_resume_reference" for row in cases)
    assert any(row["test_type"] == "regular" and "resume" in row["name"] for row in cases)
    assert any(row["launcher"] == "ft_launcher" for row in cases)
    assert any(row["training_script"].endswith("pretrain_hybrid_flex.py") for row in cases)


@pytest.mark.parametrize("tier", functional.TIERS)
def test_tiers_are_exact_not_cumulative(tier, catalogue):
    expected = {row["name"] for row in catalogue["functional"].values() if tier in row["scopes"]}
    if not expected:
        with pytest.raises(ValueError, match="1..256"):
            functional.catalogue(ROOT, ROOT, functional.select_config({"functional_scope": tier}))
    else:
        result = functional.catalogue(
            ROOT, ROOT, functional.select_config({"functional_scope": tier})
        )
        assert {row["name"] for row in result["functional"].values()} == expected


def test_alias_and_cadence_are_upstream_owned():
    parser = functional.module_from_path(
        "recipe", ROOT / "tests/test_utils/python_scripts/recipe_parser.py"
    )
    recipe = parser.dotdict(
        products=[
            {
                "test_case": ["a"],
                "products": [
                    {
                        "scope": ["mr-github", "nightly"],
                        "environment": ["dev"],
                        "platforms": ["dgx_h100"],
                    }
                ],
            }
        ]
    )
    rows = parser.flatten_products(recipe).products
    assert {row["scope"] for row in rows} == {"L1", "L2"}
    assert next(row for row in rows if row["scope"] == "L2")["cadence"] == ["nightly"]


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


def test_recipe_entrypoint_is_not_guessed(catalogue):
    case = next(row for row in catalogue["functional"].values() if row["model"] == "bert")
    spec = next(row for row in functional.load_recipes(ROOT) if row["test_case"] == case["case"])
    spec["script"] = 'TRAINING_SCRIPT_PATH=pretrain_gpt.py'
    with pytest.raises(ValueError, match="entrypoint"):
        functional.describe_case(ROOT, case["name"], [spec], functional.select_config({}))


def test_assets_prefixes_and_checkpoint(catalogue):
    assets = {
        asset["relative_path"]
        for row in catalogue["functional"].values()
        for asset in row["assets"]
    }
    assert len(assets) == 11
    assert "text/common_pile/v01_filtered_data/my-gpt3_00_text_document.bin" in assets
    assert "text/common_pile/v01_filtered_data/my-gpt3_00_text_document.idx" in assets
    assert "model/gpt_dummy_pyt/ckpt/24.10.0_bf16_teacher" in assets
    assert not any("distill_config" in path for path in assets)


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


def test_asset_report_success_is_not_numeric_success(tmp_path, catalogue):
    manifest = {**catalogue, "identity": {}, "config": functional.select_config({})}
    for row in catalogue["functional"].values():
        for requirement in row["assets"]:
            path = tmp_path / requirement["relative_path"]
            if requirement["kind"] == "directory":
                path.mkdir(parents=True, exist_ok=True)
                (path / "checkpoint").write_bytes(b"checkpoint")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
    report = functional.asset_report(manifest, tmp_path)
    assert report["ok"] and report["result"] == "assets_ready"
    assert report["numerical_validation"] is False


@pytest.mark.parametrize("precision", [None, "full", "rounded_5_decimal_places"])
def test_metric_precision_is_preserved(tmp_path, precision):
    result = read_metric(tmp_path, precision=precision)
    assert result["lm loss"]["value_precision"] == (precision or "rounded_5_decimal_places")


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


def test_iteration_warmup_nan_is_not_a_failure():
    values = {step: "nan" if step < 5 else 2.0 for step in range(1, 21)}
    result = functional.validate_region(
        {"iteration-time": {"values": values}},
        {"iteration-time": {"values": values}},
        ["iteration-time"],
    )
    assert result["iteration-time"] == list(range(5, 21))
    actual = {"iteration-time": {"values": {**values, 10: "nan"}}}
    with pytest.raises(ValueError, match="non-finite"):
        functional.validate_region(
            {"iteration-time": {"values": values}}, actual, ["iteration-time"]
        )


def test_resume_nonexecuted_half_is_excluded(tmp_path):
    reference = read_metric(tmp_path)
    actual = copy.deepcopy(reference)
    actual["lm loss"]["values"].update({1: "nan", 2: "nan"})
    assert functional.validate_region(reference, actual, ["lm loss"], after=2)["lm loss"] == [3, 4]
    actual["lm loss"]["values"][3] = "nan"
    with pytest.raises(ValueError):
        functional.validate_region(reference, actual, ["lm loss"], after=2)


@pytest.mark.parametrize(
    "precision,passes", [(None, True), ("full", False), ("rounded_5_decimal_places", True)]
)
def test_direct_upstream_precision_parity(tmp_path, upstream, precision, passes):
    common, regular, _ = upstream
    reference = common.GoldenValueMetric(**metric(precision=precision))
    actual = reference.model_copy(deep=True)
    actual.values = {key: value + 0.000001 for key, value in actual.values.items()}
    config = tmp_path / "config.yaml"
    config.write_text("METRICS: ['lm loss']\n")
    if passes:
        regular.test_regular_pipeline(
            False, {"lm loss": reference}, {"lm loss": actual}, str(config)
        )
    else:
        with pytest.raises(AssertionError):
            regular.test_regular_pipeline(
                False, {"lm loss": reference}, {"lm loss": actual}, str(config)
            )


@pytest.mark.parametrize("bad_points,passes", [(1, True), (2, False)])
def test_direct_upstream_approximate_outlier_rule(tmp_path, upstream, bad_points, passes):
    common, regular, _ = upstream
    golden = common.GoldenValueMetric(**metric(end=100))
    actual = golden.model_copy(deep=True)
    for step in range(1, bad_points + 1):
        actual.values[step] = 10.0
    config = tmp_path / "config.yaml"
    config.write_text("METRICS: ['lm loss']\n")
    if passes:
        regular.test_regular_pipeline(True, {"lm loss": golden}, {"lm loss": actual}, str(config))
    else:
        with pytest.raises(AssertionError):
            regular.test_regular_pipeline(
                True, {"lm loss": golden}, {"lm loss": actual}, str(config)
            )


def test_direct_upstream_resume_compares_current_first_run(tmp_path, upstream):
    common, _, resume = upstream
    first = common.GoldenValueMetric(**metric())
    second = common.GoldenValueMetric(**metric({"1": "nan", "2": "nan", "3": 1.0, "4": 1.0}))
    config = tmp_path / "config.yaml"
    config.write_text("METRICS: ['lm loss']\n")
    resume.test_resume_checkpoint_pipeline(
        False,
        {"lm loss": first.model_copy(deep=True)},
        {"lm loss": second.model_copy(deep=True)},
        4,
        str(config),
    )
    second.values[4] = 2.0
    with pytest.raises(AssertionError):
        resume.test_resume_checkpoint_pipeline(
            False, {"lm loss": first}, {"lm loss": second}, 4, str(config)
        )


def test_synthetic_tensorboard_uses_upstream_extractor(tmp_path, upstream):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    common, _, _ = upstream
    writer = EventFileWriter(str(tmp_path))
    for step in range(1, 5):
        writer.add_event(
            Event(
                wall_time=float(step),
                step=step,
                summary=Summary(value=[Summary.Value(tag="lm loss", simple_value=1.0)]),
            )
        )
    writer.flush()
    writer.close()
    result = common.read_tb_logs_as_list(str(tmp_path), train_iters=4, step_size=1)
    assert list(result["lm loss"].values) == [1, 2, 3, 4]
    assert all(math.isfinite(value) for value in result["lm loss"].values.values())
    assert result["lm loss"].value_precision == common.ValuePrecision.FULL


@pytest.mark.parametrize("permission", ["write", "maintain", "admin", "read"])
def test_real_authorizer_preserves_maintainer_gate_and_never_publishes_pr(permission):
    from test_ppu_quick import auth, context

    ctx = context()
    ctx["control_sha"] = ctx["sha"]
    calls = []

    def api(path, payload=None):
        calls.append(path)
        assert path.startswith("collaborators/") and payload is None
        return {"permission": permission}

    if permission == "read":
        with pytest.raises(ValueError, match="write/maintain/admin"):
            auth.authorize({}, {"test_kind": "functional"}, ctx, api)
        return
    authorized = auth.authorize({}, {"test_kind": "functional"}, ctx, api)
    assert authorized["allowed"] and authorized["test_kind"] == "functional"
    assert authorized["pr_number"] == 0 and not authorized["manual_pr"]
    before = list(calls)
    assert auth.publish_status(authorized, "success", "not a PR result", api) == "rejected"
    assert calls == before


def test_real_authorizer_rejects_functional_pr_before_reading_pr():
    from test_ppu_quick import auth, context

    def api(*_):
        raise AssertionError("Invalid mixed routing must not reach PR APIs")

    with pytest.raises(ValueError, match="PR status"):
        auth.authorize({}, {"test_kind": "functional", "pr_number": "7"}, context(), api)


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
