"""CPU-only coverage for functional selection, assets and upstream numeric parity."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

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


@pytest.mark.parametrize("async_save", [True, False, "true", "false"])
@pytest.mark.parametrize("strategy", [None, "nvrx", "mcore"])
@pytest.mark.parametrize("persistent", [True, False])
def test_checkpoint_dependencies_follow_original_flags(async_save, strategy, persistent):
    arguments = {"--async-save": async_save, "--use-persistent-ckpt-worker": persistent}
    if strategy is not None:
        arguments["--async-strategy"] = strategy
    config = {"TEST_TYPE": "regular", "MODEL_ARGS": arguments}
    before = copy.deepcopy(config)
    result = functional.nvrx_checkpoint_requirements(config)
    expected = async_save in (True, "true") and strategy in (None, "nvrx")
    assert result == (
        [{"section": "MODEL_ARGS", "persistent_worker": persistent, "cpu_shm": False}]
        if expected
        else []
    )
    assert config == before


@pytest.mark.parametrize("test_type", ["regular", "ckpt-resume"])
@pytest.mark.parametrize("first_async", [True, False])
@pytest.mark.parametrize("second_async", [True, False])
def test_checkpoint_dependencies_cover_reshard_without_merging(
    test_type, first_async, second_async
):
    config = {
        "TEST_TYPE": test_type,
        "MODEL_ARGS": {"--async-save": first_async, "--use-persistent-ckpt-worker": True},
        "MODEL_ARGS_2": {"--async-save": second_async, "--async-ckpt-use-cpu-shm": True},
    }
    result = functional.nvrx_checkpoint_requirements(config)
    expected = []
    if first_async:
        expected.append({"section": "MODEL_ARGS", "persistent_worker": True, "cpu_shm": False})
    if test_type == "ckpt-resume" and second_async:
        expected.append({"section": "MODEL_ARGS_2", "persistent_worker": False, "cpu_shm": True})
    assert result == expected


@pytest.mark.parametrize("arguments", [None, [], {"--async-strategy": "unknown"}])
def test_invalid_checkpoint_dependency_config_is_rejected(arguments):
    with pytest.raises(ValueError):
        functional.nvrx_checkpoint_requirements(
            {"TEST_TYPE": "ckpt-resume", "MODEL_ARGS": {}, "MODEL_ARGS_2": arguments}
        )


def test_real_torchrun_case_records_async_dependency_without_changing_baseline(catalogue):
    case = next(
        row
        for row in catalogue["functional"].values()
        if row["name"] == "gpt/gpt3_mcore_te_tp2_pp2_cp2"
    )
    assert case["launcher"] == "torchrun"
    assert case["nvrx_checkpoint"] == [
        {"section": "MODEL_ARGS", "persistent_worker": True, "cpu_shm": False}
    ]
    assert case["train_iters"] == 50 and case["repeat"] == 5
    for row in catalogue["functional"].values():
        path = ROOT / row["config_path"]
        assert functional.sha256(path) == row["config_sha256"]
        assert row["nvrx_checkpoint"] == functional.nvrx_checkpoint_requirements(
            yaml.safe_load(path.read_text())
        )


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


@pytest.mark.parametrize(
    "entrypoint,explicit,repository",
    [
        ("examples/bert/pretrain_bert.py", None, "bert-large-uncased"),
        ("pretrain_gpt.py", None, "gpt2"),
        ("pretrain_hybrid.py", None, "gpt2"),
        ("examples/t5/pretrain_t5.py", "BertWordPieceCase", "bert-large-cased"),
        ("pretrain_gpt.py", "NullTokenizer", None),
    ],
)
def test_tokenizer_dependencies_follow_original_entrypoint(entrypoint, explicit, repository):
    arguments = {"--mock-data": True, "--vocab-file": "${DATA_PATH}/vocab.txt"}
    if explicit:
        arguments["--tokenizer-type"] = explicit
    config = {"TEST_TYPE": "regular", "MODEL_ARGS": arguments}
    before = copy.deepcopy(config)
    dependencies = functional.tokenizer_dependencies(ROOT, config, entrypoint)
    assert [row["repository"] for row in dependencies] == ([repository] if repository else [])
    assert all(row["status"] == "source_mapping_required" for row in dependencies)
    assert config == before


@pytest.mark.parametrize("test_type", ["regular", "ckpt-resume"])
@pytest.mark.parametrize("first", ["NullTokenizer", "BertWordPieceCase"])
def test_tokenizer_reshard_uses_replacement_args_and_original_default(test_type, first):
    config = {"TEST_TYPE": test_type, "MODEL_ARGS": {"--tokenizer-type": first}, "MODEL_ARGS_2": {}}
    result = functional.tokenizer_dependencies(ROOT, config, "pretrain_gpt.py")
    expected = [] if first == "NullTokenizer" else [("MODEL_ARGS", "bert-large-cased")]
    if test_type == "ckpt-resume":
        expected.append(("MODEL_ARGS_2", "gpt2"))
    assert [(row["section"], row["repository"]) for row in result] == expected


def test_unknown_tokenizer_assets_are_not_silently_accepted():
    with pytest.raises(ValueError, match="explicit adaptation"):
        functional.tokenizer_dependencies(
            ROOT,
            {"TEST_TYPE": "regular", "MODEL_ARGS": {"--tokenizer-type": "UnknownTokenizer"}},
            "pretrain_gpt.py",
        )


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
def test_explicit_assets_cannot_hide_unmapped_tokenizer_cache(
    tmp_path, catalogue, location, monkeypatch
):
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
