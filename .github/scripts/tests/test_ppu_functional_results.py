"""Functional evidence, retry policy and unmodified pytest plugin integration."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_ppu_functional import ROOT, SCRIPTS, functional

results = functional.module_from_path(
    "ppu_functional_results", SCRIPTS / "ppu_functional_results.py"
)


@pytest.mark.parametrize(
    "message",
    [
        "DistNetworkError: connection lost",
        "ConnectionResetError: peer disconnected",
        "connection reset by peer",
        "The server socket has failed to listen on any local network address",
        "EADDRINUSE",
        "NCCL collective operation timed out",
        "Watchdog caught collective operation timeout",
    ],
)
def test_only_explicit_communication_errors_retry(message):
    assert results.classify_failure(1, message) == "infrastructure_retryable"


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: an illegal memory access was encountered",
        "Segmentation fault",
        "SIGSEGV",
        "OutOfMemoryError",
        "CUDA out of memory",
        "loss = nan",
        "gradient norm is inf",
        "AssertionError: mismatch",
        "Exact comparison of lm loss: FAILED",
        "MissingTensorboardLogsError",
        "FileNotFoundError: assets",
        "ModuleNotFoundError",
    ],
)
def test_hard_failure_wins_over_communication_error(message):
    assert results.classify_failure(1, f"DistNetworkError\n{message}") == "hard_failure"


@pytest.mark.parametrize("code", [139, -11])
def test_sigsegv_exit_without_error_text_is_never_retryable(code):
    assert results.classify_failure(code, "DistNetworkError") == "hard_failure"


@pytest.mark.parametrize("code", [130, 137, 143, -2, -9, -15])
def test_cancellation_is_not_retryable(code):
    assert results.classify_failure(code, "DistNetworkError") == "cancelled"


def test_budget_and_missing_report_are_not_retryable():
    assert results.classify_failure(124, "DistNetworkError") == "budget_exhausted"
    assert (
        results.classify_failure(1, "DistNetworkError", integrity_error=True) == "integrity_error"
    )
    assert results.classify_failure(1, "unexpected error") == "execution_failure"
    assert results.classify_failure(0, "finished") == "passed"
    assert results.classify_failure(0, "loss = nan") == "hard_failure"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def evidence(tmp_path):
    config = functional.select_config(
        {
            "functional_mode": "regression",
            "functional_repeat": 2,
            "functional_cases": "bert/bert_mcore_tp1_pp2",
        }
    )
    catalogue = functional.catalogue(ROOT, ROOT, config)
    case = next(iter(catalogue["functional"].values()))
    # 此夹具生成合成训练/比较证据，不实例化 tokenizer；真实缓存门禁另有独立测试。
    case["tokenizer_dependencies"] = []
    manifest = {
        **catalogue,
        "schema_version": 1,
        "test_kind": "functional",
        "config": config,
        "source_files": {},
        "identity": {
            "target_sha": "a" * 40,
            "control_sha": "b" * 40,
            "run_id": "123",
            "run_attempt": 1,
        },
    }
    directory = tmp_path / case["id"]
    directory.mkdir()
    return manifest, case, directory


def successful_repeat(manifest, case, directory, attempt, repeat):
    path = directory / f"attempt-{attempt}" / f"repeat-{repeat}"
    path.mkdir(parents=True)
    config = yaml.safe_load((ROOT / case["config_path"]).read_text())
    config.setdefault("ENV_VARS", {})["N_REPEAT"] = 1
    config["MODEL_ARGS"]["--exit-interval"] = case["effective_train_iters"]
    (path / "model_config.yaml").write_text(yaml.safe_dump(config))
    (path / Path(case["golden_path"]).name).write_bytes((ROOT / case["golden_path"]).read_bytes())
    (path / "driver.log").write_text("Training finished\n")
    for phase in range(1, results.phase_count(case) + 1):
        tensorboard = path / "1/tensorboard"
        if results.phase_count(case) == 2:
            tensorboard /= f"run_{phase}"
        tensorboard.mkdir(parents=True)
        (tensorboard / "events.out.tfevents.fixture").write_text("Mock event evidence")
        barrier = path / "barriers" / f"repeat_1_run_{phase}_status"
        barrier.mkdir(parents=True)
        (barrier / "node_0").write_text("0\n")
        for rank in range(8):
            logs = path / "logs" / f"run-{phase}" / "attempt_0" / str(rank)
            logs.mkdir(parents=True)
            (logs / "stdout.log").write_text("Training\n")
            (logs / "stderr.log").write_text("")
    identity = results.identity_fields(manifest, case, attempt, repeat)
    execution = {
        **identity,
        "duration_seconds": 1.0,
        "exit_code": 0,
        "cancelled": False,
        "timed_out": False,
        "runtime_config_sha256": functional.sha256(path / "model_config.yaml"),
    }
    write_json(path / "execution.json", execution)
    phases = ["regular", "resume"] if results.phase_count(case) == 2 else ["regular"]
    for phase in phases:
        name = "test_regular_pipeline" if phase == "regular" else "test_resume_checkpoint_pipeline"
        node = f"test_pretraining_{phase}_pipeline.py::{name}"
        report = {
            **identity,
            "phase": phase,
            "complete": True,
            "exitstatus": 0,
            "duration_seconds": 0.1,
            "approximate": case["approximate"],
            "collection_errors": [],
            "collected": [node],
            "tests": {node: {"setup": "passed", "call": "passed", "teardown": "passed"}},
        }
        write_json(path / f"pytest-{phase}.json", report)
    return path


def envelope(manifest, case, directory):
    write_json(
        directory / "assets-report.json",
        {
            "identity": manifest["identity"],
            "ok": True,
            "location": "pod",
            "numerical_validation": False,
            "pod_data_path": manifest["config"]["data_path"],
            "root": {"ok": True},
            "cases": [
                {
                    "id": case["id"],
                    "name": case["name"],
                    "assets": [{**asset, "ok": True} for asset in case["assets"]],
                }
            ],
        },
    )
    write_json(
        directory / "functional-environment.json",
        {
            "identity": manifest["identity"],
            "devices": 8,
            "image": manifest["config"]["image"],
            "python": "fixture",
            "packages": {"torch": "fixture"},
            "bash": "GNU Bash",
            "yq": "Mike Farah v4",
            "envsubst": "GNU gettext",
            "uv": "fixture",
        },
    )
    write_json(
        directory / "action-timing.json",
        {
            "action_outcome": "success",
            "action_status": "succeeded",
            "outputs": {
                "duration_seconds": "30",
                "queue_duration_seconds": "1",
                "occupied_ppu_card_hours": "0.01",
                "execution_ppu_card_hours": "0.01",
            },
        },
    )
    write_json(
        directory / "pod-timing.json",
        {"exit_code": 0, "started_at": 1, "completed_at": 21, "elapsed_seconds": 20},
    )


def completion(manifest, case, directory, attempts=1, **overrides):
    report = {
        "identity": manifest["identity"],
        "case_id": case["id"],
        "manifest_sha256": results.manifest_digest(manifest),
        "complete": True,
        "attempts": attempts,
        "duration_seconds": 10,
        "cancelled": False,
        "timed_out": False,
        **overrides,
    }
    write_json(directory / "case-run.json", report)
    envelope(manifest, case, directory)


def test_all_repeats_are_required_and_numerically_recomputed(evidence):
    manifest, case, directory = evidence
    successful_repeat(manifest, case, directory, 1, 1)
    completion(manifest, case, directory)
    result = results.verify_case(manifest, ROOT, case, directory)
    assert not result["success"]
    assert "required successful repetitions" in result["errors"][0]
    successful_repeat(manifest, case, directory, 1, 2)
    assert results.verify_case(manifest, ROOT, case, directory)["success"]
    path = directory / "attempt-1/repeat-2" / Path(case["golden_path"]).name
    actual = json.loads(path.read_text())
    actual["lm loss"]["values"]["10"] = 999
    write_json(path, actual)
    assert not results.verify_case(manifest, ROOT, case, directory)["success"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("exitstatus", 1),
        ("collected", []),
        ("approximate", True),
        ("manifest_sha256", "a" * 64),
        ("repeat", 7),
        ("collection_errors", ["error"]),
    ],
)
def test_invalid_pytest_evidence_fails(evidence, field, value):
    manifest, case, directory = evidence
    path = successful_repeat(manifest, case, directory, 1, 1)
    report_path = path / "pytest-regular.json"
    report = json.loads(report_path.read_text())
    report[field] = value
    write_json(report_path, report)
    with pytest.raises(ValueError):
        results.verify_repeat(manifest, ROOT, case, path, 1, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cancelled", True),
        ("timed_out", True),
        ("attempts", 4),
        ("duration_seconds", 100000),
        ("complete", False),
        ("manifest_sha256", "bad"),
    ],
)
def test_case_completion_fails_closed(evidence, field, value):
    manifest, case, directory = evidence
    for repeat in (1, 2):
        successful_repeat(manifest, case, directory, 1, repeat)
    completion(manifest, case, directory, **{field: value})
    assert not results.verify_case(manifest, ROOT, case, directory)["success"]


def failed_repeat(manifest, case, directory, attempt=1, message="DistNetworkError", code=1):
    path = successful_repeat(manifest, case, directory, attempt, 1)
    execution = json.loads((path / "execution.json").read_text())
    execution["exit_code"] = code
    write_json(path / "execution.json", execution)
    (path / "driver.log").write_text(message)
    (path / "barriers/repeat_1_run_1_status/node_0").write_text("1\n")
    return path


def test_forged_ready_report_cannot_bypass_unmapped_tokenizer(evidence):
    manifest, case, directory = evidence
    case["tokenizer_dependencies"] = functional.tokenizer_dependencies(
        ROOT, yaml.safe_load((ROOT / case["config_path"]).read_text()), case["training_script"]
    )
    for repeat in (1, 2):
        successful_repeat(manifest, case, directory, 1, repeat)
    completion(manifest, case, directory)
    result = results.verify_case(manifest, ROOT, case, directory)
    assert not result["success"]
    assert any("Unresolved tokenizer" in error for error in result["errors"])


def test_training_sigsegv_status_wins_over_generic_shell_exit(evidence):
    manifest, case, directory = evidence
    path = failed_repeat(manifest, case, directory)
    (path / "barriers/repeat_1_run_1_status/node_0").write_text("139\n")
    result = results.verify_repeat(manifest, ROOT, case, path, 1, 1)
    assert result["classification"] == "hard_failure"


def test_retry_requires_entire_new_successful_attempt(evidence):
    manifest, case, directory = evidence
    failed_repeat(manifest, case, directory)
    successful_repeat(manifest, case, directory, 2, 1)
    completion(manifest, case, directory, attempts=2)
    assert not results.verify_case(manifest, ROOT, case, directory)["success"]
    successful_repeat(manifest, case, directory, 2, 2)
    result = results.verify_case(manifest, ROOT, case, directory)
    assert result["success"] and result["retried"]
    assert len(result["attempts"]) == 2


@pytest.mark.parametrize(
    "message",
    [
        "AssertionError",
        "OutOfMemoryError",
        "CUDA error: illegal memory access",
        "MissingTensorboardLogsError",
        "OSError: AF_UNIX path too long",
    ],
)
def test_unauthorized_retry_is_rejected_even_after_success(evidence, message):
    manifest, case, directory = evidence
    failed_repeat(manifest, case, directory, message=f"DistNetworkError\n{message}")
    for repeat in (1, 2):
        successful_repeat(manifest, case, directory, 2, repeat)
    completion(manifest, case, directory, attempts=2)
    result = results.verify_case(manifest, ROOT, case, directory)
    assert not result["success"] and "Retried" in result["errors"][0]


def test_runtime_copy_cannot_change_metrics_or_training_length(evidence):
    manifest, case, directory = evidence
    original = functional.sha256(ROOT / case["config_path"])
    path = successful_repeat(manifest, case, directory, 1, 1)
    config = yaml.safe_load((path / "model_config.yaml").read_text())
    config["MODEL_ARGS"]["--train-iters"] = 4
    (path / "model_config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Runtime configuration"):
        results.verify_repeat(manifest, ROOT, case, path, 1, 1)
    assert functional.sha256(ROOT / case["config_path"]) == original


@pytest.mark.parametrize("content", [b"", None])
def test_missing_or_empty_tensorboard_is_not_success(evidence, content):
    manifest, case, directory = evidence
    path = successful_repeat(manifest, case, directory, 1, 1)
    event = path / "1/tensorboard/events.out.tfevents.fixture"
    if content is None:
        event.rename(event.with_name("not-an-event"))
    else:
        event.write_bytes(content)
    with pytest.raises(ValueError, match="TensorBoard"):
        results.verify_repeat(manifest, ROOT, case, path, 1, 1)


@pytest.mark.parametrize(
    "filename",
    ["assets-report.json", "functional-environment.json", "action-timing.json", "pod-timing.json"],
)
def test_missing_envelope_is_not_success(evidence, filename):
    manifest, case, directory = evidence
    for repeat in (1, 2):
        successful_repeat(manifest, case, directory, 1, repeat)
    completion(manifest, case, directory)
    (directory / filename).rename(directory / (filename + ".missing"))
    assert not results.verify_case(manifest, ROOT, case, directory)["success"]


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("assets-report.json", "identity", {}),
        ("assets-report.json", "ok", False),
        ("assets-report.json", "cases", []),
        ("assets-report.json", "location", "cpu"),
        ("assets-report.json", "pod_data_path", "/different"),
        ("functional-environment.json", "identity", {}),
        ("functional-environment.json", "devices", 4),
        ("functional-environment.json", "image", "different"),
        ("functional-environment.json", "packages", {}),
        ("action-timing.json", "action_outcome", "failure"),
        ("action-timing.json", "action_status", "timeout"),
        ("pod-timing.json", "exit_code", 143),
        ("pod-timing.json", "elapsed_seconds", 999),
    ],
)
def test_invalid_envelope_is_not_success(evidence, filename, field, value):
    manifest, case, directory = evidence
    envelope(manifest, case, directory)
    path = directory / filename
    report = json.loads(path.read_text())
    report[field] = value
    write_json(path, report)
    with pytest.raises(ValueError):
        results.verify_envelope(manifest, case, directory)


@pytest.mark.parametrize(
    "field",
    [
        "duration_seconds",
        "queue_duration_seconds",
        "occupied_ppu_card_hours",
        "execution_ppu_card_hours",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-1", ""])
def test_action_timings_must_be_finite_and_nonnegative(evidence, field, value):
    manifest, case, directory = evidence
    envelope(manifest, case, directory)
    path = directory / "action-timing.json"
    report = json.loads(path.read_text())
    report["outputs"][field] = value
    write_json(path, report)
    with pytest.raises(ValueError):
        results.verify_envelope(manifest, case, directory)


def test_queue_cannot_exceed_action_duration(evidence):
    manifest, case, directory = evidence
    envelope(manifest, case, directory)
    path = directory / "action-timing.json"
    report = json.loads(path.read_text())
    report["outputs"]["queue_duration_seconds"] = "31"
    write_json(path, report)
    with pytest.raises(ValueError, match="timing"):
        results.verify_envelope(manifest, case, directory)


def test_summary_cannot_report_assets_as_regression(evidence):
    manifest, _, directory = evidence
    manifest["config"]["functional_mode"] = "assets"
    with pytest.raises(ValueError, match="Asset-only"):
        results.summarize(manifest, ROOT, directory.parent)


@pytest.mark.parametrize(
    "failure,expected",
    [("asset", "asset_error"), ("timeout", "infrastructure_error"), ("cancelled", "cancelled")],
)
def test_failure_report_distinguishes_assets_and_scheduling(evidence, failure, expected):
    manifest, case, directory = evidence
    envelope(manifest, case, directory)
    if failure == "asset":
        path = directory / "assets-report.json"
        report = json.loads(path.read_text())
        report.update(ok=False, result="asset_layout_error")
    else:
        path = directory / "action-timing.json"
        report = json.loads(path.read_text())
        report["action_status"] = failure
    write_json(path, report)
    result = results.verify_case(manifest, ROOT, case, directory)
    assert not result["success"] and result["classification"] == expected
    assert result["queue_seconds"] == 1 and result["budget_seconds"] == case["budget_seconds"]
    summary = results.summarize(manifest, ROOT, directory.parent)
    assert summary["failure_categories"][expected] == [case["name"]]


def test_malformed_report_does_not_hide_other_case_reports(evidence):
    manifest, case, directory = evidence
    write_json(directory / "case-run.json", [])
    assert not results.summarize(manifest, ROOT, directory.parent)["success"]


def test_summary_rejects_missing_case_artifact(evidence):
    manifest, _, directory = evidence
    result = results.summarize(manifest, ROOT, directory.parent)
    assert not result["success"] and not result["numerical_validation"]
    assert result["numeric_failed"] == 1


@pytest.mark.parametrize("bad_numeric", [False, True])
def test_plugin_collects_real_upstream_pytest_without_changing_assertions(
    evidence, tmp_path, bad_numeric
):
    manifest, case, directory = evidence
    path = directory / "plugin-test"
    path.mkdir()
    actual = json.loads((ROOT / case["golden_path"]).read_text())
    if bad_numeric:
        actual["lm loss"]["values"]["10"] = 999
    actual_path = path / "actual.json"
    write_json(actual_path, actual)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(SCRIPTS))),
        "PPU_FUNCTIONAL_IDENTITY": json.dumps(manifest["identity"]),
        "PPU_FUNCTIONAL_CASE_ID": case["id"],
        "PPU_FUNCTIONAL_ATTEMPT": "1",
        "PPU_FUNCTIONAL_REPEAT": "1",
        "PPU_FUNCTIONAL_REPEAT_DIR": str(path),
        "PPU_FUNCTIONAL_MANIFEST_SHA256": results.manifest_digest(manifest),
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "ppu_functional_pytest",
        str(ROOT / "tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py"),
        "--golden-values-path",
        str(ROOT / case["golden_path"]),
        "--actual-values-path",
        str(actual_path),
        "--model-config-path",
        str(ROOT / case["config_path"]),
    ]
    run = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert run.returncode == (1 if bad_numeric else 0), run.stdout + run.stderr
    report = json.loads((path / "pytest-regular.json").read_text())
    assert report["complete"] and report["exitstatus"] == run.returncode
    assert len(report["tests"]) == 1
    assert next(iter(report["tests"].values()))["call"] == ("failed" if bad_numeric else "passed")
