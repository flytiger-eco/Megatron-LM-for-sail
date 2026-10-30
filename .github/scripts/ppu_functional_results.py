"""Fail-closed evidence verification and summaries for PPU functional regression."""

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import yaml

_spec = importlib.util.spec_from_file_location(
    "ppu_functional_control", Path(__file__).with_name("ppu_functional.py")
)
control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(control)

HARD_FAILURES = (
    r"CUDA error:.*(?:illegal|out of memory|misaligned|device-side assert)",
    r"\billegal memory access\b|\bsegmentation fault\b|\bsegfault\b|\bSIGSEGV\b",
    r"\bOutOfMemoryError\b|\bOutOfMemory\b|\bout of memory\b|\bOOM\b",
    r"(?:\b(?:loss|gradient|grad norm)\b[^\n]{0,80}\b(?:nan|inf)\b)",
    r"\b(?:nan|inf)\b[^\n]{0,80}(?:\b(?:loss|gradient|grad norm)\b)",
    r"\bAssertionError\b|\b(?:Exact|Approximate) comparison[^\n]*FAILED",
    r"\bMissingTensorboardLogsError\b|\bUndefinedMetricError\b",
    r"\b(?:FileNotFoundError|ModuleNotFoundError|ImportError)\b",
    r"\bAF_UNIX path too long\b",
)
RETRYABLE_FAILURES = (
    r"\bDistNetworkError\b",
    r"\bConnectionResetError\b|\bconnection reset by peer\b",
    r"(?:server socket|port)[^\n]*(?:failed to listen|address already in use)",
    r"\bEADDRINUSE\b",
    r"(?:NCCL[^\n]*collective[^\n]*tim(?:e|ed)[ -]?out)",
    r"watchdog caught collective operation timeout",
)


def classify_failure(
    exit_code: int, text: str, *, cancelled=False, timed_out=False, integrity_error=False
) -> str:
    if cancelled or exit_code in (130, 137, 143, -2, -9, -15):
        return "cancelled"
    if timed_out or exit_code == 124:
        return "budget_exhausted"
    if integrity_error:
        return "integrity_error"
    if exit_code in (139, -11) or any(
        re.search(pattern, text, re.IGNORECASE) for pattern in HARD_FAILURES
    ):
        return "hard_failure"
    if exit_code == 0:
        return "passed"
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in RETRYABLE_FAILURES):
        return "infrastructure_retryable"
    return "execution_failure"


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def identity_fields(manifest: dict, case: dict, attempt: int, repeat: int) -> dict:
    return {
        "schema_version": 1,
        "identity": manifest["identity"],
        "case_id": case["id"],
        "attempt": attempt,
        "repeat": repeat,
        "manifest_sha256": manifest_digest(manifest),
    }


def verify_identity(report: dict, manifest: dict, case: dict, attempt: int, repeat: int) -> None:
    if any(
        report.get(key) != value
        for key, value in identity_fields(manifest, case, attempt, repeat).items()
    ):
        raise ValueError("Functional evidence identity or manifest digest mismatch")


def finite_seconds(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid functional timing evidence")
    return float(value)


def phase_count(case: dict) -> int:
    return 2 if case["test_type"] == "ckpt-resume" else 1


def verify_source(manifest: dict, root: Path, case: dict) -> None:
    files = {
        **manifest["source_files"],
        case["config_path"]: case["config_sha256"],
        case["golden_path"]: case["golden_sha256"],
        case["training_script"]: case["training_script_sha256"],
        case["recipe_path"]: case["recipe_sha256"],
    }
    if case["second_golden_path"]:
        files[case["second_golden_path"]] = case["second_golden_sha256"]
    for name, expected in files.items():
        if control.sha256(control.safe_file(root, name)) != expected:
            raise ValueError(f"Functional source digest mismatch: {name}")


def verify_runtime_config(root: Path, case: dict, directory: Path) -> None:
    expected = yaml.safe_load((root / case["config_path"]).read_text())
    expected.setdefault("ENV_VARS", {})["N_REPEAT"] = 1
    expected["MODEL_ARGS"]["--exit-interval"] = case["effective_train_iters"]
    if case["observation"] == "lightweight":
        expected["ENV_VARS"]["SKIP_PYTEST"] = 1
        if case["test_type"] == "ckpt-resume":
            expected["MODEL_ARGS"]["--save-interval"] = 2
    actual = yaml.safe_load((directory / "model_config.yaml").read_text())
    if actual != expected:
        raise ValueError("Runtime configuration differs outside upstream isolation adjustments")


def read_training_statuses(directory: Path) -> dict:
    statuses = {}
    for phase in (1, 2):
        path = directory / "barriers" / f"repeat_1_run_{phase}_status" / "node_0"
        if path.exists():
            text = path.read_text().strip()
            if not re.fullmatch(r"[0-9]+", text):
                raise ValueError("Malformed training exit status")
            statuses[phase] = int(text)
    return statuses


def log_text(directory: Path) -> str:
    paths = [directory / "driver.log", *sorted((directory / "logs").rglob("*.log"))]
    return "\n".join(path.read_text(errors="replace") for path in paths if path.is_file())


def verify_pytest(
    directory: Path, phase: str, manifest: dict, case: dict, attempt: int, repeat: int
) -> None:
    report = json.loads((directory / f"pytest-{phase}.json").read_text())
    verify_identity(report, manifest, case, attempt, repeat)
    expected_node = (
        "test_regular_pipeline" if phase == "regular" else "test_resume_checkpoint_pipeline"
    )
    nodes = report.get("collected")
    if (
        report.get("complete") is not True
        or report.get("exitstatus") != 0
        or report.get("collection_errors")
        or report.get("phase") != phase
        or report.get("approximate") != case["approximate"]
        or not isinstance(nodes, list)
        or len(nodes) != 1
        or not nodes[0].endswith(f"::{expected_node}")
        or set(report.get("tests", {})) != set(nodes)
    ):
        raise ValueError(f"Missing, empty or failed upstream pytest evidence: {phase}")
    if report["tests"][nodes[0]] != {"setup": "passed", "call": "passed", "teardown": "passed"}:
        raise ValueError(f"Upstream pytest skipped or failed a test/fixture: {phase}")
    finite_seconds(report["duration_seconds"])


def compare_upstream(root: Path, case: dict, directory: Path) -> None:
    """Recompute with the original comparator, including its original precision policy."""
    sys.path.insert(0, str(root))
    try:
        common = importlib.import_module("tests.functional_tests.python_test_utils.common")
        regular = importlib.import_module(
            "tests.functional_tests.python_test_utils.test_pretraining_regular_pipeline"
        )
        resume = importlib.import_module(
            "tests.functional_tests.python_test_utils.test_pretraining_resume_checkpoint_pipeline"
        )
        for module in (common, regular, resume):
            if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
                raise ValueError("Comparator was imported from a different source tree")
        golden = common.read_golden_values_from_json(root / case["golden_path"])
        actual_path = directory / Path(case["golden_path"]).name
        actual = common.read_golden_values_from_json(actual_path)
        config_path = str(directory / "model_config.yaml")
        regular.test_regular_pipeline(
            case["approximate"], golden, copy.deepcopy(actual), config_path
        )
        if phase_count(case) == 2:
            second_path = actual_path.with_name(f"{actual_path.stem}_2nd.json")
            second = common.read_golden_values_from_json(second_path)
            resume.test_resume_checkpoint_pipeline(
                case["approximate"], actual, second, case["effective_train_iters"], config_path
            )
    finally:
        sys.path.pop(0)


def verify_repeat(
    manifest: dict, root: Path, case: dict, directory: Path, attempt: int, repeat: int
) -> dict:
    execution = json.loads((directory / "execution.json").read_text())
    verify_identity(execution, manifest, case, attempt, repeat)
    finite_seconds(execution["duration_seconds"])
    if type(execution.get("exit_code")) is not int:
        raise ValueError("Missing shell exit code")
    statuses = read_training_statuses(directory)
    text = log_text(directory)
    classification = classify_failure(
        execution["exit_code"],
        text,
        cancelled=execution["cancelled"],
        timed_out=execution["timed_out"],
    )
    if classification in ("passed", "infrastructure_retryable") and 139 in statuses.values():
        classification = "hard_failure"
    if classification != "passed":
        if classification == "infrastructure_retryable" and not statuses:
            classification = "integrity_error"
        return {
            "success": False,
            "classification": classification,
            "duration_seconds": execution["duration_seconds"],
        }
    verify_source(manifest, root, case)
    verify_runtime_config(root, case, directory)
    if statuses != {phase: 0 for phase in range(1, phase_count(case) + 1)}:
        raise ValueError("Training phase or resume phase did not finish successfully")
    if execution["runtime_config_sha256"] != control.sha256(directory / "model_config.yaml"):
        raise ValueError("Runtime configuration digest mismatch")
    for rank in range(8):
        logs = list((directory / "logs").glob(f"**/{rank}/stdout.log"))
        if len(logs) < phase_count(case) or any(
            not path.with_name("stderr.log").is_file() for path in logs
        ):
            raise ValueError(f"Missing rank/phase log evidence: rank {rank}")
    for phase in range(1, phase_count(case) + 1):
        tensorboard = directory / "1/tensorboard"
        if phase_count(case) == 2:
            tensorboard /= f"run_{phase}"
        if not any(
            path.is_file() and path.stat().st_size
            for path in tensorboard.rglob("events.out.tfevents.*")
        ):
            raise ValueError(f"Missing TensorBoard phase evidence: {phase}")
    if case["observation"]:
        for phase in range(1, phase_count(case) + 1):
            name = "observation.json" if phase == 1 else "observation_2nd.json"
            values = control.read_metrics(directory / name)
            if not any(
                control.finite(row["values"].get(case["effective_train_iters"]))
                for row in values.values()
            ):
                raise ValueError(
                    f"Observation phase {phase} did not reach its configured final training iteration"
                )
        if list(directory.glob("pytest-*.json")):
            raise ValueError("Observation unexpectedly published numeric pytest evidence")
    else:
        first_path = directory / Path(case["golden_path"]).name
        first = control.read_metrics(first_path)
        golden = control.read_metrics(root / case["golden_path"])
        control.validate_region(golden, first, case["metrics"])
        if not any(
            control.finite(row["values"].get(case["effective_train_iters"]))
            for row in first.values()
        ):
            raise ValueError("Training did not reach its configured final iteration")
        verify_pytest(directory, "regular", manifest, case, attempt, repeat)
        if phase_count(case) == 2:
            second = control.read_metrics(first_path.with_name(f"{first_path.stem}_2nd.json"))
            control.validate_region(
                first, second, case["metrics"], after=case["effective_train_iters"] // 2
            )
            if not any(
                control.finite(row["values"].get(case["effective_train_iters"]))
                for row in second.values()
            ):
                raise ValueError("Resume did not reach its configured final iteration")
            verify_pytest(directory, "resume", manifest, case, attempt, repeat)
        compare_upstream(root, case, directory)
    return {
        "success": True,
        "classification": "observation" if case["observation"] else "passed",
        "duration_seconds": execution["duration_seconds"],
    }


def action_timings(action: dict) -> dict:
    values = {
        key: finite_seconds(float(action["outputs"][key]))
        for key in (
            "duration_seconds",
            "queue_duration_seconds",
            "occupied_ppu_card_hours",
            "execution_ppu_card_hours",
        )
    }
    if values["queue_duration_seconds"] > values["duration_seconds"]:
        raise ValueError("Inconsistent action timing evidence")
    return {
        "action_seconds": values["duration_seconds"],
        "queue_seconds": values["queue_duration_seconds"],
        "occupied_ppu_card_hours": values["occupied_ppu_card_hours"],
        "execution_ppu_card_hours": values["execution_ppu_card_hours"],
    }


def verify_envelope(manifest: dict, case: dict, directory: Path) -> dict:
    assets = json.loads((directory / "assets-report.json").read_text())
    if (
        assets.get("identity") != manifest["identity"]
        or assets.get("ok") is not True
        or assets.get("location") != "pod"
        or assets.get("numerical_validation") is not False
        or assets.get("pod_data_path") != manifest["config"]["data_path"]
    ):
        raise ValueError("Missing or invalid PPU asset preflight evidence")
    rows = assets.get("cases", [])
    if len(rows) != 1 or rows[0]["id"] != case["id"] or rows[0]["name"] != case["name"]:
        raise ValueError("Asset preflight belongs to a different case")
    if (
        {row["relative_path"] for row in rows[0]["assets"]}
        != {row["relative_path"] for row in case["assets"]}
        or not assets["root"]["ok"]
        or not all(row["ok"] for row in rows[0]["assets"])
    ):
        raise ValueError("Asset preflight is incomplete")
    environment = json.loads((directory / "functional-environment.json").read_text())
    if (
        environment.get("identity") != manifest["identity"]
        or environment.get("devices") != 8
        or environment.get("image") != manifest["config"]["image"]
        or any(
            not environment.get(key)
            for key in ("python", "packages", "bash", "yq", "envsubst", "uv")
        )
    ):
        raise ValueError("Missing or invalid functional environment evidence")
    action = json.loads((directory / "action-timing.json").read_text())
    pod = json.loads((directory / "pod-timing.json").read_text())
    if (
        action.get("action_outcome") != "success"
        or action.get("action_status") != "succeeded"
        or pod.get("exit_code") != 0
    ):
        raise ValueError("Functional action or Pod did not complete successfully")
    timings = action_timings(action)
    pod_seconds = finite_seconds(pod["elapsed_seconds"])
    if pod_seconds != pod["completed_at"] - pod["started_at"]:
        raise ValueError("Inconsistent action/Pod timing evidence")
    return {
        "budget_seconds": case["budget_seconds"],
        **timings,
        "pod_seconds": pod_seconds,
        "environment": environment,
    }


def verify_case(manifest: dict, root: Path, case: dict, directory: Path) -> dict:
    report = {
        "id": case["id"],
        "name": case["name"],
        "observation": case["observation"],
        "budget_seconds": case["budget_seconds"],
        "classification": "integrity_error",
        "success": False,
        "attempts": [],
        "errors": [],
    }
    try:
        action_path = directory / "action-timing.json"
        if action_path.is_file():
            action = json.loads(action_path.read_text())
            try:
                report.update(action_timings(action))
            except (ValueError, KeyError, TypeError) as exc:
                report["timing_error"] = str(exc)
        assets_path = directory / "assets-report.json"
        if assets_path.is_file():
            assets = json.loads(assets_path.read_text())
            if assets.get("identity") == manifest["identity"] and assets.get("ok") is False:
                report["classification"] = "asset_error"
                raise ValueError(
                    f"PPU asset preflight failed: {assets.get('result', 'unavailable')}"
                )
        if action_path.is_file() and not (directory / "case-run.json").is_file():
            status = action.get("action_status")
            report["classification"] = (
                "cancelled" if status == "cancelled" else "infrastructure_error"
            )
            raise ValueError(f"PPU action did not produce case evidence: {status}")
        record = json.loads((directory / "case-run.json").read_text())
        if (
            record.get("identity") != manifest["identity"]
            or record.get("manifest_sha256") != manifest_digest(manifest)
            or record.get("case_id") != case["id"]
            or record.get("complete") is not True
        ):
            raise ValueError("Missing or mismatched case completion evidence")
        elapsed = finite_seconds(record["duration_seconds"])
        report["duration_seconds"] = elapsed
        if elapsed > case["budget_seconds"] or record.get("cancelled") or record.get("timed_out"):
            report["classification"] = (
                "cancelled" if record.get("cancelled") else "budget_exhausted"
            )
            raise ValueError("Case cancelled or total training budget exhausted")
        count = control.positive(record["attempts"], "attempt count", 3)
        total_repeats_seconds = 0.0
        for attempt in range(1, count + 1):
            attempt_dir = directory / f"attempt-{attempt}"
            repetitions = sorted(
                attempt_dir.glob("repeat-*"), key=lambda path: int(path.name.split("-")[-1])
            )
            if not repetitions or len(repetitions) > case["repeat"]:
                raise ValueError("Missing or excess functional repetitions")
            results = []
            for repeat, repeat_dir in enumerate(repetitions, 1):
                if repeat_dir.name != f"repeat-{repeat}":
                    raise ValueError("Noncontiguous functional repetitions")
                try:
                    result = verify_repeat(manifest, root, case, repeat_dir, attempt, repeat)
                except AssertionError:
                    report["classification"] = "numeric_failure"
                    raise
                results.append(result)
                total_repeats_seconds += result["duration_seconds"]
                if not result["success"] and repeat != len(repetitions):
                    raise ValueError("Continued an attempt after its first failure")
            report["attempts"].append({"index": attempt, "repeats": results})
            if attempt < count:
                if results[-1]["classification"] != "infrastructure_retryable":
                    raise ValueError("Retried a non-infrastructure failure or a successful attempt")
            elif len(results) != case["repeat"] or not all(result["success"] for result in results):
                failures = [result["classification"] for result in results if not result["success"]]
                if failures:
                    report["classification"] = failures[-1]
                raise ValueError("Final attempt lacks all required successful repetitions")
        if total_repeats_seconds > elapsed + 1:
            raise ValueError("Inconsistent case and repetition timings")
        actual_attempts = {path.name for path in directory.glob("attempt-*")}
        if actual_attempts != {f"attempt-{attempt}" for attempt in range(1, count + 1)}:
            raise ValueError("Unexpected or missing attempt evidence")
        report.update(verify_envelope(manifest, case, directory))
        report.update(
            success=True,
            retried=count > 1,
            duration_seconds=elapsed,
            classification="observation" if case["observation"] else "passed",
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AssertionError,
        AttributeError,
        IndexError,
    ) as exc:
        report["errors"].append(str(exc))
    return report


def summarize(manifest: dict, root: Path, directory: Path) -> dict:
    if (
        manifest.get("test_kind") != "functional"
        or manifest["config"]["functional_mode"] != "regression"
    ):
        raise ValueError("Asset-only preflight cannot produce a regression summary")
    if not manifest.get("functional"):
        raise ValueError("Functional manifest has no cases")
    cases = [
        verify_case(manifest, root, case, directory / case["id"])
        for case in manifest["functional"].values()
    ]
    numeric = [case for case in cases if not case["observation"]]
    observations = [case for case in cases if case["observation"]]
    return {
        "schema_version": 1,
        "identity": manifest["identity"],
        "manifest_sha256": manifest_digest(manifest),
        "success": all(case["success"] for case in cases),
        "numerical_validation": bool(numeric) and all(case["success"] for case in numeric),
        "numeric_passed": sum(case["success"] for case in numeric),
        "numeric_failed": sum(not case["success"] for case in numeric),
        "observations": observations,
        "failure_categories": {
            category: [case["name"] for case in cases if case["classification"] == category]
            for category in sorted(
                {case["classification"] for case in cases if not case["success"]}
            )
        },
        "cases": cases,
        "coverage": manifest["coverage"],
        "threshold_percent": 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = summarize(manifest, args.root, args.results)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = [
        "## PPU functional results",
        f"Numeric passed: {result['numeric_passed']}; failed: {result['numeric_failed']}; observations: {len(result['observations'])}.",
        f"Numerical validation: {result['numerical_validation']}. Integrity and execution success: {result['success']}.",
        f"Failure categories: {json.dumps(result['failure_categories'], sort_keys=True)}.",
        f"Ignored without PPU golden: {len(result['coverage']['ignored_missing_golden'])}; upstream disabled: {len(result['coverage']['upstream_disabled'])}.",
    ]
    summary += [
        f"- {row['name']}: {'passed' if row['success'] else 'failed'}{' (observation only)' if row['observation'] else ''}; queue={row.get('queue_seconds', 'unavailable')}s, training={row.get('duration_seconds', 'unavailable')}s, budget={row.get('budget_seconds', 'unavailable')}s; {'; '.join(row['errors'])}"
        for row in result["cases"]
    ]
    for key in ("ignored_missing_golden", "upstream_disabled", "baseline_not_selected"):
        summary.append(f"<details><summary>{key}</summary>\n")
        summary.extend(f"- {row['name']}: {row['reason']}" for row in result["coverage"][key])
        summary.append("</details>")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
