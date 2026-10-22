"""PPU Build/UT."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from types import ModuleType

CONTROL_ROOT = Path(os.environ.get("PPU_CONTROL_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("PPU_SOURCE_ROOT", CONTROL_ROOT)).resolve()
ENVIRONMENTS = ("dev", "lts")
OUTCOME_ORDER = {"skipped": 0, "passed": 1, "failed": 2}
RECIPE_PATH = "tests/test_utils/recipes/h100/unit-tests.yaml"
POLICY_PATH = ".github/ppu-pr-tests.json"


def load_policy() -> dict:
    policy = json.loads((CONTROL_ROOT / POLICY_PATH).read_text())
    if policy.get("schema_version") != 2 or policy.get("daily_scope") not in ("full", "quick"):
        raise ValueError("Invalid UT policy version or daily scope")
    state = policy.get("quick_state")
    if state not in ("unavailable", "candidate", "validated"):
        raise ValueError("Invalid quick validation state")
    require_baseline = policy.get("require_full_baseline", True)
    if not isinstance(require_baseline, bool):
        raise ValueError("require_full_baseline must be a boolean")
    if require_baseline and state != "unavailable" and not policy.get("baseline"):
        raise ValueError("Quick is missing full baseline evidence")
    if (state == "validated" or policy["daily_scope"] == "quick") and (
        state != "validated" or len(set(policy.get("validation_runs", []))) < 2
    ):
        raise ValueError("Daily quick requires two validation runs before activation")
    return policy


def threshold(config: dict) -> int:
    return 100 if config["scope"] == "quick" else 90


def positive(value: object, name: str, maximum: int = 10000) -> int:
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", text) or int(text) > maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return int(text)


def select_config(event: dict, event_name: str, inputs: dict) -> dict:
    if event_name not in {"pull_request_target", "push", "workflow_dispatch", "merge_group"}:
        raise ValueError(f"Unsupported trigger event: {event_name}")
    config = {"build": inputs.get("build", True), "unit_test": inputs.get("unit_test", True)}
    for key, value in config.items():
        text = str(value).lower()
        if text not in {"true", "false", "yes", "no"}:
            raise ValueError(f"{key} must be a boolean or yes/no")
        config[key] = text in {"true", "yes"}
    repeat = inputs.get("repeat", 1)
    config["repeat"] = 0 if str(repeat) == "0" else positive(repeat, "repeat", 10)
    config["timeout"] = positive(inputs.get("timeout", 30), "timeout", 1380)
    policy = load_policy()
    scope = inputs.get("scope", "auto")
    family = inputs.get("suite_family", "nvidia-dev")
    if scope not in ("auto", "quick", "full") or family not in ("nvidia-dev", "release-dev-lts"):
        raise ValueError("Invalid scope or suite_family")
    trigger = "manual"
    ref = event.get("ref", "")
    if event_name == "push":
        if ref.startswith("refs/tags/v"):
            scope, family, trigger = "full", "release-dev-lts", "release"
            config.update(build=True, unit_test=True, repeat=1)
        elif ref == "refs/heads/main":
            scope, family, trigger = "auto", "nvidia-dev", "push"
            config.update(build=True, unit_test=True, repeat=1)
        else:
            raise ValueError("Unsupported push ref")
    if event_name == "merge_group":
        if event["merge_group"]["base_ref"] != "refs/heads/main":
            raise ValueError("The merge queue base must be main")
        scope, family, trigger = "full", "nvidia-dev", "merge_group"
        config.update(build=True, unit_test=True, repeat=1)
    if event_name == "pull_request_target":
        scope, family, trigger = "auto", "nvidia-dev", "pr"
        pr = event["pull_request"]
        if pr["base"]["ref"] != "main":
            raise ValueError("The PR base must be main")
        config.update(build=True, unit_test=True, repeat=1)
        if not pr["head"].get("repo") or (
            pr["head"]["repo"]["full_name"] != pr["base"]["repo"]["full_name"]
        ):
            config["build"] = False
    if event_name == "workflow_dispatch":
        if ref != "refs/heads/main":
            raise ValueError("Manual dispatch is only allowed on main")
        number, sha, base = (inputs.get(key, "") for key in ("pr_number", "pr_sha", "pr_base_sha"))
        if any((number, sha, base)) and not all((number, sha, base)):
            raise ValueError("The PR number and full head/base SHAs must be provided together")
        if number:
            positive(number, "PR number", 10**9)
            if any(not re.fullmatch(r"[0-9a-fA-F]{40}", str(value)) for value in (sha, base)):
                raise ValueError("PR SHAs must be full 40-character hexadecimal strings")
            family, trigger = "nvidia-dev", "pr"
            config.update(build=True, unit_test=True, repeat=1)
    if scope == "auto":
        scope = "full" if family == "release-dev-lts" else policy["daily_scope"]
    if scope == "quick":
        if family != "nvidia-dev" or policy["quick_state"] == "unavailable":
            raise ValueError("Quick is unavailable for this suite or policy state")
        if policy["quick_state"] == "candidate" and trigger != "manual":
            raise ValueError("Candidate quick is limited to non-PR manual validation")
        config["repeat"] = 1
    config.update(scope=scope, suite_family=family, trigger_kind=trigger)
    bucket = inputs.get("unit_bucket", "")
    if not isinstance(bucket, str):
        raise ValueError("unit_bucket must be a full bucket string")
    if bucket:
        if (
            event_name != "workflow_dispatch"
            or trigger != "manual"
            or inputs.get("scope") != "full"
            or family != "nvidia-dev"
            or not config["build"]
            or not config["unit_test"]
            or config["repeat"] != 1
        ):
            raise ValueError(
                "A single bucket is only allowed for non-PR manual DEV/full diagnostics with a build and exactly one execution"
            )
        config.update(scope="diagnostic", unit_bucket=bucket)
    config["threshold_percent"] = threshold(config)
    config["unit_test"] = config["unit_test"] and config["build"] and config["repeat"] > 0
    return config


def load_control_module(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, CONTROL_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit_cases(config: dict) -> dict:
    import yaml

    helper = load_control_module("ppu_bucket_helper", "tests/unit_tests/find_test_cases.py")
    recipe_parser = load_control_module(
        "ppu_recipe_parser", "tests/test_utils/python_scripts/recipe_parser.py"
    )
    recipe_file = (ROOT / RECIPE_PATH).resolve()
    if not recipe_file.is_relative_to(ROOT):
        raise ValueError("Recipe path escapes the source root")
    recipe = yaml.safe_load(recipe_file.read_text())
    if recipe.get("format_version") != 1 or recipe.get("type") != "basic":
        raise ValueError("Unsupported UT recipe structure")
    for product in recipe["products"]:
        if len(product["test_case"]) != 1 or not product.get("products"):
            raise ValueError("Recipe product requires explicit adaptation")
    workloads = recipe_parser.load_and_flatten(str(recipe_file))
    buckets = [bucket for product in recipe["products"] for bucket in product["test_case"]]
    if len(buckets) != len(set(buckets)):
        raise ValueError("Duplicate UT bucket")
    for bucket in buckets:
        if not re.fullmatch(
            r"tests/unit_tests/(?:[A-Za-z0-9_]+/)*(?:[A-Za-z0-9_]+\.py|\*\*/\*\.py)", bucket
        ):
            raise ValueError(f"Unsupported UT bucket; refusing to skip silently: {bucket}")
    excluded = policy_exclusions()
    cases = {}
    environments = (
        ("dev",) if config.get("suite_family", "release-dev-lts") == "nvidia-dev" else ENVIRONMENTS
    )
    for environment in environments:
        specs = [
            workload.spec
            for workload in workloads
            if all(
                workload.spec.get(key) == value
                for key, value in {
                    "scope": "unit-tests",
                    "environment": environment,
                    "tag": "latest",
                    "platforms": "dgx_h100",
                    "model": "unit-tests",
                }.items()
            )
        ]
        selected_buckets = [spec["test_case"] for spec in specs]
        if len(selected_buckets) != len(set(selected_buckets)):
            raise ValueError(f"{environment} UT recipe has duplicate matches")
        if set(selected_buckets) != set(buckets):
            raise ValueError(
                f"{environment} UT recipe has unmatched buckets requiring explicit adaptation"
            )
        if not specs:
            raise ValueError(f"{environment} UT recipe is empty")
        for index, spec in enumerate(sorted(specs, key=lambda row: row["test_case"])):
            bucket = spec["test_case"]
            target = helper.get_base_path(bucket)
            if not (ROOT / target).exists() or not (ROOT / target).resolve().is_relative_to(
                ROOT / "tests/unit_tests"
            ):
                raise ValueError(f"UT path does not exist: {target}")
            if positive(spec["nodes"], "nodes") != 1 or positive(spec["gpus"], "gpus") != 8:
                raise ValueError(f"UT topology requires explicit adaptation: {bucket}")
            files = {
                path.relative_to(ROOT).as_posix() for path in ROOT.glob(bucket) if path.is_file()
            }
            ignored = set()
            for child in buckets:
                if child != bucket and helper.is_child_of_bucket(child, bucket):
                    ignored.update(
                        path.relative_to(ROOT).as_posix()
                        for path in ROOT.glob(child)
                        if path.is_file()
                    )
            if any(
                not (ROOT / path).resolve().is_relative_to(ROOT / "tests/unit_tests")
                for path in files
            ):
                raise ValueError("UT file path escapes the unit test root")
            if not files - ignored:
                raise ValueError(f"Empty UT bucket: {bucket}")
            case_id = f"{environment}-{index:03d}"
            minutes = (int(spec.get("time_limit_override", config["timeout"] * 60)) + 59) // 60
            positive(minutes, "bucket timeout", 1380)
            cases[case_id] = {
                "id": case_id,
                "bucket": bucket,
                "environment": environment,
                "cards": 8,
                "minutes": minutes,
                "action_minutes": minutes + 15,
                "job_minutes": minutes + 30,
                "repeat": config["repeat"],
                "args": [target] + [f"--ignore={path}" for path in sorted(files & ignored)],
                "files": sorted(files - ignored),
                "excluded": {
                    selector: reason
                    for selector, reason in excluded.items()
                    if selector.split("::")[0] in files - ignored
                },
            }
    for selector in excluded:
        if not any(selector in case["excluded"] for case in cases.values()):
            raise ValueError(f"Exclusion is outside the full test suite: {selector}")
    return cases


def selector_matches(selector: str, nodeid: str) -> bool:
    return nodeid == selector or nodeid.startswith((selector + "::", selector + "["))


def validate_selector(selector: str) -> str:
    if not isinstance(selector, str) or not re.fullmatch(
        r"tests/unit_tests/[A-Za-z0-9_/]+\.py(?:::[A-Za-z_][A-Za-z0-9_]*)*", selector
    ):
        raise ValueError(f"Invalid UT selector: {selector}")
    filename = selector.split("::", 1)[0]
    path = (ROOT / filename).resolve()
    if not path.is_relative_to(ROOT / "tests/unit_tests") or not path.is_file():
        raise ValueError(f"UT file is missing or outside the unit test root: {filename}")
    if not path.name.startswith("test_"):
        raise ValueError(f"Not a UT file: {filename}")
    return filename


def policy_exclusions() -> dict:
    excluded = load_policy().get("excluded", {})
    if not isinstance(excluded, dict):
        raise ValueError("Exclusion list must map selectors to reasons")
    for selector, reason in excluded.items():
        validate_selector(selector)
        if "::" not in selector or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Exclusions must specify an exact method or class and a reason")
        if any(old != selector and selector_matches(old, selector) for old in excluded):
            raise ValueError("Overlapping exclusion selectors")
    return excluded


def quick_cases(config: dict) -> dict:
    recipe = load_policy()
    if not recipe.get("groups"):
        raise ValueError("Invalid quick manifest version or missing groups")
    cases, seen = {}, []
    for group in recipe["groups"]:
        case_id = group["id"]
        if not re.fullmatch(r"dev-pr-[0-9]{2}", case_id) or case_id in cases:
            raise ValueError(f"Invalid or duplicate quick group ID: {case_id}")
        selectors = group["selectors"]
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"Empty quick group: {case_id}")
        files = set()
        for selector in selectors:
            filename = validate_selector(selector)
            if any(
                selector_matches(old, selector) or selector_matches(selector, old) for old in seen
            ):
                raise ValueError(f"Duplicate or overlapping selector: {selector}")
            seen.append(selector)
            files.add(filename)
        excluded = {
            selector: reason
            for selector, reason in policy_exclusions().items()
            if selector.split("::")[0] in files
        }
        if group.get("excluded"):
            raise ValueError("Exclusions must be declared in the shared exclusion list")
        if not isinstance(excluded, dict):
            raise ValueError(f"Quick exclusion list must map selectors to reasons: {case_id}")
        for selector, reason in excluded.items():
            validate_selector(selector)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Excluded tests must have a reason: {selector}")
            if not any(
                selector != parent and selector_matches(parent, selector) for parent in selectors
            ):
                raise ValueError(
                    f"Excluded tests must be a subset of a selected selector: {selector}"
                )
            if any(old != selector and selector_matches(old, selector) for old in excluded):
                raise ValueError(f"Overlapping exclusion selectors: {selector}")
        minutes = positive(group["minutes"], "quick group timeout", 35)
        cases[case_id] = {
            "id": case_id,
            "bucket": group["name"],
            "coverage": group["coverage"],
            "environment": "dev",
            "cards": 8,
            "minutes": minutes,
            "repeat": 1,
            "args": selectors,
            "excluded": excluded,
            "files": sorted(files),
            "strict_selection": True,
        }
    if sum(case["minutes"] for case in cases.values()) > 35:
        raise ValueError("The total quick test budget must not exceed 35 minutes")
    return cases


def generate(config: dict, output: Path, identity: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    quick = config.get("scope", "full") == "quick"
    cases = (quick_cases(config) if quick else unit_cases(config)) if config["unit_test"] else {}
    diagnostic = config.get("scope") == "diagnostic"
    if diagnostic or config.get("unit_bucket"):
        if (
            not diagnostic
            or config.get("trigger_kind") != "manual"
            or config.get("suite_family") != "nvidia-dev"
            or not config["build"]
            or config["repeat"] != 1
        ):
            raise ValueError("Invalid single-bucket diagnostic configuration")
        cases = {
            key: case for key, case in cases.items() if case["bucket"] == config.get("unit_bucket")
        }
        if len(cases) != 1:
            raise ValueError(
                "unit_bucket must exactly match one upstream DEV bucket; empty runs or expanded scope are not allowed"
            )
    if quick and cases:
        full = unit_cases({**config, "suite_family": "nvidia-dev", "repeat": 1, "timeout": 30})
        for case in cases.values():
            case["source_buckets"] = {}
            for selector in case["args"]:
                matches = [
                    row["bucket"]
                    for row in full.values()
                    if selector.split("::")[0] in row["files"]
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Quick selector does not belong to a unique upstream bucket: {selector}"
                    )
                case["source_buckets"][selector] = matches[0]
    provenance = {
        "recipe_path": RECIPE_PATH,
        "recipe_sha256": hashlib.sha256((ROOT / RECIPE_PATH).read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256((CONTROL_ROOT / POLICY_PATH).read_bytes()).hexdigest(),
        "excluded": policy_exclusions(),
    }
    manifest = {
        "schema_version": 3,
        "config": config,
        "identity": identity or {},
        "provenance": provenance,
        "unit": cases,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    keys = ("id", "environment", "bucket", "cards", "action_minutes", "job_minutes")
    if quick and cases:
        rows = [
            {
                "id": "quick",
                "environment": "dev",
                "bucket": "PR representative suite",
                "cards": 8,
                "action_minutes": 45,
                "job_minutes": 50,
            }
        ]
    else:
        rows = [{key: row[key] for key in keys} for row in cases.values()]
    if len(rows) > 256:
        raise ValueError("UT exceeds the GitHub limit of 256 matrix jobs")
    emit("unit_matrix", json.dumps({"include": rows or [{"id": "unused"}]}))
    return manifest


def case_results(
    case: dict, directory: Path, identity: dict | None = None, require_timing: bool = False
) -> dict:
    outcomes, errors, collected, selected = {}, [], set(), set()
    durations, phase_timings = {}, []
    elapsed = 0.0
    for iteration in range(1, case["repeat"] + 1):
        for phase in ("prod", "experimental"):
            phase_seconds, collection_seconds = 0.0, 0.0
            for rank in range(case["cards"]):
                filename = f"{phase}-{iteration}-rank-{rank}.json"
                try:
                    report = json.loads((directory / filename).read_text())
                    if (
                        report["rank"] != rank
                        or report["complete"] is not True
                        or report["collection_errors"]
                        or report.get("execution_errors")
                    ):
                        raise ValueError(
                            "Collection or fixture error, incomplete execution, or rank mismatch"
                        )
                    if report["exitstatus"] not in (0, 1, 5):
                        raise ValueError(f"Unexpected pytest exit status: {report['exitstatus']}")
                    if not isinstance(report["tests"], dict):
                        raise ValueError("Test results must be a mapping of test cases")
                    if identity:
                        for key in ("target_sha", "control_sha"):
                            if report.get(key) != identity[key]:
                                raise ValueError(f"Report {key} mismatch")
                    if case.get("strict_selection") or "selected" in report:
                        if (
                            report["case_id"] != case["id"]
                            or report["phase"] != phase
                            or report["iteration"] != iteration
                        ):
                            raise ValueError("Report group, phase, or iteration mismatch")
                        for key in ("collected", "selected"):
                            values = report[key]
                            if (
                                not isinstance(values, list)
                                or any(not isinstance(v, str) for v in values)
                                or len(values) != len(set(values))
                            ):
                                raise ValueError(f"Report {key} is invalid or contains duplicates")
                        if set(report["selected"]) != set(report["tests"]) or not (
                            set(report["selected"]) <= set(report["collected"])
                        ):
                            raise ValueError(
                                "Collected, selected, and executed tests are inconsistent"
                            )
                        collected.update(report["collected"])
                        selected.update(report["selected"])
                    if require_timing and any(
                        key not in report
                        for key in (
                            "selected",
                            "duration_seconds",
                            "collection_seconds",
                            "durations",
                        )
                    ):
                        raise ValueError("Per-rank report is missing selection or timing fields")
                    duration = float(report.get("duration_seconds", 0))
                    if not math.isfinite(duration) or duration < 0:
                        raise ValueError("Invalid report duration")
                    phase_seconds = max(phase_seconds, duration)
                    collection = float(report.get("collection_seconds", 0))
                    if not math.isfinite(collection) or not 0 <= collection <= duration:
                        raise ValueError("Invalid collection duration")
                    collection_seconds = max(collection_seconds, collection)
                    node_durations = report.get("durations", {})
                    if not isinstance(node_durations, dict):
                        raise ValueError("Invalid test duration mapping")
                    if require_timing and set(node_durations) != set(report["tests"]):
                        raise ValueError("Incomplete per-rank test durations")
                    for nodeid, seconds in node_durations.items():
                        if (
                            nodeid not in report["tests"]
                            or type(seconds) not in (int, float)
                            or not math.isfinite(seconds)
                            or seconds < 0
                        ):
                            raise ValueError("Invalid test duration")
                        key = f"{iteration}:{phase}:{nodeid}"
                        durations[key] = max(durations.get(key, 0), seconds)
                    for nodeid, outcome in report["tests"].items():
                        if not isinstance(nodeid, str) or outcome not in OUTCOME_ORDER:
                            raise ValueError("Invalid nodeid or test outcome")
                        if case.get("strict_selection") and not any(
                            selector_matches(selector, nodeid) for selector in case["args"]
                        ):
                            raise ValueError(f"Executed a test outside the manifest: {nodeid}")
                        if any(
                            selector_matches(selector, nodeid)
                            for selector in case.get("excluded", {})
                        ):
                            raise ValueError(f"Executed an explicitly excluded test: {nodeid}")
                        key = f"{iteration}:{phase}:{nodeid}"
                        if OUTCOME_ORDER[outcome] > OUTCOME_ORDER.get(outcomes.get(key), -1):
                            outcomes[key] = outcome
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{case['id']}/{filename}: {exc}")
            elapsed += phase_seconds
            phase_timings.append(
                {
                    "iteration": iteration,
                    "phase": phase,
                    "duration_seconds": phase_seconds,
                    "collection_seconds": collection_seconds,
                }
            )
    for selector in case.get("excluded", {}):
        if not any(selector_matches(selector, nodeid) for nodeid in collected):
            errors.append(f"{case['id']}: No tests collected for excluded selector: {selector}")
    executed = {key.split(":", 2)[2] for key, value in outcomes.items() if value != "skipped"}
    if not executed:
        errors.append(f"{case['id']}: No tests were actually executed in this group")
    if case.get("strict_selection"):
        for selector in case["args"]:
            if not any(selector_matches(selector, nodeid) for nodeid in collected):
                errors.append(f"{case['id']}: No tests collected for selector: {selector}")
            elif not any(selector_matches(selector, nodeid) for nodeid in selected):
                errors.append(f"{case['id']}: All tests for selector were filtered out: {selector}")
            elif not any(selector_matches(selector, nodeid) for nodeid in executed):
                errors.append(f"{case['id']}: All tests for selector were skipped: {selector}")
    return {
        "outcomes": outcomes,
        "errors": errors,
        "complete": not errors,
        "duration_seconds": elapsed,
        "durations": durations,
        "phase_timings": phase_timings,
        "excluded_count": sum(
            any(selector_matches(s, node) for s in case.get("excluded", {})) for node in collected
        ),
    }


def result_directory(manifest: dict, directory: Path, case_id: str) -> Path:
    if manifest["config"].get("scope") == "quick":
        return directory / "ppu-ut-quick" / case_id
    return directory / f"ppu-ut-{case_id}"


def read_timing(path: Path, errors: list, suite: bool = False) -> dict:
    try:
        timing = json.loads(path.read_text())
        keys = ("started_at", "completed_at", "elapsed_seconds")
        if suite:
            keys += ("prepared_at",)
        if any(
            type(timing[key]) not in (int, float)
            or not math.isfinite(timing[key])
            or timing[key] < 0
            for key in keys
        ):
            raise ValueError("Invalid run duration")
        if timing["completed_at"] - timing["started_at"] != timing["elapsed_seconds"]:
            raise ValueError("Inconsistent run timestamps")
        if suite:
            if not timing["started_at"] <= timing["prepared_at"] <= timing["completed_at"]:
                raise ValueError("Environment preparation timestamp is outside the run interval")
            timing["prepare_seconds"] = timing["prepared_at"] - timing["started_at"]
            timing["execution_seconds"] = timing["completed_at"] - timing["prepared_at"]
        return timing
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"{path.name}: {exc}")
        return {}


def summarize(manifest: dict, directory: Path) -> dict:
    scope = manifest["config"].get("scope", "full")
    threshold_percent = threshold(manifest["config"])
    if manifest["config"].get("threshold_percent", threshold_percent) != threshold_percent:
        raise ValueError("Manifest gate does not match the test scope")
    counts = dict.fromkeys(OUTCOME_ORDER, 0)
    errors, failures, groups = [], [], {}
    for case in manifest["unit"].values():
        result_dir = result_directory(manifest, directory, case["id"])
        result = case_results(
            case,
            result_dir,
            manifest.get("identity"),
            require_timing=manifest.get("schema_version", 2) >= 3,
        )
        group_errors = result["errors"]
        timing_path = result_dir / "timing.json"
        timing = (
            read_timing(timing_path, group_errors)
            if manifest.get("schema_version", 2) >= 3 or scope == "quick" or timing_path.exists()
            else {}
        )
        try:
            status = json.loads((result_dir / "status.json").read_text())
            if status["exit_code"] != 0 or status["complete"] is not True:
                raise ValueError("Bucket execution did not complete")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            group_errors.append(f"{case['id']}: {exc}")
        group_counts = dict.fromkeys(OUTCOME_ORDER, 0)
        methods = {}
        for node, seconds in result["durations"].items():
            selector = node.split(":", 2)[2].split("[", 1)[0]
            methods[selector] = methods.get(selector, 0) + seconds
        suite_timing = {}
        if scope in ("full", "diagnostic") and manifest.get("schema_version", 2) >= 3:
            suite_timing = read_timing(result_dir / "suite-timing.json", group_errors, suite=True)
            if set(result["durations"]) != set(result["outcomes"]):
                group_errors.append("The full baseline is missing complete per-test durations")
        groups[case["id"]] = {
            "counts": group_counts,
            "bucket": case.get("bucket", case["id"]),
            "environment": case.get("environment"),
            "suite_timing": suite_timing,
            "method_durations": methods,
            "phase_timings": result["phase_timings"],
            "excluded_count": result["excluded_count"],
            "duration_seconds": result["duration_seconds"],
            "timing": timing,
            "errors": group_errors,
            "excluded": case.get("excluded", {}),
        }
        errors.extend(f"{case['id']}: {error}" for error in group_errors)
        for nodeid, outcome in result["outcomes"].items():
            group_counts[outcome] += 1
            counts[outcome] += 1
            if outcome == "failed":
                failures.append(f"{case['id']}:{nodeid}")
        groups[case["id"]]["executed"] = group_counts["passed"] + group_counts["failed"]
    timings = {}
    if manifest["config"].get("scope") == "quick":
        for name in ("build", "suite"):
            path = directory / "ppu-ut-quick" / f"{name}-timing.json"
            timings[name] = read_timing(path, errors, suite=name == "suite")
    if scope in ("full", "diagnostic") and manifest.get("schema_version", 2) >= 3:
        first = next(iter(manifest["unit"]), None)
        if first:
            timings["build"] = read_timing(
                result_directory(manifest, directory, first) / "build-timing.json", errors
            )
    total = counts["passed"] + counts["failed"]
    if os.environ.get("PPU_MATRIX_RESULT", "success") != "success":
        errors.append("At least one scheduling or artifact upload job did not succeed")
    success = not errors and total > 0 and counts["passed"] * 100 >= total * threshold_percent
    return {
        **counts,
        "scope": scope,
        "suite_family": manifest["config"].get("suite_family"),
        "trigger_kind": manifest["config"].get("trigger_kind"),
        "provenance": manifest.get("provenance", {}),
        "baseline_complete": scope != "diagnostic" and not errors and total > 0,
        "threshold_percent": threshold_percent,
        "identity": manifest.get("identity", {}),
        "groups": groups,
        "timings": timings,
        "total": total,
        "pass_rate": counts["passed"] / total if total else 0,
        "success": success,
        "errors": errors,
        "failed_tests": failures,
    }


def write_summary(result: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"UT ({result['scope']}): {result['passed']}/{result['total']} = {result['pass_rate']:.2%}, "
        f"threshold {result['threshold_percent']}%; failed {result['failed']}, skipped {result['skipped']}; "
        f"result: {'PASS' if result['success'] else 'FAIL'}\n"
    )
    if result["scope"] == "diagnostic":
        text += "Single-bucket diagnostic only; not a full-suite pass, a full baseline for quick, or a successful PR gate.\n"
    if result["identity"]:
        text += f"Commit identity: {result['identity']}\n"
    text += f"Trigger: {result.get('trigger_kind')}; suite: {result.get('suite_family')}; provenance: {result.get('provenance', {})}\n"
    for case_id, group in result["groups"].items():
        text += (
            f"{case_id}: {group['counts']}, executed {group['executed']}, "
            f"pytest phase duration {group['duration_seconds']:.1f}s"
        )
        if group["timing"]:
            text += f", group wall time {group['timing']['elapsed_seconds']:.1f}s"
        if group["suite_timing"]:
            text += f", environment preparation {group['suite_timing']['prepare_seconds']:.1f}s"
        text += f", excluded nodeids {group['excluded_count']}\n"
        for selector, reason in group["excluded"].items():
            text += f"  Explicit exclusion: {selector}; reason: {reason}\n"
    if result["timings"]:
        text += f"In-pod timings (excluding uploads, hosted configuration, and aggregation): {result['timings']}\n"
    if result["errors"]:
        text += f"Execution or report errors: {len(result['errors'])}; see summary.json.\n"
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)


def emit(key: str, value: object) -> None:
    value = str(value).lower() if isinstance(value, bool) else str(value)
    print(f"{key}={value}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(f"{key}={value}\n")


def bundle_inventory(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    inventory = {}
    for path in sorted(directory.rglob("*")):
        name = path.relative_to(directory).as_posix()
        if name in ("bundle-integrity.json", ".complete"):
            continue
        if not path.resolve(strict=True).is_relative_to(directory):
            raise ValueError(f"Build artifact path escapes the bundle root: {name}")
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"Build artifact directories must not be symlinks: {name}")
            continue
        if not path.is_file():
            raise ValueError(f"Build artifact is not a regular file: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        inventory[name] = {"size": path.stat().st_size, "sha256": digest}
    for name in (
        "source.tar.gz",
        "environment.tar.gz",
        "manifest.json",
        "commit.txt",
        "control.txt",
        "image.txt",
        "build-timing.json",
        "control/ppu_ci.py",
        "control/ppu_ci_run.sh",
        "control/ppu_pytest_policy.py",
    ):
        if name not in inventory or inventory[name]["size"] == 0:
            raise ValueError(f"Missing or empty build artifact: {name}")
    return inventory


def seal_bundle(directory: Path) -> None:
    inventory = bundle_inventory(directory)
    (directory / "bundle-integrity.json").write_text(
        json.dumps({"schema_version": 1, "files": inventory}, sort_keys=True), encoding="utf-8"
    )


def verify_bundle(directory: Path, plan: Path, digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Missing artifact digest from the trusted Build job")
    raw = (directory / "bundle-integrity.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Build artifact inventory does not match the trusted Build digest")
    sealed = json.loads(raw)
    if sealed.get("schema_version") != 1 or sealed.get("files") != bundle_inventory(directory):
        raise ValueError("Build artifacts are missing, truncated, or failed content verification")
    if (directory / "manifest.json").read_bytes() != (plan / "manifest.json").read_bytes():
        raise ValueError("Build artifacts do not match the trusted test plan")
    identity = json.loads((plan / "manifest.json").read_text())["identity"]
    for filename, key in (
        (".complete", "target_sha"),
        ("commit.txt", "target_sha"),
        ("control.txt", "control_sha"),
    ):
        expected = identity[key]
        if (
            not re.fullmatch(r"[0-9a-f]{40}", expected)
            or (directory / filename).read_text().strip() != expected
        ):
            raise ValueError(f"Build artifact commit identity mismatch: {filename}")
    trusted = {
        path.relative_to(plan).as_posix(): path.read_bytes()
        for path in (plan / "control").rglob("*")
        if path.is_file()
    }
    actual = {name for name in sealed["files"] if name.startswith("control/")}
    if not trusted or actual != set(trusted):
        raise ValueError("Build artifact control script inventory mismatch")
    for name, content in trusted.items():
        if (directory / name).read_bytes() != content:
            raise ValueError(
                f"Build artifact control script does not match the trusted plan: {name}"
            )
    print(
        f"Build artifact integrity verification passed: {len(sealed['files'])} files, target={identity['target_sha']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["configure", "summarize", "case-result", "seal-bundle", "verify-bundle"]
    )
    parser.add_argument("--output", type=Path, default=Path(".ppu-ci"))
    parser.add_argument("--plan", type=Path, default=Path(".ppu-ci"))
    parser.add_argument("--bundle-digest", default="")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--case-id")
    parser.add_argument("--runner-exit", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "seal-bundle":
        seal_bundle(args.output)
        return
    if args.mode == "verify-bundle":
        verify_bundle(args.output, args.plan, args.bundle_digest)
        return
    if args.mode == "configure":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        event_name = os.environ["GITHUB_EVENT_NAME"]
        event.setdefault("ref", os.environ["GITHUB_REF"])
        inputs = json.loads(os.environ.get("PPU_DEFAULTS", "{}"))
        inputs.update(json.loads(os.environ.get("PPU_INPUTS", "{}")))
        config = select_config(event, event_name, inputs)
        authorization = json.loads((args.output / "authorization.json").read_text())
        if authorization["allowed"] is not True:
            config.update(build=False, unit_test=False)
        identity = {
            key: authorization.get(key, "")
            for key in ("target_sha", "control_sha", "pr_head_sha", "base_sha")
        }
        emit("config", json.dumps(config))
        emit("build", config["build"])
        emit("unit_test", config["unit_test"])
        emit("scope", config["scope"])
        emit("threshold_percent", config["threshold_percent"])
        emit("suite_family", config["suite_family"])
        if not config["unit_test"]:
            print("Diagnostic run: UT was skipped; no successful PR gate is provided.")
        generate(config, args.output, identity)
    else:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if args.mode == "case-result":
            result = case_results(
                manifest["unit"][args.case_id],
                args.results,
                manifest.get("identity"),
                require_timing=manifest.get("schema_version", 2) >= 3,
            )
            result["exit_code"] = args.runner_exit
            result["complete"] = result["complete"] and args.runner_exit == 0
            (args.results / "status.json").write_text(json.dumps(result), encoding="utf-8")
            raise SystemExit(0 if result["complete"] else 1)
        result = summarize(manifest, args.results)
        write_summary(result, args.results)
        raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
