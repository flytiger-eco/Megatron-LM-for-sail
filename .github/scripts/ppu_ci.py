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
        raise ValueError("UT 策略版本或日常范围无效")
    state = policy.get("quick_state")
    if state not in ("unavailable", "candidate", "validated"):
        raise ValueError("quick 验证状态无效")
    if state != "unavailable" and not policy.get("baseline"):
        raise ValueError("quick 缺少全量基线证据")
    if policy["daily_scope"] == "quick" and (
        state != "validated" or len(set(policy.get("validation_runs", []))) < 2
    ):
        raise ValueError("日常 quick 必须先完成两轮验收")
    return policy


def threshold(config: dict) -> int:
    if config["scope"] == "quick" or config.get("trigger_kind") in ("pr", "push", "merge_group"):
        return 100
    return 80


def positive(value: object, name: str, maximum: int = 10000) -> int:
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", text) or int(text) > maximum:
        raise ValueError(f"{name} 必须为 1..{maximum} 的整数")
    return int(text)


def select_config(event: dict, event_name: str, inputs: dict) -> dict:
    if event_name not in {"pull_request_target", "push", "workflow_dispatch", "merge_group"}:
        raise ValueError(f"不支持的触发事件: {event_name}")
    config = {"build": inputs.get("build", True), "unit_test": inputs.get("unit_test", True)}
    for key, value in config.items():
        text = str(value).lower()
        if text not in {"true", "false", "yes", "no"}:
            raise ValueError(f"{key} 必须是布尔值或 yes/no")
        config[key] = text in {"true", "yes"}
    repeat = inputs.get("repeat", 1)
    config["repeat"] = 0 if str(repeat) == "0" else positive(repeat, "repeat", 10)
    config["timeout"] = positive(inputs.get("timeout", 30), "timeout", 1380)
    policy = load_policy()
    scope = inputs.get("scope", "auto")
    family = inputs.get("suite_family", "nvidia-dev")
    if scope not in ("auto", "quick", "full") or family not in ("nvidia-dev", "release-dev-lts"):
        raise ValueError("scope 或 suite_family 非法")
    trigger = "manual"
    ref = event.get("ref", "")
    if event_name == "push":
        if ref.startswith("refs/tags/v"):
            scope, family, trigger = "full", "release-dev-lts", "release"
            config.update(build=True, unit_test=True, repeat=1)
        elif ref == "refs/heads/ci/ppu-validation":
            scope, family, trigger = "auto", "nvidia-dev", "push"
            config.update(build=True, unit_test=True, repeat=1)
        else:
            raise ValueError("不支持的 push ref")
    if event_name == "merge_group":
        if event["merge_group"]["base_ref"] != "refs/heads/ci/ppu-validation":
            raise ValueError("合并队列 base 必须为验证分支")
        scope, family, trigger = "full", "nvidia-dev", "merge_group"
        config.update(build=True, unit_test=True, repeat=1)
    if event_name == "pull_request_target":
        scope, family, trigger = "auto", "nvidia-dev", "pr"
        pr = event["pull_request"]
        if pr["base"]["ref"] != "ci/ppu-validation":
            raise ValueError("PR base 必须为验证分支")
        config.update(build=True, unit_test=True, repeat=1)
        if not pr["head"].get("repo") or (
            pr["head"]["repo"]["full_name"] != pr["base"]["repo"]["full_name"]
        ):
            config["build"] = False
    if event_name == "workflow_dispatch":
        if ref != "refs/heads/ci/ppu-validation":
            raise ValueError("手动入口仅允许验证分支")
        number, sha, base = (inputs.get(key, "") for key in ("pr_number", "pr_sha", "pr_base_sha"))
        if any((number, sha, base)) and not all((number, sha, base)):
            raise ValueError("PR 编号与完整 head/base SHA 必须同时填写")
        if number:
            positive(number, "PR number", 10**9)
            if any(not re.fullmatch(r"[0-9a-fA-F]{40}", str(value)) for value in (sha, base)):
                raise ValueError("PR SHA 必须为完整的 40 位十六进制字符串")
            family, trigger = "nvidia-dev", "pr"
            config.update(build=True, unit_test=True, repeat=1)
    if scope == "auto":
        scope = "full" if family == "release-dev-lts" else policy["daily_scope"]
    if scope == "quick":
        if family != "nvidia-dev" or policy["quick_state"] == "unavailable":
            raise ValueError("对应 quick 尚无有效全量基线，不能执行")
        config["repeat"] = 1
    config.update(scope=scope, suite_family=family, trigger_kind=trigger)
    bucket = inputs.get("unit_bucket", "")
    if not isinstance(bucket, str):
        raise ValueError("unit_bucket 必须是完整 bucket 字符串")
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
            raise ValueError("单 bucket 仅允许非 PR 手动 DEV/full 诊断，必须构建并执行 1 次")
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
        raise ValueError("recipe 路径越界")
    recipe = yaml.safe_load(recipe_file.read_text())
    if recipe.get("format_version") != 1 or recipe.get("type") != "basic":
        raise ValueError("不支持的 UT recipe 结构")
    for product in recipe["products"]:
        if len(product["test_case"]) != 1 or not product.get("products"):
            raise ValueError("需要显式适配的 recipe product")
    workloads = recipe_parser.load_and_flatten(str(recipe_file))
    buckets = [bucket for product in recipe["products"] for bucket in product["test_case"]]
    if len(buckets) != len(set(buckets)):
        raise ValueError("重复 UT bucket")
    for bucket in buckets:
        if not re.fullmatch(
            r"tests/unit_tests/(?:[A-Za-z0-9_]+/)*(?:[A-Za-z0-9_]+\.py|\*\*/\*\.py)", bucket
        ):
            raise ValueError(f"不支持的 UT bucket，不能静默跳过: {bucket}")
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
            raise ValueError(f"{environment} UT recipe 存在重复匹配")
        if set(selected_buckets) != set(buckets):
            raise ValueError(f"{environment} UT recipe 有未匹配 bucket，需要显式适配")
        if not specs:
            raise ValueError(f"{environment} UT recipe 为空")
        for index, spec in enumerate(sorted(specs, key=lambda row: row["test_case"])):
            bucket = spec["test_case"]
            target = helper.get_base_path(bucket)
            if not (ROOT / target).exists() or not (ROOT / target).resolve().is_relative_to(
                ROOT / "tests/unit_tests"
            ):
                raise ValueError(f"UT 路径不存在: {target}")
            if positive(spec["nodes"], "nodes") != 1 or positive(spec["gpus"], "gpus") != 8:
                raise ValueError(f"需要显式适配的 UT 拓扑: {bucket}")
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
                raise ValueError("UT 文件路径越界")
            if not files - ignored:
                raise ValueError(f"空 UT bucket: {bucket}")
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
            raise ValueError(f"排除项不属于测试全集: {selector}")
    return cases


def selector_matches(selector: str, nodeid: str) -> bool:
    return nodeid == selector or nodeid.startswith((selector + "::", selector + "["))


def validate_selector(selector: str) -> str:
    if not isinstance(selector, str) or not re.fullmatch(
        r"tests/unit_tests/[A-Za-z0-9_/]+\.py(?:::[A-Za-z_][A-Za-z0-9_]*)*", selector
    ):
        raise ValueError(f"非法 UT selector: {selector}")
    filename = selector.split("::", 1)[0]
    path = (ROOT / filename).resolve()
    if not path.is_relative_to(ROOT / "tests/unit_tests") or not path.is_file():
        raise ValueError(f"UT 文件不存在或越界: {filename}")
    if not path.name.startswith("test_"):
        raise ValueError(f"不是 UT 文件: {filename}")
    return filename


def policy_exclusions() -> dict:
    excluded = load_policy().get("excluded", {})
    if not isinstance(excluded, dict):
        raise ValueError("排除清单必须包含 selector 与原因")
    for selector, reason in excluded.items():
        validate_selector(selector)
        if "::" not in selector or not isinstance(reason, str) or not reason.strip():
            raise ValueError("排除项必须为明确方法/类并说明原因")
        if any(old != selector and selector_matches(old, selector) for old in excluded):
            raise ValueError("排除 selector 重叠")
    return excluded


def quick_cases(config: dict) -> dict:
    recipe = load_policy()
    if not recipe.get("groups"):
        raise ValueError("quick 清单版本错误或没有分组")
    cases, seen = {}, []
    for group in recipe["groups"]:
        case_id = group["id"]
        if not re.fullmatch(r"dev-pr-[0-9]{2}", case_id) or case_id in cases:
            raise ValueError(f"quick 分组 ID 非法或重复: {case_id}")
        selectors = group["selectors"]
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"quick 分组为空: {case_id}")
        files = set()
        for selector in selectors:
            filename = validate_selector(selector)
            if any(
                selector_matches(old, selector) or selector_matches(selector, old) for old in seen
            ):
                raise ValueError(f"重复或重叠 selector: {selector}")
            seen.append(selector)
            files.add(filename)
        excluded = {
            selector: reason
            for selector, reason in policy_exclusions().items()
            if selector.split("::")[0] in files
        }
        if group.get("excluded"):
            raise ValueError("排除项必须在公共排除表声明")
        if not isinstance(excluded, dict):
            raise ValueError(f"quick 排除清单必须包含 selector 与原因: {case_id}")
        for selector, reason in excluded.items():
            validate_selector(selector)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"排除用例必须说明原因: {selector}")
            if not any(
                selector != parent and selector_matches(parent, selector) for parent in selectors
            ):
                raise ValueError(f"排除用例必须属于已选 selector 的子集: {selector}")
            if any(old != selector and selector_matches(old, selector) for old in excluded):
                raise ValueError(f"排除 selector 重叠: {selector}")
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
        raise ValueError("quick 总测试预算不能超过 35 分钟")
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
            raise ValueError("单 bucket 诊断配置非法")
        cases = {
            key: case for key, case in cases.items() if case["bucket"] == config.get("unit_bucket")
        }
        if len(cases) != 1:
            raise ValueError("unit_bucket 必须精确匹配一个上游 DEV bucket，拒绝空跑或扩展范围")
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
                    raise ValueError(f"quick selector 不属于唯一上游 bucket: {selector}")
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
        raise ValueError("UT 超过 GitHub 的 256 个 matrix job 上限")
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
                        raise ValueError("采集或 fixture 错误、未完成或 rank 不匹配")
                    if report["exitstatus"] not in (0, 1, 5):
                        raise ValueError(f"异常 pytest 退出状态: {report['exitstatus']}")
                    if not isinstance(report["tests"], dict):
                        raise ValueError("测试结果不是用例映射")
                    if identity:
                        for key in ("target_sha", "control_sha"):
                            if report.get(key) != identity[key]:
                                raise ValueError(f"报告 {key} 不匹配")
                    if case.get("strict_selection") or "selected" in report:
                        if (
                            report["case_id"] != case["id"]
                            or report["phase"] != phase
                            or report["iteration"] != iteration
                        ):
                            raise ValueError("报告分组、阶段或轮次不匹配")
                        for key in ("collected", "selected"):
                            values = report[key]
                            if (
                                not isinstance(values, list)
                                or any(not isinstance(v, str) for v in values)
                                or len(values) != len(set(values))
                            ):
                                raise ValueError(f"报告 {key} 非法或重复")
                        if set(report["selected"]) != set(report["tests"]) or not (
                            set(report["selected"]) <= set(report["collected"])
                        ):
                            raise ValueError("收集、选中与执行的用例不一致")
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
                        raise ValueError("逐 rank 报告缺少选集或计时字段")
                    duration = float(report.get("duration_seconds", 0))
                    if not math.isfinite(duration) or duration < 0:
                        raise ValueError("报告耗时非法")
                    phase_seconds = max(phase_seconds, duration)
                    collection = float(report.get("collection_seconds", 0))
                    if not math.isfinite(collection) or not 0 <= collection <= duration:
                        raise ValueError("非法 collection 耗时")
                    collection_seconds = max(collection_seconds, collection)
                    node_durations = report.get("durations", {})
                    if not isinstance(node_durations, dict):
                        raise ValueError("非法用例耗时映射")
                    if require_timing and set(node_durations) != set(report["tests"]):
                        raise ValueError("逐 rank 用例耗时不完整")
                    for nodeid, seconds in node_durations.items():
                        if (
                            nodeid not in report["tests"]
                            or type(seconds) not in (int, float)
                            or not math.isfinite(seconds)
                            or seconds < 0
                        ):
                            raise ValueError("非法用例耗时")
                        key = f"{iteration}:{phase}:{nodeid}"
                        durations[key] = max(durations.get(key, 0), seconds)
                    for nodeid, outcome in report["tests"].items():
                        if not isinstance(nodeid, str) or outcome not in OUTCOME_ORDER:
                            raise ValueError("非法 nodeid 或测试结果")
                        if case.get("strict_selection") and not any(
                            selector_matches(selector, nodeid) for selector in case["args"]
                        ):
                            raise ValueError(f"执行了清单以外的用例: {nodeid}")
                        if any(
                            selector_matches(selector, nodeid)
                            for selector in case.get("excluded", {})
                        ):
                            raise ValueError(f"执行了明确排除的用例: {nodeid}")
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
            errors.append(f"{case['id']}: 排除 selector 未收集到用例: {selector}")
    executed = {key.split(":", 2)[2] for key, value in outcomes.items() if value != "skipped"}
    if not executed:
        errors.append(f"{case['id']}: 整组没有实际执行用例")
    if case.get("strict_selection"):
        for selector in case["args"]:
            if not any(selector_matches(selector, nodeid) for nodeid in collected):
                errors.append(f"{case['id']}: selector 未收集到用例: {selector}")
            elif not any(selector_matches(selector, nodeid) for nodeid in selected):
                errors.append(f"{case['id']}: selector 完全被过滤: {selector}")
            elif not any(selector_matches(selector, nodeid) for nodeid in executed):
                errors.append(f"{case['id']}: selector 全部 skipped: {selector}")
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
            raise ValueError("非法运行耗时")
        if timing["completed_at"] - timing["started_at"] != timing["elapsed_seconds"]:
            raise ValueError("运行时间戳不一致")
        if suite:
            if not timing["started_at"] <= timing["prepared_at"] <= timing["completed_at"]:
                raise ValueError("环境准备时间戳越界")
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
        raise ValueError("manifest 门禁与触发场景不一致")
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
                raise ValueError("bucket 执行未完整结束")
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
                group_errors.append("全量基线缺少完整逐用例耗时")
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
        errors.append("至少一个调度或产物上传作业未成功")
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
        f"UT ({result['scope']}): {result['passed']}/{result['total']} = {result['pass_rate']:.2%}，"
        f"阈值 {result['threshold_percent']}%；失败 {result['failed']}，跳过 {result['skipped']}；"
        f"结论：{'通过' if result['success'] else '未通过'}\n"
    )
    if result["scope"] == "diagnostic":
        text += "仅单 bucket 诊断，不代表全量通过，不作为 quick 全量基线或 PR 成功门禁。\n"
    if result["identity"]:
        text += f"提交身份: {result['identity']}\n"
    text += f"触发: {result.get('trigger_kind')}；全集: {result.get('suite_family')}；来源: {result.get('provenance', {})}\n"
    for case_id, group in result["groups"].items():
        text += (
            f"{case_id}: {group['counts']}，实际执行 {group['executed']}，"
            f"pytest 阶段耗时 {group['duration_seconds']:.1f}s"
        )
        if group["timing"]:
            text += f"，分组墙钟 {group['timing']['elapsed_seconds']:.1f}s"
        if group["suite_timing"]:
            text += f"，环境准备 {group['suite_timing']['prepare_seconds']:.1f}s"
        text += f"，排除 nodeid {group['excluded_count']} 个\n"
        for selector, reason in group["excluded"].items():
            text += f"  明确排除：{selector}；原因：{reason}\n"
    if result["timings"]:
        text += f"Pod 内计时（不含上传、hosted 配置与汇总）: {result['timings']}\n"
    if result["errors"]:
        text += f"执行或报告异常 {len(result['errors'])} 项，详见 summary.json。\n"
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
            raise ValueError(f"构建产物路径越界: {name}")
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"构建产物目录不能是符号链接: {name}")
            continue
        if not path.is_file():
            raise ValueError(f"构建产物不是普通文件: {name}")
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
            raise ValueError(f"缺少或为空的构建产物: {name}")
    return inventory


def seal_bundle(directory: Path) -> None:
    inventory = bundle_inventory(directory)
    (directory / "bundle-integrity.json").write_text(
        json.dumps({"schema_version": 1, "files": inventory}, sort_keys=True), encoding="utf-8"
    )


def verify_bundle(directory: Path, plan: Path, digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("缺少可信 Build 作业提供的产物摘要")
    raw = (directory / "bundle-integrity.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("构建产物清单与可信 Build 摘要不一致")
    sealed = json.loads(raw)
    if sealed.get("schema_version") != 1 or sealed.get("files") != bundle_inventory(directory):
        raise ValueError("构建产物缺失、截断或内容校验失败")
    if (directory / "manifest.json").read_bytes() != (plan / "manifest.json").read_bytes():
        raise ValueError("构建产物与可信测试计划不一致")
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
            raise ValueError(f"构建产物提交身份不一致: {filename}")
    trusted = {
        path.relative_to(plan).as_posix(): path.read_bytes()
        for path in (plan / "control").rglob("*")
        if path.is_file()
    }
    actual = {name for name in sealed["files"] if name.startswith("control/")}
    if not trusted or actual != set(trusted):
        raise ValueError("构建产物控制脚本清单不一致")
    for name, content in trusted.items():
        if (directory / name).read_bytes() != content:
            raise ValueError(f"构建产物控制脚本与可信计划不一致: {name}")
    print(f"构建产物完整性校验通过: {len(sealed['files'])} 个文件，target={identity['target_sha']}")


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
            print("诊断运行：UT 已跳过，不提供 PR 成功门禁。")
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
