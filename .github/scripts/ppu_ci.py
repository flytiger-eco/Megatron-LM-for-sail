"""PPU Build/UT."""

import argparse
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


def positive(value: object, name: str, maximum: int = 10000) -> int:
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", text) or int(text) > maximum:
        raise ValueError(f"{name} 必须为 1..{maximum} 的整数")
    return int(text)


def select_config(event: dict, event_name: str, inputs: dict) -> dict:
    if event_name not in {"pull_request_target", "push", "workflow_dispatch"}:
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
    scope = inputs.get("scope", "quick")
    if scope not in ("quick", "full"):
        raise ValueError("scope 必须为 quick 或 full")
    ref = event.get("ref", "")
    if event_name == "push":
        if ref.startswith("refs/tags/v"):
            scope = "full"
            config.update(build=True, unit_test=True, repeat=1)
        elif ref == "refs/heads/ci/ppu-validation":
            scope = "quick"
            config.update(build=True, unit_test=True, repeat=1)
        else:
            raise ValueError("不支持的 push ref")
    if event_name == "pull_request_target":
        scope = "quick"
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
        number, sha = inputs.get("pr_number", ""), inputs.get("pr_sha", "")
        if bool(number) != bool(sha):
            raise ValueError("PR 编号与完整 SHA 必须同时填写")
        if number:
            positive(number, "PR number", 10**9)
            if not re.fullmatch(r"[0-9a-fA-F]{40}", str(sha)):
                raise ValueError("PR SHA 必须为完整的 40 位十六进制字符串")
            scope = "quick"
            config.update(build=True, unit_test=True, repeat=1)
    if scope == "quick":
        config["repeat"] = 1
    config["scope"] = scope
    config["unit_test"] = config["unit_test"] and config["build"] and config["repeat"] > 0
    return config


def load_bucket_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ppu_bucket_helper", ROOT / "tests/unit_tests/find_test_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit_cases(config: dict) -> dict:
    import yaml

    sys.path.insert(0, str(ROOT))
    from tests.test_utils.python_scripts import recipe_parser

    helper = load_bucket_helper()
    recipe = yaml.safe_load((ROOT / "tests/test_utils/recipes/h100/unit-tests.yaml").read_text())
    buckets = [bucket for product in recipe["products"] for bucket in product["test_case"]]
    cases = {}
    for environment in ENVIRONMENTS:
        workloads = recipe_parser.load_workloads(
            container_tag="ppu-ci",
            container_image="unused",
            scope="unit-tests",
            environment=environment,
            tag="latest",
            platform="dgx_h100",
            n_repeat=config["repeat"],
            time_limit=config["timeout"] * 60,
        )
        specs = [workload.spec for workload in workloads if workload.type != "build"]
        if not specs:
            raise ValueError(f"{environment} UT recipe 为空")
        for index, spec in enumerate(sorted(specs, key=lambda row: row["test_case"])):
            bucket = spec["test_case"]
            if not bucket.startswith("tests/unit_tests/") or ".." in bucket.split("/"):
                raise ValueError(f"不支持的 UT bucket，不能静默跳过: {bucket}")
            target = helper.get_base_path(bucket)
            if not (ROOT / target).exists():
                raise ValueError(f"UT 路径不存在: {target}")
            if int(spec["nodes"]) != 1 or int(spec["gpus"]) != 8:
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
            case_id = f"{environment}-{index:03d}"
            minutes = (int(spec["time_limit"]) + 59) // 60
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
            }
    return cases


def selector_matches(selector: str, nodeid: str) -> bool:
    return nodeid == selector or nodeid.startswith((selector + "::", selector + "["))


def quick_cases(config: dict) -> dict:
    recipe = json.loads((CONTROL_ROOT / ".github/ppu-pr-tests.json").read_text())
    if recipe.get("schema_version") != 1 or not recipe.get("groups"):
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
            if any(
                selector_matches(old, selector) or selector_matches(selector, old) for old in seen
            ):
                raise ValueError(f"重复或重叠 selector: {selector}")
            seen.append(selector)
            files.add(filename)
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
    manifest = {"schema_version": 2, "config": config, "identity": identity or {}, "unit": cases}
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


def case_results(case: dict, directory: Path, identity: dict | None = None) -> dict:
    outcomes, errors, collected, selected = {}, [], set(), set()
    elapsed = 0.0
    for iteration in range(1, case["repeat"] + 1):
        for phase in ("prod", "experimental"):
            phase_seconds = 0.0
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
                    duration = float(report.get("duration_seconds", 0))
                    if not math.isfinite(duration) or duration < 0:
                        raise ValueError("报告耗时非法")
                    phase_seconds = max(phase_seconds, duration)
                    for nodeid, outcome in report["tests"].items():
                        if not isinstance(nodeid, str) or outcome not in OUTCOME_ORDER:
                            raise ValueError("非法 nodeid 或测试结果")
                        if case.get("strict_selection") and not any(
                            selector_matches(selector, nodeid) for selector in case["args"]
                        ):
                            raise ValueError(f"执行了清单以外的用例: {nodeid}")
                        key = f"{iteration}:{phase}:{nodeid}"
                        if OUTCOME_ORDER[outcome] > OUTCOME_ORDER.get(outcomes.get(key), -1):
                            outcomes[key] = outcome
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{case['id']}/{filename}: {exc}")
            elapsed += phase_seconds
    if case.get("strict_selection"):
        executed = {key.split(":", 2)[2] for key, value in outcomes.items() if value != "skipped"}
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
    counts = dict.fromkeys(OUTCOME_ORDER, 0)
    errors, failures, groups = [], [], {}
    for case in manifest["unit"].values():
        result_dir = result_directory(manifest, directory, case["id"])
        result = case_results(case, result_dir, manifest.get("identity"))
        group_errors = result["errors"]
        timing_path = result_dir / "timing.json"
        timing = (
            read_timing(timing_path, group_errors)
            if manifest["config"].get("scope") == "quick" or timing_path.exists()
            else {}
        )
        try:
            status = json.loads((result_dir / "status.json").read_text())
            if status["exit_code"] != 0 or status["complete"] is not True:
                raise ValueError("bucket 执行未完整结束")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            group_errors.append(f"{case['id']}: {exc}")
        errors.extend(f"{case['id']}: {error}" for error in group_errors)
        group_counts = dict.fromkeys(OUTCOME_ORDER, 0)
        groups[case["id"]] = {
            "counts": group_counts,
            "duration_seconds": result["duration_seconds"],
            "timing": timing,
            "errors": group_errors,
        }
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
    total = counts["passed"] + counts["failed"]
    if os.environ.get("PPU_MATRIX_RESULT", "success") != "success":
        errors.append("至少一个调度或产物上传作业未成功")
    success = not errors and total > 0 and counts["passed"] * 5 >= total * 4
    return {
        **counts,
        "scope": manifest["config"].get("scope", "full"),
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
        f"阈值 80%；失败 {result['failed']}，跳过 {result['skipped']}；"
        f"结论：{'通过' if result['success'] else '未通过'}\n"
    )
    if result["identity"]:
        text += f"待测 SHA: {result['identity']['target_sha']}；控制 SHA: {result['identity']['control_sha']}\n"
    for case_id, group in result["groups"].items():
        text += (
            f"{case_id}: {group['counts']}，实际执行 {group['executed']}，"
            f"pytest 阶段耗时 {group['duration_seconds']:.1f}s"
        )
        if group["timing"]:
            text += f"，分组墙钟 {group['timing']['elapsed_seconds']:.1f}s"
        text += "\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["configure", "summarize", "case-result"])
    parser.add_argument("--output", type=Path, default=Path(".ppu-ci"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--case-id")
    parser.add_argument("--runner-exit", type=int, default=0)
    args = parser.parse_args()
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
        identity = {key: authorization[key] for key in ("target_sha", "control_sha")}
        emit("config", json.dumps(config))
        emit("build", config["build"])
        emit("unit_test", config["unit_test"])
        emit("scope", config["scope"])
        if not config["unit_test"]:
            print("诊断运行：UT 已跳过，不提供 PR 成功门禁。")
        generate(config, args.output, identity)
    else:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if args.mode == "case-result":
            result = case_results(
                manifest["unit"][args.case_id], args.results, manifest.get("identity")
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
