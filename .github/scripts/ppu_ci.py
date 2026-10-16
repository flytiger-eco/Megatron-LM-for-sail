"""PPU Build/UT."""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = ("dev", "lts")
OUTCOME_ORDER = {"skipped": 0, "passed": 1, "failed": 2}


def positive(value: object, name: str, maximum: int = 10000) -> int:
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", text) or int(text) > maximum:
        raise ValueError(f"{name} 必须为 1..{maximum} 的整数")
    return int(text)


def select_config(event: dict, event_name: str, inputs: dict) -> dict:
    if event_name not in {
        "pull_request",
        "merge_group",
        "push",
        "schedule",
        "workflow_dispatch",
        "repository_dispatch",
    }:
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
    if event_name in {"pull_request", "merge_group"}:
        config.update(unit_test=True, repeat=1, timeout=30)
    if event_name == "push":
        config["unit_test"] = (
            config["unit_test"] and event.get("ref") == "refs/heads/ci/ppu-validation"
        )
    if event_name == "pull_request":
        pr = event["pull_request"]
        if pr["head"]["repo"]["full_name"] != pr["base"]["repo"]["full_name"]:
            config["build"] = False
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


def generate(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = unit_cases(config) if config["unit_test"] else {}
    manifest = {"config": config, "unit": cases}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    keys = ("id", "environment", "bucket", "cards", "action_minutes", "job_minutes")
    rows = [{key: row[key] for key in keys} for row in cases.values()]
    if len(rows) > 256:
        raise ValueError("UT 超过 GitHub 的 256 个 matrix job 上限")
    emit("unit_matrix", json.dumps({"include": rows or [{"id": "unused"}]}))
    return manifest


def case_results(case: dict, directory: Path) -> dict:
    outcomes, errors = {}, []
    for iteration in range(1, case["repeat"] + 1):
        for phase in ("prod", "experimental"):
            for rank in range(case["cards"]):
                filename = f"{phase}-{iteration}-rank-{rank}.json"
                try:
                    report = json.loads((directory / filename).read_text())
                    if (
                        report["rank"] != rank
                        or report["complete"] is not True
                        or report["collection_errors"]
                    ):
                        raise ValueError("采集错误、未完成或 rank 不匹配")
                    if report["exitstatus"] not in (0, 1, 5):
                        raise ValueError(f"异常 pytest 退出状态: {report['exitstatus']}")
                    if not isinstance(report["tests"], dict):
                        raise ValueError("测试结果不是用例映射")
                    for nodeid, outcome in report["tests"].items():
                        key = f"{iteration}:{phase}:{nodeid}"
                        if OUTCOME_ORDER[outcome] > OUTCOME_ORDER.get(outcomes.get(key), -1):
                            outcomes[key] = outcome
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{case['id']}/{filename}: {exc}")
    return {"outcomes": outcomes, "errors": errors, "complete": not errors}


def summarize(manifest: dict, directory: Path) -> dict:
    counts = dict.fromkeys(OUTCOME_ORDER, 0)
    errors, failures = [], []
    for case in manifest["unit"].values():
        result_dir = directory / f"ppu-ut-{case['id']}"
        result = case_results(case, result_dir)
        errors.extend(result["errors"])
        try:
            status = json.loads((result_dir / "status.json").read_text())
            if status["exit_code"] != 0 or not status["complete"]:
                raise ValueError("bucket 执行未完整结束")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{case['id']}: {exc}")
        for nodeid, outcome in result["outcomes"].items():
            counts[outcome] += 1
            if outcome == "failed":
                failures.append(f"{case['id']}:{nodeid}")
    total = counts["passed"] + counts["failed"]
    if os.environ.get("PPU_MATRIX_RESULT", "success") != "success":
        errors.append("至少一个调度或产物上传作业未成功")
    success = not errors and total > 0 and counts["passed"] * 5 >= total * 4
    return {
        **counts,
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
        f"UT: {result['passed']}/{result['total']} = {result['pass_rate']:.2%}，"
        f"阈值 80%；失败 {result['failed']}，跳过 {result['skipped']}；"
        f"结论：{'通过' if result['success'] else '未通过'}\n"
    )
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
        inputs = json.loads(os.environ.get("PPU_DEFAULTS", "{}"))
        inputs.update(
            event.get("client_payload", {})
            if event_name == "repository_dispatch"
            else json.loads(os.environ.get("PPU_INPUTS", "{}"))
        )
        config = select_config(event, event_name, inputs)
        emit("config", json.dumps(config))
        emit("build", config["build"])
        emit("unit_test", config["unit_test"])
        generate(config, args.output)
    else:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if args.mode == "case-result":
            result = case_results(manifest["unit"][args.case_id], args.results)
            result["exit_code"] = args.runner_exit
            result["complete"] = result["complete"] and args.runner_exit == 0
            (args.results / "status.json").write_text(json.dumps(result), encoding="utf-8")
            raise SystemExit(0 if result["complete"] else 1)
        result = summarize(manifest, args.results)
        write_summary(result, args.results)
        raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
