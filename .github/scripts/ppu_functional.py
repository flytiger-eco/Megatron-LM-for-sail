"""Trusted functional selection, baseline validation and read-only asset preflight."""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
from pathlib import Path

BRANCH = "refs/heads/ci/ppu-validation"
BASELINES = Path("tests/functional_tests/zhenwu_test_cases")
LOCK = Path(".github/ppu-functional-baselines.json")
POD_ASSETS = "/mnt/wl_nas/anyunge.ayg/megatron-functional-ci/artifacts-base"
CPU_ASSETS = "/wl_nas/anyunge.ayg/megatron-functional-ci/artifacts-base"
IMAGE = "pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001"
TIERS = ("L0", "L1", "L2", "L3")
SKIPPED_CASES = {
    "gpt/gpt3_mcore_te_tp2_pp1_modelopt_distill_resume": "teacher_checkpoint_unavailable"
}
DEFAULTS = {
    "functional_mode": "assets",
    "functional_scope": "all",
    "functional_cases": "all",
    "functional_repeat": "5",
    "functional_timeout": "30",
}
ENTRYPOINTS = {
    "bert": "examples/bert/pretrain_bert.py",
    "t5": "examples/t5/pretrain_t5.py",
    "gpt": "pretrain_gpt.py",
    "moe": "pretrain_gpt.py",
    "mamba": "pretrain_hybrid.py",
    "flextron": "megatron/elastification/pretrain_hybrid_flex.py",
    "multimodal-llava": "pretrain_vlm.py",
}
OUTPUT_VARIABLES = {
    "CHECKPOINT_SAVE_PATH",
    "CHECKPOINT_LOAD_PATH",
    "TENSORBOARD_PATH",
    "DATA_CACHE_PATH",
    "OUTPUT_PATH",
}
INPUT_ARGUMENTS = {
    "--data-path": "prefix",
    "--train-data-path": "prefix",
    "--valid-data-path": "prefix",
    "--test-data-path": "prefix",
    "--vocab-file": "file",
    "--merge-file": "file",
    "--tokenizer-model": "file",
    "--export-kd-teacher-load": "directory",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf = module_from_path("ppu_functional_hf", Path(__file__).with_name("ppu_functional_hf.py"))


def positive(value, name: str, maximum: int) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", str(value)) or int(value) > maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return int(value)


def enabled(value) -> bool:
    return str(value).lower() in ("true", "1")


def validate_route(event_name: str, ref: str, inputs: dict) -> str:
    kind = inputs.get("test_kind", "unit")
    if kind not in ("unit", "functional"):
        raise ValueError("test_kind must be unit or functional")
    diagnostic = str(inputs.get("compiler_diagnostic", False)).lower()
    if diagnostic not in ("true", "false"):
        raise ValueError("compiler_diagnostic must be a boolean")
    diagnostic_context = inputs.get("compiler_diagnostic_context", "bda")
    if diagnostic_context not in ("bda", "bert"):
        raise ValueError("compiler_diagnostic_context must be bda or bert")
    if diagnostic_context != "bda" and diagnostic != "true":
        raise ValueError("BERT context requires explicit compiler diagnostics")
    if diagnostic == "true" and kind != "functional":
        raise ValueError("Compiler diagnostics require test_kind=functional")
    if kind == "unit":
        if any(str(inputs.get(key, default)) != default for key, default in DEFAULTS.items()):
            raise ValueError("Functional inputs require test_kind=functional")
        return kind
    if event_name != "workflow_dispatch" or ref != BRANCH:
        raise ValueError("Functional tests require manual dispatch on the validation branch")
    if any(inputs.get(key) for key in ("pr_number", "pr_sha", "pr_base_sha")):
        raise ValueError("Functional tests cannot authorize or publish PR status")
    unit_defaults = {
        "scope": "auto",
        "suite_family": "nvidia-dev",
        "unit_bucket": "",
        "repeat": "1",
        "timeout": "30",
    }
    if any(str(inputs.get(key, default)) != default for key, default in unit_defaults.items()):
        raise ValueError("Functional requests cannot contain non-default UT selection inputs")
    for key in ("build", "unit_test"):
        if key in inputs and str(inputs[key]).lower() not in ("true", "yes"):
            raise ValueError("Functional mode controls Build and UT; do not set UT switches")
    if diagnostic == "true":
        if any(str(inputs.get(key, default)) != default for key, default in DEFAULTS.items()):
            raise ValueError("Compiler diagnostics cannot contain functional selection inputs")
        return "compiler_diagnostic"
    return kind


def select_config(inputs: dict) -> dict:
    if (
        str(inputs.get("compiler_diagnostic", False)).lower() != "false"
        or inputs.get("compiler_diagnostic_context", "bda") != "bda"
    ):
        raise ValueError("Compiler diagnostics must not enter the functional execution chain")
    mode = inputs.get("functional_mode", "assets")
    scope = inputs.get("functional_scope", "all")
    if mode not in ("assets", "regression") or scope not in ("all", *TIERS):
        raise ValueError("Invalid functional mode or scope")
    requested = inputs.get("functional_cases", "all")
    if not isinstance(requested, str) or not requested:
        raise ValueError("functional_cases must be all or full model/case names")
    names = requested.split(",")
    if requested != "all" and (
        len(names) != len(set(names))
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", name) for name in names)
    ):
        raise ValueError("Invalid or duplicate functional case request")
    return {
        "test_kind": "functional",
        "functional_mode": mode,
        "scope": scope,
        "requested_cases": names,
        "repeat": positive(inputs.get("functional_repeat", 5), "functional_repeat", 10),
        "timeout": positive(inputs.get("functional_timeout", 30), "functional_timeout", 1380),
        "build": mode == "regression",
        "unit_test": False,
        "threshold_percent": 100,
        "trigger_kind": "manual",
        "suite_family": "ppu-functional",
        "image": IMAGE,
        "data_path": POD_ASSETS,
        "cpu_data_path": CPU_ASSETS,
    }


def safe_file(root: Path, relative: str | Path) -> Path:
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe source file: {relative}")
    return path


def load_recipes(control: Path) -> list:
    import yaml

    parser = module_from_path(
        "ppu_functional_recipe_parser", control / "tests/test_utils/python_scripts/recipe_parser.py"
    )
    rows = []
    for path in sorted((control / "tests/test_utils/recipes/h100").glob("*.yaml")):
        recipe = yaml.safe_load(path.read_text())
        if recipe.get("format_version") != 1 or recipe.get("type") != "basic":
            raise ValueError(f"Unsupported recipe: {path.name}")
        if "tests/functional_tests/shell_test_utils/run_ci_test.sh" not in recipe.get(
            "spec", {}
        ).get("script", ""):
            continue
        for product in recipe.get("products", []):
            if len(product.get("test_case", [])) != 1 or not product.get("products"):
                raise ValueError(f"Recipe requires explicit products adaptation: {path.name}")
        for row in parser.filter_by_environment(parser.load_and_flatten(str(path)), "dev"):
            spec = dict(row.spec)
            if spec.get("platforms") != "dgx_h100":
                continue
            spec["recipe_path"] = path.relative_to(control).as_posix()
            spec["recipe_sha256"] = sha256(path)
            rows.append(spec)
    return rows


def input_assets(config: dict) -> list:
    """Resolve known path variables only; never execute BEFORE_SCRIPT or shell text."""
    requirements = {}
    for section, arguments in config.items():
        if not re.fullmatch(r"MODEL_ARGS(?:_\d+)?", section):
            continue
        if not isinstance(arguments, dict):
            raise ValueError(f"{section} must be a mapping")
        for argument, value in arguments.items():
            if not isinstance(value, str):
                continue
            variables = set(
                re.findall(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))", value)
            )
            variables = {left or right for left, right in variables}
            if variables - OUTPUT_VARIABLES - {"DATA_PATH"}:
                raise ValueError(f"Unresolved asset variable in {section}.{argument}: {value}")
            if argument not in INPUT_ARGUMENTS:
                if "DATA_PATH" in variables:
                    raise ValueError(f"Unknown asset argument: {section}.{argument}")
                continue
            if argument.endswith("data-path") and enabled(arguments.get("--mock-data", False)):
                continue
            values = shlex.split(value)
            for token in values:
                if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
                    continue
                match = re.fullmatch(r"\$(?:\{DATA_PATH\}|DATA_PATH)/([A-Za-z0-9_./-]+)", token)
                if not match or ".." in Path(match[1]).parts:
                    raise ValueError(
                        f"Asset requires explicit path mapping: {section}.{argument}={token}"
                    )
                relative = match[1]
                kind = INPUT_ARGUMENTS[argument]
                for suffix in ((".bin", ".idx") if kind == "prefix" else ("",)):
                    key = (relative + suffix, "file" if kind == "prefix" else kind)
                    requirements.setdefault(key, set()).add(f"{section}.{argument}")
    return [
        {"relative_path": path, "kind": kind, "purposes": sorted(purposes)}
        for (path, kind), purposes in sorted(requirements.items())
    ]


def tokenizer_dependencies(root: Path, config: dict, entrypoint: str) -> list:
    """静态读取原入口默认值及 HF 仓库名；不导入训练栈、不探测或下载缓存。"""
    defaults = []
    for node in ast.walk(ast.parse(safe_file(root, entrypoint).read_text())):
        if isinstance(node, ast.keyword) and node.arg == "args_defaults":
            value = ast.literal_eval(node.value)
            if isinstance(value, dict) and "tokenizer_type" in value:
                defaults.append(value["tokenizer_type"])
    if len(set(defaults)) != 1:
        raise ValueError(f"Ambiguous original tokenizer default: {entrypoint}")
    source = "megatron/core/tokenizers/text/libraries/megatron_hf_tokenizer.py"
    maps = []
    for node in ast.parse(safe_file(root, source).read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MEGATRON_CONFIG_MAP"
            for target in node.targets
        ):
            maps.append(ast.literal_eval(node.value))
    if len(maps) != 1 or not isinstance(maps[0], dict):
        raise ValueError("Unable to read original Megatron tokenizer repository mapping")
    sections = ["MODEL_ARGS"]
    if config["TEST_TYPE"] == "ckpt-resume" and "MODEL_ARGS_2" in config:
        sections.append("MODEL_ARGS_2")
    requirements = []
    for section in sections:
        arguments = config[section]
        kind = arguments.get("--tokenizer-type", defaults[0])
        if kind in ("NullTokenizer", "NullMultimodalTokenizer"):
            continue
        if kind not in maps[0]:
            raise ValueError(f"Tokenizer assets require explicit adaptation: {section}/{kind}")
        # 提供 vocab/merges 后原实现仍调用 AutoTokenizer.from_pretrained(repository)。
        # 固定来源由可信 control 的独立锁文件补充；显式 vocab 就绪不代表缓存就绪。
        requirements.append(
            {
                "section": section,
                "tokenizer_type": kind,
                "repository": maps[0][kind]["tokenizer_name"],
                "purpose": f"{section}.--tokenizer-type (original AutoTokenizer metadata)",
                "status": "source_mapping_required",
            }
        )
    return requirements


def nvrx_checkpoint_requirements(config: dict) -> list:
    """仅描述原配置实际执行阶段的外部 checkpoint 依赖，不改写训练参数。"""
    sections = ["MODEL_ARGS"]
    if config["TEST_TYPE"] == "ckpt-resume" and "MODEL_ARGS_2" in config:
        # 原 shell 将第二阶段参数整组替换，不与第一阶段合并。
        sections.append("MODEL_ARGS_2")
    requirements = []
    for section in sections:
        arguments = config[section]
        if not isinstance(arguments, dict):
            raise ValueError(f"{section} must be a mapping")
        strategy = arguments.get("--async-strategy", "nvrx")
        if strategy not in ("nvrx", "mcore"):
            raise ValueError(f"Unsupported async checkpoint strategy: {section}/{strategy}")
        # 上游在未启用 async-save 时会选择 mcore，无须安装 NVRx。
        if enabled(arguments.get("--async-save", False)) and strategy == "nvrx":
            requirements.append(
                {
                    "section": section,
                    "persistent_worker": enabled(
                        arguments.get("--use-persistent-ckpt-worker", False)
                    ),
                    "cpu_shm": enabled(arguments.get("--async-ckpt-use-cpu-shm", False)),
                }
            )
    return requirements


def read_metrics(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate golden key: {key}")
            result[key] = value
        return result

    content = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(content, dict) or not content:
        raise ValueError(f"Empty or malformed golden: {path}")
    result = {}
    for name, metric in content.items():
        if (
            not isinstance(metric, dict)
            or not isinstance(metric.get("values"), dict)
            or not metric["values"]
        ):
            raise ValueError(f"Invalid metric structure: {path}/{name}")
        start = positive(metric.get("start_step"), "start_step", 10**9)
        end = positive(metric.get("end_step"), "end_step", 10**9)
        interval = positive(metric.get("step_interval"), "step_interval", 10**9)
        precision = metric.get("value_precision", "rounded_5_decimal_places")
        if precision not in ("rounded_5_decimal_places", "full") or start > end:
            raise ValueError(f"Invalid metric metadata: {path}/{name}")
        values = {}
        for key, value in metric["values"].items():
            step = positive(key, "metric step", 10**9)
            if not start <= step <= end or type(value) not in (float, int, str):
                raise ValueError(f"Invalid sample: {path}/{name}/{key}")
            if isinstance(value, str) and value.lower() not in ("nan", "inf", "-inf"):
                raise ValueError(f"Invalid sample value: {path}/{name}/{key}")
            values[step] = value
        grid = {start} | {step for step in range(start + 1, end + 1) if step % interval == 0}
        if set(values) != grid:
            raise ValueError(
                f"Metric sampling grid has missing or unexpected points: {path}/{name}"
            )
        result[name] = {
            "start_step": start,
            "end_step": end,
            "step_interval": interval,
            "value_precision": precision,
            "values": values,
        }
    return result


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def comparison_steps(name: str, values: dict) -> list:
    """Mirror only the upstream comparison region, never its numeric tolerances."""
    if name != "iteration-time":
        return sorted(values)
    steps = sorted(step for step, value in values.items() if finite(value))
    if not steps:
        raise ValueError("iteration-time has no finite samples")
    window = range(5, 21) if steps[-1] <= 25 else range(30, 46)
    selected = [step for step in steps if step in window]
    selected = selected or [step for step in steps if step >= window.start][:4]
    if not selected:
        raise ValueError("iteration-time has no samples after warmup")
    return selected


def validate_region(reference: dict, actual: dict, metrics: list, *, after: int = 0) -> dict:
    regions = {}
    for name in metrics:
        if name not in reference or name not in actual:
            raise ValueError(f"Required metric missing: {name}")
        values = {step: value for step, value in reference[name]["values"].items() if step > after}
        steps = comparison_steps(name, values)
        if not steps:
            raise ValueError(f"Empty comparison region: {name}")
        for step in steps:
            if not finite(values[step]) or not finite(actual[name]["values"].get(step)):
                raise ValueError(f"Missing or non-finite comparison sample: {name}/{step}")
        regions[name] = steps
    return regions


def supported_metrics(control: Path) -> set:
    path = control / "tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CHECK_THRESHOLDS"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                return {ast.literal_eval(key) for key in node.value.keys}
    raise ValueError("Unable to read upstream CHECK_THRESHOLDS metric names")


def describe_case(root: Path, name: str, specs: list, config: dict) -> dict:
    import yaml

    model, case = name.split("/")
    normalized = []
    for spec in specs:
        normalized.append(
            {key: value for key, value in spec.items() if key not in ("scope", "cadence")}
        )
    if any(row != normalized[0] for row in normalized[1:]):
        raise ValueError(f"Conflicting recipes for {name}/dev")
    spec = specs[0]
    entries = re.findall(r'TRAINING_SCRIPT_PATH=([^\s"\']+)', spec.get("script", ""))
    if len(entries) != 1 or entries[0] != ENTRYPOINTS.get(Path(spec["recipe_path"]).stem):
        raise ValueError(f"Training entrypoint requires explicit adaptation: {name}: {entries}")
    safe_file(root, entries[0])
    directory = BASELINES / name
    path = safe_file(root, directory / "model_config.yaml")
    model_config = yaml.safe_load(path.read_text())
    if not isinstance(model_config, dict) or not isinstance(model_config.get("MODEL_ARGS"), dict):
        raise ValueError(f"Invalid PPU model configuration: {name}")
    test_type = model_config.get("TEST_TYPE")
    if (
        test_type not in ("regular", "ckpt-resume")
        or model_config.get("MODE", "pretraining") != "pretraining"
    ):
        raise ValueError(f"Unsupported PPU test type: {name}/{test_type}")
    if model_config.get("TEST_EVALUATION", "pass") != "pass":
        raise ValueError(f"Expected failures cannot count as PPU regression: {name}")
    env = model_config.get("ENV_VARS", {})
    lightweight = enabled(env.get("ENABLE_LIGHTWEIGHT_MODE", False))
    skip = enabled(env.get("SKIP_PYTEST", False))
    observation = "lightweight" if lightweight else "SKIP_PYTEST" if skip else ""
    repeat = 1 if observation else config["repeat"]
    if "N_REPEAT" in env and int(env["N_REPEAT"]) != repeat:
        raise ValueError(f"PPU configuration overrides required repeats: {name}")
    iters = positive(model_config["MODEL_ARGS"].get("--train-iters"), "train-iters", 10**9)
    metrics = model_config.get("METRICS", ["lm loss", "num-zeros"])
    if not isinstance(metrics, list) or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError(f"Invalid metric selection: {name}")
    golden_path = safe_file(root, directory / "golden_values_dev_ppu.json")
    golden = read_metrics(golden_path)
    regions = {} if observation else validate_region(golden, golden, metrics)
    if not observation and any(
        golden[metric]["end_step"] < iters for metric in metrics if metric != "iteration-time"
    ):
        raise ValueError(f"Golden ends before configured training completes: {name}")
    second = root / directory / "golden_values_dev_ppu_2nd.json"
    if second.exists():
        read_metrics(safe_file(root, second.relative_to(root)))
    seconds = positive(
        spec.get("time_limit_override", config["timeout"] * 60),
        "functional budget seconds",
        1380 * 60,
    )
    launcher = model_config.get("LAUNCHER", "torchrun")
    if launcher not in ("torchrun", "ft_launcher"):
        raise ValueError(f"Unsupported launcher: {name}/{launcher}")
    return {
        "name": name,
        "model": model,
        "case": case,
        "environment": "dev",
        "cards": 8,
        "upstream_nodes": spec.get("nodes"),
        "upstream_gpus": spec.get("gpus"),
        "recipe_path": spec["recipe_path"],
        "recipe_sha256": spec["recipe_sha256"],
        "scopes": sorted({row["scope"] for row in specs}),
        "cadence": sorted({cadence for row in specs for cadence in row["cadence"]}),
        "training_script": entries[0],
        "training_script_sha256": sha256(root / entries[0]),
        "config_path": path.relative_to(root).as_posix(),
        "config_sha256": sha256(path),
        "golden_path": golden_path.relative_to(root).as_posix(),
        "golden_sha256": sha256(golden_path),
        "second_golden_path": second.relative_to(root).as_posix() if second.exists() else None,
        "second_golden_sha256": sha256(second) if second.exists() else None,
        "second_golden_role": "provenance_only_not_resume_reference",
        "test_type": test_type,
        "train_iters": iters,
        "effective_train_iters": 4 if lightweight else iters,
        "repeat": repeat,
        "requested_repeat": config["repeat"],
        "observation": observation,
        "metrics": metrics,
        "comparison_regions": regions,
        "value_precision": {
            metric: golden[metric]["value_precision"] for metric in metrics if metric in golden
        },
        "approximate": enabled(env.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", 0))
        or enabled(env.get("NON_DETERMINSTIC_RESULTS", 0)),
        "launcher": launcher,
        "launcher_internal_max_restarts": 3 if launcher == "ft_launcher" else 0,
        "nvrx_checkpoint": nvrx_checkpoint_requirements(model_config),
        "tokenizer_dependencies": tokenizer_dependencies(root, model_config, entries[0]),
        "budget_seconds": seconds,
        "action_minutes": math.ceil(seconds / 60) + 30,
        "job_minutes": math.ceil(seconds / 60) + 45,
        "max_attempts": 3,
        "assets": input_assets(model_config),
    }


def catalogue(root: Path, control: Path, config: dict) -> dict:
    snapshot = module_from_path(
        "ppu_functional_snapshot", control / ".github/scripts/ppu_functional_snapshot.py"
    )
    files = snapshot.verify(root, control)
    names = sorted({str(Path(path).parent) for path in files})
    for path in files:
        if path.endswith(".json"):
            read_metrics(root / BASELINES / path)
    rows = load_recipes(control)
    tokenizer_sources = hf.load_sources(control)
    by_name = {}
    for row in rows:
        by_name.setdefault(f"{row.get('model')}/{row.get('test_case')}", []).append(row)
    unknown = set(names) - by_name.keys()
    if unknown:
        raise ValueError(f"PPU goldens have no DEV recipe match: {sorted(unknown)}")
    selected, disabled, ignored, outside, skipped = {}, [], [], [], []
    metric_names = supported_metrics(control)
    requested = config["requested_cases"]
    if requested != ["all"] and set(requested) - by_name.keys():
        raise ValueError(f"Unknown requested cases: {sorted(set(requested) - by_name.keys())}")
    for name, specs in sorted(by_name.items()):
        active = [row for row in specs if row.get("scope") in TIERS]
        eligible = (
            active
            if config["scope"] == "all"
            else [row for row in active if row["scope"] == config["scope"]]
        )
        if not active:
            scopes = sorted({row.get("scope", "") for row in specs})
            if any("broken" in scope or "disabled" in scope for scope in scopes):
                disabled.append({"name": name, "scopes": scopes, "reason": "upstream_disabled"})
            else:
                outside.append(name)
            continue
        if not eligible:
            outside.append(name)
            continue
        has_config = (root / BASELINES / name / "model_config.yaml").is_file()
        has_golden = (root / BASELINES / name / "golden_values_dev_ppu.json").is_file()
        if not has_golden:
            ignored.append({"name": name, "reason": "missing_PPU_golden"})
            continue
        if not has_config:
            raise ValueError(f"PPU golden has no configuration: {name}")
        if requested != ["all"] and name not in requested:
            continue
        if name in SKIPPED_CASES:
            skipped.append(
                {
                    "name": name,
                    "scopes": sorted({row["scope"] for row in eligible}),
                    "reason": SKIPPED_CASES[name],
                }
            )
            continue
        row = describe_case(root, name, eligible, config)
        for dependency in row["tokenizer_dependencies"]:
            source = tokenizer_sources.get(dependency["repository"])
            if source:
                dependency.update(source=source, status="fixed_source")
        fixed = [item for item in row["tokenizer_dependencies"] if item.get("source")]
        row["assets"].extend(hf.asset_requirements(fixed))
        if set(row["metrics"]) - metric_names:
            raise ValueError(f"Unsupported upstream metrics for {name}")
        row["id"] = f"functional-{len(selected):03d}"
        selected[row["id"]] = row
    if requested != ["all"]:
        unavailable = set(requested) & (set(outside) | {row["name"] for row in disabled})
        if unavailable:
            raise ValueError(
                f"Requested cases outside active selected scope: {sorted(unavailable)}"
            )
    if not selected and skipped:
        raise ValueError(f"No runnable functional cases; explicitly skipped: {skipped}")
    if not selected or len(selected) > 256:
        raise ValueError("Functional execution set must contain 1..256 cases with PPU goldens")
    return {
        "functional": selected,
        "coverage": {
            "ignored_missing_golden": ignored,
            "upstream_disabled": disabled,
            "skipped_by_policy": skipped,
            "outside_scope": outside,
            "baseline_configs": sum(Path(path).name == "model_config.yaml" for path in files),
            "baseline_files": len(files),
            "baseline_not_selected": [
                {
                    "name": name,
                    "scopes": sorted({row["scope"] for row in by_name[name]}),
                    "reason": SKIPPED_CASES.get(
                        name,
                        (
                            "upstream_disabled"
                            if name in {row["name"] for row in disabled}
                            else "outside_requested_scope_or_cases"
                        ),
                    ),
                }
                for name in names
                if name not in {row["name"] for row in selected.values()}
            ],
        },
        "baseline_lock_sha256": sha256(control / LOCK),
        "tokenizer_lock_sha256": sha256(control / hf.LOCK) if tokenizer_sources else None,
    }


def emit(key: str, value) -> None:
    if not isinstance(value, str):
        value = json.dumps(value, separators=(",", ":"))
    print(f"{key}={value}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"{key}={value}\n")


def generate(root: Path, control: Path, config: dict, identity: dict, output: Path) -> dict:
    for key in ("target_sha", "control_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(identity.get(key, ""))):
            raise ValueError(f"Invalid functional identity: {key}")
    positive(identity.get("run_id"), "run_id", 10**18)
    positive(identity.get("run_attempt"), "run_attempt", 10**9)
    manifest = {
        "schema_version": 1,
        "test_kind": "functional",
        "config": config,
        "identity": identity,
        **catalogue(root, control, config),
    }
    protected = [
        "tests/test_utils/python_scripts/recipe_parser.py",
        "megatron/core/tokenizers/utils/build_tokenizer.py",
        "megatron/core/tokenizers/text/libraries/megatron_hf_tokenizer.py",
        "megatron/core/tokenizers/text/libraries/huggingface_tokenizer.py",
    ]
    protected += [
        f"tests/functional_tests/shell_test_utils/{name}"
        for name in ("run_ci_test.sh", "_run_training.sh")
    ]
    protected += [
        f"tests/functional_tests/python_test_utils/{name}"
        for name in (
            "common.py",
            "conftest.py",
            "get_test_results_from_tensorboard_logs.py",
            "test_pretraining_regular_pipeline.py",
            "test_pretraining_resume_checkpoint_pipeline.py",
        )
    ]
    manifest["source_files"] = {name: sha256(safe_file(root, name)) for name in protected}
    manifest["control_files"] = {name: sha256(safe_file(control, name)) for name in protected}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for key in (
        "build",
        "unit_test",
        "scope",
        "threshold_percent",
        "suite_family",
        "test_kind",
        "functional_mode",
    ):
        emit(key, config[key])
    emit("config", config)
    emit("unit_matrix", {"include": [{"id": "unused"}]})
    emit(
        "functional_matrix",
        {
            "include": [
                {key: row[key] for key in ("id", "name", "cards", "action_minutes", "job_minutes")}
                for row in manifest["functional"].values()
            ]
        },
    )
    return manifest


def check_asset(root: Path, requirement: dict) -> dict:
    path = root / requirement["relative_path"]
    result = {
        **requirement,
        "checked_path": str(path),
        "exists": False,
        "readable": False,
        "size": None,
        "ok": False,
        "reason": "",
    }
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Asset path escapes the configured root")
        stat = path.stat()
        result.update(exists=True, readable=os.access(path, os.R_OK), size=stat.st_size)
        if not result["readable"]:
            raise ValueError("Asset is not readable")
        if requirement["kind"] == "directory":
            if not path.is_dir() or not os.access(path, os.X_OK):
                raise ValueError("Asset is not an accessible directory")
            if next(path.iterdir(), None) is None:
                raise ValueError("Asset directory is empty")
        elif not path.is_file() or stat.st_size == 0:
            raise ValueError("Asset is not a nonempty regular file")
        if requirement.get("tokenizer_cache"):
            content = hf.checked_bytes(
                root,
                requirement["relative_path"],
                {"size": requirement["expected_size"], "sha256": requirement["expected_sha256"]},
            )
            result["sha256"] = hf.digest(content)
        result["ok"] = True
    except (OSError, ValueError) as exc:
        result["reason"] = str(exc)
    return result


def validate_tokenizer_assets(case: dict) -> list:
    fixed = [item for item in case["tokenizer_dependencies"] if item.get("source")]
    expected = hf.asset_requirements(fixed)
    actual = [item for item in case["assets"] if item.get("tokenizer_cache")]
    if actual != expected:
        raise ValueError("Tokenizer asset requirements differ from the fixed source lock")
    return expected


def asset_report(
    manifest: dict, root: Path, *, location: str = "cpu", case_id: str | None = None
) -> dict:
    rows = manifest["functional"]
    if case_id:
        rows = {case_id: rows[case_id]}
    for case in rows.values():
        validate_tokenizer_assets(case)
    root_check = check_asset(
        root, {"relative_path": ".", "kind": "directory", "purposes": ["asset_root"]}
    )
    cases = [
        {
            "id": key,
            "name": case["name"],
            "observation": case["observation"],
            "assets": [check_asset(root, requirement) for requirement in case["assets"]],
            "unresolved_dependencies": [
                {
                    **requirement,
                    "relative_path": None,
                    "checked_path": None,
                    "exists": None,
                    "readable": None,
                    "size": None,
                    "ok": False,
                    "reason": (
                        "HF tokenizer fixed-revision source lock is not provisioned; the isolated "
                        "offline cache cannot be populated. No files were guessed or downloaded."
                    ),
                }
                for requirement in case["tokenizer_dependencies"]
                if requirement.get("status") != "fixed_source" or not requirement.get("source")
            ],
        }
        for key, case in rows.items()
    ]
    ok = root_check["ok"] and all(
        not case["unresolved_dependencies"] and all(asset["ok"] for asset in case["assets"])
        for case in cases
    )
    return {
        "schema_version": 1,
        "identity": manifest["identity"],
        "location": location,
        "root": root_check,
        "pod_data_path": manifest["config"]["data_path"],
        "cases": cases,
        "coverage": manifest["coverage"],
        "ok": ok,
        "numerical_validation": False,
        "result": (
            "assets_ready"
            if ok
            else (
                "cpu_preflight_unreachable"
                if location == "cpu" and not root_check["ok"]
                else (
                    "asset_source_mapping_required"
                    if any(case["unresolved_dependencies"] for case in cases)
                    else "asset_layout_error"
                )
            )
        ),
    }


def write_asset_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets-report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "## PPU functional asset preflight",
        f"Result: {report['result']}. Numerical tests were not executed.",
        f"Checked root: `{report['root']['checked_path']}`",
        f"Root error: {report['root']['reason'] or 'none'}",
        f"Selected cases: {len(report['cases'])}; observations: {sum(bool(case['observation']) for case in report['cases'])}.",
        f"Ignored without PPU golden: {len(report['coverage']['ignored_missing_golden'])}; upstream disabled: {len(report['coverage']['upstream_disabled'])}.",
        "| Case | Purpose | Expected relative path | Checked path | Exists | Readable | Bytes | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if not case["assets"] and not case["unresolved_dependencies"]:
            lines.append(
                f"| {case['name']} | mock data; no input assets | — | — | — | — | — | ready |"
            )
        for asset in case["assets"]:
            lines.append(
                f"| {case['name']} | {', '.join(asset['purposes'])} | {asset['relative_path']} | {asset['checked_path']} | {asset['exists']} | {asset['readable']} | {asset['size']} | {'ready' if asset['ok'] else asset['reason']} |"
            )
        for dependency in case["unresolved_dependencies"]:
            lines.append(
                f"| {case['name']} | {dependency['purpose']}: {dependency['repository']} | 未确认 | 未检查 | 未知 | 未知 | — | {dependency['reason']} |"
            )
    candidate = report.get("tokenizer_source_candidate")
    if candidate:
        lines.extend(
            [
                "",
                "### HF source candidate (not trusted)",
                f"Status: `{candidate['status']}`; files matching candidate: {candidate['file_count']}.",
                f"Candidate path: `{candidate['checked_path']}`.",
                "Candidate metadata is included in assets-report.json for review only; "
                "it does not replace the trusted source lock or change the asset gate.",
            ]
        )
    lines.append("")
    lines.extend(
        f"- Skipped by policy: `{case['name']}` ({case['reason']}); not executed or counted as passed."
        for case in report["coverage"].get("skipped_by_policy", [])
    )
    lines.extend(
        f"- Observation only: `{case['name']}` ({case['observation']}); excluded from the numerical pass rate."
        for case in report["cases"]
        if case["observation"]
    )
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "result": report["result"],
                "cases": len(report["cases"]),
                "skipped_by_policy": len(report["coverage"].get("skipped_by_policy", [])),
                "numerical_validation": False,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "catalogue", "assets"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("PPU_SOURCE_ROOT", Path(__file__).resolve().parents[2])),
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=Path(os.environ.get("PPU_CONTROL_ROOT", Path(__file__).resolve().parents[2])),
    )
    parser.add_argument("--output", type=Path, default=Path(".ppu-ci"))
    parser.add_argument("--assets-root", type=Path, default=Path(CPU_ASSETS))
    parser.add_argument("--tokenizer-receipts", type=Path)
    parser.add_argument("--location", choices=("cpu", "pod"), default="cpu")
    parser.add_argument("--case-id")
    args = parser.parse_args()
    if args.command == "assets":
        manifest = json.loads((args.output / "manifest.json").read_text())
        report = asset_report(
            manifest, args.assets_root, location=args.location, case_id=args.case_id
        )
        if (
            args.location == "cpu"
            and manifest["config"]["functional_mode"] == "assets"
            and report["root"]["ok"]
        ):
            report["tokenizer_source_candidate"] = hf.inspect_candidate(args.assets_root)
            if args.tokenizer_receipts:
                report["tokenizer_preparation_receipts"] = hf.inspect_receipts(
                    args.tokenizer_receipts
                )
        write_asset_report(report, args.output)
        raise SystemExit(0 if report["ok"] else 1)
    inputs = json.loads(os.environ.get("PPU_INPUTS", "{}"))
    config = select_config(inputs)
    if args.command == "catalogue":
        result = catalogue(args.root, args.control, config)
        print(
            json.dumps(
                {
                    "cases": len(result["functional"]),
                    "models": sorted({row["model"] for row in result["functional"].values()}),
                    "observations": [
                        row["name"] for row in result["functional"].values() if row["observation"]
                    ],
                    "asset_requirements": sorted(
                        {
                            asset["relative_path"]
                            for row in result["functional"].values()
                            for asset in row["assets"]
                        }
                    ),
                    "coverage": result["coverage"],
                },
                indent=2,
            )
        )
        return
    validate_route(os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF"], inputs)
    authorization = json.loads((args.output / "authorization.json").read_text())
    if not authorization["allowed"] or authorization["pr_number"] or authorization["manual_pr"]:
        raise ValueError("Functional configuration requires authorized non-PR dispatch")
    identity = {
        key: authorization[key] for key in ("target_sha", "control_sha", "run_id", "run_attempt")
    }
    generate(args.root, args.control, config, identity, args.output)


if __name__ == "__main__":
    main()
