"""Shared SAIL CI policy for production main and dev branches."""

import os
import re

MANUAL_RUN_PERMISSIONS = ("write", "maintain", "admin")

# Temporary, single-case device acceptance on the existing working branch.
VALIDATION_BRANCH = "davidsophus/ppu-ci-thin-adapter"
VALIDATION_REF = f"refs/heads/{VALIDATION_BRANCH}"
VALIDATION_CASES = (
    "gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files",
    "gpt/gpt3_mcore_te_tp2_pp2_resume_torch_dist",
)
BRANCHES = ("main", "dev", VALIDATION_BRANCH)
REFS = tuple(f"refs/heads/{branch}" for branch in BRANCHES)


def trusted_ref(ref: str) -> str:
    if ref not in REFS:
        raise ValueError("CI requires an explicitly trusted main or dev branch")
    return ref


# Orchestration and documentation are checked on hosted CPU runners. Runtime,
# dependency, recipe, golden, and hardware test changes deliberately fall through.
CPU_ONLY_PREFIXES = (
    "docs/",
    "skills/",
    ".github/workflows/",
    ".github/scripts/tests/",
    ".github/ISSUE_TEMPLATE/",
)
CPU_ONLY_FILES = {".pre-commit-config.yaml", ".editorconfig", ".github/CODEOWNERS"}


def cpu_only_path(path: str) -> bool:
    return path.endswith(".md") or path.startswith(CPU_ONLY_PREFIXES) or path in CPU_ONLY_FILES


def artifact_prefix() -> str:
    prefix = os.environ.get("PPU_ARTIFACT_PREFIX", "")
    if prefix not in ("", "sync-main-", "sync-dev-"):
        raise ValueError("Invalid sync artifact namespace")
    return prefix


# Upstream DEV's UT recipe requests eight cards for this bucket, but its script
# explicitly sets GPUS_PER_NODE=4. Match that test world size on SAIL.
UT_PROCESS_OVERRIDES = {"tests/unit_tests/generalized_tensor_parallel/**/*.py": 4}


# Shared preflight/configure selection policy. Keep hardware paths and recipe parsing
# in ppu_functional.py; admission must run with only this standard-library module.
FUNCTIONAL_TIERS = ("L0", "L1", "L2", "L3")
SCOPE_LABELS = ("Run tests", "Run functional tests")
# SAIL shares one numeric default across branch updates and PRs; upstream recipes remain authoritative.
# Values: scope, cadence, repeats, timeout in minutes, lightweight.
FUNCTIONAL_EVENT_POLICIES = {
    "push": ("L0", "bypass", 1, 30, False),
    "merge_group": ("L0", "bypass", 1, 30, False),
    "Run tests": ("L1", "bypass", 1, 30, True),
    "Run functional tests": ("L1", "bypass", 1, 30, False),
    "pr": ("L0", "bypass", 1, 30, False),
}
FUNCTIONAL_DEFAULTS = {
    "functional_mode": "assets",
    "functional_scope": "all",
    "functional_cases": "all",
    "functional_repeat": "1",
    "functional_timeout": "30",
}


def positive(value, name: str, maximum: int) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", str(value)) or int(value) > maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return int(value)


def validate_route(event_name: str, ref: str, inputs: dict) -> str:
    kind = inputs.get("test_kind", "unit")
    if ref == VALIDATION_REF:
        if event_name != "workflow_dispatch" or any(
            inputs.get(key) for key in ("pr_number", "pr_sha", "pr_base_sha")
        ):
            raise ValueError("Device acceptance only permits manual dispatch without PR status")
        if kind == "unit" and (
            inputs.get("scope") != "full"
            or inputs.get("suite_family", "nvidia-dev") != "nvidia-dev"
            or inputs.get("unit_bucket") != "tests/unit_tests/test_model_configs.py"
            or str(inputs.get("repeat", "1")) != "1"
        ):
            raise ValueError("Device acceptance requires one model-config UT bucket")
        if kind == "functional" and (
            inputs.get("functional_cases") not in VALIDATION_CASES
            or str(inputs.get("functional_repeat", "1")) != "1"
        ):
            raise ValueError("Device acceptance requires one selected FT case, repeated once")
        budget = "functional_timeout" if kind == "functional" else "timeout"
        positive(inputs.get(budget, 30), budget, 15)
    if kind not in ("unit", "functional"):
        raise ValueError("test_kind must be unit or functional")
    if {"compiler_diagnostic", "compiler_diagnostic_context"} & inputs.keys():
        raise ValueError("Compiler diagnostics have been removed from PPU CI")
    if kind == "unit":
        if any(
            str(inputs.get(key, default)) != default for key, default in FUNCTIONAL_DEFAULTS.items()
        ):
            raise ValueError("Functional inputs require test_kind=functional")
        return kind
    if event_name != "workflow_dispatch" or ref not in REFS:
        raise ValueError("Functional tests require manual dispatch on a trusted branch")
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
    return kind


def event_policy(trigger: str, labels=()) -> dict | None:
    """Functional policy of an automatic trigger; release tags and other events run UT only."""
    if trigger == "pr":
        # `Run tests` wins over `Run functional tests`, as in the upstream workflow.
        trigger = next((label for label in SCOPE_LABELS if label in labels), "pr")
    elif trigger not in ("push", "merge_group"):
        return None
    scope, cadence, repeat, timeout, lightweight = FUNCTIONAL_EVENT_POLICIES[trigger]
    return {
        "trigger": trigger,
        "functional_scope": scope,
        "cadence": cadence,
        "repeat": repeat,
        "timeout": timeout,
        "lightweight": lightweight,
    }


def functional_request(inputs: dict) -> dict:
    """Validate and normalize a manual FT request without loading the planner."""
    if {"compiler_diagnostic", "compiler_diagnostic_context"} & inputs.keys():
        raise ValueError("Compiler diagnostics have been removed from PPU CI")
    mode = inputs.get("functional_mode", "assets")
    scope = inputs.get("functional_scope", "all")
    if mode not in ("assets", "regression") or scope not in (
        "all",
        "auto",
        "mr",
        *FUNCTIONAL_TIERS,
    ):
        raise ValueError("Invalid functional mode or scope")
    cadence = "bypass"
    if scope == "auto":
        # Manual regression mirrors the upstream nightly L1 run; asset checks cover everything.
        scope, cadence = ("L1", "nightly") if mode == "regression" else ("all", "bypass")
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
        "cadence": cadence,
        "lightweight": False,
        "requested_cases": names,
        "repeat": positive(inputs.get("functional_repeat", 1), "functional_repeat", 10),
        "timeout": positive(inputs.get("functional_timeout", 30), "functional_timeout", 1380),
        "build": mode == "regression",
        "unit_test": False,
        "threshold_percent": 100,
        "trigger_kind": "manual",
        "suite_family": "ppu-functional",
    }
