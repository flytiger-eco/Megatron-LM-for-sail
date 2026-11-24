"""Shared SAIL CI policy for production main and dev branches."""

import os

BRANCHES = ("main", "dev")
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
