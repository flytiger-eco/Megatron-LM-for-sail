"""Shared SAIL CI policy; the sandbox branch is temporary validation infrastructure."""

import os

BRANCHES = ("main", "dev", "davidsophus/ppu-ci-integration")
REFS = tuple(f"refs/heads/{branch}" for branch in BRANCHES)


def trusted_ref(ref: str) -> str:
    if ref not in REFS:
        raise ValueError("CI requires an explicitly trusted main, dev, or sandbox branch")
    return ref


SANDBOX_REF = "refs/heads/davidsophus/ppu-ci-integration"
SMOKE_UNIT_BUCKET = "tests/unit_tests/test_model_configs.py"
SMOKE_FUNCTIONAL_CASE = "gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files"


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
