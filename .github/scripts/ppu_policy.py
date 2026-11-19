"""Shared SAIL CI policy; the sandbox branch is temporary validation infrastructure."""

BRANCHES = ("main", "dev", "davidsophus/ppu-ci-integration")
REFS = tuple(f"refs/heads/{branch}" for branch in BRANCHES)


def trusted_ref(ref: str) -> str:
    if ref not in REFS:
        raise ValueError("CI requires an explicitly trusted main, dev, or sandbox branch")
    return ref


SANDBOX_REF = "refs/heads/davidsophus/ppu-ci-integration"
SMOKE_UNIT_BUCKET = "tests/unit_tests/test_model_configs.py"
SMOKE_FUNCTIONAL_CASE = "gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files"
