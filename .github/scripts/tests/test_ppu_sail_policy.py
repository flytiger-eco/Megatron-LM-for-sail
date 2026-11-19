"""Branch isolation and ordering for SAIL's shared PPU execution path."""

import copy

import pytest
import yaml
from test_ppu_lifecycle import FakeGitHub
from test_ppu_quick import ROOT, auth, ci, context

BRANCHES = ("main", "dev", "davidsophus/ppu-ci-integration")


@pytest.mark.parametrize("branch", BRANCHES)
def test_manual_dispatch_is_bound_to_its_trusted_workflow(branch):
    ref = f"refs/heads/{branch}"
    result = auth.authorize({}, {}, context(ref=ref), FakeGitHub())
    assert result["allowed"] and result["base_ref"] == ref
    wrong = context(ref=ref)
    wrong["workflow_ref"] = wrong["workflow_ref"].replace("ppu-build.yml", "untrusted.yml")
    with pytest.raises(ValueError, match="trusted"):
        auth.authorize({}, {}, wrong, FakeGitHub())
    assert ci.select_config({"ref": ref}, "workflow_dispatch", {})["scope"] == "quick"


@pytest.mark.parametrize("branch", BRANCHES)
def test_pr_guard_reads_the_actual_base_branch(branch, monkeypatch):
    github = FakeGitHub()
    github.pull["base"]["ref"] = branch
    event = {"number": 7, "action": "opened", "pull_request": copy.deepcopy(github.pull)}
    result = auth.authorize(
        event, {}, context("pull_request_target", f"refs/heads/{branch}"), github
    )
    assert result["allowed"] and result["base_ref"] == f"refs/heads/{branch}"
    assert f"git/ref/heads/{branch}" in github.calls
    assert auth.current_pr(github.pull, result, github.pull["base"]["repo"]["full_name"])
    github.pull["base"]["ref"] = "untrusted"
    assert not auth.current_pr(github.pull, result, github.pull["base"]["repo"]["full_name"])


@pytest.mark.parametrize(
    "ref", ["refs/heads/feature", "refs/heads/davidsophus/unreviewed", "refs/tags/v1"]
)
def test_manual_dispatch_rejects_other_refs(ref):
    with pytest.raises(ValueError):
        auth.authorize({}, {}, context(ref=ref), FakeGitHub())


def test_combined_functional_job_requires_successful_unit_tests():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-execute.yml").read_text())
    condition = workflow["jobs"]["functional-tests"]["if"]
    assert "test_kind == 'combined' && needs.unit-tests.result == 'success'" in condition
    result = workflow["jobs"]["result"]["steps"][0]["run"]
    assert '[[ "$UT_RESULT" == success && "$UT_ENABLED" == true ]]' in result
    build = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    assert build["jobs"]["execute"]["concurrency"] == {
        "group": "sail-ppu-hardware",
        "queue": "max",
        "cancel-in-progress": False,
    }


def test_smoke_shares_build_but_cannot_publish_a_pr_gate():
    from test_ppu_functional import functional

    ref = "refs/heads/davidsophus/ppu-ci-integration"
    inputs = {"test_kind": "smoke"}
    authorization = auth.authorize({}, inputs, context(ref=ref), FakeGitHub())
    assert authorization["test_kind"] == "combined" and authorization["pr_number"] == 0
    assert functional.authorization_trigger(authorization) == "smoke"
    unit = ci.select_config({"ref": ref}, "workflow_dispatch", inputs)
    assert unit["scope"] == "diagnostic" and unit["repeat"] == 1 and unit["unit_test"]
    assert unit["unit_bucket"] == "tests/unit_tests/test_model_configs.py"
    selection = functional.select_config({}, authorization["functional_policy"])
    cases = functional.catalogue(ROOT, ROOT, selection)["functional"]
    assert len(cases) == 1
    assert next(iter(cases.values()))["name"] == functional.policy.SMOKE_FUNCTIONAL_CASE
    assert selection["repeat"] == 1 and not selection["lightweight"]
    with pytest.raises(ValueError, match="PR"):
        auth.authorize({}, {**inputs, "pr_number": "7"}, context(ref=ref), FakeGitHub())
    with pytest.raises(ValueError, match="sandbox"):
        auth.authorize({}, inputs, context(), FakeGitHub())
