"""Branch isolation and ordering for SAIL's shared PPU execution path."""

import copy
import json
import shutil
import subprocess
import sys

import pytest
import yaml
from test_ppu_lifecycle import FakeGitHub
from test_ppu_quick import ROOT, auth, ci, context

BRANCHES = ("main", "dev")


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
    result = auth.authorize(event, {}, context("pull_request_target"), github)
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


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("kind", ["smoke", "combined"])
def test_manual_dispatch_cannot_request_a_test_only_combined_route(branch, kind):
    with pytest.raises(ValueError, match="test_kind must be unit or functional"):
        auth.authorize({}, {"test_kind": kind}, context(ref=f"refs/heads/{branch}"), FakeGitHub())


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("docs/usage.rst", True),
        (".github/workflows/ppu-build.yml", True),
        (".github/scripts/tests/test_ppu_ci.py", True),
        (".github/scripts/ppu_ci_run.sh", False),
        (".github/scripts/ppu_functional_runtime.py", False),
        (".github/docker/control-requirements.txt", False),
        ("tests/functional_tests/test_cases/gpt/example/golden_values_dev_m890.json", False),
        ("tests/test_utils/recipes/h100/gpt.yaml", False),
        ("megatron/core/transformer/transformer_config.py", False),
        ("pyproject.toml", False),
    ],
)
def test_cpu_only_filter_preserves_runtime_and_dependency_coverage(filename, expected):
    pull = FakeGitHub().pull
    assert auth.cpu_only(7, pull, lambda _: [{"filename": filename}]) is expected
    if expected:
        renamed = [
            {
                "filename": filename,
                "status": "renamed",
                "previous_filename": "megatron/core/utils.py",
            }
        ]
        assert not auth.cpu_only(7, pull, lambda _: renamed)


def sync_fixture(branch):
    from test_ppu_quick import BASE_SHA, MERGE_SHA, REPOSITORY

    ref = auth.BRANCH
    ctx = context("workflow_dispatch", ref)
    ctx["workflow_ref"] = f"{REPOSITORY}/.github/workflows/copilot-upstream-sync.yml@{ref}"
    ctx["artifact_prefix"] = f"sync-{branch}-"
    base = BASE_SHA
    upstream = "e" * 40
    source = "dev" if branch == "dev" else f"sync/upstream-main/{upstream}"
    request = dict(
        target_branch=branch,
        source_ref=source,
        base_sha=base,
        upstream_sha=upstream,
        target_sha=MERGE_SHA,
    )
    state = {"head": MERGE_SHA, "base": base}

    def api(path):
        if path == f"git/ref/heads/{source}":
            return {"object": {"sha": state["head"]}}
        if path == f"git/ref/heads/{branch}":
            return {"object": {"sha": state["base"]}}
        if path == f"git/commits/{MERGE_SHA}":
            return {"parents": [{"sha": base}, {"sha": upstream}]}
        raise AssertionError(path)

    return request, ctx, api, state


@pytest.mark.parametrize("branch", ["main", "dev"])
@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_sync_handoff_uses_published_commit_and_rechecks_before_execution(branch, event):
    from test_ppu_functional import functional

    request, ctx, api, state = sync_fixture(branch)
    ctx["event_name"] = event
    result = auth.authorize_sync(request, ctx, api)
    assert result["allowed"] and result["target_sha"] == request["target_sha"]
    assert result["test_kind"] == "combined" and result["pr_number"] == 0
    assert result["functional_policy"] == functional.event_policy("push")
    assert auth.execution_guard(result, api)
    state["head"] = "f" * 40
    assert not auth.execution_guard(result, api)
    with pytest.raises(auth.StaleSnapshot):
        auth.authorize_sync(request, ctx, api)


@pytest.mark.parametrize(
    "change",
    [
        "caller",
        "ref",
        "event",
        "prefix",
        "control",
        "destination",
        "parents",
        "extra",
        "branch",
        "base",
    ],
)
def test_sync_handoff_rejects_forged_or_stale_provenance(change):
    request, ctx, api, state = sync_fixture("main")
    if change == "caller":
        ctx["workflow_ref"] = ctx["workflow_ref"].replace("copilot-upstream-sync", "ppu-build")
    elif change == "ref":
        ctx["ref"] = "refs/heads/dev"
        ctx["workflow_ref"] = ctx["workflow_ref"].replace("refs/heads/main", ctx["ref"])
    elif change == "event":
        ctx["event_name"] = "pull_request_target"
    elif change == "prefix":
        ctx["artifact_prefix"] = ""
    elif change == "control":
        ctx["control_sha"] = "f" * 40
    elif change == "destination":
        request["source_ref"] = "main"
    elif change == "parents":
        request["base_sha"] = "f" * 40
    elif change == "extra":
        request["override"] = True
    elif change == "branch":
        request["target_branch"] = "release"
    else:
        state["base"] = "f" * 40
    with pytest.raises(ValueError):
        auth.authorize_sync(request, ctx, api)


def test_sync_namespaces_are_distinct_in_artifacts_and_nas(monkeypatch, tmp_path):
    workflows = ROOT / ".github/workflows"
    for branch in ("main", "dev"):
        prefix = f"sync-{branch}-"
        monkeypatch.setenv("PPU_ARTIFACT_PREFIX", prefix)
        assert (
            ci.result_directory({"config": {"scope": "diagnostic"}}, tmp_path, "dev-001")
            == tmp_path / f"{prefix}ppu-ut-dev-001"
        )
    for name in ("ppu-build", "ppu-execute", "ppu-ut", "ppu-functional"):
        parsed = yaml.safe_load((workflows / f"{name}.yml").read_text())
        for job in parsed["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith(
                    ("actions/upload-artifact@", "actions/download-artifact@")
                ):
                    artifact = step["with"].get("name", step["with"].get("pattern"))
                    assert artifact.startswith("${{ inputs.artifact_prefix }}")
        if name != "ppu-build":
            assert (
                '/${{ inputs.artifact_prefix }}${{ github.run_id }}-${{ github.run_attempt }}/'
                in (workflows / f"{name}.yml").read_text()
            )


def test_manual_main_control_can_authorize_a_dev_pr():
    from test_ppu_quick import BASE_SHA, SHA

    github = FakeGitHub()
    github.pull["base"]["ref"] = "dev"
    request = {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA}
    result = auth.authorize({}, request, context(), github)
    assert result["allowed"] and result["base_ref"] == "refs/heads/dev"
    assert "git/ref/heads/dev" in github.calls


def test_pr_target_never_uses_dev_as_the_workflow_source():
    github = FakeGitHub()
    github.pull["base"]["ref"] = "dev"
    event = {"number": 7, "action": "opened", "pull_request": copy.deepcopy(github.pull)}
    with pytest.raises(ValueError, match="default main"):
        auth.authorize(event, {}, context("pull_request_target", "refs/heads/dev"), github)


@pytest.mark.parametrize("cards", [1, 2, 4, 8])
@pytest.mark.parametrize("escape", [False, True])
def test_upstream_dev_multimodal_bucket_keeps_its_directory_boundary(
    tmp_path, monkeypatch, escape, cards
):
    recipe = yaml.safe_load((ROOT / ci.RECIPE_PATH).read_text())
    recipe["spec"]["gpus"] = cards
    recipe["products"] = [
        {
            "test_case": ["examples/multimodal_dev/tests/**/*.py"],
            "products": [
                {
                    "environment": ["dev"],
                    "tag": ["latest"],
                    "scope": ["unit-tests"],
                    "n_repeat": [1],
                }
            ],
        }
    ]
    recipe_path = tmp_path / ci.RECIPE_PATH
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe))
    case = tmp_path / "examples/multimodal_dev/tests/test_example.py"
    case.parent.mkdir(parents=True)
    if escape:
        outside = tmp_path / "not_a_test.py"
        outside.write_text("def test_wrong_directory(): pass\n")
        case.symlink_to(outside)
    else:
        case.write_text("def test_supported_bucket(): pass\n")
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    monkeypatch.setattr(ci, "policy_exclusions", lambda: {})
    config = {"suite_family": "nvidia-dev", "repeat": 1, "timeout": 30}
    if escape:
        with pytest.raises(ValueError, match="escapes"):
            ci.unit_cases(config)
    else:
        cases = ci.unit_cases(config)
        assert len(cases) == 1
        assert next(iter(cases.values()))["cards"] == cards
        assert next(iter(cases.values()))["files"] == [
            "examples/multimodal_dev/tests/test_example.py"
        ]


@pytest.mark.parametrize("cards", [2, 4, 8])
def test_gtp_world_size_matches_upstream_script(tmp_path, monkeypatch, cards):
    recipe = yaml.safe_load((ROOT / ci.RECIPE_PATH).read_text())
    recipe["spec"]["gpus"] = cards
    recipe["products"] = [
        {
            "test_case": ["tests/unit_tests/generalized_tensor_parallel/**/*.py"],
            "products": [
                {
                    "environment": ["dev"],
                    "tag": ["latest"],
                    "scope": ["unit-tests"],
                    "n_repeat": [1],
                }
            ],
        }
    ]
    recipe_path = tmp_path / ci.RECIPE_PATH
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe))
    case = tmp_path / "tests/unit_tests/generalized_tensor_parallel/test_gtp.py"
    case.parent.mkdir(parents=True)
    case.write_text("def test_world_size(): pass\n")
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    monkeypatch.setattr(ci, "policy_exclusions", lambda: {})
    config = {"suite_family": "nvidia-dev", "repeat": 1, "timeout": 30}
    if cards == 2:
        with pytest.raises(ValueError, match="override requires adaptation"):
            ci.unit_cases(config)
    else:
        assert next(iter(ci.unit_cases(config).values()))["cards"] == 4


@pytest.mark.parametrize(
    "event,inputs",
    [
        ("push", {}),
        ("workflow_dispatch", {}),
        (
            "workflow_dispatch",
            {"test_kind": "functional", "functional_mode": "regression", "functional_scope": "L0"},
        ),
    ],
)
def test_admission_runs_without_the_functional_planner(tmp_path, event, inputs):
    # The admission artifact must work with only its standard-library policy;
    # missing planner/HF/recipe files must not break PR or manual preflight.
    for name in ("ppu_pr_authorize.py", "ppu_policy.py"):
        shutil.copyfile(ROOT / ".github/scripts" / name, tmp_path / name)
    code = """
import json, runpy, sys
module = runpy.run_path(sys.argv[1])
request = json.load(sys.stdin)
def api(path):
    assert path.startswith("collaborators/"), path
    return {"permission": "write"}
result = module["authorize"]({}, request["inputs"], request["context"], api)
print(json.dumps(result))
"""
    run = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path / "ppu_pr_authorize.py")],
        input=json.dumps({"inputs": inputs, "context": context(event)}),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["allowed"]


def test_preflight_summary_explains_automatic_scope_and_sha():
    result = auth.authorize({}, {}, context("push"), FakeGitHub())
    summary = auth.preflight_summary(result, {})
    for value in (result["target_sha"], result["control_sha"], "L0", "1 / 30", "numeric"):
        assert value in summary


def test_preflight_summary_distinguishes_manual_assets_from_numerical_tests():
    inputs = {"test_kind": "functional", "functional_mode": "assets", "functional_scope": "auto"}
    result = auth.authorize({}, inputs, context(), FakeGitHub())
    summary = auth.preflight_summary(result, inputs)
    assert "`assets`" in summary and "`all`" in summary
    assert "`numeric`" not in summary
