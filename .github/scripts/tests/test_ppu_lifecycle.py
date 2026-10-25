import copy
import json
import sys

import pytest
import yaml
from test_ppu_quick import (
    BASE_SHA,
    BRANCH,
    CONTROL_SHA,
    MERGE_SHA,
    REPOSITORY,
    ROOT,
    SHA,
    api_for,
    auth,
    context,
    pr,
)


class FakeGitHub:
    def __init__(self):
        self.pull = pr()
        self.base_ref_sha = BASE_SHA
        self.statuses = []
        self.run = {"status": "in_progress", "conclusion": None, "run_attempt": 1}
        self.parents = [{"sha": BASE_SHA}, {"sha": SHA}]
        self.writes = []
        self.calls = []
        self.mutate = lambda path: None
        self.permission = "write"

    def __call__(self, path, payload=None):
        self.calls.append(path)
        self.mutate(path)
        if payload is not None:
            assert path == f"statuses/{self.pull['head']['sha']}"
            self.writes.append(copy.deepcopy(payload))
            self.statuses.insert(0, copy.deepcopy(payload))
            return payload
        if path == "pulls/7":
            return copy.deepcopy(self.pull)
        if path.startswith("pulls/7/files?"):
            return [{"filename": ".github/scripts/ppu_ci.py"}]
        if path.startswith("actions/runs/"):
            return copy.deepcopy(self.run)
        if path.startswith("commits/") and "/statuses?" in path:
            return copy.deepcopy(self.statuses)
        if path.startswith("git/commits/"):
            return {"parents": copy.deepcopy(self.parents)}
        if path.startswith("collaborators/"):
            return {"permission": self.permission}
        if path.startswith("git/ref/"):
            return {"object": {"sha": self.base_ref_sha}}
        if path.startswith("pulls?"):
            return [{"number": 7}]
        raise AssertionError(path)


@pytest.fixture
def github(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    return FakeGitHub()


def authorization(github, action="opened", run="123"):
    ctx = context("pull_request_target")
    ctx["run_id"] = run
    event = {"number": 7, "action": action, "pull_request": copy.deepcopy(github.pull)}
    if action == "edited":
        event["changes"] = {"base": {}}
    return auth.authorize(event, {}, ctx, github, lambda _: None)


def pending(github):
    result = authorization(github)
    assert auth.publish_status(result, "pending", result["reason"], github) == "published"
    return result


@pytest.mark.parametrize(
    "action", ["opened", "synchronize", "reopened", "ready_for_review", "edited"]
)
def test_ready_events_acquire_current_pending(github, action):
    result = authorization(github, action)
    assert result["allowed"]
    assert auth.publish_status(result, "pending", result["reason"], github) == "published"
    assert auth.execution_guard(result, github)


@pytest.mark.parametrize(
    "decision,state",
    [
        ("draft", "pending"),
        ("closed", "error"),
        ("conflict", "failure"),
        ("merge_unavailable", "pending"),
        ("authorization_required", "pending"),
        ("docs_only", "success"),
        ("preflight_error", "error"),
    ],
)
def test_blocked_states_do_not_execute(github, decision, state):
    if decision == "draft":
        github.pull["draft"] = True
    elif decision == "closed":
        github.pull["state"] = "closed"
    elif decision == "conflict":
        github.pull["mergeable"] = False
    elif decision == "merge_unavailable":
        github.pull["mergeable"] = None
    elif decision == "authorization_required":
        github.pull["head"]["repo"]["full_name"] = "other/fork"
    if decision == "docs_only":
        original = github

        def api(path, payload=None):
            if "/files?" in path:
                return [{"filename": "README.md"}]
            return original(path, payload)

    else:
        api = github
    if decision == "preflight_error":
        reads = 0

        def mutate(path):
            nonlocal reads
            if path == "pulls/7":
                reads += 1
                if reads == 2:
                    raise OSError("unavailable")

        github.mutate = mutate
    result = auth.authorize(
        {"number": 7, "pull_request": copy.deepcopy(github.pull)},
        {},
        context("pull_request_target"),
        api,
        lambda _: None,
    )
    github.mutate = lambda path: None
    assert result["decision"] == decision and not result["allowed"]
    assert auth.initial_state(decision) == state
    assert auth.publish_status(result, state, result["reason"], github) == "published"
    assert not auth.execution_guard(result, github)
    assert github.statuses[0]["state"] == state
    if state != "success":
        assert auth.publish_status(result, "success", "invalid", github) == "rejected"


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_invalid_dispatch_cannot_take_ownership(github, permission):
    active = pending(github)
    github.permission = permission
    before = copy.deepcopy(github.statuses)
    with pytest.raises(ValueError):
        auth.authorize(
            {}, {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA}, context(), github
        )
    assert github.statuses == before
    assert auth.execution_guard(active, github)


@pytest.mark.parametrize("manual", [False, True])
def test_pr_rerun_is_not_a_new_authorization(github, manual):
    ctx = context() if manual else context("pull_request_target")
    ctx["run_attempt"] = "2"
    event = {"number": 7, "pull_request": pr()}
    inputs = {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA} if manual else {}
    with pytest.raises(ValueError, match="rerun"):
        auth.authorize(event, inputs, ctx, github)
    assert not github.writes


@pytest.mark.parametrize(
    "mutation", ["head", "base", "retarget", "base_repo", "head_repo", "draft", "closed"]
)
@pytest.mark.parametrize("phase", ["preflight", "parents", "publish", "guard"])
def test_every_boundary_rechecks_pr_identity(github, mutation, phase):
    result = pending(github) if phase in ("publish", "guard") else None
    original = copy.deepcopy(github.pull)

    def mutate(path):
        if phase == "preflight" and "/files?" not in path:
            return
        if phase == "parents" and not path.startswith("git/commits/"):
            return
        if phase in ("publish", "guard") and path != "pulls/7":
            return
        if mutation in ("head", "base"):
            github.pull[mutation]["sha"] = CONTROL_SHA
            if mutation == "base":
                github.base_ref_sha = CONTROL_SHA
        elif mutation == "retarget":
            github.pull["base"]["ref"] = "main"
        elif mutation == "base_repo":
            github.pull["base"]["repo"]["full_name"] = "other/repo"
        elif mutation == "head_repo":
            github.pull["head"]["repo"] = None
        elif mutation == "draft":
            github.pull["draft"] = True
        else:
            github.pull["state"] = "closed"

    github.mutate = mutate
    if phase in ("preflight", "parents"):
        with pytest.raises(auth.StaleSnapshot):
            auth.authorize(
                {"number": 7, "pull_request": original}, {}, context("pull_request_target"), github
            )
    elif phase == "publish":
        assert auth.publish_status(result, "success", "completed", github) == "stale"
    else:
        assert not auth.execution_guard(result, github)


@pytest.mark.parametrize("stage", ["initial", "terminal", "guard"])
def test_late_old_run_cannot_take_over_newer_run(github, stage):
    old = pending(github)
    newer = authorization(github, run="124")
    assert auth.publish_status(newer, "pending", newer["reason"], github) == "published"
    if stage == "guard":
        assert not auth.execution_guard(old, github)
    else:
        assert (
            auth.publish_status(old, "pending" if stage == "initial" else "success", "late", github)
            == "stale"
        )
    assert github.statuses[0]["target_url"].endswith("/124")


@pytest.mark.parametrize(
    "status,conclusion,explicit,expected",
    [
        ("in_progress", None, True, "published"),
        ("completed", "cancelled", True, "published"),
        ("completed", "cancelled", False, "stale"),
        ("completed", "success", True, "stale"),
        ("completed", "failure", True, "stale"),
    ],
)
def test_cancellation_requires_explicit_signal_and_ownership(
    github, status, conclusion, explicit, expected
):
    result = pending(github)
    github.run.update(status=status, conclusion=conclusion)
    assert auth.publish_status(result, "error", "Cancelled", github, cancelled=explicit) == expected
    assert (
        auth.publish_status(result, "success", "Not success", github, cancelled=True) == "rejected"
    )


@pytest.mark.parametrize("state", ["success", "failure", "error"])
def test_terminal_updates_are_idempotent_and_cannot_reopen_pending(github, state):
    result = pending(github)
    assert auth.publish_status(result, state, "Finished", github) == "published"
    count = len(github.writes)
    assert auth.publish_status(result, state, "Finished", github) == "published"
    assert len(github.writes) == count
    assert auth.publish_status(result, "pending", "Late pending", github) == "stale"
    assert not auth.execution_guard(result, github)


@pytest.mark.parametrize(
    "mutation",
    ["no_status", "wrong_base", "new_attempt", "run_attempt", "foreign", "merge", "parents"],
)
def test_success_requires_matching_pending_and_merge(github, mutation):
    result = pending(github)
    if mutation == "no_status":
        github.statuses = []
    elif mutation == "wrong_base":
        github.statuses[0]["description"] = f"base={CONTROL_SHA}"
    elif mutation == "new_attempt":
        github.statuses[0]["description"] = f"attempt=2 base={BASE_SHA}"
    elif mutation == "run_attempt":
        github.run["run_attempt"] = 2
    elif mutation == "foreign":
        github.statuses[0]["target_url"] = "https://example.com"
    elif mutation == "merge":
        github.pull["merge_commit_sha"] = CONTROL_SHA
    else:
        github.parents.reverse()
    assert auth.publish_status(result, "success", "Finished", github) == "stale"
    assert not auth.execution_guard(result, github)


@pytest.mark.parametrize("reason", ["Short", "x" * 500, "", "PPU cancelled; revalidate"])
def test_description_always_preserves_full_identity(reason):
    description = auth.status_description(reason, {"base_sha": BASE_SHA, "run_attempt": 999})
    assert len(description) <= 140
    assert description.endswith(f"attempt=999 base={BASE_SHA}")


@pytest.mark.parametrize("mutation", ["retarget", "closed", "draft", "repo", "new_status"])
def test_invalidation_rechecks_identity_before_write(github, monkeypatch, mutation):
    pending(github)
    github.statuses[0]["description"] = f"attempt=1 base={SHA}"
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    reads = 0

    def mutate(path):
        nonlocal reads
        if path != "pulls/7":
            return
        reads += 1
        if reads != 2:
            return
        if mutation == "retarget":
            github.pull["base"]["ref"] = "main"
        elif mutation == "closed":
            github.pull["state"] = "closed"
        elif mutation == "draft":
            github.pull["draft"] = True
        elif mutation == "repo":
            github.pull["base"]["repo"]["full_name"] = "other/repo"
        else:
            github.statuses[0]["target_url"] = auth.run_url("201")

    github.mutate = mutate
    count = len(github.writes)
    auth.invalidate_base(BASE_SHA, github)
    assert len(github.writes) == count


def test_metadata_edit_neither_publishes_nor_executes(github):
    active = pending(github)
    ignored = auth.authorize(
        {"number": 7, "action": "edited", "pull_request": pr()},
        {},
        context("pull_request_target"),
        github,
    )
    assert ignored["decision"] == "ignore"
    assert auth.publish_status(ignored, "pending", "Ignored", github) == "rejected"
    assert auth.execution_guard(active, github)


def test_status_api_failure_is_not_success(github, tmp_path, monkeypatch):
    result = authorization(github)
    (tmp_path / "authorization.json").write_text(json.dumps(result))
    monkeypatch.setattr(sys, "argv", ["ppu_pr_authorize.py", "status", "--output", str(tmp_path)])
    monkeypatch.setattr(
        auth, "github_api", lambda *args: (_ for _ in ()).throw(OSError("unavailable"))
    )
    with pytest.raises(OSError):
        auth.main()
    assert not github.writes


@pytest.mark.parametrize("draft", [False, True])
def test_base_invalidation_handles_cached_pr_base(github, monkeypatch, draft):
    previous = pending(github)
    github.pull["draft"] = draft
    github.base_ref_sha = CONTROL_SHA
    monkeypatch.setenv("GITHUB_RUN_ID", "124")
    auth.invalidate_base(CONTROL_SHA, github)
    assert github.pull["base"]["sha"] == BASE_SHA
    assert github.statuses[0]["state"] == "pending"
    assert github.statuses[0]["description"].endswith(f"base={CONTROL_SHA}")
    assert github.statuses[0]["target_url"].endswith("/124")
    assert auth.publish_status(previous, "error", "Old cancel", github, cancelled=True) == "stale"
    with pytest.raises(auth.StaleSnapshot):
        auth.authorize(
            {}, {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA}, context(), github
        )
    current = auth.authorize(
        {},
        {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": CONTROL_SHA},
        context(),
        github,
        lambda _: None,
    )
    assert current["decision"] == ("draft" if draft else "merge_unavailable")
    assert not current["allowed"] and current["base_sha"] == CONTROL_SHA


@pytest.mark.parametrize("manual", [False, True])
def test_current_merge_can_use_live_base_with_cached_pr_metadata(github, manual):
    github.base_ref_sha = CONTROL_SHA
    github.parents = [{"sha": CONTROL_SHA}, {"sha": SHA}]
    if manual:
        current = auth.authorize(
            {}, {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": CONTROL_SHA}, context(), github
        )
    else:
        current = authorization(github)
    assert current["allowed"] and current["base_sha"] == CONTROL_SHA
    assert github.pull["base"]["sha"] == BASE_SHA
    assert auth.publish_status(current, "pending", current["reason"], github) == "published"
    assert auth.execution_guard(current, github)
    assert auth.publish_status(current, "success", "Finished", github) == "published"


@pytest.mark.parametrize("state", ["pending", "success", "failure", "error"])
def test_cached_pr_base_cannot_hide_branch_advance(github, state):
    previous = pending(github)
    github.base_ref_sha = CONTROL_SHA
    writes = len(github.writes)
    assert auth.publish_status(previous, state, "Late result", github) == "stale"
    assert not auth.execution_guard(previous, github)
    assert len(github.writes) == writes


@pytest.mark.parametrize("phase", ["preflight", "parents", "publish", "guard"])
def test_branch_advance_rechecked_without_pr_api_refresh(github, phase):
    previous = pending(github) if phase in ("publish", "guard") else None
    refs = 0

    def mutate(path):
        nonlocal refs
        if path.startswith("git/ref/"):
            refs += 1
            if refs == (4 if phase == "parents" else 2):
                github.base_ref_sha = CONTROL_SHA

    github.mutate = mutate
    if phase in ("preflight", "parents"):
        with pytest.raises(auth.StaleSnapshot):
            authorization(github)
    elif phase == "publish":
        assert auth.publish_status(previous, "success", "Late success", github) == "stale"
    else:
        assert not auth.execution_guard(previous, github)


@pytest.mark.parametrize("terminal", ["success", "failure", "error"])
def test_manual_dispatch_reclaims_terminal_status_with_fresh_run(github, monkeypatch, terminal):
    previous = pending(github)
    assert auth.publish_status(previous, terminal, "Finished", github) == "published"
    ctx = context()
    ctx["run_id"] = "124"
    inputs = {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA}
    current = auth.authorize({}, inputs, ctx, github)
    assert current["manual_pr"] and current["allowed"]
    assert auth.publish_status(current, "pending", current["reason"], github) == "published"
    monkeypatch.setenv("GITHUB_RUN_ID", "124")
    assert auth.execution_guard(current, github)
    assert auth.publish_status(previous, "error", "Late cancel", github, cancelled=True) == "stale"
    assert github.statuses[0]["target_url"].endswith("/124")


def test_manual_draft_takeover_preserves_waiting_status_after_old_cancel(github):
    previous = pending(github)
    github.pull["draft"] = True
    ctx = context()
    ctx["run_id"] = "124"
    inputs = {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA}
    draft = auth.authorize({}, inputs, ctx, github)
    assert draft["decision"] == "draft" and not draft["allowed"]
    assert auth.publish_status(draft, "pending", draft["reason"], github) == "published"
    assert not auth.execution_guard(draft, github)
    assert auth.publish_status(previous, "error", "Late cancel", github, cancelled=True) == "stale"
    assert github.statuses[0]["description"].startswith("Awaiting a ready PR")
    assert github.statuses[0]["target_url"].endswith("/124")


def test_base_invalidation_requires_new_full_sha_dispatch(github, monkeypatch):
    previous = pending(github)
    github.pull["base"]["sha"] = CONTROL_SHA
    github.base_ref_sha = CONTROL_SHA
    github.parents = [{"sha": CONTROL_SHA}, {"sha": SHA}]
    monkeypatch.setenv("GITHUB_RUN_ID", "124")
    auth.invalidate_base(CONTROL_SHA, github)
    assert github.statuses[0]["description"].endswith(f"base={CONTROL_SHA}")
    assert not auth.execution_guard(previous, github)
    ctx = context()
    ctx["run_id"] = "125"
    inputs = {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA}
    with pytest.raises(ValueError, match="head/base changed"):
        auth.authorize({}, inputs, ctx, github)
    inputs["pr_base_sha"] = CONTROL_SHA
    current = auth.authorize({}, inputs, ctx, github)
    assert auth.publish_status(current, "pending", current["reason"], github) == "published"
    monkeypatch.setenv("GITHUB_RUN_ID", "125")
    assert auth.execution_guard(current, github)
    assert auth.publish_status(previous, "success", "Late success", github) == "stale"


def test_workflow_authorized_concurrency_and_cancellation_contract():
    parent = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    execute = yaml.safe_load((ROOT / ".github/workflows/ppu-execute.yml").read_text())
    call = parent["jobs"]["execute"]
    assert call["needs"] == ["authorize", "pr-pending"]
    assert "status_result == 'published'" in call["if"]
    assert "needs.authorize.result == 'success'" in call["if"]
    assert "needs.authorize.outputs.pr_number" in call["concurrency"]["group"]
    assert "ppu-validation" in call["concurrency"]["group"]
    assert call["uses"] == "./.github/workflows/ppu-execute.yml"
    assert "preflight-{0}" in parent["concurrency"]["group"]
    for field in ("pr_number", "pr_sha", "pr_base_sha"):
        assert f"inputs.{field} != ''" in parent["concurrency"]["group"]
    assert "workflow_dispatch" not in parent["concurrency"]["cancel-in-progress"]
    result = parent["jobs"]["pr-result"]
    assert "always()" in result["if"] and "!cancelled()" not in result["if"]
    assert result["timeout-minutes"] == 5
    assert any(step.get("if") == "cancelled()" for step in result["steps"])
    assert "execution_success == 'true'" in result["steps"][-1]["env"]["STATE"]
    assert "--cancelled" in result["steps"][-1]["run"]
    assert execute["jobs"]["build"]["needs"] == ["configure", "admit-build"]
    for name in ("configure", "admit-build"):
        steps = execute["jobs"][name]["steps"]
        assert any(step.get("run", "").endswith("ppu_pr_authorize.py guard") for step in steps)
        assert execute["jobs"][name]["runs-on"] == "ubuntu-latest"
    for name in ("build", "unit-tests"):
        assert "statuses" not in execute["jobs"][name].get("permissions", {})
        assert "actions" not in execute["jobs"][name].get("permissions", {})
    assert execute[True].keys() == {"workflow_call"}
    assert "workflow_run" not in parent[True]
