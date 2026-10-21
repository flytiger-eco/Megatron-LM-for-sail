import copy
import json

import pytest
import yaml
from test_ppu_quick import (
    BASE_SHA,
    BRANCH,
    CONTROL_SHA,
    EXCLUDED_UT,
    IDENTITY,
    MERGE_SHA,
    REPOSITORY,
    ROOT,
    SHA,
    api_for,
    auth,
    ci,
    context,
    pr,
    synthetic_case,
    write_reports,
)


def pr_event(pull=None, action="opened"):
    return {"number": 7, "action": action, "pull_request": pull or pr()}


@pytest.mark.parametrize(
    "change,decision",
    [
        ("draft", "draft"),
        ("closed", "closed"),
        ("conflict", "conflict"),
        ("unknown", "merge_unavailable"),
        ("parents", "merge_unavailable"),
        ("metadata", "ignore"),
    ],
)
def test_pr_preflight_decisions(change, decision):
    pull = pr()
    if change == "draft":
        pull["draft"] = True
    elif change == "closed":
        pull["state"] = "closed"
    elif change == "conflict":
        pull["mergeable"] = False
    elif change == "unknown":
        pull["mergeable"] = None
    event = pr_event(pull, "edited" if change == "metadata" else "opened")
    underlying = api_for(pull)
    calls, sleeps = [], []

    def api(path):
        calls.append(path)
        if change == "parents" and path.startswith("git/commits/"):
            return {"parents": [{"sha": SHA}, {"sha": BASE_SHA}]}
        return underlying(path)

    result = auth.authorize(event, {}, context("pull_request_target"), api, sleeps.append)
    assert not result["allowed"] and result["decision"] == decision
    assert len(sleeps) == (2 if change in ("unknown", "parents") else 0)
    if change in ("draft", "closed", "metadata"):
        assert not any("git/commits/" in path for path in calls)


@pytest.mark.parametrize(
    "files,expected",
    [
        ([{"filename": "docs/api.md"}], True),
        ([{"filename": "README.md"}], True),
        ([{"filename": "docs/conf.py"}], True),
        ([{"filename": "docs/a.md"}, {"filename": "pyproject.toml"}], False),
        ([{"filename": ".github/workflows/ci.yml"}], False),
        (
            [
                {
                    "filename": "docs/a.md",
                    "status": "renamed",
                    "previous_filename": "megatron/core/a.py",
                }
            ],
            False,
        ),
        ([{"filename": "docs/a.md", "status": "renamed"}], False),
        ([{"filename": "docs/a.md", "status": "renamed", "previous_filename": "docs/b.md"}], True),
    ],
)
def test_document_classification(files, expected):
    pull = pr()
    pull["changed_files"] = len(files)
    assert auth.docs_only(7, pull, lambda path: files) is expected


@pytest.mark.parametrize("count", [0, 3000, 3001, None, True])
def test_incomplete_document_count_never_skips(count):
    pull = {**pr(), "changed_files": count}
    assert not auth.docs_only(7, pull, lambda path: pytest.fail("must not fetch"))


@pytest.mark.parametrize("mode", ["complete", "truncated", "duplicate", "error"])
def test_document_pagination(mode):
    rows = [{"filename": f"docs/{i}.md"} for i in range(101)]
    paths = []

    def api(path):
        paths.append(path)
        if mode == "error":
            raise OSError("API unavailable")
        if path.endswith("page=1"):
            return rows[:100]
        return [] if mode == "truncated" else rows[:1] if mode == "duplicate" else rows[100:]

    assert auth.docs_only(7, {**pr(), "changed_files": 101}, api) is (mode == "complete")
    assert len(paths) == (1 if mode == "error" else 2)


@pytest.mark.parametrize("fork", [True, False])
def test_docs_only_pr_skips_without_merge_or_source(fork):
    pull = pr(fork)

    def api(path):
        if "/files?" in path:
            return [{"filename": "README.md"}]
        assert path == "pulls/7"
        return pull

    result = auth.authorize(pr_event(pull), {}, context("pull_request_target"), api)
    assert result["decision"] == "docs_only" and not result["allowed"]


def test_snapshot_drift_during_docs_check():
    calls = 0

    def api(path):
        nonlocal calls
        if "/files?" in path:
            return [{"filename": "README.md"}]
        calls += 1
        pull = pr()
        if calls > 1:
            pull["base"]["sha"] = CONTROL_SHA
        return pull

    with pytest.raises(ValueError, match="快照"):
        auth.authorize(pr_event(), {}, context("pull_request_target"), api)


def test_merge_group_identity_and_routes():
    ref = "refs/heads/gh-readonly-queue/ci/ppu-validation/pr-7"
    event = {
        "action": "checks_requested",
        "repository": {"full_name": REPOSITORY},
        "merge_group": {
            "head_ref": ref,
            "base_ref": BRANCH,
            "head_sha": CONTROL_SHA,
            "base_sha": BASE_SHA,
        },
    }
    result = auth.authorize(event, {}, context("merge_group", ref))
    assert result["allowed"] and result["target_sha"] == CONTROL_SHA
    config = ci.select_config(event, "merge_group", {"scope": "quick", "unit_test": False})
    assert config["scope"] == "full" and config["suite_family"] == "nvidia-dev"
    assert config["threshold_percent"] == 100 and config["unit_test"]
    for key, value in (
        ("base_ref", "refs/heads/main"),
        ("head_sha", SHA),
        ("head_ref", "refs/heads/foo"),
    ):
        bad = copy.deepcopy(event)
        bad["merge_group"][key] = value
        with pytest.raises(ValueError):
            auth.authorize(bad, {}, context("merge_group", ref))


@pytest.mark.parametrize(
    "trigger,scope,expected",
    [
        ("pr", "full", 100),
        ("push", "full", 100),
        ("merge_group", "full", 100),
        ("release", "full", 80),
        ("manual", "full", 80),
        ("manual", "quick", 100),
    ],
)
def test_gate_is_independent_of_selection(tmp_path, trigger, scope, expected):
    case = synthetic_case()
    config = {"scope": scope, "trigger_kind": trigger, "threshold_percent": expected}
    manifest = {"config": config, "unit": {case["id"]: case}, "identity": IDENTITY}
    write_reports(ci.result_directory(manifest, tmp_path, case["id"]), case)
    result = ci.summarize(manifest, tmp_path)
    assert result["threshold_percent"] == expected
    assert result["success"] is (expected == 80)
    manifest["config"]["threshold_percent"] = 1
    with pytest.raises(ValueError):
        ci.summarize(manifest, tmp_path)


def test_quick_not_available_before_baseline():
    with pytest.raises(ValueError, match="基线"):
        ci.select_config({"ref": BRANCH}, "workflow_dispatch", {"scope": "quick"})
    result = ci.select_config({"ref": BRANCH}, "push", {"scope": "quick"})
    assert result["scope"] == "full" and result["threshold_percent"] == 100


def test_nvidia_full_is_dev_half_of_release():
    config = {"repeat": 1, "timeout": 30}
    release = ci.unit_cases({**config, "suite_family": "release-dev-lts"})
    dev = ci.unit_cases({**config, "suite_family": "nvidia-dev"})
    assert len(dev) == 37 and len(release) == 74
    assert dev == {k: v for k, v in release.items() if v["environment"] == "dev"}
    assert sum(EXCLUDED_UT in row["excluded"] for row in dev.values()) == 1


@pytest.mark.parametrize("state", ["candidate", "validated"])
def test_daily_quick_requires_two_validation_runs(tmp_path, monkeypatch, state):
    policy = json.loads((ROOT / ci.POLICY_PATH).read_text())
    policy.update(daily_scope="quick", quick_state=state, baseline={"run_id": "123"})
    path = tmp_path / ci.POLICY_PATH
    path.parent.mkdir()
    path.write_text(json.dumps(policy))
    monkeypatch.setattr(ci, "CONTROL_ROOT", tmp_path)
    with pytest.raises(ValueError, match="两轮"):
        ci.load_policy()


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "head",
        "base",
        "draft",
        "closed",
        "newer",
        "foreign",
        "cancelled",
        "rerun",
        "retarget",
    ],
)
def test_final_status_cannot_overwrite_new_snapshot(monkeypatch, mutation):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    pull = pr()
    url = auth.run_url("124" if mutation == "newer" else "123")
    if mutation == "head":
        pull["head"]["sha"] = CONTROL_SHA
    if mutation == "base":
        pull["base"]["sha"] = CONTROL_SHA
    if mutation == "draft":
        pull["draft"] = True
    if mutation == "closed":
        pull["state"] = "closed"
    if mutation == "foreign":
        url = "https://example.com/another-check"
    if mutation == "retarget":
        pull["base"]["ref"] = "main"
    calls = []

    def api(path, payload=None):
        if payload is not None:
            calls.append((path, payload))
            return {}
        if path == "pulls/7":
            return pull
        if path == "actions/runs/123":
            return {
                "status": "completed" if mutation == "cancelled" else "in_progress",
                "run_attempt": 2 if mutation == "rerun" else 1,
            }
        return [{"context": auth.STATUS_CONTEXT, "target_url": url}]

    auth.publish_status(
        {
            "pr_number": 7,
            "pr_head_sha": SHA,
            "base_sha": BASE_SHA,
            "target_sha": MERGE_SHA,
            "run_id": "123",
            "decision": "run",
        },
        "success",
        "full completed",
        api,
    )
    assert bool(calls) is (mutation == "none")
    if calls:
        assert calls[0][0] == f"statuses/{SHA}"


@pytest.mark.parametrize("mode", ["old", "fresh", "newer", "stale_push"])
def test_base_invalidation_preserves_fresh_checks(monkeypatch, mode):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    writes = []

    def api(path, payload=None):
        if payload is not None:
            writes.append(payload)
            return {}
        if path.startswith("git/ref/"):
            return {"object": {"sha": SHA if mode == "stale_push" else BASE_SHA}}
        if path.startswith("pulls?"):
            return [{"number": 7}]
        if path == "pulls/7":
            return pr()
        return [
            {
                "context": auth.STATUS_CONTEXT,
                "target_url": auth.run_url("201" if mode == "newer" else "100"),
                "description": f"base={BASE_SHA if mode == 'fresh' else SHA}",
            }
        ]

    auth.invalidate_base(BASE_SHA, api)
    assert bool(writes) is (mode == "old")


def test_full_timing_and_rank_duration_aggregation(tmp_path):
    case = synthetic_case("dev-000")
    manifest = {
        "schema_version": 3,
        "config": {"scope": "full", "trigger_kind": "push"},
        "identity": IDENTITY,
        "unit": {case["id"]: case},
    }
    directory = ci.result_directory(manifest, tmp_path, case["id"])
    write_reports(directory, case, failed=0)
    timing = {"started_at": 100, "prepared_at": 102, "completed_at": 110, "elapsed_seconds": 10}
    (directory / "suite-timing.json").write_text(json.dumps(timing))
    (directory / "build-timing.json").write_text(json.dumps(timing))
    for path in directory.glob("*-rank-*.json"):
        report = json.loads(path.read_text())
        report["durations"] = {node: 0.1 * (report["rank"] + 1) for node in report["tests"]}
        path.write_text(json.dumps(report))
    result = ci.summarize(manifest, tmp_path)
    assert result["success"] and result["baseline_complete"]
    group = result["groups"][case["id"]]
    assert set(group["method_durations"].values()) == {0.8}
    assert group["suite_timing"]["prepare_seconds"] == 2
    (directory / "suite-timing.json").rename(directory / "missing-timing")
    assert not ci.summarize(manifest, tmp_path)["baseline_complete"]


@pytest.mark.parametrize(
    "field", ["durations", "collection_seconds", "duration_seconds", "selected", "node"]
)
def test_each_rank_requires_complete_timing(tmp_path, field):
    case = synthetic_case("dev-000")
    write_reports(tmp_path, case, failed=0)
    for path in tmp_path.glob("*-rank-*.json"):
        report = json.loads(path.read_text())
        report["durations"] = dict.fromkeys(report["tests"], 0.1)
        if report["rank"] == 7 and report["phase"] == "prod":
            if field == "node":
                report["durations"].pop(report["selected"][0])
            else:
                report.pop(field)
        path.write_text(json.dumps(report))
    assert not ci.case_results(case, tmp_path, IDENTITY, require_timing=True)["complete"]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("duplicate", "重复匹配"),
        ("missing", "未匹配"),
        ("nodes", "拓扑"),
        ("fraction", "整数"),
        ("glob", "不支持的 UT bucket"),
    ],
)
def test_recipe_changes_cannot_silently_drop_workloads(tmp_path, monkeypatch, mutation, reason):
    recipe = yaml.safe_load((ROOT / ci.RECIPE_PATH).read_text())
    if mutation == "duplicate":
        recipe["products"][0]["products"] *= 2
    elif mutation == "missing":
        recipe["products"][0]["products"][0]["environment"] = ["lts"]
    elif mutation == "nodes":
        recipe["spec"]["nodes"] = 2
    elif mutation == "fraction":
        recipe["spec"]["gpus"] = 8.5
    else:
        recipe["products"][0]["test_case"] = ["tests/unit_tests/*model*.py"]
    target = tmp_path / ci.RECIPE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(recipe))
    (tmp_path / "tests/unit_tests").mkdir()
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    monkeypatch.setattr(ci, "policy_exclusions", lambda: {})
    with pytest.raises(ValueError, match=reason):
        ci.unit_cases({"suite_family": "nvidia-dev", "repeat": 1, "timeout": 30})


@pytest.mark.parametrize("decision", ["run", "docs_only"])
def test_status_snapshot_is_rechecked_before_write(monkeypatch, decision):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    reads = 0

    def api(path, payload=None):
        nonlocal reads
        assert payload is None, "must not publish stale success"
        if path == "pulls/7":
            reads += 1
            pull = pr()
            if reads > 1:
                pull["draft"] = True
            return pull
        if path == "actions/runs/123":
            return {"status": "in_progress", "run_attempt": 1}
        return [{"context": auth.STATUS_CONTEXT, "target_url": auth.run_url("123")}]

    auth.publish_status(
        {
            "pr_number": 7,
            "pr_head_sha": SHA,
            "base_sha": BASE_SHA,
            "run_id": "123",
            "decision": decision,
        },
        "success",
        "completed",
        api,
    )
    assert reads == 2


@pytest.mark.parametrize("mutation", ["draft", "head", "base"])
def test_merge_snapshot_rechecked_after_parent_lookup(mutation):
    changed = False
    underlying = api_for()

    def api(path):
        nonlocal changed
        if path.startswith("git/commits/"):
            changed = True
        if path == "pulls/7" and changed:
            pull = pr()
            if mutation == "draft":
                pull["draft"] = True
            else:
                pull[mutation]["sha"] = CONTROL_SHA
            return pull
        return underlying(path)

    with pytest.raises(ValueError, match="校验期间"):
        auth.authorize(pr_event(), {}, context("pull_request_target"), api)
