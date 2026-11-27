"""CPU coverage of automatic functional triggers, combined plans and label authorization."""

import copy
import json
import sys

import pytest
import yaml
from test_ppu_functional import ROOT, functional
from test_ppu_functional_results import results
from test_ppu_lifecycle import FakeGitHub, github
from test_ppu_quick import BASE_SHA, CONTROL_SHA, MERGE_SHA, REPOSITORY, SHA, auth, ci, context

WORKFLOWS = ROOT / ".github/workflows"


def pr_event(github, action="opened", label=None):
    event = {"number": 7, "action": action, "pull_request": copy.deepcopy(github.pull)}
    if label is not None:
        event["label"] = {"name": label}
    return event


def authorize_pr(github, action="opened", label=None):
    return auth.authorize(
        pr_event(github, action, label), {}, context("pull_request_target"), github, lambda _: None
    )


def label(github, *names):
    github.pull["labels"] = [{"name": name} for name in names]


@pytest.mark.parametrize(
    "trigger,labels,expected",
    [
        ("push", (), ("push", "L0", "bypass", 1, 30, False)),
        ("merge_group", (), ("merge_group", "L1", "mergegroup", 1, 30, False)),
        ("pr", (), ("pr", "L0", "bypass", 1, 30, False)),
        ("pr", ("Run tests",), ("Run tests", "L1", "bypass", 1, 30, True)),
        ("pr", ("Run functional tests",), ("Run functional tests", "L1", "bypass", 5, 30, False)),
        ("pr", ("Run functional tests", "Run tests"), ("Run tests", "L1", "bypass", 1, 30, True)),
        (
            "pr",
            ("Run Merge-Queue jobs",),
            ("Run Merge-Queue jobs", "L1", "mergegroup", 1, 30, False),
        ),
        (
            "pr",
            ("Run tests", "Run functional tests", "Run Merge-Queue jobs"),
            ("Run Merge-Queue jobs", "L1", "mergegroup", 1, 30, False),
        ),
        ("pr", ("bug",), ("pr", "L0", "bypass", 1, 30, False)),
    ],
)
def test_event_policy_resolves_label_priority_and_sail_defaults(trigger, labels, expected):
    policy = functional.event_policy(trigger, labels)
    keys = ("trigger", "functional_scope", "cadence", "repeat", "timeout", "lightweight")
    assert tuple(policy[key] for key in keys) == expected


def test_policy_config_is_not_overridable_by_manual_inputs():
    policy = functional.event_policy("push")
    with pytest.raises(ValueError, match="manual inputs"):
        functional.select_config({"functional_scope": "L0"}, policy)
    config = functional.select_config({}, policy)
    assert config["functional_mode"] == "regression" and config["build"] is True
    assert config["unit_test"] is False and config["threshold_percent"] == 100
    assert (config["scope"], config["cadence"], config["repeat"], config["timeout"]) == (
        "L0",
        "bypass",
        1,
        30,
    )
    assert config["trigger_kind"] == "push" and config["requested_cases"] == ["all"]
    assert functional.functional_label(config) == "functional L0 numeric x1"


@pytest.fixture(scope="module")
def selections():
    def names(trigger, labels=()):
        config = functional.select_config({}, functional.event_policy(trigger, labels))
        return functional.catalogue(ROOT, ROOT, config)["functional"]

    return {
        "push": names("push"),
        "merge_group": names("merge_group"),
        "pr": names("pr"),
        "Run tests": names("pr", ("Run tests",)),
        "Run functional tests": names("pr", ("Run functional tests",)),
        "Run Merge-Queue jobs": names("pr", ("Run Merge-Queue jobs",)),
        "L1": functional.catalogue(
            ROOT, ROOT, functional.select_config({"functional_scope": "L1"})
        )["functional"],
    }


def test_automatic_selections_are_nonempty_and_follow_cadence(selections):
    case_names = {key: {row["name"] for row in rows.values()} for key, rows in selections.items()}
    assert all(case_names.values())
    assert case_names["merge_group"] <= case_names["L1"]
    assert case_names["Run tests"] == case_names["Run functional tests"] == case_names["L1"]
    assert case_names["push"] == case_names["pr"]
    assert case_names["Run Merge-Queue jobs"] == case_names["merge_group"]
    # Explicit labels may request the larger upstream L1 suite.
    assert case_names["push"] != case_names["L1"]
    assert all(row["observation"] for row in selections["Run tests"].values())
    assert not all(row["observation"] for row in selections["Run functional tests"].values())
    assert all(
        row["repeat"] == 5
        for row in selections["Run functional tests"].values()
        if not row["observation"]
    )
    assert all("mergegroup" in row["cadence"] for row in selections["merge_group"].values())


def test_merge_queue_filters_nightly_only_cases_but_functional_label_includes_them(
    selections, monkeypatch
):
    nightly = next(iter(selections["merge_group"].values()))["name"]
    rows = functional.load_recipes(ROOT)
    for row in rows:
        if f"{row['model']}/{row['test_case']}" == nightly:
            row["cadence"] = ["nightly"]
    monkeypatch.setattr(functional, "load_recipes", lambda _: rows)
    for trigger, labels, included in (
        ("merge_group", (), False),
        ("pr", ("Run Merge-Queue jobs",), False),
        ("pr", ("Run functional tests",), True),
    ):
        config = functional.select_config({}, functional.event_policy(trigger, labels))
        cases = functional.catalogue(ROOT, ROOT, config)["functional"]
        assert (nightly in {row["name"] for row in cases.values()}) is included


@pytest.mark.parametrize("event_name", ["pull_request_target", "workflow_dispatch"])
@pytest.mark.parametrize(
    "scope_label,repeats,cadence,lightweight",
    [
        ("Run tests", 1, "bypass", True),
        ("Run functional tests", 5, "bypass", False),
        ("Run Merge-Queue jobs", 1, "mergegroup", False),
    ],
)
def test_authorized_pr_labels_drive_full_ut_and_combined_ft(
    github, tmp_path, monkeypatch, event_name, scope_label, repeats, cadence, lightweight
):
    label(github, scope_label)
    event = pr_event(github) if event_name == "pull_request_target" else {}
    inputs = {"scope": "quick", "build": False, "unit_test": False, "repeat": 0}
    if event_name == "workflow_dispatch":
        inputs.update(pr_number="7", pr_sha=SHA, pr_base_sha=BASE_SHA)
    authorization = auth.authorize(event, inputs, context(event_name), github, lambda _: None)
    assert authorization["allowed"]
    output = tmp_path / "plan"
    output.mkdir()
    (output / "authorization.json").write_text(json.dumps(authorization))
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    for key, value in {
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "PPU_DEFAULTS": "{}",
        "PPU_INPUTS": json.dumps(inputs),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", ["ppu_ci.py", "configure", "--output", str(output)])
    ci.main()
    unit = json.loads((output / "manifest.json").read_text())
    recipe = yaml.safe_load((ROOT / ci.RECIPE_PATH).read_text())
    expected_buckets = {
        product["test_case"][0]
        for product in recipe["products"]
        if any("dev" in variant["environment"] for variant in product["products"])
    }
    assert unit["config"]["scope"] == "full"
    assert {case["bucket"] for case in unit["unit"].values()} == expected_buckets
    assert all(case["repeat"] == 1 for case in unit["unit"].values())
    plan = functional.combine(ROOT, ROOT, output)
    assert plan["identity"]["target_sha"] == unit["identity"]["target_sha"] == MERGE_SHA
    assert plan["config"]["repeat"] == repeats
    assert plan["config"]["cadence"] == cadence
    assert plan["config"]["lightweight"] is lightweight
    assert plan["functional"]
    assert all(
        case["repeat"] == (1 if case["observation"] else repeats)
        for case in plan["functional"].values()
    )


@pytest.mark.parametrize("outcome", [True, False])
def test_lightweight_summary_gates_on_execution_without_claiming_numerics(
    selections, monkeypatch, tmp_path, outcome
):
    cases = selections["Run tests"]
    manifest = {
        "test_kind": "functional",
        "config": {"functional_mode": "regression"},
        "identity": {},
        "functional": cases,
        "coverage": {"ignored_missing_golden": [], "upstream_disabled": []},
    }
    monkeypatch.setattr(
        results,
        "verify_case",
        lambda _manifest, case, _directory: {
            "name": case["name"],
            "observation": case["observation"],
            "success": outcome,
            "classification": "passed" if outcome else "training_failed",
        },
    )
    summary = results.summarize(manifest, tmp_path)
    assert summary["success"] is outcome and summary["numerical_validation"] is False
    assert summary["numeric_passed"] == summary["numeric_failed"] == 0
    assert len(summary["observations"]) == len(cases)


def combined_output(tmp_path, trigger="push", labels=(), **changes):
    output = tmp_path / ".ppu-ci"
    output.mkdir()
    identity = {"target_sha": SHA, "control_sha": CONTROL_SHA}
    authorization = {
        "allowed": True,
        "test_kind": "combined",
        "event_name": trigger,
        "pr_number": 7 if trigger == "pull_request_target" else 0,
        "labels": list(labels),
        "functional_policy": functional.event_policy(
            "pr" if trigger == "pull_request_target" else trigger, labels
        ),
        **identity,
        "run_id": "123",
        "run_attempt": 1,
        **changes,
    }
    (output / "authorization.json").write_text(json.dumps(authorization))
    top = {"schema_version": 3, "config": {"build": True, "unit_test": True}, "identity": identity}
    (output / "manifest.json").write_text(json.dumps(top, indent=2))
    return output


def test_combine_attaches_verified_functional_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    output = combined_output(tmp_path, "pull_request_target", ("Run tests",))
    plan = functional.combine(ROOT, ROOT, output)
    top = json.loads((output / "manifest.json").read_text())
    assert top["schema_version"] == 4 and top["test_kind"] == "combined"
    assert top["config"] == {"build": True, "unit_test": True}
    assert set(top["functional_plan"]) == {"path", "sha256"}
    assert functional.load_manifest(output / "manifest.json") == plan
    assert plan["config"]["lightweight"] is True and plan["config"]["trigger_kind"] == "Run tests"
    outputs = dict(line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines())
    assert outputs["test_kind"] == "combined" and outputs["functional_mode"] == "regression"
    assert outputs["functional_label"] == "functional L1 lightweight x1"
    assert len(json.loads(outputs["functional_matrix"])["include"]) == len(plan["functional"])


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"allowed": False}, "authorized combined"),
        ({"test_kind": "unit"}, "authorized combined"),
        ({"labels": ["Run tests"]}, "trusted trigger policy"),
        ({"functional_policy": {**functional.event_policy("push"), "repeat": 2}}, "trusted"),
        ({"event_name": "workflow_dispatch"}, "push, merge queue, or PR"),
        ({"control_sha": "e" * 40}, "different commit identities"),
    ],
)
def test_combine_rejects_untrusted_authorization(tmp_path, changes, match):
    output = combined_output(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        functional.combine(ROOT, ROOT, output)


@pytest.mark.parametrize(
    "top", [{"schema_version": 4}, {"test_kind": "combined"}, {"config": {"build": False}}]
)
def test_combine_requires_a_fresh_ut_manifest_with_shared_build(tmp_path, top):
    output = combined_output(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(top)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shared Build"):
        functional.combine(ROOT, ROOT, output)


@pytest.fixture(scope="module")
def combined_plan(tmp_path_factory):
    output = combined_output(tmp_path_factory.mktemp("combined"), "merge_group")
    functional.combine(ROOT, ROOT, output)
    return output


@pytest.mark.parametrize(
    "damage",
    ["sidecar", "symlink", "missing", "digest", "path", "extra", "schema", "identity", "plan"],
)
def test_load_manifest_rejects_tampered_combined_plans(combined_plan, tmp_path, damage):
    output = tmp_path / "plan"
    output.mkdir()
    for name in ("manifest.json", functional.FUNCTIONAL_MANIFEST):
        (output / name).write_bytes((combined_plan / name).read_bytes())
    top = json.loads((output / "manifest.json").read_text())
    sidecar = output / functional.FUNCTIONAL_MANIFEST
    if damage == "sidecar":
        sidecar.write_text(sidecar.read_text().replace('"repeat": 1', '"repeat": 2', 1))
    elif damage == "symlink":
        sidecar.rename(output / "real.json")
        sidecar.symlink_to("real.json")
    elif damage == "missing":
        sidecar.unlink()
    elif damage == "digest":
        top["functional_plan"]["sha256"] = "0" * 64
    elif damage == "path":
        top["functional_plan"]["path"] = "../functional-manifest.json"
    elif damage == "extra":
        top["functional_plan"]["root"] = "/"
    elif damage == "schema":
        top["schema_version"] = 3
    elif damage == "identity":
        top["identity"]["target_sha"] = "e" * 40
    else:
        del top["functional_plan"]
    (output / "manifest.json").write_text(json.dumps(top))
    with pytest.raises(ValueError):
        functional.load_manifest(output / "manifest.json")


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/tags/v1.2.3"])
def test_push_main_is_combined_but_release_tags_run_ut_only(ref):
    result = auth.authorize({}, {}, context("push", ref), lambda path: pytest.fail(path))
    assert result["allowed"]
    if ref.startswith("refs/tags/"):
        assert result["test_kind"] == "unit" and "functional_policy" not in result
    else:
        assert result["test_kind"] == "combined" and result["labels"] == []
        assert result["functional_policy"] == functional.event_policy("push")


def test_merge_queue_is_combined():
    ctx = context("merge_group", "refs/heads/gh-readonly-queue/main/pr-7-" + BASE_SHA)
    ctx["sha"] = MERGE_SHA
    event = {
        "action": "checks_requested",
        "repository": {"full_name": REPOSITORY},
        "merge_group": {
            "base_ref": "refs/heads/main",
            "head_ref": ctx["ref"],
            "head_sha": MERGE_SHA,
            "base_sha": BASE_SHA,
        },
    }
    result = auth.authorize(event, {}, ctx, lambda path: pytest.fail(path))
    assert result["test_kind"] == "combined" and result["target_sha"] == MERGE_SHA
    assert result["functional_policy"] == functional.event_policy("merge_group")


@pytest.mark.parametrize(
    "labels,trigger",
    [
        ((), "pr"),
        (("Run functional tests", "bug"), "Run functional tests"),
        (("Run Merge-Queue jobs", "Run tests"), "Run Merge-Queue jobs"),
    ],
)
def test_pr_authorization_records_label_snapshot_and_policy(github, labels, trigger):
    label(github, *labels)
    result = authorize_pr(github)
    assert result["allowed"] and result["test_kind"] == "combined"
    assert result["labels"] == [name for name in labels if name in auth.SCOPE_LABELS]
    assert result["functional_policy"]["trigger"] == trigger
    assert auth.publish_status(result, "pending", result["reason"], github) == "published"


@pytest.mark.parametrize("action", ["labeled", "unlabeled"])
@pytest.mark.parametrize(
    "scope_label", ["Run tests", "Run functional tests", "Run Merge-Queue jobs"]
)
def test_scope_label_events_rerun_and_other_labels_are_ignored(github, action, scope_label):
    label(github, *([scope_label] if action == "labeled" else []))
    assert authorize_pr(github, action, scope_label)["decision"] == "run"
    ignored = authorize_pr(github, action, "bug")
    assert ignored["decision"] == "ignore" and not ignored["allowed"]


@pytest.mark.parametrize(
    "before,after",
    [
        ((), ("Run tests",)),
        ((), ("Run Merge-Queue jobs",)),
        (("Run Merge-Queue jobs",), ()),
        (("Run functional tests",), ("Run Merge-Queue jobs",)),
    ],
)
def test_label_change_invalidates_previous_authorization(github, before, after):
    label(github, *before)
    result = authorize_pr(github)
    assert auth.publish_status(result, "pending", result["reason"], github) == "published"
    assert auth.current_pr(github.pull, result, REPOSITORY)
    label(github, *after)
    assert not auth.current_pr(github.pull, result, REPOSITORY)
    assert not auth.execution_guard(result, github)


def test_merge_queue_label_does_not_bypass_fork_authorization(github):
    github.pull["head"]["repo"]["full_name"] = "someone/fork"
    label(github, "Run Merge-Queue jobs")
    result = authorize_pr(github, "labeled", "Run Merge-Queue jobs")
    assert not result["allowed"] and result["decision"] == "authorization_required"


class DocsGitHub(FakeGitHub):
    def __call__(self, path, payload=None):
        if path.startswith("pulls/7/files?"):
            self.calls.append(path)
            return [{"filename": "docs/index.md", "status": "modified"}]
        return super().__call__(path, payload)


@pytest.mark.parametrize(
    "labels,decision",
    [((), "cpu_only"), (("Run tests",), "run"), (("Run Merge-Queue jobs",), "run")],
)
def test_scope_label_overrides_cpu_only_exemption(monkeypatch, labels, decision):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    github = DocsGitHub()
    label(github, *labels)
    assert authorize_pr(github)["decision"] == decision


def test_status_publication_rejects_functional_but_accepts_combined(github):
    result = authorize_pr(github)
    assert auth.publish_status(result, "pending", result["reason"], github) == "published"
    assert auth.publish_status({**result, "test_kind": "functional"}, "success", "x", github) == (
        "rejected"
    )
