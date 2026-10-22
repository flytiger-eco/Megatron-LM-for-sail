import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".github/scripts"
SHA = "a" * 40
CONTROL_SHA = "b" * 40
BASE_SHA = "c" * 40
MERGE_SHA = "d" * 40
IDENTITY = {"target_sha": SHA, "control_sha": CONTROL_SHA}
BRANCH = "refs/heads/ci/ppu-validation"
REPOSITORY = "flytiger-eco/Megatron-LM-for-sail"
EXCLUDED_UT = "tests/unit_tests/rl/test_rl_utils.py::TestRLUtils::test_prepare_data_for_update"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = load_module("ppu_ci")
auth = load_module("ppu_pr_authorize")


def pr(fork=False):
    return {
        "state": "open",
        "draft": False,
        "changed_files": 1,
        "mergeable": True,
        "merge_commit_sha": MERGE_SHA,
        "head": {"sha": SHA, "repo": {"full_name": "someone/fork" if fork else REPOSITORY}},
        "base": {"ref": "ci/ppu-validation", "sha": BASE_SHA, "repo": {"full_name": REPOSITORY}},
    }


def context(event="workflow_dispatch", ref=BRANCH):
    return {
        "repository": REPOSITORY,
        "event_name": event,
        "ref": ref,
        "sha": CONTROL_SHA,
        "control_sha": CONTROL_SHA,
        "workflow_ref": f"{REPOSITORY}/.github/workflows/ppu-build.yml@{ref}",
        "actor": "maintainer",
        "triggering_actor": "maintainer",
        "run_id": "123",
        "run_attempt": "1",
    }


def api_for(pull=None, permission="write"):
    def api(path):
        if path.startswith("collaborators/"):
            return {"permission": permission}
        if path.startswith("pulls/7/files?"):
            return [{"filename": "megatron/core/utils.py", "status": "modified"}]
        if path == f"git/commits/{MERGE_SHA}":
            return {"parents": [{"sha": BASE_SHA}, {"sha": SHA}]}
        assert path == "pulls/7"
        return pull or pr()

    return api


@pytest.mark.parametrize("ref,scope", [(BRANCH, "full"), ("refs/tags/v1.2.3", "full")])
def test_automatic_scope_cannot_be_downgraded(ref, scope):
    result = ci.select_config(
        {"ref": ref}, "push", {"scope": "quick", "build": False, "unit_test": False, "repeat": 0}
    )
    assert result["scope"] == scope
    assert result["build"] and result["unit_test"] and result["repeat"] == 1


@pytest.mark.parametrize("fork", [False, True])
def test_automatic_pr_filter(fork):
    result = ci.select_config({"pull_request": pr(fork)}, "pull_request_target", {"scope": "full"})
    assert result["scope"] == "full"
    assert result["build"] is not fork
    assert result["unit_test"] is not fork


@pytest.mark.parametrize(
    "inputs", [{"build": False}, {"unit_test": False}, {"scope": "full", "repeat": 0}]
)
def test_manual_diagnostic_skip(inputs):
    assert not ci.select_config({"ref": BRANCH}, "workflow_dispatch", inputs)["unit_test"]


def test_manual_pr_full_requires_complete_execution():
    result = ci.select_config(
        {"ref": BRANCH},
        "workflow_dispatch",
        {
            "scope": "full",
            "pr_number": "7",
            "pr_sha": SHA,
            "pr_base_sha": BASE_SHA,
            "build": False,
            "unit_test": False,
            "repeat": 0,
        },
    )
    assert result["scope"] == "full" and result["threshold_percent"] == 90
    assert result["unit_test"] and result["build"] and result["repeat"] == 1


@pytest.mark.parametrize(
    "inputs",
    [
        {"scope": "invalid"},
        {"build": "maybe"},
        {"timeout": "0"},
        {"repeat": "-1"},
        {"pr_number": "7"},
        {"pr_sha": SHA},
        {"pr_number": "7", "pr_sha": "abc"},
    ],
)
def test_invalid_config(inputs):
    with pytest.raises(ValueError):
        ci.select_config({"ref": BRANCH}, "workflow_dispatch", inputs)


@pytest.mark.parametrize(
    "event,ref",
    [
        ("push", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/main"),
        ("pull_request", BRANCH),
        ("schedule", BRANCH),
    ],
)
def test_inactive_routes(event, ref):
    with pytest.raises(ValueError):
        ci.select_config({"ref": ref}, event, {})


def test_quick_selection_and_single_job(tmp_path, monkeypatch):
    output = {}
    monkeypatch.setattr(ci, "emit", lambda key, value: output.update({key: value}))
    manifest = ci.generate({"scope": "quick", "unit_test": True}, tmp_path, IDENTITY)
    assert manifest["identity"] == IDENTITY
    assert list(manifest["unit"]) == [f"dev-pr-{i:02d}" for i in range(1, 7)]
    cases = list(manifest["unit"].values())
    assert sum(c["minutes"] for c in cases) == 35
    assert sum(len(c["args"]) for c in cases) == 28
    assert len({path for c in cases for path in c["files"]}) == 22
    assert all(c["cards"] == 8 and c["repeat"] == 1 and c["environment"] == "dev" for c in cases)
    rows = json.loads(output["unit_matrix"])["include"]
    assert len(rows) == 1 and rows[0]["id"] == "quick"
    assert rows[0]["action_minutes"] == 45 and rows[0]["job_minutes"] == 50
    assert set(cases[-1]["excluded"]) == {EXCLUDED_UT}
    assert all(not case["excluded"] for case in cases[:-1])
    for case in cases:
        for selector in [*case["args"], *case["excluded"]]:
            filename, *symbols = selector.split("::")
            node = ast.parse((ROOT / filename).read_text())
            for symbol in symbols:
                node = next(child for child in node.body if getattr(child, "name", None) == symbol)


@pytest.mark.parametrize(
    "bad",
    [
        "../test.py",
        "tests/unit_tests/../test.py",
        "--ignore=x",
        "tests/unit_tests/not_found.py",
        "tests/unit_tests/test_utilities.py;exit",
        "tests/unit_tests/*.py",
    ],
)
def test_bad_selector(tmp_path, monkeypatch, bad):
    recipe = json.loads((ROOT / ci.POLICY_PATH).read_text())
    recipe["groups"][0]["selectors"] = [bad]
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/ppu-pr-tests.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(ci, "CONTROL_ROOT", tmp_path)
    with pytest.raises(ValueError):
        ci.quick_cases({})


@pytest.mark.parametrize("mutation", ["overlap", "duplicate", "empty", "budget", "version", "id"])
def test_invalid_recipe(tmp_path, monkeypatch, mutation):
    recipe = json.loads((ROOT / ".github/ppu-pr-tests.json").read_text())
    group = recipe["groups"][0]
    if mutation == "overlap":
        group["selectors"].append(group["selectors"][0] + "::test_anything")
    elif mutation == "duplicate":
        recipe["groups"][1]["selectors"].append(group["selectors"][0])
    elif mutation == "empty":
        group["selectors"] = []
    elif mutation == "budget":
        group["minutes"] += 1
    elif mutation == "version":
        recipe["schema_version"] = 3
    else:
        group["id"] = "../../oops"
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/ppu-pr-tests.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(ci, "CONTROL_ROOT", tmp_path)
    with pytest.raises(ValueError):
        ci.quick_cases({})


@pytest.mark.parametrize(
    "mutation", ["type", "invalid", "reason", "outside", "whole_selector", "overlap"]
)
def test_invalid_exclusions(tmp_path, monkeypatch, mutation):
    recipe = json.loads((ROOT / ".github/ppu-pr-tests.json").read_text())
    group = recipe["groups"][-1]
    if mutation == "type":
        recipe["excluded"] = [EXCLUDED_UT]
    elif mutation == "invalid":
        recipe["excluded"] = {"../test.py": "invalid"}
    elif mutation == "reason":
        recipe["excluded"][EXCLUDED_UT] = " "
    elif mutation == "outside":
        group["excluded"] = {EXCLUDED_UT: "must use common exclusions"}
    elif mutation == "whole_selector":
        recipe["excluded"] = {group["selectors"][0]: "entire file"}
    else:
        recipe["excluded"][EXCLUDED_UT.rsplit("::", 1)[0]] = "overlapping class"
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/ppu-pr-tests.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(ci, "CONTROL_ROOT", tmp_path)
    with pytest.raises(ValueError):
        ci.quick_cases({})


@pytest.mark.parametrize("repeat", [1, 2])
def test_full_recipe_parity_with_previous_implementation(repeat):
    original = subprocess.check_output(
        ["git", "show", "826a263f9:.github/scripts/ppu_ci.py"], cwd=ROOT, text=True
    )
    namespace = {"__file__": str(SCRIPTS / "ppu_ci.py"), "__name__": "baseline_ppu_ci"}
    exec(compile(original, "baseline_ppu_ci", "exec"), namespace)
    config = {"repeat": repeat, "timeout": 30}
    actual = ci.unit_cases(config)
    assert {
        key: {k: v for k, v in row.items() if k != "excluded"} for key, row in actual.items()
    } == namespace["unit_cases"](config)
    assert len(actual) == 74
    assert sum(EXCLUDED_UT in case["excluded"] for case in actual.values()) == 2
    assert all(
        not arg.startswith("--ppu-exclude") for case in actual.values() for arg in case["args"]
    )
    assert sum(EXCLUDED_UT.split("::")[0] in case["files"] for case in actual.values()) == 2
    for environment in ("dev", "lts"):
        files = [
            path
            for case in actual.values()
            if case["environment"] == environment
            for path in case["files"]
        ]
        assert len(files) == len(set(files))
        assert {path for path in files if Path(path).name.startswith("test_")} == {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tests/unit_tests").rglob("test_*.py")
        }


def test_configure_cli_separates_control_and_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    output.mkdir()
    for original in (ROOT / "tests/unit_tests").rglob("*.py"):
        path = source / original.relative_to(ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise RuntimeError('configure must not import source')\n")
    recipe = source / ci.RECIPE_PATH
    recipe.parent.mkdir(parents=True)
    recipe.write_bytes((ROOT / ci.RECIPE_PATH).read_bytes())
    parser = source / "tests/test_utils/python_scripts/recipe_parser.py"
    parser.parent.mkdir(parents=True)
    parser.write_text("raise RuntimeError('untrusted parser imported')\n")
    control_override = source / ".github/ppu-pr-tests.json"
    control_override.parent.mkdir()
    control_override.write_text("{}")
    (output / "authorization.json").write_text(json.dumps({"allowed": True, **IDENTITY}))
    event_path = tmp_path / "event.json"
    event_path.write_text("{}")
    env = {
        **os.environ,
        "PPU_SOURCE_ROOT": str(source),
        "PPU_CONTROL_ROOT": str(ROOT),
        "PPU_INPUTS": "{}",
        "PPU_DEFAULTS": "{}",
        "GITHUB_REF": BRANCH,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPTS / "ppu_ci.py"), "configure", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["identity"] == {**IDENTITY, "pr_head_sha": "", "base_sha": ""}
    assert manifest["config"]["scope"] == "full" and len(manifest["unit"]) == 37


@pytest.mark.parametrize("state", ["pending", "success", "failure"])
def test_commit_status_uses_pr_head_sha(tmp_path, monkeypatch, state):
    (tmp_path / "authorization.json").write_text(
        json.dumps(
            {
                "pr_number": 7,
                "pr_head_sha": SHA,
                "base_sha": BASE_SHA,
                "decision": "run",
                "run_id": "123",
                **IDENTITY,
            }
        )
    )
    calls = []

    def api(path, payload=None):
        if payload is not None:
            calls.append((path, payload))
            return {}
        if path == "pulls/7":
            return pr()
        if path == "actions/runs/123":
            return {"status": "in_progress", "run_attempt": 1}
        return [
            {
                "context": auth.STATUS_CONTEXT,
                "target_url": f"https://github.com/{REPOSITORY}/actions/runs/123",
            }
        ]

    monkeypatch.setattr(auth, "github_api", api)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setattr(
        sys, "argv", ["ppu_pr_authorize.py", "status", "--output", str(tmp_path), "--state", state]
    )
    auth.main()
    assert len(calls) == 1
    assert calls[0][0] == f"statuses/{SHA}"
    assert calls[0][1]["state"] == state
    assert calls[0][1]["context"] == "PPU PR Quick"
    assert calls[0][1]["target_url"] == f"https://github.com/{REPOSITORY}/actions/runs/123"


def test_manual_fork_authorization():
    result = auth.authorize(
        {}, {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA}, context(), api_for(pr(True))
    )
    assert result["allowed"] and result["manual_pr"]
    assert result["target_sha"] == MERGE_SHA and result["control_sha"] == CONTROL_SHA
    assert result["pr_head_sha"] == SHA and result["base_sha"] == BASE_SHA
    assert result["pr_number"] == 7


def test_fork_event_never_authorizes_ppu():
    event = {"number": 7, "pull_request": pr(True)}
    result = auth.authorize(event, {}, context("pull_request_target"), api_for(pr(True)))
    assert not result["allowed"] and result["pr_head_sha"] == SHA
    assert result["decision"] == "authorization_required"


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_authorization_requires_write(permission):
    with pytest.raises(ValueError):
        auth.authorize(
            {},
            {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA},
            context(),
            api_for(permission=permission),
        )


@pytest.mark.parametrize(
    "mutation", ["rerun", "ref", "workflow", "sha", "closed", "base", "repo", "head", "pair"]
)
def test_authorization_rejects_stale_or_untrusted_inputs(mutation):
    ctx, pull, inputs = (
        context(),
        pr(True),
        {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA},
    )
    if mutation == "rerun":
        ctx["run_attempt"] = "2"
    elif mutation == "ref":
        ctx["ref"] = "refs/heads/main"
    elif mutation == "workflow":
        ctx["workflow_ref"] = "attacker/repo/.github/workflows/ppu-build.yml@" + BRANCH
    elif mutation == "sha":
        pull["head"]["sha"] = "c" * 40
    elif mutation == "closed":
        pull["state"] = "closed"
    elif mutation == "base":
        pull["base"]["ref"] = "main"
    elif mutation == "repo":
        pull["base"]["repo"]["full_name"] = "attacker/repo"
    elif mutation == "head":
        pull["head"]["repo"] = None
    else:
        inputs.pop("pr_sha")
    with pytest.raises(ValueError):
        auth.authorize({}, inputs, ctx, api_for(pull))


def synthetic_case(case_id="dev-pr-01"):
    return {
        "id": case_id,
        "cards": 8,
        "repeat": 1,
        "minutes": 1,
        "environment": "dev",
        "strict_selection": True,
        "args": ["tests/unit_tests/test_example.py"],
    }


def write_reports(directory, case, failed=1, skipped=False, total=5):
    directory.mkdir(parents=True, exist_ok=True)
    nodes = [f"{case['args'][0]}::test_{i}" for i in range(total)]
    collected = nodes + [selector + "[0]" for selector in case.get("excluded", {})]
    for phase in ("prod", "experimental"):
        for rank in range(8):
            selected = nodes if phase == "prod" else []
            tests = {
                node: "skipped" if skipped else "failed" if rank == 7 and i < failed else "passed"
                for i, node in enumerate(selected)
            }
            report = {
                "rank": rank,
                "complete": True,
                "collection_errors": [],
                "exitstatus": 1 if "failed" in tests.values() else 0,
                "case_id": case["id"],
                "phase": phase,
                "iteration": 1,
                "tests": tests,
                "collected": collected,
                "selected": selected,
                "duration_seconds": 1.0,
                "collection_seconds": 0.1,
                **IDENTITY,
            }
            (directory / f"{phase}-1-rank-{rank}.json").write_text(json.dumps(report))
    (directory / "status.json").write_text(json.dumps({"complete": True, "exit_code": 0}))
    timing = {"started_at": 100, "completed_at": 110, "elapsed_seconds": 10}
    (directory / "timing.json").write_text(json.dumps(timing))
    (directory.parent / "build-timing.json").write_text(json.dumps(timing))
    (directory.parent / "suite-timing.json").write_text(json.dumps({**timing, "prepared_at": 102}))


@pytest.mark.parametrize("scope,threshold", [("quick", 100), ("full", 90), ("diagnostic", 90)])
@pytest.mark.parametrize("failed", [0, 1, 2])
def test_gate_counts_unique_tests_and_any_rank_failure(
    tmp_path, monkeypatch, capsys, scope, threshold, failed
):
    case = synthetic_case()
    manifest = {"config": {"scope": scope}, "identity": IDENTITY, "unit": {case["id"]: case}}
    write_reports(ci.result_directory(manifest, tmp_path, case["id"]), case, failed, total=10)
    result = ci.summarize(manifest, tmp_path)
    success = failed == 0 if scope == "quick" else failed <= 1
    assert result["success"] is success
    assert result["threshold_percent"] == threshold
    assert result["total"] == 10 and result["failed"] == failed
    assert not result["errors"]
    assert result["groups"][case["id"]]["timing"]["elapsed_seconds"] == 10
    assert result["groups"][case["id"]]["executed"] == 10
    if scope == "quick":
        assert result["timings"]["suite"]["prepare_seconds"] == 2
        assert result["timings"]["suite"]["execution_seconds"] == 8
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        sys,
        "argv",
        ["ppu_ci.py", "summarize", "--output", str(tmp_path), "--results", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        ci.main()
    assert exc.value.code == (0 if success else 1)
    assert json.loads((tmp_path / "summary.json").read_text())["threshold_percent"] == threshold
    assert f"threshold {threshold}%" in capsys.readouterr().out


@pytest.mark.parametrize("scope,success", [("quick", False), ("full", True), ("diagnostic", True)])
def test_scope_gate_across_groups(tmp_path, scope, success):
    cases = {f"dev-pr-{i:02d}": synthetic_case(f"dev-pr-{i:02d}") for i in (1, 2)}
    manifest = {"config": {"scope": scope}, "identity": IDENTITY, "unit": cases}
    for index, case in enumerate(cases.values()):
        write_reports(
            ci.result_directory(manifest, tmp_path, case["id"]), case, failed=1 if index == 0 else 0
        )
    result = ci.summarize(manifest, tmp_path)
    assert result["success"] is success and result["pass_rate"] == 0.9
    assert result["groups"]["dev-pr-01"]["counts"]["failed"] == 1


@pytest.mark.parametrize("scope", ["quick", "full", "diagnostic"])
def test_existing_partial_skips_are_not_counted_as_passes(tmp_path, scope):
    case = synthetic_case()
    manifest = {"config": {"scope": scope}, "identity": IDENTITY, "unit": {case["id"]: case}}
    directory = ci.result_directory(manifest, tmp_path, case["id"])
    write_reports(directory, case, failed=0)
    for path in directory.glob("prod-*-rank-*.json"):
        report = json.loads(path.read_text())
        report["tests"][report["selected"][0]] = "skipped"
        path.write_text(json.dumps(report))
    result = ci.summarize(manifest, tmp_path)
    assert result["success"] and result["pass_rate"] == 1
    assert result["passed"] == result["total"] == 4 and result["skipped"] == 1


@pytest.mark.parametrize("mutation", ["none", "executed", "missing", "all_excluded"])
def test_exclusion_report_integrity(tmp_path, capsys, mutation):
    case = synthetic_case()
    nodes = [f"{case['args'][0]}::test_{i}" for i in range(5)]
    case["excluded"] = {nodes[0]: "Known PP mock contract defect"}
    if mutation == "missing":
        case["excluded"] = {case["args"][0] + "::test_missing": "stale exclusion"}
    elif mutation == "all_excluded":
        case["excluded"] = dict.fromkeys(nodes, "nothing left to run")
    manifest = {"config": {"scope": "quick"}, "identity": IDENTITY, "unit": {case["id"]: case}}
    directory = ci.result_directory(manifest, tmp_path, case["id"])
    write_reports(directory, case, failed=0)
    if mutation == "missing":
        for path in directory.glob("*-rank-*.json"):
            report = json.loads(path.read_text())
            report["collected"] = nodes
            path.write_text(json.dumps(report))
    if mutation != "executed":
        for path in directory.glob("prod-*-rank-*.json"):
            report = json.loads(path.read_text())
            report["selected"] = [n for n in report["selected"] if n not in case["excluded"]]
            report["tests"] = {n: v for n, v in report["tests"].items() if n in report["selected"]}
            path.write_text(json.dumps(report))
    result = ci.summarize(manifest, tmp_path)
    assert result["success"] is (mutation == "none")
    assert bool(result["errors"]) is (mutation != "none")
    assert result["groups"][case["id"]]["excluded"] == case["excluded"]
    if mutation == "none":
        assert result["passed"] == result["total"] == 4 and result["skipped"] == 0
        ci.write_summary(result, tmp_path)
        assert nodes[0] in capsys.readouterr().out


@pytest.mark.parametrize(
    "filename", ["build-timing.json", "suite-timing.json", "dev-pr-01/timing.json"]
)
@pytest.mark.parametrize("mutation", ["missing", "nan", "inconsistent", "prepared"])
def test_timing_integrity(tmp_path, filename, mutation):
    case = synthetic_case()
    manifest = {"config": {"scope": "quick"}, "identity": IDENTITY, "unit": {case["id"]: case}}
    write_reports(ci.result_directory(manifest, tmp_path, case["id"]), case, failed=0)
    path = tmp_path / "ppu-ut-quick" / filename
    if mutation == "missing":
        path.rename(path.with_suffix(".missing"))
    else:
        timing = json.loads(path.read_text())
        if mutation == "nan":
            timing["elapsed_seconds"] = float("nan")
        elif mutation == "inconsistent":
            timing["completed_at"] = 109
        elif filename == "suite-timing.json":
            timing["prepared_at"] = 111
        else:
            timing["started_at"] = True
        path.write_text(json.dumps(timing))
    result = ci.summarize(manifest, tmp_path)
    assert not result["success"] and result["errors"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_rank",
        "collection",
        "fixture",
        "missing_phase",
        "missing_status",
        "incomplete",
        "exit",
        "identity",
        "phase",
        "selected",
        "duplicate",
        "outcome",
        "duration",
        "unplanned",
        "status",
        "missing_group",
        "skipped",
        "filtered",
        "missing_selector",
    ],
)
@pytest.mark.parametrize("scope", ["quick", "full", "diagnostic"])
def test_integrity_errors_cannot_be_hidden_by_pass_rate(tmp_path, mutation, scope):
    case = synthetic_case()
    manifest = {"config": {"scope": scope}, "identity": IDENTITY, "unit": {case["id"]: case}}
    directory = ci.result_directory(manifest, tmp_path, case["id"])
    write_reports(directory, case, failed=0, skipped=mutation == "skipped")
    path = directory / "prod-1-rank-7.json"
    report = json.loads(path.read_text())
    if mutation == "missing_rank":
        path.rename(path.with_suffix(".missing"))
    elif mutation == "collection":
        report["collection_errors"] = ["broken"]
    elif mutation == "fixture":
        report["execution_errors"] = ["setup: broken"]
    elif mutation == "missing_phase":
        for phase_path in directory.glob("experimental-*.json"):
            phase_path.rename(phase_path.with_suffix(".missing"))
    elif mutation == "missing_status":
        (directory / "status.json").rename(directory / "status.missing")
    elif mutation == "incomplete":
        report["complete"] = False
    elif mutation == "exit":
        report["exitstatus"] = 2
    elif mutation == "identity":
        report["target_sha"] = CONTROL_SHA
    elif mutation == "phase":
        report["phase"] = "experimental"
    elif mutation == "selected":
        report["selected"] = []
    elif mutation == "duplicate":
        report["collected"].append(report["collected"][0])
    elif mutation == "outcome":
        report["tests"][report["selected"][0]] = "unknown"
    elif mutation == "duration":
        report["duration_seconds"] = float("nan")
    elif mutation == "unplanned":
        case["args"] = ["tests/unit_tests/test_other.py"]
    elif mutation == "status":
        (directory / "status.json").write_text('{"complete":true,"exit_code":124}')
    elif mutation == "missing_group":
        manifest["unit"]["dev-pr-02"] = synthetic_case("dev-pr-02")
    elif mutation == "missing_selector":
        case["args"].append("tests/unit_tests/test_missing.py")
    elif mutation == "filtered":
        for report_path in directory.glob("*-rank-*.json"):
            item = json.loads(report_path.read_text())
            item.update(selected=[], tests={})
            report_path.write_text(json.dumps(item))
    if mutation not in ("missing_rank", "filtered"):
        path.write_text(json.dumps(report))
    result = ci.summarize(manifest, tmp_path)
    assert result["success"] is False
    assert result["errors"]


def test_full_matrix_report_parity(tmp_path):
    cases = ci.unit_cases({"repeat": 1, "timeout": 30})
    manifest = {"config": {"scope": "full"}, "identity": IDENTITY, "unit": cases}
    for case in cases.values():
        write_reports(ci.result_directory(manifest, tmp_path, case["id"]), case)
    result = ci.summarize(manifest, tmp_path)
    assert not result["success"] and not result["errors"]
    assert result["threshold_percent"] == 90
    assert result["total"] == len(cases) * 5
    assert result["failed"] == len(cases)
    assert len(result["groups"]) == 74 and result["pass_rate"] == 0.8


@pytest.mark.parametrize("scope", ["quick", "full", "diagnostic"])
def test_matrix_failure_blocks_success(tmp_path, monkeypatch, scope):
    case = synthetic_case()
    manifest = {"config": {"scope": scope}, "identity": IDENTITY, "unit": {case["id"]: case}}
    write_reports(ci.result_directory(manifest, tmp_path, case["id"]), case, failed=0)
    monkeypatch.setenv("PPU_MATRIX_RESULT", "failure")
    assert not ci.summarize(manifest, tmp_path)["success"]


@pytest.mark.parametrize(
    "source,complete,extra",
    [
        ("def test_ok(): assert True\ndef test_failure(): assert False\n", True, []),
        ("import missing_ppu_dependency\n", False, []),
        ("import pytest\n@pytest.mark.skip\ndef test_skip(): pass\n", True, []),
        ("def test_ok(): pass\n", False, ["--keep-duplicates", "test_sample.py"]),
        ("def test_ok(): pass\n", True, ["-m", "experimental"]),
        (
            "import pytest\nclass TestExample:\n"
            "    @pytest.mark.parametrize('value', [1, 2])\n"
            "    def test_excluded(self, value): assert False\n"
            "    def test_excluded_extra(self): pass\n"
            "class TestOther:\n    def test_excluded(self): pass\n",
            True,
            ["--ppu-exclude=test_sample.py::TestExample::test_excluded"],
        ),
        (
            "class TestExample:\n"
            "    def test_excluded(self): assert False\n"
            "    def test_excluded_extra(self): pass\n"
            "class TestOther:\n    def test_excluded(self): pass\n",
            True,
            ["--ppu-exclude=test_sample.py::TestExample::test_excluded"],
        ),
        (
            "def setup_function(): raise RuntimeError('setup failed')\ndef test_ok(): pass\n",
            False,
            [],
        ),
        (
            "def teardown_function(): raise RuntimeError('teardown failed')\ndef test_ok(): pass\n",
            False,
            [],
        ),
    ],
)
def test_real_pytest_plugin_reports(tmp_path, source, complete, extra):
    (tmp_path / "test_sample.py").write_text(source)
    env = {
        **os.environ,
        "PYTHONPATH": str(SCRIPTS),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_PLUGINS": "ppu_pytest_policy",
        "PPU_CASE_ID": "dev-pr-01",
        "PPU_TEST_PHASE": "prod",
        "PPU_TEST_ITERATION": "1",
        "PPU_REPORT_DIR": str(tmp_path),
        "RANK": "0",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(tmp_path),
            "-q",
            "test_sample.py",
            *extra,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / "prod-1-rank-0.json").read_text())
    assert report["complete"] is complete, result.stdout + result.stderr
    assert (result.returncode == 0) is complete
    assert report["duration_seconds"] >= 0
    assert set(report["selected"]) == set(report["tests"])
    assert set(report["selected"]) <= set(report["collected"])
    if any(arg.startswith("--ppu-exclude=") for arg in extra):
        assert report["tests"] == {
            "test_sample.py::TestExample::test_excluded_extra": "passed",
            "test_sample.py::TestOther::test_excluded": "passed",
        }
        assert len(report["collected"]) == (4 if "parametrize" in source else 3)


def test_workflow_trust_and_resource_boundaries():
    build = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    ut = yaml.safe_load((ROOT / ".github/workflows/ppu-ut.yml").read_text())
    events = build.get("on", build.get(True))
    assert set(events) == {"push", "pull_request_target", "workflow_dispatch", "merge_group"}
    assert events["push"]["tags"] == ["v*"]
    assert events["pull_request_target"]["branches"] == ["ci/ppu-validation"]
    assert build["jobs"]["authorize"]["runs-on"] == "ubuntu-latest"
    assert "allowed == 'true'" in build["jobs"]["configure"]["if"]
    for name in ("build", "unit-tests"):
        assert "github.run_attempt == 1" in build["jobs"][name]["if"]
    assert "github.run_attempt == 1" in ut["jobs"]["test"]["if"]
    assert ut["jobs"]["test"]["strategy"]["max-parallel"] == 1
    assert "inputs.threshold_percent" in ut["jobs"]["result"]["name"]
    for field in ("scope", "threshold_percent"):
        assert build["jobs"]["configure"]["outputs"][field] == (
            "${{ steps.config.outputs." + field + " }}"
        )
        assert build["jobs"]["unit-tests"]["with"][field] == (
            "${{ needs.configure.outputs." + field + " }}"
        )
        assert (
            "needs.configure.outputs." + field
            in build["jobs"]["pr-result"]["steps"][-1]["env"]["DESCRIPTION"]
        )
    assert "GH_TOKEN" not in build["jobs"]["build"].get("env", {})
    for workflow in (build, ut):
        for job in workflow["jobs"].values():
            if job.get("permissions", {}).get("statuses") == "write":
                assert job["runs-on"] == "ubuntu-latest"
            for step in job.get("steps", []):
                if step.get("uses") == "actions/checkout@v4":
                    assert step["with"]["persist-credentials"] is False
    text = (ROOT / ".github/workflows/ppu-ut.yml").read_text()
    assert 'bash "$PPU_CONTROL_DIR/ppu_ci_run.sh"' in text
    assert "python3 -I .ppu-ci/control/ppu_ci.py summarize" in text
    build_script = (SCRIPTS / "ppu_ci_build.sh").read_text()
    assert 'install -m 0644 "$PPU_CONTROL_DIR/ppu_pytest_policy.py"' in build_script
    assert "install -m 0644 .github/scripts/ppu_pytest_policy.py" not in build_script
    assert "secrets: inherit" not in (ROOT / ".github/workflows/ppu-build.yml").read_text()
    assert "release-{0}" in build["concurrency"]["group"]
    assert "full-{0}" in build["concurrency"]["group"]
    assert "inputs.pr_number == ''" in build["concurrency"]["cancel-in-progress"]


TOOL_STUB = r'''
import json, os, pathlib, shutil, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ['TEST_ROOT'])
with (root / 'commands.jsonl').open('a') as out:
    out.write(json.dumps({'name':name,'args':args,'pid':os.getpid()})+'\n')
if name == 'mkdir':
    for value in args:
        if value.startswith(str(root)):
            pathlib.Path(value).mkdir(parents=True, exist_ok=True)
elif name == 'cp' and args[-1].endswith('/build-timing.json'):
    shutil.copyfile(args[-2], args[-1])
elif name == 'timeout':
    if os.environ['TEST_FAILURE'] == 'timeout' and os.environ['PPU_CASE_ID'] == 'dev-pr-01':
        sys.exit(124)
    sys.exit(subprocess.run(args[3:]).returncode)
elif name in ('python', 'python3'):
    if len(args) == 2 and args[0] == '-' or args and args[0].endswith('ppu_ci.py'):
        os.execv(sys.executable, [sys.executable] + args)
    if args[:3] == ['-m','torch.distributed.run','--standalone']:
        case_id, phase = os.environ['PPU_CASE_ID'], os.environ['PPU_TEST_PHASE']
        manifest = json.loads(pathlib.Path('.ppu-ci/manifest.json').read_text())
        case = manifest['unit'][case_id]
        node = case['args'][0] + '::test_example'
        collected = [node] + [s + '[0]' for s in case.get('excluded', {})]
        nodes = [node] if phase == 'prod' else []
        for rank in range(8):
            report = {'rank':rank,'case_id':case_id,'phase':phase,'iteration':1,'complete':True,
                      'collection_errors':[],'exitstatus':0,'collected':collected,'selected':nodes,
                      'tests':{n:'passed' for n in nodes},'duration_seconds':0.01,**manifest['identity']}
            path = pathlib.Path(os.environ['PPU_REPORT_DIR']) / f'{phase}-1-rank-{rank}.json'
            path.write_text(json.dumps(report))
    elif args[:3] == ['-m','coverage','combine']:
        sys.exit(1)
'''


@pytest.mark.parametrize("failure", ["none", "timeout", "budget"])
def test_suite_runner_with_fake_hardware(tmp_path, failure):
    source, artifacts, results, bin_dir = [
        tmp_path / name for name in ("source", "artifacts", "ppu-ut-quick", "bin")
    ]
    for path in (source, artifacts, results, bin_dir):
        path.mkdir()
    (artifacts / "data").mkdir()
    (artifacts / "data/.ready").touch()
    cases = {f"dev-pr-{i:02d}": synthetic_case(f"dev-pr-{i:02d}") for i in range(1, 7)}
    cases["dev-pr-06"]["excluded"] = {
        "tests/unit_tests/test_example.py::test_excluded": "known defect"
    }
    (artifacts / "build-timing.json").write_text(
        json.dumps({"started_at": 1, "completed_at": 2, "elapsed_seconds": 1})
    )
    if failure == "budget":
        for case in cases.values():
            case["minutes"] = 0
    manifest = {"config": {"scope": "quick"}, "identity": IDENTITY, "unit": cases}
    (artifacts / "manifest.json").write_text(json.dumps(manifest))
    for name in (
        "python",
        "python3",
        "mkdir",
        "tar",
        "make",
        "install",
        "cp",
        "timeout",
        "ppu-smi",
    ):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n" + TOOL_STUB)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_ROOT": str(tmp_path),
        "TEST_FAILURE": failure,
        "PPU_RESULTS_DIR": str(results),
        "PPU_SOURCE_ROOT": str(source),
        "PPU_ARTIFACT_DIR": str(artifacts),
        "PPU_CONTROL_DIR": str(SCRIPTS),
        "PPU_TARGET_SHA": SHA,
        "PPU_CONTROL_SHA": CONTROL_SHA,
        "NPROC_PER_NODE": "8",
        "NNODES": "1",
    }
    run = subprocess.run(
        ["bash", str(SCRIPTS / "ppu_ci_run.sh"), "quick"], env=env, capture_output=True, text=True
    )
    assert run.returncode == (0 if failure == "none" else 1), run.stdout + run.stderr
    commands = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
    restores = [
        c for c in commands if c['name'] == 'tar' and 'environment.tar.gz' in ' '.join(c['args'])
    ]
    assert len(restores) == 1
    calls = [
        c
        for c in commands
        if c['name'] == 'python' and c['args'][:2] == ['-m', 'torch.distributed.run']
    ]
    assert len(calls) == {"none": 12, "timeout": 10, "budget": 0}[failure]
    assert len({c['pid'] for c in calls}) == len(calls)
    for call in calls:
        args = call['args']
        experimental = '--experimental' in args
        marker = (
            'experimental and not flaky_in_dev'
            if experimental
            else 'not experimental and not flaky_in_dev'
        )
        assert marker in args
        exclusions = [arg for arg in args if arg.startswith('--ppu-exclude=')]
        if '/dev-pr-06/' in args[args.index('--log-dir') + 1]:
            assert exclusions == ['--ppu-exclude=tests/unit_tests/test_example.py::test_excluded']
        else:
            assert exclusions == []
        assert args[args.index('--nproc-per-node') + 1] == '8'
        assert args[args.index('--log-dir') + 1].endswith(
            '/experimental' if experimental else '/prod'
        )
    for case_id in cases:
        status = json.loads((results / case_id / 'status.json').read_text())
        assert status['complete'] is (
            failure == 'none' or failure == 'timeout' and case_id != 'dev-pr-01'
        )
    timeouts = [c for c in commands if c['name'] == 'timeout']
    assert all(0 < int(c['args'][2]) <= 60 for c in timeouts)
    result = ci.summarize(manifest, tmp_path)
    assert result["success"] is (failure == "none")
    assert len(result["groups"]) == 6
    if failure == "none":
        ci.write_summary(result, results)
        assert json.loads((results / "summary.json").read_text())["total"] == 6
