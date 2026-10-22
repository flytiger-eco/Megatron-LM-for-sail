import copy
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

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


def test_pla_installation_preserves_runtime_constraints(tmp_path, monkeypatch):
    script = (ROOT / ".github/scripts/ppu_ci_build.sh").read_text()
    block = next(
        block
        for block in re.findall(r"<<'PY'\n(.*?)\nPY", script, re.S)
        if "native_requirements =" in block
    )
    runtimes = {
        "torch": "2.13.0",
        "torchvision": "0.24.0",
        "triton": "3.5.0",
        "scipy": "1.16.0",
        "scikit-learn": "1.7.2",
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(metadata, "version", runtimes.__getitem__)
    artifact = tmp_path / ".ppu-ci"
    artifact.mkdir()
    exec(compile(block, "ppu-native-requirements", "exec"), {})
    requirements = (artifact / "ppu-requirements.txt").read_text().splitlines()
    constraints = (artifact / "constraints.txt").read_text().splitlines()
    assert requirements.count("pla==1.0.0+v0.1.0.ppu2.2.0") == 1
    assert set(requirements) <= set(constraints)
    assert {f"{name}=={version}" for name, version in runtimes.items()} <= set(constraints)
    assert {"numpy==1.26.4", "pyarrow==21.0.0"} <= set(constraints)
    assert not any(line.startswith(("fla==", "flash-linear-attention==")) for line in requirements)
    assert (
        "export PIP_INDEX_URL=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"
        in script
    )
    assert "PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' PIP_FIND_LINKS=''" in script
    install = script.index("-r .ppu-ci/test-requirements.txt -r .ppu-ci/ppu-requirements.txt")
    assert script.index("export PIP_CONSTRAINT=") < install
    assert (
        install
        < script.index("PLA 导入检查通过:")
        < script.index("tar -czf .ppu-ci/environment.tar.gz")
    )


@pytest.mark.parametrize("script_name", ["ppu_ci_build.sh", "ppu_ci_run.sh"])
@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "wrong_version",
        "external_path",
        "missing_package",
        "missing_module",
        "native_error",
    ],
)
def test_pla_probe_checks_version_origin_and_native_imports(monkeypatch, script_name, mode):
    script = (ROOT / ".github/scripts" / script_name).read_text()
    probe = next(
        block
        for block in re.findall(r"python -I - <<'PY'\n(.*?)\nPY", script, re.S)
        if "md.version('pla')" in block
    )
    if script_name == "ppu_ci_run.sh":
        assert script.index("tar -xzf .ppu-ci/environment.tar.gz") < script.index(probe)
        assert "pip install" not in script
    modules = ("pla", "pla.decode", "pla.prefill.flashqla", "pla.prefill.flashkdapro")
    imported = []

    def version(name):
        assert name == "pla"
        if mode == "missing_package":
            raise metadata.PackageNotFoundError(name)
        return "1.0.0" if mode == "wrong_version" else "1.0.0+v0.1.0.ppu2.2.0"

    def import_module(name):
        assert name in modules
        imported.append(name)
        if mode == "missing_module" and name == "pla.decode":
            raise ImportError("PLA 模块缺失")
        if mode == "native_error" and name == "pla.prefill.flashqla":
            raise OSError("PLA 原生扩展加载失败")
        root = Path(sys.prefix).parent if mode == "external_path" else Path(sys.prefix)
        return SimpleNamespace(__file__=str(root / "site-packages" / name / "__init__.py"))

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(importlib, "import_module", import_module)
    if mode == "valid":
        exec(compile(probe, script_name, "exec"), {})
        assert tuple(imported) == modules
    else:
        with pytest.raises((RuntimeError, ImportError, OSError)):
            exec(compile(probe, script_name, "exec"), {})


@pytest.mark.parametrize("script_name", ["ppu_ci_build.sh", "ppu_ci_run.sh"])
def test_build_and_run_embedded_python_syntax(script_name):
    script = (ROOT / ".github/scripts" / script_name).read_text()
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, re.S)
    assert blocks
    for block in blocks:
        compile(block, script_name, "exec")


GDN_BUCKET = "tests/unit_tests/ssm/test_gated_delta_net.py"


def diagnostic_config(**inputs):
    return ci.select_config(
        {"ref": BRANCH},
        "workflow_dispatch",
        {"scope": "full", "suite_family": "nvidia-dev", "unit_bucket": GDN_BUCKET, **inputs},
    )


def test_diagnostic_generates_one_unchanged_upstream_bucket(tmp_path, capsys):
    config = diagnostic_config()
    assert config["scope"] == "diagnostic" and config["trigger_kind"] == "manual"
    assert config["threshold_percent"] == 80
    full = ci.unit_cases(config)
    manifest = ci.generate(config, tmp_path, IDENTITY)
    assert list(manifest["unit"]) == ["dev-023"]
    assert manifest["unit"]["dev-023"] == full["dev-023"]
    assert manifest["unit"]["dev-023"]["args"] == [GDN_BUCKET]
    assert len(full) == 37
    output = capsys.readouterr().out
    matrix = json.loads(
        next(
            line.removeprefix("unit_matrix=")
            for line in output.splitlines()
            if line.startswith("unit_matrix=")
        )
    )
    assert len(matrix["include"]) == 1
    assert matrix["include"][0]["id"] == "dev-023"
    assert matrix["include"][0]["cards"] == 8
    workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    events = workflow.get("on", workflow.get(True))
    assert events["workflow_dispatch"]["inputs"]["unit_bucket"]["default"] == ""
    assert len(events["workflow_dispatch"]["inputs"]) <= 10


@pytest.mark.parametrize(
    "inputs",
    [
        {"scope": "auto"},
        {"scope": "quick"},
        {"suite_family": "release-dev-lts"},
        {"build": False},
        {"unit_test": False},
        {"repeat": 0},
        {"repeat": 2},
        {"unit_bucket": None},
        {"unit_bucket": [GDN_BUCKET]},
        {"pr_number": "7", "pr_sha": SHA, "pr_base_sha": BASE_SHA},
    ],
)
def test_diagnostic_rejects_incompatible_manual_inputs(inputs):
    with pytest.raises(ValueError):
        diagnostic_config(**inputs)


@pytest.mark.parametrize(
    "event_name,event",
    [
        ("push", {"ref": BRANCH}),
        ("push", {"ref": "refs/tags/v1.0"}),
        ("pull_request_target", {"pull_request": pr()}),
        ("merge_group", {"merge_group": {"base_ref": BRANCH}}),
    ],
)
def test_diagnostic_cannot_reduce_automatic_coverage(event_name, event):
    with pytest.raises(ValueError, match="仅允许非 PR 手动"):
        ci.select_config(event, event_name, {"scope": "full", "unit_bucket": GDN_BUCKET})


@pytest.mark.parametrize(
    "bucket",
    [
        GDN_BUCKET + "::test_constructor",
        "tests/unit_tests/ssm/*.py",
        "../test.py",
        GDN_BUCKET + ";echo x",
        GDN_BUCKET + "\n",
        "tests/unit_tests/ssm/missing.py",
    ],
)
def test_diagnostic_requires_exact_recipe_bucket(tmp_path, bucket):
    with pytest.raises(ValueError, match="精确匹配"):
        ci.generate(diagnostic_config(unit_bucket=bucket), tmp_path)


@pytest.mark.parametrize("state", ["passed", "all_skipped", "missing_rank", "missing_timing"])
def test_diagnostic_results_never_count_as_full_baseline(tmp_path, capsys, state):
    config = diagnostic_config()
    manifest = ci.generate(config, tmp_path / "plan", IDENTITY)
    case = manifest["unit"]["dev-023"]
    directory = ci.result_directory(manifest, tmp_path, case["id"])
    write_reports(directory, case, failed=0, skipped=state == "all_skipped")
    for path in directory.glob("*-rank-*.json"):
        report = json.loads(path.read_text())
        report["durations"] = dict.fromkeys(report["tests"], 0.1)
        path.write_text(json.dumps(report))
    timing = {"started_at": 100, "prepared_at": 102, "completed_at": 110, "elapsed_seconds": 10}
    (directory / "suite-timing.json").write_text(json.dumps(timing))
    (directory / "build-timing.json").write_text(json.dumps(timing))
    if state == "missing_rank":
        (directory / "prod-1-rank-7.json").rename(directory / "missing-rank")
    if state == "missing_timing":
        (directory / "suite-timing.json").rename(directory / "missing-timing")
    result = ci.summarize(manifest, tmp_path)
    assert result["success"] is (state == "passed")
    assert not result["baseline_complete"]
    assert result["scope"] == "diagnostic"
    ci.write_summary(result, tmp_path / "summary")
    assert "仅单 bucket 诊断" in capsys.readouterr().out


def test_diagnostic_runner_selects_only_requested_group(tmp_path, monkeypatch, capsys):
    manifest = ci.generate(diagnostic_config(), tmp_path / ".ppu-ci", IDENTITY)
    capsys.readouterr()
    script = (ROOT / ".github/scripts/ppu_ci_run.sh").read_text()
    block = next(
        block
        for block in re.findall(r"<<'PY'\n(.*?)\nPY", script, re.S)
        if "suite" not in block and "print(sum(case['minutes']" in block
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PPU_TARGET_SHA", SHA)
    monkeypatch.setenv("PPU_CONTROL_SHA", CONTROL_SHA)
    monkeypatch.setattr(sys, "argv", ["-", "dev-023"])
    exec(compile(block, "ppu-suite-settings", "exec"), {})
    assert capsys.readouterr().out.splitlines() == [
        str(manifest["unit"]["dev-023"]["minutes"] * 60),
        "dev-023",
    ]


@pytest.fixture
def sealed_bundle(tmp_path):
    plan = tmp_path / "plan"
    control = plan / "control"
    control.mkdir(parents=True)
    (plan / "manifest.json").write_text(json.dumps({"identity": IDENTITY}))
    shutil.copyfile(ROOT / ".github/scripts/ppu_ci.py", control / "ppu_ci.py")
    (control / "ppu_ci_run.sh").write_text('printf "UT_STARTED:%s\\n" "$1"\n')
    (control / "ppu_pytest_policy.py").write_text("POLICY = True\n")
    bundle = tmp_path / "nas" / "run-1" / "build"
    shutil.copytree(plan, bundle)
    payload = tmp_path / "source.py"
    payload.write_text("VALUE = 1\n")
    with tarfile.open(bundle / "source.tar.gz", "w:gz") as archive:
        archive.add(payload, arcname="source.py")
    (bundle / "environment.tar.gz").write_bytes(b"environment archive")
    (bundle / "commit.txt").write_text(SHA + "\n")
    (bundle / "control.txt").write_text(CONTROL_SHA + "\n")
    (bundle / "image.txt").write_text("fixed-image\n")
    (bundle / "build-timing.json").write_text('{"elapsed_seconds":1}')
    (bundle / "data").mkdir()
    (bundle / "data/blob").write_bytes(b"tokenizer data")
    (bundle / "data/tokenizer").symlink_to("blob")
    (bundle / "data/.ready").touch()
    ci.seal_bundle(bundle)
    (bundle / ".complete").write_text(SHA + "\n")
    return plan, bundle


def bundle_digest(bundle):
    return hashlib.sha256((bundle / "bundle-integrity.json").read_bytes()).hexdigest()


def test_bundle_seal_and_verify_preserve_all_payloads(sealed_bundle, capsys):
    plan, bundle = sealed_bundle
    ci.verify_bundle(bundle, plan, bundle_digest(bundle))
    assert "构建产物完整性校验通过" in capsys.readouterr().out
    files = json.loads((bundle / "bundle-integrity.json").read_text())["files"]
    assert files["data/blob"] == files["data/tokenizer"]
    assert "data/.ready" in files
    assert ".complete" not in files and "bundle-integrity.json" not in files


@pytest.mark.parametrize(
    "damage",
    [
        "missing_source",
        "truncated_environment",
        "missing_data",
        "changed_control",
        "extra_control",
        "wrong_digest",
        "wrong_complete",
        "missing_complete",
        "changed_plan",
        "external_symlink",
        "symlink_directory",
        "altered_integrity",
        "wrong_commit",
        "wrong_control_sha",
        "sealed_control",
        "sealed_extra_control",
        "sealed_manifest",
        "traversal_inventory",
    ],
)
def test_bundle_verify_rejects_partial_or_untrusted_payload(sealed_bundle, tmp_path, damage):
    plan, bundle = sealed_bundle
    digest = bundle_digest(bundle)
    if damage.startswith("missing_"):
        name = {
            "missing_source": "source.tar.gz",
            "missing_data": "data/tokenizer",
            "missing_complete": ".complete",
        }[damage]
        (bundle / name).rename(tmp_path / "missing")
    elif damage == "truncated_environment":
        (bundle / "environment.tar.gz").write_bytes(b"part")
    elif damage in ("changed_control", "sealed_control"):
        (bundle / "control/ppu_ci_run.sh").write_text("exit 0\n")
    elif damage in ("extra_control", "sealed_extra_control"):
        (bundle / "control/extra.py").write_text("EXTRA = True\n")
    elif damage == "wrong_digest":
        digest = "0" * 64
    elif damage == "wrong_complete":
        (bundle / ".complete").write_text(CONTROL_SHA)
    elif damage in ("changed_plan", "sealed_manifest"):
        root = plan if damage == "changed_plan" else bundle
        (root / "manifest.json").write_text(
            json.dumps({"identity": {**IDENTITY, "target_sha": CONTROL_SHA}})
        )
    elif damage == "external_symlink":
        (bundle / "external").symlink_to(tmp_path / "source.py")
    elif damage == "symlink_directory":
        (bundle / "linked-data").symlink_to("data", target_is_directory=True)
    elif damage == "altered_integrity":
        (bundle / "bundle-integrity.json").write_text("{}")
    elif damage in ("wrong_commit", "wrong_control_sha"):
        name = "commit.txt" if damage == "wrong_commit" else "control.txt"
        (bundle / name).write_text("f" * 40)
    elif damage == "traversal_inventory":
        seal = json.loads((bundle / "bundle-integrity.json").read_text())
        seal["files"]["../../source.py"] = seal["files"]["source.tar.gz"]
        (bundle / "bundle-integrity.json").write_text(json.dumps(seal))
        digest = bundle_digest(bundle)
    if damage.startswith("sealed_") or damage in ("wrong_commit", "wrong_control_sha"):
        ci.seal_bundle(bundle)
        digest = bundle_digest(bundle)
    with pytest.raises((ValueError, OSError)):
        ci.verify_bundle(bundle, plan, digest)


@pytest.mark.parametrize("name", ["source.tar.gz", "environment.tar.gz", "control/ppu_ci.py"])
def test_bundle_seal_rejects_empty_required_payload(sealed_bundle, name):
    _, bundle = sealed_bundle
    (bundle / name).write_bytes(b"")
    with pytest.raises(ValueError, match="缺少或为空"):
        ci.seal_bundle(bundle)


@pytest.mark.parametrize("damage", ["none", "missing_source", "wrong_digest", "other_run"])
def test_ut_workflow_restores_same_run_bundle_before_execution(sealed_bundle, tmp_path, damage):
    plan, bundle = sealed_bundle
    digest = bundle_digest(bundle)
    workspace = tmp_path / "pod"
    workspace.mkdir()
    shutil.copytree(plan, workspace / ".ppu-ci")
    if damage == "missing_source":
        (bundle / "source.tar.gz").rename(tmp_path / "missing-source")
    elif damage == "wrong_digest":
        digest = "0" * 64
    elif damage == "other_run":
        bundle = bundle.parent.parent / "other-run" / "build"
    workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-ut.yml").read_text())
    action = next(step for step in workflow["jobs"]["test"]["steps"] if step.get("id") == "ppu")
    command = (
        action["with"]["command"]
        .replace("cd /opt/megatron-lm", 'cd "$TEST_WORKSPACE"')
        .replace("${{ matrix.id }}", "dev-023")
    )
    env = {
        **os.environ,
        "TEST_WORKSPACE": str(workspace),
        "PPU_BUNDLE_DIR": str(bundle),
        "PPU_ARTIFACT_DIR": str(workspace / ".ppu-bundle"),
        "PPU_CONTROL_DIR": str(workspace / ".ppu-ci/control"),
        "PPU_BUNDLE_DIGEST": digest,
        "PPU_TARGET_SHA": SHA,
        "PPU_CONTROL_SHA": CONTROL_SHA,
    }
    result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is (damage == "none"), result.stdout + result.stderr
    assert ("UT_STARTED:dev-023" in result.stdout) is (damage == "none")
    if damage == "none":
        assert (workspace / "source/source.py").read_text() == "VALUE = 1\n"
        assert (workspace / ".ppu-bundle/data/tokenizer").is_symlink()
        assert not (bundle / "source").exists()


def test_bundle_workflow_digest_and_run_scoping():
    build = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    ut = yaml.safe_load((ROOT / ".github/workflows/ppu-ut.yml").read_text())
    assert (
        build["jobs"]["build"]["outputs"]["bundle_digest"]
        == "${{ steps.bundle.outputs.bundle_digest }}"
    )
    assert (
        build["jobs"]["unit-tests"]["with"]["bundle_digest"]
        == "${{ needs.build.outputs.bundle_digest }}"
    )
    downloads = [
        step["with"]["name"]
        for step in ut["jobs"]["test"]["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    ]
    assert downloads == ["ppu-plan"]
    action = next(step for step in ut["jobs"]["test"]["steps"] if step.get("id") == "ppu")
    assert (
        "PPU_BUNDLE_DIR=/mnt/wl_nas/devops/megatron-ci/${{ github.run_id }}-${{ github.run_attempt }}/build"
        in action["with"]["extra_env"]
    )
    assert "PPU_BUNDLE_DIGEST=${{ inputs.bundle_digest }}" in action["with"]["extra_env"]
    command = action["with"]["command"]
    assert (
        command.index("verify-bundle") < command.index("tar -xzf") < command.index("ppu_ci_run.sh")
    )
    script = (ROOT / ".github/scripts/ppu_ci_build.sh").read_text()
    assert script.index("seal-bundle") > script.index("tar -czf .ppu-ci/environment.tar.gz")
