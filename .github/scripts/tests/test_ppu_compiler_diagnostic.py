"""编译诊断的入口互斥、授权、预算和栈采集 CPU 测试。"""

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import yaml
from test_ppu_quick import auth, context

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".github/scripts"
spec = importlib.util.spec_from_file_location(
    "ppu_compiler_diagnostic", SCRIPTS / "ppu_compiler_diagnostic.py"
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
spec = importlib.util.spec_from_file_location(
    "ppu_functional_diagnostic_route", SCRIPTS / "ppu_functional.py"
)
functional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functional)
REQUEST = {"test_kind": "functional", "compiler_diagnostic": True}


@pytest.mark.parametrize("flag", [True, "true"])
def test_diagnostic_is_explicit_separate_route(flag):
    request = {**REQUEST, "compiler_diagnostic": flag}
    assert (
        functional.validate_route("workflow_dispatch", functional.BRANCH, request)
        == "compiler_diagnostic"
    )
    with pytest.raises(ValueError, match="must not enter"):
        functional.select_config(request)


@pytest.mark.parametrize("kind", ["unit", "functional"])
@pytest.mark.parametrize("flag", [False, "false"])
def test_default_routes_unchanged(kind, flag):
    assert (
        functional.validate_route(
            "workflow_dispatch", functional.BRANCH, {"test_kind": kind, "compiler_diagnostic": flag}
        )
        == kind
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("test_kind", "unit"),
        ("compiler_diagnostic", "yes"),
        ("compiler_diagnostic", 1),
        ("functional_mode", "regression"),
        ("functional_cases", "bert/bert_mcore_tp1_pp2"),
        ("functional_scope", "L0"),
        ("functional_repeat", 1),
        ("functional_timeout", 10),
        ("scope", "full"),
        ("suite_family", "release-dev-lts"),
        ("unit_bucket", "a"),
        ("repeat", 2),
        ("timeout", 10),
        ("build", False),
        ("unit_test", False),
        ("pr_number", "7"),
        ("pr_sha", "a" * 40),
        ("pr_base_sha", "b" * 40),
    ],
)
def test_reject_mixed_inputs(key, value):
    with pytest.raises(ValueError):
        functional.validate_route("workflow_dispatch", functional.BRANCH, {**REQUEST, key: value})


@pytest.mark.parametrize(
    "event,ref",
    [
        ("push", functional.BRANCH),
        ("pull_request_target", functional.BRANCH),
        ("merge_group", functional.BRANCH),
        ("schedule", functional.BRANCH),
        ("workflow_dispatch", "refs/heads/main"),
        ("workflow_dispatch", "refs/tags/v1"),
    ],
)
def test_reject_automatic_or_wrong_branch(event, ref):
    with pytest.raises(ValueError):
        functional.validate_route(event, ref, REQUEST)


@pytest.mark.parametrize("permission", ["write", "maintain", "admin", "read"])
def test_maintainer_authorization_and_no_pr_status(permission):
    calls = []

    def api(path, payload=None):
        calls.append(path)
        assert path.startswith("collaborators/") and payload is None
        return {"permission": permission}

    if permission == "read":
        with pytest.raises(ValueError, match="write/maintain/admin"):
            auth.authorize({}, REQUEST, context(), api)
        return
    authorized = auth.authorize({}, REQUEST, context(), api)
    assert authorized["test_kind"] == "compiler_diagnostic"
    assert authorized["allowed"] and authorized["pr_number"] == 0
    before = list(calls)
    assert auth.publish_status(authorized, "success", "diagnostic only", api) == "rejected"
    assert calls == before


@pytest.mark.parametrize(
    "key,value", [("run_attempt", "2"), ("sha", "a" * 40), ("workflow_ref", "wrong")]
)
def test_reject_stale_or_reused_authorization(key, value):
    ctx = {**context(), key: value}
    with pytest.raises(ValueError):
        auth.authorize({}, REQUEST, ctx, lambda _: {"permission": "write"})


def authorization_and_env():
    authorized = auth.authorize({}, REQUEST, context(), lambda _: {"permission": "write"})
    env = {
        "PPU_TARGET_SHA": context()["sha"],
        "PPU_CONTROL_SHA": context()["control_sha"],
        "PPU_DIAGNOSTIC_RUN_ID": "123",
        "PPU_DIAGNOSTIC_RUN_ATTEMPT": "1",
    }
    return authorized, env


def test_pod_authorization_matches_current_run():
    authorized, env = authorization_and_env()
    diagnostic.verify_authorization(authorized, env)
    diagnostic.verify_authorization(
        authorized,
        {
            "PPU_TARGET_SHA": env["PPU_TARGET_SHA"],
            "PPU_CONTROL_SHA": env["PPU_CONTROL_SHA"],
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
        },
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("allowed", False),
        ("test_kind", "functional"),
        ("event_name", "push"),
        ("pr_number", 7),
        ("manual_pr", True),
        ("run_id", "124"),
        ("run_attempt", 2),
        ("target_sha", "a" * 40),
        ("control_sha", "a" * 40),
    ],
)
def test_pod_rejects_different_identity(key, value):
    authorized, env = authorization_and_env()
    with pytest.raises(ValueError):
        diagnostic.verify_authorization({**authorized, key: value}, env)


def test_workflow_dispatch_defaults_and_job_exclusion():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    inputs = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]
    assert inputs["compiler_diagnostic"] == {
        "description": "独立编译诊断（单 Pod 八卡，最多 10 分钟；不运行 Build、UT 或正式回归）",
        "type": "boolean",
        "default": False,
    }
    jobs = workflow["jobs"]
    assert "test_kind != 'compiler_diagnostic'" in jobs["execute"]["if"]
    assert "test_kind == 'compiler_diagnostic'" in jobs["compiler-diagnostic"]["if"]
    for guard in (
        "workflow_dispatch",
        functional.BRANCH,
        "github.run_attempt == 1",
        "outputs.allowed == 'true'",
    ):
        assert guard in jobs["compiler-diagnostic"]["if"]
    assert jobs["compiler-diagnostic"]["needs"] == "authorize"
    assert "strategy" not in jobs["compiler-diagnostic"]
    assert "pr_number != '0'" in jobs["pr-result"]["if"]


def test_fixed_single_pod_budget_and_no_build_or_test_entrypoints():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-compiler-diagnostic.yml").read_text())
    assert list(workflow.get("on", workflow.get(True))) == ["workflow_call"]
    assert list(workflow["jobs"]) == ["diagnose"]
    job = workflow["jobs"]["diagnose"]
    assert job["environment"] == "ppu-ci"
    assert job["runs-on"] == "k8s-runner-group-cpu-flytiger"
    assert "strategy" not in job
    action = [step for step in job["steps"] if step.get("uses", "").startswith("flytiger-eco/")]
    assert len(action) == 1
    config = action[0]["with"]
    assert config["image"] == functional.IMAGE
    assert config["nnodes"] == 0 and config["nproc_per_node"] == 8
    assert config["node_selector"] == "board-type=ZW-M890P"
    assert config["timeout_minutes"] == 10 and config["cleanup_policy"] == "always"
    assert "--kill-after=5s 540s" in config["command"]
    assert diagnostic.BUDGET_SECONDS < 540
    assert diagnostic.VARIANT_SECONDS * len(diagnostic.VARIANTS) < diagnostic.BUDGET_SECONDS
    text = json.dumps(workflow)
    for forbidden in (
        "ppu_ci_build.sh",
        "ppu_ci_run.sh",
        "ppu_functional_run.sh",
        "pip install",
        "run_ci_test.sh",
    ):
        assert forbidden not in text


def test_cold_cache_variant_environments(tmp_path):
    base = {
        "RANK": "7",
        "MASTER_PORT": "9000",
        "TORCHINDUCTOR_COMPILE_THREADS": "99",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
    }
    default = diagnostic.variant_environment(
        base, tmp_path / "default", tmp_path / "boot", "/tmp/short", "default"
    )
    single = diagnostic.variant_environment(
        base, tmp_path / "single", tmp_path / "boot", "/tmp/short", "single_thread"
    )
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in default
    assert single["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    for key in ("RANK", "MASTER_PORT", "TORCHINDUCTOR_FX_GRAPH_CACHE"):
        assert key not in default and key not in single
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "PPU_DIAGNOSTIC_STACK_DIR"):
        assert default[key] != single[key]
    assert default["TMPDIR"] == single["TMPDIR"] == "/tmp/short"
    assert default["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_original_bda_functions_not_rewritten(monkeypatch):
    source = ROOT / diagnostic.BDA
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    fake = types.ModuleType("torch")
    fake.Tensor = object
    decorated = []

    def decorate(function):
        decorated.append(function.__name__)
        return function

    fake.compile = decorate
    monkeypatch.setitem(sys.modules, "torch", fake)
    function = diagnostic.load_bda(ROOT, fake)
    assert decorated == ["bias_dropout_add_fused_train", "bias_dropout_add_fused_inference"]
    assert function.__code__.co_filename == str(source)
    assert function.__code__.co_argcount == 3
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancel_cleanup_process_group(tmp_path, monkeypatch, cancel):
    runner = diagnostic.DiagnosticRunner()
    monkeypatch.setattr(diagnostic, "capture_processes", lambda *_: None)
    timer = threading.Timer(0.3, runner.cancel, args=(signal.SIGTERM, None)) if cancel else None
    if timer:
        timer.start()
    try:
        result = runner.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            dict(os.environ),
            tmp_path,
            time.monotonic() + (5 if cancel else 0.3),
        )
    finally:
        if timer:
            timer.join()
    assert result["cancelled"] is cancel
    assert result["timed_out"] is not cancel
    assert result["exit_code"] != 0 and result["duration_seconds"] < 5


def test_normal_process_exit_is_preserved(tmp_path):
    result = diagnostic.DiagnosticRunner().run(
        [sys.executable, "-c", "raise SystemExit(17)"],
        dict(os.environ),
        tmp_path,
        time.monotonic() + 5,
    )
    assert result["exit_code"] == 17 and not result["timed_out"]


def test_sitecustomize_emits_all_thread_python_stack(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_bytes(
        (SCRIPTS / "ppu_diagnostic_sitecustomize.py").read_bytes()
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(bootstrap),
        "PPU_DIAGNOSTIC_STACK_DIR": str(tmp_path / "stacks"),
    }
    run = subprocess.run(
        [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGUSR1)"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert run.returncode == 0, run.stderr
    stacks = list((tmp_path / "stacks").glob("python-*.log"))
    assert len(stacks) == 1
    assert "Current thread" in stacks[0].read_text()


@pytest.mark.parametrize("mode", ["complete", "missing_rank", "timeout", "cancelled"])
def test_orchestration_preserves_evidence_and_never_retries(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(
        diagnostic.subprocess,
        "check_output",
        lambda command, **_: (ROOT / command[-1].removeprefix("HEAD:")).read_bytes(),
    )
    commands = []

    def run(self, command, env, output, deadline):
        commands.append(command)
        assert command.count("--nproc-per-node=8") == 1 and "--max-restarts=0" in command
        assert 0 < deadline - time.monotonic() <= diagnostic.VARIANT_SECONDS
        for rank in range(7 if mode == "missing_rank" else 8):
            diagnostic.write_json(
                output / f"rank-{rank}.json", {"rank": rank, "phase": "warm_forward_complete"}
            )
        self.cancelled = mode == "cancelled"
        return {
            "exit_code": 0,
            "timed_out": mode == "timeout",
            "cancelled": self.cancelled,
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr(diagnostic.DiagnosticRunner, "run", run)
    output = tmp_path / "report"
    result = diagnostic.run_diagnostic(ROOT, output, {})
    report = json.loads((output / "diagnostic.json").read_text())
    assert len(commands) == (1 if mode == "cancelled" else 2)
    assert (result == 0) is (mode == "complete")
    assert report["completed"] is (mode == "complete")
    assert report["numerical_validation"] is False
    assert set(report["source_sha256"]) == {
        diagnostic.BDA,
        "megatron/core/jit.py",
        ".github/scripts/ppu_compiler_diagnostic.py",
        ".github/scripts/ppu_diagnostic_sitecustomize.py",
    }


def test_changed_source_aborts_before_gpu_process(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.subprocess, "check_output", lambda *_, **__: b"changed")
    monkeypatch.setattr(
        diagnostic.subprocess, "Popen", lambda *_, **__: pytest.fail("GPU process must not start")
    )
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="differs from committed"):
        diagnostic.run_diagnostic(ROOT, output, {})
    report = json.loads((output / "diagnostic.json").read_text())
    assert not report["completed"] and report["variants"] == []


def test_cleanup_kills_worker_after_launcher_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "capture_processes", lambda *_: None)
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"Path({str(tmp_path / 'worker.pid')!r}).write_text(str(child.pid)); time.sleep(60)"
    )
    result = diagnostic.DiagnosticRunner().run(
        [sys.executable, "-c", parent], dict(os.environ), tmp_path, time.monotonic() + 0.6
    )
    assert result["timed_out"]
    worker = int((tmp_path / "worker.pid").read_text())
    for _ in range(20):
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(worker)], capture_output=True, text=True, timeout=2
        )
        if not state.stdout.strip() or state.stdout.lstrip().startswith("Z"):
            break
        time.sleep(0.1)
    else:
        os.kill(worker, signal.SIGKILL)
        pytest.fail("编译 worker 未随诊断进程组回收")


@pytest.mark.parametrize(
    "own,ready,caught",
    [(True, True, True), (True, False, True), (True, True, False), (False, True, True)],
)
def test_stack_sampling_signals_only_registered_diagnostic_processes(
    tmp_path, monkeypatch, own, ready, caught
):
    process = tmp_path / "proc/123"
    process.mkdir(parents=True)
    mask = (1 << (signal.SIGUSR1 - 1)) if caught else 0
    (process / "status").write_text(f"SigCgt:\t{mask:016x}\n")
    stack = tmp_path / "stacks/python-123.log"
    stack.parent.mkdir()
    stack.touch()
    if ready:
        stack.with_suffix(".ready").touch()
    monkeypatch.setattr(
        diagnostic, "Path", lambda value: tmp_path / "proc" if value == "/proc" else Path(value)
    )
    monkeypatch.setattr(diagnostic.os, "getsid", lambda _: 10 if own else 20)
    signals = []
    monkeypatch.setattr(diagnostic.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    diagnostic.capture_processes(10, tmp_path)
    assert signals == ([(123, signal.SIGUSR1)] if own and ready and caught else [])


def test_kernel_stack_unavailability_is_recorded(tmp_path):
    assert diagnostic.proc_text(tmp_path / "not-present").startswith("unavailable:")
