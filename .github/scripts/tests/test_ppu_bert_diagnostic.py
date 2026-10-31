"""完整 BERT 诊断的历史身份、单次预算、冷启动采样与原入口隔离测试。"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest
import yaml
from test_ppu_compiler_diagnostic import (
    REQUEST,
    authorization_and_env,
    context,
    diagnostic,
    functional,
)
from test_ppu_functional_environment import functional_bundle
from test_ppu_functional_results import evidence
from test_ppu_functional_runtime import ROOT, SCRIPTS, runtime
from test_ppu_quick import auth
from test_ppu_upstream import sealed_bundle

bert = functional.module_from_path("ppu_bert_diagnostic", SCRIPTS / "ppu_bert_diagnostic.py")
REQUEST_BERT = {**REQUEST, "compiler_diagnostic_context": "bert"}


def authorize_bert(monkeypatch):
    authorized = auth.authorize({}, REQUEST_BERT, context(), lambda _: {"permission": "write"})
    _, env = authorization_and_env()
    env["PPU_DIAGNOSTIC_CONTEXT"] = "bert"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return authorized, env


@pytest.mark.parametrize("selected", ["bda", "bert"])
def test_context_is_bound_to_authorization(monkeypatch, selected):
    request = {**REQUEST, "compiler_diagnostic_context": selected}
    authorized = auth.authorize({}, request, context(), lambda _: {"permission": "write"})
    _, env = authorization_and_env()
    env["PPU_DIAGNOSTIC_CONTEXT"] = selected
    diagnostic.verify_authorization(authorized, env)
    other = "bert" if selected == "bda" else "bda"
    with pytest.raises(ValueError):
        diagnostic.verify_authorization(authorized, {**env, "PPU_DIAGNOSTIC_CONTEXT": other})
    with pytest.raises(ValueError):
        diagnostic.verify_authorization({**authorized, "compiler_diagnostic_context": None}, env)


@pytest.mark.parametrize("kind", ["unit", "functional"])
@pytest.mark.parametrize("value", ["bert", "unknown", "", None])
def test_context_cannot_enter_normal_execution(kind, value):
    request = {"test_kind": kind, "compiler_diagnostic_context": value}
    with pytest.raises(ValueError):
        functional.validate_route("workflow_dispatch", functional.BRANCH, request)
    with pytest.raises(ValueError):
        functional.select_config(request)


@pytest.mark.parametrize(
    "change", [{"run_attempt": "2"}, {"sha": "0" * 40}, {"ref": "refs/heads/main"}]
)
def test_bert_rejects_rerun_or_wrong_control(change):
    with pytest.raises(ValueError):
        auth.authorize({}, REQUEST_BERT, {**context(), **change}, lambda _: {"permission": "write"})


@pytest.fixture
def historical_bundle(functional_bundle, evidence, monkeypatch):
    _, bundle = functional_bundle
    manifest, _, _ = evidence
    manifest["identity"] = {
        "target_sha": bert.HISTORICAL_SHA,
        "control_sha": bert.HISTORICAL_SHA,
        "run_id": bert.HISTORICAL_RUN,
        "run_attempt": 1,
    }
    raw = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(raw)
    for name in ("commit.txt", "control.txt", ".complete"):
        (bundle / name).write_text(bert.HISTORICAL_SHA + "\n")
    (bundle / "image.txt").write_text(manifest["config"]["image"] + "\n")
    bert.bundle_control.seal_bundle(bundle)
    monkeypatch.setattr(bert, "MANIFEST_DIGEST", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        bert,
        "BUNDLE_DIGEST",
        hashlib.sha256((bundle / "bundle-integrity.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(bert, "CPU_BUNDLE", bundle)
    monkeypatch.setattr(bert, "POD_BUNDLE", bundle)
    return bundle, manifest


def test_cpu_preflight_is_read_only_and_uses_fixed_trust(historical_bundle, tmp_path):
    bundle, manifest = historical_bundle
    before = {str(path): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    bert.preflight(tmp_path / "cpu-envelope")
    after = {str(path): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    assert after == before
    report = json.loads((tmp_path / "cpu-envelope/historical-preflight.json").read_text())
    assert report["ok"] and report["historical_identity"] == manifest["identity"]
    assert report["numerical_validation"] is False


@pytest.mark.parametrize(
    "name",
    [
        "environment.tar.gz",
        "source.tar.gz",
        "manifest.json",
        "control/ppu_functional_runtime.py",
        "commit.txt",
        "control.txt",
        ".complete",
        "bundle-integrity.json",
        "image.txt",
    ],
)
@pytest.mark.parametrize("reseal", [False, True])
def test_corrupt_or_resealed_history_never_reaches_pod(
    historical_bundle, tmp_path, monkeypatch, name, reseal
):
    bundle, _ = historical_bundle
    (bundle / name).write_text("corrupt")
    if reseal and name not in ("manifest.json", "bundle-integrity.json"):
        bert.bundle_control.seal_bundle(bundle)
    monkeypatch.setattr(bert.subprocess, "Popen", lambda *_, **__: pytest.fail("预检禁止启动进程"))
    with pytest.raises((ValueError, OSError, KeyError)):
        bert.preflight(tmp_path / "cpu-envelope")
    report = json.loads((tmp_path / "cpu-envelope/historical-preflight.json").read_text())
    assert not report["ok"] and "error" in report


@pytest.mark.parametrize(
    "key,value",
    [("run_id", "other"), ("run_attempt", 2), ("target_sha", "a" * 40), ("control_sha", "b" * 40)],
)
def test_identity_has_independent_fixed_constraint(historical_bundle, monkeypatch, key, value):
    _, manifest = historical_bundle
    manifest["identity"][key] = value
    raw = json.dumps(manifest).encode()
    monkeypatch.setattr(bert, "MANIFEST_DIGEST", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="identity"):
        bert.verify_manifest(raw)


@pytest.mark.parametrize(
    "key,value",
    [
        ("name", "other"),
        ("test_type", "ckpt-resume"),
        ("effective_train_iters", 4),
        ("train_iters", 4),
        ("cards", 1),
        ("launcher", "ft_launcher"),
        ("launcher_internal_max_restarts", 3),
        ("observation", "lightweight"),
    ],
)
def test_original_case_scope_is_not_configurable(historical_bundle, monkeypatch, key, value):
    _, manifest = historical_bundle
    manifest["functional"][bert.CASE_ID][key] = value
    raw = json.dumps(manifest).encode()
    monkeypatch.setattr(bert, "MANIFEST_DIGEST", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="scope"):
        bert.verify_manifest(raw)


def test_missing_historical_nas_stops_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(bert, "CPU_BUNDLE", tmp_path / "absent")
    with pytest.raises(OSError):
        bert.preflight(tmp_path / "envelope")
    assert not (tmp_path / "absent").exists()
    assert not json.loads((tmp_path / "envelope/historical-preflight.json").read_text())["ok"]


def test_pod_verifies_copy_before_unpacking(historical_bundle, tmp_path, monkeypatch):
    bundle, _ = historical_bundle
    work, output = tmp_path / "work", tmp_path / "output"
    work.mkdir()
    output.mkdir()
    monkeypatch.setattr(bert, "VENV", tmp_path / "venv")
    calls = []

    def run(command, **kwargs):
        assert bert.verify_historical_bundle(work / "bundle")
        calls.append(command)

    monkeypatch.setattr(bert.subprocess, "run", run)
    monkeypatch.setattr(bert, "install_hook", lambda *args: {"sha256": "hook"})

    def execute(path, command, env):
        assert path == "/bin/bash" and command[2] == bert.SETUP_SCRIPT
        assert command[4:6] == [str(work / "source"), str(work / "bundle")]
        assert env["PPU_DIAGNOSTIC_STACK_DIR"] == str(output / "stacks")
        raise RuntimeError("exec boundary")

    monkeypatch.setattr(bert.os, "execve", execute)
    with pytest.raises(RuntimeError, match="exec boundary"):
        bert.execute(work, output)
    assert len(calls) == 2 and all(command[0] == "tar" for command in calls)
    assert (output / "bundle-integrity.json").read_bytes() == (
        bundle / "bundle-integrity.json"
    ).read_bytes()


def test_corruption_during_copy_prevents_extraction(historical_bundle, tmp_path, monkeypatch):
    work, output = tmp_path / "work", tmp_path / "output"
    work.mkdir()
    output.mkdir()
    original = bert.shutil.copytree

    def corrupt(source, target, *args, **kwargs):
        original(source, target, *args, **kwargs)
        if Path(target) == work / "bundle":
            (Path(target) / "environment.tar.gz").write_text("partial")

    monkeypatch.setattr(bert.shutil, "copytree", corrupt)
    monkeypatch.setattr(bert.subprocess, "run", lambda *_, **__: pytest.fail("不得展开损坏副本"))
    with pytest.raises(ValueError):
        bert.execute(work, output)


@pytest.fixture
def hooked_venv(tmp_path):
    destination = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(destination)
    python = destination / "bin/python"
    hook = SCRIPTS / "ppu_diagnostic_sitecustomize.py"
    report = bert.install_hook(python, destination, hook)
    assert report["sha256"] == hashlib.sha256(hook.read_bytes()).hexdigest()
    return destination, python


@pytest.mark.parametrize("mode", ["isolated", "fork", "new_session"])
def test_real_venv_hook_covers_isolated_and_forked_children(hooked_venv, tmp_path, mode):
    _, python = hooked_venv
    stacks = tmp_path / "stacks"
    sample = "import os,signal,threading; assert threading.active_count()==1; os.kill(os.getpid(),signal.SIGUSR1)"
    if mode == "fork":
        script = f"import os; pid=os.fork()\nif pid==0:\n exec({sample!r}); os._exit(0)\nelse:\n assert os.waitpid(pid,0)[1]==0"
    elif mode == "new_session":
        script = f"import subprocess,sys; subprocess.run([sys.executable,'-I','-c',{sample!r}],check=True,start_new_session=True)"
    else:
        script = sample
    env = {**os.environ, "PPU_DIAGNOSTIC_STACK_DIR": str(stacks), "PYTHONPATH": "/invalid/ignored"}
    result = subprocess.run(
        [str(python), "-I", "-c", script], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    logs = list(stacks.glob("*.log"))
    assert len(logs) == (1 if mode == "isolated" else 2)
    assert any("Current thread" in path.read_text() for path in logs)
    assert all(path.with_suffix(".ready").is_file() for path in logs)


def test_hook_is_inert_without_diagnostic_environment(hooked_venv, tmp_path):
    _, python = hooked_venv
    env = dict(os.environ)
    env.pop("PPU_DIAGNOSTIC_STACK_DIR", None)
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import signal; assert signal.getsignal(signal.SIGUSR1)==signal.SIG_DFL",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "stacks").exists()


@pytest.mark.parametrize("code", [0, 23, 124])
def test_train_uses_original_model_and_shell_exactly_once(
    historical_bundle, tmp_path, monkeypatch, code
):
    bundle, manifest = historical_bundle
    calls = []
    monkeypatch.setattr(bert, "load_module", lambda name, path: runtime)
    monkeypatch.setattr(runtime.control, "asset_report", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(runtime.control, "write_asset_report", lambda *args: None)
    monkeypatch.setenv("PPU_DIAGNOSTIC_STACK_DIR", str(tmp_path / "stacks"))
    case = manifest["functional"][bert.CASE_ID]
    before = (ROOT / case["config_path"]).read_bytes()

    def call(command, *, cwd, env, **kwargs):
        calls.append(command)
        assert cwd == ROOT and command == [
            "bash",
            str(ROOT / "tests/functional_tests/shell_test_utils/run_ci_test.sh"),
        ]
        config = yaml.safe_load(Path(env["TRAINING_PARAMS_PATH"]).read_text())
        expected = yaml.safe_load(before)
        expected.setdefault("ENV_VARS", {})["N_REPEAT"] = 1
        assert config == expected and config["MODEL_ARGS"]["--train-iters"] == 50
        assert env["N_REPEAT"] == "1" and env["RECORD_CHECKPOINTS"] == "false"
        assert env["GPUS_PER_NODE"] == "8" and env["NUM_NODES"] == "1"
        assert env["PPU_DIAGNOSTIC_STACK_DIR"] == str(tmp_path / "stacks")
        assert Path(env["TMPDIR"]).is_dir()
        return code

    monkeypatch.setattr(bert.subprocess, "call", call)
    output = tmp_path / "training"
    output.mkdir()
    assert bert.train(ROOT, bundle, output) == code
    assert len(calls) == 1 and (ROOT / case["config_path"]).read_bytes() == before
    rows = [json.loads(line) for line in (output / "stages.jsonl").read_text().splitlines()]
    assert rows[-1]["exit_code"] == code and all(
        row["numerical_validation"] is False for row in rows
    )


@pytest.mark.parametrize("mode", ["success", "failure", "timeout", "cancel", "exception"])
def test_single_orchestration_never_retries(tmp_path, monkeypatch, mode):
    authorized, _ = authorize_bert(monkeypatch)
    monkeypatch.setattr(
        bert.subprocess,
        "check_output",
        lambda command, **kwargs: (ROOT / command[-1].removeprefix("HEAD:")).read_bytes(),
    )
    calls = []

    def run(self, command, env, output, deadline):
        calls.append(command)
        assert command[1:4] == ["-I", str(bert.SCRIPT), "execute"]
        assert 0 < deadline - time.monotonic() <= 490
        if mode == "exception":
            raise RuntimeError("controller failed")
        self.cancelled = mode == "cancel"
        return {
            "exit_code": 0 if mode == "success" else 1,
            "timed_out": mode == "timeout",
            "cancelled": self.cancelled,
        }

    monkeypatch.setattr(bert.diagnostic.DiagnosticRunner, "run", run)
    output = tmp_path / "diagnostic"
    if mode == "exception":
        with pytest.raises(RuntimeError):
            bert.run_diagnostic(ROOT, output, authorized)
    else:
        assert (bert.run_diagnostic(ROOT, output, authorized) == 0) is (mode == "success")
    report = json.loads((output / "diagnostic.json").read_text())
    assert len(calls) == 1 and report["numerical_validation"] is False
    assert report["authorization"]["control_sha"] != report["historical_sha"]
    assert report["completed"] is (mode == "success")


@pytest.mark.parametrize("cancelled", [False, True])
def test_no_launch_after_budget_or_cancellation(tmp_path, monkeypatch, cancelled):
    runner = diagnostic.DiagnosticRunner()
    runner.cancelled = cancelled
    monkeypatch.setattr(
        diagnostic.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("不得启动")
    )
    report = runner.run(["unused"], {}, tmp_path, time.monotonic() - 1)
    assert report["exit_code"] == (143 if cancelled else 124)


@pytest.mark.parametrize("mode", ["timeout", "cancel", "success", "failure", "exception"])
def test_real_detached_workers_are_reaped_and_cleanup_recorded(tmp_path, monkeypatch, mode):
    """真实跨 session 信号测试；macOS 只替换不可用的 /proc 读取。"""
    runner = diagnostic.DiagnosticRunner()
    ready = tmp_path / "worker.pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True); "
        f"time.sleep({0.8 if mode in ('success', 'failure') else 60}); "
        f"raise SystemExit({23 if mode == 'failure' else 0})"
    )
    original_family = diagnostic.process_family
    raised = False

    def family(pid, known):
        nonlocal raised
        if Path("/proc/self/stat").is_file():
            selected = original_family(pid, known)
        else:
            candidates = {pid, *known}
            if ready.exists():
                candidates.add(int(ready.read_text()))
            selected = {}
            for candidate in candidates:
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(candidate)],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
                if not state or state.startswith("Z"):
                    continue
                try:
                    group = os.getpgid(candidate)
                except ProcessLookupError:
                    continue
                selected[candidate] = {"ppid": pid, "pgid": group, "sid": group, "start": 1}
            known.update(selected)
        if ready.exists() and int(ready.read_text()) in selected:
            if mode == "cancel":
                runner.cancel(signal.SIGTERM, None)
            if mode == "exception" and not raised:
                raised = True
                raise RuntimeError("sampler failed")
        return selected

    monkeypatch.setattr(diagnostic, "process_family", family)
    monkeypatch.setattr(diagnostic, "capture_processes", lambda *args: None)
    try:
        arguments = (
            [sys.executable, "-c", parent],
            dict(os.environ),
            tmp_path,
            time.monotonic() + (1 if mode == "timeout" else 10),
        )
        if mode == "exception":
            with pytest.raises(RuntimeError, match="sampler failed"):
                runner.run(*arguments)
        else:
            result = runner.run(*arguments)
            assert result["timed_out"] is (mode == "timeout")
            assert result["cancelled"] is (mode == "cancel")
            if mode in ("success", "failure"):
                assert result["exit_code"] == (0 if mode == "success" else 23)
        cleanup = json.loads((tmp_path / "cleanup.json").read_text())
        assert str(int(ready.read_text())) in cleanup["tracked"]
        assert cleanup["remaining"] == {}
    finally:
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_dispatch_wires_bound_context_before_only_ppu_action():
    entry = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    inputs = entry.get("on", entry.get(True))["workflow_dispatch"]["inputs"]
    assert inputs["compiler_diagnostic_context"]["default"] == "bda"
    assert inputs["compiler_diagnostic_context"]["options"] == ["bda", "bert"]
    assert (
        "inputs.compiler_diagnostic_context"
        in entry["jobs"]["compiler-diagnostic"]["with"]["diagnostic_context"]
    )
    workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-compiler-diagnostic.yml").read_text())
    steps = workflow["jobs"]["diagnose"]["steps"]
    preflight = next(step for step in steps if step.get("name") == "核验诊断授权与源码身份")
    action = next(step for step in steps if step.get("id") == "ppu")
    assert steps.index(preflight) < steps.index(action) and "if" not in action
    assert preflight["env"]["PPU_DIAGNOSTIC_CONTEXT"] == "${{ inputs.diagnostic_context }}"
    assert "PPU_DIAGNOSTIC_CONTEXT=${{ inputs.diagnostic_context }}" in action["with"]["extra_env"]
    assert "--output .ppu-diagnostic-action" in preflight["run"]
    assert "--kill-after=5s 540s" in action["with"]["command"]
    assert action["with"]["timeout_minutes"] == 10


def test_standalone_cold_start_has_all_dependencies():
    result = subprocess.run(
        [sys.executable, "-I", str(bert.SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "execute" in result.stdout
    assert "pip install" not in bert.SETUP_SCRIPT and "ppu_ci_build" not in bert.SETUP_SCRIPT
    syntax = subprocess.run(
        ["bash", "-n"], input=bert.SETUP_SCRIPT, capture_output=True, text=True, timeout=5
    )
    assert syntax.returncode == 0, syntax.stderr
