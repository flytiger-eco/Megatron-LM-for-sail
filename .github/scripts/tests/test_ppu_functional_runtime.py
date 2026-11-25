"""CPU coverage of orchestration, isolated subprocesses, budgets and workflow routing."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml
from test_ppu_functional import ROOT, SCRIPTS, functional
from test_ppu_functional_results import envelope, evidence, pytest_inputs, results

runtime = functional.module_from_path(
    "ppu_functional_runtime", SCRIPTS / "ppu_functional_runtime.py"
)


class FakeRunner:
    cancelled = False
    timed_out = False

    def __init__(self, manifest, case, codes=(0,), message="DistNetworkError"):
        self.manifest = manifest
        self.case = case
        self.codes = list(codes)
        self.message = message
        self.calls = []

    def run(self, command, *, cwd, env, log, deadline):
        self.calls.append((command, env, deadline))
        directory = Path(env["OUTPUT_PATH"])
        code = self.codes.pop(0) if self.codes else 0
        assert command == [
            "bash",
            str(ROOT / "tests/functional_tests/shell_test_utils/run_ci_test.sh"),
        ]
        assert cwd == ROOT
        assert env["DATA_PATH"] == self.manifest["config"]["data_path"]
        assert env["N_REPEAT"] == "1" and env["RECORD_CHECKPOINTS"] == "false"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["GPUS_PER_NODE"] == "8" and env["NUM_NODES"] == "1"
        temporary = Path(env["TMPDIR"])
        assert temporary.is_dir() and temporary.stat().st_mode & 0o777 == 0o700
        assert len(os.fsencode(str(temporary))) < 60
        assert env["TMP"] == env["TEMP"] == env["TMPDIR"]
        assert env["GOLDEN_VALUES_PATH"] == str(ROOT / self.case["golden_path"])
        assert env["TRAINING_SCRIPT_PATH"] == str(ROOT / self.case["training_script"])
        config = yaml.safe_load((directory / "model_config.yaml").read_text())
        config["MODEL_ARGS"]["--exit-interval"] = self.case["effective_train_iters"]
        (directory / "model_config.yaml").write_text(yaml.safe_dump(config))
        log.write_text(self.message if code else "Training completed")
        first = directory / Path(self.case["golden_path"]).name
        first.write_bytes((ROOT / self.case["golden_path"]).read_bytes())
        for phase in range(1, results.phase_count(self.case) + 1):
            tensorboard = directory / "1/tensorboard"
            if results.phase_count(self.case) == 2:
                tensorboard /= f"run_{phase}"
            tensorboard.mkdir(parents=True)
            (tensorboard / "events.out.tfevents.fixture").write_text("Mock event evidence")
            barrier = directory / "barriers" / f"repeat_1_run_{phase}_status"
            barrier.mkdir(parents=True)
            (barrier / "node_0").write_text(str(code))
            for rank in range(8):
                logs = directory / "logs" / f"run-{phase}" / "attempt_0" / str(rank)
                logs.mkdir(parents=True)
                (logs / "stdout.log").write_text("Training")
                (logs / "stderr.log").write_text("")
            kind = "regular" if phase == 1 else "resume"
            node = "test_regular_pipeline" if phase == 1 else "test_resume_checkpoint_pipeline"
            node = f"test_pretraining_{kind}_pipeline.py::{node}"
            if phase == 2:
                first.with_name(first.stem + "_2nd.json").write_bytes(first.read_bytes())
            runtime.write_json(
                directory / f"pytest-{kind}.json",
                {
                    **results.identity_fields(
                        self.manifest,
                        self.case,
                        int(env["PPU_FUNCTIONAL_ATTEMPT"]),
                        int(env["PPU_FUNCTIONAL_REPEAT"]),
                    ),
                    "phase": kind,
                    "inputs": pytest_inputs(self.case, directory, kind),
                    "train_iters": self.case["effective_train_iters"],
                    "approximate": self.case["approximate"],
                    "complete": True,
                    "exitstatus": 0,
                    "duration_seconds": 0.01,
                    "collection_errors": [],
                    "collected": [node],
                    "tests": {node: {stage: "passed" for stage in ("setup", "call", "teardown")}},
                },
            )
        return code


def prepare_case(evidence, tmp_path):
    manifest, case, _ = evidence
    assets = tmp_path / "assets"
    assets.mkdir()
    for asset in case["assets"]:
        path = assets / asset["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    manifest["config"]["data_path"] = str(assets)
    return manifest, case, tmp_path / "case-results"


def test_runtime_runs_every_repeat_and_checks_upstream_pytest_results(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    original = functional.sha256(ROOT / case["config_path"])
    runner = FakeRunner(manifest, case)
    record = runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)
    assert record["success"] and record["attempts"] == 1
    assert len(runner.calls) == case["repeat"] == 2
    envelope(manifest, case, directory)
    assert results.verify_case(manifest, case, directory)["success"]
    assert functional.sha256(ROOT / case["config_path"]) == original
    outputs = {env["OUTPUT_PATH"] for _, env, _ in runner.calls}
    assert len(outputs) == 2
    assert len({env["TMPDIR"] for _, env, _ in runner.calls}) == 2
    assert len({deadline for _, _, deadline in runner.calls}) == 1
    for _, env, _ in runner.calls:
        for key in ("DATA_CACHE_PATH", "CHECKPOINT_SAVE_PATH", "TENSORBOARD_PATH"):
            assert env[key].startswith(env["OUTPUT_PATH"] + "/")
        assert not Path(env["TMPDIR"]).exists()
        report = json.loads((Path(env["OUTPUT_PATH"]) / "execution.json").read_text())
        assert report["temporary_directory"] == env["TMPDIR"]
        assert report["shell_exit_code"] == 0 and report["extraction_exit_code"] is None
        assert report["compiler_environment"] == manifest["config"]["compiler_environment"]
        assert env["TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"] == "0"
        assert "TORCHINDUCTOR_COMPILE_THREADS" not in env


def test_retry_restarts_all_repeats_with_one_shared_deadline(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    runner = FakeRunner(manifest, case, codes=(0, 1, 0, 0))
    record = runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)
    assert record["success"] and record["attempts"] == 2
    pairs = [
        (env["PPU_FUNCTIONAL_ATTEMPT"], env["PPU_FUNCTIONAL_REPEAT"]) for _, env, _ in runner.calls
    ]
    assert pairs == [("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")]
    assert len({env["TMPDIR"] for _, env, _ in runner.calls}) == 4
    assert all(not Path(env["TMPDIR"]).exists() for _, env, _ in runner.calls)
    assert len({deadline for _, _, deadline in runner.calls}) == 1
    envelope(manifest, case, directory)
    assert results.verify_case(manifest, case, directory)["retried"]


@pytest.mark.parametrize(
    "code,message",
    [
        (1, "AssertionError: mismatch"),
        (1, "DistNetworkError\nloss = nan"),
        (1, "DistNetworkError\nOutOfMemoryError"),
        (1, "Segmentation fault"),
        (1, "MissingTensorboardLogsError"),
        (1, "ModuleNotFoundError: No module named 'google.protobuf'\nConnectionResetError"),
        (1, "OSError: AF_UNIX path too long\nConnectionResetError"),
        (143, "DistNetworkError"),
        (124, "DistNetworkError"),
    ],
)
def test_no_retry_on_numeric_integrity_crash_budget_or_cancellation(
    evidence, tmp_path, code, message
):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    runner = FakeRunner(manifest, case, codes=(code,), message=message)
    record = runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)
    assert not record["success"] and record["attempts"] == 1
    assert len(runner.calls) == 1
    assert not Path(runner.calls[0][1]["TMPDIR"]).exists()


def test_maximum_three_complete_attempts(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    runner = FakeRunner(manifest, case, codes=(1, 1, 1, 0))
    record = runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)
    assert not record["success"] and record["attempts"] == 3
    assert len(runner.calls) == 3


def test_assets_mode_never_launches_training(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    manifest["config"]["functional_mode"] = "assets"
    with pytest.raises(ValueError, match="explicit functional regression"):
        runtime.run_case(manifest, ROOT, case["id"], directory)
    assert not directory.exists()


def test_process_runner_preserves_actual_subprocess_exit_and_log(tmp_path):
    runner = runtime.ProcessRunner()
    log = tmp_path / "driver.log"
    code = runner.run(
        [sys.executable, "-c", "print('real process output'); raise SystemExit(23)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log=log,
        deadline=time.monotonic() + 10,
    )
    assert code == 23 and "real process output" in log.read_text()


def test_process_runner_timeout_stops_process_group(tmp_path):
    runner = runtime.ProcessRunner()
    code = runner.run(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log=tmp_path / "driver.log",
        deadline=time.monotonic() + 0.2,
    )
    assert code == 124 and runner.timed_out and not runner.cancelled


def test_process_runner_cancellation_stops_process_group(tmp_path):
    runner = runtime.ProcessRunner()
    timer = threading.Timer(0.2, runner.cancel, args=(signal.SIGTERM, None))
    timer.start()
    try:
        code = runner.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            log=tmp_path / "driver.log",
            deadline=time.monotonic() + 10,
        )
    finally:
        timer.join()
    assert code == 143 and runner.cancelled and not runner.timed_out


@pytest.mark.parametrize("mode", ["timeout", "cancel", "success", "failure", "exception"])
def test_process_runner_reaps_real_detached_worker(tmp_path, monkeypatch, mode):
    """A real setsid child ignores TERM; macOS only replaces the
    unavailable /proc enumeration boundary.
    """
    runner = runtime.ProcessRunner()
    ready = tmp_path / "worker.pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True); "
        f"time.sleep({0.8 if mode in ('success', 'failure') else 60}); "
        f"raise SystemExit({23 if mode == 'failure' else 0})"
    )
    original_family = runtime.process_family
    observed = set()
    raised = False

    def family(pid, known):
        nonlocal raised
        if Path("/proc/self/stat").is_file():
            selected = original_family(pid, known)
        else:
            # Signal delivery and process exit are never mocked; both always use real system calls.
            candidates = {pid, *known}
            if ready.exists():
                candidates.add(int(ready.read_text()))
            selected = {}
            for candidate in candidates:
                try:
                    group = os.getpgid(candidate)
                except ProcessLookupError:
                    continue
                selected[candidate] = {"ppid": pid, "pgid": group, "sid": group, "start": 1}
            known.update(selected)
        observed.update(selected)
        if ready.exists() and int(ready.read_text()) in known:
            if mode == "cancel":
                runner.cancel(signal.SIGTERM, None)
            elif mode == "exception" and not raised:
                raised = True
                raise RuntimeError("controller failed after tracking worker")
        return selected

    monkeypatch.setattr(runtime, "process_family", family)
    kwargs = {
        "cwd": tmp_path,
        "env": os.environ.copy(),
        "log": tmp_path / "driver.log",
        "deadline": time.monotonic() + (1 if mode == "timeout" else 10),
    }
    worker = None
    try:
        if mode == "exception":
            with pytest.raises(RuntimeError, match="controller failed"):
                runner.run([sys.executable, "-c", parent], **kwargs)
        else:
            code = runner.run([sys.executable, "-c", parent], **kwargs)
            assert code == {"timeout": 124, "cancel": 143, "success": 0, "failure": 23}[mode]
        assert runner.process is None and runner.known_processes == {}
        assert runner.cancelled is (mode == "cancel")
        assert runner.timed_out is (mode == "timeout")
        worker = int(ready.read_text())
        assert worker in observed
        for _ in range(30):
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(worker)], capture_output=True, text=True, timeout=2
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.1)
        else:
            pytest.fail("Cross-session worker survived the launcher exit")
    finally:
        if ready.exists():
            worker = int(ready.read_text())
            try:
                os.killpg(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_environment_clears_inherited_shortcuts_and_uses_runtime_copy(
    evidence, tmp_path, monkeypatch
):
    manifest, case, _ = evidence
    monkeypatch.setenv("RECORD_CHECKPOINTS", "true")
    monkeypatch.setenv("ENABLE_LIGHTWEIGHT_MODE", "true")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    cache_keys = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "HF_ASSETS_CACHE",
        "HF_MODULES_CACHE",
        "TORCH_HOME",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    )
    for key in cache_keys:
        monkeypatch.setenv(key, str(tmp_path / "unapproved-shared-cache"))
    directory = tmp_path / "repeat"
    directory.mkdir()
    with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
        assert env["RECORD_CHECKPOINTS"] == "false" and env["ENABLE_LIGHTWEIGHT_MODE"] == "false"
        assert env["PYTEST_ADDOPTS"] == "" and "SLURM_JOB_END_TIME" not in env
        assert all(Path(env[key]).is_relative_to(directory / "cache") for key in cache_keys)
        expected = yaml.safe_load((ROOT / case["config_path"]).read_text())
        expected.pop("LAUNCHER", None)  # This fixture is a M890 torchrun case.
        expected.setdefault("ENV_VARS", {})["N_REPEAT"] = 1
        assert yaml.safe_load((directory / "model_config.yaml").read_text()) == expected


def test_lightweight_policy_reaches_the_upstream_runner(evidence, tmp_path, monkeypatch):
    manifest, case, _ = evidence
    manifest = {**manifest, "config": {**manifest["config"], "lightweight": True}}
    monkeypatch.setenv("ENABLE_LIGHTWEIGHT_MODE", "false")
    directory = tmp_path / "repeat"
    directory.mkdir()
    with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
        assert env["ENABLE_LIGHTWEIGHT_MODE"] == "true"


@pytest.mark.parametrize(
    "name,launcher",
    [
        ("gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files", "torchrun"),
        (
            "gpt/gpt3_mcore_te_tp2_pp1_resume_torch_dist_multi_dist_optimizer_instances",
            "ft_launcher",
        ),
    ],
)
def test_m890_runtime_passes_the_preserved_launcher_to_upstream(evidence, tmp_path, name, launcher):
    manifest, _, _ = evidence
    catalogue = functional.catalogue(
        ROOT, ROOT, functional.select_config({"functional_cases": name})
    )
    case = next(iter(catalogue["functional"].values()))
    case["tokenizer_dependencies"] = []
    directory = tmp_path / "launcher-repeat"
    directory.mkdir()
    source = ROOT / case["config_path"]
    original = source.read_bytes()
    with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
        config = yaml.safe_load(Path(env["TRAINING_PARAMS_PATH"]).read_text())
        assert config.get("LAUNCHER", "torchrun") == launcher == case["launcher"]
        assert Path(env["GOLDEN_VALUES_PATH"]).name == "golden_values_dev_m890.json"
        config["MODEL_ARGS"]["--exit-interval"] = case["effective_train_iters"]
        (directory / "model_config.yaml").write_text(yaml.safe_dump(config))
        results.verify_runtime_config(ROOT, case, directory)
    assert source.read_bytes() == original


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("inherited", [None, "0", "1"])
def test_compiler_environment_is_scoped_without_rewriting_model(
    evidence, tmp_path, monkeypatch, legacy, inherited
):
    manifest, case, _ = evidence
    key = "TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    if legacy:
        manifest["config"].pop("compiler_environment")
        monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    if inherited is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, inherited)
    directory = tmp_path / "compiler-repeat"
    directory.mkdir()
    original = (ROOT / case["config_path"]).read_bytes()
    expected = yaml.safe_load(original)
    expected.pop("LAUNCHER", None)  # This fixture is a M890 torchrun case.
    expected.setdefault("ENV_VARS", {})["N_REPEAT"] = 1
    with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
        assert env.get(key) == (inherited if legacy else "0")
        assert ("TORCHINDUCTOR_COMPILE_THREADS" in env) is legacy
        assert yaml.safe_load((directory / "model_config.yaml").read_text()) == expected
        observed = subprocess.check_output(
            [
                "bash",
                "-c",
                '"$UV_PYTHON" -I -c \'import os; print(os.environ.get("TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"))\'',
            ],
            env=env,
            text=True,
            timeout=10,
        ).strip()
        assert observed == str(inherited if legacy else "0")
    assert os.environ.get(key) == inherited
    assert (ROOT / case["config_path"]).read_bytes() == original


@pytest.mark.parametrize("method", ["fork", "forkserver"])
def test_short_temporary_directory_supports_real_multiprocessing_managers(
    evidence, tmp_path, method
):
    manifest, case, _ = evidence
    directory = tmp_path / ("long-nas-output-" * 12) / "attempt-1/repeat-1"
    directory.mkdir(parents=True)
    legacy_tmp = directory / "tmp"
    legacy_tmp.mkdir()
    command = [
        sys.executable,
        "-c",
        f'''
import multiprocessing
import os
import sys
from contextlib import ExitStack
from multiprocessing.managers import SyncManager
from pathlib import Path
context = multiprocessing.get_context({method!r})
with ExitStack() as stack:
    managers = []
    for rank in range(8):
        if sys.argv[1] == "explicit-address":
            manager = SyncManager(address=os.path.join(os.environ["TMPDIR"], f"manager-{{rank}}.sock"), ctx=context)
        else:
            manager = context.Manager()
        managers.append(stack.enter_context(manager))
    for rank, manager in enumerate(managers):
        assert Path(manager.address).is_relative_to(os.environ["TMPDIR"])
        queue = manager.Queue()
        queue.put(rank)
        assert queue.get(timeout=5) == rank
print("eight managers ready")
''',
    ]
    # Newer Python falls back to /tmp; the explicit address keeps the long socket path
    # behaviour of remote Python 3.12.
    legacy = subprocess.run(
        [*command, "explicit-address"],
        env={**os.environ, "TMPDIR": str(legacy_tmp)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert legacy.returncode != 0 and "AF_UNIX path too long" in legacy.stderr
    with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
        process = subprocess.run(
            [*command, "default-address"], env=env, capture_output=True, text=True, timeout=30
        )
        assert process.returncode == 0, process.stderr
        assert "eight managers ready" in process.stdout
    assert not Path(env["TMPDIR"]).exists()
    assert legacy_tmp.is_dir() and (directory / "model_config.yaml").is_file()


@pytest.mark.parametrize(
    "code,cancel,archive_failure",
    [(0, False, False), (23, False, False), (0, False, True), (0, True, False)],
)
def test_runtime_shell_wrapper_archives_and_preserves_failure(
    tmp_path, code, cancel, archive_failure
):
    fixture = tmp_path / "wrapper-boundary.py"
    fixture.write_text('''
import os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-m":
    raise SystemExit(0)
if args[1].endswith("ppu_functional_environment.py"):
    output = Path(args[args.index("--output") + 1])
    (output / "environment-fixture.json").write_text("{}")
else:
    output = Path(args[args.index("--output") + 1])
    output.mkdir()
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(143))
    (output / "started").write_text("started")
    if os.environ["FIXTURE_CANCEL"] == "true":
        time.sleep(60)
    raise SystemExit(int(os.environ["FIXTURE_CODE"]))
''')
    bootstrap = tmp_path / "wrapper-boundaries.sh"
    bootstrap.write_text('''
python3() { return 0; }
tar() { return 0; }
make() { return 0; }
python() {
    if [[ "${2:-}" == */ppu_functional_runtime.py ]]; then
        exec "$FIXTURE_PYTHON" "$FIXTURE_SCRIPT" "$@"
    else
        "$FIXTURE_PYTHON" "$FIXTURE_SCRIPT" "$@"
    fi
}
chmod() { [[ "$FIXTURE_ARCHIVE_FAILURE" == false ]] && command chmod "$@"; }
''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # exec bypasses shell functions, so stub the restore bootstrap executable.
    (bin_dir / "python3").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "python3").chmod(0o755)
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "BASH_ENV": str(bootstrap),
        "FIXTURE_PYTHON": sys.executable,
        "FIXTURE_SCRIPT": str(fixture),
        "FIXTURE_CODE": str(code),
        "FIXTURE_CANCEL": str(cancel).lower(),
        "FIXTURE_ARCHIVE_FAILURE": str(archive_failure).lower(),
        "PPU_RESULTS_DIR": str(output),
        "PPU_SOURCE_ROOT": str(tmp_path),
        "PPU_CONTROL_DIR": str(SCRIPTS),
        "PPU_ARTIFACT_DIR": str(tmp_path),
        "PPU_ENVIRONMENT_DIGEST": "d" * 64,
        "PPU_IMAGE_DIGEST": "ghcr.io/flytiger-eco/megatron-lm-for-sail/ppu-ci@sha256:" + "e" * 64,
        "PPU_BASE_IMAGE_DIGEST": functional.IMAGE.rsplit(":", 1)[0] + "@sha256:" + "f" * 64,
        "PPU_CONTROL_SHA": "a" * 40,
        "PPU_TARGET_SHA": "b" * 40,
    }
    process = subprocess.Popen(
        ["bash", str(SCRIPTS / "ppu_functional_run.sh"), "functional-000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if cancel:
            deadline = time.monotonic() + 10
            while not (output / "started").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (output / "started").is_file()
            process.send_signal(signal.SIGTERM)
        stdout, _ = process.communicate(timeout=15)
        assert process.returncode == (143 if cancel else 1 if archive_failure else code), stdout
        assert (output / "environment-fixture.json").is_file()
        timing = json.loads((output / "pod-timing.json").read_text())
        assert timing["elapsed_seconds"] >= 0
        if not archive_failure:
            assert timing["exit_code"] == process.returncode
            handoff = output / "action-timing.json"
            assert handoff.read_text() == ""
            assert handoff.stat().st_mode & 0o222 == 0o222
            assert output.stat().st_mode & 0o022 == 0
            assert (output / "pod-timing.json").stat().st_mode & 0o022 == 0
            workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-functional.yml").read_text())
            step = next(
                step
                for step in workflow["jobs"]["test"]["steps"]
                if step.get("name") == "Record execution and scheduling envelope"
            )
            outcome = "success" if process.returncode == 0 else "failure"
            action_status = "succeeded" if process.returncode == 0 else "failed"
            local_control = tmp_path / ".ppu-ci/control"
            local_control.mkdir(parents=True, exist_ok=True)
            for name in ("ppu_ci_export.py", "ppu_ci_paths.py"):
                (local_control / name).write_bytes((SCRIPTS / name).read_bytes())
            local_result = tmp_path / ".ppu-action"
            (tmp_path / "ppu-control-runtime").symlink_to(Path(sys.executable).parent.parent)
            subprocess.run(
                ["bash", "-euo", "pipefail", "-c", step["run"]],
                cwd=tmp_path,
                env={
                    **os.environ,
                    "RESULT_DIR": str(local_result),
                    "RUNNER_TEMP": str(tmp_path),
                    "REQUESTED_AT": "1",
                    "ACTION_OUTCOME": outcome,
                    "ACTION_STATUS": action_status,
                    "ACTION_OUTPUTS": "{}",
                },
                check=True,
            )
            assert (
                handoff.read_text() == ""
            ), "CPU envelope must not modify the private Pod placeholder"
            envelope = json.loads((local_result / "action-timing.json").read_text())
            assert envelope["action_outcome"] == outcome
            assert envelope["action_status"] == action_status
            assert envelope["requested_at"] == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_cross_session_discovery_orphans_and_pid_reuse(tmp_path, monkeypatch):
    def process(pid, ppid, pgid, sid, start=100):
        directory = tmp_path / str(pid)
        directory.mkdir(exist_ok=True)
        fields = ["S", str(ppid), str(pgid), str(sid)] + ["0"] * 15 + [str(start)]
        (directory / "stat").write_text(f"{pid} (python worker) " + " ".join(fields))

    monkeypatch.setattr(runtime, "Path", lambda _: tmp_path)
    process(10, 1, 10, 10)
    process(20, 10, 20, 20)
    process(30, 20, 20, 20)
    process(40, 1, 40, 40)
    known = {}
    assert set(runtime.process_family(10, known)) == {10, 20, 30}
    process(20, 1, 20, 20)
    process(31, 1, 20, 20)
    process(30, 1, 40, 40, start=200)
    assert set(runtime.process_family(10, known)) == {10, 20, 31}
    assert known[30]["start"] == 100
    signals = []
    monkeypatch.setattr(runtime.os, "killpg", lambda group, sig: signals.append((group, sig)))
    monkeypatch.setattr(runtime.os, "getpgrp", lambda: 999)
    runtime.signal_family(10, known, signal.SIGTERM)
    assert set(signals) == {(10, signal.SIGTERM), (20, signal.SIGTERM)}
