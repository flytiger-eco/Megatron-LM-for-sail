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
from test_ppu_functional_results import envelope, evidence, results

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


def test_runtime_runs_every_repeat_and_recomputes_original_comparator(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    original = functional.sha256(ROOT / case["config_path"])
    runner = FakeRunner(manifest, case)
    record = runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)
    assert record["success"] and record["attempts"] == 1
    assert len(runner.calls) == case["repeat"] == 2
    envelope(manifest, case, directory)
    assert results.verify_case(manifest, ROOT, case, directory)["success"]
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
    assert results.verify_case(manifest, ROOT, case, directory)["retried"]


@pytest.mark.parametrize(
    "code,message",
    [
        (1, "AssertionError: mismatch"),
        (1, "DistNetworkError\nloss = nan"),
        (1, "DistNetworkError\nOutOfMemoryError"),
        (1, "Segmentation fault"),
        (1, "MissingTensorboardLogsError"),
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


def test_assets_and_stale_directory_prevent_training(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    manifest["config"]["data_path"] = str(tmp_path / "absent")
    runner = FakeRunner(manifest, case)
    with pytest.raises(ValueError, match="asset preflight"):
        runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)
    assert not runner.calls
    with pytest.raises(FileExistsError):
        runtime.run_case(manifest, ROOT, case["id"], directory, runner=runner)


def test_assets_mode_never_launches_training(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    manifest["config"]["functional_mode"] = "assets"
    with pytest.raises(ValueError, match="explicit functional regression"):
        runtime.run_case(manifest, ROOT, case["id"], directory)
    assert not directory.exists()


def test_budget_exhausted_before_first_repeat_records_failure(evidence, tmp_path):
    manifest, case, directory = prepare_case(evidence, tmp_path)
    runner = FakeRunner(manifest, case)
    times = iter((0, case["budget_seconds"] + 1, case["budget_seconds"] + 1))
    record = runtime.run_case(
        manifest, ROOT, case["id"], directory, runner=runner, clock=lambda: next(times)
    )
    assert not record["success"] and record["timed_out"]
    assert not runner.calls


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


def test_environment_clears_inherited_shortcuts_and_uses_runtime_copy(
    evidence, tmp_path, monkeypatch
):
    manifest, case, _ = evidence
    monkeypatch.setenv("RECORD_CHECKPOINTS", "true")
    monkeypatch.setenv("ENABLE_LIGHTWEIGHT_MODE", "true")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    directory = tmp_path / "repeat"
    directory.mkdir()
    with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
        assert env["RECORD_CHECKPOINTS"] == "false" and env["ENABLE_LIGHTWEIGHT_MODE"] == "false"
        assert env["PYTEST_ADDOPTS"] == "" and "SLURM_JOB_END_TIME" not in env
        expected = yaml.safe_load((ROOT / case["config_path"]).read_text())
        expected.setdefault("ENV_VARS", {})["N_REPEAT"] = 1
        assert yaml.safe_load((directory / "model_config.yaml").read_text()) == expected


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
    # 新版 Python 会自动回退 /tmp；显式地址保留远端 Python 3.12 的长 socket 路径行为。
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


def test_temporary_directory_is_cleaned_on_unexpected_exception(evidence, tmp_path):
    manifest, case, _ = evidence
    directory = tmp_path / "repeat"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="unexpected"):
        with runtime.repeat_environment(manifest, ROOT, case, directory, 1, 1) as env:
            temporary = Path(env["TMPDIR"])
            (temporary / "scratch").write_text("transient")
            (directory / "cache/persistent").write_text("preserve")
            raise RuntimeError("unexpected")
    assert not temporary.exists()
    assert (directory / "cache/persistent").read_text() == "preserve"


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
    output = tmp_path / "output"
    env = {
        **os.environ,
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
            subprocess.run(
                ["bash", "-euo", "pipefail", "-c", step["run"]],
                env={
                    **os.environ,
                    "RESULT_DIR": str(output),
                    "REQUESTED_AT": "1",
                    "ACTION_OUTCOME": outcome,
                    "ACTION_STATUS": action_status,
                    "ACTION_OUTPUTS": "{}",
                },
                check=True,
            )
            envelope = json.loads(handoff.read_text())
            assert envelope["action_outcome"] == outcome
            assert envelope["action_status"] == action_status
            assert envelope["requested_at"] == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_workflow_functional_route_is_manual_only_and_preserves_ut():
    parent = yaml.safe_load((ROOT / ".github/workflows/ppu-build.yml").read_text())
    execute = yaml.safe_load((ROOT / ".github/workflows/ppu-execute.yml").read_text())["jobs"]
    workflow = yaml.safe_load((ROOT / ".github/workflows/ppu-functional.yml").read_text())
    events = parent.get("on", parent.get(True))
    inputs = events["workflow_dispatch"]["inputs"]
    assert inputs["test_kind"]["default"] == "unit"
    assert inputs["functional_mode"]["default"] == "assets"
    assert inputs["functional_repeat"]["default"] == 5
    assert len(inputs) <= 25
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_call"}
    assert execute["unit-tests"]["uses"] == "./.github/workflows/ppu-ut.yml"
    assert "functional-assets" in execute["admit-build"]["needs"]
    assert "needs.functional-assets.result == 'success'" in execute["admit-build"]["if"]
    assert not any(
        "ppu-distributed-action" in step.get("uses", "")
        for step in execute["functional-assets"]["steps"]
    )
    test = workflow["jobs"]["test"]
    assert test["strategy"]["max-parallel"] == 1 and test["strategy"]["fail-fast"] is False
    action = next(
        step for step in test["steps"] if "ppu-distributed-action" in step.get("uses", "")
    )
    assert action["with"]["nproc_per_node"] == 8
    assert action["with"]["node_selector"] == "board-type=ZW-M890P"
    assert action["with"]["image"] == functional.IMAGE
    assert "verify-bundle" in action["with"]["command"]
    assert "ppu_functional_run.sh" in action["with"]["command"]
