"""Run unchanged upstream functional pipelines with isolated evidence and bounded retries."""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

_spec = importlib.util.spec_from_file_location(
    "ppu_functional_results", Path(__file__).with_name("ppu_functional_results.py")
)
results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(results)
control = results.control


class ProcessRunner:
    """Forward cancellation to the entire training process group, including all ranks."""

    def __init__(self):
        self.cancelled = False
        self.timed_out = False
        self.process = None

    def cancel(self, signum, frame):
        self.cancelled = True
        self.terminate()

    def terminate(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run(self, command, *, cwd, env, log, deadline):
        remaining = deadline - time.monotonic()
        if self.cancelled:
            return 143
        if remaining <= 0:
            self.timed_out = True
            return 124
        with log.open("a") as stream:
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while self.process.poll() is None:
                    if self.cancelled or time.monotonic() >= deadline:
                        self.timed_out = not self.cancelled
                        self.terminate()
                        try:
                            self.process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                        # The launcher may exit before all rank processes have stopped.
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self.process.wait()
                        return 143 if self.cancelled else 124
                    time.sleep(0.1)
                if self.cancelled:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return 143
                return self.process.returncode
            finally:
                self.process = None


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


@contextmanager
def repeat_environment(
    manifest: dict, root: Path, case: dict, directory: Path, attempt: int, repeat: int
) -> Iterator[dict]:
    config = yaml.safe_load((root / case["config_path"]).read_text())
    config.setdefault("ENV_VARS", {})["N_REPEAT"] = 1
    (directory / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    # Upstream consumes these through the environment; no arguments reach its eval parser.
    env = dict(os.environ)
    for key in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "SLURM_JOB_END_TIME",
        "SLURM_JOB_START_TIME",
        "RUN_CI_PHASE_INDEX",
        "KEY",
    ):
        env.pop(key, None)
    env.update(
        {
            "TRAINING_SCRIPT_PATH": str(root / case["training_script"]),
            "TRAINING_PARAMS_PATH": str(directory / "model_config.yaml"),
            "GOLDEN_VALUES_PATH": str(root / case["golden_path"]),
            "OUTPUT_PATH": str(directory),
            "TENSORBOARD_PATH": str(directory / "tensorboard"),
            "CHECKPOINT_SAVE_PATH": str(directory / "checkpoints"),
            "CHECKPOINT_LOAD_PATH": str(directory / "initial-checkpoints"),
            "DATA_PATH": manifest["config"]["data_path"],
            "DATA_CACHE_PATH": str(directory / "cache"),
            "HF_HOME": str(directory / "cache/huggingface"),
            "XDG_CACHE_HOME": str(directory / "cache/xdg"),
            "TRITON_CACHE_DIR": str(directory / "cache/triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(directory / "cache/inductor"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "RECORD_CHECKPOINTS": "false",
            "N_REPEAT": "1",
            "ENABLE_LIGHTWEIGHT_MODE": "false",
            "SKIP_PYTEST": "0",
            "ALLOW_NONDETERMINISTIC_ALGO_ARG": "",
            "NUM_NODES": "1",
            "GPUS_PER_NODE": "8",
            "NODE_RANK": "0",
            "SLURM_NODEID": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RUN_CI_BARRIER_DIR": str(directory / "barriers"),
            "RUN_CI_BARRIER_ID": f"{attempt}-{repeat}",
            "PYTHONPATH": str(root),
            "PYTHONSAFEPATH": "1",
            "PYTEST_PLUGINS": "ppu_functional_pytest",
            "PYTEST_ADDOPTS": "",
            "UV_PROJECT_ENVIRONMENT": sys.prefix,
            "UV_PYTHON": sys.executable,
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_NO_SYNC": "1",
            "PPU_FUNCTIONAL_IDENTITY": json.dumps(manifest["identity"]),
            "PPU_FUNCTIONAL_CASE_ID": case["id"],
            "PPU_FUNCTIONAL_ATTEMPT": str(attempt),
            "PPU_FUNCTIONAL_REPEAT": str(repeat),
            "PPU_FUNCTIONAL_REPEAT_DIR": str(directory),
            "PPU_FUNCTIONAL_MANIFEST_SHA256": results.manifest_digest(manifest),
        }
    )
    for name in ("cache", "initial-checkpoints"):
        (directory / name).mkdir()
    # multiprocessing 在 TMPDIR 下创建 AF_UNIX socket；NAS 输出路径会超过地址长度限制。
    # 仅瞬态临时文件使用每次重复专属的短路径，持久缓存、checkpoint 和证据仍留在输出目录。
    with tempfile.TemporaryDirectory(prefix="ppu-fn-", dir="/tmp") as temporary_directory:
        env.update({name: temporary_directory for name in ("TMPDIR", "TMP", "TEMP")})
        yield env


def run_repeat(manifest, root, case, directory, attempt, repeat, runner, deadline):
    directory.mkdir(parents=True)
    env = {}
    started = time.monotonic()
    exit_code = 1
    shell_exit_code = None
    extraction_exit_code = None
    error = ""
    try:
        with repeat_environment(manifest, root, case, directory, attempt, repeat) as env:
            command = ["bash", str(root / "tests/functional_tests/shell_test_utils/run_ci_test.sh")]
            exit_code = runner.run(
                command, cwd=root, env=env, log=directory / "driver.log", deadline=deadline
            )
            shell_exit_code = exit_code
            if exit_code == 0 and case["observation"]:
                for phase in range(1, results.phase_count(case) + 1):
                    logs = directory / "1/tensorboard"
                    if results.phase_count(case) == 2:
                        logs /= f"run_{phase}"
                    name = "observation.json" if phase == 1 else "observation_2nd.json"
                    command = [
                        sys.executable,
                        str(
                            root
                            / "tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py"
                        ),
                        "--logs-dir",
                        str(logs),
                        "--train-iters",
                        str(case["effective_train_iters"]),
                        "--output-path",
                        str(directory / name),
                        "--is-normal-test",
                        "--step-size",
                        "1",
                    ]
                    extraction_env = {**env, "PYTEST_PLUGINS": ""}
                    exit_code = runner.run(
                        command,
                        cwd=root,
                        env=extraction_env,
                        log=directory / "driver.log",
                        deadline=deadline,
                    )
                    extraction_exit_code = exit_code
                    if exit_code != 0:
                        break
    except (OSError, ValueError) as exc:
        error = str(exc)
    finally:
        write_json(
            directory / "execution.json",
            {
                **results.identity_fields(manifest, case, attempt, repeat),
                "exit_code": exit_code,
                "shell_exit_code": shell_exit_code,
                "extraction_exit_code": extraction_exit_code,
                "cancelled": runner.cancelled,
                "timed_out": runner.timed_out,
                "duration_seconds": time.monotonic() - started,
                "runtime_config_sha256": control.sha256(directory / "model_config.yaml"),
                "launcher": case["launcher"],
                "launcher_internal_max_restarts": case["launcher_internal_max_restarts"],
                "temporary_directory": env.get("TMPDIR"),
                "error": error,
            },
        )
    try:
        if error:
            raise ValueError(error)
        result = results.verify_repeat(manifest, root, case, directory, attempt, repeat)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        result = {"success": False, "classification": "integrity_error", "error": str(exc)}
    write_json(directory / "verification.json", result)
    return result


def run_case(
    manifest: dict, root: Path, case_id: str, directory: Path, *, runner=None, clock=time.monotonic
) -> dict:
    if (
        manifest.get("test_kind") != "functional"
        or manifest["config"]["functional_mode"] != "regression"
    ):
        raise ValueError("Training requires an explicit functional regression manifest")
    case = manifest["functional"][case_id]
    results.verify_source(manifest, root, case)
    directory.mkdir(parents=True, exist_ok=False)
    report = control.asset_report(
        manifest, Path(manifest["config"]["data_path"]), location="pod", case_id=case_id
    )
    control.write_asset_report(report, directory)
    if not report["ok"]:
        raise ValueError("PPU asset preflight failed; no training was started")
    runner = runner or ProcessRunner()
    started = clock()
    deadline = started + case["budget_seconds"]
    record = {
        "identity": manifest["identity"],
        "case_id": case_id,
        "manifest_sha256": results.manifest_digest(manifest),
        "attempts": 0,
        "complete": False,
        "cancelled": False,
        "timed_out": False,
        "success": False,
    }
    try:
        for attempt in range(1, case["max_attempts"] + 1):
            if runner.cancelled or clock() >= deadline:
                runner.timed_out = not runner.cancelled
                break
            record["attempts"] = attempt
            for repeat in range(1, case["repeat"] + 1):
                result = run_repeat(
                    manifest,
                    root,
                    case,
                    directory / f"attempt-{attempt}" / f"repeat-{repeat}",
                    attempt,
                    repeat,
                    runner,
                    deadline,
                )
                if not result["success"]:
                    break
            if result["success"]:
                record["success"] = True
                break
            if result["classification"] != "infrastructure_retryable":
                break
        record["complete"] = True
    finally:
        record.update(
            duration_seconds=clock() - started,
            cancelled=runner.cancelled,
            timed_out=runner.timed_out,
        )
        if (
            record["duration_seconds"] > case["budget_seconds"]
            or runner.cancelled
            or runner.timed_out
        ):
            record["success"] = False
        write_json(directory / "case-run.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for key in ("target_sha", "control_sha"):
        if manifest["identity"][key] != os.environ[f"PPU_{key.upper()}"]:
            raise ValueError(f"Runtime identity mismatch: {key}")
    runner = ProcessRunner()
    signal.signal(signal.SIGTERM, runner.cancel)
    signal.signal(signal.SIGINT, runner.cancel)
    record = run_case(
        manifest, args.root.resolve(), args.case_id, args.output.resolve(), runner=runner
    )
    raise SystemExit(0 if record["success"] else 1)


if __name__ == "__main__":
    main()
