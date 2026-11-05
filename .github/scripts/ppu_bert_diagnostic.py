"""复用固定历史产物的单次 BERT 上下文取证；永不进入正式数值汇总。"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
HISTORICAL_RUN = "41070324511"
HISTORICAL_SHA = "3eee00c953c0699e1a28bc7f53d3f0fbc017d4f5"
BUNDLE_DIGEST = "a4bff0e73d62bda207775f57986e77c8e561d5b7ceda19cdb24767b19985c7df"
MANIFEST_DIGEST = "42052aabd1591f06467232ba5b73f7ee08c1aa306851b9e63f75e29f49ca76b6"
CPU_BUNDLE = Path(f"/wl_nas/devops/megatron-ci/{HISTORICAL_RUN}-1/build")
POD_BUNDLE = Path(f"/mnt/wl_nas/devops/megatron-ci/{HISTORICAL_RUN}-1/build")
CASE_ID = "functional-000"
CASE_NAME = "bert/bert_mcore_tp1_pp2"
VENV = Path("/opt/ppu-ci-venv")
HOOK_MODULE = "ppu_bert_stack_hook"
HOOK_PTH = "ppu_bert_stack_hook.pth"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = load_module("ppu_bert_process_control", SCRIPT.with_name("ppu_compiler_diagnostic.py"))
bundle_control = load_module("ppu_bert_bundle_control", SCRIPT.with_name("ppu_ci.py"))


def verify_manifest(raw: bytes) -> dict:
    """固定原始字节摘要，区分历史被测身份与本次控制代码身份。"""
    if hashlib.sha256(raw).hexdigest() != MANIFEST_DIGEST:
        raise ValueError("Historical manifest digest mismatch")
    manifest = json.loads(raw)
    if manifest["identity"] != {
        "run_id": HISTORICAL_RUN,
        "run_attempt": 1,
        "target_sha": HISTORICAL_SHA,
        "control_sha": HISTORICAL_SHA,
    }:
        raise ValueError("Historical bundle identity mismatch")
    case = manifest["functional"][CASE_ID]
    if not (
        manifest["test_kind"] == "functional"
        and manifest["config"]["functional_mode"] == "regression"
        and case["name"] == CASE_NAME
        and case["test_type"] == "regular"
        and case["effective_train_iters"] == case["train_iters"] == 50
        and case["cards"] == 8
        and case["launcher"] == "torchrun"
        and case["launcher_internal_max_restarts"] == 0
        and not case["observation"]
    ):
        raise ValueError("Historical BERT diagnostic scope mismatch")
    return manifest


def verify_historical_bundle(directory: Path) -> dict:
    """CPU 只读检查与 Pod 副本复验共用完整摘要门禁，不信任目录名称。"""
    # 信任锚是原 BERT action 的 PPU_BUNDLE_DIGEST，不是 NAS 自报摘要。
    # 固定 inventory 认证了全部 manifest/control 字节，因此可作为自身 plan。
    # verify_bundle 仍独立检查完整清单、文件内容、.complete 与 SHA 标记。
    bundle_control.verify_bundle(directory, directory, BUNDLE_DIGEST)
    manifest = verify_manifest((directory / "manifest.json").read_bytes())
    if (directory / "image.txt").read_text().strip() != manifest["config"]["image"]:
        raise ValueError("Historical image identity mismatch")
    return manifest


def preflight(output: Path) -> None:
    """失败也归档本地 CPU envelope；绝不创建或修改 NAS run 目录。"""
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "numerical_validation": False,
        "historical_run_id": HISTORICAL_RUN,
        "bundle_digest": BUNDLE_DIGEST,
        "bundle_path": str(CPU_BUNDLE),
        "ok": False,
    }
    try:
        manifest = verify_historical_bundle(CPU_BUNDLE)
        report.update(ok=True, historical_identity=manifest["identity"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        diagnostic.write_json(output / "historical-preflight.json", report)


def stage(output: Path, name: str, **details) -> None:
    """记录环境复原和原训练入口是否实际到达。"""
    row = {"stage": name, "time": time.time(), "numerical_validation": False, **details}
    with (output / "stages.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def install_hook(python: Path, venv: Path, hook: Path) -> dict:
    """仅给本次恢复的 venv 增加观测钩子，覆盖 Python -I 与 fork。"""
    site = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
            text=True,
            timeout=10,
        ).strip()
    )
    if not site.resolve().is_relative_to(venv.resolve()):
        raise ValueError("Diagnostic hook must stay inside the restored venv")
    destination = site / f"{HOOK_MODULE}.py"
    with destination.open("xb") as stream:
        stream.write(hook.read_bytes())
    with (site / HOOK_PTH).open("x") as stream:
        stream.write(f"import {HOOK_MODULE}\n")
    return {"module": str(destination), "sha256": hashlib.sha256(hook.read_bytes()).hexdigest()}


# 复用原 functional 启动所需的镜像内环境清理与 check；无安装、Build 或重试。
SETUP_SCRIPT = r'''
set -euo pipefail
root="$1"; bundle="$2"; output="$3"; script="$4"
cd "$root"
python3 -m pip uninstall --yes cloud-tpu-client protobuf
export PATH="/opt/ppu-ci-venv/bin:$PATH"
export VIRTUAL_ENV=/opt/ppu-ci-venv UV_PROJECT_ENVIRONMENT=/opt/ppu-ci-venv
export PYTHONSAFEPATH=1 PYTHONPATH="$root" PPU_SOURCE_ROOT="$root"
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' PIP_FIND_LINKS=''
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1
unset WORLD_SIZE RANK LOCAL_RANK
python -m pip check
make -q -C megatron/core/datasets
python -I "$bundle/control/ppu_functional_environment.py" check --bundle "$bundle" --output "$output"
exec python -I "$script" train --work "$root" --bundle "$bundle" --output "$output"
'''


def execute(work: Path, output: Path) -> None:
    """整个复原阶段也在外层进程树预算内；输入 bundle 始终只读。"""
    stage(output, "historical_bundle_verifying")
    verify_historical_bundle(POD_BUNDLE)
    bundle = work / "bundle"
    shutil.copytree(POD_BUNDLE, bundle, symlinks=True)
    manifest = verify_historical_bundle(bundle)
    for name in ("manifest.json", "bundle-integrity.json"):
        shutil.copy2(bundle / name, output / name)
    stage(output, "bundle_copy_verified", historical_identity=manifest["identity"])
    root = work / "source"
    root.mkdir()
    if VENV.exists() or VENV.is_symlink():
        raise ValueError("Refusing to overwrite an existing diagnostic venv")
    subprocess.run(["tar", "-xzf", str(bundle / "source.tar.gz"), "-C", str(root)], check=True)
    subprocess.run(["tar", "-xzf", str(bundle / "environment.tar.gz"), "-C", "/opt"], check=True)
    hook = install_hook(
        VENV / "bin/python", VENV, SCRIPT.with_name("ppu_diagnostic_sitecustomize.py")
    )
    stage(output, "environment_restored", hook=hook)
    env = {**os.environ, "PPU_DIAGNOSTIC_STACK_DIR": str(output / "stacks")}
    os.execve(
        "/bin/bash",
        [
            "bash",
            "-c",
            SETUP_SCRIPT,
            "ppu-bert-diagnostic",
            str(root),
            str(bundle),
            str(output),
            str(SCRIPT),
        ],
        env,
    )


def train(root: Path, bundle: Path, output: Path) -> int:
    """只调用一次历史原 shell；不调用带重复/重试循环的 run_case。"""
    manifest = verify_manifest((bundle / "manifest.json").read_bytes())
    runtime = load_module("ppu_bert_original_runtime", bundle / "control/ppu_functional_runtime.py")
    case = manifest["functional"][CASE_ID]
    runtime.results.verify_source(manifest, root, case)
    assets = runtime.control.asset_report(
        manifest, Path(manifest["config"]["data_path"]), location="pod", case_id=CASE_ID
    )
    runtime.control.write_asset_report(assets, output)
    if not assets["ok"]:
        raise ValueError("BERT assets unavailable; original training was not started")
    directory = output / "repeat-1"
    directory.mkdir()
    with runtime.repeat_environment(manifest, root, case, directory, 1, 1) as env:
        command = ["bash", str(root / "tests/functional_tests/shell_test_utils/run_ci_test.sh")]
        stage(
            output,
            "original_shell_started",
            command=command,
            historical_identity=manifest["identity"],
            case=case["name"],
            train_iters=50,
            repeat=1,
            max_attempts=1,
            compile_threads_override=env.get("TORCHINDUCTOR_COMPILE_THREADS"),
            temporary_directory=env["TMPDIR"],
        )
        with (directory / "driver.log").open("w") as stream:
            code = subprocess.call(
                command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        stage(output, "original_shell_exited", exit_code=code)
    return code


def run_diagnostic(root: Path, output: Path, authorization: dict) -> int:
    """单个被监控进程树覆盖校验、复原、训练与清理，无自动重启。"""
    diagnostic.verify_authorization(authorization, os.environ)
    if authorization["compiler_diagnostic_context"] != "bert":
        raise ValueError("BERT diagnostic requires BERT authorization")
    started = time.monotonic()
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "context": "bert",
        "authorization": authorization,
        "historical_run_id": HISTORICAL_RUN,
        "historical_sha": HISTORICAL_SHA,
        "bundle_digest": BUNDLE_DIGEST,
        "case": CASE_NAME,
        "train_iters": 50,
        "repeat": 1,
        "max_attempts": 1,
        "budget_seconds": diagnostic.BUDGET_SECONDS,
        "numerical_validation": False,
        "completed": False,
        "source_sha256": {},
    }
    runner = diagnostic.DiagnosticRunner()
    handlers = {sig: signal.signal(sig, runner.cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for name in (
            SCRIPT.name,
            "ppu_compiler_diagnostic.py",
            "ppu_ci.py",
            "ppu_diagnostic_sitecustomize.py",
        ):
            relative = f".github/scripts/{name}"
            committed = subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={root}",
                    "-C",
                    str(root),
                    "show",
                    f"HEAD:{relative}",
                ],
                timeout=5,
            )
            if committed != SCRIPT.with_name(name).read_bytes():
                raise ValueError(f"Diagnostic control differs from committed snapshot: {name}")
            report["source_sha256"][relative] = hashlib.sha256(committed).hexdigest()
        with tempfile.TemporaryDirectory(prefix="ppu-bc-", dir="/tmp") as work:
            command = [
                sys.executable,
                "-I",
                str(SCRIPT),
                "execute",
                "--work",
                work,
                "--output",
                str(output),
            ]
            report["outcome"] = runner.run(
                command, dict(os.environ), output, started + diagnostic.BUDGET_SECONDS - 10
            )
        outcome = report["outcome"]
        report["completed"] = (
            outcome["exit_code"] == 0 and not outcome["timed_out"] and not outcome["cancelled"]
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(duration_seconds=time.monotonic() - started, cancelled=runner.cancelled)
        diagnostic.write_json(output / "diagnostic.json", report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        # 仅开放本次证据读取，不归档恢复出来的整个 venv/source，也不修改历史 NAS。
        for path in output.rglob("*"):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | 0o444 | (0o111 if path.is_dir() else 0))
    return 0 if report["completed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "train"))
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.mode == "execute":
        execute(args.work, args.output)
    else:
        raise SystemExit(train(args.work, args.bundle, args.output))


if __name__ == "__main__":
    main()
