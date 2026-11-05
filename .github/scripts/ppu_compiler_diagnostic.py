"""单 Pod 八卡有界编译诊断；不运行模型回归，不产生数值验收结论。"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

BUDGET_SECONDS = 500
VARIANT_SECONDS = 230
VARIANTS = ("default", "single_thread")
BDA = "megatron/core/fusions/fused_bias_dropout.py"
SCRIPT = Path(__file__).resolve()


def write_json(path: Path, value: dict) -> None:
    """原子保存证据。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def verify_authorization(authorization: dict, env: dict) -> None:
    """要求同一次运行的维护者授权；禁止 PR、重跑和不同源码快照。"""
    target = env.get("PPU_TARGET_SHA", "")
    run_id = env.get("PPU_DIAGNOSTIC_RUN_ID", env.get("GITHUB_RUN_ID", ""))
    attempt = env.get("PPU_DIAGNOSTIC_RUN_ATTEMPT", env.get("GITHUB_RUN_ATTEMPT", ""))
    if not (
        re.fullmatch(r"[0-9a-f]{40}", target)
        and target == env.get("PPU_CONTROL_SHA")
        and authorization.get("target_sha") == target
        and authorization.get("control_sha") == target
        and authorization.get("allowed") is True
        and authorization.get("test_kind") == "compiler_diagnostic"
        and env.get("PPU_DIAGNOSTIC_CONTEXT", "bda") in ("bda", "bert")
        and authorization.get("compiler_diagnostic_context")
        == env.get("PPU_DIAGNOSTIC_CONTEXT", "bda")
        and authorization.get("event_name") == "workflow_dispatch"
        and authorization.get("pr_number") == 0
        and authorization.get("manual_pr") is False
        and str(authorization.get("run_id")) == run_id
        and run_id.isdecimal()
        and str(authorization.get("run_attempt")) == attempt == "1"
    ):
        raise ValueError("Invalid compiler diagnostic authorization or run identity")


def load_bda(root: Path, torch_module):
    """保留原函数 AST，仅注入 torch.compile 以隔离非编译依赖导入。"""
    path = root / BDA
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "megatron.core.jit"
    ]
    if len(imports) != 1 or [(item.name, item.asname) for item in imports[0].names] != [
        ("jit_fuser", None)
    ]:
        raise ValueError("Unexpected original BDA compiler import")
    tree.body.remove(imports[0])
    module = types.ModuleType("ppu_diagnostic_original_bda")
    module.__file__ = str(path)
    module.jit_fuser = torch_module.compile
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module.bias_dropout_add_fused_train


def verify_runtime(report: dict) -> None:
    """分别核验安装包与模块版本，厂商后缀不一定出现在 torch.__version__。"""
    if not (
        report["device_count"] == 8
        and report["torch_distribution"] == "2.13.0+v0.1.0.ppu2.2.0"
        and report["torch"] in ("2.13.0", "2.13.0+v0.1.0.ppu2.2.0")
        and report["triton"] == "3.7.0+v0.1.0.ppu2.2.0"
    ):
        raise ValueError("Diagnostic requires the original eight-card PPU runtime")


def probe(root: Path, output: Path, variant: str) -> None:
    """使用原 BDA 的 BF16 形状、dropout、梯度和确定性设置。"""
    import importlib.metadata

    import torch
    import torch._inductor.config as config

    rank = int(os.environ["LOCAL_RANK"])
    report = {"rank": rank, "pid": os.getpid(), "variant": variant, "phase": "imported"}

    def phase(name: str) -> None:
        report.update(phase=name, updated_at=time.time())
        write_json(output / f"rank-{rank}.json", report)
        print(json.dumps(report), flush=True)

    try:
        report.update(
            torch=torch.__version__,
            torch_distribution=importlib.metadata.version("torch"),
            triton=importlib.metadata.version("triton"),
            compile_threads=config.compile_threads,
            worker_start_method=config.worker_start_method,
            device_count=torch.cuda.device_count(),
            numerical_validation=False,
        )
        phase("runtime_verified")
        verify_runtime(report)
        if variant == "single_thread" and config.compile_threads != 1:
            raise ValueError("Single-thread compiler setting was not applied")
        torch.cuda.set_device(rank)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        phase("device_initialized")
        fused = load_bda(root, torch)
        x = torch.randn(512, 4, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        residual = torch.randn_like(x, requires_grad=True)
        torch.cuda.synchronize()
        phase("forward_compile")
        value = fused((x, bias), residual, 0.1)
        torch.cuda.synchronize()
        phase("forward_complete")
        value.sum().backward()
        torch.cuda.synchronize()
        phase("backward_complete")
        value = fused((x, bias), residual, 0.1)
        torch.cuda.synchronize()
        phase("warm_forward_complete")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        phase("failed")
        raise


def variant_environment(
    base: dict, output: Path, bootstrap: Path, temporary: str, variant: str
) -> dict:
    """两个冷缓存变体仅改变编译线程设置，清除继承的分布式上下文。"""
    if variant not in VARIANTS:
        raise ValueError("Unknown compiler diagnostic variant")
    env = {
        key: value
        for key, value in base.items()
        if not key.startswith(("TORCHINDUCTOR_", "TRITON_"))
        and key
        not in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
    }
    env.update(
        PYTHONPATH=str(bootstrap),
        PYTHONUNBUFFERED="1",
        TMPDIR=temporary,
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
        NCCL_ALGO="Ring",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TORCHINDUCTOR_CACHE_DIR=str(output / "cache/inductor"),
        TRITON_CACHE_DIR=str(output / "cache/triton"),
        CUDA_CACHE_PATH=str(output / "cache/cuda"),
        PPU_DIAGNOSTIC_STACK_DIR=str(output / "stacks"),
        OMP_NUM_THREADS="1",
    )
    if variant == "single_thread":
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    return env


def proc_text(path: Path) -> str:
    """只读有界 /proc 字段；权限或退出竞态原样记录。"""
    try:
        with path.open(errors="replace") as stream:
            return stream.read(16384)
    except OSError as error:
        return f"unavailable: {error}"


def process_family(root_pid: int, known: dict) -> dict:
    """跟踪跨 setsid 的后代；保留会话归属以回收已成为孤儿的编译 worker。"""
    table = {}
    for path in Path("/proc").glob("[0-9]*"):
        fields = proc_text(path / "stat").rsplit(")", 1)[-1].split()
        try:
            table[int(path.name)] = {
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "sid": int(fields[3]),
                "start": int(fields[19]),
            }
        except (ValueError, IndexError):
            continue
    sessions = {root_pid, *(row["sid"] for row in known.values())}
    selected = {
        pid: row
        for pid, row in table.items()
        if row["sid"] in sessions or (pid in known and known[pid]["start"] == row["start"])
    }
    while True:
        children = {
            pid: row
            for pid, row in table.items()
            if pid not in selected and row["ppid"] in selected
        }
        if not children:
            break
        selected.update(children)
    known.update(selected)
    return selected


def signal_family(root_pid: int, known: dict, sig: int) -> None:
    """只向本次 launcher 及其后代会话的进程组转发终止信号。"""
    groups = {root_pid, *(row["pgid"] for row in process_family(root_pid, known).values())}
    for group in groups:
        if group <= 1 or group == os.getpgrp():
            raise ValueError("Refusing to signal the controller process group")
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass


def capture_processes(session: int, output: Path, known: dict | None = None) -> None:
    """记录所有后代会话的线程内核栈、等待点和 Python 采样覆盖。"""
    processes = []
    family = process_family(session, {} if known is None else known)
    for pid in family:
        path = Path("/proc") / str(pid)
        stack = output / "stacks" / f"python-{path.name}.log"
        status = proc_text(path / "status")
        caught = re.search(r"^SigCgt:\s*([0-9a-fA-F]+)$", status, re.MULTILINE)
        ready = stack.with_suffix(".ready").is_file()
        if ready and caught and int(caught[1], 16) & (1 << (signal.SIGUSR1 - 1)):
            try:
                os.kill(int(path.name), signal.SIGUSR1)
            except ProcessLookupError:
                pass
        tasks = [
            {
                "tid": task.name,
                **{name: proc_text(task / name) for name in ("comm", "wchan", "stack", "syscall")},
            }
            for task in sorted((path / "task").glob("[0-9]*"))
        ]
        processes.append(
            {
                "pid": int(path.name),
                "status": status,
                "command": proc_text(path / "cmdline").replace("\x00", " "),
                "python_stack_hook": ready and stack.is_file(),
                "tasks": tasks,
            }
        )
    with (output / "processes.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "processes": processes}) + "\n")


class DiagnosticRunner:
    """取消或预算结束时清理整个变体进程组，包括编译 worker。"""

    def __init__(self):
        self.cancelled = False

    def cancel(self, _signum, _frame) -> None:
        self.cancelled = True

    def run(self, command: list, env: dict, output: Path, deadline: float) -> dict:
        started = time.monotonic()
        timed_out = False
        if self.cancelled or started >= deadline:
            return {
                "exit_code": 143 if self.cancelled else 124,
                "timed_out": not self.cancelled,
                "cancelled": self.cancelled,
                "duration_seconds": 0,
            }
        with (output / "launcher.log").open("w") as stream:
            process = subprocess.Popen(
                command, env=env, stdout=stream, stderr=stream, start_new_session=True
            )
            known = {}
            try:
                next_capture = started + 15
                next_tracking = started
                while process.poll() is None:
                    now = time.monotonic()
                    if now >= next_tracking:
                        process_family(process.pid, known)
                        next_tracking = now + 0.2
                    if now >= next_capture:
                        capture_processes(process.pid, output, known)
                        next_capture = now + 30
                    if self.cancelled or now >= deadline:
                        timed_out = not self.cancelled
                        capture_processes(process.pid, output, known)
                        break
                    time.sleep(0.2)
            finally:
                # torchrun 的 rank 会另建会话；launcher 退出不意味着所有后代已退出。
                try:
                    signal_family(process.pid, known, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
                finally:
                    signal_family(process.pid, known, signal.SIGKILL)
                    process.wait(timeout=3)
                    remaining = {}
                    for _ in range(20):
                        remaining = {
                            pid: row
                            for pid, row in process_family(process.pid, known).items()
                            if not re.search(
                                r"^State:\s+Z", proc_text(Path(f"/proc/{pid}/status")), re.MULTILINE
                            )
                        }
                        if not remaining:
                            break
                        time.sleep(0.1)
                    write_json(
                        output / "cleanup.json",
                        {"launcher_pid": process.pid, "tracked": known, "remaining": remaining},
                    )
                    if remaining:
                        raise RuntimeError("Diagnostic descendants survived cleanup")
        return {
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "cancelled": self.cancelled,
            "duration_seconds": time.monotonic() - started,
        }


def run_diagnostic(root: Path, output: Path, authorization: dict) -> int:
    """单次执行两个变体；总预算固定，禁止重试或提交额外 Pod。"""
    started = time.monotonic()
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "authorization": authorization,
        "numerical_validation": False,
        "budget_seconds": BUDGET_SECONDS,
        "variants": [],
        "source_sha256": {},
        "probe": {
            "shape": [512, 4, 1024],
            "dtype": "bfloat16",
            "dropout": 0.1,
            "deterministic": True,
        },
        "scope": "原 BDA 计算隔离诊断；仅替换 jit_fuser 导入，不载入完整模型或分布式通信",
    }
    runner = DiagnosticRunner()
    handlers = {sig: signal.signal(sig, runner.cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for name in (
            BDA,
            "megatron/core/jit.py",
            str(SCRIPT.relative_to(root)),
            ".github/scripts/ppu_diagnostic_sitecustomize.py",
        ):
            committed = subprocess.check_output(
                ["git", "-c", f"safe.directory={root}", "-C", str(root), "show", f"HEAD:{name}"],
                timeout=5,
            )
            if committed != (root / name).read_bytes():
                raise ValueError(f"Diagnostic source differs from committed snapshot: {name}")
            report["source_sha256"][name] = hashlib.sha256(committed).hexdigest()
        bootstrap = output / "bootstrap"
        bootstrap.mkdir()
        shutil.copyfile(
            SCRIPT.with_name("ppu_diagnostic_sitecustomize.py"), bootstrap / "sitecustomize.py"
        )
        for variant in VARIANTS:
            if runner.cancelled or time.monotonic() >= started + BUDGET_SECONDS - 10:
                break
            directory = output / variant
            directory.mkdir()
            with tempfile.TemporaryDirectory(prefix="ppu-dg-", dir="/tmp") as temporary:
                env = variant_environment(
                    dict(os.environ), directory, bootstrap, temporary, variant
                )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    "--nproc-per-node=8",
                    "--max-restarts=0",
                    str(SCRIPT),
                    "probe",
                    "--root",
                    str(root),
                    "--output",
                    str(directory),
                    "--variant",
                    variant,
                ]
                outcome = runner.run(
                    command,
                    env,
                    directory,
                    min(time.monotonic() + VARIANT_SECONDS, started + BUDGET_SECONDS - 10),
                )
            ranks = [json.loads(path.read_text()) for path in sorted(directory.glob("rank-*.json"))]
            outcome.update(name=variant, ranks=ranks)
            outcome["completed"] = (
                outcome["exit_code"] == 0
                and not outcome["timed_out"]
                and not outcome["cancelled"]
                and len(ranks) == 8
                and {row["rank"] for row in ranks} == set(range(8))
                and all(row["phase"] == "warm_forward_complete" for row in ranks)
            )
            report["variants"].append(outcome)
            write_json(output / "diagnostic.json", report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(duration_seconds=time.monotonic() - started, cancelled=runner.cancelled)
        report["completed"] = len(report["variants"]) == 2 and all(
            row["completed"] for row in report["variants"]
        )
        write_json(output / "diagnostic.json", report)
        # 只开放本次输出的读取权限，CPU 归档不需要输入资产或缓存的写权限。
        for path in output.rglob("*"):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | 0o444 | (0o111 if path.is_dir() else 0))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0 if report["completed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("authorize", "run", "probe"))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    context = os.environ.get("PPU_DIAGNOSTIC_CONTEXT", "bda")
    if args.mode == "probe":
        if context != "bda":
            raise ValueError("BDA probe cannot run in BERT context")
        probe(args.root, args.output, args.variant)
        return
    authorization = json.loads(args.authorization.read_text())
    verify_authorization(authorization, os.environ)
    if context == "bert":
        spec = importlib.util.spec_from_file_location(
            "ppu_bert_diagnostic", SCRIPT.with_name("ppu_bert_diagnostic.py")
        )
        bert = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bert)
        if args.mode == "authorize":
            bert.preflight(args.output)
        else:
            raise SystemExit(bert.run_diagnostic(args.root, args.output, authorization))
    elif args.mode == "run":
        raise SystemExit(run_diagnostic(args.root, args.output, authorization))


if __name__ == "__main__":
    main()
