"""Prepare and attest functional-only tools without changing the fixed PPU image."""

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

NVRX_VERSION = "0.6.0"
YQ_VERSION = "4.47.2"
YQ = {
    "x86_64": ("amd64", "1bb99e1019e23de33c7e6afc23e93dad72aad6cf2cb03c797f068ea79814ddb0"),
    "aarch64": ("arm64", "05df1f6aed334f223bb3e6a967db259f7185e33650c3b6447625e16fea0ed31f"),
}


def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()


def nvrx_requirements(manifest: dict) -> dict:
    """Build 与执行 Pod 共用同一份清单，不能仅按 launcher 判断依赖。"""
    return {
        case_id: {"launcher": case["launcher"], "checkpoint": case["nvrx_checkpoint"]}
        for case_id, case in manifest["functional"].items()
        if case["nvrx_checkpoint"] or case["launcher"] == "ft_launcher"
    }


def check_nvrx(requirements: dict) -> dict:
    """只导入并核验原 API，不启动 worker、通信组或 checkpoint 写入。"""
    if not requirements:
        return {}
    version = importlib.metadata.version("nvidia-resiliency-ext")
    if version != NVRX_VERSION:
        raise ValueError(f"Functional NVRx version mismatch: {version} != {NVRX_VERSION}")
    stages = [stage for case in requirements.values() for stage in case["checkpoint"]]
    signatures = {}
    if stages:
        upstream = importlib.import_module("megatron.core.dist_checkpointing.strategies.torch")
        strategy, modules = upstream.get_async_strategy("nvrx")
        if strategy != "nvrx":
            raise ValueError("Original async checkpoint strategy must remain nvrx")
        queue = modules["AsyncCallsQueue"]
        request = inspect.signature(modules["AsyncRequest"])
        request.bind(None, (), [], async_fn_kwargs={}, preload_fn=None)
        signatures["AsyncRequest"] = str(request)
        init = inspect.signature(queue.__init__)
        init.bind(None, persistent=any(stage["persistent_worker"] for stage in stages))
        signatures["AsyncCallsQueue.__init__"] = str(init)
        if any(stage["cpu_shm"] for stage in stages):
            writer = inspect.signature(modules["FileSystemWriterAsync"].__init__)
            if "cpu_shm_mode" not in writer.parameters:
                raise ValueError("NVRx FileSystemWriterAsync requires cpu_shm_mode support")
            signatures["FileSystemWriterAsync.__init__"] = str(writer)
        if any(stage["persistent_worker"] for stage in stages):
            warmup = inspect.signature(queue.warmup_persistent_caller)
            kwargs = {"cpu_priority": 10, "io_priority": 3}
            if any(stage["cpu_shm"] and stage["persistent_worker"] for stage in stages):
                if "cpu_shm_mode" not in warmup.parameters:
                    raise ValueError("NVRx persistent worker requires cpu_shm_mode support")
                kwargs["cpu_shm_mode"] = True
            warmup.bind(0, **kwargs)
            results_queue = inspect.signature(modules["get_write_results_queue"])
            if "mp_mode" not in results_queue.parameters:
                raise ValueError("NVRx get_write_results_queue requires mp_mode support")
            results_queue.bind(mp_mode="fork")
            signatures["AsyncCallsQueue.warmup_persistent_caller"] = str(warmup)
            signatures["get_write_results_queue"] = str(results_queue)
    if any(case["launcher"] == "ft_launcher" for case in requirements.values()):
        if not shutil.which("ft_launcher"):
            raise ValueError("Selected case requires the original ft_launcher")
        command_output(["ft_launcher", "--help"])
    return {"version": version, "requirements": requirements, "signatures": signatures}


def prepare(bundle: Path, manifest: dict):
    tools = bundle / "functional-tools"
    tools.mkdir(exist_ok=False)
    arch, digest = YQ[platform.machine()]
    url = f"https://github.com/mikefarah/yq/releases/download/v{YQ_VERSION}/yq_linux_{arch}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Official Mike Farah yq binary digest mismatch")
    (tools / "yq").write_bytes(content)
    (tools / "yq").chmod(0o755)
    envsubst = shutil.which("envsubst")
    if not envsubst:
        raise ValueError("GNU envsubst must be installed by the Build preparation step")
    shutil.copy2(envsubst, tools / "envsubst")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            os.environ["PIP_INDEX_URL"],
            "tensorboard",
            "click",
            "pyyaml",
        ],
        check=True,
    )
    if nvrx_requirements(manifest):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                os.environ["PIP_INDEX_URL"],
                f"nvidia-resiliency-ext=={NVRX_VERSION}",
            ],
            check=True,
        )
    site = command_output(
        [sys.executable, "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))']
    )
    shutil.copy2(
        bundle / "control/ppu_functional_pytest.py", Path(site) / "ppu_functional_pytest.py"
    )


def check(bundle: Path, manifest: dict, output: Path):
    tools = bundle / "functional-tools"
    arch, expected = YQ[platform.machine()]
    if hashlib.sha256((tools / "yq").read_bytes()).hexdigest() != expected:
        raise ValueError("Functional yq binary digest mismatch")
    for name in ("yq", "envsubst"):
        shutil.copy2(tools / name, Path("/usr/local/bin") / name)
    bash = command_output(["bash", "--version"])
    if "GNU bash, version" not in bash or int(bash.split("version ")[1].split(".")[0]) < 4:
        raise ValueError("Functional tests require GNU Bash >=4")
    yq = command_output(["/usr/local/bin/yq", "--version"])
    if "mikefarah/yq" not in yq or f"v{YQ_VERSION}" not in yq:
        raise ValueError("Functional tests require the attested Mike Farah yq v4")
    envsubst = command_output(["envsubst", "--version"])
    if "GNU gettext" not in envsubst:
        raise ValueError("Functional tests require GNU envsubst")
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "tensorboard",
            "uv",
            "pytest",
            "numpy",
            "pydantic",
            "transformer-engine",
        )
    }
    for name in ("tensorboard", "yaml", "click", "ppu_functional_pytest"):
        importlib.import_module(name)
    nvrx = check_nvrx(nvrx_requirements(manifest))
    if nvrx:
        packages["nvidia-resiliency-ext"] = nvrx["version"]
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("Functional tests require exactly eight visible PPU devices")
    if int(os.environ.get("NPROC_PER_NODE", "8")) != 8 or int(os.environ.get("NNODES", "1")) != 1:
        raise ValueError("Functional tests require one eight-card node")
    # Upstream uses this path for the empty initial load only, never to save these cases.
    empty = Path("/tmp/checkpoints")
    if empty.is_symlink() or (empty.exists() and (not empty.is_dir() or any(empty.iterdir()))):
        raise ValueError("Upstream initial checkpoint path must be empty; refusing stale state")
    output.mkdir(parents=True, exist_ok=True)
    (output / "functional-environment.json").write_text(
        json.dumps(
            {
                "identity": manifest["identity"],
                "image": manifest["config"]["image"],
                "python": sys.version,
                "packages": packages,
                "nvrx": nvrx,
                "bash": bash,
                "yq": yq,
                "envsubst": envsubst,
                "uv": command_output(["uv", "--version"]),
                "architecture": arch,
                "devices": torch.cuda.device_count(),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if manifest.get("test_kind") != "functional" or not manifest["config"]["build"]:
        raise ValueError("Functional environment setup requires a regression build")
    if args.command == "prepare":
        prepare(args.bundle, manifest)
    check(args.bundle, manifest, args.output or args.bundle)


if __name__ == "__main__":
    main()
