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
NAMESPACE_PTH = "00-ppu-functional-google-namespace.pth"
# 先建立完整的 PEP 420 namespace，避免继承的 nspkg 启动钩子将 google 固定到系统目录。
NAMESPACE_BOOTSTRAP = "import importlib; importlib.invalidate_caches(); import google\n"
TRAINING_PACKAGES = {
    "pla": "1.0.0+v0.1.0.ppu2.2.0",
    "mamba-ssm": "2.3.1+v0.1.0.ppu2.2.0",
    "causal-conv1d": "1.7.0+v0.1.0.ppu2.2.0",
    "flash-linear-attention": "0.4.1+v0.1.0.ppu2.2.0",
    "fla-core": "0.4.1+v0.1.0.ppu2.2.0",
    "emerging-optimizers": "0.3.0",
    "apex": "24.4.1+v0.1.0.ppu2.2.0",
}
TRAINING_IMPORTS = {
    "pla": (),
    "pla.decode": (),
    "pla.prefill.flashqla": (),
    "pla.prefill.flashkdapro": (),
    "mamba_ssm.ops.triton.layernorm_gated": ("RMSNorm",),
    "mamba_ssm.ops.triton.ssd_combined": (
        "mamba_chunk_scan_combined",
        "mamba_split_conv1d_scan_combined",
    ),
    "selective_scan_cuda": (),
    "causal_conv1d": ("causal_conv1d_fn", "causal_conv1d_update"),
    "causal_conv1d_cuda": (),
    "fla": (),
    "fla.layers": (),
    "fla.models": (),
    "fla.modules.convolution": ("causal_conv1d", "causal_conv1d_update"),
    "fla.modules.l2norm": ("l2norm",),
    "fla.ops.gated_delta_rule": ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    "emerging_optimizers": (),
    "apex.normalization.fused_layer_norm": ("FusedLayerNormAffineFunction",),
    "apex.contrib.layer_norm.layer_norm": ("FastLayerNormFN",),
    "fused_layer_norm_cuda": ("forward_affine", "backward_affine"),
    "fast_layer_norm": ("ln_fwd", "ln_bwd"),
}
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
            # writer 侧开关名为 use_cpu_shm_for_gpu_tensors，cpu_shm_mode 只属于队列与 warmup。
            writer = inspect.signature(modules["FileSystemWriterAsync"].__init__)
            if "use_cpu_shm_for_gpu_tensors" not in writer.parameters:
                raise ValueError(
                    "NVRx FileSystemWriterAsync requires use_cpu_shm_for_gpu_tensors support"
                )
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


def check_apex_layer_norm() -> dict:
    """核验原始 LayerNorm 调用接口及后端选择，不执行训练算子。"""
    fused = importlib.import_module("megatron.core.fusions.fused_layer_norm")
    if not fused.HAVE_FUSED_LAYER_NORM or not fused.HAVE_PERSIST_LAYER_NORM:
        raise ValueError("Megatron did not enable Apex fused and persistent LayerNorm")
    backends = importlib.import_module("megatron.core.models.backends")
    selected = backends.LocalSpecProvider().layer_norm(rms_norm=False)
    if selected is not fused.FusedLayerNorm:
        raise ValueError("Original local LayerNorm backend did not select Apex FusedLayerNorm")
    signatures = {}
    for name, module_name, arguments in (
        (
            "FusedLayerNormAffineFunction",
            "apex.normalization.fused_layer_norm",
            (None, None, None, None, (1024,), 1e-5),
        ),
        ("FastLayerNormFN", "apex.contrib.layer_norm.layer_norm", (None, None, None, None, 1e-5)),
    ):
        api = getattr(importlib.import_module(module_name), name)
        if getattr(fused, name, None) is not api or not callable(getattr(api, "apply", None)):
            raise ValueError(f"Original Apex LayerNorm API mismatch: {name}")
        forward = inspect.signature(api.forward)
        if "memory_efficient" in inspect.getfullargspec(api.forward).args:
            arguments += (False,)
        forward.bind(*arguments)
        signatures[name] = str(forward)
    return {
        "backend": f"{selected.__module__}.{selected.__qualname__}",
        "fused_enabled": True,
        "persistent_enabled": True,
        "signatures": signatures,
    }


def check_training_dependencies(bundle: Path) -> dict:
    """Build 与 Pod 都检查真实导入，不能仅凭包名或被吞掉的 ImportError 放行。"""
    packages = {}
    for name, expected in TRAINING_PACKAGES.items():
        installed = importlib.metadata.version(name)
        if installed != expected:
            raise ValueError(
                f"Functional dependency version mismatch: {name}: {installed} != {expected}"
            )
        packages[name] = installed
    imports = {}
    for name, symbols in TRAINING_IMPORTS.items():
        module = importlib.import_module(name)
        for symbol in symbols:
            if not callable(getattr(module, symbol, None)):
                raise ValueError(f"Missing functional dependency API: {name}.{symbol}")
        if name == "fla.modules.l2norm":
            # 安装完整 FLA 后仍须保留已有的最后一维参数适配。
            inspect.signature(module.l2norm).bind(object(), dim=-1, eps=1e-6)
        imports[name] = str(Path(module.__file__).resolve())
    optimizer = importlib.import_module("megatron.core.optimizer.emerging_optimizers")
    if not optimizer.HAVE_EMERGING_OPTIMIZERS:
        raise ValueError(
            "Megatron did not enable emerging-optimizers; required original imports failed"
        )
    layer_norm = check_apex_layer_norm()
    installation = json.loads((bundle / "functional-package-install.json").read_text())
    if installation.get("version") != "1" or not isinstance(installation.get("install"), list):
        raise ValueError("Invalid functional package installation report")
    # 原始 pip 报告随 bundle 一并封存；没有条目只表示本次解析前已满足，不能伪造下载来源。
    sources = {
        item["metadata"]["name"].lower().replace("_", "-"): item["download_info"]
        for item in installation["install"]
        if item["metadata"]["name"].lower().replace("_", "-") in TRAINING_PACKAGES
    }
    return {
        "packages": packages,
        "imports": imports,
        "download_sources": sources,
        "satisfied_before_install": sorted(packages.keys() - sources.keys()),
        "emerging_optimizers_enabled": True,
        "apex_layer_norm": layer_norm,
    }


def check_postprocessing(root: Path) -> dict:
    """在全新 uv 子进程中先导入原比较器，避免训练依赖的导入顺序掩盖 namespace 缺陷。"""
    probe = f'''
import hashlib
import json
import sys
import sysconfig
from pathlib import Path
startup = Path(sysconfig.get_path("purelib")) / {NAMESPACE_PTH!r}
if startup.read_text() != {NAMESPACE_BOOTSTRAP!r}:
    raise ValueError("Functional namespace startup mismatch")
from tests.functional_tests.python_test_utils import common
import google.protobuf
from importlib.metadata import version
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
if not Path(google.protobuf.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
    raise ValueError("Functional protobuf must come from the verified venv")
if google.protobuf.__version__ != version("protobuf"):
    raise ValueError("Functional protobuf metadata and imported version mismatch")
event = Event(step=1, summary=Summary(value=[Summary.Value(tag="lm loss", simple_value=1.25)]))
if Event.FromString(event.SerializeToString()).summary.value[0].simple_value != 1.25:
    raise ValueError("Functional protobuf event roundtrip failed")
print(json.dumps({{
    "python": sys.executable,
    "prefix": sys.prefix,
    "protobuf_version": google.protobuf.__version__,
    "protobuf_path": google.protobuf.__file__,
    "common_path": common.__file__,
    "namespace_startup_sha256": hashlib.sha256(startup.read_bytes()).hexdigest(),
    "event_roundtrip": True,
}}))
'''
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1",
        "UV_PROJECT_ENVIRONMENT": sys.prefix,
        "UV_PYTHON": sys.executable,
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_NO_SYNC": "1",
    }
    command = ["uv", "run", "--no-sync", "python"]
    report = json.loads(
        subprocess.check_output(command + ["-c", probe], cwd=root, env=env, text=True, timeout=60)
    )
    if (
        Path(report["common_path"]).resolve()
        != (root / "tests/functional_tests/python_test_utils/common.py").resolve()
        or Path(report["prefix"]).resolve() != Path(sys.prefix).resolve()
    ):
        raise ValueError("Functional postprocessing source or interpreter mismatch")
    # 原提取器也须独立启动成功；不以父进程预热后的导入结果代替。
    extractor = (
        root / "tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py"
    )
    subprocess.run(
        command + [str(extractor), "--help"],
        cwd=root,
        env=env,
        check=True,
        timeout=60,
        stdout=subprocess.DEVNULL,
    )
    return report


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
    startup = Path(site) / NAMESPACE_PTH
    # 只新增到本次 functional venv，不删除系统 google 目录或其他分发包的启动钩子。
    with startup.open("x") as stream:
        stream.write(NAMESPACE_BOOTSTRAP)


COMPILER_PROBE = '''
import json
import os
from torch._inductor import config
print(json.dumps({
    "environment": {
        "TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL": os.environ.get("TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"),
    },
    "quiesce_async_compile_pool": config.quiesce_async_compile_pool,
    "compile_threads": config.compile_threads,
    "compile_threads_override": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
}))
'''


def check_compiler(manifest: dict) -> dict:
    """在全新解释器导入配置，验证环境开关确实生效；不执行算子或创建编译池。"""
    if "compiler_environment" not in manifest["config"]:
        return {}
    settings = manifest["config"]["compiler_environment"]
    if (
        manifest.get("test_kind") != "functional"
        or manifest["config"].get("functional_mode") != "regression"
        or settings != {"TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL": "0"}
    ):
        raise ValueError("Invalid functional compiler environment policy")
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        raise ValueError("Functional compiler policy requires default parallel compilation")
    report = json.loads(
        subprocess.check_output(
            [sys.executable, "-I", "-c", COMPILER_PROBE],
            env={**os.environ, **settings},
            text=True,
            timeout=60,
        )
    )
    if (
        not isinstance(report, dict)
        or report.get("environment") != settings
        or report.get("quiesce_async_compile_pool") is not False
        or type(report.get("compile_threads")) is not int
        or report["compile_threads"] <= 1
        or "compile_threads_override" not in report
        or report["compile_threads_override"] is not None
    ):
        raise ValueError(
            "Functional compiler environment was not applied with parallel compilation"
        )
    return report


def check(bundle: Path, manifest: dict, output: Path):
    compiler = check_compiler(manifest)
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
            "protobuf",
            "uv",
            "pytest",
            "numpy",
            "pydantic",
            "transformer-engine",
        )
    }
    postprocessing = check_postprocessing(Path(os.environ["PPU_SOURCE_ROOT"]).resolve())
    for name in ("tensorboard", "yaml", "click", "ppu_functional_pytest"):
        importlib.import_module(name)
    nvrx = check_nvrx(nvrx_requirements(manifest))
    if nvrx:
        packages["nvidia-resiliency-ext"] = nvrx["version"]
    training_dependencies = check_training_dependencies(bundle)
    packages.update(training_dependencies["packages"])
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
                "training_dependencies": training_dependencies,
                "postprocessing": postprocessing,
                "compiler": compiler,
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
