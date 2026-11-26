"""Check shared PPU dependencies and attest functional tools in the fixed image."""

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

NAMESPACE_PTH = "00-ppu-functional-google-namespace.pth"
# Build a complete PEP 420 namespace first so inherited nspkg startup hooks cannot pin
# google to the system directory.
NAMESPACE_BOOTSTRAP = "import importlib; importlib.invalidate_caches(); import google\n"
TRAINING_PACKAGES = {
    "nvidia-resiliency-ext": "0.6.0",
    "pla": "1.0.0+v0.1.0.ppu2.2.0",
    "mamba-ssm": "2.3.1+v0.1.0.ppu2.2.0",
    "causal-conv1d": "1.7.0+v0.1.0.ppu2.2.0",
    "flash-linear-attention": "0.4.1+v0.1.0.ppu2.2.0",
    "fla-core": "0.4.1+v0.1.0.ppu2.2.0",
    "emerging-optimizers": "0.3.0",
    "apex": "24.4.1+v0.1.0.ppu2.2.0",
}
TRAINING_IMPORTS = {
    "nvidia_resiliency_ext": (),
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


def check_apex_layer_norm() -> dict:
    """Verify Apex backend selection without running training kernels."""
    fused = importlib.import_module("megatron.core.ops.norm.fused_layer_norm")
    if not fused.HAVE_FUSED_LAYER_NORM or not fused.HAVE_PERSIST_LAYER_NORM:
        raise ValueError("Megatron did not enable Apex fused and persistent LayerNorm")
    backends = importlib.import_module("megatron.core.models.backends")
    selected = backends.LocalSpecProvider().layer_norm(rms_norm=False)
    if selected is not fused.FusedLayerNorm:
        raise ValueError("Original local LayerNorm backend did not select Apex FusedLayerNorm")
    return {
        "backend": f"{selected.__module__}.{selected.__qualname__}",
        "fused_enabled": True,
        "persistent_enabled": True,
    }


def check_training_dependencies(bundle: Path) -> dict:
    """Build and the Pod both check real imports; a package name or a swallowed
    ImportError is not enough to pass.
    """
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
            # The existing last-dimension argument adaptation must stay after
            # installing the full FLA package.
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
    # The raw pip report is sealed with the bundle; no entry only means it was already satisfied,
    # never a fabricated download source.
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
    """Import the original comparators first in a fresh uv subprocess so training import order
    cannot mask namespace defects.
    """
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
    # The original extractor must also start on its own; imports warmed up by
    # the parent process do not count.
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


def prepare(bundle: Path):
    """Attest the existing Dockerfile environment; never acquire software here."""
    metadata = json.loads((bundle / "docker-build.json").read_text())
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 2:
        raise ValueError("Functional preparation requires the current execution image schema")
    for name in ("yq", "envsubst"):
        path = bundle / "functional-tools" / name
        expected = metadata["files"][f"functional-tools/{name}"]
        installed = Path(sys.prefix) / "bin/yq" if name == "yq" else Path("/usr/bin/envsubst")
        if (
            path.is_symlink()
            or path.stat().st_size != expected["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
            or installed.read_bytes() != path.read_bytes()
        ):
            raise ValueError("Restored functional tool does not match the Dockerfile export")
    # Verify the build-time copy; never patch or install tools during PPU execution.
    upstream_yq = Path("/usr/local/bin") / "yq"
    if (
        upstream_yq.is_symlink()
        or not upstream_yq.is_file()
        or not os.access(upstream_yq, os.X_OK)
        or upstream_yq.read_bytes() != (bundle / "functional-tools/yq").read_bytes()
    ):
        raise ValueError("Upstream yq path does not match the Dockerfile export")
    site = Path(
        command_output(
            [sys.executable, "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))']
        )
    )
    if (site / NAMESPACE_PTH).read_text() != NAMESPACE_BOOTSTRAP:
        raise ValueError("Missing Dockerfile namespace startup hook")
    if (site / "ppu_functional_pytest.py").read_bytes() != (
        bundle / "control/ppu_functional_pytest.py"
    ).read_bytes():
        raise ValueError("Restored pytest policy does not match trusted control")


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
    """Import the config in a fresh interpreter to verify the environment switch takes
    effect; no kernels or compile pools.
    """
    settings = manifest["config"].get("compiler_environment")
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
    prepare(bundle)
    record = json.loads((bundle / "runtime-image.json").read_text())
    delivery = {}
    for field, variable in (
        ("runtime_image_digest", "PPU_IMAGE_DIGEST"),
        ("base_image_digest", "PPU_BASE_IMAGE_DIGEST"),
        ("environment_digest", "PPU_ENVIRONMENT_DIGEST"),
    ):
        if not os.environ.get(variable) or record[field] != os.environ[variable]:
            raise ValueError("Functional execution image handoff mismatch")
        delivery[field] = record[field]
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
    launchers = sorted({case["launcher"] for case in manifest["functional"].values()})
    if "ft_launcher" in launchers:
        command_output(["ft_launcher", "--help"])
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
                **delivery,
                "python": sys.version,
                "packages": packages,
                "launchers": launchers,
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
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "ppu_functional_control", Path(__file__).with_name("ppu_functional.py")
    )
    control = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(control)
        manifest = control.load_manifest(args.bundle / "manifest.json")
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    if manifest.get("test_kind") != "functional" or not manifest["config"]["build"]:
        raise ValueError("Functional environment setup requires a regression build")
    check(args.bundle, manifest, args.output or args.bundle)


if __name__ == "__main__":
    main()
