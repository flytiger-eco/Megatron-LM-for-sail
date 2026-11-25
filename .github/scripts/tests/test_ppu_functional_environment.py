"""CPU checks for functional tools, device attestation and shared bundle integrity."""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_ppu_functional import ROOT, SCRIPTS, functional

environment = functional.module_from_path(
    "ppu_functional_environment", SCRIPTS / "ppu_functional_environment.py"
)


@pytest.fixture
def tool_environment(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    site = tmp_path / "site"
    site.mkdir()
    local = tmp_path / "usr-local-bin"
    local.mkdir()
    (bundle / "control").mkdir()
    (bundle / "control/ppu_functional_pytest.py").write_text("PLUGIN = True\n")
    installation = {
        "version": "1",
        "install": [
            {
                "metadata": {"name": name.replace("-", "_"), "version": version},
                "download_info": {"url": f"https://pkg.flytiger-eco.com/{name}.whl"},
            }
            for name, version in environment.TRAINING_PACKAGES.items()
        ],
    }
    (bundle / "functional-package-install.json").write_text(json.dumps(installation))
    envsubst = tmp_path / "envsubst"
    envsubst.write_text("Mock GNU gettext")
    content = b"Attested mock yq"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(environment, "YQ", {"fixture": ("mock_arch", digest)})
    monkeypatch.setattr(environment.platform, "machine", lambda: "fixture")
    prefix = tmp_path / "runtime"
    (prefix / "bin").mkdir(parents=True)
    (bundle / "functional-tools").mkdir()
    tool_files = {}
    for name, payload in (("yq", content), ("envsubst", envsubst.read_bytes())):
        (bundle / "functional-tools" / name).write_bytes(payload)
        (prefix / "bin" / name).write_bytes(payload)
        tool_files[f"functional-tools/{name}"] = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (local / "yq").write_bytes(content)
    (local / "yq").chmod(0o755)
    (bundle / "docker-build.json").write_text(
        json.dumps({"schema_version": 2, "files": tool_files})
    )
    (site / "ppu_functional_pytest.py").write_bytes(
        (bundle / "control/ppu_functional_pytest.py").read_bytes()
    )
    (site / environment.NAMESPACE_PTH).write_text(environment.NAMESPACE_BOOTSTRAP)
    monkeypatch.setattr(
        environment,
        "sys",
        SimpleNamespace(executable=sys.executable, prefix=str(prefix), version=sys.version),
    )
    calls = []
    monkeypatch.setattr(environment.subprocess, "run", lambda args, **kwargs: calls.append(args))
    versions = {
        "bash": "GNU bash, version 5.2.21",
        str(prefix / "bin/yq"): "yq (https://github.com/mikefarah/yq/) version v4.47.2",
        "/usr/local/bin/yq": "yq (https://github.com/mikefarah/yq/) version v4.47.2",
        "envsubst": "envsubst (GNU gettext-runtime) 0.21",
        "uv": "uv 0.13.0",
        "ft_launcher": "Original launcher help",
    }
    monkeypatch.setattr(
        environment,
        "command_output",
        lambda args: str(site) if args[0] == sys.executable else versions[args[0]],
    )
    monkeypatch.setattr(
        environment.importlib.metadata,
        "version",
        lambda name: environment.TRAINING_PACKAGES.get(name, "fixture"),
    )
    training_modules = {
        name: SimpleNamespace(
            __file__=str(site / name.replace(".", "/") / "__init__.py"),
            **{
                symbol: lambda *a, **k: pytest.fail(
                    "Dependency preflight must not run training kernels"
                )
                for symbol in symbols
            },
        )
        for name, symbols in environment.TRAINING_IMPORTS.items()
    }
    training_modules["fla.modules.l2norm"].l2norm = lambda x, eps=1e-6, *, dim=-1: x
    training_modules["megatron.core.optimizer.emerging_optimizers"] = SimpleNamespace(
        HAVE_EMERGING_OPTIMIZERS=True
    )

    fused = SimpleNamespace(
        HAVE_FUSED_LAYER_NORM=True,
        HAVE_PERSIST_LAYER_NORM=True,
        FusedLayerNorm=type("FusedLayerNorm", (), {}),
    )
    fused_name = "megatron.core.fusions.fused_layer_norm"
    training_modules[fused_name] = fused
    monkeypatch.setitem(sys.modules, fused_name, fused)
    path = ROOT / "megatron/core/models/backends.py"
    provider = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "LocalSpecProvider"
    )
    selector = next(
        node
        for node in provider.body
        if isinstance(node, ast.FunctionDef) and node.name == "layer_norm"
    )
    namespace = {"LayerNormBuilder": type, "WrappedTorchNorm": object}
    exec(compile(ast.Module(body=[selector], type_ignores=[]), str(path), "exec"), namespace)
    training_modules["megatron.core.models.backends"] = SimpleNamespace(
        LocalSpecProvider=type("LocalSpecProvider", (), {"layer_norm": namespace["layer_norm"]})
    )
    original_import = environment.importlib.import_module
    monkeypatch.setattr(
        environment.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace()
            if name in ("tensorboard", "yaml", "click", "ppu_functional_pytest")
            else training_modules[name] if name in training_modules else original_import(name)
        ),
    )
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 8)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        environment,
        "Path",
        lambda value: (
            local
            if value == "/usr/local/bin"
            else tmp_path / "empty-load" if value == "/tmp/checkpoints" else Path(value)
        ),
    )
    monkeypatch.setenv("PIP_INDEX_URL", "https://packages.invalid/simple")
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("PPU_SOURCE_ROOT", str(ROOT))
    monkeypatch.setattr(environment, "check_postprocessing", lambda root: {"event_roundtrip": True})
    manifest = {
        "functional": {"case": {"launcher": "torchrun"}},
        "identity": {"run_id": "123"},
        "config": {"image": "fixed"},
    }
    delivery = {
        "runtime_image_digest": "ghcr.io/flytiger-eco/megatron-lm-for-sail/ppu-ci@sha256:"
        + "e" * 64,
        "base_image_digest": functional.IMAGE.rsplit(":", 1)[0] + "@sha256:" + "f" * 64,
        "environment_digest": "a" * 64,
    }
    (bundle / "runtime-image.json").write_text(json.dumps(delivery))
    for name, field in (
        ("PPU_IMAGE_DIGEST", "runtime_image_digest"),
        ("PPU_BASE_IMAGE_DIGEST", "base_image_digest"),
        ("PPU_ENVIRONMENT_DIGEST", "environment_digest"),
    ):
        monkeypatch.setenv(name, delivery[field])
    return bundle, site, manifest, calls, versions, cuda


def compiler_manifest():
    return {
        "test_kind": "functional",
        "config": {
            "functional_mode": "regression",
            "compiler_environment": dict(functional.COMPILER_ENVIRONMENT),
        },
    }


def compiler_report():
    return {
        "environment": dict(functional.COMPILER_ENVIRONMENT),
        "quiesce_async_compile_pool": False,
        "compile_threads": 8,
        "compile_threads_override": None,
    }


@pytest.mark.parametrize("value", ["", "1", "8"])
def test_compiler_check_rejects_inherited_thread_override(monkeypatch, value):
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", value)
    monkeypatch.setattr(
        environment.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("Must not start a probe"),
    )
    with pytest.raises(ValueError, match="default parallel compilation"):
        environment.check_compiler(compiler_manifest())


def test_compiler_probe_uses_real_isolated_cpu_interpreter(tmp_path, monkeypatch):
    """Real -I cold start; a minimal config stand-in checks import timing without
    posing as vendor Torch acceptance.
    """
    prefix = tmp_path / "compiler-venv"
    venv.EnvBuilder(with_pip=False).create(prefix)
    python = prefix / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    package = site / "torch/_inductor"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text(
        "import os, sys\nassert sys.flags.isolated\n"
        "quiesce_async_compile_pool = os.environ.get('TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL', '1') == '1'\n"
        "compile_threads = 8\n"
    )
    monkeypatch.setattr(environment.sys, "executable", str(python))
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setenv("TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL", "1")
    assert environment.check_compiler(compiler_manifest()) == compiler_report()
    assert os.environ["TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"] == "1"


@pytest.mark.parametrize("launcher", ["torchrun", "ft_launcher"])
def test_prepare_attests_prebuilt_tools_without_installing_dependencies(
    tool_environment, launcher, monkeypatch
):
    bundle, site, manifest, calls, _, _ = tool_environment
    invoked = []
    command_output = environment.command_output

    def command(args):
        invoked.append(args)
        return command_output(args)

    monkeypatch.setattr(environment, "command_output", command)
    manifest["functional"]["case"]["launcher"] = launcher
    environment.prepare(bundle)
    assert (site / "ppu_functional_pytest.py").read_bytes() == (
        bundle / "control/ppu_functional_pytest.py"
    ).read_bytes()
    assert calls == []
    environment.check(bundle, manifest, bundle / "reports")
    report = json.loads((bundle / "reports/functional-environment.json").read_text())
    assert (site / environment.NAMESPACE_PTH).read_text() == environment.NAMESPACE_BOOTSTRAP
    assert report["postprocessing"] == {"event_roundtrip": True}
    assert "protobuf" in report["packages"]
    assert report["identity"] == manifest["identity"] and report["devices"] == 8
    delivery = json.loads((bundle / "runtime-image.json").read_text())
    assert {field: report[field] for field in delivery} == delivery
    assert report["training_dependencies"]["packages"] == environment.TRAINING_PACKAGES
    assert environment.TRAINING_PACKAGES.items() <= report["packages"].items()
    assert report["training_dependencies"]["emerging_optimizers_enabled"]
    layer_norm = report["training_dependencies"]["apex_layer_norm"]
    assert layer_norm["fused_enabled"] and layer_norm["persistent_enabled"]
    assert layer_norm["backend"].endswith(".FusedLayerNorm")
    assert report["packages"]["nvidia-resiliency-ext"] == "0.6.0"
    assert report["launchers"] == [launcher]
    assert (["ft_launcher", "--help"] in invoked) == (launcher == "ft_launcher")


def test_missing_ft_launcher_fails_before_training(tool_environment, monkeypatch):
    bundle, _, manifest, _, _, _ = tool_environment
    manifest["functional"]["case"]["launcher"] = "ft_launcher"
    command_output = environment.command_output

    def command(args):
        if args == ["ft_launcher", "--help"]:
            raise FileNotFoundError("ft_launcher")
        return command_output(args)

    monkeypatch.setattr(environment, "command_output", command)
    with pytest.raises(FileNotFoundError, match="ft_launcher"):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.mark.parametrize("damage", ["missing", "modified", "not_executable", "symlink"])
def test_prepare_rejects_invalid_upstream_yq_without_repairing(tool_environment, damage):
    bundle, _, manifest, calls, _, _ = tool_environment
    local = environment.Path("/usr/local/bin") / "yq"
    if damage == "missing":
        local.unlink()
    elif damage == "modified":
        local.write_bytes(b"Changed yq")
    elif damage == "not_executable":
        local.chmod(0o644)
    else:
        local.unlink()
        local.symlink_to(bundle / "functional-tools/yq")
    before = (local.is_symlink(), local.exists(), local.read_bytes() if local.exists() else None)
    with pytest.raises(ValueError, match="Upstream yq path"):
        environment.prepare(bundle)
    assert calls == []
    assert before == (
        local.is_symlink(),
        local.exists(),
        local.read_bytes() if local.exists() else None,
    )
    assert not (bundle / "reports").exists()


@pytest.mark.parametrize(
    "variable", ["PPU_IMAGE_DIGEST", "PPU_BASE_IMAGE_DIGEST", "PPU_ENVIRONMENT_DIGEST"]
)
def test_functional_image_evidence_requires_job_identities(tool_environment, monkeypatch, variable):
    bundle, _, manifest, _, _, _ = tool_environment
    values = {
        "runtime_image_digest": "execution",
        "base_image_digest": "base",
        "environment_digest": "metadata",
    }
    (bundle / "runtime-image.json").write_text(json.dumps(values))
    monkeypatch.setenv("PPU_IMAGE_DIGEST", values["runtime_image_digest"])
    monkeypatch.setenv("PPU_BASE_IMAGE_DIGEST", values["base_image_digest"])
    monkeypatch.setenv("PPU_ENVIRONMENT_DIGEST", values["environment_digest"])
    monkeypatch.setenv(variable, "mismatch")
    with pytest.raises(ValueError, match="image handoff mismatch"):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.mark.parametrize("name", environment.TRAINING_PACKAGES)
@pytest.mark.parametrize("damage", ["missing", "version"])
def test_training_dependency_missing_or_wrong_version_fails(
    tool_environment, monkeypatch, name, damage
):
    bundle, _, manifest, _, _, _ = tool_environment
    environment.prepare(bundle)
    version = environment.importlib.metadata.version

    def installed(package):
        if package != name:
            return version(package)
        if damage == "missing":
            raise environment.importlib.metadata.PackageNotFoundError(package)
        return "999.9.9"

    monkeypatch.setattr(environment.importlib.metadata, "version", installed)
    with pytest.raises((ValueError, environment.importlib.metadata.PackageNotFoundError)):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.mark.parametrize("name", environment.TRAINING_IMPORTS)
def test_training_native_import_failure_is_not_ignored(tool_environment, monkeypatch, name):
    bundle, _, manifest, _, _, _ = tool_environment
    environment.prepare(bundle)
    original_import = environment.importlib.import_module

    def broken_import(module):
        if module == name:
            raise ImportError(f"Missing native dependency: {name}")
        return original_import(module)

    monkeypatch.setattr(environment.importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="Missing native dependency"):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.fixture
def cold_postprocessing(tmp_path, monkeypatch):
    """A real standalone interpreter emulates the system legacy
    namespace hook without parent warm-up.
    """
    prefix = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(prefix)
    python = prefix / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    parent_site = Path(sysconfig.get_path("purelib"))
    # Copy only the CPU protobuf; other installed CPU test dependencies come from the path
    # and no parent startup hook runs.
    shutil.copytree(parent_site / "google/protobuf", site / "google/protobuf")
    system = tmp_path / "system-site"
    (system / "google/shared_sibling").mkdir(parents=True)
    (system / "google/shared_sibling/__init__.py").write_text("VALUE = 'retained'\n")
    legacy = system / "google_legacy-nspkg.pth"
    legacy.write_text(
        "import sys, importlib.util, importlib.machinery; "
        f"spec = importlib.machinery.PathFinder.find_spec('google', [{str(system)!r}]); "
        "sys.modules.setdefault('google', importlib.util.module_from_spec(spec))\n"
    )
    legacy_bytes = legacy.read_bytes()
    (site / "90-cpu-dependencies.pth").write_text(
        f"{parent_site}\nimport site; site.addsitedir({str(system)!r})\n"
    )
    startup = site / environment.NAMESPACE_PTH
    startup.write_text(environment.NAMESPACE_BOOTSTRAP)
    # uv is a real tool; the test fails without it instead of mocking or
    # skipping to fake startup validation.
    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    assert Path(uv).is_file(), "CPU cold-start regression requires installing uv"
    monkeypatch.setenv("PATH", str(Path(uv).parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(environment.sys, "prefix", str(prefix))
    monkeypatch.setattr(environment.sys, "executable", str(python))
    yield prefix, site, startup, legacy
    assert legacy.read_bytes() == legacy_bytes
    assert (system / "google/shared_sibling/__init__.py").read_text() == "VALUE = 'retained'\n"


def test_real_uv_extractor_reads_tensorboard_in_fresh_process(cold_postprocessing, tmp_path):
    prefix, _, _, _ = cold_postprocessing
    logs = tmp_path / "tensorboard"
    actual = tmp_path / "actual.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1",
        "UV_PROJECT_ENVIRONMENT": str(prefix),
        "UV_PYTHON": sys.executable,
        "UV_PYTHON_DOWNLOADS": "never",
    }
    command = ["uv", "run", "--no-sync", "python"]
    writer = f'''
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
writer = EventFileWriter({str(logs)!r})
for step in range(1, 6):
    writer.add_event(Event(step=step, summary=Summary(value=[
        Summary.Value(tag="lm loss", simple_value=1.25),
        Summary.Value(tag="num-zeros", simple_value=0.0),
    ])))
writer.close()
'''
    subprocess.run(command + ["-c", writer], cwd=ROOT, env=env, check=True, timeout=30)
    extractor = (
        ROOT / "tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py"
    )
    subprocess.run(
        command
        + [
            str(extractor),
            "--logs-dir",
            str(logs),
            "--train-iters",
            "5",
            "--output-path",
            str(actual),
            "--is-normal-test",
            "--step-size",
            "1",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=30,
        capture_output=True,
    )
    report = json.loads(actual.read_text())
    assert set(report) == {"lm loss", "num-zeros"}
    for metric, value in (("lm loss", 1.25), ("num-zeros", 0.0)):
        assert report[metric]["values"] == {str(step): value for step in range(1, 6)}
        assert report[metric]["value_precision"] == "full"


@pytest.mark.parametrize("damage", ["missing", "changed", "protobuf", "version", "source"])
def test_real_cold_postprocessing_rejects_damaged_environment(
    cold_postprocessing, tmp_path, damage
):
    _, site, startup, _ = cold_postprocessing
    root = ROOT
    if damage == "missing":
        startup.rename(startup.with_suffix(".absent"))
    elif damage == "changed":
        startup.write_text(environment.NAMESPACE_BOOTSTRAP + "# changed\n")
    elif damage == "protobuf":
        (site / "google/protobuf/__init__.py").write_text("raise ImportError('broken protobuf')\n")
    elif damage == "version":
        with (site / "google/protobuf/__init__.py").open("a") as stream:
            stream.write("\n__version__ = 'wrong-version'\n")
    else:
        root = tmp_path / "wrong-source"
        root.mkdir()
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        environment.check_postprocessing(root)


@pytest.mark.parametrize(
    "damage", ["bash", "yq", "envsubst", "devices", "nodes", "stale_checkpoint", "tool_digest"]
)
def test_invalid_runtime_environment_fails_before_training(tool_environment, monkeypatch, damage):
    bundle, _, manifest, _, versions, cuda = tool_environment
    environment.prepare(bundle)
    if damage in ("bash", "yq", "envsubst"):
        versions["/usr/local/bin/yq" if damage == "yq" else damage] = "Unsupported version"
    elif damage == "devices":
        cuda.device_count = lambda: 4
    elif damage == "nodes":
        monkeypatch.setenv("NNODES", "2")
    elif damage == "stale_checkpoint":
        empty = bundle.parent / "empty-load"
        empty.mkdir()
        (empty / "old-state").write_text("Old checkpoint")
    else:
        (bundle / "functional-tools/yq").write_bytes(b"changed")
    with pytest.raises(ValueError):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()
