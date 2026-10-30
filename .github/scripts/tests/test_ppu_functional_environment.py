"""CPU checks for functional tools, device attestation and shared bundle integrity."""

import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_ppu_functional import ROOT, SCRIPTS, functional
from test_ppu_upstream import bundle_digest, ci, sealed_bundle

environment = functional.module_from_path(
    "ppu_functional_environment", SCRIPTS / "ppu_functional_environment.py"
)
FUNCTIONAL_PAYLOADS = [
    "functional-tools/yq",
    "functional-tools/envsubst",
    "functional-environment.json",
    "functional-package-install.json",
    "installed-packages.json",
    "control/ppu_functional_run.sh",
    *[
        f"control/ppu_functional{suffix}.py"
        for suffix in ("", "_results", "_runtime", "_environment", "_pytest", "_snapshot", "_hf")
    ],
]


@pytest.fixture
def functional_bundle(sealed_bundle):
    plan, bundle = sealed_bundle
    for root in (plan, bundle):
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["test_kind"] = "functional"
        (root / "manifest.json").write_text(json.dumps(manifest))
    for name in FUNCTIONAL_PAYLOADS:
        path = bundle / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("Functional payload fixture\n")
        if name.startswith("control/"):
            shutil.copy2(path, plan / name)
    ci.seal_bundle(bundle)
    return plan, bundle


def test_functional_reuses_trusted_bundle_identity(functional_bundle):
    plan, bundle = functional_bundle
    ci.verify_bundle(bundle, plan, bundle_digest(bundle))


@pytest.mark.parametrize("name", FUNCTIONAL_PAYLOADS)
@pytest.mark.parametrize("damage", ["missing", "empty", "modified"])
def test_functional_bundle_rejects_incomplete_or_changed_payload(functional_bundle, name, damage):
    plan, bundle = functional_bundle
    digest = bundle_digest(bundle)
    path = bundle / name
    if damage == "missing":
        path.rename(path.with_name(path.name + ".absent"))
    else:
        path.write_text("" if damage == "empty" else "Changed payload")
    with pytest.raises(ValueError):
        ci.verify_bundle(bundle, plan, digest)
    if damage != "modified":
        with pytest.raises(ValueError, match="Missing or empty functional"):
            ci.seal_bundle(bundle)


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
    monkeypatch.setattr(environment.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))
    monkeypatch.setattr(
        environment.shutil,
        "which",
        lambda name: str(envsubst) if name == "envsubst" else str(local / name),
    )
    calls = []
    monkeypatch.setattr(environment.subprocess, "run", lambda args, **kwargs: calls.append(args))
    versions = {
        "bash": "GNU bash, version 5.2.21",
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
        lambda name: (
            "0.6.0"
            if name == "nvidia-resiliency-ext"
            else environment.TRAINING_PACKAGES.get(name, "fixture")
        ),
    )
    training_modules = {
        name: SimpleNamespace(
            __file__=str(site / name.replace(".", "/") / "__init__.py"),
            **{
                symbol: lambda *a, **k: pytest.fail("依赖预检不应执行训练算子")
                for symbol in symbols
            },
        )
        for name, symbols in environment.TRAINING_IMPORTS.items()
    }
    training_modules["fla.modules.l2norm"].l2norm = lambda x, eps=1e-6, *, dim=-1: x
    training_modules["megatron.core.optimizer.emerging_optimizers"] = SimpleNamespace(
        HAVE_EMERGING_OPTIMIZERS=True
    )

    class AffineFunction:
        @staticmethod
        def forward(ctx, x, weight, bias, normalized_shape, eps, memory_efficient=False):
            pytest.fail("预检不应执行 LayerNorm 算子")

        @staticmethod
        def apply(*args):
            pytest.fail("预检不应执行 LayerNorm autograd")

    class FastFunction(AffineFunction):
        @staticmethod
        def forward(ctx, x, weight, bias, eps, memory_efficient=False):
            pytest.fail("预检不应执行 persistent LayerNorm 算子")

    training_modules["apex.normalization.fused_layer_norm"].FusedLayerNormAffineFunction = (
        AffineFunction
    )
    training_modules["apex.contrib.layer_norm.layer_norm"].FastLayerNormFN = FastFunction
    fused = SimpleNamespace(
        HAVE_FUSED_LAYER_NORM=True,
        HAVE_PERSIST_LAYER_NORM=True,
        FusedLayerNorm=type("FusedLayerNorm", (), {}),
        FusedLayerNormAffineFunction=AffineFunction,
        FastLayerNormFN=FastFunction,
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
        "functional": {"case": {"launcher": "torchrun", "nvrx_checkpoint": []}},
        "identity": {"run_id": "123"},
        "config": {"image": "fixed"},
    }
    return bundle, site, manifest, calls, versions, cuda


@pytest.fixture
def checkpoint_api(tool_environment, monkeypatch):
    class Queue:
        def __init__(self, persistent=True, cpu_shm_mode=False):
            pytest.fail("预检不应创建 checkpoint 队列")

        @classmethod
        def warmup_persistent_caller(cls, rank, cpu_priority=10, io_priority=3, cpu_shm_mode=False):
            pytest.fail("预检不应启动 checkpoint worker")

    class Writer:
        def __init__(self, path, cpu_shm_mode=False):
            pytest.fail("预检不应创建 checkpoint writer")

    def request(async_fn, async_fn_args, finalize_fns, async_fn_kwargs=None, preload_fn=None):
        pytest.fail("预检不应执行 async request")

    def results_queue(mp_mode="spawn"):
        pytest.fail("预检不应创建多进程资源")

    modules = {
        "core": SimpleNamespace(AsyncCallsQueue=Queue, AsyncRequest=request),
        "cached_metadata_filesystem_reader": SimpleNamespace(CachedMetadataFileSystemReader=object),
        "filesystem_async": SimpleNamespace(
            FileSystemWriterAsync=Writer, _results_queue=None, get_write_results_queue=results_queue
        ),
        "state_dict_saver": SimpleNamespace(
            CheckpointMetadataCache=object,
            save_state_dict_async_finalize=object,
            save_state_dict_async_plan=object,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(
            sys.modules, f"nvidia_resiliency_ext.checkpointing.async_ckpt.{name}", module
        )
    # 执行当前源码中的原始导入函数，避免 mock 自行定义一套不完整的 API 清单。
    path = ROOT / "megatron/core/dist_checkpointing/strategies/torch.py"
    getter = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_async_strategy"
    )
    namespace = {}
    exec(compile(ast.Module(body=[getter], type_ignores=[]), str(path), "exec"), namespace)
    requested = []

    def get_strategy(strategy):
        requested.append(strategy)
        return namespace["get_async_strategy"](strategy)

    original_import = environment.importlib.import_module
    monkeypatch.setattr(
        environment.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(get_async_strategy=get_strategy)
            if name == "megatron.core.dist_checkpointing.strategies.torch"
            else original_import(name)
        ),
    )
    return modules, requested


@pytest.mark.parametrize("launcher", ["torchrun", "ft_launcher"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_prepare_attests_tools_and_installs_selected_dependencies(
    tool_environment, checkpoint_api, launcher, checkpoint
):
    bundle, site, manifest, calls, _, _ = tool_environment
    manifest["functional"]["case"]["launcher"] = launcher
    if checkpoint:
        manifest["functional"]["case"]["nvrx_checkpoint"] = [
            {"section": "MODEL_ARGS", "persistent_worker": True, "cpu_shm": False}
        ]
    environment.prepare(bundle, manifest)
    assert (site / "ppu_functional_pytest.py").read_bytes() == (
        bundle / "control/ppu_functional_pytest.py"
    ).read_bytes()
    assert any("nvidia-resiliency-ext==0.6.0" in command for command in calls) == (
        launcher == "ft_launcher" or checkpoint
    )
    assert all(
        command[command.index("--index-url") + 1] == "https://packages.invalid/simple"
        for command in calls
    )
    environment.check(bundle, manifest, bundle / "reports")
    report = json.loads((bundle / "reports/functional-environment.json").read_text())
    assert (site / environment.NAMESPACE_PTH).read_text() == environment.NAMESPACE_BOOTSTRAP
    assert report["postprocessing"] == {"event_roundtrip": True}
    assert "protobuf" in report["packages"]
    assert report["identity"] == manifest["identity"] and report["devices"] == 8
    assert report["training_dependencies"]["packages"] == environment.TRAINING_PACKAGES
    assert environment.TRAINING_PACKAGES.items() <= report["packages"].items()
    assert report["training_dependencies"]["emerging_optimizers_enabled"]
    layer_norm = report["training_dependencies"]["apex_layer_norm"]
    assert layer_norm["fused_enabled"] and layer_norm["persistent_enabled"]
    assert layer_norm["backend"].endswith(".FusedLayerNorm")
    assert set(layer_norm["signatures"]) == {"FusedLayerNormAffineFunction", "FastLayerNormFN"}
    assert ("nvidia-resiliency-ext" in report["packages"]) == (
        launcher == "ft_launcher" or checkpoint
    )
    assert checkpoint_api[1] == (["nvrx"] if checkpoint else [])
    if checkpoint:
        assert (
            report["nvrx"]["requirements"]["case"]["checkpoint"]
            == manifest["functional"]["case"]["nvrx_checkpoint"]
        )
        assert "mp_mode" in report["nvrx"]["signatures"]["get_write_results_queue"]


@pytest.mark.parametrize(
    "module,symbol",
    [
        ("core", "AsyncCallsQueue"),
        ("core", "AsyncRequest"),
        ("cached_metadata_filesystem_reader", "CachedMetadataFileSystemReader"),
        ("filesystem_async", "FileSystemWriterAsync"),
        ("filesystem_async", "_results_queue"),
        ("filesystem_async", "get_write_results_queue"),
        ("state_dict_saver", "CheckpointMetadataCache"),
        ("state_dict_saver", "save_state_dict_async_finalize"),
        ("state_dict_saver", "save_state_dict_async_plan"),
    ],
)
def test_missing_original_checkpoint_import_fails_preflight(
    tool_environment, checkpoint_api, monkeypatch, module, symbol
):
    bundle, _, manifest, _, _, _ = tool_environment
    manifest["functional"]["case"]["nvrx_checkpoint"] = [
        {"section": "MODEL_ARGS_2", "persistent_worker": True, "cpu_shm": False}
    ]
    environment.prepare(bundle, manifest)
    monkeypatch.delattr(checkpoint_api[0][module], symbol)
    with pytest.raises(ModuleNotFoundError, match="compatible.*nvidia-resiliency-ext"):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.mark.parametrize(
    "damage", ["version", "request", "queue", "warmup", "mp_mode", "cpu_shm", "writer", "launcher"]
)
def test_incompatible_checkpoint_api_fails_before_training(
    tool_environment, checkpoint_api, monkeypatch, damage
):
    bundle, _, manifest, _, _, _ = tool_environment
    manifest["functional"]["case"]["nvrx_checkpoint"] = [
        {
            "section": "MODEL_ARGS",
            "persistent_worker": True,
            "cpu_shm": damage in ("cpu_shm", "writer"),
        }
    ]
    environment.prepare(bundle, manifest)
    modules, _ = checkpoint_api
    if damage == "version":
        monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "0.5.0")
    elif damage == "request":
        modules["core"].AsyncRequest = lambda async_fn, async_fn_args, finalize_fns: None
    elif damage == "queue":
        monkeypatch.setattr(modules["core"].AsyncCallsQueue, "__init__", lambda self: None)
    elif damage == "warmup":
        monkeypatch.setattr(
            modules["core"].AsyncCallsQueue, "warmup_persistent_caller", lambda rank: None
        )
    elif damage == "mp_mode":
        modules["filesystem_async"].get_write_results_queue = lambda: None
    elif damage == "cpu_shm":
        monkeypatch.setattr(
            modules["core"].AsyncCallsQueue,
            "warmup_persistent_caller",
            lambda rank, cpu_priority=10, io_priority=3: None,
        )
    elif damage == "writer":
        monkeypatch.setattr(
            modules["filesystem_async"].FileSystemWriterAsync, "__init__", lambda self, path: None
        )
    else:
        manifest["functional"]["case"]["launcher"] = "ft_launcher"
        monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    with pytest.raises((ValueError, TypeError)):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.mark.parametrize("kind", [None, "unit", "functional"])
def test_training_packages_match_build_constraints(tmp_path, monkeypatch, kind):
    script = (SCRIPTS / "ppu_ci_build.sh").read_text()
    block = next(
        block
        for block in re.findall(r"<<'PY'\n(.*?)\nPY", script, re.S)
        if "native_requirements =" in block
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "fixed-runtime")
    bundle = tmp_path / ".ppu-ci"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"test_kind": kind} if kind else {}))
    exec(compile(block, "functional-build-requirements", "exec"), {})
    requirements = (bundle / "ppu-requirements.txt").read_text().splitlines()
    constraints = (bundle / "constraints.txt").read_text().splitlines()
    functional_only = {"apex", "flash-linear-attention", "emerging-optimizers"}
    assert environment.TRAINING_PACKAGES["apex"] == "24.4.1+v0.1.0.ppu2.2.0"
    for name, version in environment.TRAINING_PACKAGES.items():
        expected = int(kind == "functional" or name not in functional_only)
        assert requirements.count(f"{name}=={version}") == expected
        assert constraints.count(f"{name}=={version}") == expected
    assert len(requirements) == (9 if kind == "functional" else 6)
    assert all(f"{name}==fixed-runtime" in constraints for name in ("torch", "triton"))
    assert 'functional_pip_args=(--report .ppu-ci/functional-package-install.json)' in script


@pytest.mark.parametrize("name", environment.TRAINING_PACKAGES)
@pytest.mark.parametrize("damage", ["missing", "version"])
def test_training_dependency_missing_or_wrong_version_fails(
    tool_environment, monkeypatch, name, damage
):
    bundle, _, manifest, _, _, _ = tool_environment
    environment.prepare(bundle, manifest)
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
    environment.prepare(bundle, manifest)
    original_import = environment.importlib.import_module

    def broken_import(module):
        if module == name:
            raise ImportError(f"Missing native dependency: {name}")
        return original_import(module)

    monkeypatch.setattr(environment.importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="Missing native dependency"):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.mark.parametrize(
    "name,symbol",
    [
        (name, symbol)
        for name, symbols in environment.TRAINING_IMPORTS.items()
        for symbol in symbols
    ],
)
def test_training_dependency_api_is_required(tool_environment, monkeypatch, name, symbol):
    bundle, _, _, _, _, _ = tool_environment
    module = environment.importlib.import_module(name)
    monkeypatch.setattr(module, symbol, None)
    with pytest.raises(ValueError, match="Missing functional dependency API"):
        environment.check_training_dependencies(bundle)


@pytest.mark.parametrize(
    "damage", ["fused_disabled", "persistent_disabled", "fallback", "identity", "apply", "forward"]
)
def test_apex_backend_or_api_failure_prevents_training(tool_environment, monkeypatch, damage):
    bundle, _, manifest, _, _, _ = tool_environment
    environment.prepare(bundle, manifest)
    fused = environment.importlib.import_module("megatron.core.fusions.fused_layer_norm")
    if damage == "fused_disabled":
        monkeypatch.setattr(fused, "HAVE_FUSED_LAYER_NORM", False)
    elif damage == "persistent_disabled":
        monkeypatch.setattr(fused, "HAVE_PERSIST_LAYER_NORM", False)
    elif damage == "fallback":
        backends = environment.importlib.import_module("megatron.core.models.backends")
        monkeypatch.setattr(backends.LocalSpecProvider, "layer_norm", lambda *a, **k: object)
    elif damage == "identity":
        monkeypatch.setattr(fused, "FastLayerNormFN", object)
    elif damage == "apply":
        monkeypatch.setattr(fused.FastLayerNormFN, "apply", None)
    else:
        monkeypatch.setattr(fused.FusedLayerNormAffineFunction, "forward", lambda ctx, x: None)
    with pytest.raises((ValueError, TypeError)):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


@pytest.mark.parametrize("memory_efficient", [False, True])
def test_apex_forward_signature_matches_original_call(
    tool_environment, monkeypatch, memory_efficient
):
    bundle, _, _, _, _, _ = tool_environment
    fused = environment.importlib.import_module("megatron.core.fusions.fused_layer_norm")
    if not memory_efficient:
        monkeypatch.setattr(
            fused.FusedLayerNormAffineFunction,
            "forward",
            lambda ctx, x, weight, bias, normalized_shape, eps: None,
        )
        monkeypatch.setattr(
            fused.FastLayerNormFN, "forward", lambda ctx, x, weight, bias, eps: None
        )
    report = environment.check_training_dependencies(bundle)["apex_layer_norm"]
    assert all(
        ("memory_efficient" in signature) == memory_efficient
        for signature in report["signatures"].values()
    )


@pytest.mark.parametrize("damage", ["l2norm", "optimizer", "report_missing", "report_invalid"])
def test_dependency_compatibility_and_provenance_are_required(
    tool_environment, monkeypatch, damage
):
    bundle, _, _, _, _, _ = tool_environment
    if damage == "l2norm":
        module = environment.importlib.import_module("fla.modules.l2norm")
        monkeypatch.setattr(module, "l2norm", lambda x, eps=1e-6: x)
    elif damage == "optimizer":
        module = environment.importlib.import_module("megatron.core.optimizer.emerging_optimizers")
        monkeypatch.setattr(module, "HAVE_EMERGING_OPTIMIZERS", False)
    elif damage == "report_missing":
        (bundle / "functional-package-install.json").rename(bundle / "absent-report.json")
    else:
        (bundle / "functional-package-install.json").write_text('{}')
    with pytest.raises((TypeError, ValueError, FileNotFoundError)):
        environment.check_training_dependencies(bundle)


def test_dependency_report_preserves_actual_sources(tool_environment):
    bundle, _, _, _, _, _ = tool_environment
    path = bundle / "functional-package-install.json"
    installation = json.loads(path.read_text())
    installation["install"] = [
        item for item in installation["install"] if item["metadata"]["name"] != "pla"
    ]
    installation["install"].append({"metadata": {"name": "tensorboard"}})
    path.write_text(json.dumps(installation))
    result = environment.check_training_dependencies(bundle)
    assert result["packages"] == environment.TRAINING_PACKAGES
    assert set(result["imports"]) == set(environment.TRAINING_IMPORTS)
    assert result["satisfied_before_install"] == ["pla"]
    assert result["download_sources"] == {
        item["metadata"]["name"].replace("_", "-"): item["download_info"]
        for item in installation["install"]
        if item["metadata"]["name"] != "tensorboard"
    }


@pytest.fixture
def cold_postprocessing(tmp_path, monkeypatch):
    """真实独立解释器模拟系统 legacy namespace 钩子，不借用父进程预热。"""
    prefix = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(prefix)
    python = prefix / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    parent_site = Path(sysconfig.get_path("purelib"))
    # 只复制 CPU protobuf；其余已安装的 CPU 测试依赖由路径提供，不执行父环境启动钩子。
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
    # uv 是真实工具；没有工具时测试失败，不用 mock 或跳过来冒充启动验证。
    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    assert Path(uv).is_file(), "CPU 冷启动回归需要安装 uv"
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


def test_real_cold_uv_imports_original_common_and_extractor(cold_postprocessing):
    prefix, _, startup, _ = cold_postprocessing
    report = environment.check_postprocessing(ROOT)
    assert report["event_roundtrip"]
    assert report["prefix"] == str(prefix)
    assert Path(report["protobuf_path"]).is_relative_to(prefix)
    assert report["namespace_startup_sha256"] == hashlib.sha256(startup.read_bytes()).hexdigest()
    assert (
        Path(report["common_path"]) == ROOT / "tests/functional_tests/python_test_utils/common.py"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", "import google.shared_sibling; print(google.shared_sibling.VALUE)"],
        text=True,
    )
    assert output.strip() == "retained"


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


def test_legacy_namespace_reproduces_missing_protobuf_without_bootstrap(cold_postprocessing):
    _, _, startup, _ = cold_postprocessing
    startup.rename(startup.with_suffix(".absent"))
    # 必须保留 CI 的 safe-path；否则 Python 插入 cwd 会触发 namespace 重算并掩盖缺陷。
    probe = subprocess.run(
        [sys.executable, "-c", "from tests.functional_tests.python_test_utils import common"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1"},
        text=True,
        capture_output=True,
    )
    assert probe.returncode != 0
    assert "No module named 'google.protobuf'" in probe.stderr


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


@pytest.mark.parametrize("damage", ["prefix", "common_path", "extractor"])
def test_postprocessing_route_identity_and_extractor_failure(monkeypatch, damage):
    calls = []
    report = {
        "prefix": sys.prefix,
        "common_path": str(ROOT / "tests/functional_tests/python_test_utils/common.py"),
    }
    if damage != "extractor":
        report[damage] = "/untrusted"

    def output(command, **kwargs):
        calls.append(command)
        assert command[:4] == ["uv", "run", "--no-sync", "python"]
        assert kwargs["cwd"] == ROOT
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == sys.prefix
        assert kwargs["env"]["UV_PYTHON"] == sys.executable
        assert kwargs["env"]["UV_PYTHON_DOWNLOADS"] == "never"
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert kwargs["env"]["PYTHONPATH"] == str(ROOT)
        return json.dumps(report)

    def run(command, **kwargs):
        calls.append(command)
        assert command[:4] == ["uv", "run", "--no-sync", "python"]
        assert command[4:] == [
            str(
                ROOT
                / "tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py"
            ),
            "--help",
        ]
        assert kwargs["check"] and kwargs["timeout"] == 60
        raise subprocess.CalledProcessError(23, command)

    monkeypatch.setattr(environment.subprocess, "check_output", output)
    monkeypatch.setattr(environment.subprocess, "run", run)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        environment.check_postprocessing(ROOT)
    assert len(calls) == (2 if damage == "extractor" else 1)


def test_postprocessing_failure_prevents_success_report(tool_environment, monkeypatch):
    bundle, _, manifest, _, _, _ = tool_environment
    environment.prepare(bundle, manifest)

    def fail(root):
        raise subprocess.CalledProcessError(1, ["uv", "run"])

    monkeypatch.setattr(environment, "check_postprocessing", fail)
    with pytest.raises(subprocess.CalledProcessError):
        environment.check(bundle, manifest, bundle / "reports")
    assert not (bundle / "reports/functional-environment.json").exists()


def test_namespace_prepare_refuses_to_overwrite_existing_file(tool_environment):
    bundle, site, manifest, _, _, _ = tool_environment
    startup = site / environment.NAMESPACE_PTH
    startup.write_text("existing content\n")
    with pytest.raises(FileExistsError):
        environment.prepare(bundle, manifest)
    assert startup.read_text() == "existing content\n"


def test_checkpoint_requirement_is_not_optional():
    with pytest.raises(KeyError, match="nvrx_checkpoint"):
        environment.nvrx_requirements({"functional": {"case": {"launcher": "torchrun"}}})


def test_tool_download_digest_mismatch_prevents_package_installation(tool_environment, monkeypatch):
    bundle, _, manifest, calls, _, _ = tool_environment
    monkeypatch.setattr(
        environment.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"different")
    )
    with pytest.raises(ValueError, match="digest"):
        environment.prepare(bundle, manifest)
    assert not calls


@pytest.mark.parametrize(
    "damage", ["bash", "yq", "envsubst", "devices", "nodes", "stale_checkpoint", "tool_digest"]
)
def test_invalid_runtime_environment_fails_before_training(tool_environment, monkeypatch, damage):
    bundle, _, manifest, _, versions, cuda = tool_environment
    environment.prepare(bundle, manifest)
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
