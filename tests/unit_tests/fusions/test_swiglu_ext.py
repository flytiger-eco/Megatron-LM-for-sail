# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Test shared SwiGLU extension contracts and weighted, dense, and bias paths.

Optimization is enabled by default; MEGATRON_ENABLE_PPU_SWIGLU_OPT=0 disables it.
Only sm89 inputs may use the extension. Shared contracts cover imports, reloads,
device guards, loading failures, error propagation, and independent input logging.
Both paths exercise real entry-point forward/backward calls, kernel spies, independent
FP64 references, and bitwise fallback assertions. Subprocess probes isolate default,
explicit-on, and disabled states from import caches. Accuracy checks allow dtype rounding
and input-magnitude-dependent reduction error while retaining fixed FP64 oracle tolerances.
Strict pointwise no-worse counts are diagnostic, not acceptance criteria.

Dense eligibility uses row thresholds and alignment/divisibility, not exact-shape equality.
Existing exact-h weighted kernel branches are outside this dense-only source check.
"""

import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.fusions._extensions import fused_bias_swiglu_ext as cudac

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SM89 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)
requires_sm89 = pytest.mark.skipif(not _SM89, reason="CUDA C implementation supports sm89 only")


# Shared helpers: switch isolation, reloads, tolerances, kernel spies, and subprocess probes.


@pytest.fixture(autouse=True)
def _enabled_gate(monkeypatch):
    # Enable gate/kernel unit tests explicitly; subprocesses cover default, off, and invalid values.
    monkeypatch.setattr(cudac, "_OPT_ENABLED", True)


@pytest.fixture
def _entry_cudac():
    """Reload with env=1 to install both wrappers, restoring the initial state afterward.

    Reload defines originals before install captures them, preventing wrappers from
    being mistaken for originals across repeated reloads.
    """
    saved = os.environ.get("MEGATRON_ENABLE_PPU_SWIGLU_OPT")
    os.environ["MEGATRON_ENABLE_PPU_SWIGLU_OPT"] = "1"
    import megatron.core.fusions.fused_bias_swiglu as entry

    importlib.reload(entry)
    try:
        yield entry
    finally:
        if saved is None:
            os.environ.pop("MEGATRON_ENABLE_PPU_SWIGLU_OPT", None)
        else:
            os.environ["MEGATRON_ENABLE_PPU_SWIGLU_OPT"] = saved
        importlib.reload(entry)


def _tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=3e-3, atol=3e-3)
    if dtype == torch.float32:
        return dict(rtol=2e-5, atol=2e-6)
    return dict(rtol=2e-2, atol=2e-2)  # bfloat16


def _install_spy(monkeypatch):
    """Spy on swiglu_cuda to prove native execution and return (calls, raw)."""
    ext = cudac._get_extension()
    raw = ext.swiglu_cuda
    calls = {"fwd": 0, "bwd": 0}

    def spy(y, g, bias, weights):
        calls["bwd" if g is not None else "fwd"] += 1
        return raw(y, g, bias, weights)

    monkeypatch.setattr(ext, "swiglu_cuda", spy)
    return calls, raw


def _run_probe(script, enable):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("MEGATRON_SWIGLU_INPUT_LOG_DIR", None)
    if enable is None:
        env.pop("MEGATRON_ENABLE_PPU_SWIGLU_OPT", None)
    else:
        env["MEGATRON_ENABLE_PPU_SWIGLU_OPT"] = "1" if enable else "0"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    facts = {}
    for line in proc.stdout.splitlines():
        if line.startswith("FACT "):
            _, key, value = line.split()
            facts[key] = value == "True"
    return facts


# Shared contracts: configuration, dual sm89 guards, loading/execution errors, and input logging.


_CONFIG_PROBE = r'''
import importlib
import inspect
import os
import torch
import megatron.core

cuda_initialized_before = torch.cuda.is_initialized()


def forbidden(*args, **kwargs):
    raise AssertionError("Import, install, and reload must not initialize CUDA or load the extension")


torch.cuda._lazy_init = forbidden
import megatron.core.fusions.fused_bias_swiglu as entry
from megatron.core.fusions._extensions import fused_bias_swiglu_ext as ext

assert torch.cuda.is_initialized() == cuda_initialized_before
assert ext._extension is None
assert entry._swiglu_ext is ext
assert ext.__package__ == "megatron.core.fusions._extensions"
assert ext._CSRC_DIR == ext.Path(entry.__file__).resolve().parent / "csrc"
assert all((ext._CSRC_DIR / name).is_file() for name in ("swiglu.cpp", "swiglu.cu"))
ext._get_extension = forbidden
ext._is_sm89 = forbidden
names = ("weighted_swiglu", "weighted_swiglu_back", "bias_swiglu_impl", "weighted_bias_swiglu_impl")
enabled = os.environ.get(ext._OPT_ENV, "1") == "1"
assert ext._OPT_ENABLED == enabled
assert hasattr(entry, "_orig_weighted_swiglu") == enabled
assert hasattr(entry, "_orig_bias_swiglu_impl") == enabled
if enabled:
    originals = entry._swiglu_ext_state["originals"]
else:
    originals = {name: getattr(entry, name) for name in names}
for name in names:
    assert inspect.signature(getattr(entry, name)) == inspect.signature(originals[name])
installed = {name: getattr(entry, name) for name in names}
ext.install(vars(entry))
assert all(getattr(entry, name) is fn for name, fn in installed.items())

# CPU inputs fall back before querying the device or loading the extension.
x = torch.ones(256, 2048, dtype=torch.bfloat16)
w = torch.ones(256, 1)
assert ext.cuda_extension(x, w) is None
assert ext.cuda_extension_plain(x) is None

# Unwrap repeated installs; preserve new definitions after reload instead of restoring old ones.
for value in ("0", "1", "0", "1"):
    os.environ[ext._OPT_ENV] = value
    ext.install(vars(entry))
    assert hasattr(entry, "_orig_weighted_swiglu") == (value == "1")
    if value == "0":
        assert all(getattr(entry, name) is fn for name, fn in originals.items())
    before = entry.bias_swiglu_impl
    importlib.reload(entry)
    assert entry.bias_swiglu_impl is not before
    assert hasattr(entry, "_orig_bias_swiglu_impl") == (value == "1")
    if value == "1":
        originals = entry._swiglu_ext_state["originals"]
        assert entry.bias_swiglu_impl.__wrapped__ is originals["bias_swiglu_impl"]
    else:
        originals = {name: getattr(entry, name) for name in names}
        assert not hasattr(entry, "_swiglu_ext_state")
assert ext._extension is None
assert torch.cuda.is_initialized() == cuda_initialized_before
'''


@pytest.mark.parametrize("value", [None, "1", "0", "", "true", "2"])
def test_configuration_import_and_reload(value):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("MEGATRON_SWIGLU_INPUT_LOG_DIR", None)
    if value is None:
        env.pop(cudac._OPT_ENV, None)
    else:
        env[cudac._OPT_ENV] = value
    proc = subprocess.run(
        [sys.executable, "-c", _CONFIG_PROBE],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@requires_sm89
@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0)])
@pytest.mark.parametrize("enabled", [False, True])
def test_both_gates_enforce_sm89_and_switch(monkeypatch, capability, enabled):
    y = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(256, 1, device="cuda")
    g = torch.randn(256, 1024, device="cuda", dtype=y.dtype)
    backend = object()
    loader = Mock(return_value=backend)
    device_query = Mock(return_value=capability)
    monkeypatch.setattr(cudac, "_OPT_ENABLED", enabled)
    monkeypatch.setattr(cudac, "_extension_error", None)
    monkeypatch.setattr(cudac, "_load_extension", loader)
    monkeypatch.setattr(torch.cuda, "get_device_capability", device_query)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        Mock(side_effect=AssertionError("Eligibility must not depend on device names")),
    )
    expected = backend if enabled and capability == (8, 9) else None
    assert cudac.cuda_extension(y, w) is expected
    assert cudac.cuda_extension(y, w, grad=g) is expected
    assert cudac.cuda_extension_plain(y) is expected
    assert cudac.cuda_extension_plain(y, grad=g) is expected
    assert loader.call_count == (4 if expected is backend else 0)
    assert device_query.call_count == (4 if enabled else 0)
    for call in device_query.call_args_list:
        assert call.args == (y.device,)


def test_sm89_is_queried_per_input_device(monkeypatch):
    devices = {torch.device("cuda:0"): (8, 0), torch.device("cuda:1"): (8, 9)}
    query = Mock(side_effect=devices.__getitem__)
    monkeypatch.setattr(torch.cuda, "get_device_capability", query)
    assert not cudac._is_sm89(torch.device("cuda:0"))
    assert cudac._is_sm89(torch.device("cuda:1"))
    assert not cudac._is_sm89(torch.device("cuda:0"))
    assert query.call_count == 3


def test_native_host_checks_sm89_before_launch():
    source = (cudac._CSRC_DIR / "swiglu.cu").read_text()
    guard = "properties->major == 8 && properties->minor == 9"
    assert guard in source
    assert "std::strstr(properties->name" not in source
    assert source.index("CUDAGuard guard(input.device())") < source.index(guard)
    assert source.index(guard) < source.index("auto y = input.contiguous()")


@requires_sm89
def test_loading_failure_warns_once_and_falls_back(monkeypatch):
    y = torch.ones(256, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(256, 1, device="cuda")
    loader = Mock(side_effect=RuntimeError("simulated build failure"))
    monkeypatch.setattr(cudac, "_extension", None)
    monkeypatch.setattr(cudac, "_extension_error", None)
    monkeypatch.setattr(cudac, "_get_extension", loader)
    with pytest.warns(RuntimeWarning, match="simulated build failure") as captured:
        assert cudac.cuda_extension_plain(y) is None
        assert cudac.cuda_extension_plain(y) is None
        assert cudac.cuda_extension(y, w) is None
    assert len(captured) == 1
    assert loader.call_count == 1


def test_kernel_execution_error_propagates(monkeypatch):
    kernel = Mock(side_effect=RuntimeError("simulated kernel error"))
    backend = SimpleNamespace(swiglu_cuda=kernel)
    monkeypatch.setattr(cudac, "cuda_extension", lambda *args, **kwargs: backend)
    monkeypatch.setattr(cudac, "cuda_extension_plain", lambda *args, **kwargs: backend)
    original = Mock(
        side_effect=AssertionError("Kernel errors must propagate rather than trigger fallback")
    )
    forward, backward = cudac._weighted_wrappers(original, original)
    y, w, g = torch.ones(2, 16), torch.ones(2, 1), torch.ones(2, 8)
    with pytest.raises(RuntimeError, match="simulated kernel error"):
        forward(y, w)
    with pytest.raises(RuntimeError, match="simulated kernel error"):
        backward(g, y, w)
    with pytest.raises(RuntimeError, match="simulated kernel error"):
        cudac._dense_wrapper(original)(y, None)
    original.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_logging_is_independent_and_records_each_call_once(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv(cudac._OPT_ENV, "1" if enabled else "0")
    monkeypatch.setenv("MEGATRON_SWIGLU_INPUT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    logger = cudac._InputLogger(str(tmp_path), enabled)
    monkeypatch.setattr(cudac, "_InputLogger", lambda *args: logger)
    dense = Mock(side_effect=lambda x, *args: x)
    weighted = Mock(side_effect=lambda x, *args: x)
    originals = dict(
        weighted_swiglu=Mock(),
        weighted_swiglu_back=Mock(),
        bias_swiglu_impl=dense,
        weighted_bias_swiglu_impl=weighted,
    )
    namespace = dict(originals)
    cudac.install(namespace)
    installed = namespace["bias_swiglu_impl"]
    cudac.install(namespace)
    assert namespace["bias_swiglu_impl"] is installed
    assert logger.fh is None
    x = torch.ones(2, 16)
    try:
        assert namespace["bias_swiglu_impl"](x, None) is x
        assert namespace["bias_swiglu_impl"](x.requires_grad_(), None) is x
        assert namespace["weighted_bias_swiglu_impl"](x, None, torch.ones(2, 1)) is x
        assert dense.call_count == 2 and weighted.call_count == 1
        assert logger.calls == 3
        records = [
            json.loads(line)
            for line in (tmp_path / "swiglu_inputs_rank00007.jsonl").read_text().splitlines()
        ]
        assert len(records) == 3
        header, first, last = records
        assert header["event"] == "session_start"
        assert header["cudac_enabled"] == enabled
        assert header["rank"] == 7 and header["world_size"] == 16
        assert header["rank_source"] == "RANK"
        assert first["call_index"] == 1 and last["call_index"] == 3
        assert first["fn"] == "bias_swiglu_impl"
        assert last["fn"] == "weighted_bias_swiglu_impl"
        monkeypatch.delenv("MEGATRON_SWIGLU_INPUT_LOG_DIR")
        monkeypatch.setenv(cudac._OPT_ENV, "0")
        cudac.install(namespace)
        assert all(namespace[name] is fn for name, fn in originals.items())
    finally:
        if logger.fh is not None:
            logger.fh.close()


def test_logging_failure_does_not_hide_operator_error(tmp_path):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("Occupy the intended logging directory with a file")
    logger = cudac._InputLogger(str(blocked), False)
    original = Mock(side_effect=ValueError("operator error"))
    dense, _ = cudac._logging_wrappers(original, original, logger)
    with pytest.raises(ValueError, match="operator error"):
        dense(torch.ones(2, 16), None)
    original.assert_called_once()


@requires_sm89
def test_non_sm89_entry_falls_back_without_loading(_entry_cudac, monkeypatch):
    entry = _entry_cudac
    monkeypatch.setattr(cudac, "_is_sm89", lambda device: False)
    loader = Mock(side_effect=AssertionError("Non-sm89 devices must not load the kernel"))
    monkeypatch.setattr(cudac, "_load_extension", loader)
    x = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(2048, device="cuda", dtype=x.dtype, requires_grad=True)
    g = torch.randn(256, 1024, device="cuda", dtype=x.dtype)
    out = entry.bias_swiglu_impl(x, bias)
    dx, db = torch.autograd.grad(out, (x, bias), g)
    ref = entry._orig_bias_swiglu_impl(x, bias)
    rx, rb = torch.autograd.grad(ref, (x, bias), g)
    for actual, expected in ((out, ref), (dx, rx), (db, rb)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    w = torch.ones(256, 1, device="cuda")
    with torch.no_grad():
        actual = entry.weighted_bias_swiglu_impl(x, None, w)
        expected = entry._orig_weighted_swiglu(x, w)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        dx, dw = entry.weighted_swiglu_back(g, x, w)
        rx, rw = entry._orig_weighted_swiglu_back(g, x, w)
        torch.testing.assert_close(dx, rx, rtol=0, atol=0)
        torch.testing.assert_close(dw, rw, rtol=0, atol=0)
    loader.assert_not_called()


# Weighted path: independent formulas, production patterns, eligibility, and entry-point execution.


def _dw_tol(dtype, wdtype):
    # Retain FP32 dW tolerances; only the h=512 warp path uses FP64 accumulation.
    # Check fixed tolerances and baseline-relative allowances, not strict pointwise no-worse.
    return dict(rtol=2e-4, atol=2e-4) if wdtype == torch.float32 else _tol(dtype)


@torch.no_grad()
def _weighted_reference(y, g, weights, *, round_output=True):
    """FP64 oracle for out=silu(a)*b*w, dx, and last-axis dW reduction with keepdim."""
    yy = y.double()
    a, b = yy.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(a)
    silu = a * sigmoid
    activation = silu * b
    out = activation * weights.double()
    scale = g.double() * weights.double()
    dx = torch.cat((scale * sigmoid * (1 + a * (1 - sigmoid)) * b, scale * silu), dim=-1)
    dw = (activation * g.double()).sum(dim=-1, keepdim=True)
    if round_output:
        return out.to(y.dtype), dx.to(y.dtype), dw.to(weights.dtype)
    return out, dx, dw


# Production MoE patterns: weighted_bias_swiglu_impl(fc1_output[tokens,2h],
# None, permuted_probs[tokens,1]) with BF16/FP16 activations and tokenwise weights, optionally FP32.
_REAL_PATTERNS = [
    (torch.bfloat16, 512, 1024, torch.float32),
    (torch.bfloat16, 256, 2048, torch.float32),
    (torch.bfloat16, 1024, 64, torch.float32),
    (torch.float16, 512, 1024, torch.float32),
    (torch.bfloat16, 512, 1024, torch.bfloat16),
    # h=1408 is a production expert FFN width that uses the general pack path, not h=512 warp.
    # Cover its dW reduction with both FP32 weights (W=float) and matching-dtype weights (W=T).
    (torch.bfloat16, 1024, 1408, torch.float32),
    (torch.bfloat16, 512, 1408, torch.bfloat16),
    # Cover production h=512 warp and BF16 h=768/14336 flat forward/backward paths.
    # Row counts 257/513 exercise partial four-row warp blocks and trailing flat Packs.
    (torch.bfloat16, 257, 512, torch.float32),
    (torch.bfloat16, 257, 512, torch.bfloat16),
    (torch.float16, 513, 512, torch.float32),
    (torch.float16, 513, 512, torch.float16),
    (torch.float16, 513, 1408, torch.float16),
    (torch.bfloat16, 257, 768, torch.float32),
    (torch.bfloat16, 257, 768, torch.bfloat16),
    (torch.bfloat16, 257, 14336, torch.float32),
    (torch.bfloat16, 257, 14336, torch.bfloat16),
]


@requires_sm89
@pytest.mark.parametrize("dtype,rows,h,wdtype", _REAL_PATTERNS)
def test_weighted_real_pattern_hits_kernel(dtype, rows, h, wdtype):
    torch.manual_seed(1234)
    y = torch.randn(rows, 2 * h, device="cuda", dtype=dtype)
    w = torch.rand(rows, 1, device="cuda", dtype=wdtype)
    g = torch.randn(rows, h, device="cuda", dtype=dtype)
    # Eligibility returns the lazily loaded kernel extension.
    ext = cudac.cuda_extension(y, w)
    assert ext is not None
    assert cudac.cuda_extension(y, w, grad=g) is not None
    assert cudac._extension is not None
    exp_out, exp_dx, exp_dw = _weighted_reference(y, g, w)
    # Call native directly: forward returns [0]; backward returns (dx, dw).
    out = ext.swiglu_cuda(y, None, None, w)[0]
    torch.testing.assert_close(out, exp_out, **_tol(dtype))
    dx, dw = ext.swiglu_cuda(y, g, None, w)
    torch.testing.assert_close(dx, exp_dx, **_tol(dtype))
    torch.testing.assert_close(dw, exp_dw, **_dw_tol(dtype, wdtype))


# Ineligible weighted patterns must transparently fall back. Factories return (y, w, g).
def _fb_tiny_rows():
    y = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
    return (
        y,
        torch.rand(8, 1, device="cuda"),
        torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16),
    )


def _fb_fp16_below_min():
    y = torch.randn(256, 2048, device="cuda", dtype=torch.float16)
    return (
        y,
        torch.rand(256, 1, device="cuda"),
        torch.randn(256, 1024, device="cuda", dtype=torch.float16),
    )


def _fb_fp32_activation():
    y = torch.randn(512, 2048, device="cuda", dtype=torch.float32)
    return (
        y,
        torch.rand(512, 1, device="cuda"),
        torch.randn(512, 1024, device="cuda", dtype=torch.float32),
    )


def _fb_offset_view():
    y = torch.randn(512 * 2048 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(512, 2048)
    return (
        y,
        torch.rand(512, 1, device="cuda"),
        torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16),
    )


def _fb_odd_h():
    y = torch.randn(512, 2 * 1023, device="cuda", dtype=torch.bfloat16)
    return (
        y,
        torch.rand(512, 1, device="cuda"),
        torch.randn(512, 1023, device="cuda", dtype=torch.bfloat16),
    )


_FALLBACKS = [
    ("tiny_rows", _fb_tiny_rows),
    ("fp16_below_min", _fb_fp16_below_min),
    ("fp32_activation", _fb_fp32_activation),
    ("offset_view", _fb_offset_view),
    ("odd_h", _fb_odd_h),
]


@requires_sm89
@pytest.mark.parametrize("name,make", _FALLBACKS, ids=[c[0] for c in _FALLBACKS])
def test_weighted_nonreal_gate_rejects(name, make):
    torch.manual_seed(4321)
    y, w, g = make()
    # Reject both forward and backward eligibility without calling native.
    assert cudac.cuda_extension(y, w) is None
    assert cudac.cuda_extension(y, w, grad=g) is None


@requires_sm89
def test_weighted_non_tokenwise_weights_gate_rejected():
    # Non-tokenwise [h] weights fall back to the original broadcasting semantics.
    y = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, device="cuda", dtype=torch.float32)
    assert cudac.cuda_extension(y, w) is None


@requires_sm89
@pytest.mark.parametrize("dtype,rows,h,wdtype", _REAL_PATTERNS)
def test_impl_entry_real_pattern_fwd_hits_bwd_correct(
    _entry_cudac, monkeypatch, dtype, rows, h, wdtype
):
    """Exercise production MoE patterns through weighted_bias_swiglu_impl end to end.

    A swiglu_cuda spy proves native forward/backward execution independently of numerical
    agreement: original torch.compile fallbacks never call it. Also require entry-point
    outputs to equal direct native outputs and compare gradients with an independent FP64 oracle.
    """
    entry = _entry_cudac
    calls, raw = _install_spy(monkeypatch)

    torch.manual_seed(2026)
    x = torch.randn(rows, 2 * h, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.rand(rows, 1, device="cuda", dtype=wdtype, requires_grad=True)
    g = torch.randn(rows, h, device="cuda", dtype=dtype)

    out = entry.weighted_bias_swiglu_impl(x, None, w)
    assert out.shape == (rows, h) and out.dtype == dtype
    assert calls["fwd"] >= 1  # Native forward executed; fallback would leave the count at zero.
    out_data = out.detach()
    xd, wd = x.detach(), w.detach()

    # Compare entry-point and native outputs exactly; raw bypasses the spy counter.
    with torch.no_grad():
        kern = raw(xd, None, None, wd)[0]
    torch.testing.assert_close(out_data, kern, rtol=0, atol=0)

    # Compare forward values with the independent FP64 oracle.
    exp_out, exp_dx, exp_dw = _weighted_reference(xd, g, wd)
    torch.testing.assert_close(out_data, exp_out, **_tol(dtype))

    # Backward traverses autograd, wrapper, and native; compare FP64 gradients and the weight dtype.
    out.backward(g)
    assert calls["bwd"] >= 1  # Native backward executed.
    torch.testing.assert_close(x.grad, exp_dx, **_tol(dtype))
    assert w.grad.dtype == wdtype
    torch.testing.assert_close(w.grad, exp_dw, **_dw_tol(dtype, wdtype))


# Subprocess probes isolate entry-point rebinding and dispatch from pytest import state.
# Emit FACT <key> <bool> for _run_entry_probe, with keys specific to each switch state.
_ENTRY_PROBE = r"""
import os
import torch
import torch.nn.functional as F
import megatron.core.fusions.fused_bias_swiglu as entry
import megatron.core.fusions._extensions.fused_bias_swiglu_ext as cudac

enabled = os.environ.get("MEGATRON_ENABLE_PPU_SWIGLU_OPT", "1") == "1"
dev = "cuda"


def fact(key, value):
    print("FACT %s %s" % (key, bool(value)))


# Enabled mode captures originals and wraps weighted functions; disabled mode does neither.
captured = hasattr(entry, "_orig_weighted_swiglu") and hasattr(entry, "_orig_weighted_swiglu_back")
fact("CAPTURED_ORIG", captured)
fact("WEIGHTED_REBOUND", enabled and captured
     and entry.weighted_swiglu is not entry._orig_weighted_swiglu
     and entry.weighted_swiglu_back is not entry._orig_weighted_swiglu_back)
# Low-level unweighted functions and the weighted impl/Function keep their original bindings.
fact("NONWEIGHTED_UNCAPTURED",
     not hasattr(entry, "_orig_swiglu") and not hasattr(entry, "_orig_bias_swiglu")
     and not hasattr(entry, "_orig_weighted_bias_swiglu_impl")
     and not hasattr(entry, "_orig_WeightedSwiGLUFunction"))

torch.manual_seed(7)
rows, h = 512, 1024
y = torch.randn(rows, 2 * h, device=dev, dtype=torch.bfloat16)
w = torch.rand(rows, 1, device=dev, dtype=torch.float32)
a, b = y.chunk(2, dim=-1)

# Low-level plain swiglu retains the original formula in both modes.
with torch.no_grad():
    got_plain = entry.swiglu(y)
ref_plain = (F.silu(a) * b).to(y.dtype)
fact("PLAIN_SWIGLU_ORIGINAL", torch.allclose(got_plain, ref_plain, rtol=2e-2, atol=2e-2))

if enabled:
    # Call the training entry first in a fresh process to prove lazy native loading
    # through the original impl, global lookup, wrapper, and eligibility guard.
    fact("EXTENSION_INITIALLY_NONE", cudac._extension is None)
    with torch.no_grad():
        out_impl = entry.weighted_bias_swiglu_impl(y, None, w)
    fact("IMPL_LOADED_KERNEL", cudac._extension is not None)
    ext = cudac.cuda_extension(y, w)
    with torch.no_grad():
        out_kern = ext.swiglu_cuda(y, None, None, w)[0]
    fact("IMPL_FWD_EQ_KERNEL", torch.equal(out_impl, out_kern))
    fact("IMPL_SHAPE_DTYPE", tuple(out_impl.shape) == (rows, h) and out_impl.dtype == torch.bfloat16)

    # Verify populated gradients: BF16 dx [rows,2h] and FP32 dw [rows,1].
    x = y.clone().requires_grad_(True)
    ww = w.clone().requires_grad_(True)
    out = entry.weighted_bias_swiglu_impl(x, None, ww)
    out.backward(torch.randn(rows, h, device=dev, dtype=torch.bfloat16))
    fact("IMPL_BACKWARD_GRADS",
         x.grad is not None and tuple(x.grad.shape) == (rows, 2 * h) and x.grad.dtype == torch.bfloat16
         and ww.grad is not None and tuple(ww.grad.shape) == (rows, 1) and ww.grad.dtype == torch.float32)

    # Preserve 3D view/restore semantics with tokenwise weights for flattened rows.
    x3 = torch.randn(4, 128, 2 * h, device=dev, dtype=torch.bfloat16)
    w3 = torch.rand(4 * 128, 1, device=dev, dtype=torch.float32)
    with torch.no_grad():
        out3 = entry.weighted_bias_swiglu_impl(x3, None, w3)
    fact("IMPL_3D_VIEW", tuple(out3.shape) == (4, 128, h))

    # Match the original rejection of nonempty weighted bias.
    try:
        with torch.no_grad():
            entry.weighted_bias_swiglu_impl(y, torch.randn(2 * h, device=dev, dtype=torch.bfloat16), w)
        fact("IMPL_BIAS_RAISES", False)
    except NotImplementedError:
        fact("IMPL_BIAS_RAISES", True)

    # Eligible weighted inputs produce exactly the same outputs as direct native calls.
    with torch.no_grad():
        wrap_out = entry.weighted_swiglu(y, w)
    fact("REAL_WRAPPER_EQ_KERNEL", torch.equal(wrap_out, out_kern))

    # Tiny row counts fall back exactly to the original functions.
    y2 = torch.randn(8, 2048, device=dev, dtype=torch.bfloat16)
    w2 = torch.rand(8, 1, device=dev, dtype=torch.float32)
    fact("NONREAL_GATE_REJECT", cudac.cuda_extension(y2, w2) is None)
    with torch.no_grad():
        wrap2 = entry.weighted_swiglu(y2, w2)
        orig2 = entry._orig_weighted_swiglu(y2, w2)
    fact("NONREAL_WRAPPER_EQ_ORIG", torch.equal(wrap2, orig2))
    g2 = torch.randn(8, 1024, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        dx, dw = entry.weighted_swiglu_back(g2, y2, w2)
        odx, odw = entry._orig_weighted_swiglu_back(g2, y2, w2)
    fact("NONREAL_BACK_EQ_ORIG", torch.equal(dx, odx) and torch.equal(dw, odw))
else:
    # Disabled mode neither loads nor rebinds native, even for eligible production shapes.
    with torch.no_grad():
        out = entry.weighted_swiglu(y, w)
    fact("DISABLED_NO_KERNEL", cudac._extension is None)
    ref = (F.silu(a) * b * w).to(y.dtype)
    fact("DISABLED_WEIGHTED_ORIGINAL", torch.allclose(out, ref, rtol=2e-2, atol=2e-2))
"""


def _run_entry_probe(enable):
    return _run_probe(_ENTRY_PROBE, enable)


@requires_sm89
@pytest.mark.parametrize("enable", [None, True], ids=["default", "explicit_on"])
def test_entry_enabled_routes_weighted_through_kernel(enable):
    f = _run_entry_probe(enable=enable)
    # Original-function capture and rebinding.
    assert f["CAPTURED_ORIG"]
    assert f["WEIGHTED_REBOUND"]
    assert f["NONWEIGHTED_UNCAPTURED"]
    assert f["PLAIN_SWIGLU_ORIGINAL"]
    # The original training impl reaches native through wrappers and preserves outputs and gradients.
    assert f["EXTENSION_INITIALLY_NONE"]
    assert f["IMPL_LOADED_KERNEL"]
    assert f["IMPL_FWD_EQ_KERNEL"]
    assert f["IMPL_SHAPE_DTYPE"]
    assert f["IMPL_BACKWARD_GRADS"]
    assert f["IMPL_3D_VIEW"]
    assert f["IMPL_BIAS_RAISES"]
    # Weighted wrappers use native for eligible shapes and exact original fallback otherwise.
    assert f["REAL_WRAPPER_EQ_KERNEL"]
    assert f["NONREAL_GATE_REJECT"]
    assert f["NONREAL_WRAPPER_EQ_ORIG"]
    assert f["NONREAL_BACK_EQ_ORIG"]


@requires_sm89
def test_entry_disabled_keeps_original():
    f = _run_entry_probe(enable=False)
    assert not f["CAPTURED_ORIG"]
    assert not f["WEIGHTED_REBOUND"]
    assert f["NONWEIGHTED_UNCAPTURED"]
    assert f["PLAIN_SWIGLU_ORIGINAL"]
    assert f["DISABLED_NO_KERNEL"]
    assert f["DISABLED_WEIGHTED_ORIGINAL"]


# Dense/bias paths: shape guard constraints, independent oracles, and entry-point execution.


# Reuse the guard threshold so tests adapt if the minimum row count changes.
_MIN_ROWS_BF16 = cudac._REAL_PATTERN_MIN_ROWS[torch.bfloat16]


# ---------------------------------------------------------------------------
# 1. Source checks: prohibit exact-shape equality in the guard.
# ---------------------------------------------------------------------------
def test_dense_gate_source_has_no_exact_shape_equality():
    """Reject equality against nonzero integer literals in the dense guard.

    Allow zero checks, divisibility, monotonic thresholds, and None checks.
    Negative lookahead (?!0\\b) permits `== 0` but rejects `h == 512` or `last == 12288`.
    """
    src = inspect.getsource(cudac.cuda_extension_plain)
    bad = re.findall(r"==\s*(?!0\b)\d+", src)
    assert not bad, (
        f"cuda_extension_plain contains exact-shape equality {bad}; use monotonic thresholds "
        "and alignment/divisibility, not equality against nonzero constants."
    )
    # Require actual threshold and divisibility checks so an empty guard cannot pass.
    assert re.search(r"rows\s*<\s*min_rows", src), "The guard must enforce a minimum row count"
    assert "%" in src, "The guard must check alignment/divisibility (%)"


# ---------------------------------------------------------------------------
# 2. Behavioral checks.
# ---------------------------------------------------------------------------
@requires_sm89
def test_dense_gate_monotonic_in_rows():
    """For valid h, accept exactly rows >= the dtype-specific minimum."""
    h = 1024  # last=2048 and h%8==0 satisfy alignment.
    rows_sweep = [
        1,
        2,
        8,
        64,
        128,
        _MIN_ROWS_BF16 - 1,
        _MIN_ROWS_BF16,
        _MIN_ROWS_BF16 + 1,
        512,
        1024,
        2048,
    ]
    accept = []
    for r in rows_sweep:
        y = torch.randn(r, 2 * h, device="cuda", dtype=torch.bfloat16)
        accept.append(cudac.cuda_extension_plain(y, None) is not None)
    # Acceptance must be monotonic, without isolated row counts or upper bounds.
    assert accept == sorted(
        accept
    ), f"Row acceptance is not a monotonic suffix: {list(zip(rows_sweep, accept))}"
    # Reject every row count below the minimum and accept every count at or above it.
    i = rows_sweep.index(_MIN_ROWS_BF16)
    assert not any(
        accept[:i]
    ), f"Rows below the minimum must be rejected: {list(zip(rows_sweep[:i], accept[:i]))}"
    assert all(
        accept[i:]
    ), f"Rows at or above the minimum must be accepted: {list(zip(rows_sweep[i:], accept[i:]))}"


@requires_sm89
def test_dense_gate_h_agnostic_no_island():
    """Accept every aligned h at sufficient rows; reject misalignment and odd final dimensions."""
    rows = _MIN_ROWS_BF16
    # Cover production dense widths, weighted h=512, and nonspecialized widths.
    # Any isolated exact-h eligibility would incorrectly reject other aligned widths.
    aligned_h = [64, 128, 256, 512, 1024, 1408, 2048, 4096, 6144, 8704, 9216, 12288]
    for h in aligned_h:
        assert h % 8 == 0
        y = torch.randn(rows, 2 * h, device="cuda", dtype=torch.bfloat16)
        assert (
            cudac.cuda_extension_plain(y, None) is not None
        ), f"Aligned h={h} must be accepted; rejection indicates an isolated exact-h guard"
    # Reject h%8!=0 for 128-bit Pack alignment, not exact-shape specialization.
    for h in [4, 12, 100, 1023]:
        y = torch.randn(rows, 2 * h, device="cuda", dtype=torch.bfloat16)
        assert cudac.cuda_extension_plain(y, None) is None, f"h={h} with h%8!=0 must be rejected"
    # Reject odd final dimensions.
    y = torch.randn(rows, 1023, device="cuda", dtype=torch.bfloat16)
    assert cudac.cuda_extension_plain(y, None) is None


@requires_sm89
def test_dense_gate_scope_and_alignment_rejections():
    """Reject unsupported dtypes, misaligned pointers, noncontiguity, and unfused bias shapes."""
    rows, h = _MIN_ROWS_BF16, 1024
    y = torch.randn(rows, 2 * h, device="cuda", dtype=torch.bfloat16)
    # Establish that the valid plain BF16 input is accepted.
    assert cudac.cuda_extension_plain(y, None) is not None

    # One-dimensional [2h] bias is fused without materialization using bias_stride=0.
    bias = torch.randn(2 * h, device="cuda", dtype=torch.bfloat16)
    assert cudac.cuda_extension_plain(y, bias) is not None
    # Fall back for rowwise [rows,2h], scalar, and feature-broadcast [rows,1] bias.
    assert (
        cudac.cuda_extension_plain(y, torch.randn(rows, 2 * h, device="cuda", dtype=torch.bfloat16))
        is None
    )
    assert (
        cudac.cuda_extension_plain(y, torch.randn((), device="cuda", dtype=torch.bfloat16)) is None
    )
    assert (
        cudac.cuda_extension_plain(y, torch.randn(rows, 1, device="cuda", dtype=torch.bfloat16))
        is None
    )
    # Fall back for a mismatched bias dtype.
    assert (
        cudac.cuda_extension_plain(y, torch.randn(2 * h, device="cuda", dtype=torch.float16))
        is None
    )

    # Dense native dispatch is limited to BF16; FP16/FP32 use the original implementation.
    for dt in (torch.float16, torch.float32):
        yy = torch.randn(rows, 2 * h, device="cuda", dtype=dt)
        assert cudac.cuda_extension_plain(yy, None) is None

    # Reject a contiguous input with a two-byte storage offset and no 16-byte alignment.
    y_off = torch.randn(rows * 2 * h + 1, device="cuda", dtype=torch.bfloat16)[1:].view(rows, 2 * h)
    assert y_off.is_contiguous() and y_off.data_ptr() % 16 != 0
    assert cudac.cuda_extension_plain(y_off, None) is None

    # Reject a noncontiguous transposed view with final stride != 1.
    y_nc = torch.randn(2 * h, rows, device="cuda", dtype=torch.bfloat16).t()
    assert y_nc.shape == (rows, 2 * h) and not y_nc.is_contiguous()
    assert cudac.cuda_extension_plain(y_nc, None) is None


# ===========================================================================
# Dense end-to-end hit/miss, rank routing, switch contracts, and source guard checks.
# ===========================================================================


@torch.no_grad()
def _dense_reference(y, g, bias=None, *, round_output=True):
    """Compute an independent FP64 oracle without early rounding of bias addition.

    Match backward staging for dbias: round elementwise dx to the input dtype, sum
    broadcast dimensions in FP64, then cast to the bias dtype. Do not sum unrounded dx.
    With round_output=False, retain FP64 outputs for error comparison while preserving
    elementwise rounding before the dbias reduction.
    """
    yy = y.double()
    if bias is not None:
        yy = yy + bias.double()
    a, b = yy.chunk(2, dim=-1)
    sig = torch.sigmoid(a)
    silu = a * sig
    out = silu * b
    gs = g.double()
    dx = torch.cat((gs * sig * (1 + a * (1 - sig)) * b, gs * silu), dim=-1)
    dbias = dx.to(y.dtype).double().sum_to_size(bias.shape) if bias is not None else None
    if round_output:
        return out.to(y.dtype), dx.to(y.dtype), None if dbias is None else dbias.to(bias.dtype)
    return out, dx, dbias


# Small production-shaped dense samples cover both 2D and 3D with rows>=256 and h%8==0.
_DENSE_REAL = [(512, 2048), (256, 1, 4096), (1024, 1, 2048), (257, 1024), (257, 1536), (257, 28672)]
_DBIAS_TOL = dict(rtol=2e-2, atol=2e-1)


@requires_sm89
@pytest.mark.parametrize("shape", _DENSE_REAL)
def test_dense_impl_entry_hit_fwd_bwd(_entry_cudac, monkeypatch, shape):
    """Exercise dense entry-point forward/backward with native spies and an FP64 oracle."""
    entry = _entry_cudac
    calls, raw = _install_spy(monkeypatch)
    torch.manual_seed(2026)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(shape[:-1] + (shape[-1] // 2,), device="cuda", dtype=torch.bfloat16)

    out = entry.bias_swiglu_impl(x, None)
    assert calls["fwd"] >= 1  # Native forward executed; fallback would leave the counter at zero.
    assert out.shape == shape[:-1] + (shape[-1] // 2,) and out.dtype == torch.bfloat16

    xd = x.detach()
    with torch.no_grad():  # Compare direct native output without changing spy counts.
        kern = raw(xd.reshape(-1, shape[-1]), None, None, None)[0].reshape(out.shape)
    torch.testing.assert_close(out.detach(), kern, rtol=0, atol=0)

    exp_out, exp_dx, _ = _dense_reference(xd, g)  # Compare forward values with the FP64 oracle.
    torch.testing.assert_close(out.detach(), exp_out, **_tol(torch.bfloat16))

    out.backward(g)  # Backward traverses autograd, the dense Function, and native.
    assert calls["bwd"] >= 1
    torch.testing.assert_close(x.grad, exp_dx, **_tol(torch.bfloat16))


@requires_sm89
@pytest.mark.parametrize("shape", _DENSE_REAL)
def test_dense_impl_entry_hit_fwd_bwd_with_bias(_entry_cudac, monkeypatch, shape):
    """Check [2h] bias native dispatch, FP64 out/dx/dbias, and broadcast reduction of actual dx.

    _DenseSwiGLUCudacFunction runs native forward/backward and preserves the original
    BiasSwiGLUFunction gradient contract: autograd reduces [rows,2h] gradients to [2h].
    """
    entry = _entry_cudac
    calls, raw = _install_spy(monkeypatch)
    torch.manual_seed(2026)
    last = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(last, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(tuple(shape[:-1]) + (last // 2,), device="cuda", dtype=torch.bfloat16)

    out = entry.bias_swiglu_impl(x, bias)
    assert calls["fwd"] >= 1  # [2h] bias reaches native forward.
    assert out.shape == tuple(shape[:-1]) + (last // 2,) and out.dtype == torch.bfloat16
    out.backward(g)
    assert calls["bwd"] >= 1

    xd, bd = x.detach(), bias.detach()
    with torch.no_grad():  # Compare direct native output with the same bias.
        kern = raw(xd.reshape(-1, last), None, bd, None)[0].reshape(out.shape)
    torch.testing.assert_close(out.detach(), kern, rtol=0, atol=0)

    exp_out, exp_dx, exp_dbias = _dense_reference(xd, g, bd)
    torch.testing.assert_close(out.detach(), exp_out, **_tol(torch.bfloat16))
    torch.testing.assert_close(x.grad, exp_dx, **_tol(torch.bfloat16))
    torch.testing.assert_close(bias.grad, exp_dbias, **_DBIAS_TOL)
    torch.testing.assert_close(bias.grad, x.grad.reshape(-1, last).sum(0), rtol=0, atol=0)

    # Autograd reduces bias gradients from [rows,2h] to [2h]; compare the original Function.
    assert bias.grad is not None and tuple(bias.grad.shape) == (last,)
    x2 = xd.clone().requires_grad_(True)
    b2 = bd.clone().requires_grad_(True)
    entry._orig_bias_swiglu_impl(x2, b2).backward(g)
    torch.testing.assert_close(bias.grad, b2.grad, **_DBIAS_TOL)


# Reachable gate misses: rowwise bias, tiny rows, FP16/FP32, and unaligned offsets.
# Use rowwise [rows,2h] bias because one-dimensional [2h] bias is eligible for native fusion.
def _fb_bias_rowwise():
    return (
        torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16),
        torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16),
    )


def _fb_tiny():
    return torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16), None


def _fb_fp16():
    return torch.randn(512, 2048, device="cuda", dtype=torch.float16), None


def _fb_fp32():
    return torch.randn(512, 2048, device="cuda", dtype=torch.float32), None


def _fb_offset():
    return (
        torch.randn(512 * 2048 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(512, 2048),
        None,
    )


_DENSE_FB = [
    ("bias_rowwise", _fb_bias_rowwise),
    ("tiny_rows", _fb_tiny),
    ("fp16", _fb_fp16),
    ("fp32", _fb_fp32),
    ("offset_unaligned", _fb_offset),
]


@requires_sm89
@pytest.mark.parametrize("name,make", _DENSE_FB, ids=[c[0] for c in _DENSE_FB])
def test_dense_impl_entry_fallback_bitwise_eq_pristine(_entry_cudac, monkeypatch, name, make):
    """Dense misses use the original forward/backward exactly without any native call."""
    entry = _entry_cudac
    calls, _ = _install_spy(monkeypatch)
    torch.manual_seed(99)
    x, bias = make()
    x.requires_grad_(True)  # Preserve offset/layout/dtype in place; cloning would realign storage.
    if bias is not None:
        bias.requires_grad_(True)
    g = torch.randn(x.shape[:-1] + (x.shape[-1] // 2,), device="cuda", dtype=x.dtype)

    out_c = entry.bias_swiglu_impl(x, bias)  # A rejected dense input uses the original impl.
    out_c.backward(g)
    dxc = x.grad.clone()
    dbc = bias.grad.clone() if bias is not None else None
    x.grad = None
    if bias is not None:
        bias.grad = None
    out_p = entry._orig_bias_swiglu_impl(
        x, bias
    )  # Run the same input directly through the original impl.
    out_p.backward(g)

    assert (
        calls["fwd"] == 0 and calls["bwd"] == 0
    ), f"{name} must fall back without native calls: {calls}"
    torch.testing.assert_close(out_c.detach(), out_p.detach(), rtol=0, atol=0)
    torch.testing.assert_close(dxc, x.grad, rtol=0, atol=0)
    if bias is not None:
        torch.testing.assert_close(dbc, bias.grad, rtol=0, atol=0)
        assert bias.grad.shape == bias.shape


@requires_sm89
def test_dense_impl_rank_routing(_entry_cudac, monkeypatch):
    """Ranks 2/3 preserve native view semantics; other ranks preserve original errors without native."""
    entry = _entry_cudac
    calls, _ = _install_spy(monkeypatch)
    torch.manual_seed(5)

    x2 = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        o2 = entry.bias_swiglu_impl(x2, None)
    assert o2.shape == (512, 1024) and calls["fwd"] >= 1

    calls["fwd"] = 0
    x3 = torch.randn(64, 8, 2048, device="cuda", dtype=torch.bfloat16)  # rows=512>=256
    with torch.no_grad():
        o3 = entry.bias_swiglu_impl(x3, None)
    assert o3.shape == (64, 8, 1024) and calls["fwd"] >= 1  # Restore the 3D [r0,r1,h] shape.

    calls["fwd"] = 0
    x4 = torch.randn(2, 4, 64, 2048, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):  # Preserve the original assertion that rank is 2 or 3.
        entry.bias_swiglu_impl(x4, None)
    with pytest.raises(AssertionError):
        entry._orig_bias_swiglu_impl(x4, None)
    assert calls["fwd"] == 0  # Rank 4 must not reach native.


def test_dense_entry_route_source_no_exact_shape(_entry_cudac):
    """Forbid exact-shape equality in dense wrappers, except structural rank checks."""
    assert _entry_cudac.bias_swiglu_impl.__wrapped__ is _entry_cudac._orig_bias_swiglu_impl
    src = inspect.getsource(cudac._dense_wrapper)
    assert (
        "_DenseSwiGLUCudacFunction" in src and "cuda_extension_plain" in src
    ), "Inspect the rebound dense wrapper, not the original"
    for m in re.finditer(r"==\s*(?!0\b)\d+", src):
        before = src[max(0, m.start() - 24) : m.start()]
        assert (
            ("len(" in before) or ("ndim" in before) or ("dim()" in before)
        ), f"Possible exact-shape equality in the dense entry; context: ...{before}=="


# Isolate dense rebinding in subprocesses with clean switch states.
_DENSE_PROBE = r"""
import os
import torch
import torch.nn.functional as F
import megatron.core.fusions.fused_bias_swiglu as entry
import megatron.core.fusions._extensions.fused_bias_swiglu_ext as cudac

enabled = os.environ.get("MEGATRON_ENABLE_PPU_SWIGLU_OPT", "1") == "1"
dev = "cuda"


def fact(k, v):
    print("FACT %s %s" % (k, bool(v)))


captured = hasattr(entry, "_orig_bias_swiglu_impl")
fact("DENSE_CAPTURED_ORIG", captured)
fact("DENSE_REBOUND", enabled and captured
     and entry.bias_swiglu_impl is not entry._orig_bias_swiglu_impl)

torch.manual_seed(3)
rows, h = 512, 1024
x = torch.randn(rows, 2 * h, device=dev, dtype=torch.bfloat16)
a, b = x.chunk(2, dim=-1)

if enabled:
    fact("DENSE_EXT_INITIALLY_NONE", cudac._extension is None)
    with torch.no_grad():
        out = entry.bias_swiglu_impl(x, None)
    fact("DENSE_LOADED_KERNEL", cudac._extension is not None)
    ext = cudac.cuda_extension_plain(x, None)
    with torch.no_grad():
        kern = ext.swiglu_cuda(x, None, None, None)[0]
    fact("DENSE_FWD_EQ_KERNEL", torch.equal(out, kern))
    fact("DENSE_SHAPE", tuple(out.shape) == (rows, h) and out.dtype == torch.bfloat16)
    bias = torch.randn(2 * h, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        ob = entry.bias_swiglu_impl(x, bias)              # [2h] bias reaches CUDA C.
        op = entry._orig_bias_swiglu_impl(x, bias)         # Original baseline.
        kb = cudac.cuda_extension_plain(x, bias).swiglu_cuda(x, None, bias, None)[0]
    fact("DENSE_BIAS_HIT_EQ_KERNEL", torch.equal(ob, kb))  # Exact agreement with native.
    fact("DENSE_BIAS_CLOSE_ORIG", torch.allclose(ob, op, rtol=2e-2, atol=2e-2))  # Within the original tolerance.
else:
    with torch.no_grad():
        out = entry.bias_swiglu_impl(x, None)
    fact("DENSE_DISABLED_NO_KERNEL", cudac._extension is None)
    ref = (F.silu(a) * b).to(x.dtype)
    fact("DENSE_DISABLED_ORIGINAL", torch.allclose(out, ref, rtol=2e-2, atol=2e-2))
"""


def _run_dense_probe(enable):
    return _run_probe(_DENSE_PROBE, enable)


@requires_sm89
@pytest.mark.parametrize("enable", [None, True], ids=["default", "explicit_on"])
def test_dense_entry_enabled_routes_through_kernel(enable):
    f = _run_dense_probe(enable=enable)
    assert f["DENSE_CAPTURED_ORIG"]
    assert f["DENSE_REBOUND"]
    assert f["DENSE_EXT_INITIALLY_NONE"]
    assert f["DENSE_LOADED_KERNEL"]
    assert f["DENSE_FWD_EQ_KERNEL"]
    assert f["DENSE_SHAPE"]
    assert f["DENSE_BIAS_HIT_EQ_KERNEL"]
    assert f["DENSE_BIAS_CLOSE_ORIG"]


@requires_sm89
def test_dense_entry_disabled_keeps_original():
    f = _run_dense_probe(enable=False)
    assert not f["DENSE_CAPTURED_ORIG"]
    assert not f["DENSE_REBOUND"]
    assert f["DENSE_DISABLED_NO_KERNEL"]
    assert f["DENSE_DISABLED_ORIGINAL"]


# Additional coverage: specializations, gradient layouts, numerical tolerances, and storage flags.


def _run_impl(entry, y, g, bias=None, weights=None, *, sum_loss=False, **flags):
    """Use independent leaves per run to avoid sharing gradients or graphs between implementations."""
    x = y.detach().clone().requires_grad_()
    auxiliary = weights if weights is not None else bias
    aux = None if auxiliary is None else auxiliary.detach().clone().requires_grad_()
    if weights is not None:
        out = entry.weighted_bias_swiglu_impl(x, None, aux, **flags)
    else:
        out = entry.bias_swiglu_impl(x, aux, **flags)
    if sum_loss:
        out.sum().backward()
    else:
        out.backward(g)
    result = dict(out=out.detach(), dx=x.grad)
    if aux is not None:
        result["dw" if weights is not None else "dbias"] = aux.grad
    return result


def _oracle_results(y, g, bias=None, weights=None, *, round_output=True):
    if weights is not None:
        out, dx, dw = _weighted_reference(y, g, weights, round_output=round_output)
        return dict(out=out, dx=dx, dw=dw)
    out, dx, dbias = _dense_reference(y, g, bias, round_output=round_output)
    result = dict(out=out, dx=dx)
    if bias is not None:
        result["dbias"] = dbias
    return result


def _result_tol(name, y, weights=None):
    if name == "dw":
        return _dw_tol(y.dtype, weights.dtype)
    return _DBIAS_TOL if name == "dbias" else _tol(y.dtype)


@requires_sm89
@pytest.mark.parametrize("h", [512, 768, 14336])
@pytest.mark.parametrize("family", ["dense", "bias", "weighted"])
def test_native_specialization_kernel_names(_entry_cudac, family, h, record_property):
    """Confirm warp/flat dispatch using profiler names; end-to-end cases check numerical results."""
    y = torch.ones(257, 2 * h, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(2 * h, device="cuda", dtype=y.dtype) if family == "bias" else None
    weights = torch.ones(257, 1, device="cuda") if family == "weighted" else None
    g = torch.ones(257, h, device="cuda", dtype=y.dtype)
    _run_impl(_entry_cudac, y, g, bias, weights)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        _run_impl(_entry_cudac, y, g, bias, weights)
        torch.cuda.synchronize()
    names = sorted({event.name for event in prof.events() if event.device_type.name == "CUDA"})
    record_property("native_kernels", json.dumps(names))
    if h == 512:
        assert any("swiglu_kernel<" in name and "32, 512" in name for name in names), names
    else:
        assert any(
            "swiglu_forward_flat_kernel<" in name and f", {h}>" in name for name in names
        ), names
    assert any("swiglu_kernel<" in name for name in names), names


@requires_sm89
@pytest.mark.parametrize("family", ["dense", "bias", "weighted_bf16", "weighted_fp16"])
@pytest.mark.parametrize("layout", ["sum", "transpose", "stride2", "offset"])
def test_backward_gradient_layouts(_entry_cudac, monkeypatch, family, layout):
    """Dense native handles zero-stride and other unusual gradients; weighted backward falls back."""
    torch.manual_seed(106)
    rows, h = 512, 512
    dtype = torch.float16 if family == "weighted_fp16" else torch.bfloat16
    y = torch.randn(rows, 2 * h, device="cuda", dtype=dtype)
    bias = torch.randn(2 * h, device="cuda", dtype=dtype) if family == "bias" else None
    weights = torch.rand(rows, 1, device="cuda") if family.startswith("weighted") else None
    if layout == "sum":
        g = torch.ones((), device="cuda", dtype=dtype).expand(rows, h)
        assert g.stride() == (0, 0)
    elif layout == "transpose":
        g = torch.randn(h, rows, device="cuda", dtype=dtype).t()
        assert g.stride(-1) != 1
    elif layout == "stride2":
        g = torch.randn(rows, 2 * h, device="cuda", dtype=dtype)[:, ::2]
        assert g.stride(-1) == 2
    else:
        g = torch.randn(rows * h + 1, device="cuda", dtype=dtype)[1:].view(rows, h)
        assert g.is_contiguous() and g.data_ptr() % 16 != 0
    calls, raw = _install_spy(monkeypatch)
    seen = []
    if weights is not None:
        gate = cudac.cuda_extension

        def observe(y, weights, grad=None):
            if grad is not None:
                seen.append((grad.stride(), grad.data_ptr() % 16))
            return gate(y, weights, grad)

        monkeypatch.setattr(cudac, "cuda_extension", observe)
    else:

        def observe(y, grad, bias, weights):
            if grad is not None:
                seen.append((grad.stride(), grad.data_ptr() % 16))
                calls["bwd"] += 1
            else:
                calls["fwd"] += 1
            return raw(y, grad, bias, weights)

        monkeypatch.setattr(cudac._get_extension(), "swiglu_cuda", observe)
    actual = _run_impl(_entry_cudac, y, g, bias, weights, sum_loss=layout == "sum")
    assert seen == [(g.stride(), g.data_ptr() % 16)]
    assert calls == {"fwd": 1, "bwd": int(weights is None)}
    with monkeypatch.context() as patch:
        patch.setattr(cudac, "_OPT_ENABLED", False)
        expected = _run_impl(_entry_cudac, y, g, bias, weights, sum_loss=layout == "sum")
    assert calls == {"fwd": 1, "bwd": int(weights is None)}
    oracle = _oracle_results(y, g, bias, weights)
    for name, value in actual.items():
        torch.testing.assert_close(value, oracle[name], **_result_tol(name, y, weights))
        if weights is not None and name != "out":
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    if bias is not None:
        torch.testing.assert_close(actual["dbias"], actual["dx"].sum(0), rtol=0, atol=0)


def _rounding_ulp(ref, dtype):
    """Compute target-dtype spacing at the FP64 reference binade, clamping subnormal spacing."""
    info = torch.finfo(dtype)
    _, exponent = torch.frexp(ref.abs().clamp_min(info.tiny))
    return torch.ldexp(torch.full_like(ref, info.eps), exponent - 1)


# Allow 8 FP32 epsilons times the absolute term sum for intermediate math and reordered sums.
# This engineering allowance is not a mathematical bound for arbitrary reduction topologies.
# It does not depend on candidate outputs or observed failures; fixed tolerances still apply.
_REDUCTION_EPS_FACTOR = 8


def _precision_allowances(y, g, bias, weights, oracle):
    """Allow one output ULP plus intermediate-math and rounding propagation for reductions."""
    allowances = {
        name: _rounding_ulp(ref, weights.dtype if name == "dw" else y.dtype)
        for name, ref in oracle.items()
    }
    eps = _REDUCTION_EPS_FACTOR * torch.finfo(torch.float32).eps
    if weights is not None:
        a, b = y.double().chunk(2, dim=-1)
        terms = (a * torch.sigmoid(a) * b) * g.double()
        allowances["dw"] += eps * terms.abs().sum(dim=-1, keepdim=True)
    if bias is not None:
        dx = oracle["dx"]
        # dbias reduces rounded dx; per-element ULP differences can propagate into the sum.
        allowances["dbias"] += (
            _rounding_ulp(dx, y.dtype) + eps * dx.to(y.dtype).double().abs()
        ).sum_to_size(bias.shape)
    return allowances


def _assert_precision_with_tolerance(
    actual, baseline, oracle, *, allowances, tolerances, record=None
):
    """Require error <= baseline error + allowance and fixed FP64 tolerances for both paths."""
    assert (
        actual.keys() == baseline.keys() == oracle.keys() == allowances.keys() == tolerances.keys()
    )
    reports = {}
    for name, value in actual.items():
        old, ref = baseline[name], oracle[name]
        assert value.shape == old.shape == ref.shape and value.dtype == old.dtype
        assert ref.dtype == torch.float64
        assert all(
            torch.isfinite(t).all().item() for t in (value, old, ref)
        ), "Precision comparison contains non-finite values"
        allowance = allowances[name]
        assert allowance.shape == ref.shape and allowance.dtype == torch.float64
        assert torch.isfinite(allowance).all().item() and (allowance >= 0).all().item()
        new_error, old_error = (value.double() - ref).abs(), (old.double() - ref).abs()
        worse = new_error > old_error  # Retain strict no-worse counts for diagnostics only.
        excess = new_error - old_error
        beyond = excess > allowance
        index = excess.reshape(-1).argmax().item()
        near_zero = ref.abs() < 1e-2  # Diagnostic only; does not affect acceptance.
        stats = dict(
            elements=ref.numel(),
            worse=worse.sum().item(),
            allowance_failures=beyond.sum().item(),
            max_allowance=allowance.max().item(),
            max_excess_after_allowance=(excess - allowance).max().item(),
            near_zero=near_zero.sum().item(),
            near_zero_worse=(worse & near_zero).sum().item(),
            max_abs_new=new_error.max().item(),
            max_abs_old=old_error.max().item(),
            rms_new=new_error.square().mean().sqrt().item(),
            rms_old=old_error.square().mean().sqrt().item(),
            max_excess=excess.max().item(),
            worst_flat_index=index,
            candidate=value.reshape(-1)[index].item(),
            baseline=old.reshape(-1)[index].item(),
            oracle=ref.reshape(-1)[index].item(),
        )
        tol = tolerances[name]
        rounded_ref = ref.to(value.dtype).double()
        limit = tol["atol"] + tol["rtol"] * rounded_ref.abs()
        stats["fixed_tol_failures"] = ((value.double() - rounded_ref).abs() > limit).sum().item()
        stats["baseline_fixed_tol_failures"] = (
            ((old.double() - rounded_ref).abs() > limit).sum().item()
        )
        stats["nonzero_zero_would_pass"] = (
            ((rounded_ref != 0) & (rounded_ref.abs() <= limit)).sum().item()
        )
        reports[name] = stats
    detail = json.dumps(reports, sort_keys=True)
    if record is not None:
        record("precision_report", detail)
    assert not any(s["allowance_failures"] for s in reports.values()), (
        "Floating-point allowance exceeded: " + detail
    )
    assert not any(
        s["fixed_tol_failures"] or s["baseline_fixed_tol_failures"] for s in reports.values()
    ), ("Fixed tolerance failed: " + detail)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rounding_ulp_handles_zero_subnormal_and_binade(dtype):
    info = torch.finfo(dtype)
    ref = torch.tensor([0, info.tiny / 2, info.tiny, 0.5, 1, 2, -1, info.max], dtype=torch.float64)
    ulp = _rounding_ulp(ref, dtype)
    expected = torch.tensor(
        [info.tiny * info.eps] * 3 + [info.eps / 2, info.eps, 2 * info.eps, info.eps],
        dtype=torch.float64,
    )
    torch.testing.assert_close(ulp[:-1], expected, rtol=0, atol=0)
    assert torch.isfinite(ulp).all() and (ulp > 0).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_precision_checker_accepts_one_ulp_rejects_excess_and_zeroing(dtype):
    ref = dict(out=torch.tensor([0.0, 1e-5, 1.0], dtype=torch.float64))
    baseline = dict(out=ref["out"].to(dtype))
    allowance = _rounding_ulp(ref["out"], dtype)
    kwargs = dict(allowances=dict(out=allowance), tolerances=dict(out=_tol(dtype)))
    candidate = dict(out=(baseline["out"].double() + allowance).to(dtype))
    _assert_precision_with_tolerance(candidate, baseline, ref, **kwargs)
    candidate = dict(out=(baseline["out"].double() + 4 * allowance).to(dtype))
    with pytest.raises(AssertionError, match="Floating-point allowance exceeded"):
        _assert_precision_with_tolerance(candidate, baseline, ref, **kwargs)
    candidate = dict(out=baseline["out"].clone())
    candidate["out"][1] = 0
    with pytest.raises(AssertionError, match="Floating-point allowance exceeded"):
        _assert_precision_with_tolerance(candidate, baseline, ref, **kwargs)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("target", ["actual", "baseline", "oracle"])
def test_precision_checker_rejects_nonfinite(invalid, target):
    ref = dict(out=torch.tensor([1.0], dtype=torch.float64))
    arguments = dict(
        actual=dict(out=ref["out"].float()), baseline=dict(out=ref["out"].float()), oracle=ref
    )
    arguments[target]["out"][0] = invalid
    with pytest.raises(AssertionError, match="non-finite values"):
        _assert_precision_with_tolerance(
            **arguments,
            allowances=dict(out=torch.zeros(1, dtype=torch.float64)),
            tolerances=dict(out=_tol(torch.float32)),
        )


def test_precision_checker_keeps_independent_oracle_limit():
    ref = dict(out=torch.tensor([1.0], dtype=torch.float64))
    wrong = dict(out=torch.tensor([2.0], dtype=torch.float32))
    # Matching wrong results may pass a relative check, but must fail the independent oracle.
    with pytest.raises(AssertionError, match="Fixed tolerance failed"):
        _assert_precision_with_tolerance(
            wrong,
            wrong,
            ref,
            allowances=dict(out=_rounding_ulp(ref["out"], torch.float32)),
            tolerances=dict(out=_tol(torch.float32)),
        )


def test_reduction_allowance_scales_with_input_and_bounds_cancellation():
    y = torch.tensor([[1.0, 1.0, 1.0, -1.0]], dtype=torch.bfloat16)
    g = torch.ones(1, 2, dtype=y.dtype)
    weights = torch.ones(1, 1)
    oracle = _oracle_results(y, g, weights=weights, round_output=False)
    allowance = _precision_allowances(y, g, None, weights, oracle)["dw"]
    assert oracle["dw"].item() == 0 and allowance.item() > 0
    smaller = _oracle_results(y, g / 2, weights=weights, round_output=False)
    small_allowance = _precision_allowances(y, g / 2, None, weights, smaller)["dw"]
    torch.testing.assert_close(small_allowance, allowance / 2, rtol=1e-12, atol=0)
    kwargs = dict(
        allowances=dict(dw=allowance), tolerances=dict(dw=_dw_tol(y.dtype, weights.dtype))
    )
    ref, baseline = dict(dw=oracle["dw"]), dict(dw=oracle["dw"].float())
    _assert_precision_with_tolerance(dict(dw=(allowance / 2).float()), baseline, ref, **kwargs)
    with pytest.raises(AssertionError, match="Floating-point allowance exceeded"):
        _assert_precision_with_tolerance(dict(dw=(allowance * 2).float()), baseline, ref, **kwargs)


def test_dbias_oracle_rounds_elementwise_before_reduction():
    y = torch.tensor([[0.31, 1.17], [0.73, -0.41], [-0.87, 0.63]], dtype=torch.bfloat16)
    g = torch.tensor([[0.3], [-0.7], [1.1]], dtype=y.dtype)
    bias = torch.zeros(2, dtype=y.dtype)
    _, dx64, db64 = _dense_reference(y, g, bias, round_output=False)
    expected = dx64.to(y.dtype).double().sum(0)
    torch.testing.assert_close(db64, expected, rtol=0, atol=0)
    assert not torch.equal(
        db64, dx64.sum(0)
    ), "The sample must distinguish rounded-before-reduction from unrounded reduction"
    _, _, rounded = _dense_reference(y, g, bias)
    torch.testing.assert_close(rounded, expected.to(y.dtype), rtol=0, atol=0)


def _precision_inputs(dtype, h, family, wdtype, sample):
    torch.manual_seed(20261102)
    rows = 512 if dtype == torch.float16 else 256
    y = torch.randn(rows, 2 * h, device="cuda", dtype=dtype)
    g = torch.randn(rows, h, device="cuda", dtype=dtype)
    bias = torch.randn(2 * h, device="cuda", dtype=dtype) if family == "bias" else None
    weights = torch.rand(rows, 1, device="cuda", dtype=wdtype) if family == "weighted" else None
    if sample == "near_zero":
        # Cover SiLU zero, both sides of the derivative root, negative saturation, and small b/g.
        points = torch.tensor(
            [-20.0, -12.0, -1.28125, -1.2734375, -(2**-12), 0.0, 2**-12, 1.0],
            device="cuda",
            dtype=dtype,
        )
        y[:, :h] = points.repeat(h // points.numel())
        y[:, h:] *= 1e-2
        g *= 1e-2
        if bias is not None:
            bias.zero_()
        if weights is not None:
            weights *= 1e-2
    elif sample == "cancellation":
        # Cancel dW across columns and dbias across rows, with one reproducible residual perturbation.
        y[:, 1::2] = y[:, ::2]
        y[1::2] = y[::2]
        g[:, 1::2] = -g[:, ::2]
        g[1::2] = -g[::2]
        g[0, 0] *= 0.5
        assert torch.equal(g[2::2], -g[3::2])
        assert torch.equal(g[:, 3::2], -g[:, 2::2])
    return y, g, bias, weights


_PRECISION_CASES = [
    pytest.param(torch.bfloat16, 512, "weighted", torch.float32, id="bf16-w32-warp"),
    pytest.param(torch.bfloat16, 512, "weighted", torch.bfloat16, id="bf16-wbf16-warp"),
    pytest.param(torch.float16, 512, "weighted", torch.float32, id="fp16-w32-warp"),
    pytest.param(torch.float16, 512, "weighted", torch.float16, id="fp16-wfp16-warp"),
    pytest.param(torch.bfloat16, 1408, "weighted", torch.float32, id="bf16-w32-pack"),
    pytest.param(torch.bfloat16, 14336, "weighted", torch.float32, id="bf16-w32-flat"),
    pytest.param(torch.bfloat16, 768, "dense", None, id="dense-flat"),
    pytest.param(torch.bfloat16, 14336, "bias", None, id="bias-flat"),
]


@requires_sm89
@pytest.mark.parametrize("dtype,h,family,wdtype", _PRECISION_CASES)
@pytest.mark.parametrize("sample", ["random", "near_zero", "cancellation"])
def test_precision_pointwise_with_tolerance(
    _entry_cudac, monkeypatch, record_property, dtype, h, family, wdtype, sample
):
    """Compare both implementations with one FP64 oracle using explicit error allowances."""
    y, g, bias, weights = _precision_inputs(dtype, h, family, wdtype, sample)
    calls, _ = _install_spy(monkeypatch)
    actual = _run_impl(_entry_cudac, y, g, bias, weights)
    assert calls == {"fwd": 1, "bwd": 1}
    # Avoid eager fallback caused by earlier cases exhausting Dynamo recompilation caches.
    torch.compiler.reset()
    with monkeypatch.context() as patch:
        patch.setattr(cudac, "_OPT_ENABLED", False)
        patch.setattr(torch._dynamo.config, "suppress_errors", False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            baseline = _run_impl(_entry_cudac, y, g, bias, weights)
    compiled_regions = [
        event.name for event in prof.events() if "Torch-Compiled Region" in event.name
    ]
    record_property("baseline_compiled_regions", json.dumps(compiled_regions))
    assert (
        len(compiled_regions) >= 2
    ), "Original forward and backward must execute torch.compile regions"
    assert calls == {"fwd": 1, "bwd": 1}, "The original baseline must not call the extension"
    oracle = _oracle_results(y, g, bias, weights, round_output=False)
    _assert_precision_with_tolerance(
        actual,
        baseline,
        oracle,
        allowances=_precision_allowances(y, g, bias, weights, oracle),
        tolerances={name: _result_tol(name, y, weights) for name in actual},
        record=record_property,
    )


@requires_sm89
def test_fp8_tensor_is_rejected_by_gate_and_native(monkeypatch):
    """FP8 is an activation-storage format, not a CUDA C computation input dtype."""
    y = torch.ones(256, 1024, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weights = torch.ones(256, 1, device="cuda")
    loader = Mock(side_effect=AssertionError("FP8 inputs must not trigger extension loading"))
    monkeypatch.setattr(cudac, "_load_extension", loader)
    assert cudac.cuda_extension(y, weights) is None
    assert cudac.cuda_extension_plain(y) is None
    loader.assert_not_called()
    with pytest.raises(RuntimeError, match="input dtype must be float16, bfloat16 or float32"):
        cudac._get_extension().swiglu_cuda(y, None, None, None)


@requires_sm89
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("fp8,offload", [(True, False), (False, True), (True, True)])
def test_dense_storage_flags_preserve_original(_entry_cudac, monkeypatch, with_bias, fp8, offload):
    """Check storage flags and restored gradients with CPU round-trip hooks, not FP8 computation."""
    torch.manual_seed(902)
    y = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1024, device="cuda", dtype=y.dtype) if with_bias else None
    g = torch.randn(256, 512, device="cuda", dtype=y.dtype)
    records, restored = [], []

    def pack(tensor):
        marked = bool(getattr(tensor, "activation_offloading", False))
        stored = tensor.detach().cpu() if marked else tensor.detach().clone()
        records.append((tensor.dtype, marked, stored))
        return tensor.device, stored

    def unpack(payload):
        device, stored = payload
        restored.append(stored.device.type)
        return stored.to(device)

    calls, _ = _install_spy(monkeypatch)
    flags = dict(fp8_input_store=fp8, cpu_offload_input=offload)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual = _run_impl(_entry_cudac, y, g, bias, **flags)
    assert calls == {"fwd": 0, "bwd": 0}
    assert len(records) == 1 + int(with_bias) and len(restored) == len(records)
    saved_dtype = torch.float8_e4m3fn if fp8 else y.dtype
    expected_saved = y.to(saved_dtype)
    assert records[0][0] == saved_dtype
    torch.testing.assert_close(
        records[0][2].float().cpu(), expected_saved.float().cpu(), rtol=0, atol=0
    )
    assert all(marked == offload for _, marked, _ in records)
    assert restored == (["cpu"] if offload else ["cuda"]) * len(records)
    if with_bias:
        assert records[1][0] == bias.dtype
        torch.testing.assert_close(records[1][2].cpu(), bias.cpu(), rtol=0, atol=0)
    with monkeypatch.context() as patch:
        patch.setattr(cudac, "_OPT_ENABLED", False)
        expected = _run_impl(_entry_cudac, y, g, bias, **flags)
        unquantized = _run_impl(_entry_cudac, y, g, bias)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    # Forward is unquantized; backward uses FP8-restored inputs without quantizing bias.
    with torch.no_grad():
        backward_y = expected_saved.to(y.dtype)
        dx = (
            _entry_cudac.bias_swiglu_back(g, backward_y, bias)
            if with_bias
            else _entry_cudac.swiglu_back(g, backward_y)
        )
    torch.testing.assert_close(actual["dx"], dx, rtol=0, atol=0)
    if with_bias:
        torch.testing.assert_close(actual["dbias"], dx.sum(0), rtol=0, atol=0)
    torch.testing.assert_close(actual["out"], unquantized["out"], rtol=0, atol=0)
    assert calls == {"fwd": 0, "bwd": 0}


@requires_sm89
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("same_weight_dtype", [False, True])
def test_weighted_fp8_storage_restores_before_kernel(
    _entry_cudac, monkeypatch, dtype, same_weight_dtype
):
    """Preserve weighted FP8 storage while native receives only original/restored BF16 or FP16."""
    torch.manual_seed(903)
    y = torch.randn(512, 1024, device="cuda", dtype=dtype)
    weights = torch.rand(512, 1, device="cuda", dtype=dtype if same_weight_dtype else torch.float32)
    g = torch.randn(512, 512, device="cuda", dtype=dtype)
    restored_y = y.to(torch.float8_e4m3fn).to(dtype)
    assert not torch.equal(y, restored_y), "The sample must exhibit actual FP8 quantization"
    calls, raw = _install_spy(monkeypatch)
    inputs, saved = [], []

    def spy(x, grad, bias, w):
        calls["fwd" if grad is None else "bwd"] += 1
        inputs.append(x.detach().clone())
        assert x.dtype == dtype
        return raw(x, grad, bias, w)

    def pack(tensor):
        saved.append(tensor.detach().clone())
        return tensor.detach()

    monkeypatch.setattr(cudac._get_extension(), "swiglu_cuda", spy)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = _run_impl(_entry_cudac, y, g, weights=weights, fp8_input_store=True)
    assert calls == {"fwd": 1, "bwd": 1}
    assert [tensor.dtype for tensor in saved] == [torch.float8_e4m3fn, weights.dtype]
    torch.testing.assert_close(saved[0].float(), y.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
    torch.testing.assert_close(saved[1], weights, rtol=0, atol=0)
    torch.testing.assert_close(inputs[0], y, rtol=0, atol=0)
    torch.testing.assert_close(inputs[1], restored_y, rtol=0, atol=0)
    with torch.no_grad():
        expected_out = raw(y, None, None, weights)[0]
        expected_dx, expected_dw = raw(restored_y, g, None, weights)
    for name, expected in dict(out=expected_out, dx=expected_dx, dw=expected_dw).items():
        torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
    oracle = _oracle_results(restored_y, g, weights=weights)
    oracle["out"] = _weighted_reference(y, g, weights)[0]
    for name in actual:
        torch.testing.assert_close(actual[name], oracle[name], **_result_tol(name, y, weights))
