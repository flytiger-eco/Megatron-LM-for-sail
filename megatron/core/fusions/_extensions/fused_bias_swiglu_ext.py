"""Internal SwiGLU dispatch, CUDA C loading, autograd wrappers, and optional input logging.

Eligible sm89 inputs use the extension by default. Only setting
MEGATRON_DISABLE_PPU_SWIGLU_OPT to the exact string "1" before importing the entry point
disables it. Unset and all other values leave it enabled; the former enable variable
is ignored. Other devices and inputs use the supplied original functions without
duplicating formulas or importing the entry module back.
MEGATRON_SWIGLU_INPUT_LOG_DIR independently controls input logging, which is off by default.
The first eligible call compiles the extension using the CUDA/HGGC toolchain and ninja.
Loading failures warn once and fall back; kernel execution errors propagate.
CUDA Graph acceleration requires warmup. Importing this module does not initialize CUDA.
"""

import hashlib
import os
import threading
import warnings
from functools import lru_cache, wraps
from pathlib import Path
from types import ModuleType

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode

from megatron.core.jit import jit_fuser

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_OPT_ENV = "MEGATRON_DISABLE_PPU_SWIGLU_OPT"
_OPT_ENABLED = os.environ.get(_OPT_ENV) != "1"
_extension = None
_extension_lock = threading.Lock()
_extension_error = None

# Preserve the validated dtype, row-count, and layout limits despite default enablement.
_REAL_PATTERN_MIN_ROWS = {torch.bfloat16: 256, torch.float16: 256}


def _get_extension():
    """Compile lazily after the caller checks enablement, input device, and shape."""
    global _extension
    if _extension is not None:
        return _extension
    with _extension_lock:
        if _extension is None:
            if not torch.cuda.is_available():
                raise RuntimeError("SwiGLU CUDA C requires an available CUDA/HGGC device")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up SwiGLU CUDA C with nonempty inputs before CUDA Graph capture"
                )
            from torch.utils.cpp_extension import load

            sources = [_CSRC_DIR / "swiglu.cpp", _CSRC_DIR / "swiglu.cu"]
            cuda_flags = ["-O3", "-lineinfo", "-gencode=arch=compute_89,code=sm_89"]
            source_hash = hashlib.sha256((torch.__version__ + repr(cuda_flags)).encode())
            for source in sources:
                source_hash.update(source.read_bytes())
            digest = source_hash.hexdigest()[:12]
            _extension = load(
                name=f"megatron_swiglu_cudac_{digest}",
                sources=[str(source) for source in sources],
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                with_cuda=True,
            )
    return _extension


def _load_extension():
    """Handle loading failures without catching kernel execution errors."""
    global _extension_error
    if _extension is None and torch.cuda.is_current_stream_capturing():
        return None
    try:
        return _get_extension()
    except (ImportError, OSError, RuntimeError) as error:
        _extension_error = error
        warnings.warn(
            f"SwiGLU CUDA C extension unavailable; using the original implementation: {error}",
            RuntimeWarning,
        )
        return None


@lru_cache(maxsize=None)
def _is_sm89_index(index):
    # Capabilities are immutable within a process; cache each logical device separately.
    return torch.cuda.get_device_capability(index) == (8, 9)


def _is_sm89(device):
    # Tensor devices have explicit indices. Resolve an unindexed alias before caching it.
    if device.type != "cuda":
        return False
    index = device.index
    return _is_sm89_index(torch.cuda.current_device() if index is None else index)


def cuda_extension(
    y: torch.Tensor, weights: torch.Tensor, grad: torch.Tensor | None = None
) -> ModuleType | None:
    """Return the extension for eligible weighted inputs, or None otherwise."""
    if not _OPT_ENABLED or _extension_error is not None:
        return None
    if torch.compiler.is_compiling() or torch.jit.is_tracing():
        return None
    if jit_fuser is not torch.compile:
        return None
    if _get_current_dispatch_mode() is not None or torch.overrides.has_torch_function_unary(y):
        return None
    tensors = (y, weights) if grad is None else (y, weights, grad)
    if any(type(t) not in (torch.Tensor, torch.nn.Parameter) for t in tensors):
        return None
    if torch.autograd.forward_ad._current_level >= 0 or torch._C._are_functorch_transforms_active():
        return None
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return None
    if any(t.layout != torch.strided or t.is_nested or t.is_conj() or t.is_neg() for t in tensors):
        return None
    if not y.is_cuda or y.dtype not in (torch.float16, torch.bfloat16):
        return None
    if not y.ndim or y.shape[-1] == 0 or y.shape[-1] % 2:
        return None
    last = y.shape[-1]
    rows = y.numel() // last
    if not y.is_contiguous() or y.data_ptr() % 16:
        return None
    if (last // 2) % 8:
        return None
    min_rows = _REAL_PATTERN_MIN_ROWS.get(y.dtype)
    if min_rows is None or rows < min_rows:
        return None
    if (
        weights.shape != (*y.shape[:-1], 1)
        or not weights.is_contiguous()
        or weights.data_ptr() % 16
        or weights.device != y.device
        or weights.dtype not in (torch.float32, y.dtype)
    ):
        return None
    if grad is not None and (
        not grad.is_contiguous()
        or grad.data_ptr() % 16
        or grad.shape != (*y.shape[:-1], last // 2)
        or grad.dtype != y.dtype
        or grad.device != y.device
    ):
        return None
    if torch.is_autocast_enabled() or not _is_sm89(y.device):
        return None
    return _load_extension()


def cuda_extension_plain(
    y: torch.Tensor, bias: torch.Tensor | None = None, grad: torch.Tensor | None = None
) -> ModuleType | None:
    """Return the extension for eligible dense inputs with optional bias, or None.

    Require BF16, a minimum row count, and contiguous aligned inputs. Only [2h] bias
    can be fused without materialization. Use monotonic thresholds and vector alignment
    constraints rather than an exact-shape allowlist.
    """
    if not _OPT_ENABLED or _extension_error is not None:
        return None
    if torch.compiler.is_compiling() or torch.jit.is_tracing():
        return None
    if jit_fuser is not torch.compile:
        return None
    if _get_current_dispatch_mode() is not None or torch.overrides.has_torch_function_unary(y):
        return None
    tensors = (y,) if grad is None else (y, grad)
    if bias is not None:
        tensors = tensors + (bias,)
    if any(type(t) not in (torch.Tensor, torch.nn.Parameter) for t in tensors):
        return None
    if torch.autograd.forward_ad._current_level >= 0 or torch._C._are_functorch_transforms_active():
        return None
    # Dense dispatch wraps autograd.Function at the impl level, so requires_grad is allowed.
    if any(t.layout != torch.strided or t.is_nested or t.is_conj() or t.is_neg() for t in tensors):
        return None
    if not y.is_cuda or y.dtype not in (torch.bfloat16,):
        return None
    if not y.ndim or y.shape[-1] == 0 or y.shape[-1] % 2:
        return None
    last = y.shape[-1]
    rows = y.numel() // last
    if not y.is_contiguous() or y.data_ptr() % 16:
        return None
    if (last // 2) % 8:
        return None
    min_rows = _REAL_PATTERN_MIN_ROWS.get(y.dtype)
    if min_rows is None or rows < min_rows:
        return None
    if bias is not None and (
        bias.dtype != y.dtype
        or bias.shape != (last,)
        or not bias.is_contiguous()
        or bias.data_ptr() % 16
        or bias.device != y.device
    ):
        return None
    if grad is not None and (
        not grad.is_contiguous()
        or grad.data_ptr() % 16
        or grad.shape != (*y.shape[:-1], last // 2)
        or grad.dtype != y.dtype
        or grad.device != y.device
    ):
        return None
    if torch.is_autocast_enabled() or not _is_sm89(y.device):
        return None
    return _load_extension()


class _DenseSwiGLUCudacFunction(torch.autograd.Function):
    """Internal CUDA C autograd wrapper for dense inputs with optional bias."""

    @staticmethod
    def forward(ctx, input2d, bias, extension):
        ctx.save_for_backward(input2d, bias)
        ctx.ext = extension
        return extension.swiglu_cuda(input2d, None, bias, None)[0]

    @staticmethod
    def backward(ctx, grad_output):
        y, bias = ctx.saved_tensors
        dx = ctx.ext.swiglu_cuda(y, grad_output, bias, None)[0]
        return dx, (dx if bias is not None else None), None


def _weighted_wrappers(forward, backward):
    @wraps(forward)
    def weighted_swiglu(y, weights):
        if torch.jit.is_scripting():
            return forward(y, weights)
        extension = cuda_extension(y, weights)
        if extension is None:
            return forward(y, weights)
        return extension.swiglu_cuda(y, None, None, weights)[0]

    @wraps(backward)
    def weighted_swiglu_back(g, y, weights):
        if torch.jit.is_scripting():
            return backward(g, y, weights)
        extension = cuda_extension(y, weights, grad=g)
        if extension is None:
            return backward(g, y, weights)
        dx, dw = extension.swiglu_cuda(y, g, None, weights)
        return dx, dw

    return weighted_swiglu, weighted_swiglu_back


def _dense_wrapper(original):
    @wraps(original)
    def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
        if torch.jit.is_scripting():
            return original(input, bias, fp8_input_store, cpu_offload_input)
        ori_shape = input.shape
        if len(ori_shape) not in (2, 3) or fp8_input_store or cpu_offload_input:
            return original(input, bias, fp8_input_store, cpu_offload_input)
        input2d = input.view(-1, ori_shape[-1])
        extension = cuda_extension_plain(input2d, bias)
        if extension is None:
            return original(input, bias, fp8_input_store, cpu_offload_input)
        output = _DenseSwiGLUCudacFunction.apply(input2d, bias, extension)
        return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)

    return bias_swiglu_impl


class _InputLogger:
    """Keep per-install logging state, write per-global-rank files, and deduplicate shapes."""

    def __init__(self, directory, enabled):
        self.directory = directory
        self.enabled = enabled
        self.lock = threading.Lock()
        self.fh = None
        self.seen = set()
        self.calls = 0

    @staticmethod
    def _dist_ids():
        rank, world, source = None, None, "default"
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank, world, source = dist.get_rank(), dist.get_world_size(), "torch.distributed"
        except Exception:
            pass
        if rank is None:
            for name in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "PMI_RANK", "LOCAL_RANK"):
                value = os.environ.get(name)
                if value is not None and value.lstrip("-").isdigit():
                    rank, source = int(value), name
                    break
        if world is None:
            for name in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
                value = os.environ.get(name)
                if value is not None and value.isdigit():
                    world = int(value)
                    break
        return (0 if rank is None else rank), world, source

    @staticmethod
    def _tensor_desc(x):
        if x is None:
            return None
        return dict(
            shape=tuple(x.shape),
            dtype=str(x.dtype),
            device=str(x.device),
            contiguous=bool(x.is_contiguous()),
            stride=tuple(x.stride()),
            ptr_mod16=int(x.data_ptr() % 16),
            requires_grad=bool(x.requires_grad),
        )

    def record(self, fn, input, bias=None, weights=None, **flags):
        try:
            if torch.compiler.is_compiling() or torch.jit.is_scripting():
                return
            import datetime
            import json
            import socket

            descriptors = [self._tensor_desc(x) for x in (input, bias, weights)]
            in_d, b_d, w_d = descriptors
            signature = (
                fn,
                in_d["shape"],
                in_d["dtype"],
                in_d["stride"],
                in_d["contiguous"],
                None if b_d is None else (b_d["shape"], b_d["dtype"]),
                None if w_d is None else (w_d["shape"], w_d["dtype"], w_d["contiguous"]),
                tuple(sorted((k, bool(v)) for k, v in flags.items())),
            )
            with self.lock:
                if self.fh is None:
                    self.rank, self.world, source = self._dist_ids()
                    self.host = socket.gethostname()
                    os.makedirs(self.directory, exist_ok=True)
                    path = Path(self.directory) / ("swiglu_inputs_rank%05d.jsonl" % self.rank)
                    self.fh = path.open("a", buffering=1)
                    header = dict(
                        event="session_start",
                        rank=self.rank,
                        rank_source=source,
                        world_size=self.world,
                        host=self.host,
                        pid=os.getpid(),
                        torch=torch.__version__,
                        cudac_enabled=self.enabled,
                        ts=datetime.datetime.now().isoformat(timespec="seconds"),
                    )
                    self.fh.write(json.dumps(header, ensure_ascii=False) + "\n")
                self.calls += 1
                if signature in self.seen:
                    return
                self.seen.add(signature)
                record = dict(
                    event="input",
                    fn=fn,
                    call_index=self.calls,
                    rank=self.rank,
                    world_size=self.world,
                    host=self.host,
                    ts=datetime.datetime.now().isoformat(timespec="seconds"),
                    input=descriptors[0],
                    bias=descriptors[1],
                    weights=descriptors[2],
                    flags={k: bool(v) for k, v in flags.items()},
                )
                self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.fh.flush()
        except Exception:
            # Logging is best-effort and does not swallow errors from the wrapped operator.
            pass


def _logging_wrappers(dense, weighted, logger):
    @wraps(dense)
    def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
        logger.record(
            "bias_swiglu_impl",
            input,
            bias=bias,
            fp8_input_store=fp8_input_store,
            cpu_offload_input=cpu_offload_input,
        )
        return dense(input, bias, fp8_input_store, cpu_offload_input)

    @wraps(weighted)
    def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):
        logger.record(
            "weighted_bias_swiglu_impl",
            input,
            bias=bias,
            weights=weights,
            fp8_input_store=fp8_input_store,
        )
        return weighted(input, bias, weights, fp8_input_store)

    return bias_swiglu_impl, weighted_bias_swiglu_impl


def install(namespace: dict) -> None:
    """Install wrappers after the original definitions, supporting repeated installs and reloads.

    Args:
        namespace: Entry-point globals containing the original functions, avoiding circular imports.
    """
    global _OPT_ENABLED
    enabled = os.environ.get(_OPT_ENV) != "1"
    log_dir = os.environ.get("MEGATRON_SWIGLU_INPUT_LOG_DIR") or None
    _OPT_ENABLED = enabled
    state = namespace.pop("_swiglu_ext_state", None)
    if state is not None:
        unchanged = all(namespace.get(name) is fn for name, fn in state["installed"].items())
        if unchanged and state["config"] == (enabled, log_dir):
            namespace["_swiglu_ext_state"] = state
            return
        # Preserve fresh definitions after reload; unwrap only when reinstalling unchanged wrappers.
        if unchanged:
            namespace.update(state["originals"])
    for name in ("_orig_weighted_swiglu", "_orig_weighted_swiglu_back", "_orig_bias_swiglu_impl"):
        namespace.pop(name, None)
    if not enabled and not log_dir:
        return
    names = (
        "weighted_swiglu",
        "weighted_swiglu_back",
        "bias_swiglu_impl",
        "weighted_bias_swiglu_impl",
    )
    originals = {name: namespace[name] for name in names}
    if enabled:
        for name in names[:3]:
            namespace["_orig_" + name] = originals[name]
        namespace["weighted_swiglu"], namespace["weighted_swiglu_back"] = _weighted_wrappers(
            originals["weighted_swiglu"], originals["weighted_swiglu_back"]
        )
        namespace["bias_swiglu_impl"] = _dense_wrapper(originals["bias_swiglu_impl"])
    if log_dir:
        logger = _InputLogger(log_dir, enabled)
        namespace["bias_swiglu_impl"], namespace["weighted_bias_swiglu_impl"] = _logging_wrappers(
            namespace["bias_swiglu_impl"], namespace["weighted_bias_swiglu_impl"], logger
        )
    namespace["_swiglu_ext_state"] = dict(
        config=(enabled, log_dir),
        originals=originals,
        installed={name: namespace[name] for name in names},
    )
