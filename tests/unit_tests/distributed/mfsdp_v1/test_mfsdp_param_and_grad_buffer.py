# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import os
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import (
    MegatronFSDP,
    MixedPrecisionPolicy,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    AllGatherPipeline,
    BucketingPolicy,
    FixedPoolAllocator,
    MaxPoolAllocator,
    ParamAndGradBuffer,
    ParameterGroup,
    _get_parameter_groups,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import FSDPDistributedIndex


class _ExpertTestModule(torch.nn.Module):
    """
    Mock module whose params are routed under `.experts.` to trigger
    is_expert_param=True. The outer `layer` attribute puts a dot before
    `experts` in the parameter path (e.g. `layer.experts.linear_fc1`).
    """

    def __init__(self, shapes):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.experts = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(shape)) for name, shape in shapes.items()}
        )


def _get_bucket_signatures(module):
    bucket_groups, _, _ = _get_parameter_groups(
        module, BucketingPolicy(suggested_bucket_size=None), meta_device_init_fp8_params={}
    )
    param_to_name = {param: name for name, param in module.named_parameters()}
    return [
        {
            "chunk_size_factor": group.chunk_size_factor,
            "params": [(param_to_name[param], tuple(param.shape)) for param in group.params],
        }
        for group in bucket_groups
    ]


def _make_uniform_parameter_groups(count=4):
    return [
        ParameterGroup(
            [torch.nn.Parameter(torch.empty(8, dtype=torch.bfloat16))],
            dtype=torch.bfloat16,
            fsdp_unit_id=unit_id,
        )
        for unit_id in range(count)
    ]


class _CpuMemoryBuffer:
    """CPU stand-in for the CUDA global memory buffer used by pool allocators."""

    def __init__(self):
        self.buffers = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        required_len = math.prod(tensor_shape)
        key = (name, dtype)
        if key not in self.buffers or self.buffers[key].numel() < required_len:
            self.buffers[key] = torch.empty(required_len, dtype=dtype)
        return self.buffers[key][:required_len].view(*tensor_shape)


def _allocate(allocator, bucket_id):
    return allocator.allocate(
        bucket_id=bucket_id,
        size=8,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        strict_assignments=False,
    )


def _make_all_gather_pipeline(parameter_groups, allocator_by_bucket):
    for bucket_id, parameter_group in enumerate(parameter_groups):
        parameter_group.model_weight_buffer = SimpleNamespace(
            temporary_bucket_allocator=allocator_by_bucket[bucket_id]
        )
    buffer = SimpleNamespace(
        num_buckets=len(parameter_groups),
        parameter_groups=parameter_groups,
        bucket_to_bucket_group={
            bucket_id: [bucket_id] for bucket_id in range(len(parameter_groups))
        },
        dist_index=SimpleNamespace(use_hybrid_fsdp=False),
        ddp_config=SimpleNamespace(outer_dp_sharding_strategy="no_shard"),
    )
    return AllGatherPipeline(buffer)


@pytest.mark.parametrize("allocator_cls", [FixedPoolAllocator, MaxPoolAllocator])
def test_triple_buffer_pool_capacity_and_reuse(allocator_cls, monkeypatch):
    """Three live buckets fit, a fourth waits, and a freed slot is reused."""
    cpu_memory_buffer = _CpuMemoryBuffer()
    monkeypatch.setattr(
        param_and_grad_buffer, "get_global_memory_buffer", lambda: cpu_memory_buffer
    )
    allocator = allocator_cls("triple", _make_uniform_parameter_groups(), size=3)

    live_buckets = [_allocate(allocator, bucket_id) for bucket_id in range(3)]
    assert len({bucket.data.data_ptr() for bucket in live_buckets}) == 3
    assert not allocator.can_allocate([3])
    assert allocator.can_allocate([3], releasable_bucket_ids={0})

    released_data_ptr = live_buckets[0].data.data_ptr()
    allocator.free(0)
    replacement_bucket = _allocate(allocator, 3)
    assert replacement_bucket.data.data_ptr() == released_data_ptr


@pytest.mark.parametrize("allocator_cls", [FixedPoolAllocator, MaxPoolAllocator])
def test_all_gather_capacity_check_groups_allocators_and_lazy_releases(allocator_cls, monkeypatch):
    """Capacity prediction accounts for every pool and its pending lazy releases."""
    cpu_memory_buffer = _CpuMemoryBuffer()
    monkeypatch.setattr(
        param_and_grad_buffer, "get_global_memory_buffer", lambda: cpu_memory_buffer
    )
    parameter_groups = _make_uniform_parameter_groups(count=8)
    first_allocator = allocator_cls("first", parameter_groups, size=3)
    second_allocator = allocator_cls("second", parameter_groups, size=3)
    allocator_by_bucket = {
        bucket_id: first_allocator if bucket_id < 4 else second_allocator for bucket_id in range(8)
    }
    pipeline = _make_all_gather_pipeline(parameter_groups, allocator_by_bucket)

    for bucket_id in (0, 1, 2):
        _allocate(first_allocator, bucket_id)
    for bucket_id in (4, 5, 6):
        _allocate(second_allocator, bucket_id)

    assert not pipeline._persistent_allocators_can_fit([3, 7], bwd=False)

    pipeline.bucket_can_be_released[pipeline.get_bucket_key(0, False)] = True
    assert not pipeline._persistent_allocators_can_fit([3, 7], bwd=False)

    pipeline.bucket_can_be_released[pipeline.get_bucket_key(4, False)] = True
    assert pipeline._persistent_allocators_can_fit([3, 7], bwd=False)


def test_grouped_expert_weights_split_when_chunk_size_factors_differ():
    """Grouped expert weights with mismatched chunk size factors get routed to separate buckets."""
    num_local_experts = 4
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (num_local_experts, 2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (num_local_experts, hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc1"])[1:].numel(),
            "params": [("layer.experts.linear_fc1", shapes["linear_fc1"])],
        },
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc2"])[1:].numel(),
            "params": [("layer.experts.linear_fc2", shapes["linear_fc2"])],
        },
    ]


@pytest.mark.parametrize("use_decoupled_grad", [False, True])
def test_unused_gradient_survives_scaling_and_rebinding(use_decoupled_grad):
    """Skip unused Adam updates and rebind gradients when the parameter is used again."""
    original = torch.nn.Parameter(torch.ones(2))
    master = torch.nn.Parameter(torch.ones(2))
    master.orig_param = original
    master.grad = torch.ones_like(master)
    master.decoupled_grad = torch.ones_like(master)
    optimizer = torch.optim.AdamW([master], lr=0.1, weight_decay=0.1)
    optimizer.step()
    saved_weight = master.detach().clone()
    saved_state = {key: value.clone() for key, value in optimizer.state[master].items()}

    data = torch.zeros_like(master)
    grad_buffer = SimpleNamespace(data=data, param_idx={original: 0}, get_item=lambda *a, **k: data)
    buffer = ParamAndGradBuffer.__new__(ParamAndGradBuffer)
    buffer.ddp_config = SimpleNamespace(megatron_fsdp_track_grad_usage=True)
    buffer.optimizer_named_parameters = [("conditional", master)]
    buffer.parameter_groups = [
        SimpleNamespace(
            main_grad_buffer=grad_buffer, main_weight_buffer=None, sharding_strategy="no_shard"
        )
    ]
    buffer.param_to_param_group = {original: 0}
    buffer.dist_main_grad = {"conditional": data}
    buffer.use_decoupled_grad = use_decoupled_grad
    original._fsdp_has_global_grad = False

    buffer.update_main_grads()
    buffer.scale_gradients(0.5)
    buffer.update_main_grads()
    assert master.grad is None
    assert master.decoupled_grad is None
    optimizer.step()
    torch.testing.assert_close(master, saved_weight, rtol=0, atol=0)
    for key, value in saved_state.items():
        torch.testing.assert_close(optimizer.state[master][key], value, rtol=0, atol=0)

    original._fsdp_has_global_grad = True
    buffer.update_main_grads()
    gradient = master.decoupled_grad if use_decoupled_grad else master.grad
    assert gradient is not None
    torch.testing.assert_close(gradient, torch.zeros_like(master), rtol=0, atol=0)


def test_global_gradient_usage_includes_remote_only_parameters(monkeypatch):
    """Reduce flags in stable name order and retain gradients for remote-only usage."""
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
    local, remote, unused = parameters
    local._fsdp_local_grad_used = True
    remote._fsdp_local_grad_used = False
    unused._fsdp_local_grad_used = False
    masters = [SimpleNamespace(orig_param=param) for param in parameters]
    pool = object()

    def all_reduce(flags, *, group):
        assert group is pool
        assert flags.tolist() == [1, 0, 0]
        flags.add_(torch.tensor([0, 1, 0], dtype=flags.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    buffer = SimpleNamespace(
        parameter_groups=[SimpleNamespace(requires_grad=True, params=[unused, remote, local])],
        param_to_name=dict(zip(parameters, ["a", "b", "c"])),
        optimizer_named_parameters=list(zip(["a", "b", "c"], masters)),
        update_main_grads=lambda: None,
    )
    state = SimpleNamespace(param_and_grad_buffer=buffer, device="cpu", grad_usage_group=pool)
    MegatronFSDP._apply_global_grad_usage(state)
    assert local._fsdp_has_global_grad is True
    assert remote._fsdp_has_global_grad is True
    assert unused._fsdp_has_global_grad is False


class _ConditionalBranch(torch.nn.Module):
    """Distinct FSDP units with equal-size buffers expose mismatched collective order."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, 9, dtype=torch.float32))

    def forward(self, x):
        return (self.weight * x).sum()


class _ConditionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = _ConditionalBranch()
        self.b = _ConditionalBranch()
        self.c = _ConditionalBranch()

    def forward(self, x, branch, use_c):
        value = (self.a if branch == 0 else self.b)(x)
        return value + self.c(x * 0.25) if use_c else value


class _FusedWeightContribution(torch.autograd.Function):
    """Emulate fused output wgrad plus a dummy zero_out_wgrad autograd gradient."""

    @staticmethod
    def forward(ctx, weight, x):
        ctx.weight = weight
        ctx.save_for_backward(x)
        return (weight * x).sum()

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        weight = ctx.weight
        weight.main_grad = weight.get_main_grad()
        weight.main_grad.copy_(x * gradient)
        weight.grad_added_to_main_grad = True
        return torch.zeros_like(weight), None


class _TiedWeightModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, 9, dtype=torch.float32))
        self.weight.zero_out_wgrad = True
        self.output = torch.nn.Module()
        self.output.register_parameter("weight", self.weight)

    def forward(self, x):
        # The ordinary embedding path contributes in addition to fused output wgrad.
        return _FusedWeightContribution.apply(self.output.weight, x) + (self.weight * x * 2).sum()


def _grad_usage_test_wrapper(model, mesh, tracking):
    ddp = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads",
        overlap_grad_reduce=True,
        overlap_param_gather=False,
        megatron_fsdp_track_grad_usage=tracking,
        bucket_size=8,
    )
    ddp.finalize()
    return MegatronFSDP(
        model,
        ddp_config=ddp,
        mixed_precision_policy=MixedPrecisionPolicy(),
        fsdp_unit_modules=[_ConditionalBranch],
        device=torch.device("cuda", torch.cuda.current_device()),
        dist_index=FSDPDistributedIndex(mesh, dp_shard_dim="dp", tp_dim="tp"),
        grad_usage_group=mesh["dp"].get_group() if tracking else None,
        calculate_per_token_loss=True,
        sync_model_each_microbatch=False,
    )


@pytest.fixture(scope="module")
def grad_usage_process_group():
    """Reuse the suite group, or clean up a group owned by a focused torchrun invocation."""
    dist = torch.distributed
    owned = False
    if "RANK" in os.environ and torch.cuda.is_available() and not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
        owned = True
    yield
    if owned:
        dist.destroy_process_group()


@pytest.mark.parametrize("tracking", [False, True])
def test_grad_usage_distributed_numerics(grad_usage_process_group, distributed_setup, tracking):
    """Check conditional buckets, tied fused weights, accumulation, and AdamW numerically."""
    dist = torch.distributed
    if not torch.cuda.is_available() or distributed_setup.world_size < 2:
        pytest.skip("Requires at least two CUDA ranks under torchrun.")
    from torch.distributed.device_mesh import init_device_mesh

    size = dist.get_world_size()
    rank = dist.get_rank()
    device = distributed_setup.device
    mesh = init_device_mesh("cuda", (size, 1), mesh_dim_names=("dp", "tp"))
    models = [_TiedWeightModel()]
    if tracking:
        models.append(_ConditionalModel())
    for model in models:
        model = model.to(device)
        reference = deepcopy(model)
        wrapped = _grad_usage_test_wrapper(model, mesh, tracking)
        masters = dict(wrapped.param_and_grad_buffer.optimizer_named_parameters)
        optimizer = torch.optim.AdamW(list(masters.values()), lr=0.01, weight_decay=0.1)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1)
        is_conditional = isinstance(reference, _ConditionalModel)
        if is_conditional:
            buckets = wrapped.param_and_grad_buffer
            assert (
                len({tuple(buckets.bucket_to_bucket_group[i]) for i in range(buckets.num_buckets)})
                >= 3
            )
        for step in range(3):
            wrapped.zero_grad_buffer()
            ref_optimizer.zero_grad(set_to_none=True)
            wrapped.start_param_sync(force_sync=True)
            for microbatch in range(2):
                base = torch.arange(1, 9, device=device, dtype=torch.float32) * (microbatch + 1)
                x = base * (rank + 1)
                if is_conditional:
                    wrapped(x, (rank + microbatch + step) % 2, step != 1).div(size).backward()
                    # Independent unsharded global-batch reference, including absent branches.
                    for peer in range(size):
                        reference(base * (peer + 1), (peer + microbatch + step) % 2, step != 1).div(
                            size
                        ).backward()
                else:
                    wrapped(x).div(size).backward()
                    for peer in range(size):
                        (reference.weight * base * (peer + 1) * 3).sum().div(size).backward()
            wrapped.finish_grad_sync()
            for name, param in reference.named_parameters():
                master = masters[name]
                if param.grad is None:
                    assert master.grad is None
                else:
                    torch.testing.assert_close(
                        master.grad.full_tensor(), param.grad, rtol=1e-5, atol=1e-6
                    )
            optimizer.step()
            ref_optimizer.step()
            wrapped.install_optimized_model_weights()
            for name, param in reference.named_parameters():
                master = masters[name]
                torch.testing.assert_close(master.full_tensor(), param, rtol=1e-5, atol=1e-6)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    actual = optimizer.state[master][key]
                    if hasattr(actual, "full_tensor"):
                        actual = actual.full_tensor()
                    torch.testing.assert_close(
                        actual, ref_optimizer.state[param][key], rtol=1e-5, atol=1e-6
                    )
        wrapped.grad_reduce_pipeline.reset()
        wrapped.all_gather_pipeline.reset(preserve_non_fsdp_units=False)
    dist.barrier()


def test_per_expert_2d_weights_avoid_lcm_padding():
    """Incompatible row sizes stay separate, preserving main's bounded-padding fix."""
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc1"])[1:].numel(),
            "params": [("layer.experts.linear_fc1", shapes["linear_fc1"])],
        },
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc2"])[1:].numel(),
            "params": [("layer.experts.linear_fc2", shapes["linear_fc2"])],
        },
    ]
