# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run MoELayer's BF16 PPU training interface without device kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.megakernel.te_ppu import TEPPUMegaMoE
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


def _expert_parameters(num_local_experts, config, **kwargs):
    """Mirror the per-expert TE parameter layout without initializing device libraries."""
    experts = torch.nn.Module()
    hidden, intermediate = config.hidden_size, config.moe_ffn_hidden_size
    shapes = {"linear_fc1": (2 * intermediate, hidden), "linear_fc2": (hidden, intermediate)}
    for name, shape in shapes.items():
        linear = torch.nn.Module()
        for i in range(num_local_experts):
            weight = torch.empty(shape, device="meta", dtype=torch.bfloat16)
            linear.register_parameter(f"weight{i}", torch.nn.Parameter(weight))
        experts.add_module(name, linear)
    return experts


@pytest.mark.parametrize("shared", [False, True])
def test_meta_training_call(monkeypatch, shared):
    pytest.importorskip("transformer_engine.pytorch.mega_moe")
    # Use the router's PyTorch GEMM fallback; TE cuBLAS requires a live device.
    monkeypatch.setattr(moe_utils, "te_general_gemm", None)
    group = Mock(spec=torch.distributed.ProcessGroup)
    group.group_name = "shape-demo"
    group.size.return_value = 1
    pg = ProcessGroupCollection(
        tp=group, ep=group, expt_tp=group, cp=group, tp_cp=group, tp_dp_cp=group
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_moe_experts=3,
        moe_router_topk=2,
        moe_ffn_hidden_size=8,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        use_cpu_initialization=True,
        perform_initialization=False,
        moe_router_load_balancing_type="none",
        moe_megakernel_backend="te_ppu_rtc",
        moe_shared_expert_intermediate_size=8 if shared else None,
    )
    submodules = MoESubmodules(
        experts=_expert_parameters,
        shared_experts=lambda **kwargs: torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
    )
    model = MoELayer(config, submodules, pg_collection=pg).to(device="meta")
    if shared:
        shared_forward = Mock(wraps=model.shared_experts.forward)
        monkeypatch.setattr(model.shared_experts, "forward", shared_forward)
    assert model.token_dispatcher is None
    assert not list(model.megakernel_experts.parameters())
    x = torch.empty(4, 2, 16, device="meta", dtype=torch.bfloat16, requires_grad=True)
    y, bias = model(x)
    assert bias is None and y.shape == x.shape and y.dtype == x.dtype
    y.sum().backward()
    if shared:
        shared_forward.assert_called_once_with(x)
    for tensor in (x, *model.parameters()):
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape and tensor.grad.dtype == tensor.dtype


def test_route_weights_follow_selected_ids(monkeypatch):
    """A selected expert with zero weight still receives a router gradient."""
    te = pytest.importorskip("transformer_engine.pytorch.mega_moe")
    config = SimpleNamespace(
        hidden_size=4, moe_ffn_hidden_size=2, moe_router_topk=2, num_moe_experts=3
    )
    experts = _expert_parameters(3, config)
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    probs = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.6, 0.3]], requires_grad=True)
    mask = torch.tensor([[False, False, True], [False, True, True]])

    def inspect_call(x, ids, weights, fc1, fc2, *, ep_group):
        torch.testing.assert_close(ids, torch.tensor([[2, -1], [1, 2]], dtype=torch.int32))
        torch.testing.assert_close(weights, torch.tensor([[0.0, 0.0], [0.6, 0.3]]))
        assert fc1[0] is experts.linear_fc1.weight0 and fc2[0] is experts.linear_fc2.weight0
        return x * weights.sum(dim=1, keepdim=True).bfloat16()

    monkeypatch.setattr(te, "bf16_mega_moe", inspect_call)
    backend = TEPPUMegaMoE(config, None, experts, 3)
    backend(x, probs, mask).sum().backward()
    torch.testing.assert_close(probs.grad, mask.float() * 4)
