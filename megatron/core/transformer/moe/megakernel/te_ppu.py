# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adapter for Transformer Engine's BF16 PPU MegaMoE API prototype."""

import torch

from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend


class TEPPUMegaMoE(MegakernelBackend):
    """Forward authoritative routes and the original per-expert parameters to TE."""

    def __init__(self, config, ep_group, routed_experts, num_local_experts):
        super().__init__()
        try:
            from transformer_engine.pytorch.mega_moe import bf16_mega_moe
        except ImportError as exc:
            raise ImportError("te_ppu_rtc requires TE with the BF16 MegaMoE API") from exc

        self._run = bf16_mega_moe
        self.ep_group = ep_group
        self.topk = config.moe_router_topk
        self.num_experts = config.num_moe_experts
        self.num_local_experts = num_local_experts
        # The enclosing MoELayer registers these parameters and owns their checkpoint keys.
        self._expert_modules = (routed_experts,)

    def forward(self, hidden_states, probs, routing_map):
        """Compact the router's selected expert IDs, retaining valid zero-weight routes."""
        x = hidden_states.reshape(-1, hidden_states.size(-1))
        if routing_map.dtype != torch.bool or routing_map.shape != (x.size(0), self.num_experts):
            raise ValueError("te_ppu_rtc requires a boolean [tokens, num_experts] routing map")
        if probs.shape != routing_map.shape:
            raise ValueError("router probabilities must have the same shape as the routing map")
        expert_ids = torch.arange(self.num_experts, device=x.device)
        routed_ids = torch.where(routing_map, expert_ids, self.num_experts)
        ids = routed_ids.topk(self.topk, largest=False).values
        missing = ids == self.num_experts
        weights = probs.gather(1, ids.masked_fill(missing, 0)).masked_fill(missing, 0).float()
        ids = ids.masked_fill(missing, -1).int()

        (experts,) = self._expert_modules
        names = [f"weight{i}" for i in range(self.num_local_experts)]
        fc1 = [experts.linear_fc1.get_parameter(name) for name in names]
        fc2 = [experts.linear_fc2.get_parameter(name) for name in names]
        output = self._run(x, ids, weights, fc1, fc2, ep_group=self.ep_group)
        return output.reshape_as(hidden_states)
