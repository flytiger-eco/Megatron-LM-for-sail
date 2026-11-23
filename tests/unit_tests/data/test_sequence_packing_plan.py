# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.data_schedule import SequencePackingPlan, build_sequence_packing_plan
from megatron.core.process_groups_config import ProcessGroupCollection


def group(size, rank=0, ranks=None):
    return SimpleNamespace(
        size=lambda: size, rank=lambda: rank, ranks=list(range(size)) if ranks is None else ranks
    )


def config(dynamic):
    return SimpleNamespace(
        sequence_packing_scheduler="default_dynamic_cp" if dynamic else "dp_balanced",
        dynamic_context_parallel=dynamic,
        max_seqlen_per_dp_cp_rank=64,
        min_dynamic_context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )


def groups(rank=0):
    return ProcessGroupCollection(
        tp=group(1),
        pp=group(1),
        dp=group(2),
        cp=group(2),
        dp_cp=group(4, rank),
        dynamic_cp={
            1: group(1, ranks=[rank]),
            2: group(2, rank % 2, [0, 1] if rank < 2 else [2, 3]),
            4: group(4, rank),
        },
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_plan_preserves_arbitrary_ids(dynamic):
    samples = [(100, 16), (9, 32), (801, 96), (44, 16), (13, 8)]
    plan = build_sequence_packing_plan(samples, config(dynamic), pg_collection=groups())
    counts = Counter()
    for stage in plan.assignments:
        assert len(stage) == 4
        for pack in set(stage):
            counts.update(pack)
    assert counts == Counter(gid for gid, _ in samples)
    assert plan.dynamic_cp is dynamic


def test_plan_uses_supplied_dynamic_groups(monkeypatch):
    pg = groups()
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda item: item.ranks)
    plan = build_sequence_packing_plan([(50, 96), (11, 16)], config(True), pg_collection=pg)
    for index, stage in enumerate(plan.assignments):
        runtime = plan.cp_group(index, pg)
        pack = stage[pg.dp_cp.rank()]
        assert runtime.size() == (sum(other == pack for other in stage) if pack else 1)
    assert pg.cp.size() == 2
    assert pg.dp_cp.size() == 4


@pytest.mark.parametrize("samples", [[], [(1, 0)], [(1, 16), (1, 32)]])
def test_invalid_samples_fail(samples):
    with pytest.raises(ValueError):
        build_sequence_packing_plan(samples, config(True), pg_collection=groups())


def test_plan_rejects_uninitialized_dynamic_groups():
    pg = groups()
    pg.dynamic_cp = None
    with pytest.raises(ValueError, match="dynamic_cp"):
        build_sequence_packing_plan([(1, 16)], config(True), pg_collection=pg)


def test_plan_rejects_oversized_sequence():
    with pytest.raises(ValueError):
        build_sequence_packing_plan([(1, 1024)], config(True), pg_collection=groups())


def test_empty_pack_uses_a_precreated_group_when_cp1_is_unavailable(monkeypatch):
    pg = groups(rank=2)
    pg.dynamic_cp = {size: group for size, group in pg.dynamic_cp.items() if size >= 2}
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda item: item.ranks)
    plan = SequencePackingPlan((((100,), (100,), (), ()),), True)
    assert plan.cp_group(0, pg) is pg.dynamic_cp[2]
