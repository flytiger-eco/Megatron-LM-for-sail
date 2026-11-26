# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest

from megatron.core.distributed import DistributedDataParallelConfig as MCoreFSDPConfig
from megatron.core.distributed.fsdp.mcore_adapter_v1 import FullyShardedDataParallelV1
from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig as StandaloneFSDPConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    fully_shard,
    fully_shard_model,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP


def _make_config(config_cls, **kwargs):
    """Construct a config and apply the class's resolve-and-validate step.

    MCore's DistributedDataParallelConfig resolves defaults and validates in
    finalize() since #8205; the standalone Megatron-FSDP config still does both
    in __post_init__.
    """
    config = config_cls(**kwargs)
    if config_cls is MCoreFSDPConfig:
        config.finalize()
    return config


@pytest.mark.parametrize("config_cls", [MCoreFSDPConfig, StandaloneFSDPConfig])
def test_fsdp_persistent_buffer_config_validation(config_cls):
    """Persistent buffer capacity must agree with the pool enable flag."""
    assert _make_config(config_cls).fsdp_double_buffer is False

    config = _make_config(config_cls, fsdp_double_buffer=True, fsdp_buffer_count=3)
    assert config.fsdp_buffer_count == 3

    with pytest.raises(ValueError, match="must be at least 2"):
        _make_config(config_cls, fsdp_double_buffer=True, fsdp_buffer_count=1)

    with pytest.raises(ValueError, match="may only be changed"):
        _make_config(config_cls, fsdp_double_buffer=False, fsdp_buffer_count=3)


@pytest.mark.parametrize("config_cls", [MCoreFSDPConfig, StandaloneFSDPConfig])
@pytest.mark.parametrize("enabling_option", ["nccl_ub", "megatron_fsdp_max_pool_double_buffer"])
def test_fsdp_persistent_buffer_automatic_enablement(config_cls, enabling_option, monkeypatch):
    """Options that require persistent buffers enable the pool before validation."""
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    config = _make_config(config_cls, **{enabling_option: True, "fsdp_buffer_count": 3})
    assert config.fsdp_double_buffer is True


@pytest.mark.parametrize(
    ("api", "previous_last_parameter"),
    [
        (fully_shard_model, "maxpool_double_buffer"),
        (fully_shard, "maxpool_double_buffer"),
        (MegatronFSDP.__init__, "report_nan_in_param_grad"),
    ],
)
def test_fsdp_buffer_count_preserves_positional_api_compatibility(api, previous_last_parameter):
    """New options must follow all parameters accepted by the previous public API."""
    parameters = inspect.signature(api).parameters
    assert list(parameters).index("fsdp_buffer_count") > list(parameters).index(
        previous_last_parameter
    )
    assert parameters["fsdp_buffer_count"].default == 2


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("megatron_fsdp_cuda_graph_mode", "CUDA graph mode"),
        ("fsdp_double_buffer", "unpooled gradient buffers"),
        ("nccl_ub", "unpooled gradient buffers"),
        ("megatron_fsdp_max_pool_double_buffer", "unpooled gradient buffers"),
    ],
)
def test_grad_usage_rejects_graph_and_persistent_pools(option, message):
    """Tracking cannot retain every full gradient in a finite persistent pool."""
    with pytest.raises(ValueError, match=message):
        _make_config(
            MCoreFSDPConfig,
            use_megatron_fsdp=True,
            data_parallel_sharding_strategy="optim_grads",
            megatron_fsdp_track_grad_usage=True,
            **{option: True},
        )


def test_grad_usage_rejects_full_iteration_graph_before_model_setup():
    """Fail before capture or group setup rather than synchronizing flags during capture."""
    ddp = _make_config(
        MCoreFSDPConfig,
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads",
        megatron_fsdp_track_grad_usage=True,
    )
    cfg = SimpleNamespace(moe_virtual_expert_load_balance=False, cuda_graph_impl="full_iteration")
    with pytest.raises(ValueError, match="full-iteration CUDA graphs"):
        FullyShardedDataParallelV1(cfg, ddp, module=None)
