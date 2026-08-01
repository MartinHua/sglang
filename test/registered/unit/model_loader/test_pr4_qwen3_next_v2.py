"""CPU-only checks for the Qwen3-Next weight-loader-v2 path (PR4 of #31051)."""

import sys
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

# Model imports transitively import optional kernel extension namespaces.  Stub
# them before importing any model so this module remains a lightweight CPU test.
sys.modules["sgl_kernel"] = MagicMock()
for _submodule in (
    "elementwise",
    "flash_attn",
    "flash_mla",
    "kvcacheio",
    "mamba",
    "quantization",
    "scalar_type",
    "sparse_flash_attn",
    "speculative",
    "utils",
):
    sys.modules[f"sgl_kernel.{_submodule}"] = MagicMock()

from sglang.srt.model_loader.auto_loader import (  # noqa: E402
    QWEN3_NEXT_GDN_STACKED_MAPPING,
    FusedExpertDispatch,
)
from sglang.srt.models.qwen3_next import (  # noqa: E402
    Qwen3HybridAttentionDecoderLayer,
    iter_qwen3_next_checkpoint_weights,
    remap_qwen3_next_checkpoint_name,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-b-test-cpu")


def _bare(cls):
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    return obj


def _install_param(module, name, loader):
    """Register ``name`` on ``module`` with a recording ``weight_loader``."""
    target = module
    parts = name.split(".")
    for part in parts[:-1]:
        if part not in target._modules:
            target.add_module(part, nn.Module())
        target = target._modules[part]
    param = nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = loader
    target.register_parameter(parts[-1], param)
    return param


def _shard_recording_param(recorded_shards):
    param = nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = lambda p, t, shard_id: recorded_shards.append(shard_id)
    return param


# ---------------------------------------------------------------------------
# Checkpoint name remapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,fusion,num_experts,expected",
    [
        # self_attn is folded directly onto the decoder layer.
        (
            "model.layers.0.self_attn.q_proj.weight",
            False,
            8,
            "model.layers.0.q_proj.weight",
        ),
        # Shared expert moves into the routed slot only when fusion is enabled.
        (
            "model.layers.3.mlp.shared_expert.gate_proj.weight",
            True,
            8,
            "model.layers.3.mlp.experts.8.gate_proj.weight",
        ),
        (
            "model.layers.3.mlp.shared_expert.gate_proj.weight",
            False,
            8,
            "model.layers.3.mlp.shared_expert.gate_proj.weight",
        ),
        # modelopt FP8 kv scales live on the attention module, not the proj.
        (
            "model.layers.1.self_attn.k_proj.k_scale",
            False,
            4,
            "model.layers.1.attn.k_scale",
        ),
        (
            "model.layers.1.self_attn.v_proj.v_scale",
            False,
            4,
            "model.layers.1.attn.v_scale",
        ),
    ],
)
def test_remap_checkpoint_name(name, fusion, num_experts, expected):
    assert (
        remap_qwen3_next_checkpoint_name(
            name,
            enable_shared_expert_fusion=fusion,
            num_experts=num_experts,
        )
        == expected
    )


def test_mtp_pass_keeps_only_draft_tensors():
    weights = [
        ("mtp.fc.weight", torch.zeros(1)),
        ("model.embed_tokens.weight", torch.zeros(1)),
        ("mtp.layers.0.mlp.gate_proj.weight", torch.zeros(1)),
    ]
    out = list(
        iter_qwen3_next_checkpoint_weights(
            weights,
            is_mtp=True,
            enable_shared_expert_fusion=False,
            num_experts=4,
        )
    )
    # "mtp.fc.weight" keeps its bare name; other mtp keys move onto model.*.
    assert [name for name, _ in out] == [
        "fc.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]


def test_base_pass_drops_draft_tensors():
    weights = [
        ("mtp.fc.weight", torch.zeros(1)),
        ("model.norm.weight", torch.zeros(1)),
        ("model.layers.0.rotary_emb.inv_freq", torch.zeros(1)),
    ]
    out = list(
        iter_qwen3_next_checkpoint_weights(
            weights,
            is_mtp=False,
            enable_shared_expert_fusion=False,
            num_experts=4,
        )
    )
    assert [name for name, _ in out] == ["model.norm.weight"]


def test_homeless_unit_scale_is_dropped_but_non_unit_scale_raises():
    unit = [("model.layers.0.q_proj.weight_scale", torch.ones(()))]
    assert (
        list(
            iter_qwen3_next_checkpoint_weights(
                unit,
                is_mtp=False,
                enable_shared_expert_fusion=False,
                num_experts=4,
                params_dict={},
            )
        )
        == []
    )

    off = [("model.layers.0.q_proj.weight_scale", torch.full((), 0.5))]
    with pytest.raises(AssertionError):
        list(
            iter_qwen3_next_checkpoint_weights(
                off,
                is_mtp=False,
                enable_shared_expert_fusion=False,
                num_experts=4,
                params_dict={},
            )
        )


# ---------------------------------------------------------------------------
# Packed gated-delta-net dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ckpt_name,expected_target,expected_shard",
    [
        ("in_proj_qkv.weight", "in_proj_qkvz.weight", (0, 1, 2)),
        ("in_proj_z.weight", "in_proj_qkvz.weight", 3),
        ("in_proj_b.weight", "in_proj_ba.weight", 0),
        ("in_proj_a.weight", "in_proj_ba.weight", 1),
    ],
)
def test_gdn_split_checkpoint_routes_to_fused_param(
    ckpt_name, expected_target, expected_shard
):
    recorded_shards = []
    params = {
        "in_proj_qkvz.weight": _shard_recording_param(recorded_shards),
        "in_proj_ba.weight": _shard_recording_param(recorded_shards),
    }
    target = QWEN3_NEXT_GDN_STACKED_MAPPING.try_load(ckpt_name, torch.zeros(1), params)
    assert target == expected_target
    assert recorded_shards == [expected_shard]


@pytest.mark.parametrize("ckpt_name", ["in_proj_qkvz.weight", "in_proj_ba.weight"])
def test_gdn_fused_checkpoint_is_not_captured_by_split_mapping(ckpt_name):
    """A fused checkpoint must fall through to the packed loader.

    ``in_proj_qkv.`` must not match ``in_proj_qkvz.`` and ``in_proj_b.`` must not
    match ``in_proj_ba.`` — the trailing dots in the mapping are what keep the
    fused name from being rewritten onto itself with a bogus shard id.
    """
    recorded_shards = []
    params = {ckpt_name: _shard_recording_param(recorded_shards)}
    assert (
        QWEN3_NEXT_GDN_STACKED_MAPPING.try_load(ckpt_name, torch.zeros(1), params)
        is None
    )
    assert recorded_shards == []


# ---------------------------------------------------------------------------
# Submodule hand-off
# ---------------------------------------------------------------------------


def test_attention_layer_forwards_mlp_weights_to_the_moe_loader():
    """Expert weights must reach the MoE block's own loader.

    The walker delegates a whole subtree to the first module defining
    ``load_weights``. Qwen3-Next inlines ``qkv_proj`` onto the decoder layer, so
    that layer needs a loader — and without an explicit hand-off the ``mlp.*``
    keys would be swallowed there and dropped instead of reaching expert
    dispatch.
    """
    layer = _bare(Qwen3HybridAttentionDecoderLayer)
    seen_by_mlp = []

    class _RecordingMoE(nn.Module):
        def load_weights(self, weights):
            names = [name for name, _ in weights]
            seen_by_mlp.extend(names)
            return set(names)

    layer.add_module("mlp", _RecordingMoE())
    qkv_shards = []
    _install_param(
        layer,
        "qkv_proj.weight",
        lambda p, t, shard_id: qkv_shards.append(shard_id),
    )

    loaded = layer.load_weights(
        [
            ("mlp.experts.0.gate_proj.weight", torch.zeros(1)),
            ("mlp.gate.weight", torch.zeros(1)),
            ("q_proj.weight", torch.zeros(1)),
        ]
    )

    assert seen_by_mlp == ["experts.0.gate_proj.weight", "gate.weight"]
    assert qkv_shards == ["q"]
    # Names come back re-prefixed so the caller's completeness view is correct.
    assert loaded == {
        "mlp.experts.0.gate_proj.weight",
        "mlp.gate.weight",
        "qkv_proj.weight",
    }


# ---------------------------------------------------------------------------
# Fused expert fan-out
# ---------------------------------------------------------------------------


def test_fused_expert_dispatch_fans_gate_up_into_w1_and_w3_per_expert():
    calls = []
    param = nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = lambda p, t, name, shard_id, expert_id: calls.append(
        (shard_id, expert_id)
    )
    params = {"experts.w13_weight": param}

    dispatch = FusedExpertDispatch(num_experts=2)
    target = dispatch.try_load("experts.gate_up_proj", torch.zeros(2, 4, 3), params)

    assert target == "experts.w13_weight"
    assert calls == [("w1", 0), ("w1", 1), ("w3", 0), ("w3", 1)]


def test_fused_expert_dispatch_rejects_missing_runtime_target():
    """Missing targets must fail before execution, not be silently skipped."""
    dispatch = FusedExpertDispatch(num_experts=2)
    with pytest.raises(ValueError, match="missing parameter"):
        dispatch.try_load("experts.gate_up_proj", torch.zeros(2, 4, 3), {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
