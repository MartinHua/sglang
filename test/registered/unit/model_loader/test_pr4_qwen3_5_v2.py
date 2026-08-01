"""CPU-only checks for the Qwen3.5 dense weight-loader-v2 path (PR4 of #31051)."""

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

from sglang.srt.models.qwen3_5 import (  # noqa: E402
    Qwen3_5AttentionDecoderLayer,
    iter_qwen3_5_text_checkpoint_weights,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-b-test-cpu")


def _names(weights):
    return [name for name, _ in weights]


def _stream(*names):
    return [(name, torch.zeros(1)) for name in names]


# ---------------------------------------------------------------------------
# Text-trunk selection and renaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ckpt_name,expected",
    [
        # The conditional-generation prefix is folded onto the text tree.
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.q_proj.weight",
        ),
        # self_attn is inlined onto the decoder layer.
        (
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.o_proj.weight",
        ),
        # ModelOpt kv scales move onto RadixAttention, via main's WeightsMapper.
        (
            "model.layers.2.self_attn.k_proj.k_scale",
            "model.layers.2.attn.k_scale",
        ),
        (
            "model.layers.2.self_attn.v_proj.v_scale",
            "model.layers.2.attn.v_scale",
        ),
    ],
)
def test_text_stream_renames(ckpt_name, expected):
    assert _names(iter_qwen3_5_text_checkpoint_weights(_stream(ckpt_name))) == [
        expected
    ]


@pytest.mark.parametrize(
    "dropped",
    [
        "mtp.layers.0.mlp.gate_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.layers.0.rotary_emb.inv_freq",
    ],
)
def test_non_text_streams_are_dropped(dropped):
    """MTP and the vision tower each own their own loader."""
    kept = "model.norm.weight"
    assert _names(iter_qwen3_5_text_checkpoint_weights(_stream(dropped, kept))) == [
        kept
    ]


def test_shared_expert_names_reach_the_moe_block_unrewritten():
    """Shared-expert keys must not be pre-renamed to the routed slot.

    The MoE block's loader maps ``shared_expert.*`` directly into the fused slot,
    so rewriting it to ``experts.<slot>.*`` here would produce a key that matches
    no expert mapping and be rejected as an unknown parameter.
    """
    name = "model.layers.3.mlp.shared_expert.gate_proj.weight"
    assert _names(iter_qwen3_5_text_checkpoint_weights(_stream(name))) == [name]


# ---------------------------------------------------------------------------
# Submodule hand-off
# ---------------------------------------------------------------------------


def test_attention_layer_forwards_mlp_weights_to_the_moe_loader():
    """Expert weights must reach the MLP block's own loader.

    The walker delegates a whole subtree to the first module defining
    ``load_weights``. Qwen3.5 inlines ``qkv_proj`` onto the decoder layer, so that
    layer needs a loader — and without an explicit hand-off the ``mlp.*`` keys
    would be swallowed there instead of reaching expert dispatch.
    """
    layer = Qwen3_5AttentionDecoderLayer.__new__(Qwen3_5AttentionDecoderLayer)
    nn.Module.__init__(layer)
    seen_by_mlp = []

    class _RecordingMoE(nn.Module):
        def load_weights(self, weights):
            names = _names(weights)
            seen_by_mlp.extend(names)
            return set(names)

    layer.add_module("mlp", _RecordingMoE())
    qkv_shards = []
    param = nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = lambda p, t, shard_id: qkv_shards.append(shard_id)
    qkv = nn.Module()
    qkv.register_parameter("weight", param)
    layer.add_module("qkv_proj", qkv)

    loaded = layer.load_weights(
        [
            ("mlp.experts.0.gate_proj.weight", torch.zeros(1)),
            ("q_proj.weight", torch.zeros(1)),
        ]
    )

    assert seen_by_mlp == ["experts.0.gate_proj.weight"]
    assert qkv_shards == ["q"]
    assert "mlp.experts.0.gate_proj.weight" in loaded
    assert "qkv_proj.weight" in loaded


# ---------------------------------------------------------------------------
# MTP wrapper routing
# ---------------------------------------------------------------------------


def test_mtp_splits_own_projections_from_the_draft_body():
    """The MTP wrapper keeps ``fc``/pre-fc norms and forwards the rest to the body.

    The draft shares a checkpoint with its target, so base-model keys must be
    dropped; and only the ``mtp.`` prefix may select the draft branch — a bare
    substring match could forward a malformed key to the body with a stale path.
    """
    mtp = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
    nn.Module.__init__(mtp)
    body_seen = []

    class _RecordingBody(nn.Module):
        def load_weights(self, weights):
            names = _names(weights)
            body_seen.extend(names)
            return set(names)

    mtp.add_module("model", _RecordingBody())
    for own in ("fc", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.zeros(1)))
        mtp.add_module(own, module)

    loaded = mtp._load_weights_v2(
        _stream(
            "mtp.fc.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.norm.weight",
            # Target-model weights share the stream and must not be loaded here.
            "model.layers.5.self_attn.q_proj.weight",
        )
    )

    assert body_seen == ["layers.0.self_attn.q_proj.weight", "norm.weight"]
    assert {"fc.weight", "pre_fc_norm_hidden.weight"} <= loaded
    # Body names come back re-prefixed so they match this module's param names.
    assert "model.layers.0.self_attn.q_proj.weight" in loaded
    assert not any(name.startswith("model.layers.5") for name in loaded)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
