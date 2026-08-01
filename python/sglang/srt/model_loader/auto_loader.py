# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Centralized weight loading utilities for native SGLang models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Union

import msgspec
import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper

__all__ = [
    "AutoWeightsLoader",
    "WeightsMapper",
    "StackedParamsDispatch",
    "ExpertParamsDispatch",
    "FusedExpertDispatch",
    "STANDARD_QKV_MAPPING",
    "STANDARD_GATE_UP_MAPPING",
    "STANDARD_STACKED_MAPPING",
    "LLAMA_STACKED_MAPPING",
    "QWEN3_NEXT_GDN_STACKED_MAPPING",
    "QWEN35_GDN_STACKED_MAPPING",
    "QWEN35_STACKED_MAPPING",
    "MOE_EXPERT_STACKED_SKIP_SUBSTRS",
    "normalize_qwen35_weight_name",
    "split_submodule_weights",
    "try_load_stacked_skip_moe_experts",
    "load_with_stacked_dispatch",
    "load_moe_sparse_block_weights",
    "filter_pp_weights",
    "register_weight_remap",
    "get_weight_remap",
]


class StackedParamsDispatch(msgspec.Struct, frozen=True):
    mappings: tuple[tuple[str, str, Union[int, str]], ...] = ()

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        missing_target: str | None = None
        for fused_name, source_name, shard_id in self.mappings:
            if source_name not in name:
                continue
            target = name.replace(source_name, fused_name)
            param = params_dict.get(target)
            if param is None:
                if missing_target is None:
                    missing_target = target
                continue
            param.weight_loader(param, tensor, shard_id)
            return target
        if missing_target is not None:
            raise ValueError(
                f"Mapped checkpoint weight {name!r} to missing parameter "
                f"{missing_target!r}"
            )
        return None


STANDARD_QKV_MAPPING = StackedParamsDispatch(
    mappings=(
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )
)

STANDARD_GATE_UP_MAPPING = StackedParamsDispatch(
    mappings=(
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
)

STANDARD_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
)

LLAMA_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    )
)

# Qwen3-Next / Qwen3.5 gated-delta-net packing. The checkpoint stores the mixer
# projections split (in_proj_qkv / in_proj_z, in_proj_b / in_proj_a) while the
# runtime holds them fused. Trailing dots keep "in_proj_qkv." from matching
# "in_proj_qkvz." and "in_proj_b." from matching "in_proj_ba.".
QWEN3_NEXT_GDN_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
        ("in_proj_qkvz.", "in_proj_z.", 3),
        ("in_proj_ba.", "in_proj_b.", 0),
        ("in_proj_ba.", "in_proj_a.", 1),
    )
)

# Qwen3.5 reuses the Qwen3-Next mixer packing verbatim.
QWEN35_GDN_STACKED_MAPPING = QWEN3_NEXT_GDN_STACKED_MAPPING

QWEN35_STACKED_MAPPING = StackedParamsDispatch(
    mappings=STANDARD_STACKED_MAPPING.mappings + QWEN35_GDN_STACKED_MAPPING.mappings
)

MOE_EXPERT_STACKED_SKIP_SUBSTRS: tuple[str, ...] = ("mlp.experts", "experts.")


def try_load_stacked_skip_moe_experts(
    dispatch: StackedParamsDispatch,
    name: str,
    tensor: torch.Tensor,
    params_dict: dict[str, Parameter],
    *,
    skip_substrs: tuple[str, ...] = MOE_EXPERT_STACKED_SKIP_SUBSTRS,
) -> str | None:
    missing_target: str | None = None
    for fused_name, source_name, shard_id in dispatch.mappings:
        if source_name not in name:
            continue
        if any(skip in name for skip in skip_substrs):
            continue
        target = name.replace(source_name, fused_name)
        param = params_dict.get(target)
        if param is None:
            if missing_target is None:
                missing_target = target
            continue
        param.weight_loader(param, tensor, shard_id)
        return target
    if missing_target is not None:
        raise ValueError(
            f"Mapped checkpoint weight {name!r} to missing parameter "
            f"{missing_target!r}"
        )
    return None


class ExpertParamsDispatch(msgspec.Struct, frozen=True):
    mappings: tuple[tuple[str, str, int, str], ...] = ()

    @classmethod
    def from_fused_moe_mapping(
        cls,
        expert_params_mapping: list[tuple[str, str, int, str]],
    ) -> ExpertParamsDispatch:
        return cls(mappings=tuple(expert_params_mapping))

    @classmethod
    def from_gate_up_down(
        cls,
        *,
        num_experts: int,
        ckpt_gate_proj_name: str = "gate_proj",
        ckpt_down_proj_name: str = "down_proj",
        ckpt_up_proj_name: str = "up_proj",
    ) -> ExpertParamsDispatch:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        return cls.from_fused_moe_mapping(
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name=ckpt_gate_proj_name,
                ckpt_down_proj_name=ckpt_down_proj_name,
                ckpt_up_proj_name=ckpt_up_proj_name,
                num_experts=num_experts,
            )
        )

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        missing_target: str | None = None
        for param_name, weight_name, expert_id, shard_id in self.mappings:
            if weight_name not in name:
                continue
            target = name.replace(weight_name, param_name)
            param = params_dict.get(target)
            if param is None:
                if missing_target is None:
                    missing_target = target
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                tensor,
                target,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return target
        if missing_target is not None:
            raise ValueError(
                f"Mapped checkpoint expert weight {name!r} to missing "
                f"parameter {missing_target!r}"
            )
        return None


class FusedExpertDispatch(msgspec.Struct, frozen=True):
    """Fan one fused expert checkpoint tensor out to per-expert shard records.

    Qwen3.5-style checkpoints stack every expert into a single
    ``experts.gate_up_proj`` / ``experts.down_proj`` tensor whose leading dim is
    the expert index, while the runtime keeps ``w13_weight`` / ``w2_weight``
    tensors loaded one expert at a time. ``gate_up_proj`` additionally packs w1
    and w3 along the penultimate dim.
    """

    num_experts: int
    gate_up_ckpt_substr: str = "experts.gate_up_proj"
    down_ckpt_substr: str = "experts.down_proj"
    w13_runtime_substr: str = "experts.w13_weight"
    w2_runtime_substr: str = "experts.w2_weight"

    @staticmethod
    def fan_out_to_experts(
        param: Parameter,
        loaded_weight: torch.Tensor,
        runtime_name: str,
        shard_id: str,
        num_experts: int,
    ) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        for expert_id in range(num_experts):
            weight_loader(
                param,
                loaded_weight[expert_id],
                runtime_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )

    def _resolve(
        self,
        name: str,
        ckpt_substr: str,
        runtime_substr: str,
        params_dict: dict[str, Parameter],
    ) -> tuple[str, Parameter]:
        target = name.replace(ckpt_substr, runtime_substr)
        param = params_dict.get(target)
        if param is None:
            raise ValueError(
                f"Mapped fused expert weight {name!r} to missing parameter "
                f"{target!r}"
            )
        return target, param

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        if self.gate_up_ckpt_substr in name:
            target, param = self._resolve(
                name, self.gate_up_ckpt_substr, self.w13_runtime_substr, params_dict
            )
            w1, w3 = tensor.chunk(2, dim=-2)
            self.fan_out_to_experts(param, w1, target, "w1", self.num_experts)
            self.fan_out_to_experts(param, w3, target, "w3", self.num_experts)
            return target
        if self.down_ckpt_substr in name:
            target, param = self._resolve(
                name, self.down_ckpt_substr, self.w2_runtime_substr, params_dict
            )
            self.fan_out_to_experts(param, tensor, target, "w2", self.num_experts)
            return target
        return None


def normalize_qwen35_weight_name(name: str) -> str:
    """Strip Qwen3.5/Qwen3-Next wrapper prefixes the runtime tree does not have.

    ``model.language_model.`` appears in conditional-generation checkpoints, and
    the runtime folds ``self_attn`` directly into the decoder layer.
    """
    name = name.replace("model.language_model.", "model.")
    if ".self_attn." in name:
        name = name.replace(".self_attn", "")
    return name


def split_submodule_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    prefix: str,
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Partition checkpoint entries into ``(under_prefix, remainder)``.

    Names under ``prefix`` are yielded with the prefix stripped so they can be
    handed to that submodule's own ``load_weights``. Needed when a module owns
    both fused params of its own and a child that must keep its own loader: the
    walker delegates a whole subtree to the first module defining
    ``load_weights``, so the child has to be re-entered explicitly.
    """
    under_prefix: list[tuple[str, torch.Tensor]] = []
    remainder: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if name.startswith(prefix):
            under_prefix.append((name[len(prefix) :], tensor))
        else:
            remainder.append((name, tensor))
    return under_prefix, remainder


def load_with_stacked_dispatch(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    mapping: StackedParamsDispatch,
    *,
    ignore_unexpected_suffixes: tuple[str, ...] = (".bias", ".kv_scale"),
) -> set[str]:
    """Load submodule weights via stacked dispatch, then direct param loaders."""
    loaded: set[str] = set()
    params_dict = dict(module.named_parameters())
    for name, tensor in weights:
        if name.endswith(ignore_unexpected_suffixes):
            mapped_targets = (
                name.replace(source_name, fused_name)
                for fused_name, source_name, _ in mapping.mappings
                if source_name in name
            )
            if name not in params_dict and not any(
                target in params_dict for target in mapped_targets
            ):
                continue
        target = mapping.try_load(name, tensor, params_dict)
        if target is not None:
            if target in params_dict:
                loaded.add(target)
            continue
        if name.endswith("_scale") and name not in params_dict:
            if abs(tensor.item() - 1.0) >= 1e-6:
                raise AssertionError(
                    f"Expected unit scale 1.0, got {tensor.item()} for {name}"
                )
            continue
        if name in params_dict:
            wl = getattr(params_dict[name], "weight_loader", default_weight_loader)
            wl(params_dict[name], tensor)
            loaded.add(name)
        elif not any(name.endswith(suffix) for suffix in ignore_unexpected_suffixes):
            raise ValueError(
                f"No parameter named {name!r} in {module._get_name()}."
            )
    return loaded


def load_moe_sparse_block_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    expert_dispatch: ExpertParamsDispatch,
    dense_stacked: StackedParamsDispatch = STANDARD_GATE_UP_MAPPING,
    ignore_unexpected_suffixes: tuple[str, ...] = (".bias", "_bias", ".kv_scale"),
) -> set[str]:
    loaded: set[str] = set()
    params_dict = dict(module.named_parameters())
    for name, tensor in weights:
        if name.endswith(ignore_unexpected_suffixes):
            mapped_targets = [
                name.replace(source_name, fused_name)
                for fused_name, source_name, _ in dense_stacked.mappings
                if source_name in name
            ]
            mapped_targets.extend(
                name.replace(weight_name, param_name)
                for param_name, weight_name, _, _ in expert_dispatch.mappings
                if weight_name in name
            )
            if name not in params_dict and not any(
                target in params_dict for target in mapped_targets
            ):
                continue
        target = try_load_stacked_skip_moe_experts(
            dense_stacked, name, tensor, params_dict
        )
        if target is not None:
            if target in params_dict:
                loaded.add(target)
            continue
        target = expert_dispatch.try_load(name, tensor, params_dict)
        if target is not None:
            if target in params_dict:
                loaded.add(target)
            continue
        if name.endswith("_scale") and name not in params_dict:
            if abs(tensor.item() - 1.0) >= 1e-6:
                raise AssertionError(
                    f"Expected unit scale 1.0, got {tensor.item()} for {name}"
                )
            continue
        if name not in params_dict:
            raise ValueError(
                f"No parameter named {name!r} in {module._get_name()}."
            )
        wl = getattr(params_dict[name], "weight_loader", default_weight_loader)
        wl(params_dict[name], tensor)
        loaded.add(name)
    return loaded


def filter_pp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    start_layer: int,
    end_layer: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        layer_id = get_layer_id(name)
        if layer_id is not None and (layer_id < start_layer or layer_id >= end_layer):
            continue
        yield name, tensor


_REMAP_REGISTRY: dict[str, Callable[[nn.Module], WeightsMapper]] = {}


def register_weight_remap(*class_names: str):
    def decorator(fn: Callable[[nn.Module], WeightsMapper]):
        for cn in class_names:
            _REMAP_REGISTRY[cn] = fn
        return fn

    return decorator


def get_weight_remap(model: nn.Module) -> WeightsMapper | None:
    fn = _REMAP_REGISTRY.get(type(model).__name__)
    if fn is None:
        return None
    return fn(model)


@register_weight_remap("LlamaForCausalLM")
def _llama_remap(model: nn.Module) -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_suffix={
            ".activation_scale": ".input_scale",
            ".weight_scale_inv": ".weight_scale",
        }
    )
