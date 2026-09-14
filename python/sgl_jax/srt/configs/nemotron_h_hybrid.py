"""NemotronH hybrid-recurrent config adapter for sglang-jax.

``nemotron_h`` ships its own HF config class via ``trust_remote_code``. Rather
than re-declare it, we attach the duck-typed properties the sglang-jax runner
needs (``full_attention_layer_ids`` / ``linear_layer_ids`` /
``linear_state_params``) onto the loaded HF config object, plus a
``mamba2_state_pool_spec`` marker the patched ``_build_hybrid_pools`` reads to
build a non-square ``Mamba2StatePool``.

``get_nemotron_h_config(hf_config)`` returns the (mutated) config when it is a
nemotron_h config, else ``None`` -- mirroring ``get_kimi_linear_config`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sgl_jax.srt.configs.nemotron_h_layers import parse_layer_pattern
from sgl_jax.srt.mem_cache.recurrent_state_pool import (
    LinearRecurrentStateParams,
    recurrent_state_dtype,
)


@dataclass(frozen=True)
class Mamba2StatePoolSpec:
    """Shapes for building a Mamba2StatePool (non-square SSM state)."""

    layers: list[int]
    num_heads: int  # H
    head_dim: int  # P
    ssm_state_size: int  # N
    conv_dim: int  # d_inner + 2*n_groups*N
    conv_kernel_size: int  # K


def _attach_properties(cfg) -> None:
    pattern = cfg.hybrid_override_pattern
    layers = parse_layer_pattern(pattern)
    mamba_ids = layers.mamba_layer_ids
    attn_ids = layers.attention_layer_ids

    num_heads = int(cfg.mamba_num_heads)
    head_dim = int(cfg.mamba_head_dim)
    ssm_state = int(cfg.ssm_state_size)
    n_groups = int(cfg.n_groups)
    conv_k = int(cfg.conv_kernel)
    d_inner = num_heads * head_dim
    conv_dim = d_inner + 2 * n_groups * ssm_state

    # Duck-typed attributes the runner reads.
    cfg.full_attention_layer_ids = attn_ids
    cfg.linear_layer_ids = mamba_ids
    # num_hidden_layers already on cfg.

    # Memory-budget sizing block. The pool itself is built from
    # mamba2_state_pool_spec (square formula here is only for the byte estimate;
    # head_dim padded to ssm_state so the estimate over-counts, never under).
    state_head_dim = max(head_dim, ssm_state)
    cfg.linear_state_params = LinearRecurrentStateParams(
        layers=mamba_ids,
        num_heads=num_heads,
        head_dim=state_head_dim,
        conv_kernel_size=conv_k,
        dtype=recurrent_state_dtype(),
        # Make the square-formula conv proj_size >= true conv_dim so the byte
        # estimate (used only for HBM budgeting) never under-counts.
        num_k_heads=max(1, (conv_dim - num_heads * state_head_dim + 1) // (2 * state_head_dim) + 1),
        head_k_dim=state_head_dim,
    )

    cfg.mamba2_state_pool_spec = Mamba2StatePoolSpec(
        layers=mamba_ids,
        num_heads=num_heads,
        head_dim=head_dim,
        ssm_state_size=ssm_state,
        conv_dim=conv_dim,
        conv_kernel_size=conv_k,
    )


def get_nemotron_h_config(hf_config: Any):
    """Return hf_config with hybrid properties attached if nemotron_h, else None."""
    if getattr(hf_config, "model_type", None) != "nemotron_h":
        return None
    if not hasattr(hf_config, "mamba2_state_pool_spec"):
        _attach_properties(hf_config)
    return hf_config
