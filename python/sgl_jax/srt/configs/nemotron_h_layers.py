"""Nemotron-H hybrid config helpers (Mamba2 + GQA attention + MLP/MoE).

`nemotron_h` is a FLAT stack of single-mixer layers: each char of
``hybrid_override_pattern`` is ONE layer, not a (attn+mlp) block:

    M = Mamba2 SSM      * = GQA attention      - = dense MLP      E = MoE MLP

So ``num_hidden_layers == len(pattern)`` and each layer carries its own
input RMSNorm + residual around a single mixer (matches HF modeling_nemotron_h).

This module is pure-Python config parsing (no JAX) so it can be unit-tested off
the TPU. The model file (`models/nemotron_h.py`) consumes `layer_types()` to
build the right mixer per index and to declare which layers need a recurrent
(Mamba) state slot vs a KV slot vs neither.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical char meanings in hybrid_override_pattern.
MAMBA, ATTENTION, MLP, MOE = "M", "*", "-", "E"
_KNOWN = {MAMBA, ATTENTION, MLP, MOE}


@dataclass(frozen=True)
class NemotronHLayers:
    """Per-index layer-type map derived from hybrid_override_pattern."""

    pattern: str
    types: tuple[str, ...]          # one of M/*/-/E per layer index

    @property
    def num_layers(self) -> int:
        return len(self.types)

    @property
    def mamba_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.types) if t == MAMBA]

    @property
    def attention_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.types) if t == ATTENTION]

    @property
    def mlp_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.types) if t == MLP]

    @property
    def moe_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.types) if t == MOE]

    @property
    def ffn_layer_ids(self) -> list[int]:
        """All feed-forward layers (dense MLP + MoE)."""
        return [i for i, t in enumerate(self.types) if t in (MLP, MOE)]

    @property
    def is_moe(self) -> bool:
        return MOE in self.types


def parse_layer_pattern(pattern: str) -> NemotronHLayers:
    """Parse hybrid_override_pattern into a per-index layer map.

    Tolerates whitespace; rejects unknown chars so a config typo fails loudly
    instead of silently dropping layers.
    """
    types = tuple(c for c in pattern if not c.isspace())
    bad = sorted(set(types) - _KNOWN)
    if bad:
        raise ValueError(
            f"hybrid_override_pattern has unknown layer chars {bad}; "
            f"expected a subset of {sorted(_KNOWN)}"
        )
    return NemotronHLayers(pattern=pattern, types=types)


def layer_types(hf_config) -> NemotronHLayers:
    """Derive the layer map from a HF nemotron_h config object/dict."""
    pattern = _get(hf_config, "hybrid_override_pattern")
    if pattern is None:
        raise ValueError("nemotron_h config missing hybrid_override_pattern")
    layers = parse_layer_pattern(pattern)
    n = _get(hf_config, "num_hidden_layers")
    if n is not None and n != layers.num_layers:
        raise ValueError(
            f"num_hidden_layers={n} disagrees with pattern length "
            f"{layers.num_layers} ({pattern!r})"
        )
    return layers


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ---------------------------------------------------------------------------
# Self-check: parse the two shipped Nemotron-3-Nano patterns and assert the
# layer-type counts match the published architecture. Run: python -m
# sgl_jax.srt.configs.nemotron_h  (or: python nemotron_h.py)
# ---------------------------------------------------------------------------
def _demo() -> None:
    # 4B-BF16 (dense: MLP, no MoE) and 30B-A3B-BF16 (MoE), from HF config.json.
    p4 = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
    p30 = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

    l4 = parse_layer_pattern(p4)
    assert l4.num_layers == 42, l4.num_layers
    assert len(l4.mamba_layer_ids) == 21
    assert len(l4.attention_layer_ids) == 4
    assert len(l4.mlp_layer_ids) == 17
    assert len(l4.moe_layer_ids) == 0
    assert l4.is_moe is False
    # every layer is classified exactly once
    assert (len(l4.mamba_layer_ids) + len(l4.attention_layer_ids)
            + len(l4.ffn_layer_ids)) == l4.num_layers

    l30 = parse_layer_pattern(p30)
    assert l30.num_layers == 52, l30.num_layers
    assert len(l30.mamba_layer_ids) == 23
    assert len(l30.attention_layer_ids) == 6
    assert len(l30.moe_layer_ids) == 23
    assert len(l30.mlp_layer_ids) == 0
    assert l30.is_moe is True
    assert (len(l30.mamba_layer_ids) + len(l30.attention_layer_ids)
            + len(l30.ffn_layer_ids)) == l30.num_layers

    # num_hidden_layers cross-check + bad-char rejection
    assert layer_types({"hybrid_override_pattern": p4, "num_hidden_layers": 42}).num_layers == 42
    try:
        parse_layer_pattern("MXM")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on unknown char 'X'")

    print("nemotron_h config self-check OK  (4B: 21M/4*/17- ; 30B: 23M/6*/23E)")


if __name__ == "__main__":
    _demo()
