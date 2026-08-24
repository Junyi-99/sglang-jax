"""Uniform-shape data generators for the KDA (Kimi Delta Attention) kernel.

``chunk_kda`` is varlen-only: it takes a B=1 packed layout plus ``cu_seqlens``,
so the logical batch size lives in ``len(seq_lens)`` rather than in a leading
tensor dimension. That is the same convention ``benchmark/kernels/gdn`` uses,
and the reason the sweep axes here are ``(num_seqs, seq_len)`` instead of
``(B, T)``.

Shapes produced:

  q, k:         [1, T_total, H, K]
  v:            [1, T_total, H, V]
  raw_g:        [1, T_total, H, K]   pre-activation gate projection
  beta:         [1, T_total, H]
  A_log:        [H]
  dt_bias:      [H, K]
  initial_state:[N, H, K, V]         fp32
  cu_seqlens:   i32[N + 1]

The gate helpers mirror ``kda_gate_chunk_cumsum``'s two activation paths (and
FLA's ``gate.py``) so the naive reference sees exactly what the kernel computes
internally when ``use_gate_in_kernel=True``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "activated_gate",
    "create_kda_uniform_data",
    "gate_stress_scale",
]


def create_kda_uniform_data(
    seq_lens: list[int],
    num_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    dtype=jnp.bfloat16,
    seed: int = 42,
    lower_bound: float | None = None,
) -> tuple[dict, int]:
    """Build reproducible varlen-packed KDA inputs for ``seq_lens``.

    ``lower_bound`` only affects the *scale* of ``raw_g``: the bounded gate is
    ``lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))``, which saturates
    unless raw_g spans a wide range. Widening it here is what drives the
    bounded-gate fast path over its full range instead of parking it near 0.
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(seq_lens)
    total_tokens = int(sum(seq_lens))
    H, K, V = num_heads, head_dim_k, head_dim_v

    def _unit_rows(shape):
        x = rng.standard_normal(shape)
        return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)

    q = jnp.asarray(_unit_rows((1, total_tokens, H, K)), dtype=dtype)
    k = jnp.asarray(_unit_rows((1, total_tokens, H, K)), dtype=dtype)
    v = jnp.asarray(rng.standard_normal((1, total_tokens, H, V)), dtype=dtype)

    raw_g = rng.standard_normal((1, total_tokens, H, K))
    if lower_bound is not None:
        raw_g = raw_g * gate_stress_scale()
    raw_g = jnp.asarray(raw_g, dtype=dtype)

    beta = jax.nn.sigmoid(
        jnp.asarray(rng.standard_normal((1, total_tokens, H)), dtype=jnp.float32)
    ).astype(dtype)

    A_log = jnp.asarray(-1.5 + 0.1 * rng.standard_normal((H,)), dtype=jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal((H, K)), dtype=jnp.float32)
    initial_state = jnp.asarray(rng.standard_normal((num_seqs, H, K, V)), dtype=jnp.float32)
    cu_seqlens = jnp.asarray([0] + np.cumsum(seq_lens).tolist(), dtype=jnp.int32)

    data = {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
    }
    return data, total_tokens


def gate_stress_scale() -> float:
    """Multiplier on raw_g for the bounded-gate path.

    Mirrors ``python/sgl_jax/test/kernels/kda_test.py``: ``exp(A_log) *
    (raw_g + dt_bias)`` spanning roughly +-20 puts the activated gate at ~0,
    lower_bound/2 and ~lower_bound.
    """
    return 8.0


def activated_gate(
    raw_g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    lower_bound: float | None = None,
) -> jax.Array:
    """Gate activation in log space, matching ``kda_gate_chunk_cumsum``.

    lower_bound=None -> ``-exp(A_log) * softplus(raw_g + dt_bias)``
    lower_bound=b    -> ``b * sigmoid(exp(A_log) * (raw_g + dt_bias))``
    """
    g_f32 = raw_g.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, None, :, :]
    a = jnp.exp(A_log.astype(jnp.float32))[None, None, :, None]
    if lower_bound is None:
        return -a * jax.nn.softplus(g_f32)
    return lower_bound * jax.nn.sigmoid(a * g_f32)
