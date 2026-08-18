# Adapted from https://github.com/primatrix/pallas-kernel (rev 3c691ad3)
# Vendored to remove external dependency after the upstream repository went private.
#
# This file merges the following modules into a single file:
#   - tops/utils.py (cdiv, align_up, pad_to_multiple, prepare_lens, prepare_chunk_indices, assert_shape, assert_shape_or_none)
#   - tops/ops/utils.py (exp, exp2, get_interpret)
#   - tops/ops/common/cumsum.py (chunk_local_cumsum_vector via _chunk_cumsum_kernel)
#   - tops/ops/kda/chunk_intra_fwd.py (_solve_unit_lower_triangular, _kda_fwd_intra_kernel, kda_fwd_intra)
#   - tops/ops/common/chunk_delta_h.py (_prepare_chunk_offsets, _chunk_gated_delta_rule_fwd_kernel, chunk_gated_delta_rule_fwd_h)
#   - tops/ops/gla/chunk.py (_chunk_kda_fwd_o_gk_varlen_kernel renamed to _chunk_kda_fwd_o_gk_pl_kernel, chunk_kda_fwd_o_gk_varlen renamed to chunk_kda_fwd_o_gk)
#   - tops/ops/kda/gate.py (kda_gate_chunk_cumsum, pallas_kda_gate_cumsum)
#   - tops/ops/kda/chunk_fwd.py (_align_seqs, _unalign_output, chunk_kda_fwd)
"""KDA chunked forward pass for variable-length sequences (self-contained)."""

from __future__ import annotations

import functools
import math
import os
from functools import singledispatch

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

# ============================================================================
# Utilities
# ============================================================================


@singledispatch
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@cdiv.register
def _cdiv_jax(x: jax.Array, y: int):
    return (x + y - 1) // y


def align_up(x: int, align: int):
    return cdiv(x, align) * align


@singledispatch
def pad_to_multiple(x, multiple: int, axis: int, val):
    raise NotImplementedError(f"pad_to_multiple is not implemented for type {type(x)}")


@pad_to_multiple.register
def _pad_to_multiple_jax(x: jax.Array, multiple: int | list, axis: int | list, val):
    if isinstance(multiple, int):
        multiple = [multiple]
    if isinstance(axis, int):
        axis = [axis]
    assert len(multiple) == len(axis)
    shape = list(x.shape)
    pad_width = [(0, 0)] * len(shape)
    for idx in range(len(axis)):
        ax = axis[idx]
        mu = multiple[idx]
        remainder = shape[ax] % mu
        if remainder != 0:
            pad_width[ax] = (0, mu - remainder)
    return jnp.pad(x, pad_width, constant_values=val)


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    max_T: int | None = None,
) -> jax.Array:
    lens = prepare_lens(cu_seqlens)
    n_chunks = cdiv(lens, chunk_size)
    num_seqs = len(lens)
    total_nt = max_T // chunk_size
    seq_ids = jnp.repeat(
        jnp.arange(num_seqs, dtype=jnp.int32), n_chunks, total_repeat_length=total_nt
    )
    prefix_chunks = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(n_chunks)])
    seq_offsets = jnp.repeat(prefix_chunks[:-1], n_chunks, total_repeat_length=total_nt)
    block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets
    return jnp.stack([seq_ids, block_ids], axis=1)


def assert_shape_or_none(x, expected_shape, name="tensor"):
    if x is None:
        return
    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            if tensor is not None:
                curr_name = name[i] if has_names else f"{name}_{i}"
                assert (
                    tensor.shape == expected_shape
                ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def assert_shape(x, expected_shape, name="tensor"):
    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            curr_name = name[i] if has_names else f"{name}_{i}"
            assert (
                tensor.shape == expected_shape
            ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def exp(x):
    return jnp.exp(x.astype(jnp.float32))


def exp2(x):
    return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


# ============================================================================
# Chunk-local cumulative sum (varlen Pallas kernel only)
# ============================================================================

_VMEM_HW_LIMIT_BYTES = 30 * 1024 * 1024


def _chunk_cumsum_kernel(
    s_ref,
    o_ref,
    *,
    BT: int,
    REVERSE: bool,
    HAS_SCALE: bool,
    scale: float,
):
    num_steps = int(math.log2(BT))
    s = s_ref[:, 0, :, :].astype(jnp.float32)

    if REVERSE:
        for d in range(num_steps):
            stride = 1 << d
            top = s[:, : BT - stride, :] + s[:, stride:, :]
            bot = s[:, BT - stride :, :]
            s = jnp.concatenate([top, bot], axis=1)
    else:
        for d in range(num_steps):
            stride = 1 << d
            top = s[:, :stride, :]
            bot = s[:, stride:, :] + s[:, :-stride, :]
            s = jnp.concatenate([top, bot], axis=1)

    if HAS_SCALE:
        s = s * scale

    o_ref[:, 0, dslice(0, BT), :] = s.astype(o_ref.dtype)


def chunk_local_cumsum_vector(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
) -> jax.Array:
    assert g.ndim == 4, f"g must be 4-D, got {g.ndim}-D"
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be power of 2"
    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"

    BT = chunk_size
    BS = 128
    BB = 8

    if head_first:
        B, H, T, S = g.shape
        g_flat = g.reshape(B * H, T, S)
    else:
        B, T, H, S = g.shape
        g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

    BH = B * H
    out_dtype = output_dtype or g.dtype
    HAS_SCALE = scale is not None
    scale_val = scale if scale is not None else 1.0

    interpret = get_interpret()

    pad_S = (BS - (S % BS)) % BS
    if pad_S > 0:
        g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))
    S_padded = S + pad_S
    NS = S_padded // BS

    pad_BH = (BB - (BH % BB)) % BB
    if pad_BH > 0:
        g_flat = jnp.pad(g_flat, ((0, pad_BH), (0, 0), (0, 0)))
    BH_padded = BH + pad_BH

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NC_max = len(chunk_indices)

    g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))
    T_alloc = T + BT

    cu_i32 = cu_seqlens.astype(jnp.int32)
    chunk_indices_i32 = chunk_indices.astype(jnp.int32)
    N = cu_i32.shape[0] - 1
    chunks_per_seq = (jnp.diff(cu_i32) + BT - 1) // BT

    seq_id = chunk_indices_i32[:, 0]
    local_ci = chunk_indices_i32[:, 1]
    safe_seq_id = jnp.clip(seq_id, 0, N - 1)
    chunk_valid = (
        (seq_id == safe_seq_id) & (local_ci >= 0) & (local_ci < chunks_per_seq[safe_seq_id])
    )

    bos = cu_i32[safe_seq_id]
    eos = cu_i32[safe_seq_id + 1]
    chunk_starts = jnp.where(chunk_valid, bos + local_ci * BT, 0)
    positions = chunk_starts[:, None] + jnp.arange(BT, dtype=jnp.int32)[None, :]
    token_valid = chunk_valid[:, None] & (positions < eos[:, None]) & (positions < T)

    def _gather_chunk(start):
        return jax.lax.dynamic_slice(
            g_flat,
            (0, start, 0),
            (BH_padded, BT, S_padded),
        )

    g_chunks = jax.vmap(_gather_chunk)(chunk_starts).transpose(1, 0, 2, 3)
    g_chunks = jnp.where(token_valid[None, :, :, None], g_chunks, 0)

    elem_bytes = 4
    while BB > 1 and 4 * BB * BT * BS * elem_bytes > _VMEM_HW_LIMIT_BYTES:
        BB //= 2
    NBH = BH_padded // BB

    grid = (NS, NBH, NC_max)
    kernel = functools.partial(
        _chunk_cumsum_kernel,
        BT=BT,
        REVERSE=reverse,
        HAS_SCALE=HAS_SCALE,
        scale=scale_val,
    )

    def _index_map(i_s, i_bb, i_t):
        return (i_bb, i_t, 0, i_s)

    block_shape = (BB, 1, BT, BS)

    o_chunks = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec(block_shape=block_shape, index_map=_index_map)],
            out_specs=pl.BlockSpec(block_shape=block_shape, index_map=_index_map),
        ),
        out_shape=jax.ShapeDtypeStruct(g_chunks.shape, out_dtype),
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(g_chunks)

    o_chunks = jnp.where(token_valid[None, :, :, None], o_chunks, 0)
    sentinel = jnp.minimum(cu_i32[-1], T)
    scatter_positions = jnp.where(token_valid, positions, sentinel).reshape(-1)
    scatter_values = o_chunks.reshape(BH_padded, NC_max * BT, S_padded)

    o_flat = jnp.zeros((BH_padded, T_alloc, S_padded), dtype=o_chunks.dtype)
    o_flat = o_flat.at[:, scatter_positions, :].add(scatter_values)

    o_flat = o_flat[:BH, :T, :S]

    if head_first:
        return o_flat.reshape(B, H, T, S)
    else:
        return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# ============================================================================
# KDA intra-chunk forward (varlen only)
# ============================================================================


def _solve_unit_lower_triangular(A, b):
    N, D = b.shape
    BS = 16
    num_blocks = N // BS
    A = A.astype(jnp.float32)
    b = b.astype(jnp.float32)

    blocks = jnp.split(b, num_blocks, axis=0)

    for i in range(num_blocks):
        start = i * BS
        end = (i + 1) * BS
        A_ii = A[start:end, start:end]
        x_block = blocks[i]

        rows = [x_block[r] for r in range(BS)]
        for j in range(BS):
            if j > 0:
                vec = A_ii[j, :j][None, :]
                mat = jnp.stack(rows[:j])
                correction = jax.lax.dot_general(
                    vec,
                    mat,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).squeeze(axis=0)
                rows[j] = rows[j] - correction

        x_block = jnp.stack(rows)
        blocks[i] = x_block

        if i < num_blocks - 1:
            rest_start = (i + 1) * BS
            x_rest = jnp.concatenate(blocks[i + 1 :], axis=0)
            A_rest = A[rest_start:, start:end]
            update = jax.lax.dot_general(
                A_rest,
                x_block,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            x_rest = x_rest - update
            remaining = num_blocks - 1 - i
            new_blocks = jnp.split(x_rest, remaining, axis=0)
            for idx, nb in enumerate(new_blocks):
                blocks[i + 1 + idx] = nb

    return jnp.concatenate(blocks, axis=0)


def _neumann_fused_wide(L, z, BT):
    # K3-branch solve: L is strictly lower triangular => nilpotent (L^BT = 0),
    # so (I+L)^-1 equals the finite factorization (I-L)(I+L^2)...(I+L^{BT/2})
    # exactly. Propagate Z = [I | v_beta | k_eg_beta] through the factors
    # (they are polynomials in L and commute): log2(BT)-1 power dots plus
    # log2(BT) wide [BT,BT]@[BT,BT+V+K] dots that fill the MXU lane width.
    # Runs in one bf16 MXU pass: applying factors to the RHS never
    # materializes the explicit inverse, whose compounding rounding error is
    # what makes the DIRECT bf16 form fail the 5e-4 oracle gate (measured:
    # direct 6.9e-04 FAIL, fused 2.77e-04 OK). Do not "simplify" this into
    # inverse-then-multiply.
    z = z - jax.lax.dot(L, z, preferred_element_type=jnp.float32)
    Lp = L
    for _ in range(int(math.log2(BT)) - 1):
        Lp = jax.lax.dot(Lp, Lp, preferred_element_type=jnp.float32)
        z = z + jax.lax.dot(Lp, z, preferred_element_type=jnp.float32)
    return z


def _intra_head_math(
    q,
    k,
    g,
    beta,
    v,
    a_vec,
    bias_vec,
    *,
    chunk_size,
    head_dim,
    value_dim,
    scale,
    safe_gate,
    APPLY_GATE,
    LOWER_BOUND,
    PRE_CUMSUM,
    WANT_AINV,
):
    """Compute one head and one chunk of the intra stage.

    This pure function is shared by both kernel layouts. Inputs are q/k/g
    [BT,K], beta [BT,1], v [BT,V], and optional a_vec/bias_vec [K]. It returns
    u [BT,V], w [BT,K], kg [BT,K], Aqk [BT,BT], optional A_inv [BT,BT], and
    fp32 g_cum [BT,K]. With WANT_AINV=False (the inference default), the
    safe_gate fused-wide RHS omits the identity block (320 to 256 columns), so
    (I+L)^-1 is neither computed nor returned.
    """
    dtype = q.dtype

    BT = chunk_size  # 64

    # ---- Fused stage 1: gate activation + chunk-local cumsum (log2 domain) ----
    g_f32 = g.astype(jnp.float32)
    if not PRE_CUMSUM:
        if APPLY_GATE:
            b_a = a_vec.astype(jnp.float32)  # [K] exp(A_log[h]), broadcast per channel
            b_bias = bias_vec.astype(jnp.float32)  # [K] dt_bias[h]
            if LOWER_BOUND is None:
                g_f32 = -b_a * jax.nn.softplus(g_f32 + b_bias)
            else:
                g_f32 = LOWER_BOUND * jax.nn.sigmoid(b_a * (g_f32 + b_bias))
        # Chunk-local prefix sum via a Hillis-Steele doubling scan with
        # log2(BT) shift-and-add steps. Pallas TPU does not lower jnp.cumsum.
        num_steps = int(math.log2(BT))
        assert (1 << num_steps) == BT, "chunk_size must be a power of 2 for the in-kernel scan"
        for d in range(num_steps):
            stride = 1 << d
            top = g_f32[:stride, :]
            bot = g_f32[stride:, :] + g_f32[:-stride, :]
            g_f32 = jnp.concatenate([top, bot], axis=0)
        g_f32 = g_f32 * _RCP_LN2  # [64, 128], convert to the log2 domain
    g_cum = g_f32
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32)

    # Build Aqk and L directly using exp2(g[i] - g[j]).
    # For causal (i >= j): g_cumsum[i] <= g_cumsum[j], so g[i]-g[j] <= 0,
    # giving exp2 in (0, 1].  This avoids the split-normalization overflow
    # that occurs with exp2(g-gn) when per-step gate changes exceed ~127.
    causal_bt = jnp.tril(
        jnp.ones((BT, BT), dtype=jnp.float32)
    )  # [64, 64], i >= j, including diagonal, for Aqk
    strict_bt = jnp.tril(
        jnp.ones((BT, BT), dtype=jnp.float32), k=-1
    )  # [64, 64], i > j, excluding diagonal, for L

    if safe_gate:
        # safe_gate path: Aqk/L become BT/16 per-sub-chunk GEMMs
        # [BT,K]@[K,16] on the MXU instead of a [BT,BT,K] elementwise tensor
        # on the VPU ([16,16,128]).
        SB = 16
        aqk_subchunks, l_subchunks = [], []  # Four [64, 16] blocks each
        for blk in range(BT // SB):
            cols = slice(blk * SB, (blk + 1) * SB)
            r_b = g_f32[blk * SB + SB // 2 : blk * SB + SB // 2 + 1, :]  # [1, K] = [1, 128]
            row = exp2(g_f32 - r_b)  # [64,128] - [1,128], broadcast to [BT, K]
            col = k_f32[cols] * exp2(r_b - g_f32[cols])  # [SB, K]
            #     [16,128]    * exp2([1,128] - [16,128] -> [16,128]) -> [16, 128]
            # Column factor k[j] * 2^(r-g[j]), restricted to this 16-column block.

            aqk_subchunks.append(
                jax.lax.dot_general(
                    q_f32 * row,  # [BT=64, K=128] * [BT=64, K=128]
                    col,  # [BT=64, K=128]
                    (
                        ((1,), (1,)),  # Contract dimension 1 (K=128) on both operands.
                        ((), ()),  # No batch dimensions.
                    ),
                    preferred_element_type=jnp.float32,
                )
            )
            # lhs = q_f32 * row: elementwise [64, 128]. dot_general contracts
            # dimension 1 (K) on both operands: [64,128] x [16,128] -> [64,16].
            # This is the MXU GEMM for Aqk_block[i, j_local].

            l_subchunks.append(
                jax.lax.dot_general(
                    k_f32 * row,  # [64,128] * [64,128]
                    col,  # [64,128]
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            )
            # Same contraction with lhs = k_f32 * row: [64,128] -> [64,16].

        o_i = jnp.arange(BT, dtype=jnp.int32)
        # Aqk[i, j] = scale * sum_k q[i,k] * k[j,k] * exp2(g[i,k] - g[j,k])
        Aqk = jnp.where(
            o_i[:, None] >= o_i[None, :], scale * jnp.concatenate(aqk_subchunks, axis=-1), 0.0
        )
        # L[i, j] = sum_k k[i,k] * k[j,k] * exp2(g[i,k] - g[j,k])   (i > j)
        L = jnp.where(o_i[:, None] > o_i[None, :], jnp.concatenate(l_subchunks, axis=-1), 0.0)
    else:
        # g_diff[i, j, k] = g[i, k] - g[j, k];  shape [BT, BT, K]
        g_diff = g_f32[:, None, :] - g_f32[None, :, :]
        #            [64,1,128]    -   [1, 64, 128] --> [64, 64, 128]

        # Mask anti-causal entries to -126 before exp2 to prevent overflow;
        # they will be zeroed by causal_bt / strict_bt anyway.
        g_diff = jnp.where(causal_bt[:, :, None] > 0, g_diff, -126.0)
        # Broadcast [64,64,1] to [64,64,128]. Anti-causal entries (i < j)
        # have positive g_diff, so fill them with -126 before exp2 to avoid overflow.

        decay = exp2(
            jnp.maximum(g_diff, -126.0)
        )  # [BT, BT, K], per-channel decay from position j to i

        # Aqk[i, j] = scale * sum_k q[i,k] * k[j,k] * decay[i,j,k]
        Aqk = scale * jnp.sum(q_f32[:, None, :] * decay * k_f32[None, :, :], axis=-1)

        # L[i, j] = beta[i] * sum_k k[i,k] * k[j,k] * decay[i,j,k]   (i > j)
        L = jnp.sum(k_f32[:, None, :] * decay * k_f32[None, :, :], axis=-1)

    Aqk = (Aqk * causal_bt).astype(dtype)
    L = L * beta_f32 * strict_bt

    v_beta = v.astype(jnp.float32) * beta_f32
    k_eg_beta = k_f32 * exp2(g_f32) * beta_f32
    identity = jnp.eye(BT, dtype=jnp.float32)

    if safe_gate:
        # Kimi-K3 special branch (bounded gate, lower_bound validated in
        # [-5, 0)): fused-wide finite Neumann, one bf16 MXU pass.  (PR#2)
        parts = ([identity] if WANT_AINV else []) + [v_beta, k_eg_beta]
        z = jnp.concatenate(parts, axis=-1)
        z = _neumann_fused_wide(L, z, BT)
        off = BT if WANT_AINV else 0
        A_inv = z[:, :BT] if WANT_AINV else None
        u = z[:, off : off + value_dim]
        w = z[:, off + value_dim :]
    else:
        # General KDA: sequential forward substitution (upstream original).
        parts = [v_beta, k_eg_beta] + ([identity] if WANT_AINV else [])
        combined_b = jnp.concatenate(parts, axis=-1)
        combined_x = _solve_unit_lower_triangular(L, combined_b)

        u = combined_x[:, :value_dim]
        w = combined_x[:, value_dim : value_dim + head_dim]
        A_inv = combined_x[:, value_dim + head_dim :] if WANT_AINV else None

    g_last = g_f32[BT - 1 : BT, :]
    kg = k_f32 * exp2(g_last - g_f32)

    return u, w, kg, Aqk, A_inv, g_cum


# Unified-addressing layout: arrays are [1, H, T_alloc, D], grid=(H, NC),
# and block c maps directly to rows [c*BT, (c+1)*BT). Each kernel ref is a
# [1, 1, BT, D] block selected by BlockSpec.
def _kda_fwd_intra_kernel(
    q_ref,  # [1, 1, BT, K] = [1, 1, 64, 128]
    k_ref,  # [1, 1, BT, K]
    g_ref,  # [1, 1, BT, K]
    beta_ref,  # [1, 1, BT, 1]
    v_ref,  # [1, 1, BT, V]
    a_ref,  # [1, 1, 1, K] exp(A_log), broadcast per head; None if APPLY_GATE=False
    bias_ref,  # [1, 1, 1, K] dt_bias; None if APPLY_GATE=False
    u_out_ref,  # [1, 1, BT, V]
    w_out_ref,  # [1, 1, 1, BT, K]
    qg_out_ref,  # [1, 1, 1, BT, V]
    kg_out_ref,  #
    Aqk_out_ref,  #
    Akk_inv_out_ref,  #
    g_cum_out_ref,  # [1, 1, 1, BT, K] fp32 fused stage-1 output for stages 3+4
    *,
    chunk_size,  # 64
    head_dim,  # K=128
    value_dim,  # V=128
    scale,
    disable_recompute,
    safe_gate,
    APPLY_GATE,  # Whether fused stage 1 applies gate activation in-kernel
    LOWER_BOUND,  # None -> -exp(A)*softplus; float -> lb*sigmoid
    PRE_CUMSUM,  # True for fuse=False: g is already the stage-1 log2 prefix sum
):
    # q_r, k_r, g_r, v_r:
    # beta_r: [B, H, N, D, 1]
    a_vec = a_ref[0, 0, 0] if APPLY_GATE else None
    bias_vec = bias_ref[0, 0, 0] if APPLY_GATE else None
    u, w, kg, Aqk, A_inv, g_cum = _intra_head_math(
        q_ref[0, 0],
        k_ref[0, 0],
        g_ref[0, 0],
        beta_ref[0, 0],
        v_ref[0, 0],
        a_vec,
        bias_vec,
        chunk_size=chunk_size,
        head_dim=head_dim,
        value_dim=value_dim,
        scale=scale,
        safe_gate=safe_gate,
        APPLY_GATE=APPLY_GATE,
        LOWER_BOUND=LOWER_BOUND,
        PRE_CUMSUM=PRE_CUMSUM,
        WANT_AINV=True,
    )
    g_cum_out_ref[0, 0] = g_cum.astype(g_cum_out_ref.dtype)
    u_out_ref[0, 0] = u.astype(u_out_ref.dtype)
    w_out_ref[0, 0] = w.astype(w_out_ref.dtype)
    if disable_recompute:
        # qg only exists to let backward skip recomputation; otherwise the
        # output slot is None and nothing is written (saves a full HBM store).
        qg = q_ref[0, 0].astype(jnp.float32) * exp2(g_cum)
        qg_out_ref[0, 0] = qg.astype(qg_out_ref.dtype)
    kg_out_ref[0, 0] = kg.astype(kg_out_ref.dtype)
    Aqk_out_ref[0, 0] = Aqk.astype(Aqk_out_ref.dtype)
    Akk_inv_out_ref[0, 0] = A_inv.astype(Akk_inv_out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "scale",
        "safe_gate",
        "disable_recompute",
        "use_gate_in_kernel",
        "lower_bound",
        "unified_layout",
        "pre_cumsum",
    ],
)
def kda_fwd_intra(
    q,
    k,
    v,
    gk,
    beta,
    scale,
    chunk_size=64,
    safe_gate=False,
    disable_recompute=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    unified_layout=True,
    pre_cumsum=False,
    cu_seqlens=None,
):
    """Run intra stage K1 with either addressing mode.

    With unified_layout=True, inputs and outputs use [1, H, T, D]. After
    _align_seqs, global chunk c maps directly to rows [c*BT, (c+1)*BT), so
    no gather or scatter is needed. With unified_layout=False, inputs and
    outputs use the legacy [1, T, H, D] layout: chunk_starts are derived from
    cu_seqlens, data is gathered into blocks, then scattered back along T.

    With pre_cumsum=True (the fuse=False ablation path), gk already contains
    the stage-1 log2 prefix sum, so activation and scanning are skipped here.
    """
    BT = chunk_size
    assert BT >= 16 and BT % 16 == 0

    if unified_layout:
        B, H, T_u, K = q.shape
        V = v.shape[-1]
        assert B == 1 and T_u % BT == 0
        NC = T_u // BT
        q4, k4, g4, beta4, v4 = q, k, gk, beta, v
    else:
        B, T, H, K = q.shape
        V = v.shape[-1]
        assert B == 1
        assert cu_seqlens is not None, "unified_layout=False needs cu_seqlens for gather"
        N = cu_seqlens.shape[0] - 1
        T_alloc = T + BT

        pad4d = lambda x: jnp.pad(x, ((0, 0), (0, BT), (0, 0), (0, 0)))
        q_pad, k_pad, g_pad, v_pad = pad4d(q), pad4d(k), pad4d(gk), pad4d(v)
        beta_pad = jnp.pad(beta.reshape(B, T, H, 1), ((0, 0), (0, BT), (0, 0), (0, 0)))

        cu_i32 = cu_seqlens.astype(jnp.int32)
        chunks_per_seq = (jnp.diff(cu_i32) + BT - 1) // BT
        cum_chunks = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
        total_chunks = cum_chunks[-1]
        NC = T // BT + N  # Static upper bound; extra programs read chunk 0 and write the trash row.
        flat_idx = jnp.arange(NC, dtype=jnp.int32)
        is_valid = flat_idx < total_chunks
        seq_id = jnp.minimum(jnp.searchsorted(cum_chunks[1:], flat_idx, side="right"), N - 1)
        local_ci = flat_idx - cum_chunks[seq_id]
        chunk_starts = jnp.where(is_valid, cu_i32[seq_id] + local_ci * BT, 0)

        def gather(x_pad, D):
            def extract(start):
                return jax.lax.dynamic_slice(x_pad, (0, start, 0, 0), (1, BT, H, D))[0]

            return jax.vmap(extract)(chunk_starts)  # [NC, BT, H, D]

        def to4(x_c):  # [NC, BT, H, D] -> the unified [1, H, NC*BT, D] kernel layout
            return x_c.transpose(2, 0, 1, 3).reshape(1, H, NC * BT, x_c.shape[3])

        q4 = to4(gather(q_pad, K))
        k4 = to4(gather(k_pad, K))
        g4 = to4(gather(g_pad, K))
        beta4 = to4(gather(beta_pad, 1))
        v4 = to4(gather(v_pad, V))

    # Per-head constants for fused stage 1: exp(A_log) and dt_bias,
    # broadcast to [1,H,1,K].
    if use_gate_in_kernel and not pre_cumsum:
        assert A_log is not None
        a_r = jnp.broadcast_to(
            jnp.exp(A_log.astype(jnp.float32))[None, :, None, None], (1, H, 1, K)
        )
        db = (
            jnp.zeros((H, K), jnp.float32)
            if dt_bias is None
            else dt_bias.astype(jnp.float32).reshape(H, K)
        )
        bias_r = db[None, :, None, :]
        gate_spec = pl.BlockSpec(block_shape=(1, 1, 1, K), index_map=lambda h, c: (0, h, 0, 0))
        apply_gate = True
    else:
        a_r, bias_r, gate_spec = None, None, None
        apply_gate = False

    def _spec(last_dim):
        return pl.BlockSpec(block_shape=(1, 1, BT, last_dim), index_map=lambda h, c: (0, h, c, 0))

    dt = q4.dtype
    TB = NC * BT  # T dimension of kernel arrays; equals T_u with unified addressing.
    u4, w4, qg4, kg4, Aqk4, Akk4, g_cum4 = pl.pallas_call(
        functools.partial(
            _kda_fwd_intra_kernel,
            chunk_size=BT,
            head_dim=K,
            value_dim=V,
            scale=scale,
            disable_recompute=disable_recompute,
            safe_gate=safe_gate,
            APPLY_GATE=apply_gate,
            LOWER_BOUND=lower_bound,
            PRE_CUMSUM=pre_cumsum,
        ),
        interpret=get_interpret(),
        out_shape=[
            jax.ShapeDtypeStruct((1, H, TB, V), dt),
            jax.ShapeDtypeStruct((1, H, TB, K), dt),
            jax.ShapeDtypeStruct((1, H, TB, K), dt) if disable_recompute else None,
            jax.ShapeDtypeStruct((1, H, TB, K), dt),
            jax.ShapeDtypeStruct((1, H, TB, BT), dt),
            jax.ShapeDtypeStruct((1, H, TB, BT), dt),
            jax.ShapeDtypeStruct((1, H, TB, K), jnp.float32),
        ],
        in_specs=[_spec(K), _spec(K), _spec(K), _spec(1), _spec(V), gate_spec, gate_spec],
        out_specs=[
            _spec(V),
            _spec(K),
            _spec(K) if disable_recompute else None,
            _spec(K),
            _spec(BT),
            _spec(BT),
            _spec(K),
        ],
        grid=(H, NC),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(q4, k4, g4, beta4, v4, a_r, bias_r)

    if unified_layout:
        return w4, u4, qg4, kg4, Aqk4, Akk4, g_cum4

    # Scatter back to [1, T, H, D] for the legacy layout.
    pos = chunk_starts[:, None] + jnp.arange(BT)[None, :]
    pos = jnp.where(is_valid[:, None], pos, T_alloc - 1)
    flat_pos = pos.reshape(-1)

    def scat(x4, D):
        chunks = x4.reshape(H, NC, BT, D).transpose(1, 2, 0, 3).reshape(-1, H, D)
        out = jnp.zeros((T_alloc, H, D), dtype=x4.dtype)
        out = out.at[flat_pos].add(chunks)
        return out[:T][None]

    w_out, u_out, kg_out = scat(w4, K), scat(u4, V), scat(kg4, K)
    Aqk_out, Akk_out = scat(Aqk4, BT), scat(Akk4, BT)
    g_cum_out = scat(g_cum4, K)
    qg_out = scat(qg4, K) if disable_recompute else None
    return w_out, u_out, qg_out, kg_out, Aqk_out, Akk_out, g_cum_out


# ============================================================================
# Delta-rule inter-chunk state propagation (varlen only)
# ============================================================================


def _prepare_chunk_offsets(seqlens, chunk_size):
    return jnp.pad(
        cdiv(jnp.diff(seqlens), chunk_size).astype(jnp.int32),
        (1, 0),
        constant_values=0,
    ).cumsum(-1)


def _chunk_gated_delta_rule_fwd_kernel(
    seqlens_ref,
    chunk_offsets_ref,
    k_ref,
    v_ref,
    w_ref,
    g_ref,
    gk_ref,
    h0_ref,
    h_ref,
    v_new_ref,
    ht_ref,
    scratch_ref,
    *,
    NT,
    USE_G,
    USE_GK,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    SAVE_NEW_VALUE,
    USE_EXP2,
):
    idx_n = pl.program_id(0)
    idx_nt = pl.program_id(2)

    bos = seqlens_ref[idx_n]
    eos = seqlens_ref[idx_n + 1]
    real_NT = (eos - bos) // k_ref.shape[2]

    BT = k_ref.shape[2]
    K, V = k_ref.shape[-1], v_ref.shape[-1]
    b_k = k_ref[0, 0]

    @pl.when(idx_nt == 0)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    @pl.when(idx_nt < real_NT)
    def _():
        h_ref[0, 0, 0] = scratch_ref[...].astype(h_ref.dtype)

        b_w = w_ref[0, 0]
        b_v = jnp.dot(
            b_w.astype(jnp.float32),
            scratch_ref[...],
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        b_u = v_ref[0, 0]
        b_v = b_u.astype(b_v.dtype) - b_v
        if SAVE_NEW_VALUE:
            v_new_ref[0, 0] = b_v.astype(v_new_ref.dtype)

        if USE_G:
            b_g = g_ref[0, 0, :, 0]
            b_g_last = g_ref[0, 0, BT - 1, 0].astype(jnp.float32)
            if USE_EXP2:
                b_v = b_v * exp2(b_g_last - b_g)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * exp(b_g_last - b_g)[:, None]
                b_g_last = exp(b_g_last)
            scratch_ref[...] *= b_g_last
        if USE_GK:
            b_gk_last = gk_ref[0, 0, BT - 1].astype(jnp.float32)
            if USE_EXP2:
                scratch_ref[...] *= exp2(b_gk_last)[:, None]
            else:
                scratch_ref[...] *= exp(b_gk_last)[:, None]

        scratch_ref[...] += jnp.dot(
            b_k.astype(jnp.float32).T,
            b_v.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    @pl.when(idx_nt == real_NT - 1)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


def chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    use_exp2=True,
    cu_seqlens=None,
    chunk_indices=None,
):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size

    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    assert B == 1, f"varlen mode requires B==1, got B={B}"

    N = cu_seqlens.shape[-1] - 1
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(w, (B, T, H, K), "w")
    assert_shape(u, (B, T, H, V), "u")
    assert_shape_or_none(g, (B, T, H), "g")
    assert_shape_or_none(gk, (B, T, H, K), "gk")
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    # --- Varlen launcher ---
    k = k.astype(jnp.float32)
    w = w.astype(jnp.float32)
    u_f32 = u.astype(jnp.float32)

    K_PADSIZE = int(align_up(K, 128))
    V_ALIGNED = int(align_up(V, 128))

    assert chunk_indices is not None
    NT = len(chunk_indices)
    NT_max = T // BT
    chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
    assert initial_state is None or initial_state.shape == (N, H, K, V)

    T_alloc = T + BT

    k_pad = (
        jnp.pad(k, ((0, 0), (0, BT), (0, 0), (0, K_PADSIZE - K)))
        if K_PADSIZE > K
        else jnp.pad(k, ((0, 0), (0, BT), (0, 0), (0, 0)))
    )
    w_pad = (
        jnp.pad(w, ((0, 0), (0, BT), (0, 0), (0, K_PADSIZE - K)))
        if K_PADSIZE > K
        else jnp.pad(w, ((0, 0), (0, BT), (0, 0), (0, 0)))
    )
    k_t = jnp.transpose(k_pad, (0, 2, 1, 3))
    w_t = jnp.transpose(w_pad, (0, 2, 1, 3))

    v_pad = (
        jnp.pad(u_f32, ((0, 0), (0, BT), (0, 0), (0, V_ALIGNED - V)))
        if V_ALIGNED > V
        else jnp.pad(u_f32, ((0, 0), (0, BT), (0, 0), (0, 0)))
    )
    v_t = jnp.transpose(v_pad, (0, 2, 1, 3))

    if g is not None:
        g_fp32 = g.astype(jnp.float32).reshape(B, T, H, 1)
        g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
        g_fp32 = jnp.pad(g_fp32, ((0, 0), (0, BT), (0, 0), (0, 0)))
        g_t = jnp.transpose(g_fp32, (0, 2, 1, 3))
    else:
        g_t = None

    if gk is not None:
        # NOTE: the kernel only reads gk[BT-1], but a [1,1,1,K] block is illegal
        # on TPU (sublane dim must be a multiple of 8 or the full array dim), so
        # the full [BT, K] block is shipped. The fused path avoids this entirely.
        gk_fp32 = gk.astype(jnp.float32)
        if K_PADSIZE > K:
            gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
        gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, BT), (0, 0), (0, 0)))
        gk_t = jnp.transpose(gk_fp32, (0, 2, 1, 3))
    else:
        gk_t = None

    if initial_state is not None:
        h0 = initial_state
        if V_ALIGNED > V:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
        if K_PADSIZE > K:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
    else:
        h0 = None

    g_pad_size = g_t.shape[-1] if g_t is not None else 128
    h_spec = jax.ShapeDtypeStruct([B, NT, H, K_PADSIZE, V_ALIGNED], k.dtype)
    v_new_spec = (
        jax.ShapeDtypeStruct([B, H, T_alloc, V_ALIGNED], jnp.float32) if save_new_value else None
    )
    ht_spec = (
        jax.ShapeDtypeStruct([N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
        if output_final_state
        else None
    )

    def _t_index_map(n, h, nt, seqlens_ref, chunk_offsets_ref):
        bos = pl.multiple_of(seqlens_ref[n], BT)
        block_idx = jnp.minimum(bos // BT + nt, T // BT)
        return (0, h, block_idx, 0)

    def _h_index_map(n, h, nt, seqlens_ref, chunk_offsets_ref):
        bos = pl.multiple_of(seqlens_ref[n], BT)
        chunk_idx = jnp.minimum(bos // BT + nt, NT - 1)
        return (0, chunk_idx, h, 0, 0)

    k_blockspec = pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map)
    v_blockspec = pl.BlockSpec([1, 1, BT, V_ALIGNED], index_map=_t_index_map)
    w_blockspec = pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map)
    g_blockspec = (
        pl.BlockSpec([1, 1, BT, g_pad_size], index_map=_t_index_map) if g is not None else None
    )
    gk_blockspec = (
        pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map) if gk is not None else None
    )
    h0_blockspec = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=lambda n, h, nt, *_: (n, h, 0, 0))
        if initial_state is not None
        else None
    )

    h_blockspec_out = pl.BlockSpec([1, 1, 1, K_PADSIZE, V_ALIGNED], index_map=_h_index_map)
    v_new_blockspec_out = (
        pl.BlockSpec([1, 1, BT, V_ALIGNED], index_map=_t_index_map) if save_new_value else None
    )
    ht_blockspec_out = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=lambda n, h, nt, *_: (n, h, 0, 0))
        if output_final_state
        else None
    )

    scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
    grid = (N, H, NT_max)
    interpret = get_interpret()

    h_out, v_new_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_gated_delta_rule_fwd_kernel,
            NT=NT,
            USE_G=(g is not None),
            USE_GK=(gk is not None),
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=save_new_value,
            USE_EXP2=use_exp2,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=grid,
            in_specs=[
                k_blockspec,
                v_blockspec,
                w_blockspec,
                g_blockspec,
                gk_blockspec,
                h0_blockspec,
            ],
            out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
            scratch_shapes=[scratch],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        out_shape=[h_spec, v_new_spec, ht_spec],
        interpret=interpret,
    )(cu_seqlens, chunk_offsets, k_t, v_t, w_t, g_t, gk_t, h0)

    h_out = h_out[:, :, :, :K, :V]
    v_new_out = jnp.transpose(v_new_out[:, :, :T, :V], (0, 2, 1, 3)) if save_new_value else None
    ht_out = ht_out[:, :, :K, :V] if output_final_state else None

    return h_out, v_new_out, ht_out


# ---- Flat-grid legacy stage 3, ported from PR #3 kda-fwdh-flat-grid for the
# fuse=False + flat_grid=True ablation; grid O(N x chunks) -> O(chunks). ----
def _chunk_gated_delta_rule_fwd_kernel_flat(
    seq_ids_ref,
    is_first_ref,
    is_last_ref,
    k_ref,
    v_ref,
    w_ref,
    g_ref,
    gk_ref,
    h0_ref,
    h_ref,
    v_new_ref,
    ht_ref,
    scratch_ref,
    *,
    USE_G,
    USE_GK,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    SAVE_NEW_VALUE,
    USE_EXP2,
):
    # Flat-chunk grid (h, nt): every step is a real chunk -- O(total_chunks)
    # instead of the previous O(N x total_chunks) where each sequence swept
    # the full global chunk range and idled through chunks it did not own.
    # Sequence boundaries come from the prefetched flags: reset the state
    # carry at a sequence's first chunk, emit its final state at the last.
    idx_nt = pl.program_id(1)

    BT = k_ref.shape[2]
    K, V = k_ref.shape[-1], v_ref.shape[-1]
    b_k = k_ref[0, 0]

    @pl.when(is_first_ref[idx_nt] == 1)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    h_ref[0, 0, 0] = scratch_ref[...].astype(h_ref.dtype)

    b_w = w_ref[0, 0]
    b_v = jnp.dot(
        b_w.astype(jnp.float32),
        scratch_ref[...],
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_u = v_ref[0, 0]
    b_v = b_u.astype(b_v.dtype) - b_v
    if SAVE_NEW_VALUE:
        v_new_ref[0, 0] = b_v.astype(v_new_ref.dtype)

    if USE_G:
        b_g = g_ref[0, 0, :, 0]
        b_g_last = g_ref[0, 0, BT - 1, 0].astype(jnp.float32)
        if USE_EXP2:
            b_v = b_v * exp2(b_g_last - b_g)[:, None]
            b_g_last = exp2(b_g_last)
        else:
            b_v = b_v * exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
        scratch_ref[...] *= b_g_last
    if USE_GK:
        b_gk_last = gk_ref[0, 0, BT - 1].astype(jnp.float32)
        if USE_EXP2:
            scratch_ref[...] *= exp2(b_gk_last)[:, None]
        else:
            scratch_ref[...] *= exp(b_gk_last)[:, None]

    scratch_ref[...] += jnp.dot(
        b_k.astype(jnp.float32).T,
        b_v.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    if STORE_FINAL_STATE:

        @pl.when(is_last_ref[idx_nt] == 1)
        def _():
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


def chunk_gated_delta_rule_fwd_h_flat(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    use_exp2=True,
    cu_seqlens=None,
    chunk_indices=None,
):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size

    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    assert B == 1, f"varlen mode requires B==1, got B={B}"

    N = cu_seqlens.shape[-1] - 1
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(w, (B, T, H, K), "w")
    assert_shape(u, (B, T, H, V), "u")
    assert_shape_or_none(g, (B, T, H), "g")
    assert_shape_or_none(gk, (B, T, H, K), "gk")
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    # --- Varlen launcher (flat-chunk grid) ---
    # Runs after _align_seqs, so every sequence is BT-aligned and the packed
    # chunk list is contiguous: the grid is O(total_chunks). The previous
    # (N, H, NT_max) grid swept the FULL global chunk range once per
    # sequence, idling through foreign chunks -- a per-sequence tax measured
    # at ~0.5us x (N-1) x (T/BT) x H (e.g. ~46 ms for 8x1024 packed at
    # T=8192, H=96 on v6e).
    k = k.astype(jnp.float32)
    w = w.astype(jnp.float32)
    u_f32 = u.astype(jnp.float32)

    K_PADSIZE = int(align_up(K, 128))
    V_ALIGNED = int(align_up(V, 128))

    assert chunk_indices is not None
    NT = len(chunk_indices)
    assert NT == T // BT, "flat-chunk fwd_h requires BT-aligned packing"

    cu_i32 = cu_seqlens.astype(jnp.int32)
    chunks_per_seq = jnp.diff(cu_i32) // BT
    cum_chunks = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
    flat_idx = jnp.arange(NT, dtype=jnp.int32)
    seq_ids = jnp.minimum(jnp.searchsorted(cum_chunks[1:], flat_idx, side="right"), N - 1).astype(
        jnp.int32
    )
    local_ids = flat_idx - cum_chunks[seq_ids]
    is_first = (local_ids == 0).astype(jnp.int32)
    is_last = (local_ids == chunks_per_seq[seq_ids] - 1).astype(jnp.int32)

    def _padk(x):
        if K_PADSIZE > K:
            return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
        return x

    k_t = jnp.transpose(_padk(k), (0, 2, 1, 3))
    w_t = jnp.transpose(_padk(w), (0, 2, 1, 3))
    v_pad = jnp.pad(u_f32, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V))) if V_ALIGNED > V else u_f32
    v_t = jnp.transpose(v_pad, (0, 2, 1, 3))

    if g is not None:
        g_fp32 = g.astype(jnp.float32).reshape(B, T, H, 1)
        g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
        g_t = jnp.transpose(g_fp32, (0, 2, 1, 3))
    else:
        g_t = None

    if gk is not None:
        gk_fp32 = gk.astype(jnp.float32)
        if K_PADSIZE > K:
            gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
        gk_t = jnp.transpose(gk_fp32, (0, 2, 1, 3))
    else:
        gk_t = None

    if initial_state is not None:
        h0 = initial_state
        if V_ALIGNED > V:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
        if K_PADSIZE > K:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
    else:
        h0 = None

    g_pad_size = g_t.shape[-1] if g_t is not None else 128
    h_spec = jax.ShapeDtypeStruct([B, NT, H, K_PADSIZE, V_ALIGNED], k.dtype)
    v_new_spec = jax.ShapeDtypeStruct([B, H, T, V_ALIGNED], jnp.float32) if save_new_value else None
    ht_spec = (
        jax.ShapeDtypeStruct([N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
        if output_final_state
        else None
    )

    def _t_index_map(h, nt, seq_ids_ref, is_first_ref, is_last_ref):
        return (0, h, nt, 0)

    def _h_index_map(h, nt, seq_ids_ref, is_first_ref, is_last_ref):
        return (0, nt, h, 0, 0)

    def _state_index_map(h, nt, seq_ids_ref, is_first_ref, is_last_ref):
        return (seq_ids_ref[nt], h, 0, 0)

    k_blockspec = pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map)
    v_blockspec = pl.BlockSpec([1, 1, BT, V_ALIGNED], index_map=_t_index_map)
    w_blockspec = pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map)
    g_blockspec = (
        pl.BlockSpec([1, 1, BT, g_pad_size], index_map=_t_index_map) if g is not None else None
    )
    gk_blockspec = (
        pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=_t_index_map) if gk is not None else None
    )
    h0_blockspec = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
        if initial_state is not None
        else None
    )

    h_blockspec_out = pl.BlockSpec([1, 1, 1, K_PADSIZE, V_ALIGNED], index_map=_h_index_map)
    v_new_blockspec_out = (
        pl.BlockSpec([1, 1, BT, V_ALIGNED], index_map=_t_index_map) if save_new_value else None
    )
    ht_blockspec_out = (
        pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
        if output_final_state
        else None
    )

    scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
    grid = (H, NT)
    interpret = get_interpret()

    h_out, v_new_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_gated_delta_rule_fwd_kernel_flat,
            USE_G=(g is not None),
            USE_GK=(gk is not None),
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=save_new_value,
            USE_EXP2=use_exp2,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=grid,
            in_specs=[
                k_blockspec,
                v_blockspec,
                w_blockspec,
                g_blockspec,
                gk_blockspec,
                h0_blockspec,
            ],
            out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
            scratch_shapes=[scratch],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        out_shape=[h_spec, v_new_spec, ht_spec],
        interpret=interpret,
    )(seq_ids, is_first, is_last, k_t, v_t, w_t, g_t, gk_t, h0)

    h_out = h_out[:, :, :, :K, :V]
    v_new_out = jnp.transpose(v_new_out[:, :, :, :V], (0, 2, 1, 3)) if save_new_value else None
    ht_out = ht_out[:, :, :K, :V] if output_final_state else None

    return h_out, v_new_out, ht_out


# ============================================================================
# Fused Stage 3+4: inter-chunk recurrence + output in one kernel.
# h (chunk-entry state) and v_new (delta-rule residual) stay in VMEM and never
# round-trip through HBM; o is written directly from the recurrence loop.
# ============================================================================


def _fused_step_math(q, kk, v, w, g, A, S, scale):
    """Run one fused recurrence step for one head and chunk.

    Returns o [BT,V] and S_new [K,V].
    """
    BT = q.shape[0]
    b_v = jnp.dot(
        w.astype(jnp.float32),
        S,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_v = v.astype(jnp.float32) - b_v  # [BT, V] delta-rule residual (formerly v_new)

    b_g = g.astype(jnp.float32)
    b_g_ref = b_g[0:1, :]
    b_qg = q.astype(jnp.float32) * exp2(jnp.maximum(b_g - b_g_ref, -126.0))
    b_h_scaled = S * exp2(jnp.maximum(b_g_ref[0], -126.0))[:, None]
    b_o = scale * jnp.dot(
        b_qg,
        b_h_scaled,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(m_s, A.astype(jnp.float32), 0.0)
    b_o += jnp.dot(
        b_A, b_v, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    )

    S_new = S * exp2(b_g[BT - 1])[:, None] + jnp.dot(
        kk.astype(jnp.float32).T,
        b_v,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return b_o, S_new


def _fused_h_o_chunk_step(q_ref, k_ref, v_ref, w_ref, g_ref, A_ref, o_ref, scratch_ref, scale):
    """Legacy-layout wrapper that reads [1,1,BT,*] refs and writes the result."""
    b_o, S_new = _fused_step_math(
        q_ref[0, 0],
        k_ref[0, 0],
        v_ref[0, 0],
        w_ref[0, 0],
        g_ref[0, 0],
        A_ref[0, 0],
        scratch_ref[...],
        scale,
    )
    o_ref[0, 0] = b_o.astype(o_ref.dtype)
    scratch_ref[...] = S_new


def _chunk_kda_fused_h_o_kernel(
    seq_id_ref,  # [NC] prefetch: owning sequence, consumed only by index_map
    start_flag_ref,  # [NC] prefetch: first chunk of a sequence; resets state
    end_flag_ref,  # [NC] prefetch: last chunk of a sequence; writes final_state
    q_ref,  # [1, 1, BT, K]
    k_ref,  # [1, 1, BT, K]   kg from stage 2 (k * exp2(g_last - g))
    v_ref,  # [1, 1, BT, V]   u from stage 2 (corrected values)
    w_ref,  # [1, 1, BT, K]
    g_ref,  # [1, 1, BT, K]   g_cumsum (fp32, log2 domain)
    A_ref,  # [1, 1, BT, BT]  Aqk from stage 2
    h0_ref,  # [1, 1, K, V] or None
    o_ref,  # [1, 1, BT, V] out
    ht_ref,  # [1, 1, K, V] out or None
    scratch_ref,  # [K, V] f32, persistent hidden state
    *,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
):
    """Run a flat chunk grid (flat_grid=True) with grid=(H, NC).

    The c dimension advances serially in packed order, so total work is
    O(chunks), independent of sequence count N. Start/end flags mark sequence
    boundaries. Inputs for inactive chunks are zero from K1, making their
    residual and state updates no-ops.
    """
    idx_c = pl.program_id(1)
    K, V = k_ref.shape[-1], v_ref.shape[-1]

    @pl.when(start_flag_ref[idx_c] == 1)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    _fused_h_o_chunk_step(q_ref, k_ref, v_ref, w_ref, g_ref, A_ref, o_ref, scratch_ref, scale)

    @pl.when(end_flag_ref[idx_c] == 1)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


def _chunk_kda_fused_h_o_kernel_seqgrid(
    seqlens_ref,  # [N+1] prefetch: cu_seqlens
    q_ref,
    k_ref,
    v_ref,
    w_ref,
    g_ref,
    A_ref,
    h0_ref,
    o_ref,
    ht_ref,
    scratch_ref,
    *,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
):
    """Run the legacy (N, H, NT_max) grid for flat_grid=False ablation.

    Every sequence scans the global chunk range. Steps where nt >= real_NT
    remain idle but still incur grid and DMA costs, producing O(N x chunks)
    scheduling overhead.
    """
    idx_n = pl.program_id(0)
    idx_nt = pl.program_id(2)

    bos = seqlens_ref[idx_n]
    eos = seqlens_ref[idx_n + 1]
    BT = k_ref.shape[2]
    real_NT = (eos - bos) // BT
    K, V = k_ref.shape[-1], v_ref.shape[-1]

    @pl.when(idx_nt == 0)
    def _():
        scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    @pl.when(idx_nt < real_NT)
    def _():
        _fused_h_o_chunk_step(q_ref, k_ref, v_ref, w_ref, g_ref, A_ref, o_ref, scratch_ref, scale)

    @pl.when(idx_nt == real_NT - 1)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


def chunk_kda_fused_h_o(
    q,  # unified_in=True: [1, H, T, K]; False: [1, T, H, K]
    kg,
    w,
    u,
    g_cumsum,
    A,
    scale,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
    unified_in=True,
    flat_grid=True,
):
    BT = chunk_size
    assert cu_seqlens is not None
    N = cu_seqlens.shape[-1] - 1

    if unified_in:
        B, H, T_out, K = q.shape
        V = u.shape[-1]
        assert B == 1 and T_out % BT == 0
        K_PADSIZE = int(align_up(K, 128))
        V_ALIGNED = int(align_up(V, 128))

        def _padlast(x, D, D_pad):
            if D_pad > D:
                return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, D_pad - D)))
            return x

        q_t = _padlast(q, K, K_PADSIZE)
        k_t = _padlast(kg, K, K_PADSIZE)
        w_t = _padlast(w, K, K_PADSIZE)
        v_t = _padlast(u, V, V_ALIGNED)
        g_t = _padlast(g_cumsum, K, K_PADSIZE)
        A_t = A
        if not flat_grid:
            # The legacy grid needs a trailing trash block as its clamp target.
            pad_t = lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, BT), (0, 0)))
            q_t, k_t, w_t, v_t, g_t, A_t = map(pad_t, (q_t, k_t, w_t, v_t, g_t, A_t))
    else:
        # Legacy _prep for the data-movement ablation: materialize fp32,
        # append a trailing pad, and transpose the [1,T,H,D] input.
        B, T_out, H, K = q.shape
        V = u.shape[-1]
        assert B == 1 and T_out % BT == 0
        K_PADSIZE = int(align_up(K, 128))
        V_ALIGNED = int(align_up(V, 128))

        def _prep(x, D, D_pad):
            x = x.astype(jnp.float32) if x.dtype != jnp.float32 else x
            if D_pad > D:
                x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, D_pad - D)))
            x = jnp.pad(x, ((0, 0), (0, BT), (0, 0), (0, 0)))
            return jnp.transpose(x, (0, 2, 1, 3))  # [1, H, T+BT, D_pad]

        q_t = _prep(q, K, K_PADSIZE)
        k_t = _prep(kg, K, K_PADSIZE)
        w_t = _prep(w, K, K_PADSIZE)
        v_t = _prep(u, V, V_ALIGNED)
        g_t = _prep(g_cumsum, K, K_PADSIZE)
        A_t = _prep(A, BT, BT)

    T_pad = q_t.shape[2]

    if initial_state is not None:
        h0 = initial_state
        if V_ALIGNED > V:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
        if K_PADSIZE > K:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
    else:
        h0 = None

    ht_spec = (
        jax.ShapeDtypeStruct([N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
        if output_final_state
        else None
    )
    scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
    kernel_kw = dict(
        scale=scale,
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=output_final_state,
    )
    cu_i32 = cu_seqlens.astype(jnp.int32)

    if flat_grid:
        # Flat grid: O(chunks), with sequence identity in three prefetched scalars.
        NC = T_pad // BT
        chunk_bos = jnp.arange(NC, dtype=jnp.int32) * BT
        seq_id = jnp.clip(jnp.searchsorted(cu_i32[1:], chunk_bos, side="right"), 0, N - 1).astype(
            jnp.int32
        )
        start_flag = (chunk_bos == cu_i32[seq_id]).astype(jnp.int32)
        end_flag = (chunk_bos + BT == cu_i32[seq_id + 1]).astype(jnp.int32)

        _state_index_map = lambda h, c, seq_id_ref, *_: (seq_id_ref[c], h, 0, 0)
        tspec = lambda D: pl.BlockSpec([1, 1, BT, D], index_map=lambda h, c, *_: (0, h, c, 0))
        h0_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if initial_state is not None
            else None
        )
        ht_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if output_final_state
            else None
        )

        o_out, ht_out = pl.pallas_call(
            functools.partial(_chunk_kda_fused_h_o_kernel, **kernel_kw),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                grid=(H, NC),
                in_specs=[
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(V_ALIGNED),
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(BT),
                    h0_blockspec,
                ],
                out_specs=[tspec(V_ALIGNED), ht_blockspec],
                scratch_shapes=[scratch],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
            out_shape=[jax.ShapeDtypeStruct([1, H, T_pad, V_ALIGNED], jnp.float32), ht_spec],
            interpret=get_interpret(),
        )(seq_id, start_flag, end_flag, q_t, k_t, v_t, w_t, g_t, A_t, h0)
    else:
        # Legacy (N, H, NT_max) ablation grid: O(N x chunks).
        T_ref = T_pad - BT  # Logical T; block T_ref//BT is the clamp target.
        NT_max = T_ref // BT

        def _t_index_map(n, h, nt, seqlens_ref):
            bos = pl.multiple_of(seqlens_ref[n], BT)
            return (0, h, jnp.minimum(bos // BT + nt, T_ref // BT), 0)

        _state_index_map = lambda n, h, nt, *_: (n, h, 0, 0)
        tspec = lambda D: pl.BlockSpec([1, 1, BT, D], index_map=_t_index_map)
        h0_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if initial_state is not None
            else None
        )
        ht_blockspec = (
            pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=_state_index_map)
            if output_final_state
            else None
        )

        o_out, ht_out = pl.pallas_call(
            functools.partial(_chunk_kda_fused_h_o_kernel_seqgrid, **kernel_kw),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(N, H, NT_max),
                in_specs=[
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(V_ALIGNED),
                    tspec(K_PADSIZE),
                    tspec(K_PADSIZE),
                    tspec(BT),
                    h0_blockspec,
                ],
                out_specs=[tspec(V_ALIGNED), ht_blockspec],
                scratch_shapes=[scratch],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary")
            ),
            out_shape=[jax.ShapeDtypeStruct([1, H, T_pad, V_ALIGNED], jnp.float32), ht_spec],
            interpret=get_interpret(),
        )(cu_i32, q_t, k_t, v_t, w_t, g_t, A_t, h0)

    o = jnp.transpose(o_out[:, :, :T_out, :V], (0, 2, 1, 3))
    ht_out = ht_out[:, :, :K, :V] if output_final_state else None
    return o, ht_out


# ============================================================================
# kda forward O+GK (varlen only)
# ============================================================================


def _chunk_kda_fwd_o_gk_pl_kernel(
    q_ref,
    v_ref,
    g_ref,
    h_ref,
    A_ref,
    o_ref,
    *,
    BT,
    scale,
    USE_EXP2,
):
    b_q = q_ref[0, 0]
    b_g = g_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = h_ref[0, 0]
    b_A = A_ref[0, 0]

    b_g_f32 = b_g.astype(jnp.float32)
    b_q_f32 = b_q.astype(jnp.float32)
    # Compute inter-chunk output: o = scale * q * exp2(g) @ h.
    # Use g[0] (first position, largest cumsum) as reference to avoid overflow/underflow:
    #   exp2(g[t]) = exp2(g[t] - g[0]) * exp2(g[0])
    # g[t] - g[0] <= 0 for all t (cumsum is monotonically decreasing), so exp2 is safe.
    # Factor exp2(g[0]) into h to preserve the matmul structure.
    _exp_fn = exp2 if USE_EXP2 else exp
    b_g_ref = b_g_f32[0:1, :]  # [1, K] -- reference point
    b_qg = b_q_f32 * _exp_fn(jnp.maximum(b_g_f32 - b_g_ref, -126.0))
    # Scale h rows: h_scaled[k, v] = h[k, v] * exp2(g_ref[k])
    b_h_scaled = b_h.astype(jnp.float32) * _exp_fn(jnp.maximum(b_g_ref[0], -126.0))[:, None]
    b_o = jnp.dot(
        b_qg, b_h_scaled, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    )
    b_o *= scale

    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_f32 = jnp.where(m_s, b_A, 0.0).astype(jnp.float32)
    b_o += jnp.dot(
        b_A_f32,
        b_v.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0] = b_o.astype(o_ref.dtype)


def chunk_kda_fwd_o_gk(
    q,
    v,
    g,
    A,
    h,
    scale,
    *,
    cu_seqlens,
    chunk_indices=None,
    chunk_size=64,
    use_exp2=False,
):
    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT_h = h.shape[1]
    assert B == 1
    assert T % BT == 0

    N = cu_seqlens.shape[0] - 1
    T_alloc = T + BT

    pad4d = lambda x: jnp.pad(x, ((0, 0), (0, BT), (0, 0), (0, 0)))
    q_pad, v_pad, g_pad, A_pad = pad4d(q), pad4d(v), pad4d(g), pad4d(A)

    cu_i32 = cu_seqlens.astype(jnp.int32)
    seq_lens = jnp.diff(cu_i32)
    chunks_per_seq = (seq_lens + BT - 1) // BT
    cum_chunks = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
    total_chunks = cum_chunks[-1]

    NC_max = len(chunk_indices) if chunk_indices is not None else T // BT + N
    flat_idx = jnp.arange(NC_max, dtype=jnp.int32)
    is_valid = flat_idx < total_chunks

    seq_id = jnp.minimum(jnp.searchsorted(cum_chunks[1:], flat_idx, side="right"), N - 1)
    local_ci = flat_idx - cum_chunks[seq_id]
    bos = cu_i32[seq_id]
    # After _align_seqs, every sequence is BT-aligned -- no partial chunks.
    chunk_starts = jnp.where(is_valid, bos + local_ci * BT, 0)

    def gather(x_pad, D):
        def extract(start):
            return jax.lax.dynamic_slice(x_pad, (0, start, 0, 0), (1, BT, H, D))[0]

        return jax.vmap(extract)(chunk_starts)

    q_c, v_c, g_c, A_c = gather(q_pad, K), gather(v_pad, V), gather(g_pad, K), gather(A_pad, BT)

    _q = q_c.transpose(2, 0, 1, 3)
    _v = v_c.transpose(2, 0, 1, 3)
    _g = g_c.transpose(2, 0, 1, 3)
    _A = A_c.transpose(2, 0, 1, 3)

    _h = h[0].transpose(1, 0, 2, 3)
    if NC_max > NT_h:
        _h = jnp.pad(_h, ((0, 0), (0, NC_max - NT_h), (0, 0), (0, 0)))
    elif NC_max < NT_h:
        _h = _h[:, :NC_max]

    q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    g_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    o_shape = jax.ShapeDtypeStruct([H, NC_max, BT, V], v.dtype)
    o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o_r = pl.pallas_call(
        functools.partial(_chunk_kda_fwd_o_gk_pl_kernel, BT=BT, scale=scale, USE_EXP2=use_exp2),
        grid=(H, NC_max),
        out_shape=o_shape,
        in_specs=[q_spec, v_spec, g_spec, h_spec, A_spec],
        out_specs=o_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=get_interpret(),
    )(_q, _v, _g, _h, _A)

    pos = chunk_starts[:, None] + jnp.arange(BT)[None, :]
    pos = jnp.where(is_valid[:, None], pos, T_alloc - 1)
    flat_pos = pos.reshape(-1)

    o_chunks = o_r.transpose(1, 2, 0, 3).reshape(-1, H, V)
    out = jnp.zeros((T_alloc, H, V), dtype=o_chunks.dtype)
    out = out.at[flat_pos].add(o_chunks)
    return out[:T][None]


# ============================================================================
# KDA gate cumsum helpers
# ============================================================================

_RCP_LN2 = 1.0 / math.log(2)


def kda_gate_chunk_cumsum(
    g,
    A_log,
    chunk_size,
    scale=None,
    dt_bias=None,
    cu_seqlens=None,
    output_dtype=jnp.float32,
    chunk_indices=None,
    lower_bound=None,
):
    B, T, H, K = g.shape
    assert_shape(g, (B, T, H, K), "g")
    assert A_log.shape == (H,), f"A_log shape {A_log.shape} != ({H},)"

    g_f32 = g.astype(jnp.float32)
    if dt_bias is not None:
        g_f32 = g_f32 + dt_bias.astype(jnp.float32).reshape(H, K)

    A = A_log.astype(jnp.float32)
    if lower_bound is None:
        g_act = -exp(A).reshape(1, 1, H, 1) * jax.nn.softplus(g_f32)
    else:
        g_act = lower_bound * jax.nn.sigmoid(exp(A).reshape(1, 1, H, 1) * g_f32)

    return chunk_local_cumsum_vector(
        g_act,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=False,
        output_dtype=output_dtype or jnp.float32,
        chunk_indices=chunk_indices,
    )


def pallas_kda_gate_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=_RCP_LN2,
    cu_seqlens=None,
    head_first=False,
    output_dtype=jnp.float32,
    chunk_indices=None,
):
    B, T, H, K = g.shape
    assert_shape(g, (B, T, H, K), "g")
    assert T % chunk_size == 0, f"T={T} must be divisible by chunk_size={chunk_size}"

    return chunk_local_cumsum_vector(
        g,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_first=False,
        output_dtype=jnp.float32,
    )


# ============================================================================
# Varlen alignment helpers
# ============================================================================


def _align_seqs(tensors_4d, tensors_3d, cu_seqlens, align):
    N = cu_seqlens.shape[0] - 1
    T_old = tensors_4d[0].shape[1]

    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    padded_lens = ((seg_lens + align - 1) // align) * align
    padded_cu = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(padded_lens)])
    T_new = ((T_old + N * (align - 1) + align - 1) // align) * align

    def _build_gather(i, gather_idx):
        old_start = cu_seqlens[i]
        new_start = padded_cu[i]
        sl = seg_lens[i]
        j = jnp.arange(T_new)
        in_seg = (j >= new_start) & (j < new_start + sl)
        src = old_start + (j - new_start)
        return jnp.where(in_seg, src, gather_idx)

    gather_idx = jnp.full(T_new, T_old, dtype=jnp.int32)
    gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)

    def repack_4d(t):
        return jnp.pad(t, ((0, 0), (0, T_new - T_old), (0, 0), (0, 0)))[:, gather_idx]

    def repack_3d(t):
        return jnp.pad(t, ((0, 0), (0, T_new - T_old), (0, 0)))[:, gather_idx]

    return (
        [repack_4d(t) for t in tensors_4d],
        [repack_3d(t) for t in tensors_3d],
        padded_cu,
        cu_seqlens,
    )


def _unalign_output(o, orig_cu_seqlens, aligned_cu_seqlens, T_out):
    N = orig_cu_seqlens.shape[0] - 1
    orig_seg_lens = orig_cu_seqlens[1:] - orig_cu_seqlens[:-1]

    def _build_gather(i, gather_idx):
        orig_start = orig_cu_seqlens[i]
        aligned_start = aligned_cu_seqlens[i]
        sl = orig_seg_lens[i]
        j = jnp.arange(T_out)
        in_seg = (j >= orig_start) & (j < orig_start + sl)
        src = aligned_start + (j - orig_start)
        return jnp.where(in_seg, src, gather_idx)

    gather_idx = jnp.zeros(T_out, dtype=jnp.int32)
    gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)
    return o[:, gather_idx]


# ============================================================================
# Main entry point
# ============================================================================


# ============================================================================
# Native-layout head-block (hb) mode: kernels consume [1, T, H, D] directly.
#
# A full-head [1, BT, H, D] block satisfies TPU tiling rules when H % 8 == 0;
# a single-head [1, BT, 1, D] block does not. This:
#   1. removes entry/exit transposes (~420 MB at the headline shape) because
#      the layout already matches the producer;
#   2. collapses grid (H, NC) to (NC,), reducing steps by H and amortizing the
#      roughly 1.1 us fixed cost per step;
#   3. lets the scheduler interleave H independent K2 recurrence chains and
#      fill dependency bubbles.
# Inference omits backward intermediates by default: Akk is not computed and
# the fused-wide RHS shrinks from 320 to 256 columns. See
# return_backward_intermediates.
# ============================================================================


def _kda_fwd_intra_kernel_hb(
    q_ref,  # [1, BT, H, K]
    k_ref,  # [1, BT, H, K]
    g_ref,  # [1, BT, H, K]
    beta_ref,  # [1, BT, H, 1]
    v_ref,  # [1, BT, H, V]
    a_ref,  # [H, K] or None
    bias_ref,  # [H, K] or None
    u_out_ref,  # [1, BT, H, V]
    w_out_ref,  # [1, BT, H, K]
    kg_out_ref,  # [1, BT, H, K]
    Aqk_out_ref,  # [1, BT, H, BT]
    g_cum_out_ref,  # [1, BT, H, K] f32
    *,
    chunk_size,
    head_dim,
    value_dim,
    scale,
    safe_gate,
    APPLY_GATE,
    LOWER_BOUND,
    NUM_HEADS,
):
    if not safe_gate:
        # Debug path (elementwise decay plus row-wise elimination): process
        # heads serially and keep _intra_head_math as the single source.
        for h in range(NUM_HEADS):
            a_vec = a_ref[h] if APPLY_GATE else None
            bias_vec = bias_ref[h] if APPLY_GATE else None
            u, w, kg, Aqk, _, g_cum = _intra_head_math(
                q_ref[0, :, h, :],
                k_ref[0, :, h, :],
                g_ref[0, :, h, :],
                beta_ref[0, :, h, :],
                v_ref[0, :, h, :],
                a_vec,
                bias_vec,
                chunk_size=chunk_size,
                head_dim=head_dim,
                value_dim=value_dim,
                scale=scale,
                safe_gate=safe_gate,
                APPLY_GATE=APPLY_GATE,
                LOWER_BOUND=LOWER_BOUND,
                PRE_CUMSUM=False,
                WANT_AINV=False,
            )
            u_out_ref[0, :, h, :] = u.astype(u_out_ref.dtype)
            w_out_ref[0, :, h, :] = w.astype(w_out_ref.dtype)
            kg_out_ref[0, :, h, :] = kg.astype(kg_out_ref.dtype)
            Aqk_out_ref[0, :, h, :] = Aqk.astype(Aqk_out_ref.dtype)
            g_cum_out_ref[0, :, h, :] = g_cum.astype(g_cum_out_ref.dtype)
        return

    # ---- safe_gate fast path: vectorize elementwise work across H and
    # interleave MXU stages. Each head is independent. Stage-major issue
    # traverses h within a stage, so adjacent MXU instructions have no data
    # dependency and can pipeline fill/drain. Elementwise work handles all
    # heads together, reducing instruction count by H. ----
    BT = chunk_size
    H = NUM_HEADS
    # Stage 1: gate activation and prefix sum, vectorized across H.
    g_all = g_ref[0].astype(jnp.float32)  # [BT, H, K]
    if APPLY_GATE:
        a_all = a_ref[...].astype(jnp.float32)  # [H, K]
        b_all = bias_ref[...].astype(jnp.float32)
        g_all = LOWER_BOUND * jax.nn.sigmoid(a_all[None] * (g_all + b_all[None]))
    num_steps = int(math.log2(BT))
    assert (1 << num_steps) == BT
    for d in range(num_steps):
        s = 1 << d
        g_all = jnp.concatenate([g_all[:s], g_all[s:] + g_all[:-s]], axis=0)
    g_all = g_all * _RCP_LN2
    g_cum_out_ref[0] = g_all.astype(g_cum_out_ref.dtype)

    q_all = q_ref[0].astype(jnp.float32)
    k_all = k_ref[0].astype(jnp.float32)
    v_all = v_ref[0].astype(jnp.float32)
    beta_all = beta_ref[0].astype(jnp.float32)  # [BT, H, 1]

    # Build strip rows/columns vectorized across H; interleave GEMMs by (blk, h).
    SB = 16
    o_i = jnp.arange(BT, dtype=jnp.int32)
    causal = o_i[:, None] >= o_i[None, :]
    strict = o_i[:, None] > o_i[None, :]
    dn = (((1,), (1,)), ((), ()))
    aqk_parts = [[] for _ in range(H)]
    l_parts = [[] for _ in range(H)]
    for blk in range(BT // SB):
        cols = slice(blk * SB, (blk + 1) * SB)
        r_b = g_all[blk * SB + SB // 2 : blk * SB + SB // 2 + 1]  # [1, H, K]
        row_all = exp2(g_all - r_b)  # [BT, H, K]
        col_all = k_all[cols] * exp2(r_b - g_all[cols])  # [SB, H, K]
        qrow = q_all * row_all
        krow = k_all * row_all
        for h in range(H):
            aqk_parts[h].append(
                jax.lax.dot_general(
                    qrow[:, h], col_all[:, h], dn, preferred_element_type=jnp.float32
                )
            )
            l_parts[h].append(
                jax.lax.dot_general(
                    krow[:, h], col_all[:, h], dn, preferred_element_type=jnp.float32
                )
            )

    v_beta = v_all * beta_all  # [BT, H, V]
    k_eg_beta = k_all * exp2(g_all) * beta_all  # [BT, H, K]

    Aqks, Ls, zs = [], [], []
    for h in range(H):
        Aqks.append(jnp.where(causal, scale * jnp.concatenate(aqk_parts[h], -1), 0.0))
        Ls.append(jnp.where(strict, jnp.concatenate(l_parts[h], -1), 0.0) * beta_all[:, h])
        zs.append(jnp.concatenate([v_beta[:, h], k_eg_beta[:, h]], axis=-1))

    # Fused-wide Neumann factor chain with stages interleaved across H.
    zs = [z - jax.lax.dot(L, z, preferred_element_type=jnp.float32) for L, z in zip(Ls, zs)]
    Lp = list(Ls)
    for _ in range(int(math.log2(BT)) - 1):
        Lp = [jax.lax.dot(P, P, preferred_element_type=jnp.float32) for P in Lp]
        zs = [z + jax.lax.dot(P, z, preferred_element_type=jnp.float32) for P, z in zip(Lp, zs)]

    # Assemble outputs and write them back vectorized across H.
    u_all = jnp.stack([z[:, :value_dim] for z in zs], axis=1)
    w_all = jnp.stack([z[:, value_dim:] for z in zs], axis=1)
    kg_all = k_all * exp2(g_all[BT - 1 : BT] - g_all)
    u_out_ref[0] = u_all.astype(u_out_ref.dtype)
    w_out_ref[0] = w_all.astype(w_out_ref.dtype)
    kg_out_ref[0] = kg_all.astype(kg_out_ref.dtype)
    Aqk_out_ref[0] = jnp.stack(Aqks, axis=1).astype(Aqk_out_ref.dtype)


def kda_fwd_intra_hb(
    q,
    k,
    v,
    gk,
    beta,
    scale,
    chunk_size=64,
    safe_gate=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
):
    """Run head-block K1 with native [1, T, H, D] I/O and grid=(NC,)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    assert B == 1 and T % BT == 0 and H % 8 == 0
    NC = T // BT

    beta4 = beta.reshape(B, T, H, 1)
    if use_gate_in_kernel:
        assert A_log is not None
        a_r = jnp.broadcast_to(jnp.exp(A_log.astype(jnp.float32))[:, None], (H, K))
        bias_r = (
            jnp.zeros((H, K), jnp.float32)
            if dt_bias is None
            else dt_bias.astype(jnp.float32).reshape(H, K)
        )
        gate_spec = pl.BlockSpec(block_shape=(H, K), index_map=lambda c: (0, 0))
    else:
        a_r, bias_r, gate_spec = None, None, None

    def _spec(last_dim):
        return pl.BlockSpec(block_shape=(1, BT, H, last_dim), index_map=lambda c: (0, c, 0, 0))

    dt = q.dtype
    u4, w4, kg4, Aqk4, g_cum4 = pl.pallas_call(
        functools.partial(
            _kda_fwd_intra_kernel_hb,
            chunk_size=BT,
            head_dim=K,
            value_dim=V,
            scale=scale,
            safe_gate=safe_gate,
            APPLY_GATE=use_gate_in_kernel,
            LOWER_BOUND=lower_bound,
            NUM_HEADS=H,
        ),
        interpret=get_interpret(),
        out_shape=[
            jax.ShapeDtypeStruct((1, T, H, V), dt),
            jax.ShapeDtypeStruct((1, T, H, K), dt),
            jax.ShapeDtypeStruct((1, T, H, K), dt),
            jax.ShapeDtypeStruct((1, T, H, BT), dt),
            jax.ShapeDtypeStruct((1, T, H, K), jnp.float32),
        ],
        in_specs=[_spec(K), _spec(K), _spec(K), _spec(1), _spec(V), gate_spec, gate_spec],
        out_specs=[_spec(V), _spec(K), _spec(K), _spec(BT), _spec(K)],
        grid=(NC,),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(q, k, gk, beta4, v, a_r, bias_r)
    return w4, u4, kg4, Aqk4, g_cum4


def _chunk_kda_fused_h_o_kernel_hb(
    seq_id_ref,  # [NC] prefetch
    start_flag_ref,  # [NC]
    end_flag_ref,  # [NC]
    q_ref,  # [1, BT, H, K]
    k_ref,  # [1, BT, H, K]  kg
    v_ref,  # [1, BT, H, V]  u
    w_ref,  # [1, BT, H, K]
    g_ref,  # [1, BT, H, K]  g_cumsum f32
    A_ref,  # [1, BT, H, BT]
    h0_ref,  # [1, H, K, V] or None
    o_ref,  # [1, BT, H, V] out, stored directly in input dtype
    ht_ref,  # [1, H, K, V] out or None
    scratch_ref,  # [H, K, V] fp32, with all head states resident in VMEM
    *,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    NUM_HEADS,
):
    idx_c = pl.program_id(0)

    @pl.when(start_flag_ref[idx_c] == 1)
    def _():
        scratch_ref[...] = jnp.zeros_like(scratch_ref)
        if USE_INITIAL_STATE:
            scratch_ref[...] = h0_ref[0].astype(jnp.float32)

    # Vectorize elementwise work across H and interleave MXU issue by stage:
    # A computes residuals, B/C compute o, and D updates state.
    q_all = q_ref[0].astype(jnp.float32)  # [BT, H, K]
    k_all = k_ref[0].astype(jnp.float32)
    v_all = v_ref[0].astype(jnp.float32)  # [BT, H, V]
    w_all = w_ref[0].astype(jnp.float32)
    g_all = g_ref[0].astype(jnp.float32)
    A_all = A_ref[0].astype(jnp.float32)  # [BT, H, BT]
    S_all = scratch_ref[...]  # [H, K, V]

    BT = q_ref.shape[1]
    g0 = g_all[0:1]  # [1, H, K]
    qg_all = q_all * exp2(jnp.maximum(g_all - g0, -126.0))
    h_scale = exp2(jnp.maximum(g0[0], -126.0))  # [H, K]
    g_last = g_all[BT - 1]  # [H, K]
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    A_mask = jnp.where(m_s[:, None, :], A_all, 0.0)  # [BT, H, BT]

    HI = jax.lax.Precision.HIGHEST
    bv = [
        v_all[:, h]
        - jnp.dot(w_all[:, h], S_all[h], precision=HI, preferred_element_type=jnp.float32)
        for h in range(NUM_HEADS)
    ]
    o1 = [
        scale
        * jnp.dot(
            qg_all[:, h],
            S_all[h] * h_scale[h][:, None],
            precision=HI,
            preferred_element_type=jnp.float32,
        )
        for h in range(NUM_HEADS)
    ]
    o2 = [
        jnp.dot(A_mask[:, h], bv[h], precision=HI, preferred_element_type=jnp.float32)
        for h in range(NUM_HEADS)
    ]
    o_ref[0] = jnp.stack([a + b for a, b in zip(o1, o2)], axis=1).astype(o_ref.dtype)

    upd = [
        jnp.dot(k_all[:, h].T, bv[h], precision=HI, preferred_element_type=jnp.float32)
        for h in range(NUM_HEADS)
    ]
    scratch_ref[...] = S_all * exp2(g_last)[:, :, None] + jnp.stack(upd, axis=0)

    @pl.when(end_flag_ref[idx_c] == 1)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0] = scratch_ref[...].astype(ht_ref.dtype)


def chunk_kda_fused_h_o_hb(
    q,
    kg,
    w,
    u,
    g_cumsum,
    A,
    scale,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
):
    """Run head-block K2 serially over grid=(NC,) with native [1,T,H,D] input.

    The output o is stored directly in the input dtype.
    """
    B, T, H, K = q.shape
    V = u.shape[-1]
    BT = chunk_size
    assert cu_seqlens is not None
    assert B == 1 and T % BT == 0 and H % 8 == 0
    N = cu_seqlens.shape[-1] - 1
    NC = T // BT

    cu_i32 = cu_seqlens.astype(jnp.int32)
    chunk_bos = jnp.arange(NC, dtype=jnp.int32) * BT
    seq_id = jnp.clip(jnp.searchsorted(cu_i32[1:], chunk_bos, side="right"), 0, N - 1).astype(
        jnp.int32
    )
    start_flag = (chunk_bos == cu_i32[seq_id]).astype(jnp.int32)
    end_flag = (chunk_bos + BT == cu_i32[seq_id + 1]).astype(jnp.int32)

    def _spec(last_dim):
        return pl.BlockSpec(block_shape=(1, BT, H, last_dim), index_map=lambda c, *_: (0, c, 0, 0))

    _state_map = lambda c, seq_id_ref, *_: (seq_id_ref[c], 0, 0, 0)
    h0_blockspec = (
        pl.BlockSpec([1, H, K, V], index_map=_state_map) if initial_state is not None else None
    )
    ht_blockspec = pl.BlockSpec([1, H, K, V], index_map=_state_map) if output_final_state else None
    ht_spec = jax.ShapeDtypeStruct([N, H, K, V], jnp.float32) if output_final_state else None

    o_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_kda_fused_h_o_kernel_hb,
            scale=scale,
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            NUM_HEADS=H,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(NC,),
            in_specs=[_spec(K), _spec(K), _spec(V), _spec(K), _spec(K), _spec(BT), h0_blockspec],
            out_specs=[_spec(V), ht_blockspec],
            scratch_shapes=[pltpu.VMEM((H, K, V), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        out_shape=[jax.ShapeDtypeStruct([1, T, H, V], q.dtype), ht_spec],
        interpret=get_interpret(),
    )(seq_id, start_flag, end_flag, q, kg, u, w, g_cumsum, A, initial_state)

    return o_out, ht_out


@functools.partial(
    jax.jit,
    static_argnames=(
        "scale",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "chunk_size",
        "safe_gate",
        "lower_bound",
        "use_gate_in_kernel",
        "disable_recompute",
        "return_intermediate_states",
        "cp_context",
        "transpose_state_layout",
        "fuse",
        "unified_layout",
        "flat_grid",
        "head_block",
        "return_backward_intermediates",
    ),
)
def chunk_kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    initial_state: jax.Array,
    output_final_state: bool,
    cu_seqlens: jax.Array,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: None = None,
    transpose_state_layout: bool = False,
    fuse: bool = True,
    unified_layout: bool = True,
    flat_grid: bool = True,
    head_block: bool = True,
    return_backward_intermediates: bool = False,
):
    """Run KDA chunked forward for varlen-packed B=1 inputs.

    ``cu_seqlens`` is required. Four compile-time ablation switches address
    three bottleneck classes. All enabled is the shipped configuration; all
    disabled reproduces the original upstream four-stage kernel.

    Compute: ``safe_gate`` ports the FlashKDA compute path to TPU for models
    with bounded gates. When enabled, Aqk/L construction is decomposed exactly
    into per-sub-chunk strip GEMMs on the MXU, and a finite Neumann factor chain
    (I-L)(I+L^2)... is applied directly to the RHS in a single bf16 pass by
    ``_neumann_fused_wide`` without materializing the inverse. The method comes
    from MoonshotAI/FlashKDA: finite-series inversion in one low-precision pass.
    TPU uses native bf16 and MXU-friendly BT=64/128 rather than FlashKDA's C=16,
    which targets fp16 range and GPU SM occupancy. When disabled, the upstream
    elementwise decay tensor and row-wise forward elimination are used. The
    Neumann solve is stable only for bounded gates and must remain coupled to
    safe_gate rather than being exposed on the generic softplus path.

    Data movement: with ``fuse=True``, stages 1+2 (gate activation and cumsum)
    are fused into the intra kernel, while stages 3+4 (recurrence and output)
    are fused so h/v_new stay in VMEM. ``fuse=False`` uses the original four
    stages. With ``unified_layout=True``, [1,H,T,D] identity-block addressing
    eliminates gather/scatter, transposes, and fp32 materialization between
    kernels. The legacy path retains those operations. Unified addressing
    requires fuse=True because the four-stage pipeline has per-stage glue.

    Scheduling: with ``flat_grid=True``, recurrence walks packed chunks once in
    O(chunks), with seq_id/is_first/is_last prefetched to identify boundaries.
    For fuse=False, this uses ``chunk_gated_delta_rule_fwd_h_flat`` from PR #3.
    The legacy (N, H, NT_max) grid scans the global chunk range once per
    sequence and costs O(N x chunks).

    Returns:
        12-tuple: o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
        w/u/qg/kg/v_new/h are always None. g_cumsum is returned only when
        use_gate_in_kernel=False.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size

    assert use_qk_l2norm_in_kernel is False
    assert cp_context is None
    assert not transpose_state_layout
    assert not return_intermediate_states
    assert not disable_recompute
    assert (
        fuse or not unified_layout
    ), "unified_layout=True requires fuse=True (the 4-stage pipeline has its own per-stage glue)"
    if safe_gate and use_gate_in_kernel and lower_bound is None:
        raise ValueError(
            "`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`."
        )

    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert cu_seqlens is not None, "cu_seqlens must not be None for varlen path"
    assert B == 1, f"varlen requires B=1 (packed layout), got B={B}"

    N = cu_seqlens.shape[-1] - 1
    assert_shape(beta, (B, T, H), "beta")
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")

    # Varlen alignment
    _orig_cu_seqlens = cu_seqlens
    T_input = T
    [q, k, v, g], [beta], cu_seqlens, _ = _align_seqs(
        [q, k, v, g],
        [beta],
        cu_seqlens,
        align=BT,
    )
    T = q.shape[1]

    assert T % BT == 0

    # Fix: _align_seqs pads g with 0, but softplus(0 + dt_bias) != 0 when
    # use_gate_in_kernel=True, producing non-zero gate activation at padding
    # positions.  Set padding g to a large negative so the activated gate is ~0.
    if use_gate_in_kernel:
        orig_lens = _orig_cu_seqlens[1:] - _orig_cu_seqlens[:-1]
        aligned_starts = cu_seqlens[:-1]
        pos = jnp.arange(T)
        in_range = (pos[None, :] >= aligned_starts[:, None]) & (
            pos[None, :] < (aligned_starts + orig_lens)[:, None]
        )
        valid_mask = in_range.any(axis=0)  # [T]
        g = jnp.where(valid_mask[None, :, None, None], g, -1e4)

    in_dtype = q.dtype

    # head_block uses native [1,T,H,D] with full-H blocks. It requires H%8==0
    # and otherwise falls back to unified addressing. The path removes
    # transposes, reduces grid steps by H, and interleaves H independent K2
    # chains within each block. It is inference-only: it does not produce Akk,
    # and qg/disable_recompute is unsupported as asserted above.
    #
    # The head-block fast path also requires safe_gate. With safe_gate=False,
    # unrolling full-H blocks overlaps the live [BT,BT,K] fp32 elementwise decay
    # tensors and exceeds scoped VMEM at K=128/H=16 (45 MB > 32 MB). The generic
    # path falls back to unified addressing.
    use_hb = head_block and fuse and (H % 8 == 0) and safe_gate

    if fuse and use_hb:
        w_n, u_n, kg_n, Aqk_n, gcum_n = kda_fwd_intra_hb(
            q,
            k,
            v,
            gk=g,
            beta=beta,
            scale=scale,
            chunk_size=BT,
            safe_gate=safe_gate,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
        o, final_state = chunk_kda_fused_h_o_hb(
            q=q,
            kg=kg_n,
            w=w_n,
            u=u_n,
            g_cumsum=gcum_n,
            A=Aqk_n,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=BT,
            cu_seqlens=cu_seqlens,
        )
        Aqk = Aqk_n if return_backward_intermediates else None
        Akk = None  # The head-block inference path does not compute (I+L)^-1.
        g_cumsum = gcum_n if (return_backward_intermediates and not use_gate_in_kernel) else None
    elif fuse:
        if unified_layout:
            # Unified addressing: transpose once to [1,H,T,D]; no K1/K2 copy.
            to_u = lambda x: jnp.transpose(x, (0, 2, 1, 3))
            q_u, k_u, v_u, g_u = to_u(q), to_u(k), to_u(v), to_u(g)
            beta_u = to_u(beta.reshape(B, T, H, 1))

            w_u, u_u, qg_u, kg_u, Aqk_u, Akk_u, gcum_u = kda_fwd_intra(
                q_u,
                k_u,
                v_u,
                gk=g_u,
                beta=beta_u,
                scale=scale,
                safe_gate=safe_gate,
                chunk_size=BT,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                unified_layout=True,
            )
            o, final_state = chunk_kda_fused_h_o(
                q=q_u,
                kg=kg_u,
                w=w_u,
                u=u_u,
                g_cumsum=gcum_u,
                A=Aqk_u,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=BT,
                cu_seqlens=cu_seqlens,
                unified_in=True,
                flat_grid=flat_grid,
            )
            if return_backward_intermediates:
                un_u = lambda x: jnp.transpose(x, (0, 2, 1, 3))
                Aqk, Akk = un_u(Aqk_u), un_u(Akk_u)
                g_cumsum = None if use_gate_in_kernel else un_u(gcum_u)
            else:
                Aqk, Akk, g_cumsum = None, None, None
        else:
            # Fused stages with legacy addressing to measure gather/scatter cost.
            w_, u_, qg_, kg_, Aqk, Akk, gcum_ = kda_fwd_intra(
                q,
                k,
                v,
                gk=g,
                beta=beta,
                scale=scale,
                safe_gate=safe_gate,
                chunk_size=BT,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                unified_layout=False,
                cu_seqlens=cu_seqlens,
            )
            o, final_state = chunk_kda_fused_h_o(
                q=q,
                kg=kg_,
                w=w_,
                u=u_,
                g_cumsum=gcum_,
                A=Aqk,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=BT,
                cu_seqlens=cu_seqlens,
                unified_in=False,
                flat_grid=flat_grid,
            )
            g_cumsum = None if use_gate_in_kernel else gcum_
            if not return_backward_intermediates:
                Aqk, Akk, g_cumsum = None, None, None
    else:
        # Original upstream four-stage pipeline used as the ablation baseline.
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, max_T=T)
        if use_gate_in_kernel:
            assert A_log is not None
            g_cumsum_ = kda_gate_chunk_cumsum(
                g=g,
                A_log=A_log,
                chunk_size=BT,
                scale=_RCP_LN2,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        else:
            g_cumsum_ = pallas_kda_gate_cumsum(
                g=g,
                scale=_RCP_LN2,
                chunk_size=BT,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        w_, u_, qg_, kg_, Aqk, Akk, _unused = kda_fwd_intra(
            q,
            k,
            v,
            gk=g_cumsum_,
            beta=beta,
            scale=scale,
            safe_gate=safe_gate,
            chunk_size=BT,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            unified_layout=False,
            pre_cumsum=True,
            cu_seqlens=cu_seqlens,
        )
        fwd_h = chunk_gated_delta_rule_fwd_h_flat if flat_grid else chunk_gated_delta_rule_fwd_h
        h_, v_new_, final_state = fwd_h(
            k=kg_,
            w=w_,
            u=u_,
            gk=g_cumsum_,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=BT,
            use_exp2=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        o = chunk_kda_fwd_o_gk(
            q=q,
            v=v_new_,
            g=g_cumsum_,
            A=Aqk,
            h=h_,
            scale=scale,
            chunk_size=BT,
            use_exp2=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        g_cumsum = None if use_gate_in_kernel else g_cumsum_
        if not return_backward_intermediates:
            Aqk, Akk, g_cumsum = None, None, None

    # Cast output back to input dtype (e.g. bfloat16)
    o = o.astype(in_dtype)

    # Unalign output
    o = _unalign_output(o, _orig_cu_seqlens, cu_seqlens, T_input)

    # Release intermediates
    w, u, qg, kg, v_new, h = None, None, None, None, None, None

    return o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
