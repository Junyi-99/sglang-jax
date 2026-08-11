"""Mamba2 (NemotronH SSD) linear-recurrent attention backend.

Mirrors ``GDNAttnBackend`` structurally but runs the pure-JAX Mamba2 SSD
(``mamba2_ssm``) instead of the gated-delta-rule. Stateless: the conv1d weight
(``[conv_dim, K]``) + bias, ``A_log`` / ``D`` / ``dt_bias`` recurrence params
live on the parent ``RadixLinearAttention`` and are read at call time.

Interface contract (set by the model layer, abusing the fixed q/k/v/a/b slots):
  * ``q`` = ``hidden_states_B_C``  ``[T, conv_dim]`` (pre-conv; gate already removed)
  * ``k`` = ``dt``                 ``[T, num_heads]``
  * v / a / b unused.
Returns ``(y_flat [T, num_heads*head_dim], (new_rec, [new_conv]))`` per the
linear-backend contract (model layer reshapes y back to per-head + applies the
gated output norm).

conv buffer width is ``K-1`` (vLLM/Mamba convention); we reuse
``short_convolution`` (depthwise causal conv with cu_seqlens) for both prefill
(EXTEND, ragged) and decode (single step). For TP>1, sharding is propagated by
JAX from the pool buffers (head/channel axis pinned to "tensor"); the SSD scan
is per-head independent so this is correct without an explicit shard_map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.layers.attention.linear.mamba2_ssm import (
    ssd_decode_step,
    ssd_ragged_prefill,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


def _mesh_tp_size(mesh) -> int:
    if mesh is None:
        return 1
    shape = getattr(mesh, "shape", None)
    if shape is None or "tensor" not in shape:
        return 1
    return int(shape["tensor"])


class Mamba2AttnBackend(LinearRecurrentAttnBackend):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ssm_state_size: int,
        n_groups: int,
        conv_kernel_size: int,
        mesh,
    ):
        super().__init__(mesh=mesh)
        self.num_heads = num_heads
        self.head_dim = head_dim  # P
        self.ssm_state_size = ssm_state_size  # N
        self.n_groups = n_groups
        self.conv_kernel_size = conv_kernel_size

        self.intermediate_size = num_heads * head_dim  # d_inner, 7680
        self.groups_state = n_groups * ssm_state_size  # 1024
        self.conv_dim = self.intermediate_size + 2 * self.groups_state  # 9728
        self.heads_per_group = num_heads // n_groups  # 12

    # ------------------------------------------------------------------
    def __call__(
        self,
        q: jax.Array,  # x  [T, d_inner]   P(data, tensor)
        k: jax.Array,  # B  [T, n_groups*N] P(data, tensor)
        v: jax.Array,  # C  [T, n_groups*N] P(data, tensor)
        a: jax.Array,  # dt [T, num_heads]  P(data, tensor)
        b: jax.Array | None,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        recurrent_state_pool: "RecurrentStatePool",
        **kwargs,
    ):
        x, Bm, Cm, dt_raw = q, k, v, a
        tp = _mesh_tp_size(self.mesh)
        T = x.shape[0]
        d_inner_tp = self.intermediate_size // tp
        gstate_tp = self.groups_state // tp
        # Rebuild rank-major mixed_qkv [x_r0|B_r0|C_r0 | x_r1|...] so each TP rank's
        # contiguous conv_dim slice is its own [x|B|C] block (matches the rank-major
        # striped conv1d weight from the loader). Collapses to [x|B|C] at tp=1.
        hidden_BC = jnp.concatenate(
            [
                x.reshape(T, tp, d_inner_tp),
                Bm.reshape(T, tp, gstate_tp),
                Cm.reshape(T, tp, gstate_tp),
            ],
            axis=-1,
        ).reshape(T, self.conv_dim)
        hidden_BC = jax.sharding.reshard(hidden_BC, P("data", "tensor"))

        conv_weight = layer.conv1d.weight.value  # [conv_dim, K]  P(tensor, None)
        conv_bias = layer.bias.value if layer.bias is not None else None  # [conv_dim] P(tensor)
        A_log = layer.A_log.value  # [H]  P(tensor)
        D = layer.D.value  # [H]
        dt_bias = layer.dt_bias.value  # [H]

        rec_state, conv_states = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
        conv_state = conv_states[0]  # [num_slots, conv_dim, K-1]  P(data, tensor, None)

        meta = self.forward_metadata
        state_indices = meta.recurrent_indices  # P(data)
        has_initial_state = meta.has_initial_state  # P(data)

        is_decode = forward_batch.forward_mode.is_decode()
        cu_seqlens = None if is_decode else meta.cu_q_lens  # P(data)

        out = self._run(
            hidden_BC, dt_raw, conv_weight, conv_bias, A_log, D, dt_bias,
            conv_state, rec_state, state_indices, has_initial_state, cu_seqlens,
            is_decode,
        )
        y, new_conv, new_rec = out
        T = y.shape[0]
        y = y.reshape(T, self.num_heads * self.head_dim)
        return y, (new_rec, [new_conv])

    def _run(
        self, hidden_BC, dt_raw, conv_weight, conv_bias, A_log, D, dt_bias,
        conv_state, rec_state, state_indices, has_initial_state, cu_seqlens,
        is_decode,
    ):
        tp = _mesh_tp_size(self.mesh)
        H_tp = self.num_heads // tp
        G_tp = self.n_groups // tp
        P_dim, N = self.head_dim, self.ssm_state_size
        d_inner_tp = H_tp * P_dim
        gstate_tp = G_tp * N
        hpg = self.heads_per_group
        mode = ForwardMode.DECODE if is_decode else ForwardMode.EXTEND

        def _local(
            hidden_BC_l, dt_raw_l, conv_w_l, conv_b_l, A_log_l, D_l, dt_bias_l,
            conv_state_l, rec_state_l, state_idx_l, has_init_l, cu_l,
        ):
            A_l = -jnp.exp(A_log_l.astype(jnp.float32))  # (H_tp,)

            # 1. conv1d + silu. Gather per-request conv cache by slot (local axis0).
            conv_cache_sel = conv_state_l[state_idx_l]  # [num_seqs, conv_dim_tp, K-1]
            if not is_decode:
                conv_cache_sel = jnp.where(
                    has_init_l[:, None, None], conv_cache_sel, 0.0
                )
            conv_out, new_conv_sel = short_convolution(
                hidden_BC_l, conv_w_l, conv_cache_sel,
                None if is_decode else cu_l, mode,
                bias=conv_b_l, activation="silu",
            )
            new_conv = conv_state_l.at[state_idx_l].set(new_conv_sel.astype(conv_state_l.dtype))

            # 2. split [x_tp | B_tp | C_tp].
            x = conv_out[:, :d_inner_tp]
            B_flat = conv_out[:, d_inner_tp : d_inner_tp + gstate_tp]
            C_flat = conv_out[:, d_inner_tp + gstate_tp :]
            T = conv_out.shape[0]
            x = x.reshape(T, H_tp, P_dim)
            B_mat = jnp.repeat(B_flat.reshape(T, G_tp, N), hpg, axis=1)  # (T, H_tp, N)
            C_mat = jnp.repeat(C_flat.reshape(T, G_tp, N), hpg, axis=1)

            # 3. SSD.
            if is_decode:
                y, new_rec = ssd_decode_step(
                    x, dt_raw_l, A_l, B_mat, C_mat, D_l, dt_bias_l,
                    rec_state_l, state_idx_l, has_init_l,
                )
            else:
                y, new_rec = ssd_ragged_prefill(
                    x, dt_raw_l, A_l, B_mat, C_mat, D_l, dt_bias_l,
                    cu_l, rec_state_l, state_idx_l, has_init_l,
                )
            return y, new_conv, new_rec

        return jax.shard_map(
            _local,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor"),  # hidden_BC
                P("data", "tensor"),  # dt_raw
                P("tensor", None),  # conv_weight
                P("tensor"),  # conv_bias
                P("tensor"),  # A_log
                P("tensor"),  # D
                P("tensor"),  # dt_bias
                P("data", "tensor", None),  # conv_state
                P("data", "tensor", None, None),  # rec_state
                P("data"),  # state_indices
                P("data"),  # has_initial_state
                P("data"),  # cu_seqlens
            ),
            out_specs=(
                P("data", "tensor", None),  # y [T, H_tp, P]
                P("data", "tensor", None),  # new_conv
                P("data", "tensor", None, None),  # new_rec
            ),
            check_vma=False,
        )(
            hidden_BC, dt_raw, conv_weight, conv_bias, A_log, D, dt_bias,
            conv_state, rec_state, state_indices,
            has_initial_state,
            cu_seqlens if cu_seqlens is not None else state_indices,
        )
