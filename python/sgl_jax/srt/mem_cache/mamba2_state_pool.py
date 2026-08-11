"""Mamba2 recurrent state pool.

RecurrentStatePool assumes a SQUARE per-head state ``(slots, H, d, d)`` and a
conv ``proj_size = H*head_dim + 2*num_k_heads*head_k_dim``. Mamba2's SSM state
is non-square ``(H=96, P=80, N=128)`` and its conv width is ``conv_dim=9728``
(= d_inner + 2*n_groups*ssm_state), which the GDN proj formula can't express.

This subclass keeps RecurrentStatePool's public contract
(``get_linear_recurrent_layer_cache`` / ``replace_buffer`` / ``clear`` / pytree)
but overrides the buffer shapes to the true Mamba2 layout:
  * recurrent: ``(slots, H, P, N)``  fp32
  * conv:      ``(slots, conv_dim, K-1)``  bf16
Both pinned to the ``"tensor"`` axis on the head/channel dim for TP.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.mem_cache.recurrent_state_pool import (
    RecurrentStatePool,
    recurrent_state_dtype,
)


@register_pytree_node_class
class Mamba2StatePool(RecurrentStatePool):
    def __init__(
        self,
        linear_recurrent_layer_ids: list[int],
        size: int,
        num_heads: int,  # H = 96
        head_dim: int,  # P = 80
        ssm_state_size: int,  # N = 128
        conv_dim: int,  # 9728
        conv_kernel_size: int,  # K = 4
        mesh: Mesh,
        dp_size: int = 1,
        temporal_dtype=None,
        conv_dtype=None,
    ):
        state_dtype = recurrent_state_dtype()
        self.temporal_dtype = temporal_dtype or state_dtype.temporal
        self.conv_dtype = conv_dtype or state_dtype.conv

        assert len(set(linear_recurrent_layer_ids)) == len(linear_recurrent_layer_ids)
        assert size % dp_size == 0, f"size {size} not divisible by dp_size {dp_size}"

        self.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids)
        self.layers_mapping = {
            lid: i for i, lid in enumerate(self.linear_recurrent_layer_ids)
        }
        self.num_linear_recurrent_layers = len(self.linear_recurrent_layer_ids)

        self.size = size
        self.dp_size = dp_size
        self.slots_per_rank = size // dp_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ssm_state_size = ssm_state_size
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        # Kept for RecurrentStatePool API compatibility (unused by Mamba2 math).
        self.num_k_heads = num_heads
        self.head_k_dim = head_dim
        self.proj_size = conv_dim

        self.total_slots = size + dp_size  # +1 dummy slot per DP rank

        self.mesh = mesh
        self.recurrent_partition_axis = "tensor"
        self.conv_partition_axis = "tensor"
        self.data_partition_axis = "data"

        tp = mesh.shape["tensor"]
        assert num_heads % tp == 0, f"num_heads {num_heads} not divisible by tp {tp}"
        assert conv_dim % tp == 0, f"conv_dim {conv_dim} not divisible by tp {tp}"

        # recurrent: shard heads over "tensor"; conv: shard channels over "tensor".
        self.recurrent_sharding = NamedSharding(mesh, P("data", "tensor", None, None))
        self.conv_sharding = NamedSharding(mesh, P("data", "tensor", None))

        self.recurrent_buffers, self.conv_buffers = self._create_buffers()

    def _create_buffers(self):
        rec_shape = (self.total_slots, self.num_heads, self.head_dim, self.ssm_state_size)
        conv_shape = (self.total_slots, self.conv_dim, self.conv_kernel_size - 1)
        tdt = self.temporal_dtype
        cdt = self.conv_dtype
        with self.mesh:
            rec = []
            for _ in range(self.num_linear_recurrent_layers):
                rec.append(
                    jax.jit(
                        lambda: jnp.zeros(rec_shape, dtype=tdt),
                        out_shardings=self.recurrent_sharding,
                    )()
                )
            conv = []
            for _ in range(self.num_linear_recurrent_layers):
                buf = jax.jit(
                    lambda: jnp.zeros(conv_shape, dtype=cdt),
                    out_shardings=self.conv_sharding,
                )()
                conv.append([buf])
        return rec, conv

    # --- pytree (own aux to carry Mamba2-specific shape fields) ---
    def tree_flatten(self):
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.size,
            self.dp_size,
            self.total_slots,
            self.num_heads,
            self.head_dim,
            self.ssm_state_size,
            self.conv_dim,
            self.conv_kernel_size,
            self.temporal_dtype,
            self.conv_dtype,
            self.mesh,
            self.recurrent_sharding,
            self.conv_sharding,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            lids,
            size,
            dp_size,
            total_slots,
            num_heads,
            head_dim,
            ssm_state_size,
            conv_dim,
            conv_kernel_size,
            temporal_dtype,
            conv_dtype,
            mesh,
            recurrent_sharding,
            conv_sharding,
        ) = aux
        obj = cls.__new__(cls)
        obj.linear_recurrent_layer_ids = list(lids)
        obj.layers_mapping = {lid: i for i, lid in enumerate(obj.linear_recurrent_layer_ids)}
        obj.num_linear_recurrent_layers = len(obj.linear_recurrent_layer_ids)
        obj.size = size
        obj.dp_size = dp_size
        obj.slots_per_rank = size // dp_size
        obj.total_slots = total_slots
        obj.num_heads = num_heads
        obj.head_dim = head_dim
        obj.ssm_state_size = ssm_state_size
        obj.conv_dim = conv_dim
        obj.conv_kernel_size = conv_kernel_size
        obj.num_k_heads = num_heads
        obj.head_k_dim = head_dim
        obj.proj_size = conv_dim
        obj.temporal_dtype = temporal_dtype
        obj.conv_dtype = conv_dtype
        obj.mesh = mesh
        obj.recurrent_partition_axis = "tensor"
        obj.conv_partition_axis = "tensor"
        obj.data_partition_axis = "data"
        obj.recurrent_sharding = recurrent_sharding
        obj.conv_sharding = conv_sharding
        rec, conv = children
        obj.recurrent_buffers = list(rec)
        obj.conv_buffers = [list(inner) for inner in conv]
        return obj
