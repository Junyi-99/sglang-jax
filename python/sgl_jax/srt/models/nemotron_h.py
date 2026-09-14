"""NVIDIA Nemotron-3-Nano-4B (model_type ``nemotron_h``) for sglang-jax.

A FLAT stack of 42 single-mixer layers (no MoE in the 4B):
  * Mamba2 SSM   (M, 21 layers)  -> Mamba2Mixer + Mamba2AttnBackend (recurrent state)
  * GQA attention (*, 4 layers)  -> RadixAttention (KV cache), NO RoPE
  * dense relu² MLP (-, 17 layers)

Each layer: ``x = residual + mixer(RMSNorm(x))``; final ``norm_f`` then ``lm_head``.
Weight names (HF safetensors, ``backbone`` prefix):
  backbone.embeddings.weight, backbone.layers.{i}.norm.weight,
  backbone.layers.{i}.mixer.*, backbone.norm_f.weight, lm_head.weight.

Mirrors qwen3_5.py's structure (per-index dispatch, RadixLinearAttention wiring,
deferred-residual pre-norm, model __call__ returning pool updates + load_weights).
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.nemotron_h_layers import parse_layer_pattern
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


# ===========================================================================
# Mamba2 mixer (M layers)
# ===========================================================================
class NemotronHMamba2Mixer(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype=jnp.bfloat16):
        self.mesh = mesh
        self.layer_id = layer_id
        self.dtype = dtype
        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.ssm_state_size = config.ssm_state_size
        self.n_groups = config.n_groups
        self.conv_kernel = config.conv_kernel
        self.intermediate_size = self.num_heads * self.head_dim  # d_inner 7680
        self.groups_state = self.n_groups * self.ssm_state_size  # 1024
        self.conv_dim = self.intermediate_size + 2 * self.groups_state  # 9728
        self.eps = config.layer_norm_epsilon
        # in_proj output split: [gate(d_inner) | hidden_BC(conv_dim) | dt(num_heads)]
        proj_size = self.intermediate_size + self.conv_dim + self.num_heads

        self.in_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=proj_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="in_proj",
        )
        # conv1d weight container: laid out [conv_dim, K] for short_convolution.
        self.conv1d = LinearBase(
            input_size=self.conv_dim,
            output_size=self.conv_kernel,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="conv1d",
        )
        self.conv1d_bias = nnx.Param(
            jnp.zeros((self.conv_dim,), dtype=dtype, out_sharding=P("tensor"))
        )
        self.A_log = nnx.Param(
            jnp.zeros((self.num_heads,), dtype=jnp.float32, out_sharding=P("tensor"))
        )
        self.D = nnx.Param(
            jnp.ones((self.num_heads,), dtype=jnp.float32, out_sharding=P("tensor"))
        )
        self.dt_bias = nnx.Param(
            jnp.zeros((self.num_heads,), dtype=jnp.float32, out_sharding=P("tensor"))
        )
        # Gated grouped RMSNorm over d_inner (group_size = d_inner // n_groups).
        # Stored weight is [d_inner]; we apply grouped rmsnorm then silu(gate).
        self.norm_weight = nnx.Param(
            jnp.ones((self.intermediate_size,), dtype=jnp.float32, out_sharding=P("tensor"))
        )
        self.group_size = self.intermediate_size // self.n_groups  # 960

        self.out_proj = LinearBase(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="out_proj",
        )

        self.self_attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_heads,
            num_k_heads=self.num_heads,
            num_v_heads=self.num_heads,
            head_q_dim=self.head_dim,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            conv1d=self.conv1d,
            bias=self.conv1d_bias,  # conv1d bias [conv_dim]
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )
        # Stash D on the dispatch layer for the backend (wrap so nnx keeps it a leaf).
        self.self_attn.D = nnx.data(self.D)

    def _gated_norm(self, y, gate):
        """Gated grouped RMSNorm matching MambaRMSNormGated (norm_before_gate=False):
        gate FIRST, then per-group RMSNorm, then scale.
            x = y * silu(gate);  out = rmsnorm_grouped(x) * weight
        (mamba_ssm rmsnorm_fn applies z-gate before the norm when
        norm_before_gate=False.)"""
        T = y.shape[0]
        yf = y.astype(jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))
        yf = yf.reshape(T, self.n_groups, self.group_size)
        var = jnp.mean(yf * yf, axis=-1, keepdims=True)
        yf = yf * jax.lax.rsqrt(var + self.eps)
        yf = yf.reshape(T, self.intermediate_size)
        yf = yf * self.norm_weight.value.astype(jnp.float32)
        return yf.astype(self.dtype)

    def _shard(self, x):
        # Reshard a logically-contiguous [T, C] slice so its channel axis is
        # evenly head/group-striped across "tensor" (each TP rank gets its
        # head/group shard). No-op at TP=1. Mirrors qwen3_5 GDN._shard_dt.
        return jax.sharding.reshard(x, P("data", "tensor"))

    def __call__(self, hidden_states, forward_batch, recurrent_state_pool):
        proj, _ = self.in_proj(hidden_states)  # [T, d_inner + conv_dim + H]
        d_inner = self.intermediate_size
        gstate = self.groups_state
        gate = proj[:, :d_inner]
        hbc = proj[:, d_inner : d_inner + self.conv_dim]  # [x | B | C]
        dt = proj[:, d_inner + self.conv_dim :]  # [T, H]
        x = hbc[:, :d_inner]
        Bm = hbc[:, d_inner : d_inner + gstate]
        Cm = hbc[:, d_inner + gstate :]

        # Reshard each component so it is head/group-striped per TP rank; the
        # backend re-concatenates into rank-major [x_r|B_r|C_r] for the conv
        # (matching the rank-major-striped conv1d weight from the loader).
        gate = self._shard(gate)
        dt = self._shard(dt)
        x = self._shard(x)
        Bm = self._shard(Bm)
        Cm = self._shard(Cm)

        # backend slots: q=x, k=B, v=C, a=dt; gate handled here.
        y, attn_state = self.self_attn(
            forward_batch, x, Bm, Cm, dt, None, recurrent_state_pool
        )
        # y: [T, d_inner]; gated grouped RMSNorm + silu gate, then out_proj.
        y = self._gated_norm(y, gate)
        out, _ = self.out_proj(y)
        return out, attn_state


# ===========================================================================
# GQA attention (* layers) -- NO RoPE
# ===========================================================================
class NemotronHAttention(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype=jnp.bfloat16):
        self.mesh = mesh
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        # The model runner's configure_for_tensor_parallel already mutated
        # config.num_key_value_heads to the PADDED total (= tp when tp>orig_kv,
        # e.g. 30B 2->4). Size k/v projections to that padded count so they shard
        # cleanly; the loader replicates the original-count HF weight up to it.
        self.num_kv_heads_padded = int(config.num_key_value_heads)

        self.q_proj = LinearBase(
            input_size=self.hidden_size, output_size=self.num_heads * self.head_dim,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size, output_size=self.num_kv_heads_padded * self.head_dim,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size, output_size=self.num_kv_heads_padded * self.head_dim,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim, output_size=self.hidden_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None), scope_name="o_proj",
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads_padded,
            head_dim=self.head_dim,
            scaling=self.scaling,
            layer_id=layer_id,
        )

    def __call__(self, hidden_states, forward_batch, token_to_kv_pool):
        T = hidden_states.shape[0]
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = q.reshape(T, self.num_heads, self.head_dim,
                      out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)))
        k = k.reshape(T, self.num_kv_heads_padded, self.head_dim,
                      out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)))
        v = v.reshape(T, self.num_kv_heads_padded, self.head_dim,
                      out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)))
        attn_out, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        attn_out = attn_out.reshape(T, self.num_heads * self.head_dim)
        out, _ = self.o_proj(attn_out)
        return out, kv_fused


# ===========================================================================
# dense relu² MLP (- layers)
# ===========================================================================
class NemotronHMLP(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype=jnp.bfloat16):
        self.up_proj = LinearBase(
            input_size=config.hidden_size, output_size=config.intermediate_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            input_size=config.intermediate_size, output_size=config.hidden_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None), scope_name="down_proj",
        )

    def __call__(self, hidden_states):
        h, _ = self.up_proj(hidden_states)
        h = jax.nn.relu(h)
        h = h * h
        out, _ = self.down_proj(h)
        return out


# ===========================================================================
# MoE FFN (E layers, 30B variant): non-gated relu² routed experts + shared expert.
# Router: linear gate -> softmax -> top-k -> (renorm) -> * routed_scaling_factor.
# Each expert (routed + shared) is a NON-gated relu² FFN: down(relu(up(x))**2).
# Correctness-first: dense masked dispatch over all experts via stacked-weight
# einsum (every token visits every expert, masked by routing weight). Optimize later.
# ===========================================================================
class NemotronHMoE(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype=jnp.bfloat16):
        self.mesh = mesh
        self.layer_id = layer_id
        self.dtype = dtype
        def _cfg(*names, default=None):
            for n in names:
                v = getattr(config, n, None)
                if v is not None:
                    return v
            return default

        self.hidden_size = config.hidden_size
        self.num_experts = int(_cfg("n_routed_experts", "num_experts"))
        self.top_k = int(_cfg("num_experts_per_tok", "moe_top_k", "num_experts_per_token"))
        self.moe_inter = int(_cfg("moe_intermediate_size"))
        self.norm_topk = bool(_cfg("norm_topk_prob", "moe_renormalize", default=True))
        self.routed_scale = float(_cfg("routed_scaling_factor", default=1.0))
        self.n_shared = int(_cfg("n_shared_experts", "num_shared_experts", default=0) or 0)
        shared_inter = int(
            _cfg("moe_shared_expert_intermediate_size", "shared_expert_intermediate_size",
                 default=0) or 0
        )
        # DeepSeek-V3-style noaux_tc router: sigmoid scoring + bias-corrected
        # grouped top-k. n_group/topk_group=1 here -> degenerates to plain top-k,
        # but the e_score_correction_bias is in the checkpoint so we honor it.
        self.n_group = int(_cfg("n_group", default=1) or 1)
        self.topk_group = int(_cfg("topk_group", default=1) or 1)

        # Router gate: hidden -> num_experts (no bias). Replicated.
        self.gate = LinearBase(
            input_size=self.hidden_size, output_size=self.num_experts,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, None), scope_name="moe_gate",
        )
        # Per-expert score correction bias (used ONLY for top-k selection).
        self.e_score_correction_bias = nnx.Param(
            jnp.zeros((self.num_experts,), dtype=jnp.float32, out_sharding=P(None))
        )
        # Stacked routed-expert weights: up [E, hidden, inter], down [E, inter, hidden].
        # Shard the expert intermediate dim over "tensor" (column/row parallel per expert).
        self.up = nnx.Param(
            jnp.zeros((self.num_experts, self.hidden_size, self.moe_inter), dtype=dtype,
                      out_sharding=P(None, None, "tensor"))
        )
        self.down = nnx.Param(
            jnp.zeros((self.num_experts, self.moe_inter, self.hidden_size), dtype=dtype,
                      out_sharding=P(None, "tensor", None))
        )
        # Shared expert (always applied, non-gated relu²).
        self.has_shared = self.n_shared > 0 and shared_inter > 0
        if self.has_shared:
            self.shared_up = LinearBase(
                input_size=self.hidden_size, output_size=shared_inter,
                mesh=mesh, use_bias=False, params_dtype=dtype,
                kernel_axes=(None, "tensor"), scope_name="shared_up",
            )
            self.shared_down = LinearBase(
                input_size=shared_inter, output_size=self.hidden_size,
                mesh=mesh, use_bias=False, params_dtype=dtype,
                kernel_axes=("tensor", None), scope_name="shared_down",
            )

    def __call__(self, hidden_states):
        T = hidden_states.shape[0]
        x = hidden_states
        # DeepSeek-V3 noaux_tc router: SIGMOID scores; add e_score_correction_bias
        # for SELECTION only; gate weights are the ORIGINAL sigmoid scores of the
        # selected experts; renorm; * routed_scaling_factor.
        logits, _ = self.gate(x)  # [T, E]
        scores = jax.nn.sigmoid(logits.astype(jnp.float32))  # [T, E]
        scores_for_choice = scores + self.e_score_correction_bias.value[None, :]
        # n_group/topk_group == 1 -> plain top-k over all experts (no grouping).
        _, topi = jax.lax.top_k(scores_for_choice, self.top_k)  # [T, k] selection
        topw = jnp.take_along_axis(scores, topi, axis=-1)  # original scores
        if self.norm_topk:
            topw = topw / (jnp.sum(topw, axis=-1, keepdims=True) + 1e-20)
        topw = topw * self.routed_scale

        # Dense [T, E] weight matrix via one-hot (sharding-clean; no scatter/vmap).
        onehot = jax.nn.one_hot(topi, self.num_experts, dtype=jnp.float32)  # [T, k, E]
        gate_w = jnp.sum(onehot * topw[:, :, None], axis=1)  # [T, E]

        # Dense expert FFN over all experts: up -> relu² -> down, then weight & sum.
        # h[e] = relu(x @ up[e])**2 ;  y[e] = h[e] @ down[e].
        # up weight is [E, hidden, inter] sharded on inter ("tensor"); the up einsum
        # contracts hidden (replicated) -> xe inter-sharded. The down einsum contracts
        # the inter ("tensor") axis on BOTH operands -> a partial sum needing an
        # all-reduce; specify out_sharding so XLA reduce-scatters to a replicated [E,T,H].
        xe = jnp.einsum("th,ehm->etm", x.astype(self.dtype), self.up.value)  # [E, T, inter]
        xe = jax.nn.relu(xe.astype(jnp.float32))
        xe = (xe * xe).astype(self.dtype)
        ye = jnp.einsum(
            "etm,emh->eth", xe, self.down.value,
            out_sharding=NamedSharding(self.mesh, P(None, None, None)),
        )  # [E, T, hidden] replicated
        # Weighted sum over experts: gate_w[T,E] -> [E,T,1].
        ye = ye.astype(jnp.float32) * jnp.transpose(gate_w, (1, 0))[:, :, None]
        routed = jnp.sum(ye, axis=0).astype(self.dtype)  # [T, hidden]

        if self.has_shared:
            s, _ = self.shared_up(x)
            s = jax.nn.relu(s)
            s = (s.astype(jnp.float32) * s.astype(jnp.float32)).astype(self.dtype)
            s, _ = self.shared_down(s)
            routed = routed + s
        return routed


# ===========================================================================
# Decoder layer (single mixer + input RMSNorm + residual)
# ===========================================================================
class NemotronHDecoderLayer(nnx.Module):
    def __init__(self, config, mesh, layer_id, block_type, dtype=jnp.bfloat16):
        self.layer_id = layer_id
        self.block_type = block_type  # "mamba" | "attention" | "mlp" | "moe"
        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.layer_norm_epsilon, param_dtype=jnp.float32
        )
        if block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, mesh, layer_id, dtype=dtype)
        elif block_type == "attention":
            self.mixer = NemotronHAttention(config, mesh, layer_id, dtype=dtype)
        elif block_type == "moe":
            self.mixer = NemotronHMoE(config, mesh, layer_id, dtype=dtype)
        else:
            self.mixer = NemotronHMLP(config, mesh, layer_id, dtype=dtype)

    def __call__(self, hidden_states, forward_batch, memory_pools):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        attn_state = None
        if self.block_type == "mamba":
            out, attn_state = self.mixer(
                hidden_states, forward_batch, memory_pools.recurrent_state_pool
            )
        elif self.block_type == "attention":
            out, attn_state = self.mixer(
                hidden_states, forward_batch, memory_pools.token_to_kv_pool
            )
        else:
            out = self.mixer(hidden_states)
        hidden_states = residual + out
        return hidden_states, attn_state


# ===========================================================================
# Model / CausalLM
# ===========================================================================
class NemotronHBackbone(nnx.Module):
    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        self.config = config
        layers = parse_layer_pattern(config.hybrid_override_pattern)
        self.block_types = list(layers.types)  # 'M'/'*'/'-' per index
        _CHAR = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}

        self.embeddings = Embed(
            num_embeddings=config.vocab_size, features=config.hidden_size,
            dtype=dtype, param_dtype=dtype, kernel_axes=("tensor", None), mesh=mesh,
        )
        self.layers = nnx.data(
            [
                NemotronHDecoderLayer(config, mesh, i, _CHAR[c], dtype=dtype)
                for i, c in enumerate(self.block_types)
            ]
        )
        self.norm_f = RMSNorm(
            config.hidden_size, epsilon=config.layer_norm_epsilon, param_dtype=jnp.float32
        )

    def __call__(self, forward_batch, memory_pools):
        hidden_states = self.embeddings(forward_batch.input_ids)
        layers_kv_fused = []
        layers_rec_buffers = []
        layers_conv_buffers = []
        for layer in self.layers:
            hidden_states, attn_state = layer(hidden_states, forward_batch, memory_pools)
            if layer.block_type == "attention":
                layers_kv_fused.append(attn_state)
            elif layer.block_type == "mamba":
                rec_buf, conv_buf_list = attn_state
                layers_rec_buffers.append(rec_buf)
                layers_conv_buffers.append(conv_buf_list)
        hidden_states = self.norm_f(hidden_states)
        return (
            hidden_states,
            layers_kv_fused,
            (layers_rec_buffers, layers_conv_buffers),
        )


class NemotronHForCausalLM(nnx.Module):
    def __init__(self, config: PretrainedConfig, mesh, dtype=jnp.bfloat16):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.backbone = NemotronHBackbone(config, mesh, dtype=dtype)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, dtype=dtype, param_dtype=dtype,
            kernel_axes=("tensor", None), mesh=mesh,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def __call__(self, forward_batch: ForwardBatch, memory_pools: MemoryPools,
                 logits_metadata: LogitsMetadata, pixel_values=None):
        hidden_states, layers_kv_fused, layers_rec_state = self.backbone(
            forward_batch, memory_pools
        )
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return (
            output,
            {
                "token_to_kv_pool": layers_kv_fused,
                "recurrent_state_pool": layers_rec_state,
            },
            True,
            [],
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _put(self, arr, spec, dtype=None):
        return jax.device_put(
            jnp.asarray(arr, dtype=dtype or self.dtype),
            NamedSharding(self.mesh, P(*spec)),
        )

    @staticmethod
    def _stripe_conv_dim(conv, mixer, tp):
        """Rearrange a [conv_dim, ...] tensor from component-major [x|B|C] to
        rank-major [x_r0|B_r0|C_r0 | x_r1|...] so each TP rank's contiguous
        conv_dim slice is its own [x|B|C] block (matches the backend's
        rank-major mixed_qkv). No-op at tp=1."""
        if tp <= 1:
            return conv
        d_inner = mixer.intermediate_size
        gstate = mixer.groups_state
        x_blk = conv[:d_inner]
        b_blk = conv[d_inner : d_inner + gstate]
        c_blk = conv[d_inner + gstate :]
        x_tp = d_inner // tp
        g_tp = gstate // tp
        blocks = []
        for r in range(tp):
            blocks.append(x_blk[r * x_tp : (r + 1) * x_tp])
            blocks.append(b_blk[r * g_tp : (r + 1) * g_tp])
            blocks.append(c_blk[r * g_tp : (r + 1) * g_tp])
        return np.concatenate(blocks, axis=0)

    def _load_moe_layer(self, m, src, read, key_to_file, tp):
        """Load one MoE (E) layer. Returns #tensors consumed.

        Discovers the exact HF naming from the checkpoint keys present under
        ``{src}.mixer.`` so we don't hardcode a guess; supports both
        per-expert ([E] separate tensors) and pre-stacked ([E, ...]) layouts.
        Router gate -> m.gate; routed up/down -> stacked m.up/m.down; shared
        expert up/down -> m.shared_up/m.shared_down. Expert FFN is non-gated:
        up [hidden, inter] @ x then relu² then down [inter, hidden].
        """
        E = m.num_experts
        prefix = f"{src}.mixer."
        layer_keys = [k for k in key_to_file if k.startswith(prefix)]
        consumed = 0

        def find(*cands):
            for c in cands:
                if c in key_to_file:
                    return c
            return None

        # --- router gate ---
        gate_key = find(
            f"{prefix}gate.weight", f"{prefix}router.weight",
            f"{prefix}router.gate.weight", f"{prefix}gate.gate.weight",
        )
        if gate_key is None:
            raise RuntimeError(f"MoE gate weight not found under {prefix}; keys={layer_keys[:8]}")
        # gate HF [E, hidden] -> LinearBase wants [hidden, E].
        m.gate.weight.value = self._put(read(gate_key).T, (None, None))
        consumed += 1

        # e_score_correction_bias (DeepSeek noaux_tc router selection bias).
        bias_key = find(
            f"{prefix}gate.e_score_correction_bias",
            f"{prefix}e_score_correction_bias",
            f"{prefix}gate.expert_bias",
        )
        if bias_key is not None:
            m.e_score_correction_bias.value = self._put(read(bias_key), (None,), dtype=jnp.float32)
            consumed += 1

        # --- routed experts: detect per-expert vs stacked up/down names ---
        up_names = ["up_proj", "up", "w1", "linear_fc1", "fc1", "gate_up_proj"]
        down_names = ["down_proj", "down", "w2", "linear_fc2", "fc2"]

        def _stack_experts(part_names):
            # Try pre-stacked single tensor first.
            for nm in part_names:
                k = find(f"{prefix}experts.{nm}", f"{prefix}experts.{nm}.weight")
                if k is not None:
                    return read(k), [k]  # [E, ...]
            # Else per-expert: {prefix}experts.{e}.{nm}.weight
            for nm in part_names:
                k0 = find(f"{prefix}experts.0.{nm}.weight", f"{prefix}experts.0.{nm}")
                if k0 is not None:
                    used = []
                    arrs = []
                    for e in range(E):
                        ke = find(f"{prefix}experts.{e}.{nm}.weight", f"{prefix}experts.{e}.{nm}")
                        arrs.append(read(ke))
                        used.append(ke)
                    return np.stack(arrs, axis=0), used  # [E, ...]
            return None, []

        up_stack, up_used = _stack_experts(up_names)
        down_stack, down_used = _stack_experts(down_names)
        if up_stack is None or down_stack is None:
            raise RuntimeError(
                f"MoE expert weights not found under {prefix}experts; keys={layer_keys[:12]}"
            )
        # Normalize to up [E, hidden, inter], down [E, inter, hidden].
        H = m.hidden_size
        inter = m.moe_inter
        up_stack = self._orient_expert(up_stack, E, H, inter, "up")
        down_stack = self._orient_expert(down_stack, E, inter, H, "down")
        m.up.value = self._put(up_stack, (None, None, "tensor"))
        m.down.value = self._put(down_stack, (None, "tensor", None))
        consumed += len(up_used) + len(down_used)

        # --- shared expert ---
        if m.has_shared:
            su = find(
                f"{prefix}shared_experts.up_proj.weight",
                f"{prefix}shared_expert.up_proj.weight",
                f"{prefix}shared_experts.up.weight",
            )
            sd = find(
                f"{prefix}shared_experts.down_proj.weight",
                f"{prefix}shared_expert.down_proj.weight",
                f"{prefix}shared_experts.down.weight",
            )
            if su is None or sd is None:
                raise RuntimeError(
                    f"MoE shared-expert weights not found under {prefix}; keys={layer_keys[:12]}"
                )
            m.shared_up.weight.value = self._put(read(su).T, (None, "tensor"))
            m.shared_down.weight.value = self._put(read(sd).T, ("tensor", None))
            consumed += 2
        return consumed

    @staticmethod
    def _replicate_kv(w, num_kv_heads, head_dim, replicas):
        """Block-replicate KV proj rows for tp>num_kv_heads. HF weight is
        [num_kv_heads*head_dim, hidden]; head h -> repeated `replicas` times,
        giving order [kv0,kv0,...,kv1,kv1,...] to match the KV pool's
        get_total_num_kv_heads_with_replication replicate strategy."""
        parts = []
        for h in range(num_kv_heads):
            blk = w[h * head_dim : (h + 1) * head_dim]
            for _ in range(replicas):
                parts.append(blk)
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _orient_expert(arr, E, in_dim, out_dim, which):
        """Return [E, in_dim, out_dim] from an expert weight stack that may be
        stored as [E, out_dim, in_dim] (HF nn.Linear convention) or already
        [E, in_dim, out_dim]."""
        assert arr.shape[0] == E, f"{which}: expected leading E={E}, got {arr.shape}"
        if arr.shape == (E, in_dim, out_dim):
            return arr
        if arr.shape == (E, out_dim, in_dim):
            return np.transpose(arr, (0, 2, 1))
        raise RuntimeError(
            f"{which} expert stack shape {arr.shape} matches neither "
            f"(E,{in_dim},{out_dim}) nor (E,{out_dim},{in_dim})"
        )

    def load_weights(self, model_config: ModelConfig):
        import glob
        import json
        import os
        import struct

        from sgl_jax.srt.utils.weight_utils import SequentialSafetensorManager

        cfg = model_config.hf_config
        layers = parse_layer_pattern(cfg.hybrid_override_pattern)
        block_types = list(layers.types)
        _CHAR = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}

        # Build {key: file} from safetensors headers (avoids mmap; matches the
        # WeightLoader._scan_weight_info convention).
        model_path = model_config.model_path
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise RuntimeError(f"No *.safetensors in {model_path}")
        key_to_file: dict[str, str] = {}
        for st in files:
            with open(st, "rb") as f:
                hsz = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hsz))
            for k in header:
                if k != "__metadata__":
                    key_to_file[k] = st

        tp = self.mesh.shape.get("tensor", 1)
        consumed = 0
        with SequentialSafetensorManager() as fm:
            def read(key):
                return np.asarray(fm.get_handle(key_to_file[key]).get_slice(key)[:])

            # top-level
            self.backbone.embeddings.embedding.value = self._put(
                read("backbone.embeddings.weight"), ("tensor", None)
            )
            self.backbone.norm_f.scale.value = self._put(
                read("backbone.norm_f.weight"), (None,), dtype=jnp.float32
            )
            self.lm_head.embedding.value = self._put(
                read("lm_head.weight"), ("tensor", None)
            )
            consumed += 3

            for i, c in enumerate(block_types):
                bt = _CHAR[c]
                layer = self.backbone.layers[i]
                src = f"backbone.layers.{i}"
                layer.norm.scale.value = self._put(
                    read(f"{src}.norm.weight"), (None,), dtype=jnp.float32
                )
                consumed += 1
                m = layer.mixer
                if bt == "mamba":
                    m.in_proj.weight.value = self._put(
                        read(f"{src}.mixer.in_proj.weight").T, (None, "tensor")
                    )
                    conv_w = read(f"{src}.mixer.conv1d.weight")  # [conv_dim,1,K]
                    conv_w = conv_w.reshape(conv_w.shape[0], conv_w.shape[-1])  # [conv_dim,K]
                    conv_w = self._stripe_conv_dim(conv_w, m, tp)
                    m.conv1d.weight.value = self._put(conv_w, ("tensor", None))
                    conv_b = self._stripe_conv_dim(
                        read(f"{src}.mixer.conv1d.bias")[:, None], m, tp
                    )[:, 0]
                    m.conv1d_bias.value = self._put(conv_b, ("tensor",))
                    m.A_log.value = self._put(
                        read(f"{src}.mixer.A_log"), ("tensor",), dtype=jnp.float32
                    )
                    m.D.value = self._put(
                        read(f"{src}.mixer.D"), ("tensor",), dtype=jnp.float32
                    )
                    m.dt_bias.value = self._put(
                        read(f"{src}.mixer.dt_bias"), ("tensor",), dtype=jnp.float32
                    )
                    m.norm_weight.value = self._put(
                        read(f"{src}.mixer.norm.weight"), ("tensor",), dtype=jnp.float32
                    )
                    m.out_proj.weight.value = self._put(
                        read(f"{src}.mixer.out_proj.weight").T, ("tensor", None)
                    )
                    consumed += 8
                elif bt == "attention":
                    m.q_proj.weight.value = self._put(
                        read(f"{src}.mixer.q_proj.weight").T, (None, "tensor")
                    )
                    kw = read(f"{src}.mixer.k_proj.weight")  # [orig_kv*hd, hidden] HF
                    vw = read(f"{src}.mixer.v_proj.weight")
                    # HF rows = orig_kv_heads * head_dim; pad up to the module's
                    # padded count by block-replicating each kv head.
                    orig_kv = kw.shape[0] // m.head_dim
                    if m.num_kv_heads_padded > orig_kv:
                        reps = m.num_kv_heads_padded // orig_kv
                        kw = self._replicate_kv(kw, orig_kv, m.head_dim, reps)
                        vw = self._replicate_kv(vw, orig_kv, m.head_dim, reps)
                    m.k_proj.weight.value = self._put(kw.T, (None, "tensor"))
                    m.v_proj.weight.value = self._put(vw.T, (None, "tensor"))
                    m.o_proj.weight.value = self._put(
                        read(f"{src}.mixer.o_proj.weight").T, ("tensor", None)
                    )
                    consumed += 4
                elif bt == "moe":
                    consumed += self._load_moe_layer(m, src, read, key_to_file, tp)
                else:  # mlp
                    m.up_proj.weight.value = self._put(
                        read(f"{src}.mixer.up_proj.weight").T, (None, "tensor")
                    )
                    m.down_proj.weight.value = self._put(
                        read(f"{src}.mixer.down_proj.weight").T, ("tensor", None)
                    )
                    consumed += 2

        expected = len(key_to_file)
        logger.info(
            "NemotronH load_weights: consumed=%d / checkpoint=%d tensors",
            consumed, expected,
        )
        if consumed != expected:
            raise RuntimeError(
                f"NemotronH load_weights mismatch: consumed={consumed} but "
                f"checkpoint has {expected} tensors."
            )


EntryClass = NemotronHForCausalLM
