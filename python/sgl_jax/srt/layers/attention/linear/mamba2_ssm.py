"""Pure-JAX Mamba2 SSD math (correctness-first; no Pallas).

Adapted from ``jagged-mamba2/mamba2_v6/modeling.py`` (the tested fp32 reference
scan), extended to also emit per-sequence final SSM states so the recurrent
state pool can carry them across decode steps.

Conventions (match modeling_nemotron_h.NemotronHMamba2Mixer.torch_forward):
  * dt = softplus(dt + dt_bias); NO clamp (time_step_limit = (0, inf)).
  * A = -exp(A_log)  (per head, shape (H,)).
  * grouped B/C: n_groups groups repeated to H heads (H % n_groups == 0).
  * y += D[:, None] * x  (D skip, per head).
All recurrence is done in fp32; outputs returned in fp32 (caller casts).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _discretize_inputs(dt_raw, dt_bias):
    """softplus(dt + dt_bias), no clamp. dt_raw (T, H) -> (T, H) fp32."""
    return jax.nn.softplus(dt_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32))


def ssd_ragged_prefill(
    x,  # (T, H, P) fp32  -- post-conv hidden states reshaped per head
    dt_raw,  # (T, H)      raw dt (pre-softplus)
    A,  # (H,)            = -exp(A_log)
    B_mat,  # (T, H, N)    grouped B already repeated to H heads
    C_mat,  # (T, H, N)
    D,  # (H,)
    dt_bias,  # (H,)
    cu_seqlens,  # (num_seqs + 1,) int32
    initial_states,  # (num_slots, H, P, N) fp32  recurrent buffer
    state_indices,  # (num_seqs,) int32  slot id per sequence
    has_initial_state,  # (num_seqs,) bool
):
    """Varlen prefill. Returns (y (T, H, P) fp32, new_states (num_slots,H,P,N)).

    Per-token fp32 recurrence with a per-segment reset. Each sequence may carry
    an incoming state (continued prefill) selected from ``initial_states`` by
    ``state_indices`` when ``has_initial_state`` is set. Writes the final state
    of each sequence back into a copy of ``initial_states`` at its slot.
    """
    T, H, P = x.shape
    N = B_mat.shape[-1]
    x32 = x.astype(jnp.float32)
    B32 = B_mat.astype(jnp.float32)
    C32 = C_mat.astype(jnp.float32)

    dt = _discretize_inputs(dt_raw, dt_bias)  # (T, H)
    a = jnp.exp(A.astype(jnp.float32)[None, :] * dt)  # (T, H) decay per step

    # Segment bookkeeping: which sequence each token belongs to + reset flags.
    idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seg_id = jnp.searchsorted(cu_seqlens[1:], idx, side="right")  # (T,)
    prev_seg = jnp.concatenate([jnp.full((1,), -1, dtype=seg_id.dtype), seg_id[:-1]])
    reset = seg_id != prev_seg  # (T,) True at first token of each segment

    # Initial state per token's segment (only used on the reset token).
    # Gather the incoming state for the segment that starts at this token.
    init_for_seg = initial_states[state_indices]  # (num_seqs, H, P, N)
    init_for_seg = jnp.where(
        has_initial_state[:, None, None, None],
        init_for_seg,
        jnp.zeros_like(init_for_seg),
    )

    def step(carry, inp):
        h = carry  # (H, P, N) running state for the current segment
        a_t, dt_t, x_t, B_t, C_t, reset_t, seg_t = inp
        # On a reset token, replace h with that segment's incoming state.
        h = jnp.where(reset_t, init_for_seg[seg_t], h)
        # h = a*h + dt*x*B   (dt already folded into a via exp(A*dt); x scaled by dt)
        h = a_t[:, None, None] * h + dt_t[:, None, None] * x_t[:, :, None] * B_t[:, None, :]
        y_t = jnp.einsum("hpn,hn->hp", h, C_t)  # (H, P)
        return h, (y_t, h)

    h0 = jnp.zeros((H, P, N), dtype=jnp.float32)
    _, (y, h_all) = jax.lax.scan(
        step, h0, (a, dt, x32, B32, C32, reset, seg_id)
    )
    # y: (T, H, P), h_all: (T, H, P, N) state AFTER each token.

    # D skip connection.
    y = y + D.astype(jnp.float32)[None, :, None] * x32

    # Final state per sequence = state at the last token of each segment.
    last_tok = cu_seqlens[1:] - 1  # (num_seqs,) index of each sequence's last token
    final_states = h_all[last_tok]  # (num_seqs, H, P, N)
    new_states = initial_states.at[state_indices].set(final_states.astype(initial_states.dtype))
    return y, new_states


def ssd_decode_step(
    x,  # (B, H, P) fp32  post-conv hidden states
    dt_raw,  # (B, H)
    A,  # (H,)
    B_mat,  # (B, H, N)
    C_mat,  # (B, H, N)
    D,  # (H,)
    dt_bias,  # (H,)
    rec_states,  # (num_slots, H, P, N) fp32
    state_indices,  # (B,) int32
    has_initial_state,  # (B,) bool
):
    """Single recurrence step for B requests. Returns (y (B,H,P), new_states)."""
    Bsz, H, P = x.shape
    x32 = x.astype(jnp.float32)
    B32 = B_mat.astype(jnp.float32)
    C32 = C_mat.astype(jnp.float32)

    dt = _discretize_inputs(dt_raw, dt_bias)  # (B, H)
    a = jnp.exp(A.astype(jnp.float32)[None, :] * dt)  # (B, H)

    h_prev = rec_states[state_indices]  # (B, H, P, N)
    h_prev = jnp.where(
        has_initial_state[:, None, None, None], h_prev, jnp.zeros_like(h_prev)
    )
    h_new = (
        a[:, :, None, None] * h_prev
        + dt[:, :, None, None] * x32[:, :, :, None] * B32[:, :, None, :]
    )  # (B, H, P, N)
    y = jnp.einsum("bhpn,bhn->bhp", h_new, C32)  # (B, H, P)
    y = y + D.astype(jnp.float32)[None, :, None] * x32

    new_states = rec_states.at[state_indices].set(h_new.astype(rec_states.dtype))
    return y, new_states
