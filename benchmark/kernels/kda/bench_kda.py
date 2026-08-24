"""Benchmark the KDA (Kimi Delta Attention) chunked kernel over a (B, S) sweep.

Compares:
  - ``naive_recurrent_kda``: sequential O(T) token-by-token reference (fp32 oracle)
  - ``chunk_kda``:           chunked delta-rule kernel (the production path)

Two sweep modes, both reported as latency (ms), prefill throughput (tokens/s),
speedup vs the reference, and numerical parity:

  grid     ``--batch-sizes`` x ``--seq-lens`` cartesian product. Matches
           ``benchmark/kernels/gdn/bench_gdn.py`` so KDA and GDN numbers line up.
  budget   ``--bs-seqlen-pairs``: coupled (num_seqs, seq_len) pairs at a roughly
           constant token budget. KDA's recurrent state is per-sequence
           ([N, H, K, V]), so holding the token count fixed while re-slicing it
           across B and S separates state cost from token cost -- something the
           cartesian grid cannot show.

The reference is a token-by-token scan, so it is skipped above
``--ref-max-tokens`` (default 4096); those rows report kernel latency only.

Usage:
  python benchmark/kernels/kda/bench_kda.py
  python benchmark/kernels/kda/bench_kda.py --seq-lens 512,1024,2048,4096 --batch-sizes 1,2,4
  python benchmark/kernels/kda/bench_kda.py --mode budget
  python benchmark/kernels/kda/bench_kda.py --chunk-size 32,64,128
  python benchmark/kernels/kda/bench_kda.py --profile --profile-dir /tmp/kda_profile
  SGLANG_JAX_IS_IN_CI=true python benchmark/kernels/kda/bench_kda.py   # single-point smoke

Like the sibling benchmarks (flash_attention, mla), this imports the local
``utils.py`` by bare name, so run it by script path rather than with ``-m``.
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from utils import activated_gate, create_kda_uniform_data

from sgl_jax.srt.kernels.kda import chunk_kda, naive_recurrent_kda

# --------------------------------------------------------------------------
# Sweep grids.
#
# Defaults follow the tuned-table convention used across this repo (see
# benchmark/kernels/flash_attention/get_block_spec_config_v3.py and
# python/sgl_jax/srt/utils/common_utils.py's PRECOMPILE_* ladders): powers of
# two, num_seqs capped at PRECOMPILE_DEFAULT_BS_PADDINGS' 256 and seq_len at
# PRECOMPILE_DEFAULT_TOKEN_PADDINGS' 8192.
#
# The CI variants collapse to a single point -- the benchmark is a smoke test
# there, not a perf gate. Same split as sglang's
# python/sglang/kernels/jit/benchmark/utils.py::get_benchmark_range.
# --------------------------------------------------------------------------
_FULL_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_FULL_SEQ_LENS = (512, 1024, 2048, 4096, 8192)
_FULL_CHUNK_SIZES = (64,)

_CI_BATCH_SIZES = (2,)
_CI_SEQ_LENS = (512,)
_CI_CHUNK_SIZES = (64,)

# Constant-token-budget pairs (~16K tokens), after
# sglang's benchmark/kernels/attention/bench_flash_attention_fp8.py.
_FULL_BS_SEQLEN_PAIRS = ((32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384))
_CI_BS_SEQLEN_PAIRS = ((2, 512),)

_DEFAULT_NUM_HEADS = 8
_DEFAULT_HEAD_DIM = 128
_DEFAULT_LOWER_BOUND = -5.0


def is_in_ci() -> bool:
    """Match the env-var convention used by the other benchmarks in this tree."""
    return os.environ.get("SGLANG_JAX_IS_IN_CI", "").lower() in ("1", "true", "yes")


def get_benchmark_range(full_range: tuple, ci_range: tuple) -> list:
    """Pick the full or the collapsed CI range. Mirrors sglang's helper of the same name."""
    return list(ci_range if is_in_ci() else full_range)


# Tensor argument order for the jitted callables below. Passing the tensors as
# real arguments (rather than closing over them) keeps them out of the HLO as
# constants -- at 8192 tokens the baked-in form makes compilation dominate the
# measurement. Same pattern as benchmark/kernels/gdn/bench_gdn.py.
_ARG_ORDER = ("q", "k", "v", "raw_g", "beta", "initial_state", "cu_seqlens", "A_log", "dt_bias")


def pack_args(data: dict) -> tuple:
    return tuple(data[name] for name in _ARG_ORDER)


def benchmark_kernel(
    fn,
    args: tuple,
    kwargs: dict | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> tuple[float, Any]:
    """JIT-compile, warm up, then time ``fn(*args, **kwargs)`` with block_until_ready."""
    jitted = jax.jit(functools.partial(fn, **(kwargs or {})))

    for _ in range(warmup):
        jax.block_until_ready(jitted(*args))

    t0 = time.perf_counter()
    for _ in range(iters):
        out = jitted(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters, out


def run_kda_kernel(
    q,
    k,
    v,
    raw_g,
    beta,
    initial_state,
    cu_seqlens,
    A_log,
    dt_bias,
    *,
    scale: float,
    chunk_size: int,
    lower_bound: float | None,
):
    """One ``chunk_kda`` call. Returns (output, final_state)."""
    out = chunk_kda(
        q,
        k,
        v,
        raw_g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
    )
    return out[0], out[1]


def run_naive_reference(
    q,
    k,
    v,
    raw_g,
    beta,
    initial_state,
    cu_seqlens,
    A_log,
    dt_bias,
    *,
    seq_lens: tuple[int, ...],
    scale: float,
    lower_bound: float | None,
):
    """Call ``naive_recurrent_kda`` once per logical request and re-pack.

    Mirrors ``_full_naive_reference`` in python/sgl_jax/test/kernels/kda_test.py:
    the reference is dense per-sequence, so varlen has to be unpacked by hand.
    ``seq_lens`` is a static tuple -- the slice boundaries are Python ints, so
    they must be known at trace time rather than read back from ``cu_seqlens``.
    """
    g = activated_gate(raw_g, A_log, dt_bias, lower_bound)
    boundaries = np.cumsum((0,) + tuple(seq_lens))
    outputs, final_states = [], []
    for request, length in enumerate(seq_lens):
        begin, end = int(boundaries[request]), int(boundaries[request + 1])
        assert end - begin == length
        o_i, s_i = naive_recurrent_kda(
            q[:, begin:end],
            k[:, begin:end],
            v[:, begin:end],
            g[:, begin:end],
            beta[:, begin:end],
            scale=scale,
            initial_state=initial_state[request : request + 1],
            output_final_state=True,
        )
        outputs.append(o_i)
        final_states.append(s_i)
    return jnp.concatenate(outputs, axis=1), jnp.concatenate(final_states, axis=0)


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))


def _bench_one_point(
    num_seqs: int,
    seq_len: int,
    *,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    lower_bound: float | None,
    warmup: int,
    iters: int,
    ref_max_tokens: int,
    seed: int,
) -> str:
    seq_lens = [seq_len] * num_seqs
    data, total_tokens = create_kda_uniform_data(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        seed=seed,
        lower_bound=lower_bound,
    )
    scale = head_dim**-0.5

    args = pack_args(data)
    kernel_lat_s, (kda_out, kda_state) = benchmark_kernel(
        run_kda_kernel,
        args,
        dict(scale=scale, chunk_size=chunk_size, lower_bound=lower_bound),
        warmup=warmup,
        iters=iters,
    )
    kernel_tps = total_tokens / kernel_lat_s

    # The reference is a per-token lax.scan; at large T it dominates wall clock
    # by orders of magnitude, so skip it rather than silently making the sweep
    # unrunnable. Skipped rows are marked, never reported as if verified.
    if total_tokens > ref_max_tokens:
        return (
            f"{num_seqs:4d} | {seq_len:8d} | {total_tokens:9d} | {chunk_size:3d} | "
            f"{kernel_lat_s * 1e3:12.3f} | {kernel_tps:13.1f} | "
            f"{'skipped':>12s} | {'-':>9s} | {'-':>19s}"
        )

    ref_lat_s, (ref_out, ref_state) = benchmark_kernel(
        run_naive_reference,
        args,
        dict(seq_lens=tuple(seq_lens), scale=scale, lower_bound=lower_bound),
        warmup=1,
        iters=max(1, iters // 5),
    )
    speedup = ref_lat_s / kernel_lat_s if kernel_lat_s > 0 else float("inf")
    diff_str = f"{_max_abs_diff(kda_out, ref_out):.1e} / {_max_abs_diff(kda_state, ref_state):.1e}"
    return (
        f"{num_seqs:4d} | {seq_len:8d} | {total_tokens:9d} | {chunk_size:3d} | "
        f"{kernel_lat_s * 1e3:12.3f} | {kernel_tps:13.1f} | "
        f"{ref_lat_s * 1e3:12.2f} | {speedup:8.2f}x | {diff_str:>19s}"
    )


_HEADER = (
    f"{'N':>4s} | {'T_perseq':>8s} | {'T_total':>9s} | {'BT':>3s} | "
    f"{'Kda Lat(ms)':>12s} | {'Kda (tok/s)':>13s} | "
    f"{'Ref Lat(ms)':>12s} | {'Speedup':>9s} | {'MaxDiff (Out/St)':>19s}"
)
_RULE_WIDTH = len(_HEADER)


def run_sweep(
    points: list[tuple[int, int]],
    chunk_sizes: list[int],
    *,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    warmup: int,
    iters: int,
    ref_max_tokens: int,
    seed: int,
    title: str,
):
    print("=" * _RULE_WIDTH)
    print(
        f"KDA Kernel Benchmark [{title}] "
        f"(H={num_heads}, K=V={head_dim}, "
        f"gate={'bounded lb=' + str(lower_bound) if lower_bound is not None else 'softplus'})"
    )
    print("=" * _RULE_WIDTH)
    print(_HEADER)
    print("-" * _RULE_WIDTH)
    for chunk_size in chunk_sizes:
        for num_seqs, seq_len in points:
            print(
                _bench_one_point(
                    num_seqs,
                    seq_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    chunk_size=chunk_size,
                    lower_bound=lower_bound,
                    warmup=warmup,
                    iters=iters,
                    ref_max_tokens=ref_max_tokens,
                    seed=seed,
                ),
                flush=True,
            )
    print("=" * _RULE_WIDTH)


def record_profile(
    profile_dir: str,
    *,
    num_seqs: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    lower_bound: float | None,
    seed: int,
):
    os.makedirs(profile_dir, exist_ok=True)
    print(f"\nRecording JAX profile trace (N={num_seqs}, T={seq_len}) to {profile_dir}...")
    data, _ = create_kda_uniform_data(
        seq_lens=[seq_len] * num_seqs,
        num_heads=num_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        seed=seed,
        lower_bound=lower_bound,
    )
    scale = head_dim**-0.5
    args = pack_args(data)
    jitted = jax.jit(
        functools.partial(
            run_kda_kernel, scale=scale, chunk_size=chunk_size, lower_bound=lower_bound
        )
    )
    jax.block_until_ready(jitted(*args))
    with jax.profiler.trace(profile_dir):
        for i in range(5):
            with jax.profiler.StepTraceAnnotation("kda_chunk_step", step_num=i):
                jax.block_until_ready(jitted(*args))
    print("Profile saved. View with Perfetto (https://ui.perfetto.dev) or TensorBoard/XProf.")


def _csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _csv_pairs(s: str) -> list[tuple[int, int]]:
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        n, t = item.split(":")
        out.append((int(n), int(t)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark the KDA chunked kernel over (B, S)")
    parser.add_argument(
        "--mode",
        choices=("grid", "budget", "both"),
        default="grid",
        help="grid = batch-sizes x seq-lens; budget = constant-token (N:T) pairs",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated logical batch sizes (num_seqs); empty = default ladder",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="",
        help="Comma-separated sequence lengths per request; empty = default ladder",
    )
    parser.add_argument(
        "--bs-seqlen-pairs",
        type=str,
        default="",
        help="Comma-separated N:T pairs for budget mode, e.g. '32:512,16:1024'",
    )
    parser.add_argument(
        "--chunk-size",
        type=str,
        default="",
        help="Comma-separated chunk sizes (BT) to sweep; empty = default",
    )
    parser.add_argument("--num-heads", type=int, default=_DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dim", type=int, default=_DEFAULT_HEAD_DIM, help="K and V head dim")
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=_DEFAULT_LOWER_BOUND,
        help="Bounded-gate lower bound; pass nan for the unbounded softplus gate",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--ref-max-tokens",
        type=int,
        default=4096,
        help="Skip the O(T) naive reference above this total token count",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Record a JAX trace profile")
    parser.add_argument("--profile-dir", type=str, default="/tmp/kda_profile")
    args = parser.parse_args()

    lower_bound = None if np.isnan(args.lower_bound) else args.lower_bound
    chunk_sizes = (
        _csv_ints(args.chunk_size)
        if args.chunk_size
        else get_benchmark_range(_FULL_CHUNK_SIZES, _CI_CHUNK_SIZES)
    )
    common = dict(
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        lower_bound=lower_bound,
        warmup=args.warmup,
        iters=args.iters,
        ref_max_tokens=args.ref_max_tokens,
        seed=args.seed,
    )

    if args.mode in ("grid", "both"):
        batch_sizes = (
            _csv_ints(args.batch_sizes)
            if args.batch_sizes
            else get_benchmark_range(_FULL_BATCH_SIZES, _CI_BATCH_SIZES)
        )
        seq_lens = (
            _csv_ints(args.seq_lens)
            if args.seq_lens
            else get_benchmark_range(_FULL_SEQ_LENS, _CI_SEQ_LENS)
        )
        points = [(n, t) for n in batch_sizes for t in seq_lens]
        run_sweep(points, chunk_sizes, title="grid: B x S", **common)

    if args.mode in ("budget", "both"):
        pairs = (
            _csv_pairs(args.bs_seqlen_pairs)
            if args.bs_seqlen_pairs
            else get_benchmark_range(_FULL_BS_SEQLEN_PAIRS, _CI_BS_SEQLEN_PAIRS)
        )
        run_sweep(pairs, chunk_sizes, title="budget: constant B*S", **common)

    if args.profile:
        record_profile(
            args.profile_dir,
            num_seqs=2,
            seq_len=2048,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            chunk_size=chunk_sizes[0],
            lower_bound=lower_bound,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
