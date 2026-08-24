"""Auto-tuner for the KDA chunk-size table.

Sweeps candidate ``chunk_size`` (BT) values per shape, plus any boolean
fast-path flags the installed ``chunk_kda`` happens to expose, and emits
Python-literal entries keyed the same way the RPA v3 table is keyed.

Structure mirrors ``benchmark/kernels/flash_attention/get_block_spec_config_v3.py``:
outer grid over shapes, inner enumeration of tunables, compare against the
production heuristic, emit only entries that beat it by >= --write-threshold-pct.

One deliberate difference from the RPA tuner. RPA's block sizes are a pure
tiling choice: they cannot change the result, so that tuner only times. KDA's
``chunk_size`` changes the math -- it sets the intra-chunk Neumann solve width
and the inter-chunk state-propagation granularity, so a larger BT is both
faster and less accurate. Timing alone would happily crown a BT whose output
has drifted. Every candidate is therefore checked against the fp32 recurrent
oracle at a small probe shape first, and anything past --max-abs-err is dropped
before it can win.

Usage:
    python benchmark/kernels/kda/get_block_spec_config_kda.py
    python benchmark/kernels/kda/get_block_spec_config_kda.py --num-heads 8 --head-dims 128
    # Narrow grid for fast validation:
    python benchmark/kernels/kda/get_block_spec_config_kda.py \
        --num-heads 8 --head-dims 128 --num-seqs 2,4 --seq-lens 1024,2048

Multi-worker: --shard splits the outer grid, rank from $FALCON_RANK.
    python benchmark/kernels/kda/get_block_spec_config_kda.py --shard auto,8

Like the sibling tuners this imports the local ``utils.py`` by bare name, so run
it by script path rather than with ``-m``.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import itertools
import os
from math import inf

import jax
import jax.numpy as jnp
import numpy as np
from utils import activated_gate, create_kda_uniform_data

from sgl_jax.srt.kernels.kda import chunk_kda, naive_recurrent_kda
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.common_utils import next_power_of_2
from sgl_jax.srt.utils.jax_utils import get_device_name

# --------------------------------------------------------------------------
# Default grids.
#
# Shape axes follow the ladders already used in this repo: num_seqs capped at
# PRECOMPILE_DEFAULT_BS_PADDINGS' 256 and seq_len spanning
# PRECOMPILE_DEFAULT_TOKEN_PADDINGS (see srt/utils/common_utils.py). Head counts
# match the (q, kv) combos the RPA v3 tuner sweeps, restricted to the shapes a
# KDA layer actually runs.
# --------------------------------------------------------------------------
_DEFAULT_NUM_HEADS = (8, 16, 32)
_DEFAULT_HEAD_DIMS = (128,)
_DEFAULT_NUM_SEQS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_DEFAULT_SEQ_LENS = (512, 1024, 2048, 4096, 8192)

# Tunable: chunk size. Powers of two only -- _align_seqs pads every sequence up
# to a BT multiple, so a non-power-of-two BT wastes padding without buying
# anything. 64 is the production default and therefore the heuristic baseline.
_DEFAULT_CHUNK_SIZES = (32, 64, 128, 256)
_HEURISTIC_CHUNK_SIZE = 64

_DEFAULT_LOWER_BOUND = -5.0

# Probe shape for the correctness gate: small enough that the O(T) oracle is
# cheap, large enough to span several chunks at the biggest BT in the grid.
_PROBE_NUM_SEQS = 2
_PROBE_SEQ_LEN = 1024


def _optional_bool_flags() -> list[str]:
    """Boolean fast-path kwargs the installed ``chunk_kda`` exposes.

    Upstream ``chunk_kda_fwd`` takes only chunk_size; local optimization
    branches add flags like fuse / unified_layout / flat_grid / head_block.
    Discovering them keeps one tuner working against both without pinning a
    branch-specific signature.
    """
    known = ("fuse", "unified_layout", "flat_grid", "head_block")
    try:
        params = inspect.signature(chunk_kda).parameters
    except (TypeError, ValueError):
        return []
    return [name for name in known if name in params and isinstance(params[name].default, bool)]


def _enumerate_candidates(
    chunk_sizes: list[int],
    seq_len: int,
    num_heads: int,
    flags: list[str],
) -> list[dict]:
    """Cartesian product of chunk_size x boolean flags, with shape prunes."""
    out = []
    for bt in chunk_sizes:
        if bt & (bt - 1):
            continue  # power of two only
        if bt > seq_len:
            # BT above the sequence length degenerates to a single padded chunk:
            # the measurement would be all padding, not the shape asked for.
            continue
        for combo in itertools.product((True, False), repeat=len(flags)):
            cand = {"chunk_size": bt}
            cand.update(dict(zip(flags, combo)))
            # head_block needs a TPU-shaped head tile; asking for it otherwise
            # silently falls back, producing a duplicate measurement.
            if cand.get("head_block") and num_heads % 8 != 0:
                continue
            out.append(cand)
    # Deduplicate while preserving order (flag prunes can collapse combos).
    seen, uniq = set(), []
    for cand in out:
        key = tuple(sorted(cand.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(cand)
    return uniq


def _cand_str(cand: dict) -> str:
    return "-".join(f"{k}_{v}" for k, v in sorted(cand.items()))


# Tensor argument order for the jitted kernel. Passing tensors as real
# arguments instead of closing over them keeps them out of the HLO as
# constants; at 8192 tokens the baked-in form makes compilation dominate the
# measurement. Mirrors get_block_spec_config_v3.py's jit-then-partial shape.
_ARG_ORDER = ("q", "k", "v", "raw_g", "beta", "initial_state", "cu_seqlens", "A_log", "dt_bias")


def _pack_args(data: dict) -> tuple:
    return tuple(data[name] for name in _ARG_ORDER)


def _kda_kernel(
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
    lower_bound: float | None,
    **cand,
):
    return chunk_kda(
        q,
        k,
        v,
        raw_g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
        **cand,
    )


def _jitted_kernel(scale: float, lower_bound: float | None, cand: dict):
    """jit the kernel with tunables bound as static Python values."""
    return jax.jit(functools.partial(_kda_kernel, scale=scale, lower_bound=lower_bound, **cand))


def _oracle(data: dict, seq_lens: list[int], scale: float, lower_bound: float | None):
    """fp32 token-by-token reference, unpacked per request."""
    g = activated_gate(data["raw_g"], data["A_log"], data["dt_bias"], lower_bound)
    boundaries = np.cumsum([0] + list(seq_lens))
    outs, states = [], []
    for request, length in enumerate(seq_lens):
        begin, end = int(boundaries[request]), int(boundaries[request + 1])
        assert end - begin == length
        o_i, s_i = naive_recurrent_kda(
            data["q"][:, begin:end],
            data["k"][:, begin:end],
            data["v"][:, begin:end],
            g[:, begin:end],
            data["beta"][:, begin:end],
            scale=scale,
            initial_state=data["initial_state"][request : request + 1],
            output_final_state=True,
        )
        outs.append(o_i)
        states.append(s_i)
    return jnp.concatenate(outs, axis=1), jnp.concatenate(states, axis=0)


def _max_abs_err(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))


@functools.lru_cache(maxsize=None)
def _probe_reference(num_heads: int, head_dim: int, lower_bound: float | None, seed: int):
    """Oracle output at the probe shape. Cached: it is BT-independent."""
    seq_lens = [_PROBE_SEQ_LEN] * _PROBE_NUM_SEQS
    data, _ = create_kda_uniform_data(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        seed=seed,
        lower_bound=lower_bound,
    )
    scale = head_dim**-0.5
    ref = jax.block_until_ready(_oracle(data, seq_lens, scale, lower_bound))
    return data, seq_lens, scale, ref


def check_accuracy(
    cand: dict,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    seed: int,
) -> tuple[float, float]:
    """Max abs error of (output, final_state) vs the oracle at the probe shape."""
    data, _, scale, (ref_out, ref_state) = _probe_reference(num_heads, head_dim, lower_bound, seed)
    fn = _jitted_kernel(scale, lower_bound, cand)
    out = jax.block_until_ready(fn(*_pack_args(data)))
    return _max_abs_err(out[0], ref_out), _max_abs_err(out[1], ref_state)


def benchmark_one(
    num_seqs: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    cand: dict,
    tries: int,
    seed: int,
) -> float:
    """Mean kernel time in MILLISECONDS for one (shape, candidate) pair.

    ``multiple_iteration_timeit_from_trace`` returns milliseconds
    (device_duration_ps / 1e9), and every time in this file stays in that unit
    so there is no conversion to get wrong.
    """
    data, _ = create_kda_uniform_data(
        seq_lens=[seq_len] * num_seqs,
        num_heads=num_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        seed=seed,
        lower_bound=lower_bound,
    )
    scale = head_dim**-0.5
    bound = functools.partial(_jitted_kernel(scale, lower_bound, cand), *_pack_args(data))

    # Warmup (compile).
    jax.block_until_ready(bound())

    scope = f"KDA-n_{num_seqs}-t_{seq_len}-h_{num_heads}-{_cand_str(cand)}"
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: bound(),
        data_generator=lambda: (),
        task=scope,
        tries=tries,
    )
    return float(np.mean(times)) if times else float("nan")


def sweep(
    num_seqs: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    chunk_sizes: list[int],
    flags: list[str],
    tries: int,
    max_abs_err: float,
    seed: int,
):
    """Returns (best_cand, best_ms, heuristic_cand, heuristic_ms)."""
    candidates = _enumerate_candidates(chunk_sizes, seq_len, num_heads, flags)
    heuristic = {"chunk_size": _HEURISTIC_CHUNK_SIZE}
    heuristic.update({f: inspect.signature(chunk_kda).parameters[f].default for f in flags})
    if heuristic not in candidates:
        candidates = [heuristic] + candidates

    best_time, best, heuristic_time = inf, None, inf
    for cand in candidates:
        # Accuracy gate first: chunk_size changes the math, so a fast but drifted
        # candidate must never reach the timing comparison.
        try:
            out_err, state_err = check_accuracy(cand, num_heads, head_dim, lower_bound, seed)
        except Exception as e:  # noqa: BLE001
            if cand == heuristic:
                print(
                    f"# heuristic-candidate ACCURACY FAILURE n={num_seqs} t={seq_len} "
                    f"h={num_heads} hd={head_dim}: {type(e).__name__}: {e}"
                )
            continue
        if max(out_err, state_err) > max_abs_err:
            print(
                f"# [acc-reject] n={num_seqs} t={seq_len} h={num_heads} {_cand_str(cand)}: "
                f"out={out_err:.2e} state={state_err:.2e} > {max_abs_err:.2e}"
            )
            continue

        try:
            t = benchmark_one(
                num_seqs, seq_len, num_heads, head_dim, lower_bound, cand, tries, seed
            )
        except Exception as e:  # noqa: BLE001
            # The heuristic candidate is the production baseline; if even it
            # raises, the workload itself is broken and the sweep is meaningless.
            if cand == heuristic:
                print(
                    f"# heuristic-candidate FAILURE n={num_seqs} t={seq_len} h={num_heads} "
                    f"hd={head_dim}: {type(e).__name__}: {e}"
                )
            continue
        if cand == heuristic:
            heuristic_time = t
        if t < best_time:
            best_time, best = t, cand
    return best, best_time, heuristic, heuristic_time


def _csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_shard(s: str) -> tuple[int, int]:
    """Parse --shard "RANK,TOTAL" or "auto,TOTAL" (rank from FALCON env)."""
    if not s:
        return (0, 1)
    a, b = s.split(",")
    total = int(b)
    if a == "auto":
        rank = int(os.environ.get("FALCON_RANK", os.environ.get("FALCON_JAX_PROCESS_ID", "0")))
    else:
        rank = int(a)
    if not (0 <= rank < total):
        raise SystemExit(f"--shard rank={rank} out of [0,{total})")
    return rank, total


def _table_key(
    num_heads: int,
    head_dim: int,
    num_seqs: int,
    seq_len: int,
    lower_bound: float | None,
) -> tuple:
    """Normalized lookup key, following tuned_block_sizes_v3's convention.

    Stage is always "p": ``chunk_kda`` is the chunked prefill path -- linear
    attention decodes through a separate recurrent step, not through this kernel.
    """
    return (
        "p",
        lower_bound,
        "bfloat16",
        "bfloat16",
        next_power_of_2(num_heads),
        (head_dim + 127) // 128 * 128,
        next_power_of_2(num_seqs),
        next_power_of_2(seq_len),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", default="", help="comma list; empty = full default grid")
    parser.add_argument("--head-dims", default="", help="comma list; empty = full default grid")
    parser.add_argument("--num-seqs", default="", help="comma list; empty = full default grid")
    parser.add_argument("--seq-lens", default="", help="comma list; empty = full default grid")
    parser.add_argument("--chunk-sizes", default="", help="comma list; empty = full default grid")
    parser.add_argument("--tries", type=int, default=1)
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=_DEFAULT_LOWER_BOUND,
        help="bounded-gate lower bound; pass nan for the unbounded softplus gate",
    )
    parser.add_argument(
        "--max-abs-err",
        type=float,
        default=5e-3,
        help=(
            "reject a candidate whose output or final state drifts more than this "
            "from the fp32 oracle at the probe shape"
        ),
    )
    parser.add_argument(
        "--write-threshold-pct",
        type=float,
        default=10.0,
        help="only emit a table entry if tuned is faster than heuristic by >= this %%",
    )
    parser.add_argument(
        "--shard",
        default="",
        help="split the outer grid across N workers: 'RANK,TOTAL' or 'auto,TOTAL'",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lower_bound = None if np.isnan(args.lower_bound) else args.lower_bound
    num_heads_list = _csv_ints(args.num_heads) if args.num_heads else list(_DEFAULT_NUM_HEADS)
    head_dims = _csv_ints(args.head_dims) if args.head_dims else list(_DEFAULT_HEAD_DIMS)
    num_seqs_list = _csv_ints(args.num_seqs) if args.num_seqs else list(_DEFAULT_NUM_SEQS)
    seq_lens = _csv_ints(args.seq_lens) if args.seq_lens else list(_DEFAULT_SEQ_LENS)
    chunk_sizes = _csv_ints(args.chunk_sizes) if args.chunk_sizes else list(_DEFAULT_CHUNK_SIZES)
    flags = _optional_bool_flags()

    device = get_device_name()
    shard_rank, shard_total = _parse_shard(args.shard)

    outer = list(itertools.product(num_heads_list, head_dims, num_seqs_list, seq_lens))
    my_work = outer[shard_rank::shard_total]
    print(f"# device={device!r} tunable_flags={flags or '(chunk_size only)'}")
    print(
        f"# outer-grid total={len(outer)} mine={len(my_work)} "
        f"(every {shard_total}-th starting at {shard_rank})"
    )

    rows = []
    for num_heads, head_dim, num_seqs, seq_len in my_work:
        try:
            best, best_t, heur, heur_t = sweep(
                num_seqs,
                seq_len,
                num_heads,
                head_dim,
                lower_bound,
                chunk_sizes,
                flags,
                args.tries,
                args.max_abs_err,
                args.seed,
            )
        except Exception as e:  # noqa: BLE001
            print(f"# SKIP h={num_heads} hd={head_dim} n={num_seqs} t={seq_len}: {e}")
            continue
        if best is None or heur_t == inf:
            # Don't silently drop -- heur_t==inf means every candidate including
            # the heuristic failed the accuracy gate or raised in benchmark.
            print(
                f"# DROP h={num_heads} hd={head_dim} n={num_seqs} t={seq_len}: "
                f"best={best} heur_t={heur_t} -- all candidates failed"
            )
            continue
        delta_pct = (heur_t - best_t) / heur_t * 100.0
        key = _table_key(num_heads, head_dim, num_seqs, seq_len, lower_bound)
        rows.append((key, best, best_t, heur, heur_t, delta_pct))
        win = "WIN " if delta_pct >= args.write_threshold_pct else "skip"
        print(
            f"# [{win}] {key}: "
            f"heur={_cand_str(heur)} {heur_t:.4f}ms "
            f"best={_cand_str(best)} {best_t:.4f}ms "
            f"delta={delta_pct:+.1f}%"
        )

    print()
    print(
        f"# --- Paste into TUNED_CHUNK_SIZES_KDA[{device!r}] "
        f"(>={args.write_threshold_pct}% win only) ---"
    )
    for key, best, _, _, _, delta_pct in rows:
        if delta_pct >= args.write_threshold_pct:
            print(f"    {key}: {best},")
    print()
    print("# --- All measured (for audit) ---")
    for key, best, best_t, heur, heur_t, delta_pct in rows:
        print(
            f"# {key}: best={_cand_str(best)} ({best_t:.4f}ms) "
            f"heur={_cand_str(heur)} ({heur_t:.4f}ms) delta={delta_pct:+.1f}%"
        )


if __name__ == "__main__":
    main()
