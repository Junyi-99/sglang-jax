"""Optimization variants for the KDA ablation.

The three points of the ablation. All of them run the *same* math -- identical
gate (``use_gate_in_kernel=True`` with the same ``lower_bound``), identical
delta rule -- and differ only in which optimization is switched on. That is
what makes their latencies comparable and lets one fp32 oracle validate all
three.

  baseline    the stock sglang-jax kernel: ``fuse=False`` selects the branch
              whose own comment reads "Original upstream four-stage pipeline
              used as the ablation baseline" (kda.py), with the bounded-gate
              fast path off.
  safe_gate   stage 1. ``safe_gate=True`` turns Aqk/L into BT/16 per-sub-chunk
              GEMMs (the strip-GEMM path), still on the upstream structure.
  structural  stage 2. Adds the restructured pipeline: fused h+o, unified
              layout, flat chunk grid, head blocking.

``safe_gate`` and ``lower_bound`` are independent knobs in ``chunk_kda``: the
gate activation takes ``lower_bound`` regardless of ``safe_gate``. So the
baseline can be given the same bounded gate as the other two while keeping the
strip-GEMM path off -- an apples-to-apples comparison rather than one that also
swaps the gate function.
"""

from __future__ import annotations

import inspect

__all__ = ["VARIANTS", "resolve_variants", "variant_kwargs"]

VARIANTS: dict[str, dict[str, bool]] = {
    "baseline": {"safe_gate": False, "fuse": False},
    "safe_gate": {"safe_gate": True, "fuse": False},
    "structural": {
        "safe_gate": True,
        "fuse": True,
        "unified_layout": True,
        "flat_grid": True,
        "head_block": True,
    },
}

BASELINE = "baseline"


def _supported(fn) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def variant_kwargs(name: str, fn) -> dict[str, bool]:
    """Variant kwargs restricted to what this build of ``chunk_kda`` accepts.

    Stock upstream exposes only ``safe_gate``; the structural flags live on the
    optimization branch. Filtering instead of pinning a signature keeps one
    benchmark usable against both.
    """
    supported = _supported(fn)
    return {k: v for k, v in VARIANTS[name].items() if k in supported}


def resolve_variants(names: list[str], fn) -> tuple[list[str], list[str]]:
    """Return (runnable variants, notes about ones that collapsed).

    Two requested variants can reduce to the same call when the build lacks the
    flags that separate them. Running both would report one configuration twice
    under two names, so the duplicate is dropped and named in the notes -- never
    silently folded in.
    """
    seen: dict[tuple, str] = {}
    kept: list[str] = []
    notes: list[str] = []
    missing = {k for n in names for k in VARIANTS[n]} - _supported(fn)
    if missing:
        notes.append(
            "chunk_kda does not accept "
            + ", ".join(sorted(missing))
            + " -- this build lacks those optimizations"
        )
    for name in names:
        key = tuple(sorted(variant_kwargs(name, fn).items()))
        if key in seen:
            notes.append(f"{name!r} is identical to {seen[key]!r} on this build; skipping it")
            continue
        seen[key] = name
        kept.append(name)
    return kept, notes
