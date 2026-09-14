# `bts` on fused_moe v2, TPU v6e — retracted, and why

**An earlier version of this document claimed `bts=64` gave a 1.37×/3.12×
speedup. That was a measurement artefact. It does not.**

The claim came from XProf named-scope self-time. That metric turned out to be
insensitive to how much work the kernel actually does, so it could not support
a performance claim of any size. Wall clock, measured on a jitted call, shows
no benefit.

## What the correct measurement says

GLM-4.5-Air `H=4096 F=1408 E=128 top_k=8`, `bf=128`, `ep=4`, TPU v6e-8,
`jax 0.10.1`, bf16. Median of 10 jitted calls after 4 warm-ups.

| tokens | `bts=None` (=32) | `bts=64` | |
|---:|---:|---:|---|
| 1024 | 10455 µs | 10578 µs | +1.2% |
| 2048 | 18793 µs | 19291 µs | +2.6% |

Fitting `wall = a + b·tokens` over T ∈ {1024, 2048, 4096} (residuals ±23 µs on
10–35 ms, so the fit resolves ~0.2%):

```
bts=None    2064 µs  +  8.18 µs/token
bts=64      1866 µs  +  8.51 µs/token
```

`bts=64` trades ~200 µs of fixed cost for a 4% worse per-token slope. At
T=1024 a 3.24× speedup would have predicted 4693 µs against a measured 10578 —
not a subtle miss.

## How the self-time metric failed

`scope_mix` sums exclusive self-time over XProf `XLA TraceMe` windows. Across a
4× change in problem size, the captured event count does not move:

| tokens | `bts` | trace regions | self-time/lane |
|---:|---:|---:|---:|
| 1024 | — | 341930 | 4006 µs |
| 2048 | — | 342174 | 3819 µs |
| 4096 | — | 342432 | 3879 µs |
| 1024 | 64 | 131422 | 1236 µs |
| 2048 | 64 | 131710 | 1237 µs |

The trace saturates. Self-time therefore measures "how much time sits in the
first N captured events", a fixed-size window, not the kernel. Both series are
flat in tokens, which is impossible for device time.

The apparent 3.12× was mostly the *capture ratio* between the two configs:
342k/131k = 2.6 regions, 4006/1236 = 3.24 µs. Different `bts` produces a
different density of scope windows, so the two runs filled the buffer at
different points and their self-times were never comparable.

**Rule this leaves behind:** before comparing two configs by any trace-derived
metric, vary the problem size and confirm the metric moves with it. That check
costs one extra run and would have caught this before anything was published.

## What still holds

These do not depend on the retracted metric.

**The re-fetch mechanism is real, as source.** `bts` is the token sub-tile the
kernel walks per expert, and each tile re-fetches that expert's `w1`/`w3`/`w2`
slice — `kernel.py:1169` says so directly:

```
# Weights re-prefetch per bts tile (redundant when num_bts_tiles=1, ...
```

With `bts=32` and ~128 tokens per local expert that is four fetches per slice.
What is now unsupported is the claim that removing them is worth anything
measurable at these shapes — evidently the re-fetches are already hidden.

**`TUNED_BLOCK_CONFIGS` has no `TPU v6e` section.** 48 entries, all `TPU v7`,
all `intermediate_size=2048`, all fp8, all `ep ∈ {8,16,32,128}`. Every v6e shape
falls back to `DEFAULT_V2_BLOCK_CONFIG = (32, 512, 32, 256, None)`, whose
`bf=512` divides neither `F=768` nor `F=1408`, so those shapes raise
`ValueError` before running. The v7 entries cannot be reused either: none of
their `bf` values (256/512/1024) divides 1408, and only 256 divides 768.

**`bf=1408` does not compile on v6e.** With `H=4096`, one weight tile is
4096×1408×2 B = 11.5 MB; `w1`/`w3`/`w2` plus double-buffering exceeds the
compiler's scoped VMEM limit, with or without `bts=64`:

```
E1001: CompileTimeScopedVmemOom: ... bf16[256,4096] ... bf_1408
```

This is the scoped limit, not the 128 MB physical budget; it would need
`--xla_tpu_scoped_vmem_limit_kib` raised to test.

**Cost of the kernel at this shape.** 8.2 µs/token at `H=4096 F=1408 K=8 ep=4`.
The per-device compute floor is 77 µs at T=1024 against 8.3 ms of variable
wall, so the kernel runs at roughly 1% of peak MXU. That gap, not `bts`, is
where the headroom is.

## Limits

- One shape family (GLM-4.5-Air), one `bf` (128), one `ep` (4) so far. A sweep
  over T ∈ {1024,2048,4096,8192} × `ep` ∈ {4,8} is running.
- Weights are replicated (`P()`) rather than sharded over the EP axis, which
  may not match how the model runs in serving. The absolute 8.2 µs/token should
  not be read as a serving number.
- bf16 only. The v7 table is entirely fp8.
- Wall clock includes host dispatch; the ~2 ms intercept is not attributed.
