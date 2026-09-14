# `bts` on fused_moe v2, TPU v6e — retracted, and why

**An earlier version of this document claimed `bts=64` gave a 1.37×/3.12×
speedup. That was a measurement artefact. It is slower at every point in the
T × `ep` grid.**

The claim came from XProf named-scope self-time. That metric turned out to be
insensitive to how much work the kernel actually does, so it could not support
a performance claim of any size. Jitted wall clock, over 8 (tokens, `ep`)
pairs, puts `bts=64` between 1.2% and 10.2% slower.

## What the correct measurement says

GLM-4.5-Air `H=4096 F=1408 E=128 top_k=8`, `bf=128`, TPU v6e-8, `jax 0.10.1`,
bf16. Median of 10 jitted calls after 4 warm-ups. The full grid, 8 pairs:

| `ep` | tokens | `bts=None` (=32) | tiles | `bts=64` | tiles | change |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1024 | 10455 µs | 2 | 10578 µs | 1 | +1.2% |
| 4 | 2048 | 18793 µs | 4 | 19291 µs | 2 | +2.6% |
| 4 | 4096 | 35575 µs | 8 | 36496 µs | 4 | +2.6% |
| 4 | 8192 | 68989 µs | 16 | 70974 µs | 8 | +2.9% |
| 8 | 1024 | 3146 µs | 2 | 3265 µs | 1 | +3.8% |
| 8 | 2048 | 4929 µs | 4 | 5432 µs | 2 | +10.2% |
| 8 | 4096 | 9624 µs | 8 | 9751 µs | 4 | +1.3% |
| 8 | 8192 | 18158 µs | 16 | 18551 µs | 8 | +2.2% |

`bts=64` is slower at every point. It halves the tile count everywhere, so at
T=8192 it removes eight of sixteen weight re-fetches per expert and is still
2.9% slower — the effect does not appear at scale, it inverts.

Marginal cost per token, from successive token counts at `bts=None`:

```
ep=4    8.14   8.19   8.16  µs/token
ep=8    1.74   2.29   2.08  µs/token
```

`ep=4` is linear in tokens to within 0.6%.

## An unexplained factor of ~1.9

Doubling the EP degree at fixed token count is worth far more than any block
config value tried here:

| tokens | `ep=4` | `ep=8` | ratio |
|---:|---:|---:|---:|
| 1024 | 10455 µs | 3146 µs | 3.32× |
| 2048 | 18793 µs | 4929 µs | 3.81× |
| 4096 | 35575 µs | 9624 µs | 3.70× |
| 8192 | 68989 µs | 18158 µs | 3.80× |

Per-device work should only halve. Tokens per device halve (T/ep) and experts
per device halve (E/ep), while **tokens per expert is invariant in ep** —
`(T/ep)·K/(E/ep) = T·K/E`, 256 at T=4096 for both. So the expected ratio is 2×
and the observed one is ~3.7×. The remaining ~1.9× is not accounted for by this
measurement, and is the larger effect by an order of magnitude — worth chasing
before any further block-config tuning.

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

With `bts=32` and 256 tokens per local expert that is eight fetches per slice
at T=4096. What the grid shows is that removing half of them costs time rather
than saving it, at every token count and both EP degrees — so the re-fetches
are already hidden, and the larger tile pays for itself in padding.

**`TUNED_BLOCK_CONFIGS` has no `TPU v6e` section.** 48 entries, all `TPU v7`,
all `intermediate_size=2048`, all fp8, all `ep ∈ {8,16,32,128}`. Every v6e shape
falls back to `DEFAULT_V2_BLOCK_CONFIG = (32, 512, 32, 256, None)`, whose
`bf=512` divides neither `F=768` nor `F=1408`, so those shapes raise
`ValueError` before running. The v7 entries cannot be reused either: none of
their `bf` values (256/512/1024) divides 1408, and only 256 divides 768.

**`bf=1408` does not compile on v6e.** With `H=4096`, one weight tile is
4096×1408×2 B = 11.5 MB; `w1`/`w3`/`w2` plus double-buffering exceeds the
compiler's scoped VMEM limit — all 16 `bf=1408` points in the grid failed this
way, with and without `bts=64`:

```
E1001: CompileTimeScopedVmemOom: ... bf16[256,4096] ... bf_1408
```

This is the scoped limit, not the 128 MB physical budget; it would need
`--xla_tpu_scoped_vmem_limit_kib` raised to test.

**Cost of the kernel at this shape.** 8.16 µs/token at `ep=4`, 2.0 at `ep=8`,
for `H=4096 F=1408 K=8`. The per-device compute floor is 77 µs at T=1024
against 8.3 ms of variable wall, so the kernel runs at roughly 1% of peak MXU.
That gap, not `bts`, is where the headroom is.

## Limits

- One shape family (GLM-4.5-Air) and one `bf` (128). The T × `ep` grid is
  complete; `bf` is not swept, because the only other legal value on this shape
  is 1408, which does not compile.
- Weights are replicated (`P()`) rather than sharded over the EP axis, which
  may not match how the model runs in serving. The absolute µs/token figures
  should not be read as serving numbers, and the unexplained ~1.9× above may
  be an artefact of this setup rather than a property of the kernel.
- bf16 only. The v7 table is entirely fp8.
- Wall clock includes host dispatch; the ~2 ms intercept is not attributed.
