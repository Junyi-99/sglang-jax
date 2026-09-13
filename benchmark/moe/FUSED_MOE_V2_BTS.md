# `bts=64` on fused_moe v2, TPU v6e

`FusedMoEBlockConfig.bts` defaults to `None`, which makes it equal `bt` (32).
Setting it to 64 is a one-value config change with no kernel edit.

Measured on TPU v6e-4, `jax 0.10.1`, bf16 weights, 2048 tokens, `ep=4`,
5 profiled iterations. Times are exclusive self-time per iteration, summed from
XProf named scopes with nested windows subtracted.

## Result

| model | `bts` | total self-time | weight wait | wait share | speedup |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 32 (default) | 5051 µs | 1529 µs | 30.3% | — |
| Qwen3-30B-A3B | **64** | **3684 µs** | 444 µs | 12.0% | **1.37×** |
| GLM-4.5-Air | 32 (default) | 15502 µs | 7209 µs | 46.5% | — |
| GLM-4.5-Air | **64** | **4961 µs** | 483 µs | 9.7% | **3.12×** |

Shapes: Qwen3-30B-A3B `H=2048 F=768 E=128 top_k=8`, chosen `bf=384`;
GLM-4.5-Air `H=4096 F=1408 E=128 top_k=8`, chosen `bf=128`.

## Why

`bts` is the token sub-tile the kernel walks per expert. Every bts tile
re-fetches that expert's `w1`/`w3`/`w2` slice — the kernel says so at
`kernel.py:1169`:

```
# Weights re-prefetch per bts tile (redundant when num_bts_tiles=1, ...
```

With ~128 tokens per local expert, `bts=32` gives `num_bts_tiles = 4`, so each
weight slice is fetched four times and serves only 32 tokens per fetch. At
`bts=64` it is fetched twice.

That matters because the weight DMA per step cannot be hidden behind the compute
in the same step:

| | weight bytes / bf step | DMA at 1.6 TB/s | compute available |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 4.50 MiB | 2.95 µs | 1.08 µs |
| GLM-4.5-Air | 3.00 MiB | 1.97 µs | 1.13 µs |

Deeper prefetch cannot close a 1.7–2.7× gap; fetching less weight data can.
Sustained bandwidth at `bts=32` is only 283–286 GB/s, about 18% of peak, so the
stall is redundant traffic rather than a bandwidth ceiling.

## Scope shift

Qwen3-30B-A3B, top scopes by self-time:

| scope | `bts=32` | `bts=64` |
|---|---:|---:|
| `ffn1_gate_up` | 839 µs (16.6%) | 1170 µs (31.8%) |
| `ffn2_down` | 453 µs (9.0%) | 624 µs (16.9%) |
| `w3_load_wait` | 800 µs (15.8%) | 411 µs (11.2%) |
| `w1_load_wait` | 512 µs (10.1%) | below cut |
| `expert_x_load` | 1035 µs (20.5%) | 401 µs (10.9%) |

GLM-4.5-Air:

| scope | `bts=32` | `bts=64` |
|---|---:|---:|
| `ffn1_gate_up` | 4335 µs (28.0%) | 2650 µs (53.4%) |
| `ffn2_down` | 1751 µs (11.3%) | 1061 µs (21.4%) |
| `w3_load_wait` | 4362 µs (28.1%) | 231 µs (4.7%) |
| `w1_load_wait` | 2387 µs (15.4%) | 248 µs (5.0%) |

After the change both kernels are dominated by the two matmuls, which is where
an MoE kernel should be spending its time.

## `bts=64` is the optimum, not the endpoint

Larger `bts` removes more of the stall but costs more compute, because each tile
is padded to `bts` and `num_btc_per_bts = bts/btc` grows:

| `bt`/`bts` | tiles | Qwen3 total | Qwen3 wait | GLM total | GLM wait |
|---|---:|---:|---:|---:|---:|
| 32 / 32 | 4 | 5051 µs | 30.3% | 15502 µs | 46.5% |
| 32 / **64** | 2 | **3684 µs** | 12.0% | **4961 µs** | 9.7% |
| 32 / 128 | 1 | 4204 µs | 0.9% | 5198 µs | 4.3% |
| 64 / 256 | 1 | 4717 µs | 0.5% | 5409 µs | 2.3% |
| 128 / 512 | 1 | 4696 µs | 0.3% | 5409 µs | 1.1% |

The stall falls monotonically to 0.3%, but total time turns around at 64. The
best configuration is not the one with the smallest bubble.

## VMEM is not the constraint

Estimated by the repo's own `_estimate_vmem_bytes_v2`, against a 128 MB budget:

| model | `bf=384/128`, `bts=64` | largest legal `bf`, `bts=128` |
|---|---:|---:|
| Qwen3-30B-A3B | 11.7 MB | 22.2 MB (`bf=768`) |
| GLM-4.5-Air | 10.9 MB | 74.2 MB (`bf=1408`) |

Current use is about 9% of budget. `bts` past 64 is limited by padding waste,
not by VMEM.

## Why the default is wrong here

`TUNED_BLOCK_CONFIGS` has **no `TPU v6e` section** — 48 entries, all `TPU v7`,
all `intermediate_size=2048`, all fp8, all `ep ∈ {8,16,32,128}`. Every v6e shape
falls back to `DEFAULT_V2_BLOCK_CONFIG = (32, 512, 32, 256, None)`, whose
`bf=512` does not even divide `F=768` or `F=1408`; those shapes raise
`ValueError` before running. The v7 entries cannot be reused: none of their `bf`
values (256/512/1024) divides 1408, and only 256 divides 768.

## Limits of this measurement

- One token count (2048) and one `ep` (4). A sweep over token buckets
  {128, 512, 2048, 8192} × `ep` {1, 2, 4} is running; the optimum may move.
- `bf` was picked as the largest 128-aligned divisor of `F`, not tuned. A `bf`
  sweep has not run.
- bf16 weights. The v7 table is entirely fp8, which has different tile optima.
- Self-time comes from scope-window nesting computed in the harness, not from
  the runtime. The pending sweep records independent wall-clock as a check.
- Per-execution-unit instruction lanes were not emitted by this runtime, so none
  of this is instruction-level attribution.
