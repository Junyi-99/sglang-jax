"""Scope-level profiler and block-config sweep for fused_ep_moe_v2 on TPU.

Runs the kernel at real EPMoE model shapes and reports exclusive self-time per
XProf named scope, so the weight-stall share can be separated from the matmuls.
Sweeps the block config over `bts` (token sub-tile), `bf` (intermediate tile),
token count and `ep_size`.

    PROF_SWEEP=bts|bts_wide|bf|glm_bf   which block configs to try
    PROF_SHAPES=<label>,...             restrict to some of SHAPES
    PROF_TOKENS=1024,2048,...           token counts
    PROF_EP=4,8                         ep sizes (needs that many devices)
    OUTPUT_URI=gs://...                 durable per-shard results; resumable

Findings from the first sweeps are in FUSED_MOE_V2_BTS.md.

Self-time is computed here, by building the scope containment forest per trace
lane and subtracting direct children -- summing raw window durations
double-counts, because the scopes nest. Per-execution-unit instruction lanes are
not emitted by this runtime, so nothing here is instruction-level attribution.
"""
from __future__ import annotations

import bisect
import collections
import gzip
import json
import os
import pathlib
import subprocess
import sys
import time

def load_trace(root: str) -> dict:
    d = pathlib.Path(root) / "plugins" / "profile"
    latest = max(d.iterdir(), key=os.path.getmtime)
    out: dict = {"traceEvents": []}
    for tf in sorted(latest.glob("*.trace.json.gz")):
        with gzip.open(tf, "rb") as fh:
            out["traceEvents"].extend(json.load(fh).get("traceEvents", []))
    return out


def scope_mix(trace: dict, iters: int) -> dict:
    """Exclusive self-time per scope.

    Summing window durations double-counts: the scopes nest (expert_ffn contains
    the ffn/weight-wait scopes, which contain others). Build the containment
    forest per trace lane and subtract each window's direct children, so the
    self times are additive.

    Also keeps the per-region distribution for the *_load_wait scopes: a large
    total made of many cheap waits is a scheduling artefact, one made of a few
    long waits is a real stall.
    """
    ev = trace["traceEvents"]
    tname = {(e["pid"], e["tid"]): e.get("args", {}).get("name")
             for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name" and "tid" in e}
    per_lane = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") != "X":
            continue
        if tname.get((e.get("pid"), e.get("tid"))) != "XLA TraceMe":
            continue
        n = (e.get("name") or "").split("/")[-1]
        d = e.get("dur", 0)
        if n and d >= 0:
            per_lane[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + d, n))

    incl = collections.Counter()
    excl = collections.Counter()
    cnt = collections.Counter()
    durs = collections.defaultdict(list)
    for wins in per_lane.values():
        wins.sort(key=lambda w: (w[0], -(w[1] - w[0])))
        stack: list[list] = []          # [start, end, name, child_time]
        for st, en, n in wins:
            while stack and stack[-1][1] <= st:
                done = stack.pop()
                excl[done[2]] += (done[1] - done[0]) - done[3]
                if stack:
                    stack[-1][3] += done[1] - done[0]
            incl[n] += en - st
            cnt[n] += 1
            durs[n].append(en - st)
            stack.append([st, en, n, 0.0])
        while stack:
            done = stack.pop()
            excl[done[2]] += (done[1] - done[0]) - done[3]
            if stack:
                stack[-1][3] += done[1] - done[0]

    n_lanes = max(len(per_lane), 1)
    out = {}
    for n in incl:
        v = sorted(durs[n])
        out[n] = {
            "regions": cnt[n],
            "inclusive_us_per_iter": incl[n] / iters,
            "self_us_per_iter": excl[n] / iters,
            # summed across trace lanes; divide by _lanes for a wall-clock-comparable figure
            "self_us_per_iter_per_lane": excl[n] / iters / n_lanes,
            "per_region_us": {
                "mean": incl[n] / max(cnt[n], 1),
                "p50": v[len(v) // 2],
                "p90": v[int(len(v) * 0.9)],
                "max": v[-1],
            },
        }
    return out, n_lanes


def _selfcheck():
    """Nesting arithmetic + the return arity that broke twice. `python prof_fused_moe_v2.py --selfcheck`."""
    def ev(pid, tid, name, ts, dur):
        return {"ph": "X", "pid": pid, "tid": tid, "name": name, "ts": ts, "dur": dur}
    meta = lambda pid, tid: {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                             "args": {"name": "XLA TraceMe"}}
    trace = {"traceEvents": [
        meta(0, 1), meta(0, 2),
        # lane 1: outer[0,100) contains a[10,40) and b[50,70)  -> outer self = 100-30-20 = 50
        ev(0, 1, "outer", 0, 100), ev(0, 1, "a", 10, 30), ev(0, 1, "b", 50, 20),
        # lane 2: same shape again, so totals double and lane count is 2
        ev(0, 2, "outer", 0, 100), ev(0, 2, "a", 10, 30), ev(0, 2, "b", 50, 20),
    ]}
    mix, lanes = scope_mix(trace, iters=2)
    assert lanes == 2, lanes
    assert mix["outer"]["self_us_per_iter"] == 50.0, mix["outer"]        # (50+50)/2 iters
    assert mix["outer"]["inclusive_us_per_iter"] == 100.0, mix["outer"]  # (100+100)/2
    assert mix["outer"]["self_us_per_iter_per_lane"] == 25.0, mix["outer"]
    assert mix["a"]["self_us_per_iter"] == 30.0 and mix["a"]["regions"] == 2, mix["a"]
    # self times are additive: they must sum to the wall span, not over-count
    assert sum(v["self_us_per_iter"] for v in mix.values()) == 100.0, mix
    print("selfcheck OK")


# scope_mix is pure stdlib and has silently broken twice; keep it runnable
# without a TPU, and check it before the backend is touched.
if "--selfcheck" in sys.argv:
    _selfcheck()
    raise SystemExit(0)


# LIBTPU_INIT_ARGS must be set before the backend initialises, and the accepted
# flag names differ between libtpu builds. Probe in a subprocess and keep only
# the set this runtime accepts, rather than failing the whole run on a name.
_LLO_FLAGS = [
    "--xla_enable_custom_call_region_trace=true",
    "--xla_xprof_register_llo_debug_info=true",
]


def _usable_llo_flags() -> str:
    if "LIBTPU_INIT_ARGS" in os.environ:
        return os.environ["LIBTPU_INIT_ARGS"]
    for flags in (" ".join(_LLO_FLAGS), _LLO_FLAGS[1], ""):
        env = {**os.environ, "LIBTPU_INIT_ARGS": flags, "JAX_PLATFORMS": "tpu"}
        r = subprocess.run([sys.executable, "-c", "import jax; jax.devices()"],
                           env=env, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[llo] using LIBTPU_INIT_ARGS={flags!r}", flush=True)
            return flags
        print(f"[llo] rejected {flags!r}: {r.stderr.strip().splitlines()[-1][:120]}", flush=True)
    return ""


os.environ["LIBTPU_INIT_ARGS"] = _usable_llo_flags()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Load the kernel module by path: importing it as `sgl_jax.srt.kernels...` drags
# in sgl_jax.srt.utils, which needs zmq/fastapi/transformers. The kernel itself
# needs only jax and numpy, so the stock TPU bench image runs this unmodified.
import importlib.util

_KERNEL_PY = (pathlib.Path(__file__).resolve().parents[2]
              / "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py")
_spec = importlib.util.spec_from_file_location("fused_moe_v2_kernel", _KERNEL_PY)
_k = importlib.util.module_from_spec(_spec)
sys.modules["fused_moe_v2_kernel"] = _k
_spec.loader.exec_module(_k)
FusedMoEBlockConfig = _k.FusedMoEBlockConfig
fused_ep_moe_v2 = _k.fused_ep_moe_v2

# The tuned table is what production uses, but it imports
# sgl_jax.srt.utils.jax_utils. Exec it with that one import substituted rather
# than hardcoding a block config: the default (bf=512) is invalid for every
# model whose moe_intermediate_size is not a multiple of 512.
_TUNED_PY = _KERNEL_PY.parent / "tuned_block_configs.py"


def _load_tuned(device_kind: str):
    src = _TUNED_PY.read_text()
    src = src.replace(
        "from sgl_jax.srt.utils.jax_utils import get_device_name",
        f"def get_device_name():\n    return {device_kind!r}",
    ).replace("from .kernel import FusedMoEBlockConfig", "")
    ns = {"FusedMoEBlockConfig": FusedMoEBlockConfig}
    exec(compile(src, str(_TUNED_PY), "exec"), ns)  # noqa: S102
    return ns["get_tuned_fused_moe_v2_block_config"], ns["DEFAULT_V2_BLOCK_CONFIG"]


def _valid_config(bc, *, local_tokens, F, H):
    """kernel.py:236-246."""
    problems = []
    if local_tokens % bc.bt:
        problems.append(f"local_num_tokens={local_tokens} % bt={bc.bt}")
    if F % bc.bf:
        problems.append(f"intermediate_size={F} % bf={bc.bf}")
    if H % 128:
        problems.append(f"hidden_size={H} not 128-aligned")
    if bc.bf % 128:
        problems.append(f"bf={bc.bf} not 128-aligned")
    if bc.btc % 8:
        problems.append(f"btc={bc.btc} not 8-aligned (VREG sublane)")
    if bc.bts is not None and bc.bts % bc.btc:
        problems.append(f"bts={bc.bts} % btc={bc.btc}")
    return problems


def _fallback_config(local_tokens, F):
    """Largest 128-aligned bf dividing F; bt/btc from the token count."""
    bf = max((b for b in range(128, min(F, 512) + 1, 128) if F % b == 0), default=128)
    bt = max((b for b in (32, 16, 8, 4, 2, 1) if local_tokens % b == 0))
    return FusedMoEBlockConfig(bt=bt, bf=bf, btc=32, bse=min(256, bf))

LANES = ("VALU Instructions", "XLU Instructions", "VLD Instructions", "VST Instructions",
         "MXU0 Instructions", "MXU1 Instructions", "EUP Instructions", "SALU Instructions")
# Capture every named scope the kernel emits, not just the combine: without the
# whole breakdown there is no way to tell whether acc_compute is worth tuning.
SCOPES = None  # None = keep all
OUTPUT_URI = os.environ.get("OUTPUT_URI", "").rstrip("/")

# (label, num_experts, top_k, hidden, moe_intermediate) from real EPMoE models.
# (bt, bts): bts is capped at bt*ep_size, so raising bts past bt*ep needs a
# larger bt too. bts=None reproduces today's default (bts = bt).
# PROF_SWEEP=bts sweeps the token-block size (weight re-fetch vs padding waste);
# PROF_SWEEP=bf sweeps the intermediate-block size at the bts the bts sweep picked.
# Entries are (bt, bts, bf_override); bf_override=None keeps the per-shape choice.
_SWEEPS = {
    "bts": [(32, None, None), (32, 64, None), (32, 128, None)],
    "bts_wide": [(32, None, None), (32, 64, None), (32, 128, None),
                 (64, 256, None), (128, 512, None)],
    "bf": [(32, 64, 128), (32, 64, 256), (32, 64, 384), (32, 64, 768), (32, 64, 1408)],
    # GLM-4.5-Air: F=1408 has only two 128-aligned divisors. bf=1408 gives
    # num_bf=1, which is the global_rolling_wb path.
    "glm_bf": [(32, None, 128), (32, 64, 128), (32, None, 1408), (32, 64, 1408)],
}
BLOCKS = _SWEEPS[os.environ.get("PROF_SWEEP", "bts")]
# Tuning axes. ep_size comes from the mesh, so each ep value uses that many of
# the chip's devices; num_tokens must stay divisible by it.
SWEEP_EP = [int(x) for x in os.environ.get("PROF_EP", "1,2,4").split(",")]

SHAPES = [
    ("qwen3-30b-a3b", 128, 8, 2048, 768),
    ("glm-4.5-air", 128, 8, 4096, 1408),
]
TOKENS = [int(t) for t in os.environ.get("PROF_TOKENS", "128,512,2048,8192").split(",")]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def gcs_write(rel: str, obj) -> None:
    if not OUTPUT_URI:
        log("no OUTPUT_URI, printing instead:", json.dumps(obj)[:400])
        return
    tmp = pathlib.Path("/tmp") / rel.replace("/", "_")
    tmp.write_text(json.dumps(obj, indent=1))
    subprocess.run(["gcloud", "storage", "cp", str(tmp), f"{OUTPUT_URI}/{rel}"],
                   check=True, capture_output=True)


def gcs_exists(rel: str) -> bool:
    if not OUTPUT_URI:
        return False
    r = subprocess.run(["gcloud", "storage", "ls", f"{OUTPUT_URI}/{rel}"],
                       capture_output=True)
    return r.returncode == 0


def run_one(label, E, K, H, F, T, mesh, ep_size, bt_override, bts, bf_override):
    tag = f"{label}-T{T}-ep{ep_size}-bt{bt_override}-bts{bts}-bf{bf_override or 0}"
    rel = f"shards/{tag}/SUCCESS.json"
    if gcs_exists(rel):
        log(f"skip {label} T={T} (already done)")
        return
    rng = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    repl = NamedSharding(mesh, P())
    dev = lambda x: jax.device_put(x, repl)
    tokens = dev(jax.random.normal(k1, (T, H), jnp.bfloat16))
    w1 = dev(jax.random.normal(k2, (E, H, F), jnp.bfloat16) * 0.02)
    w3 = dev(jax.random.normal(k3, (E, H, F), jnp.bfloat16) * 0.02)
    w2 = dev(jax.random.normal(k4, (E, F, H), jnp.bfloat16) * 0.02)
    ids = jax.random.randint(k5, (T, K), 0, E, dtype=jnp.int32)
    wts = dev(jnp.full((T, K), 1.0 / K, jnp.float32))
    ids = dev(ids)

    local_tokens = T // ep_size
    bc = None
    try:
        get_tuned, _default = _load_tuned(jax.devices()[0].device_kind)
        bc = get_tuned(num_tokens=T, num_experts=E, top_k=K, hidden_size=H,
                       intermediate_size=F, dtype=jnp.bfloat16,
                       weight_dtype=jnp.bfloat16, ep_size=ep_size)
        src_desc = "tuned"
    except Exception as e:  # noqa: BLE001
        log(f"tuned lookup unavailable ({type(e).__name__}: {e}); using fallback")
    if bc is None or _valid_config(bc, local_tokens=local_tokens, F=F, H=H):
        if bc is not None:
            log(f"tuned {bc} invalid: {_valid_config(bc, local_tokens=local_tokens, F=F, H=H)}")
        bc = _fallback_config(local_tokens, F)
        src_desc = "fallback"
    bf = bf_override or bc.bf
    bc = FusedMoEBlockConfig(bt=bt_override, bf=bf, btc=bc.btc, bse=min(bc.bse, bf), bts=bts)
    problems = _valid_config(bc, local_tokens=local_tokens, F=F, H=H)
    if problems:
        raise ValueError(f"no valid block config for F={F} H={H} T={T} ep={ep_size}: {problems}")
    tokens_per_expert = (T // ep_size) * K / (E // ep_size)
    eff_bts = bts if bts else bt_override
    n_bts_tiles = -(-int(tokens_per_expert) // min(eff_bts, bt_override * ep_size))
    log(f"{label} T={T} H={H} F={F} ep={ep_size} block={bc} ({src_desc}) "
        f"tok/expert={tokens_per_expert:.0f} num_bts_tiles={n_bts_tiles}")

    # jit it: called eagerly, the wall clock measures Python dispatch rather than
    # the kernel, and comes out roughly constant across block configs.
    _jitted = jax.jit(
        lambda t, a, b, c, w, i: fused_ep_moe_v2(mesh, t, a, b, c, w, i, K, block_config=bc)
    )

    def run():
        return _jitted(tokens, w1, w2, w3, wts, ids)

    out = jax.block_until_ready(run())
    for _ in range(3):
        jax.block_until_ready(run())
    wall = []
    for _ in range(10):
        t0 = time.perf_counter()
        jax.block_until_ready(run())
        wall.append((time.perf_counter() - t0) * 1e6)
    wall_us = sorted(wall)[len(wall) // 2]

    iters = 5
    root = f"/tmp/prof_{label}_{T}"
    subprocess.run(["rm", "-rf", root], check=False)
    # Same counter-sampling options bench_v2 uses; without them the trace has no
    # per-execution-unit instruction lanes and the VReg question is unanswerable.
    try:
        popts = jax.profiler.ProfileOptions()
        popts.advanced_configuration = {
            "tpu_enable_periodic_counter_sampling": True,
            "tpu_tc_perf_counter_sampling_options": (
                "interval_us:1 scaling:0 counter_size_bits:1 "
                "indices:1 indices:3 indices:4 indices:10 indices:11 "
                "indices:31 indices:32 indices:33 indices:34 indices:35 "
                "indices:37 indices:38 indices:56 indices:57 indices:58 "
                "indices:73 indices:74 indices:75 indices:105"
            ),
            "num_tensor_cores_to_trace_per_device": 1,
        }
        trace_cm = jax.profiler.trace(root, profiler_options=popts)
    except Exception as e:  # noqa: BLE001
        log(f"counter sampling unavailable ({type(e).__name__}: {e}); plain trace")
        trace_cm = jax.profiler.trace(root)
    with trace_cm:
        for _ in range(iters):
            jax.block_until_ready(run())
    mix, n_lanes = scope_mix(load_trace(root), iters)

    acc_bt = min(bc.bt, 16)
    packing = 2                      # bf16
    # one multiply + one add per element per k, over acc_bt x hidden f32 lanes
    vregs_per_tile = (acc_bt * H) / (8 * 128)
    ideal_valu = K * 2 * vregs_per_tile * (T / acc_bt)
    obs = None  # instruction lanes are not emitted by this runtime
    rec = {"label": label, "E": E, "top_k": K, "hidden": H, "intermediate": F,
           "tokens": T, "ep_size": ep_size, "wall_us": round(wall_us, 1),
           "tokens_per_expert": tokens_per_expert,
           "num_bts_tiles": n_bts_tiles, "block_config_source": src_desc, "block_config": bc.__dict__ if hasattr(bc, "__dict__") else str(bc),
           "acc_bt": acc_bt, "out_packing": packing,
           "ideal_valu_per_iter": round(ideal_valu, 1),
           "observed_valu_per_iter": (obs / iters) if obs else None,
           "valu_ratio": round((obs / iters) / ideal_valu, 2) if obs and ideal_valu else None,
           "trace_lanes": n_lanes, "scopes": mix, "output_absmax": float(np.abs(np.asarray(jax.device_get(out), np.float32)).max())}
    log(json.dumps({k: rec[k] for k in ("label", "tokens", "ideal_valu_per_iter",
                                        "observed_valu_per_iter", "valu_ratio")}))
    gcs_write(f"shards/{tag}/result.json", rec)
    gcs_write(rel, {"ok": True, "ts": time.time()})
    subprocess.run(["rm", "-rf", root], check=False)


def main():
    log("jax", jax.__version__, jax.devices())
    devices = jax.devices()
    only = [x for x in os.environ.get("PROF_SHAPES", "").split(",") if x]
    ok = failed = 0
    for label, E, K, H, F in SHAPES:
        if only and label not in only:
            continue
        for ep_size in SWEEP_EP:
            if ep_size > len(devices):
                log(f"skip ep={ep_size}: only {len(devices)} devices")
                continue
            if E % ep_size:
                log(f"skip ep={ep_size}: {E} experts not divisible")
                continue
            mesh = Mesh(np.array(devices[:ep_size]).reshape(1, ep_size),
                        axis_names=("data", "tensor"))
            for T in TOKENS:
                if T % ep_size:
                    continue
                for bt_o, bts, bf_o in BLOCKS:
                    if (T // ep_size) % bt_o:
                        log(f"skip T={T} ep={ep_size} bt={bt_o}: local not divisible")
                        continue
                    if bf_o is not None and (F % bf_o or bf_o > F):
                        log(f"skip bf={bf_o}: does not divide intermediate_size={F}")
                        continue
                    try:
                        run_one(label, E, K, H, F, T, mesh, ep_size, bt_o, bts, bf_o)
                        ok += 1
                    except Exception as e:  # noqa: BLE001
                        failed += 1
                        log(f"FAILED {label} T={T} ep={ep_size} bt={bt_o} bts={bts} "
                            f"bf={bf_o}: {type(e).__name__}: {str(e)[:150]}")
                        gcs_write(
                            f"shards/{label}-T{T}-ep{ep_size}-bt{bt_o}-bts{bts}-bf{bf_o or 0}/ERROR.json",
                            {"error": f"{type(e).__name__}: {str(e)[:300]}"})
    log(f"done: {ok} ok, {failed} failed")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()


