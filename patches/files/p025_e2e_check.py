#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-025 gb10-mhc-prefill-sp] END-TO-END CPU suite (no GPU, no CUDA init, no TileLang; torch on CPU + gloo).

Added in rev 2 to close the rev-1 verifier's B1: the boot self-test and the data-flow sections of p025_cpu_check.py
exercised only the RS/AG primitives, so a layer-WIRING bug in sp_layer_forward / the shard / the gather / the
suppression handles (an unsuppressed o_proj / MoE / dense all-reduce, a missing / rolled gather, the wrong ffn
weights, a skipped last-layer hc_post, a missing exit gather, a ref without the all-gather) produced bitwise
rank-consistent output and still satisfied sp == ref. This suite runs the REAL patched model.py forward functions
(Glm5NextModel.forward, Glm5NextDecoderLayer.forward / hc_pre / hc_post / hc_fused_post_pre, Glm5NextMoE.forward
(016), Glm5NextMLP.forward) and the REAL vllm/_gb10_prefill_sp.py (the production _TPWorld over a gloo-backed fake TP
group, incl. its canonical reduce-scatter over a gloo-backed fake pynccl send/recv, the prs neutralisation and the
full boot self-test) on 8 gloo ranks, against the image path (gate off) of the SAME code. The REAL patch-022 prs module
is hooked into a toy pre-norm GEMM. Rev 3 adds: the fake TP group's device communicator (pynccl send/recv + a patch-012
one-shot all-reduce emulation, same arithmetic), a counter that proves sp never touches the torch-c10d device group,
the toy layers' layer_kind / input_size_per_partition (so the boot self-test's live checks run on layer 0 = KDA +
dense and layer 3 = MLA + MoE, and check 2 on their o_proj / down_proj), the boot-only mode and the HANG
mutants (a rank-local exception at each stage of the boot self-test: every one must end in a clean raise on ALL ranks).

Modes:
  launch  one config (base or a mutant module via --sp-module)        -> the E1..E18 checks on 8 gloo ranks
          --bootonly MODE,RS,T: only the first eligible begin_forward (envelope + preflight + the boot self-test, as in
          the profile run) on the toy model's live layers with the real 022 module; every rank writes its verdict and
          waits for the others (no rank exits early, so a hang stays a hang); summary.json classifies the run:
          OK (no rank raised), CLEAN-RAISE (every rank raised a [patch-025] error), PARTIAL (some ranks raised —
          would wedge the next collective in production), HANG (a rank never finished within --timeout), FATAL.
  worker  one gloo rank (spawned by launch)
  sweep   the base + every negative-control mutant; exit 0 iff the base passes AND every REQUIRED mutant is caught
          (by this e2e, by the boot self-test raising, or by p025_cpu_check.py's module-logic sections B/C), AND the
          boot-only base is OK for RS=nccl and RS=canonical AND every HANG mutant is a CLEAN-RAISE in each RS mode it
          applies to. Two mutants are documented as not catchable on CPU: begin_no_capture_check (a CUDA-graph-capture
          guard; covered by p025_gpu_selftest.py section G) and canon_reverse_order (reversing a fixed-rank-order
          fp32 sum whose bf16 output is unchanged — a benign reordering, not a defect; a reorder that DID move the
          bf16 output is caught by the canonical==rank-order-fp32-sum self-test).

  CUDA_VISIBLE_DEVICES= python3 files/p025_e2e_check.py sweep --patches <patches dir> --vsrc <v0.32.0 tree>/vllm --work DIR      (--vsrc defaults to $VLLM_TREE, else vllm-v0.32.0/vllm)
  CUDA_VISIBLE_DEVICES= python3 files/p025_e2e_check.py launch --patches <patches dir> --vsrc .../vllm --work DIR
  CUDA_VISIBLE_DEVICES= python3 files/p025_e2e_check.py launch ... --bootonly sp,nccl,8192 --barrier-s 20 --timeout 180

Toy model (the rev-1 independent verifier's, plus rev 3's layer_kind / input_size_per_partition and an exp-based
sigmoid): 6 layers (0-2 dense MLP, 3-5 MoE; layer 3 causal softmax 'MLA', others causal linear 'KDA'), H=64, n_hc=4,
8 heads (1 per rank), MoE 16 experts top-4 + shared expert (TP-sharded intermediate), vocab-parallel embedding, final
RMSNorm, plus one MTP-style layer after the forward. mHC ops are per-row torch (shard-invariant: rev 3 replaced
torch.sigmoid, whose CPU vectorized body and scalar tail round differently, by 1/(1+exp(-t))). The image path's
all-reduce is 'canonical' (all-gather + rank-order fp32 sum = patch-012 arithmetic) or gloo's native all_reduce.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = 64
NHC = 4
HEADS = 8
HD = 16
VOCAB = 1024
E = 16
TOPK = 4
I_EXP = 32
I_SHARED = 64
I_DENSE = 64
L = 6
MIN_T = 1024

# -- negative-control mutants of gb10_prefill_sp.py (source rewrites; each must match exactly once) ------------------
# Each is applied to a FRESH copy of the module, so the shared anchors (e.g. the canonical acc loop) do not collide.
MUTANTS = {
    # sp_layer_forward wiring
    "skip_mlp_gather": (
        '    x_full = gather_full(x, handle.world, handle.plan) if handle.mode == "sp" else x\n'
        '    with _mlp_reduce_handle(',
        '    x_full = x\n'
        '    with _mlp_reduce_handle('),
    "skip_attn_gather": (
        '    x_full = gather_full(x, handle.world, handle.plan) if handle.mode == "sp" else x\n'
        '    with _attn_reduce_handle(',
        '    x_full = x\n'
        '    with _attn_reduce_handle('),
    "ffn_pre_uses_attn_weights": (
        "        x, residual, post, comb, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,",
        "        x, residual, post, comb, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,"),
    "last_layer_skip_post": (
        "        x = layer.hc_post(x, residual, post, comb)\n        x = _hc_contract(x, n)",
        "        x = _hc_contract(residual, n)"),
    # row plan / shard / gather
    "shard_rolled_by_one": (
        "    return xp[plan.start(rank): plan.start(rank) + plan.S].contiguous()",
        "    return torch.roll(xp, -1, 0)[plan.start(rank): plan.start(rank) + plan.S].contiguous()"),
    "gather_drops_head": (
        "    return g[: plan.T]",
        "    return g[plan.pad: plan.pad + plan.T] if plan.pad else g[: plan.T]"),
    "pad_with_ones": (
        "    return torch.nn.functional.pad(x.contiguous(), pad)",
        "    return torch.nn.functional.pad(x.contiguous(), pad, value=1.0)"),
    # reduction / exit gather
    "ref_mode_no_allgather": (
        '    if handle.mode == "ref":\n        return gather_full(shard, handle.world, handle.plan)\n    return shard',
        '    return shard'),
    "no_final_gather": (
        "    return gather_full(hidden_states, handle.world, handle.plan)\n",
        "    return hidden_states\n"),
    "rs_double_count_own_rows": (
        "        acc = recv[0].to(torch.float32)\n        for p in range(1, W):\n"
        "            acc = acc + recv[p].to(torch.float32)\n        return acc.to(dtype)",
        "        acc = recv[0].to(torch.float32)\n        for p in range(1, W):\n"
        "            acc = acc + recv[p].to(torch.float32)\n        acc = acc + recv[r].to(torch.float32)\n"
        "        return acc.to(dtype)"),
    "nccl_rs_rotated_rows": (
        "        return self.group.reduce_scatter(x, dim=0)",
        "        return self.group.reduce_scatter(torch.roll(x, x.shape[0] // self.size, 0), dim=0)"),
    # all-reduce suppression
    "no_attn_suppress": (
        '    return _SuppressReduce(o_proj, "reduce_results", False)',
        '    return _SuppressReduce(o_proj, "reduce_results", True)'),
    "no_moe_suppress": (
        '        return _SuppressReduce(cfg, "skip_final_all_reduce", True)',
        '        return _SuppressReduce(cfg, "skip_final_all_reduce", False)'),
    "no_dense_suppress": (
        '    return _SuppressReduce(down, "reduce_results", False)',
        '    return _SuppressReduce(down, "reduce_results", True)'),
    # region bookkeeping
    "end_no_restore_prs": (
        "    _restore_prs()\n    _STATE.active_handle = None",
        "    _STATE.active_handle = None"),
    "no_prs_neutralise": (
        "    prs._STATE.min_t = MIN_T_CEIL << 20",
        "    pass  # mutant: prs not neutralised"),
    # self-test weakened (does the suite notice a self-test that cannot fail?)
    "selftest_always_ok": (
        '    bad = [r for r in allr if not r["ok"]]',
        "    bad = []"),
    # rev 3: live layers / check 2 / canonical gating
    "moe_handle_wrong_target": (
        '        return _SuppressReduce(cfg, "skip_final_all_reduce", True)',
        '        return _SuppressReduce(mlp.shared_experts.down_proj, "reduce_results", False)'),
    "live_layers_none": (
        "    return first, mla_moe\n",
        "    return None, None\n"),
    "live_no_tail_T": (
        "LIVE_T_EXTRA = 3 ",
        "LIVE_T_EXTRA = 0 "),
    "canonical_check_ungated": (
        '        if self.rs == "canonical":\n            self.step(f"{name} canonical reduce-scatter"',
        '        if True:\n            self.step(f"{name} canonical reduce-scatter"'),
    # not catchable on CPU (documented):
    "begin_no_capture_check": (
        "    if _capturing(probe):\n        st.capture_skips += 1\n        return None\n",
        ""),
    "canon_reverse_order": (
        "        acc = recv[0].to(torch.float32)\n        for p in range(1, W):\n"
        "            acc = acc + recv[p].to(torch.float32)\n        return acc.to(dtype)",
        "        acc = recv[W - 1].to(torch.float32)\n        for p in range(W - 2, -1, -1):\n"
        "            acc = acc + recv[p].to(torch.float32)\n        return acc.to(dtype)"),
}
# begin_no_capture_check: no CUDA-graph capture exists on CPU, so removing the guard changes nothing here (covered by
# p025_gpu_selftest.py section G). canon_reverse_order: a reversed fixed-order fp32 sum rounds to the same bf16 for
# well-conditioned partials, so its output is unchanged (not a defect). Every OTHER mutant is REQUIRED to be caught.
BENIGN_ON_CPU = {"begin_no_capture_check", "canon_reverse_order"}


def _raise7(msg, indent, cond="r == 7"):
    return f"{' ' * indent}if {cond}:\n{' ' * indent}    raise RuntimeError(\"planted rank-local failure: {msg}\")\n"


# -- HANG mutants (rev 3): a rank-local exception (or a wrong local shape) at each stage of the boot self-test. Each is
# run boot-only (the first eligible begin_forward, like the profile run) for RS=nccl and RS=canonical; each MUST end in
# a CLEAN-RAISE on all 8 ranks (rev 2 hung on h_plan_start_shift2, the rev-2 verifier's plan_start_shift2). (old, new):
# `new` replaces `old` (exactly one match in the module).
HANG_MUTANTS = {
    "h_envelope_r7": (
        "def check_envelope(world, is_sequence_parallel: bool = False) -> None:\n",
        "def check_envelope(world, is_sequence_parallel: bool = False) -> None:\n"
        + _raise7("envelope check", 4, "world.rank == 7")),
    "h_geometry_r7": (
        "    W = world.size\n    geo = {",
        _raise7("geometry", 4, "world.rank == 7") + "    W = world.size\n    geo = {"),
    "h_plan_S_r7": (
        "    S = padded // W\n    return Plan(T, W, S, padded - T)",
        "    S = padded // W + (2 if os.environ.get('RANK') == '7' and T > 64 else 0)\n"
        "    return Plan(T, W, S, padded - T)"),
    "h_case_prep_r7": (
        "        def prep():\n            check_plan(plan)\n",
        "        def prep():\n" + _raise7("case prep", 12) + "            check_plan(plan)\n"),
    "h_case_prep_r0": (
        "        def prep():\n            check_plan(plan)\n",
        "        def prep():\n" + _raise7("case prep on rank 0", 12, "r == 0") + "            check_plan(plan)\n"),
    "h_ref_after_ag_r7": (
        '            allp = self.sched.all_gather_rows(c["padded"]).view(W, plan.padded, H)\n',
        '            allp = self.sched.all_gather_rows(c["padded"]).view(W, plan.padded, H)\n'
        + _raise7("after the reference all-gather", 12)),
    "h_canonical_r7": (
        "        def canonical():\n",
        "        def canonical():\n" + _raise7("canonical check", 12)),
    "h_rs_production_r7": (
        "    padded = pad_rows(partial_full, plan.padded)\n    if rs_mode == \"canonical\":",
        _raise7("production reduce-scatter (before the collective)", 4, "getattr(world, 'rank', -1) == 7")
        + "    padded = pad_rows(partial_full, plan.padded)\n    if rs_mode == \"canonical\":"),
    "h_sensitivity_r7": (
        "        def sensitivity():\n",
        "        def sensitivity():\n" + _raise7("double-reduction probe", 12)),
    "h_final_gather_shape_r7": (
        '            c["full"] = gather_full(c["got"], self.sched, plan)\n',
        '            c["full"] = gather_full(c["got"][:-2] if r == 7 else c["got"], self.sched, plan)\n'),
    "h_plan_start_shift2": (
        "        return r * self.S\n",
        "        return r * self.S + (2 if r else 0)\n"),
    "h_case_verdict_r7": (
        '        def verdict():\n            full = c["full"]\n',
        '        def verdict():\n' + _raise7("case verdict", 12) + '            full = c["full"]\n'),
    "h_live_inputs_r7": (
        '        def inputs():\n            c["res"]',
        '        def inputs():\n' + _raise7("live inputs", 12) + '            c["res"]'),
    "h_live_pre_args_r7": (
        "        def pre_args():\n",
        "        def pre_args():\n" + _raise7("hc_pre full-T arguments", 12)),
    "h_live_fused_args_r7": (
        "        def fused_args():\n",
        "        def fused_args():\n" + _raise7("hc_fused_post_pre full-T arguments", 12)),
    "h_live_shards_r7": (
        "        def shards():\n",
        "        def shards():\n" + _raise7("live shard calls", 12)),
    "h_suppression_r7": (
        "    rows: list[dict] = []\n    tag = f\"L{getattr(layer, 'layer_idx', '?')}\"\n",
        "    rows: list[dict] = []\n    tag = f\"L{getattr(layer, 'layer_idx', '?')}\"\n"
        + _raise7("suppression cases", 4, "os.environ.get('RANK') == '7'")),
    "h_check2_partial_r7": (
        "        def partial():\n",
        "        def partial():\n" + _raise7("check-2 partial", 12)),
    "h_check2_ar_r7": (
        "        def decode_ar():\n",
        "        def decode_ar():\n" + _raise7("check-2 decode all-reduce", 12)),
    "h_verdict_build_r7": (
        '        mine = {"rank": world.rank, "ok": all(r["ok"] for r in st.rows),',
        _raise7("verdict build", 8, "world.rank == 7")
        + '        mine = {"rank": world.rank, "ok": all(r["ok"] for r in st.rows),'),
    "h_make_world_r7": (
        "def _make_world():\n    return",
        "def _make_world():\n" + _raise7("world creation (bounded by the monitored barrier)", 4,
                                         "os.environ.get('RANK') == '7'") + "    return"),
}
# stages that only exist with RS=canonical (under RS=nccl the planted line never runs: OK expected there)
HANG_CANONICAL_ONLY = {"h_canonical_r7"}


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def pin(sh, name):
    import re
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", Path(sh).read_text(), re.M)
    return m.group(1)


# ----------------------------------------------------------------------------------------------------------------
# launcher
# ----------------------------------------------------------------------------------------------------------------
def build_patched(patches, vsrc, work):
    """v0.32.0 original common/model.py -> p025_apply.apply; sha-checked against the .sh pins."""
    rel = "models/glm5next/common/model.py"
    orig = (vsrc / rel).read_text()
    sh25 = patches / "025-gb10-mhc-prefill-sp.sh"
    assert sha_text(orig) == pin(sh25, "MODEL_ORIG"), "model.py != MODEL_ORIG pin"
    spec = importlib.util.spec_from_file_location("p025_apply_e2e", str(patches / "files" / "p025_apply.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    p25 = P.apply(orig)
    assert sha_text(p25) == pin(sh25, "MODEL_ORIG_P025"), "025 result sha != pin"
    (work / "model_p025.py").write_text(p25)
    return work / "model_p025.py"


def launch(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    patches = Path(a.patches)
    model_path = build_patched(patches, Path(a.vsrc), work)
    sp_module = Path(a.sp_module) if a.sp_module else patches / "files" / "gb10_prefill_sp.py"
    prs_module = patches / "files" / "gb10_mhc_prs.py"
    out = work / f"out_{a.tag}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    env = dict(os.environ)
    env.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(a.port), "WORLD_SIZE": str(a.world),
                "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "GLOO_SOCKET_IFNAME": ("lo0" if sys.platform == "darwin" else "lo"),
                "PYTHONDONTWRITEBYTECODE": "1"})
    procs = []
    t0 = time.time()
    for r in range(a.world):
        e = dict(env, RANK=str(r))
        cmd = [sys.executable, __file__, "worker", "--model", str(model_path), "--sp-module", str(sp_module),
               "--prs-module", str(prs_module), "--out", str(out), "--world", str(a.world), "--tlist", a.tlist,
               "--timeout", str(a.timeout)]
        if a.bootonly:
            cmd += ["--bootonly", a.bootonly]
        if a.barrier_s:
            cmd += ["--barrier-s", str(a.barrier_s)]
        procs.append(subprocess.Popen(cmd, env=e, stdout=open(out / f"rank{r}.log", "w"), stderr=subprocess.STDOUT))
    rcs = []
    deadline = t0 + a.timeout                    # ONE deadline for the whole world (a hang is a hang)
    for p in procs:
        try:
            rcs.append(p.wait(timeout=max(1.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs.append("timeout")
    for p in procs:
        if p.poll() is None:
            p.kill()
    res = []
    for r in range(a.world):
        f = out / f"rank{r}.json"
        res.append(json.loads(f.read_text()) if f.exists() else None)
    if a.bootonly:
        return bootonly_summary(a.tag, res, rcs, out, time.time() - t0), False
    boot_raised = any(x is not None and x.get("fatal") and "BOOT SELF-TEST FAILED" in x["fatal"] for x in res)
    rc = summarize(a.tag, res, rcs, out)
    return rc, boot_raised


def classify_boot(res, rcs) -> str:
    """OK | CLEAN-RAISE | PARTIAL | HANG | FATAL for a boot-only run (see the module docstring)."""
    if any(rc == "timeout" for rc in rcs) or any(x is None for x in res):
        return "HANG"
    if any(x.get("fatal") for x in res):
        return "FATAL"
    raised = [x for x in res if x.get("boot_exc")]
    if not raised:
        return "OK"
    if len(raised) == len(res) and all("[patch-025]" in x["boot_exc"] for x in raised):
        return "CLEAN-RAISE"
    return "PARTIAL"


def bootonly_summary(tag, res, rcs, out, wall) -> int:
    kind = classify_boot(res, rcs)
    per = []
    for r, x in enumerate(res):
        if x is None:
            per.append({"rank": r, "verdict": "NO-RESULT", "rc": rcs[r]})
        elif x.get("fatal"):
            per.append({"rank": r, "verdict": "FATAL", "why": x["fatal"].strip().splitlines()[-1][:240]})
        elif x.get("boot_exc"):
            per.append({"rank": r, "verdict": "RAISED", "boot_s": x.get("boot_s"),
                        "why": x["boot_exc"].strip().splitlines()[-1][:240]})
        else:
            per.append({"rank": r, "verdict": "OK", "boot_s": x.get("boot_s"), "prs_booted": x.get("prs_booted"),
                        "report": x.get("report"), "flags_default": x.get("flags_default"),
                        "p2p": (x.get("coll") or {}).get("p2p"), "devgroup": (x.get("coll") or {}).get("devgroup")})
    summary = {"tag": tag, "kind": kind, "wall_s": round(wall, 1), "rcs": [str(rc) for rc in rcs], "ranks": per}
    (out / "summary.json").write_text(json.dumps(summary, default=str, indent=1))
    print(f"BOOTONLY [{tag}] {kind} wall={wall:.1f}s rcs={summary['rcs']}")
    for p in per[:8]:
        print("   " + json.dumps({k: v for k, v in p.items() if k != "report"}, default=str)[:300])
    return 0 if kind == "OK" else 1


def summarize(tag, res, rcs, out):
    checks = []

    def chk(name, ok, info=""):
        checks.append((name, bool(ok), info))

    ok_ranks = [r for r in res if r is not None]
    chk("all ranks finished", len(ok_ranks) == len(res) and all(rc == 0 for rc in rcs), f"rcs={rcs}")
    if not ok_ranks:
        if (out / "rank0.log").exists():
            print("\n".join((out / "rank0.log").read_text().splitlines()[-20:]))
        return report(tag, checks)
    r0 = ok_ranks[0]
    if r0.get("fatal"):
        chk("no fatal exception", False, r0["fatal"].strip().splitlines()[-1][:200] if r0["fatal"] else "")
        return report(tag, checks)
    incons = []
    for key in r0["hash"]:
        if len({r["hash"].get(key) for r in ok_ranks}) != 1:
            incons.append(key)
    chk("every output bit-identical across the 8 ranks (all configs, all T)", not incons, str(incons[:6]))
    for name, (ok, info) in r0["checks"].items():
        allok = all(r["checks"].get(name, [False])[0] for r in ok_ranks)
        chk(name, allok, info)
    return report(tag, checks)


def report(tag, checks):
    npass = sum(1 for c in checks if c[1])
    for n, ok, info in checks:
        print(("PASS " if ok else "FAIL ") + f"[{tag}] " + n + (f"  [{info}]" if info else ""))
    print(f"[{tag}] {npass}/{len(checks)} passed")
    return 0 if npass == len(checks) else 1


# ----------------------------------------------------------------------------------------------------------------
# worker (one gloo rank): executes the REAL patched functions vs the image path
# ----------------------------------------------------------------------------------------------------------------
def worker(a):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    torch.set_num_threads(1)
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", rank=rank, world_size=W)
    out = {"rank": rank, "hash": {}, "checks": {}}
    bf = torch.bfloat16

    def put(name, ok, info=""):
        out["checks"][name] = [bool(ok), str(info)]

    try:
        for k in [k for k in list(sys.modules) if k == "vllm" or k.startswith("vllm.")]:
            del sys.modules[k]

        def mkmod(name, pkg=False):
            m = types.ModuleType(name)
            if pkg:
                m.__path__ = []
            sys.modules[name] = m
            return m

        vllm = mkmod("vllm", True)
        for nm in ("vllm.distributed", "vllm.model_executor", "vllm.model_executor.layers", "vllm.config"):
            mkmod(nm, True)
        ps = mkmod("vllm.distributed.parallel_state")
        mhcmod = mkmod("vllm.model_executor.layers.mhc")
        sys.modules["vllm.config"].get_current_vllm_config_or_none = lambda: None

        def hc_expand(x, n):
            return x.unsqueeze(1).expand(-1, n, -1).contiguous()

        def hc_contract(x, n):
            return x.mean(dim=1)

        mhcmod.hc_expand, mhcmod.hc_contract = hc_expand, hc_contract

        COLL = {"ag": 0, "rs": 0, "ar": 0, "p2p": 0, "oneshot": 0, "dc_ar": 0}
        DEVG = {"n": 0}          # torch-c10d device-group accesses (never reset): must stay 0
        BOOTS = {}               # per (mode, rs) boot: the self-test report + the P2P / one-shot traffic it issued

        class FakePyNccl:
            """vLLM's PyNcclCommunicator send/recv/group surface over gloo: ops queued between group_start and
            group_end run as one batch_isend_irecv (NCCL group semantics)."""

            def __init__(self):
                self.rank, self.world_size, self.disabled = rank, W, False
                self._q = None

            def group_start(self):
                assert self._q is None, "nested pynccl group"
                self._q = []

            def group_end(self):
                q, self._q = self._q, None
                if q:
                    for w_ in dist.batch_isend_irecv(q):
                        w_.wait()

            def send(self, t, dst, stream=None):
                assert self._q is not None and t.is_contiguous(), "pynccl send outside a group / non-contiguous"
                COLL["p2p"] += 1
                self._q.append(dist.P2POp(dist.isend, t, dst))

            def recv(self, t, src, stream=None):
                assert self._q is not None and t.is_contiguous(), "pynccl recv outside a group / non-contiguous"
                self._q.append(dist.P2POp(dist.irecv, t, src))

        class FakeOneShot:
            """patch 012's one-shot all-reduce (gb10_oneshot_ar.py) emulated: the same should_use() rule and the same
            per-element arithmetic (acc = 0; acc += own + got over ranks 0..W-1 in fp32; one bf16 rounding)."""
            max_elems = 65536 // 2

            def should_use(self, x):
                return x.dtype == torch.bfloat16 and x.is_contiguous() and 0 < x.numel() <= self.max_elems

            def __call__(self, x):
                COLL["oneshot"] += 1
                flat = x.contiguous().view(-1)
                g = torch.empty(W * flat.numel(), dtype=flat.dtype)
                dist.all_gather_into_tensor(g, flat)
                g = g.view(W, -1)
                acc = torch.zeros(flat.numel(), dtype=torch.float32)
                zero = torch.zeros_like(acc)
                for r_ in range(W):
                    own = flat.float() if r_ == rank else zero
                    got = g[r_].float() if r_ != rank else zero
                    acc = acc + (own + got)
                return acc.to(torch.bfloat16).view_as(x)

        class FakeDevComm:
            def __init__(self):
                self.pynccl_comm = FakePyNccl()
                self._gb10_oneshot = FakeOneShot()

            def all_reduce(self, x):
                if self._gb10_oneshot.should_use(x):
                    return self._gb10_oneshot(x)
                COLL["dc_ar"] += 1
                o = x.contiguous().clone()
                dist.all_reduce(o)
                return o

        class FakeTPGroup:
            def __init__(self):
                self.rank_in_group = rank
                self.world_size = W
                self.ranks = list(range(W))
                self.cpu_group = dist.group.WORLD
                self.device_communicator = FakeDevComm()

            @property
            def device_group(self):
                # rev 3: sp must never use the torch-c10d device group (a new NCCL communicator at boot); E15 checks 0.
                DEVG["n"] += 1
                return dist.group.WORLD

            def all_reduce(self, x):
                return self.device_communicator.all_reduce(x)

            def all_gather(self, x, dim=-1):
                assert dim == 0
                x = x.contiguous()
                o = torch.empty((x.shape[0] * W,) + tuple(x.shape[1:]), dtype=x.dtype)
                dist.all_gather_into_tensor(o, x)
                COLL["ag"] += 1
                return o

            def reduce_scatter(self, x, dim=-1):
                assert dim == 0
                x = x.contiguous()
                assert x.shape[0] % W == 0
                o = torch.empty((x.shape[0] // W,) + tuple(x.shape[1:]), dtype=x.dtype)
                dist.reduce_scatter_tensor(o, x, op=dist.ReduceOp.SUM)
                COLL["rs"] += 1
                return o

        TPG = FakeTPGroup()
        PPG = types.SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
        ps.get_tp_group = lambda: TPG
        ps.get_pp_group = lambda: PPG

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m
            spec.loader.exec_module(m)
            setattr(vllm, name.split(".")[-1], m)
            return m

        os.environ["VLLM_GB10_MHC_SP"] = "sp"
        os.environ["VLLM_GB10_MHC_PRS"] = "1"
        SP = load("vllm._gb10_prefill_sp", a.sp_module)
        PRS = load("vllm._gb10_mhc_prs", a.prs_module)
        if a.barrier_s:
            SP.SELFTEST_BARRIER_S = float(a.barrier_s)   # tests only: the bounded gloo barrier before each exchange

        # ---- toy collectives for the image path --------------------------------------------------------------
        AR_MODE = {"v": "canonical"}

        def toy_all_reduce(t):
            COLL["ar"] += 1
            t = t.contiguous()
            if AR_MODE["v"] == "gloo":
                o = t.clone()
                dist.all_reduce(o)
                return o
            g = torch.empty((W,) + tuple(t.shape), dtype=t.dtype)
            dist.all_gather_into_tensor(g.view(W * t.shape[0], *t.shape[1:]), t)
            acc = g[0].to(torch.float32)
            for p in range(1, W):
                acc = acc + g[p].to(torch.float32)
            return acc.to(t.dtype)

        gen = torch.Generator().manual_seed(25025)

        def rnd(*shape, s=1.0):
            return torch.randn(*shape, generator=gen) * s

        class Param:
            def __init__(self, t):
                self.data = t

        class RMSNorm:
            def __init__(self, w, eps=1e-5):
                self.weight = Param(w)
                self.variance_epsilon = eps

            def __call__(self, x, residual=None):
                if residual is not None:
                    s = (x.float() + residual.float())
                    y = (s * torch.rsqrt(s.square().mean(-1, keepdim=True) + self.variance_epsilon)
                         * self.weight.data.float()).to(bf)
                    return y, s.to(bf)
                xf = x.float()
                return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.variance_epsilon)
                        * self.weight.data.float()).to(bf)

        class RowParallel:
            def __init__(self, w_full, in_dim_shard=True):
                inl = w_full.shape[0] // W
                self.w = w_full[rank * inl:(rank + 1) * inl].contiguous()
                self.reduce_results = True
                self.input_size_per_partition = inl

            def __call__(self, x):
                o = (x.float() @ self.w).to(bf)
                if self.reduce_results:
                    o = toy_all_reduce(o)
                return o, None

        class ColParallel:
            def __init__(self, w_full_list):
                parts = []
                for w in w_full_list:
                    ol = w.shape[1] // W
                    parts.append(w[:, rank * ol:(rank + 1) * ol])
                self.w = torch.cat(parts, dim=1).contiguous()

            def __call__(self, x):
                return (x.float() @ self.w).to(bf), None

        class SiluAndMul:
            def __call__(self, x):
                d = x.shape[-1] // 2
                return F.silu(x[..., :d]) * x[..., d:]

        def rope(x, pos):
            half = x.shape[-1] // 2
            f = torch.exp(-torch.arange(half, dtype=torch.float32) / half * 4.0)
            ang = pos.float().unsqueeze(-1) * f
            c, s = torch.cos(ang), torch.sin(ang)
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

        SEGS = {"v": None}

        class Attn:
            def __init__(self, kind):
                self.kind = kind
                wq, wk, wv = (rnd(H, HEADS * HD, s=H ** -0.5) for _ in range(3))
                self.qkv = ColParallel([wq, wk, wv])
                self.o_proj = RowParallel(rnd(HEADS * HD, H, s=(HEADS * HD) ** -0.5))
                self.calls = 0

            def __call__(self, hidden_states, positions):
                self.calls += 1
                T = hidden_states.shape[0]
                assert positions.shape[0] == T, (positions.shape, T)
                qkv, _ = self.qkv(hidden_states)
                q, k, v = qkv.float().split(HD, dim=-1)
                seg = SEGS["v"]
                if self.kind == "mla":
                    q, k = rope(q, positions), rope(k, positions)
                    outs = []
                    for r0 in range(0, T, 1024):
                        r1 = min(T, r0 + 1024)
                        sc = (q[r0:r1] @ k.t()) * HD ** -0.5
                        ti = torch.arange(r0, r1).unsqueeze(1)
                        tj = torch.arange(T).unsqueeze(0)
                        mask = tj > ti
                        if seg is not None:
                            mask = mask | (seg[r0:r1].unsqueeze(1) != seg.unsqueeze(0))
                        sc = sc.masked_fill(mask, float("-inf"))
                        outs.append(torch.softmax(sc, dim=-1) @ v)
                    o = torch.cat(outs)
                else:
                    pk, pq = F.elu(k) + 1, F.elu(q) + 1
                    vv = v * (1.0 + 1e-4 * positions.float().unsqueeze(-1))
                    if seg is None:
                        kv = torch.cumsum(pk.unsqueeze(-1) * vv.unsqueeze(-2), dim=0)
                        z = torch.cumsum(pk, dim=0)
                    else:
                        kv = torch.empty(T, HD, HD)
                        z = torch.empty(T, HD)
                        for sid in seg.unique():
                            idx = (seg == sid).nonzero().squeeze(1)
                            kv[idx] = torch.cumsum(pk[idx].unsqueeze(-1) * vv[idx].unsqueeze(-2), dim=0)
                            z[idx] = torch.cumsum(pk[idx], dim=0)
                    o = (pq.unsqueeze(-2) @ kv).squeeze(-2) / ((pq * z).sum(-1, keepdim=True) + 1e-6)
                o, _ = self.o_proj(o.to(bf))
                return o

        text = Path(a.model).read_text()
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)

        def src_of(node):
            return textwrap.dedent("".join(lines[node.lineno - 1:node.end_lineno]))

        cls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

        def method(c, m):
            hits = [n for n in cls[c].body if isinstance(n, ast.FunctionDef) and n.name == m]
            assert len(hits) == 1, (c, m)
            return src_of(hits[0])

        gate_nodes = [n for n in tree.body if (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_P025_ON" for t in n.targets))
            or (isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "_P025_ON")]
        assert len(gate_nodes) == 2, gate_nodes

        class IntermediateTensors(dict):
            pass

        def _never(*a_, **k_):
            raise AssertionError("image SP-MoE path must not run")

        NS = {"torch": torch, "nn": torch.nn, "hc_expand": hc_expand, "hc_contract": hc_contract,
              "sp_all_gather": _never, "sp_reduce_scatter": _never, "sp_shard": _never,
              "get_pp_group": lambda: PPG, "IntermediateTensors": IntermediateTensors,
              "sequence_parallel_chunk": _never, "tensor_model_parallel_all_gather": _never}
        for n in gate_nodes:
            exec(compile(ast.Module(body=[n], type_ignores=[]), "model.py:gate", "exec"), NS)
        assert NS["_P025_ON"] is True and NS["_p025"] is SP, "the real gate block did not bind the module"

        def get_fn(c, m):
            exec(method(c, m), NS)
            return NS.pop(m)

        LayerFwd = get_fn("Glm5NextDecoderLayer", "forward")
        HcPre = get_fn("Glm5NextDecoderLayer", "hc_pre")
        HcPost = get_fn("Glm5NextDecoderLayer", "hc_post")
        HcFused = get_fn("Glm5NextDecoderLayer", "hc_fused_post_pre")
        ModelFwd = get_fn("Glm5NextModel", "forward")
        MoEFwd = get_fn("Glm5NextMoE", "forward")
        MLPFwd = get_fn("Glm5NextMLP", "forward")

        class MLP:
            forward = MLPFwd

            def __init__(self, inter, reduce_results=True):
                self.gate_up_proj = ColParallel([rnd(H, inter, s=H ** -0.5), rnd(H, inter, s=H ** -0.5)])
                self.down_proj = RowParallel(rnd(inter, H, s=inter ** -0.5))
                self.down_proj.reduce_results = reduce_results
                self.act_fn = SiluAndMul()

            def __call__(self, *a_, **k_):
                return self.forward(*a_, **k_)

        class Gate:
            def __init__(self):
                self.w = rnd(H, E, s=H ** -0.5)

            def __call__(self, x):
                return x.float() @ self.w, None

        class Runner:
            def __init__(self, gate, shared):
                self.gate = gate
                self.shared = shared
                self.w1 = [ColParallel([rnd(H, I_EXP, s=H ** -0.5), rnd(H, I_EXP, s=H ** -0.5)]) for _ in range(E)]
                self.w2 = [RowParallel(rnd(I_EXP, H, s=I_EXP ** -0.5)) for _ in range(E)]
                for w in self.w2:
                    w.reduce_results = False
                self.moe_config = types.SimpleNamespace(skip_final_all_reduce=False, is_sequence_parallel=False,
                                                        tp_size=W, ep_size=1)
                self._fused_output_is_reduced = False
                self.act = SiluAndMul()

            def __call__(self, hidden_states, router_logits):
                logits, _ = self.gate(hidden_states)
                probs = torch.softmax(logits, dim=-1)
                tw, ti = probs.topk(TOPK, dim=-1)
                tw = tw / tw.sum(-1, keepdim=True)
                fused = torch.zeros(hidden_states.shape[0], H, dtype=torch.float32)
                for e in range(E):
                    rows, slot = (ti == e).nonzero(as_tuple=True)
                    if rows.numel() == 0:
                        continue
                    gu, _ = self.w1[e](hidden_states[rows])
                    y, _ = self.w2[e](self.act(gu))
                    fused.index_add_(0, rows, y.float() * tw[rows, slot].unsqueeze(-1))
                fused = (fused * 1.5).to(bf)
                result = self.shared(hidden_states) + fused
                if self.moe_config.skip_final_all_reduce:
                    assert not self._fused_output_is_reduced
                if not self.moe_config.skip_final_all_reduce:
                    result = toy_all_reduce(result)
                return result

        class MoE:
            forward = MoEFwd

            def __init__(self):
                self.is_sequence_parallel = False
                self.gate = Gate()
                self.shared_experts = MLP(I_SHARED, reduce_results=False)
                self.experts = Runner(self.gate, self.shared_experts)

            def __call__(self, *a_, **k_):
                return self.forward(*a_, **k_)

        PRS_ON = {"v": True}
        PRENORM_CALLS = {"split": 0, "full": 0, "rows": []}

        def toy_kernel(x, fn, out_, sq, hidden_size, hc_mult, n_out, n_thr, tile_n, block_m):
            T_ = x.shape[0]
            for r0 in range(0, T_, 256):
                xf = x[r0:r0 + 256].float()
                out_[0, r0:r0 + 256].copy_((xf.unsqueeze(1) * fn.unsqueeze(0)).sum(-1))
                sq[0, r0:r0 + 256].copy_((xf * xf).sum(-1))

        class ToyKObj:
            """HcPrenormGemmTileLangKernel stand-in for the 022 hook: dispatch() always picks block_m=2, kernel() hands
            out the toy bm2 function (the v0.32.0 hook receives the JIT-class instance)."""

            def dispatch(self, *, num_tokens, hc_hidden_size, n_thr=512, tile_n=12, n_splits=1, **kw):
                return types.SimpleNamespace(use_block_m=True, block_m=2, n_splits=n_splits, n_thr=n_thr, tile_n=tile_n)

            def kernel(self, key):
                return toy_kernel

        TOY_K = ToyKObj()

        def prenorm(x2d, fn, out_, sq, hidden_size, hc_mult):
            PRENORM_CALLS["rows"].append(int(x2d.shape[0]))
            if PRS_ON["v"] and PRS.prenorm_rowsplit(TOY_K, x2d, fn, out_, sq, hidden_size, hc_mult):
                PRENORM_CALLS["split"] += 1
                return
            PRENORM_CALLS["full"] += 1
            toy_kernel(x2d, fn, out_, sq, hidden_size, hc_mult, fn.shape[0], 512, 12, 2)

        def mhc_pre_impl(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                         hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=0.0):
            T_, n, Hh = residual.shape
            o_ = torch.empty(1, T_, fn.shape[0])
            s_ = torch.empty(1, T_)
            prenorm(residual.reshape(T_, n * Hh).contiguous(), fn, o_, s_, Hh, n)
            mixes = o_[0] * torch.rsqrt(s_[0].unsqueeze(-1) / (n * Hh) + rms_eps)
            # rev 3: 1/(1+exp(-t)), not torch.sigmoid - CPU torch.sigmoid returns different bits for the same input
            # in its vectorized body vs its scalar tail, so the toy was NOT per-row (row results depended on the call's
            # row count); the rev-3 live check at T = MIN_T + 3 (rank 7's 117-row tail vs prs's split) exposed it.
            # exp / + / / are position-independent (probe: 0 vs 136 mismatches in 1,600 full-vs-shard trials).
            pre = 1.0 / (1.0 + torch.exp(-(mixes[:, :n] * hc_scale[0] + hc_base[:n]))) + hc_pre_eps
            post = 1.0 / (1.0 + torch.exp(-(mixes[:, n:2 * n] * hc_scale[1] + hc_base[n:2 * n]))) * hc_post_mult_value
            cl = mixes[:, 2 * n:].reshape(T_, n, n) * hc_scale[2] + hc_base[2 * n:].reshape(1, n, n)
            comb = torch.softmax(cl, dim=-1) + hc_sinkhorn_eps
            comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
            for _ in range(sinkhorn_repeat - 1):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
            rf = residual.float()
            li = pre[:, 0:1] * rf[:, 0]
            for i in range(1, n):
                li = li + pre[:, i:i + 1] * rf[:, i]
            if norm_weight is not None:
                li = li * torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps) * norm_weight.float()
            return post.reshape(T_, n, 1), comb.reshape(T_, n, n), li.to(bf)

        def mhc_post_impl(x, residual, post, comb):
            T_, n, Hh = residual.shape
            rf = residual.float()
            cf = comb.float()
            outs = []
            for j in range(n):
                acc = post.reshape(T_, n)[:, j:j + 1].float() * x.float()
                for i in range(n):
                    acc = acc + cf[:, i, j:j + 1] * rf[:, i]
                outs.append(acc)
            return torch.stack(outs, dim=1).to(residual.dtype)

        class PreOp:
            def __call__(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                         hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=0.0):
                return mhc_pre_impl(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                    hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps)

        class PostOp:
            def __call__(self, x, residual, post, comb):
                return mhc_post_impl(x, residual, post, comb)

        class FusedOp:
            def __call__(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
                         hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1,
                         norm_weight=None, norm_eps=0.0):
                assert n_splits == 1
                rc = mhc_post_impl(x, residual, post_layer_mix, comb_res_mix)
                p, c, li = mhc_pre_impl(rc, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                        hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps)
                return rc, p, c, li

        class Layer:
            forward = LayerFwd
            hc_pre = HcPre
            hc_post = HcPost
            hc_fused_post_pre = HcFused

            def __init__(self, idx, mtp=False):
                self.layer_idx = idx
                self.num_hidden_layers = L
                self.is_mtp_layer = mtp
                self.mhc = True
                self.is_sequence_parallel = False
                self.rms_norm_eps = 1e-5
                self.layer_kind = "mla" if (idx == 3 or mtp) else "kda"     # as model.py: config.is_kda_layer()
                self.self_attn = Attn(self.layer_kind)
                self.mlp = MoE() if (idx >= 3 or mtp) else MLP(I_DENSE)
                self._mlp_is_moe = isinstance(self.mlp, MoE)
                self.input_layernorm = RMSNorm(1.0 + 0.1 * rnd(H))
                self.post_attention_layernorm = RMSNorm(1.0 + 0.1 * rnd(H))
                if not mtp:
                    self.n = NHC
                    mix = (2 + NHC) * NHC
                    self.hc_eps = 1e-6
                    self.mhc_post_mult_value = 2.0
                    self.mhc_sinkhorn_iterations = 4
                    self.hc_attn_fn = rnd(mix, NHC * H, s=0.05)
                    self.hc_attn_base = rnd(mix, s=0.1)
                    self.hc_attn_scale = torch.rand(3, generator=gen) + 0.5
                    self.hc_ffn_fn = rnd(mix, NHC * H, s=0.05)
                    self.hc_ffn_base = rnd(mix, s=0.1)
                    self.hc_ffn_scale = torch.rand(3, generator=gen) + 0.5
                    self.mhc_pre_op = PreOp()
                    self.mhc_post_op = PostOp()
                    self.mhc_fused_post_pre_op = FusedOp()

            def __call__(self, *a_, **k_):
                return self.forward(*a_, **k_)

        class Model:
            forward = ModelFwd

            def __init__(self):
                self.config = types.SimpleNamespace(hidden_size=H)
                self.is_sequence_parallel = False
                self.layers = [Layer(i) for i in range(L)]
                self._active_layers = self.layers
                self.start_layer, self.end_layer = 0, L          # v0.31+ loop: enumerate(self._active_layers, start=...)
                self.aux_hidden_state_layers = ()                # EagleModelMixin default (MTP: no aux taps)
                self.norm = RMSNorm(1.0 + 0.1 * rnd(H))
                emb = rnd(VOCAB, H).to(bf)
                vl = VOCAB // W
                self.emb_local = emb[rank * vl:(rank + 1) * vl]
                self.vstart, self.vend = rank * vl, (rank + 1) * vl

            def embed_input_ids(self, ids):
                m = (ids >= self.vstart) & (ids < self.vend)
                loc = torch.where(m, ids - self.vstart, torch.zeros_like(ids))
                return toy_all_reduce(self.emb_local[loc] * m.unsqueeze(-1).to(bf))

            def __call__(self, *a_, **k_):
                return self.forward(*a_, **k_)

        model = Model()
        mtp = Layer(L, mtp=True)

        def flags_default():
            bad = []
            for ly in model.layers + [mtp]:
                if ly.self_attn.o_proj.reduce_results is not True:
                    bad.append(("o_proj", ly.layer_idx))
                if ly._mlp_is_moe:
                    if ly.mlp.experts.moe_config.skip_final_all_reduce is not False:
                        bad.append(("moe", ly.layer_idx))
                elif ly.mlp.down_proj.reduce_results is not True:
                    bad.append(("down", ly.layer_idx))
            return bad

        if a.bootonly:
            # BOOT-ONLY: the first eligible begin_forward (envelope + preflight + boot self-test + prs neutralisation),
            # as in the profile run, on the toy model's live layers with the real 022 module. Every rank records its
            # verdict and then WAITS for all ranks' verdict files (no early exit: a peer's exit could unblock a gloo
            # wait and turn a hang into an error) - a rank that never finishes is a HANG for the launcher.
            bmode, brs, bT = a.bootonly.split(",")
            SP._reset_for_test(mode=bmode, min_t=MIN_T, rs_mode=brs)
            NS["_P025_ON"] = True
            out["prs_booted_before"] = PRS._STATE.world is not None
            t0 = time.time()
            try:
                hb = SP.begin_forward(int(bT), H, layers=model._active_layers, is_sequence_parallel=False)
                out["handle"] = hb is not None
                out["prs_min_t_in_region"] = PRS._STATE.min_t
                SP.end_forward(hb)
                out["report"] = SP._STATE.report
            except Exception:  # noqa: BLE001
                out["boot_exc"] = traceback.format_exc()
            out["boot_s"] = round(time.time() - t0, 2)
            out["prs_booted"] = PRS._STATE.world is not None
            out["flags_default"] = not flags_default()
            out["coll"] = dict(COLL, devgroup=DEVG["n"])
            Path(a.out, f"rank{rank}.json").write_text(json.dumps(out, default=str))
            end = time.time() + a.timeout
            while time.time() < end and len(list(Path(a.out).glob("rank*.json"))) < W:
                time.sleep(0.5)
            os._exit(0)

        def run(T, cfg, ar_mode, prs_on=True, seg=False, pos0=0):
            AR_MODE["v"] = ar_mode
            PRS_ON["v"] = prs_on
            g2 = torch.Generator().manual_seed(1000 + T)
            ids = torch.randint(0, VOCAB, (T,), generator=g2)
            positions = torch.arange(pos0, pos0 + T)
            SEGS["v"] = (torch.arange(T) >= T // 3).long() if seg else None
            if cfg == "image":
                NS["_P025_ON"] = False
            else:
                NS["_P025_ON"] = True
                mode, rs = cfg
                if SP._STATE.mode != mode or SP._STATE.rs_mode != rs:
                    SP._reset_for_test(mode=mode, min_t=MIN_T, rs_mode=rs)
                    # pre-boot with the real layers so the boot self-test's LIVE-weights checks run (checks 2 and 3).
                    p2p0, os0 = COLL["p2p"], COLL["oneshot"]
                    hb = SP.begin_forward(MIN_T, H, layers=model._active_layers, is_sequence_parallel=False)
                    SP.end_forward(hb)
                    BOOTS[f"{mode},{rs}"] = {"report": SP._STATE.report, "p2p": COLL["p2p"] - p2p0,
                                             "oneshot": COLL["oneshot"] - os0}
            for k_ in COLL:
                COLL[k_] = 0
            PRENORM_CALLS.update(split=0, full=0, rows=[])
            hs = model(ids, positions, None)
            coll = dict(COLL)
            pcalls = dict(PRENORM_CALLS)
            mtp_out = mtp(positions=positions, hidden_states=hs, residual=None)
            NS["_P025_ON"] = True
            return hs, mtp_out[0], coll, pcalls

        def h(t):
            return hashlib.sha256(t.float().contiguous().numpy().tobytes()).hexdigest()[:16]

        def mism(x, y):
            if x.shape != y.shape:
                return f"shape {tuple(x.shape)} vs {tuple(y.shape)}"
            return int((x.contiguous().view(torch.int16) != y.contiguous().view(torch.int16)).sum())

        def rel(x, y):
            if x.shape != y.shape:
                return float("inf")
            return float((x.float() - y.float()).abs().amax() / y.float().abs().amax().clamp_min(1e-12))

        tlist = [int(t) for t in a.tlist.split(",")]
        res_exact, res_ref_exact, res_nccl_ref, rel_nccl, inert, coll_bad = [], [], [], [], [], []
        flags_bad, active_bad, prs_state_bad, prs_split_in_sp, shard_rows_bad, mtp_bad = [], [], [], [], [], []
        canon_fwd_p2p = []
        for T in tlist:
            seg = T >= 2048
            pos0 = 4096 if T == 3853 else 0
            img_c, mtp_img_c, coll_img, pc_img = run(T, "image", "canonical", seg=seg, pos0=pos0)
            out["hash"][f"img_c_{T}"] = h(img_c)
            sp_c, mtp_sp_c, coll_sp, pc_sp = run(T, ("sp", "canonical"), "canonical", seg=seg, pos0=pos0)
            out["hash"][f"sp_c_{T}"] = h(sp_c)
            if T < MIN_T:
                inert.append((T, mism(sp_c, img_c), coll_sp["rs"]))
            else:
                res_exact.append((T, mism(sp_c, img_c)))
                canon_fwd_p2p.append((T, coll_sp["p2p"]))
                if coll_sp["ar"] != 1:
                    coll_bad.append((T, "ar", coll_sp["ar"]))
                plan = SP.plan_rows(T, W)
                if any(r != plan.S for r in pc_sp["rows"]):
                    shard_rows_bad.append((T, sorted(set(pc_sp["rows"]))[:4], plan.S))
                if pc_sp["split"]:
                    prs_split_in_sp.append((T, pc_sp["split"]))
            if mism(mtp_sp_c, mtp_img_c) != 0:
                mtp_bad.append((T, mism(mtp_sp_c, mtp_img_c)))
            rf_c, _, _, _ = run(T, ("ref", "canonical"), "canonical", seg=seg, pos0=pos0)
            out["hash"][f"ref_c_{T}"] = h(rf_c)
            if T >= MIN_T:
                res_ref_exact.append((T, mism(rf_c, img_c)))
            img_g, _, _, _ = run(T, "image", "gloo", seg=seg, pos0=pos0)
            floor = rel(img_c, img_g)
            sp_n, _, coll_spn, _ = run(T, ("sp", "nccl"), "gloo", seg=seg, pos0=pos0)
            rf_n, _, _, _ = run(T, ("ref", "nccl"), "gloo", seg=seg, pos0=pos0)
            out["hash"][f"sp_n_{T}"] = h(sp_n)
            out["hash"][f"ref_n_{T}"] = h(rf_n)
            if T >= MIN_T:
                res_nccl_ref.append((T, mism(sp_n, rf_n)))
                rows_dev = int(((sp_n.float() - img_g.float()).abs().amax(-1)
                                > 0.05 * img_g.float().abs().amax()).sum())
                rel_nccl.append((T, round(rel(sp_n, img_g), 5), round(floor, 5), rows_dev))
                if coll_spn["rs"] != 2 * L or coll_spn["ag"] != 2 * L + 1:
                    coll_bad.append((T, "nccl rs/ag", coll_spn["rs"], coll_spn["ag"]))
            fb = flags_default()
            if fb:
                flags_bad.append((T, fb[:3]))
            if SP.sp_active():
                active_bad.append(T)
            if PRS._STATE.min_t not in (None, MIN_T):
                prs_state_bad.append((T, PRS._STATE.min_t))

        put("E1 sp(canonical RS) == image path (canonical AR) BITWISE for every T >= MIN_T (real model.py + module)",
            all(m == 0 for _, m in res_exact), str(res_exact))
        put("E2 ref(canonical) == image path BITWISE (T >= MIN_T)", all(m == 0 for _, m in res_ref_exact),
            str(res_ref_exact))
        put("E3 sp(nccl-RS) == ref(nccl-RS) BITWISE (sp == ref with the same RS)",
            all(m == 0 for _, m in res_nccl_ref), str(res_nccl_ref))
        put("E4 sp(nccl-RS) vs image(native AR) within the reduction-order A/A floor (rel <= max(3e-2, 3x floor))",
            all(r <= max(3e-2, 3 * fl) for _, r, fl, _ in rel_nccl), str(rel_nccl))
        put("E5 T < MIN_T: inert, BITWISE the image path, no reduce-scatter",
            all(m == 0 and n == 0 for _, m, n in inert), str(inert))
        put("E6 collectives: sp forward = 1 AR (embedding) + 2L RS + (2L+1) AG", not coll_bad, str(coll_bad[:4]))
        put("E7 every mHC pre-norm call inside sp runs on exactly S rows", not shard_rows_bad, str(shard_rows_bad[:3]))
        put("E8 prs (real 022 module) never splits inside an sp region", not prs_split_in_sp, str(prs_split_in_sp))
        put("E9 after every forward: suppression flags at defaults, sp inactive, prs MIN_T restored",
            not flags_bad and not active_bad and not prs_state_bad,
            f"{flags_bad[:2]} {active_bad[:2]} {prs_state_bad[:2]}")
        put("E10 MTP layer after the forward: BITWISE identical sp vs image (canonical)", not mtp_bad, str(mtp_bad[:3]))
        put("E11 prs (real 022 module) booted + split in the image runs (gate off), never inside an sp region",
            PRS._STATE.world is not None and not prs_split_in_sp,
            f"world={getattr(PRS._STATE, 'world', None) is not None} calls={PRS.stats().get('calls')}")
        put("E12 sp boot self-test ran on the production _TPWorld (gloo) and passed",
            SP._STATE.booted and SP._STATE.report is not None, str(SP._STATE.report)[:200])
        put("E14 boot self-test LIVE-weights mHC shard-invariance + suppression handles ran and passed",
            bool(SP._STATE.report and SP._STATE.report.get("live") and SP._STATE.report["live"]["ok"]),
            str(SP._STATE.report.get("live") if SP._STATE.report else None))
        # rev 3
        put("E15 sp never touched the torch-c10d device group; the canonical RS ran over the pynccl send/recv path",
            DEVG["n"] == 0 and bool(canon_fwd_p2p) and all(n > 0 for _, n in canon_fwd_p2p),
            f"device_group uses={DEVG['n']} canonical-forward p2p sends={canon_fwd_p2p[:4]}")
        want_layers = [[0, "kda", False, NHC], [3, "mla", True, NHC]]
        want_targets = ["L0.o_proj", "L0.down_proj", "L3.o_proj"]
        e16_bad = []
        for key, b in sorted(BOOTS.items()):
            rep_ = b["report"] or {}
            lv, c2 = rep_.get("live") or {}, rep_.get("check2") or {}
            if (lv.get("layers") != want_layers or lv.get("t") != [MIN_T, MIN_T + 3]
                    or c2.get("targets") != want_targets or c2.get("oneshot") is not True
                    or c2.get("bitwise") != key.endswith(",canonical") or not b["oneshot"]):
                e16_bad.append((key, lv.get("layers"), lv.get("t"), c2, b["oneshot"]))
        put("E16 boot self-test live checks on layer 0 (KDA+dense) and layer 3 (MLA+MoE) at T=MIN_T and MIN_T+3 + "
            "check 2 (suppressed o_proj/down_proj partial: sp RS+AG vs the patch-012 one-shot, bitwise for canonical)",
            len(BOOTS) == 4 and not e16_bad, f"boots={sorted(BOOTS)} bad={e16_bad[:2]}")
        put("E17 canonical check gated on RS=canonical: RS=nccl boots issue no P2P, RS=canonical boots do",
            all((b["p2p"] == 0) if k.endswith(",nccl") else (b["p2p"] > 0) for k, b in BOOTS.items()),
            str({k: b["p2p"] for k, b in sorted(BOOTS.items())}))

        # E18 (v0.32.0) aux_hidden_state_layers set (EAGLE-3 taps): the forward is never sharded - the image path runs,
        # BITWISE, no reduce-scatter, the aux list is returned as upstream does, begin_forward counts it
        SP._reset_for_test(mode="sp", min_t=MIN_T, rs_mode="canonical")
        NS["_P025_ON"] = True
        model.aux_hidden_state_layers = (2, 4)
        model._aux_hidden_state = lambda hs, res, post, comb: hs          # the real tap needs the widened streams; any tensor will do
        T18 = 2048
        ids18 = torch.randint(0, VOCAB, (T18,), generator=torch.Generator().manual_seed(18))
        pos18 = torch.arange(T18)
        SEGS["v"] = None
        AR_MODE["v"] = "canonical"
        for k_ in COLL:
            COLL[k_] = 0
        out_aux = model(ids18, pos18, None)
        coll_aux = dict(COLL)
        aux_skips = SP.stats()["aux_skips"]
        model.aux_hidden_state_layers = ()
        NS["_P025_ON"] = False
        out_img = model(ids18, pos18, None)
        NS["_P025_ON"] = True
        put("E18 aux_hidden_state_layers set: never sharded (begin_forward -> None, counted), no reduce-scatter, BITWISE the "
            "image path, aux list returned (EAGLE-3 taps would otherwise see this rank's rows only)",
            isinstance(out_aux, tuple) and len(out_aux[1]) == 2 and aux_skips == 1 and coll_aux["rs"] == 0
            and mism(out_aux[0], out_img) == 0 and not SP.sp_active(),
            f"tuple={isinstance(out_aux, tuple)} aux_skips={aux_skips} rs={coll_aux['rs']}")

        # E13 exception inside the sp region: flags restored, region closed, prs restored, next small forward OK
        SP._reset_for_test(mode="sp", min_t=MIN_T, rs_mode="nccl")
        NS["_P025_ON"] = True
        boom = {"armed": True}
        orig_call = Attn.__call__

        def exploding(self, hidden_states, positions):
            if boom["armed"] and self.kind == "mla":
                raise RuntimeError("planted failure inside attention")
            return orig_call(self, hidden_states, positions)

        Attn.__call__ = exploding
        SEGS["v"] = None
        raised = False
        try:
            model(torch.randint(0, VOCAB, (2048,)), torch.arange(2048), None)
        except RuntimeError as e:
            raised = "planted" in str(e)
        Attn.__call__ = orig_call
        ok_after = not SP.sp_active() and not flags_default() and PRS._STATE.min_t in (None, MIN_T)
        small_ok = True
        try:
            model(torch.randint(0, VOCAB, (100,)), torch.arange(100), None)
        except Exception as e:  # noqa: BLE001
            small_ok = f"{type(e).__name__}: {e}"
        put("E13 exception inside the sp region: flags/region/prs restored, next forward OK",
            raised and ok_after and small_ok is True, f"raised={raised} after={ok_after} small={small_ok}")

    except Exception:  # noqa: BLE001
        out["fatal"] = traceback.format_exc()
    Path(a.out, f"rank{rank}.json").write_text(json.dumps(out))
    try:
        dist.barrier()
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    return 0


# ----------------------------------------------------------------------------------------------------------------
# sweep: base + every mutant; each REQUIRED mutant must be caught
# ----------------------------------------------------------------------------------------------------------------
def _run_e2e(patches, vsrc, work, tag, sp_module, port, timeout, tlist):
    cmd = [sys.executable, __file__, "launch", "--patches", str(patches), "--vsrc", str(vsrc),
           "--work", str(work / tag), "--tag", tag, "--port", str(port), "--timeout", str(timeout), "--tlist", tlist]
    if sp_module:
        cmd += ["--sp-module", str(sp_module)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    boot_raised = "BOOT SELF-TEST FAILED" in (p.stdout + p.stderr) \
        or any("BOOT SELF-TEST FAILED" in (Path(work / tag / f"out_{tag}" / f"rank{r}.json").read_text()
               if (work / tag / f"out_{tag}" / f"rank{r}.json").exists() else "") for r in range(8))
    passed = p.returncode == 0
    return passed, boot_raised, p.stdout


def _run_builder_bc(patches, vsrc, work, tag, sp_module):
    """p025_cpu_check.py module-logic sections B/C against a (possibly mutant) module via P025_MODULE_OVERRIDE."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if sp_module:
        env["P025_MODULE_OVERRIDE"] = str(sp_module)
    cmd = [sys.executable, str(patches / "files" / "p025_cpu_check.py"), "--vllm-root", str(vsrc),
           "--work", str(work / f"{tag}-bc"), "--world", "8", "--only", "BC"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode == 0, p.stdout


def _run_bootonly(patches, vsrc, work, tag, sp_module, port, timeout, barrier_s, spec):
    """One boot-only launch (8 gloo ranks); returns its summary.json (classification + per-rank verdicts)."""
    cmd = [sys.executable, __file__, "launch", "--patches", str(patches), "--vsrc", str(vsrc),
           "--work", str(work / tag), "--tag", tag, "--port", str(port), "--timeout", str(timeout),
           "--bootonly", spec, "--barrier-s", str(barrier_s), "--tlist", "8192"]
    if sp_module:
        cmd += ["--sp-module", str(sp_module)]
    subprocess.run(cmd, capture_output=True, text=True)
    f = work / tag / f"out_{tag}" / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {"kind": "NO-SUMMARY", "ranks": [], "wall_s": None}


def hang_sweep(a, patches, vsrc, work, src, port):
    """Boot-only base (must be OK) + every HANG mutant (must be a CLEAN-RAISE on all ranks) for RS=nccl/canonical."""
    mut_dir = work / "mut"
    mut_dir.mkdir(exist_ok=True)
    rows, ok = [], True
    for rs in ("nccl", "canonical"):
        port += 1
        s = _run_bootonly(patches, vsrc, work, f"hb-base-{rs}", None, port, a.hang_timeout, a.barrier_s,
                          f"sp,{rs},8192")
        r0 = (s.get("ranks") or [{}])[0]
        good = s["kind"] == "OK" and all(x.get("prs_booted") for x in s.get("ranks", []))
        ok &= good
        rows.append((f"base ({rs})", s["kind"], "OK", good,
                     f"wall {s.get('wall_s')}s, prs booted inside the sp boot on all ranks: "
                     f"{all(x.get('prs_booted') for x in s.get('ranks', []))}, p2p={r0.get('p2p')}, "
                     f"device-group uses={r0.get('devgroup')}"))
    for name, (old, new) in HANG_MUTANTS.items():
        if src.count(old) != 1:
            rows.append((name, "ANCHOR", "CLEAN-RAISE", False, f"anchor matches {src.count(old)}x"))
            ok = False
            continue
        mp = mut_dir / f"{name}.py"
        mp.write_text(src.replace(old, new, 1))
        for rs in ("nccl", "canonical"):
            port += 1
            want = "OK" if (name in HANG_CANONICAL_ONLY and rs == "nccl") else "CLEAN-RAISE"
            s = _run_bootonly(patches, vsrc, work, f"{name}-{rs}", mp, port, a.hang_timeout, a.barrier_s,
                              f"sp,{rs},8192")
            good = s["kind"] == want
            darwin_note = ""
            if (not good and sys.platform == "darwin" and name == "h_make_world_r7" and s["kind"] == "HANG"
                    and all(x.get("verdict") == "RAISED" for x in s.get("ranks", []) if x.get("rank") in (0, 7))):
                # macOS torch/gloo: dist.monitored_barrier blocks the NON-zero ranks until rank 0 releases them and does not time
                # them out when rank 0 raises (a plain-torch 8-process probe reproduces it, no vLLM / patch code involved); on
                # Linux gloo the other 7 ranks raise after the barrier (21.8 s with the 20 s test barrier, measured on node1).
                # Rank 0 (the monitor) and rank 7 raise cleanly here; accepted as a documented platform exception.
                good = True
                darwin_note = " [macOS exception: ranks 1-6 stay blocked in gloo monitored_barrier (plain torch reproduces it)]"
            ok &= good
            why = ""
            raised = [x for x in s.get("ranks", []) if x.get("verdict") == "RAISED"]
            if raised:
                why = raised[0].get("why", "")[:150]
            why += darwin_note
            slow = max((x.get("boot_s") or 0.0) for x in s.get("ranks", [])) if s.get("ranks") else 0.0
            bounded = " [bounded: raised after the monitored barrier]" if slow >= a.barrier_s else ""
            rows.append((f"{name} ({rs})", s["kind"], want, good, f"wall {s.get('wall_s')}s{bounded}; {why}"))
    print("\n== HANG mutants (boot-only, 8 gloo ranks): a rank-local failure at each self-test stage must end in a "
          "CLEAN-RAISE on ALL ranks (never a hang / a partial raise)")
    for name, kind, want, good, info in rows:
        print(f"   {'OK ' if good else 'BAD'} {name:38s} {kind:12s} (want {want}) {info}")
    n_hang = sum(1 for r in rows if not r[0].startswith("base"))
    print(f"\n   hang mutants: {n_hang} runs, {sum(1 for r in rows if not r[0].startswith('base') and r[3])} as "
          f"required; base boot-only: {', '.join(r[1] for r in rows if r[0].startswith('base'))}")
    return ok, port


def sweep(a):
    patches = Path(a.patches)
    vsrc = Path(a.vsrc)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    src = (patches / "files" / "gb10_prefill_sp.py").read_text()
    mut_dir = work / "mut"
    mut_dir.mkdir(exist_ok=True)
    port = a.port

    hang_ok = True
    if a.hang in ("yes", "only"):
        hang_ok, port = hang_sweep(a, patches, vsrc, work, src, port + 100)
        if a.hang == "only":
            print("HANG SWEEP OK" if hang_ok else "HANG SWEEP FAILED")
            return 0 if hang_ok else 1

    print("== base (unmutated): e2e + builder B/C must PASS")
    base_e2e, _, base_out = _run_e2e(patches, vsrc, work, "base", None, a.port, a.timeout, a.tlist)
    base_bc, base_bc_out = _run_builder_bc(patches, vsrc, work, "base", None)
    base_last = base_out.strip().splitlines()[-1] if base_out.strip() else ""
    print(f"   base e2e {'PASS' if base_e2e else 'FAIL'} ({base_last}); builder B/C {'PASS' if base_bc else 'FAIL'}")

    results = []
    for name, (old, new) in MUTANTS.items():
        port += 1
        if src.count(old) != 1:
            results.append((name, "ANCHOR", src.count(old), False))
            continue
        mp = mut_dir / f"{name}.py"
        mp.write_text(src.replace(old, new, 1))
        e2e_pass, boot_raised, _ = _run_e2e(patches, vsrc, work, name, mp, port, a.timeout, a.tlist)
        bc_pass, _ = _run_builder_bc(patches, vsrc, work, name, mp)
        caught = (not e2e_pass) or (not bc_pass) or boot_raised
        results.append((name, "caught" if caught else "passed", f"e2e={'P' if e2e_pass else 'F'}"
                        f" bc={'P' if bc_pass else 'F'} boot_raised={boot_raised}", caught))

    print("\n== mutant table (each REQUIRED mutant must be caught = fail the suite or raise the boot self-test)")
    required_missed = []
    for name, status, info, caught in results:
        req = name not in BENIGN_ON_CPU
        tag = "REQUIRED" if req else "benign/gpu-only"
        mark = "OK " if (caught or (not req and not caught)) else "MISS"
        if req and not caught:
            required_missed.append(name)
        print(f"   {mark} {name:28s} {status:7s} [{tag}] {info}")
    base_ok = base_e2e and base_bc
    ok = base_ok and not required_missed and hang_ok
    print(f"\n   base: e2e {'PASS' if base_e2e else 'FAIL'}, builder B/C {'PASS' if base_bc else 'FAIL'}; "
          f"{len(MUTANTS) - len(BENIGN_ON_CPU)} required mutants, {len(required_missed)} missed "
          f"{required_missed}; {len(BENIGN_ON_CPU)} documented not-catchable-on-CPU {sorted(BENIGN_ON_CPU)}; "
          f"hang mutants {'OK' if hang_ok else 'FAILED'}{'' if a.hang != 'no' else ' (not run: --hang no)'}")
    print("SWEEP OK" if ok else "SWEEP FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "worker", "sweep"])
    ap.add_argument("--patches", default=str(HERE.parent))
    ap.add_argument("--vsrc", default=os.environ.get("VLLM_TREE", "vllm-v0.32.0/vllm"))
    ap.add_argument("--work")
    ap.add_argument("--sp-module", default=None)
    ap.add_argument("--prs-module")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--port", type=int, default=29631)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--tlist", default="1023,1024,1027,3840,3853,8192")
    ap.add_argument("--bootonly", default="", help="MODE,RS,T: boot-only launch (see the module docstring)")
    ap.add_argument("--barrier-s", type=float, default=0.0, help="tests: override SELFTEST_BARRIER_S")
    ap.add_argument("--hang", choices=["yes", "no", "only"], default="yes", help="sweep: run the HANG mutants")
    ap.add_argument("--hang-timeout", type=int, default=240, help="sweep: per boot-only launch deadline (s)")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if a.cmd == "worker":
        sys.exit(worker(a))
    if a.cmd == "sweep":
        if not a.barrier_s:
            a.barrier_s = 20.0
        if not a.work:
            a.work = os.environ.get("TMPDIR", "/tmp") + "/p025-e2e-sweep"
        sys.exit(sweep(a))
    if not a.work:
        a.work = os.environ.get("TMPDIR", "/tmp") + "/p025-e2e"
    rc, _ = launch(a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
