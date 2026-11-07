#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p025_gpu_selftest.py - patch 025 (VLLM_GB10_MHC_SP, mode 'sp'/'ref') on-device checks on the v0.32.0 image's own vLLM +
TileLang (v0.32.0 port: the kernels are the JIT-class kernels of kernels/mhc/tilelang_kernels.py; the model is
models/glm5next/common/model.py). WRITTEN, NOT RUN in the authoring session (no GPU / cluster access).

WHAT A SINGLE IDLE RANK CONTAINER CAN PROVE (this harness), and WHAT IT CANNOT:
  * CAN (runs here): (K) the real mHC TileLang kernels are PER-ROW / SHARD-INVARIANT on the device, on ZERO-PADDED
    S-row shards exactly as sp runs them - mhc_pre_tilelang, mhc_fused_post_pre_tilelang AND mhc_post_tilelang (+
    hc_contract) on an even-start S-row shard (S = 2*ceil(T/16)) are BITWISE equal to the full-T call sliced, incl.
    the uneven tail shard. This is the numerics claim sp shares with patch 022 (same kernels, even shard starts),
    re-proved at the exact sp shard shapes for all three mHC ops the boot self-test checks on the live weights. (D)
    the module's collective data flow (shard -> all-gather -> reduce-scatter -> all-gather) reconstructs an
    INDEPENDENT fp32 reference over 8 DISTINCT per-rank partials and is bit-identical across the 8 shards on GPU
    tensors; canonical reduce-scatter == the rank-order fp32 sum, bitwise. (G) CUDA-graph safety: during capture
    begin_forward / the hook are ineligible (no collective, no split) and MUST NOT raise.
  * CANNOT (needs the live 8-rank boot): the FULL sp vs ref per-layer comparison. Attention is head-sharded and the
    MoE is expert-sharded, so one rank holds only its own heads / experts - the all-reduce the RS replaces cannot be
    reproduced on one GPU. That end-to-end check is the module's boot self-test (collective, on all 8 ranks: it
    reconstructs the reduction, checks the live-weights mHC shard-invariance + the suppression handles, and asserts
    the gathered result is bit-identical across ranks, fail-closed) plus the CPU end-to-end files/p025_e2e_check.py
    (the real patched model.py functions on 8 gloo ranks, with the wiring mutants as negative controls), plus the
    operator A/B: boot mode=ref and mode=sp and compare logits (sp == ref within the A/A floor, or bitwise with
    canonical RS + the determinism patch 023), and ref vs the previous base for the pure reduction-order delta. See PATCHES.md
    (patch 025).

Run INSIDE the CURRENT rank container while the server is idle (the 025 gate is NOT set there; nothing on disk
changes; no NCCL, no network; ~1-2 min). The harness pops VLLM_GB10_MHC_PRS and sets VLLM_USE_DEEP_GEMM=0 before it
imports tilelang, so the patch-022 prenorm hook (which would reach an uninitialised TP group in a bare docker exec)
stays inert and the prenorm takes the bm2 path, not DeepGEMM:

  ssh node1 'docker exec <rank-0 container> python3 /opt/recipe-patches/files/p025_gpu_selftest.py --out /tmp/p025-gpu.json'

It reads the INSTALLED vllm/models/glm5next/common/model.py only to record its sha; it does NOT need to patch it to run
K/D/G (those exercise the TileLang mHC ops and vllm/_gb10_prefill_sp.py directly). The TileLang kernels compile on
first use (cached under TILELANG_CACHE_DIR); the mHC class kernels are warmed lazily here (no JIT warmup registry in a
bare `docker exec`).

CPU validation of this harness (no GPU, no TileLang; fake row-local kernels):
  CUDA_VISIBLE_DEVICES= python3 files/p025_gpu_selftest.py --emulate
Exit: 0 PASS | 1 FAIL | 2 setup error | 3 refused (server busy).
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
import traceback
import types
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD_NAME = "vllm._gb10_prefill_sp"
T_MAIN = (1024, 1027, 2048, 3840, 3853, 8192)
HC_MULT = 4
HIDDEN = 4096
STATUS_URL = os.environ.get("STATUS_URL", "http://127.0.0.1:8000/metrics")   # the API server's /metrics


def log(*a):
    print(*a, flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(sh_path, name):
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", Path(sh_path).read_text(), re.M)
    return m.group(1) if m else None


def server_busy(url, timeout=3.0):
    try:
        t = urllib.request.urlopen(url, timeout=timeout).read().decode()
    except Exception:  # noqa: BLE001 - no server reachable: fine, we are not perturbing it
        return False
    tot = 0.0
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        for m in re.finditer(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", t, re.M):
            tot += float(m.group(1))
    return tot > 0


def load_module(vllm_root, emulate):
    """Load vllm/_gb10_prefill_sp.py. On a real install it binds to the real vllm; under --emulate a stub vllm."""
    if emulate:
        for k in [k for k in list(sys.modules) if k == "vllm" or k.startswith("vllm.")]:
            del sys.modules[k]
        v = types.ModuleType("vllm"); v.__path__ = []; sys.modules["vllm"] = v
        for name in ("vllm.model_executor", "vllm.model_executor.layers"):
            m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m
        mhc = types.ModuleType("vllm.model_executor.layers.mhc")
        mhc.hc_expand = lambda x, n: x.unsqueeze(1).expand(-1, n, -1).contiguous()
        mhc.hc_contract = lambda x, n: x.mean(1)
        sys.modules["vllm.model_executor.layers.mhc"] = mhc
    spec = importlib.util.spec_from_file_location(MOD_NAME, HERE / "gb10_prefill_sp.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[MOD_NAME] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------------------------------
# K. mHC kernel shard-invariance on the real TileLang ops (GPU only)
# ---------------------------------------------------------------------------------------------------------------------
def _biteq(torch, a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return bool(torch.equal(a.contiguous().view(iv), b.contiguous().view(iv)))


def section_k(M, torch, device, res):
    """Per-row / shard-invariance of the three live mHC ops the boot self-test checks, on ZERO-PADDED S-row shards
    (exactly how sp feeds them): mhc_pre, mhc_fused_post_pre and mhc_post(+contract). shard[:real] == full[lo:lo+real]
    bitwise, for the sp shard sizes S = 2*ceil(T/16) incl. the uneven tail."""
    import importlib
    tl = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang")
    from vllm.model_executor.layers.mhc import hc_contract
    n_out = (2 + HC_MULT) * HC_MULT
    eps = (1e-5, 1e-6, 1e-6, 2.0, 20)       # rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    bad_pre, bad_fused, bad_post = [], [], []

    def pad_shard(full, S, lo, real):
        sh = torch.zeros((S, *full.shape[1:]), dtype=full.dtype, device=full.device)
        sh[:real] = full[lo:lo + real]
        return sh

    for T in T_MAIN:
        plan = M.plan_rows(T, 8)
        S = plan.S
        g = torch.Generator(device="cpu").manual_seed(T)
        residual = (torch.randn(T, HC_MULT, HIDDEN, generator=g) * 3).to(torch.bfloat16).to(device)
        x = (torch.randn(T, HIDDEN, generator=g) * 2).to(torch.bfloat16).to(device)
        fn_a = (torch.randn(n_out, HC_MULT * HIDDEN, generator=g) * 0.02).to(device)
        fn_f = (torch.randn(n_out, HC_MULT * HIDDEN, generator=g) * 0.02).to(device)
        scale = (torch.rand(3, generator=g) + 0.5).to(device)
        base = (torch.randn(n_out, generator=g) * 0.1).to(device)
        nw = (torch.rand(HIDDEN, generator=g) + 0.5).to(torch.bfloat16).to(device)
        # full-T calls
        pre_full = tl.mhc_pre_tilelang(residual, fn_a, scale, base, *eps, 1, nw, 1e-5)
        post_f, comb_f, _ = pre_full
        fused_full = tl.mhc_fused_post_pre_tilelang(x, residual, post_f, comb_f, fn_f, scale, base, *eps, 1, 1,
                                                    nw, 1e-5)
        post_out_full = hc_contract(tl.mhc_post_tilelang(x, residual, post_f, comb_f), HC_MULT)
        for r in range(8):
            real = plan.real_rows(r)
            if not real:
                continue
            lo = plan.start(r)
            rs = pad_shard(residual, S, lo, real)
            xs = pad_shard(x, S, lo, real)
            ps = pad_shard(post_f, S, lo, real)
            cs = pad_shard(comb_f, S, lo, real)
            pre_sh = tl.mhc_pre_tilelang(rs, fn_a, scale, base, *eps, 1, nw, 1e-5)
            if not all(_biteq(torch, a[:real], b[lo:lo + real]) for a, b in zip(pre_sh, pre_full)):
                bad_pre.append((T, r))
            fused_sh = tl.mhc_fused_post_pre_tilelang(xs, rs, ps, cs, fn_f, scale, base, *eps, 1, 1, nw, 1e-5)
            if not all(_biteq(torch, a[:real], b[lo:lo + real]) for a, b in zip(fused_sh, fused_full)):
                bad_fused.append((T, r))
            post_sh = hc_contract(tl.mhc_post_tilelang(xs, rs, ps, cs), HC_MULT)
            if not _biteq(torch, post_sh[:real], post_out_full[lo:lo + real]):
                bad_post.append((T, r))
    res.append(("K mhc_pre_tilelang shard-invariant (bitwise) on zero-padded S-row shards, all T + tails",
                not bad_pre, str(bad_pre[:4])))
    res.append(("K mhc_fused_post_pre_tilelang shard-invariant (bitwise) on zero-padded S-row shards",
                not bad_fused, str(bad_fused[:4])))
    res.append(("K mhc_post_tilelang + hc_contract shard-invariant (bitwise) on zero-padded S-row shards",
                not bad_post, str(bad_post[:4])))


# ---------------------------------------------------------------------------------------------------------------------
# D. collective data flow on GPU tensors (single-GPU emulated 8-rank world)
# ---------------------------------------------------------------------------------------------------------------------
def section_d(M, torch, device, res):
    """Non-tautological: drive the module's OWN reduce_scatter_partial + gather_full over 8 DISTINCT per-rank partials
    through a world that sums them independently (fp32), and compare the reconstructed full-T tensor to that
    reference. The gather uses the calling rank's actual module output in its own slot, so a wrong shard shows up."""
    H = 256
    bad_rel = 0.0
    cross_bad = 0
    can_bad = 0

    class MultiShardWorld(M.EmulatedWorld):
        """Holds all 8 ranks' padded partials; reduce_scatter_rows / canonical return THIS rank's rows of the fp32
        sum over the 8 DISTINCT partials; all_gather_rows gathers the calling rank's actual output + the others'
        (computed) shards -> the full padded sum."""

        def __init__(self, parts_padded, size, rank):
            super().__init__(size, rank)
            self.parts = parts_padded

        def _sum_shard(self, r, S):
            acc = self.parts[0][r * S:(r + 1) * S].to(torch.float32).clone()
            for p in range(1, self.size):
                acc = acc + self.parts[p][r * S:(r + 1) * S].to(torch.float32)
            return acc

        def reduce_scatter_rows(self, x):
            S = x.shape[0] // self.size
            return self._sum_shard(self.rank, S).to(x.dtype)

        def canonical_reduce_scatter_rows(self, x, plan):
            return self._sum_shard(self.rank, plan.S).to(x.dtype)

        def all_gather_rows(self, x):
            S = x.shape[0]
            outs = []
            for r in range(self.size):
                outs.append(x if r == self.rank else self._sum_shard(r, S).to(x.dtype))
            return torch.cat(outs)

    for T in T_MAIN:
        plan = M.plan_rows(T, 8)
        parts = [M.synthetic_rows(T, H, 9000 + T + 7 * r, device).to(torch.float32) * (1 + 0.1 * r) for r in range(8)]
        parts_bf = [p.to(torch.bfloat16) for p in parts]
        parts_padded = [M.pad_rows(p, plan.padded) for p in parts_bf]
        ref = sum(x.to(torch.float32) for x in parts_padded)[:T].to(torch.bfloat16)
        for rs in ("nccl", "canonical"):
            fulls = []
            for r in range(8):
                w = MultiShardWorld(parts_padded, 8, r)
                shard = M.reduce_scatter_partial(parts_bf[r], w, plan, rs)        # the module's reduce path
                full = M.gather_full(shard, w, plan)                             # the module's gather path
                fulls.append(full)
                bad_rel = max(bad_rel, float((full.to(torch.float32) - ref.to(torch.float32)).abs().amax()
                                             / ref.to(torch.float32).abs().amax().clamp_min(1e-9)))
            if any(M.mismatches(f, fulls[0]) for f in fulls):
                cross_bad += 1
            if rs == "canonical":
                for r in range(8):
                    w = MultiShardWorld(parts_padded, 8, r)
                    can = w.canonical_reduce_scatter_rows(M.pad_rows(parts_bf[r], plan.padded), plan)
                    loc = sum(parts_padded[p][plan.start(r):plan.start(r) + plan.S].to(torch.float32)
                              for p in range(8))
                    if M.mismatches(can, loc.to(torch.bfloat16)) != 0:
                        can_bad += 1
    res.append(("D reduce_scatter_partial + gather_full reconstruct an independent fp32 reference (rel <= 2e-2)",
                bad_rel <= 2e-2, f"rel={bad_rel:.2e}"))
    res.append(("D reconstructed full-T bit-identical across the 8 shards", cross_bad == 0, str(cross_bad)))
    res.append(("D canonical reduce-scatter == rank-order fp32 sum (bitwise)", can_bad == 0, str(can_bad)))


# ---------------------------------------------------------------------------------------------------------------------
# G. CUDA-graph safety
# ---------------------------------------------------------------------------------------------------------------------
def section_g(M, torch, device, res):
    M._reset_for_test(mode="sp", min_t=1024, rs_mode="nccl")
    M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
    # decode-sized / short forward -> no sp (the small_skips path); no CUDA needed.
    inert = M.begin_forward(48, HIDDEN, device=device) is None
    captured_inert = True
    cap_info = ""
    if device.type == "cuda":
        # During capture begin_forward must be inert (return None) and MUST NOT raise (no collective, no split): if
        # the capture raises, that is a FAILURE, not a silent pass.
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                captured = M.begin_forward(4096, HIDDEN, device=device)
            if captured is not None:
                captured_inert = False
                cap_info = "begin_forward returned a handle during capture"
        except Exception as e:  # noqa: BLE001 - a raise during capture is a failure
            captured_inert = False
            cap_info = f"begin_forward raised during capture: {type(e).__name__}: {e}"
    else:
        # CPU emulate: no CUDA init; _capturing returns False for a CPU tensor (is_cuda False).
        captured_inert = M._capturing(torch.empty(1)) is False
    M._set_world_factory_for_test(None)
    M._reset_for_test()
    res.append(("G short forward (T=48) -> no sp; capture -> ineligible and does not raise",
                inert and captured_inert, cap_info))


def run(args):
    out = {"when": dt.datetime.now(dt.timezone.utc).isoformat(), "emulate": args.emulate, "sections": {}}
    if not args.emulate and server_busy(STATUS_URL) and not args.force:
        log("[refused] server has in-flight requests; rerun when idle or pass --force (3)")
        return 3
    # In the serving container VLLM_GB10_MHC_PRS=1 is set, so calling the mHC TileLang ops would fire patch 022's
    # prenorm hook -> _TPWorld -> get_tp_group() which is NOT initialised in a bare `docker exec` (crash). Neutralise
    # before importing tilelang (as p022_gpu_selftest.py). The 025 module is driven explicitly, not by its env gate, so
    # pop it too; force the TileLang prenorm path (VLLM_USE_DEEP_GEMM=0 -> is_deep_gemm_supported() False).
    for k in ("VLLM_GB10_MHC_PRS", "VLLM_GB10_MHC_SP"):
        os.environ.pop(k, None)
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    out["env_neutralised"] = ["VLLM_GB10_MHC_PRS", "VLLM_GB10_MHC_SP", "VLLM_USE_DEEP_GEMM=0"]
    M = load_module(args.vllm_root, args.emulate)
    res = []
    if args.emulate:
        import torch
        torch.set_num_threads(1)
        device = torch.device("cpu")
        # K needs the real TileLang kernels; under --emulate only D and G run (harness logic) - unless --emulate-tree
        # points at a v0.32.0 tree: then K also runs on the REAL JIT classes over fake row-local kernels
        # (pmhc_testlib), at reduced sizes, to validate this harness's call shapes against the real op signatures.
        if args.emulate_tree:
            global HIDDEN, T_MAIN
            sys.path.insert(0, str(HERE))
            import pmhc_testlib as L
            HIDDEN, T_MAIN = 128, (1024, 1027)
            envx = L.build_env(Path(args.emulate_tree))
            mhc_stub = types.ModuleType("vllm.model_executor.layers.mhc")
            mhc_stub.hc_contract = lambda x, n: x.mean(dim=1)
            mhc_stub.hc_expand = lambda x, n: x.unsqueeze(1).expand(-1, n, -1).contiguous()
            envx.modules["vllm.model_executor.layers.mhc"] = mhc_stub
            with envx.activate():
                section_k(M, torch, device, res)
        section_d(M, torch, device, res)
        section_g(M, torch, device, res)
    else:
        import torch
        device = torch.device("cuda")
        base = sha_file(Path(args.vllm_root) / "models/glm5next/common/model.py")
        out["model_sha"] = base
        section_k(M, torch, device, res)
        section_d(M, torch, device, res)
        section_g(M, torch, device, res)
    npass = sum(1 for r in res if r[1])
    for name, ok, *info in res:
        log(("PASS " if ok else "FAIL ") + name + (f"  [{info[0]}]" if info and info[0] else ""))
        out["sections"][name] = {"ok": bool(ok), "info": (info[0] if info else "")}
    out["pass"] = npass
    out["total"] = len(res)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    log(f"p025 GPU self-test: {npass}/{len(res)} passed")
    return 0 if npass == len(res) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", default="/usr/local/lib/python3.12/dist-packages/vllm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emulate", action="store_true")
    ap.add_argument("--emulate-tree", default=None, help="with --emulate: a v0.32.0 `vllm` dir; also runs section K on CPU")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.emulate:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        return run(args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
