#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p022_gpu_selftest.py - patch 022 (VLLM_GB10_MHC_PRS=1, mode 'prs') GPU proof on the v0.32.0 image's own vLLM + TileLang:
the mHC pre-norm GEMM split by token rows over 8 (emulated) TP ranks + gather is BIT-IDENTICAL to the unsplit call, for the
kernel alone and for the whole mHC block path. WRITTEN, NOT RUN in the authoring session (no GPU / cluster access).

Run INSIDE a rank container with patch 022 installed (started with VLLM_GB10_MHC_PRS=1, or patches applied by hand) while the
server is idle (no NCCL, no network: all 8 ranks' shards run on this one GPU and the all-gather is emulated by concatenation,
byte for byte what NCCL's all_gather returns; ~1-3 min). The script sets the gate itself before importing, pops the 025 gate and
forces VLLM_USE_DEEP_GEMM=0 (which makes `_hc_prenorm_gemm_outputs` take the TileLang class branch the hook sits in):

  ssh node1 'docker exec <rank-0 container> python3 /opt/recipe-patches/files/p022_gpu_selftest.py --out /tmp/p022-gpu.json'

It does not re-exec source text: the v0.32.0 gate is a module global of kernels/mhc/tilelang.py read at CALL time, so the SAME
installed functions are run with the gate on (the patched path) and off (upstream) and compared. The kernels are the image's TileLang
kernels (compiled on first use, cached under TILELANG_CACHE_DIR).

Checks (every comparison bitwise: int32 / int16 views; outputs NaN-poisoned before every call):
  K kernel   the JIT function the class's dispatch() names for the call (hc_prenorm_gemm_block_m_tilelang) on all T rows vs the same
             function on the 8 row shards + gather + unpack, T in {1024, 1280, 2048, 3840, 8192} + {1027, 3853, 8191} (uneven
             tails) x x-kinds {gauss, wide, outlier} x fn {random bf16-exact, random full-mantissa fp32} (+ the checkpoint's real fn
             with --model-dir, default $MODEL_DIR): out + sqrsum identical, fully written, repeat identical; and every shard ALONE == the unsplit rows
  R  route   the class's dispatch() routes T >= MIN_T to block_m=2 for the installed threshold; prenorm_rowsplit returns False for a
             call it does not route to block_m=2 (counted not_bm2)
  S  boot    the module's boot path on the first eligible call (T = 8192, the profile-run size): envelope, rank fingerprint and the
             collective self-test (9 cases) pass in the emulated 8-rank world; ACTIVE logged once
  B  block   gate on vs gate off: mhc_fused_post_pre_tilelang (num_tokens > 32) and layer-0 mhc_pre_tilelang, T as in K: all
             outputs identical; one split per call (stats()['calls']); a 12-step chain (layer-0 pre, then fused calls) identical
  G  graphs  the op captured at T = 1024 >= MIN_T: the hook returns False during capture (capture_skips + 1, no split captured),
             replays == the gate-off eager op on fresh inputs; decode sizes 24, 48 captured gate on vs off: identical replays
  N  controls must be detected: one flipped bit in the gathered buffer (exactly 1 mismatch); shards run on the small-T kernel config
             (n_thr 1024, tile_n 4) instead of the block_m=2 kernel (expected NOT bit-identical: why 022 hands the shard the full
             call's own kernel). Info: shards starting at odd rows
  T  timing  (skip with --no-bench) DRAM-cold: unsplit at T, the largest shard, the unpack; the all-gather ESTIMATE from nccl-tests
             numbers (65 us + 7 x bytes/rank / 12.1 GB/s); per call and x 90 per forward. The real split + gather time is logged by
             the boot self-test on the first 022 boot.
Exit: 0 PASS | 1 FAIL | 2 setup error.

CPU validation of this harness (the REAL classes over fake row-local kernels, no GPU; graphs/timing skipped):
  PYTHONDONTWRITEBYTECODE=1 python3 files/p022_gpu_selftest.py --emulate-tree <v0.32.0>/vllm
"""
import argparse
import datetime as dt
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MODEL_DIR", "<MODEL_DIR>")   # checkpoint directory (holds model.safetensors.index.json); --model-dir overrides
T_MAIN = (1024, 1280, 2048, 3840, 8192)
T_EXTRA = (1027, 3853, 8191)
CALLS_PER_FORWARD = 90


def log(*a):
    print(*a, flush=True)


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else repr(o)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def biteq(torch, a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return bool(torch.equal(a.contiguous().view(iv), b.contiguous().view(iv)))


def run(args):  # noqa: C901
    os.environ["VLLM_GB10_MHC_PRS"] = "1"
    os.environ.pop("VLLM_GB10_MHC_SP", None)
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    import torch
    emu = args.emulate_tree is not None
    dev = torch.device("cpu" if emu else "cuda")
    out = {"when": dt.datetime.now(dt.timezone.utc).isoformat(), "emulate": emu, "sections": {}}
    ctx = None
    if emu:
        sys.path.insert(0, str(HERE))
        import pmhc_testlib as L
        tree = Path(args.emulate_tree)
        wd = Path(tempfile.mkdtemp(prefix="p022gpu-"))
        spec = importlib.util.spec_from_file_location("p022_apply", HERE / "p022_apply.py")
        P22 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P22)
        (wd / "tl.py").write_text(P22.apply((tree / "model_executor/kernels/mhc/tilelang.py").read_text()))
        envx = L.build_env(tree, tl_path=wd / "tl.py", extra=[("vllm._gb10_mhc_prs", HERE / "gb10_mhc_prs.py")])
        ctx = envx.activate()
        ctx.__enter__()
    try:
        tl = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang")
        tk = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang_kernels")
        if not getattr(tl, "_P022_ON", False):
            log("[setup] the installed tilelang.py does not carry patch 022 with the gate on (is 022 installed?)")
            return 2
        P = importlib.import_module("vllm._gb10_mhc_prs")
        hidden, hc = (128 if emu else args.hidden), 4
        n_out = (2 + hc) * hc
        K = hidden * hc
        Kc = tk._HC_PRENORM_GEMM_TILELANG_KERNEL
        P._reset_for_test(min_t=1024)
        P._set_world_factory_for_test(lambda: P.EmulatedWorld(8))
        W = P.EmulatedWorld(8)
        res = []
        ts = list(T_MAIN) + list(T_EXTRA)
        if emu:
            ts = [1024, 1027, 2048]
        g = torch.Generator().manual_seed(5)

        def mkx(kind, T):
            x = torch.randn(T, K, generator=g) * 3
            if kind == "wide":
                x = x * (2.0 ** torch.randint(-5, 6, (T, K), generator=g).float())
            elif kind == "outlier":
                x[:, ::89] *= 250.0
            return x.to(torch.bfloat16).to(dev)

        def mkfn(kind):
            fn = torch.randn(n_out, K, generator=g) * 0.02
            if kind == "bf16":
                fn = fn.to(torch.bfloat16).float()
            return fn.contiguous().to(dev)

        def poison(shape):
            return torch.full(shape, float("nan"), dtype=torch.float32, device=dev)

        # ---- K ------------------------------------------------------------------------------------------------
        bad_k, bad_rep, bad_shard, unwritten = [], [], [], []
        route_bad = []
        fns = [("bf16", mkfn("bf16")), ("full", mkfn("full"))]
        if not emu and Path(args.model_dir, "model.safetensors.index.json").is_file():
            import json as _j
            from safetensors import safe_open
            wm = _j.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))["weight_map"]
            name = sorted(k for k in wm if k.endswith("layers.3.hc_ffn_fn"))[0]
            with safe_open(os.path.join(args.model_dir, wm[name]), framework="pt", device="cpu") as fh:
                real = torch.empty(fh.get_tensor(name).shape, dtype=torch.float32)
                real.copy_(fh.get_tensor(name))
            fns.append(("checkpoint", real.to(dev)))
        for T in ts:
            key = Kc.dispatch(num_tokens=T, hc_hidden_size=K, hidden_size=hidden, hc_mult=hc, n_out=n_out)
            if not key.use_block_m:
                route_bad.append((T, "not block_m=2"))
                continue
            kfn = Kc.kernel(key)
            plan = P.plan_rows(T, 8)
            for xk in ("gauss", "wide", "outlier"):
                x = mkx(xk, T)
                for fk, fn in fns:
                    ro, rs = poison((1, T, n_out)), poison((1, T))
                    kfn(x, fn, ro, rs, hidden, hc, n_out, key.n_thr, key.tile_n, key.block_m)
                    o1, s1 = poison((1, T, n_out)), poison((1, T))
                    P.split_into(kfn, x, fn, o1, s1, hidden, hc, key.n_thr, key.tile_n, W, plan)
                    o2, s2 = poison((1, T, n_out)), poison((1, T))
                    P.split_into(kfn, x, fn, o2, s2, hidden, hc, key.n_thr, key.tile_n, W, plan)
                    if not (biteq(torch, o1, ro) and biteq(torch, s1, rs)):
                        bad_k.append((T, xk, fk))
                    if not (biteq(torch, o2, o1) and biteq(torch, s2, s1)):
                        bad_rep.append((T, xk, fk))
                    if torch.isnan(o1).any() or torch.isnan(s1).any() or torch.isnan(ro).any():
                        unwritten.append((T, xk, fk))
                    if xk == "gauss" and fk == "full":
                        for r in range(8):
                            b, e = plan.starts[r], plan.ends[r]
                            if e > b:
                                so, ss = poison((1, e - b, n_out)), poison((1, e - b))
                                kfn(x[b:e], fn, so, ss, hidden, hc, n_out, key.n_thr, key.tile_n, key.block_m)
                                if not (biteq(torch, so[0], ro[0, b:e]) and biteq(torch, ss[0], rs[0, b:e])):
                                    bad_shard.append((T, r))
        res.append(("K split + gather + unpack == unsplit (bitwise), T list x 3 x-kinds x fn kinds, all written", not bad_k and not unwritten,
                    f"bad={bad_k[:4]} unwritten={unwritten[:3]}"))
        res.append(("K repeat bit-identical", not bad_rep, str(bad_rep[:3])))
        res.append(("K every shard ALONE == the unsplit rows (row invariance, directly)", not bad_shard, str(bad_shard[:4])))
        # ---- R ------------------------------------------------------------------------------------------------
        thr = getattr(tk, "_P011_BM2_MIN_T", 1024)
        x_small = mkx("gauss", 512)
        P._reset_for_test(min_t=256)
        P._set_world_factory_for_test(lambda: P.EmulatedWorld(8))
        o_, s_ = poison((1, 512, n_out)), poison((1, 512))
        r512 = P.prenorm_rowsplit(Kc, x_small, fns[0][1], o_, s_, hidden, hc)
        want = thr <= 512
        res.append((f"R routing: threshold {thr}: a T=512 call (MIN_T 256) is {'split' if want else 'NOT split (not_bm2)'}",
                    r512 is want and (P.stats()["not_bm2"] == (0 if want else 1)), f"returned {r512} stats {P.stats()}"))
        res.append(("R every T >= 1024 in the list is routed to block_m=2 by the class's dispatch()", not route_bad, str(route_bad)))
        P._reset_for_test(min_t=1024)
        P._set_world_factory_for_test(lambda: P.EmulatedWorld(8))
        # ---- S ------------------------------------------------------------------------------------------------
        import logging
        import io
        stream = io.StringIO()
        h = logging.StreamHandler(stream)
        P.logger.addHandler(h)
        P.logger.setLevel(logging.INFO)
        Tb = 8192 if not emu else 2048
        xb, fnb = mkx("gauss", Tb), fns[0][1]
        key = Kc.dispatch(num_tokens=Tb, hc_hidden_size=K, hidden_size=hidden, hc_mult=hc, n_out=n_out)
        ro, rs = poison((1, Tb, n_out)), poison((1, Tb))
        Kc.kernel(key)(xb, fnb, ro, rs, hidden, hc, n_out, key.n_thr, key.tile_n, key.block_m)
        ob, sb = poison((1, Tb, n_out)), poison((1, Tb))
        okb = P.prenorm_rowsplit(Kc, xb, fnb, ob, sb, hidden, hc)
        st = P.stats()
        res.append(("S boot on the first eligible call: envelope + fingerprint + self-test pass, live output exact, ACTIVE logged once",
                    okb is True and st["active"] and st["report"]["ranks"] == 8 and biteq(torch, ob, ro) and biteq(torch, sb, rs)
                    and stream.getvalue().count("ACTIVE") == 1, str(st["report"])))
        P.logger.removeHandler(h)
        # ---- B ------------------------------------------------------------------------------------------------
        def block_args(T, seed, kind):
            gg = torch.Generator().manual_seed(seed)
            x = (torch.randn(T, hidden, generator=gg) * 2).to(torch.bfloat16).to(dev)
            r = (torch.randn(T, hc, hidden, generator=gg) * 4).to(torch.bfloat16).to(dev)
            post = (torch.sigmoid(torch.randn(T, hc, 1, generator=gg)) * 2).contiguous().to(dev)
            comb = torch.softmax(torch.randn(T, hc, hc, generator=gg), -1).contiguous().to(dev)
            fn = (torch.randn(n_out, K, generator=gg) * 0.02).to(torch.bfloat16).float().to(dev)
            scale = (torch.rand(3, generator=gg) + 0.5).to(dev)
            base = (torch.randn(n_out, generator=gg) * 0.1).to(dev)
            nw = (torch.rand(hidden, generator=gg) + 0.5).to(torch.bfloat16).to(dev)
            eps = (1e-5, 1e-6, 1e-6, 2.0, 20)
            if kind == "fused":
                return (x, r, post, comb, fn, scale, base, *eps, 1, 1, nw, 1e-5)
            return (r, fn, scale, base, *eps, 1, nw, 1e-5)

        def call(kind, a, gate):
            tl._P022_ON = gate
            try:
                return (tl.mhc_fused_post_pre_tilelang if kind == "fused" else tl.mhc_pre_tilelang)(*a)
            finally:
                tl._P022_ON = True

        bad_b, bad_calls = [], []
        for T in ts:
            for kind in ("fused", "pre"):
                a = block_args(T, T, kind)
                c0 = P.stats()["calls"]
                o_new = call(kind, a, True)
                n_split = P.stats()["calls"] - c0
                o_ref = call(kind, a, False)
                if not all(biteq(torch, u, v) for u, v in zip(o_new, o_ref)):
                    bad_b.append((T, kind))
                if n_split != 1:
                    bad_calls.append((T, kind, n_split))
        res.append(("B block op gate on == gate off (all outputs, T list; layer-0 pre + fused M > 32)", not bad_b, str(bad_b[:4])))
        res.append(("B exactly one row-split per pre-norm call", not bad_calls, str(bad_calls[:4])))
        # chain: layer-0 pre, then 11 fused calls feeding each other
        T = ts[1]
        a0 = block_args(T, 99, "pre")
        outs = {}
        for gate in (True, False):
            _post, _comb, li = call("pre", a0, gate)[:3]
            r_ = a0[0]
            state = (li, r_, _post, _comb)
            for i in range(11):
                fa = block_args(T, 200 + i, "fused")
                state_in = (state[0], state[1], state[2], state[3]) + fa[4:]
                r2, p2, c2, x2 = call("fused", state_in, gate)
                state = (x2, r2, p2, c2)
            outs[gate] = state
        res.append(("B 12-step chain identical", all(biteq(torch, u, v) for u, v in zip(outs[True], outs[False])), f"T={T}"))
        # ---- G (GPU only) ---------------------------------------------------------------------------------------
        if not emu:
            P._reset_for_test(min_t=1024)
            P._set_world_factory_for_test(lambda: P.EmulatedWorld(8))
            call("fused", block_args(2048, 7, "fused"), True)          # boot outside any capture
            cs0 = P.stats()["capture_skips"]
            a = list(block_args(1024, 8, "fused"))
            ref_in = block_args(1024, 9, "fused")
            ref = call("fused", ref_in, False)
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                cap = call("fused", tuple(a), True)
            for i_ in (0, 1, 2, 3):
                a[i_].copy_(ref_in[i_])
            gph.replay()
            torch.cuda.synchronize()
            res.append(("G T=1024 captured: hook returns False during capture (capture_skips + 1), replay == gate-off eager",
                        P.stats()["capture_skips"] > cs0 and all(biteq(torch, u, v) for u, v in zip(cap, ref)),
                        str(P.stats())))
            calls0 = P.stats()["calls"]
            bad_g = []
            for m in (24, 48):
                a = list(block_args(m, 30 + m, "fused"))
                ref_in = block_args(m, 40 + m, "fused")
                ref = call("fused", ref_in, False)
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    cap = call("fused", tuple(a), True)
                for i_ in (0, 1, 2, 3):
                    a[i_].copy_(ref_in[i_])
                gph.replay()
                torch.cuda.synchronize()
                if not all(biteq(torch, u, v) for u, v in zip(cap, ref)):
                    bad_g.append(m)
            res.append(("G decode sizes 24 / 48 captured gate on vs off: identical; the module never split", not bad_g
                        and P.stats()["calls"] == calls0, str(bad_g)))
        # ---- N ------------------------------------------------------------------------------------------------
        T = 1027
        key = Kc.dispatch(num_tokens=T, hc_hidden_size=K, hidden_size=hidden, hc_mult=hc, n_out=n_out)
        kfn = Kc.kernel(key)
        x, fn = mkx("gauss", T), fns[1][1]
        ro, rs = poison((1, T, n_out)), poison((1, T))
        kfn(x, fn, ro, rs, hidden, hc, n_out, key.n_thr, key.tile_n, key.block_m)

        class Corrupt(P.EmulatedWorld):
            def gather_shards(self, make_send):
                gt = super().gather_shards(make_send)
                gt.view(torch.int32)[3] ^= 1
                return gt
        o, s = poison((1, T, n_out)), poison((1, T))
        P.split_into(kfn, x, fn, o, s, hidden, hc, key.n_thr, key.tile_n, Corrupt(8))
        flips = P.mismatches(o, ro) + P.mismatches(s, rs)
        res.append(("N control: one flipped bit in the gathered buffer is detected (exactly 1 mismatch)", flips == 1, f"{flips}"))
        if not emu and K % 1024 == 0:
            plan = P.plan_rows(T, 8)
            bad_small = 0
            for r in range(8):
                b, e = plan.starts[r], plan.ends[r]
                so, ss = poison((1, e - b, n_out)), poison((1, e - b))
                tk.hc_prenorm_gemm_tilelang(x[b:e], fn, so, ss, hidden, hc, n_out, 1024, 4, 1)
                bad_small += P.mismatches(so[0], ro[0, b:e])
            res.append(("N control (info): shards on the small-T kernel config (n_thr 1024, tile_n 4) are NOT bit-identical "
                        "(expected: why the shard runs the full call's own kernel)", bad_small > 0, f"{bad_small} elements differ"))
        # ---- T ------------------------------------------------------------------------------------------------
        if not emu and not args.no_bench:
            flush = torch.empty(96 * 2**20 // 4, dtype=torch.float32, device=dev)

            def med(f, n=7):
                v = []
                for _ in range(n):
                    flush.zero_()
                    a_, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a_.record()
                    f()
                    b_.record()
                    b_.synchronize()
                    v.append(a_.elapsed_time(b_))
                return statistics.median(v)
            rows = []
            for T in (1024, 2048, 3840):
                key = Kc.dispatch(num_tokens=T, hc_hidden_size=K, hidden_size=hidden, hc_mult=hc, n_out=n_out)
                kfn = Kc.kernel(key)
                x, fn = mkx("gauss", T), fns[0][1]
                plan = P.plan_rows(T, 8)
                o, s = torch.empty(1, T, n_out, device=dev), torch.empty(1, T, device=dev)
                so, ss = torch.empty(1, plan.S, n_out, device=dev), torch.empty(1, plan.S, device=dev)
                t_un = med(lambda: kfn(x, fn, o, s, hidden, hc, n_out, key.n_thr, key.tile_n, key.block_m))
                t_sh = med(lambda: kfn(x[:plan.S], fn, so, ss, hidden, hc, n_out, key.n_thr, key.tile_n, key.block_m))
                est_ag = 0.065 + 7 * (plan.S * 25 * 4) / 12.1e9 * 1000
                rows.append((T, round(t_un, 4), round(t_sh, 4), round(est_ag, 4), round((t_un - t_sh - est_ag) * CALLS_PER_FORWARD, 1)))
                log(f"T timing T={T}: unsplit {t_un:.3f} ms, one shard ({plan.S} rows) {t_sh:.3f} ms, all-gather est {est_ag:.3f} ms -> "
                    f"saving ~{(t_un - t_sh - est_ag) * CALLS_PER_FORWARD:.0f} ms per {CALLS_PER_FORWARD}-call forward")
            out["timing"] = rows
        npass = sum(1 for r_ in res if r_[1])
        for name, ok, info in res:
            log(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
            out["sections"][name] = {"ok": bool(ok), "info": info}
        out["pass"], out["total"] = npass, len(res)
        if args.out:
            Path(args.out).write_text(json.dumps(clean(out), indent=2))
        log(f"p022 GPU self-test: {npass}/{len(res)} passed")
        return 0 if npass == len(res) else 1
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--emulate-tree", default=None, help="a v0.32.0 `vllm` dir: run on CPU over fake kernels (harness check)")
    args = ap.parse_args()
    if args.emulate_tree:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        return run(args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
