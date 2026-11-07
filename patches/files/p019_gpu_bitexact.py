#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p019_gpu_bitexact.py - patch 019 (VLLM_GB10_MHC_FN_BF16=1) GPU proof on the v0.32.0 image's own vLLM + TileLang: the fused
mHC decode op reading bf16 copies of hc_*_fn is BIT-IDENTICAL to the upstream fp32 path, eagerly and under CUDA-graph replay,
for num_tokens = 1..48. WRITTEN, NOT RUN in the authoring session (no GPU / cluster access).

Run INSIDE a rank container (patches 019 [and optionally 011/022] installed by the entrypoint, i.e. started with
VLLM_GB10_MHC_FN_BF16=1; or a container whose patches/ were applied by hand) while the server is idle; ~1-3 min incl. the first
TileLang compile of the bf16 kernel (cached under TILELANG_CACHE_DIR). The script sets the gate itself before importing, pops the
022/025 gates (their collectives need an initialised TP group, which a bare `docker exec` does not have) and forces
VLLM_USE_DEEP_GEMM=0:

  ssh node1 'docker exec <rank-0 container> python3 /opt/recipe-patches/files/p019_gpu_bitexact.py --out /tmp/p019-bitexact.json'

Unlike the 487ecf187-era script it does not re-exec source text: the v0.32.0 gate is a module global of kernels/mhc/tilelang.py
read at CALL time (`if _P019_ON:`), so the SAME installed function is run twice - gate on (the patched path: bf16 class
MhcFusedBf16wTileLangKernel on the cached copy) and gate off (the upstream fp32 class MhcFusedTileLangKernel) - and the outputs
compared bitwise. Every comparison is bitwise (int32 / int16 views). Checks:
  P  premise    (--model-dir, default $MODEL_DIR) all 90 hc_*_fn, loaded like vLLM (bf16 checkpoint tensor -> fp32 param.copy_), survive fp32 ->
                bf16 -> fp32 bitwise; without a checkpoint synthetic bf16-exact weights are used (stated in the output)
  S  self-test  the module's boot self-test (both kernels, raw JIT + class path, M = 1, 7, 8, 15, 16, 24, 32) ran on the first
                conversion and passed; stats() consistent
  K  kernel     the raw JIT functions mhc_fused_tilelang vs mhc_fused_bf16w_tilelang, and the production classes, M = 1..32 at the
                op's own (tile_n, n_splits, n_thr), 3 weights x 2 activation kinds: yp_out, rp_out, residual_out identical and
                fully written (NaN sentinel on the raw launches)
  O  op         mhc_fused_post_pre_tilelang(M tokens), M = 1..48 x kinds: gate on vs gate off identical in all four outputs;
                routing traced (gate on: the bf16 class launches for M <= 32 and never the fp32 one; M > 32 the fp32 pre-norm
                path; gate off: the fp32 class only); the first call is at M = 48 (conversion at M > 32, like the profile run)
  G  graphs     every capture size of the recipe (1 2 4 6 8 16 24 32 40 48): the gate-on op captured in a torch.cuda.CUDAGraph
                with static inputs; fresh inputs copied in place; each replay == the gate-off eager output; no conversion and
                no capture miss during capture; M <= 32 graphs captured the bf16 class
  F  fallback   cache cleared: a weight first seen INSIDE a capture -> the fp32 kernel is captured (capture_misses 1), replay ==
                reference
  C  chain      one CUDA graph of 89 chained fused calls (M = 3 and 16), gate on vs gate off: final state identical after every replay
  T  timing     (skip with --no-bench) DRAM-cold per call, fp32 vs bf16 raw kernel, M = 1, 3, 8, 16, 32, cycling 89 distinct
                weights (134 / 67 MiB >> L2), and the C chain graphs interleaved -> ms saved per target forward
Exit: 0 PASS | 1 bit-exact FAIL | 2 setup error.

CPU validation of this harness (the REAL classes over fake kernels, no GPU; graphs/timing skipped):
  PYTHONDONTWRITEBYTECODE=1 python3 files/p019_gpu_bitexact.py --emulate-tree <v0.32.0>/vllm
"""
import argparse
import datetime as dt
import importlib
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
CAPTURE_SIZES = (1, 2, 4, 6, 8, 16, 24, 32, 40, 48)
FUSED_MAX = 32


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


def load_checkpoint_weights(model_dir, layers, dev, torch):
    """{(L, 'attn'|'ffn'): (fn fp32 as vLLM holds it, scale, base, norm)} for the given layers (+ premise stats)."""
    from safetensors import safe_open
    wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for L in layers:
        for kind, norm in (("attn", "input_layernorm.weight"), ("ffn", "post_attention_layernorm.weight")):
            cands = sorted(k for k in wm if k.endswith(f"layers.{L}.hc_{kind}_fn"))
            cands.sort(key=lambda k: 0 if k.startswith("model.language_model.") else 1)
            pre = cands[0][: -len(f"hc_{kind}_fn")]
            names = {"fn": cands[0], "scale": pre + f"hc_{kind}_scale", "base": pre + f"hc_{kind}_base", "norm": pre + norm}
            t = {}
            for k, n in names.items():
                with safe_open(os.path.join(model_dir, wm[n]), framework="pt", device="cpu") as fh:
                    t[k] = fh.get_tensor(n)
            fn = torch.empty(t["fn"].shape, dtype=torch.float32, device=dev)
            fn.copy_(t["fn"])           # default_weight_loader: param.data.copy_(loaded_weight)
            out[(L, kind)] = (fn, t["scale"].float().to(dev), t["base"].float().to(dev), t["norm"].to(torch.bfloat16).to(dev),
                              str(t["fn"].dtype))
    return out


def synth_weights(n, hidden, hc, dev, torch, seed=7):
    g = torch.Generator().manual_seed(seed)
    n_out = (2 + hc) * hc
    out = {}
    for i in range(n):
        fn = (torch.randn(n_out, hc * hidden, generator=g) * 0.02).to(torch.bfloat16).float()
        out[(i, "x")] = (fn.to(dev), (torch.rand(3, generator=g) + 0.5).to(dev), (torch.randn(n_out, generator=g) * 0.1).to(dev),
                         (torch.rand(hidden, generator=g) + 0.5).to(torch.bfloat16).to(dev), "synthetic")
    return out


def run(args):  # noqa: C901
    # the gate is read at import: set it (and neutralise the collective patches) BEFORE the mHC modules are imported
    os.environ["VLLM_GB10_MHC_FN_BF16"] = "1"
    for k in ("VLLM_GB10_MHC_PRS", "VLLM_GB10_MHC_SP"):
        os.environ.pop(k, None)
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
        wd = Path(tempfile.mkdtemp(prefix="p019gpu-"))
        P = importlib.util.spec_from_file_location("p019_apply", HERE / "p019_apply.py")
        Pm = importlib.util.module_from_spec(P)
        P.loader.exec_module(Pm)
        (wd / "tl.py").write_text(Pm.apply_tl((tree / "model_executor/kernels/mhc/tilelang.py").read_text()))
        (wd / "tk.py").write_text(Pm.apply_tk((tree / "model_executor/kernels/mhc/tilelang_kernels.py").read_text()))
        envx = L.build_env(tree, tk_path=wd / "tk.py", tl_path=wd / "tl.py")
        envx.load_module("vllm._gb10_mhc_fn_bf16", HERE / "gb10_mhc_fn_bf16.py")
        ctx = envx.activate()
        ctx.__enter__()
        args.hidden = 128
    try:
        tl = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang")
        tk = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang_kernels")
        if not getattr(tl, "_P019_ON", False) or "_p019" not in open(tl.__file__).read():
            log("[setup] the installed tilelang.py does not carry patch 019 with the gate on (is 019 installed?)")
            return 2
        M = importlib.import_module("vllm._gb10_mhc_fn_bf16")
        hidden, hc = args.hidden, 4
        n_out = (2 + hc) * hc
        if emu:
            def validate_cpu(fn, hc_, hid_):
                if fn.dtype != torch.float32 or tuple(fn.shape) != (hc_ * (hc_ + 2), hc_ * hid_) or not fn.is_contiguous():
                    raise RuntimeError("[patch-019] unexpected mHC weight (CPU stand-in)")
            M._validate = validate_cpu
            torch.cuda.is_current_stream_capturing = lambda: False
        res = []
        # ---- weights --------------------------------------------------------------------------------------------
        have_ckpt = (not emu) and Path(args.model_dir, "model.safetensors.index.json").is_file()
        if have_ckpt:
            cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
            tc = cfg.get("text_config", cfg)
            hidden, hc = int(tc["hidden_size"]), int(tc.get("hc_mult", tc.get("mhc_num_residual_streams", 4)))
            W3 = load_checkpoint_weights(args.model_dir, (1, 20, 43), dev, torch)
            src = "checkpoint (3 layers x attn/ffn)"
        else:
            W3 = synth_weights(6, hidden, hc, dev, torch)
            src = "SYNTHETIC bf16-exact weights (no checkpoint here)"
        log(f"weights: {src}; hidden {hidden}, hc {hc}")
        out["weights"] = src
        keys = list(W3)

        def exact(fn):
            w, bad = M._to_bf16_exact(fn)
            return bad == 0
        res.append(("P every weight used survives fp32 -> bf16 -> fp32 bitwise", all(exact(W3[k][0]) for k in keys), src))
        # ---- inputs ---------------------------------------------------------------------------------------------
        g = torch.Generator().manual_seed(11)

        def inputs(m, kind):
            x = torch.randn(m, hidden, generator=g)
            res_ = torch.randn(m, hc, hidden, generator=g) * 4
            if kind == "wide":
                res_ = res_ * (2.0 ** torch.randint(-4, 5, (m, hc, hidden), generator=g).float())
            post = torch.rand(m, hc, 1, generator=g) * 2
            comb = torch.softmax(torch.randn(m, hc, hc, generator=g), -1)
            return (x.to(torch.bfloat16).to(dev), res_.to(torch.bfloat16).to(dev), post.to(dev), comb.to(dev))

        def op_args(m, kind, wkey):
            fn, scale, base, norm = W3[wkey][:4]
            x, r, p, c = inputs(m, kind)
            return (x, r, p, c, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)

        # trace the class launches
        counts = {"fp32": 0, "bf16": 0}
        fp32_cls, bf16_cls = tk._MHC_FUSED_TILELANG_KERNEL, M.MHC_FUSED_BF16W_KERNEL
        for nm, inst in (("fp32", fp32_cls), ("bf16", bf16_cls)):
            orig = inst.launch

            def counting(jit_impl, a, kw=None, _o=orig, _n=nm):
                counts[_n] += 1
                return _o(jit_impl, a, kw)
            inst.launch = counting

        def call_op(a, gate):
            tl._P019_ON = gate
            try:
                return tl.mhc_fused_post_pre_tilelang(*a)
            finally:
                tl._P019_ON = True

        # ---- O (first call at M = 48 converts) --------------------------------------------------------------------
        bad_op, bad_route = [], []
        first = True
        for m in range(48, 0, -1):
            for kind in ("gauss", "wide"):
                for wk in keys[:2]:
                    a = op_args(m, kind, wk)
                    counts["fp32"] = counts["bf16"] = 0
                    o_off = call_op(a, False)
                    c_off = dict(counts)
                    counts["fp32"] = counts["bf16"] = 0
                    o_on = call_op(a, True)
                    c_on = dict(counts)
                    if not all(biteq(torch, x, y) for x, y in zip(o_on, o_off)):
                        bad_op.append((m, kind))
                    fused = m <= FUSED_MAX
                    if not first:
                        if c_off != {"fp32": int(fused), "bf16": 0} or c_on != {"fp32": 0, "bf16": int(fused)}:
                            bad_route.append((m, c_off, c_on))
                    first = False
        res.append(("S module self-test ran on the first conversion and passed", M.stats()["selftest"] and M.stats()["converted"] >= 1,
                    str(M.stats())))
        res.append(("O op gate on == gate off, all 4 outputs, M = 1..48 x kinds x weights", not bad_op, str(bad_op[:4])))
        res.append(("O routing: gate on -> bf16 class for M <= 32 (never the fp32 one); gate off -> fp32 class", not bad_route,
                    str(bad_route[:3])))
        # ---- K ------------------------------------------------------------------------------------------------------
        bad_k, bad_kc, unwritten = [], [], []
        for m in range(1, FUSED_MAX + 1):
            tile_n, n_splits, n_thr = tk.mhc_fused_post_pre_split_config(m, hidden, hc)
            for wk in keys[:3]:
                for kind in ("gauss", "wide"):
                    x, r, p, c = inputs(m, kind)
                    fn = W3[wk][0]
                    w16 = M.weight_for(fn, hc, hidden)
                    raw = []
                    for kern, w in ((tk.mhc_fused_tilelang, fn), (M.mhc_fused_bf16w_tilelang, w16)):
                        yp = torch.full((n_splits, m, n_out), float("nan"), device=dev)
                        rp = torch.full((n_splits, m), float("nan"), device=dev)
                        ro = torch.full((m, hc, hidden), float("nan"), dtype=torch.bfloat16, device=dev)
                        kern(c, r, p.view(m, hc), x, w.view(n_out, hc, hidden), yp, rp, ro, hc, hidden, n_out,
                             n_thr=n_thr, tile_n=tile_n)
                        raw.append((yp, rp, ro))
                    if not all(biteq(torch, a_, b_) for a_, b_ in zip(raw[0], raw[1])):
                        bad_k.append((m, kind))
                    if any(torch.isnan(t_).any() for t_ in raw[1]):
                        unwritten.append((m, kind))
                    cl = [fp32_cls(c, r, p.view(m, hc), x, fn.view(n_out, hc, hidden), hc, hidden, n_out),
                          bf16_cls(c, r, p.view(m, hc), x, w16.view(n_out, hc, hidden), hc, hidden, n_out)]
                    if not (all(biteq(torch, a_, b_) for a_, b_ in zip(cl[0], cl[1]))
                            and all(biteq(torch, a_, b_) for a_, b_ in zip(cl[0], raw[0]))):
                        bad_kc.append((m, kind))
        res.append(("K raw JIT kernels fp32 vs bf16, M = 1..32 at the op's config: identical and fully written",
                    not bad_k and not unwritten, f"bad={bad_k[:4]} unwritten={unwritten[:3]}"))
        res.append(("K production classes fp32 vs bf16 identical, and == the raw launch", not bad_kc, str(bad_kc[:4])))
        # ---- G / F / C / T (GPU only) -----------------------------------------------------------------------------
        if not emu:
            bad_g = []
            for m in CAPTURE_SIZES:
                a = op_args(m, "gauss", keys[0])
                static = list(a)
                ref_inputs = op_args(m, "wide", keys[0])
                ref = call_op(ref_inputs, False)
                call_op(tuple(static), True)                      # eager warm-up (the V2 runner does one before capture)
                st0 = dict(M.stats())
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    cap_out = call_op(tuple(static), True)
                for i_ in (0, 1, 2, 3):
                    static[i_].copy_(ref_inputs[i_])
                gph.replay()
                torch.cuda.synchronize()
                st1 = M.stats()
                if not all(biteq(torch, x, y) for x, y in zip(cap_out, ref)) or st1["converted"] != st0["converted"] \
                        or st1["capture_misses"] != st0["capture_misses"]:
                    bad_g.append(m)
            res.append(("G CUDA-graph capture + replay (all recipe sizes) == gate-off eager; no conversion / capture miss",
                        not bad_g, str(bad_g)))
            # F fallback: a weight first seen inside a capture
            M._reset_for_test()
            m = 3
            a = list(op_args(m, "gauss", keys[0]))
            ref = call_op(tuple(a), False)
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                cap_out = call_op(tuple(a), True)
            gph.replay()
            torch.cuda.synchronize()
            res.append(("F weight first seen inside a capture -> fp32 kernel captured (capture_misses 1), replay == reference",
                        M.stats()["capture_misses"] == 1 and biteq(torch, cap_out[0], ref[0]) and biteq(torch, cap_out[3], ref[3]),
                        str(M.stats())))
            for k in keys:                                        # re-convert eagerly for C / T
                M.weight_for(W3[k][0], hc, hidden)
            # C chain of 89 calls
            n_chain = 89
            chain_w = [W3[keys[i % len(keys)]] for i in range(n_chain)]
            bad_c = []
            for m in (3, 16):
                x0, r0, p0, c0 = inputs(m, "gauss")

                def chain(gate, x_, r_, p_, c_):
                    for fn_, sc_, ba_, no_ in ((w[0], w[1], w[2], w[3]) for w in chain_w):
                        r_, p_, c_, x_ = call_op((x_, r_, p_, c_, fn_, sc_, ba_, 1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, no_, 1e-5), gate)
                    return r_, p_, c_, x_

                ref_state = chain(False, x0, r0, p0, c0)
                sx = [t_.clone() for t_ in (x0, r0, p0, c0)]
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    cap = chain(True, *sx)
                gph.replay()
                torch.cuda.synchronize()
                if not all(biteq(torch, x, y) for x, y in zip(cap, ref_state)):
                    bad_c.append(m)
            res.append(("C 89-call chain in one CUDA graph (M = 3, 16): final state identical", not bad_c, str(bad_c)))
            if not args.no_bench:
                flush = torch.empty(96 * 2**20 // 4, dtype=torch.float32, device=dev)
                rows = []
                for m in (1, 3, 8, 16, 32):
                    tile_n, n_splits, n_thr = tk.mhc_fused_post_pre_split_config(m, hidden, hc)
                    x, r, p, c = inputs(m, "gauss")
                    yp = torch.empty(n_splits, m, n_out, device=dev)
                    rp = torch.empty(n_splits, m, device=dev)
                    ro = torch.empty(m, hc, hidden, dtype=torch.bfloat16, device=dev)
                    times = {}
                    for nm in ("fp32", "bf16"):
                        v = []
                        for i in range(n_chain):
                            w = chain_w[i][0]
                            w16 = M.weight_for(w, hc, hidden)
                            kern, wt = (tk.mhc_fused_tilelang, w) if nm == "fp32" else (M.mhc_fused_bf16w_tilelang, w16)
                            flush.zero_()
                            a_, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            a_.record()
                            kern(c, r, p.view(m, hc), x, wt.view(n_out, hc, hidden), yp, rp, ro, hc, hidden, n_out,
                                 n_thr=n_thr, tile_n=tile_n)
                            b_.record()
                            b_.synchronize()
                            v.append(a_.elapsed_time(b_) * 1000)
                        times[nm] = statistics.median(v)
                    rows.append((m, round(times["fp32"], 2), round(times["bf16"], 2), round((times["fp32"] - times["bf16"]) * n_chain / 1000, 3)))
                    log(f"T M={m}: fp32 {times['fp32']:.2f} us, bf16 {times['bf16']:.2f} us per call (cold) -> "
                        f"{(times['fp32'] - times['bf16']) * n_chain / 1000:.3f} ms per {n_chain}-call forward")
                out["timing_us"] = rows
        else:
            log("[emulate] G / F / C / T need a GPU: skipped")
        npass = sum(1 for r_ in res if r_[1])
        for name, ok, info in res:
            log(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
            out["sections"][name] = {"ok": bool(ok), "info": info}
        out["pass"], out["total"] = npass, len(res)
        if args.out:
            Path(args.out).write_text(json.dumps(clean(out), indent=2))
        log(f"p019 GPU bit-exact check: {npass}/{len(res)} passed")
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
