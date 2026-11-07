#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p011_gpu_check.py - patch 011 (VLLM_GB10_MHC_BM2_MIN_T) on-device proof on the v0.32.0 image's own TileLang: the block_m=2
pre-norm GEMM kernel is BIT-IDENTICAL to the generic per-token kernel (n_thr 512, tile_n 12, n_splits 1) for every T the
patch re-routes (128 <= T < 1,024). WRITTEN, NOT RUN in the authoring session (no GPU / cluster access).

Run INSIDE a rank container while the server is idle (the kernels compile on first use, cached under TILELANG_CACHE_DIR):

  ssh node1 'docker exec <rank-0 container> python3 /opt/recipe-patches/files/p011_gpu_check.py --out /tmp/p011-gpu.json'

It imports the INSTALLED vllm.model_executor.kernels.mhc.tilelang_kernels (patched or not; the numbers do not depend on the
routing, it calls the two JIT functions directly) and checks, with NaN-poisoned outputs and bitwise comparisons (int32 views):
  K  kernel   hc_prenorm_gemm_block_m_tilelang(block_m=2) vs hc_prenorm_gemm_tilelang(n_thr=512, tile_n=12, n_splits=1) for
              T in --ts (default 1 .. 3,853 incl. every boundary 127/128/129, 255/256/257, 511/512/513, 1023/1024/1025) x x-kinds
              {gauss, wide, outlier} x fn {random bf16-exact, random full-mantissa fp32}: out + sqrsum identical, written
  C  class    HcPrenormGemmTileLangKernel(x, fn, out, sqrsum, hidden, hc) (the call the model makes) == the kernel its own
              dispatch() names, bitwise, for the same T list; and its dispatch routes T >= the installed threshold to
              block_m=2 (reports the threshold: 1,024 unpatched, VLLM_GB10_MHC_BM2_MIN_T when 011 is installed)
  I  info     the generic small-T kernel (n_thr 1024, tile_n 4, what T < 128 uses) vs the per-token kernel: reported only
              (a different accumulation order: NOT expected to be bit-identical; this is why 011 keeps T < 128 on it)
  T  timing   (skip with --no-bench) DRAM-cold (L2 flushed) per call: per-token vs block_m=2 at T = 128 .. 1,023
Exit: 0 PASS | 1 FAIL | 2 setup error.

CPU validation of this harness (fake row-local kernels over the REAL classes, no GPU):
  PYTHONDONTWRITEBYTECODE=1 python3 files/p011_gpu_check.py --emulate-tree <v0.32.0>/vllm
"""
import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TS_DEFAULT = (1, 8, 64, 127, 128, 129, 255, 256, 257, 300, 511, 512, 513, 700, 1000, 1023, 1024, 1025, 1027, 2048, 3840, 3853)


def log(*a):
    print(*a, flush=True)


def biteq(torch, a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return bool(torch.equal(a.contiguous().view(iv), b.contiguous().view(iv)))


def mkx(torch, kind, T, K, dev, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, K, generator=g)
    if kind == "wide":
        x = x * (2.0 ** torch.randint(-6, 7, (T, K), generator=g).float())
    elif kind == "outlier":
        x = x * 0.5
        x[:, ::97] *= 300.0
    return x.to(torch.bfloat16).to(dev)


def mkfn(torch, kind, n_out, K, dev, seed):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(n_out, K, generator=g) * 0.02
    if kind == "bf16":
        fn = fn.to(torch.bfloat16).float()
    return fn.contiguous().to(dev)


def run(args):
    import torch
    dev = torch.device("cuda" if not args.emulate_tree else "cpu")
    if args.emulate_tree:
        sys.path.insert(0, str(HERE))
        import pmhc_testlib as L
        envx = L.build_env(Path(args.emulate_tree))
        ctx = envx.activate()
    else:
        import contextlib
        ctx = contextlib.nullcontext()
    out = {"when": dt.datetime.now(dt.timezone.utc).isoformat(), "emulate": bool(args.emulate_tree), "sections": {}}
    res = []
    with ctx:
        import importlib
        tk = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang_kernels")
        Kc = tk._HC_PRENORM_GEMM_TILELANG_KERNEL
        hidden, hc, n_out = args.hidden, 4, 24
        K = hidden * hc
        ts = [int(t) for t in args.ts.split(",")]
        thr = getattr(tk, "_P011_BM2_MIN_T", 1024)
        out["installed_threshold"] = thr
        out["patch_011_installed"] = hasattr(tk, "_P011_BM2_MIN_T")
        log(f"installed threshold: {thr} ({'patch 011' if out['patch_011_installed'] else 'upstream 1,024'})")

        def poison(shape):
            return torch.full(shape, float("nan"), dtype=torch.float32, device=dev)

        bad_k, bad_c, nonwritten, bad_route = [], [], [], []
        info_small = {}
        for T in ts:
            for xk in ("gauss", "wide", "outlier"):
                x = mkx(torch, xk, T, K, dev, 1000 + T)
                for fk in ("bf16", "full"):
                    fn = mkfn(torch, fk, n_out, K, dev, 2000 + T)
                    o1, s1 = poison((1, T, n_out)), poison((1, T))
                    o2, s2 = poison((1, T, n_out)), poison((1, T))
                    tk.hc_prenorm_gemm_tilelang(x, fn, o1, s1, hidden, hc, n_out, 512, 12, 1)
                    tk.hc_prenorm_gemm_block_m_tilelang(x, fn, o2, s2, hidden, hc, n_out, 512, 12, 2)
                    if not (biteq(torch, o1, o2) and biteq(torch, s1, s2)):
                        bad_k.append((T, xk, fk))
                    if torch.isnan(o2).any() or torch.isnan(s2).any() or torch.isnan(o1).any():
                        nonwritten.append((T, xk, fk))
                    o3, s3 = poison((1, T, n_out)), poison((1, T))
                    Kc(x, fn, o3, s3, hidden, hc)
                    key = Kc.dispatch(num_tokens=T, hc_hidden_size=K, hidden_size=hidden, hc_mult=hc, n_out=n_out)
                    if key.use_block_m != (T >= thr):
                        bad_route.append((T, key.use_block_m))
                    ref = (o2, s2) if key.use_block_m else (o1, s1)
                    if key.n_thr == 1024:                                   # small-T kernel: its own order
                        ref = None
                    if ref is not None and not (biteq(torch, o3, ref[0]) and biteq(torch, s3, ref[1])):
                        bad_c.append((T, xk, fk))
                    if xk == "gauss" and fk == "full" and T >= 8 and hidden * hc % 1024 == 0:
                        o4, s4 = poison((1, T, n_out)), poison((1, T))
                        tk.hc_prenorm_gemm_tilelang(x, fn, o4, s4, hidden, hc, n_out, 1024, 4, 1)
                        info_small[T] = int((o4.view(torch.int32) != o1.view(torch.int32)).sum())
        res.append(("K block_m=2 kernel == per-token kernel (bitwise), T in the list x 3 x-kinds x 2 fn kinds, all written",
                    not bad_k and not nonwritten, f"bad={bad_k[:4]} unwritten={nonwritten[:3]}"))
        res.append(("C the class call (the model's) == the kernel its own dispatch names (bitwise); routing == T >= threshold",
                    not bad_c and not bad_route, f"bad={bad_c[:4]} route={bad_route[:4]}"))
        log(f"I small-T kernel (n_thr 1024, tile_n 4) vs per-token: mismatching elements per T (info, not a check): {info_small}")
        out["info_small_tile_vs_pertoken_mismatch"] = info_small
        if not args.no_bench and dev.type == "cuda":
            flush = torch.empty(96 * 2**20 // 4, dtype=torch.float32, device=dev)
            rows = []
            for T in (128, 256, 512, 1023):
                x = mkx(torch, "gauss", T, K, dev, 5)
                fn = mkfn(torch, "bf16", n_out, K, dev, 6)
                o, s = torch.empty(1, T, n_out, device=dev), torch.empty(1, T, device=dev)
                tt = {}
                for nm, f in (("per_token", lambda: tk.hc_prenorm_gemm_tilelang(x, fn, o, s, hidden, hc, n_out, 512, 12, 1)),
                              ("block_m2", lambda: tk.hc_prenorm_gemm_block_m_tilelang(x, fn, o, s, hidden, hc, n_out, 512, 12, 2))):
                    v = []
                    for _ in range(9):
                        flush.zero_()
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        f()
                        b.record()
                        b.synchronize()
                        v.append(a.elapsed_time(b) * 1000)
                    tt[nm] = sorted(v)[len(v) // 2]
                rows.append((T, round(tt["per_token"], 1), round(tt["block_m2"], 1)))
                log(f"T timing T={T}: per-token {tt['per_token']:.1f} us, block_m=2 {tt['block_m2']:.1f} us (cold, median of 9)")
            out["timing_us"] = rows
    npass = sum(1 for r in res if r[1])
    for name, ok, info in res:
        log(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
        out["sections"][name] = {"ok": bool(ok), "info": info}
    out["pass"], out["total"] = npass, len(res)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    log(f"p011 GPU check: {npass}/{len(res)} passed")
    return 0 if npass == len(res) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ts", default=",".join(str(t) for t in TS_DEFAULT))
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--emulate-tree", default=None, help="a v0.32.0 `vllm` dir: run on CPU over fake row-local kernels")
    args = ap.parse_args()
    if args.emulate_tree:
        args.hidden, args.ts = 128, "1,8,64,127,128,129,255,256,257,300,511,512,513,1000,1023,1024,1025"
    try:
        return run(args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
