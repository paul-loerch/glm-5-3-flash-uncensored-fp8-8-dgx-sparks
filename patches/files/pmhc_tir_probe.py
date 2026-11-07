#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""pmhc_tir_probe.py - TileLang-frontend (TIR level) evidence for the mHC family's bit-exactness claims, runnable on a Mac.

TileLang 0.1.12 (the version vLLM v0.32.0 pins for CUDA, requirements/cuda.txt) ships a macOS arm64 wheel. Its Python frontend
elaborates a @tilelang.jit kernel into a TIR PrimFunc (`get_tir(*args)`) without a GPU, nvcc or the CUDA backend. This script builds
the TIR of the kernels the patches depend on - straight from the REAL v0.32.0 source text and from the vendored 019 module - and
checks the arithmetic structure:

  019   mhc_fused_tilelang (fp32 weight_t) vs mhc_fused_bf16w_tilelang (bf16 weight_t), for the op's two (tile_n, n_thr) configs:
        the TIR differs in exactly three lines - the function name, the weight_t buffer dtype and the FMA statement, where the bf16
        element is wrapped in an explicit T.Cast("float32", ...) before the multiplication with the fp32 new_r (an exact
        widening). Everything else - loop nest, index expressions, reductions, shared-memory staging - is textually identical.
  011   hc_prenorm_gemm_tilelang (per-token, n_splits=1) vs hc_prenorm_gemm_block_m_tilelang: per (row, column) both run
        `acc = acc + x_val * fn[col, i_k]` (x_val = T.Cast("float32", x[row, i_k])) for it = 0..31 in order with
        i_k = it * 512 + tid, then T.warp_reduce_sum, then warp 0 sums the 16 warp partials in order from 0.0, in both kernels;
        sqrsum likewise.
  022   the same bm2 TIR: `num_tokens` appears only in the row-validity predicate (token_idx < num_tokens) and in the output
        strides; no arithmetic reads it, the row offset or another row.

Usage (needs torch + tilelang importable; PYTHONPATH or --tilelang-path):
  python3 pmhc_tir_probe.py --tree <v0.32.0>/vllm [--module files/gb10_mhc_fn_bf16.py] [--tilelang-path DIR] [--out-dir DIR]
Prints one JSON object; exit 0 = all structural facts hold, 1 = a fact failed, 3 = tilelang not importable (SKIP).
Install on a Mac: pip install --target DIR tilelang==0.1.12 "apache-tvm-ffi>=0.1.10,<=0.1.11" cloudpickle ml-dtypes psutil tqdm
  typing-extensions "z3-solver>=4.13.0,<4.15.5" setuptools   (torch/numpy from the interpreter's own environment).
"""
import argparse
import ast
import difflib
import json
import re
import sys
import tempfile
from pathlib import Path


def seg(src, name):
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([d.lineno for d in fn.decorator_list] + [fn.lineno])
    return "\n".join(src.splitlines()[start - 1:fn.end_lineno])


HEAD = '''from __future__ import annotations
import math
import tilelang
import tilelang.language as T

ENABLE_PDL = False
tilelang_jit = tilelang.jit


'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--module", default=str(Path(__file__).resolve().parent / "gb10_mhc_fn_bf16.py"))
    ap.add_argument("--tilelang-path", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.tilelang_path:
        sys.path.insert(0, a.tilelang_path)
    try:
        import tilelang  # noqa: F401
        import torch
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"skip": f"tilelang/torch not importable: {type(e).__name__}: {e}"[:200]}))
        return 3
    tk_src = (Path(a.tree) / "model_executor/kernels/mhc/tilelang_kernels.py").read_text()
    mod_src = Path(a.module).read_text()
    wd = Path(a.out_dir) if a.out_dir else Path(tempfile.mkdtemp(prefix="pmhc-tir-"))
    wd.mkdir(parents=True, exist_ok=True)
    probe = wd / "mhc_probe_kernels.py"
    probe.write_text(HEAD + "\n\n\n".join([seg(tk_src, "mhc_fused_tilelang"), seg(mod_src, "mhc_fused_bf16w_tilelang"),
                                          seg(tk_src, "hc_prenorm_gemm_tilelang"),
                                          seg(tk_src, "hc_prenorm_gemm_block_m_tilelang")]) + "\n")
    sys.path.insert(0, str(wd))
    import mhc_probe_kernels as kp

    hc, hidden, n_out = 4, 4096, 24
    K = hc * hidden
    out = {"tilelang": getattr(tilelang, "__version__", "?"), "fused": {}, "prenorm": {}}
    ok = True
    # ---- 019: fp32 vs bf16 fused kernel ----
    for tag, (M, tile_n, n_thr, n_splits) in {"M3_tile2": (3, 2, 128, 8), "M24_tile6": (24, 6, 128, 8)}.items():
        comb = torch.zeros(M, hc, hc)
        res = torch.zeros(M, hc, hidden, dtype=torch.bfloat16)
        post = torch.zeros(M, hc)
        x = torch.zeros(M, hidden, dtype=torch.bfloat16)
        yp = torch.zeros(n_splits, M, n_out)
        rp = torch.zeros(n_splits, M)
        ro = torch.zeros(M, hc, hidden, dtype=torch.bfloat16)
        w32 = torch.zeros(n_out, hc, hidden)
        w16 = torch.zeros(n_out, hc, hidden, dtype=torch.bfloat16)
        s32 = kp.mhc_fused_tilelang.get_tir(comb, res, post, x, w32, yp, rp, ro, hc, hidden, n_out, n_thr=n_thr, tile_n=tile_n).script()
        s16 = kp.mhc_fused_bf16w_tilelang.get_tir(comb, res, post, x, w16, yp, rp, ro, hc, hidden, n_out, n_thr=n_thr,
                                                  tile_n=tile_n).script()
        (wd / f"tir_fused_fp32_{tag}.txt").write_text(s32)
        (wd / f"tir_fused_bf16_{tag}.txt").write_text(s16)
        l32, l16 = s32.splitlines(), s16.splitlines()
        diff = [d for d in difflib.unified_diff(l32, l16, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("---", "+++")]
        minus = [d[1:] for d in diff if d[0] == "-"]
        plus = [d[1:] for d in diff if d[0] == "+"]
        fma_old = [m for m in minus if "acc[n] = acc[n] + weight_t_1[" in m]
        fma_new = [m for m in plus if 'acc[n] = acc[n] + T.Cast("float32", weight_t_1[' in m]
        cast_exact = bool(fma_old and fma_new and fma_old[0].replace("weight_t_1[", 'T.Cast("float32", weight_t_1[').replace(
            "] * new_r[j]", "]) * new_r[j]") == fma_new[0])
        defs = [m for m in minus if m.lstrip().startswith("def mhc_fused_tilelang(")]
        defs_new = [m for m in plus if m.lstrip().startswith("def mhc_fused_bf16w_tilelang(")]
        buf = [m for m in plus if "weight_t_1 = T.match_buffer" in m and '"bfloat16"' in m]
        fact = {"diff_lines": len(diff), "differing_statements": {"def": len(defs), "buffer_dtype": len(buf), "fma": len(fma_new)},
                "fma_is_explicit_cast_of_the_same_expression": cast_exact,
                "other_fp32_math_uses_weight_t": sum(1 for l in l16 if "weight_t_1[" in l)}
        good = (len(diff) == 6 and len(defs) == len(defs_new) == len(buf) == len(fma_new) == len(fma_old) == 1 and cast_exact
                and fact["other_fp32_math_uses_weight_t"] == 1)
        fact["ok"] = good
        ok &= good
        out["fused"][tag] = fact
    # ---- 011 / 022: the two pre-norm kernels ----
    T_ = 1024
    x = torch.zeros(T_, K, dtype=torch.bfloat16)
    fn = torch.zeros(n_out, K)
    o = torch.zeros(1, T_, n_out)
    s = torch.zeros(1, T_)
    t_tok = kp.hc_prenorm_gemm_tilelang.get_tir(x, fn, o, s, hidden, hc, n_out, 512, 12, 1).script()
    t_bm = kp.hc_prenorm_gemm_block_m_tilelang.get_tir(x, fn, o, s, hidden, hc, n_out, 512, 12, 2).script()
    (wd / "tir_prenorm_tok.txt").write_text(t_tok)
    (wd / "tir_prenorm_bm2.txt").write_text(t_bm)
    facts = {}
    facts["both_k_loop_is_32_iterations_in_order"] = "for it in range(32):" in t_tok and "for it in range(32):" in t_bm
    facts["k_index_tok"] = "i_k: T.int32 = bz * 16384 + it * 512 + tid" in t_tok            # bz == 0 at n_splits = 1
    facts["k_index_bm2"] = "i_k: T.int32 = it * 512 + tid" in t_bm
    facts["tok_grid_bz_is_1"] = 'for bz in T.thread_binding(1, thread="blockIdx.z")' in t_tok
    facts["x_widened_explicitly_in_both"] = ('x_val: T.float32 = T.Cast("float32", x_1[bx, i_k])' in t_tok
                                             and 'x_val: T.float32 = T.Cast("float32", x_1[token_idx, i_k])' in t_bm)
    facts["acc_update_tok"] = "acc[i_o] = acc[i_o] + x_val * fn_1[out_idx, i_k]" in t_tok
    facts["acc_update_bm2"] = ("acc[i_m, i_o] = acc[i_m, i_o] + x_val * fn_val[i_o]" in t_bm
                               and "fn_val[i_o] = fn_1[out_idx, i_k]" in t_bm)
    facts["sqr_update"] = "sqr[0] = sqr[0] + x_val * x_val" in t_tok and "sqr[i_m] = sqr[i_m] + x_val * x_val" in t_bm
    facts["warp_reduce_in_both"] = "acc[i_o] = T.warp_reduce_sum(acc[i_o])" in t_tok and \
        "acc[i_m, i_o] = T.warp_reduce_sum(acc[i_m, i_o])" in t_bm
    facts["cross_warp_sum_in_order_from_zero"] = (
        "reduced_acc[0] = T.float32(0.0)" in t_tok and "for i_w in T.unroll(16" in t_tok
        and "reduced_acc[0] = reduced_acc[0] + warp_acc[i_w, lane]" in t_tok
        and "T.fill(T.region(reduced_acc[0], 2, 2), 0)" in t_bm
        and "reduced_acc[i_m] = reduced_acc[i_m] + warp_acc[i_w, i_m, lane]" in t_bm)
    # num_tokens appears only in the row predicate / grid / output strides (022's row independence)
    uses = [l.strip() for l in t_bm.splitlines() if "num_tokens" in l]
    allowed = ("num_tokens = T.int32()", "x_1 = T.match_buffer", "out_1 = T.match_buffer", "sqrsum_1 = T.match_buffer",
               "for bx in T.thread_binding((num_tokens + 2 - 1) // 2", "if token_idx < num_tokens:")
    facts["bm2_num_tokens_only_in_predicate_grid_strides"] = all(any(u.startswith(p) for p in allowed) for u in uses)
    arith = [l for l in t_bm.splitlines() if ("acc[" in l or "sqr[" in l or "reduced_" in l) and "num_tokens" in l]
    facts["bm2_no_arithmetic_reads_num_tokens"] = not arith
    out["prenorm"] = facts
    ok &= all(facts.values())
    out["ok"] = bool(ok)
    out["out_dir"] = str(wd)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
