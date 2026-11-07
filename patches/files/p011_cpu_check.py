#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-011 mhc-prenorm-bm2-threshold] CPU checks (v0.32.0 port). python3 + torch (CPU); no GPU, no TileLang, no vLLM install.

  PYTHONDONTWRITEBYTECODE=1 python3 files/p011_cpu_check.py --vllm-root <v0.32.0 tree>/vllm [--work DIR]
  (needs a Python with torch; without torch only sections A and A5 run, B is a SKIP)

--vllm-root holds the v0.32.0 originals (sha-checked against the .sh pins). Sections:
  A   pins + applier: tilelang_kernels.py == TK_ORIG; apply == the pinned result for the original and the 019 result
      (019's own applier builds the second base); revert is the exact inverse; re-apply and every damaged anchor are
      refused; AST scope: only HcPrenormGemmTileLangKernel.{dispatch,get_warmup_keys,warmup_inputs} changed and one
      module-level assignment was added; the 011/019 results commute.
  A5  the two kernels' arithmetic (AST): the per-token kernel (n_splits=1) and the block_m=2 kernel accumulate x*fn with the
      same index expression, the same cast, the same sqr update and the same warp / cross-warp reductions.
  B   behaviour on the REAL JIT-class machinery (jit_warmup.py, jit_warmup_tilelang_helper.py, the patched
      tilelang_kernels.py exec'd over fake TileLang): for the unset default and thresholds 128/129/200/512/1024/2048:
      the dispatch table over num_tokens 1..8192 equals the closed-form rule; unset == upstream for EVERY num_tokens;
      the warmed compile keys equal the set of keys the dispatch can produce for 1..max_tokens (no key missing, none
      extra); a REAL JitWarmupRegistry warmup completes (no 'different compile key') and compiles exactly those keys.
Exit 0 = ALL PASS.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
SH11 = PDIR / "011-mhc-prenorm-bm2-threshold.sh"
TK_REL = "model_executor/kernels/mhc/tilelang_kernels.py"
RESULTS = {"pass": 0, "fail": 0, "skip": 0, "failed": []}
sys.dont_write_bytecode = True


def check(name, ok, info=""):
    ok = bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    RESULTS["pass" if ok else "fail"] += 1
    if not ok:
        RESULTS["failed"].append(name)
    return ok


def skip(name, why):
    print(f"SKIP {name}  [{why}]", flush=True)
    RESULTS["skip"] += 1


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def pin(name, sh=SH11):
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", Path(sh).read_text(), re.M)
    if not m:
        raise SystemExit(f"no pin {name} in {sh}")
    return m.group(1)


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def refused(P, text):
    try:
        P.apply(text)
    except P.Refused:
        return True
    return False


def kernel_fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    root = Path(a.vllm_root)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="p011-"))
    work.mkdir(parents=True, exist_ok=True)
    tk_text = (root / TK_REL).read_text()
    P = load_file_module("p011_apply", HERE / "p011_apply.py")
    P19 = load_file_module("p019_apply", HERE / "p019_apply.py")

    print("== A pins + applier")
    check("A tilelang_kernels.py == TK_ORIG", sha_text(tk_text) == pin("TK_ORIG"))
    check("A files/p011_apply.py == APPLIER_SHA", hashlib.sha256((HERE / "p011_apply.py").read_bytes()).hexdigest()
          == pin("APPLIER_SHA"))
    r_orig = P.apply(tk_text)
    check("A apply(original) == TK_P011", sha_text(r_orig) == pin("TK_P011"), sha_text(r_orig)[:16])
    b19 = P19.apply_tk(tk_text)
    check("A 019's applier on the original == TK_P019", sha_text(b19) == pin("TK_P019"))
    r_19 = P.apply(b19)
    check("A apply(019 result) == TK_P011_P019", sha_text(r_19) == pin("TK_P011_P019"), sha_text(r_19)[:16])
    check("A 011 and 019 commute (019(011(x)) == 011(019(x)))", P19.apply_tk(r_orig) == r_19)
    check("A revert(apply(original)) == original", P.revert(r_orig) == tk_text)
    check("A revert(apply(019 result)) == the 019 result", P.revert(r_19) == b19)
    check("A re-apply refused", refused(P, r_orig))
    for name, old, _new in P.EDITS:
        dmg = tk_text.replace(old, old.replace("1024", "1000").replace("ENABLE_PDL", "ENABLE_PDX")
                              .replace("num_tokens = 128", "num_tokens = 127"), 1)
        check(f"A damaged anchor refused: {name}", dmg != tk_text and refused(P, dmg))
        dup = tk_text + "\n" + old
        check(f"A duplicated anchor refused: {name}", refused(P, dup))
    # AST scope
    t0, t1 = ast.parse(tk_text), ast.parse(r_orig)
    c0 = next(n for n in t0.body if isinstance(n, ast.ClassDef) and n.name == P.CLASS)
    c1 = next(n for n in t1.body if isinstance(n, ast.ClassDef) and n.name == P.CLASS)
    ch = sorted(m0.name for m0, m1 in zip(c0.body, c1.body)
                if isinstance(m0, ast.FunctionDef) and ast.dump(m0) != ast.dump(m1))
    d0 = {getattr(n, "name", None) or ast.dump(n): ast.dump(n) for n in t0.body if n is not c0}
    d1 = {getattr(n, "name", None) or ast.dump(n): ast.dump(n) for n in t1.body if n is not c1}
    added = [k for k in d1 if k not in d0]
    check("A AST scope: only dispatch / get_warmup_keys / warmup_inputs changed; one assignment added; nothing else",
          ch == ["dispatch", "get_warmup_keys", "warmup_inputs"] and len(c0.body) == len(c1.body)
          and len(added) == 1 and "_P011_BM2_MIN_T" in added[0] and all(d0[k] == d1[k] for k in d0),
          f"changed={ch} added={len(added)}")

    print("== A5 kernel arithmetic (AST): per-token kernel vs block_m=2 kernel")
    k_tok = kernel_fn(t0, "hc_prenorm_gemm_tilelang")
    k_bm = kernel_fn(t0, "hc_prenorm_gemm_block_m_tilelang")
    src_tok, src_bm = ast.unparse(k_tok), ast.unparse(k_bm)
    must_both = ["T.warp_reduce_sum(", "T.sync_threads()", "lane = tid % 32", "warp_id = tid // 32",
                 "num_warps = n_thr // 32", "k_iters = hc_hidden_size // n_thr"]
    check("A5 both kernels: warp_reduce_sum, sync, lane/warp split, k_iters = K // n_thr",
          all(s in src_tok and s in src_bm for s in must_both) or all(
              s in src_tok and s in src_bm for s in must_both[:5]) and "k_iters" in src_tok and "k_iters" in src_bm)
    cast_ok = ("T.cast(x[i_n, i_k], T.float32)" in src_tok) and ("T.cast(x[token_idx, i_k], T.float32)" in src_bm)
    check("A5 both kernels widen x with an explicit T.cast(.., T.float32) (same operand type as the other kernel)", cast_ok)
    check("A5 per-token accumulates acc[i_o] += x_val * fn[out_idx, i_k]; bm2 acc[i_m, i_o] += x_val * fn_val[i_o] with "
          "fn_val[i_o] = fn[out_idx, i_k] (same operand, same order of the k loop)",
          "acc[i_o] += x_val * fn[out_idx, i_k]" in src_tok and "acc[i_m, i_o] += x_val * fn_val[i_o]" in src_bm
          and "fn_val[i_o] = fn[out_idx, i_k]" in src_bm)
    check("A5 k index: per-token i_k = i_s * k_per_split + it * n_thr + tid (i_s = 0, k_per_split = K at n_splits = 1); "
          "bm2 i_k = it * n_thr + tid",
          "i_k = i_s * k_per_split + it * n_thr + tid" in src_tok and "i_k = it * n_thr + tid" in src_bm
          and "k_per_split = hc_hidden_size // n_splits" in src_tok)
    check("A5 sqr accumulates sqr += x_val * x_val (token row) in both",
          "sqr[0] += x_val * x_val" in src_tok and "sqr[i_m] += x_val * x_val" in src_bm)
    check("A5 final reduction: warp 0 sums warp partials in order i_w = 0..num_warps-1 from 0.0 in both",
          "for i_w in T.unroll(num_warps)" in src_tok and "for i_w in T.unroll(num_warps)" in src_bm
          and "init=0.0" in src_tok and "T.clear(reduced_acc)" in src_bm and "T.clear(reduced_sqr)" in src_bm)

    print("== A5T kernel TIR (real TileLang 0.1.12 frontend, no GPU): per-token vs block_m=2 arithmetic structure")
    try:
        import torch  # noqa: F401
    except ImportError:
        skip("A5T", "torch is not installed")
    else:
        sys.path.insert(0, str(HERE))
        import pmhc_testlib as L0

        st, info = L0.tir_probe(root)
        if st == "skip":
            skip("A5T TIR facts", f"{info.get('skip')}; set TILELANG_PROBE_PATH (see pmhc_tir_probe.py)")
        else:
            facts = info.get("prenorm", {})
            check("A5T the TIR of both kernels: same k order (it = 0..31, i_k = it*512 + tid), same explicit widening, "
                  "same `acc + x_val * fn` update, same warp_reduce_sum, same in-order cross-warp sum from 0.0",
                  st == "ok" and bool(facts) and all(facts.values()), json.dumps({k: v for k, v in facts.items() if not v}))

    print("== B behaviour on the real JIT-class machinery")
    try:
        import torch  # noqa: F401
    except ImportError:
        skip("B real-class checks", "torch is not installed (use a Python with torch)")
        return finish()
    sys.path.insert(0, str(HERE))
    import pmhc_testlib as L

    max_tokens = 8192
    cfg = L.fake_vllm_config(max_tokens)
    patched_path = work / "tilelang_kernels.p011.py"
    patched_path.write_text(r_orig)
    base_env = L.build_env(root)
    K0 = base_env.tk._HC_PRENORM_GEMM_TILELANG_KERNEL

    def table(env, K, ns=None):
        with env.activate():
            rows = {}
            for n in ns or range(1, max_tokens + 1):
                k = K.dispatch(num_tokens=n, hc_hidden_size=16384, hidden_size=4096, hc_mult=4, n_out=24)
                rows[n] = (k.n_thr, k.tile_n, k.n_splits, k.use_block_m, k.block_m)
            return rows

    base_table = table(base_env, K0)
    for thr in (None, 128, 129, 200, 512, 1024, 2048):
        saved = os.environ.get("VLLM_GB10_MHC_BM2_MIN_T")
        if thr is None:
            os.environ.pop("VLLM_GB10_MHC_BM2_MIN_T", None)
        else:
            os.environ["VLLM_GB10_MHC_BM2_MIN_T"] = str(thr)
        try:
            env = L.build_env(root, tk_path=patched_path)
        finally:
            if saved is None:
                os.environ.pop("VLLM_GB10_MHC_BM2_MIN_T", None)
            else:
                os.environ["VLLM_GB10_MHC_BM2_MIN_T"] = saved
        eff = 1024 if thr is None else thr
        K = env.tk._HC_PRENORM_GEMM_TILELANG_KERNEL
        check(f"B[{thr}] module constant _P011_BM2_MIN_T == {eff}", env.tk._P011_BM2_MIN_T == eff)
        tb = table(env, K)

        def rule(n):
            bm = n >= eff
            small = (not bm) and n < 128
            return (1024 if small else 512, 4 if small else 12, 1, bm, 2 if bm else 1)

        bad = [n for n in tb if tb[n] != rule(n)]
        check(f"B[{thr}] dispatch table over num_tokens 1..{max_tokens} == the closed-form rule", not bad, str(bad[:5]))
        if thr in (None, 1024):
            diff = [n for n in tb if tb[n] != base_table[n]]
            check(f"B[{thr}] == upstream dispatch for EVERY num_tokens (unset / 1024 is inert)", not diff, str(diff[:5]))
        with env.activate():
            keys = set(K.get_warmup_keys(cfg, hidden_size=4096, hc_mult=4, n_out=24))
            reach = {K.dispatch(num_tokens=n, hc_hidden_size=16384, hidden_size=4096, hc_mult=4, n_out=24)
                     for n in range(1, max_tokens + 1)}
        check(f"B[{thr}] warmed keys == the keys the dispatch can produce for 1..{max_tokens} ({len(reach)})",
              keys == reach, f"missing={len(reach - keys)} extra={len(keys - reach)}")
        for mt in (64, 127, 128, 130, 1500):
            with env.activate():
                keys_m = set(K.get_warmup_keys(L.fake_vllm_config(mt), hidden_size=4096, hc_mult=4, n_out=24))
                reach_m = {K.dispatch(num_tokens=n, hc_hidden_size=16384, hidden_size=4096, hc_mult=4, n_out=24)
                           for n in range(1, mt + 1)}
            if keys_m != reach_m:
                check(f"B[{thr}] warmed keys == reachable keys at max_num_batched_tokens={mt}", False,
                      f"missing={len(reach_m - keys_m)} extra={len(keys_m - reach_m)}")
                break
        else:
            check(f"B[{thr}] warmed keys == reachable keys at max_num_batched_tokens in (64, 127, 128, 130, 1500)", True)
        with env.activate():
            env.reset_jits()
            try:
                L.run_registry_warmup(env, lambda: K.register_warmup(cfg, hidden_size=4096, hc_mult=4, n_out=24))
                ok, err = True, ""
            except Exception as e:  # noqa: BLE001
                ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
            n_comp = sum(len(env.jit(n).compiles) for n in ("hc_prenorm_gemm_tilelang",
                                                            "hc_prenorm_gemm_block_m_tilelang"))
        check(f"B[{thr}] a REAL registry warmup completes and compiles each reachable key once ({n_comp})",
              ok and n_comp == len(reach), err or f"compiles={n_comp} keys={len(reach)}")
        if thr is not None and thr != 1024:
            bm_n = [args[9] for args, _kw in env.jit("hc_prenorm_gemm_block_m_tilelang").compiles]
            check(f"B[{thr}] the block_m=2 key is compiled with block_m=2", bm_n and all(b == 2 for b in bm_n))

    print("== C negative controls: the checks above must notice a patch with one edit missing")
    # EDITS[1..3]: dispatch, warmup lambda, warmup input. The lambda and the input only matter for thresholds above
    # the old 1,024 bucket (below it the sample points 128 / 1,024 already reach every key).
    for skipped, thr in ((1, 200), (2, 2048), (3, 2048)):
        txt = tk_text
        for i, (_n, old, new) in enumerate(P.EDITS):
            if i != skipped:
                txt = txt.replace(old, new, 1)
        mp = work / f"tilelang_kernels.mut{skipped}.py"
        mp.write_text(txt)
        os.environ["VLLM_GB10_MHC_BM2_MIN_T"] = str(thr)
        try:
            env = L.build_env(root, tk_path=mp)
        finally:
            os.environ.pop("VLLM_GB10_MHC_BM2_MIN_T", None)
        K = env.tk._HC_PRENORM_GEMM_TILELANG_KERNEL
        caught = []
        with env.activate():
            keys = set(K.get_warmup_keys(cfg, hidden_size=4096, hc_mult=4, n_out=24))
            reach = {K.dispatch(num_tokens=n, hc_hidden_size=16384, hidden_size=4096, hc_mult=4, n_out=24)
                     for n in range(1, max_tokens + 1)}
            if keys != reach:
                caught.append("keys differ")
            try:
                L.run_registry_warmup(env, lambda: K.register_warmup(cfg, hidden_size=4096, hc_mult=4, n_out=24))
            except Exception as e:  # noqa: BLE001
                caught.append(type(e).__name__)
        check(f"C mutant without edit {skipped + 1} ({P.EDITS[skipped][0]}) is caught at threshold {thr}", bool(caught),
              ", ".join(caught))
    print("   (at thresholds 128..1,024 edits 3 and 4 change nothing observable: the old 128 / 1,024 sample points already "
          "reach every key; they keep the warmup consistent for any threshold)")
    return finish()


def finish():
    print(f"---- {RESULTS['pass']} passed, {RESULTS['fail']} failed, {RESULTS['skip']} skipped"
          + (f": {RESULTS['failed']}" if RESULTS["fail"] else ""))
    print("ALL PASS" if not RESULTS["fail"] else "SOME FAILED")
    return 1 if RESULTS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
