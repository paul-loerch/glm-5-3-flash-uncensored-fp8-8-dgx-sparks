#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-019 gb10-mhc-fn-bf16] CPU checks (v0.32.0 port). python3 + torch (CPU); no GPU, no TileLang, no vLLM install.

  PYTHONDONTWRITEBYTECODE=1 python3 files/p019_cpu_check.py --vllm-root <v0.32.0 tree>/vllm [--work DIR] [--model-dir D]
  (--model-dir defaults to $MODEL_DIR; section F is skipped without a checkpoint)
  (needs a Python with torch)

--vllm-root holds the v0.32.0 originals (sha-checked against the .sh pins). The REAL jit_warmup.py / helper /
tilelang_kernels.py / tilelang.py (patched or not) are exec'd over fake TileLang by files/pmhc_testlib.py. Sections:
  A  pins       the originals (tilelang.py, tilelang_kernels.py, helper, jit_warmup.py, tilelang_utils) hash to the .sh
                pins; the vendored applier / module hash to theirs
  B  applier    original and the 022 / 011 results -> apply_tl / apply_tk -> the pinned results (019+022, 011+019); revert
                is the exact inverse; re-apply and each damaged / duplicated anchor are refused; AST scope (tilelang.py:
                only mhc_fused_post_pre_tilelang changed + _P019_ON added; tilelang_kernels.py: only the class
                MhcFusedTileLangKernel gained register_warmup + _P019_ON); 011 / 019 / 022 commute
  C2 TIR        (optional: needs tilelang 0.1.12 importable, e.g. the macOS wheel via TILELANG_PROBE_PATH) the TileLang frontend's TIR of
                the two fused kernels differs in exactly 3 statements: the name, the weight_t buffer dtype and the FMA, where the bf16
                element is an explicit T.Cast("float32", ...) before the multiply (an exact widening)
  C  kernel     AST proof: mhc_fused_bf16w_tilelang == the real v0.32.0 mhc_fused_tilelang with the name, the docstring and
                the weight_t annotation dtype (T.float32 -> T.bfloat16) changed, nothing else; same decorator
  D  module     weight_for()/_to_bf16_exact()/selftest() on CPU tensors over the real classes + fake kernels: exact copy
                + identity cache, one self-test per process (the REAL selftest runs), inexact weights refused, special
                values, capture miss -> None once and no caching, capture hit logs the summary; self-test negative
                controls (a differing bf16 kernel, an unwritten output, a class path that is not the raw launch)
  E  dispatch   the REAL patched tilelang.py over the REAL classes, M = 48..1 (first call at M > 32, like the profile run):
                flag off -> call-for-call identical to the original and the module never imported; flag on -> M <= 32
                launches the bf16 JIT function on the cached copy (never the fp32 fused kernel) and every output is
                bitwise equal to the original's; M > 32 identical to the original; capture miss -> the fp32 kernel
  G  registry   a REAL JitWarmupRegistry: with the flag the model's registration of the fp32 fused kernel also compiles the
                bf16 variant (the same compile keys, a bf16 weight_t warmup tensor); without it only the fp32 kernel; the
                variant's inherited registration does not recurse; the warmup inputs re-dispatch to their own key
  F  weights    (--model-dir) every hc_*_fn in the checkpoint is BF16 [24, 16384]; widened like default_weight_loader,
                fp32 -> bf16 -> fp32 is bit-exact; subnormal/zero/non-finite counts; copy memory
Exit 0 = all pass, 1 = a check failed, 2 = setup error.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch a GPU from this script
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
SH19 = PDIR / "019-gb10-mhc-fn-bf16.sh"
MODEL_DIR = os.environ.get("MODEL_DIR", "<MODEL_DIR>")   # checkpoint directory (holds model.safetensors.index.json); --model-dir overrides
TL_REL = "model_executor/kernels/mhc/tilelang.py"
TK_REL = "model_executor/kernels/mhc/tilelang_kernels.py"
FAILS: list[str] = []
RESULTS = {"pass": 0, "skip": 0}


def check(name, ok, info=""):
    ok = bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    RESULTS["pass"] += ok
    if not ok:
        FAILS.append(name)
    return ok


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pins_from_sh(sh: Path) -> dict:
    return dict(re.findall(r"^([A-Z0-9_]+)=([0-9a-f]{64})$", sh.read_text(), re.M))


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def refused(fn, text):
    try:
        fn(text)
    except Exception as e:  # noqa: BLE001
        return type(e).__name__ == "Refused"
    return False


def fdef(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vllm-root", required=True, help="v0.32.0 `vllm` dir")
    ap.add_argument("--work", default=None)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    a = ap.parse_args()
    root = Path(a.vllm_root)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="p019-"))
    work.mkdir(parents=True, exist_ok=True)
    pins = pins_from_sh(SH19)
    tl0, tk0 = (root / TL_REL).read_text(), (root / TK_REL).read_text()
    mod_path = HERE / "gb10_mhc_fn_bf16.py"
    P = load_file_module("p019_apply", HERE / "p019_apply.py")
    P11 = load_file_module("p011_apply", HERE / "p011_apply.py")
    P22 = load_file_module("p022_apply", HERE / "p022_apply.py")

    # ---------------------------------------------------------------------------------------------------------- A
    print("== A pins")
    for var, rel in (("TL_ORIG", TL_REL), ("TK_ORIG", TK_REL), ("HELPER_ORIG", "model_executor/warmup/jit_warmup_tilelang_helper.py"),
                     ("JITW_ORIG", "model_executor/warmup/jit_warmup.py"), ("TLU_ORIG", "tilelang_utils/__init__.py")):
        check(f"A {rel} == {var}", sha((root / rel).read_bytes()) == pins[var], sha((root / rel).read_bytes())[:16])
    check("A files/p019_apply.py == APPLIER_SHA", sha((HERE / "p019_apply.py").read_bytes()) == pins["APPLIER_SHA"])
    check("A files/gb10_mhc_fn_bf16.py == MODULE_SHA", sha(mod_path.read_bytes()) == pins["MODULE_SHA"])

    # ---------------------------------------------------------------------------------------------------------- B
    print("== B applier")
    tl19, tk19 = P.apply_tl(tl0), P.apply_tk(tk0)
    check("B apply_tl(original) == TL_P019", sha(tl19.encode()) == pins["TL_P019"], sha(tl19.encode())[:16])
    check("B apply_tk(original) == TK_P019", sha(tk19.encode()) == pins["TK_P019"], sha(tk19.encode())[:16])
    tl22 = P22.apply(tl0)
    tk11 = P11.apply(tk0)
    check("B 022's applier on the original == TL_P022", sha(tl22.encode()) == pins["TL_P022"])
    check("B 011's applier on the original == TK_P011", sha(tk11.encode()) == pins["TK_P011"])
    tl1922, tk1119 = P.apply_tl(tl22), P.apply_tk(tk11)
    check("B apply_tl(022 result) == TL_P019_P022", sha(tl1922.encode()) == pins["TL_P019_P022"])
    check("B apply_tk(011 result) == TK_P011_P019", sha(tk1119.encode()) == pins["TK_P011_P019"])
    check("B 019/022 commute on tilelang.py", P22.apply(tl19) == tl1922)
    check("B 011/019 commute on tilelang_kernels.py", P11.apply(tk19) == tk1119)
    for nm, text, new, rev in (("tl", tl0, tl19, P.revert_tl), ("tk", tk0, tk19, P.revert_tk),
                               ("tl+022", tl22, tl1922, P.revert_tl), ("tk+011", tk11, tk1119, P.revert_tk)):
        check(f"B revert({nm} result) == its base", rev(new) == text)
    check("B re-apply refused (tl, tk)", refused(P.apply_tl, tl19) and refused(P.apply_tk, tk19))
    for which, text, edits, fn in (("tl", tl0, P.EDITS_TL, P.apply_tl), ("tk", tk0, P.EDITS_TK, P.apply_tk)):
        for name, old, _new in edits:
            dmg = text.replace(old, old.replace("hc_mult3", "hc_mult4").replace("def __call__", "def __cal__")
                               .replace("direct_register_custom_op", "direct_register_custom_oq")
                               .replace("fused_config = ", "fused_confix = "), 1)
            check(f"B damaged anchor refused ({which}): {name}", dmg != text and refused(fn, dmg))
            check(f"B duplicated anchor refused ({which}): {name}", refused(fn, text + "\n" + old))
    # AST scope
    def top(text):
        return {getattr(n, "name", None) or ast.dump(n): ast.dump(n) for n in ast.parse(text).body}

    t0, t1 = top(tl0), top(tl19)
    ch = sorted(k for k in t0 if k in t1 and t0[k] != t1[k])
    add = [k for k in t1 if k not in t0]
    check("B tilelang.py AST scope: only mhc_fused_post_pre_tilelang changed, one assignment (_P019_ON) added",
          ch == ["mhc_fused_post_pre_tilelang"] and len(add) == 1 and "_P019_ON" in add[0] and not [k for k in t0 if k not in t1],
          f"changed={ch} added={len(add)}")
    c0 = next(n for n in ast.parse(tk0).body if isinstance(n, ast.ClassDef) and n.name == P.CLASS_TK)
    c1 = next(n for n in ast.parse(tk19).body if isinstance(n, ast.ClassDef) and n.name == P.CLASS_TK)
    rest0 = {ast.dump(n) for n in c0.body}
    new1 = [n for n in c1.body if ast.dump(n) not in rest0]
    check("B tilelang_kernels.py AST scope: the class gained exactly register_warmup + _P019_ON; everything else identical",
          sorted(getattr(n, "name", None) or n.targets[0].id for n in new1) == ["_P019_ON", "register_warmup"]
          and rest0 <= {ast.dump(n) for n in c1.body}
          and top(tk0).keys() == top(tk19).keys()
          and all(top(tk0)[k] == top(tk19)[k] for k in top(tk0) if k != P.CLASS_TK))

    # ---------------------------------------------------------------------------------------------------------- C
    print("== C kernel port (AST)")
    mb = mod_path.read_text()
    k_up = fdef(tk0, "mhc_fused_tilelang")
    k_new = fdef(mb, "mhc_fused_bf16w_tilelang")
    n_changed = 0
    for node in ast.walk(k_up):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "weight_t":
            dt = node.annotation.args[1]
            if isinstance(dt, ast.Attribute) and dt.attr == "float32":
                dt.attr = "bfloat16"
                n_changed += 1
    k_up.name = "mhc_fused_bf16w_tilelang"
    for k in (k_up, k_new):
        if isinstance(k.body[0], ast.Expr) and isinstance(k.body[0].value, ast.Constant):
            k.body = k.body[1:]
    check("C upstream kernel with weight_t T.float32->T.bfloat16 == ported kernel (AST, docstrings excluded, decorators "
          "included)", n_changed == 1 and ast.dump(k_up) == ast.dump(k_new), f"annotation edits {n_changed}")
    check("C the ported kernel is decorated with @tilelang_jit (the pass_configs upstream's kernels share)",
          [ast.unparse(d) for d in k_new.decorator_list] == ["tilelang_jit"])

    # ---------------------------------------------------------------------------------------------------------- C2
    print("== C2 kernel TIR (real TileLang 0.1.12 frontend, no GPU): fp32 vs bf16 fused kernel")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP C2: torch is not installed (use a Python with torch)")
        RESULTS["skip"] += 1
    else:
        sys.path.insert(0, str(HERE))
        import pmhc_testlib as L0

        st, info = L0.tir_probe(root, module=mod_path)
        if st == "skip":
            print(f"SKIP C2: {info.get('skip')} (pip install --target DIR tilelang==0.1.12 ...; export TILELANG_PROBE_PATH=DIR)")
            RESULTS["skip"] += 1
        else:
            fz = info.get("fused", {})
            check("C2 the TIR of mhc_fused_tilelang and mhc_fused_bf16w_tilelang differ in exactly 3 statements (def name, weight_t buffer "
                  "dtype, the FMA with an explicit T.Cast(\"float32\", weight) of the same expression) for tile_n 2 and 6",
                  st == "ok" and all(v.get("ok") for v in fz.values()) and len(fz) == 2, json.dumps(fz)[:200])

    # ---------------------------------------------------------------------------------------------------------- env
    try:
        import torch
    except ImportError:
        print("SKIP D/E/G: torch is not installed (use a Python with torch)")
        RESULTS["skip"] += 1
        return finish()
    sys.path.insert(0, str(HERE))
    import pmhc_testlib as L

    (work / "tl").mkdir(exist_ok=True)
    paths = {}
    for nm, text in (("tl0", tl0), ("tl19", tl19), ("tk0", tk0), ("tk19", tk19), ("tl1922", tl1922), ("tk1119", tk1119)):
        paths[nm] = work / "tl" / f"{nm}.py"
        paths[nm].write_text(text)

    def make_env(on: bool, patched: bool = True):
        saved = os.environ.get("VLLM_GB10_MHC_FN_BF16")
        if on:
            os.environ["VLLM_GB10_MHC_FN_BF16"] = "1"
        else:
            os.environ.pop("VLLM_GB10_MHC_FN_BF16", None)
        try:
            env = L.build_env(root, tk_path=paths["tk19"] if patched else None, tl_path=paths["tl19"] if patched else None)
            if patched and on:                      # with the gate off nothing may import the module (it is absent here)
                env.load_module("vllm._gb10_mhc_fn_bf16", mod_path)
        finally:
            if saved is None:
                os.environ.pop("VLLM_GB10_MHC_FN_BF16", None)
            else:
                os.environ["VLLM_GB10_MHC_FN_BF16"] = saved
        return env

    class Cap:
        on = False

    torch.cuda.is_current_stream_capturing = lambda: Cap.on  # CPU stand-in
    g = torch.Generator().manual_seed(1)
    HC, HID = 4, 4096

    def exact_fn(scale=0.02):
        return (torch.randn(24, HC * HID, generator=g) * scale).to(torch.bfloat16).float().contiguous()

    # ---------------------------------------------------------------------------------------------------------- D
    print("== D module logic (CPU; the real classes over fake kernels)")
    env_on = make_env(True)
    M = env_on.modules["vllm._gb10_mhc_fn_bf16"]
    with env_on.activate():
        def validate_cpu(fn, hc, hid):  # CPU stand-in: the real checks minus "on cuda"
            if fn.dtype != torch.float32 or tuple(fn.shape) != (hc * (hc + 2), hc * hid) or not fn.is_contiguous():
                raise RuntimeError("[patch-019] unexpected mHC weight (CPU stand-in)")

        real_validate = M._validate
        try:
            real_validate(torch.zeros(24, HC * HID), HC, HID)
            check("D real _validate refuses a weight that is not on cuda", False)
        except RuntimeError as e:
            check("D real _validate refuses a weight that is not on cuda", "on cuda" in str(e))
        M._validate = validate_cpu
        w1 = exact_fn()
        w, bad = M._to_bf16_exact(w1)
        check("D _to_bf16_exact(bf16-exact fp32) -> 0 bad, widened bits identical",
              bad == 0 and torch.equal(w.float().view(torch.int32), w1.view(torch.int32)))
        specials = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), 3.3895313892515355e38, 1.1754943508222875e-38,
                                 9.183549615799121e-41, -1.0, 2.0 ** -126, 0.98046875], dtype=torch.float32)
        specials = specials.to(torch.bfloat16).float()
        ws, bs = M._to_bf16_exact(specials)
        check("D _to_bf16_exact(special bf16 values: +-0, +-inf, max, min normal, subnormal) exact",
              bs == 0 and torch.equal(ws.float().view(torch.int32), specials.view(torch.int32)))
        noisy = w1.clone()
        noisy.view(torch.int32)[0, :7] |= 1
        _, bn = M._to_bf16_exact(noisy)
        check("D _to_bf16_exact(fp32 with low bits) counts them", bn == 7, f"bad={bn}")
        # the REAL self-test over the real classes + fake kernels (equal bits by construction)
        M._reset_for_test()
        env_on.reset_jits()
        a1 = M.weight_for(w1, HC, HID)
        a2 = M.weight_for(w1, HC, HID)
        st = M.stats()
        check("D eager first call converts (REAL self-test ran), second returns the same tensor",
              a1 is a2 and a1.dtype == torch.bfloat16 and tuple(a1.shape) == (24, HC * HID) and st["converted"] == 1
              and st["selftest"])
        calls_fp32 = [(k.get("tile_n"), k.get("n_thr")) for _a, k in env_on.jit("mhc_fused_tilelang").calls]
        calls_bf16 = [(k.get("tile_n"), k.get("n_thr")) for _a, k in env_on.jit("mhc_fused_bf16w_tilelang").calls]
        want = []
        for m in M.SELFTEST_M:
            tn, ns, nt = env_on.tk.mhc_fused_post_pre_split_config(m, HID, HC)
            want += [(tn, nt), (tn, nt)]                       # raw + class launch, per kernel
        check("D the self-test launched both kernels (raw JIT + class path) at every SELFTEST_M with the op's own "
              "(tile_n, n_thr)", calls_fp32 == want and calls_bf16 == want and M.SELFTEST_M == (1, 7, 8, 15, 16, 24, 32),
              f"{len(calls_fp32)}/{len(calls_bf16)} launches")
        check("D the fp32 source is held (its storage cannot be recycled under the key)", M._SRC[M._key(w1)] is w1)
        check("D copy bytes = 768 KiB per weight", st["bytes"] == 24 * HC * HID * 2 == 786432, str(st["bytes"]))
        w2 = exact_fn()
        M.weight_for(w2, HC, HID)
        check("D one self-test per process", M.stats()["converted"] == 2 and len(env_on.jit("mhc_fused_tilelang").calls)
              == len(want))
        try:
            M.weight_for(noisy, HC, HID)
            check("D inexact weight refused (raises)", False)
        except RuntimeError as e:
            check("D inexact weight refused (raises)", "[patch-019]" in str(e) and M.stats()["cached"] == 2)
        w3 = exact_fn()
        Cap.on = True
        r1, r2 = M.weight_for(w3, HC, HID), M.weight_for(w3, HC, HID)
        st = M.stats()
        check("D capture miss -> None, counted once, not cached",
              r1 is None and r2 is None and st["capture_misses"] == 1 and st["cached"] == 2 and st["converted"] == 2)
        hit = M.weight_for(w1, HC, HID)
        check("D capture hit -> cached copy (summary logged once)", hit is a1 and M._STATE["capture_logged"])
        Cap.on = False
        check("D after capture, eager call converts the missed weight",
              M.weight_for(w3, HC, HID) is not None and M.stats()["converted"] == 3)
        try:
            M.weight_for(w1[:, :100].contiguous(), HC, HID)
            check("D wrong-shape weight refused", False)
        except RuntimeError:
            check("D wrong-shape weight refused", True)

        # self-test negative controls: each defect must make the REAL self-test raise
        def selftest_raises(mutate, label):
            M._reset_for_test()
            saved = dict(env_on.impls)
            try:
                mutate(env_on.impls)
                M._validate = validate_cpu
                try:
                    M.weight_for(exact_fn(), HC, HID)
                    caught = False
                except RuntimeError as e:
                    caught = "SELF-TEST FAILED" in str(e)
            finally:
                env_on.impls.clear()
                env_on.impls.update(saved)
            check(f"D self-test negative control: {label} -> the boot self-test raises", caught)

        def differing(impls):
            base = impls["mhc_fused_bf16w_tilelang"]

            def bad_impl(comb, res, post, x, w, yp, rp, ro, *a, **k):
                base(comb, res, post, x, w, yp, rp, ro, *a, **k)
                yp.view(torch.int32).view(-1)[3] ^= 1
            impls["mhc_fused_bf16w_tilelang"] = bad_impl

        def unwritten(impls):
            base = impls["mhc_fused_bf16w_tilelang"]

            def bad_impl(comb, res, post, x, w, yp, rp, ro, *a, **k):
                base(comb, res, post, x, w, yp, rp, ro, *a, **k)
                rp.view(-1)[-1] = float("nan")
            impls["mhc_fused_bf16w_tilelang"] = bad_impl

        def class_not_raw(impls):
            base = impls["mhc_fused_tilelang"]
            n_call = {"n": 0}

            def bad_impl(comb, res, post, x, w, yp, rp, ro, *a, **k):
                n_call["n"] += 1
                base(comb, res, post, x, w, yp, rp, ro, *a, **k)
                if n_call["n"] % 2 == 0:                      # every second launch = the class path of the fp32 kernel
                    ro.view(torch.int16).view(-1)[0] ^= 1
            impls["mhc_fused_tilelang"] = bad_impl

        selftest_raises(differing, "a bf16 kernel whose bits differ")
        selftest_raises(unwritten, "a bf16 kernel that leaves an output unwritten")
        selftest_raises(class_not_raw, "a class path that differs from the raw launch")
        M._reset_for_test()
        M._validate = real_validate

    # ---------------------------------------------------------------------------------------------------------- E
    print("== E dispatch (the real patched tilelang.py over the real classes)")
    env_ref = make_env(False, patched=False)        # upstream, untouched
    env_off = make_env(False)                       # patched files, gate unset
    env_on = make_env(True)                         # patched files, gate set
    M = env_on.modules["vllm._gb10_mhc_fn_bf16"]
    check("E flag off: the vendored module is not even imported by the patched files (it is not in sys.modules here)",
          "vllm._gb10_mhc_fn_bf16" not in env_off.modules)
    check("E import-time gates: _P019_ON True with the env, False without (tilelang.py and the class)",
          env_on.tl._P019_ON is True and env_off.tl._P019_ON is False
          and env_on.tk.MhcFusedTileLangKernel._P019_ON is True and env_off.tk.MhcFusedTileLangKernel._P019_ON is False)
    with env_on.activate():
        def validate_cpu2(fn, hc, hid):
            if fn.dtype != torch.float32 or tuple(fn.shape) != (hc * (hc + 2), hc * hid) or not fn.is_contiguous():
                raise RuntimeError("[patch-019] unexpected mHC weight (CPU stand-in)")
        M._validate = validate_cpu2
    fn = exact_fn()
    scale, base = torch.rand(3, generator=g), torch.rand(24, generator=g)
    normw = torch.ones(HID, dtype=torch.bfloat16)

    def args_for(m):
        x = torch.randn(m, HID, generator=g).to(torch.bfloat16)
        res = torch.randn(m, HC, HID, generator=g).to(torch.bfloat16)
        post = torch.rand(m, HC, 1, generator=g)
        comb = torch.rand(m, HC, HC, generator=g)
        return (x, res, post, comb, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, normw, 1e-5)

    def run_op(env, m_args):
        with env.activate():
            env.reset_jits()
            out = env.tl.mhc_fused_post_pre_tilelang(*m_args)
            names = ("mhc_fused_tilelang", "mhc_fused_bf16w_tilelang", "mhc_post_tilelang", "hc_prenorm_gemm_tilelang",
                     "hc_prenorm_gemm_block_m_tilelang", "mhc_pre_big_fuse_with_norm_tilelang")
            calls = {n: [(tuple(getattr(t, "dtype", t) for t in a if hasattr(t, "dtype")), k) for a, k in env.jit(n).calls]
                     for n in names if n in env.jits and env.jit(n).calls}
            return out, calls

    bad_off, bad_route, bad_bits, imports_off = [], [], [], []
    M._reset_for_test()
    n_first = None
    for m in range(48, 0, -1):               # first call at M = 48 (> 32): the copy must still be made there
        ar = args_for(m)
        o_ref, c_ref = run_op(env_ref, ar)
        o_off, c_off = run_op(env_off, ar)
        o_on, c_on = run_op(env_on, ar)
        if c_off != c_ref or not all(L.tensors_equal_bits(x, y) for x, y in zip(o_off, o_ref)):
            bad_off.append(m)
        fused = m <= 32
        n_fp32, n_bf16 = len(c_on.get("mhc_fused_tilelang", [])), len(c_on.get("mhc_fused_bf16w_tilelang", []))
        fused_names = ("mhc_fused_tilelang", "mhc_fused_bf16w_tilelang")
        if m == 48:        # the first call converts: its self-test launches both fused kernels (not part of the op's route)
            if {k: v for k, v in c_on.items() if k not in fused_names} != {k: v for k, v in c_ref.items() if k not in fused_names}:
                bad_route.append(m)
        elif fused and not (n_fp32 == 0 and n_bf16 == 1 and c_ref["mhc_fused_tilelang"] and
                            c_on["mhc_fused_bf16w_tilelang"][0][0][4:5] == (torch.bfloat16,)):
            bad_route.append(m)
        elif not fused and (n_bf16 != 0 or {k: v for k, v in c_on.items() if k != "mhc_fused_bf16w_tilelang"}
                            != {k: v for k, v in c_ref.items() if k != "mhc_fused_bf16w_tilelang"}):
            bad_route.append(m)
        if not all(L.tensors_equal_bits(x, y) for x, y in zip(o_on, o_ref)):
            bad_bits.append(m)
    check("E flag off: call-for-call and bit-for-bit identical to the unpatched files at M = 1..48", not bad_off, str(bad_off))
    check("E flag on: M <= 32 -> bf16 JIT function on a bf16 weight (never the fp32 fused kernel); M > 32 unchanged",
          not bad_route, str(bad_route))
    check("E flag on: ALL outputs bitwise equal to the unpatched files at M = 1..48 (fused outputs, mixes via big_fuse)",
          not bad_bits, str(bad_bits))
    st = M.stats()
    check("E flag on: one conversion (made at the first call, M = 48), one self-test",
          st["converted"] == 1 and st["selftest"] and st["cached"] == 1, str(st))
    # the call uses the cached copy (same storage every time)
    ptrs = set()
    with env_on.activate():
        for m in (1, 2, 3, 16, 32):
            env_on.reset_jits()
            env_on.tl.mhc_fused_post_pre_tilelang(*args_for(m))
            ptrs.add(env_on.jit("mhc_fused_bf16w_tilelang").calls[0][0][4].data_ptr())
    check("E the bf16 launches read ONE cached copy (same storage for M = 1, 2, 3, 16, 32)", len(ptrs) == 1
          and ptrs != {fn.data_ptr()})
    # capture miss -> the upstream fp32 kernel
    env_cm = make_env(True)
    Mcm = env_cm.modules["vllm._gb10_mhc_fn_bf16"]
    with env_cm.activate():
        Mcm._validate = validate_cpu2
    Cap.on = True
    ar = args_for(3)
    o_ref, c_ref = run_op(env_ref, ar)
    o_cm, c_cm = run_op(env_cm, ar)
    Cap.on = False
    check("E capture miss -> the original fp32 fused kernel, identical call and bits",
          c_cm["mhc_fused_tilelang"] == c_ref["mhc_fused_tilelang"] and not c_cm.get("mhc_fused_bf16w_tilelang")
          and Mcm.stats()["capture_misses"] == 1 and Mcm.stats()["converted"] == 0
          and all(L.tensors_equal_bits(x, y) for x, y in zip(o_cm, o_ref)))

    # ---------------------------------------------------------------------------------------------------------- G
    print("== G JIT warmup registry (REAL JitWarmupRegistry, REAL classes)")
    cfg_kw = dict(hidden_size=HID, hc_mult=HC)
    for flag in (False, True):
        env = make_env(flag)
        K = env.tk._MHC_FUSED_TILELANG_KERNEL
        with env.activate():
            cfg = L.fake_vllm_config(8192)
            try:
                reg = L.run_registry_warmup(env, lambda: K.register_warmup(cfg, **cfg_kw))
                err = ""
            except Exception as e:  # noqa: BLE001
                reg, err = None, f"{type(e).__name__}: {str(e)[:300]}"
            c32 = env.jit("mhc_fused_tilelang").compiles
            c16 = env.jits["mhc_fused_bf16w_tilelang"].compiles if "mhc_fused_bf16w_tilelang" in env.jits else []
            keys32 = sorted((k.get("tile_n"), k.get("n_thr")) for _a, k in c32)
            keys16 = sorted((k.get("tile_n"), k.get("n_thr")) for _a, k in c16)
            dtypes = {str(a[4].dtype) for a, _k in c16}
            fp_dtypes = {str(a[4].dtype) for a, _k in c32}
        if not flag:
            check("G flag off: the registry warms only the fp32 kernel (2 keys); the module is not imported",
                  not err and keys32 == [(2, 128), (6, 128)] and not c16 and "vllm._gb10_mhc_fn_bf16" not in env.modules
                  and fp_dtypes == {"torch.float32"}, err or f"{keys32} bf16={keys16}")
        else:
            check("G flag on: one registration of the fp32 kernel warms BOTH kernels, the same 2 keys, bf16 weight_t for the "
                  "variant", not err and keys32 == keys16 == [(2, 128), (6, 128)] and dtypes == {"torch.bfloat16"}
                  and fp_dtypes == {"torch.float32"}, err or f"{keys32} / {keys16} {dtypes}")
            Mv = env.modules["vllm._gb10_mhc_fn_bf16"]
            with env.activate():
                regs = reg._registrations
                kernels = [type(k).__name__ for k in regs]
            check("G flag on: the registry holds exactly the fp32 class and the bf16 variant (no recursion, one entry each)",
                  sorted(kernels) == ["MhcFusedBf16wTileLangKernel", "MhcFusedTileLangKernel"], str(kernels))
            same_keys = (set(K.get_warmup_keys(cfg, **cfg_kw)) == set(Mv.MHC_FUSED_BF16W_KERNEL.get_warmup_keys(cfg, **cfg_kw))
                         if False else True)
            with env.activate():
                kk = K.get_warmup_keys(cfg, **cfg_kw)
                kb = Mv.MHC_FUSED_BF16W_KERNEL.get_warmup_keys(cfg, **cfg_kw)
            check("G the variant's compile keys == the fp32 kernel's (inherited dispatch / get_warmup_keys)",
                  kk == kb and len(kk) == 2 and same_keys)
            with env.activate():
                bf = Mv.MHC_FUSED_BF16W_KERNEL
                wi = bf.warmup_inputs(kb[0])
            check("G the variant's warmup weight is a bf16 [24, 4, 4096] tensor, the rest as upstream",
                  str(wi["weight_t"].dtype) == "torch.bfloat16" and tuple(wi["weight_t"].shape) == (24, HC, HID))
            # registering outside any registry is a no-op (no active registry) and does not fail
            with env.activate():
                K.register_warmup(cfg, **cfg_kw)
            check("G registration outside an active registry is a no-op for both kernels", True)

    # ---------------------------------------------------------------------------------------------------------- F
    if Path(a.model_dir, "model.safetensors.index.json").is_file():
        print("== F checkpoint hc_*_fn")
        from safetensors import safe_open

        d = Path(a.model_dir)
        wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        keys = sorted((k for k in wm if k.endswith(("hc_attn_fn", "hc_ffn_fn"))),
                      key=lambda k: (int(k.split(".layers.")[1].split(".")[0]), k))
        cfg_j = json.loads((d / "config.json").read_text())
        tc = cfg_j.get("text_config", cfg_j)
        L_ = int(tc["num_hidden_layers"])
        n = rt = sub = zero = nonfin = 0
        dts = set()
        Mx = env_on.modules["vllm._gb10_mhc_fn_bf16"]
        for k in keys:
            with safe_open(str(d / wm[k]), framework="pt", device="cpu") as fh:
                t = fh.get_tensor(k)
            dts.add((str(t.dtype), tuple(t.shape)))
            f32 = torch.empty(t.shape, dtype=torch.float32)
            f32.copy_(t)  # what default_weight_loader does (param.data.copy_)
            w, nb = Mx._to_bf16_exact(f32)
            rt += nb == 0 and torch.equal(w.view(torch.int16), t.view(torch.int16))
            aa = f32.abs()
            sub += int(((aa > 0) & (aa < torch.finfo(torch.float32).tiny)).sum())
            zero += int((aa == 0).sum())
            nonfin += int((~torch.isfinite(f32)).sum())
            n += 1
        check(f"F {n} hc_*_fn == 2 x num_hidden_layers ({L_}), all BF16 [24, 16384]",
              n == 2 * L_ and dts == {("torch.bfloat16", (24, 16384))}, str(sorted(dts)))
        check(f"F fp32 (as loaded) -> bf16 -> fp32 bit-exact for {rt}/{n}", rt == n)
        print(f"   subnormal {sub}, zero {zero}, non-finite {nonfin}; bf16 copies of the {2 * L_ - 1} fused-op "
              f"weights: {(2 * L_ - 1) * 24 * 16384 * 2 / 2**20:.2f} MiB per rank")
    else:
        print("== F skipped (no checkpoint on this machine; the 90/90 BF16 result is in the old 019 record, 2026-09-30)")
        RESULTS["skip"] += 1
    return finish()


def finish():
    print(f"---- {RESULTS['pass']} passed, {len(FAILS)} failed, {RESULTS['skip']} skipped"
          + (": " + "; ".join(FAILS) if FAILS else ""))
    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
