#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-020 gb10-sparse-mla-triton-prefill] CPU checks, REVISION 4 (vLLM v0.32.0 port).
Mac / any CPU: python3 >= 3.10 + numpy; NO torch, NO triton, NO GPU, NO vLLM install, no container.

  python3 files/p020_cpu_check.py --vllm-root <v0.32.0 tree>/vllm [--baked-root <baked tree>/vllm] [--quick] [--mutants]
                                  [--only ABCDEF]

--vllm-root holds the PRISTINE v0.32.0 files (sha-checked against the .sh pin): v1/attention/backends/mla/
flashinfer_mla_sparse_sm90.py and platforms/cuda.py. The deployable target is the FROZEN SM121 bake's file: with
--baked-root (a tree produced by running image/glm53_sm121_v032.py on a copy of the pristine one; run_checks.sh does
this) that file is used as the BAKED base; without it the base is derived by replaying the bake's two SM90-file edits
(asserted byte-identical to the image build's published post-bake sha 6e97595e...). The MiaAI-Lab spelling of the bake
(ALT: three edits incl. the fp8 version gate) is a third, unpinned base. Needs Python >= 3.10 (it executes the real
v0.32.0 backend source, which uses `X | Y` annotations at run time).

The REAL stock and patched backend source is exec'd into a STUBBED vllm namespace over a numpy-backed fake torch
(files/p020_testlib.py), so the REAL upstream FlashInferMLASparseSM90Builder.build / _kv_lens_host and
Impl.__init__ / forward_mqa run next to the patch's hooks and the vendored module. Sections:

  A  pins + applier: the .sh pins == the files; stock / baked bases + results == the pins; apply / idempotent /
     revert (byte-exact) / on-disk / refusals (every anchor mutated or duplicated); the anchors do not overlap the
     SM121-bake edits; AST placement of the five hooks; the stock lines survive verbatim in the gate-off branches.
  B  the fail-closed import block (exec'd on its own).
  C  the backend probe (capability 12.1 accepted? cuda.py lists the backend?) on stock and baked files.
  D  the REAL patched backend: gate-off parity with stock (builder plan log, forward_mqa call log, __init__) over
     thousands of simulated V2 batches x {sparse-MHA routing, host bound present / absent, adaptive verification};
     gate-on routing (nd == 0 -> no plan; nd > 0 -> upstream's plan over the first nd rows; pure decode -> the
     stock plan and the stock call sequence), plan -> forward consistency, init hook, runtime env gating.
  E  module units: plan_step, forward_split plumbing + refusals, startup_check, the self-test under the model
     loader's device/dtype context, the split-LSE merge, the frozen config, and the SHA of every unchanged function's
     source against revision 3 (the kernel is byte-identical to the GB10-validated one).
  F  (--mutants) planted bugs in the patched file / module / applier: every one must make the suite fail.
Exit 0 = ALL PASS.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import os
import random
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
SH = PDIR / "020-gb10-sparse-mla-triton-prefill.sh"
sys.path.insert(0, str(HERE))
if sys.version_info < (3, 10):
    raise SystemExit("[020] p020_cpu_check.py needs Python >= 3.10 (it executes the real v0.32.0 backend source)")

import numpy as np  # noqa: E402

import p020_testlib as T  # noqa: E402

SM_REL = "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
CUDA_REL = "platforms/cuda.py"
MODULE_PATH = Path(os.environ.get("P020_MODULE_OVERRIDE") or (HERE / "gb10_sparse_mla_prefill.py"))
APPLIER_PATH = Path(os.environ.get("P020_APPLIER_OVERRIDE") or (HERE / "p020_apply.py"))
PATCHED_OVERRIDE = os.environ.get("P020_PATCHED_OVERRIDE")  # a ready-made patched file replacing apply(stock)
GATE = "VLLM_GB10_SPARSE_MLA_TRITON_PREFILL"

RESULTS = {"pass": 0, "fail": 0, "failed": []}
QUIET = {"on": False}


def check(name, ok, info=""):
    ok = bool(ok)
    if not QUIET["on"] or not ok:
        shown = info if (not ok or len(info) <= 70) else info[:67] + "..."
        print(("PASS " if ok else "FAIL ") + name + (f"  [{shown}]" if info else ""), flush=True)
    RESULTS["pass" if ok else "fail"] += 1
    if not ok:
        RESULTS["failed"].append(name)
    return ok


def raises(fn, exc=Exception, match=None):
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return match is None or re.search(match, str(e)) is not None
    return False


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def pin(name):
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", SH.read_text(), re.M)
    if not m:
        raise SystemExit(f"no pin {name} in {SH}")
    return m.group(1)


def setgate(on: bool):
    if on:
        os.environ[GATE] = "1"
    else:
        os.environ.pop(GATE, None)


# SHA-256 of the exact SOURCE TEXT of every module-level object that did not change since revision 3
# (sha 6346bd7b...), the GB10-validated module: kernel, host wrapper, merge, self-test, config check.
UNCHANGED_SHA = {
    "_sparse_mla_prefill_kernel": "c88c2565c5f2ef03dac7098108f006e3f575c87dbbe2ab4c602a0f469ae888ce",
    "_choose_num_splits": "5d7155d2999caf37765ac1ed8664150a5d336c54f60e08053709c0ddb2015200",
    "sparse_mla_prefill_triton": "ced39c0d481fcc2d81907d4a5c48ada6dc44eb26b453066d66b15097963f9ccf",
    "_merge_splits": "19b53ab0c7e3e5f2903642a282911457d3688744a69cb3b7891b838ada0119f7",
    "_check_impl_for_triton": "115d5b32121a52965ce4f7dff524d1da29a185e93db670fa41d658f0fafe60e2",
    "_torch_reference": "ac141e06119dc99ecace79a1cc5f699285b09b5618571622d4a7c8fbd7334942",
    "kernel_self_test": "59ce7969f3e30bce3e0640ec1ead2a1b8e9d181a82d26574895a51f1fceeb3ab",
    "Patch020Error": "71da5867a5ea7b5b02edbade16c146f8f023d84c8426d08561d457898aec3206",
    "_require": "6e2628fad928061433100808d276f16d97a01cfd6816dc13553da14e053b6cea",
    "_flag": "e52ac7629d2d66962bfc64554e5f5bad4761d556e815f2d27f02bc7c11e74f4a",
    "CONSTANTS": "3b573a53f1f0251cabf56f372f362578f1fa40ab1d0087b46a2265e394fbebad",
}


def source_hashes(text: str) -> dict:
    tree = ast.parse(text)
    out = {}
    consts = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(text, n)
            # decorators are not part of get_source_segment's range for FunctionDef in some versions
            if getattr(n, "decorator_list", None):
                first = min(d.lineno for d in n.decorator_list)
                seg = "\n".join(text.split("\n")[first - 1 : n.end_lineno])
            out[n.name] = hashlib.sha256(seg.encode()).hexdigest()
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in (
            "_D_LATENT", "_BLOCK_H", "_BLOCK_N", "_NUM_WARPS", "_NUM_STAGES", "_TWO_SPLIT_MAX_ROWS"
        ):
            consts.append(ast.get_source_segment(text, n))
    out["CONSTANTS"] = hashlib.sha256("\n".join(consts).encode()).hexdigest()
    return out


# ------------------------------------------------------------------------------------------------
# bases
# ------------------------------------------------------------------------------------------------
# The REAL bake of this port set (image/glm53_sm121_v032.py group `attn`, ids sm90cap + sm90fa2) — its
# published post-bake sha is 6e97595e... in image/pins-after-bake.txt; the two replacements reproduce it byte for byte.
IMAGE_TASK_BAKED_SHA = "6e97595e2c3b2e6265cc3ea578a2d1654f6504adcdc2ba30f814d0505d72950e"
CAP_OLD = "        return capability.major == 9\n"
CAP_NEW = ("        # SM121 bake: SM12x (GB10) runs the portable FA2 kernel behind the same\n"
           "        # wrapper (see _SM90State); FA3 stays Hopper-only.\n"
           "        return capability.major in (9, 12)\n")
FA_OLD = '            backend="fa3",\n'
FA_NEW = ("            # SM121 bake: FA3 (wgmma) is Hopper-only; FA2 behind the same wrapper is\n"
          "            # portable to SM12x and shares MLAPlan/MLAPlanInfo with FA3, so the\n"
          "            # sync-free plan and the kv_end clamp in this file apply to both.\n"
          "            backend=(\n"
          '                "fa3" if torch.cuda.get_device_capability(device)[0] == 9 else "fa2"\n'
          "            ),\n")
# ALT = the MiaAI-Lab spelling of the same edits plus the fp8-only FlashInfer-version gate (not pinned in the .sh)
ALT_CAP_OLD = ("    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
               "        return capability.major == 9\n")
ALT_CAP_NEW = ("    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
               "        return capability.major in (9, 12)\n")
ALT_FA_NEW = '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),\n'
ALT_GATE_OLD = '''        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )'''
ALT_GATE_NEW = '''        if kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 fp8 KV requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )'''
CUDA_OLD = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
CUDA_NEW = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                # SM121 bake: the NoPE sparse-MLA backend on FlashInfer FA2 (plain
                # fp8_e4m3 KV, dense-MHA prefill) ahead of the packed-cache SM120 path.
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""


def _once(text, old, new, what):
    assert text.count(old) == 1, f"{what}: {text.count(old)} matches"
    return text.replace(old, new, 1)


def make_bases(stock: str) -> dict:
    baked = _once(_once(stock, CAP_OLD, CAP_NEW, "cap"), FA_OLD, FA_NEW, "fa2")
    alt = _once(_once(_once(stock, ALT_CAP_OLD, ALT_CAP_NEW, "alt cap"), FA_OLD, ALT_FA_NEW, "alt fa2"),
                ALT_GATE_OLD, ALT_GATE_NEW, "alt gate")
    return {"STOCK": stock, "BAKED": baked, "ALT": alt}


def load_applier():
    spec = importlib.util.spec_from_file_location("p020_apply", APPLIER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p020_apply"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------
# A. pins + applier
# ------------------------------------------------------------------------------------------------
def test_pins_and_applier(ap, stock: str, bases: dict, baked_from_script: bool):
    check("A1 pristine v0.32.0 file sha == SM_STOCK pin (pristine = bake absent = inert, never patched)",
          sha_text(stock) == pin("SM_STOCK"), sha_text(stock)[:16])
    check("A1 the frozen bake's SM90 file == the image build's published post-bake sha 6e97595e... == SM_BAKED pin"
          + (" [file produced by running glm53_sm121_v032.py on a copy of the tree]" if baked_from_script
             else " [derived by replaying the two edits; pass --baked-root to use the bake script's own output]"),
          sha_text(bases["BAKED"]) == IMAGE_TASK_BAKED_SHA == pin("SM_BAKED"), sha_text(bases["BAKED"])[:16])
    check("A1 the MiaAI-Lab spelling of the bake (ALT, 3 edits) also applies to v0.32.0 exactly once each", True)
    check("A2 applier / module shas in the .sh == the shipped files",
          hashlib.sha256(APPLIER_PATH.read_bytes()).hexdigest() == pin("APPLIER_SHA")
          and hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest() == pin("MODULE_SHA"))
    patched = {}
    for key, base in bases.items():
        out = ap.apply(base)
        patched[key] = out
        if key == "BAKED":  # the only deployable target; STOCK is inert in the .sh, ALT exercises the re-pin flow
            check("A3 BAKED: base sha == SM_BAKED", sha_text(base) == pin("SM_BAKED"))
            check("A3 BAKED: result sha == SM_BAKED_P020", sha_text(out) == pin("SM_BAKED_P020"))
        check(f"A4 {key}: {ap.N_MARKERS} markers, compiles, idempotent, revert is byte-exact",
              out.count(ap.MARKER) == ap.N_MARKERS == 5 and compile(out, "x", "exec") is not None
              and ap.apply(out) == out and ap.revert(out) == base)
        # on disk
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.py"
            p.write_text(base)
            os.chmod(p, 0o640)
            ok1 = ap.main(["p020_apply.py", "apply", str(p)]) == 0 and p.read_text() == out
            mode = os.stat(p).st_mode & 0o7777
            ok2 = ap.main(["p020_apply.py", "revert", str(p)]) == 0 and p.read_text() == base
            check(f"A5 {key}: on-disk apply / revert, mode preserved", ok1 and ok2 and mode == 0o640)
    # anchors independent of the bake edits (disjoint character ranges in the stock file)
    anchors = [old for _, old, _ in ap.EDITS]
    bake_olds = [CAP_OLD, FA_OLD, ALT_CAP_OLD, ALT_GATE_OLD]
    ranges = lambda s: [(stock.index(a), stock.index(a) + len(a)) for a in s]  # noqa: E731
    ra, rb = ranges(anchors), ranges(bake_olds)
    overlap = [(x, y) for x in ra for y in rb if x[0] < y[1] and y[0] < x[1]]
    check("A6 the five anchors do not overlap the SM121-bake edits (either spelling)", not overlap, str(overlap))
    check("A6 every anchor occurs exactly once in stock, the real bake and the ALT bake",
          all(b.count(a) == 1 for b in bases.values() for a in anchors))
    # refusals
    base = bases["BAKED"]
    for i, (name, old, _) in enumerate(ap.EDITS):
        mut = base.replace(old, old.rstrip("\n") + "  # drifted\n", 1)
        check(f"A7 anchor {i + 1} ({name}) mutated -> refused", raises(lambda m=mut: ap.apply(m), SystemExit, "anchor check failed"))
        dup = base + "\n" + old
        check(f"A7 anchor {i + 1} ({name}) duplicated -> refused", raises(lambda m=dup: ap.apply(m), SystemExit, "anchor check failed"))
    # a file carrying the marker is returned as is (idempotence); the .sh refuses unknown shas
    check("A8 marker present -> apply is a no-op", ap.apply(patched["STOCK"]) == patched["STOCK"])
    # AST placement
    check_placement(ap, patched["BAKED"], "BAKED")
    check_placement(ap, patched["STOCK"], "STOCK (applier-level only: deployed never)")
    # the gate-off branches keep the stock statements verbatim; the diff is additions + one rename
    p = patched["BAKED"]
    stock = bases["BAKED"]
    keep = [
        "        num_rows, kv_lens, exact = self._kv_lens_host(common_attn_metadata)\n",
        "        state.pack_indices(topk_slots)\n",
        "        out = state.wrapper.run(q_nope, q_pe, ckv, kpe, **scale_kwargs)\n        return out, None\n",
        "            self.state.plan(num_rows, kv_lens, None, None)\n",
    ]
    check("A9 stock lines survive verbatim in the gate-off paths", all(k in p for k in keep))
    import difflib

    removed = [l for l in difflib.unified_diff(stock.split("\n"), p.split("\n"), lineterm="", n=0)
               if l.startswith("-") and not l.startswith("---")]
    check("A9 the patch removes exactly one stock line (the `topk_slots, _ =` -> `_p020_valid_counts` rename)",
          removed == ["-        topk_slots, _ = triton_convert_req_index_to_global_index("], removed)
    return patched


def _find(tree, cls_name, fn_name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)


def _is_gate_if(s):
    return isinstance(s, ast.If) and isinstance(s.test, ast.Call) and getattr(s.test.func, "id", "") == "_p020_triton_on"


def check_placement(ap, text, label):
    tree = ast.parse(text)
    defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigns = {x.id for n in tree.body if isinstance(n, (ast.Try, ast.Assign)) for m in ast.walk(n)
               if isinstance(m, ast.Assign) for x in m.targets if isinstance(x, ast.Name)}
    ok_mod = {"_p020_triton_on", "_p020_module"} <= defs and {"_P020", "_P020_IMPORT_ERROR"} <= assigns
    check(f"A10 {label}: gate helper, module accessor and _P020 live at module level", ok_mod)
    build = _find(tree, "FlashInferMLASparseSM90Builder", "build")
    i_if = next((i for i, s in enumerate(build.body) if _is_gate_if(s)), None)
    i_kv = next((i for i, s in enumerate(build.body)
                 if isinstance(s, ast.Assign) and "_kv_lens_host" in ast.dump(s.value)), None)
    body = build.body[i_if].body if i_if is not None else []
    ok = (i_if is not None and i_kv is not None and i_if + 1 == i_kv and len(body) == 3
          and "plan_step" in ast.dump(body[0]) and "state" in ast.dump(body[1]) and isinstance(body[2], ast.Return))
    check(f"A10 {label}: builder hook = `if gate: plan_step; metadata.state = self.state; return` right before "
          "upstream's _kv_lens_host planning", ok)
    fm = _find(tree, "FlashInferMLASparseSM90Impl", "forward_mqa")
    i_conv = next((i for i, s in enumerate(fm.body) if isinstance(s, ast.Assign)
                   and "triton_convert_req_index_to_global_index" in ast.dump(s.value)), None)
    tgt = ast.dump(fm.body[i_conv].targets[0]) if i_conv is not None else ""
    i_st = next((i for i, s in enumerate(fm.body) if isinstance(s, ast.Assign)
                 and "attn_metadata" in ast.dump(s.value) and getattr(s.targets[0], "id", "") == "state"), None)
    i_fgate = next((i for i, s in enumerate(fm.body) if _is_gate_if(s)), None)
    i_pack = next((i for i, s in enumerate(fm.body) if isinstance(s, ast.Expr) and "pack_indices" in ast.dump(s)), None)
    ok = (i_conv is not None and "_p020_valid_counts" in tgt and i_st is not None and i_fgate is not None
          and i_pack is not None and i_st < i_fgate < i_pack and "forward_split" in ast.dump(fm.body[i_fgate]))
    check(f"A10 {label}: forward_mqa keeps valid_counts, and the split hook sits between `state = "
          "attn_metadata.state` and `state.pack_indices(...)`", ok)
    init = _find(tree, "FlashInferMLASparseSM90Impl", "__init__")
    ok = _is_gate_if(init.body[-1]) and "startup_check" in ast.dump(init.body[-1])
    check(f"A10 {label}: Impl.__init__ ends with the startup_check hook", ok)


# ------------------------------------------------------------------------------------------------
# B. import block, C. probe
# ------------------------------------------------------------------------------------------------
def test_import_block(ap):
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm._gb10_sparse_mla_prefill")}
    fake_pkg = types.ModuleType("vllm")
    fake_pkg.__path__ = []
    try:
        sys.modules["vllm"] = fake_pkg
        sys.modules.pop("vllm._gb10_sparse_mla_prefill", None)
        setgate(False)
        ns = {"__name__": "p020_import_block"}
        exec(compile(ap.IMPORT_BLOCK, "import_block", "exec"), ns)
        ok = ns["_P020"] is None and not ns["_p020_triton_on"]()
        ok &= raises(ns["_p020_module"], RuntimeError, r"\[patch-020\]")
        check("B1 gate unset: a missing module is tolerated; _p020_module() raises if used", ok)
        setgate(True)
        check("B2 gate set + unloadable module -> RuntimeError at import",
              raises(lambda: exec(compile(ap.IMPORT_BLOCK, "import_block", "exec"), {"__name__": "x"}),
                     RuntimeError, r"\[patch-020\].*failed to import"))
        fake_mod = types.ModuleType("vllm._gb10_sparse_mla_prefill")
        sys.modules["vllm._gb10_sparse_mla_prefill"] = fake_mod
        fake_pkg._gb10_sparse_mla_prefill = fake_mod
        ns = {"__name__": "y"}
        exec(compile(ap.IMPORT_BLOCK, "import_block", "exec"), ns)
        check("B3 gate set + loadable module -> used", ns["_p020_module"]() is fake_mod and ns["_p020_triton_on"]())
        setgate(False)
        os.environ["VLLM_GB10_SPARSE_MLA_NOSYNC"] = "1"
        try:
            ns = {"__name__": "z"}
            exec(compile(ap.IMPORT_BLOCK, "import_block", "exec"), ns)
            check("B4 the obsolete NOSYNC gate does not enable anything", not ns["_p020_triton_on"]())
        finally:
            os.environ.pop("VLLM_GB10_SPARSE_MLA_NOSYNC", None)
    finally:
        setgate(False)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_probe(ap, vllm_root: Path, bases: dict, baked_root: Path | None):
    cuda_stock = (vllm_root / CUDA_REL).read_text()
    cuda_baked = _once(cuda_stock, CUDA_OLD, CUDA_NEW, "cuda.py")
    if baked_root is not None:
        real_cuda = (baked_root / CUDA_REL).read_text()
        check("C0 the bake script's own platforms/cuda.py: SM90 listed ahead of SM120 for capability 12, sha 22fcde66...",
              ap.probe(bases["BAKED"], real_cuda) == ("yes", "yes")
              and sha_text(real_cuda) == "22fcde666f46d5955186209d0366930026926a14d11ddbfd573b2e7d24312e7a")
    got = {k: ap.probe(b, cuda_stock) for k, b in bases.items()}
    check("C1 stock SM90 backend: capability 12.1 rejected, cuda.py does not list it",
          got["STOCK"] == ("no", "no"), str(got["STOCK"]))
    check("C2 baked SM90 backend (real bake and ALT): capability 12.1 accepted",
          got["BAKED"][0] == "yes" and got["ALT"][0] == "yes")
    check("C3 baked cuda.py (the real bake's capability-12 list) names the SM90 backend",
          ap.probe(bases["BAKED"], cuda_baked) == ("yes", "yes"))
    check("C4 garbage / missing inputs -> unknown, never an exception",
          ap.probe("def broken(:", None) == ("unknown", "unknown") and ap.probe("", "") == ("unknown", "unknown"))
    check("C5 an unrelated expression is evaluated safely (no builtins)",
          ap.probe(bases["STOCK"].replace("return capability.major == 9", "return __import__('os').getpid() > 0"), None)[0] == "unknown")
    late = cuda_stock.replace("AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n            ]\n        else:",
                              "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,\n            ]\n        else:", 1)
    check("C6 SM90 listed AFTER the SM120 backend -> 'late' (SM120 would win the auto-selection)",
          late != cuda_stock and ap.probe(bases["BAKED"], late)[1] == "late")
    commented = cuda_stock.replace("                AttentionBackendEnum.TRITON_MLA,\n                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,",
                                   "                AttentionBackendEnum.TRITON_MLA,\n                # FLASHINFER_MLA_SPARSE_SM90 is not listed here\n                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,", 1)
    check("C7 a comment naming the SM90 backend does not count as listing it", commented != cuda_stock and ap.probe(bases["BAKED"], commented)[1] == "no")


# ------------------------------------------------------------------------------------------------
# D. the REAL patched backend over the fake torch
# ------------------------------------------------------------------------------------------------
class Fixture:
    def __init__(self, ap, patched_text: str, stock_text: str, module_path: Path = MODULE_PATH):
        setgate(False)
        self.env = T.install_stubs()
        self.mod = T.load_file(self.env, "vllm_gb10_sparse_mla_prefill", module_path,
                               install_as="vllm._gb10_sparse_mla_prefill")
        self.stock = T.load_source(self.env, "sm90_stock", stock_text, "stock_sm90.py")
        self.patched = T.load_source(self.env, "sm90_patched", patched_text, "patched_sm90.py")
        self.torch = self.env.torch

    def builder(self, m, use_sparse_mha=False, adaptive=False):
        b = object.__new__(m.FlashInferMLASparseSM90Builder)
        b.state = T.FakeState()
        b._adaptive_verification = adaptive
        b._attention_layer = T.FakeLayer(use_sparse_mha)
        return b

    def impl(self, m, heads=8, fp8=True, rope=0, head_size=512, widths=2176, tmax=8192):
        buf = T.Tensor(np.full((tmax, widths), -1, dtype=np.int32), self.torch.int32, T._Device("cpu"))
        impl = m.FlashInferMLASparseSM90Impl(
            heads, head_size, 1.0 / 16, 1, None, None, "fp8_e4m3" if fp8 else "auto", None, "decoder", None,
            topk_indices_buffer=buf, indexer=None, kv_lora_rank=512, qk_rope_head_dim=rope)
        return impl, buf


def test_build_parity(ap, fx: Fixture, n_batches: int):
    """Gate off: the patched build() == the stock build() (plan log, metadata fields, state tag)."""
    rng = random.Random(20261023)
    total = bad = 0
    for _ in range(n_batches):
        reqs = T.random_batch(rng)
        if not reqs:
            continue
        for host in (True, False):
            for mha in (False, True):
                for adaptive in (False, True):
                    cam = T.make_cam(fx.env, reqs, with_host_bound=host)
                    bs, bp = fx.builder(fx.stock, mha, adaptive), fx.builder(fx.patched, mha, adaptive)
                    setgate(False)
                    ms, mp = bs.build(0, cam), bp.build(0, cam)
                    same = (bs.state.log == bp.state.log and ms.state is bs.state and mp.state is bp.state
                            and ms.num_decode_tokens == mp.num_decode_tokens and ms.num_prefills == mp.num_prefills
                            and not hasattr(bp.state, "_p020_plan_rows"))
                    total += 1
                    bad += not same
    check(f"D1 gate off: patched build() == stock build() on {total} builds "
          f"({n_batches} V2 batches x host bound/exact x sparse-MHA x adaptive)", bad == 0 and total > 0, f"{bad} differ")


def _stock_plan(fx, reqs, host, mha, adaptive):
    cam = T.make_cam(fx.env, reqs, with_host_bound=host)
    b = fx.builder(fx.stock, mha, adaptive)
    setgate(False)
    m = b.build(0, cam)
    return cam, m, b.state.log


def test_build_gate_on(ap, fx: Fixture, n_batches: int):
    rng = random.Random(7)
    n0 = nmix = ndec = nbad = 0
    reasons = []
    for _ in range(n_batches):
        reqs = T.random_batch(rng)
        if not reqs:
            continue
        for host in (True, False):
            for mha in (False, True):
                for adaptive in (False, True):
                    cam, ms, slog = _stock_plan(fx, reqs, host, mha, adaptive)
                    nd = ms.num_decode_tokens
                    rows = int(cam.num_actual_tokens)
                    b = fx.builder(fx.patched, mha, adaptive)
                    setgate(True)
                    try:
                        mp = b.build(0, cam)
                    finally:
                        setgate(False)
                    log = b.state.log
                    ok = mp.state is b.state and getattr(b.state, "_p020_plan_rows", None) == (min(nd, rows) if nd else 0)
                    if nd == 0:
                        n0 += 1
                        ok &= log == []
                    else:
                        # upstream's own plan over the first n = min(nd, rows) rows: the stock log when it
                        # planned ALL rows is sliced; when stock already planned the decode prefix (sparse
                        # MHA with prefill requests) it is equal
                        s = slog[0]
                        n = min(nd, rows)
                        want = ("plan", n, s[2][:n], s[3], s[4])
                        ok &= len(log) == 1 and log[0] == want
                        if nd == rows:
                            ndec += 1
                            ok &= log == slog  # pure decode: bit-identical plan call
                        else:
                            nmix += 1
                    nbad += not ok
                    if not ok and len(reasons) < 3:
                        reasons.append((nd, rows, host, mha, adaptive, log[:1], slog[:1]))
    check(f"D2 gate on: nd == 0 -> no plan ({n0}); nd > 0 -> upstream's plan over the first nd rows ({nmix} mixed); "
          f"pure decode == the stock plan call ({ndec})", nbad == 0 and n0 > 50 and nmix > 50 and ndec > 20, str(reasons))
    # independent oracle on the exact path: decode-prefix lens == the device truth
    rng = random.Random(8)
    bad = 0
    cnt = 0
    for _ in range(300):
        reqs = T.random_batch(rng, pure_prefill=False)
        if not reqs or reqs[0]["q"] > T.THRESH:
            continue
        cam = T.make_cam(fx.env, reqs, with_host_bound=False)
        b = fx.builder(fx.patched)
        setgate(True)
        try:
            mp = b.build(0, cam)
        finally:
            setgate(False)
        nd = mp.num_decode_tokens
        if nd == 0:
            continue
        cnt += 1
        oracle = T.exact_lens(reqs, 2048, 4)[:nd]
        bad += b.state.log[0][2] != oracle
    check(f"D3 gate on, exact path: the planned decode lengths equal an independent first-principles oracle ({cnt} batches)",
          bad == 0 and cnt > 50, f"{bad} differ")
    # runtime env gating: gate on then off in one process
    reqs = [dict(q=4, gpu_nc=100, cpu_nc=101), dict(q=900, gpu_nc=0, cpu_nc=0)]
    cam = T.make_cam(fx.env, reqs)
    b = fx.builder(fx.patched)
    setgate(True)
    b.build(0, cam)
    n_on = len(b.state.log)
    b.state.log.clear()
    setgate(False)
    b.build(0, cam)
    check("D4 the gate is read at runtime: on -> decode-prefix plan; off -> the stock all-rows plan again",
          n_on == 1 and b.state.log[0][1] == 904, str(b.state.log[:1]))


def _inputs(fx, T_rows, nd, heads=8, D=512, num_blocks=16, width=2176, seed=0, strided_q=False):
    """q (q_nope, q_rope), kv cache, topk buffer rows, attn_metadata for the impl's forward_mqa."""
    torch = fx.torch
    rng = np.random.default_rng(seed)
    if strided_q:
        base = rng.standard_normal((heads, T_rows, D)).astype(np.float32)
        qn = T.Tensor(T.round_bf16(base), torch.bfloat16, T._Device("cpu")).transpose(0, 1)
    else:
        qn = T.Tensor(T.round_bf16(rng.standard_normal((T_rows, heads, D)).astype(np.float32)), torch.bfloat16, T._Device("cpu"))
    qr = T.Tensor(np.zeros((T_rows, heads, 0), dtype=np.float32), torch.bfloat16, T._Device("cpu"))
    kv = T.Tensor(rng.integers(0, 120, size=(num_blocks, 64, D), dtype=np.uint8), torch.uint8, T._Device("cpu"))
    return qn, qr, kv


def _fill_topk(buf, T_rows, rng, num_rows):
    valid = rng.integers(1, buf.shape[1] + 1, size=T_rows)
    for r in range(T_rows):
        v = int(valid[r])
        buf.arr[r, :v] = rng.integers(0, num_rows, size=v)
        buf.arr[r, v:] = -1
    return valid


def _md(m, torch, state, nd, T_rows, block_size=64):
    return m.FlashInferMLASparseSM90Metadata(
        num_reqs=1, num_actual_tokens=T_rows, num_decode_tokens=nd, num_decodes=1 if nd else 0,
        num_prefills=1 if T_rows > nd else 0, block_size=block_size,
        req_id_per_token=T.Tensor(np.zeros(T_rows, dtype=np.int32), torch.int32, T._Device("cpu")),
        block_table=T.Tensor(np.zeros((1, 8), dtype=np.int32), torch.int32, T._Device("cpu")), state=state)


def _recorders(fx):
    """replace the module's Triton launch with a recorder that writes 3.0"""
    rec = []

    def fake_triton(q, latent, slots, vcount, sm_scale, k_scale=1.0, v_scale=None, out=None):
        rec.append(dict(q=q, latent=latent, slots=slots, vcount=vcount, sm_scale=sm_scale, k_scale=k_scale,
                        v_scale=v_scale, out=out))
        out.fill_(3.0)
        return out

    fx.mod.sparse_mla_prefill_triton = fake_triton
    return rec


def test_forward_parity_and_routing(ap, fx: Fixture):
    torch = fx.torch
    rng = np.random.default_rng(3)
    layer = T.FakeLayer(k_scale=0.5)
    ok_all = []
    # --- gate off: stock vs patched forward_mqa, identical call logs and outputs
    for T_rows, nd in ((6, 6), (7, 2), (5, 0)):
        qn, qr, kv = _inputs(fx, T_rows, nd, seed=T_rows)
        res = []
        for m in (fx.stock, fx.patched):
            impl, buf = fx.impl(m)
            _fill_topk(buf, T_rows, np.random.default_rng(1), 16 * 64)
            st = T.FakeState()
            setgate(False)
            out, lse = impl.forward_mqa((qn, qr), kv, _md(m, torch, st, nd, T_rows), layer)
            res.append((st.log, out.arr.copy(), lse))
        ok_all.append(res[0][0] == res[1][0] and np.array_equal(res[0][1], res[1][1]) and res[0][2] is None and res[1][2] is None)
    check("D5 gate off: patched forward_mqa == stock forward_mqa (pack/run call log, output, lse) for pure-decode / "
          "mixed / pure-prefill row splits", all(ok_all), str(ok_all))
    # --- gate on
    rec = _recorders(fx)
    # (a) pure decode: bit-identical call sequence to stock (the same pack + run, same tensors)
    T_rows = 6
    qn, qr, kv = _inputs(fx, T_rows, T_rows, seed=11)
    logs = []
    for m, on in ((fx.stock, False), (fx.patched, True)):
        impl, buf = fx.impl(m)
        _fill_topk(buf, T_rows, np.random.default_rng(2), 16 * 64)
        st = T.FakeState()
        setgate(on)
        if on:
            setattr(st, "_p020_plan_rows", T_rows)
        try:
            out, _ = impl.forward_mqa((qn, qr), kv, _md(m, torch, st, T_rows, T_rows), layer)
        finally:
            setgate(False)
        logs.append((st.log, out.arr.copy()))
    check("D6 gate on, pure decode: the same pack_indices + wrapper.run call sequence and output as stock; Triton not launched",
          logs[0][0] == logs[1][0] and np.array_equal(logs[0][1], logs[1][1]) and not rec, str(logs[1][0][:2]))
    # (b) mixed
    for T_rows, nd in ((9, 4), (130, 3), (8, 1)):
        qn, qr, kv = _inputs(fx, T_rows, nd, seed=20 + T_rows, strided_q=(T_rows == 130))
        impl, buf = fx.impl(fx.patched)
        valid = _fill_topk(buf, T_rows, np.random.default_rng(5), 16 * 64)
        st = T.FakeState()
        setattr(st, "_p020_plan_rows", nd)
        rec.clear()
        setgate(True)
        try:
            out, lse = impl.forward_mqa((qn, qr), kv, _md(fx.patched, torch, st, nd, T_rows), layer)
        finally:
            setgate(False)
        log = st.log
        r = rec[0] if rec else None
        ok = (lse is None and out.shape == (T_rows, 8, 512) and len(rec) == 1
              and [x[0] for x in log] == ["pack", "run"]
              and log[0][1] == (nd, 2176) and log[1][1] == (nd, 8, 512) and log[1][3] == (nd, 8, 0)
              and np.all(out.arr[:nd] == 7.0) and np.all(out.arr[nd:] == 3.0)
              and r["q"].shape == (T_rows - nd, 8, 512) and r["q"].data_ptr() == qn[nd:].data_ptr()
              and r["q"].stride() == qn.stride()
              and r["slots"].shape == (T_rows - nd, 2176) and r["vcount"].shape == (T_rows - nd,)
              and r["out"].data_ptr() == out[nd:].data_ptr()
              and r["sm_scale"] == 1.0 / 16 and r["k_scale"] == 0.5 and r["v_scale"] == 0.5
              and r["latent"].dtype is torch.float8_e4m3fn and r["latent"].shape == ((16 - 1) * 64 + 64, 512)
              and r["latent"].stride() == (512, 1)
              and np.array_equal(r["vcount"].arr, valid[nd:].astype(np.int32)))
        # the slots handed to Triton are the converted prefill rows, compacted, with the -1 tail
        if ok:
            for j in range(T_rows - nd):
                v = int(r["vcount"].arr[j])
                ok &= bool((r["slots"].arr[j, :v] >= 0).all() and (r["slots"].arr[j, v:] == -1).all())
        ok_all.append(ok)
        check(f"D7 gate on, mixed T={T_rows} nd={nd}: FA2 packs+runs ONLY rows [0,{nd}), Triton gets rows [{nd},{T_rows}) "
              "(strided q view, out[nd:] in place, K and V scale, row-view latent, valid_counts slice)", ok, str(log[:2]))
    # (c) pure prefill
    T_rows = 5
    qn, qr, kv = _inputs(fx, T_rows, 0, seed=31)
    impl, buf = fx.impl(fx.patched)
    _fill_topk(buf, T_rows, np.random.default_rng(6), 16 * 64)
    st = T.FakeState()
    setattr(st, "_p020_plan_rows", 0)
    rec.clear()
    setgate(True)
    try:
        out, _ = impl.forward_mqa((qn, qr), kv, _md(fx.patched, torch, st, 0, T_rows), layer)
    finally:
        setgate(False)
    check("D8 gate on, pure prefill (nd == 0): no FA2 call at all, Triton gets every row, output is the Triton one",
          st.log == [] and len(rec) == 1 and rec[0]["q"].shape[0] == T_rows and np.all(out.arr == 3.0))
    # (d) no plan for this metadata -> refuse
    st = T.FakeState()
    qn, qr, kv = _inputs(fx, 9, 4, seed=32)
    impl, buf = fx.impl(fx.patched)
    _fill_topk(buf, 9, np.random.default_rng(7), 16 * 64)
    setgate(True)
    try:
        r1 = raises(lambda: impl.forward_mqa((qn, qr), kv, _md(fx.patched, torch, st, 4, 9), layer), fx.mod.Patch020Error, "FA2 plan covers None rows")
        setattr(st, "_p020_plan_rows", 9)  # planned for all rows but forward runs 4 on FA2
        r2 = raises(lambda: impl.forward_mqa((qn, qr), kv, _md(fx.patched, torch, st, 4, 9), layer), fx.mod.Patch020Error, "plan covers 9 rows")
    finally:
        setgate(False)
    check("D9 gate on, a mixed batch whose metadata was not planned for exactly nd rows is refused (fail closed)", r1 and r2)


def test_plan_forward_chain(ap, fx: Fixture, n_batches: int):
    """build() then forward_mqa() on the same metadata for random V2 batches, both routings."""
    torch = fx.torch
    rng = random.Random(99)
    rec = _recorders(fx)
    layer = T.FakeLayer(k_scale=1.0)
    bad = n = 0
    why = []
    for _ in range(n_batches):
        reqs = T.random_batch(rng)
        if not reqs:
            continue
        cam = T.make_cam(fx.env, reqs)
        if int(cam.num_actual_tokens) > 8192:
            continue
        for mha in (False, True):
            b = fx.builder(fx.patched, mha)
            setgate(True)
            try:
                md = b.build(0, cam)
                nd = md.num_decode_tokens
                T_rows = int(cam.num_actual_tokens)
                mqa_rows = nd if (mha and md.num_prefills > 0) else T_rows  # rows forward_mqa receives
                if mqa_rows == 0:
                    continue
                qn, qr, kv = _inputs(fx, mqa_rows, nd, seed=n)
                setgate(False)
                impl, buf = fx.impl(fx.patched)  # (the gate-on __init__ would run the real self-test)
                setgate(True)
                _fill_topk(buf, mqa_rows, np.random.default_rng(n), 16 * 64)
                rec.clear()
                out, _ = impl.forward_mqa((qn, qr), kv, md, layer)
                fa2_rows = nd if nd < mqa_rows else mqa_rows
                runs = [x for x in b.state.log if x[0] == "run"]
                packs = [x for x in b.state.log if x[0] == "pack"]
                tri_rows = sum(r["q"].shape[0] for r in rec)
                ok = (tri_rows == mqa_rows - nd) and (len(runs) == (1 if fa2_rows else 0)) \
                    and all(x[1][0] == fa2_rows for x in runs + packs)
                # the planned rows equal the rows FA2 runs
                plans = [x for x in b.state.log if x[0] == "plan"]
                ok &= (plans[0][1] == fa2_rows) if fa2_rows else not plans
            except Exception as e:  # noqa: BLE001
                ok = False
                why.append(repr(e))
            finally:
                setgate(False)
            n += 1
            bad += not ok
    check(f"D10 build() -> forward_mqa() on the same metadata, {n} runs: planned rows == FA2 rows, Triton rows == the rest, "
          "no refusal", bad == 0 and n > 40, str(why[:2]))


def test_init_hook(ap, fx: Fixture):
    torch = fx.torch
    calls = []
    fx.mod.startup_check = lambda impl, device: calls.append((impl, str(device)))
    setgate(False)
    impl, _ = fx.impl(fx.patched)
    off = list(calls)
    setgate(True)
    try:
        impl2, _ = fx.impl(fx.patched)
    finally:
        setgate(False)
    check("D11 Impl.__init__: gate off -> no startup_check; gate on -> startup_check(impl, topk buffer device) once",
          off == [] and len(calls) == 1 and calls[0][0] is impl2 and calls[0][1] == "cpu")
    # stock __init__ is untouched by the gate
    setgate(True)
    try:
        calls.clear()
        fx.impl(fx.stock)
    finally:
        setgate(False)
    check("D12 the stock Impl never calls the module", calls == [])


# ------------------------------------------------------------------------------------------------
# E. module units
# ------------------------------------------------------------------------------------------------
def fresh_module():
    env = T.install_stubs()
    setgate(False)
    mod = T.load_file(env, "vllm_gb10_sparse_mla_prefill", MODULE_PATH, install_as="vllm._gb10_sparse_mla_prefill")
    return env, mod


class FB:  # fake builder for plan_step
    def __init__(self, env, rows, lens, exact):
        self.state = T.FakeState()
        self.rows, self.lens, self.exact = rows, lens, exact
        self.calls = 0

    def _kv_lens_host(self, cam):
        self.calls += 1
        return self.rows, self.lens, self.exact


def test_plan_step_units():
    env, mod = fresh_module()
    torch = env.torch
    lens = torch.tensor(list(range(100, 100 + 10)), dtype=torch.int64)
    cam = types.SimpleNamespace()
    md = lambda nd: types.SimpleNamespace(num_decode_tokens=nd, req_id_per_token=torch.arange(10, dtype=torch.int32))  # noqa: E731
    out = []
    b = FB(env, 10, lens, False)
    mod.plan_step(b, cam, md(0))
    out.append(b.state.log == [] and b.calls == 0 and b.state._p020_plan_rows == 0)
    b = FB(env, 10, lens, False)
    mod.plan_step(b, cam, md(4))
    out.append(len(b.state.log) == 1 and b.state.log[0][:4] == ("plan", 4, [100, 101, 102, 103], True)
               and b.state._p020_plan_rows == 4)
    b = FB(env, 10, lens, True)
    mod.plan_step(b, cam, md(10))
    out.append(b.state.log[0][:4] == ("plan", 10, list(range(100, 110)), False) and b.state.log[0][4] is None)
    b = FB(env, 10, lens, False)
    mod.plan_step(b, cam, md(25))  # nd > rows: plan every row (n = rows), the forward check then refuses
    out.append(b.state.log[0][1] == 10 and b.state._p020_plan_rows == 10)
    b = FB(env, 10, lens, False)
    out.append(raises(lambda: mod.plan_step(b, cam, types.SimpleNamespace()), mod.Patch020Error, "num_decode_tokens"))
    check("E1 plan_step: nd == 0 -> nothing (no host bound either); nd < rows -> first nd rows with the sync-free args; "
          "exact -> (None, None); nd > rows clamps; missing nd refused", all(out), str(out))


def _impl_ns(**kw):
    d = dict(num_heads=8, kv_lora_rank=512, head_size=512, qk_rope_head_dim=0, scale=1.0 / 16, use_fp8_kv_cache=True)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_forward_split_units():
    env, mod = fresh_module()
    torch = env.torch
    T_rows, nd, H, D = 6, 2, 8, 512
    seen = {}

    def fake_fa2(impl_, state_, q_nope, q_pe, topk_slots, kv_cache, layer_, n):
        seen["fa2"] = (n, q_nope is qn, topk_slots is slots)
        return T.Tensor(np.full((n, H, D), 7.0, dtype=np.float32), torch.bfloat16, T._Device("cpu"))

    def fake_triton(q, latent, slots_, vcount, sm_scale, k_scale=1.0, v_scale=None, out=None):
        seen["tri"] = dict(q=q, latent=latent, sm_scale=sm_scale, k_scale=k_scale, v_scale=v_scale, out=out, vcount=vcount)
        out.fill_(3.0)
        return out

    mod._fa2_run, mod.sparse_mla_prefill_triton = fake_fa2, fake_triton
    qn = T.Tensor(np.zeros((T_rows, H, D), dtype=np.float32), torch.bfloat16, T._Device("cpu"))
    qp = T.Tensor(np.zeros((T_rows, H, 0), dtype=np.float32), torch.bfloat16, T._Device("cpu"))
    slots = T.Tensor(np.zeros((T_rows, 64), dtype=np.int32), torch.int32, T._Device("cpu"))
    vc = T.Tensor(np.full(T_rows, 5, dtype=np.int32), torch.int32, T._Device("cpu"))
    kv = T.Tensor(np.zeros((300, D), dtype=np.uint8), torch.uint8, T._Device("cpu"))
    layer = types.SimpleNamespace(_k_scale_float=0.25)
    state = types.SimpleNamespace(_p020_plan_rows=nd)
    md = lambda n: types.SimpleNamespace(num_decode_tokens=n)  # noqa: E731
    # mixed
    out = mod.forward_split(_impl_ns(), state, qn, qp, slots, vc, kv, layer, md(nd))
    check("E2a forward_split mixed: FA2 on nd rows, Triton on the rest, K and V scale = layer._k_scale_float, out in place",
          seen["fa2"] == (nd, True, True) and np.all(out.arr[:nd] == 7.0) and np.all(out.arr[nd:] == 3.0)
          and seen["tri"]["k_scale"] == 0.25 == seen["tri"]["v_scale"] and seen["tri"]["sm_scale"] == 1.0 / 16
          and seen["tri"]["out"].data_ptr() == out[nd:].data_ptr() and seen["tri"]["latent"].shape == (300, D)
          and seen["tri"]["latent"].dtype is torch.float8_e4m3fn)
    # pure decode never touches Triton and passes the UNSLICED tensors
    seen.clear()
    mod.forward_split(_impl_ns(), types.SimpleNamespace(), qn, qp, slots, vc, kv, layer, md(T_rows))
    check("E2b forward_split pure decode: FA2 only, on all rows, no plan-rows check, no Triton", seen.get("fa2") == (T_rows, True, True) and "tri" not in seen)
    # pure prefill
    seen.clear()
    out = mod.forward_split(_impl_ns(), types.SimpleNamespace(_p020_plan_rows=0), qn, qp, slots, vc, kv, layer, md(0))
    check("E2c forward_split pure prefill: no FA2, all rows on Triton", "fa2" not in seen and seen["tri"]["q"].shape[0] == T_rows and np.all(out.arr == 3.0))
    # refusals
    refusals = []
    for bad, why in ((dict(use_fp8_kv_cache=False), "bf16 cache"), (dict(qk_rope_head_dim=64), "rope MLA"),
                     (dict(kv_lora_rank=256), "latent dim")):
        refusals.append(raises(lambda b=bad: mod.forward_split(_impl_ns(**b), state, qn, qp, slots, vc, kv, layer, md(nd)),
                               mod.Patch020Error))
    refusals.append(raises(lambda: mod.forward_split(_impl_ns(), state, qn, qp, slots, None, kv, layer, md(nd)), mod.Patch020Error, "valid_counts"))
    refusals.append(raises(lambda: mod.forward_split(_impl_ns(), types.SimpleNamespace(), qn, qp, slots, vc, kv, layer, md(nd)),
                           mod.Patch020Error, "covers None rows"))
    refusals.append(raises(lambda: mod.forward_split(_impl_ns(), types.SimpleNamespace(_p020_plan_rows=T_rows), qn, qp, slots, vc, kv, layer, md(nd)),
                           mod.Patch020Error, "covers 6 rows"))
    T._STATE["capturing"] = True
    try:
        refusals.append(raises(lambda: mod.forward_split(_impl_ns(), state, qn, qp, slots, vc, kv, layer, md(nd)), mod.Patch020Error, "CUDA graph capture"))
        cap_decode = True
        mod.forward_split(_impl_ns(), types.SimpleNamespace(), qn, qp, slots, vc, kv, layer, md(T_rows))  # decode inside capture is fine
    finally:
        T._STATE["capturing"] = False
    check("E2d forward_split refusals: bf16 cache / rope MLA / latent dim / valid_counts None / unplanned / wrongly planned / "
          "mixed batch inside graph capture (pure decode inside capture is allowed)", all(refusals) and cap_decode, str(refusals))
    # first-use log, once
    cap = T.LogCapture().attach(mod.logger.name)
    mod._FIRST_USE_LOGGED[0] = False
    mod.forward_split(_impl_ns(), state, qn, qp, slots, vc, kv, layer, md(nd))
    mod.forward_split(_impl_ns(), state, qn, qp, slots, vc, kv, layer, md(nd))
    check("E2e the first Triton batch is logged exactly once", len([r for r in cap.records if "first Triton prefill batch" in r]) == 1, str(cap.records))


def test_fa2_run_units():
    env, mod = fresh_module()
    torch = env.torch
    T_rows, H, D = 5, 8, 512
    st = T.FakeState()
    qn = T.Tensor(np.zeros((T_rows, H, D), dtype=np.float32), torch.bfloat16, T._Device("cpu"))
    qp = T.Tensor(np.zeros((T_rows, H, 0), dtype=np.float32), torch.bfloat16, T._Device("cpu"))
    slots = T.Tensor(np.zeros((T_rows, 32), dtype=np.int32), torch.int32, T._Device("cpu"))
    kv = T.Tensor(np.zeros((40, D), dtype=np.uint8), torch.uint8, T._Device("cpu"))
    layer = types.SimpleNamespace(_k_scale_float=0.0)
    mod._fa2_run(_impl_ns(), st, qn, qp, slots, kv, layer, T_rows)
    mod._fa2_run(_impl_ns(), st, qn, qp, slots, kv, layer, 2)
    mod._fa2_run(_impl_ns(use_fp8_kv_cache=False), st, qn, qp, slots, kv, layer, T_rows)
    l = st.log
    full = hashlib.sha1(np.ascontiguousarray(slots.arr).tobytes()).hexdigest()[:12]
    part = hashlib.sha1(np.ascontiguousarray(slots.arr[:2]).tobytes()).hexdigest()[:12]
    check("E3 _fa2_run: pack + run with upstream's arguments (unsliced for all rows, sliced for n < rows; ckv_scale "
          "falls back to 1.0; no scale kwargs for a bf16 cache)",
          [x[0] for x in l] == ["pack", "run"] * 3 and l[0][1] == (T_rows, 32) and l[0][2] == full and l[1][2] == qn.data_ptr()
          and l[2][1] == (2, 32) and l[2][2] == part and l[3][1] == (2, 8, 512) and l[3][8] == (("ckv_scale", 1.0), ("kpe_scale", 1.0))
          and l[1][5] == "torch.float8_e4m3fn" and l[5][8] == () and l[5][5] == "torch.uint8", str(l[:2]))


def test_startup_units():
    env, mod = fresh_module()
    torch = env.torch
    calls, warm = [], []
    mod.kernel_self_test = lambda device, num_heads, sm_scale, k_scale=0.5: calls.append((str(device), num_heads, sm_scale)) or {"P64_S1": {"finite": True}}
    mod._warm_unaligned_variant = lambda device, num_heads, sm_scale: warm.append((str(device), num_heads, sm_scale))
    cap = T.LogCapture().attach(mod.logger.name)
    refuse = []
    for bad in (dict(use_fp8_kv_cache=False), dict(qk_rope_head_dim=64), dict(head_size=576)):
        mod._STARTUP_OK.clear()
        refuse.append(raises(lambda b=bad: mod.startup_check(_impl_ns(**b), "cpu"), mod.Patch020Error))
    check("E4a startup_check refuses a bf16 cache, rope MLA and another head size; no kernel run", all(refuse) and not calls and not warm)
    mod._STARTUP_OK.clear()
    mod.startup_check(_impl_ns(), "cpu")
    mod.startup_check(_impl_ns(), "cpu")  # the MTP draft impl: same config -> no second self-test, no second line
    mod.startup_check(_impl_ns(num_heads=16), "cpu")
    mod.startup_check(_impl_ns(), "cuda:0")  # the cuda branch (`with torch.cuda.device(dev)`): a distinct (config, device) key
    act = [r for r in cap.records if r.startswith("[patch-020] ACTIVE")]
    ok = (calls == [("cpu", 8, 1.0 / 16), ("cpu", 16, 1.0 / 16), ("cuda:0", 8, 1.0 / 16)] and warm == calls and len(act) == 3
          and any("self-test OK" in r for r in cap.records) and all("NOSYNC is ignored" in r for r in act))
    check("E4b startup_check: self-test + pre-compile of the unaligned variant + one ACTIVE line per distinct (config, device), "
          "not per impl", ok, str(cap.records[:3]))
    # a failing warm-up is non-fatal: one warning, the ACTIVE line still appears
    mod._STARTUP_OK.clear()
    cap.records.clear()

    def boom(device, num_heads, sm_scale):
        raise RuntimeError("compile failed")

    mod._warm_unaligned_variant = boom
    mod.startup_check(_impl_ns(), "cpu")
    ok = (len([r for r in cap.records if "could not pre-compile the unaligned valid_counts variant" in r]) == 1
          and len([r for r in cap.records if r.startswith("[patch-020] ACTIVE")]) == 1)
    check("E4c a failing pre-compile of the unaligned variant is non-fatal: one warning, ACTIVE still logged", ok, str(cap.records))
    # a failing self-test is fatal (propagates), nothing is cached, no ACTIVE line
    mod._STARTUP_OK.clear()
    cap.records.clear()

    def bad_selftest(device, num_heads, sm_scale, k_scale=0.5):
        raise mod.Patch020Error("[patch-020] Triton sparse-MLA kernel self-test failed")

    mod.kernel_self_test = bad_selftest
    r = raises(lambda: mod.startup_check(_impl_ns(), "cpu"), mod.Patch020Error, "self-test failed")
    check("E4d a failing kernel self-test is fatal and leaves no ACTIVE line / cache entry", r and not mod._STARTUP_OK
          and not [x for x in cap.records if x.startswith("[patch-020] ACTIVE")])


def test_warm_unaligned_units():
    """The pre-compile of the second Triton specialisation: valid_counts pointer 4 bytes past a 16-byte boundary,
    everything else aligned, S == 1, inputs explicit on the CPU in fp32 (loader context)."""
    env, mod = fresh_module()
    torch = env.torch
    seen = []

    def fake_kernel(q, latent, slots, valid, sm_scale, k_scale=1.0, v_scale=None, out=None):
        seen.append(dict(q=q, latent=latent, slots=slots, valid=valid, sm_scale=sm_scale, k_scale=k_scale))
        return q

    mod.sparse_mla_prefill_triton = fake_kernel
    old_dt = torch.get_default_dtype()
    T._STATE["strict"] = True
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            mod._warm_unaligned_variant("cpu", 8, 1.0 / 16)
    finally:
        T._STATE["strict"] = False
        torch.set_default_dtype(old_dt)
    s = seen[0] if seen else None
    ok = (len(seen) == 1 and s["valid"].data_ptr() % 16 == 4 and s["q"].data_ptr() % 16 == 0 and s["slots"].data_ptr() % 16 == 0
          and s["latent"].dtype is torch.float8_e4m3fn and s["q"].shape[0] == mod._TWO_SPLIT_MAX_ROWS + 16 > mod._TWO_SPLIT_MAX_ROWS
          and s["valid"].shape == (s["q"].shape[0],) and s["slots"].shape[0] == s["q"].shape[0]
          and s["valid"].dtype is torch.int32 and s["slots"].dtype is torch.int32 and s["q"].dtype is torch.bfloat16
          and mod._choose_num_splits(s["q"].shape[0]) == 1)
    check("E4e _warm_unaligned_variant: one S=1 launch whose valid_counts pointer is 4 bytes past a 16-byte boundary while q / "
          "slots stay 16-byte aligned; inputs built explicitly on the CPU under a meta device + bf16 default dtype", ok,
          str({k: (v.data_ptr() % 16 if hasattr(v, "data_ptr") else v) for k, v in (s or {}).items()}))


def test_selftest_under_loader_context():
    """kernel_self_test runs from Impl.__init__ inside `with torch.device(cuda:N)` + a bf16 default dtype: every
    input must carry an explicit CPU device and dtype (a CPU generator on another default device raises)."""
    env, mod = fresh_module()
    torch = env.torch
    seen = []

    def fake_kernel(q, latent, slots, valid, sm_scale, k_scale=1.0, v_scale=None, out=None):
        seen.append((q.device.type, q.dtype.name, latent.device.type, slots.device.type, valid.device.type))
        return mod._torch_reference(q, latent, slots, valid, sm_scale, k_scale, k_scale).to(torch.bfloat16)

    mod.sparse_mla_prefill_triton = fake_kernel
    old_dt = torch.get_default_dtype()
    T._STATE["strict"] = True
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            rep = mod.kernel_self_test("cpu", 8, 1.0 / 16)
    finally:
        T._STATE["strict"] = False
        torch.set_default_dtype(old_dt)
    ok = len(seen) == 2 and all(s == ("cpu", "bfloat16", "cpu", "cpu", "cpu") for s in seen) and all(v["finite"] for v in rep.values())
    check("E5 kernel_self_test under a meta default device + bf16 default dtype: every input on the CPU, 2 cases finite "
          "and within the sanity tolerance (against the module's own fp64 reference)", ok, str(seen))
    # and the strict mode really bites when an input forgets the device
    T._STATE["strict"] = True
    try:
        with torch.device("meta"):
            bites = raises(lambda: torch.zeros(3, dtype=torch.float32), RuntimeError, "explicit device")
    finally:
        T._STATE["strict"] = False
    check("E5b (control) the strict fake-torch mode rejects a tensor created without an explicit device", bites)


def test_merge_units():
    env, mod = fresh_module()
    torch = env.torch
    rng = np.random.default_rng(0)
    P, H, D, N = 5, 8, 64, 130
    q = rng.standard_normal((P, H, D))
    k = rng.standard_normal((N, D))
    scores = np.einsum("phd,nd->phn", q, k) / 16
    e = np.exp(scores - scores.max(-1, keepdims=True))
    ref = np.einsum("phn,nd->phd", e / e.sum(-1, keepdims=True), k)

    def part(lo, hi):
        s = scores[..., lo:hi]
        m = s.max(-1)
        ee = np.exp(s - m[..., None])
        l = ee.sum(-1)
        o = np.einsum("phn,nd->phd", ee, k[lo:hi]) / np.maximum(l, 1e-30)[..., None]
        return o, np.where(l > 0, m + np.log(np.maximum(l, 1e-300)), -3.4e38)

    a0, l0 = part(0, 64)
    a1, l1 = part(64, N)
    mk = lambda a: T.Tensor(a.astype(np.float32), torch.float32, T._Device("cpu"))  # noqa: E731
    merged = mod._merge_splits(mk(np.stack([a0, a1])), mk(np.stack([l0, l1])))
    err = float(np.abs(merged.arr - ref).max())
    z = mod._merge_splits(mk(np.zeros((2, 1, H, D))), mk(np.full((2, 1, H), -3.4e38)))
    check(f"E6 split-LSE merge vs direct softmax: max abs err {err:.1e} < 1e-4; an all-empty row stays 0",
          err < 1e-4 and np.all(z.arr == 0.0))
    check("E6b _choose_num_splits: S=2 iff 0 < P <= 48", [mod._choose_num_splits(p) for p in (0, 1, 48, 49, 200)] == [1, 2, 2, 1, 1])


def test_unchanged_sources():
    cur = source_hashes(MODULE_PATH.read_text())
    bad = [k for k, v in UNCHANGED_SHA.items() if cur.get(k) != v]
    check("E7 the Triton kernel, host wrapper, merge, self-test, config check and frozen config are source-identical "
          f"to revision 3 (GB10-validated, sha 6346bd7b...): {len(UNCHANGED_SHA)} objects", not bad, str(bad))
    t = MODULE_PATH.read_text()
    gone = [s for s in ("nosync_enabled", "_kv_lens_from_upper_bound", "_check_part_a_invariants", "_assert_host_lens_match_device",
                        "_spec_max_query_len", "_assert_async") if s in t]
    check("E8 Part A (NOSYNC) is gone from the module", not gone, str(gone))
    check("E8b the module reads only the TRITON_PREFILL gate", re.findall(r'"(VLLM_GB10_[A-Z0-9_]+)"', t) == [GATE], str(re.findall(r'"(VLLM_GB10_[A-Z0-9_]+)"', t)))


# ------------------------------------------------------------------------------------------------
# F. mutants
# ------------------------------------------------------------------------------------------------
MUTANTS = [
    # (id, target, old, new, description) — target: 'patched' (the applied backend), 'module', 'applier'
    ("hook-before-state", "patched", "        state = attn_metadata.state\n        assert state is not None\n        # [patch-020 gb10-sparse-mla-triton-prefill] prefill rows -> Triton, decode rows -> FA2 (opt-in)\n",
     "        # [patch-020 gb10-sparse-mla-triton-prefill] prefill rows -> Triton, decode rows -> FA2 (opt-in)\n        state = attn_metadata.state\n        assert state is not None\n",
     "comment moved above `state = ...` (harmless control: the suite must stay green)"),
    ("swap-valid-counts", "patched", "                topk_slots,\n                _p020_valid_counts,\n", "                _p020_valid_counts,\n                topk_slots,\n",
     "forward hook passes valid_counts and topk_slots swapped"),
    ("forget-state-tag", "patched", "            _p020_module().plan_step(self, common_attn_metadata, metadata)\n            metadata.state = self.state\n",
     "            _p020_module().plan_step(self, common_attn_metadata, metadata)\n", "builder hook does not tag metadata.state"),
    ("fall-through", "patched", "            metadata.state = self.state\n            return metadata\n        num_rows, kv_lens, exact",
     "            metadata.state = self.state\n        num_rows, kv_lens, exact", "builder hook does not return: upstream plans again"),
    ("ungated-init", "patched", "        if _p020_triton_on():\n            _p020_module().startup_check(", "        if True:\n            _p020_module().startup_check(",
     "startup hook ignores the gate"),
    ("ungated-plan", "patched", "        if _p020_triton_on():\n            _p020_module().plan_step(", "        if True:\n            _p020_module().plan_step(",
     "plan hook ignores the gate"),
    ("wrong-rows-plan", "module", "    n = min(nd, int(num_rows))\n", "    n = int(num_rows)\n", "plan_step plans ALL rows"),
    ("plan-nd0", "module", "    if nd == 0:\n        setattr(state, _PLAN_ROWS_ATTR, 0)\n        return\n",
     "    if nd == 0:\n        setattr(state, _PLAN_ROWS_ATTR, 0)\n", "plan_step does not return on nd == 0"),
    ("no-slice-lens", "module", "    if n < int(num_rows):\n        kv_lens = kv_lens[:n]\n", "", "plan_step forgets to slice kv_lens"),
    ("fa2-no-qpe-slice", "module", "        q_nope, q_pe, topk_slots = q_nope[:n], q_pe[:n], topk_slots[:n]\n",
     "        q_nope, topk_slots = q_nope[:n], topk_slots[:n]\n", "_fa2_run does not slice q_pe"),
    ("fa2-pack-all", "module", "        q_nope, q_pe, topk_slots = q_nope[:n], q_pe[:n], topk_slots[:n]\n    state.pack_indices(topk_slots)\n",
     "        q_nope, q_pe = q_nope[:n], q_pe[:n]\n    state.pack_indices(topk_slots)\n", "_fa2_run packs every row"),
    ("tri-all-slots", "module", "        topk_slots[nd:],\n        valid_counts[nd:],\n", "        topk_slots,\n        valid_counts[nd:],\n", "Triton gets all slot rows"),
    ("tri-v-scale", "module", "        v_scale=k_scale,\n        out=out[nd:],\n", "        v_scale=1.0,\n        out=out[nd:],\n", "V not dequantised"),
    ("tri-latent-slice", "module", "[:, : impl.kv_lora_rank]\n    k_scale", "[:, :256]\n    k_scale", "latent sliced to 256 columns"),
    ("no-plan-check", "module", "    planned = getattr(state, _PLAN_ROWS_ATTR, None)\n    _require(\n        planned == nd,",
     "    planned = getattr(state, _PLAN_ROWS_ATTR, None)\n    _require(\n        True,", "plan-rows consistency check disabled"),
    ("decode-via-triton", "module", "    if nd == num_tokens:\n        # Pure decode", "    if nd == num_tokens + 1:\n        # Pure decode", "pure decode no longer takes the FA2-only path"),
    ("no-capture-check", "module", "    _require(\n        not _capturing(),", "    _require(\n        True,", "graph-capture refusal disabled"),
    ("startup-no-fp8-check", "module", "    _require(\n        bool(impl.use_fp8_kv_cache),", "    _require(\n        True,", "bf16 cache accepted"),
    ("warm-aligned", "module", "sparse_mla_prefill_triton(q, lat8, slots, valid[1:], sm_scale, k_scale=1.0)  # valid[1:]",
     "sparse_mla_prefill_triton(q, lat8, slots, valid[:P], sm_scale, k_scale=1.0)  # valid[1:]", "pre-compile launches the ALIGNED variant"),
    ("warm-fatal", "module", "    except Exception as e:  # noqa: BLE001 - only costs a one-off compile inside the first mixed step\n",
     "    except ZeroDivisionError as e:  # noqa: BLE001 - only costs a one-off compile inside the first mixed step\n", "pre-compile failure becomes fatal"),
    ("active-twice", "module", "    if key in _STARTUP_OK:\n        return\n", "", "startup_check repeats the self-test"),
    ("import-fail-open", "applier", "    if _p020_triton_on():\n        raise RuntimeError(", "    if False:\n        raise RuntimeError(", "import block fails open"),
]


def run_mutants(vllm_root: Path, baked_text: str, ap, baked_root: Path | None = None) -> bool:
    base_patched = ap.apply(baked_text)
    module_text = MODULE_PATH.read_text()
    applier_text = APPLIER_PATH.read_text()
    ok_all = True
    detected = 0
    expected = 0
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for mid, target, old, new, desc in MUTANTS:
            control = mid == "hook-before-state"  # a harmless comment move: the suite must stay GREEN
            env = dict(os.environ, P020_MUTANT_RUN="1")
            if target == "patched":
                assert base_patched.count(old) == 1, f"mutant {mid}: pattern x{base_patched.count(old)}"
                p = d / f"{mid}.py"
                p.write_text(base_patched.replace(old, new, 1))
                env["P020_PATCHED_OVERRIDE"] = str(p)
            elif target == "module":
                assert module_text.count(old) == 1, f"mutant {mid}: pattern x{module_text.count(old)}"
                p = d / f"{mid}_module.py"
                p.write_text(module_text.replace(old, new, 1))
                env["P020_MODULE_OVERRIDE"] = str(p)
            else:
                if mid == "import-fail-open":
                    src = applier_text
                    old2 = '    "    if _p020_triton_on():\\n"\n    "        raise RuntimeError(\\n"'
                    assert src.count(old2) == 1, f"mutant {mid}: pattern x{src.count(old2)}"
                    p = d / f"{mid}_apply.py"
                    p.write_text(src.replace(old2, '    "    if False:\\n"\n    "        raise RuntimeError(\\n"', 1))
                env["P020_APPLIER_OVERRIDE"] = str(p)
            cmd = [sys.executable, "-I", str(HERE / "p020_cpu_check.py"), "--vllm-root", str(vllm_root),
                   "--quick", "--quiet", "--only", "BCDEF" if target == "applier" else "DE"]
            if baked_root is not None:
                cmd += ["--baked-root", str(baked_root)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            expected += 1
            caught = r.returncode != 0
            good = (not caught) if control else caught
            detected += good
            ok_all &= good
            tag = ("CONTROL OK " if good else "CONTROL BAD") if control else ("CAUGHT     " if caught else "MISSED     ")
            print(f"{tag} {mid}: {desc}", flush=True)
            if not good:
                print(r.stdout[-400:])
    check(f"F mutants: {detected}/{expected} planted bugs make the suite fail (and the harmless control stays green)", ok_all)
    return ok_all


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--vllm-root", required=True, type=Path, help="<v0.32.0>/vllm (PRISTINE files)")
    ap_.add_argument("--baked-root", type=Path, default=None,
                     help="<tree>/vllm produced by running the image build's glm53_sm121_v032.py on a copy of the pristine "
                          "tree; its SM90 file replaces the derived BAKED base (sha must be the pin)")
    ap_.add_argument("--quick", action="store_true")
    ap_.add_argument("--quiet", action="store_true")
    ap_.add_argument("--mutants", action="store_true")
    ap_.add_argument("--only", default="ABCDEF")
    args = ap_.parse_args()
    QUIET["on"] = args.quiet
    n = 60 if args.quick else 400
    ap = load_applier()
    stock = (args.vllm_root / SM_REL).read_text()
    only = args.only
    bases = make_bases(stock)
    if args.baked_root is not None:
        bases["BAKED"] = (args.baked_root / SM_REL).read_text()
    if "A" in only:
        test_pins_and_applier(ap, stock, bases, args.baked_root is not None)
    if "B" in only:
        test_import_block(ap)
    if "C" in only:
        test_probe(ap, args.vllm_root, bases, args.baked_root)
    if "D" in only:
        # the deployable target is the BAKED file; STOCK / ALT only prove the hooks do not depend on the bake's spelling
        labels = [("BAKED", bases["BAKED"])] + ([] if (args.quick or PATCHED_OVERRIDE) else
                                               [("STOCK", stock), ("ALT", bases["ALT"])])
        for label, base_text in labels:
            ptxt = Path(PATCHED_OVERRIDE).read_text() if PATCHED_OVERRIDE else ap.apply(base_text)
            if not QUIET["on"]:
                print(f"-- D: {label} base", flush=True)
            fx = Fixture(ap, ptxt, base_text)
            test_build_parity(ap, fx, n)
            test_build_gate_on(ap, fx, n)
            fx = Fixture(ap, ptxt, base_text)
            test_forward_parity_and_routing(ap, fx)
            fx = Fixture(ap, ptxt, base_text)
            test_plan_forward_chain(ap, fx, n)
            fx = Fixture(ap, ptxt, base_text)
            test_init_hook(ap, fx)
    if "E" in only:
        test_plan_step_units()
        test_forward_split_units()
        test_fa2_run_units()
        test_startup_units()
        test_warm_unaligned_units()
        test_selftest_under_loader_context()
        test_merge_units()
        test_unchanged_sources()
    if "F" in only and args.mutants:
        run_mutants(args.vllm_root, bases["BAKED"], ap, args.baked_root)
    setgate(False)
    print(f"[020] CPU checks: {RESULTS['pass']} passed, {RESULTS['fail']} failed" +
          ("" if not RESULTS["failed"] else "  FAILED: " + "; ".join(RESULTS["failed"][:8])))
    print("[020] CPU self-test:", "ALL PASS" if RESULTS["fail"] == 0 else "FAILURES")
    return 0 if RESULTS["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
