#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 011 applier (v0.32.0 port): four anchored edits to vllm/model_executor/kernels/mhc/tilelang_kernels.py.

  apply(text) -> text     v0.32.0 original (or the patch-019 result) -> + patch 011; fail-closed: every anchor must
                          match exactly once, the result must carry exactly N_MARKERS markers and compile, and the
                          AST-level scope is checked (only HcPrenormGemmTileLangKernel.{dispatch,get_warmup_keys,
                          warmup_inputs} changed, plus one module-level assignment)
  revert(text) -> text    exact inverse (the tests rebuild the base from a patched file)

CLI (used by 011-mhc-prenorm-bm2-threshold.sh):
  p011_apply.py apply  <tilelang_kernels.py>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on refusal
  p011_apply.py check  <tilelang_kernels.py>   dry run: print the sha256 the file would have after apply()
  p011_apply.py revert <tilelang_kernels.py>   in-place inverse (tests / manual rollback)

Why a rewrite: in vLLM <= 0.28 the routing was `if n_splits == 1 and use_default_config and x.shape[0] >= 1024:` in
tilelang.py::_tilelang_hc_prenorm_gemm. Since #50178 (v0.30.0) the kernels are class-based JIT kernels with a warmup
registry; the routing decision is `HcPrenormGemmTileLangKernel.dispatch` (an AST-traced compile-key function) and the
warmup keys are derived from the same threshold. v0.32.0 edits (image line numbers, tilelang_kernels.py sha256 61a384a0...):
  1. after `ENABLE_PDL = ...` (:24): module constant _P011_BM2_MIN_T = int(VLLM_GB10_MHC_BM2_MIN_T or 1024), read once
     at import (the traced dispatch resolves module globals, so no per-call os.environ lookup);
  2. dispatch (:1148): `use_block_m = use_default_config and num_tokens >= 1024` -> `>= _P011_BM2_MIN_T`;
  3. get_warmup_keys (:1182): the WarmupIntRange bucket lambda `128 if v < 128 else 1024 if v < 1024 else max_tokens+1`
     -> the routed threshold (so the warmed compile keys are exactly the ones the dispatch can produce);
  4. warmup_inputs (:1194): the block_m=2 warmup tensor must itself route to block_m=2:
     `num_tokens = 1024` -> `max(1024, _P011_BM2_MIN_T)` (only differs for thresholds above 1,024; the warmup helper
     raises "TileLang warmup inputs produced a different compile key" if the inputs re-dispatch to another key).
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-011 mhc-prenorm-bm2-threshold]"
N_MARKERS = 4
ENV = "VLLM_GB10_MHC_BM2_MIN_T"
CLASS = "HcPrenormGemmTileLangKernel"

# ---- Edit 1: module constant ----
CONST_OLD = "ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()\n"
CONST_NEW = (
    CONST_OLD
    + "\n"
    + f"{MARKER} opt-in {ENV}=<tokens> (read once, at import; >= 128): the mHC pre-norm\n"
    + "# GEMM takes the block_m=2 kernel from this many tokens instead of 1,024 (bit-identical per token: both kernels\n"
    + "# accumulate k = it*512 + tid in the same order with the same warp / cross-warp reductions; the generic per-token\n"
    + "# kernel re-reads its 786 KB fp32 fn slice per CTA, 33 GB/s on GB10). Unset -> 1,024 (upstream).\n"
    + f"_P011_BM2_MIN_T = int(__import__(\"os\").environ.get(\"{ENV}\", \"1024\"))\n"
)

# ---- Edit 2: dispatch ----
DISPATCH_OLD = "        use_block_m = use_default_config and num_tokens >= 1024\n"
DISPATCH_NEW = (
    f"        {MARKER} routed threshold (upstream: 1024)\n"
    "        use_block_m = use_default_config and num_tokens >= _P011_BM2_MIN_T\n"
)

# ---- Edit 3: warmup bucket lambda ----
LAMBDA_OLD = "                    128 if value < 128 else 1024 if value < 1024 else max_tokens + 1\n"
LAMBDA_NEW = (
    f"                    {MARKER} block_m=2 bucket at the routed threshold\n"
    "                    128 if value < 128 else _P011_BM2_MIN_T if value < _P011_BM2_MIN_T else max_tokens + 1\n"
)

# ---- Edit 4: warmup_inputs ----
INPUTS_OLD = (
    "        if compile_key.use_block_m:\n"
    "            num_tokens = 1024\n"
)
INPUTS_NEW = (
    "        if compile_key.use_block_m:\n"
    f"            {MARKER} the warmup tensor must re-dispatch to block_m=2\n"
    "            num_tokens = max(1024, _P011_BM2_MIN_T)\n"
)

EDITS = [
    ("module constant (after ENABLE_PDL)", CONST_OLD, CONST_NEW),
    ("HcPrenormGemmTileLangKernel.dispatch threshold", DISPATCH_OLD, DISPATCH_NEW),
    ("HcPrenormGemmTileLangKernel.get_warmup_keys bucket lambda", LAMBDA_OLD, LAMBDA_NEW),
    ("HcPrenormGemmTileLangKernel.warmup_inputs block_m=2 tensor", INPUTS_OLD, INPUTS_NEW),
]


class Refused(ValueError):
    pass


def _class_method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    cs = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    if len(cs) != 1:
        raise Refused(f"class {cls} defined {len(cs)}x at module level (expected 1)")
    ms = [n for n in cs[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(ms) != 1:
        raise Refused(f"{cls}.{name} defined {len(ms)}x (expected 1)")
    return ms[0]


def _strip_doc(body):
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def check_structure(text: str, patched: bool) -> None:
    """AST checks of the base (patched=False) or the result (patched=True)."""
    tree = ast.parse(text)
    disp = _class_method(tree, CLASS, "dispatch")
    wk = _class_method(tree, CLASS, "get_warmup_keys")
    wi = _class_method(tree, CLASS, "warmup_inputs")
    # The dispatch is AST-traced by the warmup registry: only plain assignments, then ONE `return CompileKey(...)`.
    body = _strip_doc(disp.body)
    if not body or not isinstance(body[-1], ast.Return) or not isinstance(body[-1].value, ast.Call):
        raise Refused(f"{CLASS}.dispatch does not end in `return CompileKey(...)` (the traced-dispatch contract)")
    if not all(isinstance(s, (ast.Assign, ast.AnnAssign)) for s in body[:-1]):
        raise Refused(f"{CLASS}.dispatch has a non-assignment statement before the return (traced-dispatch contract)")
    ub = [s for s in body[:-1] if isinstance(s, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "use_block_m" for t in s.targets)]
    if len(ub) != 1:
        raise Refused(f"{CLASS}.dispatch: expected one `use_block_m = ...` assignment, found {len(ub)}")
    names = [n.id for n in ast.walk(ub[0].value) if isinstance(n, ast.Name)]
    consts = [n.value for n in ast.walk(ub[0].value) if isinstance(n, ast.Constant)]
    has_global = "_P011_BM2_MIN_T" in names
    if not patched:
        if has_global or any("_P011_BM2_MIN_T" in ast.dump(n) for n in (disp, wk, wi)):
            raise Refused("base already mentions _P011_BM2_MIN_T")
        if 1024 not in consts or "num_tokens" not in names:
            raise Refused("base use_block_m is not `use_default_config and num_tokens >= 1024`")
        return
    if not has_global or 1024 in consts:
        raise Refused("patched use_block_m does not compare num_tokens with _P011_BM2_MIN_T")
    gates = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_P011_BM2_MIN_T" for t in n.targets)]
    cls_idx = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef) and n.name == CLASS)
    if len(gates) != 1 or tree.body.index(gates[0]) > cls_idx:
        raise Refused("expected one module-level _P011_BM2_MIN_T assignment before " + CLASS)
    for fn, nm in ((wk, "get_warmup_keys"), (wi, "warmup_inputs")):
        if "_P011_BM2_MIN_T" not in ast.dump(fn):
            raise Refused(f"{CLASS}.{nm} does not use _P011_BM2_MIN_T")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-011 marker")
    bad = [(name, text.count(old)) for name, old, _ in EDITS if text.count(old) != 1]
    if bad:
        raise Refused("; ".join(f"anchor '{n}' found {c}x (expected 1)" for n, c in bad))
    check_structure(text, patched=False)
    out = text
    for name, old, new in EDITS:
        if out.count(old) != 1:
            raise Refused(f"anchor '{name}' changed by an earlier edit")
        out = out.replace(old, new, 1)
    if out.count(MARKER) != N_MARKERS:
        raise Refused(f"expected {N_MARKERS} markers after the edits, found {out.count(MARKER)}")
    compile(out, "tilelang_kernels.py", "exec")  # in memory: no .pyc side effects
    check_structure(out, patched=True)
    return out


def revert(text: str) -> str:
    out = text
    for name, old, new in reversed(EDITS):
        if out.count(new) != 1:
            raise Refused(f"patched block '{name}' found {out.count(new)}x (expected 1)")
        out = out.replace(new, old, 1)
    if MARKER in out:
        raise Refused("marker still present after revert")
    compile(out, "tilelang_kernels.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p011_apply.py {apply|check|revert} <tilelang_kernels.py>", file=sys.stderr)
        return 2
    p = Path(argv[2])
    try:
        src = p.read_text(encoding="utf-8")
        new = revert(src) if argv[1] == "revert" else apply(src)
    except (Refused, SyntaxError, OSError) as e:
        print(f"[FAIL] {p}: {e}", file=sys.stderr)
        return 1
    if argv[1] == "check":
        print(_sha(new.encode()))
        return 0
    tmp = p.with_name(p.name + ".p011.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if argv[1] == "revert" else "applied"
    print(f" [OK] {len(EDITS)} anchored edits {what} on {p.name} "
          "(_P011_BM2_MIN_T; HcPrenormGemmTileLangKernel dispatch + warmup keys + warmup inputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
