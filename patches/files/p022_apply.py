#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 022 applier (v0.32.0 port): two anchored edits to vllm/model_executor/kernels/mhc/tilelang.py.

  apply(text) -> text     v0.32.0 original / patch-019 result -> + patch 022; fail-closed: every anchor must match
                          exactly once, the hook must land in the TileLang branch of _hc_prenorm_gemm_outputs
                          (AST-checked), the result must carry exactly N_MARKERS markers and compile
  revert(text) -> text    exact inverse (the tests use it to rebuild the base from a patched file)

CLI (used by 022-gb10-mhc-prefill-rowsplit.sh):
  p022_apply.py apply  <tilelang.py>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p022_apply.py check  <tilelang.py>   dry run: print the sha256 the file would have after apply()
  p022_apply.py revert <tilelang.py>   in-place inverse (tests / manual rollback)

Edits (v0.32.0 line numbers, tilelang.py sha256 204e1920...). The image-era choke point `_tilelang_hc_prenorm_gemm` is
gone (#50178): every GLM5Next pre-norm GEMM - layer-0 mhc_pre_tilelang and the num_tokens > 32 branch of
mhc_fused_post_pre_tilelang - now goes through `_hc_prenorm_gemm_outputs`, which takes the DeepGEMM
`tf32_hc_prenorm_gemm` when `is_deep_gemm_supported()` and otherwise the TileLang class instance
`_HC_PRENORM_GEMM_TILELANG_KERNEL` (n_splits = 1). With VLLM_USE_DEEP_GEMM=0 (the GB10 recipe) `is_deep_gemm_supported()`
is False, so the TileLang branch is the one every call takes; the hook sits there (the DeepGEMM branch is never hooked:
a split-k tf32 kernel with its own numerics, not the per-row kernel the bit-identity argument covers).
  1. before `def _hc_prenorm_gemm_outputs(` (:22): module gate _P022_ON, read once from VLLM_GB10_MHC_PRS at import,
     and - only when set - `from vllm import _gb10_mhc_prs as _p022` (gate set + module missing -> ImportError at
     import: fail closed);
  2. in the TileLang branch of _hc_prenorm_gemm_outputs (:59-70), before the kernel call: if _P022_ON and
     _p022.prenorm_rowsplit(_HC_PRENORM_GEMM_TILELANG_KERNEL, x, fn, out, sqrsum, hidden_size, hc_mult): return the
     filled (out, sqrsum). Otherwise the upstream call runs, unchanged. The module asks the class's own dispatch() whether
     this call routes to the block_m=2 kernel and runs exactly that JIT function on this rank's rows.
The delayed variants (mhc_pre_delayed_tilelang / mhc_fused_post_pre_delayed_tilelang, DeepSeek-V4 shifted mHC) call the
class directly and are not hooked (GLM5Next never reaches them). Patch 019 edits `mhc_fused_post_pre_tilelang` and a
module gate elsewhere in the same file; the anchors are disjoint, so 019 and 022 commute.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-022 gb10-mhc-prefill-rowsplit]"
N_MARKERS = 2
ENV = "VLLM_GB10_MHC_PRS"
FUNC = "_hc_prenorm_gemm_outputs"
KERNEL = "_HC_PRENORM_GEMM_TILELANG_KERNEL"

GATE_OLD = "\n\ndef _hc_prenorm_gemm_outputs(\n"
GATE_NEW = (
    "\n\n"
    f"{MARKER} opt-in {ENV}=1 (read once, at import): prefill calls of the TileLang\n"
    f"# pre-norm GEMM below (T >= {ENV}_MIN_T, default 1,024, routed to the block_m=2 kernel; eager only, never inside\n"
    "# a CUDA-graph capture) run on this TP rank's token rows and all-gather the per-token outputs\n"
    "# (vllm/_gb10_mhc_prs.py) - bit-identical: the same kernel computes every row with the same arithmetic. Unset ->\n"
    "# unchanged. Gate set + module missing -> ImportError. VLLM_USE_DEEP_GEMM=0 (GB10) makes this the only path.\n"
    f"_P022_ON = __import__(\"os\").environ.get(\"{ENV}\", \"0\") == \"1\"\n"
    "if _P022_ON:\n"
    "    from vllm import _gb10_mhc_prs as _p022\n"
    "\n\ndef _hc_prenorm_gemm_outputs(\n"
)

CALL = (
    f"        {KERNEL}(\n"
    "            x,\n"
    "            fn,\n"
    "            out,\n"
    "            sqrsum,\n"
    "            hidden_size,\n"
    "            hc_mult,\n"
    "        )\n"
    "    return out, sqrsum\n"
)
HOOK_OLD = CALL
HOOK_NEW = (
    f"        {MARKER} {ENV}=1: in prefill this rank runs the SAME block_m=2 JIT\n"
    "        # kernel on its own token rows and the outputs are all-gathered over the TP group (bit-identical); False ->\n"
    "        # the unchanged call below (T < threshold, not routed to block_m=2, CUDA-graph capture, unexpected layout).\n"
    "        if _P022_ON and _p022.prenorm_rowsplit(\n"
    f"            {KERNEL}, x, fn, out, sqrsum, hidden_size, hc_mult\n"
    "        ):\n"
    "            return out, sqrsum\n"
    + CALL
)

EDITS = [
    ("module gate (before def _hc_prenorm_gemm_outputs)", GATE_OLD, GATE_NEW),
    ("row-split hook (TileLang branch of _hc_prenorm_gemm_outputs)", HOOK_OLD, HOOK_NEW),
]


class Refused(ValueError):
    pass


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    hits = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(hits) != 1:
        raise Refused(f"{name} defined {len(hits)}x at module level (expected 1)")
    return hits[0]


def _is_call_to(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == name)


def _tilelang_branch(fn: ast.FunctionDef) -> list:
    """orelse of the unique top-level `if use_deep_gemm:` whose else-branch calls the TileLang class instance."""
    hits = [n for n in fn.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "use_deep_gemm"
            and any(_is_call_to(s, KERNEL) for s in n.orelse)]
    if len(hits) != 1:
        raise Refused(f"{FUNC}: {len(hits)} `if use_deep_gemm:` branches whose else calls {KERNEL} (expected 1)")
    return hits[0].orelse


def check_structure(text: str, patched: bool) -> None:
    """AST checks of the base (patched=False) or the result (patched=True)."""
    tree = ast.parse(text)
    fn = _func(tree, FUNC)
    branch = _tilelang_branch(fn)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not patched:
        if not (len(branch) == 2 and isinstance(branch[0], ast.ImportFrom) and _is_call_to(branch[1], KERNEL)):
            raise Refused(f"{FUNC}: the TileLang branch is not [import, kernel call]")
        if "_P022_ON" in names or "_p022" in names:
            raise Refused("base already mentions _P022_ON / _p022")
        return
    if not (len(branch) == 3 and isinstance(branch[0], ast.ImportFrom) and isinstance(branch[1], ast.If)
            and _is_call_to(branch[2], KERNEL)):
        raise Refused(f"{FUNC}: patched TileLang branch is not [import, hook, kernel call]")
    hook = branch[1]
    calls = [n for n in ast.walk(hook.test) if isinstance(n, ast.Call)]
    if not (len(calls) == 1 and isinstance(calls[0].func, ast.Attribute) and calls[0].func.attr == "prenorm_rowsplit"
            and isinstance(calls[0].func.value, ast.Name) and calls[0].func.value.id == "_p022"
            and isinstance(calls[0].args[0], ast.Name) and calls[0].args[0].id == KERNEL
            and len(hook.body) == 1 and isinstance(hook.body[0], ast.Return) and not hook.orelse):
        raise Refused(f"{FUNC}: patched hook is not `if _P022_ON and _p022.prenorm_rowsplit({KERNEL}, ...): return`")
    gates = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_P022_ON" for t in n.targets)]
    if len(gates) != 1 or tree.body.index(gates[0]) > tree.body.index(fn):
        raise Refused("expected one module-level _P022_ON assignment before " + FUNC)


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-022 marker")
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
        raise Refused(f"expected {N_MARKERS} markers after the edit, found {out.count(MARKER)}")
    compile(out, "tilelang.py", "exec")  # in memory: no .pyc side effects
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
    compile(out, "tilelang.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p022_apply.py {apply|check|revert} <tilelang.py>", file=sys.stderr)
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
    tmp = p.with_name(p.name + ".p022.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if argv[1] == "revert" else "applied"
    print(f" [OK] {len(EDITS)} anchored edits {what} on {p.name} "
          "(module gate _P022_ON; row-split hook in the TileLang branch of _hc_prenorm_gemm_outputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
