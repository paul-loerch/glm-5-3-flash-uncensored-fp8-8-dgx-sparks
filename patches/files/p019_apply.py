#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 019 applier (v0.32.0 port): anchored edits to two files of vllm/model_executor/kernels/mhc/.

  tilelang.py          (3 edits)  module gate _P019_ON; the bf16 conversion hook and the kernel selection inside
                                  mhc_fused_post_pre_tilelang (the non-delayed fused post + pre-norm op GLM5Next uses)
  tilelang_kernels.py  (1 edit)   MhcFusedTileLangKernel.register_warmup override: whoever registers the fp32 fused
                                  kernel for JIT warmup also registers the bf16 variant (vllm/_gb10_mhc_fn_bf16.py)

  apply_tl(text) / apply_tk(text) -> text     v0.32.0 original (or the 011 / 022 result) -> + patch 019; fail-closed: every
                          anchor must match exactly once, the result must carry exactly the expected markers, compile,
                          and pass the AST scope checks
  revert_tl / revert_tk(text) -> text         exact inverses (the tests rebuild the base from a patched file)

CLI (used by 019-gb10-mhc-fn-bf16.sh):
  p019_apply.py apply  {tl|tk} <file>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p019_apply.py check  {tl|tk} <file>   dry run: print the sha256 the file would have after the edit
  p019_apply.py revert {tl|tk} <file>   in-place inverse (tests / manual rollback)

Edits (v0.32.0 line numbers; tilelang.py sha256 204e1920..., tilelang_kernels.py sha256 61a384a0...):
  tilelang.py
    1. after `from vllm.utils.torch_utils import direct_register_custom_op` (:6): module gate _P019_ON, read once from
       VLLM_GB10_MHC_FN_BF16 at import;
    2. mhc_fused_post_pre_tilelang, after `fused_config = mhc_fused_post_pre_split_config(...)` (:893): when the gate is
       on, `_p019_w = _p019.weight_for(fn, hc_mult, hidden_size)` - the cached bf16 copy, made on the FIRST EAGER call of
       each weight at ANY num_tokens (the pre-capture profile run at 8,192 tokens, where the op is not fused, converts
       every weight; so does the V2 runner's eager warm-up before each capture); None (gate off, or a weight first seen
       inside a CUDA-graph capture) keeps the upstream fp32 kernel;
    3. the fused launch (:914-924): `_MHC_FUSED_TILELANG_KERNEL(..., fn.view(...), ...)` becomes
       `_fused_kernel(..., _fused_fn.view(...), ...)` with (_fused_kernel, _fused_fn) = (the bf16 class instance, the
       copy) when a copy exists, else (the upstream instance, fn) - the call, its arguments and its outputs are
       otherwise untouched. num_tokens > 32 (mhc_post + the fp32 pre-norm GEMM, where patches 011 / 022 act) and
       layer-0 mhc_pre_tilelang are untouched.
  tilelang_kernels.py
    4. MhcFusedTileLangKernel gains `register_warmup`, which forwards to the bf16 variant's registration when the gate
       is on (class attribute _P019_ON read once at import; the variant imports lazily, at registration time = model
       construction under the runner's registry, so no TileLang module is imported earlier than before).
Patch 011 edits only HcPrenormGemmTileLangKernel and a module constant in tilelang_kernels.py, patch 022 only
_hc_prenorm_gemm_outputs / a module gate in tilelang.py: no anchor overlaps, so the three commute.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-019 gb10-mhc-fn-bf16]"
ENV = "VLLM_GB10_MHC_FN_BF16"
N_MARKERS_TL = 3
N_MARKERS_TK = 1
CLASS_TK = "MhcFusedTileLangKernel"
FUNC_TL = "mhc_fused_post_pre_tilelang"

# ------------------------------------------------------------------------------------------------------ tilelang.py
GATE_OLD = "from vllm.utils.torch_utils import direct_register_custom_op\n"
GATE_NEW = GATE_OLD + (
    "\n"
    f"{MARKER} opt-in {ENV}=1 (read once, at import): the fused mHC\n"
    "# decode kernel (num_tokens <= 32) reads a bf16 copy of hc_*_fn (vllm/_gb10_mhc_fn_bf16.py); unset -> unchanged.\n"
    f"_P019_ON = __import__(\"os\").environ.get(\"{ENV}\", \"0\") == \"1\"\n"
)

CONVERT_OLD = (
    "    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)\n"
    "\n"
    "    fused_config = mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)\n"
)
CONVERT_NEW = (
    CONVERT_OLD
    + "\n"
    + f"    {MARKER} {ENV}=1: a cached bf16 copy of fn for the fused kernel below -\n"
    + "    # bit-identical: hc_*_fn are BF16 in the checkpoint (each copy is checked exact) and the kernel widens every\n"
    + "    # weight back to fp32 before the same FMA. The copy is made on the first EAGER call of each weight at any\n"
    + "    # num_tokens (the profile run); None (flag off, or a weight first seen inside a CUDA-graph capture) -> the\n"
    + "    # upstream fp32 kernel.\n"
    + "    _p019_w = None\n"
    + "    if _P019_ON:\n"
    + "        from vllm import _gb10_mhc_fn_bf16 as _p019\n"
    + "\n"
    + "        _p019_w = _p019.weight_for(fn, hc_mult, hidden_size)\n"
)

_FUSED_ARGS_HEAD = (
    "            comb_res_mix_flat,\n"
    "            residual_flat,\n"
    "            post_layer_mix_flat,\n"
    "            x_flat,\n"
)
_FUSED_ARGS_TAIL = (
    "            hc_mult,\n"
    "            hidden_size,\n"
    "            hc_mult3,\n"
    "        )\n"
)
LAUNCH_OLD = (
    "    if fused_config is not None:\n"
    "        gemm_out_mul, gemm_out_sqrsum, residual_cur = _MHC_FUSED_TILELANG_KERNEL(\n"
    + _FUSED_ARGS_HEAD
    + "            fn.view(hc_mult3, hc_mult, hidden_size),\n"
    + _FUSED_ARGS_TAIL
)
LAUNCH_NEW = (
    "    if fused_config is not None:\n"
    f"        {MARKER} the bf16-weight variant of the same JIT class when a copy exists\n"
    "        _fused_kernel, _fused_fn = _MHC_FUSED_TILELANG_KERNEL, fn\n"
    "        if _p019_w is not None:\n"
    "            _fused_kernel, _fused_fn = _p019.MHC_FUSED_BF16W_KERNEL, _p019_w\n"
    "        gemm_out_mul, gemm_out_sqrsum, residual_cur = _fused_kernel(\n"
    + _FUSED_ARGS_HEAD
    + "            _fused_fn.view(hc_mult3, hc_mult, hidden_size),\n"
    + _FUSED_ARGS_TAIL
)

EDITS_TL = [
    ("module gate (after the direct_register_custom_op import)", GATE_OLD, GATE_NEW),
    ("bf16 conversion hook (mhc_fused_post_pre_tilelang, after fused_config)", CONVERT_OLD, CONVERT_NEW),
    ("fused launch (mhc_fused_post_pre_tilelang, fused_config is not None)", LAUNCH_OLD, LAUNCH_NEW),
]

# ------------------------------------------------------------------------------------------------ tilelang_kernels.py
REG_OLD = (
    "    @kernel_launcher\n"
    "    def __call__(\n"
    "        self,\n"
    "        comb_mix: Any,\n"
    "        residual_in: Any,\n"
)
REG_NEW = (
    f"    {MARKER} opt-in {ENV}=1 (class attribute, read once at import): registering the\n"
    "    # fp32 fused kernel for JIT warmup (the model does it under the runner's registry) also registers the bf16-weight\n"
    "    # variant (vllm/_gb10_mhc_fn_bf16.py), so its compile keys are warmed before CUDA-graph capture. The type check\n"
    "    # keeps the variant's own (inherited) registration from recursing. Unset -> unchanged.\n"
    f"    _P019_ON = __import__(\"os\").environ.get(\"{ENV}\", \"0\") == \"1\"\n"
    "\n"
    "    def register_warmup(self, *args: Any, **kwargs: Any) -> None:\n"
    "        super().register_warmup(*args, **kwargs)\n"
    "        if self._P019_ON and type(self) is MhcFusedTileLangKernel:\n"
    "            from vllm import _gb10_mhc_fn_bf16 as _p019\n"
    "\n"
    "            _p019.register_warmup(*args, **kwargs)\n"
    "\n"
    + REG_OLD
)
EDITS_TK = [
    ("MhcFusedTileLangKernel.register_warmup (bf16 variant fan-out)", REG_OLD, REG_NEW),
]


class Refused(ValueError):
    pass


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    hits = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(hits) != 1:
        raise Refused(f"{name} defined {len(hits)}x at module level (expected 1)")
    return hits[0]


def _cls(tree: ast.Module, name: str) -> ast.ClassDef:
    hits = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(hits) != 1:
        raise Refused(f"class {name} defined {len(hits)}x at module level (expected 1)")
    return hits[0]


def check_structure_tl(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    fn = _func(tree, FUNC_TL)
    gate_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not patched:
        if "_P019_ON" in gate_names or "_p019" in gate_names or "_p019_w" in gate_names:
            raise Refused("base already mentions _P019_ON / _p019")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_MHC_FUSED_TILELANG_KERNEL"]
        if len(calls) != 1:
            raise Refused(f"{FUNC_TL}: expected one _MHC_FUSED_TILELANG_KERNEL(...) call, found {len(calls)}")
        return
    gates = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_P019_ON" for t in n.targets)]
    if len(gates) != 1 or tree.body.index(gates[0]) > tree.body.index(fn):
        raise Refused("expected one module-level _P019_ON assignment before " + FUNC_TL)
    sels = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_fused_kernel"]
    if len(sels) != 1:
        raise Refused(f"{FUNC_TL}: expected one _fused_kernel(...) launch, found {len(sels)}")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_MHC_FUSED_TILELANG_KERNEL"]
    if calls:
        raise Refused(f"{FUNC_TL}: the upstream fused kernel is still called directly")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    if not {"weight_for", "MHC_FUSED_BF16W_KERNEL"} <= attrs:
        raise Refused(f"{FUNC_TL}: expected _p019.weight_for and _p019.MHC_FUSED_BF16W_KERNEL")


def check_structure_tk(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    cls = _cls(tree, CLASS_TK)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "register_warmup"]
    if not patched:
        if methods or any("_P019_ON" in ast.dump(n) for n in cls.body):
            raise Refused(f"base {CLASS_TK} already defines register_warmup / _P019_ON")
        return
    if len(methods) != 1:
        raise Refused(f"{CLASS_TK}: expected one register_warmup after the edit, found {len(methods)}")
    flag = [n for n in cls.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_P019_ON" for t in n.targets)]
    if len(flag) != 1:
        raise Refused(f"{CLASS_TK}: expected one class-level _P019_ON assignment")


def _apply(text, edits, marker_count, check, what):
    if MARKER in text:
        raise Refused("input already carries the patch-019 marker")
    bad = [(name, text.count(old)) for name, old, _ in edits if text.count(old) != 1]
    if bad:
        raise Refused("; ".join(f"anchor '{n}' found {c}x (expected 1)" for n, c in bad))
    check(text, patched=False)
    out = text
    for name, old, new in edits:
        if out.count(old) != 1:
            raise Refused(f"anchor '{name}' changed by an earlier edit")
        out = out.replace(old, new, 1)
    if out.count(MARKER) != marker_count:
        raise Refused(f"expected {marker_count} markers after the edits, found {out.count(MARKER)}")
    compile(out, what, "exec")  # in memory: no .pyc side effects
    check(out, patched=True)
    return out


def _revert(text, edits, what):
    out = text
    for name, old, new in reversed(edits):
        if out.count(new) != 1:
            raise Refused(f"patched block '{name}' found {out.count(new)}x (expected 1)")
        out = out.replace(new, old, 1)
    if MARKER in out:
        raise Refused("marker still present after revert")
    compile(out, what, "exec")
    return out


def apply_tl(text: str) -> str:
    return _apply(text, EDITS_TL, N_MARKERS_TL, check_structure_tl, "tilelang.py")


def revert_tl(text: str) -> str:
    return _revert(text, EDITS_TL, "tilelang.py")


def apply_tk(text: str) -> str:
    return _apply(text, EDITS_TK, N_MARKERS_TK, check_structure_tk, "tilelang_kernels.py")


def revert_tk(text: str) -> str:
    return _revert(text, EDITS_TK, "tilelang_kernels.py")


FUNCS = {("apply", "tl"): apply_tl, ("apply", "tk"): apply_tk, ("check", "tl"): apply_tl, ("check", "tk"): apply_tk,
         ("revert", "tl"): revert_tl, ("revert", "tk"): revert_tk}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in ("apply", "check", "revert") or argv[2] not in ("tl", "tk"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p019_apply.py {apply|check|revert} {tl|tk} <tilelang.py | tilelang_kernels.py>", file=sys.stderr)
        return 2
    cmd, which, p = argv[1], argv[2], Path(argv[3])
    try:
        new = FUNCS[(cmd, which)](p.read_text(encoding="utf-8"))
    except (Refused, SyntaxError, OSError) as e:
        print(f"[FAIL] {p}: {e}", file=sys.stderr)
        return 1
    if cmd == "check":
        print(_sha(new.encode()))
        return 0
    tmp = p.with_name(p.name + ".p019.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if cmd == "revert" else "applied"
    n = len(EDITS_TL if which == "tl" else EDITS_TK)
    print(f" [OK] {n} anchored edit(s) {what} on {p.name} "
          + ("(module gate _P019_ON; bf16 copy hook + kernel selection in mhc_fused_post_pre_tilelang)"
             if which == "tl" else "(MhcFusedTileLangKernel.register_warmup fan-out to the bf16 variant)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
