#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch 014 gb10-skinny-indexer] CPU checks (Mac / any CPU: python3 >= 3.10, stdlib only; NO torch, NO triton, NO vLLM install).

  python3 -I -B files/p014_cpu_check.py [--vllm-root <v0.32.0 vllm package dir>]    (default: $VLLM_TREE, else vllm-v0.32.0/vllm)

  A  pins + applier on the REAL v0.32.0 models/glm5next/common/attention.py: original == the .sh pin, apply() == the pinned
     result, the diff is exactly the three hooks (the carried block is the v0.32.0 text byte for byte, re-indented),
     3.12 grammar, revert() byte-exact, a second apply is refused, applier / module shas.
  B  refusals: anchors missing / duplicated, the carried block altered (e.g. the day-0 `getattr(self, "_wp_fp32", None)` text),
     the gate_score line changed, class renamed, marker present -> Refused; the patched structure check rejects tampering.
  C  the REAL .sh end to end in a scratch VLLM_ROOT: gate unset = inert (exit 0, nothing written), gate 1 = banner, module
     installed, [OK] lines, result == pinned sha, second run [skip], unknown base / stale edit / tampered applier or module ->
     exit 1 with attention.py untouched.
  D  the REAL patched Indexer.forward run over symbolic stubs: gate off -> exactly the original call sequence (one
     wk_weights_proj call, the fp32 twin built once and cached, mm, F.linear); gate on + use() True -> project() output is what
     flows to _fused_indexer_k_norm / _fused_indexer_weight_scale / indexer_op(gate_score=...) and none of the original
     projections runs; gate on + use() False -> the original sequence; the module-level gate (env unset / '1' / module
     missing -> ImportError / 'true' -> off).
  E  the module (gb10_skinny_idx.py): compiles with the 3.12 grammar, the per-M route table, and the pure-Python helpers.
Exit 0 = ALL PASS."""
from __future__ import annotations

import ast
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psmall_lib as L  # noqa: E402

SH = L.PDIR / "014-gb10-skinny-indexer.sh"
APPLIER = HERE / "p014_apply.py"
MODULE = HERE / "gb10_skinny_idx.py"
REL = "models/glm5next/common/attention.py"
OLD_LINE = '        if getattr(self, "_wp_fp32", None) is None:\n'


class S:
    """A symbolic tensor: every operation returns another S; nothing is computed."""

    def __init__(self, name="t"):
        self.name = name

    def __getattr__(self, a):
        if a.startswith("__"):
            raise AttributeError(a)
        return S(f"{self.name}.{a}")

    def __call__(self, *a, **k):
        return S(f"{self.name}()")

    def __getitem__(self, i):
        return S(f"{self.name}[]")

    def __iter__(self):
        return iter((S(f"{self.name}.0"), S(f"{self.name}.1")))

    def __mul__(self, o):
        return S("mul")

    __rmul__ = __add__ = __radd__ = __sub__ = __truediv__ = __mul__


def run_forward(patched_src: str, idx_mod, *, wp_cached=False):
    """Run the REAL (patched) Indexer.forward with the module-level name _GB10_IDX = idx_mod. -> (trail, result, self_)."""
    tree = ast.parse(patched_src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    src = ast.unparse(ast.Module(body=[fn], type_ignores=[]))
    trail = []

    def rec(name, ret=None):
        def f(*a, **k):
            trail.append((name, a, k))
            return ret(*a, **k) if callable(ret) else (ret if ret is not None else S(name))
        return f

    knorm_args = []
    ns = {
        "_GB10_IDX": idx_mod,
        "torch": types.SimpleNamespace(mm=rec("torch.mm", lambda x, w: S("weights_orig")),
                                       split=lambda t, sizes, dim=-1: (S("p0"), S("p1")),
                                       cat=lambda ts, dim=-1: S("cat")),
        "F": types.SimpleNamespace(linear=rec("F.linear", lambda x, w: S("gate_orig"))),
        "_fused_indexer_k_norm": lambda k, w, b, d, eps: (knorm_args.append(k) or S("knorm")),
        "_fused_indexer_weight_scale": rec("weight_scale", lambda w, qs, c: S("wscale")),
        "_pad_indexer_heads": lambda t, p: t,
        "fwht128_quant_fp8": lambda q: (S("qfp8"), S("qscale")),
    }
    exec(compile(src, "Indexer.forward(patched)", "exec"), ns)
    fw = ns["forward"]

    class WK:
        def __init__(self):
            self.weight = S("wk.weight")
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            trail.append(("wk_weights_proj", (), {}))
            return S("kw"), None

    sf = types.SimpleNamespace(
        n_head=32, head_dim=128, rope_dim=0, quant_block_size=128, scale_fmt="ue8m0", softmax_scale=0.1, index_kpool=16,
        topk_tokens=2048, wq_b=lambda qr: (S("q"), None), wk_weights_proj=WK(),
        _wp_fp32=(S("wp_cached") if wp_cached else None), k_norm=types.SimpleNamespace(weight=S("w"), bias=None, eps=1e-6),
        index_kpool_compress_gate=S("gate"), index_kpool_compress_ape=S("ape"),
        indexer_op=lambda *a, **k: (trail.append(("indexer_op", a, k)) or "OUT"))
    res = fw(sf, S("hidden"), S("qr"), "positions", None)
    return trail, res, sf, knorm_args


def main(argv) -> int:
    args = L.parse_args(argv)
    vroot = Path(args.vllm_root)
    ap = L.load_module("p014_apply", APPLIER)
    orig = (vroot / REL).read_text()
    check = L.check

    # ---------------------------------------------------------------- A
    check("A1 v0.32.0 common/attention.py == the AT_ORIG pin", L.sha_file(vroot / REL) == L.pin(SH, "AT_ORIG"))
    check("A2 applier == APPLIER_SHA, module == MOD_SHA", L.sha_file(APPLIER) == L.pin(SH, "APPLIER_SHA")
          and L.sha_file(MODULE) == L.pin(SH, "MOD_SHA"))
    new = ap.apply(orig)
    check("A3 apply() == the AT_P014 pin", L.sha_bytes(new.encode()) == L.pin(SH, "AT_P014"), L.sha_bytes(new.encode())[:16])
    check("A4 3 markers; compiles with the python 3.12 grammar", new.count(ap.MARKER) == 3 and L.compiles_312(new, REL))
    check("A5 revert(apply(x)) == x byte for byte", ap.revert(new) == orig)
    check("A6 a second apply() is refused", L.raises(lambda: ap.apply(new), ap.Refused))
    check("A7 the carried block is the v0.32.0 text: `if self._wp_fp32 is None:` (the day-0 getattr form is gone)",
          'if self._wp_fp32 is None:' in orig and OLD_LINE not in orig and ap.BLOCK_OLD in orig)
    nl = new.splitlines()
    check("A8 the original 10 lines survive unchanged inside the else branch (4 more spaces)",
          all(("    " + ln) in new for ln in ap.BLOCK_OLD.splitlines() if ln.strip()))
    check("A9 each anchor occurs exactly once in v0.32.0", all(orig.count(old) == 1 for _, old, _ in ap.EDITS))

    # ---------------------------------------------------------------- B
    check("B1 day-0 form of the carried block (getattr) -> Refused (anchor 2 not found)",
          L.raises(lambda: ap.apply(orig.replace("        if self._wp_fp32 is None:\n", OLD_LINE)), ap.Refused, "found 0x"))
    check("B2 logger anchor missing -> Refused", L.raises(lambda: ap.apply(orig.replace(ap.IMPORT_OLD, "log = 1\n")), ap.Refused, "found 0x"))
    check("B3 gate_score line duplicated -> Refused", L.raises(lambda: ap.apply(orig + "\n" + ap.GATE_OLD), ap.Refused, "found 2x"))
    check("B4 gate_score line changed -> Refused", L.raises(
        lambda: ap.apply(orig.replace(ap.GATE_OLD, "        gate_score = F.linear(hidden_states, self.index_kpool_compress_gate.T)\n")), ap.Refused))
    check("B5 class Indexer renamed -> Refused (AST)", L.raises(lambda: ap.apply(orig.replace("class Indexer(", "class IndexerX(")), ap.Refused))
    check("B6 marker present -> Refused", L.raises(lambda: ap.apply(orig + "\n" + ap.MARKER + "\n"), ap.Refused, "marker"))
    tamper = new.replace("            k, weights, gate_score = _GB10_IDX.project(self, hidden_states)\n",
                         "            k, weights, gate_score = _GB10_IDX.project(self, hidden_states[:1])\n")
    check("B7 patched structure check rejects a changed per-M branch", L.raises(lambda: ap.check_structure(tamper, patched=True), ap.Refused))
    tamper2 = new.replace("        if not _p014:", "        if True:")
    check("B8 patched structure check rejects an unguarded gate_score", L.raises(lambda: ap.check_structure(tamper2, patched=True), ap.Refused))
    tamper3 = new.replace("                    .float()\n", "                    .half()\n")
    check("B9 patched structure check rejects an edited else branch", L.raises(lambda: ap.check_structure(tamper3, patched=True), ap.Refused))
    check("B10 revert of an unpatched file is refused", L.raises(lambda: ap.revert(orig), ap.Refused))

    # ---------------------------------------------------------------- C
    roots = []

    def fresh(text=None):
        r = L.make_root(vroot, [REL])
        roots.append(r)
        if text is not None:
            (r / REL).write_text(text)
        return r

    root = fresh()
    b0 = (root / REL).read_bytes()
    r0 = L.run_sh(SH, root)
    check("C1 gate unset: inert (exit 0, skipped banner, nothing written, no module)",
          r0.returncode == 0 and "skipped (VLLM_GB10_SKINNY_IDX=unset); files untouched" in r0.stdout and (root / REL).read_bytes() == b0
          and not (root / "_gb10_skinny_idx.py").exists())
    r1 = L.run_sh(SH, root, {"VLLM_GB10_SKINNY_IDX": "1"})
    out = r1.stdout
    check("C2 gate 1: exit 0; banner, module install, [OK] line, verification, final line",
          r1.returncode == 0 and "=== Patch 014: GB10 skinny indexer, per-M kernels for the DSA indexer projections at M <= 16 (opt-in VLLM_GB10_SKINNY_IDX=1) ===" in out
          and " [OK] installed _gb10_skinny_idx.py (sha256 " in out
          and " [OK] 3 anchored edits applied (module import, Indexer.forward dispatch, gate_score guard)" in out
          and "[OK] patch 014 in place (active only when VLLM_GB10_SKINNY_IDX=1; now: 1)" in out, r1.stderr[-300:] + out[-300:])
    check("C3 attention.py == the pinned result; module installed byte-identical",
          L.sha_file(root / REL) == L.pin(SH, "AT_P014") and L.sha_file(root / "_gb10_skinny_idx.py") == L.pin(SH, "MOD_SHA"))
    before = (root / REL).read_bytes()
    r2 = L.run_sh(SH, root, {"VLLM_GB10_SKINNY_IDX": "1"})
    check("C4 second run: [skip], exit 0, file untouched", r2.returncode == 0 and " [skip] attention.py already wired" in r2.stdout
          and (root / REL).read_bytes() == before)
    root2 = fresh(orig + "\n# local change\n")
    b2 = (root2 / REL).read_bytes()
    r3 = L.run_sh(SH, root2, {"VLLM_GB10_SKINNY_IDX": "1"})
    check("C5 unknown base -> exit 1, file untouched, module not installed", r3.returncode == 1 and (root2 / REL).read_bytes() == b2
          and not (root2 / "_gb10_skinny_idx.py").exists())
    root3 = fresh(new + "\n# edited after patching\n")
    b3 = (root3 / REL).read_bytes()
    r4 = L.run_sh(SH, root3, {"VLLM_GB10_SKINNY_IDX": "1"})
    check("C6 stale patch-014 edit (marker, unpinned sha) -> exit 1 'recreate the container'",
          r4.returncode == 1 and (root3 / REL).read_bytes() == b3 and "recreate the container" in r4.stderr)
    for what, fn in (("applier", "p014_apply.py"), ("module", "gb10_skinny_idx.py")):
        pdir = Path(tempfile.mkdtemp(prefix="smallchk_p_"))
        roots.append(pdir)
        shutil.copy2(SH, pdir / SH.name)
        (pdir / "files").mkdir()
        for f in ("p014_apply.py", "gb10_skinny_idx.py"):
            shutil.copy2(HERE / f, pdir / "files" / f)
        (pdir / "files" / fn).write_text((HERE / fn).read_text() + "\n# tampered\n")
        rr = fresh()
        bb = (rr / REL).read_bytes()
        r5 = L.run_sh(pdir / SH.name, rr, {"VLLM_GB10_SKINNY_IDX": "1"})
        check(f"C7 tampered {what} -> exit 1 before any write", r5.returncode == 1 and (rr / REL).read_bytes() == bb
              and not (rr / "_gb10_skinny_idx.py").exists(), r5.stderr[-160:])
    check("C8 gate '0' / 'true' / '' -> inert", all(
        L.run_sh(SH, fresh(), {"VLLM_GB10_SKINNY_IDX": v}).returncode == 0 and "skipped" in L.run_sh(SH, fresh(), {"VLLM_GB10_SKINNY_IDX": v}).stdout
        for v in ("0", "true", "")))

    # ---------------------------------------------------------------- D
    class Idx:
        def __init__(self, use):
            self._use, self.trail = use, []

        def use(self, layer, x):
            self.trail.append("use")
            return self._use

        def project(self, layer, x):
            self.trail.append("project")
            return S("kproj"), S("wproj"), S("gproj")

    trail, res, sf, knorm_args = run_forward(new, None)
    names = [t[0] for t in trail]
    check("D1 gate off (_GB10_IDX None): the original sequence wk_weights_proj, torch.mm, F.linear, one of each",
          names.count("wk_weights_proj") == 1 and names.count("torch.mm") == 1 and names.count("F.linear") == 1
          and names.index("wk_weights_proj") < names.index("torch.mm") < names.index("F.linear") and res == "OUT", str(names))
    check("D2 gate off: the fp32 twin is built once (cached on self) and reused by the next call",
          sf._wp_fp32 is not None and run_forward(new, None, wp_cached=True)[0] != [] and
          [t[0] for t in run_forward(new, None, wp_cached=True)[0]].count("torch.mm") == 1)
    idx = Idx(True)
    trail, res, sf, knorm_args = run_forward(new, idx)
    names = [t[0] for t in trail]
    iop = next(t for t in trail if t[0] == "indexer_op")
    check("D3 gate on + use() True: use then project; no wk_weights_proj / torch.mm / F.linear", idx.trail == ["use", "project"]
          and "wk_weights_proj" not in names and "torch.mm" not in names and "F.linear" not in names, str(names))
    check("D4 project()'s k goes to the k-norm, its weights to the weight scaling, its gate_score to indexer_op unchanged",
          knorm_args and knorm_args[0].name == "kproj" and trail[names.index("weight_scale")][1][0].name == "wproj"
          and iop[2]["gate_score"].name == "gproj", str(iop[2]))
    check("D5 gate on, the _wp_fp32 twin is not built by the patched path", sf._wp_fp32 is None)
    idx = Idx(False)
    trail, res, sf, knorm_args = run_forward(new, idx)
    names = [t[0] for t in trail]
    check("D6 gate on + use() False (prefill / M > 16): use only, then exactly the original sequence",
          idx.trail == ["use"] and names.count("wk_weights_proj") == 1 and names.count("torch.mm") == 1 and names.count("F.linear") == 1)
    base_trail, base_res, _, _ = run_forward(orig, None)
    check("D7 gate off: the call sequence of the PATCHED forward equals the unpatched v0.32.0 forward's",
          [t[0] for t in base_trail] == [t[0] for t in run_forward(new, None)[0]])

    # module-level gate, run for real
    tree = ast.parse(new)
    lg = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "logger")
    gate_src = ast.unparse(ast.Module(body=tree.body[lg + 1:lg + 3], type_ignores=[]))

    def gate(env: dict, module_present: bool):
        import os
        os.environ.pop("VLLM_GB10_SKINNY_IDX", None)
        os.environ.update(env)
        saved = {k: sys.modules.get(k) for k in ("vllm", "vllm._gb10_skinny_idx")}
        vl = types.ModuleType("vllm")
        vl.__path__ = []
        sys.modules["vllm"] = vl
        sys.modules.pop("vllm._gb10_skinny_idx", None)
        if module_present:
            m = types.ModuleType("vllm._gb10_skinny_idx")
            sys.modules["vllm._gb10_skinny_idx"] = m
            vl._gb10_skinny_idx = m
        ns = {}
        try:
            exec(compile(gate_src, "attention.py(gate)", "exec"), ns)
            return ns["_GB10_IDX"], None
        except ImportError as e:
            return None, e
        finally:
            os.environ.pop("VLLM_GB10_SKINNY_IDX", None)
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    v, e = gate({}, True)
    check("D8 module gate: env unset -> _GB10_IDX None (module not imported)", v is None and e is None)
    v, e = gate({"VLLM_GB10_SKINNY_IDX": "1"}, True)
    check("D9 module gate: env '1' + module present -> imported", v is not None and v.__name__ == "vllm._gb10_skinny_idx")
    v, e = gate({"VLLM_GB10_SKINNY_IDX": "1"}, False)
    check("D10 module gate: env '1' + module missing -> ImportError at import (the boot aborts)", v is None and e is not None)
    v, e = gate({"VLLM_GB10_SKINNY_IDX": "true"}, True)
    check("D11 module gate: only the exact value '1' enables it", v is None and e is None)

    # ---------------------------------------------------------------- E
    mtext = MODULE.read_text()
    check("E1 module compiles with the python 3.12 grammar", L.compiles_312(mtext, "gb10_skinny_idx.py"))
    # exec only the pure-python route table helpers
    mt = ast.parse(mtext)
    keep = [n for n in mt.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
        "HIDDEN", "ROUTES", "_SK_BS256", "_SK_BS128_KU2", "_TRITON_HG", "MAX_M", "N_WK") for t in n.targets))
        or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)) or (isinstance(n, ast.AnnAssign) and n.target.id == "ROUTES")
        or (isinstance(n, ast.FunctionDef) and n.name == "route_table")]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "gb10_skinny_idx(routes)", "exec"), ns)
    table = ns["route_table"]()
    check("E2 route table: M=1-2 / 3 / 4-8 / 9-16 rows with the measured routes",
          "M=1-2" in table and "M=3 " in table and "M=4-8" in table and "M=9-16" in table
          and "skinny bs256 opb1 ku1 vw8" in table and "skinny bs128 opb1 ku2 vw8" in table and "ll_bf16 dotprod bs256" in table
          and "ll_bf16 dotprod bs128" in table and "triton mma16 BN16 BK64 S32 w4" in table and table.count("\n") == 3, table)
    check("E3 ROUTES cover exactly M = 1..16", sorted(ns["ROUTES"]) == list(range(1, 17)))
    check("E4 the module's CuTe/Triton surface names are all still defined in v0.32.0's cute_dsl sources (static grep)", all(
        s in (vroot / "model_executor/kernels/linear/cute_dsl/ll_bf16.py").read_text() for s in (
            "class CompileKey", "def compile(", "_compiled_cache", "def _stream()", "def is_available()", "ll_bf16_gemm_kernel = LLBf16Gemm()"))
          and all(s in (vroot / "model_executor/kernels/linear/cute_dsl/skinny_gemm.py").read_text() for s in (
              "class SkinnyGemmConfig", "def is_available()", "shape_dynamic_skinny_gemm = ShapeDynamicSkinnyGemm()", "self._compiled[")))

    for d in roots:
        shutil.rmtree(d, ignore_errors=True)
    return L.finish("p014_cpu_check")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
