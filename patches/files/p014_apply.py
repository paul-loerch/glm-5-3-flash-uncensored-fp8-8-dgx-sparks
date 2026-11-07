#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 014 applier (vLLM v0.32.0 port): three anchored edits to vllm/models/glm5next/common/attention.py (the file moved
from models/glm5next/nvidia/attention.py in v0.31.0, upstream #56176).

  apply(text) -> text     v0.32.0 original -> + patch 014; fail-closed: every anchor must match exactly once, the three
                          hooks must land where they are meant to (AST-checked: the module gate right after the logger, the
                          dispatch around the wk_weights_proj / head-gate block of Indexer.forward, the gate_score guard in the
                          same function), the result must carry N_MARKERS markers and compile
  revert(text) -> text    exact inverse (the tests rebuild the base from the patched file)

CLI (used by 014-gb10-skinny-indexer.sh):
  p014_apply.py apply  <attention.py>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p014_apply.py check  <attention.py>   dry run: print the sha256 the file would have after apply()
  p014_apply.py revert <attention.py>   in-place inverse (tests / manual rollback)

Edits (v0.32.0 line numbers):
  1. after `logger = init_logger(__name__)` (:40): `_GB10_IDX = None`; when VLLM_GB10_SKINNY_IDX == "1" (read once, at import)
     `from vllm import _gb10_skinny_idx as _GB10_IDX` (module missing -> ImportError: the boot aborts on that rank).
  2. Indexer.forward (:297), the block that projects hidden_states through wk_weights_proj and builds the fp32 head gate (:305-314)
     becomes `_p014 = _GB10_IDX is not None and _GB10_IDX.use(self, hidden_states)` /
     `if _p014: k, weights, gate_score = _GB10_IDX.project(self, hidden_states)` / `else: <the original 10 lines, unchanged>`.
  3. `gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)` (:361) runs only when not _p014.
The ONE text change against the day-0 port is inside the carried block: v0.32.0 initialises `self._wp_fp32` in Indexer.__init__
(:255) and tests `if self._wp_fp32 is None:` where the day-0 image wrote `if getattr(self, "_wp_fp32", None) is None:`; the carried
block is the v0.32.0 text byte for byte (re-indented under `else:`).
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-014 gb10-skinny-indexer]"
N_MARKERS = 3
CLASS, FUNC = "Indexer", "forward"

IMPORT_OLD = "logger = init_logger(__name__)\n"
IMPORT_NEW = IMPORT_OLD + (
    "\n"
    f"{MARKER} opt-in VLLM_GB10_SKINNY_IDX=1: decode-sized (M <= 16) Indexer\n"
    "# projections on per-M kernels (vllm/_gb10_skinny_idx.py). An import error aborts the boot.\n"
    "_GB10_IDX = None\n"
    "if __import__(\"os\").environ.get(\"VLLM_GB10_SKINNY_IDX\", \"0\") == \"1\":\n"
    "    from vllm import _gb10_skinny_idx as _GB10_IDX\n"
)

BLOCK_OLD = (
    "        kw, _ = self.wk_weights_proj(hidden_states)\n"
    "        k = kw[:, : self.head_dim]\n"
    "        if self._wp_fp32 is None:\n"
    "            self._wp_fp32 = (\n"
    "                self.wk_weights_proj.weight.data[self.head_dim :, :]\n"
    "                .t()\n"
    "                .contiguous()\n"
    "                .float()\n"
    "            )\n"
    "        weights = torch.mm(hidden_states.float(), self._wp_fp32)\n"
)
BLOCK_NEW = (
    f"        {MARKER} M <= 16, bf16, contiguous: k (a [M,160]\n"
    "        # view), the fp32 head gate and gate_score come from per-M kernels with the same shapes,\n"
    "        # dtypes and strides; prefill and everything else keep the original lines below.\n"
    "        _p014 = _GB10_IDX is not None and _GB10_IDX.use(self, hidden_states)\n"
    "        if _p014:\n"
    "            k, weights, gate_score = _GB10_IDX.project(self, hidden_states)\n"
    "        else:\n"
    + "".join(("    " + ln) if ln.strip() else ln for ln in BLOCK_OLD.splitlines(keepends=True))
)

GATE_OLD = "        gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)\n"
GATE_NEW = (
    f"        if not _p014:  {MARKER} the per-M path set gate_score above\n"
    "            gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)\n"
)

EDITS = [
    ("module gate (after logger = init_logger(__name__))", IMPORT_OLD, IMPORT_NEW),
    ("Indexer.forward dispatch (wk_weights_proj + head-gate block)", BLOCK_OLD, BLOCK_NEW),
    ("Indexer.forward gate_score guard", GATE_OLD, GATE_NEW),
]


class Refused(ValueError):
    pass


def _forward(tree: ast.Module):
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS]
    if len(cls) != 1:
        raise Refused(f"class {CLASS} found {len(cls)} times (expected 1)")
    fns = [n for n in cls[0].body if isinstance(n, ast.FunctionDef) and n.name == FUNC]
    if len(fns) != 1:
        raise Refused(f"{CLASS}.{FUNC} found {len(fns)} times (expected 1)")
    return fns[0]


def _src(nodes) -> list[str]:
    return [ast.dump(n) for n in nodes]


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    fn = _forward(tree)
    loggers = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "logger"]
    if len(loggers) != 1:
        raise Refused(f"expected one module-level `logger = ...`, found {len(loggers)}")
    # the carried statements, parsed from the anchor text itself (dedented by 8)
    carried = ast.parse("".join(ln[8:] if ln.strip() else ln for ln in BLOCK_OLD.splitlines(keepends=True))).body
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not patched:
        if "_GB10_IDX" in names or "_p014" in names:
            raise Refused("base already mentions _GB10_IDX / _p014")
        body = fn.body
        k0 = next((i for i, s in enumerate(body) if _src(body[i:i + len(carried)]) == _src(carried)), None)
        if k0 is None:
            raise Refused("the wk_weights_proj / head-gate block is not a run of statements of Indexer.forward")
        later = [i for i, s in enumerate(body) if i > k0 and isinstance(s, ast.Assign)
                 and ast.unparse(s) == "gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)"]
        if len(later) != 1:
            raise Refused("the gate_score assignment does not follow the block in Indexer.forward")
        return
    # patched: module gate right after the logger
    k = tree.body.index(loggers[0])
    nxt = tree.body[k + 1: k + 3]
    ok = (len(nxt) == 2 and isinstance(nxt[0], ast.Assign) and ast.unparse(nxt[0]) == "_GB10_IDX = None"
          and isinstance(nxt[1], ast.If) and len(nxt[1].body) == 1 and isinstance(nxt[1].body[0], ast.ImportFrom)
          and nxt[1].body[0].module == "vllm"
          and [(a.name, a.asname) for a in nxt[1].body[0].names] == [("_gb10_skinny_idx", "_GB10_IDX")]
          and not nxt[1].orelse
          and ast.unparse(nxt[1].test) == "__import__('os').environ.get('VLLM_GB10_SKINNY_IDX', '0') == '1'")
    if not ok:
        raise Refused("the module gate is not `_GB10_IDX = None; if <env> == '1': from vllm import _gb10_skinny_idx as _GB10_IDX` "
                      "right after the logger")
    body = fn.body
    i = next((i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s) ==
              "_p014 = _GB10_IDX is not None and _GB10_IDX.use(self, hidden_states)"), None)
    if i is None or len([1 for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "_p014" and isinstance(n.ctx, ast.Store)]) != 1:
        raise Refused("the `_p014 = ...` dispatch assignment is not a statement of Indexer.forward (or is assigned twice)")
    br = body[i + 1]
    if not (isinstance(br, ast.If) and ast.unparse(br.test) == "_p014" and len(br.body) == 1
            and ast.unparse(br.body[0]) == "k, weights, gate_score = _GB10_IDX.project(self, hidden_states)"):
        raise Refused("the per-M branch is not `if _p014: k, weights, gate_score = _GB10_IDX.project(self, hidden_states)`")
    if _src(br.orelse) != _src(carried):
        raise Refused("the else branch is not the carried original block, statement for statement")
    guards = [s for s in body[i + 2:] if isinstance(s, ast.If) and ast.unparse(s.test) == "not _p014"]
    if len(guards) != 1 or [ast.unparse(x) for x in guards[0].body] != [
            "gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)"] or guards[0].orelse:
        raise Refused("the gate_score assignment is not guarded by exactly `if not _p014:`")
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "_GB10_IDX"]
    if sorted(n.attr for n in uses) != ["project", "use"]:
        raise Refused("_GB10_IDX is used by something other than use() / project()")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-014 marker")
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
    compile(out, "attention.py", "exec")
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
    compile(out, "attention.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p014_apply.py {apply|check|revert} <attention.py>", file=sys.stderr)
        return 2
    p = Path(argv[2])
    try:
        src = p.read_text(encoding="utf-8")
        new = revert(src) if argv[1] == "revert" else apply(src)
    except (Refused, SyntaxError, OSError, UnicodeDecodeError) as e:
        print(f"[FAIL] {p}: {e}", file=sys.stderr)
        return 1
    if argv[1] == "check":
        print(_sha(new.encode()))
        return 0
    tmp = p.with_name(p.name + ".p014.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    print(" [OK] 3 anchored edits applied (module import, Indexer.forward dispatch, gate_score guard)"
          if argv[1] == "apply" else " [OK] 3 anchored edits reverted")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
