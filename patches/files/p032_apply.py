#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 032 applier: two anchored edits to vllm/models/glm5next/common/model.py (written here; vLLM v0.32.0 port).

  apply(text) -> text     any model.py base the entrypoint can hand over after 001-031 (the v0.32.0 original, or patch
                          025's result) -> + patch 032; fail-closed: each anchor must match exactly once, the gate must
                          land at module level before class Glm5NextModel and the class choice inside
                          Glm5NextModel.__init__'s `if get_pp_group().is_first_rank:` branch (AST-checked), the result
                          must carry N_MARKERS markers and compile
  revert(text) -> text    exact inverse (the tests rebuild every base from its patched file)

CLI (used by 032-gb10-repl-embed.sh):
  p032_apply.py apply <model.py>    in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p032_apply.py check <model.py>    dry run: print the sha256 the file would have after apply()
  p032_apply.py revert <model.py>   in-place inverse (tests / manual rollback)

Edits (v0.32.0 line numbers, model.py sha256 f337b984...; the day-0 image's file had the same two anchors at :95 / :632):
  1. after `logger = init_logger(__name__)` (:102): `_p032 = None`; when VLLM_GB10_REPL_EMBED == "1" (read once, at
     import) `from vllm import _gb10_repl_embed as _p032` (module missing -> ImportError: the boot aborts).
  2. Glm5NextModel.__init__ (:724-729): `_p032_cls = VocabParallelEmbedding if _p032 is None else
     _p032.embedding_class(config)` and `self.embed_tokens = _p032_cls(<the original three arguments, unchanged>)`.
Neither anchor moved in v0.32.0 (the file moved from models/glm5next/nvidia/ to models/glm5next/common/ and gained ~450
lines elsewhere; both anchors are byte-identical and unique). Patch 025 (the only other model.py writer in the new set:
module gate before class Glm5NextDecoderLayer, the decoder-layer forward, the Glm5NextModel.forward layer loop) touches
neither anchor, so both match once on the 025 result too and revert() gives each base back byte for byte. Patch 033
(runs after 032) inserts its own gate directly after the same logger line; it is applied AFTER this one, so this applier
never sees it (and revert() must be called in reverse order: 033 first).
mtp.py is NOT edited (see vllm/_gb10_repl_embed.py: the draft shares the target's embed_tokens through load_eagle_model).
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-032 gb10-repl-embed]"
N_MARKERS = 2
ENV = "VLLM_GB10_REPL_EMBED"
CLASS = "Glm5NextModel"
INIT = "Glm5NextModel.__init__"

# ---- Edit 1: module gate, right after the module logger ----
GATE_OLD = "\nlogger = init_logger(__name__)\n"
GATE_NEW = (
    "\nlogger = init_logger(__name__)\n"
    "\n"
    f"{MARKER} opt-in {ENV}=1 (read once, at import): Glm5NextModel builds embed_tokens\n"
    "# with vllm/_gb10_repl_embed.py's replicated subclass of VocabParallelEmbedding (the full vocabulary on every TP\n"
    "# rank, no all-reduce); its fail-closed boot self-test after weight loading keeps it or restores the image's\n"
    "# vocab-parallel layer. Unset/0 -> unchanged. Gate set + module missing -> ImportError at import (boot aborts).\n"
    "_p032 = None\n"
    f"if __import__(\"os\").environ.get(\"{ENV}\", \"0\") == \"1\":\n"
    "    from vllm import _gb10_repl_embed as _p032\n"
)

# ---- Edit 2: the embed_tokens class, Glm5NextModel.__init__ ----
CTOR_OLD = (
    "        if get_pp_group().is_first_rank:\n"
    "            self.embed_tokens = VocabParallelEmbedding(\n"
    "                config.vocab_size,\n"
    "                config.hidden_size,\n"
    '                prefix=f"{prefix}.embed_tokens",\n'
    "            )\n"
)
CTOR_NEW = (
    "        if get_pp_group().is_first_rank:\n"
    f"            {MARKER} {ENV}=1: embedding_class() returns the replicated\n"
    "            # subclass (full vocabulary on every TP rank, no all-reduce; fail-closed boot self-test after weight\n"
    "            # loading) or VocabParallelEmbedding when this config is outside its validated envelope.\n"
    "            _p032_cls = VocabParallelEmbedding if _p032 is None else _p032.embedding_class(config)\n"
    "            self.embed_tokens = _p032_cls(\n"
    "                config.vocab_size,\n"
    "                config.hidden_size,\n"
    '                prefix=f"{prefix}.embed_tokens",\n'
    "            )\n"
)

EDITS = [
    ("module gate (after logger = init_logger(__name__))", GATE_OLD, GATE_NEW),
    ("embed_tokens class (Glm5NextModel.__init__)", CTOR_OLD, CTOR_NEW),
]


class Refused(ValueError):
    pass


def _node(tree: ast.Module, qualname: str):
    body, node = tree.body, None
    for part in qualname.split("."):
        hits = [n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part]
        if len(hits) != 1:
            raise Refused(f"{qualname}: {part} found {len(hits)} times (expected 1)")
        node = hits[0]
        body = node.body
    return node


def _span(text: str, node) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[: node.lineno - 1])), sum(map(len, lines[: node.end_lineno]))


def _is_first_rank_test(test) -> bool:
    return (isinstance(test, ast.Attribute) and test.attr == "is_first_rank" and isinstance(test.value, ast.Call)
            and isinstance(test.value.func, ast.Name) and test.value.func.id == "get_pp_group")


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    init = _node(tree, INIT)
    cls = _node(tree, CLASS)
    loggers = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "logger"]
    if len(loggers) != 1:
        raise Refused(f"expected one module-level `logger = ...`, found {len(loggers)}")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not patched:
        if "_p032" in names or "_p032_cls" in names:
            raise Refused("base already mentions _p032 / _p032_cls")
        a, b = _span(text, init)
        i = text.find(CTOR_OLD)
        if not a <= i < b:
            raise Refused(f"the embed_tokens construction is not inside {INIT}")
        return
    # patched: `_p032 = None` + `if <gate>: from vllm import _gb10_repl_embed as _p032` at module level, right after
    # the logger and before class Glm5NextModel
    k = tree.body.index(loggers[0])
    nxt = tree.body[k + 1: k + 3]
    ok = (len(nxt) == 2 and isinstance(nxt[0], ast.Assign) and isinstance(nxt[0].targets[0], ast.Name)
          and nxt[0].targets[0].id == "_p032" and isinstance(nxt[0].value, ast.Constant) and nxt[0].value.value is None
          and isinstance(nxt[1], ast.If) and len(nxt[1].body) == 1 and isinstance(nxt[1].body[0], ast.ImportFrom)
          and nxt[1].body[0].module == "vllm"
          and [(a.name, a.asname) for a in nxt[1].body[0].names] == [("_gb10_repl_embed", "_p032")]
          and not nxt[1].orelse)
    if not ok:
        raise Refused("the module gate is not `_p032 = None; if ...: from vllm import _gb10_repl_embed as _p032` "
                      "right after the logger")
    if tree.body.index(nxt[1]) > tree.body.index(cls):
        raise Refused("the module gate must precede class Glm5NextModel")
    gate_src = ast.unparse(nxt[1].test)
    if gate_src != f"__import__('os').environ.get('{ENV}', '0') == '1'":
        raise Refused(f"unexpected gate expression {gate_src}")
    # the class choice: one `_p032_cls = ...` inside Glm5NextModel.__init__, in the is_first_rank branch, and the
    # embed_tokens construction calls it
    ifs = [n for n in ast.walk(init) if isinstance(n, ast.If) and _is_first_rank_test(n.test)
           and any(isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "_p032_cls"
                   for s in n.body)]
    if len(ifs) != 1:
        raise Refused(f"{INIT}: expected the _p032_cls choice inside `if get_pp_group().is_first_rank:`")
    body = ifs[0].body
    if not (len(body) == 2 and isinstance(body[1], ast.Assign)
            and ast.unparse(body[1].targets[0]) == "self.embed_tokens"
            and isinstance(body[1].value, ast.Call) and isinstance(body[1].value.func, ast.Name)
            and body[1].value.func.id == "_p032_cls"):
        raise Refused(f"{INIT}: embed_tokens must be built by _p032_cls right after the choice")
    if ast.unparse(body[0].value) != ("VocabParallelEmbedding if _p032 is None else "
                                      "_p032.embedding_class(config)"):
        raise Refused(f"{INIT}: unexpected class choice {ast.unparse(body[0].value)}")
    allassign = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ("_p032", "_p032_cls") for t in n.targets)]
    if len(allassign) != 2:
        raise Refused(f"expected exactly two assignments to _p032 / _p032_cls, found {len(allassign)}")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-032 marker")
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
    compile(out, "model.py", "exec")
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
    compile(out, "model.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p032_apply.py {apply|check|revert} <model.py>", file=sys.stderr)
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
    tmp = p.with_name(p.name + ".p032.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if argv[1] == "revert" else "applied"
    print(f" [OK] {len(EDITS)} anchored edits {what} on {p.name} (module gate _p032; embed_tokens class choice in "
          f"{INIT})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
