#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 033 applier (vLLM v0.32.0 port): three anchored edits to vllm/models/glm5next/common/model.py.

  apply(text) -> text     any of the model.py bases the entrypoint can hand over after 001-032 on v0.32.0 (the tag's original;
                          + patch 025; + patch 032; + 025 + 032) -> + patch 033; fail-closed:
                          each anchor must match exactly once, every hook must land inside its function (AST-checked),
                          the result must carry N_MARKERS markers and compile
  revert(text) -> text    exact inverse (the tests rebuild every base from its patched file)

CLI (used by 033-gb10-mla-w8a16.sh):
  p033_apply.py apply <model.py>    in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p033_apply.py check <model.py>    dry run: print the sha256 the file would have after apply()
  p033_apply.py revert <model.py>   in-place inverse (tests / manual rollback)

Edits (line numbers of the v0.32.0 original, model.py sha256 f337b984...; the file moved from models/glm5next/nvidia/model.py
in v0.31.0, upstream #56176; the three anchors and the three hooks are byte-identical to the day-0 port):
  1. after `logger = init_logger(__name__)` (:102): `_p033 = None`; when VLLM_GB10_MLA_W8A16 == "1" (read once, at import,
     whitespace stripped) `from vllm import _gb10_mla_w8a16 as _p033` (module missing -> ImportError: the boot aborts).
  2. Glm5NextDecoderLayer.__init__, directly before the "# MTP layers sit past ..." comment (:403), for MLA layers (the MTP
     layer is a Glm5NextDecoderLayer too): `_p033.install(self.self_attn, vllm_config, layer_idx, is_mtp_layer)`.
  3. _try_load_fp8_attn_proj (:1469-1471), between `block_size = ...` and the BF16 dequant: `_p033.capture(params_dict[
     target_w], key, weight_fp8, scale_inv, kv_a_pad_size)` (the lookup raises exactly where the original
     `param = params_dict[target_w]` below would; capture() itself never raises). Both callers of this function
     (Glm5NextModel.load_weights, common/mtp.py Glm5NextMTP.load_weights) share it; _try_load_fp8_indexer_wk is not touched.
Patch 025's (decoder-layer forward, a module gate before class Glm5NextDecoderLayer, the Glm5NextModel.forward layer loop) and
patch 032's (a second gate block after the logger, the embed_tokens construction in Glm5NextModel.__init__) edits touch none of
these anchors, so all three match once on every base and revert() gives each base back byte for byte. mtp.py and attention.py
are not edited.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-033 gb10-mla-w8a16]"
N_MARKERS = 3
ENV = "VLLM_GB10_MLA_W8A16"
LAYER = "Glm5NextDecoderLayer"
INIT = "Glm5NextDecoderLayer.__init__"
LOADER = "_try_load_fp8_attn_proj"

# ---- Edit 1: module gate, right after the module logger ----
GATE_OLD = "logger = init_logger(__name__)\n"
GATE_NEW = (
    "logger = init_logger(__name__)\n"
    "\n"
    f"{MARKER} opt-in {ENV}=1 (read once, at import): the MLA layers' (and the MTP layer's)\n"
    "# q_a|kv_a / q_b / o_proj keep their checkpoint block-FP8 weights as Marlin W8A16 copies for decode-sized batches\n"
    "# (vllm/_gb10_mla_w8a16.py: capture at load, build + boot self-test in process_weights_after_loading, fail closed);\n"
    "# the BF16 weights stay for larger M. Unset/0 -> unchanged. Gate set + module missing -> ImportError at import.\n"
    "_p033 = None\n"
    f"if __import__(\"os\").environ.get(\"{ENV}\", \"0\").strip() == \"1\":\n"
    "    from vllm import _gb10_mla_w8a16 as _p033\n"
)

# ---- Edit 2: install the dispatching quant methods, Glm5NextDecoderLayer.__init__ ----
INST_OLD = "        # MTP layers sit past the base model's hidden layers (layer_idx >=\n"
INST_NEW = (
    f"        {MARKER} MLA layers (the MTP layer included): install the\n"
    "        # dispatching quant methods on fused_qkv_a_proj / q_b_proj / o_proj (Marlin W8A16 for M <= MAX_M, else the\n"
    "        # image's BF16 GEMM); the boot self-test runs in process_weights_after_loading. Never raises.\n"
    "        if _p033 is not None and self.layer_kind == \"mla\":\n"
    "            _p033.install(self.self_attn, vllm_config, layer_idx, is_mtp_layer)\n"
    "\n" + INST_OLD
)

# ---- Edit 3: keep the checkpoint FP8 shard + scales before the BF16 dequant, _try_load_fp8_attn_proj ----
CAP_OLD = (
    "    buf[layer_prefix].pop(key, None)\n"
    "    block_size = weight_fp8.shape[1] // scale_inv.shape[1]\n"
    "    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)\n"
)
CAP_NEW = (
    "    buf[layer_prefix].pop(key, None)\n"
    "    block_size = weight_fp8.shape[1] // scale_inv.shape[1]\n"
    f"    {MARKER} keep this projection's checkpoint FP8 shard + block scales for the\n"
    "    # decode-sized Marlin copy (capture() never raises: a capture problem only keeps BF16 for this layer; the\n"
    "    # params_dict lookup raises exactly where the original `param = params_dict[target_w]` below would).\n"
    "    if _p033 is not None:\n"
    "        _p033.capture(params_dict[target_w], key, weight_fp8, scale_inv, kv_a_pad_size)\n"
    "    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)\n"
)

EDITS = [
    ("module gate (after logger = init_logger(__name__))", GATE_OLD, GATE_NEW),
    ("install hook (Glm5NextDecoderLayer.__init__)", INST_OLD, INST_NEW),
    ("capture hook (_try_load_fp8_attn_proj)", CAP_OLD, CAP_NEW),
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


def _p033_calls(node, attr: str) -> list:
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == attr and isinstance(c.func.value, ast.Name) and c.func.value.id == "_p033"]


def _enclosing_if(node, call):
    ifs = [n for n in ast.walk(node) if isinstance(n, ast.If) and any(c is call for s in n.body for c in ast.walk(s))]
    if len(ifs) != 1:
        raise Refused(f"expected the _p033 call inside exactly one `if`, found {len(ifs)}")
    return ifs[0]


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    init = _node(tree, INIT)
    loader = _node(tree, LOADER)
    layer = _node(tree, LAYER)
    loggers = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "logger"]
    if len(loggers) != 1:
        raise Refused(f"expected one module-level `logger = ...`, found {len(loggers)}")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not patched:
        if "_p033" in names:
            raise Refused("base already mentions _p033")
        a, b = _span(text, init)
        if not a <= text.find(INST_OLD) < b:
            raise Refused(f"the MTP-comment anchor is not inside {INIT}")
        a, b = _span(text, loader)
        if not a <= text.find(CAP_OLD) < b:
            raise Refused(f"the dequant anchor is not inside {LOADER}")
        return
    # patched: `_p033 = None` + `if <gate>: from vllm import _gb10_mla_w8a16 as _p033` at module level, right after the
    # logger and before class Glm5NextDecoderLayer
    k = tree.body.index(loggers[0])
    nxt = tree.body[k + 1: k + 3]
    ok = (len(nxt) == 2 and isinstance(nxt[0], ast.Assign) and isinstance(nxt[0].targets[0], ast.Name)
          and nxt[0].targets[0].id == "_p033" and isinstance(nxt[0].value, ast.Constant) and nxt[0].value.value is None
          and isinstance(nxt[1], ast.If) and len(nxt[1].body) == 1 and isinstance(nxt[1].body[0], ast.ImportFrom)
          and nxt[1].body[0].module == "vllm"
          and [(a.name, a.asname) for a in nxt[1].body[0].names] == [("_gb10_mla_w8a16", "_p033")]
          and not nxt[1].orelse)
    if not ok:
        raise Refused("the module gate is not `_p033 = None; if ...: from vllm import _gb10_mla_w8a16 as _p033` "
                      "right after the logger")
    if tree.body.index(nxt[1]) > tree.body.index(layer):
        raise Refused(f"the module gate must precede class {LAYER}")
    gate_src = ast.unparse(nxt[1].test)
    if gate_src != f"__import__('os').environ.get('{ENV}', '0').strip() == '1'":
        raise Refused(f"unexpected gate expression {gate_src}")
    # the install hook: one call, inside Glm5NextDecoderLayer.__init__, guarded by `_p033 is not None and layer_kind == mla`
    inst = _p033_calls(tree, "install")
    if len(inst) != 1 or not any(c is inst[0] for c in ast.walk(init)):
        raise Refused(f"expected exactly one _p033.install call, inside {INIT}")
    cond = ast.unparse(_enclosing_if(init, inst[0]).test)
    if cond != "_p033 is not None and self.layer_kind == 'mla'":
        raise Refused(f"unexpected install guard {cond}")
    if ast.unparse(inst[0]) != "_p033.install(self.self_attn, vllm_config, layer_idx, is_mtp_layer)":
        raise Refused(f"unexpected install call {ast.unparse(inst[0])}")
    # the capture hook: one call, inside _try_load_fp8_attn_proj, before the BF16 dequant call
    cap = _p033_calls(tree, "capture")
    if len(cap) != 1 or not any(c is cap[0] for c in ast.walk(loader)):
        raise Refused(f"expected exactly one _p033.capture call, inside {LOADER}")
    if ast.unparse(_enclosing_if(loader, cap[0]).test) != "_p033 is not None":
        raise Refused("the capture hook must be guarded by `if _p033 is not None:`")
    if ast.unparse(cap[0]) != "_p033.capture(params_dict[target_w], key, weight_fp8, scale_inv, kv_a_pad_size)":
        raise Refused(f"unexpected capture call {ast.unparse(cap[0])}")
    deq = [c for c in ast.walk(loader) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
           and c.func.id == "_dequant_fp8_block"]
    if len(deq) != 1 or not cap[0].lineno < deq[0].lineno:
        raise Refused("the capture call must precede the (single) _dequant_fp8_block call")
    uses = [c for c in ast.walk(tree) if isinstance(c, ast.Attribute) and isinstance(c.value, ast.Name)
            and c.value.id == "_p033"]
    if len(uses) != 2:
        raise Refused("_p033 is used by something other than the two hooks")
    allassign = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_p033" for t in n.targets)]
    if len(allassign) != 1:
        raise Refused(f"expected exactly one assignment to _p033, found {len(allassign)}")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-033 marker")
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
        print("usage: p033_apply.py {apply|check|revert} <model.py>", file=sys.stderr)
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
    tmp = p.with_name(p.name + ".p033.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if argv[1] == "revert" else "applied"
    print(f" [OK] {len(EDITS)} anchored edits {what} on {p.name} (module gate _p033; install hook in {INIT}; "
          f"capture hook in {LOADER})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
