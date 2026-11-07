#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 029 applier: anchored edits to two vLLM files (written here; ported to v0.32.0 as module rev 3; stdlib only).

  apply(kind, text) -> text     v0.32.0 original -> + patch 029; fail-closed: every anchor must match exactly once, the
                                hooks must land in the expected functions (AST-checked), the result must carry the
                                expected number of markers and compile
  revert(kind, text) -> text    exact inverse (tests rebuild the base from a patched file)

kind 'mtp'  = vllm/models/glm5next/common/mtp.py (v0.32.0 sha256 55f33f99...; moved from nvidia/ in v0.31.0; no other
  patch of the v0.32.0 set edits it), 3 edits:
  1. after the imports (the `from .model import (...)` block, whose `.ops.fused_eh_norm` import became absolute in
     v0.32.0), before `class Glm5NextMultiTokenPredictorLayer`: module gate `_P029_ON` (env read once at import) and,
     only when set, `from vllm import _gb10_draft_compress as _p029` (gate + module missing -> ImportError at import:
     fail closed);
  2. Glm5NextMultiTokenPredictorLayer.forward: `hidden_states = self.eh_proj(eh_input)` ->
     `_p029.eh_proj_forward(self, eh_input) if _P029_ON else self.eh_proj(eh_input)`;
  3. Glm5NextMultiTokenPredictor.compute_logits (the MTP DRAFT logits only; Glm5NextForCausalLM.compute_logits lives
     in model.py and is not touched): `if _P029_ON: return _p029.draft_logits(self, mtp_layer, hidden_states)` before
     the image `return self.logits_processor(mtp_layer.shared_head.head, hidden_states)`.
kind 'spec' = vllm/v1/worker/gpu/spec_decode/mtp/speculator.py (v0.32.0 sha256 b730010c..., no other patch edits it),
  1 edit: MTPSpeculator.load_draft_model, right after `draft_model = load_eagle_model(target_model, self.vllm_config)`
  (i.e. after the target's lm_head was shared into the MTP layers): when the gate is set, `_p029.prepare(draft_model,
  self.vllm_config)` (builds the FP8 head copy + the eh_proj row split and runs the collective boot self-test; never
  raises).

CLI (used by 029-gb10-draft-compress.sh):
  p029_apply.py apply  {mtp|spec} <file>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p029_apply.py check  {mtp|spec} <file>   dry run: print the sha256 the file would have after apply()
  p029_apply.py revert {mtp|spec} <file>   in-place inverse (tests / manual rollback)
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-029 gb10-draft-compress]"
ENV = "VLLM_GB10_DRAFT_COMPRESS"

# ---- mtp.py --------------------------------------------------------------------------------------------------------
MTP_GATE_OLD = (
    "    get_spec_layer_idx_from_weight_name,\n"
    ")\n"
    "\n"
    "\n"
    "class Glm5NextMultiTokenPredictorLayer(nn.Module):\n"
)
MTP_GATE_NEW = (
    "    get_spec_layer_idx_from_weight_name,\n"
    ")\n"
    "\n"
    f"{MARKER} opt-in {ENV}=1 (read once, at import): the MTP DRAFT\n"
    "# passes use a draft-only FP8 (W8A16, Marlin) copy of the shared lm_head shard and a weight-row split of eh_proj\n"
    "# assembled by patch 012's one-shot all-reduce (vllm/_gb10_draft_compress.py; built and boot-self-tested by\n"
    "# MTPSpeculator.load_draft_model, fail closed). The target's lm_head / logits / verification are untouched.\n"
    "# Unset/0 -> unchanged. Gate set + module missing -> ImportError at import (fail closed).\n"
    f"_P029_ON = __import__(\"os\").environ.get(\"{ENV}\", \"0\").strip() == \"1\"\n"
    "if _P029_ON:\n"
    "    from vllm import _gb10_draft_compress as _p029\n"
    "\n"
    "\n"
    "class Glm5NextMultiTokenPredictorLayer(nn.Module):\n"
)
MTP_EH_OLD = "        hidden_states = self.eh_proj(eh_input)\n"
MTP_EH_NEW = (
    f"        {MARKER} M <= the one-shot limit: this rank's\n"
    "        # eh_proj row block + patch 012's one-shot (an exact gather); else the replicated eh_proj.\n"
    "        hidden_states = (\n"
    "            _p029.eh_proj_forward(self, eh_input) if _P029_ON else self.eh_proj(eh_input)\n"
    "        )\n"
)
MTP_LOGITS_OLD = (
    "        # second RMSNorm.\n"
    "        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)\n"
)
MTP_LOGITS_NEW = (
    "        # second RMSNorm.\n"
    f"        {MARKER} DRAFT logits only: the FP8 copy of the\n"
    "        # shared lm_head shard through the same LogitsProcessor (all-gather, vocab slice); the image head for\n"
    "        # ineligible shapes or when the boot self-test refused.\n"
    "        if _P029_ON:\n"
    "            return _p029.draft_logits(self, mtp_layer, hidden_states)\n"
    "        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)\n"
)

# ---- mtp/speculator.py ---------------------------------------------------------------------------------------------
SPEC_OLD = "        draft_model = load_eagle_model(target_model, self.vllm_config)\n"
SPEC_NEW = (
    "        draft_model = load_eagle_model(target_model, self.vllm_config)\n"
    f"        {MARKER} {ENV}=1: after load_eagle_model shared\n"
    "        # the target's lm_head into the MTP layers, build the draft-only FP8 head copy + the eh_proj row split and\n"
    "        # run the collective boot self-test (never raises: a refusal keeps the image draft path, logged as\n"
    "        # '[patch-029] ... INACTIVE (fail closed)').\n"
    f"        if __import__(\"os\").environ.get(\"{ENV}\", \"0\").strip() == \"1\":\n"
    "            from vllm import _gb10_draft_compress as _p029\n"
    "\n"
    "            _p029.prepare(draft_model, self.vllm_config)\n"
)

EDITS = {
    "mtp": [
        ("module gate (before class Glm5NextMultiTokenPredictorLayer)", MTP_GATE_OLD, MTP_GATE_NEW),
        ("eh_proj hook (Glm5NextMultiTokenPredictorLayer.forward)", MTP_EH_OLD, MTP_EH_NEW),
        ("draft-logits hook (Glm5NextMultiTokenPredictor.compute_logits)", MTP_LOGITS_OLD, MTP_LOGITS_NEW),
    ],
    "spec": [
        ("prepare hook (MTPSpeculator.load_draft_model)", SPEC_OLD, SPEC_NEW),
    ],
}
N_MARKERS = {"mtp": 3, "spec": 1}


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


def _calls(node, attr: str) -> list:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr and isinstance(n.func.value, ast.Name) and n.func.value.id == "_p029"]


def check_structure(kind: str, text: str, patched: bool) -> None:
    tree = ast.parse(text)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if kind == "mtp":
        fwd = _node(tree, "Glm5NextMultiTokenPredictorLayer.forward")
        cl = _node(tree, "Glm5NextMultiTokenPredictor.compute_logits")
        _node(tree, "Glm5NextMultiTokenPredictor.get_top_tokens")
        if not patched:
            if "_P029_ON" in names or "_p029" in names:
                raise Refused("base already mentions _P029_ON / _p029")
            ft = ast.get_source_segment(text, fwd) or ""
            if "self.eh_proj(eh_input)" not in ft:
                raise Refused("Glm5NextMultiTokenPredictorLayer.forward: the eh_proj call is absent; refusing")
            ct = ast.get_source_segment(text, cl) or ""
            if "self.logits_processor(mtp_layer.shared_head.head, hidden_states)" not in ct:
                raise Refused("Glm5NextMultiTokenPredictor.compute_logits: the head call is absent; refusing")
            return
        gates = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_P029_ON" for t in n.targets)]
        if len(gates) != 1:
            raise Refused("expected exactly one module-level _P029_ON assignment")
        cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextMultiTokenPredictorLayer"]
        if not cls or tree.body.index(gates[0]) > tree.body.index(cls[0]):
            raise Refused("_P029_ON must be assigned before class Glm5NextMultiTokenPredictorLayer")
        if len(_calls(tree, "eh_proj_forward")) != 1 or len(_calls(fwd, "eh_proj_forward")) != 1:
            raise Refused("expected one _p029.eh_proj_forward call, inside Glm5NextMultiTokenPredictorLayer.forward")
        if len(_calls(tree, "draft_logits")) != 1 or len(_calls(cl, "draft_logits")) != 1:
            raise Refused("expected one _p029.draft_logits call, inside Glm5NextMultiTokenPredictor.compute_logits")
        if _calls(tree, "prepare"):
            raise Refused("mtp.py must not call _p029.prepare")
        return
    # spec
    ld = _node(tree, "MTPSpeculator.load_draft_model")
    if not patched:
        if "_p029" in names:
            raise Refused("base already mentions _p029")
        return
    calls = _calls(tree, "prepare")
    if len(calls) != 1 or len(_calls(ld, "prepare")) != 1:
        raise Refused("expected one _p029.prepare call, inside MTPSpeculator.load_draft_model")
    # it must come after the load_eagle_model assignment and before the return
    stmts = ld.body
    i_load = next((i for i, s in enumerate(stmts) if isinstance(s, ast.Assign)
                   and isinstance(s.value, ast.Call) and getattr(s.value.func, "id", None) == "load_eagle_model"), None)
    i_hook = next((i for i, s in enumerate(stmts) if isinstance(s, ast.If) and _calls(s, "prepare")), None)
    i_ret = next((i for i, s in enumerate(stmts) if isinstance(s, ast.Return)), None)
    if i_load is None or i_hook is None or i_ret is None or not (i_load < i_hook < i_ret) or i_hook != i_load + 1:
        raise Refused(f"prepare hook placement: load_eagle_model at {i_load}, hook at {i_hook}, return at {i_ret}")


def apply(kind: str, text: str) -> str:
    if kind not in EDITS:
        raise Refused(f"unknown kind {kind!r}")
    if MARKER in text:
        raise Refused("input already carries the patch-029 marker")
    bad = [(name, text.count(old)) for name, old, _ in EDITS[kind] if text.count(old) != 1]
    if bad:
        raise Refused("; ".join(f"anchor '{n}' found {c}x (expected 1)" for n, c in bad))
    check_structure(kind, text, patched=False)
    out = text
    for name, old, new in EDITS[kind]:
        if out.count(old) != 1:
            raise Refused(f"anchor '{name}' changed by an earlier edit")
        out = out.replace(old, new, 1)
    if out.count(MARKER) != N_MARKERS[kind]:
        raise Refused(f"expected {N_MARKERS[kind]} markers after the edits, found {out.count(MARKER)}")
    compile(out, f"{kind}.py", "exec")
    check_structure(kind, out, patched=True)
    return out


def revert(kind: str, text: str) -> str:
    if kind not in EDITS:
        raise Refused(f"unknown kind {kind!r}")
    out = text
    for name, old, new in reversed(EDITS[kind]):
        if out.count(new) != 1:
            raise Refused(f"patched block '{name}' found {out.count(new)}x (expected 1)")
        out = out.replace(new, old, 1)
    if MARKER in out:
        raise Refused("marker still present after revert")
    compile(out, f"{kind}.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in ("apply", "check", "revert") or argv[2] not in EDITS:
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p029_apply.py {apply|check|revert} {mtp|spec} <file>", file=sys.stderr)
        return 2
    mode, kind, p = argv[1], argv[2], Path(argv[3])
    try:
        src = p.read_text(encoding="utf-8")
        new = revert(kind, src) if mode == "revert" else apply(kind, src)
    except (Refused, SyntaxError, OSError, UnicodeDecodeError) as e:
        print(f"[FAIL] {p}: {e}", file=sys.stderr)
        return 1
    if mode == "check":
        print(_sha(new.encode()))
        return 0
    tmp = p.with_name(p.name + ".p029.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    print(f" [OK] {len(EDITS[kind])} anchored edit(s) {'reverted' if mode == 'revert' else 'applied'} on {p.name} "
          f"({kind})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
