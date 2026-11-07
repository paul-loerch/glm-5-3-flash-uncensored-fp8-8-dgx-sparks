#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 025 applier (v0.32.0 port): four anchored edits to vllm/models/glm5next/common/model.py (written here).

  apply(text) -> text     v0.32.0 original (or any base that still carries the four anchors) -> + patch 025; fail-closed:
                          every anchor must match exactly once, the hooks must land in Glm5NextDecoderLayer.forward and
                          Glm5NextModel.forward (AST-checked), the result must carry N_MARKERS markers and compile
  revert(text) -> text    exact inverse (the tests rebuild the base from a patched file)

CLI (used by 025-gb10-mhc-prefill-sp.sh):
  p025_apply.py apply <model.py>    in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p025_apply.py check <model.py>    dry run: print the sha256 the file would have after apply()
  p025_apply.py revert <model.py>   in-place inverse (tests / manual rollback)

Edits (v0.32.0 line numbers, common/model.py sha256 f337b984...):
  1. before `class Glm5NextDecoderLayer` (:345): module gate `_P025_ON` (env read once at import) and, only when set,
     `from vllm import _gb10_prefill_sp as _p025` (gate set + module missing -> ImportError at import: fail closed).
  2. first statements of Glm5NextDecoderLayer.forward (:531): when the gate is on and an sp region is active (set by
     Glm5NextModel.forward for an eligible prefill) and this is an mHC, non-MTP layer, delegate to
     _p025.sp_layer_forward; otherwise the upstream forward runs unchanged.
  3. Glm5NextModel.forward, the layer loop (:822-833; v0.31+ shape with the EAGLE-3 aux taps and enumerate(...,
     start=self.start_layer)): begin the sp forward (passing self._active_layers, self.is_sequence_parallel and
     self.aux_hidden_state_layers, so the first-eligible-call boot self-test checks the envelope + collective + the LIVE
     layer-weight mHC shard-invariance, and a forward that captures aux layers is never sharded), then, inside a
     try/finally that ends the sp region, shard the residual stream and run the (unchanged) loop. Inert (handle None)
     unless the gate is on AND the forward is an eligible, eager, not-ubatched, >= MIN_T prefill.
  4. Glm5NextModel.forward, the exit (:851-854): `elif _p025_handle is not None:` all-gather this rank's rows before the
     norm (the upstream `if self.is_sequence_parallel:` branch above it is the EP+DP>1 sequence-parallel-MoE path, which
     the envelope refuses together with sp).
Upstream's own sequence-parallel code (sp_shard / sp_all_gather / sp_reduce_scatter, use_sequence_parallel_moe) is inert
at DP=1: the module gate, the hooks and the envelope compose with it only as "refuse".
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-025 gb10-mhc-prefill-sp]"
N_MARKERS = 4
ENV = "VLLM_GB10_MHC_SP"
MODEL_FORWARD = "Glm5NextModel.forward"
LAYER_FORWARD = "Glm5NextDecoderLayer.forward"

# ---- Edit 1: module gate, before class Glm5NextDecoderLayer ----
GATE_OLD = "\n\nclass Glm5NextDecoderLayer(nn.Module):\n"
GATE_NEW = (
    "\n\n"
    f"{MARKER} opt-in {ENV}=sp/ref (read once, at import): during eligible prefills (T >=\n"
    f"# {ENV}_MIN_T, default 1,024; eager, never inside a CUDA-graph capture) the residual stream is sharded across\n"
    "# the 8 TP ranks and the attention / MLP all-reduces become reduce-scatter (vllm/_gb10_prefill_sp.py) - a\n"
    "# NUMERICS CHANGE vs the all-reduce path (serving needs the owner's OK). Unset/0 -> unchanged. Gate set +\n"
    "# module missing / an invalid value -> ImportError / ValueError at import: fail closed.\n"
    f"_P025_ON = __import__(\"os\").environ.get(\"{ENV}\", \"0\").strip().lower() not in "
    "(\"\", \"0\", \"off\", \"false\", \"no\")\n"
    "if _P025_ON:\n"
    "    from vllm import _gb10_prefill_sp as _p025\n"
    "\n\nclass Glm5NextDecoderLayer(nn.Module):\n"
)

# ---- Edit 2: decoder-layer forward hook ----
LAYER_OLD = (
    "        # 70B or MTP layers: KDA + MoE without HC.\n"
    "        if not self.mhc or self.is_mtp_layer:\n"
)
LAYER_NEW = (
    f"        {MARKER} {ENV}=sp/ref + an active sp region (set by\n"
    "        # Glm5NextModel.forward for an eligible prefill): run this mHC layer sequence-parallel (the residual\n"
    "        # stream stays on this rank's rows; attention / MLP all-reduces -> reduce-scatter). MTP and non-mHC\n"
    "        # layers are unchanged. See vllm/_gb10_prefill_sp.py.\n"
    "        if _P025_ON and _p025.sp_active() and self.mhc and not self.is_mtp_layer:\n"
    "            return _p025.sp_layer_forward(\n"
    "                self, positions, hidden_states, residual, post, comb\n"
    "            )\n"
    "\n"
    "        # 70B or MTP layers: KDA + MoE without HC.\n"
    "        if not self.mhc or self.is_mtp_layer:\n"
)

# ---- Edit 3: model forward loop ----
LOOP_OLD = (
    "        aux_hidden_states: list[torch.Tensor] = []\n"
    "        for idx, layer in enumerate(self._active_layers, start=self.start_layer):\n"
    "            if idx in self.aux_hidden_state_layers:\n"
    "                aux_hidden_state = self._aux_hidden_state(\n"
    "                    hidden_states, residual, post, comb\n"
    "                )\n"
    "                if self.is_sequence_parallel:\n"
    "                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]\n"
    "                aux_hidden_states.append(aux_hidden_state)\n"
    "            hidden_states, residual, post, comb = layer(\n"
    "                positions, hidden_states, residual, post, comb\n"
    "            )\n"
)
LOOP_NEW = (
    f"        {MARKER} {ENV}=sp/ref: shard the residual stream across the TP\n"
    "        # group for an eligible prefill (T >= MIN_T, eager, not ubatched, no aux taps), run the layers\n"
    "        # sequence-parallel, then all-gather before the norm. Inert (handle None) for decode / short / captured /\n"
    "        # micro-batched / aux-capturing forwards and when the gate is unset. begin_forward runs the fail-closed boot\n"
    "        # self-test on the first eligible forward (the envelope + the collective + the LIVE layer-weight mHC\n"
    "        # shard-invariance via `layers`) and neutralises patch 022 (prs) for the region; end_forward restores it.\n"
    "        # The envelope refuses PP > 1, so when a handle is live this rank is always the last PP rank and the final\n"
    "        # all-gather below runs.\n"
    "        _p025_handle = _p025.begin_forward(\n"
    "            full_num_tokens, self.config.hidden_size, device=hidden_states.device,\n"
    "            layers=self._active_layers, is_sequence_parallel=self.is_sequence_parallel,\n"
    "            aux_layers=self.aux_hidden_state_layers,\n"
    "        ) if _P025_ON else None\n"
    "        aux_hidden_states: list[torch.Tensor] = []\n"
    "        try:\n"
    "            if _p025_handle is not None:\n"
    "                hidden_states = _p025.shard_model_input(\n"
    "                    hidden_states, _p025_handle, _p025_handle.world.rank\n"
    "                )\n"
    "            for idx, layer in enumerate(self._active_layers, start=self.start_layer):\n"
    "                if idx in self.aux_hidden_state_layers:\n"
    "                    aux_hidden_state = self._aux_hidden_state(\n"
    "                        hidden_states, residual, post, comb\n"
    "                    )\n"
    "                    if self.is_sequence_parallel:\n"
    "                        aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]\n"
    "                    aux_hidden_states.append(aux_hidden_state)\n"
    "                hidden_states, residual, post, comb = layer(\n"
    "                    positions, hidden_states, residual, post, comb\n"
    "                )\n"
    "        finally:\n"
    "            if _p025_handle is not None:\n"
    "                _p025.end_forward(_p025_handle)\n"
)

# ---- Edit 4: model forward exit gather ----
EXIT_OLD = (
    "        if self.is_sequence_parallel:\n"
    "            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]\n"
    "\n"
    "        hidden_states = self.norm(hidden_states)\n"
)
EXIT_NEW = (
    "        if self.is_sequence_parallel:\n"
    "            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]\n"
    f"        {MARKER}: this rank's rows -> the full [T, H] before the norm\n"
    "        elif _p025_handle is not None:\n"
    "            hidden_states = _p025.gather_model_output(hidden_states, _p025_handle)\n"
    "\n"
    "        hidden_states = self.norm(hidden_states)\n"
)

EDITS = [
    ("module gate (before class Glm5NextDecoderLayer)", GATE_OLD, GATE_NEW),
    ("decoder-layer sp hook (Glm5NextDecoderLayer.forward)", LAYER_OLD, LAYER_NEW),
    ("model sp wrap (Glm5NextModel.forward layer loop)", LOOP_OLD, LOOP_NEW),
    ("model sp exit gather (Glm5NextModel.forward, before the norm)", EXIT_OLD, EXIT_NEW),
]


class Refused(ValueError):
    pass


def _method(tree: ast.Module, qualname: str) -> ast.FunctionDef:
    body, node = tree.body, None
    for part in qualname.split("."):
        hits = [n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part]
        if len(hits) != 1:
            raise Refused(f"{qualname}: {part} found {len(hits)} times (expected 1)")
        node = hits[0]
        body = node.body
    return node


def _span(text: str, qualname: str) -> tuple[int, int]:
    node = _method(ast.parse(text), qualname)
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[: node.lineno - 1])), sum(map(len, lines[: node.end_lineno]))


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    # The two forwards must exist exactly once each (guards against a 007 base that reshaped them).
    lf = _method(tree, LAYER_FORWARD)
    mf = _method(tree, MODEL_FORWARD)
    has_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not patched:
        if "_P025_ON" in has_names or "_p025" in has_names:
            raise Refused("base already mentions _P025_ON / _p025")
        # Glm5NextModel.forward must still shard via sp_shard and gather via sp_all_gather (the image SP skeleton we
        # wrap); a reshaped forward (e.g. patch 007) would fail this and be refused.
        mtext = text[slice(*_span(text, MODEL_FORWARD))]
        if "sp_shard(hidden_states)" not in mtext or "sp_all_gather(hidden_states)" not in mtext:
            raise Refused(f"{MODEL_FORWARD}: the upstream SP skeleton (sp_shard / sp_all_gather) is absent; refusing")
        if "for idx, layer in enumerate(self._active_layers, start=self.start_layer):" not in mtext:
            raise Refused(f"{MODEL_FORWARD}: the expected layer loop is absent; refusing")
        ltext = text[slice(*_span(text, LAYER_FORWARD))]
        if "if not self.mhc or self.is_mtp_layer:" not in ltext:
            raise Refused(f"{LAYER_FORWARD}: the expected MTP / non-mHC guard is absent; refusing")
        return
    # Patched: the gate assignment precedes Glm5NextDecoderLayer; the hooks are the first real statements.
    gates = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_P025_ON" for t in n.targets)]
    if len(gates) != 1:
        raise Refused("expected exactly one module-level _P025_ON assignment")
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextDecoderLayer"]
    if not cls or tree.body.index(gates[0]) > tree.body.index(cls[0]):
        raise Refused("_P025_ON must be assigned before class Glm5NextDecoderLayer")
    # the decoder hook: an `if ... _p025.sp_layer_forward(...)` returning before the image body
    lf_calls = [n for n in ast.walk(lf) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "sp_layer_forward"]
    if len(lf_calls) != 1:
        raise Refused(f"{LAYER_FORWARD}: expected one _p025.sp_layer_forward call")
    mf_names = {n.attr for n in ast.walk(mf) if isinstance(n, ast.Attribute)}
    if not {"begin_forward", "shard_model_input", "end_forward", "gather_model_output"} <= mf_names:
        raise Refused(f"{MODEL_FORWARD}: expected begin_forward / shard_model_input / end_forward / "
                      "gather_model_output")
    # the try/finally wrapping the loop must be present
    if not any(isinstance(n, ast.Try) for n in ast.walk(mf)):
        raise Refused(f"{MODEL_FORWARD}: the layer loop is not wrapped in try/finally")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-025 marker")
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
        print("usage: p025_apply.py {apply|check|revert} <model.py>", file=sys.stderr)
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
    tmp = p.with_name(p.name + ".p025.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if argv[1] == "revert" else "applied"
    print(f" [OK] {len(EDITS)} anchored edits {what} on {p.name} "
          "(module gate _P025_ON; sp hook in Glm5NextDecoderLayer.forward; sp wrap in Glm5NextModel.forward)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
