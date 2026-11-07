#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# [patch-020 gb10-sparse-mla-triton-prefill] applier, REVISION 4 (vLLM v0.32.0 port) — five anchored
# edits to vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py (the SM121-baked file's sha is pinned in
# the .sh; no other patch of this set touches the file). Commands: apply / revert / probe. Fail-closed: every
# anchor must match exactly once against the ORIGINAL text before anything is written; the result is compiled
# in memory; the caller re-checks the sha. Idempotent via the marker. Used by 020-*.sh, files/p020_repin.sh and
# files/p020_cpu_check.py. Runs on Python >= 3.9 (the image has 3.12).
#
#   1. module level: gate helper + import of vllm._gb10_sparse_mla_prefill. With the gate set an import
#      failure RAISES (backend import = engine start-up); only with the gate unset is it tolerated.
#   2. FlashInferMLASparseSM90Builder.build(): _p020_module().plan_step(...) + early return when the gate
#      is set (decode-prefix-only plan), else upstream's planning lines verbatim.
#   3. FlashInferMLASparseSM90Impl.forward_mqa(): keep the valid_counts the convert kernel already returns
#      (upstream discards them with `_`).
#   4. FlashInferMLASparseSM90Impl.forward_mqa(): _p020_module().forward_split(...) when the gate is set,
#      else upstream's pack_indices + wrapper.run verbatim.
#   5. FlashInferMLASparseSM90Impl.__init__(): _p020_module().startup_check(...) when the gate is set
#      (config refusal + on-device kernel self-test / warm-up before serving).
#
#   probe <sm90.py> [cuda.py] prints "<cap> <list>" (read-only facts for the .sh, never raises): <cap> = can the
#   target's FlashInferMLASparseSM90Backend be selected on capability 12.1 (yes/no/unknown; the method's own return
#   expression is evaluated), <list> = does platforms/cuda.py list it for capability 12 ahead of the SM120 backend
#   (yes / late = after it / no / unknown).
#
# Revision 3 (vLLM 487ecf187) had four edits and two gates (NOSYNC + TRITON_PREFILL); NOSYNC is upstream's
# own sync-free plan since #58684 and is gone. The anchors below are independent of the SM121-bake edits
# (supports_compute_capability -> (9, 12), backend="fa3" -> fa2 off Hopper; the community bake also relaxed the
# fp8 FlashInfer-version gate), so the same applier works on the pristine file, the frozen bake's file and the
# MiaAI-Lab spelling of the bake.
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

MARKER = "# [patch-020 gb10-sparse-mla-triton-prefill]"
N_MARKERS = 5

_IMPORT_ANCHOR = (
    "# device-side clamp restores the exact lengths.\n"
    "_PLAN_SLACK = 32\n"
)
# The block inserted after the module constants (also exec'd on its own by p020_cpu_check.py).
IMPORT_BLOCK = (
    f"{MARKER} opt-in GB10 sparse-MLA prefill on Triton\n"
    "# (VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1); module installed by patch 020. The gate is read at runtime;\n"
    "# unset, every hook below runs the upstream code verbatim. FAIL CLOSED with the gate set: an import\n"
    "# failure raises here (backend import = engine start-up) and no hook falls back to the upstream path.\n"
    "import os as _p020_os\n"
    "\n"
    "\n"
    "def _p020_triton_on() -> bool:\n"
    '    return _p020_os.environ.get("VLLM_GB10_SPARSE_MLA_TRITON_PREFILL", "0") == "1"\n'
    "\n"
    "\n"
    "try:\n"
    "    from vllm import _gb10_sparse_mla_prefill as _P020\n"
    "\n"
    "    _P020_IMPORT_ERROR = None\n"
    "except Exception as _p020_exc:  # tolerated only while the gate is unset\n"
    "    _P020 = None\n"
    "    _P020_IMPORT_ERROR = repr(_p020_exc)\n"
    "    if _p020_triton_on():\n"
    "        raise RuntimeError(\n"
    '            "[patch-020] VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1 but "\n'
    '            "vllm/_gb10_sparse_mla_prefill.py failed to import; refusing the upstream path"\n'
    "        ) from _p020_exc\n"
    "\n"
    "\n"
    "def _p020_module():\n"
    "    if _P020 is None:\n"
    "        raise RuntimeError(\n"
    '            "[patch-020] VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1 but "\n'
    '            f"vllm/_gb10_sparse_mla_prefill.py is not loaded ({_P020_IMPORT_ERROR})"\n'
    "        )\n"
    "    return _P020\n"
)

# (name, old, new) — applied in this order; each `old` must occur exactly once in the original.
EDITS = [
    # 1. module gate helper + fail-closed import (after the _PLAN_SLACK constant).
    ("module import", _IMPORT_ANCHOR, _IMPORT_ANCHOR + "\n" + IMPORT_BLOCK),
    # 2. builder.build(): decode-prefix-only plan through the module, then return (gate set).
    (
        "builder plan hook",
        "        # refreshed buffers.\n"
        "        num_rows, kv_lens, exact = self._kv_lens_host(common_attn_metadata)\n",
        "        # refreshed buffers.\n"
        f"        {MARKER} decode-prefix-only plan when prefill rows run on Triton (opt-in)\n"
        "        if _p020_triton_on():\n"
        "            _p020_module().plan_step(self, common_attn_metadata, metadata)\n"
        "            metadata.state = self.state\n"
        "            return metadata\n"
        "        num_rows, kv_lens, exact = self._kv_lens_host(common_attn_metadata)\n",
    ),
    # 3. impl.forward_mqa(): keep the valid_counts the convert kernel returns.
    (
        "valid_counts capture",
        "        topk_slots, _ = triton_convert_req_index_to_global_index(\n",
        f"        {MARKER} keep valid_counts: the Triton prefill kernel's loop bound\n"
        "        topk_slots, _p020_valid_counts = triton_convert_req_index_to_global_index(\n",
    ),
    # 4. impl.forward_mqa(): split decode/prefill when the gate is set (else the upstream path).
    (
        "forward split hook",
        "        state = attn_metadata.state\n"
        "        assert state is not None\n",
        "        state = attn_metadata.state\n"
        "        assert state is not None\n"
        f"        {MARKER} prefill rows -> Triton, decode rows -> FA2 (opt-in)\n"
        "        if _p020_triton_on():\n"
        "            out = _p020_module().forward_split(\n"
        "                self,\n"
        "                state,\n"
        "                q_nope,\n"
        "                q_pe,\n"
        "                topk_slots,\n"
        "                _p020_valid_counts,\n"
        "                kv_c_and_k_pe_cache,\n"
        "                layer,\n"
        "                attn_metadata,\n"
        "            )\n"
        "            return out, None\n",
    ),
    # 5. impl.__init__(): fail closed at start-up (config refusal + on-device kernel self-test).
    (
        "startup check hook",
        "        self.use_fp8_kv_cache = self.kv_cache_dtype in _FP8_KV_DTYPES\n",
        "        self.use_fp8_kv_cache = self.kv_cache_dtype in _FP8_KV_DTYPES\n"
        f"        {MARKER} fail closed at start-up when the gate is set\n"
        "        if _p020_triton_on():\n"
        "            _p020_module().startup_check(self, self.topk_indices_buffer.device)\n",
    ),
]


def apply(text: str) -> str:
    if MARKER in text:
        return text  # idempotent
    # Validate every anchor against the original before writing anything.
    bad = [(name, text.count(old)) for name, old, _ in EDITS if text.count(old) != 1]
    if bad:
        raise SystemExit(
            "[020] anchor check failed: "
            + ", ".join(f"{n!r} x{c}" for n, c in bad)
        )
    out = text
    for name, old, new in EDITS:
        if out.count(old) != 1:
            raise SystemExit(f"[020] anchor {name!r} changed by an earlier edit")
        out = out.replace(old, new, 1)
    if out.count(MARKER) != N_MARKERS:
        raise SystemExit(f"[020] expected {N_MARKERS} markers, found {out.count(MARKER)}")
    compile(out, "flashinfer_mla_sparse_sm90.py", "exec")  # in memory, no .pyc
    return out


def revert(text: str) -> str:
    # Each `new` fully contains `old` plus the marker, so new -> old removes the patch cleanly.
    out = text
    for name, old, new in EDITS:
        if new in out:
            out = out.replace(new, old, 1)
    if MARKER in out:
        raise SystemExit("[020] revert incomplete: marker still present")
    return out


def probe(sm_text: str, cuda_text: str | None = None) -> tuple[str, str]:
    """Read-only facts about the image the .sh prints a state line from. NEVER raises.

    cap  -- "yes"/"no"/"unknown": does FlashInferMLASparseSM90Backend.supports_compute_capability accept
            capability 12.1? The method's single `return` expression is evaluated as written (stock
            v0.32.0: `capability.major == 9`; SM121 bake: `capability.major in (9, 12)`).
    lst  -- "yes"/"late"/"no"/"unknown": does platforms/cuda.py's capability-12 backend priority list name
            FLASHINFER_MLA_SPARSE_SM90 BEFORE FLASHINFER_MLA_SPARSE_SM120 (or without it) = auto-selection on GB10
            picks the SM90 backend; "late" = listed after SM120 (vLLM takes the first valid backend by priority,
            so SM120 would win); "no" = not listed (explicit --attention-backend only).
    """
    import re
    import types

    cap = "unknown"
    try:
        tree = ast.parse(sm_text)
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FlashInferMLASparseSM90Backend"
        )
        fn = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "supports_compute_capability"
        )
        rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        assert len(rets) == 1 and rets[0].value is not None
        ok = eval(  # noqa: S307 - the vendored file's own one-line expression, nothing else in scope
            compile(ast.Expression(rets[0].value), "<cap>", "eval"),
            {"__builtins__": {}},
            {"capability": types.SimpleNamespace(major=12, minor=1)},
        )
        cap = "yes" if ok else "no"
    except Exception:  # noqa: BLE001
        pass
    lst = "unknown"
    try:
        if cuda_text is not None:
            m = re.search(r"device_capability\.major == 12:\n(.*?)\n        else:", cuda_text, re.S)
            if m:
                block = re.sub(r"#[^\n]*", "", m.group(1))  # comments may name backends
                i90, i120 = block.find("FLASHINFER_MLA_SPARSE_SM90"), block.find("FLASHINFER_MLA_SPARSE_SM120")
                if i90 < 0:
                    lst = "no"
                else:
                    lst = "yes" if (i120 < 0 or i90 < i120) else "late"
    except Exception:  # noqa: BLE001
        pass
    return cap, lst


def main(argv: list[str]) -> int:
    if len(argv) in (3, 4) and argv[1] == "probe":
        sm_text = Path(argv[2]).read_text() if Path(argv[2]).is_file() else ""
        cuda_text = None
        if len(argv) == 4 and Path(argv[3]).is_file():
            cuda_text = Path(argv[3]).read_text()
        print(" ".join(probe(sm_text, cuda_text)))
        return 0
    if len(argv) != 3 or argv[1] not in ("apply", "revert"):
        print("usage: p020_apply.py {apply|revert} <flashinfer_mla_sparse_sm90.py>\n"
              "       p020_apply.py probe <flashinfer_mla_sparse_sm90.py> [platforms/cuda.py]", file=sys.stderr)
        return 2
    mode, path = argv[1], Path(argv[2])
    src = path.read_text()
    out = apply(src) if mode == "apply" else revert(src)
    ast.parse(out)
    if out != src:
        tmp = path.with_suffix(path.suffix + ".p020.tmp")
        tmp.write_text(out)
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
