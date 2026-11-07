#!/usr/bin/env bash
# 014-gb10-skinny-indexer.sh — opt-in (VLLM_GB10_SKINNY_IDX=1) per-M kernels for the three narrow projections of the
# GLM-5.3-Flash DSA indexer at decode M.
#
# WHY: Indexer.forward projects hidden_states [M,4096] through wk_weights_proj [160,4096] (bf16), the fp32 head gate
# torch.mm(x.float(), _wp_fp32[4096,32]) and the kpool gate [128,4096] (bf16). At decode M cuBLAS runs them on 1-16 CTAs
# (the head gate is gemmSN_NN on ONE CTA): 132 us per call at M=3, 1.62 ms per k=2 decode step in a decode profile.
# a microbenchmark (not included) on node1 (DRAM-cold CUDA-graph timing, real weights): today 31.4 / 130.6 / 93.2 /
# 92.9 / 89.1 us per call at M = 1 / 3 / 4 / 8 / 16 -> 12.4 / 12.7 / ~21 / ~22 / ~24 us with the routes below.
# Projected ~1.43 ms per batch-1 k=2 step (12 calls at M=3 + 1 at M=1; ~3 % of the 46.5 ms step), ~0.8 ms per step at
# batch 8 (12 calls at M=16).
#
# WHAT: installs files/gb10_skinny_idx.py as vllm/_gb10_skinny_idx.py and makes 3 anchored edits (files/p014_apply.py) to
# vllm/models/glm5next/common/attention.py:
#   1. after `logger = init_logger(__name__)`: import the module when VLLM_GB10_SKINNY_IDX=1 (an ImportError aborts the boot)
#   2. Indexer.forward: the wk_weights_proj + head-gate block becomes
#        _p014 = _GB10_IDX is not None and _GB10_IDX.use(self, hidden_states)
#        if _p014: k, weights, gate_score = _GB10_IDX.project(self, hidden_states)
#        else: <the original 10 lines, unchanged>
#   3. `gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)` runs only when not _p014
# use() is True only for 1 <= M <= 16, bf16, contiguous, 16-byte aligned hidden_states [M,4096]; prefill and everything
# else keep the original code. Routes (per M): M<=3 in-tree CuTe skinny GEMM for wk_weights_proj and the gate + in-tree
# ll_bf16 dot-product for the head gate (on the W160[128:160] bf16 slice, no fp32 twin); M 4-8 today's cuBLAS for wk and
# the gate + ll_bf16 head gate; M 9-16 today's cuBLAS + the bench's Triton split-K head gate. Outputs keep today's
# contract: k = [:, :128] view of a [M,160] bf16 buffer, weights fp32 [M,32] contiguous, gate_score bf16 [M,128] contiguous.
#
# CUDA graphs (cudagraph_num_of_warmups = 0): the first call of each Indexer outside capture (vLLM's eager profile run
# reaches every target and MTP indexer) compiles every kernel variant for M = 1..16 once per process, then self-tests that
# layer against the original path at every M (bench rules: fp32 head gate within 2x today's rounding error of the exact
# result; bf16 outputs within 1 bf16 ulp of today's and of exact; deterministic; same layout) and raises on failure. A first
# call inside capture raises. Boot log: "[patch-014] skinny indexer ACTIVE" + the per-M table, one self-test line per layer.
#
# Numerics: head gate fp32 -> fp32 with a different summation order (products are exact); k / gate within 1 bf16 ulp at
# M<=3 (rare rounding flips), bit-identical at M>=4. Deterministic and identical on all ranks. NOT bit-exact against the
# upstream v0.32.0 path (the same statement holds against the day-0 path); gate with decode-kl, needles, BFCL like 011/012.
#
# vLLM v0.32.0 PORT (base image vllm/vllm-openai:v0.32.0-aarch64): the target moved from models/glm5next/nvidia/attention.py
# to models/glm5next/common/attention.py (upstream #56176); the three anchors match exactly once, with ONE changed line inside
# the carried block (`if self._wp_fp32 is None:` instead of `if getattr(self, "_wp_fp32", None) is None:`; v0.32.0 initialises
# `self._wp_fp32 = None` in Indexer.__init__). The module (CuTe DSL / Triton kernels, self-test, log lines) is unchanged; its
# dependencies on v0.32.0's cute_dsl/ll_bf16.py and skinny_gemm.py were re-read (see the module docstring). The edit is now
# applied by files/p014_apply.py (statement-level AST checks of the carried block and of both guards) and the file is pinned by
# sha256 (v0.32.0 original / patch-014 result). Order: no other patch of the v0.32.0 set (001 003 011 012 014 019 020B 022 025
# 029 031' 032 033) edits attention.py (the day-0 chain's patch 006 is dropped), so there is exactly one base.
# Inert without VLLM_GB10_SKINNY_IDX=1 (exits before touching anything).
# Fail-closed (pinned applier / module / target shas, every anchor must match exactly once before anything is written; the
# result must compile), idempotent (the patched file's sha).
set -euo pipefail

if [ "${VLLM_GB10_SKINNY_IDX:-0}" != "1" ]; then
  echo "=== Patch 014: GB10 skinny indexer — skipped (VLLM_GB10_SKINNY_IDX=${VLLM_GB10_SKINNY_IDX:-unset}); files untouched ==="
  exit 0
fi

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
AT="$VLLM_ROOT/models/glm5next/common/attention.py"
APPLIER="$SRC_DIR/p014_apply.py"
MODULE_SRC="$SRC_DIR/gb10_skinny_idx.py"
MODULE_DST="$VLLM_ROOT/_gb10_skinny_idx.py"
MARKER="# [patch-014 gb10-skinny-indexer]"

# pins (sha256): attention.py of vllm/vllm-openai:v0.32.0-aarch64 and its patch-014 result; the vendored applier and module
AT_ORIG=e83364215dccc2871e47906a8a5286852bd6a922a6a322394ff8f27db80952e0
AT_P014=40a75d21fc76663e748a1cd1db94f07bf234b426f671f083dfa80a92d79a4847
APPLIER_SHA=9593e14ae158366772031353acb5e2aea30298cc8de22a13273934dfbabdfd9e
MOD_SHA=8afd9a5d4b19992bb8b675f7c73dab4a029b0d32202cd31f227dd44527942ce5

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[FAIL] [014] $*" >&2; exit 1; }

echo "=== Patch 014: GB10 skinny indexer, per-M kernels for the DSA indexer projections at M <= 16 (opt-in VLLM_GB10_SKINNY_IDX=1) ==="
[ -f "$AT" ] || { echo "[FAIL] $AT not found" >&2; exit 1; }
[ -f "$MODULE_SRC" ] || { echo "[FAIL] $MODULE_SRC not found" >&2; exit 1; }
[ -f "$APPLIER" ] || { echo "[FAIL] $APPLIER not found" >&2; exit 1; }
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p014_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MODULE_SRC")" = "$MOD_SHA" ] || fail "files/gb10_skinny_idx.py sha256 $(sha "$MODULE_SRC") != pinned $MOD_SHA"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MODULE_SRC" \
  || { echo "[FAIL] $MODULE_SRC does not parse" >&2; exit 1; }

now="$(sha "$AT")"
mode=""
if [ "$now" = "$AT_P014" ]; then
  mode=installed
elif [ "$now" = "$AT_ORIG" ]; then
  mode=apply
else
  if grep -qF "$MARKER" "$AT"; then
    fail "attention.py carries a patch-014 edit that is not this revision's pinned result (sha ${now:0:16}...); recreate the container from the image"
  fi
  fail "attention.py sha256 $now is neither the v0.32.0 original ($AT_ORIG) nor patch 014's result ($AT_P014); refusing (a different vLLM build? re-pin after reading the diff)"
fi

install -m 0644 "$MODULE_SRC" "$MODULE_DST"
[ "$(sha "$MODULE_DST")" = "$MOD_SHA" ] || fail "installed $MODULE_DST sha256 mismatch"
echo " [OK] installed $(basename "$MODULE_DST") (sha256 $(sha "$MODULE_DST" | cut -c1-12))"

if [ "$mode" = installed ]; then
  echo " [skip] attention.py already wired"
else
  python3 "$APPLIER" apply "$AT" || fail "p014_apply.py refused attention.py"
  got="$(sha "$AT")"
  [ "$got" = "$AT_P014" ] || fail "patched attention.py sha256 $got != pinned $AT_P014"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$AT")" = 3 ] || { echo "[FAIL] expected 3 markers in $AT" >&2; exit 1; }
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$AT" "$MODULE_DST" \
  || { echo "[FAIL] patched files do not parse" >&2; exit 1; }
echo "[OK] patch 014 in place (active only when VLLM_GB10_SKINNY_IDX=1; now: ${VLLM_GB10_SKINNY_IDX:-unset})"
