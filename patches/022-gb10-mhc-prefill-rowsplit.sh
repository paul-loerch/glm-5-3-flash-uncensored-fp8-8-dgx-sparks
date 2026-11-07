#!/usr/bin/env bash
# 022-gb10-mhc-prefill-rowsplit.sh — mHC prefill row-split (mode 'prs');
# v0.32.0 PORT (rewrite: the hook moved from _tilelang_hc_prenorm_gemm to the JIT class path).
# Opt-in (VLLM_GB10_MHC_PRS=1), LOSSLESS (bit-identical outputs): during prefill the mHC pre-norm GEMM is split by
# token rows across the 8 TP ranks and its small per-token outputs are all-gathered over the TP group.
#
# WHY: every mHC block (layer-0 hc_pre + 89 fused hc_post/hc_pre per forward; common/model.py:562/571/597 ->
# kernels/mhc/tilelang.py:624/953) runs the pre-norm GEMM on the full residual x [T, 16384] bf16 on EVERY rank (mHC is
# replicated under TP). In prefill it is the slowest mHC kernel: hc_prenorm_gemm_block_m (block_m=2, patch 011 routes
# T >= 128 to it) 90 x 1.88 ms = 169 ms per 3,840-token chunk (a profile of the previous base; ~67 GB/s), ~10 %
# of a ~1.6 s chunk there. Split 8 ways it costs ~0.23 ms + one all-gather (S x 100 B per rank) per call.
#
# WHICH PATH v0.32.0 TAKES (confirmed against the source): every GLM5Next pre-norm GEMM goes through
# tilelang.py::_hc_prenorm_gemm_outputs (layer-0 mhc_pre_tilelang; the num_tokens > 32 branch of
# mhc_fused_post_pre_tilelang). It uses DeepGEMM's tf32_hc_prenorm_gemm iff vllm.utils.deep_gemm.is_deep_gemm_supported()
# = platform.support_deep_gemm() [capability 9.0 / family 10x / family 12x, so True on sm_121] AND envs.VLLM_USE_DEEP_GEMM
# AND has_deep_gemm(). The recipe sets VLLM_USE_DEEP_GEMM=0, so is_deep_gemm_supported() is False and the TileLang class
# instance _HC_PRENORM_GEMM_TILELANG_KERNEL (n_splits = 1) is the only path: that is where the hook sits. With
# VLLM_USE_DEEP_GEMM=1 (and a deep_gemm install) the DeepGEMM branch would run instead - a split-k tf32 kernel with other
# numerics - and 022 would be silently inert; this script warns about it (and 025 refuses it outright).
#
# WHAT: installs files/gb10_mhc_prs.py as vllm/_gb10_mhc_prs.py and makes 2 anchored edits (files/p022_apply.py) to
# vllm/model_executor/kernels/mhc/tilelang.py (v0.32.0 line numbers):
#   :22  before def _hc_prenorm_gemm_outputs: _P022_ON = (VLLM_GB10_MHC_PRS == "1"), read once at import; when set,
#        `from vllm import _gb10_mhc_prs as _p022` (module missing -> ImportError at import: fail closed);
#   :59  first statement of the TileLang branch: if _P022_ON and _p022.prenorm_rowsplit(_HC_PRENORM_GEMM_TILELANG_KERNEL,
#        x, fn, out, sqrsum, hidden_size, hc_mult): return out, sqrsum; else the upstream call, unchanged.
#   prenorm_rowsplit() returns False (upstream path) for T < VLLM_GB10_MHC_PRS_MIN_T (default 1,024), inside a CUDA-graph
#   capture (decode graphs are <= 48 tokens and never reach it anyway), for an unexpected layout, or when the class's own
#   dispatch() does not route this call to the block_m=2 kernel (so the 011 threshold is never duplicated). Otherwise rank r
#   runs that SAME JIT function (hc_prenorm_gemm_block_m_tilelang, taken from the class's compile key) on rows
#   [r*S, min((r+1)*S, T)), S = 2*ceil(ceil(T/2)/8) (shard starts even: every shard CTA is a CTA of the full call), and
#   one NCCL all-gather (pynccl, TP group, current stream) of [24 mixes | sqrsum] fp32 rebuilds out / sqrsum on every rank.
#   mhc_post, the all-reduces, big_fuse and every decode path are untouched. The shards call the JIT function directly with
#   the same positional arguments the class's launch uses, so they reuse the kernel the warmup registry compiled for the
#   block_m=2 key (TileLang's call cache is keyed on dtypes / static arguments; T is T.dynamic) - no new compile, no
#   warmup-registry entry. The delayed variants (DeepSeek-V4 shifted mHC) call the class directly and are not hooked.
#
# EXACT (bit-identical, not "close"): the bm2 kernel (v0.32.0 tilelang_kernels.py:926-1017; arithmetic identical to the
# image kernel 022 was proven on, except `x_val = T.cast(x[...], T.float32)` is now explicit [p022_cpu_check.py A5]) computes
# each (row, column) with per-thread fp32 accumulators over it = 0..31 in order, tl::warp_reduce_sum, then 16 warp partials
# summed in order from 0.0f; num_tokens is a runtime int used only in the row-validity predicate (one compiled kernel for
# every T). No arithmetic reads T, the row offset or another row; shards run the kernel object the full-T dispatch chose, at
# any shard size (neither patch 011's 128-row threshold nor the class's small-tile rule can move a shard to another
# kernel); the gather and the unpack are copies. Boot self-test (collective, the first eligible call = the 8,192-token
# profile run): the live call + synthetic inputs at T = MIN_T, MIN_T + 3, 3,840, 3,853 (live fn and a full-mantissa fp32 fn)
# must be bitwise equal to the unsplit kernel on every rank, repeat-stable, fully written, and the unsplit reference
# identical across ranks; a gloo rank fingerprint (module sha, gate, MIN_T, 011 threshold, TP size) must match; TP must
# be 8, PP 1, no sequence-parallel MoE. Any failure raises on EVERY rank -> the boot aborts.
# Boot log: "[patch-022] mHC prefill row-split ACTIVE (rank r/8, T >= 1024 ...): boot self-test bit-exact ...".
#
# ORDER / STACKING: runs after 019 (tilelang.py, other function + a module gate; anchors disjoint) and 011 (tilelang_kernels.py,
# read-only here): tilelang.py may be the v0.32.0 original or the 019 result, each mapped to one pinned 022 result;
# tilelang_kernels.py may be any of the original / 011 / 019 / 011+019 results (pinned set: the kernels are byte-identical in
# all four, the edits touch the dispatch, the warmup lambdas, a module constant and an extra method). 025 (sp) composes with
# this patch by neutralising it per forward (see 025). A REUSED container that already carries 019 / 022 results is
# accepted idempotently (every combination is pinned in 019's and 022's .sh); an unknown tilelang.py sha is refused
# (recreate the container; launch/up.sh always does). No other patch touches tilelang.py. Fail-closed: pinned applier/module shas, pinned target AND result shas (a marked file that is not a pinned
# result is refused), anchors + AST scope validated before any write, result compiled in memory, written atomically,
# re-hashed; any failure or TERM/INT/HUP (the applier child is reaped first) restores tilelang.py AND the previous module
# bytes (or removes the module this run installed) and exits non-zero -> the entrypoint aborts. Idempotent (sha). Inert
# unless VLLM_GB10_MHC_PRS=1. Set the flags on EVERY rank (M_DOCKER_EXTRA): the rank fingerprint catches a mismatch at boot.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
TL="$VLLM_ROOT/model_executor/kernels/mhc/tilelang.py"
TK="$VLLM_ROOT/model_executor/kernels/mhc/tilelang_kernels.py"
APPLIER="$SRC_DIR/p022_apply.py"
MOD_SRC="$SRC_DIR/gb10_mhc_prs.py"
MOD_DST="$VLLM_ROOT/_gb10_mhc_prs.py"
MARKER="# [patch-022 gb10-mhc-prefill-rowsplit]"
MARKERS=2
# tilelang.py bases: vLLM v0.32.0 original, patch-019 result, and this patch's two results
TL_ORIG=204e19209db5b19eb3dd25ea6ae10c84425dee02ef5bc896eca3f83da6149a8e
TL_P019=05a8b792dc27fb01544e199a6e329a5092433387fa13e2216c1dc5ce2b64f146
TL_ORIG_P022=af5d8dac025be6175fee132f55188f35e4518227c20ce9529f2f47ececa47f77
TL_P019_P022=4bea7f8ebe797ce7b82909b5dae03d7988e59fb6b845edc7b02d64317cae88d6
# tilelang_kernels.py (read-only dependency: hc_prenorm_gemm_block_m_tilelang, whose row independence 022 relies on, and the
# class dispatch it asks): the v0.32.0 original and the 011 / 019 / 011+019 results (== the pins in 011 / 019)
TK_KNOWN="61a384a091344d8207a04b2b7c74d30a81626eec4c068cf1dd034699cbfc34b5 \
8e9d71a1f4c3db3a312191105d0f2b15d2d8713bfec6c9e8a2e5207777ed4e95 \
77f41a656775d6f3db928e5900c78f6f6e4f38b84b29c7ef59a48c9a31615125 \
e80f60b8de9d5f82547f8eb92afc71e45e4d5406ff89812efd10c2835a0abd8c"
# vendored sources (this patch owns them)
APPLIER_SHA=17ccd2735ed1086c6bc4d13fbdae68ebeb887a56d11406af3c4bd262ae33bb23
MOD_SHA=084d90ec8c6ebe45cdd5fde2d2caaa45f644005a0a476afa6bb23cf0f4a510f7

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[022] FATAL: $*" >&2; exit 1; }

echo "=== Patch 022: GB10 mHC prefill row-split pre-norm GEMM (VLLM_GB10_MHC_PRS=${VLLM_GB10_MHC_PRS:-unset}, VLLM_GB10_MHC_PRS_MIN_T=${VLLM_GB10_MHC_PRS_MIN_T:-unset -> 1024}) ==="
if [ "${VLLM_GB10_MHC_PRS:-0}" != "1" ]; then
  if grep -qF "$MARKER" "$TL" 2>/dev/null; then
    echo " [note] VLLM_GB10_MHC_PRS unset but tilelang.py already carries patch 022 (reused container?);"
    echo "        its import-time gate (_P022_ON) keeps the upstream unsplit pre-norm GEMM."
  fi
  echo "[OK] patch 022 skipped (VLLM_GB10_MHC_PRS unset); files untouched"
  exit 0
fi
MIN_T="${VLLM_GB10_MHC_PRS_MIN_T:-1024}"
case "$MIN_T" in ''|*[!0-9]*) fail "VLLM_GB10_MHC_PRS_MIN_T='$MIN_T' must be an integer";; esac
{ [ "${#MIN_T}" -le 8 ] && [ "$MIN_T" -ge 256 ] && [ "$MIN_T" -le 1048576 ]; } \
  || fail "VLLM_GB10_MHC_PRS_MIN_T=$MIN_T outside [256, 1048576] (default 1024; below 256 the gather costs more than it saves)"
DG="$(printf '%s' "${VLLM_USE_DEEP_GEMM:-1}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$DG" in
  0|off|false|no) ;;
  *) echo " [warn] VLLM_USE_DEEP_GEMM='${VLLM_USE_DEEP_GEMM:-unset -> 1}': if vLLM then takes DeepGEMM's tf32_hc_prenorm_gemm (capability 12.x counts as supported when" \
          "deep_gemm is installed) the row-split hook, which sits in the TileLang branch, is never reached and 022 stays inert (no '[patch-022] ... ACTIVE' line). The recipe sets VLLM_USE_DEEP_GEMM=0." >&2 ;;
esac

for f in "$TL" "$TK" "$APPLIER" "$MOD_SRC"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p022_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MOD_SRC")" = "$MOD_SHA" ] || fail "files/gb10_mhc_prs.py sha256 $(sha "$MOD_SRC") != pinned $MOD_SHA"
tk_now="$(sha "$TK")"; tk_ok=0; for s in $TK_KNOWN; do [ "$tk_now" = "$s" ] && tk_ok=1 && break; done
[ "$tk_ok" = 1 ] || fail "tilelang_kernels.py sha256 $tk_now is not a known v0.32.0 variant (original or a 011 / 019 result) whose block_m=2 kernel and class dispatch 022's bit-identity argument covers; refusing"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MOD_SRC" || fail "$MOD_SRC does not parse"

tl_now="$(sha "$TL")"
mode=""; expect=""; base=""
case "$tl_now" in
  "$TL_ORIG_P022"|"$TL_P019_P022") mode=installed ;;
  "$TL_ORIG") mode=apply; expect="$TL_ORIG_P022"; base="v0.32.0 original" ;;
  "$TL_P019") mode=apply; expect="$TL_P019_P022"; base="patch-019 result" ;;
  *)
    if grep -qF "$MARKER" "$TL"; then
      fail "tilelang.py carries a patch-022 edit that is not one of this revision's pinned results (sha ${tl_now:0:16}...); recreate the container from the image"
    fi
    fail "tilelang.py sha256 $tl_now is neither the v0.32.0 original nor a pinned 019 result (a new 019 revision must be re-validated and re-pinned here); refusing"
    ;;
esac

# Back up the target and any pre-existing module bytes; restore all of them on any failure / signal.
had_mod=0; if [ -e "$MOD_DST" ]; then had_mod=1; cp -p "$MOD_DST" "$MOD_DST.p022.orig"; fi
cp -p "$TL" "$TL.p022.orig"
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    rm -f "$TL.p022.tmp"
    mv -f "$TL.p022.orig" "$TL"
    if [ "$had_mod" = 1 ]; then mv -f "$MOD_DST.p022.orig" "$MOD_DST"; else rm -f "$MOD_DST"; fi
    echo "[022] tilelang.py and vllm/_gb10_mhc_prs.py restored to their previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

install -m 0644 "$MOD_SRC" "$MOD_DST"
[ "$(sha "$MOD_DST")" = "$MOD_SHA" ] || fail "installed $MOD_DST sha256 mismatch"
if [ "$mode" = installed ]; then
  echo " [skip] tilelang.py already carries patch 022 (sha ${tl_now:0:16}...)"
else
  echo " [state] tilelang.py is the $base (${tl_now:0:16}...)"
  python3 "$APPLIER" apply "$TL" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p022_apply.py refused tilelang.py ($base)"
  APPLIER_PID=""
  got="$(sha "$TL")"
  [ "$got" = "$expect" ] || fail "patched tilelang.py sha256 $got != pinned $expect ($base + 022)"
  echo " [OK] tilelang.py: $base + 022 (sha256 ${got:0:16}...)"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$TL")" = "$MARKERS" ] || fail "expected $MARKERS markers in $TL"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$TL" "$MOD_DST" \
  || fail "patched/installed files do not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$TL.p022.orig" "$MOD_DST.p022.orig"
echo "[OK] patch 022 in place (VLLM_GB10_MHC_PRS=1, MIN_T=$MIN_T: mHC pre-norm GEMM row-split across the TP group in prefill; boot self-test logs '[patch-022] ... ACTIVE')"
