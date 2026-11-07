#!/usr/bin/env bash
# 019-gb10-mhc-fn-bf16.sh — v0.32.0 PORT (rewrite as a kernel variant inside the JIT-class / warmup
# registry). Opt-in (VLLM_GB10_MHC_FN_BF16=1), LOSSLESS: the fused mHC post + pre-norm decode kernel reads a cached bf16
# copy of hc_attn_fn / hc_ffn_fn instead of the model's fp32 Parameter.
#
# WHY: every Glm5NextDecoderLayer call of hc_fused_post_pre (common/model.py:571/597, 89 per target forward: layer-0 ffn +
# attn/ffn of layers 1-44) reaches torch.ops.vllm.mhc_fused_post_pre_tilelang (layers/mhc.py MHCFusedPostPreOp ->
# kernels/mhc/tilelang.py:814). For num_tokens <= 32 (tilelang_kernels.py mhc_fused_post_pre_split_config: every decode
# step; CUDA-graph sizes 1..32 of the recipe) mHC post + the pre-norm GEMM run as ONE TileLang kernel, mhc_fused_tilelang
# (tilelang_kernels.py:668, launched through the JIT class MhcFusedTileLangKernel: grid [M, 24/tile_n, n_splits], tile_n 2
# below 16 tokens else 6, n_splits 8, n_thr 128; then mhc_pre_big_fuse_with_norm), whose weight operand is the fp32
# Parameter fn [24, 16384] viewed [24, 4, 4096] (kernel annotation weight_t T.float32, still fp32 in v0.32.0): 1.5 MiB of
# fp32 per call. The checkpoint stores all 90 hc_*_fn as BF16 [24, 16384] (safetensors headers of
# orcarouter/GLM-5.3-Flash-Uncensored-FP8 @3cec42d6, checked 2026-09-30: 90/90 BF16, 0 subnormals/zeros/non-finite);
# common/model.py:457/464 declares them fp32 and default_weight_loader widens (param.data.copy_), so half of each fp32
# word is zero. A bf16 copy halves the bytes: kindlingai measured 11.9 -> 8.2 us per call on GB10 (their image) and
# p019_gpu_bitexact.py measured -0.21 ms (M=3) .. -0.50 ms (M=16) per decode step on ours (89 calls, 0.4-0.5 % of a step).
#
# WHAT: installs files/gb10_mhc_fn_bf16.py as vllm/_gb10_mhc_fn_bf16.py (the bf16-weight kernel mhc_fused_bf16w_tilelang,
# the kernel CLASS MhcFusedBf16wTileLangKernel(MhcFusedTileLangKernel) overriding only kernel() and warmup_inputs(), and the
# copy cache) and makes four anchored edits (files/p019_apply.py):
#   tilelang.py  :6    after the direct_register_custom_op import: _P019_ON = (VLLM_GB10_MHC_FN_BF16 == "1"), read once
#   tilelang.py  :893  mhc_fused_post_pre_tilelang, after fused_config: if _P019_ON, _p019_w = weight_for(fn) (the cached
#                      bf16 copy, made on the FIRST EAGER call of each weight at ANY num_tokens - the pre-capture profile
#                      run at 8,192 tokens, and the V2 runner's eager warm-up before each capture)
#   tilelang.py  :914  the fused launch: (kernel, weight) = (the bf16 class instance, the copy) when a copy exists, else the
#                      upstream instance and fn; the call, its arguments and its outputs are otherwise untouched
#   tilelang_kernels.py :1735  MhcFusedTileLangKernel.register_warmup: whoever registers the fp32 fused kernel for JIT warmup
#                      (the model does, under the runner's active registry) also registers the bf16 variant, so its two
#                      compile keys (tile_n 2 / 6, n_thr 128; n_splits is a runtime shape) are compiled in kernel_warmup,
#                      before CUDA-graph capture. Without the registry (enable_jit_warmup off) the first conversion's
#                      self-test compiles them in the profile run (also before capture).
# num_tokens > 32 (mhc_post + the pre-norm GEMM, where patches 011 / 022 act) and layer-0 mhc_pre_tilelang are untouched.
#
# EXACT (bit-identical outputs vs the v0.32.0 fp32 path, not "close"): (1) each fp32 fn element has zero low 16 bits, so
# fn.to(bf16) is exact; weight_for() checks fp32 -> bf16 -> fp32 bitwise for EVERY weight and raises otherwise; (2)
# mhc_fused_bf16w_tilelang is upstream's mhc_fused_tilelang with only the weight_t annotation changed to T.bfloat16
# (p019_cpu_check.py proves it by AST against the real v0.32.0 file; same @tilelang_jit pass configs, same PDL setting); in
# `acc += weight_t * new_r` TIR promotes the bf16 weight to fp32 (exact, the same promotion the kernel already does for
# x_in / residual_in), so every FMA gets the identical fp32 operands in the identical order, same reductions -> identical
# yp_out / rp_out / residual_out bits; mhc_pre_big_fuse_with_norm then sees identical inputs; (3) boot self-test: on the
# first converted weight both kernels run at M = 1, 7, 8, 15, 16, 24, 32 (both configs and their boundaries), as raw JIT
# functions with NaN-poisoned outputs AND through the production classes, and the process raises unless every output is
# bitwise equal, written, and the class path equals the raw launch. CUDA graphs: a weight first seen INSIDE a capture gets
# the upstream fp32 kernel in that graph (exact, no gain; warning); copies are never freed.
# Boot log: "[patch-019] mHC fn bf16 ACTIVE: ... self-test bit-exact ..." then, at the first capture,
# "[patch-019] 89 bf16 hc_*_fn copies (66.75 MiB) serve the fused mHC decode kernel (M <= 32) in CUDA graphs; ... first seen
# inside a capture: 0". (v0.32.0 wording: M <= 32 where the image said M <= 16 - upstream raised the fusion limit.)
#
# MEMORY: 768 KiB per weight x 89 fused weights = 66.75 MiB per rank (the fp32 Parameters stay: the num_tokens > 32 pre-norm
# GEMM and layer-0 mhc_pre still read them). KV pool unaffected (--kv-cache-memory-bytes is pinned).
#
# PROVENANCE: kernel = port of vLLM mhc_fused_tilelang (v0.32.0 tilelang_kernels.py, sha256 61a384a0..., Apache-2.0,
# Copyright contributors to the vLLM project; changes listed in the module header). Idea from
# github.com/kindlingai/glm-5.3-flash-gx10 @ 3ed8fea (#47, experimental/fixes/mhc_tilelang.py `_fn_bf16` +
# tilelang_kernels.py; Apache-2.0 vLLM-derived files); no code copied from it. See PATCHES.md.
#
# ORDER / STACKING: runs after 011, which edits tilelang_kernels.py (HcPrenormGemmTileLangKernel + a module constant), and
# before 022, which edits tilelang.py (_hc_prenorm_gemm_outputs + a module gate). All anchors are disjoint, so each file may
# be the v0.32.0 original or any earlier patch's result; each combination maps to one pinned result (tilelang_kernels.py:
# original / 011 / 019 / 011+019; tilelang.py: original / 019 / 022 / 019+022). jit_warmup_tilelang_helper.py,
# tilelang_utils/__init__.py and jit_warmup.py are read-only and pinned (the variant's class and kernel use their API).
# Fail-closed: pinned sources, pinned input/result sha256, anchors must match exactly once (checked before anything is
# written), results compiled in memory, written atomically, re-hashed; any failure or TERM/INT/HUP restores both files (and
# removes a module this run installed) and exits non-zero -> the entrypoint aborts the boot. Idempotent (sha). Inert when
# unset. Set the flag on EVERY rank (M_DOCKER_EXTRA); it is numerically invisible, so a per-rank mismatch only loses speed.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
TL="$VLLM_ROOT/model_executor/kernels/mhc/tilelang.py"
TK="$VLLM_ROOT/model_executor/kernels/mhc/tilelang_kernels.py"
HELPER="$VLLM_ROOT/model_executor/warmup/jit_warmup_tilelang_helper.py"
JITW="$VLLM_ROOT/model_executor/warmup/jit_warmup.py"
TLU="$VLLM_ROOT/tilelang_utils/__init__.py"
APPLIER="$SRC_DIR/p019_apply.py"
MODULE_SRC="$SRC_DIR/gb10_mhc_fn_bf16.py"
MODULE_DST="$VLLM_ROOT/_gb10_mhc_fn_bf16.py"
MARKER="# [patch-019 gb10-mhc-fn-bf16]"
MARKERS_TL=3
MARKERS_TK=1
# tilelang.py: vLLM v0.32.0 original, patch-022 result, and this patch's two results (019 and 019+022)
TL_ORIG=204e19209db5b19eb3dd25ea6ae10c84425dee02ef5bc896eca3f83da6149a8e
TL_P022=af5d8dac025be6175fee132f55188f35e4518227c20ce9529f2f47ececa47f77
TL_P019=05a8b792dc27fb01544e199a6e329a5092433387fa13e2216c1dc5ce2b64f146
TL_P019_P022=4bea7f8ebe797ce7b82909b5dae03d7988e59fb6b845edc7b02d64317cae88d6
# tilelang_kernels.py: vLLM v0.32.0 original, patch-011 result, and this patch's two results (019 and 011+019)
TK_ORIG=61a384a091344d8207a04b2b7c74d30a81626eec4c068cf1dd034699cbfc34b5
TK_P011=8e9d71a1f4c3db3a312191105d0f2b15d2d8713bfec6c9e8a2e5207777ed4e95
TK_P019=77f41a656775d6f3db928e5900c78f6f6e4f38b84b29c7ef59a48c9a31615125
TK_P011_P019=e80f60b8de9d5f82547f8eb92afc71e45e4d5406ff89812efd10c2835a0abd8c
# read-only dependencies (the variant class / kernel are written against these exact v0.32.0 files)
HELPER_ORIG=6a473790e0f80c1e209fc52fb90aa8549490d7018f2bb442915156bcb0afe4a3
JITW_ORIG=a5fade0712c9032e8ed5f61454e39782fc69035be8a4ab8ec4fca4dfa25acfd0
TLU_ORIG=69ca819bbf68aa4e0ffdf6aff2995a3b8cd91c10a92922c051a165c9c625e726
# vendored sources (this patch owns them)
APPLIER_SHA=0f064e36bbcde71397af4d40a2b68d913dd19ca4794bce243bae3bfd2caff19a
MODULE_SHA=3919494fb605695c9fef45d3a4bc7dcf79db38f99434d8c52bde5ace792cad2a

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[019] FATAL: $*" >&2; exit 1; }

echo "=== Patch 019: GB10 mHC fused decode kernel reads a bf16 copy of hc_*_fn (opt-in VLLM_GB10_MHC_FN_BF16; now: ${VLLM_GB10_MHC_FN_BF16:-unset}) ==="
if [ "${VLLM_GB10_MHC_FN_BF16:-0}" != "1" ]; then
  if grep -qF "$MARKER" "$TL" 2>/dev/null || grep -qF "$MARKER" "$TK" 2>/dev/null; then
    echo " [note] VLLM_GB10_MHC_FN_BF16 unset but tilelang.py / tilelang_kernels.py already carry patch 019 (reused container?);"
    echo "        their import-time gates (_P019_ON) keep the upstream fp32 kernel."
  fi
  echo "[OK] patch 019 skipped (VLLM_GB10_MHC_FN_BF16=${VLLM_GB10_MHC_FN_BF16:-unset}); files untouched"
  exit 0
fi

for f in "$TL" "$TK" "$HELPER" "$JITW" "$TLU" "$APPLIER" "$MODULE_SRC"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p019_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MODULE_SRC")" = "$MODULE_SHA" ] || fail "files/gb10_mhc_fn_bf16.py sha256 $(sha "$MODULE_SRC") != pinned $MODULE_SHA"
[ "$(sha "$HELPER")" = "$HELPER_ORIG" ] || fail "jit_warmup_tilelang_helper.py sha256 $(sha "$HELPER") != v0.32.0 original $HELPER_ORIG (the variant class uses its API); refusing"
[ "$(sha "$JITW")" = "$JITW_ORIG" ] || fail "jit_warmup.py sha256 $(sha "$JITW") != v0.32.0 original $JITW_ORIG (the variant class uses its API); refusing"
[ "$(sha "$TLU")" = "$TLU_ORIG" ] || fail "tilelang_utils/__init__.py sha256 $(sha "$TLU") != v0.32.0 original $TLU_ORIG (tilelang_jit / pass configs the bf16 kernel must share); refusing"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MODULE_SRC" || fail "$MODULE_SRC does not parse"

tl_now="$(sha "$TL")"; tk_now="$(sha "$TK")"
tl_mode=""; tl_expect=""; tl_base=""; tk_mode=""; tk_expect=""; tk_base=""
case "$tl_now" in
  "$TL_P019"|"$TL_P019_P022") tl_mode=installed ;;
  "$TL_ORIG") tl_mode=apply; tl_expect="$TL_P019"; tl_base="v0.32.0 original" ;;
  "$TL_P022") tl_mode=apply; tl_expect="$TL_P019_P022"; tl_base="patch-022 result" ;;
  *) fail "tilelang.py sha256 $tl_now is neither the v0.32.0 original, the patch-022 result nor a patch-019 result; refusing" ;;
esac
case "$tk_now" in
  "$TK_P019"|"$TK_P011_P019") tk_mode=installed ;;
  "$TK_ORIG") tk_mode=apply; tk_expect="$TK_P019"; tk_base="v0.32.0 original" ;;
  "$TK_P011") tk_mode=apply; tk_expect="$TK_P011_P019"; tk_base="patch-011 result" ;;
  *) fail "tilelang_kernels.py sha256 $tk_now is neither the v0.32.0 original, the patch-011 result nor a patch-019 result; refusing" ;;
esac
# a half-applied pair (one file patched, the other not) is refused: the pair is applied and rolled back together
if [ "$tl_mode" != "$tk_mode" ]; then
  fail "tilelang.py is '$tl_mode' but tilelang_kernels.py is '$tk_mode' (patch 019 edits both together); recreate the container from the image"
fi

had_module=0; [ -e "$MODULE_DST" ] && had_module=1
if [ "$had_module" = 1 ]; then cp -p "$MODULE_DST" "$MODULE_DST.p019.orig"; fi
if [ "$tl_mode" = apply ]; then cp -p "$TL" "$TL.p019.orig"; cp -p "$TK" "$TK.p019.orig"; fi
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    if [ "$tl_mode" = apply ]; then
      rm -f "$TL.p019.tmp" "$TK.p019.tmp"
      [ -e "$TL.p019.orig" ] && mv -f "$TL.p019.orig" "$TL"
      [ -e "$TK.p019.orig" ] && mv -f "$TK.p019.orig" "$TK"
    fi
    if [ "$had_module" = 1 ]; then [ -e "$MODULE_DST.p019.orig" ] && mv -f "$MODULE_DST.p019.orig" "$MODULE_DST"; else rm -f "$MODULE_DST"; fi
    echo "[019] tilelang.py, tilelang_kernels.py and vllm/_gb10_mhc_fn_bf16.py restored to their previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

install -m 0644 "$MODULE_SRC" "$MODULE_DST"
[ "$(sha "$MODULE_DST")" = "$MODULE_SHA" ] || fail "installed $MODULE_DST sha256 mismatch"
if [ "$tl_mode" = installed ]; then
  echo " [skip] tilelang.py and tilelang_kernels.py already carry patch 019 (sha ${tl_now:0:16}... / ${tk_now:0:16}...)"
else
  echo " [state] tilelang.py is the $tl_base (${tl_now:0:16}...), tilelang_kernels.py the $tk_base (${tk_now:0:16}...)"
  python3 "$APPLIER" apply tl "$TL" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p019_apply.py refused tilelang.py ($tl_base)"
  APPLIER_PID=""
  python3 "$APPLIER" apply tk "$TK" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p019_apply.py refused tilelang_kernels.py ($tk_base)"
  APPLIER_PID=""
  got_tl="$(sha "$TL")"; got_tk="$(sha "$TK")"
  [ "$got_tl" = "$tl_expect" ] || fail "patched tilelang.py sha256 $got_tl != expected $tl_expect ($tl_base + 019)"
  [ "$got_tk" = "$tk_expect" ] || fail "patched tilelang_kernels.py sha256 $got_tk != expected $tk_expect ($tk_base + 019)"
  echo " [OK] tilelang.py: $tl_base + 019 (sha256 ${got_tl:0:16}...); tilelang_kernels.py: $tk_base + 019 (sha256 ${got_tk:0:16}...)"
fi
echo " [OK] installed _gb10_mhc_fn_bf16.py (sha256 ${MODULE_SHA:0:12}...)"

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$TL")" = "$MARKERS_TL" ] || fail "expected $MARKERS_TL markers in $TL"
[ "$(grep -cF "$MARKER" "$TK")" = "$MARKERS_TK" ] || fail "expected $MARKERS_TK marker in $TK"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$TL" "$TK" "$MODULE_DST" \
  || fail "patched/installed files do not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$TL.p019.orig" "$TK.p019.orig" "$MODULE_DST.p019.orig"
echo "[OK] patch 019 in place (fused mHC decode kernel reads bf16 hc_*_fn copies; boot self-test logs '[patch-019] ... ACTIVE')"
