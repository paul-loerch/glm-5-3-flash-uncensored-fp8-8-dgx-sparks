#!/usr/bin/env bash
# 020-gb10-sparse-mla-triton-prefill.sh — REVISION 4: the vLLM v0.32.0 port
# (2026-10-23). One opt-in feature for the GLM-5.3-Flash sparse-MLA (NoPE, DSA kpool) attention path on
# GB10 (sm_121), where every prefill row of a long prompt runs the FlashInfer FA2 MQA wrapper (masked MHA
# needs SM100 — sparse_mla_attention._is_masked_mha_available is family-100 only —, and the dense-MHA
# prefill only fires while the prefill context is <= index_topk = 2,048 tokens).
#
# PART B  VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1  (CHANGES NUMERICS) — route prefill rows [nd, T)
#   (nd = num_decode_tokens) of the sparse MQA path to a Triton fused sparse-attention kernel; decode
#   rows [0, nd) keep the FA2 wrapper (planned for nd rows). Pure-decode steps (nd == T) route nothing to
#   Triton and run upstream's own pack_indices + wrapper.run (bit-identical to stock + CUDA-graph-safe);
#   pure-prefill builds (nd == 0) skip the FA2 plan (nothing reads it) and route everything to Triton;
#   mixed steps split (the decode rows get a different FA2 schedule than a T-row plan: an expected ~ulp
#   change). The kernel reads the E4M3 latent cache directly (fp8 -> fp16), dequantises K AND V with
#   k_scale (as FA2's ckv_scale), gives q an exact per-head power-of-two pre-scale, tiles heads (any head
#   count), clamps valid_counts to the index width, accumulates in fp32, rounds P to fp16. THE KERNEL IS
#   BYTE-IDENTICAL TO REVISION 3 (measured on GB10 2026-10-01 against the old FlashInfer: Triton 1.28e-3
#   mean-rel vs an fp64 reference, FA2 2.24e-3, Triton vs FA2 1.7e-3; prefill +13..18 % at 4k..64k); on
#   this image the boot self-test and files/p020_gpu_selftest.py --kernel/--fa2/--split are the acceptance.
#
# PART A (VLLM_GB10_SPARSE_MLA_NOSYNC) IS GONE: vLLM #58684 (v0.31.0) plans every build sync-free from the
#   host upper bound and clamps each FA2 work's kv_end on the device, for decode, prefill and mixed steps
#   alike; the env var is ignored (a note is printed when it is set). Patch 021 (the same idea, plus its
#   rank fingerprint / VERIFY machinery) is dropped for the same reason: remove VLLM_GB10_SPARSE_MLA_NOSYNC
#   / _SYNCFREE / _SYNCFREE_VERIFY from the recipe.
#
# WHAT: installs files/gb10_sparse_mla_prefill.py as vllm/_gb10_sparse_mla_prefill.py (kernel + planner +
# router + start-up check) and makes 5 anchored edits (files/p020_apply.py) to
#     vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py  (sha-pinned; NO other patch touches it):
#     1. module level: gate helper + import of _gb10_sparse_mla_prefill — FAIL CLOSED: with the gate set an
#        import failure raises (backend import = engine start-up); tolerated only with the gate unset;
#     2. FlashInferMLASparseSM90Builder.build(): plan_step(...) (decode-prefix-only plan, none when
#        nd == 0) and return when the gate is 1, else upstream's planning lines;
#     3. Impl.forward_mqa(): keep the valid_counts upstream discards;
#     4. Impl.forward_mqa(): forward_split(...) when the gate is 1, else upstream's pack + run;
#     5. Impl.__init__(): startup_check(...) when the gate is 1: refuses an unsupported config (bf16
#        cache, rope MLA, other dims) and runs an on-device kernel self-test that also compiles both
#        kernel variants before serving; logs one "[patch-020] ACTIVE" line per rank.
# The hooks read the env gate at RUNTIME, so a reused container with the edit applied but the env unset
# behaves exactly like stock vLLM.
#
# THIS BACKEND ONLY EXISTS ON GB10 BECAUSE OF THE SM121 IMAGE BAKE: stock v0.32.0 allows
# FLASHINFER_MLA_SPARSE_SM90 on capability 9 only (supports_compute_capability: major == 9, FA3) and lists
# only TRITON_MLA / FLASHINFER_MLA_SPARSE_SM120 for capability 12. The bake (image/glm53_sm121_v032.py, frozen) edits this
# very file (capability (9, 12), backend "fa2" off Hopper) and platforms/cuda.py (SM90 listed ahead of SM120).
# The anchors of this patch are independent of those edits. Accepted target shas (below): the FROZEN bake's file
# (6e97595e..., patched here), a spare re-pin slot (files/p020_repin.sh), and the PRISTINE file, which means
# "bake absent" and is NEVER patched: with the pristine file (or any file whose supports_compute_capability rejects
# capability 12.1) the backend cannot be selected on GB10, so this script prints one "[state] patch 020 inert" line
# and touches NOTHING. If another backend ends up selected at run time the hooks never execute (they live only in
# FlashInferMLASparseSM90{Builder,Impl}): no "[patch-020] ACTIVE" line appears in the rank logs.
#
# PROVENANCE (see PATCHES.md): the Triton kernel is a port of vLLM's ROCm-aiter sparse-MLA prefill kernel
# (vllm/v1/attention/ops/rocm_aiter_mla_sparse.py _sparse_attn_prefill_ragged_kernel; Apache-2.0,
# "Copyright contributors to the vLLM project"), changed to a valid_counts bound, an fp8 E4M3 latent
# gather with k_scale on K and V, an exact per-head power-of-two q pre-scale, int64 slot offsets, BLOCK_H 16
# head tiles / BLOCK_N 32; the fixed-order fp32 two-split + LSE merge follows vLLM PR #47629's idea
# (re-implemented). The licence-restricted kindlingai image/patches were NOT read for code.
#
# Fail-closed: pinned module/applier sha, sha-pinned target (the baked base or this patch's result
# only; any other patch-020 revision is refused — recreate the container), anchors validated before any
# write, result compiled in memory + written atomically + re-hashed; any failure restores the original
# file and the previous module (or removes the one this run installed), then exits 1 (entrypoint aborts the
# boot). Idempotent (sha/marker). Inert unless the gate is set. Set the flag on EVERY rank
# (M_DOCKER_EXTRA): a per-rank mismatch would desync attention numerics.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
SM="$VLLM_ROOT/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
CUDA_PLAT="$VLLM_ROOT/platforms/cuda.py"
APPLIER="$SRC_DIR/p020_apply.py"
MODULE_SRC="$SRC_DIR/gb10_sparse_mla_prefill.py"
MODULE_DST="$VLLM_ROOT/_gb10_sparse_mla_prefill.py"
MARKER="# [patch-020 gb10-sparse-mla-triton-prefill]"
MARKERS=5
# flashinfer_mla_sparse_sm90.py — vLLM v0.32.0 (5df63a7203cd). SM_STOCK = the pristine file: the SM121 bake is ABSENT,
# the SM90 backend cannot run on capability 12 without it, so a pristine file means "inert" (never patched).
# SM_BAKED = the FROZEN SM121 bake of this port set (image/glm53_sm121_v032.py group `attn`, ids sm90cap +
# sm90fa2: capability (9, 12) and FA2 off Hopper; pins-after-bake.txt: 6e97595e...), SM_BAKED_P020 = this patch's result
# on it. SM_REAL is a spare slot for a re-baked image file (fill both with files/p020_repin.sh; all zeros = unused).
SM_STOCK=480f2060f82313f32e149af00e8e988d01e37b95e96e45a696c07b25b7a32327
SM_BAKED=6e97595e2c3b2e6265cc3ea578a2d1654f6504adcdc2ba30f814d0505d72950e
SM_BAKED_P020=c6fb753706b5e36f3c250789a9660d1674e7ff975e94e59554f95b98df754d3f
SM_REAL=0000000000000000000000000000000000000000000000000000000000000000
SM_REAL_P020=0000000000000000000000000000000000000000000000000000000000000000
# vendored sources (this patch owns them)
APPLIER_SHA=ebc48ec64f672eca2f65032241f5b17c3a6d7687c171185784b1b8e867ed6b0b
MODULE_SHA=1f3c04f812f4f543f960eff7e084acc202466bfb77cf4e41e251beec23cd2a17

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[020] FATAL: $*" >&2; exit 1; }

_triton="${VLLM_GB10_SPARSE_MLA_TRITON_PREFILL:-unset}"
echo "=== Patch 020 rev 4 (vLLM v0.32.0): GB10 sparse-MLA Triton prefill (TRITON_PREFILL=$_triton) ==="
if [ "${VLLM_GB10_SPARSE_MLA_NOSYNC:-0}" = "1" ]; then
  echo " [note] VLLM_GB10_SPARSE_MLA_NOSYNC=1 is obsolete on vLLM >= 0.31: upstream plans every build sync-free (#58684); ignored, remove it from the recipe"
fi
if [ "${VLLM_GB10_SPARSE_MLA_SYNCFREE:-0}" = "1" ]; then
  echo " [note] VLLM_GB10_SPARSE_MLA_SYNCFREE=1 belonged to patch 021, which is dropped (upstream #58684 is the same plan); ignored, remove it from the recipe"
fi
if [ "${VLLM_GB10_SPARSE_MLA_TRITON_PREFILL:-0}" != "1" ]; then
  if grep -qF "$MARKER" "$SM" 2>/dev/null; then
    echo " [note] gate unset but flashinfer_mla_sparse_sm90.py already carries patch 020 (reused container?);"
    echo "        the runtime env check keeps the stock path (upstream plan + FA2 forward)."
  fi
  echo "[OK] patch 020 skipped (VLLM_GB10_SPARSE_MLA_TRITON_PREFILL unset); files untouched"
  exit 0
fi

for f in "$SM" "$APPLIER" "$MODULE_SRC"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p020_apply.py sha $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MODULE_SRC")" = "$MODULE_SHA" ] || fail "files/gb10_sparse_mla_prefill.py sha $(sha "$MODULE_SRC") != pinned $MODULE_SHA"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MODULE_SRC" || fail "$MODULE_SRC does not parse"

# Backend probe (read-only, stdlib only; never fails the boot): can this image's FLASHINFER_MLA_SPARSE_SM90
# backend be selected on capability 12.1? The class method is evaluated as written (stock: major == 9;
# SM121 bake: major in (9, 12)).
probe="$(python3 "$APPLIER" probe "$SM" "$CUDA_PLAT" 2>/dev/null || echo "unknown unknown")"
probe_cap="${probe%% *}"; probe_list="${probe##* }"
if [ "$probe_cap" = "no" ]; then
  echo "[state] patch 020 inert: this image's FlashInferMLASparseSM90Backend.supports_compute_capability rejects capability 12.1 (SM121 bake absent), so vLLM cannot select the SM90 sparse-MLA backend on GB10; nothing patched, the ranks will log no ACTIVE line for this patch"
  exit 0
fi
if [ "$probe_list" = "no" ]; then
  echo " [note] platforms/cuda.py's capability-12 backend list does not name FLASHINFER_MLA_SPARSE_SM90: it is selected only with an explicit --attention-backend FLASHINFER_MLA_SPARSE_SM90 (or once the bake lists it)"
elif [ "$probe_list" = "late" ]; then
  echo " [note] platforms/cuda.py lists FLASHINFER_MLA_SPARSE_SM90 AFTER FLASHINFER_MLA_SPARSE_SM120 for capability 12: vLLM takes the first valid backend by priority (SM120), so this patch stays inert unless --attention-backend FLASHINFER_MLA_SPARSE_SM90 is given or the bake reorders the list"
fi

sm_now="$(sha "$SM")"
mode=""; expect=""
case "$sm_now" in
  "$SM_BAKED_P020"|"$SM_REAL_P020") mode=installed ;;
  "$SM_BAKED") mode=apply; expect="$SM_BAKED_P020"; base="SM121-baked v0.32.0 (capability + FA2)" ;;
  "$SM_REAL") mode=apply; expect="$SM_REAL_P020"; base="re-pinned REAL baked image" ;;
  "$SM_STOCK")
    echo "[state] patch 020 inert: flashinfer_mla_sparse_sm90.py is the PRISTINE v0.32.0 file (SM121 bake absent; sha ${sm_now:0:16}...), the SM90 sparse-MLA backend cannot run on capability 12 without the bake; nothing patched, the ranks will log no ACTIVE line for this patch"
    exit 0
    ;;
  *)
    if grep -qF "$MARKER" "$SM"; then
      fail "flashinfer_mla_sparse_sm90.py carries a patch-020 edit that is not this revision's output (sha ${sm_now:0:16}...); recreate the container from the image"
    fi
    fail "flashinfer_mla_sparse_sm90.py sha $sm_now is neither the pinned SM121-baked v0.32.0 file nor the patch-020 result; refusing (re-pin with files/p020_repin.sh after validating the new bake)"
    ;;
esac

if [ "$mode" = installed ]; then
  echo " [skip] flashinfer_mla_sparse_sm90.py already patched (sha ${sm_now:0:16}...)"
  install -m 0644 "$MODULE_SRC" "$MODULE_DST"
  [ "$(sha "$MODULE_DST")" = "$MODULE_SHA" ] || fail "installed $MODULE_DST sha mismatch"
else
  echo " [state] target is the $base file (${sm_now:0:16}...)"
  had_module=0
  if [ -e "$MODULE_DST" ]; then had_module=1; cp -p "$MODULE_DST" "$MODULE_DST.p020.orig"; fi
  cp -p "$SM" "$SM.p020.orig"
  ROLLBACK=1
  rollback() {
    if [ "${ROLLBACK:-0}" = 1 ]; then
      mv -f "$SM.p020.orig" "$SM"; rm -f "$SM.p020.tmp"
      if [ "$had_module" = 1 ]; then mv -f "$MODULE_DST.p020.orig" "$MODULE_DST"; else rm -f "$MODULE_DST"; fi
      echo "[020] flashinfer_mla_sparse_sm90.py and _gb10_sparse_mla_prefill.py restored" >&2
    fi
  }
  trap rollback EXIT
  # a TERM/INT/HUP must not roll back while the applier still writes: stop and reap it first, then
  # exit (which runs the EXIT rollback)
  APPLIER_PID=""
  on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
  trap on_signal TERM INT HUP
  install -m 0644 "$MODULE_SRC" "$MODULE_DST"
  [ "$(sha "$MODULE_DST")" = "$MODULE_SHA" ] || fail "installed $MODULE_DST sha mismatch"
  python3 "$APPLIER" apply "$SM" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "applier refused flashinfer_mla_sparse_sm90.py"
  APPLIER_PID=""
  got="$(sha "$SM")"
  [ "$got" = "$expect" ] || fail "patched sha $got != expected $expect"
  ROLLBACK=0; trap - EXIT TERM INT HUP
  rm -f "$SM.p020.orig" "$MODULE_DST.p020.orig"
  echo " [OK] flashinfer_mla_sparse_sm90.py patched (sha ${got:0:16}...) + installed _gb10_sparse_mla_prefill.py"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$SM")" = "$MARKERS" ] || fail "expected $MARKERS markers in $SM"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$SM" "$MODULE_DST" \
  || fail "patched/installed files do not parse"
echo "[OK] patch 020 in place (TRITON_PREFILL: prefill rows on Triton, decode rows on FA2; sync-free planning is upstream's; start-up self-test armed, each rank logs one ACTIVE line)"
