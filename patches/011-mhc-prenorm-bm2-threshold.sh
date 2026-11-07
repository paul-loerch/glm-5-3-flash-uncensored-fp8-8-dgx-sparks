#!/usr/bin/env bash
# 011-mhc-prenorm-bm2-threshold.sh — v0.32.0 PORT (rewrite). Opt-in
# (VLLM_GB10_MHC_BM2_MIN_T=<tokens>, >= 128, e.g. 128) routing of the mHC pre-norm GEMM to the block_m=2 TileLang kernel
# from <tokens> instead of 1,024.
#
# WHY: for 128 <= T < 1,024 (agent-turn appends, chat prompts, prefill tail chunks) the dispatch falls through to the
# generic per-token kernel (grid [T, 2, 1], every CTA re-reads its 786 KB fp32 fn slice from L2): 33 GB/s on GB10.
# The block_m=2 kernel halves the fn re-reads. Microbenchmark on node1 (a microbenchmark not included here; DRAM-streamed x, the
# 487ecf187 kernels): T=128 132 -> 79 us, T=256 252 -> 120 us, T=512 509 -> 240 us; outputs BIT-IDENTICAL to the
# unrouted dispatch at every T tested (per token both kernels accumulate k = it*512 + tid in the same order, same warp and
# cross-warp reductions). 90 calls per forward -> ~25 ms per 512-token chunk. Larger block_m is slower on GB10 (register
# pressure), so only the threshold changes. T < 128 keeps its small-T kernel (n_thr 1024, tile_n 4); decode never
# reaches this path.
#
# v0.32.0: the routing moved. `_tilelang_hc_prenorm_gemm` (tilelang.py) no longer exists; since #50178 the pre-norm GEMM
# is the JIT class HcPrenormGemmTileLangKernel in kernels/mhc/tilelang_kernels.py, whose AST-traced dispatch() holds the
# `use_block_m = use_default_config and num_tokens >= 1024` rule and whose get_warmup_keys() / warmup_inputs() derive the
# warmed compile keys from the same 128 / 1024 buckets. The patch (files/p011_apply.py) makes four anchored edits to
# tilelang_kernels.py: a module constant _P011_BM2_MIN_T (env read once at import; the traced dispatch resolves module
# globals), the dispatch comparison, the warmup bucket lambda, and the block_m warmup tensor (max(1024, threshold)).
# Upstream keeps the T.cast(x, float32) of the v0.32.0 kernels; the arithmetic of the two kernels stays per-row and
# order-identical (re-checked on the generated structure by p011_cpu_check.py A5; the GPU check p011_gpu_check.py re-proves
# bit-identity of the two kernels on the real device for T = 128..1,023).
#
# NUMERICS vs v0.32.0: BIT-IDENTICAL by the argument above (the kernels differ only in grid mapping / register blocking);
# unset -> not even the code changes (inert, files untouched). With the routing on the warmup registry compiles the
# block_m=2 key from T = 128 (WarmupIntRange buckets 1, 128 [, threshold, ...]); the 128 <= T < threshold default-config
# key is warmed only for thresholds above 128.
# ORDER / STACKING: runs before 019, which edits the same file (an extra method in MhcFusedTileLangKernel); the anchors
# are disjoint and the two results commute (tilelang_kernels.py may be the v0.32.0 original or the 019 result; each maps
# to one pinned result). No other patch touches tilelang_kernels.py. Fail-closed: pinned input and result sha256, anchors
# must match exactly once (checked before anything is written), result compiled in memory, written atomically, re-hashed;
# any failure restores the file and exits 1 -> the entrypoint aborts the boot. Idempotent (sha). Inert when unset.
# Set the variable on EVERY rank (M_DOCKER_EXTRA); it is read once per process and routes only a bit-identical kernel
# choice, so a per-rank mismatch only changes speed (patch 022's rank fingerprint includes it and refuses a mismatch).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
TK="$VLLM_ROOT/model_executor/kernels/mhc/tilelang_kernels.py"
APPLIER="$SRC_DIR/p011_apply.py"
MARKER="# [patch-011 mhc-prenorm-bm2-threshold]"
MARKERS=4
# tilelang_kernels.py: vLLM v0.32.0 original, patch-019 result, and this patch's two results (the 011+019 pin is 019's too)
TK_ORIG=61a384a091344d8207a04b2b7c74d30a81626eec4c068cf1dd034699cbfc34b5
TK_P019=77f41a656775d6f3db928e5900c78f6f6e4f38b84b29c7ef59a48c9a31615125
TK_P011=8e9d71a1f4c3db3a312191105d0f2b15d2d8713bfec6c9e8a2e5207777ed4e95
TK_P011_P019=e80f60b8de9d5f82547f8eb92afc71e45e4d5406ff89812efd10c2835a0abd8c
# vendored source (this patch owns it)
APPLIER_SHA=a850885b634ab9789e88886bf44fc719d5158178711cc611328650a01e22c99c

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[011] FATAL: $*" >&2; exit 1; }

echo "=== Patch 011: mHC prenorm block_m=2 kernel from T >= ${VLLM_GB10_MHC_BM2_MIN_T:-unset} (opt-in VLLM_GB10_MHC_BM2_MIN_T) ==="
if [ -z "${VLLM_GB10_MHC_BM2_MIN_T:-}" ]; then
  if grep -qF "$MARKER" "$TK" 2>/dev/null; then
    echo " [note] VLLM_GB10_MHC_BM2_MIN_T unset but tilelang_kernels.py already carries patch 011 (reused container?);"
    echo "        its import-time constant reads the variable, so the upstream threshold 1,024 applies."
  fi
  echo "[OK] patch 011 skipped (VLLM_GB10_MHC_BM2_MIN_T unset); files untouched"
  exit 0
fi
case "$VLLM_GB10_MHC_BM2_MIN_T" in ''|*[!0-9]*) fail "VLLM_GB10_MHC_BM2_MIN_T must be an integer";; esac
{ [ "${#VLLM_GB10_MHC_BM2_MIN_T}" -le 8 ] && [ "$VLLM_GB10_MHC_BM2_MIN_T" -ge 128 ]; } \
  || fail "VLLM_GB10_MHC_BM2_MIN_T=$VLLM_GB10_MHC_BM2_MIN_T must be an integer >= 128 (below that the small-T kernel is faster)"

for f in "$TK" "$APPLIER"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p011_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"

tk_now="$(sha "$TK")"
mode=""; expect=""; base=""
case "$tk_now" in
  "$TK_P011"|"$TK_P011_P019") mode=installed ;;
  "$TK_ORIG") mode=apply; expect="$TK_P011"; base="v0.32.0 original" ;;
  "$TK_P019") mode=apply; expect="$TK_P011_P019"; base="patch-019 result" ;;
  *)
    if grep -qF "$MARKER" "$TK"; then
      fail "tilelang_kernels.py carries a patch-011 edit that is not one of this revision's pinned results (sha ${tk_now:0:16}...); recreate the container from the image"
    fi
    fail "tilelang_kernels.py sha256 $tk_now is neither the v0.32.0 original $TK_ORIG nor the patch-019 result; refusing"
    ;;
esac

if [ "$mode" = installed ]; then
  echo " [skip] tilelang_kernels.py already carries patch 011 (sha ${tk_now:0:16}...)"
else
  echo " [state] tilelang_kernels.py is the $base (${tk_now:0:16}...)"
  cp -p "$TK" "$TK.p011.orig"
  ROLLBACK=1
  rollback() {
    if [ "${ROLLBACK:-0}" = 1 ]; then
      rm -f "$TK.p011.tmp"
      mv -f "$TK.p011.orig" "$TK"
      echo "[011] tilelang_kernels.py restored ($base)" >&2
    fi
  }
  trap rollback EXIT
  APPLIER_PID=""
  on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
  trap on_signal TERM INT HUP
  python3 "$APPLIER" apply "$TK" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p011_apply.py refused tilelang_kernels.py ($base)"
  APPLIER_PID=""
  got="$(sha "$TK")"
  [ "$got" = "$expect" ] || fail "patched tilelang_kernels.py sha256 $got != pinned $expect ($base + 011)"
  ROLLBACK=0; trap - EXIT TERM INT HUP
  rm -f "$TK.p011.orig"
  echo " [OK] tilelang_kernels.py: $base + 011 (sha256 ${got:0:16}...)"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$TK")" = "$MARKERS" ] || fail "expected $MARKERS markers in $TK"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$TK" || fail "patched file does not parse"
echo "[OK] patch 011 in place (block_m=2 pre-norm kernel from T >= $VLLM_GB10_MHC_BM2_MIN_T; read at import)"
