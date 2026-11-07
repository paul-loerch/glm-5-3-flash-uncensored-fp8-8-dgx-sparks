#!/usr/bin/env bash
# 031-gb10-idx-shard-gate.sh — v0.32.0 port of patch 031 (indexer prefill row-sharding with a DEPTH gate). Opt-in
# (VLLM_GB10_IDX_SHARD=1; VLLM_GB10_IDX_SHARD_MIN_CTX 4096..4194304, default 32768; VLLM_GB10_IDX_SHARD_MIN_ROWS 64..8192,
# default 256). Replaces the old vendored-module patch 031 (031-gb10-idx-prefill-shard.sh, vLLM 487ecf187): v0.32.0 contains the
# mechanism itself (vLLM #54951), so this patch only moves ITS GATE.
#
# WHY: the DSA indexer (11 target layers + the MTP layer; 32 heads x 128, kpool 4, top-512 pools) is replicated under TP: in
# prefill every rank scores every query row against all pooled keys (DeepGEMM fp8 MQA logits) and selects each row's top-512
# pools. That work grows with depth (the previous base's prefill fit t/N = 0.372 ms + 1.71e-7 ms x N puts it at ~32 % of a 1M prefill;
# the old patch measured +3 / +8 / +20 / +39 % cold prefill at 128k / 258k / 512k / 1M). Upstream v0.32.0 shards the
# prefill rows of a step across the TP group (balanced_prefill_row_shard in v1/attention/backends/mla/indexer.py: contiguous
# row counts balanced by indexer MQA work; PrefillRowShard.narrow / exchange_topk = TP all_gatherv in the GLM5Next indexer op),
# but only when ONE step holds >= 16,384 x TP rows -- at --max-num-batched-tokens 8192 that never happens. Upstream main
# (#61647) made the row part configurable (attention_config.indexer_tp_shard_min_rows, default 65,536 total rows) and has no
# depth part. This patch applies the old depth gate to upstream's decision.
#
# WHAT: ONE file, four anchored edits (files/p031_apply.py) to vllm/v1/attention/backends/mla/indexer.py:
#   A  after MIN_TP_SHARD_ROWS_PER_RANK: the gate block (env read once at import, validated; _p031_row_gate, _p031_announce,
#      _p031_note_engaged);
#   B  balanced_prefill_row_shard: `num_rows < MIN_TP_SHARD_ROWS_PER_RANK * tp_size` -> `not _p031_row_gate(...)`:
#      gate on  -> shard iff rows >= max(MIN_ROWS, tp) and the deepest prefill request of the step has context >= MIN_CTX;
#      unset/0  -> upstream's own decision, unchanged;
#   C  DeepseekV32IndexerMetadataBuilder.__init__: one boot line per rank,
#        "[patch-031] indexer prefill row-shard ACTIVE (rank r/8 ..."  or  "... INACTIVE (rank r/8): <why>"
#      (INACTIVE when upstream's tp_prefill_row_sharding_supported() is False: cudagraph mixed mode FULL, symmetric-memory NCCL,
#      batch-invariant mode, DCP/PCP ...);
#   D  the function's return: the FIRST sharded step of the process is logged once,
#        "[patch-031] first sharded indexer prefill step: N rows -> per-rank rows [...], deepest context C tokens".
# No module is installed, no kernel is touched, nothing is added to the forward path except upstream's own sharded branch.
#
# DIFFERENCES to the old patch 031 (all in upstream's mechanism, recorded for the A/B):
#   - split: upstream balances contiguous row counts by MQA work (first_key + row, per request), the old patch cut equal row pieces
#     starting 16 rows from the chunk start; per-row logits are independent either way (see EXACT);
#   - exchange: upstream gathers the EXPANDED token indices ([rows, 2048 + kpool - 1] int32 = ~8.2 KB per row; all_gatherv = 8
#     ncclBroadcasts), the old patch gathered the [rows, 512] pool ids (2 KB per row) and expanded on every rank: ~4x more bytes per
#     sharded call (~55 MB received per rank per call at 7,680 rows, ~5 ms on 100G; x12 indexer calls per chunk). First estimate from
#     the old patch's numbers (old gate 32,768: 64k +0 %, 128k +3.4 %, 258k +8 %, 512k +20 %): the break-even depth moves from ~50k to
#     ~90k, so 32k..64k may lose 0..2 % and 128k+ keeps most of the gain. MIN_CTX is a runtime knob (no re-pin): A/B 32768 vs
#     98304 on the 32k / 64k / 128k cold-prefill cells before settling the default;
#   - no boot self-test, no cross-rank fingerprint, no VERIFY mode: the sharded branch is upstream's tested code; the gate is a
#     pure function of CPU metadata that is identical on every rank (rank-0 scheduler output). The flags must still be set on EVERY
#     rank (M_DOCKER_EXTRA); a rank with a different value would skip the all_gatherv the others enter (hang until the 600 s execute
#     timeout) -- hence the .sh refuses typos and the recipe sets the variable once for all ranks.
#
# EXACT / NUMERICS: gate unset -> bit-for-bit upstream v0.32.0 (files/p031_cpu_check.py G1: 4,000 random inputs against the
# unpatched function). Gate on -> the sharded steps use upstream's path: per row the same kernels (DeepGEMM MQA logits, top-k
# per row, pool expansion) on the same data; logits are one-output-row-per-query-row, so a row's logits do not depend on which
# rank scores it (the old patch verified this bitwise at boot with pieces aligned to the CTA row group; upstream's split is NOT
# aligned to 16 rows -- the first GPU A/B must therefore compare pool sets / decode-KL with the gate on and off). The image's
# top-k kernel orders its 512 pools through shared-memory atomics, so the ORDER and, on exact ties at the 512th value, the chosen
# members already vary run to run and rank to rank on the replicated path: pool SETS identical,
# ties = equally valid exact top-512 selections. Decode, short prefills (context <= 2,048), every CUDA-graph path and the KV writes
# are untouched. Prefill only.
#
# ORDER / STACKING: indexer.py is edited by no other patch of this set (020B edits flashinfer_mla_sparse_sm90.py; 014 edits
# glm5next/common/attention.py). Read-only contracts pinned by sha256: model_executor/layers/sparse_attn_indexer.py
# (PrefillRowShard, exchange_topk, _reserve_row_shard_exchange_transient) and models/glm5next/nvidia/sparse_indexer.py (the op
# that narrows chunks and calls exchange_topk; the SM121 bake edits it, both its pristine and its baked sha are accepted). A
# change of either must be re-validated and re-pinned here.
#
# Fail-closed install: pinned applier / base / result shas, anchors + AST placement validated before any write, result compiled
# in memory, written atomically, re-hashed; any failure or TERM/INT/HUP (the applier child is reaped first) restores the file and
# exits non-zero -> the entrypoint aborts. Idempotent (sha). A REUSED container carrying another revision's edit is refused
# (launch/up.sh always recreates). Inert unless VLLM_GB10_IDX_SHARD=1 (unset / 0 -> nothing is read or written; any other value
# -> refused, so a typo cannot silently disable it).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
IDX="$VLLM_ROOT/v1/attention/backends/mla/indexer.py"
OPFILE="$VLLM_ROOT/model_executor/layers/sparse_attn_indexer.py"
GLMIDX="$VLLM_ROOT/models/glm5next/nvidia/sparse_indexer.py"
APPLIER="$SRC_DIR/p031_apply.py"
MARKER="# [patch-031 gb10-idx-shard-gate]"
MARKERS=4
# indexer.py: the v0.32.0 original (vllm-project/vllm tag v0.32.0, commit 5df63a7203cd; the SM121 bake does not touch it) and
# this patch's result
IDX_ORIG=aabcb7f2e78b1136072edc87223b0d6e3bbd359b91dda5e1d0c86cba22f98f0e
IDX_P031=eb5860ce0c57e1c805ba0502a8626d7f819fa03c8e48a914b616b86dc442ad56
# read-only contracts: sparse_attn_indexer.py is the v0.32.0 original; models/glm5next/nvidia/sparse_indexer.py is the original
# OR the SM121 bake's result (image/glm53_sm121_v032.py groups topk_neg1_prefill + topk_neg1_decode turn the two
# `pool_topk = torch.empty(...)` into `torch.full(..., -1, ...)`; the built image carries the baked one, a `--skip topk_neg1`
# build the original). Neither bake edit touches PrefillRowShard.narrow / exchange_topk or the sharded call structure.
OP_ORIG=75d14da5e2b8c237f7fc0a807d2b72677660a945c70d23825da3517801d9ca10
GLMIDX_ORIG=cf659b6fdcfa1057ef6a61c3d31d302d2dd15dfea6f3c96743697664d87e3e44
GLMIDX_BAKED=3d97e02cb49a21402f3814cee2b4895af661fb1d5c4b5f7080c6b0bbc6db32a5
# vendored source (this patch owns it)
APPLIER_SHA=c5a3facf041152521f55d1de17c20879b6577582b6dad955b12c35a11834452c

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[031] FATAL: $*" >&2; exit 1; }

echo "=== Patch 031 (v0.32 port): GB10 indexer prefill row-shard depth gate (VLLM_GB10_IDX_SHARD=${VLLM_GB10_IDX_SHARD:-unset}, MIN_CTX=${VLLM_GB10_IDX_SHARD_MIN_CTX:-unset -> 32768}, MIN_ROWS=${VLLM_GB10_IDX_SHARD_MIN_ROWS:-unset -> 256}) ==="
gate="$(printf '%s' "${VLLM_GB10_IDX_SHARD:-0}" | tr -d '[:space:]')"
case "$gate" in
  ''|0)
    if grep -qF "$MARKER" "$IDX" 2>/dev/null; then
      echo " [note] VLLM_GB10_IDX_SHARD unset but indexer.py already carries patch 031 (reused container?);"
      echo "        its import-time gate keeps upstream's own row gate (16,384 x TP rows)."
    fi
    echo "[OK] patch 031 skipped (VLLM_GB10_IDX_SHARD unset); files untouched"
    exit 0 ;;
  1) ;;
  *) fail "VLLM_GB10_IDX_SHARD='${VLLM_GB10_IDX_SHARD}' must be unset/0 (inert) or 1" ;;
esac
# the module block raises on a bad value too; refusing here makes a typo visible before the boot
chk_int() {  # chk_int NAME VALUE LO HI
  local n=$1 v=$2 lo=$3 hi=$4
  v="$(printf '%s' "$v" | tr -d '[:space:]')"
  [ -z "$v" ] && return 0
  case "$v" in *[!0-9]*) fail "$n='$v' must be a plain non-negative integer";; esac
  { [ "${#v}" -le 9 ] && [ "$v" -ge "$lo" ] && [ "$v" -le "$hi" ]; } || fail "$n=$v outside [$lo, $hi]"
}
chk_int VLLM_GB10_IDX_SHARD_MIN_CTX "${VLLM_GB10_IDX_SHARD_MIN_CTX:-}" 4096 4194304
chk_int VLLM_GB10_IDX_SHARD_MIN_ROWS "${VLLM_GB10_IDX_SHARD_MIN_ROWS:-}" 64 8192

for f in "$IDX" "$OPFILE" "$GLMIDX" "$APPLIER"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p031_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$OPFILE")" = "$OP_ORIG" ] \
  || fail "sparse_attn_indexer.py sha256 $(sha "$OPFILE") != pinned $OP_ORIG: PrefillRowShard / exchange_topk (the code this gate turns on) changed; re-validate and re-pin"
glm_now="$(sha "$GLMIDX")"
case "$glm_now" in
  "$GLMIDX_BAKED"|"$GLMIDX_ORIG") ;;
  *) fail "models/glm5next/nvidia/sparse_indexer.py sha256 $glm_now is neither the v0.32.0 original ($GLMIDX_ORIG) nor the SM121-baked file ($GLMIDX_BAKED): the op that narrows chunks and calls exchange_topk changed; re-validate and re-pin" ;;
esac

idx_now="$(sha "$IDX")"
mode=""
case "$idx_now" in
  "$IDX_P031") mode=installed ;;
  "$IDX_ORIG") mode=apply ;;
  *)
    if grep -qF "$MARKER" "$IDX"; then
      fail "indexer.py carries a patch-031 edit that is not this revision's pinned result (sha ${idx_now:0:16}...); recreate the container from the image"
    fi
    fail "indexer.py sha256 $idx_now is not the pinned v0.32.0 original $IDX_ORIG (the image's file differs: re-validate the anchors and re-pin); refusing"
    ;;
esac

# Back up the target; restore it on any failure / signal.
cp -p "$IDX" "$IDX.p031.orig"
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    rm -f "$IDX.p031.tmp"
    mv -f "$IDX.p031.orig" "$IDX"
    echo "[031] indexer.py restored to its previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

if [ "$mode" = installed ]; then
  echo " [skip] indexer.py already carries patch 031 (sha ${idx_now:0:16}...)"
else
  echo " [state] indexer.py is the v0.32.0 original (${idx_now:0:16}...)"
  python3 "$APPLIER" apply "$IDX" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p031_apply.py refused indexer.py"
  APPLIER_PID=""
  got="$(sha "$IDX")"
  [ "$got" = "$IDX_P031" ] || fail "patched indexer.py sha256 $got != pinned $IDX_P031"
  echo " [OK] indexer.py: v0.32.0 original + 031 (sha256 ${got:0:16}...)"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$IDX")" = "$MARKERS" ] || fail "expected $MARKERS markers in $IDX"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$IDX" \
  || fail "patched indexer.py does not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$IDX.p031.orig"
echo "[OK] patch 031 in place (VLLM_GB10_IDX_SHARD=1: prefill steps whose deepest request has context >= ${VLLM_GB10_IDX_SHARD_MIN_CTX:-32768} tokens and >= ${VLLM_GB10_IDX_SHARD_MIN_ROWS:-256} rows run upstream's TP-sharded indexer prefill; boot logs '[patch-031] indexer prefill row-shard ACTIVE' or '... INACTIVE: why')"
