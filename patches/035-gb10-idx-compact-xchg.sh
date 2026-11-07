#!/usr/bin/env bash
# 035-gb10-idx-compact-xchg.sh — compact pool-id exchange for the TP-sharded indexer prefill (vLLM v0.32.0 image, GLM-5.3-Flash TP=8).
# Opt-in (VLLM_GB10_IDX_COMPACT_XCHG=1; VLLM_GB10_IDX_COMPACT_XCHG_VERIFY 0..1000, default 3, 0 = off). Restores the exchange design of
# the old patch 031 (all_gather the 512 POOL ids per row, expand on every rank) on top of upstream's PrefillRowShard, which 031
# (v0.32 port, VLLM_GB10_IDX_SHARD=1) switches on at depth. NO numerics change: the result is bitwise the original path's (EXACT below).
#
# WHY (measured 2026-10-23, same tool, same cells, cold, one user): the v0.32.0 port prefills 6-8 % faster than the old stack at
# 8k-32k but 6.5-7.7 % SLOWER at 128k-1M (1M TTFT 358 s vs 330 s) although upstream's TP-sharded indexer prefill is engaged (031's depth
# gate fires: context >= 32768 and >= 256 rows). In models/glm5next/nvidia/sparse_indexer.py the GLM5Next kpool path exchanges the
# EXPANDED token ids after the chunk loop: row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1) = [rows, 2051]
# int32 (8,204 B per row) through get_tp_group().all_gatherv (8 ncclBroadcasts), in 12 indexer calls (11 target layers + the MTP layer)
# per prefill chunk of 7,680 rows: ~63 MB per call. The old 031 exchanged select_k = topk_tokens // index_kpool = 512 pool ids per row
# (int32, 2,048 B per row, ~15.7 MB per call, 4.006x fewer bytes) and let every rank expand all rows locally.
#
# WHAT: ONE file, five anchored edits (files/p035_apply.py) to vllm/models/glm5next/nvidia/sparse_indexer.py as the SM121-baked image
# carries it (sha 3d97e02c..., = upstream cf659b6f... + the bake's topk_neg1 groups). Only when row_shard is not None AND index_kpool > 1
# AND positions is not None (the served layout) and the host assumptions below hold:
#   A  the helper block (env read once at import, the plan class, the VERIFY machinery) before `sparse_attn_indexer_kpool`;
#   B  before the chunk loop: pool_stage = torch.full((sum(sizes), select_k), -1, int32) on topk_indices_buffer's device (row 0 = global
#      row num_decode_tokens), inside the plan;
#   C  in the loop, after top_k_per_row_prefill and _merge_dcp_topk_global exactly as now: pool_topk is copied to pool_stage[row_start -
#      nd : row_end - nd] INSTEAD of being expanded and written into topk_indices_buffer (the -1 pool rows of a
#      local_total_seq_lens == 0 chunk, filled by topk_dst.fill_(-1), are staged too);
#   D  the original expansion block now writes to _p035_out (== topk_indices_buffer; the scratch buffer in a VERIFY call);
#   E  after the loop, the width-2051 exchange becomes: (a) all_gatherv of this rank's contiguous pool_stage slice [row_shard.start - nd,
#      row_shard.stop - nd) at width select_k (the same group call as PrefillRowShard.exchange_topk; the local tensor stays alive until
#      the expansion that consumes the gather has been enqueued, as upstream's comment requires), (b) ONE
#      kpool_ops.expand_pools_and_append_tail(gathered.to(torch.int64), positions[nd:end].to(torch.int32) + 1, index_kpool) over ALL
#      prefill rows, (c) topk_indices_buffer[nd:end, :expanded.shape[-1]] = expanded.
# Decode, short prefill (row_shard is None), the CUDA-graph paths (declined while capturing), the positions-is-None branch and
# index_kpool == 1 are never touched; with the gate unset the added statements are inert (files/p035_cpu_check.py F).
#
# EXACT (bitwise identical to the original path): _expand_pools_and_append_tail_kernel is one program per (row, column tile): row r of
# its output is a function of pool_ids[r, :] and seq_lens[r] ONLY, integer arithmetic (floor-div, mul, add, compares, where, int32
# store), no cross-row access (files/p035_cpu_check.py K7 checks this on the kernel source). The original path expands row r on the rank
# that owns it from its own pool_topk[r] with q_seq[r] = positions[r] + 1, then all_gatherv concatenates the per-rank expanded slices in
# rank order = global row order, which is a plain byte copy. The compact path all_gathers the SAME pool_topk rows (int32, a byte copy,
# rank order = global row order) and expands each of them with the SAME positions[r] + 1 through the SAME kernel. So
# expand(concat_r(P_r), q_seq) == concat_r(expand(P_r, q_seq_r)): row by row identical, and the buffer region written is the same
# [nd, end) x [0, topk_tokens + index_kpool - 1). Columns beyond and the rows outside stay -1 as before. Asserted on the host (cheap;
# every input is metadata or a shape that is identical on all TP ranks, so the decision is too), else the ORIGINAL exchange runs:
#   (1) end == nd + sum(row_shard.sizes), with end = num_tokens (the step's tokens, slot_mapping.shape[0]): the sharded rows are exactly
#       the prefill rows, so positions[nd:end] and the gathered rows line up;
#   (2) select_k * index_kpool == topk_tokens: pool expansion has no remainder;
#   (3) expanded width == topk_tokens + index_kpool - 1: implied by (2) (the kernel writes select_k * index_kpool + index_kpool - 1
#       columns) and asserted again on the result before anything is written -- the width of the original exchange;
#   (4) the chunks tile [nd, end) exactly: a row outside every chunk (the builder drops a chunk whose requests all have < index_kpool
#       tokens) keeps the original's all -1 row, while the compact path would expand its -1 pool row into a tail-only row;
#   (5) this rank's slice, positions and the top-k buffer cover [nd, end) x width, and no CUDA graph is being captured.
# The pool rows are the same values in both paths; in a VERIFY call the SAME pool_topk feeds both, so a racy top-k order (the image's
# kernel orders its 512 pools through shared-memory atomics) can not cause a false mismatch.
#
# VERIFY (default on, N = 3): the first N calls per process that take the compact path ALSO run the original path -- the loop's own
# expansion statements write into a separate scratch buffer [end, 2051], then upstream's row_shard.exchange_topk(scratch, 2051) --
# and compare rows [nd, end) with torch.equal. The 0/1 mismatch flag is reduced with ONE all_reduce(MAX) of a 1-element int32 CPU
# tensor on the TP group's gloo cpu_group (upstream does the same for a TP-uniform vote in moriio_connector.py), so EVERY rank takes the
# same decision; a per-rank decision would put the ranks into different collectives and hang them. A mismatch on any rank: every rank
# logs an ERROR, uses the ORIGINAL rows for this call and the compact path is permanently off in the process. VERIFY calls sync the
# host (prefill is eager, so this is free of graph constraints) and cost ~2x the exchange for 3 calls.
#
# LOG LINES (rank 0 unless noted; first sharded prefill step after the depth gate):
#   [patch-035] compact exchange ACTIVE (width 512 instead of 2051)
#   [patch-035] verify 1/3: bitwise equal (7680 rows, deepest context C)        (also 2/3, 3/3)
#   [patch-035] verify k/3: MISMATCH ... DISABLED                               (ERROR, EVERY rank, only on a failure)
#   [patch-035] compact exchange declined (<why>): the original exchange runs   (WARNING, once per reason; not for "no sharded step")
# No ACTIVE line in a run that engaged 031's shard (look for "[patch-031] first sharded indexer prefill step") means the compact path was
# declined: the warning says why.
#
# MEMORY: _reserve_row_shard_exchange_transient (profiling) reserves 2 x rows x buffer_width(2176) x 4 B = 17,408 B per row (142.6 MB at
# 8,192 rows) for the old exchange; it is left alone. The old exchange's transient is 9,229 B per row (its local slice + the gathered
# copy). The compact call's peak in finish() is pool_stage 2,048 + int64 pool ids 4,096 + expanded 8,204 = 14,348 B per row (117.5 MB at
# 8,192 rows: +42 MB over the old exchange, inside the reserve; the last chunk's logits, <= 64 MB, are alive in both). The 3 VERIFY calls
# peak at 25,637 B per row (scratch + expanded + the original exchange's local slice and gather = 210 MB at 8,192 rows), 67 MB ABOVE
# the reserve, once per process; VERIFY=0 removes that.
#
# ORDER / STACKING: sparse_indexer.py is edited by no other patch of this set; 031 only READS it as a contract (it accepts the original
# and the baked sha, so a REUSED container that already carries 035 is refused by 031 on the next boot; launch/up.sh always recreates).
# Read-only contracts pinned by sha256: model_executor/layers/sparse_attn_indexer.py (PrefillRowShard.start/stop/sizes/num_decode_tokens,
# exchange_topk, the narrowing the loop relies on) and models/glm5next/nvidia/ops/kpool_compress.py (the SM121-baked per-row expand
# kernel the exactness argument is about). A change of either must be re-validated and re-pinned here. Needs 031 (or upstream's own
# gate) to have a sharded step at all; harmless without.
# Set BOTH variables on EVERY rank (M_DOCKER_EXTRA): a rank on the original path would enter an all_gatherv of another width than its
# peers (hang until the 600 s execute timeout, or garbage); the .sh refuses typos and the recipe sets the variables once for all ranks.
#
# Fail-closed install: pinned applier / base / result shas, anchors + AST placement validated before any write, result compiled in
# memory, written atomically, re-hashed; any failure or TERM/INT/HUP (the applier child is reaped first) restores the file and exits
# non-zero -> the entrypoint aborts. Idempotent (sha). Inert unless VLLM_GB10_IDX_COMPACT_XCHG=1 (unset / 0 -> nothing is read or
# written; any other value -> refused, so a typo cannot silently disable it).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
TARGET="$VLLM_ROOT/models/glm5next/nvidia/sparse_indexer.py"
OPFILE="$VLLM_ROOT/model_executor/layers/sparse_attn_indexer.py"
KPOOL="$VLLM_ROOT/models/glm5next/nvidia/ops/kpool_compress.py"
APPLIER="$SRC_DIR/p035_apply.py"
MARKER="# [patch-035 gb10-idx-compact-xchg]"
MARKERS=5
# sparse_indexer.py: the file as the SM121-baked image carries it (upstream cf659b6f... + the bake's topk_neg1 groups; the pristine
# upstream file is NOT accepted -- it means the bake did not run) and this patch's result
TARGET_ORIG=3d97e02cb49a21402f3814cee2b4895af661fb1d5c4b5f7080c6b0bbc6db32a5
TARGET_PRISTINE=cf659b6fdcfa1057ef6a61c3d31d302d2dd15dfea6f3c96743697664d87e3e44
TARGET_P035=9b70e533b9a1ec89d724472bc0b29b0f92a89a6c4b4dccac11b7dac6de8930c9
# read-only contracts: sparse_attn_indexer.py is the v0.32.0 original (the bake does not touch it); kpool_compress.py is the SM121-baked
# file (the bake's kpool_pid group adds the `pid < pool_len` bound to _expand_pools_and_append_tail_kernel)
OP_ORIG=75d14da5e2b8c237f7fc0a807d2b72677660a945c70d23825da3517801d9ca10
KPOOL_BAKED=276d3df25fce804ad45f727fe2b0149db82fa327c1392ce1eea2bf6dafa47ec4
KPOOL_PRISTINE=e5d1baa2f8fac4e490cc2f22a5ca05f67d4dea03a420efdae05f3a2bbeb6616f
# vendored source (this patch owns it)
APPLIER_SHA=4cbbd68e0406838b0b983999fc07e29d300bbb91e93fe402a11bea5459c040fc

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[035] FATAL: $*" >&2; exit 1; }

echo "=== Patch 035: GB10 compact pool-id exchange for the TP-sharded indexer prefill (VLLM_GB10_IDX_COMPACT_XCHG=${VLLM_GB10_IDX_COMPACT_XCHG:-unset}, VERIFY=${VLLM_GB10_IDX_COMPACT_XCHG_VERIFY:-unset -> 3}) ==="
gate="$(printf '%s' "${VLLM_GB10_IDX_COMPACT_XCHG:-0}" | tr -d '[:space:]')"
case "$gate" in
  ''|0)
    if [ -f "$TARGET" ] && grep -qF "$MARKER" "$TARGET" 2>/dev/null; then
      echo " [note] VLLM_GB10_IDX_COMPACT_XCHG unset but sparse_indexer.py already carries patch 035 (reused container?);"
      echo "        its import-time gate keeps upstream's original exchange."
    fi
    echo "[state] patch 035 inert (VLLM_GB10_IDX_COMPACT_XCHG=${VLLM_GB10_IDX_COMPACT_XCHG:-unset}); files untouched"
    exit 0 ;;
  1) ;;
  *) fail "VLLM_GB10_IDX_COMPACT_XCHG='${VLLM_GB10_IDX_COMPACT_XCHG}' must be unset/0 (inert) or 1" ;;
esac
# the module block raises on a bad value too; refusing here makes a typo visible before the boot
vfy="$(printf '%s' "${VLLM_GB10_IDX_COMPACT_XCHG_VERIFY:-}" | tr -d '[:space:]')"
if [ -n "$vfy" ]; then
  case "$vfy" in *[!0-9]*) fail "VLLM_GB10_IDX_COMPACT_XCHG_VERIFY='$vfy' must be a plain non-negative integer (0 = off)";; esac
  { [ "${#vfy}" -le 4 ] && [ "$vfy" -le 1000 ]; } || fail "VLLM_GB10_IDX_COMPACT_XCHG_VERIFY=$vfy outside [0, 1000]"
fi
if [ "$(printf '%s' "${VLLM_GB10_IDX_SHARD:-0}" | tr -d '[:space:]')" != 1 ]; then
  echo " [note] VLLM_GB10_IDX_SHARD is not 1: no sharded indexer prefill step happens (upstream's own gate needs 131,072 rows at TP=8), so this patch will log no ACTIVE line"
fi

for f in "$TARGET" "$OPFILE" "$KPOOL" "$APPLIER"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p035_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$OPFILE")" = "$OP_ORIG" ] \
  || fail "sparse_attn_indexer.py sha256 $(sha "$OPFILE") != pinned $OP_ORIG: PrefillRowShard / exchange_topk (the exchange this patch replaces and compares against) changed; re-validate and re-pin"
kp_now="$(sha "$KPOOL")"
[ "$kp_now" != "$KPOOL_PRISTINE" ] \
  || fail "kpool_compress.py is the PRISTINE upstream v0.32.0 file ($kp_now): the SM121 bake (kpool_pid group) did not run; this patch is pinned to the baked file ($KPOOL_BAKED); rebuild the image"
[ "$kp_now" = "$KPOOL_BAKED" ] \
  || fail "kpool_compress.py sha256 $kp_now != pinned $KPOOL_BAKED: _expand_pools_and_append_tail_kernel (the per-row kernel the exactness argument rests on) changed; re-validate and re-pin"

tgt_now="$(sha "$TARGET")"
mode=""
case "$tgt_now" in
  "$TARGET_P035") mode=installed ;;
  "$TARGET_ORIG") mode=apply ;;
  "$TARGET_PRISTINE")
    fail "sparse_indexer.py is the PRISTINE upstream v0.32.0 file ($tgt_now): the SM121 bake (topk_neg1 groups) did not run; this patch is pinned to the baked file ($TARGET_ORIG); rebuild the image" ;;
  *)
    if grep -qF "$MARKER" "$TARGET"; then
      fail "sparse_indexer.py carries a patch-035 edit that is not this revision's pinned result (sha ${tgt_now:0:16}...); recreate the container from the image"
    fi
    fail "sparse_indexer.py sha256 $tgt_now is not the pinned SM121-baked v0.32.0 file $TARGET_ORIG (the image's file differs: re-validate the anchors and re-pin); refusing"
    ;;
esac

# Back up the target; restore it on any failure / signal.
cp -p "$TARGET" "$TARGET.p035.orig"
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    rm -f "$TARGET.p035.tmp"
    mv -f "$TARGET.p035.orig" "$TARGET"
    echo "[035] sparse_indexer.py restored to its previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

if [ "$mode" = installed ]; then
  echo " [skip] sparse_indexer.py already carries patch 035 (sha ${tgt_now:0:16}...)"
else
  echo " [state] sparse_indexer.py is the SM121-baked v0.32.0 file (${tgt_now:0:16}...)"
  python3 "$APPLIER" apply "$TARGET" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p035_apply.py refused sparse_indexer.py"
  APPLIER_PID=""
  got="$(sha "$TARGET")"
  [ "$got" = "$TARGET_P035" ] || fail "patched sparse_indexer.py sha256 $got != pinned $TARGET_P035"
  echo " [OK] sparse_indexer.py: baked v0.32.0 + 035 (sha256 ${got:0:16}...)"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$TARGET")" = "$MARKERS" ] || fail "expected $MARKERS markers in $TARGET"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$TARGET" \
  || fail "patched sparse_indexer.py does not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$TARGET.p035.orig"
echo "[OK] patch 035 in place (VLLM_GB10_IDX_COMPACT_XCHG=1, VERIFY=${vfy:-3}: a TP-sharded indexer prefill step exchanges the 512 pool ids per row instead of the 2051 expanded ids and expands all rows once; rank 0 logs '[patch-035] compact exchange ACTIVE (width 512 instead of 2051)' at the first sharded step)"
