#!/usr/bin/env bash
# 032-gb10-repl-embed.sh — replicated input embedding (from an 8-rank profile of the previous base);
# PORTED TO vLLM v0.32.0 (module revision 2). Opt-in (VLLM_GB10_REPL_EMBED=1).
# Bit-identical: the replicated lookup returns, for every int32 token id, the bits the unpatched vocab-parallel lookup +
# all-reduce returns (proven below, re-checked at boot).
#
# WHY: each decode step replays 3 CUDA graphs (target, MTP draft prefill, MTP draft decode). The FIRST NCCL kernel of each
# replay waits 0.5-0.9 ms on NCCL's per-plan host-callback node (a GPU -> CPU -> GPU round trip, longer after CPU idle):
# 511 / 915 / 621 us on the critical path + ~0.54 ms of slower first floors per step. In all three replays that first NCCL
# op is the vocab-parallel embedding all-reduce (5-14 us into the replay). With the embedding replicated it needs no
# all-reduce: the first NCCL op becomes the first layer's (244 / 742 / 742 us of compute later), so the host round trip
# runs under compute. Modelled -1.8 to -2.2 ms/step (+4.7-5.8 % decode; optimistic, only partly hidden in the
# target and draft-1 passes; -0.4 to -0.7 with 029's eh split, refused below), prefill -8.7 ms per 3,840-token chunk
# (the target's and the MTP's 31.5 MB embedding all-reduces). Cost: +1.03 GiB per rank ("Model loading took" 39.23 ->
# ~40.27 GiB on the day-0 image; the same +1.03 GiB applies to v0.32.0's own baseline: launch/readiness.txt must accept
# the larger value).
#
# v0.32.0 PORT: models/glm5next/nvidia/model.py moved to models/glm5next/common/model.py (both anchors byte-identical and
# unique there). v0.32.0's VocabParallelEmbedding.forward at TP > 1 is ONE fused kernel (ops.vocab_parallel_embedding)
# plus the all-reduce, so revision 2 of the module runs the REAL forward of the pinned vocab_parallel_embedding.py on a
# view object for BOTH the replicated lookup (reduce_results=False) and the boot's reference (this rank's shard, then the
# layer's own tensor_model_parallel_all_reduce); the replicated layer therefore takes exactly the branch/kernel the
# unpatched layer takes at TP=8. Pinned read-only contracts are re-derived for v0.32.0 (below); the new loader steps
# (maybe_retie_word_embeddings, update_param_tp_status, requires_device_loading, supports_pre_processed_weights) were
# read and do not interfere (module docstring + files/p032_cpu_check.py).
#
# WHAT: installs files/p032_gb10_repl_embed.py as vllm/_gb10_repl_embed.py and makes 2 anchored edits
# (files/p032_apply.py) to vllm/models/glm5next/common/model.py:
#   :102  after `logger = init_logger(__name__)`: `_p032 = None`; when VLLM_GB10_REPL_EMBED == "1" (read once, at import)
#         `from vllm import _gb10_repl_embed as _p032` (module missing -> ImportError: the boot aborts on that rank);
#   :725  Glm5NextModel.__init__: `_p032_cls = VocabParallelEmbedding if _p032 is None else
#         _p032.embedding_class(config)`, then `self.embed_tokens = _p032_cls(<the original three arguments>)`.
# embedding_class() returns the unpatched class (logged, no extra memory) outside the validated envelope (TP 8, PP 1, no
# LoRA, tie_word_embeddings False and not untied-by-checkpoint, vocab a multiple of 64 and of TP, bf16), else
# ReplicatedEmbedding: VocabParallelEmbedding built with disable_tp=True (tp_size 1, shard = the whole vocab, weight
# [154880, 4096] loaded by the upstream weight_loader's narrow(0, 0, V): the FULL table on every rank). Its forward is
# upstream's TP>1 lookup for the full-vocab indices WITHOUT the all-reduce (the fused kernel; zero rows for out-of-range ids).
# The MTP drafter needs no edit and mtp.py is NOT edited by this patch: v0.32.0's load_eagle_model ->
# maybe_share_target_embed (spec_decode/eagle/utils.py) hands it the TARGET's embed_tokens object (Glm5NextMTP never sets
# has_own_embed_tokens, so _should_share is True) and deletes its own vocab-parallel copy (common/mtp.py:136). The lm_head
# (ParallelLMHead, its own "lm_head.weight", tie_word_embeddings False) is another module: untouched.
#
# EXACT: for an in-range id the unpatched layer all-reduces W[t] (its owner rank's row: the fused kernel is a byte copy,
# -0.0 included) with seven all-zero rows (+0.0; `vec_t()` in csrc/vocab_parallel_embedding_kernels.cu). Patch 012's
# one-shot (fp32 acc +0.0, acc += x + 0.0 in rank order, <= 8 tokens) and NCCL's bf16 sum both return x exactly for every
# finite non-zero x and +0.0 for -0.0; out-of-range ids give +0.0 rows. So the unpatched output is W[t] with -0.0 turned
# into +0.0: the boot rewrites the replicated table's -0.0 elements (bit pattern 0x8000) as +0.0 (masked_fill_, nothing
# else written), after which the replicated lookup is BITWISE the unpatched one for every id. Subnormal flushing in a
# reduction (none expected) is not assumed away: the boot compares against the real all-reduce.
#
# BOOT SELF-TEST (the replicated layer's quant method, from process_weights_after_loading: once, after ALL weights, every
# rank, the loader's module order; FAIL CLOSED, hang-free): local preflight (envelope, layout, exactly one full-vocab
# load, table scan: -0.0 / subnormal / non-finite (refused) counts + an int64 digest) -> gloo exchange (refusal,
# fingerprint, content: a checkpoint copy that differs across nodes is caught here) -> clone this rank's image shard,
# canonicalise the zeros, then the replicated lookup vs the image computation of the same ids (the REAL
# VocabParallelEmbedding.forward on a view of the cloned shard + the image's tensor_model_parallel_all_reduce) BITWISE for
# all 154,880 ids (19 all-reduces of <= 8,192 tokens = 64 MiB, the largest runtime one) + probe calls of 1-8 (one-shot), 16,
# 48 and 3,840 tokens (boundary rows, -0.0 / subnormal rows, random and 7 out-of-range ids) + a CUDA graph of the
# replicated lookup replayed on 4 id sets; every planned all-reduce is issued on every rank whatever fails locally ->
# gloo verdicts -> ACTIVE iff every rank passed, else every rank turns the layer back into the unpatched
# VocabParallelEmbedding (class, TP shard indices, sizes, use_fused_embedding, the cloned uncanonicalised shard,
# UnquantizedEmbeddingMethod) and frees the full table.
# Boot log per rank: one "[patch-032] replicated input embedding ACTIVE (rank r/8; ...)" or one
# "[patch-032] replicated input embedding INACTIVE (rank r/8): <why>; ..." line.
#
# ORDER / STACKING: runs after 025, the only patch of the v0.32.0 set (001, 003, 011, 012, 014, 019, 020B, 022, 025, 029,
# 031', 032, 033) that edits model.py before it (enumerated in files/p032_cpu_check.py A3; 025's edits - module gate
# before class Glm5NextDecoderLayer, the decoder-layer forward, the Glm5NextModel.forward layer loop - overlap neither
# anchor). The bases are the v0.32.0 original and the 025 result; each maps to one pinned 032 result (the 025 pair and
# MTP_P029 are recomputed by `python3 files/p032_cpu_check.py --vllm-root <v0.32.0 vllm> --print-pins`; the checks fail on
# a stale value). Patch 033 runs after this one and pins this patch's results as its own bases. Read-only contracts pinned
# by content:
# vocab_parallel_embedding.py (the class 032 subclasses and whose forward it runs), spec_decode/eagle/utils.py (the MTP
# shares the target's embed_tokens), mtp.py (v0.32.0 original or patch 029's result; its forward embeds via
# self.embed_tokens), model_loader/utils.py + base_loader.py (process_weights_after_loading runs once after
# load_weights, every module with a QuantizeMethodBase), model_loader/weight_tying.py (maybe_retie_word_embeddings is a
# no-op for this layer: the flag is False and the shapes differ).
# CONFLICT: 029's eh_proj row split (VLLM_GB10_DRAFT_COMPRESS=1 with the eh part, the default) adds a one-shot all-reduce
# ~50 us into each draft replay, which would become the drafts' first NCCL op again; refused here - use
# VLLM_GB10_DRAFT_COMPRESS_PARTS=head. A REUSED container that already carries 032 is refused by 025's .sh on the next
# boot (unknown model.py sha; launch/up.sh always recreates).
# Fail-closed install: pinned applier / module shas, pinned base AND result shas, anchors + AST placement validated before
# any write, result compiled in memory, written atomically, re-hashed; any failure or TERM/INT/HUP (the applier child is
# reaped first) restores model.py AND the previous module bytes (or removes the module this run installed) and exits
# non-zero -> the entrypoint aborts. Idempotent (sha). Inert unless VLLM_GB10_REPL_EMBED=1 (unset / 0 -> nothing is read
# or written beyond the gate; any other value -> refused, so a typo cannot silently disable it). Set the flag on EVERY rank
# (M_DOCKER_EXTRA): a rank without it never joins the boot exchange and the others fail the boot at the gloo timeout.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
MODEL="$VLLM_ROOT/models/glm5next/common/model.py"
VPE="$VLLM_ROOT/model_executor/layers/vocab_parallel_embedding.py"
EAGLE="$VLLM_ROOT/v1/worker/gpu/spec_decode/eagle/utils.py"
MTP="$VLLM_ROOT/models/glm5next/common/mtp.py"
LUTIL="$VLLM_ROOT/model_executor/model_loader/utils.py"
BLOAD="$VLLM_ROOT/model_executor/model_loader/base_loader.py"
WTIE="$VLLM_ROOT/model_executor/model_loader/weight_tying.py"
APPLIER="$SRC_DIR/p032_apply.py"
MOD_SRC="$SRC_DIR/p032_gb10_repl_embed.py"
MOD_DST="$VLLM_ROOT/_gb10_repl_embed.py"
MARKER="# [patch-032 gb10-repl-embed]"
MARKERS=2
# model.py: every base the entrypoint can hand over after 001-031 (the v0.32.0 original; 025 = opt-in mHC prefill sp) and
# each + 032. MODEL_P025 / MODEL_P025_P032 / MTP_P029 are derived from files/p025_apply.py / files/p029_apply.py: refresh
# with `python3 files/p032_cpu_check.py --vllm-root <v0.32.0 vllm> --write-pins` whenever one of those appliers changes (a
# non-hex value such as `pending-025-port` is not a sha, so that branch can never match = fail closed).
MODEL_ORIG=f337b984e25c8cb3d1546dbb4bf7ac8a5c057b19967adb0bb037c416b252b7bb
MODEL_P025=92ab069cd173dad39ded4c35c3c3fbb374a855cd472b3592f13f92eb481131fb
MODEL_ORIG_P032=3fc6088119997ad8961c93e60c11a90597d0eebab7a2f2de227b9b20c6e3c98d
MODEL_P025_P032=45cc5b89636b599e8a842aec23911d2fcd55b335c3bdd39a96d7eed3b46899ff
# read-only contracts (v0.32.0 originals; mtp.py may also be patch 029's result)
VPE_ORIG=10e17024bd5a5915e0d486e7a78a96d62e77bce803e1afd27feae7ad2042edc0
EAGLE_ORIG=21d1f84c1f5f4d95fe3eca85cb43da9e1517287fc8b32450d091e1670f5adef4
MTP_ORIG=55f33f9912c7fd127aaf914df21b060113e04b7d6d1686ab982e1d5b288b1c10
MTP_P029=ff508c4d52864e7c52379572c3461fa04638af41ebc98c8c8f28d309ba77e35a
LUTIL_ORIG=3091a9f64cbe42ad0e7fc0907fbbcb9197d2d56d5da4008779c5a16d9b8649a6
BLOAD_ORIG=cba765f6e669c51834ea5ac4146d5de9648635cfbc00c987601009865e869b36
WTIE_ORIG=8b950c56edb5f7586b522d13a045c415d391ce715e4590b16dbdf073e38a7af4
# vendored sources (this patch owns them)
APPLIER_SHA=2d823c1221566b38caeabc48d8a2e462c4ddf255b0551f6b1203e3af1bd23f69
MOD_SHA=74e6fe95a96cf70a80c5355fb00fde5c1a4e4045d3c0c48be12364e68a55338b

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[032] FATAL: $*" >&2; exit 1; }

echo "=== Patch 032: GB10 full-vocabulary input embedding per TP rank (VLLM_GB10_REPL_EMBED=${VLLM_GB10_REPL_EMBED:-unset}) ==="
case "${VLLM_GB10_REPL_EMBED:-0}" in
  ''|0)
    if grep -qF "$MARKER" "$MODEL" 2>/dev/null; then
      echo " [note] VLLM_GB10_REPL_EMBED unset but model.py already carries patch 032 (reused container?);"
      echo "        its import-time gate keeps the image's vocab-parallel embedding."
    fi
    echo "[OK] patch 032 skipped (VLLM_GB10_REPL_EMBED unset); files untouched"
    exit 0 ;;
  1) ;;
  *) fail "VLLM_GB10_REPL_EMBED='${VLLM_GB10_REPL_EMBED}' must be unset/0 (inert) or 1" ;;
esac
# 029's eh_proj row split would put an all-reduce at the front of both draft replays again
dc="$(printf '%s' "${VLLM_GB10_DRAFT_COMPRESS:-0}" | tr -d '[:space:]')"
if [ "$dc" = 1 ]; then
  parts="$(printf '%s' "${VLLM_GB10_DRAFT_COMPRESS_PARTS:-eh,head}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
  [ -n "$parts" ] || parts="eh,head"
  case ",$parts," in
    *,eh,*) fail "VLLM_GB10_REPL_EMBED=1 with patch 029's eh_proj split (VLLM_GB10_DRAFT_COMPRESS=1, parts '$parts'): its one-shot all-reduce would become the draft replays' first NCCL op and undo most of 032; set VLLM_GB10_DRAFT_COMPRESS_PARTS=head" ;;
  esac
fi

for f in "$MODEL" "$VPE" "$EAGLE" "$MTP" "$LUTIL" "$BLOAD" "$WTIE" "$APPLIER" "$MOD_SRC"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p032_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MOD_SRC")" = "$MOD_SHA" ] || fail "files/p032_gb10_repl_embed.py sha256 $(sha "$MOD_SRC") != pinned $MOD_SHA"
[ "$(sha "$VPE")" = "$VPE_ORIG" ] || fail "vocab_parallel_embedding.py sha256 $(sha "$VPE") != image original $VPE_ORIG (the class 032 subclasses and the TP>1 lookup whose forward it runs); refusing"
[ "$(sha "$EAGLE")" = "$EAGLE_ORIG" ] || fail "spec_decode/eagle/utils.py sha256 $(sha "$EAGLE") != image original $EAGLE_ORIG (the MTP's embed_tokens sharing contract); refusing"
mtp_now="$(sha "$MTP")"
[ "$mtp_now" = "$MTP_ORIG" ] || [ "$mtp_now" = "$MTP_P029" ] || fail "mtp.py sha256 $mtp_now is neither the image original nor patch 029's result (the draft embeds via self.embed_tokens); refusing"
[ "$(sha "$LUTIL")" = "$LUTIL_ORIG" ] || fail "model_loader/utils.py sha256 $(sha "$LUTIL") != image original $LUTIL_ORIG (process_weights_after_loading runs 032's boot self-test); refusing"
[ "$(sha "$BLOAD")" = "$BLOAD_ORIG" ] || fail "model_loader/base_loader.py sha256 $(sha "$BLOAD") != image original $BLOAD_ORIG (process_weights_after_loading after load_weights); refusing"
[ "$(sha "$WTIE")" = "$WTIE_ORIG" ] || fail "model_loader/weight_tying.py sha256 $(sha "$WTIE") != image original $WTIE_ORIG (maybe_retie_word_embeddings must stay a no-op for the replicated table); refusing"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MOD_SRC" || fail "$MOD_SRC does not parse"

m_now="$(sha "$MODEL")"
mode=""; expect=""; base=""
case "$m_now" in
  "$MODEL_ORIG_P032"|"$MODEL_P025_P032")
    mode=installed ;;
  "$MODEL_ORIG") mode=apply; expect="$MODEL_ORIG_P032"; base="image original" ;;
  "$MODEL_P025") mode=apply; expect="$MODEL_P025_P032"; base="patch-025 result" ;;
  *)
    if grep -qF "$MARKER" "$MODEL"; then
      fail "model.py carries a patch-032 edit that is not one of this revision's pinned results (sha ${m_now:0:16}...); recreate the container from the image"
    fi
    fail "model.py sha256 $m_now is none of the two bases 001-031 can hand over (v0.32.0 original, 025 result); a new model.py writer must be enumerated and pinned here (025 pin pair: $MODEL_P025); refusing"
    ;;
esac

# Back up the target and any pre-existing module bytes; restore all of them on any failure / signal.
had_mod=0; if [ -e "$MOD_DST" ]; then had_mod=1; cp -p "$MOD_DST" "$MOD_DST.p032.orig"; fi
cp -p "$MODEL" "$MODEL.p032.orig"
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    rm -f "$MODEL.p032.tmp"
    mv -f "$MODEL.p032.orig" "$MODEL"
    if [ "$had_mod" = 1 ]; then mv -f "$MOD_DST.p032.orig" "$MOD_DST"; else rm -f "$MOD_DST"; fi
    echo "[032] model.py and vllm/_gb10_repl_embed.py restored to their previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

install -m 0644 "$MOD_SRC" "$MOD_DST"
[ "$(sha "$MOD_DST")" = "$MOD_SHA" ] || fail "installed $MOD_DST sha256 mismatch"
if [ "$mode" = installed ]; then
  echo " [skip] model.py already carries patch 032 (sha ${m_now:0:16}...)"
else
  echo " [state] model.py is the $base (${m_now:0:16}...)"
  python3 "$APPLIER" apply "$MODEL" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p032_apply.py refused model.py ($base)"
  APPLIER_PID=""
  got="$(sha "$MODEL")"
  [ "$got" = "$expect" ] || fail "patched model.py sha256 $got != pinned $expect ($base + 032)"
  echo " [OK] model.py: $base + 032 (sha256 ${got:0:16}...)"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$MODEL")" = "$MARKERS" ] || fail "expected $MARKERS markers in $MODEL"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$MODEL" "$MOD_DST" \
  || fail "patched/installed files do not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$MODEL.p032.orig" "$MOD_DST.p032.orig"
echo "[OK] patch 032 in place (VLLM_GB10_REPL_EMBED=1, model.py sha256 $(sha "$MODEL" | cut -c1-16)...; the full-vocab table and its per-rank boot decision come at weight loading)"
