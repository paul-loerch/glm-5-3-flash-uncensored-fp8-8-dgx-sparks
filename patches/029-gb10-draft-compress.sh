#!/usr/bin/env bash
# 029-gb10-draft-compress.sh — draft-path compression; REVISION 3 = the vLLM v0.32.0
# port of REVISION 2.
#
# REVISION 3 (vLLM v0.32.0 port; the module's REVISION constant is 3; env gates and log-line texts are rev 2's, only the
# 'rev N' / 'module rev N' number in the [OK] and boot-log lines changed):
#   * targets moved: vllm/models/glm5next/common/mtp.py (was nvidia/mtp.py; the module-gate anchor is now the end of the
#     `from .model import (...)` block) and vllm/v1/worker/gpu/spec_decode/mtp/speculator.py (same path, same hook anchor).
#   * Marlin API drift handled in the MODULE (files/p029_draft_compress.py): prepare_fp8_layer_for_marlin no longer creates
#     layer.workspace, so the FP8 head allocates its own lock buffer at load and always passes it to apply_fp8_marlin_linear
#     (workspace=None would use the workspace manager's per-stream locks, allocated lazily - possibly inside a CUDA-graph
#     capture). The pinned read-only contracts are v0.32.0's marlin_utils_fp8.py / marlin_utils.py / logits_processor.py /
#     eagle/utils.py plus (new) the draft-sampling base spec_decode/speculator.py, whose sample_draft decides that the
#     hooked model.compute_logits is the served draft-logits path (draft_sample_method=probabilistic).
#   * served configuration: VLLM_GB10_DRAFT_COMPRESS=1 VLLM_GB10_DRAFT_COMPRESS_PARTS=head (with patch 032). The eh part is
#     carried unchanged from rev 2: it needs patch 012 (module pinned below), excludes patch 032, CPU-verified only.
#   * upstream main has speculative_config.draft_lm_head_quantization (#59973); v0.32.0 does not, so this patch is still
#     needed here and is not needed on a base that has that flag.
#   * ORDER / STACKING (new patch set 001, 003, 011, 012, 014, 019, 020B, 022, 025, 029, 031', 032, 033): no other patch edits
#     mtp.py or mtp/speculator.py, so each target has exactly one base (the v0.32.0 original); patch 032 and 033 read mtp.py
#     as a contract and accept this patch's result.
# Opt-in (VLLM_GB10_DRAFT_COMPRESS=1; VLLM_GB10_DRAFT_COMPRESS_PARTS = comma list of head / eh, default both;
# VLLM_GB10_DRAFT_COMPRESS_EH_GATE = graph (default) / numerics). DRAFT-ONLY numerics change: the served output
# distribution is unchanged (rejection sampling with the draft probabilities the drafts were sampled from); only the
# MTP acceptance rate can move.
#
# REVISION 2 (module only: the applier, both targets and their pinned results are rev 1's, byte for byte):
#   * E5 (the eh part's speed gate) is timed inside CUDA graphs captured at boot on a side stream: per M = 1, 3,
#     G_split = [one-shot] + 8 x [eh row block -> pad -> one-shot] vs G_full = [one-shot] + 8 x [replicated eh_proj]
#     (the leading one-shot stands for the draft embedding's all-reduce that precedes eh_proj in every served draft
#     replay and absorbs NCCL's per-replay host-callback round trip in both graphs); a gloo capture verdict (nobody
#     replays unless every rank captured), 2 + 10 replays each in a fixed collective order, pass iff median(G_split)
#     <= 0.9 x median(G_full). Rev 1 refused the eh part on every rank because it timed it EAGERLY ("E5 eh M=3 split
#     faster: 0.437 vs 0.295 ms", launch-bound); a microbenchmark measured it inside graphs (291 us replicated vs 42 us
#     split + ~47 us one-shot). The eager timing stays in the log as information. VLLM_GB10_DRAFT_COMPRESS_EH_GATE=
#     numerics (operator fallback, if the boot-time capture ever misbehaves) gates the eh part on its numerics only.
#   * Patch 032 (VLLM_GB10_REPL_EMBED=1, replicated input embedding) removes the all-reduce that precedes eh_proj in
#     the draft replays on purpose: the eh one-shot would become their first NCCL op and pay the round trip 032 hides.
#     032's .sh refuses VLLM_GB10_DRAFT_COMPRESS=1 with the eh part; this .sh now refuses the same combination before
#     any write (whichever runs first), and the module refuses the eh part at runtime when the draft's embed_tokens
#     does not all-reduce over the TP group. With 032 use VLLM_GB10_DRAFT_COMPRESS_PARTS=head.
#   * No [OK] / ACTIVE line quotes another gate's text any more (rev 1's [OK] line quoted the module's ACTIVE /
#     INACTIVE texts, which a gate's negative grep counts); the module defangs every reason / exception text it logs
#     (every launch/readiness.txt fail and require rule, ACTIVE / INACTIVE / [OK] / [NNN]).
#   * Verification of rev 2 (before any boot): a FAILED boot-time capture no longer leaves its private pool registered
#     with torch's caching allocator (capture_end raises before unregistering an invalidated capture; vLLM's next
#     torch.cuda.empty_cache() would then assert and kill the boot instead of refusing eh); each graph must record
#     exactly the one-shots its replays are budgeted for; _safe() covers the 3 fail rules (and the require rules) it
#     missed; the [OK] line names the selected parts only.
#
# WHY: at batch 1 (k=2) each decode step runs two MTP draft passes. Per pass and rank they stream (a) the full BF16
# lm_head shard, 19,360 x 4,096 = 158.6 MB (load_eagle_model shares the TARGET's lm_head into the MTP layer), and
# (b) eh_proj, an nn.Linear(8,192 -> 4,096) replicated on every rank, 67.1 MB. Both are pure weight streaming at
# M = 1..4 rows. Expected: -1.0 to -1.2 ms per step (+2.6-3.1 % decode), acceptance within 1 %.
#
# WHAT: installs files/p029_draft_compress.py as vllm/_gb10_draft_compress.py and makes anchored edits
# (files/p029_apply.py) to two vLLM files that no other patch of the set edits:
#   vllm/models/glm5next/common/mtp.py (3 edits): module gate _P029_ON (read once at import; set -> `from vllm import
#     _gb10_draft_compress as _p029`, missing module -> ImportError: fail closed); Glm5NextMultiTokenPredictorLayer.
#     forward: eh_proj via _p029.eh_proj_forward; Glm5NextMultiTokenPredictor.compute_logits (the MTP draft logits;
#     the target's Glm5NextForCausalLM.compute_logits lives in model.py and is NOT touched): _p029.draft_logits.
#   vllm/v1/worker/gpu/spec_decode/mtp/speculator.py (1 edit): MTPSpeculator.load_draft_model, right after
#     load_eagle_model (the lm_head is shared by then): _p029.prepare(draft_model, ...) builds the FP8 head copy and
#     the eh_proj row split and runs the collective boot self-test.
#   (a) head: a draft-only FP8 e4m3 copy of this rank's lm_head shard (per vocab row, scale bf16(amax/448)), applied
#       with the image's Marlin W8A16 path (marlin_utils_fp8: prepare_fp8_layer_for_marlin + apply_fp8_marlin_linear,
#       measured on GB10 in this image at 220-229 GB/s for M = 1..16) through the real LogitsProcessor (its all-gather
#       and vocab slice unchanged). 79.4 MB instead of 158.6 MB per draft pass. Rows M <= 64, else the image head.
#   (b) eh: weight-ROW split: rank r keeps rows [512r, 512r+512) of eh_proj (8.4 MB, a view, no copy), computes its
#       [M, 512] output block with the full-K dot product, zero-pads it to [M, 4,096], and patch 012's one-shot
#       all-reduce (canonical rank-order fp32 sum) assembles [M, 4,096] on every rank = an EXACT gather (7 of 8
#       addends are zeros). M <= the one-shot limit (8 rows: both draft passes at batch 1, the one pass at k=1 up
#       to batch 4); larger M (a prompt chunk's draft pass, batch >= 5) keeps the replicated eh_proj. Needs
#       VLLM_GB10_ONESHOT_AR=1 (patch 012), else (b) stays off (fail closed, logged).
#
# EXACTNESS: gumbel_sample() samples the draft token FROM the logits compute_logits returns and stores the same bf16
# tensor (pre-temperature) in the draft-logits cache the rejection sampler reads, so the acceptance test uses exactly
# the distribution the draft was drawn from; patch 017 keeps the draft noise disjoint from the target's resample
# noise. The target model, its lm_head (only READ once at load), logits and verification are untouched.
#
# BOOT SELF-TEST (in prepare(), at model load, every rank, HANG-FREE, FAIL CLOSED = refuse + log, never raise):
# local H1-H5 on synthetic hidden states (Marlin vs exact dequant relRMSE <= 0.01; vs the image BF16 head relRMSE
# <= 0.05 and max|diff| within the worst-case FP8 bound; top-1 >= 0.5 on 128 random rows, 100 % on peaked rows;
# deterministic; FP8 <= 0.9 x BF16 time at M = 1), then a gloo preflight exchange (fingerprint + local verdicts ->
# plan), then the planned collectives (E1-E4: eh at M = 1/3/4/8 own block exact, within 2^-8 (|a|+|b|) + 2^-10 |x||W|
# of the replicated eh_proj, finite; E5: eager timing (information) + the CUDA-graph timing above with its capture
# exchange; G1-G4: the FP8 and BF16 heads through the real LogitsProcessor all-gather at M = 1/3/16, own slice exact,
# gathered relRMSE <= 0.05), a verdict exchange and a commit round. A part activates only if EVERY rank passed it and
# the gathered / assembled outputs are bit-identical across ranks; else the image path. One boot-log line per rank
# from the module (its ACTIVE or fail-closed form; module rev 3 at its end).
#
# ORDER / STACKING: no patch 001-028 of the v0.32.0 set edits mtp.py or spec_decode/mtp/speculator.py (enumerated in
# files/p029_cpu_check.py A5 and files/p029_sh_check.sh), so each target has exactly one base: the v0.32.0 original.
# Read-only dependencies pinned (their contracts are what 029 relies on): marlin_utils_fp8.py / marlin_utils.py (the
# FP8 Marlin prepare + apply), logits_processor.py (_apply_head -> lm_head.quant_method.apply, tp_size gather),
# eagle/utils.py (load_eagle_model shares the target lm_head into shared_head.head), spec_decode/speculator.py
# (sample_draft -> compute_draft_logits -> model.compute_logits), and patch 012's installed
# vllm/_gb10_oneshot_ar.py (the one-shot's should_use / max_elems / canonical sum). Fail-closed: pinned applier /
# module / target / result / dependency shas; anchors + AST placement validated before any write; results compiled
# in memory, written atomically, re-hashed; any failure or TERM/INT/HUP (the applier child is reaped first) restores
# BOTH targets and the previous module bytes (or removes the module this run installed) and exits non-zero -> the
# entrypoint aborts. Idempotent (sha + marker). Inert unless VLLM_GB10_DRAFT_COMPRESS=1. Set the SAME flags on EVERY
# rank (M_DOCKER_EXTRA): a fingerprint mismatch (module / parts / shapes) makes every rank refuse (image path); a rank
# WITHOUT the gate never joins the boot exchange, so the others give up at the bounded barrier (rank 0 after 600 s,
# the rest at gloo's own timeout) and stay on the image path too (slow boot, no wrong output).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
MTP="$VLLM_ROOT/models/glm5next/common/mtp.py"
SPEC="$VLLM_ROOT/v1/worker/gpu/spec_decode/mtp/speculator.py"
MU8="$VLLM_ROOT/model_executor/layers/quantization/utils/marlin_utils_fp8.py"
MU="$VLLM_ROOT/model_executor/layers/quantization/utils/marlin_utils.py"
LP="$VLLM_ROOT/model_executor/layers/logits_processor.py"
EAGLE="$VLLM_ROOT/v1/worker/gpu/spec_decode/eagle/utils.py"
DSPEC="$VLLM_ROOT/v1/worker/gpu/spec_decode/speculator.py"
OS_MOD="$VLLM_ROOT/_gb10_oneshot_ar.py"
APPLIER="$SRC_DIR/p029_apply.py"
MOD_SRC="$SRC_DIR/p029_draft_compress.py"
MOD_DST="$VLLM_ROOT/_gb10_draft_compress.py"
MARKER="# [patch-029 gb10-draft-compress]"
MTP_MARKERS=3
SPEC_MARKERS=1

# targets: v0.32.0 originals (vllm/vllm-openai:v0.32.0-aarch64, vLLM 5df63a7203cd; no other patch of the set edits them)
# + this patch's results
MTP_ORIG=55f33f9912c7fd127aaf914df21b060113e04b7d6d1686ab982e1d5b288b1c10
MTP_P029=ff508c4d52864e7c52379572c3461fa04638af41ebc98c8c8f28d309ba77e35a
SPEC_ORIG=b730010cad9d54772c03ca8c6e562192ce3a0faf18b511720c9ec7b0f0b9ec44
SPEC_P029=9f12057c31857fd4e695c88d94913b57717c6cc7ab27433930098b1a76b0e708
# read-only dependencies (v0.32.0 originals; no served patch edits them)
MU8_ORIG=a5cc60882dc32851b2cb7d573e2b28980319a99ef816fd6ed9925b6fabfffd9a
MU_ORIG=e60713c15080347b01425f90eaf03840ca80a1cd226ec57e526e8f6286eeeb22
LP_ORIG=35cf8235cb529ba2d37358fe9b9e5a29f40b5c3cbcc9e6750c8480c51bd7da83
EAGLE_ORIG=21d1f84c1f5f4d95fe3eca85cb43da9e1517287fc8b32450d091e1670f5adef4
DSPEC_ORIG=b25efadde6376b2168a2d5284f788d239079b0ef1480e68fafc25d557d44389a
# patch 012's module as installed by 012-gb10-oneshot-allreduce.sh (files/gb10_oneshot_ar.py)
OS_MOD_SHA=d5c5f5d6801d3fa02f518de37d3fa86ac6cfac98a305033b1ebc99b28817db37
# vendored sources (this patch owns them)
APPLIER_SHA=967eebd8b670b50946c463c92c95fc8b897a47a13e0fada6758606caef09e6b9
MOD_SHA=06ae4c18f6ae449ed88a7c438755bfec4d1ee682556a8206d7612a83af2625f9

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[029] FATAL: $*" >&2; exit 1; }

echo "=== Patch 029 rev 3: GB10 draft-path compression: FP8 draft lm_head + eh_proj row split (VLLM_GB10_DRAFT_COMPRESS=${VLLM_GB10_DRAFT_COMPRESS:-unset}, PARTS=${VLLM_GB10_DRAFT_COMPRESS_PARTS:-unset -> eh,head}, EH_GATE=${VLLM_GB10_DRAFT_COMPRESS_EH_GATE:-unset -> graph}) ==="
gate="$(printf '%s' "${VLLM_GB10_DRAFT_COMPRESS:-0}" | tr -d '[:space:]')"
case "$gate" in
  ''|0)
    if grep -qF "$MARKER" "$MTP" 2>/dev/null || grep -qF "$MARKER" "$SPEC" 2>/dev/null; then
      echo " [note] VLLM_GB10_DRAFT_COMPRESS unset but mtp.py / mtp/speculator.py already carry patch 029 (reused"
      echo "        container?); their gates (_P029_ON, the load_draft_model check) keep the image draft path."
    fi
    echo "[OK] patch 029 skipped (VLLM_GB10_DRAFT_COMPRESS=${VLLM_GB10_DRAFT_COMPRESS:-unset}); files untouched"
    exit 0 ;;
  1) ;;
  *) fail "VLLM_GB10_DRAFT_COMPRESS='${VLLM_GB10_DRAFT_COMPRESS}' must be unset/0 (inert) or 1" ;;
esac
PARTS="$(printf '%s' "${VLLM_GB10_DRAFT_COMPRESS_PARTS:-eh,head}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
[ -n "$PARTS" ] || PARTS="eh,head"
for p in $(printf '%s' "$PARTS" | tr ',' ' '); do
  case "$p" in head|eh) ;; *) fail "VLLM_GB10_DRAFT_COMPRESS_PARTS='${VLLM_GB10_DRAFT_COMPRESS_PARTS}': a comma list of head / eh";; esac
done
case ",$PARTS," in *,,*) fail "VLLM_GB10_DRAFT_COMPRESS_PARTS='${VLLM_GB10_DRAFT_COMPRESS_PARTS}': empty element";; esac
EH_GATE="$(printf '%s' "${VLLM_GB10_DRAFT_COMPRESS_EH_GATE:-graph}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
[ -n "$EH_GATE" ] || EH_GATE=graph
case "$EH_GATE" in
  graph|numerics) ;;
  *) fail "VLLM_GB10_DRAFT_COMPRESS_EH_GATE='${VLLM_GB10_DRAFT_COMPRESS_EH_GATE}': graph (default: E5 timed inside CUDA graphs) or numerics (the eh part gated on its numerics only)" ;;
esac
# patch 032 (replicated input embedding) and the eh part exclude each other: with 032 the eh one-shot would become the
# first NCCL op of each draft replay (032's .sh refuses the same combination; whichever runs first stops the boot)
if [ "${VLLM_GB10_REPL_EMBED:-0}" = 1 ]; then
  case ",$PARTS," in
    *,eh,*) fail "VLLM_GB10_REPL_EMBED=1 (patch 032, replicated input embedding) with patch 029's eh_proj split (parts '$PARTS'): the split's one-shot would become the draft replays' first NCCL op and undo 032; set VLLM_GB10_DRAFT_COMPRESS_PARTS=head" ;;
  esac
  echo " [note] VLLM_GB10_REPL_EMBED=1 (patch 032): parts=$PARTS (no eh split), as 032 requires"
fi

for f in "$MTP" "$SPEC" "$MU8" "$MU" "$LP" "$EAGLE" "$DSPEC" "$APPLIER" "$MOD_SRC"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p029_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MOD_SRC")" = "$MOD_SHA" ] || fail "files/p029_draft_compress.py sha256 $(sha "$MOD_SRC") != pinned $MOD_SHA"
[ "$(sha "$MU8")" = "$MU8_ORIG" ] || fail "marlin_utils_fp8.py sha256 $(sha "$MU8") != image original $MU8_ORIG (the FP8 Marlin prepare/apply contract); refusing"
[ "$(sha "$MU")" = "$MU_ORIG" ] || fail "marlin_utils.py sha256 $(sha "$MU") != image original $MU_ORIG (Marlin padding / workspace contract); refusing"
[ "$(sha "$LP")" = "$LP_ORIG" ] || fail "logits_processor.py sha256 $(sha "$LP") != image original $LP_ORIG (lm_head.quant_method.apply + tp_size gather contract); refusing"
[ "$(sha "$EAGLE")" = "$EAGLE_ORIG" ] || fail "spec_decode/eagle/utils.py sha256 $(sha "$EAGLE") != image original $EAGLE_ORIG (the target lm_head sharing contract); refusing"
[ "$(sha "$DSPEC")" = "$DSPEC_ORIG" ] || fail "spec_decode/speculator.py sha256 $(sha "$DSPEC") != image original $DSPEC_ORIG (sample_draft -> compute_draft_logits -> model.compute_logits: the hooked draft-logits path); refusing"
if [ -e "$OS_MOD" ]; then
  [ "$(sha "$OS_MOD")" = "$OS_MOD_SHA" ] || fail "vllm/_gb10_oneshot_ar.py (patch 012) sha256 $(sha "$OS_MOD") != pinned $OS_MOD_SHA (the eh split calls its should_use / max_elems / canonical sum); refusing"
else
  echo " [note] vllm/_gb10_oneshot_ar.py (patch 012) is not installed: the eh part stays off at boot (fail closed)"
fi
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MOD_SRC" || fail "$MOD_SRC does not parse"

state_of() {  # file orig result -> installed | apply | <fail message>
  local f=$1 o=$2 r=$3 now; now="$(sha "$f")"
  if [ "$now" = "$r" ]; then echo installed
  elif [ "$now" = "$o" ]; then echo apply
  elif grep -qF "$MARKER" "$f"; then echo "stale"
  else echo "unknown"; fi
}
st_mtp="$(state_of "$MTP" "$MTP_ORIG" "$MTP_P029")"
st_spec="$(state_of "$SPEC" "$SPEC_ORIG" "$SPEC_P029")"
for pair in "mtp.py:$st_mtp" "mtp/speculator.py:$st_spec"; do
  case "${pair#*:}" in
    stale) fail "${pair%%:*} carries a patch-029 edit that is not this revision's pinned result; recreate the container from the image" ;;
    unknown) fail "${pair%%:*} is neither the image original nor the pinned patch-029 result (a new earlier patch editing it must be enumerated and pinned here); refusing" ;;
  esac
done

had_mod=0; if [ -e "$MOD_DST" ]; then had_mod=1; cp -p "$MOD_DST" "$MOD_DST.p029.orig"; fi
cp -p "$MTP" "$MTP.p029.orig"; cp -p "$SPEC" "$SPEC.p029.orig"
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    rm -f "$MTP.p029.tmp" "$SPEC.p029.tmp"
    mv -f "$MTP.p029.orig" "$MTP"; mv -f "$SPEC.p029.orig" "$SPEC"
    if [ "$had_mod" = 1 ]; then mv -f "$MOD_DST.p029.orig" "$MOD_DST"; else rm -f "$MOD_DST"; fi
    echo "[029] mtp.py, mtp/speculator.py and vllm/_gb10_draft_compress.py restored to their previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

install -m 0644 "$MOD_SRC" "$MOD_DST"
[ "$(sha "$MOD_DST")" = "$MOD_SHA" ] || fail "installed $MOD_DST sha256 mismatch"
apply_one() {  # kind file state expect label
  local kind=$1 f=$2 st=$3 expect=$4 label=$5 got
  if [ "$st" = installed ]; then
    echo " [skip] $label already carries patch 029 (sha $(sha "$f" | cut -c1-16)...)"
    return 0
  fi
  echo " [state] $label is the image original ($(sha "$f" | cut -c1-16)...)"
  python3 "$APPLIER" apply "$kind" "$f" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p029_apply.py refused $label"
  APPLIER_PID=""
  got="$(sha "$f")"
  [ "$got" = "$expect" ] || fail "patched $label sha256 $got != pinned $expect"
  echo " [OK] $label: image original + 029 (sha256 ${got:0:16}...)"
}
apply_one mtp "$MTP" "$st_mtp" "$MTP_P029" "mtp.py"
apply_one spec "$SPEC" "$st_spec" "$SPEC_P029" "mtp/speculator.py"

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$MTP")" = "$MTP_MARKERS" ] || fail "expected $MTP_MARKERS markers in $MTP"
[ "$(grep -cF "$MARKER" "$SPEC")" = "$SPEC_MARKERS" ] || fail "expected $SPEC_MARKERS markers in $SPEC"
[ "$(sha "$MTP")" = "$MTP_P029" ] && [ "$(sha "$SPEC")" = "$SPEC_P029" ] || fail "result shas changed after the apply"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$MTP" "$SPEC" "$MOD_DST" \
  || fail "patched/installed files do not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$MTP.p029.orig" "$SPEC.p029.orig" "$MOD_DST.p029.orig"
what="draft-only FP8 lm_head + eh_proj row split"
case ",$PARTS," in *,eh,*) ;; *) what="draft-only FP8 lm_head" ;; esac
case ",$PARTS," in *,head,*) ;; *) what="eh_proj row split" ;; esac
echo "[OK] patch 029 in place (rev 3; VLLM_GB10_DRAFT_COMPRESS=1, parts=$PARTS, eh gate $EH_GATE: $what; the module's boot self-test logs one verdict line per rank at model load)"
