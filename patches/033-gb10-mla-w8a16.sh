#!/usr/bin/env bash
# 033-gb10-mla-w8a16.sh — "MLA projections as FP8 W8A16" (E3 below = the 2026-10-03 pre-integration microbenchmark).
# Opt-in (VLLM_GB10_MLA_W8A16=1; VLLM_GB10_MLA_W8A16_MAX_M = 1..64, default 16; VLLM_GB10_MLA_W8A16_SPEED_GATE = graph
# (default, enforced) / info). NUMERICS CHANGE, minimal: the checkpoint's own FP8 values with the block scale rounded to
# bf16 (<= 1 bf16 ulp) through a different GEMM (Marlin W8A16); E3 output relRMSE vs fp32 0.00346 vs 0.00247 for the BF16 path.
# NOT bit-exact against the upstream v0.32.0 path (which dequantises these projections to BF16 at load and runs BF16 GEMMs).
#
# vLLM v0.32.0 PORT (base image vllm/vllm-openai:v0.32.0-aarch64; module REVISION 2). What moved: the model file is now
# models/glm5next/common/model.py (the three anchors and the three hooks are byte-identical to the day-0 port; the old applier
# applies unchanged); mtp.py is models/glm5next/common/mtp.py (its loader still calls model.py's _try_load_fp8_attn_proj, which
# is the hook point; unchanged contract); and the Marlin helpers drifted (upstream #57421): prepare_fp8_layer_for_marlin() no
# longer attaches a workspace, apply_fp8_marlin_linear() takes workspace: Tensor | None and goes through the custom op
# torch.ops.vllm.marlin_gemm(b_q_type_id=...), gptq_marlin_repack lost `perm`. The module therefore allocates its own explicit
# lock buffer per projection (marlin_make_workspace_new(device, MARLIN_MAX_BLOCKS_PER_SM)) at weight-processing time and passes
# it, instead of relying on upstream's lazily allocated per-stream buffer (see the module header and files/p033_mla_w8a16.py).
# Read-only contracts are now pinned against the v0.32.0 tree (linear.py: UnquantizedLinearMethod now resolves its GEMM impl in
# __init__, forward paths unchanged; marlin_utils.py / marlin_utils_fp8.py / _custom_ops.py: the new signatures; model_loader/
# utils.py + base_loader.py: process_weights_after_loading runs once per load; eagle/utils.py). deepseek_v2.py (the fused q_a|kv_a
# module: byte-identical class to the day-0 image, its min-latency kernel needs shape (2112, 7168) so it never engages here),
# layers/mla.py (call pattern unchanged: fused_qkv_a_proj(x)[0], q_b_proj(q_c)[0], o_proj(attn_out)[0]) and parameter.py
# are covered by the boot self-test (V2 for the loaders' narrow rules, C8 for the module -> quant_method.apply seam) and
# recorded as [note] lines at every boot, as before.
#
# WHY: the 11 MLA layers (3, 7, ..., 43) and the MTP layer (45) store q_a_proj / kv_a_proj_with_mqa / q_b_proj / o_proj as
# block-FP8 (e4m3, 128 x 128, weight_scale_inv) but the model is built with quant_config=None and dequantises them to BF16 at
# load (model.py _try_load_fp8_attn_proj; the comment at the quant_config=None line is stale), so every decode step streams
# 39.8 MB of BF16 per MLA layer for weights that exist as 19.9 MB of FP8. E3 (real weights, DRAM-cold, inside CUDA graphs,
# M=3): 183.5 us -> 97.2 us per layer (Marlin 0.53 x BF16) = -1.18 ms per step with the MTP layer. REALISTIC range with patch
# 032 active: 0.95-1.18 ms/step = +2.5-3.1 % decode, because 032 moved each replay's first NCCL op behind compute and
# the two draft replays' first ~742 us (which the MTP layer's own qkv_a + q_b + o_proj share) are the cover of NCCL's
# host-callback wait: draft prefill gains ~0, draft decode at most 121 of 144 us; the target's 949 us (layers 3..43) lie
# after its first NCCL op and are fully gained. The decode A/B/A against the BF16 projections is the verdict, not this estimate.
#
# WHAT: installs files/p033_mla_w8a16.py as vllm/_gb10_mla_w8a16.py and makes 3 anchored edits (files/p033_apply.py) to
# vllm/models/glm5next/common/model.py (nothing else is edited: the MTP layer is a Glm5NextDecoderLayer and its loader calls
# the same _try_load_fp8_attn_proj). Line numbers are those of the v0.32.0 original:
#   :102  after `logger = init_logger(__name__)`: `_p033 = None`; when VLLM_GB10_MLA_W8A16 == "1" (read once, at import)
#         `from vllm import _gb10_mla_w8a16 as _p033` (module missing -> ImportError: the boot aborts on that rank);
#   :403  Glm5NextDecoderLayer.__init__, for MLA layers: `_p033.install(self.self_attn, vllm_config, layer_idx, is_mtp_layer)`
#         replaces the quant_method of fused_qkv_a_proj (the replicated q_a|kv_a), q_b_proj and o_proj by a subclass of
#         UnquantizedLinearMethod whose apply() runs the Marlin W8A16 GEMM (apply_fp8_marlin_linear) when the input is
#         a 2-D bf16 matrix with 0 < M <= MAX_M and no bias, else the BF16 GEMM on the BF16 weight that stays resident;
#   :1470 _try_load_fp8_attn_proj, before the BF16 dequant: `_p033.capture(...)` copies the checkpoint FP8 shard + block
#         scales (explicit copies, sharded exactly like the BF16 loader) so the Marlin copy is built from the checkpoint
#         bytes, not re-quantised.
# DECODE vs PREFILL: CUDA-graph dispatch keys on the token count, so ANY batch of <= MAX_M tokens (decode, but also a short
# prompt, a short follow-up turn with prefix caching, a tail chunk) uses Marlin; prefill chunks LONGER than MAX_M tokens, and
# graph sizes 24..48 (17..48 tokens), run the unchanged BF16 GEMMs (numerics unchanged). Memory: +0.2261 GiB per rank steady
# (36 Marlin copies), ~+0.45 GiB transient peak until each group's self-test has finished (the raw FP8 shard is kept for
# its fp32 references; the initial estimate of a +0.25 GiB peak assumed it was dropped right after the build).
#
# BOOT SELF-TEST (process_weights_after_loading; every rank; FAIL CLOSED, hang-free; one decision line per rank): per module
# V1-V4 (the capture equals the served BF16 weight BITWISE, an independent re-quantisation, scales), the Marlin build, then
# C1-C9 on the real weights at M = 1,2,3,4,6,8,16 (+MAX_M) over three seeded draws (error vs fp32 / vs the BF16 path / vs
# Marlin's own dequant, row cosine, determinism, dispatch fidelity, strided input, the real module call reaches
# quant_method.apply exactly once, memory), cross-rank digests of the replicated qkv_a, an in-graph 11-layer speed gate
# (Marlin <= 0.85 x BF16 at M=3; E3: 0.53), a gloo exchange per group and a commit round. A type whose real module call does
# not reach quant_method.apply (qkv_a only, if deepseek_v2.py routed it to a dedicated kernel) falls back ALONE; any other
# failure keeps the BF16 GEMMs on every rank. The Mac tests model the Marlin layout only through a stand-in tile
# permutation: the first GPU boot's logged C1-C3 numbers are the layout acceptance (on v0.32.0 too).
#
# ORDER / STACKING: runs after 025 / 032, the only patches of the v0.32.0 set that edit model.py (enumerated in
# files/p033_cpu_check.py A3 over every NNN-*.sh and vendored python file). The bases below: the v0.32.0 original and the
# 025 result (025 is ported by another task: until its applier exists the +025 pins read `pending-025-port`, a value that can
# never equal a sha, i.e. those branches are dead and the .sh fails closed on a 025 base; files/p033_cpu_check.py --print-pins
# prints the lines to paste), each with and without 032 (opt-in replicated embedding), and each + 033. mtp.py (the v0.32.0
# original or 029's result) is a read-only contract: the MTP layer's loader must call _try_load_fp8_attn_proj.
# Read-only contracts pinned by content: linear.py (LinearBase.forward -> quant_method.apply, the loader choice at
# construction, RowParallelLinear.reduce_results), marlin_utils_fp8.py / marlin_utils.py (prepare / apply / padding / the
# atomic-add gate / marlin_make_workspace_new), _custom_ops.py (marlin_gemm / gptq_marlin_repack signatures), model_loader/utils.py
# + base_loader.py (process_weights_after_loading runs once per load), spec_decode/eagle/utils.py (the draft is loaded through the
# normal loader).
# Fail-closed install: pinned applier / module shas, pinned base AND result shas, anchors + AST placement validated before any
# write, result compiled in memory, written atomically, re-hashed; any failure or TERM/INT/HUP (the applier child is reaped
# first) restores model.py AND the previous module bytes (or removes the module this run installed) and exits non-zero ->
# the entrypoint aborts. Idempotent (sha). A REUSED container that already carries 033 is refused by 025's / 032's .sh
# on the next boot (unknown model.py sha; launch/up.sh always recreates). Inert unless VLLM_GB10_MLA_W8A16=1 (unset / 0 ->
# nothing is read or written beyond the gate; any other value -> refused, so a typo cannot silently disable it). Set the
# SAME flags on EVERY rank (M_DOCKER_EXTRA): a rank without the gate never joins the boot exchange, so the others give up at
# the bounded barrier (600 s) and keep the BF16 GEMMs (slow boot, no wrong output).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
MODEL="$VLLM_ROOT/models/glm5next/common/model.py"
MTP="$VLLM_ROOT/models/glm5next/common/mtp.py"
LINEAR="$VLLM_ROOT/model_executor/layers/linear.py"
MU8="$VLLM_ROOT/model_executor/layers/quantization/utils/marlin_utils_fp8.py"
MU="$VLLM_ROOT/model_executor/layers/quantization/utils/marlin_utils.py"
COPS="$VLLM_ROOT/_custom_ops.py"
LUTIL="$VLLM_ROOT/model_executor/model_loader/utils.py"
BLOAD="$VLLM_ROOT/model_executor/model_loader/base_loader.py"
EAGLE="$VLLM_ROOT/v1/worker/gpu/spec_decode/eagle/utils.py"
APPLIER="$SRC_DIR/p033_apply.py"
MOD_SRC="$SRC_DIR/p033_mla_w8a16.py"
MOD_DST="$VLLM_ROOT/_gb10_mla_w8a16.py"
MARKER="# [patch-033 gb10-mla-w8a16]"
MARKERS=3
# model.py: every base the entrypoint can hand over after 001-032 (v0.32.0 original; 025 = opt-in mHC prefill sp; each with and
# without 032 = opt-in replicated embedding) and each + 033
MODEL_ORIG=f337b984e25c8cb3d1546dbb4bf7ac8a5c057b19967adb0bb037c416b252b7bb
MODEL_P025=92ab069cd173dad39ded4c35c3c3fbb374a855cd472b3592f13f92eb481131fb
MODEL_ORIG_P032=3fc6088119997ad8961c93e60c11a90597d0eebab7a2f2de227b9b20c6e3c98d
MODEL_P025_P032=45cc5b89636b599e8a842aec23911d2fcd55b335c3bdd39a96d7eed3b46899ff
MODEL_ORIG_P033=7b1ca7eafea2ecd491fcddeba215bc1db7e9c541386ecb3c70be0e041d938567
MODEL_P025_P033=a879a06550f2640e448c0714d5bb72911ed8b99b0272eb6cd86f3f582c06f6e3
MODEL_ORIG_P032_P033=74cca58c9b42d707e835cd9108211ee275f7fd619da042473219735a3da59d32
MODEL_P025_P032_P033=a45306274743905377ef5fc8a3f3d4ed794f7d4acb308e73efae8d761af06407
# read-only contracts (v0.32.0 originals; mtp.py may also be patch 029's result)
MTP_ORIG=55f33f9912c7fd127aaf914df21b060113e04b7d6d1686ab982e1d5b288b1c10
MTP_P029=ff508c4d52864e7c52379572c3461fa04638af41ebc98c8c8f28d309ba77e35a
LINEAR_ORIG=7a9b90937865fa35d955ffc2017f23d3997bc4fba954e2667870b1b4854a38e0
MU8_ORIG=a5cc60882dc32851b2cb7d573e2b28980319a99ef816fd6ed9925b6fabfffd9a
MU_ORIG=e60713c15080347b01425f90eaf03840ca80a1cd226ec57e526e8f6286eeeb22
COPS_ORIG=36428c27bad9bf5ffe5debf9e34bccbc888adb68ad3eb972f6c9ff080e7f683e
LUTIL_ORIG=3091a9f64cbe42ad0e7fc0907fbbcb9197d2d56d5da4008779c5a16d9b8649a6
BLOAD_ORIG=cba765f6e669c51834ea5ac4146d5de9648635cfbc00c987601009865e869b36
EAGLE_ORIG=21d1f84c1f5f4d95fe3eca85cb43da9e1517287fc8b32450d091e1670f5adef4
# vendored sources (this patch owns them)
APPLIER_SHA=888405955929ab1664536dcd1aec0d29ca976e3fb696af44cdf345a07fa54122
MOD_SHA=210b8ea1f9fac7ce7914594353d3226cae55342a9ea0dbcec43a6e6c6e3de1ee

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[033] FATAL: $*" >&2; exit 1; }

echo "=== Patch 033: GB10 MLA projections as FP8 W8A16 (Marlin) in decode (VLLM_GB10_MLA_W8A16=${VLLM_GB10_MLA_W8A16:-unset}, MAX_M=${VLLM_GB10_MLA_W8A16_MAX_M:-unset -> 16}, SPEED_GATE=${VLLM_GB10_MLA_W8A16_SPEED_GATE:-unset -> graph}) ==="
gate="$(printf '%s' "${VLLM_GB10_MLA_W8A16:-0}" | tr -d '[:space:]')"
case "$gate" in
  ''|0)
    if grep -qF "$MARKER" "$MODEL" 2>/dev/null; then
      echo " [note] VLLM_GB10_MLA_W8A16 unset but model.py already carries patch 033 (reused container?);"
      echo "        its import-time gate keeps the image's BF16 projections."
    fi
    echo "[OK] patch 033 skipped (VLLM_GB10_MLA_W8A16=${VLLM_GB10_MLA_W8A16:-unset}); files untouched"
    exit 0 ;;
  1) ;;
  *) fail "VLLM_GB10_MLA_W8A16='${VLLM_GB10_MLA_W8A16}' must be unset/0 (inert) or 1" ;;
esac
MAX_M="$(printf '%s' "${VLLM_GB10_MLA_W8A16_MAX_M:-}" | tr -d '[:space:]')"
if [ -n "$MAX_M" ]; then
  case "$MAX_M" in
    *[!0-9]*|0*) fail "VLLM_GB10_MLA_W8A16_MAX_M='${VLLM_GB10_MLA_W8A16_MAX_M}' must be an integer 1..64 without leading zeros" ;;
  esac
  { [ "${#MAX_M}" -le 2 ] && [ "$MAX_M" -le 64 ]; } || fail "VLLM_GB10_MLA_W8A16_MAX_M='${VLLM_GB10_MLA_W8A16_MAX_M}' must be an integer 1..64"
else
  MAX_M=16
fi
SPEED="$(printf '%s' "${VLLM_GB10_MLA_W8A16_SPEED_GATE:-graph}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
[ -n "$SPEED" ] || SPEED=graph
case "$SPEED" in
  graph|info) ;;
  *) fail "VLLM_GB10_MLA_W8A16_SPEED_GATE='${VLLM_GB10_MLA_W8A16_SPEED_GATE}': graph (default: Marlin <= 0.85 x BF16 inside CUDA graphs, enforced) or info (timed and logged, not enforced)" ;;
esac

for f in "$MODEL" "$MTP" "$LINEAR" "$MU8" "$MU" "$COPS" "$LUTIL" "$BLOAD" "$EAGLE" "$APPLIER" "$MOD_SRC"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p033_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MOD_SRC")" = "$MOD_SHA" ] || fail "files/p033_mla_w8a16.py sha256 $(sha "$MOD_SRC") != pinned $MOD_SHA"
[ "$(sha "$LINEAR")" = "$LINEAR_ORIG" ] || fail "linear.py sha256 $(sha "$LINEAR") != v0.32.0 original $LINEAR_ORIG (LinearBase.forward -> quant_method.apply, UnquantizedLinearMethod, the loader choice at construction, RowParallelLinear.reduce_results); refusing"
[ "$(sha "$MU8")" = "$MU8_ORIG" ] || fail "marlin_utils_fp8.py sha256 $(sha "$MU8") != v0.32.0 original $MU8_ORIG (the FP8 Marlin prepare/apply contract); refusing"
[ "$(sha "$MU")" = "$MU_ORIG" ] || fail "marlin_utils.py sha256 $(sha "$MU") != v0.32.0 original $MU_ORIG (Marlin padding / workspace / atomic-add contract, torch.ops.vllm.marlin_gemm); refusing"
[ "$(sha "$COPS")" = "$COPS_ORIG" ] || fail "_custom_ops.py sha256 $(sha "$COPS") != v0.32.0 original $COPS_ORIG (marlin_gemm / gptq_marlin_repack signatures); refusing"
[ "$(sha "$LUTIL")" = "$LUTIL_ORIG" ] || fail "model_loader/utils.py sha256 $(sha "$LUTIL") != v0.32.0 original $LUTIL_ORIG (process_weights_after_loading runs 033's boot self-test); refusing"
[ "$(sha "$BLOAD")" = "$BLOAD_ORIG" ] || fail "model_loader/base_loader.py sha256 $(sha "$BLOAD") != v0.32.0 original $BLOAD_ORIG (process_weights_after_loading after load_weights); refusing"
[ "$(sha "$EAGLE")" = "$EAGLE_ORIG" ] || fail "spec_decode/eagle/utils.py sha256 $(sha "$EAGLE") != v0.32.0 original $EAGLE_ORIG (the MTP draft is loaded through the normal loader); refusing"
mtp_now="$(sha "$MTP")"
[ "$mtp_now" = "$MTP_ORIG" ] || [ "$mtp_now" = "$MTP_P029" ] || fail "mtp.py sha256 $mtp_now is neither the v0.32.0 original nor patch 029's result (the MTP layer's loader must call model.py _try_load_fp8_attn_proj); refusing"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MOD_SRC" || fail "$MOD_SRC does not parse"
# files the design depends on but the boot self-test, not a pin, covers: recorded at every boot
for rel in model_executor/models/deepseek_v2.py model_executor/layers/mla.py model_executor/parameter.py; do
  if [ -f "$VLLM_ROOT/$rel" ]; then
    echo " [note] $rel sha256 $(sha "$VLLM_ROOT/$rel") (not pinned: covered by the boot self-test; recorded)"
  else
    echo " [note] $rel not found under $VLLM_ROOT (not pinned)"
  fi
done

m_now="$(sha "$MODEL")"
mode=""; expect=""; base=""
case "$m_now" in
  "$MODEL_ORIG_P033"|"$MODEL_P025_P033"|"$MODEL_ORIG_P032_P033"|"$MODEL_P025_P032_P033")
    mode=installed ;;
  "$MODEL_ORIG") mode=apply; expect="$MODEL_ORIG_P033"; base="v0.32.0 original" ;;
  "$MODEL_P025") mode=apply; expect="$MODEL_P025_P033"; base="patch-025 result" ;;
  "$MODEL_ORIG_P032") mode=apply; expect="$MODEL_ORIG_P032_P033"; base="patch-032 result" ;;
  "$MODEL_P025_P032") mode=apply; expect="$MODEL_P025_P032_P033"; base="patch-025 + 032 result" ;;
  *)
    if grep -qF "$MARKER" "$MODEL"; then
      fail "model.py carries a patch-033 edit that is not one of this revision's pinned results (sha ${m_now:0:16}...); recreate the container from the image"
    fi
    fail "model.py sha256 $m_now is none of the four bases 001-032 can hand over (v0.32.0 original, 025 result, each with and without 032); a new model.py writer must be enumerated and pinned here; refusing"
    ;;
esac

# Back up the target and any pre-existing module bytes; restore all of them on any failure / signal.
had_mod=0; if [ -e "$MOD_DST" ]; then had_mod=1; cp -p "$MOD_DST" "$MOD_DST.p033.orig"; fi
cp -p "$MODEL" "$MODEL.p033.orig"
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    rm -f "$MODEL.p033.tmp"
    mv -f "$MODEL.p033.orig" "$MODEL"
    if [ "$had_mod" = 1 ]; then mv -f "$MOD_DST.p033.orig" "$MOD_DST"; else rm -f "$MOD_DST"; fi
    echo "[033] model.py and vllm/_gb10_mla_w8a16.py restored to their previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

install -m 0644 "$MOD_SRC" "$MOD_DST"
[ "$(sha "$MOD_DST")" = "$MOD_SHA" ] || fail "installed $MOD_DST sha256 mismatch"
if [ "$mode" = installed ]; then
  echo " [skip] model.py already carries patch 033 (sha ${m_now:0:16}...)"
else
  echo " [state] model.py is the $base (${m_now:0:16}...)"
  python3 "$APPLIER" apply "$MODEL" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p033_apply.py refused model.py ($base)"
  APPLIER_PID=""
  got="$(sha "$MODEL")"
  [ "$got" = "$expect" ] || fail "patched model.py sha256 $got != pinned $expect ($base + 033)"
  echo " [OK] model.py: $base + 033 (sha256 ${got:0:16}...)"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$MODEL")" = "$MARKERS" ] || fail "expected $MARKERS markers in $MODEL"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$MODEL" "$MOD_DST" \
  || fail "patched/installed files do not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$MODEL.p033.orig" "$MOD_DST.p033.orig"
echo "[OK] patch 033 in place (VLLM_GB10_MLA_W8A16=1, MAX_M=$MAX_M, speed gate $SPEED, model.py sha256 $(sha "$MODEL" | cut -c1-16)...; the module logs one boot-decision line per rank at model load)"
