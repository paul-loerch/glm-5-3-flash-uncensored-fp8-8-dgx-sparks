#!/usr/bin/env bash
# 025-gb10-mhc-prefill-sp.sh — mHC prefill sequence parallelism (mode 'sp');
# v0.32.0 PORT (revision 4: re-anchored on vllm/models/glm5next/common/model.py, hooks + boot self-test semantics unchanged).
# REVISION 3 (2026-10-02): hang-free boot self-test (every collective issued on every rank with a planned, pre-checked
# shape; rank-local failures become failing rows; gloo preflight + verdict behind a bounded monitored barrier), the
# canonical RS over pynccl send/recv and checked at boot only with RS=canonical, live checks on layers 0 AND 3 at
# T = MIN_T and MIN_T+3 + check 2 (a live suppressed partial), fail-closed live-layer lookup.
# REVISION 4 (v0.32.0): model.py -> models/glm5next/common/model.py; the Glm5NextModel.forward layer loop gained the
# EAGLE-3 aux taps (a forward that captures them is never sharded) and enumerate(..., start=self.start_layer); the decoder
# layer's mHC forward is unchanged. 4 anchored edits (module gate, decoder-layer hook, layer-loop wrap, exit gather).
# Upstream's own sequence-parallel code (sp_shard / sp_all_gather / sp_reduce_scatter, use_sequence_parallel_moe =
# EP + DP > 1) is inert at DP=1 and is refused together with sp by the envelope.
# Opt-in (VLLM_GB10_MHC_SP=sp|ref), NUMERICS CHANGE (NOT bit-identical to the all-reduce path): during eligible
# prefills the residual stream is sharded by token rows across the 8 TP ranks, every mHC block runs on this rank's
# T/8 rows, and the per-block attention / MoE all-reduce becomes reduce-scatter (+ a final all-gather before the
# norm). On v0.32.0 it is, as on the previous base, a numerics
# change relative to the v0.32.0 all-reduce path (it changes the prefill reduction order).
#
# WHY: at TP=8 every mHC block (layer-0 hc_pre + 89 fused hc_post/hc_pre) and the 91 all-reduces per forward are
# REPLICATED on all 8 ranks (common/model.py:553-623; mHC ~357 ms = ~22 % of a 3,840-token chunk in a profile of the
# previous base). Full SP turns the 90 block all-reduces into reduce-scatter + all-gather (SAME bytes: a ring
# AR moves 2*(7/8)*S per rank, RS and AG move (7/8)*S each) and runs mHC on T/8 rows. A microbenchmark (GO):
# NCCL RS+AG = 1.00/0.92/0.99x one all-reduce at 8/16/32 MiB; estimated sp +17.9/+21.6/+24.5 %
# prefill at 1k/2k/3,840-token chunks (measured +15.6 % at 32k / 64k on the previous base). Not bit-identical: NCCL ring RS
# does not reproduce the ring AR's element order (differ on ~52 % of elements, both valid sums);
# VLLM_GB10_MHC_SP_RS=canonical makes RS an all-to-all + fixed rank-order fp32 sum (patch 012's arithmetic), so
# sp == ref bitwise and prefill gets decode's reduction semantics.
#
# WHAT: installs files/gb10_prefill_sp.py as vllm/_gb10_prefill_sp.py and makes 4 anchored edits (files/p025_apply.py)
# to vllm/models/glm5next/common/model.py (v0.32.0 line numbers):
#   :345 before class Glm5NextDecoderLayer: module gate _P025_ON (VLLM_GB10_MHC_SP read once at import; set -> `from vllm
#        import _gb10_prefill_sp as _p025`; set + module missing / invalid value -> ImportError / ValueError: fail closed);
#   :531 first statements of Glm5NextDecoderLayer.forward: `if _P025_ON and _p025.sp_active() and self.mhc and not
#        self.is_mtp_layer: return _p025.sp_layer_forward(...)` (MTP / non-mHC layers unchanged);
#   :822 Glm5NextModel.forward layer loop: begin_forward (boot self-test on the first eligible call + prs neutralisation)
#        then, in a try/finally that calls end_forward, shard_model_input -> the (unchanged) loop; begin_forward returns a
#        handle only for T >= VLLM_GB10_MHC_SP_MIN_T (default 1,024; validated floor 1,024), eager, not inside a CUDA-graph
#        capture, not micro-batched (DBO/ubatch) and with no aux_hidden_state_layers; the first-eligible-call boot
#        self-test refuses TP != 8, PP > 1, prefill/decode context parallel, sequence-parallel MoE and VLLM_USE_DEEP_GEMM=1;
#   :851 the exit: `elif _p025_handle is not None:` gather_model_output before the norm.
#   Otherwise None (the upstream path runs, unchanged). Decode (T <= 48) and short prefills never shard. mode=ref keeps
#   the mHC replicated and only changes the reduction (the A/B).
#
# COMPOSITION WITH 022 (prs): both gates may be set (the previous base served prs; sp is added on top). Inside an sp region the residual
# stream is already this rank's rows, so prs must not split the pre-norm GEMM again: begin_forward raises patch 022's
# effective MIN_T to infinity for the region (its public _STATE.min_t) and end_forward restores it, so prs returns the
# upstream (unsplit) path on the shard (the hook sits in _hc_prenorm_gemm_outputs; prenorm_rowsplit's first test is
# T < _STATE.min_t); for inert forwards (decode / T < MIN_T) prs is exactly as without sp. Both default to MIN_T=1,024,
# so they never act on the same forward; begin_forward also asserts prs is inactive for this T after neutralising. prs
# DOES boot under sp — exactly once, INSIDE sp's boot self-test: the live-weights check calls hc_pre / hc_fused_post_pre
# on the full T = MIN_T (and MIN_T+3) rows as its reference before any sp region exists, and those T are >= prs's MIN_T,
# so prs runs its own collective bit-identity boot self-test there (all 8 ranks) and row-splits those reference calls (the
# check's shard calls run with prs neutralised, as in an sp region). Every later eligible forward is an sp region with prs
# neutralised, so prs never splits while serving; expect one "[patch-022] ... ACTIVE" line per rank, logged inside the
# sp boot. Patch 012 (one-shot AR for <= 64 KiB decode messages) is untouched: sp only rewrites the >= MIN_T prefill
# reductions; decode all-reduces still take the patch-012 path (its `device_communicator._gb10_oneshot` is read by the
# boot's check 2 when present). RS=canonical uses 012's transport (grouped pynccl send/recv on the TP group's existing NCCL
# communicator; sp never touches the torch-c10d device group).
#
# NUMERICS (why every rank still computes identical final outputs): row t is owned by exactly one rank (t // S) and
# all of its mHC arithmetic runs there (the kernels are per-row). The attention / MLP inputs are the all-gathered
# full-T x, bit-identical on every rank (all-gather is a copy). The reduce-scatter sum for row t is deterministic
# (fixed order for a fixed owner, NCCL; fixed rank order, canonical). The final all-gather before the norm copies each
# owner's rows to all ranks, so all ranks hold the same [T,H] into the norm (and the MTP pass / LM head see the full
# sequence). Shard-invariance of the mHC (S is even -> every shard start is even -> each pre-norm-GEMM CTA of a shard
# is a CTA of the full call; mhc_post / big_fuse read only their own row): sp == ref bitwise. With patch 011 the shard
# (S >= 128 rows) is routed to the same block_m=2 kernel as the full call; without 011 the full call is block_m=2 and the
# shard the per-token kernel, bitwise equal per row (011's equivalence). The boot self-test re-checks on the device
# (fail-closed on every rank, HANG-FREE): the collective reconstruction + a double-reduction sensitivity probe,
# rank-consistency (bitwise), canonical == the rank-order fp32 sum (RS=canonical only), the LIVE weights of layer 0 (KDA +
# dense) and layer 3 (MLA + MoE): mHC ops (hc_pre / hc_fused_post_pre / hc_post+contract) on a zero-padded S-row shard ==
# the full call sliced at T = MIN_T and MIN_T+3, the real suppression handles flip the live o_proj / MoE / down_proj flags
# off and restore them, and check 2: a live o_proj / down_proj suppressed partial reduced by sp (RS + AG) == the
# decode all-reduce of it (patch 012's one-shot; bitwise with RS=canonical). The whole sp_layer_forward wiring vs the
# upstream path is proven on CPU by files/p025_e2e_check.py (the real patched model.py functions on 8 gloo ranks, with
# the wiring mutants as required negative controls).
#
# ORDER / STACKING: model.py is the v0.32.0 original when 025 runs: no earlier patch (001-022) edits it (019 / 022 edit
# tilelang.py and tilelang_kernels.py, 011 tilelang_kernels.py). Patches 029 / 032 / 033 edit model.py / mtp.py AFTER 025
# (their appliers' anchors are disjoint from 025's four; the chain model.py states are pinned in their own .sh and in
# pins.txt). A reused container carrying 025 + later model.py edits is refused by the next boot's 025 (unknown model.py
# sha; launch/up.sh always recreates). Fail-closed: pinned applier/module/target/result shas, the patch-022 module sha (025
# neutralises prs at runtime), and read-only dependency shas (moe_runner.py, linear.py, mhc.py, fused_moe/config.py — the
# all-reduce-suppression + per-row contracts; tilelang_kernels.py — the per-row prenorm kernels; tilelang.py — one of the
# known mHC-dispatch variants); VLLM_USE_DEEP_GEMM=0 enforced; anchors + AST scope validated before any write; result
# compiled in memory, written atomically, re-hashed; any failure or TERM/INT/HUP (the applier child is reaped first)
# restores model.py AND the previous module bytes (or removes the module this run installed) and exits non-zero -> the
# entrypoint aborts. Idempotent (sha + marker). Inert unless VLLM_GB10_MHC_SP is set to a mode. Set the SAME flags on EVERY
# rank (M_DOCKER_EXTRA): among ranks that run the gate the boot fingerprint (module sha, mode, MIN_T, RS, prs gate, TP
# size) must match or every participating rank raises; a rank left entirely without the gate never joins the sp
# collectives and the others block to the gloo / execute timeout (as 018/021/022) — up.sh's patch-set check and the
# shared M_DOCKER_EXTRA make that practically impossible.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
MODEL="$VLLM_ROOT/models/glm5next/common/model.py"
MR="$VLLM_ROOT/model_executor/layers/fused_moe/runner/moe_runner.py"
LINEAR="$VLLM_ROOT/model_executor/layers/linear.py"
MHC="$VLLM_ROOT/model_executor/layers/mhc.py"
TL="$VLLM_ROOT/model_executor/kernels/mhc/tilelang.py"
TK="$VLLM_ROOT/model_executor/kernels/mhc/tilelang_kernels.py"
MOECFG="$VLLM_ROOT/model_executor/layers/fused_moe/config.py"
PRS_MOD="$SRC_DIR/gb10_mhc_prs.py"
APPLIER="$SRC_DIR/p025_apply.py"
MOD_SRC="$SRC_DIR/gb10_prefill_sp.py"
MOD_DST="$VLLM_ROOT/_gb10_prefill_sp.py"
MARKER="# [patch-025 gb10-mhc-prefill-sp]"
MARKERS=4

# model.py: vLLM v0.32.0 original and this patch's result (no earlier patch edits model.py; 029 / 032 / 033 come after)
MODEL_ORIG=f337b984e25c8cb3d1546dbb4bf7ac8a5c057b19967adb0bb037c416b252b7bb
MODEL_ORIG_P025=92ab069cd173dad39ded4c35c3c3fbb374a855cd472b3592f13f92eb481131fb
# read-only dependencies (sp's all-reduce suppression + per-row contracts are written against these exact v0.32.0 files)
MR_ORIG=20b69fe514bebd6723a1378f89a564c8fec4144c30244101f629c3f32c188183
LINEAR_ORIG=7a9b90937865fa35d955ffc2017f23d3997bc4fba954e2667870b1b4854a38e0
MHC_ORIG=433d750a4d669ad468679b1faf0a3f1e846410198fb4a1eeee2d14cfe55e6a55
MOECFG_ORIG=575248839aa351e8a1ab47a0965882b32c35932eb3e6e780271606e5b3451d9d
# tilelang_kernels.py: the per-row bm2 / per-token prenorm kernels whose shard-invariance sp relies on. Accept exactly the
# known-good variants: the v0.32.0 original and every 011 / 019 / 011+019 result (the kernels are byte-identical in all
# four; 011 moves a threshold and the warmup buckets, 019 adds a warmup-registration method). Pins from 011 / 019.
TK_KNOWN="61a384a091344d8207a04b2b7c74d30a81626eec4c068cf1dd034699cbfc34b5 \
8e9d71a1f4c3db3a312191105d0f2b15d2d8713bfec6c9e8a2e5207777ed4e95 \
77f41a656775d6f3db928e5900c78f6f6e4f38b84b29c7ef59a48c9a31615125 \
e80f60b8de9d5f82547f8eb92afc71e45e4d5406ff89812efd10c2835a0abd8c"
# tilelang.py: the mHC-op dispatch (prenorm routing, mhc_post / fused). sp does NOT edit it, but depends on the per-row
# routing. Accept exactly the known-good variants: the v0.32.0 original and every 019 / 022 / 019+022 result (the pins in
# 019 / 022): 019 only the num_tokens <= 32 fused decode bf16 weight path, 022 adds the prenorm hook that is the upstream
# path once prs is neutralised.
TL_KNOWN="204e19209db5b19eb3dd25ea6ae10c84425dee02ef5bc896eca3f83da6149a8e \
05a8b792dc27fb01544e199a6e329a5092433387fa13e2216c1dc5ce2b64f146 \
af5d8dac025be6175fee132f55188f35e4518227c20ce9529f2f47ececa47f77 \
4bea7f8ebe797ce7b82909b5dae03d7988e59fb6b845edc7b02d64317cae88d6"
# patch-022 module source: 025 neutralises it at runtime by writing its _STATE.min_t, so the 022 revision is locked
# here (a changed prs module could break the neutralisation silently).
PRS_MOD_SHA=084d90ec8c6ebe45cdd5fde2d2caaa45f644005a0a476afa6bb23cf0f4a510f7
# vendored sources (this patch owns them)
APPLIER_SHA=1c4f630a30814ced4a25deae944f2a6e1868f9d7bafa3c75c42ed8886a459ef4
MOD_SHA=7054a08312a5984859622419a562f92e3779ab9a2daacd32997948f6be055c3d

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[025] FATAL: $*" >&2; exit 1; }

echo "=== Patch 025: GB10 mHC prefill sequence-parallel (VLLM_GB10_MHC_SP=${VLLM_GB10_MHC_SP:-unset}, MIN_T=${VLLM_GB10_MHC_SP_MIN_T:-unset -> 1024}, RS=${VLLM_GB10_MHC_SP_RS:-unset -> nccl}) ==="
mode_raw="$(printf '%s' "${VLLM_GB10_MHC_SP:-0}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$mode_raw" in
  ''|0|off|false|no)
    if grep -qF "$MARKER" "$MODEL" 2>/dev/null; then
      echo " [note] VLLM_GB10_MHC_SP unset but model.py already carries patch 025 (reused container?);"
      echo "        its import-time gate (_P025_ON) keeps upstream's replicated mHC + all-reduce path."
    fi
    echo "[OK] patch 025 skipped (VLLM_GB10_MHC_SP unset); files untouched"
    exit 0 ;;
  1|sp|on|true|yes) mode=sp ;;
  ref) mode=ref ;;
  *) fail "VLLM_GB10_MHC_SP='${VLLM_GB10_MHC_SP}' must be unset/0 (inert), 1/sp (full SP) or ref (reduction-only)" ;;
esac

MIN_T="${VLLM_GB10_MHC_SP_MIN_T:-1024}"
case "$MIN_T" in ''|*[!0-9]*) fail "VLLM_GB10_MHC_SP_MIN_T='$MIN_T' must be an integer";; esac
{ [ "${#MIN_T}" -le 8 ] && [ "$MIN_T" -ge 1024 ] && [ "$MIN_T" -le 1048576 ]; } \
  || fail "VLLM_GB10_MHC_SP_MIN_T=$MIN_T outside [1024, 1048576] (default 1024; the validated floor is 1024 — below it the shard hits the small-T prenorm kernel and the sp/prs overlap is not validated)"
RS="$(printf '%s' "${VLLM_GB10_MHC_SP_RS:-nccl}" | tr 'A-Z' 'a-z')"
case "$RS" in nccl|canonical) ;; *) fail "VLLM_GB10_MHC_SP_RS='$RS' must be nccl or canonical";; esac
DG="$(printf '%s' "${VLLM_USE_DEEP_GEMM:-0}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$DG" in 1|on|true|yes) fail "VLLM_GB10_MHC_SP needs VLLM_USE_DEEP_GEMM=0: the DeepGEMM prenorm (tf32, split-k) is not the per-row bm2 kernel sp's shard-invariance is proven against";; esac

for f in "$MODEL" "$MR" "$LINEAR" "$MHC" "$TL" "$TK" "$MOECFG" "$PRS_MOD" "$APPLIER" "$MOD_SRC"; do [ -f "$f" ] || fail "$f not found"; done
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p025_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MOD_SRC")" = "$MOD_SHA" ] || fail "files/gb10_prefill_sp.py sha256 $(sha "$MOD_SRC") != pinned $MOD_SHA"
[ "$(sha "$PRS_MOD")" = "$PRS_MOD_SHA" ] || fail "files/gb10_mhc_prs.py (patch 022) sha256 $(sha "$PRS_MOD") != pinned $PRS_MOD_SHA (025 neutralises prs at runtime; the 022 revision is locked here); refusing"
# read-only dependencies: refuse if the suppression / per-row contract / mHC-kernel files changed
[ "$(sha "$MR")" = "$MR_ORIG" ] || fail "moe_runner.py sha256 $(sha "$MR") != v0.32.0 original $MR_ORIG (skip_final_all_reduce contract); refusing"
[ "$(sha "$LINEAR")" = "$LINEAR_ORIG" ] || fail "linear.py sha256 $(sha "$LINEAR") != v0.32.0 original $LINEAR_ORIG (RowParallelLinear.reduce_results contract); refusing"
[ "$(sha "$MHC")" = "$MHC_ORIG" ] || fail "mhc.py sha256 $(sha "$MHC") != v0.32.0 original $MHC_ORIG (per-row mHC ops); refusing"
tk_now="$(sha "$TK")"; tk_ok=0; for s in $TK_KNOWN; do [ "$tk_now" = "$s" ] && tk_ok=1 && break; done
[ "$tk_ok" = 1 ] || fail "tilelang_kernels.py sha256 $tk_now is not a known v0.32.0 variant (original or an 011 / 019 result); the per-row prenorm kernels sp's shard-invariance relies on are unverified for it; refusing"
tl_now="$(sha "$TL")"; tl_ok=0; for s in $TL_KNOWN; do [ "$tl_now" = "$s" ] && tl_ok=1 && break; done
[ "$tl_ok" = 1 ] || fail "tilelang.py sha256 $tl_now is not a known mHC-dispatch variant (v0.32.0 original or a 019 / 022 result); the prenorm routing sp relies on is unverified for it; refusing"
[ "$(sha "$MOECFG")" = "$MOECFG_ORIG" ] || fail "fused_moe/config.py sha256 $(sha "$MOECFG") != v0.32.0 original $MOECFG_ORIG (skip_final_all_reduce field); refusing"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MOD_SRC" || fail "$MOD_SRC does not parse"

m_now="$(sha "$MODEL")"
mode_state=""; expect=""; base=""
case "$m_now" in
  "$MODEL_ORIG_P025") mode_state=installed ;;
  "$MODEL_ORIG") mode_state=apply; expect="$MODEL_ORIG_P025"; base="v0.32.0 original" ;;
  *)
    if grep -qF "$MARKER" "$MODEL"; then
      fail "model.py carries a patch-025 edit that is not one of this revision's pinned results (sha ${m_now:0:16}...); recreate the container from the image"
    fi
    fail "model.py sha256 $m_now is not the v0.32.0 original (any other model.py revision - a patch that ran before 025 - is refused: its anchors and the aux-tap contract are unverified); refusing"
    ;;
esac

had_mod=0; if [ -e "$MOD_DST" ]; then had_mod=1; cp -p "$MOD_DST" "$MOD_DST.p025.orig"; fi
cp -p "$MODEL" "$MODEL.p025.orig"
ROLLBACK=1
rollback() {
  if [ "${ROLLBACK:-0}" = 1 ]; then
    rm -f "$MODEL.p025.tmp"
    mv -f "$MODEL.p025.orig" "$MODEL"
    if [ "$had_mod" = 1 ]; then mv -f "$MOD_DST.p025.orig" "$MOD_DST"; else rm -f "$MOD_DST"; fi
    echo "[025] model.py and vllm/_gb10_prefill_sp.py restored to their previous bytes" >&2
  fi
}
trap rollback EXIT
APPLIER_PID=""
on_signal() { if [ -n "$APPLIER_PID" ]; then kill "$APPLIER_PID" 2>/dev/null || true; wait "$APPLIER_PID" 2>/dev/null || true; fi; exit 130; }
trap on_signal TERM INT HUP

install -m 0644 "$MOD_SRC" "$MOD_DST"
[ "$(sha "$MOD_DST")" = "$MOD_SHA" ] || fail "installed $MOD_DST sha256 mismatch"
if [ "$mode_state" = installed ]; then
  echo " [skip] model.py already carries patch 025 (sha ${m_now:0:16}...)"
else
  echo " [state] model.py is the $base (${m_now:0:16}...)"
  python3 "$APPLIER" apply "$MODEL" & APPLIER_PID=$!
  wait "$APPLIER_PID" || fail "p025_apply.py refused model.py ($base)"
  APPLIER_PID=""
  got="$(sha "$MODEL")"
  [ "$got" = "$expect" ] || fail "patched model.py sha256 $got != pinned $expect ($base + 025)"
  echo " [OK] model.py: $base + 025 (sha256 ${got:0:16}...)"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$MODEL")" = "$MARKERS" ] || fail "expected $MARKERS markers in $MODEL"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$MODEL" "$MOD_DST" \
  || fail "patched/installed files do not parse"
ROLLBACK=0; trap - EXIT TERM INT HUP
rm -f "$MODEL.p025.orig" "$MOD_DST.p025.orig"
echo "[OK] patch 025 in place (VLLM_GB10_MHC_SP=$mode, MIN_T=$MIN_T, RS=$RS: mHC prefill sequence-parallel; boot self-test logs '[patch-025] ... ACTIVE'; SERVING needs the owner's OK — numerics change)"
