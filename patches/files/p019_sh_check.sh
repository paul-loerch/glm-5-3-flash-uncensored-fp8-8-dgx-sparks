#!/usr/bin/env bash
# [patch-019 gb10-mhc-fn-bf16] .sh scenario checks (v0.32.0 port). Runs patches/019 (and 011 / 022 for stacking, in the
# entrypoint's numeric order) against scratch VLLM_ROOT trees seeded with the v0.32.0 originals; never touches a real vLLM
# install. Needs bash, python3 (no torch).
#   bash files/p019_sh_check.sh --tree <v0.32.0>/vllm --work /tmp/p019-sh-check
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; TREE=""; WORK=""
while [ $# -gt 0 ]; do
  case "$1" in --tree) TREE="$2"; shift 2 ;; --work) WORK="$2"; shift 2 ;; --patches) PDIR="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;; esac
done
[ -d "$TREE" ] && [ -n "$WORK" ] || { echo "usage: $0 --tree <vllm dir> --work DIR [--patches DIR]" >&2; exit 2; }
. "$PDIR/files/pmhc_sh_lib.sh"; pmhc_init
TL_REL=model_executor/kernels/mhc/tilelang.py
TK_REL=model_executor/kernels/mhc/tilelang_kernels.py
HELPER_REL=model_executor/warmup/jit_warmup_tilelang_helper.py
JITW_REL=model_executor/warmup/jit_warmup.py
TLU_REL=tilelang_utils/__init__.py
FILES="$TL_REL $TK_REL $HELPER_REL $JITW_REL $TLU_REL"
M19="# [patch-019 gb10-mhc-fn-bf16]"
S19="$P/019-gb10-mhc-fn-bf16.sh"; S11="$P/011-mhc-prenorm-bm2-threshold.sh"; S22="$P/022-gb10-mhc-prefill-rowsplit.sh"
no_leftovers() { ! ls "$1"/*.p019.* "$1/$(dirname "$TL_REL")"/*.p019.* >/dev/null 2>&1; }
mod_absent() { [ ! -e "$1/_gb10_mhc_fn_bf16.py" ]; }
mod_ok() { [ "$(sha "$1/_gb10_mhc_fn_bf16.py")" = "$(sha "$P/files/gb10_mhc_fn_bf16.py")" ]; }
TLO=$(sha "$TREE/$TL_REL"); TKO=$(sha "$TREE/$TK_REL")
G=VLLM_GB10_MHC_FN_BF16=1; E11=VLLM_GB10_MHC_BM2_MIN_T=128; E22=VLLM_GB10_MHC_PRS=1

[ "$TLO" = "$(pin 019 TL_ORIG)" ] && [ "$TKO" = "$(pin 019 TK_ORIG)" ] && ok "S0 the tree's tilelang.py / tilelang_kernels.py are the pinned v0.32.0 originals" || bad "S0 shas"

r=$(mkroot s1 $FILES); rc=$(run "$S19" "$r")
[ "$rc" = 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && mod_absent "$r" && grep -q "skipped" "$r.log" \
  && ok "S1 gate unset: skip, untouched, no module" || bad "S1 rc=$rc"

r=$(mkroot s2 $FILES); rc=$(run "$S19" "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$TL_REL")" = "$(pin 019 TL_P019)" ] && [ "$(sha "$r/$TK_REL")" = "$(pin 019 TK_P019)" ] \
  && [ "$(grep -cF "$M19" "$r/$TL_REL")" = 3 ] && [ "$(grep -cF "$M19" "$r/$TK_REL")" = 1 ] && mod_ok "$r" && no_leftovers "$r" \
  && ok "S2 apply on the originals: both files == their pinned results (3 + 1 markers), module installed, no backups left" \
  || { bad "S2 rc=$rc"; tail -5 "$r.log"; }
S2=$r
rc=$(run "$S19" "$S2" $G)
[ "$rc" = 0 ] && [ "$(sha "$S2/$TL_REL")" = "$(pin 019 TL_P019)" ] && grep -q "already carry patch 019" "$S2.log" && no_leftovers "$S2" \
  && ok "S3 idempotent re-run" || bad "S3 rc=$rc"
rc=$(run "$S19" "$S2")
[ "$rc" = 0 ] && grep -q "unset but tilelang.py / tilelang_kernels.py already carry patch 019" "$S2.log" && [ "$(sha "$S2/$TL_REL")" = "$(pin 019 TL_P019)" ] \
  && ok "S3b gate unset on patched files: note + skip, untouched" || bad "S3b rc=$rc"
# S3c stale 019 edit in one file -> refuse, both untouched
r=$(mkroot s3c $FILES); cp "$S2/$TL_REL" "$r/$TL_REL"; printf '\n# stale 019 revision\n' >> "$r/$TL_REL"
before=$(sha "$r/$TL_REL"); rc=$(run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$before" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && mod_absent "$r" && no_leftovers "$r" \
  && ok "S3c stale 019 edit in tilelang.py refused, both files untouched, no module" || bad "S3c rc=$rc"
# S3d half-applied pair (tilelang.py patched, tilelang_kernels.py original) -> refused
r=$(mkroot s3d $FILES); cp "$S2/$TL_REL" "$r/$TL_REL"; rc=$(run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && mod_absent "$r" && grep -q "edits both together" "$r.log" \
  && ok "S3d half-applied pair refused, untouched" || bad "S3d rc=$rc"
# S3e installed targets + a stale module -> module re-installed with the pinned bytes
echo "# old module revision" > "$S2/_gb10_mhc_fn_bf16.py"; rc=$(run "$S19" "$S2" $G)
[ "$rc" = 0 ] && mod_ok "$S2" && [ "$(sha "$S2/$TK_REL")" = "$(pin 019 TK_P019)" ] && no_leftovers "$S2" \
  && ok "S3e installed targets + stale module: module replaced by the pinned bytes" || bad "S3e rc=$rc"

# M the entrypoint's numeric order on fresh originals for every {011, 019, 022} gate set; 019 sees each earlier result
for g11 in 0 1; do for g19 in 0 1; do for g22 in 0 1; do
  e=""; [ $g11 = 1 ] && e="$e $E11"; [ $g19 = 1 ] && e="$e $G"; [ $g22 = 1 ] && e="$e $E22"
  r=$(mkroot "m_$g11$g19$g22" $FILES); a=$(run "$S11" "$r" $e); c=$(run "$S19" "$r" $e); d=$(run "$S22" "$r" $e)
  case "$g11$g19" in 00) tk=TK_ORIG;; 10) tk=TK_P011;; 01) tk=TK_P019;; 11) tk=TK_P011_P019;; esac
  case "$g19$g22" in 00) tl=TL_ORIG;; 10) tl=TL_P019;; 01) tl=TL_P022;; 11) tl=TL_P019_P022;; esac
  wtk=$(pin 019 $tk); wtl=$(pin 019 $tl)
  if [ "$a$c$d" = 000 ] && [ "$(sha "$r/$TK_REL")" = "$wtk" ] && [ "$(sha "$r/$TL_REL")" = "$wtl" ] && no_leftovers "$r"; then
    a2=$(run "$S11" "$r" $e); c2=$(run "$S19" "$r" $e); d2=$(run "$S22" "$r" $e)
    if [ "$a2$c2$d2" = 000 ] && [ "$(sha "$r/$TK_REL")" = "$wtk" ] && [ "$(sha "$r/$TL_REL")" = "$wtl" ]; then
      ok "M 011=$g11 019=$g19 022=$g22: entrypoint order -> TK $tk, TL $tl; re-run of all three idempotent"
    else bad "M $g11$g19$g22 re-run rc=$a2$c2$d2"; fi
  else bad "M $g11$g19$g22: rc 011=$a 019=$c 022=$d"; tail -3 "$r.log"; fi
done; done; done
# M2 019 AFTER 022 (not the entrypoint order, but a reused container): accepted, same pinned result
r=$(mkroot m2 $FILES); run "$S22" "$r" $E22 >/dev/null; c=$(run "$S19" "$r" $G)
[ "$c" = 0 ] && [ "$(sha "$r/$TL_REL")" = "$(pin 019 TL_P019_P022)" ] && ok "M2 019 on a 022-patched tilelang.py -> TL_P019_P022" || bad "M2 rc=$c"

# S5 unexpected targets -> refuse, untouched, no module
r=$(mkroot s5 $FILES); printf '\n# local edit\n' >> "$r/$TL_REL"; before=$(sha "$r/$TL_REL"); rc=$(run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$before" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "tilelang.py sha256" "$r.log" \
  && ok "S5 unexpected tilelang.py refused, untouched" || bad "S5 rc=$rc"
r=$(mkroot s5b $FILES); printf '\n# kernel edit\n' >> "$r/$TK_REL"; rc=$(run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && mod_absent "$r" && grep -q "tilelang_kernels.py sha256" "$r.log" \
  && ok "S5b unexpected tilelang_kernels.py refused, untouched" || bad "S5b rc=$rc"
for rel in $HELPER_REL $JITW_REL $TLU_REL; do
  r=$(mkroot "s5c_$(basename $rel)" $FILES); printf '\n# dep edit\n' >> "$r/$rel"; rc=$(run "$S19" "$r" $G)
  [ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && mod_absent "$r" && grep -q "v0.32.0 original" "$r.log" \
    && ok "S5c modified read-only dependency $(basename $rel) refused before anything is touched" || bad "S5c $rel rc=$rc"
done

# S6 tampered applier / module -> refuse before touching anything
for f in p019_apply.py gb10_mhc_fn_bf16.py; do
  T="$WORK/tamper"; rm -rf "$T"; cp -R "$P" "$T"; printf '\n# tamper\n' >> "$T/files/$f"
  r=$(mkroot "s6_$f" $FILES); rc=$(run "$T/019-gb10-mhc-fn-bf16.sh" "$r" $G)
  [ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && mod_absent "$r" && ok "S6 tampered files/$f refused, untouched" || bad "S6 $f rc=$rc"
done

# S8 rollback: simulated failure of the SECOND applier call (tilelang_kernels.py) after tilelang.py was edited and the
# module installed -> both files restored byte for byte, a pre-existing module's bytes restored, a new one removed
shim_make "$WORK/shim" "p019_apply.py apply tk" fail
r=$(mkroot s8 $FILES); echo "# old module bytes" > "$r/_gb10_mhc_fn_bf16.py"
rc=$(SHIM="$WORK/shim:" run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && grep -qx "# old module bytes" "$r/_gb10_mhc_fn_bf16.py" \
  && no_leftovers "$r" && grep -q "restored to their previous bytes" "$r.log" \
  && ok "S8 failure of the 2nd applier -> BOTH files restored + a pre-existing module's bytes restored" || { bad "S8 rc=$rc"; tail -4 "$r.log"; }
r=$(mkroot s8b $FILES); rc=$(SHIM="$WORK/shim:" run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && mod_absent "$r" && no_leftovers "$r" \
  && ok "S8b rollback removes a module this run installed" || bad "S8b rc=$rc"
shim_make "$WORK/shim2" "p019_apply.py apply tl" corrupt
r=$(mkroot s8c $FILES); rc=$(SHIM="$WORK/shim2:" run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "!= expected" "$r.log" \
  && ok "S8c result sha != pin after the tilelang.py applier -> rollback, untouched" || bad "S8c rc=$rc"
shim_make "$WORK/shim3" "p019_apply.py apply tk" corrupt
r=$(mkroot s8d $FILES); rc=$(SHIM="$WORK/shim3:" run "$S19" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && mod_absent "$r" && no_leftovers "$r" \
  && ok "S8d result sha != pin after the tilelang_kernels.py applier -> rollback of BOTH files" || bad "S8d rc=$rc"
# S9 patched files compile; the applier CLI round-trips
r=$(mkroot s9 $FILES); run "$S19" "$r" $G >/dev/null
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$r/$TL_REL" "$r/$TK_REL" "$r/_gb10_mhc_fn_bf16.py" \
  && python3 "$P/files/p019_apply.py" revert tl "$r/$TL_REL" >/dev/null && python3 "$P/files/p019_apply.py" revert tk "$r/$TK_REL" >/dev/null \
  && [ "$(sha "$r/$TL_REL")" = "$TLO" ] && [ "$(sha "$r/$TK_REL")" = "$TKO" ] && ok "S9 patched files + module parse; applier revert restores both originals byte for byte" || bad "S9"
pmhc_summary
