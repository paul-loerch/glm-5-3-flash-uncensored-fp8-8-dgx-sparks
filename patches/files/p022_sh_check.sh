#!/usr/bin/env bash
# [patch-022 gb10-mhc-prefill-rowsplit] .sh scenario checks (v0.32.0 port). Runs patches/022 (and 011 / 019 for stacking, in
# the entrypoint's numeric order) against scratch VLLM_ROOT trees seeded with the v0.32.0 originals; never touches a real
# vLLM install. Needs bash, python3 (no torch).
#   bash files/p022_sh_check.sh --tree <v0.32.0>/vllm --work /tmp/p022-sh-check
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
M22="# [patch-022 gb10-mhc-prefill-rowsplit]"
S22="$P/022-gb10-mhc-prefill-rowsplit.sh"; S19="$P/019-gb10-mhc-fn-bf16.sh"; S11="$P/011-mhc-prenorm-bm2-threshold.sh"
no_leftovers() { ! ls "$1"/*.p022.* "$1/$(dirname "$TL_REL")"/*.p022.* >/dev/null 2>&1; }
mod_absent() { [ ! -e "$1/_gb10_mhc_prs.py" ]; }
mod_ok() { [ "$(sha "$1/_gb10_mhc_prs.py")" = "$(sha "$P/files/gb10_mhc_prs.py")" ]; }
ORIG=$(sha "$TREE/$TL_REL")
G=VLLM_GB10_MHC_PRS=1; E11=VLLM_GB10_MHC_BM2_MIN_T=128; E19=VLLM_GB10_MHC_FN_BF16=1

[ "$ORIG" = "$(pin 022 TL_ORIG)" ] && [ "$(sha "$TREE/$TK_REL")" = "$(pin 011 TK_ORIG)" ] \
  && ok "S0 the tree's tilelang.py / tilelang_kernels.py are the pinned v0.32.0 originals" || bad "S0 shas"

r=$(mkroot s1 $FILES); rc=$(run "$S22" "$r")
[ "$rc" = 0 ] && [ "$(sha "$r/$TL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "skipped" "$r.log" \
  && ok "S1 gate unset: skip, untouched, no module" || bad "S1 rc=$rc"

r=$(mkroot s2 $FILES); rc=$(run "$S22" "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$TL_REL")" = "$(pin 022 TL_ORIG_P022)" ] && [ "$(grep -cF "$M22" "$r/$TL_REL")" = 2 ] \
  && mod_ok "$r" && no_leftovers "$r" && grep -q "v0.32.0 original + 022" "$r.log" \
  && ok "S2 apply on the original: sha == pinned TL_ORIG_P022, module installed, no backups left" || { bad "S2 rc=$rc"; tail -5 "$r.log"; }
S2=$r
rc=$(run "$S22" "$S2" $G)
[ "$rc" = 0 ] && [ "$(sha "$S2/$TL_REL")" = "$(pin 022 TL_ORIG_P022)" ] && grep -q "already carries patch 022" "$S2.log" && no_leftovers "$S2" \
  && ok "S3 idempotent re-run" || bad "S3 rc=$rc"
rc=$(run "$S22" "$S2")
[ "$rc" = 0 ] && grep -q "unset but tilelang.py already carries patch 022" "$S2.log" && [ "$(sha "$S2/$TL_REL")" = "$(pin 022 TL_ORIG_P022)" ] \
  && ok "S3b gate unset on a patched file: note + skip, untouched" || bad "S3b rc=$rc"
r=$(mkroot s3c $FILES); cp "$S2/$TL_REL" "$r/$TL_REL"; printf '\n# stale 022 revision\n' >> "$r/$TL_REL"
before=$(sha "$r/$TL_REL"); rc=$(run "$S22" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "not one of this revision's pinned results" "$r.log" \
  && ok "S3c stale 022 edit refused, file untouched, no module" || bad "S3c rc=$rc"
echo "# old module revision" > "$S2/_gb10_mhc_prs.py"; rc=$(run "$S22" "$S2" $G)
[ "$rc" = 0 ] && mod_ok "$S2" && [ "$(sha "$S2/$TL_REL")" = "$(pin 022 TL_ORIG_P022)" ] && no_leftovers "$S2" \
  && ok "S3d installed target + stale module: module replaced by the pinned bytes" || bad "S3d rc=$rc"

# M the entrypoint's numeric order on fresh originals for every {011, 019, 022} gate set; each 022 step re-run
for g11 in 0 1; do for g19 in 0 1; do for g22 in 0 1; do
  e=""; [ $g11 = 1 ] && e="$e $E11"; [ $g19 = 1 ] && e="$e $E19"; [ $g22 = 1 ] && e="$e $G"
  case "$g19$g22" in 00) tl=TL_ORIG;; 10) tl=TL_P019;; 01) tl=TL_ORIG_P022;; 11) tl=TL_P019_P022;; esac
  want=$(pin 022 $tl)
  r=$(mkroot "m_$g11$g19$g22" $FILES); a=$(run "$S11" "$r" $e); c=$(run "$S19" "$r" $e); d=$(run "$S22" "$r" $e)
  if [ "$a$c$d" = 000 ] && [ "$(sha "$r/$TL_REL")" = "$want" ] && no_leftovers "$r" && { [ $g22 = 1 ] && mod_ok "$r" || { [ $g22 = 0 ] && mod_absent "$r"; }; }; then
    d2=$(run "$S22" "$r" $e)
    if [ "$d2" = 0 ] && [ "$(sha "$r/$TL_REL")" = "$want" ]; then ok "M 011=$g11 019=$g19 022=$g22: entrypoint order -> $tl (${want:0:12}...), 022 re-run idempotent"
    else bad "M $g11$g19$g22 re-run rc=$d2"; fi
  else bad "M $g11$g19$g22: rc 011=$a 019=$c 022=$d sha $(sha "$r/$TL_REL" | cut -c1-12) want ${want:0:12}"; tail -3 "$r.log"; fi
done; done; done
# M2 a REUSED container holding 011 + 019 + 022 booted again with the same gates: every step skips (idempotent)
r="$WORK/roots/m_111"; before=$(sha "$r/$TL_REL"); a=$(run "$S11" "$r" $E11 $E19 $G); c=$(run "$S19" "$r" $E11 $E19 $G); d=$(run "$S22" "$r" $E11 $E19 $G)
[ "$a$c$d" = 000 ] && [ "$(sha "$r/$TL_REL")" = "$before" ] && ok "M2 reused 011+019+022 container: all three skip, file intact" || bad "M2 rc $a$c$d"

# S5 unexpected targets / modified tilelang_kernels.py -> refuse, untouched, no module
r=$(mkroot s5 $FILES); printf '\n# local edit\n' >> "$r/$TL_REL"; before=$(sha "$r/$TL_REL"); rc=$(run "$S22" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "is neither the v0.32.0 original" "$r.log" \
  && ok "S5 unexpected tilelang.py refused, untouched" || bad "S5 rc=$rc"
r=$(mkroot s5b $FILES); printf '\n# kernel edit\n' >> "$r/$TK_REL"; rc=$(run "$S22" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "tilelang_kernels.py sha256" "$r.log" \
  && ok "S5b modified tilelang_kernels.py refused before anything is touched" || bad "S5b rc=$rc"
# S5c every pinned tilelang_kernels.py variant is accepted (original, 011, 019, 011+019 results)
for v in "" 11 19 "11 19"; do
  r=$(mkroot "s5c_${v// /_}" $FILES); e=""; for x in $v; do [ $x = 11 ] && e="$e $E11"; [ $x = 19 ] && e="$e $E19"; done
  [ -n "$v" ] && { for x in $v; do [ $x = 11 ] && run "$S11" "$r" $e >/dev/null; done; for x in $v; do [ $x = 19 ] && run "$S19" "$r" $e >/dev/null; done; }
  rc=$(run "$S22" "$r" $G $e)
  [ "$rc" = 0 ] && grep -q "tilelang.py is the" "$r.log" && continue || bad "S5c tilelang_kernels.py variant '$v' rc=$rc"
done
ok "S5c all four pinned tilelang_kernels.py variants (original / 011 / 019 / 011+019) accepted"

# S6 tampered applier / module -> refuse before touching anything
for f in p022_apply.py gb10_mhc_prs.py; do
  T="$WORK/tamper"; rm -rf "$T"; cp -R "$P" "$T"; printf '\n# tamper\n' >> "$T/files/$f"
  r=$(mkroot "s6_$f" $FILES); rc=$(run "$T/022-gb10-mhc-prefill-rowsplit.sh" "$r" $G)
  [ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$ORIG" ] && mod_absent "$r" && ok "S6 tampered files/$f refused, untouched" || bad "S6 $f rc=$rc"
done

# S7 MIN_T validation (before anything is touched); empty = default
for v in abc 100 255 1048577 99999999999 -5 1.5; do
  r=$(mkroot "s7_$v" $FILES); rc=$(run "$S22" "$r" $G VLLM_GB10_MHC_PRS_MIN_T="$v")
  [ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$ORIG" ] && mod_absent "$r" || bad "S7 MIN_T=$v rc=$rc"
done
r=$(mkroot s7ok $FILES); rc=$(run "$S22" "$r" $G VLLM_GB10_MHC_PRS_MIN_T=256); rc2=$(run "$(mkroot s7e $FILES >/dev/null; echo "$S22")" "$WORK/roots/s7e" $G VLLM_GB10_MHC_PRS_MIN_T=)
[ "$rc" = 0 ] && [ "$rc2" = 0 ] && grep -q "MIN_T=256" "$r.log" && grep -q "MIN_T=1024" "$WORK/roots/s7e.log" \
  && ok "S7 MIN_T: abc/100/255/1048577/99999999999/-5/1.5 refused untouched; 256 and empty (-> 1024) accepted" || bad "S7 ok rc=$rc rc2=$rc2"

# S7b VLLM_USE_DEEP_GEMM warning (DeepGEMM would bypass the hook): warns, still applies
r=$(mkroot s7b $FILES); rc=$(run "$S22" "$r" $G VLLM_USE_DEEP_GEMM=1)
[ "$rc" = 0 ] && grep -q "stays inert" "$r.log" && ok "S7b VLLM_USE_DEEP_GEMM=1: warns that 022 would stay inert, applies" || bad "S7b rc=$rc"
r=$(mkroot s7c $FILES); rc=$(run "$S22" "$r" $G VLLM_USE_DEEP_GEMM=0)
[ "$rc" = 0 ] && ! grep -q "stays inert" "$r.log" && ok "S7c VLLM_USE_DEEP_GEMM=0: no warning" || bad "S7c rc=$rc"

# S8 rollback after the module was installed (simulated applier failure): the target AND a PRE-EXISTING module's bytes are
# restored; a module that did not exist is removed; a result-sha mismatch rolls back too
shim_make "$WORK/shim" "p022_apply.py apply" fail
r=$(mkroot s8 $FILES); echo "# old module bytes" > "$r/_gb10_mhc_prs.py"
rc=$(SHIM="$WORK/shim:" run "$S22" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$ORIG" ] && grep -qx "# old module bytes" "$r/_gb10_mhc_prs.py" && no_leftovers "$r" \
  && grep -q "restored to their previous bytes" "$r.log" && ok "S8 rollback restores the target + a pre-existing module's bytes" || { bad "S8 rc=$rc"; tail -4 "$r.log"; }
r=$(mkroot s8b $FILES); rc=$(SHIM="$WORK/shim:" run "$S22" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$ORIG" ] && mod_absent "$r" && no_leftovers "$r" && ok "S8b rollback removes a module this run installed" || bad "S8b rc=$rc"
shim_make "$WORK/shim2" "p022_apply.py apply" corrupt
r=$(mkroot s8c $FILES); rc=$(SHIM="$WORK/shim2:" run "$S22" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TL_REL")" = "$ORIG" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "!= pinned" "$r.log" \
  && ok "S8c result sha != pin after the applier -> rollback, untouched" || bad "S8c rc=$rc"

# S9 TERM during the applier: the child is reaped, exit 130, everything restored
cat > "$WORK/shim_term_python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *p022_apply.py\ apply*) exec sleep 30 ;; esac
exec "$REALPY" "\$@"
EOF
mkdir -p "$WORK/shim_term"; mv "$WORK/shim_term_python3" "$WORK/shim_term/python3"; chmod +x "$WORK/shim_term/python3"
r=$(mkroot s9 $FILES); ( env -i PATH="$WORK/shim_term:$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" $G bash "$S22" >>"$r.log" 2>&1 & echo $! > "$WORK/s9.pid"; wait ) &
sleep 2; pid=$(cat "$WORK/s9.pid" 2>/dev/null); [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null; sleep 2; wait 2>/dev/null
[ "$(sha "$r/$TL_REL")" = "$ORIG" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "restored to their previous bytes" "$r.log" \
  && ok "S9 TERM during the applier (child reaped): rollback ran - target restored, module removed, no backups left" || { bad "S9 TERM scenario"; tail -5 "$r.log"; }

# S10 byte-exact revert of every result through the applier CLI
for st in "" 19; do
  r=$(mkroot "s10_$st" $FILES); [ -n "$st" ] && run "$S19" "$r" $E19 >/dev/null
  base=$(sha "$r/$TL_REL"); run "$S22" "$r" $G $([ -n "$st" ] && echo $E19) >/dev/null
  python3 "$P/files/p022_apply.py" revert "$r/$TL_REL" >/dev/null && [ "$(sha "$r/$TL_REL")" = "$base" ] \
    && ok "S10 applier revert of the ${st:-original}+022 result restores the base byte for byte" || bad "S10 $st"
done
pmhc_summary
