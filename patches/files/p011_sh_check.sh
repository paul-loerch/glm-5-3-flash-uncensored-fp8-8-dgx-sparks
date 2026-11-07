#!/usr/bin/env bash
# [patch-011 mhc-prenorm-bm2-threshold] .sh scenario checks (v0.32.0 port). Runs patches/011 against scratch VLLM_ROOT trees
# seeded with the v0.32.0 tilelang_kernels.py; never touches a real vLLM install. Needs bash, python3 (no torch).
#   bash files/p011_sh_check.sh --tree <v0.32.0>/vllm --work /tmp/p011-sh-check
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; TREE=""; WORK=""
while [ $# -gt 0 ]; do
  case "$1" in --tree) TREE="$2"; shift 2 ;; --work) WORK="$2"; shift 2 ;; --patches) PDIR="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;; esac
done
[ -d "$TREE" ] && [ -n "$WORK" ] || { echo "usage: $0 --tree <vllm dir> --work DIR [--patches DIR]" >&2; exit 2; }
. "$PDIR/files/pmhc_sh_lib.sh"; pmhc_init
TK_REL=model_executor/kernels/mhc/tilelang_kernels.py
M11="# [patch-011 mhc-prenorm-bm2-threshold]"
S11="$P/011-mhc-prenorm-bm2-threshold.sh"
no_leftovers() { ! ls "$1/$(dirname "$TK_REL")"/*.p011.* >/dev/null 2>&1; }
ORIG=$(sha "$TREE/$TK_REL")
G=VLLM_GB10_MHC_BM2_MIN_T=128

[ "$ORIG" = "$(pin 011 TK_ORIG)" ] && ok "S0 the tree's tilelang_kernels.py is the pinned v0.32.0 original" || bad "S0 sha $ORIG != TK_ORIG"

r=$(mkroot s1 $TK_REL); rc=$(run "$S11" "$r")
[ "$rc" = 0 ] && [ "$(sha "$r/$TK_REL")" = "$ORIG" ] && grep -q "skipped" "$r.log" && ok "S1 gate unset: skip, untouched" || bad "S1 rc=$rc"

r=$(mkroot s2 $TK_REL); rc=$(run "$S11" "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$TK_REL")" = "$(pin 011 TK_P011)" ] && [ "$(grep -cF "$M11" "$r/$TK_REL")" = 4 ] && no_leftovers "$r" \
  && grep -q "v0.32.0 original + 011" "$r.log" && ok "S2 apply on the original: sha == pinned TK_P011, 4 markers, no backups left" \
  || { bad "S2 rc=$rc"; tail -5 "$r.log"; }
S2=$r
rc=$(run "$S11" "$S2" $G)
[ "$rc" = 0 ] && [ "$(sha "$S2/$TK_REL")" = "$(pin 011 TK_P011)" ] && grep -q "already carries patch 011" "$S2.log" && no_leftovers "$S2" \
  && ok "S3 idempotent re-run" || bad "S3 rc=$rc"
rc=$(run "$S11" "$S2")
[ "$rc" = 0 ] && grep -q "unset but tilelang_kernels.py already carries patch 011" "$S2.log" && [ "$(sha "$S2/$TK_REL")" = "$(pin 011 TK_P011)" ] \
  && ok "S3b gate unset on a patched file: note + skip, untouched" || bad "S3b rc=$rc"
r=$(mkroot s3c $TK_REL); cp "$S2/$TK_REL" "$r/$TK_REL"; printf '\n# stale 011 revision\n' >> "$r/$TK_REL"
before=$(sha "$r/$TK_REL"); rc=$(run "$S11" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TK_REL")" = "$before" ] && no_leftovers "$r" && grep -q "not one of this revision's pinned results" "$r.log" \
  && ok "S3c stale 011 edit refused, file untouched" || bad "S3c rc=$rc"

# S4 stacking with 019 (its applier builds the 019 base; the real 011 -> 019 order is in p019_sh_check.sh)
r=$(mkroot s4 $TK_REL); python3 "$P/files/p019_apply.py" apply tk "$r/$TK_REL" >/dev/null
rc=$(run "$S11" "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$TK_REL")" = "$(pin 011 TK_P011_P019)" ] && grep -q "patch-019 result + 011" "$r.log" && no_leftovers "$r" \
  && ok "S4 apply on the 019 result: sha == pinned TK_P011_P019" || { bad "S4 rc=$rc"; tail -4 "$r.log"; }
rc=$(run "$S11" "$r" $G)
[ "$rc" = 0 ] && grep -q "already carries patch 011" "$r.log" && [ "$(sha "$r/$TK_REL")" = "$(pin 011 TK_P011_P019)" ] \
  && ok "S4b 011+019 result: idempotent" || bad "S4b rc=$rc"

# S5 unexpected target (local edit) -> refuse, untouched
r=$(mkroot s5 $TK_REL); printf '\n# local edit\n' >> "$r/$TK_REL"; before=$(sha "$r/$TK_REL"); rc=$(run "$S11" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TK_REL")" = "$before" ] && no_leftovers "$r" && grep -q "is neither the v0.32.0 original" "$r.log" \
  && ok "S5 unexpected tilelang_kernels.py refused, untouched" || bad "S5 rc=$rc"
r=$(mkroot s5b); rc=$(run "$S11" "$r" $G)
[ "$rc" != 0 ] && grep -q "not found" "$r.log" && ok "S5b missing target refused" || bad "S5b rc=$rc"

# S6 tampered applier -> refuse before touching anything
T="$WORK/tamper"; rm -rf "$T"; cp -R "$P" "$T"; printf '\n# tamper\n' >> "$T/files/p011_apply.py"
r=$(mkroot s6 $TK_REL); rc=$(run "$T/011-mhc-prenorm-bm2-threshold.sh" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TK_REL")" = "$ORIG" ] && ok "S6 tampered files/p011_apply.py refused, untouched" || bad "S6 rc=$rc"

# S7 threshold validation (before anything is touched); 128 and larger accepted
for v in abc 0 100 127 -5 1.5 99999999999 "1 2"; do
  r=$(mkroot "s7_$v" $TK_REL); rc=$(run "$S11" "$r" VLLM_GB10_MHC_BM2_MIN_T="$v")
  if [ "$rc" != 0 ] && [ "$(sha "$r/$TK_REL")" = "$ORIG" ]; then :; else bad "S7 BM2_MIN_T='$v' rc=$rc"; fi
done
r=$(mkroot s7ok $TK_REL); rc=$(run "$S11" "$r" VLLM_GB10_MHC_BM2_MIN_T=2048)
[ "$rc" = 0 ] && [ "$(sha "$r/$TK_REL")" = "$(pin 011 TK_P011)" ] \
  && ok "S7 abc/0/100/127/-5/1.5/99999999999/'1 2' refused untouched; 2048 accepted (the patched file does not depend on the value)" \
  || bad "S7 ok rc=$rc"

# S8 rollback: simulated applier failure, then a result-sha mismatch -> the file is restored byte for byte
shim_make "$WORK/shim" "p011_apply.py apply" fail
r=$(mkroot s8 $TK_REL); rc=$(SHIM="$WORK/shim:" run "$S11" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TK_REL")" = "$ORIG" ] && no_leftovers "$r" && grep -q "restored" "$r.log" \
  && ok "S8 applier failure -> rollback restores the file" || { bad "S8 rc=$rc"; tail -4 "$r.log"; }
shim_make "$WORK/shim2" "p011_apply.py apply" corrupt
r=$(mkroot s8c $TK_REL); rc=$(SHIM="$WORK/shim2:" run "$S11" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$TK_REL")" = "$ORIG" ] && no_leftovers "$r" && grep -q "!= pinned" "$r.log" \
  && ok "S8c result sha != pin after the applier -> rollback, untouched" || bad "S8c rc=$rc"

# S9 the patched file compiles and the applier's CLI round-trips
r=$(mkroot s9 $TK_REL); run "$S11" "$r" $G >/dev/null
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$r/$TK_REL" && python3 "$P/files/p011_apply.py" revert "$r/$TK_REL" >/dev/null \
  && [ "$(sha "$r/$TK_REL")" = "$ORIG" ] && ok "S9 patched file parses; applier revert restores the original byte for byte" || bad "S9"
pmhc_summary
