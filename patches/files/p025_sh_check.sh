#!/usr/bin/env bash
# [patch-025 gb10-mhc-prefill-sp] .sh scenario checks (v0.32.0 port). Runs patches/025 (and 011 / 019 / 022 for the
# stacked tilelang.py / tilelang_kernels.py states, in the entrypoint's numeric order) against scratch VLLM_ROOT trees
# seeded with the v0.32.0 originals of common/model.py + the read-only dependency files; never touches a real vLLM
# install. Needs bash, python3 (no torch), sha256sum or shasum.
#   bash files/p025_sh_check.sh --vllm <v0.32.0 tree>/vllm --work /tmp/p025-sh-check
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM=""; WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --vllm) VLLM="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --patches) PDIR="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -d "$VLLM" ] && [ -n "$WORK" ] || { echo "usage: $0 --vllm <vllm-src>/vllm --work DIR [--patches DIR]" >&2; exit 2; }
MODEL_REL=models/glm5next/common/model.py
MR_REL=model_executor/layers/fused_moe/runner/moe_runner.py
LINEAR_REL=model_executor/layers/linear.py
MHC_REL=model_executor/layers/mhc.py
TK_REL=model_executor/kernels/mhc/tilelang_kernels.py
TL_REL=model_executor/kernels/mhc/tilelang.py
CFG_REL=model_executor/layers/fused_moe/config.py
# rev 2: dropped sequence_parallel.py (unused); the hard-pinned deps are moe_runner / linear / mhc / tilelang_kernels
# / fused_moe-config. tilelang.py is checked for TL_KNOWN membership (its own case below).
DEP_RELS="$MR_REL $LINEAR_REL $MHC_REL $CFG_REL"
# the 019 readers (helper / jit_warmup / tilelang_utils) so the real 011 / 019 / 022 scripts can run in the same roots
EXTRA_RELS="model_executor/warmup/jit_warmup_tilelang_helper.py model_executor/warmup/jit_warmup.py tilelang_utils/__init__.py"
M25="# [patch-025 gb10-mhc-prefill-sp]"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
pin() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$PDIR/025-gb10-mhc-prefill-sp.sh"; }
tl_known_set() { sed -n '/^TL_KNOWN="/,/"/p' "$PDIR/025-gb10-mhc-prefill-sp.sh" | grep -oE '[0-9a-f]{64}'; }
tk_known_set() { sed -n '/^TK_KNOWN="/,/"/p' "$PDIR/025-gb10-mhc-prefill-sp.sh" | grep -oE '[0-9a-f]{64}'; }
rm -rf "$WORK"; mkdir -p "$WORK/roots"
P="$WORK/patches"; cp -R "$PDIR" "$P"; rm -rf "$P/files/__pycache__"
ORIG=$(sha "$VLLM/$MODEL_REL"); REALPY="$(command -v python3)"
pass=0; fail=0
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }
mkroot() {
  local r="$WORK/roots/$1"; rm -rf "$r"
  for rel in "$MODEL_REL" $DEP_RELS "$TL_REL" "$TK_REL" $EXTRA_RELS; do mkdir -p "$r/$(dirname "$rel")"; cp "$VLLM/$rel" "$r/$rel"; done
  echo "$r"
}
run() { local s=$1 r=$2; shift 2; env -i PATH="${SHIM:-}$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" "$@" bash "$s" >>"$r.log" 2>&1; echo $?; }
run25() { local r=$1; shift; run "$P/025-gb10-mhc-prefill-sp.sh" "$r" "$@"; }
run11() { local r=$1; shift; run "$P/011-mhc-prenorm-bm2-threshold.sh" "$r" "$@"; }
run19() { local r=$1; shift; run "$P/019-gb10-mhc-fn-bf16.sh" "$r" "$@"; }
run22() { local r=$1; shift; run "$P/022-gb10-mhc-prefill-rowsplit.sh" "$r" "$@"; }
mod_absent() { [ ! -e "$1/_gb10_prefill_sp.py" ]; }
mod_ok() { [ "$(sha "$1/_gb10_prefill_sp.py")" = "$(sha "$P/files/gb10_prefill_sp.py")" ]; }
no_leftovers() { ! ls "$1"/*.p025.* "$1/$(dirname "$MODEL_REL")"/*.p025.* >/dev/null 2>&1; }
G=VLLM_GB10_MHC_SP=sp; GR=VLLM_GB10_MHC_SP=ref; E11=VLLM_GB10_MHC_BM2_MIN_T=128; E19=VLLM_GB10_MHC_FN_BF16=1; E22=VLLM_GB10_MHC_PRS=1

[ "$ORIG" = "$(pin MODEL_ORIG)" ] && ok "S0 --vllm model.py is the pinned v0.32.0 original" || bad "S0 model sha $ORIG != MODEL_ORIG"
for rel in $DEP_RELS; do
  nm=$(basename "$rel"); case "$nm" in moe_runner.py) k=MR_ORIG;; linear.py) k=LINEAR_ORIG;; mhc.py) k=MHC_ORIG;; config.py) k=MOECFG_ORIG;; esac
  [ "$(sha "$VLLM/$rel")" = "$(pin $k)" ] && ok "S0 dep $nm == $k" || bad "S0 dep $nm sha"
done
TLS=$(sha "$VLLM/$TL_REL"); tl_known_set | grep -qx "$TLS" && ok "S0 dep tilelang.py in TL_KNOWN" || bad "S0 dep tilelang.py sha $TLS not in TL_KNOWN"
TKS=$(sha "$VLLM/$TK_REL"); tk_known_set | grep -qx "$TKS" && ok "S0 dep tilelang_kernels.py in TK_KNOWN" || bad "S0 dep tilelang_kernels.py sha $TKS not in TK_KNOWN"
[ "$(tl_known_set | wc -l | tr -d ' ')" = 4 ] && [ "$(tk_known_set | wc -l | tr -d ' ')" = 4 ] && ok "S0 TL_KNOWN and TK_KNOWN pin 4 states each" || bad "S0 known-set sizes"

r=$(mkroot s1); rc=$(run25 "$r")
[ "$rc" = 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "skipped" "$r.log" \
  && ok "S1 gate unset: skip, untouched, no module" || bad "S1 rc=$rc"

r=$(mkroot s2); rc=$(run25 "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$(pin MODEL_ORIG_P025)" ] && [ "$(grep -cF "$M25" "$r/$MODEL_REL")" = 4 ] \
  && mod_ok "$r" && no_leftovers "$r" && grep -q "v0.32.0 original + 025" "$r.log" \
  && ok "S2 apply on the original (mode sp): sha == pinned MODEL_ORIG_P025, module installed, no backups" || { bad "S2 rc=$rc"; tail -5 "$r.log"; }
S2=$r
rc=$(run25 "$S2" $G)
[ "$rc" = 0 ] && [ "$(sha "$S2/$MODEL_REL")" = "$(pin MODEL_ORIG_P025)" ] && grep -q "already carries patch 025" "$S2.log" && no_leftovers "$S2" \
  && ok "S3 idempotent re-run" || bad "S3 rc=$rc"
rc=$(run25 "$S2")
[ "$rc" = 0 ] && grep -q "unset but model.py already carries patch 025" "$S2.log" && [ "$(sha "$S2/$MODEL_REL")" = "$(pin MODEL_ORIG_P025)" ] \
  && ok "S3b gate unset on a patched file: note + skip, untouched" || bad "S3b rc=$rc"
# S3r mode=ref applies the same edits (sha identical: the mode is a runtime gate, not a source change)
r=$(mkroot s3r); rc=$(run25 "$r" $GR)
[ "$rc" = 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$(pin MODEL_ORIG_P025)" ] && mod_ok "$r" && grep -q "VLLM_GB10_MHC_SP=ref" "$r.log" \
  && ok "S3r mode=ref: same pinned result (runtime gate), module installed" || bad "S3r rc=$rc"
# S3c a reused container carrying a stale 025 edit -> refuse, untouched
r=$(mkroot s3c); cp "$S2/$MODEL_REL" "$r/$MODEL_REL"; printf '\n# stale 025 revision\n' >> "$r/$MODEL_REL"
before=$(sha "$r/$MODEL_REL"); rc=$(run25 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "not one of this revision's pinned results" "$r.log" \
  && ok "S3c stale 025 edit refused, file untouched, no module" || bad "S3c rc=$rc"
# S3d installed target + a stale module -> module re-installed with the pinned bytes
echo "# old module revision" > "$S2/_gb10_prefill_sp.py"; rc=$(run25 "$S2" $G)
[ "$rc" = 0 ] && mod_ok "$S2" && [ "$(sha "$S2/$MODEL_REL")" = "$(pin MODEL_ORIG_P025)" ] && no_leftovers "$S2" \
  && ok "S3d installed target + stale module: module replaced by the pinned bytes" || bad "S3d rc=$rc"

# M the entrypoint's numeric order (011, 019, 022, 025) on fresh originals: 025 accepts every stacked tilelang.py /
# tilelang_kernels.py state and always yields the same pinned model.py result
for g11 in 0 1; do for g19 in 0 1; do for g22 in 0 1; do
  e=""; [ $g11 = 1 ] && e="$e $E11"; [ $g19 = 1 ] && e="$e $E19"; [ $g22 = 1 ] && e="$e $E22"
  r=$(mkroot "m_$g11$g19$g22"); a=$(run11 "$r" $e); b=$(run19 "$r" $e); c=$(run22 "$r" $e); d=$(run25 "$r" $e $G)
  want=$(pin MODEL_ORIG_P025)
  if [ "$a$b$c$d" = 0000 ] && [ "$(sha "$r/$MODEL_REL")" = "$want" ] && no_leftovers "$r" && mod_ok "$r"; then
    d2=$(run25 "$r" $e $G)
    if [ "$d2" = 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$want" ]; then ok "M 011=$g11 019=$g19 022=$g22: entrypoint order -> pinned MODEL_ORIG_P025, 025 re-run idempotent"
    else bad "M $g11$g19$g22 re-run rc=$d2"; fi
  else bad "M $g11$g19$g22: rc 011=$a 019=$b 022=$c 025=$d"; tail -3 "$r.log"; fi
done; done; done

# S5 unexpected target (local edit) / changed dependency -> refuse, untouched, no module
r=$(mkroot s5); printf '\n# local edit\n' >> "$r/$MODEL_REL"; before=$(sha "$r/$MODEL_REL"); rc=$(run25 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "is not the v0.32.0 original" "$r.log" \
  && ok "S5 unexpected model.py refused, untouched" || bad "S5 rc=$rc"
for rel in $DEP_RELS; do
  nm=$(basename "$rel"); r=$(mkroot "s5_$nm"); printf '\n# dep edit\n' >> "$r/$rel"; rc=$(run25 "$r" $G)
  [ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "$nm sha256" "$r.log" \
    && ok "S5 changed dependency $nm refused before any write" || bad "S5 dep $nm rc=$rc"
done
# S5t a tilelang.py / tilelang_kernels.py that is not in TL_KNOWN / TK_KNOWN -> refuse (the per-row prenorm routing sp relies on
# is unverified for it)
r=$(mkroot s5_tl); printf '\n# tl edit\n' >> "$r/$TL_REL"; rc=$(run25 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "not a known mHC-dispatch variant" "$r.log" \
  && ok "S5t unknown tilelang.py refused before any write" || bad "S5t tl rc=$rc"
r=$(mkroot s5_tk); printf '\n# tk edit\n' >> "$r/$TK_REL"; rc=$(run25 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "not a known v0.32.0 variant" "$r.log" \
  && ok "S5t unknown tilelang_kernels.py refused before any write" || bad "S5t tk rc=$rc"
# S5v a model.py that a 025-unaware patch edited first (any other revision) -> refused with the reason
r=$(mkroot s5_v); python3 - "$r/$MODEL_REL" <<'PY'
import sys
p = sys.argv[1]; t = open(p).read(); open(p, "w").write(t.replace("class Glm5NextMLP(nn.Module):", "class Glm5NextMLP(nn.Module):  # edited", 1))
PY
rc=$(run25 "$r" $G)
[ "$rc" != 0 ] && grep -q "is not the v0.32.0 original" "$r.log" && ok "S5v a model.py edited by another patch before 025 is refused (unverified anchors / aux contract)" || bad "S5v rc=$rc"
# S5p a changed patch-022 module source -> refuse (the neutralisation contract is locked to its sha)
T5="$WORK/tamper_prs"; rm -rf "$T5"; cp -R "$P" "$T5"; printf '\n# prs edit\n' >> "$T5/files/gb10_mhc_prs.py"
r=$(mkroot s5_prs); rc=$(run "$T5/025-gb10-mhc-prefill-sp.sh" "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "patch 022" "$r.log" \
  && ok "S5p changed patch-022 module source refused" || bad "S5p rc=$rc"

# S6 tampered applier / module -> refuse before touching anything
for f in p025_apply.py gb10_prefill_sp.py; do
  T="$WORK/tamper_$f"; rm -rf "$T"; cp -R "$P" "$T"; printf '\n# tamper\n' >> "$T/files/$f"
  r=$(mkroot "s6_$f"); rc=$(run "$T/025-gb10-mhc-prefill-sp.sh" "$r" $G)
  [ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && ok "S6 tampered files/$f refused, untouched" || bad "S6 $f rc=$rc"
done

# S7 MODE / MIN_T / RS validation (before anything is touched)
for v in xyz 2 sp,ref; do
  r=$(mkroot "s7m_$v"); rc=$(run25 "$r" VLLM_GB10_MHC_SP="$v")
  [ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" || { bad "S7 mode=$v rc=$rc"; }
done
# rev 2: the MIN_T floor is 1024 -> 512 and 1023 are now REJECTED (were accepted in rev 1).
for v in abc 100 255 512 1023 1048577 -5; do
  r=$(mkroot "s7t_$v"); rc=$(run25 "$r" $G VLLM_GB10_MHC_SP_MIN_T="$v")
  [ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" || { bad "S7 MIN_T=$v rc=$rc"; }
done
r=$(mkroot s7rs); rc=$(run25 "$r" $G VLLM_GB10_MHC_SP_RS=bogus)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] || bad "S7 RS=bogus rc=$rc"
# S7d DeepGEMM on -> refuse untouched
r=$(mkroot s7dg); rc=$(run25 "$r" $G VLLM_USE_DEEP_GEMM=1)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && grep -q "VLLM_USE_DEEP_GEMM=0" "$r.log" \
  && ok "S7d VLLM_USE_DEEP_GEMM=1 refused untouched" || bad "S7d rc=$rc"
r=$(mkroot s7okc); rc=$(run25 "$r" $G VLLM_GB10_MHC_SP_RS=canonical VLLM_GB10_MHC_SP_MIN_T=2048)
[ "$rc" = 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$(pin MODEL_ORIG_P025)" ] && grep -q "MIN_T=2048, RS=canonical" "$r.log" \
  && ok "S7 validation: bad mode/MIN_T(incl 512,1023)/RS/DeepGEMM refused untouched; RS=canonical + MIN_T=2048 accepted" || bad "S7 ok rc=$rc"

# S8 rollback after the module was installed (simulated applier failure): the target AND a PRE-EXISTING module's
# bytes are restored; a module that did not exist is removed
mkdir -p "$WORK/shim"
cat > "$WORK/shim/python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *p025_apply.py\ apply*) echo "shim: simulated applier failure" >&2; exit 1 ;; esac
exec "$REALPY" "\$@"
EOF
chmod +x "$WORK/shim/python3"
r=$(mkroot s8); echo "# old module bytes" > "$r/_gb10_prefill_sp.py"
rc=$(SHIM="$WORK/shim:" run25 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && grep -qx "# old module bytes" "$r/_gb10_prefill_sp.py" && no_leftovers "$r" \
  && grep -q "restored to their previous bytes" "$r.log" && ok "S8 rollback restores the target + a pre-existing module's bytes" \
  || { bad "S8 rc=$rc"; tail -4 "$r.log"; }
r=$(mkroot s8b); rc=$(SHIM="$WORK/shim:" run25 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && no_leftovers "$r" \
  && ok "S8b rollback removes a module this run installed" || bad "S8b rc=$rc"
# S8c a result-sha mismatch after the applier -> rollback
mkdir -p "$WORK/shim2"
cat > "$WORK/shim2/python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *p025_apply.py\ apply*) printf '\n# unexpected\n' >> "\${@: -1}"; exit 0 ;; esac
exec "$REALPY" "\$@"
EOF
chmod +x "$WORK/shim2/python3"
r=$(mkroot s8c); rc=$(SHIM="$WORK/shim2:" run25 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "!= pinned" "$r.log" \
  && ok "S8c result sha != pin after the applier -> rollback, untouched" || bad "S8c rc=$rc"

# S9 TERM while the applier runs: child reaped, target restored, new module removed, exit 130
mkdir -p "$WORK/shim3"
cat > "$WORK/shim3/python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *p025_apply.py\ apply*) exec sleep 47.25 ;; esac
exec "$REALPY" "\$@"
EOF
chmod +x "$WORK/shim3/python3"
r=$(mkroot s9)
env -i PATH="$WORK/shim3:$PATH" HOME="$HOME" VLLM_ROOT="$r" $G bash "$P/025-gb10-mhc-prefill-sp.sh" >>"$r.log" 2>&1 &
spid=$!
for _ in $(seq 1 50); do pgrep -f "sleep 47.25" >/dev/null && break; python3 -c "import time; time.sleep(0.1)"; done
kill -TERM "$spid"; wait "$spid"; rc=$?
python3 -c "import time; time.sleep(0.3)"
leftover=$(pgrep -f "sleep 47.25" || true)
[ "$rc" = 130 ] && [ -z "$leftover" ] && [ "$(sha "$r/$MODEL_REL")" = "$ORIG" ] && mod_absent "$r" && no_leftovers "$r" \
  && ok "S9 TERM during the applier: child reaped, target restored, module removed, exit 130" \
  || { bad "S9 rc=$rc leftover=[$leftover]"; tail -4 "$r.log"; [ -n "$leftover" ] && kill $leftover; }

# S10 CLI revert is byte-exact on the pinned result
cp "$WORK/roots/m_000/$MODEL_REL" "$WORK/rev.py"; "$REALPY" "$P/files/p025_apply.py" revert "$WORK/rev.py" >/dev/null
[ "$(sha "$WORK/rev.py")" = "$(pin MODEL_ORIG)" ] && ok "S10 revert(MODEL_ORIG_P025) == MODEL_ORIG" || bad "S10 revert"
echo "sh scenarios: $pass passed, $fail failed"
[ "$fail" = 0 ]
