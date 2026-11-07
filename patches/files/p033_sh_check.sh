#!/usr/bin/env bash
# [patch-033 gb10-mla-w8a16] .sh scenario checks (vLLM v0.32.0 port). Runs patches/033 against scratch VLLM_ROOT trees seeded
# with the files it reads at boot; never touches a real vLLM install and never deletes anything (every scenario gets a fresh
# mktemp directory under --work; clean --work up by hand). Needs bash, python3 (no torch), sha256sum or shasum.
#   bash files/p033_sh_check.sh --vllm <v0.32.0 tree>/vllm --work <new dir> [--bases <dir of the bases>] [--patches <patches dir>]
# --vllm is the v0.32.0 `vllm/` package directory (model.py, mtp.py, linear.py, marlin_utils{,_fp8}.py, _custom_ops.py,
# model_loader/{utils,base_loader}.py, eagle/utils.py: every pinned file is the REAL one, no stand-ins any more).
# The model.py bases (v0.32.0 original; + 025; + 032; + 025 + 032) come from --bases (orig/ p025/ orig_p032/ p025_p032/, each with a
# model.py, e.g. written by `p033_cpu_check.py --dump-bases DIR`) or are derived here through that same option; a base whose
# earlier patch has no applier yet is reported as [SKIP]. The REAL 025 / 032 .sh run over a full tree in the stack check
# (all patch scripts in entrypoint order over one tree), not here: S11 / S12 are always [SKIP] in this script.
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM=""; WORK=""; BASES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --vllm) VLLM="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --bases) BASES="$2"; shift 2 ;;
    --patches) PDIR="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -d "$VLLM" ] && [ -n "$WORK" ] || { echo "usage: $0 --vllm <vllm-src>/vllm --work DIR [--bases DIR] [--patches DIR]" >&2; exit 2; }
mkdir -p "$WORK"
RUN="$(mktemp -d "$WORK/run.XXXXXX")"
REALPY="$(command -v python3)"
M_REL=models/glm5next/common/model.py
MTP_REL=models/glm5next/common/mtp.py
LIN_REL=model_executor/layers/linear.py
MU8_REL=model_executor/layers/quantization/utils/marlin_utils_fp8.py
MU_REL=model_executor/layers/quantization/utils/marlin_utils.py
COPS_REL=_custom_ops.py
LU_REL=model_executor/model_loader/utils.py
BL_REL=model_executor/model_loader/base_loader.py
EAGLE_REL=v1/worker/gpu/spec_decode/eagle/utils.py
DEPS="$M_REL $MTP_REL $LIN_REL $MU8_REL $MU_REL $COPS_REL $LU_REL $BL_REL $EAGLE_REL"
MOD=_gb10_mla_w8a16.py
M33="# [patch-033 gb10-mla-w8a16]"
ALLBASEVARS="MODEL_ORIG MODEL_P025 MODEL_ORIG_P032 MODEL_P025_P032"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
pass=0; fail=0; skp=0
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }
skipped() { echo "[SKIP] $*"; skp=$((skp+1)); }
bdir() {
  case "$1" in
    MODEL_ORIG) echo orig ;; MODEL_P025) echo p025 ;; MODEL_ORIG_P032) echo orig_p032 ;; MODEL_P025_P032) echo p025_p032 ;;
  esac
}

# ---- the patch tree copy under test
P="$RUN/patches"; cp -R "$PDIR" "$P"
SH33="$P/033-gb10-mla-w8a16.sh"
pin_orig() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$PDIR/033-gb10-mla-w8a16.sh"; }
pin() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$SH33"; }
mkdir -p "$RUN/shim"

# ---- the bases
if [ -z "$BASES" ]; then
  BASES="$RUN/bases"; mkdir -p "$BASES"
  "$REALPY" "$PDIR/files/p033_cpu_check.py" --vllm-root "$VLLM" --dump-bases "$BASES" --work "$RUN/dump" >"$RUN/dump.log" 2>&1 \
    || { echo "[FAIL] could not derive the bases (see $RUN/dump.log)"; exit 1; }
fi
BASEVARS=""
for v in $ALLBASEVARS; do
  if [ -f "$BASES/$(bdir "$v")/model.py" ]; then BASEVARS="$BASEVARS $v"; else skipped "S0 base $v: not derivable (the earlier patch's applier is absent or refuses it)"; fi
done
mkroot() {
  local r; r="$(mktemp -d "$RUN/root.$1.XXXXXX")"
  for rel in $DEPS; do
    mkdir -p "$r/$(dirname "$rel")"
    cp "$VLLM/$rel" "$r/$rel"
  done
  : > "$r.log"
  echo "$r"
}
seed() { cp "$BASES/$(bdir "$2")/model.py" "$1/$M_REL"; }
run() { local s=$1 r=$2; shift 2; env -i PATH="${SHIM:-}$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" "$@" bash "$s" >>"$r.log" 2>&1; echo $?; }
run33() { local r=$1; shift; run "$SH33" "$r" "$@"; }
msha() { sha "$1/$M_REL"; }
mod_absent() { [ ! -e "$1/$MOD" ]; }
mod_ok() { [ "$(sha "$1/$MOD")" = "$(sha "$P/files/p033_mla_w8a16.py")" ]; }
no_leftovers() { ! ls "$1"/*.p033.* "$1/$(dirname "$M_REL")"/*.p033.* >/dev/null 2>&1; }
G=VLLM_GB10_MLA_W8A16=1

# S0 inputs and pins
for kv in "$M_REL:MODEL_ORIG" "$MTP_REL:MTP_ORIG" "$LIN_REL:LINEAR_ORIG" "$MU8_REL:MU8_ORIG" "$MU_REL:MU_ORIG" "$COPS_REL:COPS_ORIG"; do
  rel=${kv%%:*}; pn=${kv#*:}
  [ "$(sha "$VLLM/$rel")" = "$(pin_orig "$pn")" ] && ok "S0 --vllm $rel == pin $pn" || bad "S0 $rel != $pn"
done
for kv in "$LU_REL:LUTIL_ORIG" "$BL_REL:BLOAD_ORIG" "$EAGLE_REL:EAGLE_ORIG"; do
  rel=${kv%%:*}; pn=${kv#*:}
  [ "$(sha "$VLLM/$rel")" = "$(pin_orig "$pn")" ] && ok "S0 --vllm $rel == pin $pn" || bad "S0 $rel != $pn"
done
for v in $BASEVARS; do
  [ "$(sha "$BASES/$(bdir "$v")/model.py")" = "$(pin_orig "$v")" ] && ok "S0 base $v == its pin" || bad "S0 base $v != pin"
done
[ "$(sha "$P/files/p033_apply.py")" = "$(pin_orig APPLIER_SHA)" ] && [ "$(sha "$P/files/p033_mla_w8a16.py")" = "$(pin_orig MOD_SHA)" ] \
  && ok "S0 APPLIER_SHA / MOD_SHA == the vendored files" || bad "S0 vendored shas"
for kv in "LUTIL_ORIG:032-gb10-repl-embed.sh" "BLOAD_ORIG:032-gb10-repl-embed.sh" "EAGLE_ORIG:029-gb10-draft-compress.sh" "MU8_ORIG:029-gb10-draft-compress.sh" \
          "MU_ORIG:029-gb10-draft-compress.sh" "LINEAR_ORIG:025-gb10-mhc-prefill-sp.sh" "MTP_ORIG:029-gb10-draft-compress.sh" "MTP_P029:029-gb10-draft-compress.sh"; do
  pn=${kv%%:*}; other=${kv#*:}
  theirs=$(sed -n "s/^$pn=\([0-9a-f]\{64\}\)$/\1/p" "$PDIR/$other" 2>/dev/null)
  if [ -z "$theirs" ]; then skipped "S0 pin $pn vs $other: that patch is absent or its pin is not a sha"
  else [ "$(pin_orig "$pn")" = "$theirs" ] && ok "S0 pin $pn == $other's own pin" || bad "S0 pin provenance $pn vs $other"; fi
done

# S1 gate unset / 0 / blank -> skipped, untouched (several bases)
for v in $BASEVARS; do
  for val in "" 0 " "; do
    r=$(mkroot "s1_${v}"); seed "$r" "$v"; before=$(msha "$r")
    rc=$(run33 "$r" VLLM_GB10_MLA_W8A16="$val")
    [ "$rc" = 0 ] && [ "$(msha "$r")" = "$before" ] && mod_absent "$r" && grep -q "\[OK\] patch 033 skipped" "$r.log" \
      && ok "S1 $v, VLLM_GB10_MLA_W8A16='$val': skipped, model.py untouched, no module" || { bad "S1 $v '$val' rc=$rc"; tail -3 "$r.log"; }
  done
done
r=$(mkroot s1_unset); seed "$r" MODEL_ORIG; rc=$(run33 "$r")
[ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && ok "S1 VLLM_GB10_MLA_W8A16 not in the environment: skipped" || bad "S1 unset rc=$rc"
# the skip line never quotes the words the boot gates count
grep -q "ACTIVE" "$r.log" && bad "S1 the skip output contains ACTIVE" || ok "S1 the skip output contains neither ACTIVE nor INACTIVE"

# S2 invalid values refused before anything is written; whitespace-padded valid values accepted
for v in 2 true yes 11 "1 1"; do
  r=$(mkroot "s2_$(printf '%s' "$v" | tr -c 'a-z0-9' '_')"); seed "$r" MODEL_ORIG; rc=$(run33 "$r" VLLM_GB10_MLA_W8A16="$v")
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "must be unset/0 (inert) or 1" "$r.log" \
    && ok "S2 VLLM_GB10_MLA_W8A16='$v' refused, untouched" || bad "S2 '$v' rc=$rc"
done
for v in " 1" "1 " " 1 "; do
  r=$(mkroot "s2w_$(printf '%s' "$v" | tr ' ' '_')"); seed "$r" MODEL_ORIG; rc=$(run33 "$r" VLLM_GB10_MLA_W8A16="$v")
  [ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG_P033)" ] && ok "S2 VLLM_GB10_MLA_W8A16='$v': whitespace stripped (as the module's gate), applied" || bad "S2w '$v' rc=$rc"
done
for v in 0 65 08 100 abc 1.5 -3; do
  r=$(mkroot "s2m_$v"); seed "$r" MODEL_ORIG; rc=$(run33 "$r" $G VLLM_GB10_MLA_W8A16_MAX_M="$v")
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "VLLM_GB10_MLA_W8A16_MAX_M" "$r.log" \
    && ok "S2 MAX_M='$v' refused, untouched" || bad "S2 MAX_M '$v' rc=$rc"
done
for v in 1 8 " 16 " 64; do
  r=$(mkroot "s2mo_$(printf '%s' "$v" | tr ' ' '_')"); seed "$r" MODEL_ORIG; rc=$(run33 "$r" $G VLLM_GB10_MLA_W8A16_MAX_M="$v")
  [ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG_P033)" ] && ok "S2 MAX_M='$v' accepted" || bad "S2 MAX_M ok '$v' rc=$rc"
done
for v in numerics fast ""; do
  r=$(mkroot "s2s_$v"); seed "$r" MODEL_ORIG; rc=$(run33 "$r" $G VLLM_GB10_MLA_W8A16_SPEED_GATE="$v")
  if [ -z "$v" ]; then
    [ "$rc" = 0 ] && ok "S2 SPEED_GATE='' means the default (graph), accepted" || bad "S2 SPEED_GATE '' rc=$rc"
  else
    [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "SPEED_GATE" "$r.log" \
      && ok "S2 SPEED_GATE='$v' refused, untouched" || bad "S2 SPEED_GATE '$v' rc=$rc"
  fi
done
for v in info " GRAPH "; do
  r=$(mkroot "s2so_$(printf '%s' "$v" | tr ' ' '_')"); seed "$r" MODEL_ORIG; rc=$(run33 "$r" $G VLLM_GB10_MLA_W8A16_SPEED_GATE="$v")
  [ "$rc" = 0 ] && ok "S2 SPEED_GATE='$v' accepted" || bad "S2 SPEED_GATE ok '$v' rc=$rc"
done
# invalid MAX_M / SPEED_GATE are ignored when the patch is off (nothing reads them)
r=$(mkroot s2off); seed "$r" MODEL_ORIG; rc=$(run33 "$r" VLLM_GB10_MLA_W8A16_MAX_M=999 VLLM_GB10_MLA_W8A16_SPEED_GATE=zzz)
[ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && ok "S2 gate off: MAX_M / SPEED_GATE are not even read" || bad "S2 off rc=$rc"

# S3 every available base: pinned result, module, markers, idempotent, gate-unset note, revert
for v in $BASEVARS; do
  r=$(mkroot "s3_$v"); seed "$r" "$v"
  [ "$(msha "$r")" = "$(pin "$v")" ] || { bad "S3 base $v not seeded"; continue; }
  c=$(run33 "$r" $G)
  if [ "$c" = 0 ] && [ "$(msha "$r")" = "$(pin "${v}_P033")" ] && mod_ok "$r" && no_leftovers "$r" \
     && [ "$(grep -cF "$M33" "$r/$M_REL")" = 3 ] && [ "$(grep -c "\[OK\] patch 033 in place" "$r.log")" = 1 ]; then
    ok "S3 $v -> 033: model.py == pinned ${v}_P033, module installed, 3 markers, one [OK] line, no leftovers"
  else bad "S3 $v rc=$c sha $(msha "$r" | cut -c1-16)"; tail -4 "$r.log"; fi
  grep -q "ACTIVE" "$r.log" && bad "S3 $v: the [OK] output quotes ACTIVE (a gate's negative grep would count it)" || ok "S3 $v: no ACTIVE / INACTIVE words in the .sh output"
  d=$(run33 "$r" $G)
  [ "$d" = 0 ] && [ "$(msha "$r")" = "$(pin "${v}_P033")" ] && mod_ok "$r" && grep -q "\[skip\] model.py already carries patch 033" "$r.log" \
    && ok "S3 $v: 033 re-run on its own result: idempotent [skip], same bytes" || bad "S3 re-run $v rc=$d"
  e=$(run33 "$r")
  [ "$e" = 0 ] && [ "$(msha "$r")" = "$(pin "${v}_P033")" ] && grep -q "already carries patch 033 (reused container?)" "$r.log" \
    && ok "S3 $v: gate unset on its result: note + skipped, untouched" || bad "S3 unset re-run $v rc=$e"
  python3 "$P/files/p033_apply.py" revert "$r/$M_REL" >/dev/null 2>&1
  [ "$(msha "$r")" = "$(pin "$v")" ] && ok "S3 $v: applier revert restores the base" || bad "S3 revert $v"
done

# S4 mtp.py: the image original and patch 029's result (REAL p029_apply.py) are accepted, anything else refused
S4BASE=MODEL_ORIG_P032; case " $BASEVARS " in *" $S4BASE "*) ;; *) S4BASE=MODEL_ORIG ;; esac
r=$(mkroot s4_029); seed "$r" $S4BASE; python3 "$P/files/p029_apply.py" apply mtp "$r/$MTP_REL" >/dev/null 2>&1
if [ "$(sha "$r/$MTP_REL")" = "$(pin MTP_P029)" ]; then
  rc=$(run33 "$r" $G)
  [ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin ${S4BASE}_P033)" ] && ok "S4 mtp.py == patch 029's pinned result (REAL p029_apply.py): accepted" || bad "S4 mtp029 rc=$rc"
else bad "S4 p029_apply.py did not produce MTP_P029 ($(sha "$r/$MTP_REL" | cut -c1-16))"; fi
r=$(mkroot s4_bad); seed "$r" MODEL_ORIG; printf '\n# drift\n' >> "$r/$MTP_REL"; rc=$(run33 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "neither the v0.32.0 original nor patch 029's result" "$r.log" \
  && ok "S4 an unknown mtp.py refused, untouched" || bad "S4 bad mtp rc=$rc"

# S5 unknown bases / stale 033 files
r=$(mkroot s5a); seed "$r" MODEL_ORIG; printf '\n# local edit\n' >> "$r/$M_REL"; before=$(msha "$r"); rc=$(run33 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "none of the four bases" "$r.log" \
  && ok "S5 a model.py that is no pinned base is refused, every file untouched" || bad "S5a rc=$rc"
r=$(mkroot s5b); seed "$r" MODEL_ORIG; run33 "$r" $G >/dev/null; printf '\n# drift\n' >> "$r/$M_REL"; before=$(msha "$r"); rc=$(run33 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$before" ] && grep -q "not one of this revision's pinned results" "$r.log" \
  && ok "S5 a 033-marked model.py that is not a pinned result refused (recreate the container)" || bad "S5b rc=$rc"
r=$(mkroot s5c); seed "$r" $S4BASE; run33 "$r" $G >/dev/null; rc=$(run33 "$r")
[ "$rc" = 0 ] && ok "S5 a reused container (033 applied, gate now unset) is skipped with a note, not failed" || bad "S5c rc=$rc"

# S6 read-only contracts: any drift refused before a write
for rel in "$LIN_REL" "$MU8_REL" "$MU_REL" "$COPS_REL" "$LU_REL" "$BL_REL" "$EAGLE_REL"; do
  r=$(mkroot "s6_$(basename "$rel" .py)"); seed "$r" MODEL_ORIG; printf '\n# drift\n' >> "$r/$rel"; rc=$(run33 "$r" $G)
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "refusing" "$r.log" \
    && ok "S6 $rel changed: refused, untouched" || bad "S6 $rel rc=$rc"
  r=$(mkroot "s6m_$(basename "$rel" .py)"); seed "$r" MODEL_ORIG; rm "$r/$rel"; rc=$(run33 "$r" $G)
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "not found" "$r.log" \
    && ok "S6 $rel missing: refused, untouched" || bad "S6 $rel missing rc=$rc"
done

# S7 vendored files tampered -> refused before any write
for f in p033_apply.py p033_mla_w8a16.py; do
  cp "$P/files/$f" "$RUN/$f.keep"; printf '\n# tamper\n' >> "$P/files/$f"
  r=$(mkroot "s7_$f"); seed "$r" MODEL_ORIG; rc=$(run33 "$r" $G)
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "!= pinned" "$r.log" \
    && ok "S7 tampered files/$f refused, untouched" || bad "S7 $f rc=$rc"
  cp "$RUN/$f.keep" "$P/files/$f"
done

# S8 rollback: the applier fails after writing garbage -> model.py restored, module removed / previous bytes restored
cat > "$RUN/shim/python3" <<EOF
#!/usr/bin/env bash
if [ "\${2:-}" = apply ] && [ "\$(basename "\${1:-}")" = p033_apply.py ]; then echo garbage > "\$3"; exit 1; fi
exec "$REALPY" "\$@"
EOF
chmod +x "$RUN/shim/python3"
r=$(mkroot s8a); seed "$r" MODEL_ORIG; rc=$(SHIM="$RUN/shim:" run33 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "restored to their previous bytes" "$r.log" \
  && ok "S8 applier failure: model.py restored, module removed, no leftovers" || bad "S8a rc=$rc"
r=$(mkroot s8b); seed "$r" MODEL_ORIG; echo "# an older module" > "$r/$MOD"; oldm=$(sha "$r/$MOD"); rc=$(SHIM="$RUN/shim:" run33 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && [ "$(sha "$r/$MOD")" = "$oldm" ] && no_leftovers "$r" \
  && ok "S8 applier failure with a pre-existing module: its previous bytes restored" || bad "S8b rc=$rc"

# S9 TERM during the applier -> rollback, exit 130
cat > "$RUN/shim/python3" <<EOF
#!/usr/bin/env bash
if [ "\${2:-}" = apply ] && [ "\$(basename "\${1:-}")" = p033_apply.py ]; then echo partial > "\$3"; sleep 30; exit 0; fi
exec "$REALPY" "\$@"
EOF
chmod +x "$RUN/shim/python3"
r=$(mkroot s9); seed "$r" MODEL_ORIG
( env -i PATH="$RUN/shim:$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" $G bash "$SH33" >>"$r.log" 2>&1; echo $? > "$r.rc" ) &
bgpid=$!
for _ in $(seq 1 50); do grep -q "\[state\]" "$r.log" 2>/dev/null && break; sleep 0.1; done
sleep 0.5
pkill -TERM -f "bash $SH33" 2>/dev/null || true
wait "$bgpid" 2>/dev/null
rc=$(cat "$r.rc" 2>/dev/null || echo none)
[ "$rc" = 130 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && no_leftovers "$r" \
  && ok "S9 TERM during the applier: exit 130, model.py restored, module removed" || bad "S9 rc=$rc sha $(msha "$r" | cut -c1-16)"
pkill -f "sleep 30" 2>/dev/null || true

# S10 SF9: the dependencies covered by the self-test are recorded as [note] lines with their sha256 (or reported absent)
r=$(mkroot s10a); seed "$r" MODEL_ORIG; run33 "$r" $G >/dev/null
n_abs=$(grep -c "\[note\] .* not found under .* (not pinned)" "$r.log")
[ "$n_abs" = 3 ] && ok "S10 three recorded dependencies absent: three '[note] ... not found' lines, the patch still applies" || bad "S10 absent notes: $n_abs"
r=$(mkroot s10b); seed "$r" MODEL_ORIG
for rel in model_executor/models/deepseek_v2.py model_executor/layers/mla.py model_executor/parameter.py; do
  mkdir -p "$r/$(dirname "$rel")"; cp "$VLLM/$rel" "$r/$rel"
done
run33 "$r" $G >/dev/null
okn=1
for rel in model_executor/models/deepseek_v2.py model_executor/layers/mla.py model_executor/parameter.py; do
  grep -q "\[note\] $rel sha256 $(sha "$r/$rel") (not pinned: covered by the boot self-test; recorded)" "$r.log" || okn=0
done
[ "$okn" = 1 ] && ok "S10 the real v0.32.0 deepseek_v2.py / mla.py / parameter.py sha256 are printed as [note] lines" || bad "S10 notes"

# S11 container entrypoint over a full tree
skipped "S11 REAL 001 .. 033 chain through the container entrypoint over a full v0.32.0 tree: done by the stack check, not here"
skipped "S12 the REAL 025 / 032 .sh refusing a model.py that carries 033 (reused container): covered by their own checks"

echo "== $pass PASS, $fail FAIL, $skp SKIP (SKIP = not verifiable on this Mac)"
[ "$fail" = 0 ]
