#!/usr/bin/env bash
# [patch-032 gb10-repl-embed] .sh scenario checks, vLLM v0.32.0 port. Runs patches/032 against scratch VLLM_ROOT trees seeded with
# the files it reads at boot (and, for the 025 base, the result of the REAL files/p025_apply.py); never touches a real vLLM
# install and never deletes anything (every scenario gets a fresh mktemp directory under --work; clean --work up by hand).
# Needs bash, python3 (no torch), sha256sum or shasum.
#   bash files/p032_sh_check.sh --vllm <v0.32.0 vllm> --work <new dir> [--patches DIR] [--full-tree <v0.32.0 vllm>]
# --full-tree (optional, a complete copy of the v0.32.0 vllm package): S11 simulates the container entrypoint on a copy of it
# (APFS clone where available) - every real NNN-*.sh up to 032 in glob order with the env of the new recipe (gates of the
# carried patches + VLLM_GB10_REPL_EMBED=1) and variants, and checks what happens to model.py / mtp.py.
# Chain of the v0.32.0 set before 032: 001, 003, 011, 012, 014, 019, 020, 022, 025, 029, 031; only 025 writes model.py and only
# 029 writes mtp.py among the files 032 reads (S3 / S6 / S11). No 007 / 016 exist any more.
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM=""; WORK=""; FULL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --vllm) VLLM="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --patches) PDIR="$2"; shift 2 ;;
    --full-tree) FULL="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -d "$VLLM" ] && [ -n "$WORK" ] || { echo "usage: $0 --vllm <v0.32.0 vllm> --work DIR [--patches DIR] [--full-tree <v0.32.0 vllm>]" >&2; exit 2; }
FULL="${FULL%/}"
[ -z "$FULL" ] || [ -f "$FULL/v1/worker/gpu/model_runner.py" ] || { echo "--full-tree must be a complete vllm package copy" >&2; exit 2; }
mkdir -p "$WORK"
RUN="$(mktemp -d "$WORK/run.XXXXXX")"
M_REL=models/glm5next/common/model.py
MTP_REL=models/glm5next/common/mtp.py
VPE_REL=model_executor/layers/vocab_parallel_embedding.py
EAGLE_REL=v1/worker/gpu/spec_decode/eagle/utils.py
LU_REL=model_executor/model_loader/utils.py
BL_REL=model_executor/model_loader/base_loader.py
WT_REL=model_executor/model_loader/weight_tying.py
DEPS="$M_REL $MTP_REL $VPE_REL $EAGLE_REL $LU_REL $BL_REL $WT_REL"
MOD=_gb10_repl_embed.py
M32="# [patch-032 gb10-repl-embed]"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
pin() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$PDIR/032-gb10-repl-embed.sh"; }
P="$RUN/patches"; cp -R "$PDIR" "$P"
mkdir -p "$RUN/shim"
REALPY="$(command -v python3)"
pass=0; fail=0; skipped=0
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }
skp() { echo "[SKIP] $*"; skipped=$((skipped+1)); }
mkroot() {
  local r; r="$(mktemp -d "$RUN/root.$1.XXXXXX")"
  for rel in $DEPS; do mkdir -p "$r/$(dirname "$rel")"; cp "$VLLM/$rel" "$r/$rel"; done
  : > "$r.log"
  echo "$r"
}
run() { local s=$1 r=$2; shift 2; env -i PATH="${SHIM:-}$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" "$@" bash "$s" >>"$r.log" 2>&1; echo $?; }
run32() { local r=$1; shift; run "$P/032-gb10-repl-embed.sh" "$r" "$@"; }
msha() { sha "$1/$M_REL"; }
mod_absent() { [ ! -e "$1/$MOD" ]; }
mod_ok() { [ "$(sha "$1/$MOD")" = "$(sha "$P/files/p032_gb10_repl_embed.py")" ]; }
no_leftovers() { ! ls "$1"/*.p032.* "$1/$(dirname "$M_REL")"/*.p032.* >/dev/null 2>&1; }
G=VLLM_GB10_REPL_EMBED=1
HAVE025=0; [ -f "$P/files/p025_apply.py" ] && [ -n "$(pin MODEL_P025)" ] && [ -n "$(pin MODEL_P025_P032)" ] && HAVE025=1

# S0 inputs and pins
for kv in "$M_REL:MODEL_ORIG" "$VPE_REL:VPE_ORIG" "$EAGLE_REL:EAGLE_ORIG" "$MTP_REL:MTP_ORIG" "$LU_REL:LUTIL_ORIG" "$BL_REL:BLOAD_ORIG" "$WT_REL:WTIE_ORIG"; do
  rel=${kv%%:*}; p=${kv#*:}
  [ "$(sha "$VLLM/$rel")" = "$(pin "$p")" ] && ok "S0 --vllm $rel == pin $p" || bad "S0 $rel != $p"
done
[ "$(sha "$P/files/p032_apply.py")" = "$(pin APPLIER_SHA)" ] && [ "$(sha "$P/files/p032_gb10_repl_embed.py")" = "$(pin MOD_SHA)" ] \
  && ok "S0 APPLIER_SHA / MOD_SHA == the vendored files" || bad "S0 vendored shas"
if [ "$HAVE025" = 1 ]; then ok "S0 the 025 pins are shas and files/p025_apply.py exists: the 025 base is exercised"
else skp "S0 the 025 base (files/p025_apply.py absent or the MODEL_P025 / MODEL_P025_P032 pins are placeholders)"; fi

# S1 gate unset / 0 -> skipped, untouched
for v in "" 0; do
  r=$(mkroot "s1_$v"); before=$(msha "$r")
  rc=$(run32 "$r" VLLM_GB10_REPL_EMBED="$v")
  [ "$rc" = 0 ] && [ "$(msha "$r")" = "$before" ] && mod_absent "$r" && grep -q "\[OK\] patch 032 skipped" "$r.log" \
    && ok "S1 VLLM_GB10_REPL_EMBED='$v': skipped, model.py untouched, no module" || { bad "S1 '$v' rc=$rc"; tail -3 "$r.log"; }
done
r=$(mkroot s1_unset); rc=$(run32 "$r")
[ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && ok "S1 VLLM_GB10_REPL_EMBED not in the environment: skipped" || bad "S1 unset rc=$rc"
if [ "$HAVE025" = 1 ]; then
  r=$(mkroot s1_025); python3 "$P/files/p025_apply.py" apply "$r/$M_REL" >/dev/null 2>&1; before=$(msha "$r"); rc=$(run32 "$r")
  [ "$rc" = 0 ] && [ "$(msha "$r")" = "$before" ] && [ "$before" = "$(pin MODEL_P025)" ] && mod_absent "$r" && ok "S1 on the 025 base, gate unset: skipped, untouched" || bad "S1 025 base rc=$rc"
fi

# S2 invalid gate values refused before anything is written
for v in 2 true yes " 1" "1 "; do
  r=$(mkroot "s2_$(printf '%s' "$v" | tr -c 'a-z0-9' '_')"); rc=$(run32 "$r" VLLM_GB10_REPL_EMBED="$v")
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "must be unset/0 (inert) or 1" "$r.log" \
    && ok "S2 VLLM_GB10_REPL_EMBED='$v' refused, untouched" || bad "S2 '$v' rc=$rc"
done

# S3 every base: pinned results, module, markers, idempotent, revert
s3() {  # s3 TAG BASE_PIN [025]
  local tag=$1 b=$2 with025=${3:-} r c d e
  r=$(mkroot "s3_$tag")
  if [ -n "$with025" ]; then python3 "$P/files/p025_apply.py" apply "$r/$M_REL" >/dev/null 2>&1; fi
  [ "$(msha "$r")" = "$(pin "$b")" ] || { bad "S3 base $b not produced"; return; }
  c=$(run32 "$r" $G)
  if [ "$c" = 0 ] && [ "$(msha "$r")" = "$(pin "${b}_P032")" ] && mod_ok "$r" && no_leftovers "$r" \
     && [ "$(grep -cF "$M32" "$r/$M_REL")" = 2 ] && [ "$(grep -c "\[OK\] patch 032 in place" "$r.log")" = 1 ]; then
    ok "S3 $b -> 032: model.py == pinned ${b}_P032, module installed, 2 markers, one [OK] line, no leftovers"
  else bad "S3 $b rc=$c sha $(msha "$r" | cut -c1-16)"; tail -4 "$r.log"; fi
  d=$(run32 "$r" $G)
  [ "$d" = 0 ] && [ "$(msha "$r")" = "$(pin "${b}_P032")" ] && mod_ok "$r" && grep -q "\[skip\] model.py already carries patch 032" "$r.log" \
    && ok "S3 032 re-run on ${b}_P032: idempotent [skip], same bytes" || bad "S3 re-run $b rc=$d"
  e=$(run32 "$r")
  [ "$e" = 0 ] && [ "$(msha "$r")" = "$(pin "${b}_P032")" ] && grep -q "already carries patch 032 (reused container?)" "$r.log" \
    && ok "S3 032 with the gate unset on ${b}_P032: note + skipped, untouched" || bad "S3 unset re-run $b rc=$e"
  python3 "$P/files/p032_apply.py" revert "$r/$M_REL" >/dev/null 2>&1
  [ "$(msha "$r")" = "$(pin "$b")" ] && ok "S3 applier revert restores $b" || bad "S3 revert $b"
}
s3 orig MODEL_ORIG
if [ "$HAVE025" = 1 ]; then s3 p025 MODEL_P025 025; fi

# S4 the 029 eh_proj conflict (each case: '|'-separated env assignments, so values may hold spaces)
s4env() { local IFS='|'; S4=($1); }
for kv in "VLLM_GB10_DRAFT_COMPRESS=1" "VLLM_GB10_DRAFT_COMPRESS=1|VLLM_GB10_DRAFT_COMPRESS_PARTS=eh" \
          "VLLM_GB10_DRAFT_COMPRESS=1|VLLM_GB10_DRAFT_COMPRESS_PARTS=head,eh" "VLLM_GB10_DRAFT_COMPRESS=1|VLLM_GB10_DRAFT_COMPRESS_PARTS=EH" \
          "VLLM_GB10_DRAFT_COMPRESS=1|VLLM_GB10_DRAFT_COMPRESS_PARTS=" "VLLM_GB10_DRAFT_COMPRESS= 1|VLLM_GB10_DRAFT_COMPRESS_PARTS=eh, head"; do
  s4env "$kv"
  r=$(mkroot "s4bad_$(printf '%s' "$kv" | tr -c 'A-Za-z0-9' '_')"); rc=$(run32 "$r" $G "${S4[@]}")
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "set VLLM_GB10_DRAFT_COMPRESS_PARTS=head" "$r.log" \
    && ok "S4 refused (029 eh split on): $kv" || { bad "S4 $kv rc=$rc"; tail -2 "$r.log"; }
done
for kv in "VLLM_GB10_DRAFT_COMPRESS=1|VLLM_GB10_DRAFT_COMPRESS_PARTS=head" "VLLM_GB10_DRAFT_COMPRESS=1|VLLM_GB10_DRAFT_COMPRESS_PARTS= HEAD " \
          "VLLM_GB10_DRAFT_COMPRESS=0|VLLM_GB10_DRAFT_COMPRESS_PARTS=eh" "VLLM_GB10_DRAFT_COMPRESS_PARTS=eh" "VLLM_GB10_DRAFT_COMPRESS=|VLLM_GB10_DRAFT_COMPRESS_PARTS=eh"; do
  s4env "$kv"
  r=$(mkroot "s4ok_$(printf '%s' "$kv" | tr -c 'A-Za-z0-9' '_')"); rc=$(run32 "$r" $G "${S4[@]}")
  [ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG_P032)" ] && ok "S4 accepted: $kv" || { bad "S4 $kv rc=$rc"; tail -2 "$r.log"; }
done

# S5 unknown bases / stale 032 files
r=$(mkroot s5a); printf '\n# local edit\n' >> "$r/$M_REL"; before=$(msha "$r"); rc=$(run32 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$before" ] && mod_absent "$r" && grep -q "none of the two bases" "$r.log" \
  && ok "S5 a model.py that is no pinned base refused, untouched" || bad "S5a rc=$rc"
r=$(mkroot s5b); run32 "$r" $G >/dev/null; printf '\n# drift\n' >> "$r/$M_REL"; before=$(msha "$r"); rc=$(run32 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$before" ] && grep -q "not one of this revision's pinned results" "$r.log" \
  && ok "S5 a 032-marked model.py that is not a pinned result refused (recreate the container)" || bad "S5b rc=$rc"
# a placeholder pin can never match: a copy of the .sh whose 025 pins are placeholders refuses the 025 result
if [ "$HAVE025" = 1 ]; then
  cp -R "$P" "$RUN/patches_ph"
  sed -i.bak -e 's/^MODEL_P025=.*/MODEL_P025=pending-025-port/' -e 's/^MODEL_P025_P032=.*/MODEL_P025_P032=pending-025-port/' "$RUN/patches_ph/032-gb10-repl-embed.sh"
  r=$(mkroot s5c); python3 "$P/files/p025_apply.py" apply "$r/$M_REL" >/dev/null 2>&1; before=$(msha "$r")
  rc=$(run "$RUN/patches_ph/032-gb10-repl-embed.sh" "$r" $G)
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$before" ] && mod_absent "$r" && grep -q "pending-025-port" "$r.log" \
    && ok "S5 placeholder 025 pins (pending-025-port) never match: the 025 result is refused, untouched (fail closed)" || bad "S5c rc=$rc"
fi

# S6 read-only contracts
for rel in "$VPE_REL" "$EAGLE_REL" "$MTP_REL" "$LU_REL" "$BL_REL" "$WT_REL"; do
  r=$(mkroot "s6_$(basename "$rel" .py)"); printf '\n# drift\n' >> "$r/$rel"; rc=$(run32 "$r" $G)
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "refusing" "$r.log" \
    && ok "S6 $rel changed: refused, untouched" || bad "S6 $rel rc=$rc"
done
if [ -f "$P/files/p029_apply.py" ] && [ -n "$(pin MTP_P029)" ]; then
  r=$(mkroot s6_mtp029); python3 "$P/files/p029_apply.py" apply mtp "$r/$MTP_REL" >/dev/null 2>&1
  if [ "$(sha "$r/$MTP_REL")" = "$(pin MTP_P029)" ]; then
    rc=$(run32 "$r" $G)
    [ "$rc" = 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG_P032)" ] && ok "S6 mtp.py == patch 029's pinned result (REAL p029_apply.py): accepted" || bad "S6 mtp029 rc=$rc"
  else bad "S6 p029_apply.py did not produce MTP_P029 ($(sha "$r/$MTP_REL" | cut -c1-16)): refresh the pin with files/p032_cpu_check.py --print-pins"; fi
else skp "S6 mtp.py == patch 029's result (files/p029_apply.py absent or MTP_P029 is a placeholder)"; fi

# S7 vendored files tampered -> refused before any write
for f in p032_apply.py p032_gb10_repl_embed.py; do
  cp "$P/files/$f" "$RUN/$f.keep"; printf '\n# tamper\n' >> "$P/files/$f"
  r=$(mkroot "s7_$f"); rc=$(run32 "$r" $G)
  [ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && grep -q "!= pinned" "$r.log" \
    && ok "S7 tampered files/$f refused, untouched" || bad "S7 $f rc=$rc"
  cp "$RUN/$f.keep" "$P/files/$f"
done

# S8 rollback: the applier fails after writing garbage -> model.py restored, module removed / previous bytes restored
cat > "$RUN/shim/python3" <<EOF
#!/usr/bin/env bash
if [ "\${2:-}" = apply ] && [ "\$(basename "\${1:-}")" = p032_apply.py ]; then echo garbage > "\$3"; exit 1; fi
exec "$REALPY" "\$@"
EOF
chmod +x "$RUN/shim/python3"
r=$(mkroot s8a); rc=$(SHIM="$RUN/shim:" run32 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "restored to their previous bytes" "$r.log" \
  && ok "S8 applier failure: model.py restored, module removed, no leftovers" || bad "S8a rc=$rc"
r=$(mkroot s8b); echo "# an older module" > "$r/$MOD"; oldm=$(sha "$r/$MOD"); rc=$(SHIM="$RUN/shim:" run32 "$r" $G)
[ "$rc" != 0 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && [ "$(sha "$r/$MOD")" = "$oldm" ] && no_leftovers "$r" \
  && ok "S8 applier failure with a pre-existing module: its previous bytes restored" || bad "S8b rc=$rc"

# S9 TERM during the applier -> rollback, exit 130
cat > "$RUN/shim/python3" <<EOF
#!/usr/bin/env bash
if [ "\${2:-}" = apply ] && [ "\$(basename "\${1:-}")" = p032_apply.py ]; then echo partial > "\$3"; sleep 30; exit 0; fi
exec "$REALPY" "\$@"
EOF
chmod +x "$RUN/shim/python3"
r=$(mkroot s9)
( env -i PATH="$RUN/shim:$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" $G bash "$P/032-gb10-repl-embed.sh" >>"$r.log" 2>&1; echo $? > "$r.rc" ) &
bgpid=$!
for _ in $(seq 1 50); do grep -q "\[state\]" "$r.log" 2>/dev/null && break; sleep 0.1; done
sleep 0.5
pkill -TERM -f "bash $P/032-gb10-repl-embed.sh" 2>/dev/null || true
wait "$bgpid" 2>/dev/null
rc=$(cat "$r.rc" 2>/dev/null || echo none)
[ "$rc" = 130 ] && [ "$(msha "$r")" = "$(pin MODEL_ORIG)" ] && mod_absent "$r" && no_leftovers "$r" \
  && ok "S9 TERM during the applier: exit 130, model.py restored, module removed" || bad "S9 rc=$rc sha $(msha "$r" | cut -c1-16)"
pkill -f "sleep 30" 2>/dev/null || true

# S10 the .sh text carries no gate text on any line it prints (the gates count [patch-032] lines of the python module only)
if ! grep -nE '(echo|fail) "[^"]*(ACTIVE|\[patch-032\]|replicated input embedding|never ran)' "$PDIR/032-gb10-repl-embed.sh" >/dev/null; then
  ok "S10 no echo / fail line of the .sh carries gate text (ACTIVE / [patch-032] / replicated input embedding / never ran)"
else bad "S10 a line of the .sh quotes gate text"; fi

# S11 (optional, --full-tree): the container entrypoint on a scratch copy of the FULL tree - every real NNN-*.sh up to 032 in
# glob order with the env of the new recipe's carried gates (+ additions), then 032
if [ -n "$FULL" ]; then
  clone() { cp -cR "$1" "$2" 2>/dev/null || cp -R "$1" "$2"; }
  BASEENV="VLLM_GB10_BLOCKING_SYNC=1 VLLM_GB10_ONESHOT_AR=1 VLLM_GB10_SKINNY_IDX=1 VLLM_GB10_DRAFT_COMPRESS=1 VLLM_GB10_DRAFT_COMPRESS_PARTS=head"
  for cs in "plain|$BASEENV|skip|" "gate|$BASEENV $G|apply|MODEL_ORIG_P032" \
            "dchead|$BASEENV $G|apply|MODEL_ORIG_P032" "dcdefault|VLLM_GB10_DRAFT_COMPRESS=1 $G|refuse|" \
            "sp|$BASEENV $G VLLM_GB10_MHC_SP=1|apply|MODEL_P025_P032"; do
    IFS='|' read -r tag envs want wpin <<<"$cs"
    fr="$(mktemp -d "$RUN/full_$tag.XXXXXX")"; clone "$FULL" "$fr/vllm"; cp -R "$P" "$fr/patches"
    rcs=""; last=""
    for s in "$fr"/patches/[0-9][0-9][0-9]-*.sh; do
      n=$(basename "$s" | cut -c1-3); [ "$((10#$n))" -le 32 ] || continue
      env -i PATH="$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$fr/vllm" $envs bash "$s" >>"$fr.log" 2>&1
      rc=$?; last=$(basename "$s"); [ "$rc" = 0 ] || { rcs="$last rc=$rc"; break; }
    done
    case "$want" in
      skip)
        [ -z "$rcs" ] && grep -q "\[OK\] patch 032 skipped" "$fr.log" && [ ! -e "$fr/vllm/$MOD" ] \
          && ok "S11 full-tree entrypoint, 032 gate unset: 032 skipped, no module" || { bad "S11 full-tree $tag [$rcs]"; tail -5 "$fr.log"; } ;;
      refuse)  # whichever of 029 / 032 runs first refuses (029 runs first in glob order); 032 never installs anything
        { [ "$last" = 032-gb10-repl-embed.sh ] || [ "$last" = 029-gb10-draft-compress.sh ]; } && [ -n "$rcs" ] \
          && grep -q "set VLLM_GB10_DRAFT_COMPRESS_PARTS=head" "$fr.log" && [ ! -e "$fr/vllm/$MOD" ] \
          && [ "$(sha "$fr/vllm/$M_REL")" = "$(pin MODEL_ORIG)" ] \
          && ok "S11 full-tree entrypoint, 029 default parts + 032: refused by $last (the entrypoint aborts), model.py untouched, no module" \
          || { bad "S11 full-tree $tag [$rcs] last=$last"; tail -5 "$fr.log"; } ;;
      *)
        if [ "$tag" = sp ] && [ "$HAVE025" != 1 ]; then skp "S11 full-tree $tag (025 pins are placeholders)"; continue; fi
        [ -z "$rcs" ] && [ "$last" = 032-gb10-repl-embed.sh ] && grep -q "\[OK\] patch 032 in place" "$fr.log" \
          && [ "$(sha "$fr/vllm/$M_REL")" = "$(pin "$wpin")" ] && [ "$(sha "$fr/vllm/$MOD")" = "$(sha "$P/files/p032_gb10_repl_embed.py")" ] \
          && ok "S11 full-tree entrypoint [$tag]: the real 001-031 hand 032 its base; model.py == $wpin" \
          || { bad "S11 full-tree $tag [$rcs] last=$last"; tail -5 "$fr.log"; } ;;
    esac
  done
  # S12 reused container: the REAL 025 .sh on a tree whose model.py already carries 032 refuses it (documented rule: the
  # launch always recreates the container), model.py intact
  if [ -f "$P/025-gb10-mhc-prefill-sp.sh" ]; then
    fr="$(mktemp -d "$RUN/full_reuse.XXXXXX")"; clone "$FULL" "$fr/vllm"
    r1=$(env -i PATH="$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$fr/vllm" $G bash "$P/032-gb10-repl-embed.sh" >"$fr.log" 2>&1; echo $?)
    after32=$(sha "$fr/vllm/$M_REL")
    r2=$(env -i PATH="$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$fr/vllm" VLLM_GB10_MHC_SP=1 bash "$P/025-gb10-mhc-prefill-sp.sh" >>"$fr.log" 2>&1; echo $?)
    [ "$r1" = 0 ] && [ "$after32" = "$(pin MODEL_ORIG_P032)" ] && [ "$r2" != 0 ] && [ "$(sha "$fr/vllm/$M_REL")" = "$after32" ] \
      && ok "S12 reused container: the REAL 025 .sh refuses a model.py that carries 032 (recreate the container), file intact" \
      || { bad "S12 reused container r1=$r1 r2=$r2"; tail -3 "$fr.log"; }
  else skp "S12 reused container (025 .sh absent)"; fi
else
  echo "[skip] S11 full-tree entrypoint simulation (pass --full-tree <v0.32.0 vllm> to run it)"
fi

echo "== $pass PASS, $fail FAIL, $skipped SKIP"
[ "$fail" = 0 ]
