#!/usr/bin/env bash
# [patch-029 gb10-draft-compress] .sh scenario checks (vLLM v0.32.0 port, module rev 3). Runs patches/029 against scratch
# VLLM_ROOT trees seeded with the v0.32.0 ORIGINAL files (common/mtp.py, mtp/speculator.py and the 5 read-only
# dependencies) + patch 012's installed module; never touches a real vLLM install. Needs bash, python3 (stdlib only),
# sha256sum or shasum, rsync.
#   bash files/p029_sh_check.sh --image-root <v0.32.0 source tree>/vllm --work /tmp/p029-sh-check [--patches DIR]
#   (--vllm-root is an alias of --image-root)
# S14 also runs every earlier NNN-*.sh (NNN < 029) that exists in the patches dir in the entrypoint's numeric order on a
# copy of the WHOLE source tree, with every gate set, with every gate unset and with the served gate set, and checks that
# none of them changed 029's targets (whatever their own exit codes on this host), then that 029 applies on top to the
# pinned results.
# Rev 2: S2g the .sh output against the boot-gate scripts' patterns (exactly one '[OK] patch 029 in place', nothing a
# negative grep counts, no ACTIVE / INACTIVE text); S9e VLLM_GB10_DRAFT_COMPRESS_EH_GATE values; S15 the patch-032
# combination refused before any write (whichever of 029 / 032 runs first); S16 the REAL 032 .sh after 029 rev 2 on a
# whole-tree copy (parts=head: both apply, 032 accepts 029's mtp.py result; eh: 029 refuses first, 032 alone refuses
# too). Verification of rev 2: S9w the [OK] line names the selected parts only.
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG=""; WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --image-root|--vllm-root) IMG="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --patches) PDIR="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -d "$IMG" ] && [ -n "$WORK" ] || { echo "usage: $0 --image-root DIR --work DIR [--patches DIR]" >&2; exit 2; }
MTP_REL=models/glm5next/common/mtp.py
SPEC_REL=v1/worker/gpu/spec_decode/mtp/speculator.py
DEPS="model_executor/layers/quantization/utils/marlin_utils_fp8.py model_executor/layers/quantization/utils/marlin_utils.py model_executor/layers/logits_processor.py v1/worker/gpu/spec_decode/eagle/utils.py v1/worker/gpu/spec_decode/speculator.py"
M29="# [patch-029 gb10-draft-compress]"
SHN=029-gb10-draft-compress.sh
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
pin() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$PDIR/$SHN"; }
rm -rf "$WORK"; mkdir -p "$WORK/roots"
P="$WORK/patches"; mkdir -p "$P"; rsync -a --exclude __pycache__ "$PDIR/" "$P/"
REALPY="$(command -v python3)"
pass=0; fail=0
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }
mkroot() {
  local r="$WORK/roots/$1" f
  rm -rf "$r"; mkdir -p "$r"
  for f in $MTP_REL $SPEC_REL $DEPS; do mkdir -p "$r/$(dirname "$f")"; cp "$IMG/$f" "$r/$f"; done
  cp "$P/files/gb10_oneshot_ar.py" "$r/_gb10_oneshot_ar.py"     # as patch 012 installs it
  echo "$r"
}
run() { local s=$1 r=$2; shift 2; env -i PATH="${SHIM:-}$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" "$@" bash "$s" >>"$r.log" 2>&1; echo $?; }
run29() { local r=$1; shift; run "${SH29:-$P/$SHN}" "$r" "$@"; }
mod_absent() { [ ! -e "$1/_gb10_draft_compress.py" ]; }
mod_ok() { [ "$(sha "$1/_gb10_draft_compress.py")" = "$(sha "$P/files/p029_draft_compress.py")" ]; }
no_leftovers() { ! find "$1" -name '*.p029.*' | grep -q .; }
orig_both() { [ "$(sha "$1/$MTP_REL")" = "$(pin MTP_ORIG)" ] && [ "$(sha "$1/$SPEC_REL")" = "$(pin SPEC_ORIG)" ]; }
patched_both() { [ "$(sha "$1/$MTP_REL")" = "$(pin MTP_P029)" ] && [ "$(sha "$1/$SPEC_REL")" = "$(pin SPEC_P029)" ]; }
G=VLLM_GB10_DRAFT_COMPRESS=1

# S0 pins
[ "$(sha "$IMG/$MTP_REL")" = "$(pin MTP_ORIG)" ] && [ "$(sha "$IMG/$SPEC_REL")" = "$(pin SPEC_ORIG)" ] \
  && ok "S0 v0.32.0 common/mtp.py / mtp/speculator.py == pinned originals" || bad "S0 image target shas"
i=0; for f in $DEPS; do
  k=$(echo MU8_ORIG MU_ORIG LP_ORIG EAGLE_ORIG DSPEC_ORIG | cut -d' ' -f$((i+1))); i=$((i+1))
  [ "$(sha "$IMG/$f")" = "$(pin $k)" ] && ok "S0 dependency $f == pinned $k" || bad "S0 dependency $f"
done
[ "$(sha "$P/files/gb10_oneshot_ar.py")" = "$(pin OS_MOD_SHA)" ] && ok "S0 patch 012's vendored module == pinned OS_MOD_SHA" || bad "S0 012 module"
[ "$(sha "$P/files/p029_apply.py")" = "$(pin APPLIER_SHA)" ] && [ "$(sha "$P/files/p029_draft_compress.py")" = "$(pin MOD_SHA)" ] \
  && ok "S0 applier / module == pinned" || bad "S0 applier/module pins"

# S1 gate unset / 0
for v in "" 0 " 0 "; do
  r=$(mkroot "s1_${#v}"); if [ -z "$v" ]; then rc=$(run29 "$r"); else rc=$(run29 "$r" "VLLM_GB10_DRAFT_COMPRESS=$v"); fi
  [ "$rc" = 0 ] && orig_both "$r" && mod_absent "$r" && grep -q "\[OK\] patch 029 skipped" "$r.log" \
    && ok "S1 gate '${v:-unset}': skipped, both targets untouched, no module" || bad "S1 gate '$v' rc=$rc"
done

# S2 apply
r=$(mkroot s2); rc=$(run29 "$r" $G)
[ "$rc" = 0 ] && patched_both "$r" && mod_ok "$r" && no_leftovers "$r" \
  && [ "$(grep -cF "$M29" "$r/$MTP_REL")" = 3 ] && [ "$(grep -cF "$M29" "$r/$SPEC_REL")" = 1 ] \
  && grep -q "\[OK\] patch 029 in place" "$r.log" \
  && ok "S2 apply: both results == pins, module installed, markers 3/1, no backups, '[OK] patch 029 in place'" \
  || { bad "S2 rc=$rc"; tail -5 "$r.log"; }
# S2g the gate patterns of the boot-gate script and the counting rules of the other gates on the .sh output
n_ok=$(grep -c "\[OK\] patch 029 in place" "$r.log"); n_bad=$(grep -cE "\[patch-029\].*(INACTIVE|not active|exchange failed|unexpected)|\[029\] FATAL" "$r.log")
n_act=$(grep -c "draft-path compression ACTIVE" "$r.log"); n_any=$(grep -cE "ACTIVE|\[patch-029\]" "$r.log")
[ "$n_ok" = 1 ] && [ "$n_bad" = 0 ] && [ "$n_act" = 0 ] && [ "$n_any" = 0 ] && grep -q "\[OK\] patch 029 in place (rev 3; VLLM_GB10_DRAFT_COMPRESS=1, parts=eh,head, eh gate graph:" "$r.log" \
  && ok "S2g .sh output vs the gate patterns: '[OK] patch 029 in place' once, 0 negative-grep hits, no ACTIVE / INACTIVE / [patch-029] text anywhere (rev 1's [OK] line quoted them)" \
  || bad "S2g ok=$n_ok bad=$n_bad act=$n_act any=$n_any"
# for the record: rev 1's [OK] line alone tripped the boot gate's negative grep on every node (badl=1), ACTIVE or not
R1OK="[OK] patch 029 in place (VLLM_GB10_DRAFT_COMPRESS=1, parts=eh,head: draft-only FP8 lm_head + eh_proj row split; boot self-test logs '[patch-029] draft-path compression ACTIVE' or '... INACTIVE (fail closed)')"
[ "$(printf '%s\n' "$R1OK" | grep -cE "\[patch-029\].*(INACTIVE|not active|exchange failed|unexpected)|\[029\] FATAL")" = 1 ] \
  && [ "$(grep "\[OK\] patch 029 in place" "$r.log" | grep -cE "\[patch-029\].*(INACTIVE|not active|exchange failed|unexpected)|\[029\] FATAL")" = 0 ] \
  && ok "S2g regression: rev 1's [OK] text matches the boot gate's negative pattern (its gate stopped on every node whatever the verdict), rev 2's does not" \
  || bad "S2g rev-1 [OK] regression"
S2=$r
rc=$(run29 "$S2" $G)
[ "$rc" = 0 ] && patched_both "$S2" && [ "$(grep -c "already carries patch 029" "$S2.log")" = 2 ] && no_leftovers "$S2" \
  && ok "S3 idempotent re-run (both targets skipped)" || bad "S3 rc=$rc"
rc=$(run29 "$S2")
[ "$rc" = 0 ] && patched_both "$S2" && grep -q "unset but mtp.py / mtp/speculator.py already carry patch 029" "$S2.log" \
  && ok "S3b gate unset on patched files: note + skipped, untouched" || bad "S3b rc=$rc"
echo "# old module revision" > "$S2/_gb10_draft_compress.py"; rc=$(run29 "$S2" $G)
[ "$rc" = 0 ] && mod_ok "$S2" && patched_both "$S2" && no_leftovers "$S2" \
  && ok "S3c installed targets + a stale module: module replaced by the pinned bytes" || bad "S3c rc=$rc"
r=$(mkroot s3d); cp "$S2/$MTP_REL" "$r/$MTP_REL"; rc=$(run29 "$r" $G)
[ "$rc" = 0 ] && patched_both "$r" && grep -q "mtp.py already carries patch 029" "$r.log" && grep -q "mtp/speculator.py: image original + 029" "$r.log" \
  && ok "S3d mixed state (mtp.py patched, speculator.py original): only speculator.py applied, both pinned" || bad "S3d rc=$rc"
r=$(mkroot s3e); cp "$S2/$MTP_REL" "$r/$MTP_REL"; printf '\n# stale 029 revision\n' >> "$r/$MTP_REL"; before=$(sha "$r/$MTP_REL")
rc=$(run29 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MTP_REL")" = "$before" ] && [ "$(sha "$r/$SPEC_REL")" = "$(pin SPEC_ORIG)" ] && mod_absent "$r" && no_leftovers "$r" \
  && grep -q "not this revision's pinned result" "$r.log" && ok "S3e stale 029 edit refused, both untouched, no module" || bad "S3e rc=$rc"

# S4 unknown bases
r=$(mkroot s4); printf '\n# another patch\n' >> "$r/$MTP_REL"; before=$(sha "$r/$MTP_REL"); rc=$(run29 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$MTP_REL")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "neither the image original" "$r.log" \
  && ok "S4 unknown mtp.py base refused, untouched, no module" || bad "S4 rc=$rc"
r=$(mkroot s4b); printf '\n# another patch\n' >> "$r/$SPEC_REL"; before=$(sha "$r/$SPEC_REL"); rc=$(run29 "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$SPEC_REL")" = "$before" ] && [ "$(sha "$r/$MTP_REL")" = "$(pin MTP_ORIG)" ] && mod_absent "$r" \
  && ok "S4b unknown mtp/speculator.py base refused, untouched" || bad "S4b rc=$rc"

# S5 / S6 tampered vendored files (a copy of the patches dir)
for which in p029_apply.py p029_draft_compress.py; do
  T="$WORK/tamper_$which"; rm -rf "$T"; rsync -a "$P/" "$T/"; printf '\n# tampered\n' >> "$T/files/$which"
  r=$(mkroot "s5_$which"); rc=$(SH29="$T/$SHN" run29 "$r" $G)
  [ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && grep -q "!= pinned" "$r.log" \
    && ok "S5 tampered files/$which refused before any write" || bad "S5 $which rc=$rc"
done

# S7 dependencies
for f in $DEPS; do
  r=$(mkroot "s7_$(basename "$f" .py)"); printf '\n# changed\n' >> "$r/$f"; rc=$(run29 "$r" $G)
  [ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && grep -q "refusing" "$r.log" \
    && ok "S7 changed read-only dependency $(basename "$f") refused" || bad "S7 $f rc=$rc"
done

# S8 patch 012's module
r=$(mkroot s8); printf '\n# other 012 revision\n' >> "$r/_gb10_oneshot_ar.py"; rc=$(run29 "$r" $G)
[ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && grep -q "_gb10_oneshot_ar.py (patch 012)" "$r.log" \
  && ok "S8 a different patch-012 module refused" || bad "S8 rc=$rc"
r=$(mkroot s8b); rm -f "$r/_gb10_oneshot_ar.py"; rc=$(run29 "$r" $G)
[ "$rc" = 0 ] && patched_both "$r" && mod_ok "$r" && grep -q "eh part stays off at boot (fail closed)" "$r.log" && ! grep -q "INACTIVE" "$r.log" \
  && ok "S8b patch 012's module absent: note (eh stays off at boot), 029 applied" || bad "S8b rc=$rc"

# S9 env validation
for v in 2 yes true on; do
  r=$(mkroot "s9_$v"); rc=$(run29 "$r" "VLLM_GB10_DRAFT_COMPRESS=$v")
  [ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && ok "S9 gate '$v' refused" || bad "S9 gate $v rc=$rc"
done
for v in all "head,x" "head,,eh" "eh;head"; do
  r=$(mkroot "s9p_$(echo "$v" | tr -c 'a-z' '_')"); rc=$(run29 "$r" $G "VLLM_GB10_DRAFT_COMPRESS_PARTS=$v")
  [ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && ok "S9 PARTS '$v' refused" || bad "S9 PARTS $v rc=$rc"
done
for v in head eh "HEAD, eh" ""; do
  r=$(mkroot "s9q_$(echo "$v" | tr -c 'a-zA-Z' '_')"); rc=$(run29 "$r" $G "VLLM_GB10_DRAFT_COMPRESS_PARTS=$v")
  [ "$rc" = 0 ] && patched_both "$r" && ok "S9 PARTS '$v' accepted" || bad "S9 PARTS '$v' rc=$rc"
done
# (verification of rev 2) the [OK] line names the selected parts only
for pv in "head:draft-only FP8 lm_head;" "eh:eh_proj row split;" "eh,head:draft-only FP8 lm_head + eh_proj row split;"; do
  v=${pv%%:*}; want=${pv#*:}
  r=$(mkroot "s9w_$(echo "$v" | tr -c 'a-z' '_')"); rc=$(run29 "$r" $G "VLLM_GB10_DRAFT_COMPRESS_PARTS=$v")
  [ "$rc" = 0 ] && [ "$(grep -c '^\[OK\] patch 029 in place' "$r.log")" = 1 ] \
    && grep '^\[OK\] patch 029 in place' "$r.log" | grep -qF "parts=$v, eh gate graph: $want the module's" \
    && ok "S9w PARTS '$v': the [OK] line names '$want'" || bad "S9w PARTS '$v' rc=$rc"
done
# S9e rev 2: the eh speed-gate mode
for v in eager 1 "graph,numerics" off; do
  r=$(mkroot "s9e_$(echo "$v" | tr -c 'a-z0-9' '_')"); rc=$(run29 "$r" $G "VLLM_GB10_DRAFT_COMPRESS_EH_GATE=$v")
  [ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && grep -q "VLLM_GB10_DRAFT_COMPRESS_EH_GATE='$v'" "$r.log" \
    && ok "S9e EH_GATE '$v' refused before any write" || bad "S9e EH_GATE $v rc=$rc"
done
for v in graph numerics " Graph " NUMERICS ""; do
  want=$(printf '%s' "${v:-graph}" | tr 'A-Z' 'a-z' | tr -d ' '); [ -n "$want" ] || want=graph
  r=$(mkroot "s9f_$(echo "$v" | tr -c 'a-zA-Z' '_')"); rc=$(run29 "$r" $G "VLLM_GB10_DRAFT_COMPRESS_EH_GATE=$v")
  [ "$rc" = 0 ] && patched_both "$r" && grep -q "eh gate $want:" "$r.log" && ok "S9e EH_GATE '$v' accepted (eh gate $want)" || bad "S9e EH_GATE '$v' rc=$rc"
done

# S10 rollback after the first target was written (a copy of the .sh with a wrong SPEC_P029 pin)
T="$WORK/badpin"; rm -rf "$T"; rsync -a "$P/" "$T/"
sed -i.bak "s/^SPEC_P029=.*/SPEC_P029=$(printf '0%.0s' $(seq 1 64))/" "$T/$SHN"; rm -f "$T/$SHN.bak"
r=$(mkroot s10); rc=$(SH29="$T/$SHN" run29 "$r" $G)
[ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && no_leftovers "$r" && grep -q "restored to their previous bytes" "$r.log" \
  && ok "S10 failure after mtp.py was written: BOTH targets restored, the new module removed, no backups" || bad "S10 rc=$rc"
r=$(mkroot s10b); echo "# previous module bytes" > "$r/_gb10_draft_compress.py"; prev=$(sha "$r/_gb10_draft_compress.py")
rc=$(SH29="$T/$SHN" run29 "$r" $G)
[ "$rc" != 0 ] && orig_both "$r" && [ "$(sha "$r/_gb10_draft_compress.py")" = "$prev" ] && no_leftovers "$r" \
  && ok "S10b failure with a pre-existing module: its previous bytes restored" || bad "S10b rc=$rc"

# S12 TERM while the applier runs: child reaped, exit 130, both targets restored
SH_DIR="$WORK/shim"; mkdir -p "$SH_DIR"; FLAG="$WORK/shim.flag"; rm -f "$FLAG"
cat > "$SH_DIR/python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *p029_apply.py\ apply\ spec*) touch "$FLAG"; sleep 20 ;; esac
exec "$REALPY" "\$@"
EOF
chmod +x "$SH_DIR/python3"
r=$(mkroot s12)
env -i PATH="$SH_DIR:$PATH" HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 VLLM_ROOT="$r" $G bash "$P/$SHN" >>"$r.log" 2>&1 &
bpid=$!
for _ in $(seq 1 100); do [ -e "$FLAG" ] && break; sleep 0.1; done
sleep 0.3; kill -TERM "$bpid" 2>/dev/null; wait "$bpid"; rc=$?
sleep 0.2
[ "$rc" = 130 ] && orig_both "$r" && mod_absent "$r" && no_leftovers "$r" && ! pgrep -f "p029_apply.py apply spec $r" >/dev/null \
  && ok "S12 TERM during the applier: exit 130, applier child reaped, both targets restored, no module" || bad "S12 rc=$rc"
pkill -f "sleep 20" 2>/dev/null || true

# S13 byte-exact revert of both pinned results through the applier CLI
cp "$S2/$MTP_REL" "$WORK/m.py"; cp "$S2/$SPEC_REL" "$WORK/s.py"
"$REALPY" "$P/files/p029_apply.py" revert mtp "$WORK/m.py" >/dev/null && "$REALPY" "$P/files/p029_apply.py" revert spec "$WORK/s.py" >/dev/null
[ "$(sha "$WORK/m.py")" = "$(pin MTP_ORIG)" ] && [ "$(sha "$WORK/s.py")" = "$(pin SPEC_ORIG)" ] \
  && ok "S13 applier revert of both pinned results == the v0.32.0 originals" || bad "S13"

# S14 every earlier patch (NNN < 029, whatever exists in the patches dir) in entrypoint order on a full v0.32.0 source-tree copy
# (all gates set / all unset / the served v0.32.0 gate set): 029's targets untouched by all of them, then 029 applies to the pins
GATES_ALL="VLLM_GB10_BLOCKING_SYNC=1 VLLM_GB10_MHC_BM2_MIN_T=128 VLLM_GB10_ONESHOT_AR=1 VLLM_GB10_SKINNY_IDX=1 VLLM_GB10_MHC_FN_BF16=1 VLLM_GB10_SPARSE_MLA_NOSYNC=1 VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1 VLLM_GB10_MHC_PRS=1 VLLM_GB10_MHC_SP=1 VLLM_GB10_IDX_SHARD=1 VLLM_GB10_KPOOL_FIX=1 VLLM_GLM5NEXT_AUX_CAPTURE=1 VLLM_DFLASH2_PATCH=1 VLLM_EAGLE_KEEP_LAST_BLOCK=1 VLLM_GB10_DYNSD_FIX=1 VLLM_GB10_KPOOL_SLOTMAP_FIX=1 VLLM_GLM5NEXT_ROUTER_ONCE=1 VLLM_GB10_DRAFT_SALT=1 VLLM_GB10_DRAFT_K=1 VLLM_GB10_ADAPTIVE_K=1 VLLM_GB10_SPARSE_MLA_SYNCFREE=1 VLLM_GB10_DET_REMAP=1 VLLM_GLM53_REASONING_ALWAYS=1 VLLM_GB10_MAMBA_ALIGN_FREE=1 VLLM_GB10_KPOOL_WORKSPACE=1 VLLM_GB10_ADAPTIVE_PREFILL=1"
# the served gate set of the v0.32.0 recipe (gate variables of the carried patches only; 029 + 032 are the later ones)
GATES_PREV="VLLM_GB10_BLOCKING_SYNC=1 VLLM_GB10_ONESHOT_AR=1 VLLM_GB10_SKINNY_IDX=1 VLLM_GB10_MHC_BM2_MIN_T=128 VLLM_GB10_MHC_FN_BF16=1 VLLM_GB10_SPARSE_MLA_NOSYNC=1 VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1 VLLM_GB10_MHC_PRS=1 VLLM_GB10_MHC_SP=1 VLLM_GB10_IDX_SHARD=1"
for mode in all none prev; do
  r="$WORK/roots/s14_$mode"; rm -rf "$r"; mkdir -p "$r"; rsync -a --exclude __pycache__ "$IMG/" "$r/"
  cp "$P/files/gb10_oneshot_ar.py" "$r/_gb10_oneshot_ar.py" 2>/dev/null
  before_m=$(sha "$r/$MTP_REL"); before_s=$(sha "$r/$SPEC_REL"); touched=""; rcs=""
  for s in $(ls "$P"/[0-9][0-9][0-9]-*.sh | sort); do
    n=$(basename "$s" | cut -c1-3); [ "$((10#$n))" -lt 29 ] || continue
    case $mode in
      all) rc=$(run "$s" "$r" $GATES_ALL) ;;
      prev) rc=$(run "$s" "$r" $GATES_PREV) ;;
      *) rc=$(run "$s" "$r") ;;
    esac
    rcs="$rcs $n:$rc"
    [ "$(sha "$r/$MTP_REL")" = "$before_m" ] && [ "$(sha "$r/$SPEC_REL")" = "$before_s" ] || touched="$touched $n"
  done
  rc=$(run29 "$r" $G)
  [ -z "$touched" ] && [ "$rc" = 0 ] && patched_both "$r" \
    && ok "S14 gates $mode: every earlier patch run in order (rc$rcs) left mtp.py + mtp/speculator.py untouched; 029 then applied to the pins" \
    || bad "S14 $mode touched by:$touched 029 rc=$rc"
done

# S15 rev 2: patch 032 (VLLM_GB10_REPL_EMBED=1) and the eh part exclude each other; 029 refuses before any write
for v in "" "eh,head" "EH" "head,eh"; do
  r=$(mkroot "s15_$(echo "${v:-default}" | tr -c 'a-zA-Z' '_')")
  if [ -z "$v" ]; then rc=$(run29 "$r" $G VLLM_GB10_REPL_EMBED=1); else rc=$(run29 "$r" $G VLLM_GB10_REPL_EMBED=1 "VLLM_GB10_DRAFT_COMPRESS_PARTS=$v"); fi
  [ "$rc" != 0 ] && orig_both "$r" && mod_absent "$r" && no_leftovers "$r" && grep -q "set VLLM_GB10_DRAFT_COMPRESS_PARTS=head" "$r.log" \
    && ok "S15 VLLM_GB10_REPL_EMBED=1 + parts '${v:-default eh,head}': refused before any write" || bad "S15 parts '$v' rc=$rc"
done
r=$(mkroot s15_head); rc=$(run29 "$r" $G VLLM_GB10_REPL_EMBED=1 VLLM_GB10_DRAFT_COMPRESS_PARTS=head)
[ "$rc" = 0 ] && patched_both "$r" && mod_ok "$r" && grep -q "VLLM_GB10_REPL_EMBED=1 (patch 032): parts=head (no eh split)" "$r.log" \
  && ok "S15 VLLM_GB10_REPL_EMBED=1 + parts=head: applied (note)" || bad "S15 head rc=$rc"
for v in 0 "" yes; do
  r=$(mkroot "s15o_${v:-empty}"); rc=$(run29 "$r" $G "VLLM_GB10_REPL_EMBED=$v")
  [ "$rc" = 0 ] && patched_both "$r" && ! grep -q "patch 032" "$r.log" \
    && ok "S15 VLLM_GB10_REPL_EMBED='$v' (not 1; 032 validates its own values): 029 applies" || bad "S15 REPL '$v' rc=$rc"
done
r=$(mkroot s15_skip); rc=$(run29 "$r" VLLM_GB10_REPL_EMBED=1)
[ "$rc" = 0 ] && orig_both "$r" && mod_absent "$r" && ok "S15 VLLM_GB10_REPL_EMBED=1 with 029's gate unset: 029 skipped, untouched" || bad "S15 skip rc=$rc"

# S16 the REAL 032 .sh after 029 rev 3 on a whole-tree copy. Runs only when the v0.32.0 port of 032 is in the patches dir (its
# MODEL_ORIG pin is the v0.32.0 common/model.py); with the day-0 032 (or none) there is nothing meaningful to chain: SKIP.
S32="$P/032-gb10-repl-embed.sh"
MODEL_REL=models/glm5next/common/model.py
pin32() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$S32" 2>/dev/null; }
if [ -f "$S32" ] && [ -f "$IMG/$MODEL_REL" ] && [ -n "$(pin32 MODEL_ORIG)" ] && [ "$(pin32 MODEL_ORIG)" = "$(sha "$IMG/$MODEL_REL")" ]; then
  r="$WORK/roots/s16_head"; rm -rf "$r"; mkdir -p "$r"; rsync -a --exclude __pycache__ "$IMG/" "$r/"; cp "$P/files/gb10_oneshot_ar.py" "$r/_gb10_oneshot_ar.py"
  rc29=$(run29 "$r" $G VLLM_GB10_REPL_EMBED=1 VLLM_GB10_DRAFT_COMPRESS_PARTS=head)
  rc32=$(run "$S32" "$r" $G VLLM_GB10_REPL_EMBED=1 VLLM_GB10_DRAFT_COMPRESS_PARTS=head)
  want32="$(pin32 MODEL_ORIG_P032)"; m_after="$(sha "$r/$MODEL_REL")"
  [ "$rc29" = 0 ] && [ "$rc32" = 0 ] && patched_both "$r" \
    && { { [ -n "$want32" ] && [ "$m_after" = "$want32" ]; } || { [ -z "$want32" ] && [ "$m_after" != "$(sha "$IMG/$MODEL_REL")" ]; }; } \
    && { [ -z "$(pin32 MTP_P029)" ] || [ "$(pin32 MTP_P029)" = "$(pin MTP_P029)" ]; } \
    && ok "S16 parts=head + VLLM_GB10_REPL_EMBED=1: 029 rev 3 then the REAL 032 both apply (032 accepts 029's mtp.py result $(pin MTP_P029 | cut -c1-8))" \
    || bad "S16 head rc29=$rc29 rc32=$rc32"
  r="$WORK/roots/s16_eh"; rm -rf "$r"; mkdir -p "$r"; rsync -a --exclude __pycache__ "$IMG/" "$r/"; cp "$P/files/gb10_oneshot_ar.py" "$r/_gb10_oneshot_ar.py"
  m0=$(sha "$r/$MODEL_REL")
  rc29=$(run29 "$r" $G VLLM_GB10_REPL_EMBED=1)
  rc32=$(run "$S32" "$r" $G VLLM_GB10_REPL_EMBED=1)
  [ "$rc29" != 0 ] && [ "$rc32" != 0 ] && orig_both "$r" && mod_absent "$r" && [ "$(sha "$r/$MODEL_REL")" = "$m0" ] \
    && grep -q "VLLM_GB10_REPL_EMBED=1 (patch 032, replicated input embedding) with patch 029's eh_proj split" "$r.log" \
    && grep -q "with patch 029's eh_proj split (VLLM_GB10_DRAFT_COMPRESS=1, parts 'eh,head')" "$r.log" \
    && ok "S16 eh part + VLLM_GB10_REPL_EMBED=1: 029 refuses first and the REAL 032 refuses on its own too; mtp.py, speculator.py, model.py untouched" \
    || bad "S16 eh rc29=$rc29 rc32=$rc32"
else
  echo "[SKIP] S16 the v0.32.0 port of 032 is not in $P yet (no 032 .sh pinning the v0.32.0 common/model.py): not run"
fi

echo "== $pass passed, $fail failed"
[ "$fail" = 0 ]
