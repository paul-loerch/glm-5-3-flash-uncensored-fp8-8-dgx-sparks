#!/usr/bin/env bash
# [patch-035 gb10-idx-compact-xchg] .sh scenario checks. Runs patches/035-gb10-idx-compact-xchg.sh against scratch VLLM_ROOT trees seeded
# with the three files it reads at boot (the image's sparse_indexer.py, sparse_attn_indexer.py and kpool_compress.py); never touches a
# real vLLM install and never deletes anything outside its own scratch dirs. Needs bash, python3 (no torch, no numpy), sha256sum or shasum.
#   bash files/p035_sh_check.sh --work <existing or new dir> [--files <dir with the 3 image files>] [--patches <patches dir>]
# --files defaults to $IMAGE_FILES, else ./image-files (the files exactly as the SM121-baked image carries them).
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILES=""; WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --files) FILES="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --patches) PDIR="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$FILES" ] || FILES="${IMAGE_FILES:-image-files}"
[ -d "$FILES" ] && [ -n "$WORK" ] || { echo "usage: $0 --work DIR [--files DIR] [--patches DIR]" >&2; exit 2; }
mkdir -p "$WORK"
RUN="$(mktemp -d "$WORK/p035run.XXXXXX")"
TGT_REL=models/glm5next/nvidia/sparse_indexer.py
OP_REL=model_executor/layers/sparse_attn_indexer.py
KP_REL=models/glm5next/nvidia/ops/kpool_compress.py
SH="$PDIR/035-gb10-idx-compact-xchg.sh"
BAKE="$PDIR/../image/glm53_sm121_v032.py"
MARK="# [patch-035 gb10-idx-compact-xchg]"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
pin() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$SH"; }
pass=0; fail=0
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }
seedfile() {  # seedfile <root> <rel> <source file>
  mkdir -p "$1/$(dirname "$2")"; cp "$3" "$1/$2"
}
n=0
mkroot() {  # mkroot -> prints a new VLLM_ROOT seeded with the three originals
  n=$((n+1)); local r="$RUN/root$n"
  seedfile "$r" "$TGT_REL" "$FILES/sparse_indexer.py"
  seedfile "$r" "$OP_REL" "$FILES/sparse_attn_indexer.py"
  seedfile "$r" "$KP_REL" "$FILES/kpool_compress.py"
  echo "$r"
}
run() {  # run <root> [VAR=VAL ...] -> stdout+stderr in $OUT, exit code in $RC
  local root=$1; shift
  OUT="$(env -u VLLM_GB10_IDX_COMPACT_XCHG -u VLLM_GB10_IDX_COMPACT_XCHG_VERIFY -u VLLM_GB10_IDX_SHARD VLLM_ROOT="$root" "$@" bash "$SH" 2>&1)"; RC=$?
}
count() { printf '%s\n' "$OUT" | grep -cE "$1" || true; }
leftovers() { ls "$1/$(dirname "$TGT_REL")" | grep -E 'p035\.(orig|tmp)' || true; }

ORIG="$(pin TARGET_ORIG)"; PATCHED="$(pin TARGET_P035)"
[ "$(sha "$FILES/sparse_indexer.py")" = "$ORIG" ] && ok "S0 the sparse_indexer.py in --files is the pinned SM121-baked original ($ORIG)" || bad "S0 --files sparse_indexer.py is not the pinned original"
[ "$(sha "$FILES/sparse_attn_indexer.py")" = "$(pin OP_ORIG)" ] && [ "$(sha "$FILES/kpool_compress.py")" = "$(pin KPOOL_BAKED)" ] && ok "S0b the two read-only contract files in --files are the pinned ones" || bad "S0b contract files differ from their pins"
[ -n "$PATCHED" ] && [ "$PATCHED" != "0000000000000000000000000000000000000000000000000000000000000000" ] && ok "S0c TARGET_P035 is pinned (not the placeholder)" || bad "S0c TARGET_P035 is a placeholder"

# T1: gate unset / 0 / empty -> inert, nothing touched, exactly one state line, no OK line
for spec in "unset:" "zero:VLLM_GB10_IDX_COMPACT_XCHG=0" "empty:VLLM_GB10_IDX_COMPACT_XCHG=" "space:VLLM_GB10_IDX_COMPACT_XCHG= "; do
  label=${spec%%:*}; assign=${spec#*:}
  r="$(mkroot)"; before="$(sha "$r/$TGT_REL")"
  if [ -n "$assign" ]; then run "$r" "$assign"; else run "$r"; fi
  after="$(sha "$r/$TGT_REL")"
  [ $RC -eq 0 ] && [ "$(count '^\[state\] patch 035 inert \(VLLM_GB10_IDX_COMPACT_XCHG=')" = 1 ] && [ "$(count '^\[OK\] patch 035')" = 0 ] && [ "$before" = "$after" ] \
    && [ -z "$(leftovers "$r")" ] && ok "T1 gate $label: '[state] patch 035 inert', exit 0, sparse_indexer.py untouched, no leftovers" || bad "T1 gate $label rc=$RC: $OUT"
done
# T1b: the inert mode reads nothing (no file of the tree needs to exist)
run "$RUN/does-not-exist"
[ $RC -eq 0 ] && [ "$(count '^\[state\] patch 035 inert')" = 1 ] && ok "T1b inert mode needs no file of the vLLM tree" || bad "T1b rc=$RC: $OUT"

# T2: gate 1 -> applied, pinned result, one [OK] line, no leftovers
r="$(mkroot)"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
got="$(sha "$r/$TGT_REL")"
[ $RC -eq 0 ] && [ "$got" = "$PATCHED" ] && [ "$(count '^\[OK\] patch 035 in place')" = 1 ] && [ "$(count '^\[state\] patch 035 inert')" = 0 ] && [ "$(grep -cF "$MARK" "$r/$TGT_REL")" = 5 ] \
  && ok "T2 gate 1: applied, sha256 == pinned result, exactly one '[OK] patch 035 in place', 5 markers" || bad "T2 rc=$RC got=$got: $OUT"
[ -z "$(leftovers "$r")" ] && ok "T2b no .orig / .tmp left behind" || bad "T2b leftovers: $(leftovers "$r")"
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$r/$TGT_REL" && ok "T2c the patched file parses" || bad "T2c parse"
[ "$(sha "$r/$OP_REL")" = "$(pin OP_ORIG)" ] && [ "$(sha "$r/$KP_REL")" = "$(pin KPOOL_BAKED)" ] && ok "T2d the two read-only contract files are untouched" || bad "T2d contract files changed"
echo "$OUT" | head -1 | grep -q 'VERIFY=unset -> 3' && echo "$OUT" | grep -q "VERIFY=3: a TP-sharded indexer prefill step exchanges the 512 pool ids" && ok "T2e the banner and the OK line show the VERIFY default 3" || bad "T2e banner: $(echo "$OUT" | head -1)"
echo "$OUT" | grep -q "compact exchange ACTIVE (width 512 instead of 2051)" && ok "T2f the OK line names the ACTIVE boot-log line to look for" || bad "T2f"
echo "$OUT" | grep -q "VLLM_GB10_IDX_SHARD is not 1" && ok "T2g without VLLM_GB10_IDX_SHARD=1 a [note] says no ACTIVE line will appear" || bad "T2g note missing"
r2="$(mkroot)"; run "$r2" VLLM_GB10_IDX_COMPACT_XCHG=1 VLLM_GB10_IDX_SHARD=1
[ $RC -eq 0 ] && [ "$(count 'VLLM_GB10_IDX_SHARD is not 1')" = 0 ] && ok "T2h with VLLM_GB10_IDX_SHARD=1 there is no such note" || bad "T2h rc=$RC: $OUT"

# T3: second run (reused container) is idempotent
run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
[ $RC -eq 0 ] && [ "$(sha "$r/$TGT_REL")" = "$PATCHED" ] && [ "$(count '^\[OK\] patch 035 in place')" = 1 ] && [ "$(count '\[skip\] sparse_indexer.py already carries patch 035')" = 1 ] \
  && ok "T3 second run: '[skip] already carries', pinned sha unchanged, one [OK] line" || bad "T3 rc=$RC: $OUT"
# T4: gate unset on an already patched file: note + inert, file untouched
run "$r"
[ $RC -eq 0 ] && [ "$(sha "$r/$TGT_REL")" = "$PATCHED" ] && [ "$(count '\[note\] VLLM_GB10_IDX_COMPACT_XCHG unset but sparse_indexer.py already carries patch 035')" = 1 ] && [ "$(count '^\[state\] patch 035 inert')" = 1 ] \
  && ok "T4 gate unset on a patched file: note + inert, file untouched" || bad "T4 rc=$RC: $OUT"

# T5: bad gate values / bad VERIFY values are refused before anything is touched
for v in "2" "yes" "true" "on" "-1" "1.0"; do
  r="$(mkroot)"; before="$(sha "$r/$TGT_REL")"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG="$v"
  [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "must be unset/0 (inert) or 1" && ok "T5 VLLM_GB10_IDX_COMPACT_XCHG='$v' refused (rc=$RC), file untouched" || bad "T5 '$v' rc=$RC: $OUT"
done
for v in "abc" "-1" "1.5" "1001" "99999999999" "1e3" "0x10"; do
  r="$(mkroot)"; before="$(sha "$r/$TGT_REL")"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1 VLLM_GB10_IDX_COMPACT_XCHG_VERIFY="$v"
  [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "VLLM_GB10_IDX_COMPACT_XCHG_VERIFY" && ok "T5 VERIFY='$v' refused (rc=$RC), file untouched" || bad "T5 VERIFY '$v' rc=$RC: $OUT"
done
for v in "0" "1" "3" "1000" " 5 " "007"; do
  r="$(mkroot)"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1 VLLM_GB10_IDX_COMPACT_XCHG_VERIFY="$v"
  [ $RC -eq 0 ] && [ "$(sha "$r/$TGT_REL")" = "$PATCHED" ] && ok "T5 VERIFY='$v' accepted (read at runtime, the file is the same)" || bad "T5 VERIFY '$v' rc=$RC: $OUT"
done
r="$(mkroot)"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1 VLLM_GB10_IDX_COMPACT_XCHG_VERIFY=0
echo "$OUT" | grep -q "in place (VLLM_GB10_IDX_COMPACT_XCHG=1, VERIFY=0:" && ok "T5b the [OK] line shows the overridden VERIFY" || bad "T5b: $(echo "$OUT" | tail -1)"
# a bad VERIFY with the gate off is not looked at (inert)
r="$(mkroot)"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG_VERIFY=abc
[ $RC -eq 0 ] && [ "$(count '^\[state\] patch 035 inert')" = 1 ] && ok "T5c a bad VERIFY while the gate is off is ignored (inert)" || bad "T5c rc=$RC: $OUT"

# T6: a changed base is refused and left alone
r="$(mkroot)"; printf '\n# drift\n' >> "$r/$TGT_REL"; before="$(sha "$r/$TGT_REL")"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "is not the pinned SM121-baked v0.32.0 file" && [ -z "$(leftovers "$r")" ] && ok "T6 drifted sparse_indexer.py refused, file untouched, no leftovers" || bad "T6 rc=$RC: $OUT"
# T6b: the PRISTINE upstream file (the bake did not run) is refused with a pointed message. Derived by undoing the bake's two topk_neg1 edits.
if [ -f "$BAKE" ]; then
  r="$(mkroot)"
  python3 - "$BAKE" "$r/$TGT_REL" "$TGT_REL" <<'PYEOF' || bad "T6b could not derive the pristine file"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("bake", sys.argv[1]); m = importlib.util.module_from_spec(spec)
sys.modules["bake"] = m; spec.loader.exec_module(m)
p = sys.argv[2]; rel = sys.argv[3]; t = open(p).read()
for e in m.EDITS:
    if e.pkg == "vllm" and e.rel == rel:
        assert t.count(e.new) == 1; t = t.replace(e.new, e.old, 1)
open(p, "w").write(t)
PYEOF
  if [ "$(sha "$r/$TGT_REL")" = "$(pin TARGET_PRISTINE)" ]; then
    before="$(sha "$r/$TGT_REL")"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
    [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "PRISTINE upstream" && ok "T6b the pristine upstream sparse_indexer.py (bake missing) is refused with a pointed message, file untouched" || bad "T6b rc=$RC: $OUT"
  else
    echo "[SKIP] T6b the derived pristine file does not match TARGET_PRISTINE ($(sha "$r/$TGT_REL"))"
  fi
else
  echo "[SKIP] T6b pristine upstream file ($BAKE not found)"
fi
# T7: a foreign revision's marker is refused
r="$(mkroot)"; printf '\n%s foreign edit\n' "$MARK" >> "$r/$TGT_REL"; before="$(sha "$r/$TGT_REL")"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "not this revision's pinned result" && ok "T7 foreign patch-035 edit refused, file untouched" || bad "T7 rc=$RC: $OUT"
# T8: read-only contracts drifted -> refused before any write
for rel in "$OP_REL" "$KP_REL"; do
  r="$(mkroot)"; printf '\n# drift\n' >> "$r/$rel"; before="$(sha "$r/$TGT_REL")"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
  [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "re-validate and re-pin" && [ -z "$(leftovers "$r")" ] && ok "T8 drifted $rel refused before any write" || bad "T8 $rel rc=$RC: $OUT"
done
# T8b: the PRISTINE kpool_compress.py (the bake did not run) is refused with the same pointed message
if [ -f "$BAKE" ]; then
  r="$(mkroot)"
  python3 - "$BAKE" "$r/$KP_REL" "$KP_REL" <<'PYEOF' || bad "T8b could not derive the pristine kpool_compress.py"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("bake", sys.argv[1]); m = importlib.util.module_from_spec(spec)
sys.modules["bake"] = m; spec.loader.exec_module(m)
p = sys.argv[2]; rel = sys.argv[3]; t = open(p).read()
for e in m.EDITS:
    if e.pkg == "vllm" and e.rel == rel:
        assert t.count(e.new) == 1; t = t.replace(e.new, e.old, 1)
open(p, "w").write(t)
PYEOF
  if [ "$(sha "$r/$KP_REL")" = "$(pin KPOOL_PRISTINE)" ]; then
    before="$(sha "$r/$TGT_REL")"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
    [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "kpool_compress.py is the PRISTINE upstream" && [ -z "$(leftovers "$r")" ] && ok "T8b the pristine upstream kpool_compress.py (bake missing) is refused with a pointed message, before any write" || bad "T8b rc=$RC: $OUT"
  else
    echo "[SKIP] T8b the derived pristine kpool_compress.py does not match KPOOL_PRISTINE ($(sha "$r/$KP_REL"))"
  fi
else
  echo "[SKIP] T8b pristine kpool_compress.py ($BAKE not found)"
fi
# T9: missing file
for rel in "$TGT_REL" "$OP_REL" "$KP_REL"; do
  r="$(mkroot)"; rm "$r/$rel"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
  [ $RC -ne 0 ] && echo "$OUT" | grep -q "not found" && ok "T9 missing $rel refused" || bad "T9 $rel rc=$RC: $OUT"
done

# T10: a tampered applier (pinned sha) and a wrong result pin (rollback)
P2="$RUN/patches_tampered"; cp -R "$PDIR" "$P2"; printf '\n# x\n' >> "$P2/files/p035_apply.py"
r="$(mkroot)"; before="$(sha "$r/$TGT_REL")"
OUT="$(env -u VLLM_GB10_IDX_COMPACT_XCHG VLLM_ROOT="$r" VLLM_GB10_IDX_COMPACT_XCHG=1 bash "$P2/035-gb10-idx-compact-xchg.sh" 2>&1)"; RC=$?
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "p035_apply.py sha256" && ok "T10 tampered applier refused, file untouched" || bad "T10 rc=$RC: $OUT"
P3="$RUN/patches_badpin"; cp -R "$PDIR" "$P3"
sed -i.bak -E "s/^TARGET_P035=.*/TARGET_P035=0000000000000000000000000000000000000000000000000000000000000000/" "$P3/035-gb10-idx-compact-xchg.sh"; rm -f "$P3/035-gb10-idx-compact-xchg.sh.bak"
r="$(mkroot)"; before="$(sha "$r/$TGT_REL")"
OUT="$(env -u VLLM_GB10_IDX_COMPACT_XCHG VLLM_ROOT="$r" VLLM_GB10_IDX_COMPACT_XCHG=1 bash "$P3/035-gb10-idx-compact-xchg.sh" 2>&1)"; RC=$?
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && echo "$OUT" | grep -q "restored to its previous bytes" && [ -z "$(leftovers "$r")" ] \
  && ok "T10b result sha != pin: failure AFTER the write restores sparse_indexer.py byte for byte, no leftovers" || bad "T10b rc=$RC: $OUT"
# a pin for the contract that does not match -> refused before any write
P5="$RUN/patches_badop"; cp -R "$PDIR" "$P5"
sed -i.bak -E "s/^OP_ORIG=.*/OP_ORIG=1111111111111111111111111111111111111111111111111111111111111111/" "$P5/035-gb10-idx-compact-xchg.sh"; rm -f "$P5/035-gb10-idx-compact-xchg.sh.bak"
r="$(mkroot)"; before="$(sha "$r/$TGT_REL")"
OUT="$(env -u VLLM_GB10_IDX_COMPACT_XCHG VLLM_ROOT="$r" VLLM_GB10_IDX_COMPACT_XCHG=1 bash "$P5/035-gb10-idx-compact-xchg.sh" 2>&1)"; RC=$?
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$TGT_REL")" ] && [ -z "$(leftovers "$r")" ] && ok "T10c a changed contract pin refuses before the backup/write" || bad "T10c rc=$RC: $OUT"

# T11: the applier mid-run is TERMinated -> the file is restored (a stub applier that sleeps; its sha is re-pinned in the copy)
P4="$RUN/patches_term"; cp -R "$PDIR" "$P4"
cat > "$P4/files/p035_apply.py" <<'PYEOF'
import sys, time, pathlib
p = pathlib.Path(sys.argv[2]); p.write_text(p.read_text() + "\n# half-written\n"); time.sleep(30)
PYEOF
sed -i.bak -E "s/^APPLIER_SHA=.*/APPLIER_SHA=$(sha "$P4/files/p035_apply.py")/" "$P4/035-gb10-idx-compact-xchg.sh"; rm -f "$P4/035-gb10-idx-compact-xchg.sh.bak"
r="$(mkroot)"; before="$(sha "$r/$TGT_REL")"
( env -u VLLM_GB10_IDX_COMPACT_XCHG VLLM_ROOT="$r" VLLM_GB10_IDX_COMPACT_XCHG=1 bash "$P4/035-gb10-idx-compact-xchg.sh" >"$RUN/term.out" 2>&1 & echo $! >"$RUN/term.pid"; wait )&
sleep 2; kill -TERM "$(cat "$RUN/term.pid")" 2>/dev/null; sleep 2
[ "$before" = "$(sha "$r/$TGT_REL")" ] && [ -z "$(leftovers "$r")" ] && ok "T11 SIGTERM while the applier runs: sparse_indexer.py restored byte for byte, no leftovers" || bad "T11 sparse_indexer.py not restored: $(cat "$RUN/term.out")"
pkill -f "$P4/files/p035_apply.py" 2>/dev/null || true

# T12: banner / header facts
head -1 "$SH" | grep -q '^#!/usr/bin/env bash' && ok "T12 shebang" || bad "T12 shebang"
bash -n "$SH" && ok "T12b bash -n" || bad "T12b bash -n"
grep -q '^set -euo pipefail$' "$SH" && ok "T12c set -euo pipefail" || bad "T12c"
# T13: every terminal path prints exactly one of the two final-state lines (OK in place / state inert) -- counted over the runs above
r="$(mkroot)"; run "$r" VLLM_GB10_IDX_COMPACT_XCHG=1
a="$(count '^\[OK\] patch 035 in place')"; b="$(count '^\[state\] patch 035 inert')"
run "$r"; c="$(count '^\[OK\] patch 035 in place')"; d="$(count '^\[state\] patch 035 inert')"
[ "$a" = 1 ] && [ "$b" = 0 ] && [ "$c" = 0 ] && [ "$d" = 1 ] && ok "T13 exactly one '[OK] patch 035 in place' with the gate on, exactly one '[state] patch 035 inert' with it off" || bad "T13 counts $a $b $c $d"

echo "# p035_sh_check: $pass passed, $fail failed (scratch: $RUN)"
[ "$fail" -eq 0 ]
