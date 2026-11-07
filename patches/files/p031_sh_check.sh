#!/usr/bin/env bash
# [patch-031 gb10-idx-shard-gate] .sh scenario checks. Runs patches/031-gb10-idx-shard-gate.sh against scratch VLLM_ROOT trees seeded with
# the three v0.32.0 files it reads at boot; never touches a real vLLM install and never deletes anything outside its own scratch dirs.
# Needs bash, python3 (no torch, no numpy), sha256sum or shasum.
#   bash files/p031_sh_check.sh --vllm <v0.32.0 vllm/ package dir> --work <existing or new dir> [--patches <patches dir>]
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
[ -d "$VLLM" ] && [ -n "$WORK" ] || { echo "usage: $0 --vllm <v0.32.0 vllm/ dir> --work DIR [--patches DIR]" >&2; exit 2; }
mkdir -p "$WORK"
RUN="$(mktemp -d "$WORK/p031run.XXXXXX")"
IDX_REL=v1/attention/backends/mla/indexer.py
OP_REL=model_executor/layers/sparse_attn_indexer.py
GLM_REL=models/glm5next/nvidia/sparse_indexer.py
SH="$PDIR/031-gb10-idx-shard-gate.sh"
MARK="# [patch-031 gb10-idx-shard-gate]"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
pin() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$SH"; }
pass=0; fail=0
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }

# fresh sandbox root
n=0
mkroot() {  # mkroot -> prints a new VLLM_ROOT seeded with the three originals
  n=$((n+1)); local r="$RUN/root$n"
  for rel in "$IDX_REL" "$OP_REL" "$GLM_REL"; do mkdir -p "$r/$(dirname "$rel")"; cp "$VLLM/$rel" "$r/$rel"; done
  echo "$r"
}
run() {  # run <root> [VAR=VAL ...] -> stdout+stderr in $OUT, exit code in $RC
  local root=$1; shift
  OUT="$(env -u VLLM_GB10_IDX_SHARD -u VLLM_GB10_IDX_SHARD_MIN_CTX -u VLLM_GB10_IDX_SHARD_MIN_ROWS VLLM_ROOT="$root" "$@" bash "$SH" 2>&1)"; RC=$?
}
count() { printf '%s\n' "$OUT" | grep -cE "$1" || true; }

ORIG="$(pin IDX_ORIG)"; PATCHED="$(pin IDX_P031)"
[ "$(sha "$VLLM/$IDX_REL")" = "$ORIG" ] && ok "S0 the v0.32.0 indexer.py in --vllm is the pinned original ($ORIG)" || bad "S0 --vllm indexer.py is not the pinned original"

# T1/T2: gate unset or 0 -> skipped, nothing touched
for spec in "unset:" "zero:VLLM_GB10_IDX_SHARD=0" "empty:VLLM_GB10_IDX_SHARD="; do
  label=${spec%%:*}; assign=${spec#*:}
  r="$(mkroot)"; before="$(sha "$r/$IDX_REL")"
  if [ -n "$assign" ]; then run "$r" "$assign"; else run "$r"; fi
  after="$(sha "$r/$IDX_REL")"
  [ $RC -eq 0 ] && [ "$(count '^\[OK\] patch 031 skipped \(VLLM_GB10_IDX_SHARD unset\); files untouched$')" = 1 ] && [ "$before" = "$after" ] \
    && ls "$r/$IDX_REL".* >/dev/null 2>&1 && bad "T1 gate $label: stray backup file left behind" \
    || { [ $RC -eq 0 ] && [ "$(count '^\[OK\] patch 031 skipped')" = 1 ] && [ "$before" = "$after" ] && ok "T1 gate $label: '[OK] patch 031 skipped', exit 0, indexer.py untouched" || bad "T1 gate $label rc=$RC: $OUT"; }
done

# T3: gate 1 -> applied, pinned result, one [OK] line, no leftovers
r="$(mkroot)"; run "$r" VLLM_GB10_IDX_SHARD=1
got="$(sha "$r/$IDX_REL")"
[ $RC -eq 0 ] && [ "$got" = "$PATCHED" ] && [ "$(count '^\[OK\] patch 031 in place')" = 1 ] && [ "$(grep -cF "$MARK" "$r/$IDX_REL")" = 4 ] \
  && ok "T3 gate 1: applied, sha256 == pinned result, exactly one '[OK] patch 031 in place', 4 markers" || bad "T3 rc=$RC got=$got: $OUT"
[ -z "$(ls "$r/$(dirname "$IDX_REL")" | grep -E 'p031\.(orig|tmp)')" ] && ok "T3b no .orig / .tmp left behind" || bad "T3b leftovers: $(ls "$r/$(dirname "$IDX_REL")")"
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$r/$IDX_REL" && ok "T3c the patched file parses" || bad "T3c parse"
[ "$(sha "$r/$OP_REL")" = "$(pin OP_ORIG)" ] && [ "$(sha "$r/$GLM_REL")" = "$(pin GLMIDX_ORIG)" ] && ok "T3d the two read-only contract files are untouched" || bad "T3d contract files changed"
echo "$OUT" | grep -q 'MIN_CTX=unset -> 32768' && ok "T3e the banner shows the defaults" || bad "T3e banner: $(echo "$OUT" | head -1)"

# T4: second run (reused container) is idempotent
run "$r" VLLM_GB10_IDX_SHARD=1
[ $RC -eq 0 ] && [ "$(sha "$r/$IDX_REL")" = "$PATCHED" ] && [ "$(count '^\[OK\] patch 031 in place')" = 1 ] && [ "$(count '\[skip\] indexer.py already carries patch 031')" = 1 ] \
  && ok "T4 second run: '[skip] already carries', pinned sha unchanged, one [OK] line" || bad "T4 rc=$RC: $OUT"

# T5: gate unset on an already patched file: note + skipped, file untouched
run "$r"
[ $RC -eq 0 ] && [ "$(sha "$r/$IDX_REL")" = "$PATCHED" ] && [ "$(count '\[note\] VLLM_GB10_IDX_SHARD unset but indexer.py already carries patch 031')" = 1 ] && [ "$(count '^\[OK\] patch 031 skipped')" = 1 ] \
  && ok "T5 gate unset on a patched file: note + skipped, file untouched" || bad "T5 rc=$RC: $OUT"

# T6: bad gate values / bad integers are refused before anything is touched
for v in "2" "yes" "true" "on"; do
  r="$(mkroot)"; before="$(sha "$r/$IDX_REL")"; run "$r" VLLM_GB10_IDX_SHARD="$v"
  [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$IDX_REL")" ] && echo "$OUT" | grep -q "must be unset/0 (inert) or 1" && ok "T6 VLLM_GB10_IDX_SHARD='$v' refused (rc=$RC), file untouched" || bad "T6 '$v' rc=$RC: $OUT"
done
for kv in "VLLM_GB10_IDX_SHARD_MIN_CTX=100" "VLLM_GB10_IDX_SHARD_MIN_CTX=abc" "VLLM_GB10_IDX_SHARD_MIN_CTX=99999999999" "VLLM_GB10_IDX_SHARD_MIN_CTX=4194305" \
          "VLLM_GB10_IDX_SHARD_MIN_ROWS=5" "VLLM_GB10_IDX_SHARD_MIN_ROWS=8193" "VLLM_GB10_IDX_SHARD_MIN_ROWS=1.5"; do
  r="$(mkroot)"; before="$(sha "$r/$IDX_REL")"; run "$r" VLLM_GB10_IDX_SHARD=1 "$kv"
  [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$IDX_REL")" ] && ok "T6 $kv refused (rc=$RC), file untouched" || bad "T6 $kv rc=$RC: $OUT"
done
for kv in "VLLM_GB10_IDX_SHARD_MIN_CTX=65536" "VLLM_GB10_IDX_SHARD_MIN_CTX=4096" "VLLM_GB10_IDX_SHARD_MIN_ROWS=8192" "VLLM_GB10_IDX_SHARD_MIN_ROWS=64"; do
  r="$(mkroot)"; run "$r" VLLM_GB10_IDX_SHARD=1 "$kv"
  [ $RC -eq 0 ] && [ "$(sha "$r/$IDX_REL")" = "$PATCHED" ] && ok "T6 $kv accepted (the env is read at runtime, the file is the same)" || bad "T6 $kv rc=$RC: $OUT"
done
r="$(mkroot)"; run "$r" VLLM_GB10_IDX_SHARD=1 VLLM_GB10_IDX_SHARD_MIN_CTX=65536
echo "$OUT" | grep -q "in place (VLLM_GB10_IDX_SHARD=1: .*context >= 65536 tokens and >= 256 rows" && ok "T6b the [OK] line shows the overridden MIN_CTX" || bad "T6b: $(echo "$OUT" | tail -1)"

# T7: a changed base is refused and left alone
r="$(mkroot)"; printf '\n# drift\n' >> "$r/$IDX_REL"; before="$(sha "$r/$IDX_REL")"; run "$r" VLLM_GB10_IDX_SHARD=1
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$IDX_REL")" ] && echo "$OUT" | grep -q "is not the pinned v0.32.0 original" && ok "T7 drifted indexer.py refused, file untouched" || bad "T7 rc=$RC: $OUT"
# T8: a foreign revision's marker is refused
r="$(mkroot)"; printf '\n%s foreign edit\n' "$MARK" >> "$r/$IDX_REL"; before="$(sha "$r/$IDX_REL")"; run "$r" VLLM_GB10_IDX_SHARD=1
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$IDX_REL")" ] && echo "$OUT" | grep -q "not this revision's pinned result" && ok "T8 foreign patch-031 edit refused, file untouched" || bad "T8 rc=$RC: $OUT"
# T9: read-only contracts drifted
for rel in "$OP_REL" "$GLM_REL"; do
  r="$(mkroot)"; printf '\n# drift\n' >> "$r/$rel"; before="$(sha "$r/$IDX_REL")"; run "$r" VLLM_GB10_IDX_SHARD=1
  [ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$IDX_REL")" ] && echo "$OUT" | grep -q "re-validate and re-pin" && ok "T9 drifted $rel refused before any write" || bad "T9 $rel rc=$RC: $OUT"
done
# T9c: the SM121-baked sparse_indexer.py (what the built image carries) is accepted
BAKE="$PDIR/../image/glm53_sm121_v032.py"
if [ -f "$BAKE" ]; then
  r="$(mkroot)"
  python3 - "$BAKE" "$r/$GLM_REL" "$GLM_REL" <<'PYEOF' || bad "T9c could not derive the baked file"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("bake", sys.argv[1]); m = importlib.util.module_from_spec(spec)
sys.modules["bake"] = m; spec.loader.exec_module(m)
p = sys.argv[2]; rel = sys.argv[3]; t = open(p).read()
for e in m.EDITS:
    if e.pkg == "vllm" and e.rel == rel:
        assert t.count(e.old) == 1; t = t.replace(e.old, e.new, 1)
open(p, "w").write(t)
PYEOF
  [ "$(sha "$r/$GLM_REL")" = "$(pin GLMIDX_BAKED)" ] && ok "T9c the derived baked file matches the GLMIDX_BAKED pin" || bad "T9c baked sha $(sha "$r/$GLM_REL") != pin $(pin GLMIDX_BAKED)"
  run "$r" VLLM_GB10_IDX_SHARD=1
  [ $RC -eq 0 ] && [ "$(sha "$r/$IDX_REL")" = "$PATCHED" ] && ok "T9c the SM121-baked sparse_indexer.py is accepted: 031 applies" || bad "T9c rc=$RC: $OUT"
else
  echo "[SKIP] T9c SM121-baked sparse_indexer.py ($BAKE not found)"
fi
# T10: missing file
r="$(mkroot)"; rm "$r/$GLM_REL"; run "$r" VLLM_GB10_IDX_SHARD=1
[ $RC -ne 0 ] && echo "$OUT" | grep -q "not found" && ok "T10 missing models/glm5next/nvidia/sparse_indexer.py refused" || bad "T10 rc=$RC: $OUT"

# T11: a tampered applier (pinned sha) and a wrong result pin (rollback)
P2="$RUN/patches_tampered"; cp -R "$PDIR" "$P2"; printf '\n# x\n' >> "$P2/files/p031_apply.py"
r="$(mkroot)"; before="$(sha "$r/$IDX_REL")"
OUT="$(env -u VLLM_GB10_IDX_SHARD VLLM_ROOT="$r" VLLM_GB10_IDX_SHARD=1 bash "$P2/031-gb10-idx-shard-gate.sh" 2>&1)"; RC=$?
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$IDX_REL")" ] && echo "$OUT" | grep -q "p031_apply.py sha256" && ok "T11 tampered applier refused, file untouched" || bad "T11 rc=$RC: $OUT"
P3="$RUN/patches_badpin"; cp -R "$PDIR" "$P3"
sed -i.bak -E "s/^IDX_P031=.*/IDX_P031=0000000000000000000000000000000000000000000000000000000000000000/" "$P3/031-gb10-idx-shard-gate.sh"; rm -f "$P3/031-gb10-idx-shard-gate.sh.bak"
r="$(mkroot)"; before="$(sha "$r/$IDX_REL")"
OUT="$(env -u VLLM_GB10_IDX_SHARD VLLM_ROOT="$r" VLLM_GB10_IDX_SHARD=1 bash "$P3/031-gb10-idx-shard-gate.sh" 2>&1)"; RC=$?
[ $RC -ne 0 ] && [ "$before" = "$(sha "$r/$IDX_REL")" ] && echo "$OUT" | grep -q "restored to its previous bytes" && [ -z "$(ls "$r/$(dirname "$IDX_REL")" | grep -E 'p031\.(orig|tmp)')" ] \
  && ok "T11b result sha != pin: failure AFTER the write restores indexer.py byte for byte, no leftovers" || bad "T11b rc=$RC: $OUT"

# T12: the applier mid-run is TERMinated -> the file is restored (a stub applier that sleeps; its sha is re-pinned in the copy)
P4="$RUN/patches_term"; cp -R "$PDIR" "$P4"
cat > "$P4/files/p031_apply.py" <<'PYEOF'
import sys, time, pathlib
p = pathlib.Path(sys.argv[2]); p.write_text(p.read_text() + "\n# half-written\n"); time.sleep(30)
PYEOF
sed -i.bak -E "s/^APPLIER_SHA=.*/APPLIER_SHA=$(sha "$P4/files/p031_apply.py")/" "$P4/031-gb10-idx-shard-gate.sh"; rm -f "$P4/031-gb10-idx-shard-gate.sh.bak"
r="$(mkroot)"; before="$(sha "$r/$IDX_REL")"
( env -u VLLM_GB10_IDX_SHARD VLLM_ROOT="$r" VLLM_GB10_IDX_SHARD=1 bash "$P4/031-gb10-idx-shard-gate.sh" >"$RUN/term.out" 2>&1 & echo $! >"$RUN/term.pid"; wait )&
sleep 2; kill -TERM "$(cat "$RUN/term.pid")" 2>/dev/null; sleep 2
[ "$before" = "$(sha "$r/$IDX_REL")" ] && ok "T12 SIGTERM while the applier runs: indexer.py restored byte for byte" || bad "T12 indexer.py not restored: $(cat "$RUN/term.out")"
pkill -f "$P4/files/p031_apply.py" 2>/dev/null || true

# T13: banner / header facts
head -1 "$SH" | grep -q '^#!/usr/bin/env bash' && ok "T13 shebang" || bad "T13 shebang"
bash -n "$SH" && ok "T13b bash -n" || bad "T13b bash -n"

echo "# p031_sh_check: $pass passed, $fail failed (scratch: $RUN)"
[ "$fail" -eq 0 ]
