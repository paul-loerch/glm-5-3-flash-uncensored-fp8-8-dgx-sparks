#!/usr/bin/env bash
# [patch-020 gb10-sparse-mla-triton-prefill] .sh scenario checks (revision 4, vLLM v0.32.0 port). Runs
# patches/020-gb10-sparse-mla-triton-prefill.sh against scratch VLLM_ROOT trees seeded with the STOCK v0.32.0
# flashinfer_mla_sparse_sm90.py / platforms/cuda.py and with the SM121-baked variants derived from them; never
# touches a real vLLM install. Needs bash (3.2 is fine), python3 + numpy (no torch), sha256sum or shasum.
#   bash files/p020_sh_check.sh --vllm-root <pristine v0.32.0 tree>/vllm --work <scratch dir> [--patches <dir>]
#                               [--baked-root <tree>/vllm produced by running the image build's bake script on a copy]
set -u
PDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=""; WORK=""; BAKED_ROOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --vllm-root) ROOT="$2"; shift 2 ;;
    --baked-root) BAKED_ROOT="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --patches) PDIR="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
REL=v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py
CREL=platforms/cuda.py
[ -f "$ROOT/$REL" ] && [ -f "$ROOT/$CREL" ] && [ -n "$WORK" ] || { echo "usage: $0 --vllm-root <vllm dir with $REL and $CREL> --work DIR [--patches DIR]" >&2; exit 2; }
SH=020-gb10-sparse-mla-triton-prefill.sh
M="# [patch-020 gb10-sparse-mla-triton-prefill]"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
pin() { sed -n "s/^$1=\([0-9a-f]\{64\}\)$/\1/p" "$PDIR/$SH"; }
rm -rf "$WORK"; mkdir -p "$WORK/roots" "$WORK/bases"
P="$WORK/patches"; cp -R "$PDIR" "$P"; rm -rf "$P/files/__pycache__"
REALPY="$(command -v python3)"
pass=0; fail=0
ok()  { echo "[PASS] $*"; pass=$((pass+1)); }
bad() { echo "[FAIL] $*"; fail=$((fail+1)); }

# the baked variants and a baked cuda.py (the MiaAI-Lab edits, copied into p020_cpu_check.py)
"$REALPY" - "$ROOT" "$WORK/bases" "$PDIR/files" "$BAKED_ROOT" <<'EOF' || { echo "cannot derive the baked variants"; exit 2; }
import sys
from pathlib import Path
sys.dont_write_bytecode = True
root, out, files, baked_root = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, files)
import p020_cpu_check as C
stock = (root / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py").read_text()
bases = C.make_bases(stock)
cuda = (root / "platforms/cuda.py").read_text()
cuda_baked = C._once(cuda, C.CUDA_OLD, C.CUDA_NEW, "cuda.py")
if baked_root:  # the bake script's own output replaces the derived files
    bases["BAKED"] = (Path(baked_root) / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py").read_text()
    cuda_baked = (Path(baked_root) / "platforms/cuda.py").read_text()
for k, v in bases.items():
    (out / f"sm90_{k}.py").write_text(v)
(out / "cuda_STOCK.py").write_text(cuda)
(out / "cuda_BAKED.py").write_text(cuda_baked)
EOF
B="$WORK/bases"
STOCK=$(sha "$B/sm90_STOCK.py"); BAKED=$(sha "$B/sm90_BAKED.py"); ALT=$(sha "$B/sm90_ALT.py")
P_BAKED=$(pin SM_BAKED_P020)

mkroot() { # name base cuda
  local r="$WORK/roots/$1"; rm -rf "$r"; mkdir -p "$r/$(dirname "$REL")" "$r/platforms"
  cp "$B/sm90_$2.py" "$r/$REL"; [ -n "${3:-}" ] && cp "$B/cuda_$3.py" "$r/$CREL"; echo "$r"
}
run() { local r=$1; shift; env -i PATH="${SHIM:-}$PATH" HOME="$HOME" VLLM_ROOT="$r" "$@" bash "$P/$SH" >>"$r.log" 2>&1; echo $?; }
runp() { local s=$1 r=$2; shift 2; env -i PATH="${SHIM:-}$PATH" HOME="$HOME" VLLM_ROOT="$r" "$@" bash "$s" >>"$r.log" 2>&1; echo $?; }
mod_absent() { [ ! -e "$1/_gb10_sparse_mla_prefill.py" ]; }
no_leftovers() { ! ls "$1"/*.p020.* "$1/$(dirname "$REL")"/*.p020.* >/dev/null 2>&1; }
final_lines() { grep -cE '^\[(OK|state)\] patch 020' "$1.log"; }
G=VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1

[ "$STOCK" = "$(pin SM_STOCK)" ] && [ "$BAKED" = "$(pin SM_BAKED)" ] && [ "$ALT" != "$BAKED" ] \
  && ok "S0 --vllm-root is the pinned pristine v0.32.0 file; the frozen-bake file ${BAKED_ROOT:+(produced by the bake script) }carries the pinned sha (the ALT spelling is not pinned)" \
  || bad "S0 shas stock=$STOCK baked=$BAKED alt=$ALT"

# S1 gate unset: skip on every base, nothing touched
for v in STOCK BAKED ALT; do
  r=$(mkroot "s1_$v" $v BAKED); rc=$(run "$r"); got=$(sha "$r/$REL")
  [ "$rc" = 0 ] && [ "$got" = "$(sha "$B/sm90_$v.py")" ] && mod_absent "$r" && [ "$(final_lines "$r")" = 1 ] \
    && grep -q "skipped (VLLM_GB10_SPARSE_MLA_TRITON_PREFILL unset)" "$r.log" \
    && ok "S1 $v: gate unset -> skipped, untouched, one result line" || bad "S1 $v rc=$rc"
done
# S1b the obsolete gates alone do not enable anything but are called out
r=$(mkroot s1b BAKED BAKED); rc=$(run "$r" VLLM_GB10_SPARSE_MLA_NOSYNC=1 VLLM_GB10_SPARSE_MLA_SYNCFREE=1)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$BAKED" ] && mod_absent "$r" && grep -q "NOSYNC=1 is obsolete" "$r.log" && grep -q "SYNCFREE=1 belonged to patch 021" "$r.log" \
  && ok "S1b NOSYNC / SYNCFREE alone: skipped + a note each, untouched" || bad "S1b rc=$rc"

# S2 apply on the real bake: result sha, markers, module, no leftovers, exactly one result line
r=$(mkroot s2_BAKED BAKED BAKED); rc=$(run "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$P_BAKED" ] && [ "$(grep -cF "$M" "$r/$REL")" = 5 ] \
  && [ "$(sha "$r/_gb10_sparse_mla_prefill.py")" = "$(sha "$P/files/gb10_sparse_mla_prefill.py")" ] && no_leftovers "$r" \
  && [ "$(final_lines "$r")" = 1 ] && grep -q "^\[OK\] patch 020 in place" "$r.log" && ! grep -q "platforms/cuda.py's capability-12" "$r.log" \
  && ok "S2 BAKED: apply -> pinned result sha, 5 markers, module installed, no backups, one '[OK] patch 020 in place' line" \
  || { bad "S2 BAKED rc=$rc"; tail -4 "$r.log"; }
# S2b the ALT spelling of the bake is a different file: not pinned -> refused with the re-pin hint (S11 re-pins it)
r=$(mkroot s2b ALT BAKED); rc=$(run "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$REL")" = "$ALT" ] && mod_absent "$r" && grep -q "p020_repin.sh" "$r.log" \
  && ok "S2b an unpinned bake spelling (ALT) is refused, untouched" || bad "S2b rc=$rc"
S2="$WORK/roots/s2_BAKED"
rc=$(run "$S2" $G)
[ "$rc" = 0 ] && [ "$(sha "$S2/$REL")" = "$P_BAKED" ] && grep -q "already patched" "$S2.log" && no_leftovers "$S2" && ok "S3 idempotent re-run" || bad "S3 rc=$rc"
rc=$(run "$S2")
[ "$rc" = 0 ] && grep -q "gate unset but flashinfer_mla_sparse_sm90.py already carries patch 020" "$S2.log" && [ "$(sha "$S2/$REL")" = "$P_BAKED" ] \
  && ok "S3b gate unset on a patched file: note + skip, hooks stay inert" || bad "S3b rc=$rc"
# S3c a missing cuda.py (probe 'unknown') does not stop the apply
r=$(mkroot s3c BAKED ""); rm -f "$r/$CREL"; rc=$(run "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$P_BAKED" ] && ok "S3c no platforms/cuda.py -> apply proceeds (probe unknown)" || bad "S3c rc=$rc"
# S3d baked backend but a stock cuda.py: apply + the note
r=$(mkroot s3d BAKED STOCK); rc=$(run "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$P_BAKED" ] && grep -q "capability-12 backend list does not name FLASHINFER_MLA_SPARSE_SM90" "$r.log" && [ "$(final_lines "$r")" = 1 ] \
  && ok "S3d baked backend + stock cuda.py: applied, one note about explicit --attention-backend" || bad "S3d rc=$rc"

# S4 PRISTINE file (SM121 bake absent): inert, ONE state line, nothing touched, rc 0
r=$(mkroot s4 STOCK STOCK); rc=$(run "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$STOCK" ] && mod_absent "$r" && no_leftovers "$r" && [ "$(final_lines "$r")" = 1 ] \
  && grep -q "^\[state\] patch 020 inert: .*rejects capability 12.1 (SM121 bake absent)" "$r.log" \
  && ok "S4 pristine SM90 backend (rejects capability 12.1): inert, one '[state] patch 020 inert' line, untouched" || { bad "S4 rc=$rc"; tail -3 "$r.log"; }
# S4b ... a pristine-based file that some other tool already hooked (capability still rejected): still inert, untouched
r=$(mkroot s4b STOCK STOCK); cp "$B/sm90_STOCK.py" "$r/$REL"; "$REALPY" "$P/files/p020_apply.py" apply "$r/$REL"
before=$(sha "$r/$REL"); rc=$(run "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$before" ] && grep -q "^\[state\] patch 020 inert" "$r.log" && ok "S4b a hooked pristine-based file (capability still rejected): inert, untouched" || bad "S4b rc=$rc"
# S4c the probe itself unavailable (simulated failure): the PRISTINE sha still means 'bake absent' -> inert, never patched
mkdir -p "$WORK/shimp"
cat > "$WORK/shimp/python3" <<EOF3
#!/usr/bin/env bash
case "\$*" in *p020_apply.py\ probe*) echo "shim: probe failure" >&2; exit 1 ;; esac
exec "$REALPY" "\$@"
EOF3
chmod +x "$WORK/shimp/python3"
r=$(mkroot s4c STOCK STOCK); rc=$(SHIM="$WORK/shimp:" run "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$STOCK" ] && mod_absent "$r" && [ "$(final_lines "$r")" = 1 ] && grep -q "is the PRISTINE v0.32.0 file (SM121 bake absent" "$r.log" \
  && ok "S4c probe unavailable + pristine sha: inert by the pin alone, untouched" || { bad "S4c rc=$rc"; tail -3 "$r.log"; }

# S5 reused container carrying a patch-020 edit that is not this revision's output -> refuse, untouched
r=$(mkroot s5 BAKED BAKED); cp "$S2/$REL" "$r/$REL"; printf '\n# another 020 revision\n' >> "$r/$REL"
before=$(sha "$r/$REL"); rc=$(run "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$REL")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "not this revision's output" "$r.log" \
  && ok "S5 patch-020-marked file with another sha refused (recreate the container), untouched" || bad "S5 rc=$rc"
# S6 unexpected target (local edit) -> refuse, untouched, no module
r=$(mkroot s6 BAKED BAKED); printf '\n# local edit\n' >> "$r/$REL"; before=$(sha "$r/$REL"); rc=$(run "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$REL")" = "$before" ] && mod_absent "$r" && no_leftovers "$r" && grep -q "neither the pinned SM121-baked v0.32.0 file nor the patch-020 result" "$r.log" \
  && ok "S6 unexpected (re-baked / locally edited) target refused with the re-pin hint, untouched" || bad "S6 rc=$rc"
# S6b a baked file whose anchor drifted (different sha) -> refused by the pin, never half-applied
r=$(mkroot s6b BAKED BAKED)
"$REALPY" - "$r/$REL" <<'EOF'
import sys; p = sys.argv[1]; t = open(p).read()
old = "        self.use_fp8_kv_cache = self.kv_cache_dtype in _FP8_KV_DTYPES\n"
assert t.count(old) == 1; open(p, "w").write(t.replace(old, old + "        self._x = 1\n"))
EOF
before=$(sha "$r/$REL"); rc=$(run "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$REL")" = "$before" ] && mod_absent "$r" && ok "S6b drifted baked file refused by the pin, untouched" || bad "S6b rc=$rc"
# S6c gate set but the target is missing
r="$WORK/roots/s6c"; rm -rf "$r"; mkdir -p "$r"; rc=$(run "$r" $G)
[ "$rc" != 0 ] && grep -q "not found" "$r.log" && ok "S6c gate set + no target file -> FATAL" || bad "S6c rc=$rc"

# S7 tampered applier / module -> refuse before touching anything
for f in p020_apply.py gb10_sparse_mla_prefill.py; do
  T="$WORK/tamper"; rm -rf "$T"; cp -R "$P" "$T"; printf '\n# tamper\n' >> "$T/files/$f"
  r=$(mkroot "s7_$f" BAKED BAKED); rc=$(runp "$T/$SH" "$r" $G)
  [ "$rc" != 0 ] && [ "$(sha "$r/$REL")" = "$BAKED" ] && mod_absent "$r" && ok "S7 tampered files/$f refused, untouched" || bad "S7 $f rc=$rc"
done

# S8 rollback after the module was installed (simulated applier failure): the target AND a PRE-EXISTING
# module's bytes are restored; a module that did not exist is removed
mkdir -p "$WORK/shim"
cat > "$WORK/shim/python3" <<EOF
#!/usr/bin/env bash
case "\$*" in *p020_apply.py\ apply*) echo "shim: simulated applier failure" >&2; exit 1 ;; esac
exec "$REALPY" "\$@"
EOF
chmod +x "$WORK/shim/python3"
r=$(mkroot s8 BAKED BAKED); echo "# old module bytes" > "$r/_gb10_sparse_mla_prefill.py"
rc=$(SHIM="$WORK/shim:" run "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$REL")" = "$BAKED" ] && grep -qx "# old module bytes" "$r/_gb10_sparse_mla_prefill.py" && no_leftovers "$r" && grep -q "restored" "$r.log" \
  && ok "S8 rollback restores the target + a PRE-EXISTING module's bytes" || { bad "S8 rc=$rc"; tail -4 "$r.log"; }
r=$(mkroot s8b BAKED BAKED); rc=$(SHIM="$WORK/shim:" run "$r" $G)
[ "$rc" != 0 ] && [ "$(sha "$r/$REL")" = "$BAKED" ] && mod_absent "$r" && no_leftovers "$r" && ok "S8b rollback removes a module this run installed" || bad "S8b rc=$rc"

# S9 applier round trip: apply -> revert is byte-exact on every base; the BAKED result is the pinned one
for v in STOCK BAKED ALT; do
  cp "$B/sm90_$v.py" "$WORK/rev.py"; "$REALPY" "$P/files/p020_apply.py" apply "$WORK/rev.py"
  got=$(sha "$WORK/rev.py"); want=$got; [ "$v" = BAKED ] && want=$P_BAKED
  "$REALPY" "$P/files/p020_apply.py" revert "$WORK/rev.py"
  [ "$got" = "$want" ] && [ "$(sha "$WORK/rev.py")" = "$(sha "$B/sm90_$v.py")" ] && ok "S9 $v: apply -> revert == base (BAKED: and the result == the pinned SM_BAKED_P020; STOCK / ALT are applier-level only)" || bad "S9 $v"
done

# S10 the installed file compiles under the system python (syntax of the patched backend + module)
r="$WORK/roots/s2_BAKED"
"$REALPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$r/$REL" "$r/_gb10_sparse_mla_prefill.py" \
  && ok "S10 patched backend + installed module parse" || bad "S10 parse"

# S11 the re-pin flow for the REAL baked image file: repin prints SM_REAL / SM_REAL_P020, pasting them into a copy of
# the .sh makes it apply that file (pinned result), and the original pins still refuse it
REAL="$WORK/bases/sm90_REAL.py"; { cat "$B/sm90_ALT.py"; printf '\n# some other bake edit elsewhere in the file\n'; } > "$REAL"
out=$(bash "$P/files/p020_repin.sh" "$REAL" "$B/cuda_BAKED.py" 2>&1); rc=$?
real_base=$(printf '%s\n' "$out" | sed -n 's/^SM_REAL=\([0-9a-f]\{64\}\)$/\1/p'); real_res=$(printf '%s\n' "$out" | sed -n 's/^SM_REAL_P020=\([0-9a-f]\{64\}\)$/\1/p')
[ "$rc" = 0 ] && [ "$real_base" = "$(sha "$REAL")" ] && [ -n "$real_res" ] && printf '%s' "$out" | grep -q "probe .*: yes yes" \
  && ok "S11 p020_repin.sh on a re-baked file: base + result pins printed, probe 'yes yes' with the baked cuda.py" || bad "S11 repin rc=$rc: $out"
T="$WORK/repinned"; rm -rf "$T"; cp -R "$P" "$T"
sed -i.bak -e "s/^SM_REAL=0\{64\}$/SM_REAL=$real_base/" -e "s/^SM_REAL_P020=0\{64\}$/SM_REAL_P020=$real_res/" "$T/$SH"; rm -f "$T/$SH.bak"
r="$WORK/roots/s11"; rm -rf "$r"; mkdir -p "$r/$(dirname "$REL")" "$r/platforms"; cp "$REAL" "$r/$REL"; cp "$B/cuda_BAKED.py" "$r/$CREL"
rc=$(runp "$T/$SH" "$r" $G)
[ "$rc" = 0 ] && [ "$(sha "$r/$REL")" = "$real_res" ] && [ "$(grep -cF "$M" "$r/$REL")" = 5 ] && ok "S11b the re-pinned .sh applies the real baked file (result == printed pin)" || { bad "S11b rc=$rc"; tail -3 "$r.log"; }
rc=$(runp "$T/$SH" "$r" $G)
[ "$rc" = 0 ] && grep -q "already patched" "$r.log" && ok "S11c ... and is idempotent on it" || bad "S11c rc=$rc"
# repin refuses a file whose anchor drifted
DRIFT="$WORK/bases/sm90_DRIFT.py"
"$REALPY" - "$B/sm90_BAKED.py" "$DRIFT" <<'EOF2'
import sys; t = open(sys.argv[1]).read()
old = "        state = attn_metadata.state\n        assert state is not None\n"
assert t.count(old) == 1; open(sys.argv[2], "w").write(t.replace(old, "        state = attn_metadata.state  # drift\n        assert state is not None\n"))
EOF2
bash "$P/files/p020_repin.sh" "$DRIFT" >/dev/null 2>&1; rc=$?
[ "$rc" != 0 ] && ok "S11d p020_repin.sh refuses a file whose anchor drifted (exit $rc)" || bad "S11d rc=$rc"

echo "sh scenarios: $pass passed, $fail failed"
[ "$fail" = 0 ]
