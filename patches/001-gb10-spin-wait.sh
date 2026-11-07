#!/usr/bin/env bash
# hotfix-gb10-spin-wait.sh — Reduce vLLM IPC spin-wait CPU load / SoC heat on GB10
#
# ROOT CAUSE: vLLM's cross-process IPC wait path (shm_broadcast.SpinCondition)
# spins on performance cores for `busy_loop_s` (default 1s) before sleeping.
# Under decode the message gap is always < 1s, so the sleep path never engages
# and 3–4 P-cores spin at max clock (issue #79).
#
# FIX (one value):
#   vllm/distributed/device_communicators/shm_broadcast.py
#   SpinCondition.__init__  busy_loop_s: float = 1 → 0.002  (2ms grace, then sleep via notify socket)
#
# TP>=2 only (this recipe). TP=1 uses the uni executor and never forms this
# path. Wait policy only — model outputs unchanged.
#
# vLLM v0.32.0 PORT (base image vllm/vllm-openai:v0.32.0-aarch64): the edited line is byte-identical to the day-0 image's
# (`busy_loop_s: float = 1,` in SpinCondition.__init__, v0.32.0 shm_broadcast.py line 138); upstream still has no knob for
# it (PR #52814 unmerged), so the patch is carried as is. The edit is applied by files/p001_apply.py (one anchored edit,
# AST-checked: the anchor must be the default of SpinCondition.__init__'s busy_loop_s parameter). The file is now pinned by
# sha256 (v0.32.0 original and patch-001 result; any other content is refused: fail closed, no half state). Messages
# and the `busy_loop_s = 0.002` verification line (a launch/readiness.txt `require` rule) are unchanged. The old header's
# "Skip: DSPARK_SKIP_SPIN_WAIT_HOTFIX=1" was never read by this script and is gone.
#
# Applied from the recipe entrypoint before exec vllm. Idempotent.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
SPIN_WAIT_FILE="$VLLM_ROOT/distributed/device_communicators/shm_broadcast.py"
APPLIER="$SRC_DIR/p001_apply.py"
READY_NEW="busy_loop_s: float = 0.002"

# pins (sha256): the file in vllm/vllm-openai:v0.32.0-aarch64 (= the v0.32.0 tag's vllm/distributed/...), its patch-001 result,
# and the vendored applier
SHM_ORIG=8b122fa8a8ae05a489c2a15069e67d17b626da5d9939524f088cda2ef3c1d75b
SHM_P001=c2d10cb80e0c8085f642fd3b4c6635552ad701f419b2f77cff7300bda0d9cfcd
APPLIER_SHA=d6d0889f4c9ab735b0245f54511d70e58cce44fb46336091752645efca989627

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[001] FATAL: $*" >&2; exit 1; }

if [ ! -f "$SPIN_WAIT_FILE" ]; then
  echo "ERROR: vLLM shm_broadcast.py not found at $SPIN_WAIT_FILE" >&2
  exit 1
fi

echo "=== Hotfix: GB10 vLLM IPC spin-wait (CPU load / SoC heat, issue #79) ==="
echo "vLLM file: $SPIN_WAIT_FILE"

[ -f "$APPLIER" ] || fail "$APPLIER not found"
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p001_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
now="$(sha "$SPIN_WAIT_FILE")"
if [ "$now" = "$SHM_P001" ]; then
  echo " [skip] busy_loop_s already set to 0.002"
elif [ "$now" = "$SHM_ORIG" ]; then
  python3 "$APPLIER" apply "$SPIN_WAIT_FILE" || fail "p001_apply.py refused shm_broadcast.py"
  got="$(sha "$SPIN_WAIT_FILE")"
  [ "$got" = "$SHM_P001" ] || fail "patched shm_broadcast.py sha256 $got != pinned $SHM_P001"
else
  fail "shm_broadcast.py sha256 $now is neither the v0.32.0 original ($SHM_ORIG) nor patch 001's result ($SHM_P001); refusing (a different vLLM build? re-pin after reading the diff)"
fi

echo "=== Verification ==="
if grep -qF "$READY_NEW" "$SPIN_WAIT_FILE"; then
  echo "[OK] busy_loop_s = 0.002 — IPC wait sleeps after 2ms instead of spinning 1s."
else
  echo "[FAIL] busy_loop_s not set to 0.002" >&2
  exit 1
fi
