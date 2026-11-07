#!/usr/bin/env bash
# 003-gb10-blocking-sync.sh — Opt-in CUDA blocking-sync for vLLM worker processes.
#
# WHY: under async scheduling the SM90 sparse-MLA builder waits for the GPU once per step via a D2H copy
# (flashinfer_mla_sparse_sm90.py:350 on the day-0 image). CUDA's default schedule spins during that wait, pegging one P-core
# per rank for ~the whole GPU step (py-spy + per-thread sampling). BLOCKING_SYNC makes the
# thread sleep until the GPU signals; it keeps async scheduling (unlike the --no-async-scheduling workaround).
#
# WHAT: installs files/gb10_blocking_sync.py as vllm/_gb10_blocking_sync.py and wires two calls into
# vllm/v1/worker/gpu_worker.py Worker.init_device around `torch.accelerator.set_device_index(self.device)`
# (the call that creates the primary context) through files/p003_apply.py. Both calls are no-ops unless
# VLLM_GB10_BLOCKING_SYNC=1, so this patch changes nothing for configs that don't set the env var.
# Fail-closed: pinned applier / module / target shas, anchor missing -> exit 1 (entrypoint aborts the boot); env set but
# flag not verified -> the worker raises at init. Idempotent (the patched file's sha).
#
# vLLM v0.32.0 PORT (base image vllm/vllm-openai:v0.32.0-aarch64): the anchor pair (`self.device = torch.device(f"cuda:...")`
# directly followed by `torch.accelerator.set_device_index(self.device)`, now lines 500-501 of gpu_worker.py) is unchanged
# and unique; nothing in Worker.init_device before it creates the CUDA context (the diff of init_device against the day-0
# image adds the PCP world-size term to the local-rank arithmetic BEFORE the anchor, pure Python, and
# warn_if_flashinfer_kernels_missing(), the external-weight-memory request and the _init_workspace_and_model_runner()
# split AFTER it). The module file is unchanged except for its docstring.
# NOTE for the A/B: v0.32.0's SM90 sparse-MLA builder is sync-free under async scheduling (seq_lens_cpu_upper_bound,
# upstream #58684), so the wait this patch was written for is mostly gone; the benefit must be re-measured, the patch is
# opt-in. Messages are those of the day-0 port. The file is pinned by sha256 (v0.32.0 original / patch-003 result).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
WORKER="$VLLM_ROOT/v1/worker/gpu_worker.py"
APPLIER="$SRC_DIR/p003_apply.py"
MODULE_SRC="$SRC_DIR/gb10_blocking_sync.py"
MODULE_DST="$VLLM_ROOT/_gb10_blocking_sync.py"
MARKER="# [patch-003 gb10-blocking-sync]"

# pins (sha256): gpu_worker.py of vllm/vllm-openai:v0.32.0-aarch64 and its patch-003 result; the vendored applier and module
WORKER_ORIG=e970aa29a738aa15cd963d9f9202ca6d3d504e98fd344959fd998e940472a5a0
WORKER_P003=7b03ce97d577d6d78cf0ab49936204edafdcdc738b015616d77243cfb35ee9b3
APPLIER_SHA=7bf6579b04cd87c648e43ab47b1917ab1f9f10aa88ce7a4c0c78f551dc1c4094
MOD_SHA=4c26f3d92d84b59ec0deb76e320843df38b55cf31acb906291428efed8c234d9

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[FAIL] [003] $*" >&2; exit 1; }

echo "=== Patch 003: GB10 CUDA blocking-sync hook (opt-in VLLM_GB10_BLOCKING_SYNC=1) ==="
[ -f "$WORKER" ] || { echo "[FAIL] $WORKER not found" >&2; exit 1; }
[ -f "$MODULE_SRC" ] || { echo "[FAIL] $MODULE_SRC not found" >&2; exit 1; }
[ -f "$APPLIER" ] || { echo "[FAIL] $APPLIER not found" >&2; exit 1; }
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p003_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MODULE_SRC")" = "$MOD_SHA" ] || fail "files/gb10_blocking_sync.py sha256 $(sha "$MODULE_SRC") != pinned $MOD_SHA"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MODULE_SRC" || fail "$MODULE_SRC does not parse"

now="$(sha "$WORKER")"
mode=""
if [ "$now" = "$WORKER_P003" ]; then
  mode=installed
elif [ "$now" = "$WORKER_ORIG" ]; then
  mode=apply
else
  if grep -qF "$MARKER" "$WORKER"; then
    fail "gpu_worker.py carries a patch-003 edit that is not this revision's pinned result (sha ${now:0:16}...); recreate the container from the image"
  fi
  fail "gpu_worker.py sha256 $now is neither the v0.32.0 original ($WORKER_ORIG) nor patch 003's result ($WORKER_P003); refusing (a different vLLM build? re-pin after reading the diff)"
fi

install -m 0644 "$MODULE_SRC" "$MODULE_DST"
[ "$(sha "$MODULE_DST")" = "$MOD_SHA" ] || fail "installed $MODULE_DST sha256 mismatch"
echo " [OK] installed $(basename "$MODULE_DST") (sha256 $(sha "$MODULE_DST" | cut -c1-12))"

if [ "$mode" = installed ]; then
  echo " [skip] gpu_worker.py already wired"
else
  python3 "$APPLIER" apply "$WORKER" || fail "p003_apply.py refused gpu_worker.py"
  got="$(sha "$WORKER")"
  [ "$got" = "$WORKER_P003" ] || fail "patched gpu_worker.py sha256 $got != pinned $WORKER_P003"
fi

echo "=== Verification ==="
grep -qF "$MARKER" "$WORKER" || { echo "[FAIL] marker missing after patch" >&2; exit 1; }
python3 -c "import ast,sys; ast.parse(open('$WORKER').read()); ast.parse(open('$MODULE_DST').read())" \
  || { echo "[FAIL] patched files do not parse" >&2; exit 1; }
echo "[OK] patch 003 in place (active only when VLLM_GB10_BLOCKING_SYNC=1; now: ${VLLM_GB10_BLOCKING_SYNC:-unset})"
