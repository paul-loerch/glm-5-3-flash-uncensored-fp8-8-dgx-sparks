#!/usr/bin/env bash
# 012-gb10-oneshot-allreduce.sh — opt-in one-shot all-reduce for small TP messages (VLLM_GB10_ONESHOT_AR=1).
#
# WHY: decode all-reduces (8-64 KiB, ~96 per TP=8 step) are latency-bound: NCCL's tree makes ~6 sequential host-staged
# hops over 100 GbE (no GPUDirect RDMA on GB10). Grouped send/recv to all peers + a local fp32 sum is one network step.
# Measured in CUDA-graph replay (nccl-tests -G, 8 nodes): tree 79 / 98 / 121 / 124 us vs one-shot 52 / 49 / 64 / 86 us at
# 8 / 16 / 32 / 64 KiB. Expected ~4.5 ms per k=2 decode step.
#
# WHAT: installs files/gb10_oneshot_ar.py as vllm/_gb10_oneshot_ar.py and wires two anchored edits (files/p012_apply.py) into
# vllm/distributed/device_communicators/cuda_communicator.py:
#   __init__    after the PyNcclCommunicator block: self._gb10_oneshot = maybe_create(...) (None unless the env var is 1
#               and the group is the TP group; the constructor runs a collective, self-checking warm-up)
#   all_reduce  first statement: bf16 contiguous tensors <= VLLM_GB10_ONESHOT_AR_MAX_BYTES (default 65536) take the one-shot
# Inert without VLLM_GB10_ONESHOT_AR=1 (maybe_create returns None, all_reduce unchanged). Numerics: canonical rank-order fp32
# sum, one bf16 rounding, bit-identical on all ranks — different from NCCL's per-hop bf16 adds, so greedy outputs are not
# bit-equal to earlier runs (nor to a stock v0.32.0 pynccl all-reduce); gate with logprob/KL + quality checks.
# Fail-closed (pinned applier / module / target shas, anchors must match exactly once, AST placement checked, patched files
# must parse), idempotent (the patched file's sha).
#
# vLLM v0.32.0 PORT (base image vllm/vllm-openai:v0.32.0-aarch64). Both anchors are unchanged and unique:
#   __init__    `if is_symmetric_memory_enabled(): register_nccl_symmetric_ops(self.pynccl_comm)` (lines 81-82)
#   all_reduce  `def all_reduce(self, input_):` (line 362)
# v0.32.0 turned CudaCommunicator.all_reduce into a dispatch chain (NCCL symmetric memory, quick reduce, FlashInfer PCIe-IPC,
# FlashInfer all-reduce [default ON: VLLM_ALLREDUCE_USE_FLASHINFER=1], AITER, custom all-reduce, torch symm mem, pynccl).
# The hook is the FIRST statement, so an eligible message (TP group, bf16, contiguous, <= 64 KiB) never reaches any of them.
# Everything else (prefill-sized bf16 messages, other dtypes) does: with VLLM_ALLREDUCE_USE_FLASHINFER=0 (set by the recipe)
# CudaCommunicator.__init__ builds no FlashInferAllReduce (use_flashinfer_allreduce False -> fi_ar_comm None -> use_fi_ar
# False) and with --disable-custom-all-reduce no ca_comm, so the chain ends in pynccl.all_reduce exactly as on the day-0 image;
# with the default (1) a multi-node TP group would get FlashInfer's MNNVL workspace attempt (needs NVSwitch multicast, absent
# on GB10): this script prints a [WARN] line then (the boot is not refused). Changes against the day-0 port: pins, applier
# file, the [WARN] line, and the module's TP-group test (exact 'tp:<n>' like v0.32.0's own gate instead of the substring 'tp').
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
COMM="$VLLM_ROOT/distributed/device_communicators/cuda_communicator.py"
APPLIER="$SRC_DIR/p012_apply.py"
MODULE_SRC="$SRC_DIR/gb10_oneshot_ar.py"
MODULE_DST="$VLLM_ROOT/_gb10_oneshot_ar.py"
MARKER="# [patch-012 gb10-oneshot-ar]"

# pins (sha256): cuda_communicator.py of vllm/vllm-openai:v0.32.0-aarch64 and its patch-012 result; the vendored applier and module
COMM_ORIG=2f5368d6ca1856a7b2b49fc423cc4728483d0c9289f913a7bc7cbe0baa040ec0
COMM_P012=022076750806ea2ac5a83f969bac8b386c0735a56d3b4cfe705d4a3879007215
APPLIER_SHA=d2e9abc2d0b65480d53afb52eb4ca4cd6edd2af53003b288ea111018266d39de
MOD_SHA=d5c5f5d6801d3fa02f518de37d3fa86ac6cfac98a305033b1ebc99b28817db37

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[FAIL] [012] $*" >&2; exit 1; }

echo "=== Patch 012: GB10 one-shot all-reduce for small TP messages (opt-in VLLM_GB10_ONESHOT_AR=1; now: ${VLLM_GB10_ONESHOT_AR:-unset}) ==="
[ -f "$COMM" ] || { echo "[FAIL] $COMM not found" >&2; exit 1; }
[ -f "$MODULE_SRC" ] || { echo "[FAIL] $MODULE_SRC not found" >&2; exit 1; }
[ -f "$APPLIER" ] || { echo "[FAIL] $APPLIER not found" >&2; exit 1; }
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p012_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MODULE_SRC")" = "$MOD_SHA" ] || fail "files/gb10_oneshot_ar.py sha256 $(sha "$MODULE_SRC") != pinned $MOD_SHA"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MODULE_SRC" || fail "$MODULE_SRC does not parse"

now="$(sha "$COMM")"
mode=""
if [ "$now" = "$COMM_P012" ]; then
  mode=installed
elif [ "$now" = "$COMM_ORIG" ]; then
  mode=apply
else
  if grep -qF "$MARKER" "$COMM"; then
    fail "cuda_communicator.py carries a patch-012 edit that is not this revision's pinned result (sha ${now:0:16}...); recreate the container from the image"
  fi
  fail "cuda_communicator.py sha256 $now is neither the v0.32.0 original ($COMM_ORIG) nor patch 012's result ($COMM_P012); refusing (a different vLLM build? re-pin after reading the diff)"
fi

if [ "${VLLM_GB10_ONESHOT_AR:-0}" = 1 ] && [ "${VLLM_ALLREDUCE_USE_FLASHINFER:-1}" != 0 ]; then
  echo " [WARN] VLLM_GB10_ONESHOT_AR=1 but VLLM_ALLREDUCE_USE_FLASHINFER='${VLLM_ALLREDUCE_USE_FLASHINFER:-unset (v0.32.0 default 1)}':" \
       "messages the one-shot does not take (prefill-sized, non-bf16) may be routed to FlashInfer's all-reduce, which a multi-node TP group on GB10 cannot use; set -e VLLM_ALLREDUCE_USE_FLASHINFER=0" >&2
fi

install -m 0644 "$MODULE_SRC" "$MODULE_DST"
[ "$(sha "$MODULE_DST")" = "$MOD_SHA" ] || fail "installed $MODULE_DST sha256 mismatch"
echo " [OK] installed $(basename "$MODULE_DST") (sha256 $(sha "$MODULE_DST" | cut -c1-12))"

if [ "$mode" = installed ]; then
  echo " [skip] cuda_communicator.py already wired"
else
  python3 "$APPLIER" apply "$COMM" || fail "p012_apply.py refused cuda_communicator.py"
  got="$(sha "$COMM")"
  [ "$got" = "$COMM_P012" ] || fail "patched cuda_communicator.py sha256 $got != pinned $COMM_P012"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$COMM")" = 2 ] || { echo "[FAIL] expected 2 markers in $COMM" >&2; exit 1; }
python3 -c "import ast; ast.parse(open('$COMM').read()); ast.parse(open('$MODULE_DST').read())" \
  || { echo "[FAIL] patched files do not parse" >&2; exit 1; }
echo "[OK] patch 012 in place (active only when VLLM_GB10_ONESHOT_AR=1)"
