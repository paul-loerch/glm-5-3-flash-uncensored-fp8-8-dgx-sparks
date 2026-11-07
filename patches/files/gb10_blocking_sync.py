"""GB10 blocking-sync hook (recipe patch 003).

vLLM v0.32.0 port: this module is unchanged (the file below is the day-0 text). Only its call sites moved by a few lines:
patches/003-gb10-blocking-sync.sh wires apply_before_context()/verify_after_context() around
`torch.accelerator.set_device_index(self.device)` in Worker.init_device of v1/worker/gpu_worker.py (v0.32.0 lines 500-501).
The per-step D2H copy named below (flashinfer_mla_sparse_sm90.py:350) is the DAY-0 image's. v0.32.0's SM90 sparse-MLA builder
(upstream #58684) plans from the host-side upper bounds `seq_lens_cpu_upper_bound` under async scheduling and copies
`cam.seq_lens[:num_reqs].cpu()` only when no upper bound exists (flashinfer_mla_sparse_sm90.py:499-501), so the gain of this
patch on v0.32.0 has to be re-measured (decode A/B/A against VLLM_GB10_BLOCKING_SYNC unset). It only changes the CUDA
context's wait policy (sleep instead of spin), never an output.

Makes this worker process's CUDA primary context wait for the GPU by SLEEPING (CU_CTX_SCHED_BLOCKING_SYNC)
instead of spinning. Why: under async scheduling the SM90 sparse-MLA metadata builder does one D2H copy per
step (`cam.seq_lens[:num_reqs].cpu()`, flashinfer_mla_sparse_sm90.py:350) that waits for the previous step's
forward. With CUDA's default schedule (Auto -> spin when contexts <= cores) that wait pegs one P-core per rank
for ~the whole GPU step: the CPU-cluster hotspot behind the thermal ABORTs.

Opt-in: VLLM_GB10_BLOCKING_SYNC=1 (unset/0 = no-op, so recipes that don't set it are unchanged).
Fail-closed: if the flag cannot be set or verified, the worker raises instead of silently spinning.
Called from vllm/v1/worker/gpu_worker.py Worker.init_device (inserted by patches/003-gb10-blocking-sync.sh):
apply_before_context() right before the primary context is created, verify_after_context() right after.
"""
from __future__ import annotations

import ctypes
import logging
import os

ENV = "VLLM_GB10_BLOCKING_SYNC"
CU_CTX_SCHED_MASK = 0x07
CU_CTX_SCHED_BLOCKING_SYNC = 0x04
_NAMES = {0x00: "AUTO", 0x01: "SPIN", 0x02: "YIELD", 0x04: "BLOCKING_SYNC"}

logger = logging.getLogger("vllm.gb10_blocking_sync")
_state: dict[str, int] = {}


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _cuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def _fn(cu: ctypes.CDLL, *names: str):
    for n in names:
        f = getattr(cu, n, None)
        if f is not None:
            return f
    raise RuntimeError(f"[gb10-blocking-sync] libcuda lacks {' / '.join(names)}")


def _ok(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"[gb10-blocking-sync] {what} failed (CUresult {rc})")


def _sched(flags: int) -> str:
    return _NAMES.get(flags & CU_CTX_SCHED_MASK, hex(flags & CU_CTX_SCHED_MASK))


def apply_before_context(ordinal: int) -> None:
    """Set the primary-context scheduling flag for CUDA device `ordinal` (the visible-device index)."""
    if not enabled():
        return
    cu = _cuda()
    _ok(_fn(cu, "cuInit")(ctypes.c_uint(0)), "cuInit")
    dev = ctypes.c_int()
    _ok(_fn(cu, "cuDeviceGet")(ctypes.byref(dev), ctypes.c_int(int(ordinal))), f"cuDeviceGet({ordinal})")
    _ok(_fn(cu, "cuDevicePrimaryCtxSetFlags_v2", "cuDevicePrimaryCtxSetFlags")(
        dev, ctypes.c_uint(CU_CTX_SCHED_BLOCKING_SYNC)), "cuDevicePrimaryCtxSetFlags")
    _state["dev"] = dev.value
    _state["ordinal"] = int(ordinal)


def verify_after_context() -> None:
    """Check the primary context (now created by the caller) really uses BLOCKING_SYNC; raise otherwise."""
    if not enabled():
        return
    if "dev" not in _state:
        raise RuntimeError("[gb10-blocking-sync] verify called without apply (patch wiring broken)")
    cu = _cuda()
    flags, active = ctypes.c_uint(), ctypes.c_int()
    _ok(_fn(cu, "cuDevicePrimaryCtxGetState")(ctypes.c_int(_state["dev"]), ctypes.byref(flags), ctypes.byref(active)),
        "cuDevicePrimaryCtxGetState")
    effective = None
    if active.value:
        cur = ctypes.c_uint()
        if _fn(cu, "cuCtxGetFlags")(ctypes.byref(cur)) == 0:
            effective = cur.value
            if effective & CU_CTX_SCHED_MASK != CU_CTX_SCHED_BLOCKING_SYNC:
                # The context existed before apply() (flags recorded but not applied): set it on the live one.
                set_flags = getattr(cu, "cuCtxSetFlags", None)
                if set_flags is not None and set_flags(ctypes.c_uint(CU_CTX_SCHED_BLOCKING_SYNC)) == 0:
                    _ok(_fn(cu, "cuCtxGetFlags")(ctypes.byref(cur)), "cuCtxGetFlags")
                    effective = cur.value
    sched = (effective if effective is not None else flags.value) & CU_CTX_SCHED_MASK
    if sched != CU_CTX_SCHED_BLOCKING_SYNC:
        raise RuntimeError(
            f"[gb10-blocking-sync] {ENV}=1 but the CUDA context schedule is {_sched(sched)} "
            f"(primary flags {_sched(flags.value)}, active={active.value}, effective="
            f"{_sched(effective) if effective is not None else 'n/a'}); refusing to run spinning")
    logger.info("[gb10-blocking-sync] device %d: CUDA context schedule = BLOCKING_SYNC (primary flags %s, "
                "active=%d, effective %s)", _state["ordinal"], _sched(flags.value), active.value,
                _sched(effective) if effective is not None else "n/a")
