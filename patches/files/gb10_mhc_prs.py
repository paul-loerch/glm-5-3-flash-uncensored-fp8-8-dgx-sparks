# SPDX-License-Identifier: Apache-2.0
# [patch-022 gb10-mhc-prefill-rowsplit] installed as vllm/_gb10_mhc_prs.py by patches/022-gb10-mhc-prefill-rowsplit.sh
# (mode 'prs' of the mHC prefill split). Written for this recipe.
# REVISION 2 (v0.32.0 port): the hook moved from the image's _tilelang_hc_prenorm_gemm to the TileLang branch of
# _hc_prenorm_gemm_outputs and receives the JIT-class INSTANCE (HcPrenormGemmTileLangKernel); the block_m=2 routing is
# asked of the class's own dispatch(). The row plan, the all-gather, the boot self-test and the numerics are unchanged.
"""GB10, TP=8: the mHC pre-norm GEMM runs on this rank's token rows only during prefill (opt-in, BIT-IDENTICAL).

Used only by the patch-022 edit of vllm/model_executor/kernels/mhc/tilelang.py (_hc_prenorm_gemm_outputs, TileLang
branch - the branch every call takes with VLLM_USE_DEEP_GEMM=0, i.e. whenever is_deep_gemm_supported() is False), and
only when VLLM_GB10_MHC_PRS=1.

WHAT. Every mHC block of GLM-5.3-Flash (layer-0 hc_pre + 89 fused hc_post/hc_pre per forward, M > 16 path) computes
the pre-norm GEMM out[t, :] = x[t, :] @ fn.T and sqrsum[t] = sum_k x[t, k]^2 over x = residual [T, 4 * 4096] bf16 -
REPLICATED on all 8 TP ranks. In prefill it is the slowest mHC kernel (profile, 3,840-token chunk: 90 x 1.88 ms
= 169 ms of 357 ms mHC). For T >= VLLM_GB10_MHC_PRS_MIN_T (default 1,024), outside CUDA-graph capture, rank r now
runs the SAME JIT function (hc_prenorm_gemm_block_m_tilelang, block_m=2, the args the class's launch would pass) on the rows
[r*S, min((r+1)*S, T)) only, S = 2 * ceil(ceil(T/2) / 8) (even), and one NCCL all-gather over the TP group (~S x 100 B
per rank: 24 fp32 mixes + 1 fp32 sqrsum per token, the sqrsum block 256-byte aligned) rebuilds out / sqrsum on every
rank. mhc_post, the all-reduces and big_fuse are untouched. T % 16 != 0: the last non-empty shard is shorter; its
padding is never read.

WHY BIT-IDENTICAL (the v0.32.0 kernel, tilelang_kernels.py:926-1017, sha256 61a384a0...; its arithmetic is the image
kernel's [03aeb3f7...] except `x_val = T.cast(x[...], T.float32)` made explicit; generated CUDA of the image kernel in the
served TileLang 0.1.12 cache, kernel aba0cbe9..., hc_prenorm_gemm_block_m_tilelang_kernel(fn, out, sqrsum, x, int num_tokens)):
  * CTA (bx, by) owns rows 2*bx + {0, 1} and output columns 12*by + {0..11}; for each owned row the thread tid runs
    acc += float(x[row, it*512 + tid]) * fn[col, it*512 + tid] for it = 0..31 in order (fn values independent of the
    row), then tl::warp_reduce_sum, then warp 0 adds the 16 warp partials in order 0..15 starting from 0.0f, and
    writes out[row, col]; sqrsum the same way. Nothing in that arithmetic reads T, the row offset or another row.
  * num_tokens is a runtime int used only in the predicate `2*bx + i_m < num_tokens`; one compiled kernel serves
    every T (T.dynamic), so a shard runs the same machine code as the full call.
  * shard starts are multiples of block_m = 2, so every CTA of a shard is, row for row, a CTA of the full call
    (identical data, identical predicate: interior shard ends are even, the last shard ends at T) - the result
    does not even rely on the two rows of a CTA being computed alike.
  * the shard runs the kernel the full-T dispatch chose (the hook asks the class's dispatch() and only acts when the
    full call routes to block_m=2, then calls that very JIT function), at ANY shard size - so neither patch 011's
    128-row threshold nor the class's small-tile rule (T < 128) can move a shard to another kernel;
  * the all-gather and the unpack are copies.
  Measured: patch 011 found bm2 == per-token kernel bitwise for M = 1..2,048 (same per-row order).
  The boot self-test below re-proves it on the device for every boot.

BOOT (first eligible call = the pre-capture profile run at 8,192 tokens; collective on all TP ranks):
  1. envelope: TP group size 8, PP 1, no sequence-parallel MoE / context parallel (if the vLLM config is visible);
  2. rank fingerprint (gloo all-gather of module sha, gate, MIN_T, patch 011's threshold, TP size): any mismatch
     raises on every rank;
  3. self-test: the live call (real activation, real fn) plus synthetic inputs (device-side integer hash, identical on
     all ranks) at T = MIN_T, MIN_T + 3, 3,840, 3,853 with the live fn and a full-mantissa fp32 fn: split vs the
     unsplit kernel BITWISE on every rank, a repeat bit-identical, every output written, the unsplit reference
     bit-identical across ranks (the mHC input really is replicated), a comparator self-check; per-rank verdicts are
     all-gathered (gloo) and EVERY rank raises unless all passed -> the boot fails closed;
  4. timing of split + gather vs unsplit at the live T (logged), then "[patch-022] ... ACTIVE".
A rank that lacks the gate never joins the collectives: the others block until the gloo / execute timeout (as 018 /
021; up.sh's patch-set check and the identical M_DOCKER_EXTRA make it practically impossible).

CUDA graphs: captures are <= 48 tokens (recipe) and glm5next is eager otherwise; a call while the current stream is
capturing returns False (the image path, no collective). Decode never reaches this code (T <= 48 < MIN_T).
"""
from __future__ import annotations

import functools
import hashlib
import logging
import os
from collections.abc import Callable
from typing import NamedTuple

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001 - CPU tests outside a vLLM install
    logger = logging.getLogger(__name__)

TAG = "[patch-022]"
ENV = "VLLM_GB10_MHC_PRS"
ENV_MIN_T = "VLLM_GB10_MHC_PRS_MIN_T"
ENV_BM2 = "VLLM_GB10_MHC_BM2_MIN_T"  # patch 011's threshold: part of the rank fingerprint
REVISION = 2
DEFAULT_MIN_T = 1024
MIN_T_FLOOR = 256  # below it the gather latency (~65-90 us) eats the saving; values < 1,024 are for sweeps
MIN_T_CEIL = 1 << 20
REQUIRED_TP = 8  # the validated envelope (8 x GB10, TP=8)
BLOCK_M = 2  # hc_prenorm_gemm_block_m_tilelang(..., block_m=2) as launched by HcPrenormGemmTileLangKernel
SELFTEST_T = (3840, 3853)  # the served chunk and an uneven size; MIN_T and MIN_T + 3 are added at boot
CALLS_PER_FORWARD = 90  # GLM-5.3-Flash: layer-0 hc_pre + 89 fused post/pre (log arithmetic only)


def parse_min_t(raw: str | None) -> int:
    """VLLM_GB10_MHC_PRS_MIN_T: unset/empty -> 1,024; else an integer in [256, 2**20], otherwise ValueError."""
    if raw is None or raw.strip() == "":
        return DEFAULT_MIN_T
    s = raw.strip()
    if not s.isdigit():
        raise ValueError(f"{TAG} {ENV_MIN_T}={raw!r} is not a non-negative integer")
    v = int(s)
    if not MIN_T_FLOOR <= v <= MIN_T_CEIL:
        raise ValueError(f"{TAG} {ENV_MIN_T}={v} outside [{MIN_T_FLOOR}, {MIN_T_CEIL}]")
    return v


def _own_sha16() -> str:
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


MODULE_SHA16 = _own_sha16()
if os.environ.get(ENV, "0") == "1":
    parse_min_t(os.environ.get(ENV_MIN_T))  # fail closed at import (tilelang.py imports us only with the gate set)


# ---------------------------------------------------------------------------------------------------------------------
# Row plan
# ---------------------------------------------------------------------------------------------------------------------
class Plan(NamedTuple):
    T: int
    W: int
    S: int  # rows per full shard (a multiple of BLOCK_M); the all-gather moves S rows per rank
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def rows(self, r: int) -> int:
        return self.ends[r] - self.starts[r]


@functools.lru_cache(maxsize=256)
def plan_rows(T: int, W: int, block_m: int = BLOCK_M) -> Plan:
    """Contiguous row shards [r*S, min((r+1)*S, T)) with S = block_m * ceil(ceil(T / block_m) / W).

    Starts are multiples of block_m (the bm2 kernel's CTA row pairs stay the full call's pairs). Shards before the
    tail are full (S rows), the tail shard ends at T, any later shard is empty (needs T <= (W-1)*S; never for
    T >= 256 at W = 8). With T % (block_m * W) == 0 this is exactly rows [r*T/W, (r+1)*T/W)."""
    if T < 1 or W < 1 or block_m < 1:
        raise ValueError(f"{TAG} bad plan request T={T} W={W} block_m={block_m}")
    pairs = -(-T // block_m)
    S = -(-pairs // W) * block_m
    starts = tuple(min(r * S, T) for r in range(W))
    ends = tuple(min((r + 1) * S, T) for r in range(W))
    return Plan(T, W, S, starts, ends)


def check_plan(p: Plan, block_m: int = BLOCK_M) -> None:
    """Raise AssertionError unless p partitions [0, T) as described in plan_rows()."""
    assert p.S % block_m == 0 and p.S > 0, p
    assert p.starts[0] == 0 and p.ends[-1] == p.T, p
    for r in range(p.W):
        n = p.rows(r)
        assert 0 <= n <= p.S, p
        if r + 1 < p.W:
            assert p.ends[r] == p.starts[r + 1], p
        if n:
            assert p.starts[r] == r * p.S and p.starts[r] % block_m == 0, p
            assert p.ends[r] == p.T or n == p.S, p  # only the tail shard is short
    assert sum(p.rows(r) for r in range(p.W)) == p.T, p


# ---------------------------------------------------------------------------------------------------------------------
# Worlds: the vLLM TP group (production) and an in-process emulation of all ranks (tests)
# ---------------------------------------------------------------------------------------------------------------------
class _TPWorld:
    """The vLLM tensor-parallel group: NCCL all-gather (pynccl, current stream) + gloo object all-gather."""

    kind = "tp"

    def __init__(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        self.group = get_tp_group()
        self.rank = int(self.group.rank_in_group)
        self.size = int(self.group.world_size)
        self.pp_size = int(get_pp_group().world_size)

    def gather_shards(self, make_send: Callable[[int], torch.Tensor]) -> torch.Tensor:
        return self.group.all_gather(make_send(self.rank), dim=0)

    def all_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        return self.group.all_gather(t.contiguous().view(-1), dim=0)

    def all_gather_object(self, obj):
        import torch.distributed as dist

        out = [None] * self.size
        dist.all_gather_object(out, obj, group=self.group.cpu_group)
        return out


class EmulatedWorld:
    """Tests only: all `size` TP ranks in this one process. gather_shards() builds every rank's send buffer in rank
    order and concatenates them - byte for byte what NCCL's all_gather hands to every rank. Cross-rank checks are
    trivially satisfied (one process, one input)."""

    kind = "emulated"

    def __init__(self, size: int = REQUIRED_TP, rank: int = 0, pp_size: int = 1) -> None:
        self.size, self.rank, self.pp_size = size, rank, pp_size

    def gather_shards(self, make_send: Callable[[int], torch.Tensor]) -> torch.Tensor:
        return torch.cat([make_send(r) for r in range(self.size)])

    def all_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous().view(-1)
        return torch.cat([t] * self.size)

    def all_gather_object(self, obj):
        return [obj for _ in range(self.size)]


_WORLD_FACTORY: Callable[[], object] | None = None  # tests only


def _make_world():
    return _WORLD_FACTORY() if _WORLD_FACTORY is not None else _TPWorld()


# ---------------------------------------------------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------------------------------------------------
def layout(S: int, n_out: int) -> tuple[int, int]:
    """Per-rank send buffer (fp32 elements): out rows [0, S * n_out), sqrsum at sq_off = S * n_out rounded up to 64
    elements (both kernel output views start 256-byte aligned in a fresh allocation), total width sq_off + S."""
    sq_off = -(-(S * n_out) // 64) * 64
    return sq_off, sq_off + S


def split_into(kernel, x, fn, out, sqrsum, hidden_size, hc_mult, n_thr, tile_n, world, plan: Plan | None = None):
    """out[0] / sqrsum[0] for all T rows of x, each rank running `kernel` on its own row shard + one all-gather.

    `kernel` is called exactly like the image's bm2 call: kernel(x_rows, fn, out_rows, sqrsum_rows, hidden_size,
    hc_mult, fn.shape[0], n_thr, tile_n, 2). x_rows is a row view of x (contiguous rows, stride K, starting at an
    even row; K = 16,384 bf16 -> 64 KiB-aligned), out_rows / sqrsum_rows are contiguous views of one flat fp32 send
    buffer (layout())."""
    T = int(x.shape[0])
    plan = plan if plan is not None else plan_rows(T, world.size)
    if plan.T != T or plan.W != world.size:
        raise RuntimeError(f"{TAG} plan {plan} does not match T={T} W={world.size}")
    S, n_out = plan.S, int(fn.shape[0])
    sq_off, width = layout(S, n_out)
    dev = x.device

    def make_send(r: int) -> torch.Tensor:
        send = torch.empty(width, dtype=torch.float32, device=dev)
        b, e = plan.starts[r], plan.ends[r]
        n = e - b
        if n > 0:
            kernel(x[b:e], fn, send[: n * n_out].view(1, n, n_out), send[sq_off: sq_off + n].view(1, n),
                   hidden_size, hc_mult, n_out, n_thr, tile_n, BLOCK_M)
        return send

    recv = world.gather_shards(make_send)
    if recv.numel() != plan.W * width:
        raise RuntimeError(f"{TAG} all-gather returned {recv.numel()} elements, expected {plan.W * width}")
    unpack(recv.view(plan.W, width), plan, n_out, out, sqrsum)


def unpack(recv: torch.Tensor, plan: Plan, n_out: int, out: torch.Tensor, sqrsum: torch.Tensor) -> None:
    """recv [W, width] (rank-major, layout()) -> out[0, :T] / sqrsum[0, :T]. Rows r*S.. of rank r land at r*S..: the
    q = T // S full shards in one strided copy per tensor, the tail shard (if any) in one more."""
    T, S = plan.T, plan.S
    sq_off, _ = layout(S, n_out)
    q, tail = divmod(T, S)
    o, s = out[0], sqrsum[0]
    if q:
        o[: q * S].view(q, S, n_out).copy_(recv[:q, : S * n_out].view(q, S, n_out))
        s[: q * S].view(q, S).copy_(recv[:q, sq_off: sq_off + S])
    if tail:
        o[q * S: T].copy_(recv[q, : tail * n_out].view(tail, n_out))
        s[q * S: T].copy_(recv[q, sq_off: sq_off + tail])


def ineligible(x, fn, out, sqrsum, hidden_size, hc_mult) -> str | None:
    """None if the split path applies, else why not (depends only on shapes/dtypes/layout: identical on all ranks)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return f"x {tuple(x.shape)} {x.dtype} contiguous={x.is_contiguous()}"
    T, K = int(x.shape[0]), int(x.shape[1])
    if K != hc_mult * hidden_size:
        return f"x width {K} != hc_mult * hidden_size"
    if fn.dim() != 2 or fn.dtype != torch.float32 or not fn.is_contiguous() or int(fn.shape[1]) != K:
        return f"fn {tuple(fn.shape)} {fn.dtype} contiguous={fn.is_contiguous()}"
    n_out = int(fn.shape[0])
    if tuple(out.shape) != (1, T, n_out) or out.dtype != torch.float32 or not out.is_contiguous():
        return f"out {tuple(out.shape)} {out.dtype} contiguous={out.is_contiguous()}"
    if tuple(sqrsum.shape) != (1, T) or sqrsum.dtype != torch.float32 or not sqrsum.is_contiguous():
        return f"sqrsum {tuple(sqrsum.shape)} {sqrsum.dtype} contiguous={sqrsum.is_contiguous()}"
    if not (x.device == fn.device == out.device == sqrsum.device):
        return "tensors on different devices"
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Boot: envelope, rank fingerprint, collective self-test
# ---------------------------------------------------------------------------------------------------------------------
def fingerprint(min_t: int, world_size: int) -> str:
    parts = ["p022", f"rev={REVISION}", f"module={MODULE_SHA16}", f"gate={os.environ.get(ENV, '0')}",
             f"min_t={min_t}", f"bm2_min_t={os.environ.get(ENV_BM2, 'unset')}", f"tp={world_size}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _vllm_config_or_none():
    try:
        from vllm.config import get_current_vllm_config_or_none

        return get_current_vllm_config_or_none()
    except Exception:  # noqa: BLE001
        return None


def check_envelope(world) -> None:
    """Refuse (raise; the same verdict on every rank) outside the validated envelope."""
    if world.size != REQUIRED_TP:
        raise RuntimeError(f"{TAG} {ENV}=1 is validated for TP={REQUIRED_TP} only; this TP group has {world.size} "
                           f"ranks: refusing - unset {ENV}")
    if getattr(world, "pp_size", 1) != 1:
        raise RuntimeError(f"{TAG} {ENV}=1 needs pipeline parallel size 1 (got {world.pp_size}): unset {ENV}")
    cfg = _vllm_config_or_none()
    if cfg is None:
        return
    pc = getattr(cfg, "parallel_config", None)
    if pc is None:
        return
    if bool(getattr(pc, "use_sequence_parallel_moe", False)):
        raise RuntimeError(f"{TAG} {ENV}=1 with sequence-parallel MoE: the mHC rows are already sharded per rank; "
                           f"refusing - unset {ENV}")
    for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        v = getattr(pc, name, 1)
        if v not in (None, 1):
            raise RuntimeError(f"{TAG} {ENV}=1 needs {name} == 1 (got {v}): unset {ENV}")


def check_fingerprint(world, min_t: int) -> str:
    fp = fingerprint(min_t, world.size)
    got = world.all_gather_object(fp)
    if any(g != fp for g in got):
        raise RuntimeError(
            f"{TAG} rank fingerprint mismatch across the TP group: {got} (this rank {world.rank} = {fp}). A rank has a "
            f"different patch-022 module, {ENV} / {ENV_MIN_T} or {ENV_BM2} setting: the mHC pre-norm collectives "
            "would desync. Apply the same recipe on every node.")
    return fp


def _hash_u32(idx: torch.Tensor, seed: int) -> torch.Tensor:
    """int64 idx (0 <= idx < 2**31) -> int64 in [0, 2**32): exact integer arithmetic, identical on every rank."""
    m = 0xFFFFFFFF
    h = (idx * 0x9E3779B1 + (seed * 0x85EBCA6B & m)) & m
    h = h ^ (h >> 16)
    h = (h * 0x045D9F3B) & m
    h = h ^ (h >> 16)
    h = (h * 0x045D9F3B) & m
    return h ^ (h >> 16)


def synthetic_x(T: int, K: int, seed: int, device) -> torch.Tensor:
    """[T, K] bf16 in +-64 with 8 binary exponents, from the hash of the flat index (row chunks of 64)."""
    x = torch.empty(T, K, dtype=torch.bfloat16, device=device)
    for r0 in range(0, T, 64):
        r1 = min(T, r0 + 64)
        h = _hash_u32(torch.arange(r0 * K, r1 * K, dtype=torch.int64, device=device), seed)
        v = (h & 0xFFFF).to(torch.float32) * (1.0 / 4096.0) - 8.0
        e = ((h >> 16) & 7).to(torch.float32) - 4.0
        x[r0:r1].copy_((v * torch.exp2(e)).view(r1 - r0, K))
    return x


def synthetic_fn(n_out: int, K: int, seed: int, device) -> torch.Tensor:
    """[n_out, K] fp32 with full 24-bit mantissas in [-0.5, 0.5) (the checkpoint's fn are bf16-exact: harder test)."""
    h = _hash_u32(torch.arange(n_out * K, dtype=torch.int64, device=device), seed)
    return ((h & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0) - 0.5).view(n_out, K).contiguous()


def mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements whose bits differ (fp32 tensors of one shape)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel(), 1)
    return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum())


def _poison(shape, device) -> torch.Tensor:
    return torch.full(shape, float("nan"), dtype=torch.float32, device=device)


def selftest(kernel, x, fn, out, sqrsum, hidden_size, hc_mult, n_thr, tile_n, world, min_t: int,
             timing: bool = True) -> dict:
    """Collective boot self-test; fills out / sqrsum for the live call through the production split path.

    Every rank runs the same cases in the same order (same collectives). Raises RuntimeError on EVERY rank unless
    every rank passed every case."""
    dev, K, n_out, T_live = x.device, int(x.shape[1]), int(fn.shape[0]), int(x.shape[0])
    args = (hidden_size, hc_mult, n_thr, tile_n)

    def run_case(name, xc, fnc, dst_o=None, dst_s=None):
        T = int(xc.shape[0])
        plan = plan_rows(T, world.size)
        check_plan(plan)
        ref_o, ref_s = _poison((1, T, n_out), dev), _poison((1, T), dev)
        kernel(xc, fnc, ref_o, ref_s, hidden_size, hc_mult, n_out, n_thr, tile_n, BLOCK_M)
        o1 = dst_o if dst_o is not None else _poison((1, T, n_out), dev)
        s1 = dst_s if dst_s is not None else _poison((1, T), dev)
        split_into(kernel, xc, fnc, o1, s1, *args, world, plan)
        o2, s2 = _poison((1, T, n_out), dev), _poison((1, T), dev)
        split_into(kernel, xc, fnc, o2, s2, *args, world, plan)
        g = world.all_gather_tensor(torch.cat([ref_o.view(-1), ref_s.view(-1)])).view(world.size, -1)
        cross = int((g.view(torch.int32) != g[:1].view(torch.int32)).sum())
        probe = o1.clone()
        probe.view(-1).view(torch.int32)[0] ^= 1
        row = {"case": name, "T": T, "S": plan.S,
               "mismatch": mismatches(o1, ref_o) + mismatches(s1, ref_s),
               "repeat_mismatch": mismatches(o2, o1) + mismatches(s2, s1),
               "cross_rank_mismatch": cross,
               "written": bool(torch.isfinite(ref_o).all() and torch.isfinite(ref_s).all()
                               and torch.isfinite(o1).all() and torch.isfinite(s1).all()),
               "comparator_ok": mismatches(probe, o1) == 1}
        row["ok"] = (row["mismatch"] == 0 and row["repeat_mismatch"] == 0 and row["cross_rank_mismatch"] == 0
                     and row["written"] and row["comparator_ok"])
        return row

    rows = [run_case(f"live T={T_live}", x, fn, out, sqrsum)]
    fn_syn = synthetic_fn(n_out, K, 2201, dev)
    for i, T in enumerate(sorted({min_t, min_t + 3, *SELFTEST_T})):
        xs = synthetic_x(T, K, 1001 + i, dev)
        rows.append(run_case(f"synthetic T={T} live fn", xs, fn))
        rows.append(run_case(f"synthetic T={T} fp32 fn", xs, fn_syn))
        del xs
    mine = {"rank": world.rank, "ok": all(r["ok"] for r in rows), "cases": rows}
    allr = world.all_gather_object(mine)
    bad = [r for r in allr if not r["ok"]]
    if bad:
        detail = "; ".join(f"rank {r['rank']}: " + ", ".join(f"{c['case']} {c}" for c in r["cases"] if not c["ok"])
                           for r in bad)
        raise RuntimeError(f"{TAG} {ENV}=1: BOOT SELF-TEST FAILED (row-split pre-norm GEMM is not bit-identical to the "
                           f"unsplit kernel, or the inputs differ across ranks): {detail}; refusing to serve - unset "
                           f"{ENV}")
    report = {"cases": len(rows), "ranks": len(allr), "live_T": T_live, "live_S": plan_rows(T_live, world.size).S}
    if timing and x.is_cuda and getattr(world, "kind", "") == "tp":  # an emulated world runs all 8 shards here
        report.update(_time_live(kernel, x, fn, args, world, n_out))
    return report


def _time_live(kernel, x, fn, args, world, n_out, reps: int = 3) -> dict:
    hidden_size, hc_mult, n_thr, tile_n = args
    T, dev = int(x.shape[0]), x.device
    to, ts = torch.empty(1, T, n_out, dtype=torch.float32, device=dev), torch.empty(1, T, dtype=torch.float32,
                                                                                     device=dev)
    plan = plan_rows(T, world.size)

    def med(f):
        vals = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            f()
            b.record()
            b.synchronize()
            vals.append(a.elapsed_time(b))
        return sorted(vals)[len(vals) // 2]

    torch.cuda.synchronize(dev)
    t_un = med(lambda: kernel(x, fn, to, ts, hidden_size, hc_mult, n_out, n_thr, tile_n, BLOCK_M))
    t_sp = med(lambda: split_into(kernel, x, fn, to, ts, hidden_size, hc_mult, n_thr, tile_n, world, plan))
    return {"unsplit_ms": round(t_un, 4), "split_gather_ms": round(t_sp, 4),
            "est_saving_ms_per_forward": round((t_un - t_sp) * CALLS_PER_FORWARD, 2)}


# ---------------------------------------------------------------------------------------------------------------------
# The hook (called from the patch-022 edit of tilelang.py, TileLang branch of _hc_prenorm_gemm_outputs)
# ---------------------------------------------------------------------------------------------------------------------
class _State:
    def __init__(self) -> None:
        self.min_t: int | None = None
        self.world = None
        self.report: dict | None = None
        self.calls = 0
        self.rows_local = 0
        self.rows_total = 0
        self.capture_skips = 0
        self.small_skips = 0
        self.ineligible = 0
        self.not_bm2 = 0
        self.warned: set[str] = set()


_STATE = _State()


def configured_min_t() -> int:
    if _STATE.min_t is None:
        _STATE.min_t = parse_min_t(os.environ.get(ENV_MIN_T))
    return _STATE.min_t


def _capturing(x: torch.Tensor) -> bool:
    return bool(x.is_cuda and torch.cuda.is_current_stream_capturing())


def route_of(kernel_obj, x, fn, hidden_size, hc_mult):
    """(jit_function, n_thr, tile_n) the UNSPLIT call would run if it routes to the block_m=2 kernel, else None.

    Asks the JIT class's own dispatch() with the arguments _hc_prenorm_gemm_outputs's call uses (n_splits = 1, the default
    n_thr / tile_n), so the threshold (upstream 1,024, patch 011's VLLM_GB10_MHC_BM2_MIN_T) is never duplicated here."""
    T, K, n_out = int(x.shape[0]), int(x.shape[1]), int(fn.shape[0])
    key = kernel_obj.dispatch(num_tokens=T, hc_hidden_size=K, hidden_size=hidden_size, hc_mult=hc_mult, n_out=n_out)
    if not key.use_block_m:
        return None
    if int(key.block_m) != BLOCK_M or int(key.n_splits) != 1:
        raise RuntimeError(f"{TAG} the block_m=2 dispatch changed (block_m {key.block_m}, n_splits {key.n_splits}); "
                           f"refusing - unset {ENV}")
    return kernel_obj.kernel(key), int(key.n_thr), int(key.tile_n)


def prenorm_rowsplit(kernel_obj, x, fn, out, sqrsum, hidden_size, hc_mult) -> bool:
    """True: out / sqrsum hold the bm2 kernel's result for all rows of x, computed row-split across the TP group.
    False: nothing done; the caller runs the upstream unsplit call (T < MIN_T, CUDA-graph capture, unexpected layout,
    or a call the class does not route to block_m=2). `kernel_obj` is HcPrenormGemmTileLangKernel's instance.
    The decision depends only on T, shapes and the capture state: identical on every TP rank."""
    st = _STATE
    T = int(x.shape[0])
    if T < configured_min_t():
        st.small_skips += 1
        return False
    if _capturing(x):
        st.capture_skips += 1
        return False
    why = ineligible(x, fn, out, sqrsum, hidden_size, hc_mult)
    if why is not None:
        st.ineligible += 1
        if why not in st.warned:
            st.warned.add(why)
            logger.warning("%s row split skipped (upstream path kept): %s", TAG, why)
        return False
    route = route_of(kernel_obj, x, fn, hidden_size, hc_mult)
    if route is None:
        st.not_bm2 += 1
        return False
    kernel, n_thr, tile_n = route
    return _rowsplit(kernel, x, fn, out, sqrsum, hidden_size, hc_mult, n_thr, tile_n)


def _rowsplit(kernel, x, fn, out, sqrsum, hidden_size, hc_mult, n_thr, tile_n) -> bool:
    """The split itself (and the boot on the first eligible call): `kernel` is the block_m=2 JIT function."""
    st = _STATE
    T = int(x.shape[0])
    if st.world is None:
        world = _make_world()
        check_envelope(world)
        fp = check_fingerprint(world, st.min_t)
        st.report = selftest(kernel, x, fn, out, sqrsum, hidden_size, hc_mult, n_thr, tile_n, world, st.min_t)
        st.world = world
        st.calls += 1
        st.rows_total += T
        st.rows_local += plan_rows(T, world.size).rows(world.rank)
        r = st.report
        timing = (f"; split+gather {r['split_gather_ms']:.3f} ms vs unsplit {r['unsplit_ms']:.3f} ms per call at "
                  f"T={r['live_T']} ({r['live_S']} rows/rank), ~{r['est_saving_ms_per_forward']:.0f} ms per "
                  f"{CALLS_PER_FORWARD}-call forward" if "unsplit_ms" in r else "")
        logger.info(
            "%s mHC prefill row-split ACTIVE (rank %d/%d, T >= %d, eager prefill only; fingerprint %s): boot self-test "
            "bit-exact on all %d ranks (%d cases: live T=%d + synthetic T=%s, live and full-mantissa fn; repeat "
            "bit-stable; unsplit reference identical across ranks)%s",
            TAG, world.rank, world.size, st.min_t, fp, r["ranks"], r["cases"], r["live_T"],
            "/".join(str(t) for t in sorted({st.min_t, st.min_t + 3, *SELFTEST_T})), timing)
        return True
    split_into(kernel, x, fn, out, sqrsum, hidden_size, hc_mult, n_thr, tile_n, st.world)
    st.calls += 1
    st.rows_total += T
    st.rows_local += plan_rows(T, st.world.size).rows(st.world.rank)
    return True


def stats() -> dict:
    st = _STATE
    return {"active": st.world is not None, "world": getattr(st.world, "kind", None), "min_t": st.min_t,
            "calls": st.calls, "rows_local": st.rows_local, "rows_total": st.rows_total,
            "capture_skips": st.capture_skips, "small_skips": st.small_skips, "ineligible": st.ineligible,
            "not_bm2": st.not_bm2, "report": st.report}


def _set_world_factory_for_test(factory: Callable[[], object] | None) -> None:
    global _WORLD_FACTORY
    _WORLD_FACTORY = factory


def _reset_for_test(min_t: int | None = None) -> None:
    global _STATE
    _STATE = _State()
    _STATE.min_t = min_t
