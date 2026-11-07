# SPDX-License-Identifier: Apache-2.0
# [patch-025 gb10-mhc-prefill-sp] installed as vllm/_gb10_prefill_sp.py by patches/025-gb10-mhc-prefill-sp.sh
# (mode 'sp' of the mHC prefill sequence parallelism). Written for this recipe.
# REVISION 4 (v0.32.0 port): re-anchored on vllm/models/glm5next/common/model.py (the layer loop gained the EAGLE-3 aux
# taps, so begin_forward takes aux_layers and an sp region is never opened while they are set; everything else - the
# hooks' contract, the 43-case boot self-test, the gloo preflight, the hang-free schedule guard, the prs (022)
# neutralisation - is rev 3 unchanged). Read-only dependencies re-validated against v0.32.0 (see the .sh).
# REVISION 3 (2026-10-02): the boot self-test is HANG-FREE - every collective of it is issued on every rank in a fixed
# order with a planned, pre-checked shape (a rank-local exception or a bad local shape becomes a failing row, never a
# missing or mismatched collective), the envelope / fingerprint / geometry and the per-rank verdicts are exchanged over
# gloo behind a bounded monitored barrier, and the verdict is a pure function of the exchanged data (identical on every
# rank). The canonical reduce-scatter runs over the TP group's pynccl send/recv (patch 012's path, no new
# communicator) and the boot checks it only when RS=canonical. The live checks now cover the first MoE + MLA layer
# too, at T = MIN_T and MIN_T + 3 (the zero-padded tail shard), plus check 2 (a live suppressed partial,
# reduced by sp vs the decode all-reduce), and fail closed when layers are passed but no live layer is found.
# REVISION 2: live-layer mHC shard-invariance + suppression handles in the boot self-test, MIN_T floor 1,024, the
# envelope read from the groups / forward context, prs inactive asserted; the CPU e2e (files/p025_e2e_check.py).
"""GB10, TP=8: full sequence-parallel (token-split) mHC during prefill (opt-in, NUMERICS CHANGE — not bit-identical
to today's serving; owner OK required to SERVE).

Used only by the patch-025 edits of vllm/models/glm5next/common/model.py (Glm5NextModel.forward,
Glm5NextDecoderLayer.forward), and only when VLLM_GB10_MHC_SP is set to 'sp'/'1' or 'ref'.

WHAT (mode 'sp'). Today every mHC block (layer-0 hc_pre + 89 fused hc_post/hc_pre per forward) and the attention /
MoE all-reduces are REPLICATED on all 8 TP ranks: each rank reduces the per-block TP partial to the full [T, H] and
runs the whole mHC (mhc_post -> pre-norm GEMM -> big_fuse) on all T rows. 'sp' keeps the residual stream
[T, 4, H] sharded: rank r holds rows [r*S, (r+1)*S) (S = 2*ceil(T/16), even, uniform; the tail rows past T are
padding). Per mHC layer, for T >= VLLM_GB10_MHC_SP_MIN_T (default 1,024), eager prefill only, never inside a
CUDA-graph capture:

      hc_fused_post_pre on this rank's S rows            (per-row kernels, no collective)
   -> all-gather x  [S,H] -> [T,H]                       (attention needs the whole sequence)
   -> attention with o_proj all-reduce SUPPRESSED -> partial [T,H]
   -> reduce-scatter partial [T,H] -> [S,H]              (this rank's reduced attention rows)
   -> hc_fused_post_pre on this rank's S rows
   -> all-gather x  [S,H] -> [T,H]
   -> MLP/MoE with the final all-reduce SUPPRESSED -> partial [T,H]
   -> reduce-scatter partial [T,H] -> [S,H]

The embedding (already all-reduced) is sharded with a local slice at model entry; the last layer's hc_post +
contract run on the S rows and the final [S,H] is all-gathered to [T,H] before the norm (the MTP pass and the LM
head then see the full sequence, unchanged). Per forward: 90 reduce-scatters + 91 all-gathers, the SAME bytes as the
91 all-reduces they replace (a ring all-reduce moves 2*(7/8)*S per rank; RS and AG move (7/8)*S each). mhc_post,
big_fuse, the KDA conv/recurrence, the sparse-MLA indexer and the MoE router/top-k are untouched — they run on the
gathered full-T x, exactly as today.

WHAT (mode 'ref'). The same reduction change WITHOUT the sharding: the mHC stays replicated on all T rows, but each
attention / MLP all-reduce becomes reduce-scatter + all-gather (so the reduced value uses the RS element order, then
is broadcast). No speed-up; it isolates the numerics delta (reduction order) from the sharding, for the A/B that
decides whether 'sp' is worth serving.

WHY NOT BIT-IDENTICAL to today. NCCL's ring reduce-scatter does not reproduce the ring all-reduce's
per-element summation order, so the reduced attention / MoE outputs differ bitwise (~52 % of elements in the
microbenchmark) — both are valid sums of the same 8 partials. With VLLM_GB10_MHC_SP_RS=canonical the reduce-scatter
is a grouped pynccl send/recv of row slices to their owner (patch 012's transport, on the TP group's existing NCCL
communicator) + a fixed rank-order fp32 sum (patch 012's one-shot arithmetic), deterministic and one bf16 rounding,
so 'sp' == 'ref' bitwise and the prefill reduction gets decode's (patch-012) semantics.

WHY EVERY RANK STILL COMPUTES IDENTICAL FINAL OUTPUTS. Row t is owned by exactly one rank (t // S). All of row t's
mHC arithmetic runs on that owner (the kernels are per-row: row t reads only row t — mhc_post, the pre-norm GEMM
and big_fuse all accumulate within a row). The attention / MLP inputs are the all-gathered full-T x, bit-identical
on every rank (all-gather is a copy of the same bytes). The reduce-scatter sum for row t is a deterministic function
of the 8 ranks' partials (fixed order for a fixed owner, NCCL; fixed rank order, canonical), computed once by the
owner. The final all-gather before the norm copies each owner's rows to all ranks. So all ranks hold the same
[T, H] going into the norm; divergence is impossible unless a collective or a kernel misbehaves — which the boot
self-test checks on the device and fails closed.

BITWISE SHARD-INVARIANCE of the mHC kernels (patch 022's proof, extended). The pre-norm GEMM
(hc_prenorm_gemm_block_m_tilelang, block_m=2) computes each (row, col) with per-row fp32 accumulators over a fixed k
order; num_tokens is only a row-validity predicate; CTA (bx,by) owns rows 2bx+{0,1}; because S is even every shard
start is even, so each CTA of a shard is a CTA of the full call with the same data and predicate. mhc_post and
big_fuse read only their own row. So the mHC on a shard is bitwise the mHC on the full call sliced -> 'sp' == 'ref'
bitwise. The boot self-test re-proves this on the device (check 3) on the LIVE weights of the first mHC
layer (layer 0: KDA + dense MLP) and of the first MoE + MLA layer (layer 3): hc_pre / hc_fused_post_pre /
hc_post+contract on a zero-padded S-row shard == the full call sliced, bitwise, at T = MIN_T and T = MIN_T + 3 (the
latter gives rank 7 a zero-padded tail shard); the CPU e2e (files/p025_e2e_check.py) proves the collective data flow
AND the whole sp_layer_forward wiring vs the image path.

COMPOSITION WITH PATCH 022 (prs). Both gates may be set (the previous base served prs; sp is added on top). When an 'sp'/'ref' forward
is eligible, the residual stream is already sharded per rank, so the pre-norm GEMM input really is this rank's rows:
prs must NOT split it again. For the duration of an sp region this module raises patch 022's effective MIN_T to
infinity (its public _STATE.min_t) and asserts prs is now inactive for this T, so prs returns the image (unsplit)
path on the shard; the value is restored in a finally. For forwards where sp is inert (decode, T < MIN_T), prs
behaves exactly as without sp. Both default to MIN_T = 1,024, so they never act on the same forward. prs DOES boot
under sp — exactly once, INSIDE this module's boot self-test: the live-weights check calls hc_pre /
hc_fused_post_pre on the full T = MIN_T (and MIN_T + 3) rows as its reference before any sp region exists, those T
are >= prs's MIN_T, so prs runs its own collective bit-identity boot self-test there (all 8 ranks, symmetric) and
row-splits those reference calls; the shard calls of the check run with prs neutralised, exactly as inside an sp
region. Every later eligible forward is an sp region with prs neutralised, so prs never splits while serving; its
"[patch-022] ... ACTIVE" line appears once per rank, logged inside the sp boot. The CPU suite tests the
neutralisation (prs returns False on shards while sp is active) and the e2e runs the real 022 module.

BOOT SELF-TEST (first eligible forward = the pre-capture profile run; collective on all TP ranks; fail closed):
  0. preflight, exchanged over gloo behind a bounded monitored barrier: the envelope verdict of every rank (TP=8,
     PP=1, no context parallel, no SP-MoE, no DBO/ubatch, DeepGEMM off), the rank fingerprint and the self-test
     geometry (plans, live layers, check-2 targets, patch-012 one-shot engagement). Any envelope refusal, fingerprint
     or geometry mismatch raises on EVERY rank before the first NCCL collective;
  1. synthetic reduction cases (T = MIN_T, MIN_T + 3, 3,840, 7,680): the production reduce-scatter reconstructs this
     rank's rows of the all-ranks sum (bitwise for canonical); RS=canonical only: canonical RS == the local
     rank-order fp32 sum (bitwise); sp: a double reduction is detectably different (the probe can fail); the final
     all-gather is bit-identical across ranks; a comparator self-check;
  2. live layers 0 and 3 (fail closed if layers are passed but either is missing): the suppression handles flip the
     live o_proj (KDA, MLA) / down_proj (dense) / MoE skip_final_all_reduce flags and restore them; the mHC
     shard-invariance above; check 2 — a live o_proj / down_proj SUPPRESSED partial at T = 8 (64 KiB at
     H = 4,096, patch 012's one-shot size) reduced by sp's production reduce-scatter + all-gather vs the decode
     all-reduce of the same partial (patch 012's one-shot): bitwise when RS=canonical and the one-shot is engaged,
     within tolerance otherwise; the partials must differ across ranks (the module output was not already reduced);
  3. the per-rank verdicts are exchanged over gloo behind a bounded monitored barrier; every rank raises unless every
     rank passed every case and every cross-rank result is bit-identical.
  HANG-FREE: every collective of steps 1-2 goes through a schedule guard that issues it on every rank in a fixed order
  with its planned shape; a rank-local exception or a wrong local shape becomes a failing row and the planned
  collective is still issued (with a zero input); the full-T reference mHC calls (which carry prs's collectives) are
  always made, with well-formed fallback inputs if the local preparation failed. A rank that dies or wedges before an
  exchange makes every other rank raise after SELFTEST_BARRIER_S (gloo, bounded) instead of blocking.

CUDA graphs. Captures are <= 48 tokens (recipe) and glm5next is eager otherwise; an eligible forward is >= 1,024
tokens and eager. begin_forward returns ineligible while the stream is capturing and asserts the default (non-sp)
flags on every non-sp forward, so a capture can never carry sp state. Decode never reaches sp (T <= 48 < MIN_T).
"""
from __future__ import annotations

import datetime
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

TAG = "[patch-025]"
ENV = "VLLM_GB10_MHC_SP"
ENV_MIN_T = "VLLM_GB10_MHC_SP_MIN_T"
ENV_RS = "VLLM_GB10_MHC_SP_RS"
ENV_PRS = "VLLM_GB10_MHC_PRS"            # patch 022: neutralised inside sp regions
ENV_PRS_MIN_T = "VLLM_GB10_MHC_PRS_MIN_T"
ENV_DEEP_GEMM = "VLLM_USE_DEEP_GEMM"
REVISION = 4
DEFAULT_MIN_T = 1024
MIN_T_FLOOR = 1024      # rev 2: the validated floor. T/8 >= 128 keeps the shard on the bm2 prenorm kernel (patch 011);
MIN_T_CEIL = 1 << 20    #        below 1,024 the shard-invariance proof and the prs-overlap reasoning are not validated.
REQUIRED_TP = 8
BLOCK_M = 2                              # the pre-norm GEMM CTA row pairing; S is a multiple of it (even starts)
MODES = ("sp", "ref")
RS_MODES = ("nccl", "canonical")
SELFTEST_T = (3840, 7680)               # the served chunk and a 2-chunk size; MIN_T and MIN_T+3 added at boot
LIVE_T_EXTRA = 3                        # live mHC checks at T = MIN_T and MIN_T + 3 (rank 7: a zero-padded tail shard)
CHECK2_T = 8                            # check 2 rows: 8 x 4,096 bf16 = 64 KiB = patch 012's one-shot limit
SELFTEST_BARRIER_S = 600.0              # bounded gloo barrier before each boot exchange (tests may lower it)
CALLS_PER_FORWARD = 90


# ---------------------------------------------------------------------------------------------------------------------
# Env parsing (fail closed, at import under the gate)
# ---------------------------------------------------------------------------------------------------------------------
def parse_mode(raw: str | None) -> str | None:
    """VLLM_GB10_MHC_SP: unset/'0'/'' -> None (inert); '1'/'sp' -> 'sp'; 'ref' -> 'ref'; else ValueError."""
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("", "0", "off", "false", "no"):
        return None
    if s in ("1", "sp", "on", "true", "yes"):
        return "sp"
    if s == "ref":
        return "ref"
    raise ValueError(f"{TAG} {ENV}={raw!r} must be unset/0 (inert), 1/sp (full SP) or ref (reduction-only)")


def parse_min_t(raw: str | None) -> int:
    if raw is None or raw.strip() == "":
        return DEFAULT_MIN_T
    s = raw.strip()
    if not s.isdigit():
        raise ValueError(f"{TAG} {ENV_MIN_T}={raw!r} is not a non-negative integer")
    v = int(s)
    if not MIN_T_FLOOR <= v <= MIN_T_CEIL:
        raise ValueError(f"{TAG} {ENV_MIN_T}={v} outside [{MIN_T_FLOOR}, {MIN_T_CEIL}] "
                         f"(the validated floor is {MIN_T_FLOOR}; below it the shard hits the small-T prenorm kernel "
                         f"and the sp/prs overlap is not validated)")
    return v


def parse_rs(raw: str | None) -> str:
    if raw is None or raw.strip() == "":
        return "nccl"
    s = raw.strip().lower()
    if s not in RS_MODES:
        raise ValueError(f"{TAG} {ENV_RS}={raw!r} must be one of {RS_MODES}")
    return s


def _own_sha16() -> str:
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


MODULE_SHA16 = _own_sha16()
if parse_mode(os.environ.get(ENV)) is not None:
    # Fail closed at import (model.py imports us only when the gate is set to a mode).
    parse_min_t(os.environ.get(ENV_MIN_T))
    parse_rs(os.environ.get(ENV_RS))


# ---------------------------------------------------------------------------------------------------------------------
# Row plan: uniform, even starts (every shard start is a multiple of BLOCK_M -> the pre-norm GEMM CTA pairs align;
# a uniform shard size is what NCCL all-gather / reduce-scatter require). Rows [T, W*S) of the last shards are padding.
# ---------------------------------------------------------------------------------------------------------------------
class Plan(NamedTuple):
    T: int
    W: int
    S: int                # rows per rank (even, uniform); padded length is W*S >= T
    pad: int              # W*S - T

    @property
    def padded(self) -> int:
        return self.W * self.S

    def start(self, r: int) -> int:
        return r * self.S

    def real_rows(self, r: int) -> int:
        """Rows of this shard that are < T (the rest is padding)."""
        return max(0, min((r + 1) * self.S, self.T) - r * self.S)


@functools.lru_cache(maxsize=256)
def plan_rows(T: int, W: int, block_m: int = BLOCK_M) -> Plan:
    """S = block_m * ceil(T / (block_m * W)): the smallest multiple of block_m with W*S >= T. Uniform, even starts."""
    if T < 1 or W < 1 or block_m < 1:
        raise ValueError(f"{TAG} bad plan request T={T} W={W} block_m={block_m}")
    unit = block_m * W
    padded = -(-T // unit) * unit
    S = padded // W
    return Plan(T, W, S, padded - T)


def check_plan(p: Plan, block_m: int = BLOCK_M) -> None:
    assert p.S % block_m == 0 and p.S > 0, p
    assert p.W >= 1 and p.padded >= p.T and p.padded - p.T == p.pad, p
    assert p.pad < p.W * block_m, p            # the plan is minimal: no fully-empty trailing shard beyond one unit
    assert sum(p.real_rows(r) for r in range(p.W)) == p.T, p
    for r in range(p.W):
        assert p.start(r) % block_m == 0, p


# ---------------------------------------------------------------------------------------------------------------------
# Row-space collective primitives (dim-0). The production path uses the vLLM TP group (NCCL / canonical pynccl
# send/recv); tests inject an in-process or gloo world with the same interface.
# ---------------------------------------------------------------------------------------------------------------------
def pad_rows(x: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Zero-pad a [*, ...] tensor along dim 0 up to n_rows (contiguous)."""
    if x.shape[0] == n_rows:
        return x.contiguous()
    pad = (0, 0) * (x.dim() - 1) + (0, n_rows - x.shape[0])
    return torch.nn.functional.pad(x.contiguous(), pad)


def shard_entry(x_full: torch.Tensor, plan: Plan, rank: int) -> torch.Tensor:
    """Local slice of the (already reduced) full-T tensor -> this rank's S rows (pad the last shard). No collective."""
    xp = pad_rows(x_full, plan.padded)
    return xp[plan.start(rank): plan.start(rank) + plan.S].contiguous()


def gather_full(x_shard: torch.Tensor, world, plan: Plan) -> torch.Tensor:
    """All-gather this rank's S rows -> the full [T, ...] (identical on every rank)."""
    g = world.all_gather_rows(x_shard.contiguous())
    if g.shape[0] != plan.padded:
        raise RuntimeError(f"{TAG} all-gather returned {g.shape[0]} rows, expected {plan.padded}")
    return g[: plan.T]


def reduce_scatter_partial(partial_full: torch.Tensor, world, plan: Plan, rs_mode: str) -> torch.Tensor:
    """Reduce (sum) the per-rank partial [T, ...] across the TP group and scatter rows -> this rank's [S, ...]."""
    padded = pad_rows(partial_full, plan.padded)
    if rs_mode == "canonical":
        out = world.canonical_reduce_scatter_rows(padded, plan)
    else:
        out = world.reduce_scatter_rows(padded)
    if out.shape[0] != plan.S:
        raise RuntimeError(f"{TAG} reduce-scatter returned {out.shape[0]} rows, expected {plan.S}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------------------------------------------------
class _TPWorld:
    """The vLLM tensor-parallel group: pynccl all-gather / reduce-scatter / all-reduce on the current stream (dim 0),
    the canonical reduce-scatter as a grouped pynccl send/recv (patch 012's transport, on the group's existing NCCL
    communicator) + a fixed rank-order fp32 sum (patch-012 arithmetic), and gloo object all-gather / bounded barrier.
    sp never uses the torch-c10d device group (no new communicator)."""

    kind = "tp"

    def __init__(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        self.group = get_tp_group()
        self.rank = int(self.group.rank_in_group)
        self.size = int(self.group.world_size)
        self.pp_size = int(get_pp_group().world_size)

    def all_gather_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.group.all_gather(x, dim=0)

    def reduce_scatter_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.group.reduce_scatter(x, dim=0)

    def all_reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """The decode all-reduce path (GroupCoordinator.all_reduce: patch 012's one-shot for <= 64 KiB bf16)."""
        return self.group.all_reduce(x)

    def oneshot_engaged(self, x: torch.Tensor) -> bool:
        """True iff the TP group's all-reduce takes patch 012's one-shot for a tensor like x."""
        os_ = getattr(getattr(self.group, "device_communicator", None), "_gb10_oneshot", None)
        return bool(os_ is not None and os_.should_use(x))

    def _pynccl(self):
        comm = getattr(getattr(self.group, "device_communicator", None), "pynccl_comm", None)
        if comm is None or getattr(comm, "disabled", True):
            raise RuntimeError(f"{TAG} {ENV_RS}=canonical needs the TP group's pynccl communicator (patch 012's "
                               "send/recv transport); it is missing or disabled - refusing")
        if int(comm.world_size) != self.size or int(comm.rank) != self.rank:
            raise RuntimeError(f"{TAG} pynccl communicator rank {comm.rank}/{comm.world_size} does not match the TP "
                               f"group rank {self.rank}/{self.size} - refusing")
        return comm

    def canonical_reduce_scatter_rows(self, x: torch.Tensor, plan: Plan) -> torch.Tensor:
        """Grouped pynccl send/recv of row slices to their owner (the patch-012 pattern: send to r+d, receive from
        r-d, d = 1..W-1, one NCCL group), then sum the W contributions to THIS rank's rows in fp32, rank order 0..W-1,
        one bf16 rounding (the one-shot all-reduce arithmetic, patches/files/gb10_oneshot_ar.py)."""
        S, W, r = plan.S, self.size, self.rank
        x = x.contiguous()
        dtype = x.dtype
        tail = x.shape[1:]
        comm = self._pynccl()
        recv = [torch.empty((S, *tail), dtype=dtype, device=x.device) for _ in range(W)]
        recv[r].copy_(x[r * S:(r + 1) * S])
        comm.group_start()
        try:
            for d in range(1, W):
                dst, src = (r + d) % W, (r - d) % W
                comm.send(x[dst * S:(dst + 1) * S], dst)
                comm.recv(recv[src], src)
        finally:
            comm.group_end()
        acc = recv[0].to(torch.float32)
        for p in range(1, W):
            acc = acc + recv[p].to(torch.float32)
        return acc.to(dtype)

    def all_gather_object(self, obj):
        import torch.distributed as dist

        out = [None] * self.size
        dist.all_gather_object(out, obj, group=self.group.cpu_group)
        return out

    def barrier_bounded(self, timeout_s: float) -> None:
        """gloo monitored barrier: returns when every rank arrived; raises on every present rank after timeout_s if
        one did not (instead of blocking in the following all-gather for the 30 min gloo default)."""
        import torch.distributed as dist

        dist.monitored_barrier(group=self.group.cpu_group, timeout=datetime.timedelta(seconds=timeout_s),
                               wait_all_ranks=True)


class EmulatedWorld:
    """Tests only: all `size` ranks live in this one process. all_gather concatenates every rank's buffer in rank
    order (byte for byte what NCCL hands back); the reductions sum across ranks. Cross-rank checks are trivially
    satisfied (one process, one input)."""

    kind = "emulated"

    def __init__(self, size: int = REQUIRED_TP, rank: int = 0, pp_size: int = 1) -> None:
        self.size, self.rank, self.pp_size = size, rank, pp_size

    def all_gather_rows(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.contiguous()] * self.size)

    def reduce_scatter_rows(self, x: torch.Tensor) -> torch.Tensor:
        # One process holds one rank's copy; the "sum across ranks" is the rank-order sequential fp32 sum of W
        # identical slices (so the emulation matches the all-gather-then-sum reference bitwise, as a real ring RS
        # matches it within a few ulps). Scatter this rank's slice.
        return self._seq_sum(x, x.shape[0] // self.size)

    def canonical_reduce_scatter_rows(self, x: torch.Tensor, plan: Plan) -> torch.Tensor:
        return self._seq_sum(x, plan.S)

    def all_reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        acc = x.to(torch.float32).clone()
        for _ in range(1, self.size):
            acc = acc + x.to(torch.float32)
        return acc.to(x.dtype)

    def oneshot_engaged(self, x: torch.Tensor) -> bool:
        return False

    def _seq_sum(self, x: torch.Tensor, S: int) -> torch.Tensor:
        sl = x[self.rank * S:(self.rank + 1) * S].to(torch.float32)
        acc = sl.clone()
        for _ in range(1, self.size):
            acc = acc + sl
        return acc.to(x.dtype)

    def all_gather_object(self, obj):
        return [obj for _ in range(self.size)]

    def barrier_bounded(self, timeout_s: float) -> None:
        return None


_WORLD_FACTORY: Callable[[], object] | None = None  # tests only


def _make_world():
    return _WORLD_FACTORY() if _WORLD_FACTORY is not None else _TPWorld()


# ---------------------------------------------------------------------------------------------------------------------
# Block all-reduce suppression (eager, runtime attributes). The o_proj / MoE / dense-MLP final reduce is turned off
# so the block hands back its per-rank partial; sp reduce-scatters it instead. Fail closed: refuse an unknown layout.
# ---------------------------------------------------------------------------------------------------------------------
class _SuppressReduce:
    """Context manager: set the module's final-all-reduce flag off, restore it afterwards (even on exception)."""

    def __init__(self, attr_owner, attr_name: str, off_value):
        self.owner, self.name, self.off = attr_owner, attr_name, off_value
        self.saved = getattr(attr_owner, attr_name)

    def __enter__(self):
        setattr(self.owner, self.name, self.off)
        return self

    def __exit__(self, *exc):
        setattr(self.owner, self.name, self.saved)
        return False


def _attn_reduce_handle(self_attn):
    o_proj = getattr(self_attn, "o_proj", None)
    if o_proj is None or not hasattr(o_proj, "reduce_results"):
        raise RuntimeError(f"{TAG} self_attn.o_proj has no reduce_results flag (unexpected attention layout)")
    return _SuppressReduce(o_proj, "reduce_results", False)


def _mlp_reduce_handle(mlp, is_moe: bool):
    if is_moe:
        experts = getattr(mlp, "experts", None)
        cfg = getattr(experts, "moe_config", None)
        if cfg is None or not hasattr(cfg, "skip_final_all_reduce"):
            raise RuntimeError(f"{TAG} mlp.experts.moe_config has no skip_final_all_reduce (unexpected MoE layout)")
        return _SuppressReduce(cfg, "skip_final_all_reduce", True)
    down = getattr(mlp, "down_proj", None)
    if down is None or not hasattr(down, "reduce_results"):
        raise RuntimeError(f"{TAG} mlp.down_proj has no reduce_results flag (unexpected dense-MLP layout)")
    return _SuppressReduce(down, "reduce_results", False)


# ---------------------------------------------------------------------------------------------------------------------
# Per-forward state
# ---------------------------------------------------------------------------------------------------------------------
class _Handle(NamedTuple):
    mode: str             # 'sp' or 'ref'
    rs_mode: str
    plan: Plan
    world: object


class _State:
    def __init__(self) -> None:
        self.mode: str | None = None
        self.min_t: int | None = None
        self.rs_mode: str | None = None
        self.world = None
        self.report: dict | None = None
        self.active_handle: _Handle | None = None   # set only inside a forward (begin..end)
        self.prs_saved_min_t = "__unset__"
        self.forwards = 0
        self.layers = 0
        self.capture_skips = 0
        self.small_skips = 0
        self.ubatch_skips = 0
        self.aux_skips = 0
        self.booted = False
        self.warned: set[str] = set()


_STATE = _State()


def configured() -> tuple[str | None, int, str]:
    st = _STATE
    if st.mode is None and st.min_t is None:
        st.mode = parse_mode(os.environ.get(ENV))
        st.min_t = parse_min_t(os.environ.get(ENV_MIN_T))
        st.rs_mode = parse_rs(os.environ.get(ENV_RS))
    return st.mode, st.min_t or DEFAULT_MIN_T, st.rs_mode or "nccl"


def sp_active() -> bool:
    """True while a decoder layer is executing inside an eligible sp/ref forward (set by begin_forward)."""
    return _STATE.active_handle is not None


def active_handle() -> _Handle | None:
    return _STATE.active_handle


# ---------------------------------------------------------------------------------------------------------------------
# prs (patch 022) neutralisation for the sp region
# ---------------------------------------------------------------------------------------------------------------------
def _neutralise_prs(T: int) -> None:
    """Raise patch 022's effective MIN_T to infinity so its hook returns the image (unsplit) path on our shards, and
    assert prs is now inactive for this T (fail closed: if prs would still split, the residual would be double-split)."""
    if os.environ.get(ENV_PRS, "0") != "1":
        return
    try:
        import vllm._gb10_mhc_prs as prs
    except Exception:  # noqa: BLE001 - prs not installed; nothing to neutralise
        return
    _STATE.prs_saved_min_t = getattr(prs._STATE, "min_t", "__unset__")
    prs._STATE.min_t = MIN_T_CEIL << 20
    # prs.prenorm_rowsplit() returns the image path when T < prs._STATE.min_t; verify that holds for this T and for
    # every shard size we will feed it (S <= T). A violation means prs would re-split the already-sharded residual.
    eff = getattr(prs._STATE, "min_t", 0)
    if not (isinstance(eff, int) and eff > T):
        raise RuntimeError(f"{TAG} prs neutralisation failed: patch 022 effective MIN_T {eff!r} does not exceed "
                           f"T={T}; refusing to shard (prs would re-split the residual). Unset {ENV} or {ENV_PRS}.")


def _restore_prs() -> None:
    if _STATE.prs_saved_min_t == "__unset__":
        return
    try:
        import vllm._gb10_mhc_prs as prs

        prs._STATE.min_t = None if _STATE.prs_saved_min_t == "__unset__" else _STATE.prs_saved_min_t
    except Exception:  # noqa: BLE001
        pass
    finally:
        _STATE.prs_saved_min_t = "__unset__"


class _PrsNeutral:
    """Boot self-test only: prs inactive for the enclosed (shard) calls, exactly as inside an sp region; restored on
    exit. Independent of the region's _STATE.prs_saved_min_t bookkeeping."""

    def __enter__(self):
        self.prs, self.saved = None, None
        if os.environ.get(ENV_PRS, "0") != "1":
            return self
        try:
            import vllm._gb10_mhc_prs as prs
        except Exception:  # noqa: BLE001 - prs not installed
            return self
        self.prs, self.saved = prs, getattr(prs._STATE, "min_t", None)
        setattr(prs._STATE, "min_t", MIN_T_CEIL << 20)
        return self

    def __exit__(self, *exc):
        if self.prs is not None:
            self.prs._STATE.min_t = self.saved
        return False


def _capturing(x: torch.Tensor) -> bool:
    return bool(x.is_cuda and torch.cuda.is_current_stream_capturing())


# ---------------------------------------------------------------------------------------------------------------------
# Boot: envelope, rank fingerprint, geometry (preflight exchange), collective self-test
# ---------------------------------------------------------------------------------------------------------------------
def fingerprint(mode: str, min_t: int, rs_mode: str, world_size: int) -> str:
    parts = ["p025", f"rev={REVISION}", f"module={MODULE_SHA16}", f"mode={mode}", f"min_t={min_t}",
             f"rs={rs_mode}", f"prs={os.environ.get(ENV_PRS, '0')}", f"tp={world_size}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _vllm_config_or_none():
    try:
        from vllm.config import get_current_vllm_config_or_none

        return get_current_vllm_config_or_none()
    except Exception:  # noqa: BLE001
        return None


def _context_parallel_sizes() -> tuple[int, int]:
    """(prefill_cp, decode_cp) read from the live groups (always initialised on a real boot). Returns (1, 1) when the
    groups are unavailable (CPU tests): the config-path check below still covers an injected config."""
    pcp = dcp = 1
    try:
        from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group

        pcp = int(get_pcp_group().world_size)
        dcp = int(get_dcp_group().world_size)
    except Exception:  # noqa: BLE001
        pass
    return pcp, dcp


def _ubatching_active() -> bool:
    """True when this forward is micro-batched (DBO / ubatch): the residual stream is already split another way.
    Read from the forward context (the only per-forward signal; the vLLM config is None during the forward)."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return False
        return get_forward_context().ubatch_slices is not None
    except Exception:  # noqa: BLE001
        return False


def _deep_gemm_on() -> bool:
    """True when the prenorm GEMM would take the DeepGEMM (tf32, split-k) path instead of the bm2 kernel sp's
    shard-invariance proof covers. False on GB10 (sm_121 is not a DeepGEMM arch) and when VLLM_USE_DEEP_GEMM=0."""
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        return bool(is_deep_gemm_supported())
    except Exception:  # noqa: BLE001
        # No vLLM deep_gemm module (CPU tests): fall back to the env the recipe pins to 0.
        return os.environ.get(ENV_DEEP_GEMM, "0").strip().lower() in ("1", "on", "true", "yes")


def check_envelope(world, is_sequence_parallel: bool = False) -> None:
    if world.size != REQUIRED_TP:
        raise RuntimeError(f"{TAG} {ENV} is validated for TP={REQUIRED_TP} only; this group has {world.size} ranks: "
                           f"refusing - unset {ENV}")
    if getattr(world, "pp_size", 1) != 1:
        raise RuntimeError(f"{TAG} {ENV} needs pipeline parallel size 1 (got {world.pp_size}): unset {ENV}")
    if is_sequence_parallel:
        raise RuntimeError(f"{TAG} {ENV} with the image's sequence-parallel-MoE path (use_sequence_parallel_moe): "
                           f"refusing - the two shard the residual stream differently. Unset one.")
    if _deep_gemm_on():
        raise RuntimeError(f"{TAG} {ENV} needs {ENV_DEEP_GEMM}=0: the DeepGEMM prenorm (tf32, split-k) is not the "
                           f"per-row bm2 kernel sp's shard-invariance is proven against. Unset {ENV} or set "
                           f"{ENV_DEEP_GEMM}=0.")
    pcp, dcp = _context_parallel_sizes()
    if pcp != 1:
        raise RuntimeError(f"{TAG} {ENV} needs prefill_context_parallel_size == 1 (got {pcp}): unset {ENV}")
    if dcp != 1:
        raise RuntimeError(f"{TAG} {ENV} needs decode_context_parallel_size == 1 (got {dcp}): unset {ENV}")
    if _ubatching_active():
        raise RuntimeError(f"{TAG} {ENV} with DBO / micro-batching (the forward is ubatched): refusing - the residual "
                           f"stream is already split per micro-batch. Unset {ENV} or disable DBO/ubatching.")
    cfg = _vllm_config_or_none()
    if cfg is None:
        return
    # Defensive: if a future vLLM makes the config visible during the forward, re-check it here too.
    pc = getattr(cfg, "parallel_config", None)
    if pc is None:
        return
    if bool(getattr(pc, "use_sequence_parallel_moe", False)):
        raise RuntimeError(f"{TAG} {ENV} with use_sequence_parallel_moe (the image's EP-SP path): refusing - the two "
                           f"shard the residual stream differently. Unset one.")
    if bool(getattr(pc, "enable_dbo", False)) or int(getattr(pc, "ubatch_size", 0) or 0) > 1:
        raise RuntimeError(f"{TAG} {ENV} with DBO / micro-batching (parallel_config): refusing - unset {ENV}.")
    for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        v = getattr(pc, name, 1)
        if v not in (None, 1):
            raise RuntimeError(f"{TAG} {ENV} needs {name} == 1 (got {v}): unset {ENV}")


def _hash_u32(idx: torch.Tensor, seed: int) -> torch.Tensor:
    m = 0xFFFFFFFF
    h = (idx * 0x9E3779B1 + (seed * 0x85EBCA6B & m)) & m
    h = h ^ (h >> 16)
    h = (h * 0x045D9F3B) & m
    h = h ^ (h >> 16)
    h = (h * 0x045D9F3B) & m
    return h ^ (h >> 16)


def synthetic_rows(T: int, width: int, seed: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """[T, width] with values in +-64 over 8 binary exponents, from the hash of the flat index (identical on all
    ranks). Row t depends only on t, so slicing a shard gives that shard's rows exactly."""
    out = torch.empty(T, width, dtype=dtype, device=device)
    for r0 in range(0, T, 64):
        r1 = min(T, r0 + 64)
        h = _hash_u32(torch.arange(r0 * width, r1 * width, dtype=torch.int64, device=device), seed)
        v = (h & 0xFFFF).to(torch.float32) * (1.0 / 4096.0) - 8.0
        e = ((h >> 16) & 7).to(torch.float32) - 4.0
        out[r0:r1].copy_((v * torch.exp2(e)).view(r1 - r0, width).to(dtype))
    return out


def mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel(), 1)
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return int((a.contiguous().view(iv) != b.contiguous().view(iv)).sum())


def _amax_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    denom = b32.abs().amax().clamp_min(1e-12)
    return float((a32 - b32).abs().amax() / denom)


# The sp attention/MLP reduction replaces one ring all-reduce by reduce-scatter (+ an implicit second rounding in the
# NCCL case). Over 90 blocks the residual stream drifts from the all-reduce reference by at most a few bf16 ulps per
# block; the boot tolerance below is generous enough to pass a correct reduction and tight enough to catch a wrong
# one (a missing term, a double reduction or an unsuppressed all-reduce fail by >> this). Rank-consistency is checked
# bitwise (exact), which is the invariant that actually matters.
SELFTEST_RTOL = 2.0e-2


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.to(torch.float32).contiguous().cpu().numpy().tobytes()).hexdigest()


def _dev(device) -> torch.device:
    """Normalise ('cuda' -> 'cuda:N') so the self-test's device checks compare like with like."""
    return torch.empty(0, device=device).device


def _case_ts(min_t: int) -> list[int]:
    return sorted({min_t, min_t + 3, *SELFTEST_T})


def _live_ts(min_t: int) -> tuple[int, int]:
    return (min_t, min_t + LIVE_T_EXTRA)


def _live_layers(layers):
    """(the first mHC non-MTP layer, the first mHC non-MTP layer with MLA attention AND a MoE MLP) with the hc_*
    weights loaded - the boot self-test's live-weights sources (GLM-5.3-Flash: layer 0 = KDA + dense, layer 3 =
    MLA + MoE). Either may be None (the caller fails closed)."""
    first = mla_moe = None
    for ly in layers or ():
        if not (getattr(ly, "mhc", False) and not getattr(ly, "is_mtp_layer", False)
                and getattr(ly, "n", None) and hasattr(ly, "hc_pre")):
            continue
        if first is None:
            first = ly
        if mla_moe is None and getattr(ly, "layer_kind", None) == "mla" and bool(getattr(ly, "_mlp_is_moe", False)):
            mla_moe = ly
        if first is not None and mla_moe is not None:
            break
    return first, mla_moe


def _check2_targets(layer) -> list[list]:
    """check 2 sources on a live layer: its attention o_proj and, for a dense MLP, its down_proj (both
    RowParallelLinear). The MoE's suppressed partial needs the forward context (the fused-MoE custom op), so the MoE
    is covered by the flag check + the CPU e2e."""
    out = [["o_proj", int(getattr(getattr(getattr(layer, "self_attn", None), "o_proj", None),
                                  "input_size_per_partition", 0) or 0)]]
    if not bool(getattr(layer, "_mlp_is_moe", False)):
        out.append(["down_proj", int(getattr(getattr(getattr(layer, "mlp", None), "down_proj", None),
                                             "input_size_per_partition", 0) or 0)])
    return out


def _check2_geometry(world, hidden: int, device) -> tuple[int, bool]:
    """Rows for check 2 and whether the decode all-reduce takes patch 012's one-shot at that size (the largest
    T <= CHECK2_T it engages for; CHECK2_T when it is not active)."""
    if not hasattr(world, "oneshot_engaged"):
        return CHECK2_T, False
    for t in range(CHECK2_T, 0, -1):
        if world.oneshot_engaged(torch.empty(t, hidden, dtype=torch.bfloat16, device=device)):
            return t, True
    return CHECK2_T, False


def _geometry(world, mode, min_t, rs_mode, hidden, device, layers) -> tuple[dict, list]:
    """The self-test's collective geometry (identical on every rank or the preflight refuses) + the live layer
    objects. Raises (-> reported in the preflight, every rank refuses) when layers are passed but a live layer is
    missing: fail closed, the live-weights checks cannot run."""
    W = world.size
    geo = {"W": W, "H": int(hidden), "mode": mode, "rs": rs_mode, "min_t": int(min_t),
           "prs": os.environ.get(ENV_PRS, "0"), "cases": [list(plan_rows(T, W)) for T in _case_ts(min_t)],
           "layers_passed": layers is not None, "live": [], "live_t": [], "check2": None}
    live: list = []
    if layers is not None:
        first, mla_moe = _live_layers(layers)
        if first is None or mla_moe is None:
            raise RuntimeError(
                f"{TAG} boot self-test: model layers were passed but no live "
                f"{'mHC' if first is None else 'MLA + MoE mHC'} layer was found (fail closed: the live-weights checks "
                f"- mHC shard-invariance, suppression handles, check 2 - cannot run on this layer layout)")
        live = [first] if first is mla_moe else [first, mla_moe]
        geo["live"] = [[int(getattr(ly, "layer_idx", -1)), str(getattr(ly, "layer_kind", "?")),
                        bool(getattr(ly, "_mlp_is_moe", False)), int(ly.n), _check2_targets(ly)] for ly in live]
        geo["live_t"] = [list(plan_rows(T, W)) for T in _live_ts(min_t)]
        t2, engaged = _check2_geometry(world, hidden, device)
        geo["check2"] = {"plan": list(plan_rows(t2, W)), "oneshot": engaged}
    return geo, live


def _exchange(world, obj, what: str):
    """gloo all-gather of a per-rank object behind a bounded monitored barrier."""
    try:
        world.barrier_bounded(SELFTEST_BARRIER_S)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{TAG} {ENV}: boot {what}: not every TP rank reached the exchange within "
                           f"{SELFTEST_BARRIER_S:.0f} s ({type(e).__name__}: {str(e)[:300]}); refusing - a rank failed "
                           "or is stuck before it (see that rank's log)") from e
    return world.all_gather_object(obj)


def _preflight(world, mode, min_t, rs_mode, hidden, device, layers, is_sequence_parallel) -> tuple[str, dict, list]:
    """Envelope + fingerprint + geometry of EVERY rank, exchanged before the first NCCL collective; every rank takes
    the same decision from the same exchanged data (a refusal on one rank raises on all, never a hang)."""
    fp = fingerprint(mode, min_t, rs_mode, world.size)
    env_err = geo = geo_err = None
    live: list = []
    try:
        check_envelope(world, is_sequence_parallel=is_sequence_parallel)
    except Exception as e:  # noqa: BLE001 - reported to every rank below
        env_err = str(e) or type(e).__name__
    try:
        geo, live = _geometry(world, mode, min_t, rs_mode, hidden, device, layers)
    except Exception as e:  # noqa: BLE001 - reported to every rank below
        geo_err = f"{type(e).__name__}: {e}"
    allp = _exchange(world, {"rank": world.rank, "fp": fp, "env_err": env_err, "geo": geo, "geo_err": geo_err},
                     "preflight")
    errs = [(p["rank"], p["env_err"]) for p in allp if p["env_err"]]
    if errs:
        raise RuntimeError(f"{TAG} boot refused on rank(s) {[r for r, _ in errs]}: {errs[0][1]}"
                           + ("" if len({e for _, e in errs}) == 1 else f" (and: {sorted({e for _, e in errs[1:]})})"))
    fps = [p["fp"] for p in allp]
    if len(set(fps)) != 1:
        raise RuntimeError(
            f"{TAG} rank fingerprint mismatch across the TP group: {fps}. A rank has a different patch-025 module, "
            f"{ENV} / {ENV_MIN_T} / {ENV_RS} or {ENV_PRS} setting: the sp collectives would desync. Apply the same "
            "recipe on every node.")
    gerrs = [(p["rank"], p["geo_err"]) for p in allp if p["geo_err"]]
    if gerrs:
        raise RuntimeError(f"{TAG} boot self-test refused on rank(s) {[r for r, _ in gerrs]}: {gerrs[0][1]}")
    if any(p["geo"] != allp[0]["geo"] for p in allp):
        diff = sorted({k for p in allp for k in (p["geo"] or {}) if p["geo"].get(k) != allp[0]["geo"].get(k)})
        raise RuntimeError(f"{TAG} boot self-test geometry differs across the TP group ({diff}): the self-test "
                           "collectives would not match - refusing")
    return fp, geo, live


class _LocalFail(RuntimeError):
    """A rank-local self-test failure: recorded as a failing row, never allowed to skip a collective."""


def _valid_or(t, shape, dtype, device) -> torch.Tensor:
    """t if it is a contiguous tensor of exactly this shape / dtype / device, else zeros of that shape. Keeps an opaque
    (possibly collective-bearing) call well-formed on every rank; the case still fails through its own rows."""
    if (isinstance(t, torch.Tensor) and tuple(t.shape) == tuple(shape) and t.dtype == dtype and t.device == device
            and t.is_contiguous()):
        return t
    return torch.zeros(tuple(shape), dtype=dtype, device=device)


class _Sched:
    """The boot self-test's collective schedule guard. Every collective of the self-test goes through _issue(): it is
    issued on every rank in the same order with the planned shape, whatever the local code did. A step declares its
    planned collectives [(op, rows, plan)]; a call that does not match the plan (another op, an extra call, or an input
    of another shape / dtype / device) raises locally and the PLANNED collective is issued with a zero input; the
    collectives a step did not reach (a local exception) are issued with zero inputs when the step ends. So a
    rank-local failure becomes a failing row on that rank and garbage (-> failing rows) for its peers, never a
    missing or mismatched collective. The world interface matches what the production primitives call."""

    kind = "selftest-guard"

    def __init__(self, world, hidden: int, device) -> None:
        self.w = world
        self.H = int(hidden)
        self.device = device
        self.ops: list = []
        self.i = 0
        self.issued = 0

    @property
    def rank(self) -> int:
        return self.w.rank

    @property
    def size(self) -> int:
        return self.w.size

    def all_gather_rows(self, x):
        return self._issue("ag", x)

    def reduce_scatter_rows(self, x):
        return self._issue("rs", x)

    def canonical_reduce_scatter_rows(self, x, plan):
        return self._issue("crs", x, plan)

    def all_reduce_rows(self, x):
        return self._issue("ar", x)

    def _problem(self, x, rows) -> str | None:
        if not isinstance(x, torch.Tensor):
            return f"input is {type(x).__name__}, not a tensor"
        if tuple(x.shape) != (rows, self.H) or x.dtype != torch.bfloat16 or x.device != self.device:
            return f"input {tuple(x.shape)} {x.dtype} {x.device} != planned ({rows}, {self.H}) bf16 {self.device}"
        if not x.is_contiguous():
            return "input not contiguous"
        return None

    def _zeros(self, rows: int) -> torch.Tensor:
        return torch.zeros(rows, self.H, dtype=torch.bfloat16, device=self.device)

    def _real(self, op, x, plan):
        self.issued += 1
        if op == "ag":
            return self.w.all_gather_rows(x)
        if op == "rs":
            return self.w.reduce_scatter_rows(x)
        if op == "crs":
            return self.w.canonical_reduce_scatter_rows(x, plan)
        if op == "ar":
            return self.w.all_reduce_rows(x)
        raise AssertionError(op)

    def _issue(self, op, x, plan=None):
        if self.i >= len(self.ops):
            raise _LocalFail(f"unplanned extra collective '{op}' (not issued)")
        pop, rows, pplan = self.ops[self.i]
        if op != pop:
            raise _LocalFail(f"collective '{op}' where '{pop}' was planned (the planned one is issued)")
        self.i += 1
        bad = self._problem(x, rows)
        if bad is None and op == "crs" and (plan is None or plan.S != pplan.S or plan.W != pplan.W):
            bad = f"canonical plan {plan} != planned {pplan}"
        if bad is not None:
            self._real(pop, self._zeros(rows), pplan)      # keep the group in step
            raise _LocalFail(f"'{op}' {bad} (a zero input of the planned shape was issued instead)")
        return self._real(op, x, pplan)

    def run(self, rows_out: list, name: str, ops, fn):
        """Run one step: fn() with `ops` as its planned collectives. NEVER raises; returns fn()'s result, or None
        after recording a failing row."""
        self.ops, self.i = list(ops), 0
        res, err = None, None
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001 - a rank-local failure becomes a failing row
            err = f"{type(e).__name__}: {e}"
        try:
            if err is None and self.i < len(self.ops):
                err = f"planned collective(s) {[o[:2] for o in self.ops[self.i:]]} not issued by the step"
            while self.i < len(self.ops):
                op, nrows, pplan = self.ops[self.i]
                self.i += 1
                self._real(op, self._zeros(nrows), pplan)
        except Exception as e:  # noqa: BLE001 - the transport itself failed locally; nothing more can be done here
            err = (err + "; " if err else "") + f"drain failed: {type(e).__name__}: {e}"
        self.ops, self.i = [], 0
        if err is not None:
            rows_out.append({"case": name, "ok": False, "err": err[:400]})
            return None
        return res

    def opaque(self, rows_out: list, name: str, make_args, fallback, call):
        """An opaque, possibly collective-bearing call (a full-T mHC op: patch 022's row split + all-gather when prs
        is on). The call is made on EVERY rank: if preparing its arguments fails locally, the well-formed fallback
        arguments are used (and the step fails); only the call itself can still fail (recorded)."""
        def fn():
            prep_err = None
            try:
                args = make_args()
            except Exception as e:  # noqa: BLE001
                prep_err = f"argument preparation: {type(e).__name__}: {e}"
                args = fallback()
            out = call(args)
            if prep_err is not None:
                raise _LocalFail(prep_err + " (the call was still made with fallback arguments)")
            return out
        return self.run(rows_out, name, (), fn)


def _suppression_cases(layer, is_moe: bool) -> list[dict]:
    """The real suppression handles flip the live module's final-all-reduce flag off inside the context and restore
    it after. Catches a handle that targets the wrong attribute or does not suppress (the on-device analog of the
    no_*_suppress wiring mutants, which the CPU e2e drives end to end)."""
    rows: list[dict] = []
    tag = f"L{getattr(layer, 'layer_idx', '?')}"

    def one(name, make_handle, read):
        before = read()
        inside = None
        try:
            with make_handle():
                inside = read()
        except Exception as e:  # noqa: BLE001
            rows.append({"case": f"suppress-{tag}-{name}", "ok": False, "err": repr(e)[:120]})
            return
        after = read()
        off = (inside is False) if name != "moe" else (inside is True)
        rows.append({"case": f"suppress-{tag}-{name}", "ok": bool(off and after == before),
                     "before": before, "inside": inside, "after": after})

    one("o_proj", lambda: _attn_reduce_handle(layer.self_attn),
        lambda: layer.self_attn.o_proj.reduce_results)
    if is_moe:
        one("moe", lambda: _mlp_reduce_handle(layer.mlp, True),
            lambda: layer.mlp.experts.moe_config.skip_final_all_reduce)
    else:
        one("down_proj", lambda: _mlp_reduce_handle(layer.mlp, False),
            lambda: layer.mlp.down_proj.reduce_results)
    return rows


class _SelfTest:
    """The collective boot self-test. Straight-line: every method issues a FIXED sequence of guarded
    steps (no data-dependent branching; only the agreed geometry / mode / RS select steps), so all ranks issue the
    same collectives in the same order."""

    def __init__(self, world, mode, min_t, rs_mode, hidden, device, geo, live) -> None:
        self.w, self.W, self.r = world, world.size, world.rank
        self.mode, self.min_t, self.rs = mode, min_t, rs_mode
        self.H, self.dev, self.geo, self.live = int(hidden), device, geo, live
        self.rs_op = "crs" if rs_mode == "canonical" else "rs"
        self.sched = _Sched(world, self.H, device)
        self.rows: list[dict] = []
        self.shas: dict[str, str] = {}      # must be bit-identical across ranks
        self.psh: dict[str, str] = {}       # suppressed partials: must differ across ranks

    def step(self, name, ops, fn):
        return self.sched.run(self.rows, name, ops, fn)

    # (1) synthetic reduction cases ------------------------------------------------------------------------------
    def reduction_case(self, T: int) -> None:
        W, r, H, dev = self.W, self.r, self.H, self.dev
        plan = plan_rows(T, W)
        name = f"T={T}"
        c: dict = {}

        def prep():
            check_plan(plan)
            # A rank-dependent partial so the cross-rank sum is non-trivial (in a real / gloo world the ranks differ;
            # in the single-process EmulatedWorld every rank is this process, so the "sum of W copies" matches too).
            part = synthetic_rows(T, H, 9001 + T, dev).to(torch.float32) * (1.0 + 0.1 * r)
            c["partial"] = part.to(torch.bfloat16)
            c["padded"] = pad_rows(c["partial"], plan.padded)

        def reference():
            # this rank's rows of the sum of ALL ranks' partials, rank order, fp32 -> bf16
            allp = self.sched.all_gather_rows(c["padded"]).view(W, plan.padded, H)
            lo, hi = plan.start(r), plan.start(r) + plan.S
            acc = allp[0, lo:hi].to(torch.float32)
            for p in range(1, W):
                acc = acc + allp[p, lo:hi].to(torch.float32)
            c["ref"] = acc.to(torch.bfloat16)

        def canonical():
            # RS=canonical only: canonical == the local rank-order fp32 sum (bitwise)
            c["can_exact"] = mismatches(self.sched.canonical_reduce_scatter_rows(c["padded"], plan), c["ref"])

        def production():
            c["got"] = reduce_scatter_partial(c["partial"], self.sched, plan, self.rs)
            c["rel"] = _amax_rel(c["got"], c["ref"])

        def sensitivity():
            # an unsuppressed all-reduce would feed the RS an already-summed partial -> a double reduction; it must be
            # detectably different from the single reduction (so the reconstruction check can fail)
            dbl = reduce_scatter_partial(gather_full(c["got"], self.sched, plan).to(torch.bfloat16), self.sched,
                                         plan, self.rs)
            c["sens_ok"] = _amax_rel(dbl, c["ref"]) > SELFTEST_RTOL

        def final():
            c["full"] = gather_full(c["got"], self.sched, plan)

        def verdict():
            full = c["full"]
            probe = full.clone()
            probe.view(-1).view(torch.int16)[0] ^= 1
            comparator_ok = mismatches(probe, full) == 1
            can_exact = c["can_exact"] if self.rs == "canonical" else 0
            rel = c["rel"]
            sens_ok = c["sens_ok"] if self.mode == "sp" else True
            ok = (can_exact == 0 and rel <= SELFTEST_RTOL and (self.rs == "nccl" or rel == 0.0)
                  and comparator_ok and sens_ok and bool(torch.isfinite(full).all()))
            self.shas[name] = _tensor_sha(full)                       # bit-identical across ranks (post-exchange)
            self.rows.append({"case": name, "T": T, "S": plan.S,
                              "canonical_exact": can_exact if self.rs == "canonical" else "n/a (rs=nccl)",
                              "rs_rel": round(rel, 6), "comparator_ok": comparator_ok, "sensitivity_ok": sens_ok,
                              "ok": bool(ok)})

        self.step(f"{name} prep", (), prep)
        self.step(f"{name} reference all-gather", [("ag", plan.padded, None)], reference)
        if self.rs == "canonical":
            self.step(f"{name} canonical reduce-scatter", [("crs", plan.padded, plan)], canonical)
        self.step(f"{name} production reduce-scatter", [(self.rs_op, plan.padded, plan)], production)
        if self.mode == "sp":
            self.step(f"{name} double-reduction probe", [("ag", plan.S, None), (self.rs_op, plan.padded, plan)],
                      sensitivity)
        self.step(f"{name} final all-gather", [("ag", plan.S, None)], final)
        self.step(f"{name} verdict", (), verdict)

    # (2a) live mHC shard-invariance -----------------------------------------------------------------------------
    def live_mhc_case(self, ly, T: int) -> None:
        """check 3 on the LIVE layer weights: hc_pre, hc_fused_post_pre and hc_post+contract on a
        ZERO-PADDED S-row shard; the shard's real rows must be BITWISE the full-T call sliced. The full-T reference
        calls are opaque (prs row-splits them, collectively, when prs is on) and are made on every rank; the shard
        calls run with prs neutralised (as inside an sp region) and are local."""
        W, r, H, dev = self.W, self.r, self.H, self.dev
        plan = plan_rows(T, W)
        n = int(ly.n)
        tag = f"live-L{getattr(ly, 'layer_idx', '?')} T={T} S={plan.S}"
        c: dict = {}

        def inputs():
            c["res"] = synthetic_rows(T, n * H, 70251 + T, dev).view(T, n, H)
            c["x"] = synthetic_rows(T, H, 70252 + T, dev)

        def res_full():
            return _valid_or(c.get("res"), (T, n, H), torch.bfloat16, dev)

        def x_full():
            return _valid_or(c.get("x"), (T, H), torch.bfloat16, dev)

        def pre_args():
            return (res_full(),)

        def pre_fallback():
            return (torch.zeros(T, n, H, dtype=torch.bfloat16, device=dev),)

        def pre_call(a):
            out = ly.hc_pre(a[0], ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base,
                            norm_weight=ly.input_layernorm.weight.data, norm_eps=ly.input_layernorm.variance_epsilon)
            c["pf"] = out
            return out

        def fused_args():
            pf = c["pf"]
            for i, shp in ((0, (T, n, 1)), (1, (T, n, n))):
                if _valid_or(pf[i], shp, torch.float32, dev) is not pf[i]:
                    # fail closed: never weaken the check silently on a valid run (an mHC op contract change)
                    raise _LocalFail(f"hc_pre output {i} {tuple(getattr(pf[i], 'shape', ()))} "
                                     f"{getattr(pf[i], 'dtype', None)} is not the expected contiguous {shp} fp32")
            c["post"], c["comb"] = pf[0], pf[1]
            return (x_full(), res_full(), c["post"], c["comb"])

        def fused_fallback():
            c["post"] = torch.zeros(T, n, 1, dtype=torch.float32, device=dev)
            c["comb"] = torch.zeros(T, n, n, dtype=torch.float32, device=dev)
            return (torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                    torch.zeros(T, n, H, dtype=torch.bfloat16, device=dev), c["post"], c["comb"])

        def fused_call(a):
            out = ly.hc_fused_post_pre(a[0], a[1], a[2], a[3], ly.hc_ffn_fn, ly.hc_ffn_scale, ly.hc_ffn_base,
                                       norm_weight=ly.post_attention_layernorm.weight.data,
                                       norm_eps=ly.post_attention_layernorm.variance_epsilon)
            c["ff"] = out
            return out

        def shards():
            pf, ff = c["pf"], c["ff"]
            res, x, post, comb = c["res"], c["x"], c["post"], c["comb"]
            lo, real = plan.start(r), plan.real_rows(r)
            if real <= 0:
                raise _LocalFail(f"no real rows on rank {r} (plan {plan})")

            def shard_of(full):
                # zero-padded S-row shard: rows [lo, lo+real) are real, [lo+real, lo+S) are padding (never read)
                return pad_rows(full[lo:lo + real], plan.S)

            def cmp(nm, full_out, shard_out):
                mm = mismatches(shard_out[:real], full_out[lo:lo + real])
                self.rows.append({"case": f"{tag} {nm}", "mismatch": mm, "real_rows": real,
                                  "ok": mm == 0 and bool(torch.isfinite(shard_out[:real]).all())})

            with _PrsNeutral():
                ps = ly.hc_pre(shard_of(res), ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base,
                               norm_weight=ly.input_layernorm.weight.data,
                               norm_eps=ly.input_layernorm.variance_epsilon)
                for i, nm in enumerate(("hc_pre.post", "hc_pre.comb", "hc_pre.layer_input")):
                    cmp(nm, pf[i], ps[i])
                fs = ly.hc_fused_post_pre(shard_of(x), shard_of(res), shard_of(post), shard_of(comb),
                                          ly.hc_ffn_fn, ly.hc_ffn_scale, ly.hc_ffn_base,
                                          norm_weight=ly.post_attention_layernorm.weight.data,
                                          norm_eps=ly.post_attention_layernorm.variance_epsilon)
                for i, nm in enumerate(("fused.residual", "fused.post", "fused.comb", "fused.layer_input")):
                    cmp(nm, ff[i], fs[i])
                post_full = _hc_contract(ly.hc_post(x, res, post, comb), n)
                post_shard = _hc_contract(ly.hc_post(shard_of(x), shard_of(res), shard_of(post), shard_of(comb)), n)
                cmp("hc_post+contract", post_full, post_shard)

        self.step(f"{tag} inputs", (), inputs)
        self.sched.opaque(self.rows, f"{tag} hc_pre full-T reference", pre_args, pre_fallback, pre_call)
        self.sched.opaque(self.rows, f"{tag} hc_fused_post_pre full-T reference", fused_args, fused_fallback,
                          fused_call)
        self.step(f"{tag} shard calls", (), shards)

    # (2b) check 2 ---------------------------------------------------------------------------------------
    def check2_case(self, ly, which: str) -> None:
        """A live o_proj / down_proj SUPPRESSED partial (reduce_results off -> no collective inside the module),
        reduced by sp's production reduce-scatter + all-gather vs the decode all-reduce of the same partial (patch
        012's one-shot when engaged): bitwise when RS=canonical and the one-shot is engaged (prefill == decode
        semantics), within SELFTEST_RTOL otherwise. The module is called only after its flag reads off inside the
        handle, so a handle that does not fire can never all-reduce on one rank only."""
        W, r, H, dev = self.W, self.r, self.H, self.dev
        plan2 = Plan(*self.geo["check2"]["plan"])
        T2 = plan2.T
        engaged = bool(self.geo["check2"]["oneshot"])
        tag = f"check2-L{getattr(ly, 'layer_idx', '?')}.{which} T={T2}"
        c: dict = {}

        def partial():
            if which == "o_proj":
                mod, handle = ly.self_attn.o_proj, _attn_reduce_handle(ly.self_attn)
            else:
                mod, handle = ly.mlp.down_proj, _mlp_reduce_handle(ly.mlp, False)
            k = int(getattr(mod, "input_size_per_partition", 0) or 0)
            if k <= 0:
                raise _LocalFail(f"{which} has no input_size_per_partition (not a RowParallelLinear?)")
            x2 = synthetic_rows(T2, k, 7025 + T2 + (0 if which == "o_proj" else 1), dev)
            with handle:
                if getattr(mod, "reduce_results", None) is not False:
                    raise _LocalFail(f"{which}.reduce_results is {getattr(mod, 'reduce_results', None)!r} inside the "
                                     "suppression handle: not calling it (it would all-reduce)")
                out = mod(x2)
            part = out[0] if isinstance(out, tuple) else out
            if not (isinstance(part, torch.Tensor) and tuple(part.shape) == (T2, H) and part.dtype == torch.bfloat16):
                raise _LocalFail(f"{which} partial {getattr(part, 'shape', None)} {getattr(part, 'dtype', None)} != "
                                 f"({T2}, {H}) bf16")
            c["part"] = part.contiguous()
            self.psh[tag] = _tensor_sha(c["part"])

        def sp_reduce():
            c["red"] = gather_full(reduce_scatter_partial(c["part"], self.sched, plan2, self.rs), self.sched, plan2)

        def decode_ar():
            c["ar"] = self.sched.all_reduce_rows(c["part"])

        def verdict():
            red, ar = c["red"], c["ar"]
            exact = mismatches(red, ar)
            rel = _amax_rel(red, ar)
            bitwise = self.rs == "canonical" and engaged
            finite = bool(torch.isfinite(red).all() and torch.isfinite(ar).all())
            ok = finite and (exact == 0 if bitwise else rel <= SELFTEST_RTOL)
            self.shas[f"{tag} sp-reduced"] = _tensor_sha(red)
            self.shas[f"{tag} decode-all-reduce"] = _tensor_sha(ar)
            self.rows.append({"case": tag, "mismatch": exact, "rel": round(rel, 6), "oneshot": engaged,
                              "bitwise_required": bitwise, "ok": bool(ok)})

        self.step(f"{tag} suppressed partial", (), partial)
        self.step(f"{tag} sp reduce-scatter + all-gather", [(self.rs_op, plan2.padded, plan2), ("ag", plan2.S, None)],
                  sp_reduce)
        self.step(f"{tag} decode all-reduce", [("ar", T2, None)], decode_ar)
        self.step(f"{tag} verdict", (), verdict)

    def run(self) -> None:
        for T in _case_ts(self.min_t):
            self.reduction_case(T)
        for ly, desc in zip(self.live, self.geo["live"]):
            is_moe = bool(desc[2])
            self.step(f"L{desc[0]} suppression handles", (),
                      lambda ly=ly, is_moe=is_moe: self.rows.extend(_suppression_cases(ly, is_moe)))
            for T in _live_ts(self.min_t):
                self.live_mhc_case(ly, T)
            for which, _k in desc[4]:
                self.check2_case(ly, which)


def _evaluate(allr: list) -> list[str]:
    """Pure function of the exchanged verdicts (identical on every rank): the list of problems."""
    problems = []
    bad = [r for r in allr if not r["ok"]]
    for r in bad:
        problems.append(f"rank {r['rank']}: " + ", ".join(f"{c['case']} {c}" for c in r["cases"] if not c["ok"]))
    keys = sorted(set().union(*(set(r["shas"]) for r in allr)))
    for k in keys:
        if len({r["shas"].get(k) for r in allr}) != 1:
            problems.append(f"{k}: the result is not bit-identical across ranks")
    pkeys = sorted(set().union(*(set(r["psh"]) for r in allr)))
    for k in pkeys:
        vals = [r["psh"].get(k) for r in allr]
        if None in vals or len(set(vals)) < 2:
            problems.append(f"{k}: the suppressed partial is identical on every rank or missing (the module output "
                            "was already reduced?)")
    return problems


def selftest(world, mode, min_t, rs_mode, hidden: int, device, layers=None, geo=None, live=None) -> dict:
    """Collective boot self-test; raises on EVERY rank unless every rank passed every case (see the module
    docstring for the checks). Hang-free: every collective is issued on every rank through the schedule guard, the
    verdicts are exchanged over gloo behind a bounded barrier, and the decision is a pure function of them."""
    device = _dev(device)
    if geo is None:
        geo, live = _geometry(world, mode, min_t, rs_mode, hidden, device, layers)
    st = _SelfTest(world, mode, min_t, rs_mode, hidden, device, geo, live or [])
    st.run()
    try:
        mine = {"rank": world.rank, "ok": all(r["ok"] for r in st.rows), "cases": st.rows, "shas": st.shas,
                "psh": st.psh}
    except Exception as e:  # noqa: BLE001 - still take part in the exchange
        mine = {"rank": world.rank, "ok": False, "shas": {}, "psh": {},
                "cases": [{"case": "verdict build", "ok": False, "err": f"{type(e).__name__}: {e}"[:300]}]}
    allr = _exchange(world, mine, "self-test verdict exchange")
    problems = _evaluate(allr)
    if problems:
        detail = "; ".join(problems)
        raise RuntimeError(f"{TAG} {ENV}={mode}: BOOT SELF-TEST FAILED (the sp data flow does not reconstruct the "
                           f"reduction reference, ranks disagree, a live mHC op is not shard-invariant, a suppression "
                           f"handle did not fire, or a step failed on a rank): {detail[:6000]}; refusing to serve - "
                           f"unset {ENV}")
    live_rep = None
    if geo["live"]:
        live_rep = {"layers": [d[:4] for d in geo["live"]], "t": [p[0] for p in geo["live_t"]],
                    "cases": sum(1 for r in st.rows if r["case"].startswith(("live-", "suppress-", "check2-"))),
                    "ok": True}
    check2 = None
    if geo["check2"] is not None and geo["live"]:
        check2 = {"T": geo["check2"]["plan"][0], "oneshot": bool(geo["check2"]["oneshot"]),
                  "bitwise": rs_mode == "canonical" and bool(geo["check2"]["oneshot"]),
                  "targets": [f"L{d[0]}.{w}" for d in geo["live"] for w, _k in d[4]]}
    return {"cases": len(st.rows), "ranks": len(allr), "mode": mode, "rs_mode": rs_mode, "live": live_rep,
            "check2": check2, "collectives": st.sched.issued}


# ---------------------------------------------------------------------------------------------------------------------
# The hooks (called from the patch-025 edits of model.py)
# ---------------------------------------------------------------------------------------------------------------------
def begin_forward(T: int, hidden: int, device=None, layers=None, is_sequence_parallel: bool = False,
                  aux_layers=()) -> _Handle | None:
    """Called at the top of Glm5NextModel.forward (before sharding). Returns a handle if this forward is an eligible
    sp/ref forward (and sets sp_active), else None (the model runs its normal replicated path). Boots on the first
    eligible forward. The decision depends only on T, the capture/ubatch state and the config: identical on every
    rank. `layers` feeds the boot self-test's live-weights checks (checks 2 and 3). `aux_layers` is the model's
    aux_hidden_state_layers (EAGLE-3 taps, v0.31+): their per-layer taps would see this rank's rows only, so a forward
    that captures them is never sharded (empty with MTP, the served speculator)."""
    st = _STATE
    mode, min_t, rs_mode = configured()
    if mode is None:
        return None
    if T < min_t:
        st.small_skips += 1
        _assert_default_flags()
        return None
    if aux_layers:
        st.aux_skips += 1
        _assert_default_flags()
        return None
    probe = torch.empty(1, device=device) if device is not None else torch.empty(1)
    if _capturing(probe):
        st.capture_skips += 1
        return None
    if _ubatching_active():
        # A micro-batched forward splits the residual another way; never shard on top of it.
        st.ubatch_skips += 1
        _assert_default_flags()
        return None
    if st.world is None:
        try:
            world = _make_world()
        except Exception as e:  # noqa: BLE001 - this rank cannot even join the preflight
            raise RuntimeError(f"{TAG} {ENV}: could not set up the TP world on this rank ({type(e).__name__}: {e}); "
                               f"the other ranks refuse at the preflight barrier after {SELFTEST_BARRIER_S:.0f} s - "
                               f"refusing") from e
        dev = _dev(probe.device)
        fp, geo, live = _preflight(world, mode, min_t, rs_mode, hidden, dev, layers, is_sequence_parallel)
        st.report = selftest(world, mode, min_t, rs_mode, hidden, dev, layers=layers, geo=geo, live=live)
        st.world = world
        st.booted = True
        rep = st.report
        live_txt = ""
        if rep["live"] is not None:
            lys = ", ".join("%d (%s/%s)" % (d[0], d[1], "moe" if d[2] else "dense") for d in rep["live"]["layers"])
            c2 = rep["check2"] or {}
            live_txt = ("; live layers %s: mHC shard-invariant at T=%s, suppression handles fire, suppressed-partial "
                        "RS+AG %s the decode all-reduce at T=%s (%s)") % (
                lys, "/".join(str(t) for t in rep["live"]["t"]),
                "== (bitwise)" if c2.get("bitwise") else "~= (within tolerance)", c2.get("T"),
                "patch-012 one-shot" if c2.get("oneshot") else "NCCL all-reduce")
        logger.info(
            "%s mHC prefill sequence-parallel ACTIVE (rank %d/%d, mode=%s, rs=%s, T >= %d, eager prefill only; "
            "fingerprint %s): boot self-test OK on all %d ranks (%d cases, %d guarded collectives: %sdata flow "
            "reconstructs the reduction reference, result bit-identical across ranks%s). NUMERICS CHANGE vs the "
            "all-reduce path - serving needs the owner's OK.",
            TAG, world.rank, world.size, mode, rs_mode, min_t, fp, rep["ranks"], rep["cases"], rep["collectives"],
            "canonical==rank-order fp32 sum bitwise, " if rs_mode == "canonical" else "", live_txt)
    plan = plan_rows(T, st.world.size)
    handle = _Handle(mode, rs_mode, plan, st.world)
    st.active_handle = handle
    st.forwards += 1
    _neutralise_prs(T)
    return handle


def end_forward(handle: _Handle | None) -> None:
    """Called in a finally around the layer loop. Clears sp_active and restores prs; idempotent."""
    _restore_prs()
    _STATE.active_handle = None


def _assert_default_flags() -> None:
    """On a non-sp forward the suppression flags must be at their defaults (a prior sp forward restored them)."""
    # Cheap invariant: if a handle leaked we would still be 'active'; begin_forward only reaches here when inert.
    assert _STATE.active_handle is None, f"{TAG} sp handle leaked into a non-sp forward"


def shard_model_input(hidden_states: torch.Tensor, handle: _Handle, rank: int) -> torch.Tensor:
    """Entry: the all-reduced embedding [T,H] -> this rank's [S,H]. (ref mode keeps the full T.)"""
    if handle.mode == "ref":
        return pad_rows(hidden_states, handle.plan.padded)[: handle.plan.T]
    return shard_entry(hidden_states, handle.plan, rank)


def gather_model_output(hidden_states: torch.Tensor, handle: _Handle) -> torch.Tensor:
    """Exit: this rank's [S,H] (last layer's contracted output) -> the full [T,H] before the norm."""
    if handle.mode == "ref":
        return hidden_states[: handle.plan.T]
    return gather_full(hidden_states, handle.world, handle.plan)


def _reduce_block(partial_full: torch.Tensor, handle: _Handle) -> torch.Tensor:
    """Attention / MLP partial [T,H] -> this rank's reduced rows. sp: [S,H]; ref: reduce-scatter then all-gather back
    to the full [T,H] (the replicated mHC needs all rows; this is the reduction-only experiment)."""
    shard = reduce_scatter_partial(partial_full, handle.world, handle.plan, handle.rs_mode)
    if handle.mode == "ref":
        return gather_full(shard, handle.world, handle.plan)
    return shard


def sp_layer_forward(layer, positions, hidden_states, residual, post, comb):
    """Full replacement of Glm5NextDecoderLayer.forward for an mHC layer inside an eligible sp/ref forward. Mirrors
    the upstream forward (common/model.py:553-623) with the residual stream kept on this rank's rows (sp) and the attention /
    MLP all-reduces replaced by reduce-scatter (sp) or reduce-scatter+all-gather (ref)."""
    handle = _STATE.active_handle
    assert handle is not None, f"{TAG} sp_layer_forward called outside an sp region"
    n = layer.n
    x = hidden_states

    # --- attention mHC pre (fused with the previous layer's deferred ffn hc_post) ---
    if post is None:
        if layer.layer_idx == 0:
            x = _hc_expand(x, n)
        residual = x
        post, comb, x = layer.hc_pre(
            x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            norm_weight=layer.input_layernorm.weight.data, norm_eps=layer.input_layernorm.variance_epsilon)
    else:
        residual, post, comb, x = layer.hc_fused_post_pre(
            x, residual, post, comb, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            norm_weight=layer.input_layernorm.weight.data, norm_eps=layer.input_layernorm.variance_epsilon)

    # --- attention on the whole sequence; o_proj all-reduce suppressed -> reduce-scatter ---
    x_full = gather_full(x, handle.world, handle.plan) if handle.mode == "sp" else x
    with _attn_reduce_handle(layer.self_attn):
        x_partial = layer.self_attn(hidden_states=x_full, positions=positions)
    x = _reduce_block(x_partial, handle)

    # --- ffn mHC pre (fused with this layer's attention hc_post) ---
    residual, post, comb, x = layer.hc_fused_post_pre(
        x, residual, post, comb, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
        norm_weight=layer.post_attention_layernorm.weight.data,
        norm_eps=layer.post_attention_layernorm.variance_epsilon)

    # --- MLP / MoE on the whole sequence; final all-reduce suppressed -> reduce-scatter ---
    x_full = gather_full(x, handle.world, handle.plan) if handle.mode == "sp" else x
    with _mlp_reduce_handle(layer.mlp, layer._mlp_is_moe):
        if layer._mlp_is_moe:
            x_partial = layer.mlp(x_full, already_sequence_parallel=False)
        else:
            x_partial = layer.mlp(x_full)
    x = _reduce_block(x_partial, handle)

    _STATE.layers += 1

    # --- mHC end ---
    if layer.layer_idx == layer.num_hidden_layers - 1:
        x = layer.hc_post(x, residual, post, comb)
        x = _hc_contract(x, n)
        return x, None, None, None
    return x, residual, post, comb


def _hc_expand(x, n):
    from vllm.model_executor.layers.mhc import hc_expand

    return hc_expand(x, n)


def _hc_contract(x, n):
    from vllm.model_executor.layers.mhc import hc_contract

    return hc_contract(x, n)


def stats() -> dict:
    st = _STATE
    return {"booted": st.booted, "mode": st.mode, "min_t": st.min_t, "rs_mode": st.rs_mode,
            "world": getattr(st.world, "kind", None), "forwards": st.forwards, "layers": st.layers,
            "capture_skips": st.capture_skips, "small_skips": st.small_skips, "ubatch_skips": st.ubatch_skips,
            "aux_skips": st.aux_skips, "active": sp_active(), "report": st.report}


def _set_world_factory_for_test(factory: Callable[[], object] | None) -> None:
    global _WORLD_FACTORY
    _WORLD_FACTORY = factory


def _reset_for_test(mode: str | None = None, min_t: int | None = None, rs_mode: str | None = None) -> None:
    global _STATE
    _STATE = _State()
    _STATE.mode, _STATE.min_t, _STATE.rs_mode = mode, min_t, rs_mode
