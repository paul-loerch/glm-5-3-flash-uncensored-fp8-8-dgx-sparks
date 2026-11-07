# SPDX-License-Identifier: Apache-2.0
# [patch-029 gb10-draft-compress] installed as vllm/_gb10_draft_compress.py by patches/029-gb10-draft-compress.sh
# (draft-path compression; revision 2). Written for this recipe.
"""GB10, TP=8: compress the two MTP draft passes per decode step (opt-in VLLM_GB10_DRAFT_COMPRESS=1).

REVISION 3 = the vLLM v0.32.0 port of rev 2 (everything below is rev 2's text unless it says v0.32.0). What changed:
  * targets: vllm/models/glm5next/common/mtp.py (moved from nvidia/) and vllm/v1/worker/gpu/spec_decode/mtp/speculator.py;
    _find_mtp checks the class' module name glm5next.common.mtp.
  * Marlin API drift (#57421 and neighbours): marlin_utils_fp8.prepare_fp8_layer_for_marlin no longer creates layer.workspace
    and apply_fp8_marlin_linear takes workspace=None meaning "the workspace manager's persistent per-stream locks"
    (marlin_utils.get_marlin_workspace); the FP8 head allocates its OWN int32 lock buffer before any CUDA-graph capture
    (marlin_make_workspace_new(device, MARLIN_MAX_BLOCKS_PER_SM), default 4) and always passes it, so nothing is allocated
    inside a capture. The op-presence check also requires torch.ops.vllm.marlin_gemm (registered by marlin_utils).
  * the served configuration is VLLM_GB10_DRAFT_COMPRESS_PARTS=head. The eh part (eh_proj row split + patch 012's one-shot) is
    carried UNCHANGED from rev 2: it needs patch 012, excludes patch 032 (replicated embedding), and is verified on the CPU
    suite only - it has not been looked at on a GPU with v0.32.0.
  * upstream vLLM main has speculative_config.draft_lm_head_quantization (#59973, MRV2, per-row FP8 draft head); v0.32.0 does
    not, so this module still builds the draft-only FP8 head. On a base that has that flag this patch is not needed.
  * limitation (unchanged): only the logits path model.compute_logits is hooked. v0.32.0's DraftModelSpeculator.sample_draft
    takes it for draft_sample_method=probabilistic (the served config) and for greedy drafting WITHOUT
    use_local_argmax_reduction; with use_local_argmax_reduction=True (greedy only; refused for probabilistic) it calls
    model.get_top_tokens, which stays on the BF16 head (the patch is then simply not engaged for the draft head).

REVISION 2. The revision-1 boot refused the eh part on every rank by its E5 speed gate ("E5 eh M=3 split faster:
0.437 vs 0.295 ms"): rev 1 timed split + one-shot EAGERLY, where the extra one-shot all-reduce is launch-bound; a
microbenchmark measured inside CUDA graphs 291 us (replicated eh_proj, M=3) vs 42 us (1/8 row split + pad) + ~47 us
(one-shot). Rev 2 times E5 inside CUDA graphs captured at boot on a side stream (one leading one-shot, which stands
for the draft graph's embedding all-reduce that already precedes eh_proj in every served draft replay, then N calls of
the split path / of the replicated eh_proj; replayed in a fixed collective order; medians compared; the gloo capture
verdict and the existing verdict / commit rounds keep every rank on the same decision). The eager timing is kept as
information (and as the side stream's warm-up). VLLM_GB10_DRAFT_COMPRESS_EH_GATE=numerics (operator fallback) gates
the eh part on its numerics only. The eh part now also requires the draft's embed_tokens to be vocab-parallel over
the TP group (refused with patch 032's replicated embedding, whose .sh refuses this combination too). Log lines never
quote other gates' texts (see _safe()). Unchanged from rev 1: the hooks, the FP8 head, E1-E4, G1-G4, the protocol.

Used only by the patch-029 edits of vllm/models/glm5next/common/mtp.py (Glm5NextMultiTokenPredictorLayer.forward,
Glm5NextMultiTokenPredictor.compute_logits) and of vllm/v1/worker/gpu/spec_decode/mtp/speculator.py
(MTPSpeculator.load_draft_model -> prepare()).

WHY. At batch 1 with k=2 every decode step runs two MTP draft passes. Each one reads, per rank,
  (a) the full BF16 lm_head shard: 19,360 x 4,096 bf16 = 158.6 MB (load_eagle_model shares the TARGET's lm_head into
      the MTP layer's shared_head.head), and
  (b) the eh_proj weight, an nn.Linear(8,192 -> 4,096) REPLICATED on every rank: 67.1 MB.
Both are pure weight streaming at M = 1..4 rows (~215-230 GB/s on GB10).

WHAT.
  (a) "head": a draft-only FP8 copy of this rank's lm_head shard: per-output-channel (vocab row) e4m3, scale =
      amax/448 rounded to bf16 (the scale Marlin applies), W8A16 through the image's Marlin FP8 GEMM
      (marlin_utils_fp8.prepare_fp8_layer_for_marlin(size_k_first=False) + apply_fp8_marlin_linear: the path
      measured on GB10 in this image, 220-229 GB/s at M = 1..16, kda-fp8-bench 2026-09-28). 79.4 MB instead of
      158.6 MB per draft pass. compute_logits of the MTP model hands the real LogitsProcessor a stand-in head whose
      quant_method.apply runs the FP8 GEMM; the LogitsProcessor's all-gather and vocab slice are unchanged.
  (b) "eh": a weight-ROW (output-feature) split of eh_proj: rank r keeps rows [r*H/W, (r+1)*H/W) of the [H, 2H] weight
      (512 x 8,192 = 8.4 MB), computes its [M, H/W] block of the output with the full-K dot product, zero-pads it to
      [M, H] at its column offset, and patch 012's one-shot all-reduce (canonical rank-order fp32 sum, one bf16
      rounding) assembles the full [M, H] on every rank. The other 7 blocks are exact zeros, so the "sum" is an EXACT
      gather: every output element is its owner rank's full-K dot product, bit for bit (only the GEMM kernel's own
      accumulation order can differ from the replicated nn.Linear). M <= the one-shot limit (64 KiB = 8 rows at
      H = 4,096: both draft passes at batch 1, the single draft pass at k=1 up to batch 4); larger M (the draft pass
      over a prompt chunk, batch >= 5) keeps the replicated eh_proj, unchanged.
  The TARGET model's lm_head, logits and verification are untouched: the shared lm_head module is only READ once at
  load to build the separate FP8 copy; its weight / quant_method are never modified; Glm5NextForCausalLM.compute_logits
  is not edited.

EXACTNESS OF THE SERVED DISTRIBUTION. The draft token is sampled by gumbel_sample() FROM the logits this
compute_logits returns, and the same tensor (bf16, pre-temperature) is what gumbel_sample stores in the draft-logits
cache that the rejection sampler reads (gumbel.py: "Consumers ... reproduce the value used below bitwise"). So the
draft probabilities used for acceptance are exactly the ones the draft was sampled from, whatever the draft logits
are; rejection sampling then emits exactly the target distribution. Patch 017's salt keeps the draft Philox stream
disjoint from the target's resample stream. Only the acceptance rate can move. The draft path stays deterministic
(Marlin with fp32 global reduce, no atomics; the one-shot's fixed-order sum) and identical on every rank (each rank
computes its own vocab shard, the all-gather is a copy; the one-shot output is bit-identical on all ranks).

BOOT SELF-TEST (prepare(), at load time, right after load_eagle_model shared the lm_head; collective over the TP
group; HANG-FREE; FAIL CLOSED = refuse to activate and log why, NEVER raise into the engine):
  0. local: envelope (TP >= 2, PP = 1, GLM5Next MTP draft, bf16 weights, image LogitsProcessor with all-gather,
     unquantized lm_head, not VLLM_BATCH_INVARIANT, Marlin FP8 ops present, patch 012's one-shot present for "eh",
     the draft's embed_tokens vocab-parallel over the TP group for "eh", a valid VLLM_GB10_DRAFT_COMPRESS_EH_GATE),
     FP8 build + the local head checks on synthetic hidden states (unit-RMS rows x the live shared_head.norm weight;
     identical on every rank):
       H1 Marlin output vs the exact fp32 dequant GEMM (q*s): rel-RMSE <= 0.01 (kernel / packing / scale-permutation)
       H2 FP8 vs the image BF16 head: rel-RMSE <= 0.05, max |diff| / worst-case FP8 bound <= 1, finite, bf16, shape
       H3 top-1 agreement: >= 0.5 on 128 random rows (a gross-failure floor; real-size simulation 0.92: random
          hidden states give flat logits, so a correct FP8 head flips the rows whose top-2 margin is below its
          ~0.4 max|diff|); 100 % on >= 16 rows built to peak at a chosen vocab row (64 candidates, each lifted
          along w_v - the shard's mean row (centred: robust to an anisotropic unembedding) to a margin of
          max(3, 8 x max|diff|); kept if the BF16 head then leads by >= half of it)
       H4 determinism: two runs bitwise equal
       H5 speed: median Marlin FP8 <= 0.9 x median image BF16 head at M = 1 (DRAM-bound: 158.6 MB > 24 MB L2)
  1. preflight exchange (gloo, bounded monitored barrier): fingerprint (module sha, gate, parts, eh gate, TP size,
     shapes, one-shot limit) + every rank's local verdicts -> a plan that is a pure function of the exchanged rows.
  2. collective checks, issued on every rank in the planned order with planned shapes (a rank-local failure feeds
     zeros and fails its rows, it never skips or reshapes a collective):
       E1-E4 eh at M in {1, 3, 4, 8} (<= the one-shot limit): own output block bitwise == the local block, max |split
       - replicated| <= 2^-8 (|split| + |full|) + 2^-10 (|x| @ |W|^T) (simulated <= 0.7 of it; a wrong / missing /
       double block is >= 90x over), finite, sha for the cross-rank identity;
       E5 speed at M = 1 and 3, on a side stream: (a) eager timing of split + one-shot vs the replicated eh_proj
       (INFORMATION only since rev 2; a raising split still fails the row; it also warms the side stream: cuBLAS
       workspace, Triton JIT, NCCL connections of these sizes); (b) eh gate "graph" (default): per M two CUDA graphs
       captured locally on the side stream (thread-local capture mode; nothing is communicated while capturing),
       G_split = [one-shot] + N x [block -> pad -> one-shot] and G_full = [one-shot] + N x [eh_proj] (N = 8; the
       leading one-shot absorbs NCCL's per-replay host-callback round trip in BOTH graphs, as the draft embedding's
       all-reduce does in the served draft replays; each graph must record exactly N + 1 / 1 one-shots, and a failed
       capture's private pool is unregistered from the caching allocator if torch's capture_end left it registered);
       a gloo capture verdict (no rank replays unless EVERY rank captured every graph); W + I = 2 + 10 replays each in
       a fixed interleaved order (collective, every replayed one-shot counted against the plan; a replay that fails
       to launch has its one-shots issued eagerly with zeros); pass iff median(G_split) <= 0.9 x median(G_full) (the
       shared leading one-shot makes that stricter than 0.9 per call). Eh gate "numerics": (a) only, no speed gate.
       G1-G4 head through the REAL LogitsProcessor (its NCCL all-gather) at M = 1, 3, 16: this rank's slice of the
       gathered FP8 logits == its local FP8 shard (values), shape [M, vocab], finite, gathered FP8 vs gathered BF16
       relRMSE <= 0.05, sha for the cross-rank identity, full-vocab top-1 agreement vs the BF16 head (logged).
  3. verdict exchange (gloo, bounded): "head" activates iff every rank passed every head row and the gathered
     logits are bit-identical across ranks; "eh" likewise for its rows and the assembled outputs. The decision is a
     pure function of the exchanged rows, so every rank takes the same one. Anything else (a refused envelope, a
     fingerprint mismatch, a failed or timed-out exchange) leaves that part on the image path.
  Log: one line per rank, "[patch-029] draft-path compression ACTIVE (...)" or "... INACTIVE (fail closed): ...".

CUDA graphs: the hooks choose the path from shapes / dtypes only (identical at capture and replay, and on every
rank); the FP8 weights, scales, Marlin workspace and the eh slice are allocated at load, before any capture; the
one-shot is graph-capturable (patch 012). The E5 boot graphs use their own private memory pools and are reset
(NCCL's persistent plans released) before prepare() returns; a capture that fails leaves no pool registered with the
caching allocator (_unregister_pool: else vLLM's next torch.cuda.empty_cache() would assert). Memory: +79.4 MB (FP8
head) + 8.4 MB (eh slice) per rank; the self-test temporaries are freed before the KV cache is sized.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import math
import os
import re
from collections.abc import Callable
from typing import NamedTuple

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001 - CPU tests outside a vLLM install
    logger = logging.getLogger(__name__)

TAG = "[patch-029]"
ENV = "VLLM_GB10_DRAFT_COMPRESS"
ENV_PARTS = "VLLM_GB10_DRAFT_COMPRESS_PARTS"
ENV_BATCH_INVARIANT = "VLLM_BATCH_INVARIANT"
ENV_EH_GATE = "VLLM_GB10_DRAFT_COMPRESS_EH_GATE"
REVISION = 3
PARTS = ("eh", "head")
EH_GATES = ("graph", "numerics")         # E5: CUDA-graph timed speed gate (default) / numerics only (operator fallback)
FP8_MAX = 448.0
QUANT_CHUNK_ROWS = 2048                  # FP8 build / reference GEMMs in row chunks (bounded temporaries)
HEAD_MAX_M = 64                          # draft-logits rows eligible for the FP8 head (served: num_reqs <= 8)
HEAD_TEST_RANDOM = 128                   # H2/H3 random rows
HEAD_TEST_PEAKED = 16                    # H3 peaked rows: at least this many must be built (and all must agree)
HEAD_PEAK_CANDIDATES = 64                # H3 peaked-row candidates (deterministic local vocab rows; a row that a
#                                          lift along its centred w_v cannot peak - another vocab row projects more
#                                          on it, e.g. a small-norm rare token - is skipped, not failed)
HEAD_GATHER_M = (1, 3, 16)               # G1-G4 through the real LogitsProcessor (collective all-gather)
EH_TEST_M = (1, 3, 4, 8)                 # E1-E4 (filtered to <= the one-shot limit)
EH_TIME_M = (1, 3)                       # E5 timing rows (collective, fixed iteration / replay counts)
TIME_WARMUP = 3
TIME_ITERS = 10
PERF_RATIO_MAX = 0.9                     # H5: the FP8 head must be at least 10 % faster (both GPU-bound at M = 1)
# E5 (rev 2): CUDA-graph timing. The eager timing (TIME_WARMUP + TIME_ITERS calls) is INFORMATION only: there the
# split path is launch-bound (0.437 vs 0.295 ms at M = 3, refused by rev 1) while inside graphs it is
# ~42 + ~47 us vs 291 us (microbenchmark).
EH_GRAPH_CALLS = 8                       # path calls per captured graph, after ONE leading one-shot in both graphs
EH_GRAPH_WARMUP = 2                      # untimed replays per graph
EH_GRAPH_ITERS = 10                      # timed replays per graph (median)
EH_GRAPH_RATIO_MAX = 0.9                 # median(G_split) <= 0.9 x median(G_full); both carry the same leading
#                                          one-shot, so this is stricter than 0.9 per call (expected ~0.45)
CAPTURE_ERROR_MODE = "thread_local"      # our own unsafe calls fail the capture; other threads are not affected
HEAD_KERNEL_REL_MAX = 0.01               # H1 (GB10 measurement of the same Marlin path: 0.0017-0.0022)
HEAD_REL_MAX = 0.05                      # H2 (per-channel e4m3: ~0.026-0.028 simulated / measured on KDA weights)
HEAD_TOP1_RANDOM_MIN = 0.5               # H3 random rows: a gross-failure floor only (real-size simulation 0.92;
#                                          random hidden states give flat logits, so a CORRECT FP8 head flips the
#                                          rows whose top-2 margin is below its own ~0.4 max|diff|; H1/H2 carry the
#                                          numerical guarantee). 128 rows: binomial noise 0.03, no false refusals.
PEAK_MARGIN_MIN = 3.0
BOUND_RATIO_MAX = 1.0                    # H2 / E3: rigorous worst-case bounds (typ. 0.06 / 0.7)
SELFTEST_BARRIER_S = 600.0               # bounded gloo barrier before each exchange (ranks reach load at different
#                                          times: the target weights load for ~8-11 min on each node)
SEED = 0x029D


def _own_sha16() -> str:
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


MODULE_SHA16 = _own_sha16()


# ---------------------------------------------------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------------------------------------------------
def gate_on(raw: str | None = None) -> bool:
    raw = os.environ.get(ENV, "0") if raw is None else raw
    return str(raw).strip() == "1"


def parse_parts(raw: str | None) -> tuple[str, ...]:
    """VLLM_GB10_DRAFT_COMPRESS_PARTS: comma list of 'head' / 'eh' (default both). Raises ValueError on anything else
    (the .sh validates the same grammar before the boot; prepare() turns a ValueError into INACTIVE)."""
    if raw is None or str(raw).strip() == "":
        return PARTS
    toks = [t.strip().lower() for t in str(raw).split(",")]
    if not toks or any(t not in PARTS for t in toks):
        raise ValueError(f"{ENV_PARTS}={raw!r}: expected a comma list of {list(PARTS)}")
    return tuple(sorted(set(toks)))


def parse_eh_gate(raw: str | None) -> str:
    """VLLM_GB10_DRAFT_COMPRESS_EH_GATE: 'graph' (default: E5 timed inside CUDA graphs) or 'numerics' (operator
    fallback: the eh part is gated on E1-E4 only; the eager timing is logged). Raises ValueError on anything else
    (the .sh refuses the same values before the boot; prepare() turns it into an eh refusal)."""
    v = "" if raw is None else re.sub(r"\s+", "", str(raw)).lower()
    if v == "":
        return EH_GATES[0]
    if v not in EH_GATES:
        raise ValueError(f"{ENV_EH_GATE}={raw!r}: expected one of {list(EH_GATES)}")
    return v


# Texts other gates grep for (every launch/readiness.txt fail AND require rule, the boot-gate scripts' ACTIVE /
# INACTIVE / [OK] counts): a quoted exception or reason must never reproduce one inside a [patch-029] line (rev 2;
# rev-1 residual risk 2). Applied to every reason / exception text this module logs; the information stays readable.
_DEFANG = (
    (re.compile(r"INACTIVE"), "inactive"),
    (re.compile(r"ACTIVE"), "active"),
    (re.compile(r"\[OK\]"), "(ok)"),
    (re.compile(r"\[(patch-)?0\d\d\]"), lambda mo: "(" + mo.group(0)[1:-1] + ")"),
    (re.compile(r"\[recipe-entrypoint\]"), "(recipe-entrypoint)"),
    (re.compile(r"OutOfMemoryError"), "out-of-memory error"),
    (re.compile(r"NCCL (WARN|error)"), lambda mo: "NCCL-" + mo.group(1).lower()),
    (re.compile(r"no kernel image"), "no-kernel-image"),
    (re.compile(r"cudaFuncSetAttribute"), "cudaFuncSetAttr"),
    (re.compile(r"NV_ERR_NO_MEMORY"), "NV-ERR-NO-MEMORY"),
    (re.compile(r"EngineDeadError"), "engine-dead error"),
    (re.compile(r"Engine core initialization failed"), "engine-core init failed"),
    (re.compile(r"died unexpectedly"), "died-unexpectedly"),
    (re.compile(r"invalid choice"), "invalid-choice"),
    (re.compile(r"Too many open files"), "too-many-open-files"),
    (re.compile(r"FilteredTopK fallback requires"), "FilteredTopK-fallback requires"),
    (re.compile(r"Free memory on device"), "free-memory on device"),
    (re.compile(r"Graph capturing finished"), "graph-capturing finished"),
    (re.compile(r"Application startup complete"), "application-startup complete"),
    # (verification of rev 2) the remaining launch/readiness.txt fail rules and the require rules (a quoted require
    # text would satisfy that rule falsely): "Using <X> attention / MoE backend", "pe_dim must be 64 ...", ...
    (re.compile(r"\bUsing\b"), "using"),
    (re.compile(r"pe_dim must be"), "pe-dim must be"),
    (re.compile(r"Unknown SF transformation"), "unknown SF transformation"),
    (re.compile(r"busy_loop_s = "), "busy_loop_s="),
    (re.compile(r"world_size="), "world_size:"),
    (re.compile(r"Model loading took"), "model loading took"),
    (re.compile(r"Setting attention block size"), "setting attention block size"),
    (re.compile(r"for KV Cache"), "for KV-cache"),
    (re.compile(r"GPU KV cache size"), "GPU KV-cache size"),
    (re.compile(r"init engine \("), "init-engine ("),
    (re.compile(r"Starting vLLM server"), "starting vLLM server"),
    # (v0.32.0 recipe) its readiness file adds two fail rules: vLLM's FlashInfer all-reduce init failure and the stale-env-var warning
    (re.compile(r"Failed to initialize FlashInfer Allreduce"), "failed to initialize FlashInfer Allreduce"),
    (re.compile(r"Unknown vLLM environment variable"), "unknown vLLM environment variable"),
)


def _safe(text) -> str:
    s = str(text)
    for pat, rep in _DEFANG:
        s = pat.sub(rep, s)
    return s


# ---------------------------------------------------------------------------------------------------------------------
# Worlds: the TP group (production) or an injected test world with the same interface
# ---------------------------------------------------------------------------------------------------------------------
class _TPWorld:
    """vLLM's tensor-parallel group: rank / size, patch 012's one-shot object, and gloo object all-gather behind a
    bounded monitored barrier (the group's CPU group; no new communicator)."""

    kind = "tp"

    def __init__(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        self.group = get_tp_group()
        self.rank = int(self.group.rank_in_group)
        self.size = int(self.group.world_size)
        self.pp_size = int(get_pp_group().world_size)

    def oneshot(self):
        return getattr(getattr(self.group, "device_communicator", None), "_gb10_oneshot", None)

    def all_gather_object(self, obj):
        import torch.distributed as dist

        out = [None] * self.size
        dist.all_gather_object(out, obj, group=self.group.cpu_group)
        return out

    def barrier_bounded(self, timeout_s: float) -> None:
        import torch.distributed as dist

        dist.monitored_barrier(group=self.group.cpu_group, timeout=datetime.timedelta(seconds=timeout_s),
                               wait_all_ranks=True)


_WORLD_FACTORY: Callable[[], object] | None = None   # tests only


def _make_world():
    return _WORLD_FACTORY() if _WORLD_FACTORY is not None else _TPWorld()


def _marlin_fp8():
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as MU8

    return MU8


def _make_workspace(device):
    """A zeroed int32 Marlin lock buffer (num SMs x MARLIN_MAX_BLOCKS_PER_SM, default 4: what get_marlin_workspace uses)."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils as MU

    return MU.marlin_make_workspace_new(device, int(getattr(MU, "MARLIN_MAX_BLOCKS_PER_SM", 4)))


# ---------------------------------------------------------------------------------------------------------------------
# (a) the draft-only FP8 lm_head
# ---------------------------------------------------------------------------------------------------------------------
def quantize_rows_fp8(w: torch.Tensor, chunk: int = QUANT_CHUNK_ROWS) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row FP8 e4m3 of a [N, K] bf16 weight. The scale amax/448 is rounded to bf16 FIRST (the value Marlin
    multiplies by), then q = clamp(w / scale, +-448) cast to e4m3 (round-to-nearest-even; the clamp keeps the cast in
    range: e4m3fn has no inf). dequant = q * scale is exact in fp32. Zero rows get q = 0."""
    n, k = int(w.shape[0]), int(w.shape[1])
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((n,), dtype=torch.bfloat16, device=w.device)
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        wf = w[r0:r1].float()
        sc = (wf.abs().amax(dim=1) / FP8_MAX).clamp_min(2.0 ** -100).to(torch.bfloat16)
        q[r0:r1] = (wf / sc.float().unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        s[r0:r1] = sc
    return q, s


class _Fp8Head:
    """Stand-in lm_head for the MTP compute_logits ONLY: LogitsProcessor._apply_head calls
    lm_head.quant_method.apply(lm_head, x, bias) and _get_logits reads lm_head.tp_size; both are this object. The
    weight is the Marlin repack of the e4m3 copy of `head.weight` (this rank's vocab shard)."""

    def __init__(self, head) -> None:
        w = head.weight
        self.head = head                      # identity check in draft_logits(): the shared lm_head it mirrors
        self.size_n, self.size_k = int(w.shape[0]), int(w.shape[1])
        self.tp_size = int(head.tp_size)
        self.q, self.s = quantize_rows_fp8(w)  # kept for the boot self-test only (release_reference())
        L = torch.nn.Module()
        L.weight = torch.nn.Parameter(self.q, requires_grad=False)
        L.weight_scale = torch.nn.Parameter(self.s, requires_grad=False)
        L.orig_dtype = torch.bfloat16
        L.output_size_per_partition = self.size_n
        L.input_size_per_partition = self.size_k
        L.weight_block_size = None            # channel-wise scales
        _marlin_fp8().prepare_fp8_layer_for_marlin(L, size_k_first=False)
        self.w = L.weight.data
        self.ws = L.weight_scale.data
        # v0.32.0: prepare_fp8_layer_for_marlin no longer creates layer.workspace; apply_fp8_marlin_linear(workspace=None)
        # would take the workspace manager's per-stream locks (allocated lazily, possibly inside a CUDA-graph capture).
        # Own int32 lock buffer, allocated here (load time, before any capture), always passed explicitly.
        self.workspace = _make_workspace(w.device)
        self.quant_method = self
        self.bytes = int(self.w.numel() * self.w.element_size() + self.ws.numel() * self.ws.element_size())

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        out = _marlin_fp8().apply_fp8_marlin_linear(
            input=x.contiguous(), weight=self.w, weight_scale=self.ws, workspace=self.workspace,
            size_n=self.size_n, size_k=self.size_k, bias=None,
        )
        return out if bias is None else out + bias

    def eligible(self, lp, h: torch.Tensor) -> bool:
        return bool(
            h.dim() == 2 and h.dtype == torch.bfloat16 and int(h.shape[1]) == self.size_k
            and 0 < int(h.shape[0]) <= HEAD_MAX_M and h.is_cuda
            and getattr(lp, "head_dtype", None) in (None, torch.bfloat16)
            and not getattr(lp, "logits_as_input", False)
        )

    def release_reference(self) -> None:
        self.q = self.s = None


# ---------------------------------------------------------------------------------------------------------------------
# (b) the eh_proj weight-row split + patch 012's one-shot all-reduce (exact gather of disjoint blocks)
# ---------------------------------------------------------------------------------------------------------------------
class _EhSplit:
    def __init__(self, eh_proj, rank: int, world_size: int, oneshot) -> None:
        w = eh_proj.weight                    # [H, 2H]
        self.H, self.K = int(w.shape[0]), int(w.shape[1])
        if self.H % world_size:
            raise ValueError(f"eh_proj out features {self.H} not divisible by the TP size {world_size}")
        self.n = self.H // world_size
        self.off = rank * self.n
        self.left, self.right = self.off, self.H - self.off - self.n
        self.w = w[self.off:self.off + self.n].contiguous()
        self.oneshot = oneshot
        self.max_tokens = int(oneshot.max_elems) // self.H
        self.bytes = int(self.w.numel() * self.w.element_size())

    def eligible(self, x: torch.Tensor) -> bool:
        return bool(x.dim() == 2 and x.dtype == torch.bfloat16 and int(x.shape[1]) == self.K
                    and 0 < int(x.shape[0]) <= self.max_tokens and x.is_cuda)

    def block(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.w)

    def pad(self, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(y, (self.left, self.right))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.oneshot(self.pad(self.block(x)))


# ---------------------------------------------------------------------------------------------------------------------
# Runtime hooks (called from the patched mtp.py; shapes / dtypes only -> identical at capture, replay, on every rank)
# ---------------------------------------------------------------------------------------------------------------------
def eh_proj_forward(layer, x: torch.Tensor) -> torch.Tensor:
    sp = getattr(layer, "_p029_eh", None)
    if sp is not None and sp.eligible(x):
        return sp(x)
    return layer.eh_proj(x)


def draft_logits(predictor, mtp_layer, hidden_states: torch.Tensor):
    lp = predictor.logits_processor
    head = mtp_layer.shared_head.head
    fh = getattr(predictor, "_p029_head", None)
    if fh is not None and fh.head is head and fh.eligible(lp, hidden_states):
        return lp(fh, hidden_states)
    return lp(head, hidden_states)


# ---------------------------------------------------------------------------------------------------------------------
# Boot self-test helpers
# ---------------------------------------------------------------------------------------------------------------------
def _synthetic(rows: int, cols: int, seed: int, device, scale: torch.Tensor | None = None) -> torch.Tensor:
    """[rows, cols] bf16, unit RMS per row (x scale[cols] when given), from a seeded CPU generator: identical on every
    rank (same seed, same torch build)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    x = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    x = x / (x * x).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    x = x.to(device)
    if scale is not None:
        x = x * scale.float().to(device).unsqueeze(0)
    return x.to(torch.bfloat16)


def _mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return max(a.numel(), b.numel(), 1)
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return int((a.contiguous().view(iv) != b.contiguous().view(iv)).sum().item())


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    d = a.float() - ref.float()
    den = float((ref.float() * ref.float()).sum().item())
    return math.sqrt(float((d * d).sum().item()) / max(den, 1e-30))


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t.float()).all().item())


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def _cuda_timer(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """median ms of fn() over iters CUDA-event-timed calls (after warmup calls)."""
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    return float(ts[len(ts) // 2])


_TIMER: Callable[[Callable[[], object], int, int], float] = _cuda_timer   # tests replace it


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _capture(calls: list, stream, dev=None):
    """One CUDA graph of calls() on `stream` (its own private memory pool, from a handle kept here; thread-local
    capture mode: an unsafe call of THIS thread invalidates the capture, other threads are not affected). Nothing runs
    or communicates while capturing: the NCCL p2p of a captured one-shot is issued at every replay. Raises on failure
    (graph reset; the pool unregistered from the caching allocator if torch's capture_end left it registered)."""
    g = torch.cuda.CUDAGraph()
    pool = _new_pool()
    try:
        with torch.cuda.graph(g, pool=pool, stream=stream, capture_error_mode=CAPTURE_ERROR_MODE):
            for c in calls:
                c()
    except BaseException:
        _reset_graph(g)
        _unregister_pool(pool, dev)
        raise
    return g


def _new_pool():
    try:
        return torch.cuda.graph_pool_handle()
    except Exception:  # noqa: BLE001 - torch.cuda.graph then makes its own pool (nothing to unregister by hand)
        return None


def _unregister_pool(pool, dev) -> str:
    """After a FAILED capture only. torch's CUDAGraph::capture_end raises at cudaStreamEndCapture (an INVALIDATED
    capture: a prohibited call inside it) before it unregisters the capture's private pool from the caching allocator
    (endAllocateToPool). The allocator then keeps that capture 'underway', and every later torch.cuda.empty_cache() -
    vLLM's DeviceMemoryProfiler calls it right after load_model - fails its captures_underway.empty() assert: the boot
    would die instead of refusing the eh part. Unregister the pool with the binding the image's own vLLM uses
    (distributed/device_communicators/pynccl_allocator.py: torch._C._cuda_endAllocateToPool(device index, pool id)),
    then drop the use count capture_begin took (_cuda_releasePool; reset() does not, the capture never ended) so the
    pool can be freed. If capture_end did unregister it (a clean end, a Python error inside a valid capture), the
    binding refuses ('not currently recording') and nothing else is touched. -> 'unregistered' / 'not registered' /
    why it was not tried."""
    if pool is None:
        return "no pool handle"
    end = getattr(getattr(torch, "_C", None), "_cuda_endAllocateToPool", None)
    if end is None:
        return "no _cuda_endAllocateToPool binding"
    try:
        idx = int(dev.index) if getattr(dev, "index", None) is not None else int(torch.cuda.current_device())
        end(idx, pool)
    except Exception:  # noqa: BLE001 - not registered: capture_end unregistered it, nothing to repair
        return "not registered"
    rel = getattr(getattr(torch, "_C", None), "_cuda_releasePool", None)
    if rel is not None:
        try:
            rel(idx, pool)
        except Exception:  # noqa: BLE001
            pass
    return "unregistered"


def _reset_graph(g) -> None:
    try:
        if g is not None:
            g.reset()
    except Exception:  # noqa: BLE001
        pass


def _median(xs) -> float:
    s = sorted(float(v) for v in xs)
    return s[len(s) // 2] if s else float("nan")


def _bf16_head_apply(head, h: torch.Tensor) -> torch.Tensor:
    """The image's draft head GEMM (UnquantizedEmbeddingMethod.apply -> dispatch_unquantized_gemm)."""
    return head.quant_method.apply(head, h)


def _abs_gemm(h: torch.Tensor, w: torch.Tensor, chunk: int = QUANT_CHUNK_ROWS) -> torch.Tensor:
    """|h| @ |w|^T in fp32 (row chunks of w; bf16 GEMM of exact |.| values, x 1.01 for its output rounding)."""
    ha = h.abs()
    out = torch.empty((int(h.shape[0]), int(w.shape[0])), dtype=torch.float32, device=h.device)
    for r0 in range(0, int(w.shape[0]), chunk):
        r1 = min(int(w.shape[0]), r0 + chunk)
        out[:, r0:r1] = torch.nn.functional.linear(ha, w[r0:r1].abs()).float() * 1.01
    return out


def _row_mean(w: torch.Tensor, chunk: int = QUANT_CHUNK_ROWS) -> torch.Tensor:
    """fp32 mean of the rows of a [N, K] weight (row chunks: bounded temporaries)."""
    n = int(w.shape[0])
    acc = torch.zeros((int(w.shape[1]),), dtype=torch.float32, device=w.device)
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        acc = acc + w[r0:r1].float().sum(dim=0)
    return acc / float(max(n, 1))


def _dequant_gemm(h: torch.Tensor, q: torch.Tensor, s: torch.Tensor, chunk: int = QUANT_CHUNK_ROWS) -> torch.Tensor:
    """h @ (q * s)^T in fp32: the exact product of the e4m3 weights and their bf16 scales (row chunks)."""
    hf = h.float()
    out = torch.empty((int(h.shape[0]), int(q.shape[0])), dtype=torch.float32, device=h.device)
    for r0 in range(0, int(q.shape[0]), chunk):
        r1 = min(int(q.shape[0]), r0 + chunk)
        deq = q[r0:r1].float() * s[r0:r1].float().unsqueeze(1)
        out[:, r0:r1] = torch.nn.functional.linear(hf, deq)
    return out


class _Rows:
    """Per-part pass/fail rows of this rank (plain data: exchanged over gloo)."""

    def __init__(self) -> None:
        self.fail: list[str] = []
        self.info: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if not ok:
            self.fail.append(f"{name}{(': ' + detail) if detail else ''}")
        return bool(ok)

    @property
    def ok(self) -> bool:
        return not self.fail


class _Plan(NamedTuple):
    head: bool
    eh: bool
    eh_ms: tuple
    eh_time_ms: tuple
    head_ms: tuple
    why: tuple
    eh_gate: str = EH_GATES[0]


def _find_mtp(draft_model):
    """-> (predictor, [mtp layers], logits_processor) of a GLM5Next MTP draft model, else raises ValueError."""
    if type(draft_model).__name__ != "Glm5NextMTP" or not type(draft_model).__module__.endswith(
            "models.glm5next.common.mtp"):
        raise ValueError(f"draft model {type(draft_model).__module__}.{type(draft_model).__name__} is not "
                         "glm5next.common.mtp.Glm5NextMTP")
    pred = draft_model.model
    layers = list(getattr(pred, "_mtp_layers", None) or [])
    if not layers:
        raise ValueError("Glm5NextMultiTokenPredictor has no _mtp_layers")
    lp = getattr(pred, "logits_processor", None)
    if lp is None:
        raise ValueError("Glm5NextMultiTokenPredictor has no logits_processor")
    return pred, layers, lp


def _head_envelope(pred, layers, lp, world) -> list[str]:
    why = []
    heads = {id(ly.shared_head.head) for ly in layers}
    if len(heads) != 1:
        why.append(f"{len(heads)} distinct draft heads (expected the one shared lm_head)")
    head = layers[0].shared_head.head
    w = getattr(head, "weight", None)
    if w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or not w.is_cuda:
        why.append("draft lm_head weight is not a 2-D bf16 CUDA tensor")
    qm = getattr(head, "quant_method", None)
    if type(qm).__name__ != "UnquantizedEmbeddingMethod":
        why.append(f"draft lm_head quant_method is {type(qm).__name__} (expected UnquantizedEmbeddingMethod)")
    if getattr(head, "bias", None) is not None:
        why.append("draft lm_head has a bias")
    if int(getattr(head, "tp_size", -1)) != world.size:
        why.append(f"draft lm_head tp_size {getattr(head, 'tp_size', None)} != TP size {world.size}")
    if type(lp).__name__ != "LogitsProcessor" or not type(lp).__module__.endswith("layers.logits_processor"):
        why.append(f"logits processor is {type(lp).__module__}.{type(lp).__name__} (an OOT replacement?)")
    if getattr(lp, "logits_as_input", False) or getattr(lp, "soft_cap", None) is not None:
        why.append("logits processor has logits_as_input / soft_cap")
    if getattr(lp, "head_dtype", None) not in (None, torch.bfloat16):
        why.append(f"head_dtype {getattr(lp, 'head_dtype', None)} (FP8 path needs the bf16 head path)")
    if not getattr(lp, "use_all_gather", False):
        why.append("logits processor gathers to rank 0 only (use_all_gather False)")
    try:
        MU8 = _marlin_fp8()
        if not MU8.is_fp8_marlin_supported():
            why.append("is_fp8_marlin_supported() is False")
        for op in ("marlin_gemm", "gptq_marlin_repack"):
            if not hasattr(torch.ops._C, op):
                why.append(f"torch.ops._C.{op} missing")
        if not hasattr(torch.ops.vllm, "marlin_gemm"):      # v0.32.0: apply_fp8_marlin_linear -> torch.ops.vllm.marlin_gemm
            why.append("torch.ops.vllm.marlin_gemm missing")
    except Exception as e:  # noqa: BLE001
        why.append(f"marlin_utils_fp8 import failed: {type(e).__name__}: {e}")
    if not hasattr(torch, "float8_e4m3fn"):
        why.append("torch has no float8_e4m3fn")
    return why


def _embed_refusal(pred, world) -> str | None:
    """The eh one-shot must not become the FIRST NCCL op of a draft replay: in the served draft graphs the input
    embedding's all-reduce precedes eh_proj and absorbs NCCL's per-replay host-callback round trip (0.5-0.9 ms, the
    E2 profile), which E5's leading one-shot stands for. Patch 032's replicated embedding (disable_tp, tp_size 1, no
    all-reduce) removes that op on purpose; with it the split would pay the round trip (032's .sh refuses the
    combination; this is the runtime guard)."""
    emb = getattr(pred, "embed_tokens", None)
    tp = getattr(emb, "tp_size", None)
    try:
        tp_ok = int(tp) == int(world.size)
    except Exception:  # noqa: BLE001
        tp_ok = False
    if emb is not None and tp_ok and not bool(getattr(emb, "disable_tp", False)):
        return None
    return (f"the draft's embed_tokens ({type(emb).__name__}, tp_size {tp}) does not all-reduce over the TP group "
            "(patch 032's replicated embedding?): the eh one-shot would become the first NCCL op of each draft replay "
            f"and pay the host-callback round trip; use {ENV_PARTS}=head")


def _eh_envelope(pred, layers, world, oneshot) -> list[str]:
    why = []
    r = _embed_refusal(pred, world)
    if r:
        why.append(r)
    if oneshot is None:
        why.append("patch 012's one-shot all-reduce is not active on the TP group (VLLM_GB10_ONESHOT_AR=1 needed)")
    else:
        for a in ("should_use", "max_elems", "rank", "world_size"):
            if not hasattr(oneshot, a):
                why.append(f"one-shot object lacks {a}")
        if not why and (int(oneshot.rank) != world.rank or int(oneshot.world_size) != world.size):
            why.append(f"one-shot rank {oneshot.rank}/{oneshot.world_size} != TP rank {world.rank}/{world.size}")
    for ly in layers:
        eh = getattr(ly, "eh_proj", None)
        w = getattr(eh, "weight", None)
        if type(eh) is not torch.nn.Linear or w is None or w.dim() != 2 or w.dtype != torch.bfloat16 \
                or not w.is_cuda or getattr(eh, "bias", None) is not None:
            why.append("eh_proj is not a bias-free bf16 CUDA nn.Linear")
            continue
        H, K = int(w.shape[0]), int(w.shape[1])
        if K != 2 * H or H % world.size:
            why.append(f"eh_proj shape [{H}, {K}] (expected [H, 2H], H divisible by {world.size})")
        elif oneshot is not None and hasattr(oneshot, "max_elems") and int(oneshot.max_elems) // H < 1:
            why.append(f"one-shot limit {oneshot.max_elems} elements < one row of {H}")
    return why


def _common_envelope(world, parts) -> list[str]:
    why = []
    if world.size < 2:
        why.append(f"TP size {world.size} < 2")
    if getattr(world, "pp_size", 1) != 1:
        why.append(f"PP size {world.pp_size} != 1")
    if str(os.environ.get(ENV_BATCH_INVARIANT, "0")).strip() not in ("", "0"):
        why.append(f"{ENV_BATCH_INVARIANT} is set (the draft path would become batch-shape dependent)")
    return why


def fingerprint(parts, world_size: int, shapes: dict, eh_gate: str = EH_GATES[0]) -> str:
    """What must be identical on every rank for the planned collectives to line up (module revision, gate, parts,
    eh speed-gate mode (rev 2: it decides whether the capture exchange and the graph replays happen), TP size,
    shapes, batch-invariant mode). A per-rank one-shot difference is not part of it: it only turns the eh part off
    (every rank sees the preflight rows), the head part stays independent."""
    items = ["p029", f"rev={REVISION}", f"module={MODULE_SHA16}", f"gate={os.environ.get(ENV, '0').strip()}",
             f"parts={','.join(parts)}", f"ehgate={eh_gate}", f"tp={world_size}",
             "shapes=" + ",".join(f"{k}:{v}" for k, v in sorted(shapes.items())),
             f"bi={os.environ.get(ENV_BATCH_INVARIANT, '0').strip()}"]
    return hashlib.sha256("|".join(items).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------
# Local (no collective) head checks H1-H5
# ---------------------------------------------------------------------------------------------------------------------
def _head_local_checks(fh: _Fp8Head, head, norm_w, rows: _Rows) -> None:
    dev = head.weight.device
    W = head.weight
    h = _synthetic(HEAD_TEST_RANDOM, fh.size_k, SEED + 1, dev, norm_w)
    lb = _bf16_head_apply(head, h)
    lf = fh.apply(fh, h)
    lf2 = fh.apply(fh, h)
    rows.check("H2 shape/dtype", tuple(lf.shape) == (HEAD_TEST_RANDOM, fh.size_n) and lf.dtype == torch.bfloat16,
               f"{tuple(lf.shape)} {lf.dtype}")
    rows.check("H2 finite", _finite(lf) and _finite(lb))
    rows.check("H4 deterministic", _mismatches(lf, lf2) == 0, f"{_mismatches(lf, lf2)} elements differ")
    ref = _dequant_gemm(h, fh.q, fh.s)
    krel = _rel(lf, ref)
    rows.check("H1 kernel vs exact dequant", krel <= HEAD_KERNEL_REL_MAX, f"relRMSE {krel:.4g}")
    rel = _rel(lf, lb)
    diff = (lf.float() - lb.float()).abs()
    maxabs = float(diff.amax().item())
    l1 = h.float().abs().sum(dim=1, keepdim=True)
    absw = _abs_gemm(h, W)
    bound = (absw * (2.0 ** -4 + 2.0 ** -11) + l1 * fh.s.float().unsqueeze(0) * (2.0 ** -6)
             + (lf.float().abs() + lb.float().abs()) * (2.0 ** -8) + 1e-6)
    ratio = float((diff / bound).amax().item())
    rows.check("H2 relRMSE vs BF16", rel <= HEAD_REL_MAX, f"{rel:.4g}")
    rows.check("H2 max|diff| within the FP8 bound", ratio <= BOUND_RATIO_MAX, f"ratio {ratio:.3g}")
    t1 = float((lf.float().argmax(dim=1) == lb.float().argmax(dim=1)).float().mean().item())
    rows.check("H3 top-1 random rows", t1 >= HEAD_TOP1_RANDOM_MIN, f"{t1:.3f}")
    # peaked rows: lift a chosen local vocab row v to max(lb) + margin along d_v = w_v - the shard's mean row; a
    # candidate counts only if the BF16 head then puts v on top by >= margin / 2 (>= 4 max|diff|: no correct FP8 head
    # can flip it). Centred because a trained unembedding is anisotropic (its rows share a mean direction): a lift
    # along w_v itself raises every row by its share of that direction and v seldom leads (19,360 x 4,096 shard
    # simulation, mean row cosine 0.2 / 0.3: 13 / 1 of 64 built -> a false refusal; centred 63 / 63). Along d_v the
    # shared part c * (mean . d_v) lifts every logit equally, so the argmax and the leads depend on d_v only.
    margin = max(PEAK_MARGIN_MIN, 8.0 * maxabs)
    npk = min(HEAD_PEAK_CANDIDATES, HEAD_TEST_RANDOM)
    vs = [int((i * 7919 + 13) % fh.size_n) for i in range(npk)]
    hp = h[:npk].float().clone()
    lbf = lb[:npk].float()
    wmean = _row_mean(W)
    for i, v in enumerate(vs):
        wv = W[v].float()
        dv = wv - wmean
        nw2 = float((wv * wv).sum().item())
        nn2 = float((dv * dv).sum().item())
        if nw2 <= 0.0 or nn2 <= 1e-6 * nw2:       # a zero row / a row equal to the mean row: cannot be peaked
            continue
        target = float(lbf[i].amax().item()) + margin
        hp[i] = hp[i] + dv * ((target - float(lbf[i, v].item())) / nn2)
    hp = hp.to(torch.bfloat16)
    lbp_all = _bf16_head_apply(head, hp).float()
    lbp = lbp_all.argmax(dim=1)
    lfp = fh.apply(fh, hp).float().argmax(dim=1)
    valid = []
    for i, v in enumerate(vs):
        if int(lbp[i].item()) != v:
            continue
        row = lbp_all[i].clone()
        top = float(row[v].item())
        row[v] = float("-inf")
        if top - float(row.amax().item()) >= 0.5 * margin:
            valid.append(i)
    agree = sum(1 for i in valid if int(lfp[i].item()) == vs[i])
    rows.check("H3 peaked rows built", len(valid) >= HEAD_TEST_PEAKED, f"{len(valid)}/{npk}")
    rows.check("H3 top-1 peaked rows", agree == len(valid), f"{agree}/{len(valid)}")
    h1 = h[:1].contiguous()
    t_bf = _TIMER(lambda: _bf16_head_apply(head, h1), TIME_WARMUP, TIME_ITERS)
    t_fp = _TIMER(lambda: fh.apply(fh, h1), TIME_WARMUP, TIME_ITERS)
    rows.check("H5 FP8 faster at M=1", t_fp <= PERF_RATIO_MAX * t_bf, f"{t_fp:.3f} vs {t_bf:.3f} ms")
    rows.info.update({"kernel_rel": round(krel, 5), "rel": round(rel, 5), "maxabs": round(maxabs, 4),
                      "bound_ratio": round(ratio, 4), "top1_random": round(t1, 3),
                      "top1_peaked": f"{agree}/{len(valid)}", "ms_fp8": round(t_fp, 4), "ms_bf16": round(t_bf, 4),
                      "mb": round(fh.bytes / 1e6, 1)})


# ---------------------------------------------------------------------------------------------------------------------
# Collective checks (fixed schedule: every rank issues exactly the planned collectives, in order, with planned shapes)
# ---------------------------------------------------------------------------------------------------------------------
def _value_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose VALUES differ (-0.0 == +0.0; NaN never equal). The one-shot's fp32 sum turns a -0.0 block entry
    into +0.0 (0.0 + -0.0), so the exact-gather checks compare values, with finiteness checked separately."""
    if tuple(a.shape) != tuple(b.shape):
        return max(a.numel(), b.numel(), 1)
    return int((a.float() != b.float()).sum().item())


class _OneshotGuard:
    """Self-test only: stands in for patch 012's one-shot inside the _EhSplit under test, so that the self-test issues
    EXACTLY the planned one-shot calls with the planned [m, H] bf16 shape, whatever the local code does: a call over
    budget raises locally without communicating; a malformed input is replaced by zeros (the call is still issued)
    and raises; flush() issues the calls a step did not reach (a local exception) with zeros.
    Rev 2, CUDA graphs: a call while the stream is capturing communicates nothing (it is recorded into the graph): it
    is shape-checked (a malformed input raises = a failed capture, reported in the capture exchange) and not counted;
    replayed(n) counts the n one-shots a successful graph replay issued against the budget."""

    def __init__(self, inner, H: int, dev) -> None:
        self.inner, self.H, self.dev = inner, H, dev
        self.max_elems, self.rank, self.world_size = inner.max_elems, inner.rank, inner.world_size
        self.m = 1
        self.budget = 0
        self.issued = 0
        self.captured = 0

    def should_use(self, x) -> bool:
        return bool(self.inner.should_use(x))

    def expect(self, m: int, count: int) -> None:
        self.flush()
        self.m, self.budget = int(m), int(count)

    def _zeros(self) -> torch.Tensor:
        return torch.zeros((self.m, self.H), dtype=torch.bfloat16, device=self.dev)

    def _planned(self, x) -> bool:
        return bool(isinstance(x, torch.Tensor) and tuple(x.shape) == (self.m, self.H) and x.dtype == torch.bfloat16
                    and x.is_contiguous() and x.device == self.dev)

    def __call__(self, x):
        if _capturing():
            if not self._planned(x):
                raise RuntimeError(f"captured one-shot input {tuple(getattr(x, 'shape', ()))} is not the planned "
                                   f"[{self.m}, {self.H}] bf16 contiguous tensor")
            self.captured += 1
            return self.inner(x)
        if self.budget <= 0:
            raise RuntimeError("unplanned one-shot call in the boot self-test (not issued)")
        self.budget -= 1
        self.issued += 1
        if not self._planned(x):
            self.inner(self._zeros())
            raise RuntimeError(f"self-test one-shot input {tuple(getattr(x, 'shape', ()))} is not the planned "
                               f"[{self.m}, {self.H}] bf16 contiguous tensor")
        return self.inner(x)

    def replayed(self, n: int) -> None:
        n = int(n)
        if n > self.budget:
            raise RuntimeError(f"graph replay of {n} one-shots exceeds the planned budget {self.budget}")
        self.budget -= n
        self.issued += n

    def flush(self) -> None:
        while self.budget > 0:
            self.budget -= 1
            self.issued += 1
            self.inner(self._zeros())


def _eh_scale(ly):
    try:
        return torch.cat([ly.enorm.weight.float(), ly.hnorm.weight.float()])
    except Exception:  # noqa: BLE001
        return None


def _eh_collective_checks(plan: _Plan, layers, splits, oneshot, H: int, K: int, dev, rows: _Rows,
                          world) -> list[str]:
    """E1-E4 on every MTP layer through the PRODUCTION call _EhSplit.__call__ (block -> pad -> one-shot), then E5
    (_eh_speed_checks). Returns the shas of the assembled outputs (cross-rank identity, compared in the verdict
    exchange)."""
    shas = []
    guard = _OneshotGuard(oneshot, H, dev)
    items = [(li, ly, splits[li] if (splits is not None and li < len(splits)) else None)
             for li, ly in enumerate(layers or [])]
    saved = [sp.oneshot if sp is not None else None for _, _, sp in items]
    for _, _, sp in items:
        if sp is not None:
            sp.oneshot = guard
    try:
        scales = [_eh_scale(ly) for _, ly, _ in items]
        for li, ly, sp in items:
            scale = scales[li]
            for m in plan.eh_ms:
                guard.expect(m, 1)
                sha = None
                try:
                    if sp is None:
                        raise RuntimeError("no eh split on this rank")
                    x = _synthetic(m, K, SEED + 100 + m, dev, scale)
                    out = sp(x)                                   # production path (one planned one-shot)
                    sha = _sha(out)
                    y_loc = sp.block(x)
                    full = ly.eh_proj(x)                          # the image's replicated nn.Linear
                    own = out[:, sp.off:sp.off + sp.n]
                    mm = _value_mismatches(own, y_loc)
                    rows.check(f"E1 eh M={m} own block exact", mm == 0, f"{mm} elements")
                    rows.check(f"E4 eh M={m} finite", _finite(out) and _finite(full))
                    rows.check(f"E4 eh M={m} one-shot engaged", bool(oneshot.should_use(sp.pad(y_loc))))
                    d = (out.float() - full.float()).abs()
                    bnd = ((out.float().abs() + full.float().abs()) * (2.0 ** -8)
                           + _abs_gemm(x, ly.eh_proj.weight) * (2.0 ** -10) + 1e-6)
                    ratio = float((d / bnd).amax().item())
                    rows.check(f"E3 eh M={m} vs replicated", ratio <= BOUND_RATIO_MAX,
                               f"ratio {ratio:.3g}, max|diff| {float(d.amax().item()):.3g}")
                    rows.info.setdefault("eh_ratio", []).append(round(ratio, 4))
                    rows.info.setdefault("eh_ulpdiff", []).append(_value_mismatches(out, full))
                except Exception as e:  # noqa: BLE001 - rank-local: the guard keeps the schedule
                    rows.check(f"E0 eh M={m}", False, f"{type(e).__name__}: {str(e)[:200]}")
                guard.flush()
                shas.append(sha or "none")
        if plan.eh_time_ms:
            _eh_speed_checks(plan, items, guard, H, K, dev, rows, world, scales)
    finally:
        for (_, _, sp), s in zip(items, saved):
            if sp is not None:
                sp.oneshot = s
    rows.info["oneshot_calls"] = guard.issued
    rows.info["oneshot_captured"] = guard.captured
    return shas


def _eh_speed_checks(plan: _Plan, items, guard: _OneshotGuard, H: int, K: int, dev, rows: _Rows, world,
                     scales) -> None:
    """E5 on a side stream. Planned collectives, identical on every rank whatever fails locally: (a) TIME_WARMUP +
    TIME_ITERS eager one-shots per (layer, M); eh gate 'graph': (b) ONE capture exchange (gloo), then - only if
    EVERY rank captured every graph - (EH_GRAPH_WARMUP + EH_GRAPH_ITERS) x (EH_GRAPH_CALLS + 2) replayed one-shots per
    (layer, M). The graphs are reset and the side stream joined before returning."""
    side, why_local, ctx, entered = None, "", None, False
    try:
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream())
        ctx = torch.cuda.stream(side)
        ctx.__enter__()
        entered = True
    except Exception as e:  # noqa: BLE001 - the eager part then runs on the current stream, the capture is refused
        side, why_local = None, f"side stream: {type(e).__name__}: {str(e)[:200]}"
    graphs: list = []
    try:
        _eh_eager_timing(plan, items, guard, K, dev, rows, scales)
        if plan.eh_gate == "graph":
            if not why_local:
                graphs, why_local = _eh_capture(plan, items, guard, H, K, dev, side, scales)
            ok_all, why_all = _capture_verdict(world, why_local)
            if ok_all:
                _eh_graph_timing(graphs, guard, rows)
            else:
                for m in plan.eh_time_ms:
                    rows.check(f"E5 eh M={m} CUDA-graph timing", False, why_all)
    finally:
        for rec in graphs:
            _reset_graph(rec[2])
            _reset_graph(rec[3])
        graphs = []
        if entered:
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        try:
            if side is not None:
                torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass


def _eh_eager_timing(plan: _Plan, items, guard: _OneshotGuard, K: int, dev, rows: _Rows, scales) -> None:
    """E5 (a): eager split + one-shot vs the replicated eh_proj per (layer, M) - INFORMATION (launch-bound; rev 1
    gated on it and refused a working split at revision 1). A raising split path still fails the row."""
    for li, ly, sp in items:
        for m in plan.eh_time_ms:
            guard.expect(m, TIME_WARMUP + TIME_ITERS)
            try:
                if sp is None:
                    raise RuntimeError("no eh split on this rank")
                x = _synthetic(m, K, SEED + 200 + m, dev, scales[li])
                t_sp = _TIMER(lambda: sp(x), TIME_WARMUP, TIME_ITERS)    # collective: planned call count
                guard.flush()
                t_full = _TIMER(lambda: ly.eh_proj(x), TIME_WARMUP, TIME_ITERS)
                rows.info.setdefault("eh_eager_ms", []).append([m, round(t_sp, 4), round(t_full, 4)])
            except Exception as e:  # noqa: BLE001
                rows.check(f"E5 eh M={m} eager run", False, f"{type(e).__name__}: {str(e)[:200]}")
            guard.flush()


def _eh_capture(plan: _Plan, items, guard: _OneshotGuard, H: int, K: int, dev, side, scales):
    """E5 (b), local: per (layer, M) G_split = [one-shot] + N x [split path] and G_full = [one-shot] + N x [eh_proj],
    captured on the side stream (static inputs allocated before the capture). Each graph must have recorded exactly
    the one-shots its replays are budgeted for (N + 1 / 1), else this rank reports a failed capture (no rank replays).
    -> (records, '') or ([], why); after a failure the caching allocator is probed (vLLM's next empty_cache())."""
    graphs = []
    try:
        for li, ly, sp in items:
            if sp is None:
                raise RuntimeError("no eh split on this rank")
            for m in plan.eh_time_ms:
                x = _synthetic(m, K, SEED + 400 + m, dev, scales[li])
                lead = torch.zeros((m, H), dtype=torch.bfloat16, device=dev)
                guard.expect(m, 0)
                rec = [li, m, None, None, x, lead]
                graphs.append(rec)
                c0 = guard.captured
                rec[2] = _capture([lambda: guard(lead)] + [lambda: sp(x)] * EH_GRAPH_CALLS, side, dev)
                c1 = guard.captured
                rec[3] = _capture([lambda: guard(lead)] + [lambda: ly.eh_proj(x)] * EH_GRAPH_CALLS, side, dev)
                if (c1 - c0, guard.captured - c1) != (EH_GRAPH_CALLS + 1, 1):
                    raise RuntimeError(f"the graphs recorded {c1 - c0} / {guard.captured - c1} one-shots, planned "
                                       f"{EH_GRAPH_CALLS + 1} / 1")
        return graphs, ""
    except Exception as e:  # noqa: BLE001
        for rec in graphs:
            _reset_graph(rec[2])
            _reset_graph(rec[3])
        why = f"{type(e).__name__}: {str(e)[:200]}"
        try:
            torch.cuda.empty_cache()
        except Exception as e2:  # noqa: BLE001 - reported: vLLM's own next empty_cache() would fail the same way
            why += f"; caching allocator not restored after the failed capture ({type(e2).__name__}: {str(e2)[:120]})"
        return [], why


def _capture_verdict(world, why_local: str) -> tuple[bool, str]:
    """Gloo exchange of the capture results: replay only if EVERY rank captured every graph (a pure function of the
    exchanged rows: every rank takes the same decision; a rank that failed never leaves the others waiting)."""
    got = _exchange(world, {"rank": world.rank, "cap_ok": not why_local, "cap_why": _safe(why_local)[:300]},
                    "graph-capture", "the eh part stays off (its E5 rows fail)")
    if got is None:
        return False, "graph-capture exchange failed on this rank"
    bad = sorted((r for r in got if not r.get("cap_ok")), key=lambda r: int(r.get("rank", -1)))
    if bad:
        return False, f"CUDA-graph capture failed on rank(s) {[r.get('rank') for r in bad]}: {bad[0].get('cap_why')}"
    return True, ""


def _eh_graph_timing(graphs, guard: _OneshotGuard, rows: _Rows) -> None:
    """E5 (b), collective: replay G_split / G_full in a fixed interleaved order, compare the medians."""
    n_split, n_full = EH_GRAPH_CALLS + 1, 1
    for _li, m, g_s, g_f, _x, _lead in graphs:
        guard.expect(m, (EH_GRAPH_WARMUP + EH_GRAPH_ITERS) * (n_split + n_full))
        try:
            ts, tf = _replay_timed(((g_s, n_split), (g_f, n_full)), guard)
            t_s, t_f = _median(ts), _median(tf)
            ok = (math.isfinite(t_s) and math.isfinite(t_f) and t_s > 0.0 and t_f > 0.0
                  and t_s <= EH_GRAPH_RATIO_MAX * t_f)
            rows.check(f"E5 eh M={m} CUDA graphs: split <= {EH_GRAPH_RATIO_MAX} x replicated", ok,
                       f"{t_s:.3f} vs {t_f:.3f} ms per graph (one-shot + {EH_GRAPH_CALLS} calls)")
            rows.info.setdefault("eh_graph_ms", []).append(
                [m, round(t_s, 4), round(t_f, 4), round((t_f - t_s) / EH_GRAPH_CALLS, 4)])
        except Exception as e:  # noqa: BLE001 - rank-local: flush() issues the one-shots of the replays not launched
            rows.check(f"E5 eh M={m} CUDA-graph timing", False, f"{type(e).__name__}: {str(e)[:200]}")
        guard.flush()


def _replay_timed(entries, guard: _OneshotGuard) -> list:
    """entries: ((graph, one-shots per replay), ...) -> per entry the CUDA-event times (ms) of its EH_GRAPH_ITERS timed
    replays. Order (identical on every rank): for each round, every entry once; EH_GRAPH_WARMUP untimed rounds first.
    A replay counts against the guard's plan only once it launched."""
    evs = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(EH_GRAPH_ITERS)] for _ in entries]
    for i in range(EH_GRAPH_WARMUP + EH_GRAPH_ITERS):
        j = i - EH_GRAPH_WARMUP
        for k, (g, n_os) in enumerate(entries):
            if j >= 0:
                evs[k][j][0].record()
            g.replay()
            guard.replayed(n_os)
            if j >= 0:
                evs[k][j][1].record()
    torch.cuda.synchronize()
    return [[a.elapsed_time(b) for a, b in ev] for ev in evs]


class _GuardedHead:
    """G-checks only: a stand-in lm_head whose apply() never raises (zeros of the planned [m, n] bf16 shape instead,
    failing the row), so the LogitsProcessor's all-gather is always issued with the planned shape."""

    def __init__(self, apply_fn, n: int, rows: _Rows, tp_size: int, name: str) -> None:
        self.apply_fn, self.n, self.rows, self.tp_size, self.name = apply_fn, n, rows, tp_size, name
        self.quant_method = self

    def apply(self, layer, x, bias=None):
        try:
            out = self.apply_fn(x)
            if tuple(out.shape) != (int(x.shape[0]), self.n) or out.dtype != torch.bfloat16:
                raise RuntimeError(f"output {tuple(out.shape)} {out.dtype}")
            return out
        except Exception as e:  # noqa: BLE001
            self.rows.check(f"G0 {self.name} local", False, f"{type(e).__name__}: {str(e)[:200]}")
            return torch.zeros((int(x.shape[0]), self.n), dtype=torch.bfloat16, device=x.device)


def _head_collective_checks(plan: _Plan, fh, head, lp, norm_w, world, rows: _Rows) -> list[str]:
    """G1-G4 through the REAL LogitsProcessor: per M, one FP8 and one BF16 all-gather, always issued."""
    shas = []
    n, k = int(head.weight.shape[0]), int(head.weight.shape[1])
    dev = head.weight.device

    def fp8_apply(x):
        if fh is None:
            raise RuntimeError("no FP8 head on this rank")
        return fh.apply(fh, x)

    gf = _GuardedHead(fp8_apply, n, rows, int(head.tp_size), "head fp8")
    gb = _GuardedHead(lambda x: _bf16_head_apply(head, x), n, rows, int(head.tp_size), "head bf16")
    agree = total = 0
    gmax = 0.0
    for m in plan.head_ms:
        try:
            h = _synthetic(m, k, SEED + 300 + m, dev, norm_w)
        except Exception as e:  # noqa: BLE001
            rows.check(f"G0 head M={m} input", False, f"{type(e).__name__}: {e}")
            h = torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
        a = lp(gf, h)                                         # FP8 head through the real LogitsProcessor (all-gather)
        b = lp(gb, h)                                         # the image BF16 head, same processor (all-gather)
        try:
            shas.append(_sha(a))
            ok_shape = (a is not None and b is not None and tuple(a.shape) == tuple(b.shape)
                        and a.dim() == 2 and int(a.shape[0]) == m and a.dtype == torch.bfloat16)
            rows.check(f"G1 head M={m} shape", ok_shape, f"{None if a is None else tuple(a.shape)}")
            if not ok_shape:
                continue
            lo = world.rank * n
            hi = min(lo + n, int(a.shape[1]))
            if hi > lo:
                loc = fp8_apply(h)[:, :hi - lo]
                mm = _value_mismatches(a[:, lo:hi], loc)
                rows.check(f"G2 head M={m} own slice of the gathered logits exact", mm == 0, f"{mm} elements")
            rows.check(f"G3 head M={m} finite", _finite(a))
            grel = _rel(a, b)
            rows.check(f"G4 head M={m} gathered FP8 vs BF16 relRMSE", grel <= HEAD_REL_MAX, f"{grel:.4g}")
            agree += int((a.float().argmax(dim=1) == b.float().argmax(dim=1)).sum().item())
            total += m
            gmax = max(gmax, float((a.float() - b.float()).abs().amax().item()))
        except Exception as e:  # noqa: BLE001 - local only (both all-gathers of this M were issued)
            rows.check(f"G0 head M={m} checks", False, f"{type(e).__name__}: {str(e)[:200]}")
    rows.info.update({"top1_global": f"{agree}/{total}", "maxabs_global": round(gmax, 4)})
    return shas


# ---------------------------------------------------------------------------------------------------------------------
# prepare(): the boot entry point (MTPSpeculator.load_draft_model, after load_eagle_model)
# ---------------------------------------------------------------------------------------------------------------------
def _exchange(world, obj, what: str, consequence: str = "draft-path compression stays INACTIVE here"):
    try:
        world.barrier_bounded(SELFTEST_BARRIER_S)
        out = world.all_gather_object(obj)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s boot %s exchange failed on this rank (%s: %s); %s", TAG, what, type(e).__name__,
                       _safe(str(e)[:300]), consequence)
        return None
    if not isinstance(out, list) or len(out) != world.size or any(not isinstance(r, dict) for r in out):
        logger.warning("%s boot %s exchange returned malformed rows; %s", TAG, what, consequence)
        return None
    return out


def _make_plan(pre_rows, parts) -> _Plan:
    why = []
    ranks = sorted(int(r.get("rank", -1)) for r in pre_rows)
    if ranks != list(range(len(pre_rows))):
        return _Plan(False, False, (), (), (), (f"preflight ranks {ranks}",))
    fps = {r.get("fp") for r in pre_rows}
    if len(fps) != 1:
        return _Plan(False, False, (), (), (), (f"rank fingerprint mismatch {sorted(str(f) for f in fps)}",))
    head = "head" in parts and all(r.get("head_ok") for r in pre_rows)
    eh = "eh" in parts and all(r.get("eh_ok") for r in pre_rows)
    if "head" in parts and not head:
        bad = [(r["rank"], r.get("head_why")) for r in pre_rows if not r.get("head_ok")]
        why.append(f"head refused on rank(s) {[b[0] for b in bad]}: {bad[0][1]}")
    if "eh" in parts and not eh:
        bad = [(r["rank"], r.get("eh_why")) for r in pre_rows if not r.get("eh_ok")]
        why.append(f"eh refused on rank(s) {[b[0] for b in bad]}: {bad[0][1]}")
    mt = {int(r.get("max_tokens", 0)) for r in pre_rows}
    if eh and len(mt) != 1:
        eh = False
        why.append(f"one-shot limits differ across ranks {sorted(mt)}")
    gates = {str(r.get("eh_gate", "")) for r in pre_rows}
    eh_gate = next(iter(gates)) if len(gates) == 1 else ""
    if eh and eh_gate not in EH_GATES:
        eh = False
        why.append(f"eh speed-gate modes differ across ranks or are invalid {sorted(gates)}")
    max_tokens = min(mt) if mt else 0
    eh_ms = tuple(m for m in EH_TEST_M if m <= max_tokens) if eh else ()
    eh_time = tuple(m for m in EH_TIME_M if m <= max_tokens) if eh else ()
    if eh and not eh_ms:
        eh = False
        why.append("no eh test size fits the one-shot limit")
    return _Plan(head, eh, eh_ms, eh_time, HEAD_GATHER_M if head else (), tuple(why),
                 eh_gate if eh_gate in EH_GATES else EH_GATES[0])


def _decide(plan: _Plan, ver_rows) -> tuple[bool, bool, list[str]]:
    why = list(plan.why)
    head = plan.head and all(r.get("head_ok") for r in ver_rows)
    eh = plan.eh and all(r.get("eh_ok") for r in ver_rows)
    if plan.head and not head:
        bad = [r for r in ver_rows if not r.get("head_ok")]
        why.append(f"head self-test failed on rank(s) {[r['rank'] for r in bad]}: {bad[0].get('head_fail')}")
    if plan.eh and not eh:
        bad = [r for r in ver_rows if not r.get("eh_ok")]
        why.append(f"eh self-test failed on rank(s) {[r['rank'] for r in bad]}: {bad[0].get('eh_fail')}")
    if head and len({tuple(r.get("head_sha") or ()) for r in ver_rows}) != 1:
        head = False
        why.append("gathered FP8 draft logits differ across ranks")
    if eh and len({tuple(r.get("eh_sha") or ()) for r in ver_rows}) != 1:
        eh = False
        why.append("assembled eh_proj outputs differ across ranks")
    return head, eh, why


def prepare(draft_model, vllm_config=None) -> dict:
    """Build + self-test + (de)activate. NEVER raises (fail closed). Returns a summary dict (logged; tests read it)."""
    summary = {"head": False, "eh": False, "why": []}
    try:
        with torch.no_grad():
            summary = _prepare(draft_model)
    except Exception as e:  # noqa: BLE001 - last resort: the image path
        try:
            pred = draft_model.model
            for ly in list(getattr(pred, "_mtp_layers", []) or []):
                ly._p029_eh = None
            pred._p029_head = None
        except Exception:  # noqa: BLE001
            pass
        summary = {"head": False, "eh": False, "why": [f"unexpected {type(e).__name__}: {str(e)[:300]}"]}
        logger.warning("%s draft-path compression INACTIVE (fail closed): unexpected %s: %s (module rev %d)", TAG,
                       type(e).__name__, _safe(str(e)[:300]), REVISION)
    return summary


class _Local:
    """This rank's local phase (no collectives): envelopes, the FP8 build + H1-H5, the eh split build."""

    def __init__(self) -> None:
        self.parts: tuple = ()
        self.eh_gate: str = ""
        self.pred = self.layers = self.lp = self.oneshot = self.fh = self.splits = self.dev = None
        self.head_why: list[str] = []
        self.eh_why: list[str] = []
        self.shapes: dict = {}
        self.H = self.K = self.max_tokens = 0
        self.hrows, self.erows = _Rows(), _Rows()


def _local_phase(draft_model, world) -> _Local:
    loc = _Local()
    rank, size = world.rank, world.size
    why_local = []
    try:
        loc.parts = parse_parts(os.environ.get(ENV_PARTS))
    except ValueError as e:
        why_local.append(str(e))
    try:
        loc.pred, loc.layers, loc.lp = _find_mtp(draft_model)
    except Exception as e:  # noqa: BLE001
        why_local.append(f"{type(e).__name__}: {e}")
    try:
        loc.oneshot = world.oneshot()
    except Exception as e:  # noqa: BLE001
        why_local.append(f"one-shot lookup: {type(e).__name__}: {e}")
    common = _common_envelope(world, loc.parts)
    loc.head_why = list(why_local) + list(common)
    loc.eh_why = list(why_local) + list(common)
    try:
        loc.eh_gate = parse_eh_gate(os.environ.get(ENV_EH_GATE))
    except ValueError as e:
        loc.eh_gate = "invalid"
        loc.eh_why.append(str(e))
    layers = loc.layers
    if layers:
        loc.shapes["layers"] = len(layers)
        try:
            loc.head_why += _head_envelope(loc.pred, layers, loc.lp, world)
            hw = layers[0].shared_head.head.weight
            loc.shapes["head"] = f"{int(hw.shape[0])}x{int(hw.shape[1])}"
            loc.dev = hw.device
        except Exception as e:  # noqa: BLE001
            loc.head_why.append(f"head envelope: {type(e).__name__}: {e}")
        try:
            loc.eh_why += _eh_envelope(loc.pred, layers, world, loc.oneshot)
            ew = layers[0].eh_proj.weight
            loc.H, loc.K = int(ew.shape[0]), int(ew.shape[1])
            loc.shapes["eh"] = f"{loc.H}x{loc.K}"
            loc.dev = loc.dev if loc.dev is not None else ew.device
        except Exception as e:  # noqa: BLE001
            loc.eh_why.append(f"eh envelope: {type(e).__name__}: {e}")
    if "head" not in loc.parts:
        loc.head_why = ["not selected"] + loc.head_why
    if "eh" not in loc.parts:
        loc.eh_why = ["not selected"] + loc.eh_why
    if loc.oneshot is not None and loc.H and hasattr(loc.oneshot, "max_elems"):
        loc.max_tokens = int(loc.oneshot.max_elems) // loc.H
    if not loc.head_why:
        try:
            head = layers[0].shared_head.head
            loc.fh = _Fp8Head(head)
            norm_w = getattr(getattr(layers[0].shared_head, "norm", None), "weight", None)
            _head_local_checks(loc.fh, head, norm_w, loc.hrows)
        except Exception as e:  # noqa: BLE001
            loc.hrows.check("H0 build/local checks", False, f"{type(e).__name__}: {str(e)[:300]}")
        if not loc.hrows.ok:
            loc.head_why.append("; ".join(loc.hrows.fail[:4]))
    if not loc.eh_why:
        try:
            loc.splits = [_EhSplit(ly.eh_proj, rank, size, loc.oneshot) for ly in layers]
        except Exception as e:  # noqa: BLE001
            loc.eh_why.append(f"eh build: {type(e).__name__}: {e}")
            loc.splits = None
    return loc


def _prepare(draft_model) -> dict:
    world = _make_world()
    rank, size = world.rank, world.size
    try:
        loc = _local_phase(draft_model, world)
    except Exception as e:  # noqa: BLE001 - still take part in the preflight (as a refusal): no rank is left waiting
        loc = _Local()
        msg = f"local phase: unexpected {type(e).__name__}: {str(e)[:200]}"
        loc.head_why, loc.eh_why = [msg], [msg]
        try:
            loc.pred = draft_model.model
            loc.layers = list(getattr(loc.pred, "_mtp_layers", []) or [])
        except Exception:  # noqa: BLE001
            loc.pred, loc.layers = None, None
    parts, pred, layers, lp, oneshot = loc.parts, loc.pred, loc.layers, loc.lp, loc.oneshot
    head_why, eh_why, shapes, H, K, dev = loc.head_why, loc.eh_why, loc.shapes, loc.H, loc.K, loc.dev
    max_tokens, fh, splits, hrows, erows = loc.max_tokens, loc.fh, loc.splits, loc.hrows, loc.erows

    # 1. preflight exchange -> plan (pure function of the exchanged rows)
    fp = fingerprint(parts, size, shapes, loc.eh_gate)
    pre = {"rank": rank, "fp": fp, "head_ok": not head_why, "head_why": "; ".join(head_why)[:400],
           "eh_ok": not eh_why, "eh_why": "; ".join(eh_why)[:400], "max_tokens": max_tokens,
           "eh_gate": loc.eh_gate}
    pre_rows = _exchange(world, pre, "preflight")
    if pre_rows is None:
        plan = _Plan(False, False, (), (), (), ("preflight exchange failed on this rank",))
    else:
        plan = _make_plan(pre_rows, parts)

    # 2. collective checks (planned, identical on every rank)
    eh_sha, head_sha = [], []
    if plan.eh:
        eh_sha = _eh_collective_checks(plan, layers, splits, oneshot, H, K, dev, erows, world)
    if plan.head:
        norm_w = getattr(getattr(layers[0].shared_head, "norm", None), "weight", None)
        head_sha = _head_collective_checks(plan, fh, layers[0].shared_head.head, lp, norm_w, world, hrows)

    # 3. verdict exchange -> decision (pure function of the exchanged rows)
    ver = {"rank": rank, "head_ok": plan.head and hrows.ok, "head_fail": "; ".join(hrows.fail[:4])[:400],
           "head_sha": head_sha, "eh_ok": plan.eh and erows.ok, "eh_fail": "; ".join(erows.fail[:4])[:400],
           "eh_sha": eh_sha}
    if plan.head or plan.eh:
        ver_rows = _exchange(world, ver, "verdict")
        if ver_rows is None:
            head_on, eh_on, why = False, False, list(plan.why) + ["verdict exchange failed on this rank"]
        else:
            head_on, eh_on, why = _decide(plan, ver_rows)
        # commit round: every rank reports the decision it took; any disagreement (a rank whose verdict exchange
        # failed locally) or a failed commit leaves BOTH parts off on every rank that sees it.
        com_rows = _exchange(world, {"rank": rank, "head_on": bool(head_on), "eh_on": bool(eh_on)}, "commit")
        if com_rows is None or len({(bool(r.get("head_on")), bool(r.get("eh_on"))) for r in com_rows}) != 1:
            if head_on or eh_on:
                why.append("commit round failed or the ranks' decisions differ")
            head_on = eh_on = False
    else:
        head_on, eh_on, why = False, False, list(plan.why)

    # 4. install (or not) + log
    if fh is not None:
        fh.release_reference()
    pred_ok = pred is not None
    if pred_ok:
        pred._p029_head = fh if head_on else None
        for li, ly in enumerate(layers or []):
            ly._p029_eh = splits[li] if (eh_on and splits is not None) else None
    if not head_on:
        fh = loc.fh = None                    # drop the refused FP8 copy BEFORE empty_cache (79 MB per rank)
    if not eh_on:
        splits = loc.splits = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    summary = {"head": head_on, "eh": eh_on, "why": why, "plan": plan._asdict(), "head_info": hrows.info,
               "eh_info": erows.info, "parts": parts, "fp": fp, "rank": rank, "rev": REVISION}
    _log(rank, size, parts, head_on, eh_on, why, hrows, erows, H, K, plan.eh_gate)
    return summary


def _log(rank, size, parts, head_on, eh_on, why, hrows, erows, H, K, eh_gate=EH_GATES[0]) -> None:
    """One line per rank. The ACTIVE line never contains INACTIVE / [OK] / readiness fail texts: every reason is
    passed through _safe() (a refused part is reported as '; not active: ...', which the gate scripts look for)."""
    hi, ei = hrows.info, erows.info
    head_s = ("FP8 draft lm_head ON [" + f"{hi.get('mb')} MB e4m3 per-channel, Marlin W8A16; kernel-vs-dequant "
              f"relRMSE {hi.get('kernel_rel')}, vs BF16 relRMSE {hi.get('rel')} max|diff| {hi.get('maxabs')} "
              f"(bound ratio {hi.get('bound_ratio')}), top-1 random {hi.get('top1_random')} peaked "
              f"{hi.get('top1_peaked')} global {hi.get('top1_global')}, M=1 {hi.get('ms_fp8')} vs "
              f"{hi.get('ms_bf16')} ms]") if head_on else "FP8 draft lm_head off"
    if eh_gate == "graph":
        e5 = (f"E5 CUDA graphs (one-shot + {EH_GRAPH_CALLS} calls, median of {EH_GRAPH_ITERS} replays, bound "
              f"{EH_GRAPH_RATIO_MAX}) [M, split, replicated, saved per call] ms {ei.get('eh_graph_ms')}")
    else:
        e5 = f"E5 speed gate off ({ENV_EH_GATE}=numerics)"
    eh_s = ("eh_proj row split ON [" + f"{H // max(size, 1)}x{K} per rank + patch-012 one-shot; vs replicated "
            f"bound ratio {max(ei.get('eh_ratio') or [0])}, ulp-diff elements {ei.get('eh_ulpdiff')}; {e5}; eager "
            f"[M, split, replicated] ms {ei.get('eh_eager_ms')} (information)]") if eh_on else "eh_proj split off"
    reasons = _safe(" | ".join(why)) if why else ""
    if head_on or eh_on:
        logger.info("%s draft-path compression ACTIVE (rank %d/%d, parts %s): %s; %s; target lm_head / logits / "
                    "verification untouched; module rev %d%s", TAG, rank, size, "+".join(parts), head_s, eh_s,
                    REVISION, ("; not active: " + reasons) if reasons else "")
    else:
        logger.warning("%s draft-path compression INACTIVE (fail closed) (rank %d/%d): %s - the image draft path "
                       "runs unchanged (module rev %d)", TAG, rank, size, reasons or "nothing selected", REVISION)
