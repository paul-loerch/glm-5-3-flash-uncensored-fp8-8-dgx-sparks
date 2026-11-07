# SPDX-License-Identifier: Apache-2.0
# [patch-033 gb10-mla-w8a16] installed as vllm/_gb10_mla_w8a16.py by patches/033-gb10-mla-w8a16.sh
# ("MLA projections as FP8 W8A16"; E3 below = the 2026-10-03 pre-integration microbenchmark). Written for this recipe.
"""GB10, TP=8: serve the MLA attention projections as FP8 W8A16 (Marlin) in decode (opt-in VLLM_GB10_MLA_W8A16=1).

REVISION 2 = the vLLM v0.32.0 port (base image vllm/vllm-openai:v0.32.0-aarch64). What changed against revision 1 (day-0
image 487ecf187), all of it forced by upstream API drift in the Marlin helpers (upstream #57421 "shared Marlin workspaces"):
  * marlin_utils_fp8.prepare_fp8_layer_for_marlin() no longer creates `layer.workspace`, and apply_fp8_marlin_linear() now takes
    `workspace: Tensor | None` and calls torch.ops.vllm.marlin_gemm(b_q_type_id=...) (a custom op registered by marlin_utils.py
    that falls back to upstream's persistent per-STREAM lock buffer, allocated on first use through the workspace manager,
    when workspace is None). _build() therefore allocates its OWN zero-initialised lock buffer per projection with
    marlin_utils.marlin_make_workspace_new(device, MARLIN_MAX_BLOCKS_PER_SM) (4 blocks per SM, upstream's value for every dense
    call; the image used 1) at weight-processing time, outside any CUDA-graph capture, and passes it explicitly: a lazily
    allocated per-stream buffer could be created INSIDE a capture on the capture stream (and the boot self-test runs on other
    streams). 36 projections x (SMs x 4) int32 = a few KiB per rank. Marlin resets its locks after every call, and every
    projection runs on the one stream of the graph replay, so a private buffer per projection is as safe as before.
  * ops.gptq_marlin_repack lost its `perm` argument and marlin_make_workspace_new its `existing` argument: only called by the
    upstream helpers this module uses (prepare_fp8_layer_for_marlin), never directly.
  * the construction check also requires torch.ops.vllm.marlin_gemm (registered when marlin_utils is imported) and refuses
    dual-batch overlap / ubatching (parallel_config.use_ubatching): with one private lock buffer per projection, two
    micro-batches of one layer running on two streams would share it (upstream keys its persistent locks by stream, ubatch
    and workspace lane). The validated envelope never enabled it.
  * the model file is models/glm5next/common/model.py (was nvidia/model.py); UnquantizedLinearMethod.__init__ now resolves the
    GEMM implementation once (self._gemm_impl = dispatch_unquantized_gemm(kernel_config.linear_backend)); _Method calls
    super().__init__() under the model-construction vLLM config exactly as the layer's own method was, so the fallback BF16
    GEMM is the same implementation the unpatched layer would have used.
Everything else (capture, V1-V4, C1-C9, the speed gate, the exchanges, the decision, the log lines) is unchanged.

WHY. The 11 MLA layers (3, 7, ..., 43) and the MTP layer (45) store q_a_proj, kv_a_proj_with_mqa, q_b_proj and o_proj as
block-FP8 (e4m3, 128 x 128 blocks, weight_scale_inv). The served model dequantises them to BF16 at load
(model.py _try_load_fp8_attn_proj; the MLA projections are built with quant_config=None), so every decode step streams
39.8 MB of BF16 per MLA layer for weights that exist as 19.9 MB of FP8. E3 (real weights, Marlin W8A16, DRAM-cold, inside
CUDA graphs): 183.5 us -> 97.2 us per layer at M=3 (-1.18 ms per step with the MTP layer), error vs fp32 0.00346 (BF16
path 0.00247). Realistic range after patch 032's first-NCCL-op cover: 0.95-1.18 ms/step = +2.5-3.1 %.

WHAT. Per projection TYPE of every MLA layer (qkv_a = the replicated fused q_a|kv_a, q_b, o_proj; kv_b_proj is BF16 in
the checkpoint and not touched):
  * capture(): the checkpoint's FP8 shard + block scales are copied (explicitly: never a view that would pin the whole
    checkpoint tensor) at load, in model.py's _try_load_fp8_attn_proj, before the BF16 dequant, sharded exactly like the
    BF16 loader shards them;
  * process_weights_after_loading (per module): the capture is verified BITWISE against the served BF16 weight (V1-V4),
    then the Marlin group-128 copy is built with the image's prepare_fp8_layer_for_marlin (the path E3 measured);
  * after the last module of the target group (and of the MTP group): a boot self-test on the real weights (C1-C9 below),
    a gloo exchange, one decision for target + MTP, a commit round, one log line per rank.
  * dispatch (_Method.apply, the module's quant_method): Marlin when the input is a 2-D bf16 matrix with 0 < M <=
    MAX_M (default 16) and no bias, else the image's BF16 GEMM (UnquantizedLinearMethod.apply) on the BF16 weight that
    stays resident. Prefill chunks longer than MAX_M tokens are therefore BF16 with unchanged numerics; SHORTER ones (short
    prompts, short follow-up turns with prefix caching, tails) use Marlin, because CUDA-graph dispatch keys on the token
    count: a 3-token prefill and a 3-token decode are the same captured graph. Do not switch to capture gating.

STATES (per projection; MF1 of the design review). pending (BF16; the state from construction until the boot decision)
-> probe (entered ONLY inside the boot self-test, with try/finally, restored to pending on every path; dispatches like
active and counts seam hits) -> active | off. `hits` is counted only in probe, so the seam check "one real module call =
exactly one hit" and the strided-input check compare real Marlin results, never the BF16 path with itself.
RELOAD. A second process_weights_after_loading of an ACTIVE projection (weights loaded again after the decision; not
reachable in this recipe) sends that projection back to the image BF16 GEMM but never frees its Marlin tensors: captured
CUDA graphs hold raw pointers to them (_retire_after_reload parks them in the state for the life of the process) and keep
replaying the OLD checkpoint FP8; one loud warning per rank says to restart the server.

NUMERICS. The checkpoint FP8 values are exact; the only change is the scale rounded to bf16 (<= 1 bf16 ulp, 0.4 % rel.)
and a different GEMM (E3: output relRMSE 1.40 x the BF16 path's, 0.00346 vs 0.00247). Re-quantising the served BF16
weight with the checkpoint scales reproduces the checkpoint FP8 bit for bit (exhaustive over all 254 finite e4m3 codes x
10^8 scales); V3 uses that as an independent cross-check of the captured bytes. The abliteration edited o_proj tensors;
they are still the checkpoint's FP8 values.

BOOT SELF-TEST (fail closed, hang-free, never raises into the engine). Per module, local, no collective:
  V1 completeness/shape/storage (the capture holds exactly the shard: storage bytes == numel), V2 captured FP8 x scale
  == the served BF16 weight BITWISE (int views), V3 enc(served / s) == the captured bytes (blocks with s < 2^-100 are
  excluded and must serve ~0), V4 scales finite, 0 <= s < 128, no NaN codes. Build: scratch holder Module +
  prepare_fp8_layer_for_marlin(size_k_first=False), weight_block_size [128, 128] (exactly bench_mla.py prep("marlin")).
  Per module and M in {1,2,3,4,6,8,16} + MAX_M, three seeded draws (unit / heavy_chan / heavy_t3), through the dispatch in
  the probe state:
    C1 relRMSE(Marlin, x @ (fp8 x fp32 scale)^T; float64 reference rounded to fp32) <= 0.0060   (E3 worst 0.0035)
    C2 relRMSE(Marlin, the served BF16 path)            <= 0.0080   (E3 0.0042)
    C3 relRMSE(Marlin, x @ (fp8 x bf16(s))^T in fp32)   <= 0.0045   (E3 0.0025; a layout/scale bug gives ~1)
    C4 relRMSE(BF16 path, fp32)                         <= 0.0045   (E3 0.0025)
    C5 min row cosine(Marlin, fp32) >= 0.99998;  C6 finite, two calls bitwise equal, [M, N] bf16 contiguous
    C7 dispatch fidelity: M = MAX_M + 1, 2 x MAX_M, bias, wrong dtype, 3-D input return BITWISE what
       UnquantizedLinearMethod.apply returns (no hit counted); M = 0 does not raise; a strided input equals the contiguous
       input's Marlin result bitwise (two hits)
    C8 seam: the REAL module call (fused_qkv_a_proj(x), q_b_proj(x), o_proj(x) with reduce_results False) reaches
       quant_method.apply exactly once and returns the Marlin result. A type that fails only here falls back ALONE
       (the other types stay active); anything else fails the whole patch.
    C9 memory (storage bytes) <= 0.5 GiB, layer.weight digests unchanged, every module back in pending
  plus: should_use_atomic_add_reduce is False for every tested shape (the graph-capture assumption), the replicated qkv_a
  digests (captured FP8 + scales, Marlin outputs) are identical on all 8 ranks, and (target group, speed gate "graph") an
  11-layer DRAM-cold sweep inside CUDA graphs at M=3 requires Marlin <= 0.85 x BF16 (E3: 0.53).
The real Marlin kernel's tile permutation is NOT modelled by the Mac tests: the first GPU boot's logged C1-C3 numbers are
the layout acceptance.

LOG LINES (one decision line per rank; every reason/exception text is defanged by _safe()):
  "[patch-033] MLA FP8 W8A16 (Marlin) ACTIVE (rank r/8): ..." | "... INACTIVE (fail closed) (rank r/8): <reason>; ..."
  plus one INFO line after the target group, and one "first small-M call" line per projection type and rank (SF8).
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001 - CPU tests outside a vLLM install
    logger = logging.getLogger(__name__)

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402

TAG = "[patch-033]"
ENV = "VLLM_GB10_MLA_W8A16"
ENV_MAX_M = "VLLM_GB10_MLA_W8A16_MAX_M"
ENV_SPEED = "VLLM_GB10_MLA_W8A16_SPEED_GATE"
ENV_BATCH_INVARIANT = "VLLM_BATCH_INVARIANT"
REVISION = 2
REQUIRED_TP = 8
BLOCK = 128                              # checkpoint block-FP8 geometry == Marlin group size
MAX_M_DEFAULT = 16
MAX_M_LIMIT = 64
SPEED_MODES = ("graph", "info")          # graph: enforce the in-graph speed gate; info: time + log, do not enforce
TYPES = ("qkv_a", "q_b", "o_proj")
SELFTEST_M = (1, 2, 3, 4, 6, 8, 16)      # 6 is a real capture size of this recipe (SF7); 3 is the eager verify size
DRAWS = ("unit", "heavy_chan", "heavy_t3")
DIGEST_M = (1, 3, 16)
C1_MAX = 0.0060
C2_MAX = 0.0080
C3_MAX = 0.0045
C4_MAX = 0.0045
C5_MIN = 0.99998
SCALE_MAX = 128.0                        # 2^120-folded bf16 scales overflow above ~255: 2x margin
V3_MIN_SCALE = 2.0 ** -100               # below this the re-quantisation lemma is not applied (served ~ 0 is required)
V3_TINY = 2.0 ** -90
MEM_LIMIT_BYTES = int(0.5 * 2 ** 30)
CHUNK_ROWS = 512                         # row chunks of every reference GEMM / dequant (bounded fp32 temporaries)
SPEED_M = 3
SPEED_RATIO_MAX = 0.85                   # Marlin <= 0.85 x BF16 inside graphs (E3 measured 0.53)
SPEED_WARMUP = 2
SPEED_ITERS = 10
E3_REF_US = (183.5, 97.2)                # BF16, Marlin us per MLA layer at M=3 (printed for regression reading)
SELFTEST_BARRIER_S = 600.0               # ranks reach process_weights_after_loading at different times (~8-11 min)
CAPTURE_ERROR_MODE = "thread_local"
SEED = 0x033A
PENDING = "pending"
PROBE = "probe"
ACTIVE = "active"
OFF = "off"
MODES = (PENDING, PROBE, ACTIVE, OFF)
FP8_NAN_MASK = 0x7F


def _own_sha16() -> str:
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


MODULE_SHA16 = _own_sha16()


def _sha16_file(path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# Env parsing (the .sh validates the same grammar before the boot)
# ---------------------------------------------------------------------------------------------------------------------
def gate_on(raw: str | None = None) -> bool:
    raw = os.environ.get(ENV, "0") if raw is None else raw
    return str(raw).strip() == "1"


def parse_max_m(raw: str | None) -> int:
    v = "" if raw is None else re.sub(r"\s+", "", str(raw))
    if v == "":
        return MAX_M_DEFAULT
    if not re.fullmatch(r"[1-9][0-9]*", v) or int(v) > MAX_M_LIMIT:
        raise ValueError(f"{ENV_MAX_M}={raw!r}: expected an integer 1..{MAX_M_LIMIT}")
    return int(v)


def parse_speed_gate(raw: str | None) -> str:
    v = "" if raw is None else re.sub(r"\s+", "", str(raw)).lower()
    if v == "":
        return SPEED_MODES[0]
    if v not in SPEED_MODES:
        raise ValueError(f"{ENV_SPEED}={raw!r}: expected one of {list(SPEED_MODES)}")
    return v


# Texts other gates grep for (every launch/readiness.txt fail AND require rule, the boot-gate scripts' ACTIVE /
# INACTIVE / [OK] counts): a quoted exception or reason must never reproduce one inside a [patch-033] line (029 rev 2).
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
    """vLLM's tensor-parallel group: rank / size, and gloo object all-gather behind a bounded monitored barrier (the
    group's CPU group; no new communicator, no NCCL)."""

    kind = "tp"

    def __init__(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        self.group = get_tp_group()
        self.rank = int(self.group.rank_in_group)
        self.size = int(self.group.world_size)
        self.pp_size = int(get_pp_group().world_size)

    def all_gather_object(self, obj):
        import torch.distributed as dist

        out = [None] * self.size
        dist.all_gather_object(out, obj, group=self.group.cpu_group)
        return out

    def barrier_bounded(self, timeout_s: float) -> None:
        import torch.distributed as dist

        dist.monitored_barrier(group=self.group.cpu_group, timeout=datetime.timedelta(seconds=timeout_s),
                               wait_all_ranks=True)


_WORLD_FACTORY: Callable[[], object] | None = None      # tests only
_CTX_KEY_FN: Callable[[], object] | None = None         # tests only: one state per emulated rank thread


def _make_world():
    return _WORLD_FACTORY() if _WORLD_FACTORY is not None else _TPWorld()


def _tp_rank_size() -> tuple[int, int]:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())


def _marlin_fp8():
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as MU8

    return MU8


def _marlin_utils():
    from vllm.model_executor.layers.quantization.utils import marlin_utils as MU

    return MU


def _make_workspace(device):
    """v0.32.0: prepare_fp8_layer_for_marlin() no longer attaches a workspace. One zero-initialised int32 lock buffer of
    SMs x MARLIN_MAX_BLOCKS_PER_SM (= what marlin_utils.get_marlin_workspace() would hand out), allocated here at weight
    processing time (never inside a capture)."""
    MU = _marlin_utils()
    return MU.marlin_make_workspace_new(device, getattr(MU, "MARLIN_MAX_BLOCKS_PER_SM", 4))


# ---------------------------------------------------------------------------------------------------------------------
# State (one per process; tests keep one per emulated rank)
# ---------------------------------------------------------------------------------------------------------------------
class _Entry:
    """Capture registry entry of one installed module weight. Keyed by the Parameter OBJECT (the MTP layer's parameter
    names carry `mtp_block.`, its module prefix does not: a name key would not match)."""

    def __init__(self, param, typ: str, n: int, k: int, n_q: int, tp_rank: int, tp_size: int) -> None:
        self.param = param                 # strong reference: the id cannot be reused while registered
        self.typ, self.n, self.k, self.n_q = typ, n, k, n_q
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.w8 = None                     # [N, K] float8_e4m3fn, this rank's shard (own storage)
        self.sc = None                     # [N/128, K/128] float32
        self.have: set = set()
        self.bad: str | None = None
        self.calls = 0


class _Proj:
    def __init__(self, typ: str, group: str, layer_idx: int, module, entry: _Entry, max_m: int) -> None:
        self.typ, self.group, self.layer_idx = typ, group, layer_idx
        self.module, self.entry, self.max_m = module, entry, max_m
        self.n, self.k = entry.n, entry.k
        self.orig = module.quant_method
        self.method = None
        self.mode = PENDING
        self.hits = 0
        self.w = self.s = self.ws = None
        self.built_bytes = 0
        self.bad: str | None = None
        self.processed = False

    @property
    def label(self) -> str:
        return f"{self.group} layer {self.layer_idx} {self.typ}"


class _Group:
    def __init__(self, name: str) -> None:
        self.name = name
        self.projs: list[_Proj] = []
        self.done = 0
        self.verdict: dict | None = None
        self.complete = False
        self.selftest_s = 0.0                # wall time of this rank's self-test (logged: the float64 references are estimated)


class _State:
    def __init__(self) -> None:
        self.groups: dict[str, _Group] = {"target": _Group("target"), "mtp": _Group("mtp")}
        self.by_param: dict[int, _Entry] = {}
        self.expect_mtp = False
        self.disabled: str | None = None
        self.finalized = False
        self.rank = self.size = "?"
        self.max_m = MAX_M_DEFAULT
        self.speed = SPEED_MODES[0]
        self.failed: list[str] = []          # reasons of groups whose verdict was not ok (decision waits for all groups)
        self.first_logged: set = set()
        self.pending_warned = False
        self.late_logged = False
        self.refusal_logged = False
        self.reload_logged = False
        self.retired: list = []              # (weight, scales, workspace) of projections whose weights were reloaded after
        #                                      the decision: kept alive for the life of the process (see _retire_after_reload)
        self.final: dict | None = None


_STATES: dict = {}


def _state() -> _State:
    key = _CTX_KEY_FN() if _CTX_KEY_FN is not None else 0
    st = _STATES.get(key)
    if st is None:
        st = _STATES[key] = _State()
    return st


def _all_projs(st: _State) -> list[_Proj]:
    return [p for g in st.groups.values() for p in g.projs]


# ---------------------------------------------------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------------------------------------------------
def _nbytes(t) -> int:
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return int(t.numel() * t.element_size())


def _i16(t):
    return t.view(torch.int16)


def _u8(t):
    return t.view(torch.uint8)


def _expand_scales(sc, r0: int, r1: int, k: int):
    """fp32 [r1 - r0, k]: the block scale of rows r0:r1 (multiples of BLOCK) broadcast to elements (the formula of
    model.py _dequant_fp8_block)."""
    s = sc[r0 // BLOCK:(r1 + BLOCK - 1) // BLOCK]
    return s.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)[: r1 - r0, :k]


def _count(t) -> int:
    return int(t.sum().item())


def _digest_int(t) -> str:
    """Content digest of a 2-D fp8 / int16 / int32 tensor: row sums and position-weighted row sums as int64, chunked."""
    n = int(t.shape[0])
    d1 = d2 = 0
    for r0 in range(0, n, CHUNK_ROWS):
        r1 = min(n, r0 + CHUNK_ROWS)
        x = t[r0:r1]
        x = _u8(x) if x.dtype == torch.float8_e4m3fn else x
        rs = x.to(torch.int64).sum(dim=1)
        d1 += int(rs.sum().item())
        pos = torch.arange(r0 + 1, r1 + 1, dtype=torch.int64, device=t.device)
        d2 += int((rs * pos).sum().item())
    return f"{d1}:{d2}"


def _wdigest(w) -> str:
    return _digest_int(_i16(w))


def _randn(m: int, k: int, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(m, k, generator=g, dtype=torch.float32)


def _synth(draw: str, m: int, k: int, seed: int, device):
    """[m, k] bf16 from a seeded CPU generator (identical on every rank): E3's draws."""
    if draw == "unit":
        x = _randn(m, k, seed)
    elif draw == "heavy_chan":
        ch = (1.5 * _randn(1, k, seed + 7919)).exp()
        x = _randn(m, k, seed) * ch
    elif draw == "heavy_t3":
        z = _randn(m, k, seed)
        c1, c2, c3 = _randn(m, k, seed + 11), _randn(m, k, seed + 13), _randn(m, k, seed + 17)
        x = z / ((c1 * c1 + c2 * c2 + c3 * c3) / 3.0).sqrt()
        for j in range(4):
            idx = (j * 7919 + 13) % k
            x[:, idx] = x[:, idx] * 20.0
    else:
        raise ValueError(draw)
    return x.to(device).to(torch.bfloat16)


def _rel(a, ref) -> float:
    d = a.float() - ref.float()
    den = float((ref.float() * ref.float()).sum().item())
    return math.sqrt(float((d * d).sum().item()) / max(den, 1e-30))


def _mismatches(a, b) -> int:
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return max(int(a.numel()), int(b.numel()), 1)
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return int((a.contiguous().view(iv) != b.contiguous().view(iv)).sum().item())


def _sha16_tensor(t) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------
# The quant method installed on the three modules of every MLA layer
# ---------------------------------------------------------------------------------------------------------------------
def _marlin(p: _Proj, x):
    """The Marlin W8A16 GEMM for a 2-D contiguous-or-not bf16 input (host metadata only; no allocation of our own)."""
    return _marlin_fp8().apply_fp8_marlin_linear(
        input=x if x.is_contiguous() else x.contiguous(), weight=p.w, weight_scale=p.s, workspace=p.ws,
        size_n=p.n, size_k=p.k, bias=None)


def _bf16(p: _Proj, x, bias=None):
    """The image's BF16 GEMM on the resident BF16 weight (what _Method.apply falls back to)."""
    return UnquantizedLinearMethod.apply(p.method, p.module, x, bias)


class _Method(UnquantizedLinearMethod):
    """UnquantizedLinearMethod + the decode dispatch + the boot hook. The three modules keep `isinstance(...,
    UnquantizedLinearMethod)`; the weight-loader choice was made at construction and does not look at this object."""

    def __init__(self, proj: _Proj) -> None:
        super().__init__()
        self._p = proj

    def apply(self, layer, x, bias=None):
        p = self._p
        mode = p.mode
        if mode == ACTIVE or mode == PROBE:
            if bias is None and x.dim() == 2 and x.dtype == torch.bfloat16 and 0 < x.shape[0] <= p.max_m:
                if mode == PROBE:
                    p.hits += 1
                else:
                    _first_call(p, "marlin", x)
                return _marlin(p, x)
            if mode == ACTIVE:
                _first_call(p, "bf16", x, bias)
        elif mode == PENDING:
            _pending_call(p, x)
        return super().apply(layer, x, bias)

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        _module_pwal(self._p, layer)


def _first_call(p: _Proj, route: str, x, bias=None) -> None:
    """SF8: one line per projection type and rank, from the first real call that could use Marlin (0 < rows <=
    MAX_M): marlin, or bf16 with the reason (a 3-D / non-bf16 / biased call through an unpinned wrapper)."""
    st = _state()
    if p.typ in st.first_logged:
        return
    try:
        k = int(x.shape[-1])
        rows = int(x.numel()) // max(k, 1)
        if not 0 < rows <= p.max_m:
            return
        st.first_logged.add(p.typ)
        why = ""
        if route == "bf16":
            why = f" ({'bias' if bias is not None else 'dim ' + str(x.dim()) if x.dim() != 2 else str(x.dtype)})"
        logger.info("%s first small-M call on rank %s/%s: %s -> %s%s (M=%d, N=%d, K=%d)", TAG, st.rank, st.size, p.typ,
                    route, why, rows, p.n, p.k)
    except Exception:  # noqa: BLE001 - logging must never break a forward
        pass


def _pending_call(p: _Proj, x) -> None:
    st = _state()
    if st.pending_warned:
        return
    try:
        if 0 < int(x.numel()) // max(int(x.shape[-1]), 1) <= p.max_m:
            st.pending_warned = True
            logger.warning("%s a decode-sized call reached %s before the boot decision completed on rank %s/%s: the "
                           "BF16 GEMM runs (group %s: %d of %d modules processed)", TAG, p.label, st.rank, st.size,
                           p.group, st.groups[p.group].done, len(st.groups[p.group].projs))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------------------------------
# install() - Glm5NextDecoderLayer.__init__ (model.py), every MLA layer, the MTP layer included
# ---------------------------------------------------------------------------------------------------------------------
def _expect_mtp(vllm_config) -> bool:
    sc = getattr(vllm_config, "speculative_config", None)
    if sc is None:
        return False
    try:   # vLLM names the family "mtp" or "<model>_mtp" depending on the target's model type
        return "mtp" in str(getattr(sc, "method", "")).lower() and int(getattr(sc, "num_speculative_tokens", 0) or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def _module_refusal(typ: str, m, attn, tp: int) -> str | None:
    """Config / structure checks only (identical on every rank)."""
    qm = getattr(m, "quant_method", None)
    if type(qm) is not UnquantizedLinearMethod:
        return f"{typ}: quant_method is {type(qm).__name__} (expected exactly UnquantizedLinearMethod)"
    w = getattr(m, "weight", None)
    if w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or not w.is_cuda:
        return f"{typ}: weight is not a 2-D bf16 CUDA tensor"
    if getattr(m, "bias", None) is not None:
        return f"{typ}: has a bias"
    n, k = int(w.shape[0]), int(w.shape[1])
    if n % BLOCK or k % BLOCK:
        return f"{typ}: shape {n}x{k} is not a multiple of the {BLOCK}x{BLOCK} block"
    if typ == "qkv_a":
        want_tp = 1
        n_q = int(attn.q_lora_rank)
        n_kv = int(attn.kv_lora_rank) + int(attn.qk_rope_head_dim)
        if n != n_q + n_kv:
            return f"{typ}: {n} rows != q_lora_rank {n_q} + kv_a rows {n_kv}"
        if n_q % BLOCK or n_kv % BLOCK:
            return f"{typ}: q_a / kv_a rows {n_q} / {n_kv} are not multiples of {BLOCK}"
    else:
        want_tp = tp
    if int(getattr(m, "tp_size", -1)) != want_tp:
        return f"{typ}: tp_size {getattr(m, 'tp_size', None)} != {want_tp}"
    if typ == "o_proj" and tp > 1 and not hasattr(m, "reduce_results"):
        return f"{typ}: no reduce_results attribute (the seam probe could not avoid its all-reduce)"
    MU = _marlin_utils()
    if tuple(MU.marlin_padded_nk(n, k, BLOCK)) != (n, k):
        return f"{typ}: shape {n}x{k} needs Marlin tile padding"
    return None


def _construction_refusal(attn, vllm_config, size: int) -> str | None:
    """The validated envelope; depends on the config and the module structure only -> identical on every rank."""
    if size != REQUIRED_TP:
        return f"validated for TP={REQUIRED_TP} only (this TP group has {size} ranks)"
    from vllm.distributed import get_pp_group

    pp = int(get_pp_group().world_size)
    if pp != 1:
        return f"needs pipeline parallel size 1 (got {pp})"
    if torch.get_default_dtype() != torch.bfloat16:
        return f"model dtype {torch.get_default_dtype()} (validated for bfloat16 only)"
    if str(os.environ.get(ENV_BATCH_INVARIANT, "0")).strip() not in ("", "0"):
        return f"{ENV_BATCH_INVARIANT} is set (the batch-invariant GEMM path replaces the dispatch)"
    if getattr(vllm_config, "lora_config", None) is not None:
        return "LoRA is enabled"
    pc = getattr(vllm_config, "parallel_config", None)
    if pc is not None and bool(getattr(pc, "use_ubatching", False)):
        # v0.32.0 only: the projections own one Marlin lock buffer each; dual-batch overlap would run two micro-batches of the
        # same layer on two streams at once and share it (upstream keys its persistent locks by stream / ubatch / lane)
        return "dual-batch overlap / ubatching is enabled (two streams would share one projection's Marlin lock buffer)"
    try:
        parse_max_m(os.environ.get(ENV_MAX_M))
        parse_speed_gate(os.environ.get(ENV_SPEED))
    except ValueError as e:
        return str(e)
    if getattr(attn, "q_lora_rank", None) is None:
        return "q_lora_rank is None (no fused q_a|kv_a projection)"
    for a in ("fused_qkv_a_proj", "q_b_proj", "o_proj"):
        if not hasattr(attn, a):
            return f"attention module has no {a}"
    try:
        MU8 = _marlin_fp8()
        if not MU8.is_fp8_marlin_supported():
            return "is_fp8_marlin_supported() is False"
        for op in ("marlin_gemm", "gptq_marlin_repack"):
            if not hasattr(torch.ops._C, op):
                return f"torch.ops._C.{op} missing"
        if not hasattr(torch.ops.vllm, "marlin_gemm"):
            return "torch.ops.vllm.marlin_gemm missing (marlin_utils registers it when imported)"
        if int(torch.cuda.get_device_capability()[0]) != 12:
            return f"CUDA capability family {torch.cuda.get_device_capability()} (validated on 12.x / sm_121)"
    except Exception as e:  # noqa: BLE001
        return f"Marlin import/capability check: {type(e).__name__}: {e}"
    if not hasattr(torch, "float8_e4m3fn"):
        return "torch has no float8_e4m3fn"
    return None


def install(attn, vllm_config, layer_idx: int, is_mtp_layer: bool) -> None:
    """model.py E2: called once per MLA layer (target and MTP). NEVER raises."""
    try:
        _install(attn, vllm_config, int(layer_idx), bool(is_mtp_layer))
    except Exception as e:  # noqa: BLE001
        try:
            _refuse(_state(), f"install raised {type(e).__name__}: {str(e)[:300]}")
        except Exception:  # noqa: BLE001
            pass


def _install(attn, vllm_config, layer_idx: int, is_mtp: bool) -> None:
    st = _state()
    if st.disabled is not None:
        return
    if st.finalized:
        if not st.late_logged:
            st.late_logged = True
            logger.info("%s layer %d arrived after the boot decision on rank %s/%s: left as the image's BF16 GEMMs",
                        TAG, layer_idx, st.rank, st.size)
        return
    rank, size = _tp_rank_size()
    st.rank, st.size = str(rank), str(size)
    why = _construction_refusal(attn, vllm_config, size)
    if why is None:
        for typ, name in zip(TYPES, ("fused_qkv_a_proj", "q_b_proj", "o_proj")):
            why = _module_refusal(typ, getattr(attn, name), attn, size)
            if why:
                break
    if why is None and is_mtp and not st.groups["target"].projs:
        why = "the MTP layer was built before any target MLA layer"
    if why is not None:
        _refuse(st, why)
        return
    st.max_m = parse_max_m(os.environ.get(ENV_MAX_M))
    st.speed = parse_speed_gate(os.environ.get(ENV_SPEED))
    if not is_mtp:
        st.expect_mtp = _expect_mtp(vllm_config)
    grp = st.groups["mtp" if is_mtp else "target"]
    n_q = int(attn.q_lora_rank)
    for typ, name in zip(TYPES, ("fused_qkv_a_proj", "q_b_proj", "o_proj")):
        m = getattr(attn, name)
        w = m.weight
        ent = _Entry(w, typ, int(w.shape[0]), int(w.shape[1]), n_q if typ == "qkv_a" else 0,
                     int(getattr(m, "tp_rank", 0)), int(getattr(m, "tp_size", 1)))
        st.by_param[id(w)] = ent
        p = _Proj(typ, grp.name, layer_idx, m, ent, st.max_m)
        p.method = _Method(p)
        m.quant_method = p.method
        grp.projs.append(p)


def _refuse(st: _State, why: str) -> None:
    """Identical on every rank (config / structure only): restore whatever was installed, log once, disable."""
    if st.disabled is None:
        st.disabled = why
    _restore(st, _all_projs(st))
    if not st.refusal_logged:
        st.refusal_logged = True
        _log_inactive(st, [why])


def _restore(st: _State, projs: list[_Proj]) -> None:
    for p in projs:
        try:
            if p.module.quant_method is p.method:
                p.module.quant_method = p.orig
        except Exception:  # noqa: BLE001
            pass
        p.mode = OFF
        p.w = p.s = p.ws = None
        p.built_bytes = 0
        p.entry.w8 = p.entry.sc = None


def _retire_after_reload(st: _State, p: _Proj) -> None:
    """The model's weights were loaded AGAIN after the boot decision (not reachable in this recipe: no sleep mode, no reload
    endpoint). CUDA graphs captured while the projection was active hold RAW POINTERS to its Marlin weight, scales and
    workspace (vLLM keeps the workspace address stable across reloads for the same reason), so those tensors are never freed
    here: they are parked in st.retired for the life of the process (freeing them would let the caching allocator hand the
    memory to something else while the graphs keep replaying, i.e. silent garbage). Only the dispatch goes back to the image
    BF16 GEMM, for eager calls and for any graph captured from now on. Graphs captured BEFORE keep replaying the OLD checkpoint
    FP8 weights, hence the loud warning: restart the server."""
    st.retired.append((p.w, p.s, p.ws))
    _restore(st, [p])
    if not st.reload_logged:
        st.reload_logged = True
        logger.warning("%s weights of %s were loaded again after the boot decision on rank %s/%s: this projection returns to "
                       "the image's BF16 GEMM for new calls (the other 033 projections of this rank follow as their weights "
                       "are loaded, without further lines), BUT the CUDA graphs captured earlier still read the OLD "
                       "checkpoint FP8 weights through raw pointers (their buffers are kept alive for that reason): restart "
                       "the server to get a consistent model", TAG, p.label, st.rank, st.size)


# ---------------------------------------------------------------------------------------------------------------------
# capture() - model.py _try_load_fp8_attn_proj, before the BF16 dequant
# ---------------------------------------------------------------------------------------------------------------------
def capture(param, key: str, weight_fp8, scale_inv, kv_a_pad_size: int) -> None:
    """Keep this projection's checkpoint FP8 shard + block scales. NEVER raises; a problem is recorded on the entry
    (the group then refuses at the boot self-test). Parameters that install() did not register are ignored (a KDA
    layer's o_proj is BF16 in this checkpoint and holds nothing)."""
    try:
        st = _state()
        ent = st.by_param.get(id(param))
        if ent is None or ent.param is not param or st.finalized or st.disabled is not None:
            return
        try:
            _capture(ent, key, weight_fp8, scale_inv, int(kv_a_pad_size))
        except Exception as e:  # noqa: BLE001
            ent.bad = ent.bad or f"capture({key}): {type(e).__name__}: {str(e)[:300]}"
    except Exception:  # noqa: BLE001
        pass


def _capture(ent: _Entry, key: str, weight_fp8, scale_inv, kv_a_pad_size: int) -> None:
    ent.calls += 1
    if kv_a_pad_size > 0:
        raise ValueError(f"kv_a_pad_size {kv_a_pad_size} > 0 (the pad rows have no scale)")
    if weight_fp8.dtype != torch.float8_e4m3fn or weight_fp8.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(f"checkpoint tensors {weight_fp8.dtype} {tuple(weight_fp8.shape)} / "
                         f"{tuple(scale_inv.shape)} are not 2-D float8_e4m3fn + 2-D scale")
    cn, ck = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    if cn % BLOCK or ck % BLOCK or tuple(scale_inv.shape) != (cn // BLOCK, ck // BLOCK):
        raise ValueError(f"checkpoint {cn}x{ck} with scale {tuple(scale_inv.shape)} is not a {BLOCK}x{BLOCK} block "
                         "layout (partial blocks unsupported)")
    N, K = ent.n, ent.k
    dev = ent.param.device
    if ent.w8 is None:
        ent.w8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=dev)
        ent.sc = torch.empty((N // BLOCK, K // BLOCK), dtype=torch.float32, device=dev)
    if ent.typ == "qkv_a":
        rows = {"q_a": (0, ent.n_q), "kv_a": (ent.n_q, ent.n)}
        if key not in rows:
            raise ValueError(f"unexpected checkpoint part {key!r} for the fused q_a|kv_a projection")
        r0, r1 = rows[key]
        if (cn, ck) != (r1 - r0, K):
            raise ValueError(f"{key} checkpoint {cn}x{ck} != {r1 - r0}x{K}")
        src_w, src_s = weight_fp8, scale_inv
        dst_w, dst_s = ent.w8[r0:r1], ent.sc[r0 // BLOCK:r1 // BLOCK]
    elif ent.typ == "q_b":
        if key != "q_b" or (cn, ck) != (N * ent.tp_size, K):
            raise ValueError(f"{key} checkpoint {cn}x{ck} != {N * ent.tp_size}x{K} (rank {ent.tp_rank}/{ent.tp_size})")
        r0 = ent.tp_rank * N
        s0 = ent.tp_rank * (N // BLOCK)
        src_w, src_s = weight_fp8[r0:r0 + N], scale_inv[s0:s0 + N // BLOCK]
        dst_w, dst_s = ent.w8, ent.sc
    elif ent.typ == "o_proj":
        if key != "o_proj" or (cn, ck) != (N, K * ent.tp_size):
            raise ValueError(f"{key} checkpoint {cn}x{ck} != {N}x{K * ent.tp_size} (rank {ent.tp_rank}/{ent.tp_size})")
        c0 = ent.tp_rank * K
        t0 = ent.tp_rank * (K // BLOCK)
        src_w, src_s = weight_fp8[:, c0:c0 + K], scale_inv[:, t0:t0 + K // BLOCK]
        dst_w, dst_s = ent.w8, ent.sc
    else:
        raise ValueError(f"unknown projection type {ent.typ}")
    if tuple(src_w.shape) != tuple(dst_w.shape) or tuple(src_s.shape) != tuple(dst_s.shape):
        raise ValueError(f"{key} shard {tuple(src_w.shape)} / {tuple(src_s.shape)} != target "
                         f"{tuple(dst_w.shape)} / {tuple(dst_s.shape)}")
    dst_w.copy_(src_w)                      # explicit copies into our own storage (SF5: no view pins the checkpoint)
    dst_s.copy_(src_s.to(torch.float32))
    ent.have.add(key)


# ---------------------------------------------------------------------------------------------------------------------
# Per-module process_weights_after_loading: V1-V4 + the Marlin build
# ---------------------------------------------------------------------------------------------------------------------
def _module_pwal(p: _Proj, layer) -> None:
    """NEVER raises. The last module of a group runs the group self-test."""
    st = _state()
    g = st.groups[p.group]
    if st.disabled is not None and st.finalized:
        return
    if p.processed:                                    # a second load into an already decided model
        if p.mode == ACTIVE:
            _retire_after_reload(st, p)
        return
    p.processed = True
    try:
        with torch.no_grad():
            _build(p, layer)
    except Exception as e:  # noqa: BLE001
        p.bad = p.bad or f"{type(e).__name__}: {_safe(str(e)[:300])}"
    g.done += 1
    if g.done == len(g.projs) and not g.complete:
        g.complete = True
        try:
            with torch.no_grad():
                _group_complete(st, g)
        except Exception as e:  # noqa: BLE001 - last resort: fail closed on this rank
            _decide_local_off(st, f"unexpected {type(e).__name__}: {_safe(str(e)[:300])}")


def _build(p: _Proj, layer) -> None:
    """V1-V4, then the Marlin group-128 copy. Raises on any problem (recorded as p.bad)."""
    st = _state()
    ent = p.entry
    w = layer.weight
    if ent.param is not w or st.by_param.get(id(w)) is not ent:
        raise RuntimeError("layer.weight is not the Parameter that install() registered")
    if ent.bad:
        raise RuntimeError(ent.bad)
    need = {"qkv_a": {"q_a", "kv_a"}, "q_b": {"q_b"}, "o_proj": {"o_proj"}}[p.typ]
    if ent.have != need or ent.w8 is None:
        raise RuntimeError(f"checkpoint parts captured {sorted(ent.have)}, expected {sorted(need)} "
                           "(no FP8 checkpoint tensors reached the loader hook)")
    N, K = p.n, p.k
    if tuple(w.shape) != (N, K) or w.dtype != torch.bfloat16:
        raise RuntimeError(f"layer.weight {tuple(w.shape)} {w.dtype} != ({N}, {K}) bfloat16")
    # V1 storage: the capture holds exactly its shard (not a view of a checkpoint tensor)
    if _nbytes(ent.w8) != N * K or _nbytes(ent.sc) != 4 * (N // BLOCK) * (K // BLOCK):
        raise RuntimeError(f"captured storage {_nbytes(ent.w8)} / {_nbytes(ent.sc)} bytes != the shard "
                           f"{N * K} / {4 * (N // BLOCK) * (K // BLOCK)}")
    sc = ent.sc
    # V4 scales
    if not bool(torch.isfinite(sc).all().item()):
        raise RuntimeError("V4: non-finite block scale")
    if float(sc.amin().item()) < 0.0 or float(sc.amax().item()) >= SCALE_MAX:
        raise RuntimeError(f"V4: block scale outside [0, {SCALE_MAX}): min {float(sc.amin().item()):.4g} max "
                           f"{float(sc.amax().item()):.4g}")
    bad2 = bad3 = nan = 0
    for r0 in range(0, N, CHUNK_ROWS):
        r1 = min(N, r0 + CHUNK_ROWS)
        w8 = ent.w8[r0:r1]
        nan += _count(((_u8(w8) & FP8_NAN_MASK) == FP8_NAN_MASK))
        s_full = _expand_scales(sc, r0, r1, K)
        served = (w8.to(torch.float32) * s_full).to(torch.bfloat16)
        # V2 capture == served, bitwise on int views (NaN / -0.0 safe)
        bad2 += _count(_i16(served) != _i16(w[r0:r1]))
        # V3 independent re-quantisation (SF4): only where the lemma holds; tiny / zero-scale blocks must serve ~0
        small = s_full < V3_MIN_SCALE
        safe = torch.where(small, torch.ones_like(s_full), s_full)
        enc = (served.to(torch.float32) / safe).to(torch.float8_e4m3fn)
        bad3 += _count((_u8(enc) != _u8(w8)) & ~small)
        bad3 += _count(small & (served.to(torch.float32).abs() > V3_TINY))
    if nan:
        raise RuntimeError(f"V4: {nan} NaN codes in the captured FP8 weight")
    if bad2:
        raise RuntimeError(f"V2: captured FP8 x scale differs from the served BF16 weight in {bad2} elements "
                           "(wrong shard / order / stale scale)")
    if bad3:
        raise RuntimeError(f"V3: re-quantising the served weight reproduces the captured FP8 bytes in all but {bad3} "
                           "elements")
    # build (exactly bench_mla.py prep("marlin")): a scratch holder, never the real layer
    MU8 = _marlin_fp8()
    H = torch.nn.Module()
    H.weight = torch.nn.Parameter(ent.w8, requires_grad=False)
    H.weight_scale_inv = torch.nn.Parameter(sc, requires_grad=False)
    H.orig_dtype = torch.bfloat16
    H.output_size_per_partition = N
    H.input_size_per_partition = K
    H.weight_block_size = [BLOCK, BLOCK]
    MU8.prepare_fp8_layer_for_marlin(H, size_k_first=False)
    p.w, p.s = H.weight.data, H.weight_scale_inv.data
    p.ws = _make_workspace(p.w.device)             # v0.32.0: not attached to the holder any more (see the header)
    del H
    if tuple(p.w.shape) != (K // 16, N * 4) or p.w.dtype != torch.int32:
        raise RuntimeError(f"Marlin weight {tuple(p.w.shape)} {p.w.dtype} != ({K // 16}, {N * 4}) int32")
    if tuple(p.s.shape) != (K // BLOCK, N) or p.s.dtype != torch.bfloat16:
        raise RuntimeError(f"Marlin scales {tuple(p.s.shape)} {p.s.dtype} != ({K // BLOCK}, {N}) bfloat16")
    if p.ws.dtype != torch.int32 or int(p.ws.numel()) < 1:
        raise RuntimeError("Marlin workspace is not a non-empty int32 tensor")
    p.built_bytes = _nbytes(p.w) + _nbytes(p.s) + _nbytes(p.ws)


# ---------------------------------------------------------------------------------------------------------------------
# The boot self-test of one group (local; C1-C9)
# ---------------------------------------------------------------------------------------------------------------------
class _Rows:
    """Pass/fail rows of this rank (plain data: exchanged over gloo)."""

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


@contextlib.contextmanager
def _probing(projs: list[_Proj]):
    """The ONLY place a module is in the probe state: restored to pending on every path (MF1)."""
    try:
        for p in projs:
            p.mode, p.hits = PROBE, 0
        yield
    finally:
        for p in projs:
            p.mode, p.hits = PENDING, 0


def _case_list(max_m: int) -> list[tuple[str, int]]:
    ms = sorted({m for m in SELFTEST_M if m <= max_m} | {max_m})
    return [(d, m) for d in DRAWS for m in ms]


def _refs(p: _Proj, X):
    """fp32 [R, N] references for all rows of X: x @ (fp8 x fp32 scale)^T and x @ (fp8 x bf16(scale))^T, row chunks.

    Computed in float64 (cuBLAS DGEMM; R ~ 120 rows x N x K per module is negligible at boot) and rounded ONCE to fp32, so the
    references are exact to fp32 rounding whatever the process's TF32 / matmul-precision setting is: this module never reads or
    writes torch.backends.*.matmul.* (in a process that has used the new fp32_precision API, even READING the legacy
    allow_tf32 flag raises RuntimeError, which would have refused the patch on every rank for no numerical reason)."""
    ent = p.entry
    R = int(X.shape[0])
    Xd = X.to(torch.float64)
    r32 = torch.empty((R, p.n), dtype=torch.float32, device=X.device)
    rdq = torch.empty((R, p.n), dtype=torch.float32, device=X.device)
    sbf = ent.sc.to(torch.bfloat16).to(torch.float32)
    for r0 in range(0, p.n, CHUNK_ROWS):
        r1 = min(p.n, r0 + CHUNK_ROWS)
        f = ent.w8[r0:r1].to(torch.float64)
        r32[:, r0:r1] = torch.nn.functional.linear(
            Xd, f * _expand_scales(ent.sc, r0, r1, p.k).to(torch.float64)).to(torch.float32)
        rdq[:, r0:r1] = torch.nn.functional.linear(
            Xd, f * _expand_scales(sbf, r0, r1, p.k).to(torch.float64)).to(torch.float32)
    return r32, rdq


def _same_outcome(f1, f2) -> tuple[bool, str]:
    """Both calls raise the same exception type, or both return bitwise equal results."""
    r1 = r2 = e1 = e2 = None
    try:
        r1 = f1()
    except Exception as e:  # noqa: BLE001
        e1 = e
    try:
        r2 = f2()
    except Exception as e:  # noqa: BLE001
        e2 = e
    if e1 is not None or e2 is not None:
        return (type(e1) is type(e2)), f"{type(e1).__name__ if e1 else 'ok'} vs {type(e2).__name__ if e2 else 'ok'}"
    mm = _mismatches(r1, r2)
    return mm == 0, f"{mm} elements differ"


def _check_proj(p: _Proj, rows: _Rows, worst: dict, seam: dict, digests: dict) -> None:
    """C1-C8 of one module, in the probe state (the caller wraps this in _probing)."""
    dev = p.module.weight.device
    layer = p.module
    cases = _case_list(p.max_m)
    xs = [_synth(d, m, p.k, SEED + 1000 * (1 + DRAWS.index(d)) + m, dev) for d, m in cases]
    X = torch.cat(xs, dim=0)
    r32, rdq = _refs(p, X)
    off = 0
    w = worst.setdefault(p.typ, {"c1": 0.0, "c2": 0.0, "c3": 0.0, "c4": 0.0, "c5": 1.0})
    for (d, m), x in zip(cases, xs):
        tag = f"{p.label} {d} M={m}"
        h0 = p.hits
        y = p.method.apply(layer, x)
        y2 = p.method.apply(layer, x)
        rows.check(f"C8 {tag}: Marlin dispatched twice", p.hits == h0 + 2, f"hits {p.hits - h0}")
        yb = _bf16(p, x)
        a32, adq = r32[off:off + m], rdq[off:off + m]
        off += m
        ok_shape = (tuple(y.shape) == (m, p.n) and y.dtype == torch.bfloat16 and y.is_contiguous())
        rows.check(f"C6 {tag}: output [M, N] bf16 contiguous", ok_shape, f"{tuple(y.shape)} {y.dtype}")
        if not ok_shape:
            continue
        rows.check(f"C6 {tag}: finite", bool(torch.isfinite(y.float()).all().item()))
        rows.check(f"C6 {tag}: deterministic", _mismatches(y, y2) == 0, f"{_mismatches(y, y2)} elements")
        c1, c2, c3, c4 = _rel(y, a32), _rel(y, yb), _rel(y, adq), _rel(yb, a32)
        yf, af = y.float(), a32.float()
        den = (yf * yf).sum(dim=1).sqrt() * (af * af).sum(dim=1).sqrt()
        cos = torch.where(den > 0.0, (yf * af).sum(dim=1) / den.clamp_min(1e-30), torch.ones_like(den)).amin()
        c5 = float(cos.item())                       # an all-zero row on both sides (a zero-scale block row) is cos 1
        rows.check(f"C1 {tag}: Marlin vs fp32 relRMSE <= {C1_MAX}", c1 <= C1_MAX, f"{c1:.5f}")
        rows.check(f"C2 {tag}: Marlin vs BF16 path relRMSE <= {C2_MAX}", c2 <= C2_MAX, f"{c2:.5f}")
        rows.check(f"C3 {tag}: Marlin vs its own dequant relRMSE <= {C3_MAX}", c3 <= C3_MAX, f"{c3:.5f}")
        rows.check(f"C4 {tag}: BF16 path vs fp32 relRMSE <= {C4_MAX}", c4 <= C4_MAX, f"{c4:.5f}")
        rows.check(f"C5 {tag}: min row cosine >= {C5_MIN}", c5 >= C5_MIN, f"{c5:.7f}")
        w["c1"], w["c2"], w["c3"], w["c4"] = max(w["c1"], c1), max(w["c2"], c2), max(w["c3"], c3), max(w["c4"], c4)
        w["c5"] = min(w["c5"], c5)
    _check_dispatch(p, rows, dev)
    _check_seam(p, rows, seam, dev)
    if p.typ == "qkv_a":
        for m in sorted({min(m, p.max_m) for m in DIGEST_M}):
            x = _synth("unit", m, p.k, SEED + 5000 + m, dev)
            digests.setdefault("out", []).append(f"{p.layer_idx}/{m}:{_sha16_tensor(p.method.apply(layer, x))}")


def _check_dispatch(p: _Proj, rows: _Rows, dev) -> None:
    """C7: the dispatch rule and its fidelity (probe state: a wrong branch would show as a hit or a different result)."""
    layer, K = p.module, p.k
    h0 = p.hits
    for m in (p.max_m + 1, 2 * p.max_m):
        x = _synth("unit", m, K, SEED + 6000 + m, dev)
        ok, why = _same_outcome(lambda: p.method.apply(layer, x), lambda: _bf16(p, x))
        rows.check(f"C7 {p.label} M={m} (> MAX_M) is the BF16 path bitwise", ok, why)
    x3 = _synth("unit", 3, K, SEED + 6100, dev)
    bias = torch.zeros(p.n, dtype=torch.bfloat16, device=dev)
    ok, why = _same_outcome(lambda: p.method.apply(layer, x3, bias), lambda: _bf16(p, x3, bias))
    rows.check(f"C7 {p.label} bias guard is the BF16 path bitwise", ok, why)
    xd = x3.to(torch.float32)
    ok, why = _same_outcome(lambda: p.method.apply(layer, xd), lambda: _bf16(p, xd))
    rows.check(f"C7 {p.label} dtype guard is the BF16 path (same outcome)", ok, why)
    x3d = x3.reshape(1, 3, K)
    ok, why = _same_outcome(lambda: p.method.apply(layer, x3d), lambda: _bf16(p, x3d))
    rows.check(f"C7 {p.label} 3-D guard is the BF16 path bitwise", ok, why)
    rows.check(f"C7 {p.label} no hit counted by any guard / M > MAX_M call", p.hits == h0, f"{p.hits - h0} hits")
    try:
        z = p.method.apply(layer, torch.zeros((0, K), dtype=torch.bfloat16, device=dev))
        rows.check(f"C7 {p.label} M=0 returns an empty result", int(z.shape[0]) == 0 and p.hits == h0)
    except Exception as e:  # noqa: BLE001
        rows.check(f"C7 {p.label} M=0 does not raise", False, f"{type(e).__name__}: {str(e)[:120]}")
    big = _synth("unit", 3, K + 64, SEED + 6200, dev)
    xs = big[:, :K]
    rows.check(f"C7 {p.label} strided input is not contiguous", not xs.is_contiguous())
    h1 = p.hits
    ys, yc = p.method.apply(layer, xs), p.method.apply(layer, xs.contiguous())
    rows.check(f"C7 {p.label} strided input == its contiguous copy bitwise (both Marlin)",
               _mismatches(ys, yc) == 0 and p.hits == h1 + 2, f"{_mismatches(ys, yc)} differ, {p.hits - h1} hits")


def _check_seam(p: _Proj, rows: _Rows, seam: dict, dev) -> None:
    """C8: the REAL module call reaches quant_method.apply exactly once. A failure marks only this TYPE (qkv_a falls back
    alone); the other checks of the module still ran."""
    layer = p.module
    x = _synth("unit", 3, p.k, SEED + 7000, dev)
    old = None
    had = hasattr(layer, "reduce_results")
    try:
        if p.typ == "o_proj" and had:
            old = layer.reduce_results
            layer.reduce_results = False                       # no all-reduce at boot (restored below)
        h0 = p.hits
        out = layer(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        direct = p.method.apply(layer, x)
        hits = p.hits - h0 - 1                                  # the direct call above counted one
        if hits != 1:
            raise RuntimeError(f"the module call reached quant_method.apply {hits} times (expected exactly 1)")
        if _mismatches(out, direct) != 0:
            raise RuntimeError("the module call result differs from the Marlin result")
    except Exception as e:  # noqa: BLE001
        seam.setdefault(p.typ, f"{p.label}: {type(e).__name__}: {_safe(str(e)[:200])}")
    finally:
        if p.typ == "o_proj" and had:
            layer.reduce_results = old


def _atomic_add_refusal(p: _Proj) -> str | None:
    MU = _marlin_utils()
    ms = sorted({m for m in SELFTEST_M if m <= p.max_m} | {p.max_m, p.max_m + 1, 2 * p.max_m})
    dev = p.module.weight.device
    for m in ms:
        if MU.should_use_atomic_add_reduce(m=m, n=p.n, k=p.k, device=dev, dtype=torch.bfloat16):
            return (f"should_use_atomic_add_reduce is True for M={m}, N={p.n}, K={p.k} (the atomic-add Marlin variant "
                    "is not what E3 measured nor what graph capture was reviewed for)")
    return None


def _content_digest(p: _Proj) -> str:
    return f"{p.layer_idx}:{_digest_int(p.entry.w8)}|{_digest_int(p.entry.sc.view(torch.int32))}"


def _fingerprint(st: _State, g: _Group, world) -> str:
    dims = ",".join(f"{t}:{next((p.n for p in g.projs if p.typ == t), 0)}x{next((p.k for p in g.projs if p.typ == t), 0)}"
                    for t in TYPES)
    parts = ["p033", f"rev={REVISION}", f"module={MODULE_SHA16}", f"target={_sha16_file(_target_file())}",
             f"gate={os.environ.get(ENV, '0').strip()}", f"group={g.name}", f"maxm={st.max_m}", f"speed={st.speed}",
             f"tp={world.size}", f"pp={getattr(world, 'pp_size', '?')}", f"dims={dims}",
             "layers=" + ",".join(str(p.layer_idx) for p in g.projs if p.typ == "qkv_a"),
             f"selftest={SELFTEST_M}", f"draws={DRAWS}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _target_file() -> str | None:
    m = sys.modules.get("vllm.models.glm5next.common.model")
    return getattr(m, "__file__", None)


def _dep_sha16s() -> str:
    """SF9: the modules the design depends on and the .sh only records (their call patterns are covered by the boot self-test:
    V2 for the loaders' narrow rules, C8 for the module -> quant_method.apply seam), recorded at every real boot."""
    out = []
    for label, name in (("deepseek_v2.py", "vllm.model_executor.models.deepseek_v2"),
                        ("layers/mla.py", "vllm.model_executor.layers.mla"),
                        ("parameter.py", "vllm.model_executor.parameter")):
        m = sys.modules.get(name)
        f = getattr(m, "__file__", None)
        out.append(f"{label} {_sha16_file(f) if f else 'not loaded'}")
    return ", ".join(out)


def _selftest_group(st: _State, g: _Group, world) -> dict:
    """Everything local, no collective. Returns this rank's plain-data row."""
    rows = _Rows()
    worst: dict = {}
    seam: dict = {}
    digests: dict = {}
    row = {"rank": world.rank, "group": g.name, "fp": None, "ok": False, "why": "", "type_ok": {t: False for t in TYPES},
           "type_why": {}, "qkv_digest": "", "out_digest": "", "worst": {}, "bytes": 0, "speed": None}
    try:
        row["fp"] = _fingerprint(st, g, world)
        for p in g.projs:
            if p.bad:
                rows.check(f"build {p.label}", False, p.bad)
        if rows.ok:
            for p in g.projs:
                why = _atomic_add_refusal(p)
                rows.check(f"atomic-add gate {p.label}", why is None, why or "")
        if rows.ok:
            pre = [_wdigest(p.module.weight) for p in g.projs]
            tally = sum(p.built_bytes for p in g.projs)
            with _probing(g.projs):
                for p in g.projs:
                    try:
                        _check_proj(p, rows, worst, seam, digests)
                    except Exception as e:  # noqa: BLE001 - a local failure of this module (never skips the exchange)
                        rows.check(f"C0 {p.label}", False, f"{type(e).__name__}: {_safe(str(e)[:200])}")
            post = [_wdigest(p.module.weight) for p in g.projs]
            rows.check("C9 layer.weight untouched by the self-test", pre == post)
            rows.check("C9 every module back in pending", all(p.mode == PENDING and p.hits == 0 for p in g.projs))
            rows.check(f"C9 memory tally <= {MEM_LIMIT_BYTES / 2 ** 30:.1f} GiB", tally <= MEM_LIMIT_BYTES,
                       f"{tally / 2 ** 30:.3f} GiB")
            row["bytes"] = tally
            row["qkv_digest"] = ";".join(_content_digest(p) for p in g.projs if p.typ == "qkv_a")
            row["out_digest"] = ";".join(digests.get("out", []))
            for t in TYPES:
                present = any(p.typ == t for p in g.projs)
                row["type_ok"][t] = present and t not in seam
                if t in seam:
                    row["type_why"][t] = seam[t]
            if g.name == "target" and rows.ok:
                row["speed"] = _speed_gate(st, g, rows)
    except Exception as e:  # noqa: BLE001
        rows.check("C0 self-test", False, f"{type(e).__name__}: {_safe(str(e)[:200])}")
    row["worst"] = {t: {k: round(v, 5) for k, v in d.items()} for t, d in worst.items()}
    row["ok"] = rows.ok
    row["why"] = _safe("; ".join(rows.fail[:4]))[:600]
    return row


# ---------------------------------------------------------------------------------------------------------------------
# SF6: the speed gate, inside CUDA graphs on a side stream (DRAM-cold: 11 layers x 39.8 MB >> the 24 MB L2)
# ---------------------------------------------------------------------------------------------------------------------
def _new_pool():
    try:
        return torch.cuda.graph_pool_handle()
    except Exception:  # noqa: BLE001
        return None


def _unregister_pool(pool, dev) -> str:
    """After a FAILED capture only (029 rev 2): torch's capture_end raises before unregistering an invalidated capture's
    private pool from the caching allocator; vLLM's next torch.cuda.empty_cache() would then assert."""
    if pool is None:
        return "no pool handle"
    end = getattr(getattr(torch, "_C", None), "_cuda_endAllocateToPool", None)
    if end is None:
        return "no _cuda_endAllocateToPool binding"
    try:
        idx = int(dev.index) if getattr(dev, "index", None) is not None else int(torch.cuda.current_device())
        end(idx, pool)
    except Exception:  # noqa: BLE001 - not registered: capture_end unregistered it
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


def _capture_graph(calls: list, stream, dev):
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


def _median(xs) -> float:
    s = sorted(float(v) for v in xs)
    return s[len(s) // 2] if s else float("nan")


def _speed_gate(st: _State, g: _Group, rows: _Rows) -> dict:
    """One graph per arm over EVERY module of the group at M=3 (BF16 path = the image GEMM, Marlin path), interleaved
    replays, medians. Enforced in mode 'graph' (Marlin <= 0.85 x BF16); 'info' only logs."""
    projs = g.projs
    dev = projs[0].module.weight.device
    info = {"mode": st.speed, "layers": len({p.layer_idx for p in projs})}
    side = ctx = None
    entered = False
    graphs: list = []
    try:
        xs = {id(p): _synth("unit", min(SPEED_M, p.max_m), p.k, SEED + 8000, dev) for p in projs}
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream())
        ctx = torch.cuda.stream(side)
        ctx.__enter__()
        entered = True
        arms = (("marlin", [(lambda p=p: _marlin(p, xs[id(p)])) for p in projs]),
                ("bf16", [(lambda p=p: _bf16(p, xs[id(p)])) for p in projs]))
        for _name, calls in arms:                        # eager warm-up (cuBLAS workspace, lazy kernels)
            for c in calls:
                c()
        torch.cuda.synchronize()
        for _name, calls in arms:
            graphs.append(_capture_graph(calls, side, dev))
        evs = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(SPEED_ITERS)] for _ in graphs]
        for i in range(SPEED_WARMUP + SPEED_ITERS):
            j = i - SPEED_WARMUP
            for k, gr in enumerate(graphs):
                if j >= 0:
                    evs[k][j][0].record()
                gr.replay()
                if j >= 0:
                    evs[k][j][1].record()
        torch.cuda.synchronize()
        t_m = _median(a.elapsed_time(b) for a, b in evs[0])
        t_b = _median(a.elapsed_time(b) for a, b in evs[1])
        ratio = t_m / t_b if t_b > 0 else float("inf")
        info.update({"marlin_ms": round(t_m, 4), "bf16_ms": round(t_b, 4), "ratio": round(ratio, 4),
                     "saved_ms": round(t_b - t_m, 4), "e3_ratio": round(E3_REF_US[1] / E3_REF_US[0], 3)})
        if st.speed == "graph":
            rows.check(f"S1 CUDA-graph sweep: Marlin <= {SPEED_RATIO_MAX} x BF16 at M={SPEED_M}",
                       math.isfinite(ratio) and t_m > 0 and ratio <= SPEED_RATIO_MAX,
                       f"{t_m:.3f} vs {t_b:.3f} ms per sweep, ratio {ratio:.3f}; E3 reference {E3_REF_US[1]} vs "
                       f"{E3_REF_US[0]} us per layer; set {ENV_SPEED}=info to log without enforcing")
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {_safe(str(e)[:200])}"
        if st.speed == "graph":
            rows.check("S1 CUDA-graph sweep ran", False, f"{info['error']}; set {ENV_SPEED}=info to log without "
                                                        "enforcing")
    finally:
        for gr in graphs:
            _reset_graph(gr)
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
    return info


# ---------------------------------------------------------------------------------------------------------------------
# Group completion, exchange, verdict, commit
# ---------------------------------------------------------------------------------------------------------------------
def _exchange(world, obj, what: str):
    try:
        world.barrier_bounded(SELFTEST_BARRIER_S)
        out = world.all_gather_object(obj)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s boot %s exchange failed on this rank (%s: %s); the patch stays inactive here", TAG, what,
                       type(e).__name__, _safe(str(e)[:300]))
        return None
    if not isinstance(out, list) or len(out) != world.size or any(not isinstance(r, dict) for r in out):
        logger.warning("%s boot %s exchange returned malformed rows; the patch stays inactive here", TAG, what)
        return None
    return out


def _group_verdict(rows: list | None, size: int, name: str) -> dict:
    """Pure function of the gathered rows: identical on every rank that saw the same rows."""
    v = {"ok": False, "why": [], "types": [], "fp": None, "worst": {}, "bytes": 0, "speed": None}
    if rows is None:
        v["why"].append(f"{name} group exchange failed on this rank")
        return v
    ranks = sorted(int(r.get("rank", -1)) for r in rows)
    if ranks != list(range(size)):
        v["why"].append(f"{name} group ranks {ranks}")
        return v
    fps = {str(r.get("fp")) for r in rows}
    if len(fps) != 1 or "None" in fps:
        v["why"].append(f"{name} group fingerprint mismatch {sorted(fps)} (module / model.py / env / shapes)")
        return v
    bad = [r for r in rows if not r.get("ok")]
    if bad:
        b = sorted(bad, key=lambda r: int(r["rank"]))
        v["why"].append(f"{name} group self-test failed on rank(s) {[r['rank'] for r in b]}: {b[0].get('why')}")
        return v
    if len({str(r.get("qkv_digest")) for r in rows}) != 1:
        v["why"].append(f"{name} group: the replicated qkv_a checkpoint FP8 differs across ranks")
        return v
    if len({str(r.get("out_digest")) for r in rows}) != 1:
        v["why"].append(f"{name} group: the replicated qkv_a Marlin outputs differ across ranks")
        return v
    types = [t for t in TYPES if all(r.get("type_ok", {}).get(t) for r in rows)]
    for t in TYPES:
        if t not in types:
            why = next((r.get("type_why", {}).get(t) for r in rows if r.get("type_why", {}).get(t)), "not installed")
            v["why"].append(f"{t} not active ({name} group seam check: {why})")
    if not types:
        return v
    worst: dict = {}
    for r in rows:
        for t, d in (r.get("worst") or {}).items():
            w = worst.setdefault(t, dict(d))
            for k, x in d.items():
                w[k] = min(w[k], x) if k == "c5" else max(w[k], x)
    v.update({"ok": True, "types": types, "fp": next(iter(fps)), "worst": worst,
              "bytes": max(int(r.get("bytes", 0)) for r in rows), "speed": rows[0].get("speed")})
    return v


def _group_complete(st: _State, g: _Group) -> None:
    """The exchange schedule is fixed by the CONFIG, not by local outcomes: target group, MTP group (when expected),
    commit round - every rank issues all of them whatever failed on it (a failed group only turns its row, and the
    final decision, to refuse), so a local failure can never leave the other ranks waiting for a missing exchange."""
    try:
        world = _make_world()
    except Exception as e:  # noqa: BLE001 - no TP group: nothing collective can run (identically on every rank)
        _decide_local_off(st, f"no TP group ({type(e).__name__}: {_safe(str(e)[:200])})")
        return
    st.rank, st.size = str(world.rank), str(world.size)
    if st.failed and g.name == "mtp":
        row = {"rank": world.rank, "group": g.name, "fp": "skipped", "ok": False,
               "why": "self-test not run: the target group was refused", "type_ok": {t: False for t in TYPES},
               "type_why": {}, "qkv_digest": "", "out_digest": "", "worst": {}, "bytes": 0, "speed": None}
    else:
        t0 = time.perf_counter()
        row = _selftest_group(st, g, world)
        g.selftest_s = time.perf_counter() - t0
    rows = _exchange(world, row, f"{g.name} group")
    g.verdict = _group_verdict(rows, world.size, g.name)
    for p in g.projs:                                   # the raw FP8 is only needed by the self-test
        p.entry.w8 = p.entry.sc = None
    _empty_cache()
    if not g.verdict["ok"]:
        st.failed += list(g.verdict["why"])
    if g.name == "target" and st.expect_mtp and not st.groups["mtp"].complete:
        if g.verdict["ok"]:
            logger.info("%s target MLA group built and self-tested on rank %d/%d (%d projections, %.3f GiB FP8 copies, "
                        "worst relRMSE vs fp32 %s, self-test %.1f s); waiting for the MTP draft group before the boot "
                        "decision; recorded (covered by the self-test, not pinned): %s", TAG, world.rank, world.size,
                        len(g.projs), g.verdict["bytes"] / 2 ** 30, _worst_s(g.verdict["worst"], "c1"), g.selftest_s,
                        _dep_sha16s())
        else:
            logger.info("%s target MLA group refused on rank %d/%d (%s); the MTP group's exchange and the commit round "
                        "still follow, then the image BF16 GEMMs stay", TAG, world.rank, world.size,
                        _safe("; ".join(g.verdict["why"])))
        return
    _decide(st, world)


def _worst_s(worst: dict, key: str) -> str:
    return "/".join(f"{t} {worst[t][key]:.5f}" for t in TYPES if t in worst) or "n/a"


def _decide(st: _State, world) -> None:
    """Every expected group is in: the decision is a pure function of the verdicts (any refusal refuses all; the active
    types are the types every group's seam check passed), then the commit round."""
    vs = [st.groups["target"].verdict]
    if st.groups["mtp"].projs:
        vs.append(st.groups["mtp"].verdict)
    why: list[str] = []
    for v in vs:
        why += [w for w in (v or {}).get("why", []) if w not in why]
    types = [t for t in TYPES if all(v and v.get("ok") and t in v["types"] for v in vs)]
    if st.failed or any(not (v and v.get("ok")) for v in vs):
        types = []
    if st.expect_mtp and not st.groups["mtp"].projs:
        why.append("the MTP draft layer was expected but never installed")
        types = []
    _finalize(st, world, types, why)


def _empty_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _decide_local_off(st: _State, why: str) -> None:
    if st.finalized:
        return
    st.disabled = st.disabled or why
    _restore(st, _all_projs(st))
    st.finalized = True
    st.final = {"types": [], "why": [why]}
    _empty_cache()
    _log_inactive(st, [why])


def _finalize(st: _State, world, types: list, why: list) -> None:
    if st.finalized:
        return
    fp_final = hashlib.sha256(("|".join([str(st.groups["target"].verdict and st.groups["target"].verdict.get("fp")),
                                         ",".join(types)])).encode()).hexdigest()[:16]
    com = _exchange(world, {"rank": world.rank, "types": list(types), "fp": fp_final}, "commit")
    if com is None or len({(tuple(r.get("types", ())), str(r.get("fp"))) for r in com}) != 1:
        if types:
            why = list(why) + ["commit round failed or the ranks' decisions differ"]
        types = []
    st.finalized = True
    st.final = {"types": list(types), "why": list(why), "fp": fp_final}
    projs = _all_projs(st)
    keep = [p for p in projs if p.typ in types and p.bad is None]
    drop = [p for p in projs if p not in keep]
    _restore(st, drop)
    for p in keep:
        p.mode = ACTIVE
    _empty_cache()
    if keep:
        _log_active(st, world, types, why, fp_final)
    else:
        st.disabled = st.disabled or "; ".join(why) or "no projection type active"
        _log_inactive(st, why or ["no projection type active"])


def _log_active(st: _State, world, types: list, why: list, fp: str) -> None:
    tv, mv = st.groups["target"].verdict or {}, st.groups["mtp"].verdict or {}
    nt = sum(1 for p in st.groups["target"].projs if p.typ in types)
    nm = sum(1 for p in st.groups["mtp"].projs if p.typ in types)
    layers = len({p.layer_idx for p in st.groups["target"].projs})
    mlayers = len({p.layer_idx for p in st.groups["mtp"].projs})
    nbytes = sum(p.built_bytes for p in _all_projs(st) if p.mode == ACTIVE)
    worst: dict = {}
    for v in (tv, mv):
        for t, d in (v.get("worst") or {}).items():
            w = worst.setdefault(t, dict(d))
            for k, x in d.items():
                w[k] = min(w[k], x) if k == "c5" else max(w[k], x)
    sp = tv.get("speed") or {}
    speed = (f"; speed gate ({sp.get('mode')}): Marlin {sp.get('marlin_ms')} ms vs BF16 {sp.get('bf16_ms')} ms per "
             f"{sp.get('layers')}-layer sweep at M={SPEED_M} (ratio {sp.get('ratio')}; E3 reference {sp.get('e3_ratio')})"
             if sp.get("ratio") is not None else ("; speed gate not run" if not sp else
                                                   f"; speed gate error {sp.get('error')}"))
    reasons = _safe(" | ".join(why)) if why else ""
    logger.info(
        "%s MLA FP8 W8A16 (Marlin) ACTIVE (rank %d/%d): decode M<=%d on %d target + %d MTP MLA layers, active types "
        "{%s} (%d projections, %.3f GiB FP8 copies, BF16 weights kept for M>%d); boot self-test worst relRMSE vs fp32 "
        "[%s] (BF16 path [%s]), vs BF16 [%s], vs own dequant [%s], cos_min [%s], deterministic, dispatch fidelity and "
        "seam hit exactly once on every module%s; qkv_a digests equal on %d ranks; self-test %.1f s target + %.1f s MTP on "
        "this rank; module rev %d, fingerprint %s%s",
        TAG, world.rank, world.size, st.max_m, layers, mlayers, ", ".join(t for t in TYPES if t in types), nt + nm,
        nbytes / 2 ** 30, st.max_m, _worst_s(worst, "c1"), _worst_s(worst, "c4"), _worst_s(worst, "c2"),
        _worst_s(worst, "c3"), _worst_s(worst, "c5"), speed, world.size, st.groups["target"].selftest_s,
        st.groups["mtp"].selftest_s, REVISION, fp,
        ("; not active: " + reasons) if reasons else "")


def _log_inactive(st: _State, why: list) -> None:
    logger.warning("%s MLA FP8 W8A16 INACTIVE (fail closed) (rank %s/%s): %s; the image BF16 GEMMs run unchanged "
                   "(module rev %d)", TAG, st.rank, st.size, _safe(" | ".join(str(w) for w in why)) or "refused",
                   REVISION)


# ---------------------------------------------------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------------------------------------------------
def _set_world_factory_for_test(factory) -> None:
    global _WORLD_FACTORY
    _WORLD_FACTORY = factory


def _set_ctx_key_for_test(fn) -> None:
    global _CTX_KEY_FN
    _CTX_KEY_FN = fn
    _STATES.clear()
