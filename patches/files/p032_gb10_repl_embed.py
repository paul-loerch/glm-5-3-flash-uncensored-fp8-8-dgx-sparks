# SPDX-License-Identifier: Apache-2.0
"""[patch-032 gb10-repl-embed] Replicated input embedding for GLM-5.3 (Glm5NextModel.embed_tokens), TP=8 on GB10.
(vLLM v0.32.0 port, module revision 2.)

Installed by patches/032-gb10-repl-embed.sh as vllm/_gb10_repl_embed.py. model.py (vllm/models/glm5next/common/model.py)
imports it only when VLLM_GB10_REPL_EMBED=1 (read once, at import) and then builds embed_tokens with
`embedding_class(config)(...)`.

v0.32.0 PORT (what changed against revision 1, written for the day-0 image's vocab_parallel_embedding.py): v0.32.0's
VocabParallelEmbedding.forward at TP > 1 is `ops.vocab_parallel_embedding` (ONE fused CUDA kernel: range mask, id shift,
row gather, zero rows for ids outside the shard; csrc/vocab_parallel_embedding_kernels.cu) followed by the TP all-reduce,
selected by `use_fused_embedding` (tp_size > 1, CUDA, UnquantizedEmbeddingMethod); the masked_fill path of the day-0 image
is its fallback. Revision 2 therefore no longer re-implements the lookup: both the replicated layer's forward and the boot
self-test's reference run the REAL `VocabParallelEmbedding.forward` of the pinned vocab_parallel_embedding.py on a small view
object (`_view`: tp_size, shard_indices, weight, quant_method, use_fused_embedding, embedding_dim, reduce_results,
parallel_group) - the same code, the same kernel, the same branch as the unpatched layer at TP=8 - with reduce_results=False
for the replicated lookup (no all-reduce) and the reference followed by the layer's own tensor_model_parallel_all_reduce.
Also new: `construction_refusal` refuses model_config.word_embeddings_untied_by_checkpoint (v0.32.0 may re-tie the lm_head to
the embedding after loading), `_P032EmbeddingMethod.supports_pre_processed_weights = False` (the boot transforms the table;
the v0.32.0 loader rejects such a method under weights_already_processed()), `_to_vocab_parallel` restores the attributes
v0.32.0's constructor sets (output_size_per_partition, output_partition_sizes, use_fused_embedding, reduce_results,
parallel_group), the fingerprint covers the fused-kernel decision, and the model file it fingerprints is
vllm.models.glm5next.common.model. mtp.py needs no edit (see WHAT, last bullet). The boot protocol, plan, probe ids, log
lines and fail-closed rules are unchanged.

WHY (8-rank profile of the previous base): every decode step replays 3 CUDA graphs (target, draft prefill, draft
decode). The FIRST NCCL kernel of each replay waits 0.5-0.9 ms on NCCL's per-plan host-callback node (GPU -> CPU -> GPU,
worsened by CPU idle-state wake-ups): ~2.05 ms/step + ~0.54 ms of slower first floors. In all three replays that first
NCCL op is the vocab-parallel embedding all-reduce (one non-zero row per token, seven zero rows), issued 5-14 us into
the replay. With the table replicated the embedding needs no all-reduce, the replay's first NCCL op becomes the first
layer's all-reduce (244 / 742 / 742 us of compute later) and the host-callback round trip runs under that compute.
Modelled -1.8 to -2.2 ms/step (partly hidden only in the target and draft-1 passes), prefill -8.7 ms per
3,840-token chunk (the target's and the MTP's embedding all-reduces of 31.5 MB each). Cost: +1.03 GiB per rank.

WHAT:
  * embedding_class(config): ReplicatedEmbedding when the config is in the validated envelope (TP 8, PP 1, no LoRA,
    tie_word_embeddings False, vocab a multiple of 64 and of TP, bf16), else the image's VocabParallelEmbedding (an
    INACTIVE line, no extra memory).
  * ReplicatedEmbedding = VocabParallelEmbedding built with disable_tp=True: tp_size 1, shard_indices = the whole vocab
    ([0, V), no padding), weight [V, H] on every rank, loaded by the IMAGE weight_loader (narrow(0, 0, V): the full
    table). Its quant method (_P032EmbeddingMethod, an UnquantizedEmbeddingMethod) runs the collective boot self-test
    from process_weights_after_loading, i.e. once, after ALL weights are loaded, on every rank, in the loader's module
    order (model_loader/utils.py, pinned). Glm5NextModel.load_weights may be entered several times (AutoWeightsLoader
    groups CONSECUTIVE checkpoint keys), so no hook sits there.
  * forward, replicated mode: the REAL VocabParallelEmbedding.forward (vocab_parallel_embedding.py, pinned) on a view with
    the full-vocab shard indices ([0, V), no padding) and reduce_results=False, i.e. upstream's own TP>1 lookup WITHOUT the
    all-reduce: the fused `ops.vocab_parallel_embedding` kernel on CUDA (the masked-gather fallback otherwise). Out-of-range
    ids give zero rows exactly as in the unpatched layer (the kernel compares the ids as int64 against the shard range and
    writes `vec_t()` = all-zero bytes = +0.0 for every other row; never an out-of-bounds gather).
  * the MTP drafter needs no change and mtp.py is NOT edited: v0.32.0's load_eagle_model -> maybe_share_target_embed
    (v1/worker/gpu/spec_decode/eagle/utils.py, pinned) gives the draft the TARGET's embed_tokens object whenever the draft
    does not declare `has_own_embed_tokens` (_should_share returns True; Glm5NextMTP never sets it) and deletes the draft's
    own vocab-parallel copy (common/mtp.py:136, built before the share), so both draft passes use this module too. mtp.py is
    a read-only pinned contract (v0.32.0 original, or patch 029's pinned result).

EXACTNESS (re-stated for v0.32.0): for an in-range id t the unpatched layer's output is the all-reduce of W[t] (the
owner rank's fused-kernel row: a pure byte copy of the weight row, -0.0 elements included) and seven all-zero rows (the
kernel's `vec_t()` stores for ids outside the shard = +0.0). Patch 012's one-shot (fp32 acc = +0.0; acc += x_r + 0.0 in
rank order; one bf16 rounding) and NCCL's bf16 sum both return x + 0 = x exactly for every finite non-zero x, and +0.0 for
x = -0.0 (-0.0 + +0.0 = +0.0 in round-to-nearest). Out-of-range ids: eight zero rows -> +0.0. So image(t) == canon(W[t])
bitwise, where canon maps -0.0 -> +0.0 (and assumes no subnormal flushing in the reduction - checked, not assumed, below).
The boot therefore sets every -0.0 of the replicated table to +0.0 (masked_fill_ of the 0x8000 positions; nothing else is
written) and the replicated lookup (the same kernel on the full table, no reduction) is BITWISE the unpatched embedding
output for every int32 id. The lm_head (ParallelLMHead, its own "lm_head.weight", tie_word_embeddings False) is a
different module and is not touched.

BOOT SELF-TEST (process_weights_after_loading, every rank; FAIL CLOSED, hang-free):
  1. local preflight: TP 8 / PP 1 / rank as at construction, weight [V, H] bf16 CUDA contiguous, exactly one full-vocab
     weight_loader call, then a scan of the table (negative zeros, subnormals, non-finite values (refused), rows holding
     -0.0 / subnormals, an int64 content digest). gloo all-gather of {fingerprint, content, refusal}: any refusal, a
     fingerprint mismatch (module / model.py sha, env, shapes, plan) or a content mismatch (the checkpoint copies differ
     across nodes) -> every rank restores the image's vocab-parallel embedding.
  2. on the GPU, with the planned collectives issued on every rank (a rank-local failure substitutes zero rows of the
     planned shape, so the NCCL sequence never diverges; an all-reduce that raises = a broken TP group -> re-raised):
     clone this rank's rows (the image's own shard bytes), canonicalise the zeros, then compare the replicated lookup
     BITWISE with the image computation of the same ids (the REAL VocabParallelEmbedding.forward on a view of the cloned
     shard with the image's TP indices, reduce_results=False, then the image's tensor_model_parallel_all_reduce, i.e.
     patch 012's one-shot for <= 8 tokens and NCCL above):
       - every id 0..V-1 in calls of CHUNK_IDS = 8,192 (= --max-num-batched-tokens: the largest runtime embedding
         all-reduce, 64 MiB); this also proves every rank's full table equals the rows its owners would have loaded;
       - probe calls of 1..8, 16, 48 and 3,840 tokens over the shard-boundary ids, the rows holding -0.0 /
         subnormals, 32 pseudo-random ids and 7 out-of-range ids (-1, -2, V, V+1, -V, 2^31-1, -2^31);
       - one CUDA graph of the replicated lookup (4 tokens) replayed on 4 id sets == the image results.
  3. gloo all-gather of the verdicts: ACTIVE iff every rank passed; otherwise every rank restores the image layer
     (class, TP shard indices, the cloned UNcanonicalised shard as its weight, UnquantizedEmbeddingMethod) and frees the
     full table, so "Model loading took" is the image's again.
  Log per rank: "[patch-032] replicated input embedding ACTIVE (rank r/8; ...)" or "[patch-032] replicated input
  embedding INACTIVE (rank r/8): <why>; ...".
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from types import SimpleNamespace

import torch
from torch.nn.parameter import Parameter

import vllm.model_executor.layers.vocab_parallel_embedding as _vpe
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    pad_vocab_size,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

TAG = "[patch-032]"
ENV = "VLLM_GB10_REPL_EMBED"
REVISION = 2                     # 1 = day-0 image (487ecf187); 2 = vLLM v0.32.0 port
REQUIRED_TP = 8                  # the validated envelope: 8 x GB10, TP=8
CHUNK_IDS = 8192                 # exhaustive check: ids per all-reduce (= --max-num-batched-tokens, 64 MiB at H=4096)
PROBE_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 16, 48, 3840)   # 1..8 tokens = patch 012's one-shot at H=4096, the rest NCCL
MAX_SPECIAL_ROWS = 32            # rows holding -0.0 / subnormals that join the probe list
N_RANDOM_IDS = 32
GRAPH_N = 4                      # tokens in the CUDA-graph check (the decode target pass at batch 1, k=2, padded)
GRAPH_REPLAYS = 4
DIGEST_MOD = 65521
BF16_NEG_ZERO = -32768           # 0x8000 as int16: the only bf16 pattern the image's all-reduce changes (-> +0.0)
BF16_EXP = 0x7F80                # exponent bits: 0 = zero / subnormal, all ones = inf / NaN
BF16_MAG = 0x7FFF
MODE_PENDING = "pending"         # built, boot self-test not run yet: the image computation on this rank's rows
MODE_REPLICATED = "replicated"   # boot passed on every rank: full-vocab lookup, no all-reduce
MODE_VOCAB_PARALLEL = "vocab-parallel"   # boot refused: the layer is the image's VocabParallelEmbedding again


def _sha16_file(path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


MODULE_SHA16 = _sha16_file(__file__)


# ---------------------------------------------------------------------------------------------------------------------
# World: the vLLM TP group (production); tests may install their own through _set_world_factory_for_test
# ---------------------------------------------------------------------------------------------------------------------
class _TPWorld:
    """The image's own TP all-reduce (vocab_parallel_embedding.tensor_model_parallel_all_reduce: patch 012's one-shot
    for small bf16 messages, else pynccl) + gloo object all-gather on the TP group's cpu_group."""

    kind = "tp"

    def __init__(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        self.group = get_tp_group()
        self.rank = int(self.group.rank_in_group)
        self.size = int(self.group.world_size)
        try:
            self.pp_size = int(get_pp_group().world_size)
        except Exception:  # noqa: BLE001
            self.pp_size = -1

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        return _vpe.tensor_model_parallel_all_reduce(t)

    def all_gather_object(self, obj):
        import torch.distributed as dist

        out = [None] * self.size
        dist.all_gather_object(out, obj, group=self.group.cpu_group)
        return out


_WORLD_FACTORY = None


def _make_world():
    return _WORLD_FACTORY() if _WORLD_FACTORY is not None else _TPWorld()


def _tp_rank_size() -> tuple[int, int]:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())


def _safe_rank_size() -> tuple[str, str]:
    try:
        r, s = _tp_rank_size()
        return str(r), str(s)
    except Exception:  # noqa: BLE001
        return "?", "?"


def _inactive(rank, size, why: str, tail: str) -> None:
    logger.warning("%s replicated input embedding INACTIVE (rank %s/%s): %s; %s", TAG, rank, size, why, tail)


# ---------------------------------------------------------------------------------------------------------------------
# Construction (model.py: Glm5NextModel.__init__, `if get_pp_group().is_first_rank:` branch)
# ---------------------------------------------------------------------------------------------------------------------
def construction_refusal(config) -> str | None:
    """Envelope checks that need no weights; identical inputs on every rank -> identical decisions."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed import get_pp_group

    rank, size = _tp_rank_size()
    if size != REQUIRED_TP:
        return f"validated for TP={REQUIRED_TP} only (this TP group has {size} ranks)"
    pp = int(get_pp_group().world_size)
    if pp != 1:
        return f"needs pipeline parallel size 1 (got {pp})"
    vc = get_current_vllm_config_or_none()
    if vc is None:
        return "no current vLLM config at model construction"
    if getattr(vc, "lora_config", None) is not None:
        return "LoRA is enabled (its embedding wrapper expects the vocab-parallel layer)"
    if getattr(getattr(vc, "model_config", None), "word_embeddings_untied_by_checkpoint", False):
        return ("the config ties the word embeddings but the checkpoint ships an lm_head (v0.32.0 may re-tie the "
                "lm_head to the embedding after loading)")
    if getattr(config, "tie_word_embeddings", False):
        return "tie_word_embeddings=True (an lm_head tied to this table would see the full vocabulary)"
    V, H = getattr(config, "vocab_size", None), getattr(config, "hidden_size", None)
    if not (isinstance(V, int) and V > 0 and isinstance(H, int) and H > 0):
        return f"vocab_size {V!r} / hidden_size {H!r}"
    if pad_vocab_size(V, DEFAULT_VOCAB_PADDING_SIZE) != V:
        return f"vocab_size {V} is not a multiple of {DEFAULT_VOCAB_PADDING_SIZE} (padded rows are not validated)"
    if V % size:
        return f"vocab_size {V} is not a multiple of the TP size {size}"
    dt = torch.get_default_dtype()
    if dt != torch.bfloat16:
        return f"model dtype {dt} (validated for bfloat16 only)"
    return None


def embedding_class(config):
    """The class model.py builds embed_tokens with: ReplicatedEmbedding, or the image's VocabParallelEmbedding."""
    try:
        why = construction_refusal(config)
    except Exception as e:  # noqa: BLE001
        why = f"construction check raised {e!r}"[:400]
    if why is None:
        return ReplicatedEmbedding
    r, s = _safe_rank_size()
    _inactive(r, s, why, "embed_tokens is built as the image's VocabParallelEmbedding (no extra memory)")
    return VocabParallelEmbedding


class _P032EmbeddingMethod(UnquantizedEmbeddingMethod):
    """UnquantizedEmbeddingMethod + the boot hook. process_weights_after_loading() runs once per load, after every
    weight is loaded, on every rank, in the loader's module order (model_loader/utils.py, pinned).

    supports_pre_processed_weights = False: the boot transforms the table (zero canonicalisation, possible fallback to the
    vocab-parallel shard), so v0.32.0's process_weights_after_loading must reject this method under
    weights_already_processed() (the weight-cache IPC loader) instead of skipping the boot silently (fail closed)."""

    supports_pre_processed_weights = False

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        if isinstance(layer, ReplicatedEmbedding):
            boot(layer)


class ReplicatedEmbedding(VocabParallelEmbedding):
    """embed_tokens with the full vocabulary on every TP rank (disable_tp=True) and no all-reduce once the boot
    self-test has passed on every rank; until then (and if it fails) the image's vocab-parallel computation."""

    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = "") -> None:
        super().__init__(num_embeddings, embedding_dim, prefix=prefix, disable_tp=True)
        rank, size = _tp_rank_size()
        self._p032_rank, self._p032_size = rank, size
        self._p032_rows = self.num_embeddings_padded // size
        self._p032_tp_indices = self._get_indices(self.num_embeddings_padded, self.org_vocab_size_padded,
                                                  self.num_embeddings, self.org_vocab_size, rank, size)
        self._p032_mode = MODE_PENDING
        self._p032_full_loads = 0
        self._p032_booted = False
        self._p032_warned = False
        self._p032_report = None
        self.quant_method = _P032EmbeddingMethod()
        logger.info("%s embed_tokens built with the full vocabulary on rank %d/%d (%d x %d %s = %.2f GiB; the "
                    "vocab-parallel shard would be %d rows); the boot self-test runs after weight loading", TAG, rank,
                    size, self.num_embeddings_padded, embedding_dim, str(self.params_dtype).replace("torch.", ""),
                    self.weight.numel() * self.weight.element_size() / 2**30, self._p032_rows)

    def weight_loader(self, param, loaded_weight) -> None:
        super().weight_loader(param, loaded_weight)
        if (getattr(param, "output_dim", None) == 0 and self.tp_size == 1
                and int(loaded_weight.shape[0]) == self.org_vocab_size):
            self._p032_full_loads += 1

    def forward(self, input_):
        if self._p032_mode == MODE_REPLICATED:
            return _lookup(self._p032_replicated_view(), input_)
        return self._p032_pending_forward(input_)

    def _p032_replicated_view(self):
        """The view the replicated forward runs on (cached; rebuilt if the weight Parameter object was replaced)."""
        v = self.__dict__.get("_p032_view")
        if v is None or v.weight is not self.weight:
            v = self.__dict__["_p032_view"] = _view(self.weight, self.shard_indices, self._p032_size,
                                                    self.embedding_dim, reduce=False)
        return v

    def _p032_pending_forward(self, input_):
        if not self._p032_warned:
            self._p032_warned = True
            logger.warning("%s boot self-test never ran on rank %d/%d (no process_weights_after_loading call): "
                           "embed_tokens computes the image's vocab-parallel lookup + all-reduce on this rank's rows "
                           "of the full table", TAG, self._p032_rank, self._p032_size)
        start = self._p032_rank * self._p032_rows
        view = _image_view(self.weight.narrow(0, start, self._p032_rows), self._p032_tp_indices, self._p032_size,
                           self.embedding_dim, reduce=True)
        return _lookup(view, input_)


def _vanilla_use_fused(size: int) -> bool:
    """VocabParallelEmbedding.__init__'s own `use_fused_embedding` (v0.32.0): tp_size > 1 and CUDA and an
    UnquantizedEmbeddingMethod. Every layer this module builds or restores has that method."""
    return bool(size > 1 and current_platform.is_cuda())


def _view(weight, si, size: int, embedding_dim: int, *, reduce: bool):
    """The attributes VocabParallelEmbedding.forward (v0.32.0) reads, for the shard layout `si` at TP size `size`.
    Running the REAL forward on this object IS upstream's TP>1 lookup (the fused kernel on CUDA, the masked gather
    otherwise); reduce=False drops the final all-reduce, reduce=True keeps tensor_model_parallel_all_reduce."""
    return SimpleNamespace(tp_size=size, shard_indices=si, quant_method=UnquantizedEmbeddingMethod(), weight=weight,
                           use_fused_embedding=_vanilla_use_fused(size), embedding_dim=embedding_dim,
                           reduce_results=reduce, parallel_group=None)


def _image_view(weight, si, size: int, embedding_dim: int, *, reduce: bool):
    """_view for the image's TP layout of this rank (`weight` = this rank's shard rows, `si` = the image's indices)."""
    return _view(weight, si, size, embedding_dim, reduce=reduce)


def _lookup(view, input_):
    """The unpatched layer's forward (upstream code) on `view`."""
    return VocabParallelEmbedding.forward(view, input_)


# ---------------------------------------------------------------------------------------------------------------------
# Table scan / canonicalisation / plan
# ---------------------------------------------------------------------------------------------------------------------
def _scan_table(W) -> dict:
    """Counts of -0.0 / subnormal / non-finite elements, the rows holding -0.0 or subnormals, an int64 digest. Pure
    integer tests on the int16 view of the bf16 table (no float kernel involved)."""
    V = int(W.shape[0])
    dev = W.device
    neg = torch.zeros((), dtype=torch.int64, device=dev)
    sub = torch.zeros((), dtype=torch.int64, device=dev)
    nonf = torch.zeros((), dtype=torch.int64, device=dev)
    dig1 = torch.zeros((), dtype=torch.int64, device=dev)
    dig2 = torch.zeros((), dtype=torch.int64, device=dev)
    rowflag = torch.zeros(V, dtype=torch.bool, device=dev)
    for c0 in range(0, V, CHUNK_IDS):
        c1 = min(V, c0 + CHUNK_IDS)
        xi = W[c0:c1].view(torch.int16)
        ex = xi & BF16_EXP
        nz = xi == BF16_NEG_ZERO
        sb = (ex == 0) & ((xi & BF16_MAG) != 0)
        neg += nz.sum()
        sub += sb.sum()
        nonf += (ex == BF16_EXP).sum()
        rowflag[c0:c1] = (nz | sb).any(dim=1)
        rs = xi.sum(dim=1, dtype=torch.int64)
        dig1 += rs.sum()
        dig2 += (rs * (torch.arange(c0, c1, dtype=torch.int64, device=dev) % DIGEST_MOD + 1)).sum()
    special = rowflag.nonzero().flatten()
    return {"negzero": int(neg.item()), "subnormal": int(sub.item()), "nonfinite": int(nonf.item()),
            "special_rows": int(special.numel()), "special": [int(i) for i in special[:MAX_SPECIAL_ROWS].tolist()],
            "digest": f"{int(dig1.item())}:{int(dig2.item())}"}


def _canonicalize_zeros(W) -> None:
    """Every -0.0 of the table becomes +0.0 (the value the image's all-reduce produces); nothing else is written."""
    for c0 in range(0, int(W.shape[0]), CHUNK_IDS):
        x = W[c0:c0 + CHUNK_IDS]
        x.masked_fill_(x.view(torch.int16) == BF16_NEG_ZERO, 0.0)


def _count_negzero(W) -> int:
    n = torch.zeros((), dtype=torch.int64, device=W.device)
    for c0 in range(0, int(W.shape[0]), CHUNK_IDS):
        n += (W[c0:c0 + CHUNK_IDS].view(torch.int16) == BF16_NEG_ZERO).sum()
    return int(n.item())


def probe_ids(V: int, rows: int, size: int, special) -> list[int]:
    ids = [0, 1, V - 2, V - 1]
    for r in range(1, size):
        ids += [r * rows - 1, r * rows]
    ids += list(special)[:MAX_SPECIAL_ROWS]
    x = 0x9E3779B9
    for _ in range(N_RANDOM_IDS):
        x = (1103515245 * x + 12345) % (1 << 31)
        ids.append(x % V)
    ids += oob_ids(V)
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def oob_ids(V: int) -> list[int]:
    return [-1, -2, V, V + 1, -V, (1 << 31) - 1, -(1 << 31)]


def build_plan(V: int, rows: int, size: int, special) -> list[tuple]:
    """Every collective of the self-test, identical on every rank: ("all", c0, c1) or ("probe", n, ids)."""
    plan = [("all", c0, min(V, c0 + CHUNK_IDS)) for c0 in range(0, V, CHUNK_IDS)]
    probe = probe_ids(V, rows, size, special)
    L = len(probe)
    for n in PROBE_SIZES:
        if n >= L:
            plan.append(("probe", n, tuple(probe[i % L] for i in range(n))))
            continue
        for s in range(0, L, n):
            plan.append(("probe", n, tuple(probe[(s + i) % L] for i in range(n))))
    return plan


def _plan_sig(plan) -> str:
    h = hashlib.sha256()
    for c in plan:
        h.update(repr(c).encode())
    return h.hexdigest()[:16]


def _call_rows(call) -> int:
    return call[2] - call[1] if call[0] == "all" else call[1]


def _call_ids(call, device):
    if call[0] == "all":
        return torch.arange(call[1], call[2], dtype=torch.int32, device=device)
    return torch.tensor(list(call[2]), dtype=torch.int32, device=device)


def _call_label(call) -> str:
    if call[0] == "all":
        return f"ids {call[1]}..{call[2] - 1}"
    ids = call[2]
    return f"{call[1]}-token probe {list(ids[:6])}{'...' if len(ids) > 6 else ''}"


# ---------------------------------------------------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------------------------------------------------
def _target_file() -> str | None:
    m = sys.modules.get("vllm.models.glm5next.common.model")
    return getattr(m, "__file__", None)


def fingerprint(layer, world, plan_sig: str) -> str:
    parts = ["p032", f"rev={REVISION}", f"module={MODULE_SHA16}", f"target={_sha16_file(_target_file())}",
             f"gate={os.environ.get(ENV, '0')}", f"tp={world.size}", f"pp={getattr(world, 'pp_size', '?')}",
             f"V={layer.org_vocab_size}", f"Vp={layer.num_embeddings_padded}", f"H={layer.embedding_dim}",
             f"dtype={layer.weight.dtype}", f"fused={_vanilla_use_fused(world.size)}", f"chunk={CHUNK_IDS}",
             f"probe={PROBE_SIZES}", f"graph={GRAPH_N}", f"plan={plan_sig}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _envelope_refusal(layer, world) -> str | None:
    if world.size != REQUIRED_TP:
        return f"validated for TP={REQUIRED_TP} only (this TP group has {world.size} ranks)"
    if world.size != layer._p032_size or world.rank != layer._p032_rank:
        return (f"TP rank {world.rank}/{world.size} != the construction-time {layer._p032_rank}/"
                f"{layer._p032_size}")
    if getattr(world, "pp_size", 1) != 1:
        return f"needs pipeline parallel size 1 (got {world.pp_size})"
    return None


def _layout_refusal(layer) -> str | None:
    W = layer.weight
    V, H = layer.org_vocab_size, layer.embedding_dim
    if layer.tp_size != 1 or layer.num_embeddings_padded != V or layer.num_embeddings != V:
        return f"layer layout tp_size {layer.tp_size}, padded vocab {layer.num_embeddings_padded} (want 1, {V})"
    if not isinstance(W, Parameter) or tuple(W.shape) != (V, H):
        return f"weight {type(W).__name__} {tuple(W.shape)} (want Parameter ({V}, {H}))"
    if W.dtype != torch.bfloat16 or not W.is_cuda or not W.is_contiguous():
        return f"weight dtype {W.dtype} / device {W.device} / contiguous {W.is_contiguous()}"
    if layer._p032_full_loads != 1:
        return (f"embed_tokens.weight went through the full-vocab weight_loader {layer._p032_full_loads} times "
                f"(want exactly 1)")
    if type(layer.quant_method) is not _P032EmbeddingMethod:
        return f"quant method {type(layer.quant_method).__name__}"
    return None


def _exchange(world, obj, what: str):
    """gloo all-gather of a small object. A failure here means the TP group's gloo group is broken (vLLM's own
    collectives fail next): logged and re-raised, never turned into a per-rank decision."""
    try:
        return world.all_gather_object(obj)
    except Exception as e:  # noqa: BLE001
        logger.error("%s boot %s exchange over the TP group failed on rank %s (%r): the TP group is broken - "
                     "re-raising (not a patch-032 refusal)", TAG, what, getattr(world, "rank", "?"), e)
        raise


def _graph_check(layer, refs) -> int:
    """Capture the replicated lookup of GRAPH_N tokens in one CUDA graph, replay it on the probe id sets `refs`
    [(ids, image_ref)] and count the bits that differ from the image results."""
    dev = layer.weight.device
    static = torch.zeros(GRAPH_N, dtype=torch.int32, device=dev)
    view = layer._p032_replicated_view()
    _lookup(view, static)                                       # eager warm-up (lazy compile / allocator) first
    torch.cuda.synchronize(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        out = _lookup(view, static)
    bad = 0
    try:
        for ids, ref in refs:
            static.copy_(ids)
            g.replay()
            bad += int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())
    finally:
        g.reset()
        del out
        del g
    return bad


def _selftest(layer, world, plan) -> tuple[dict, object]:
    """Phase 2 on this rank. Every planned all-reduce is issued whatever fails locally. Returns (report, orig_shard)."""
    rep = {"rank": world.rank, "ok": False, "fails": [], "calls": len(plan), "mism": 0, "graph": None}
    fails = rep["fails"]
    dev = layer.weight.device
    H = layer.embedding_dim
    rows, start = layer._p032_rows, world.rank * layer._p032_rows
    si_tp = layer._p032_tp_indices
    orig = None
    ok = True
    try:
        with torch.no_grad():
            orig = layer.weight.narrow(0, start, rows).clone()     # this rank's image shard bytes (uncanonicalised)
            _canonicalize_zeros(layer.weight)
            left = _count_negzero(layer.weight)
        if left:
            ok = False
            fails.append(f"{left} negative zeros left after canonicalisation")
    except Exception as e:  # noqa: BLE001
        ok = False
        fails.append(f"clone / canonicalise raised {e!r}"[:300])
    graph_refs = []
    with torch.inference_mode():
        mm = None
        if ok:
            try:
                mm = torch.zeros(len(plan), dtype=torch.int64, device=dev)
            except Exception as e:  # noqa: BLE001
                ok = False
                fails.append(f"allocation raised {e!r}"[:300])
        for k, call in enumerate(plan):
            ids = contrib = None
            if ok:
                try:
                    ids = _call_ids(call, dev)
                    contrib = _lookup(_image_view(orig, si_tp, world.size, H, reduce=False), ids)
                except Exception as e:  # noqa: BLE001
                    ok = False
                    fails.append(f"{_call_label(call)}: image lookup raised {e!r}"[:300])
                    contrib = None
            if contrib is None:      # keep the collective schedule: zero rows of the planned shape
                contrib = torch.zeros(_call_rows(call), H, dtype=layer.weight.dtype, device=dev)
            ref = world.all_reduce(contrib)       # raises only if the TP group is broken -> propagates
            if not ok:
                continue
            try:
                out = _lookup(layer._p032_replicated_view(), ids)
                mm[k:k + 1].copy_((out.view(torch.int16) != ref.view(torch.int16)).sum().view(1))
                if call[0] == "probe" and call[1] == GRAPH_N and len(graph_refs) < GRAPH_REPLAYS:
                    graph_refs.append((ids, ref))
            except Exception as e:  # noqa: BLE001
                ok = False
                fails.append(f"{_call_label(call)}: replicated lookup / compare raised {e!r}"[:300])
        if ok:
            counts = mm.tolist()
            bad = [(plan[k], int(c)) for k, c in enumerate(counts) if c]
            rep["mism"] = sum(c for _, c in bad)
            if bad:
                ok = False
                fails.append(f"{len(bad)} of {len(plan)} calls differ from the image's lookup + all-reduce ("
                             + ", ".join(f"{_call_label(c)}: {n} elements" for c, n in bad[:3]) + ")")
        if ok:
            try:
                if len(graph_refs) != GRAPH_REPLAYS:
                    raise RuntimeError(f"only {len(graph_refs)} probe sets of {GRAPH_N} tokens")
                gbad = _graph_check(layer, graph_refs)
                rep["graph"] = gbad
                if gbad:
                    ok = False
                    fails.append(f"CUDA-graph replay differs from the image results in {gbad} elements")
            except Exception as e:  # noqa: BLE001
                ok = False
                fails.append(f"CUDA-graph check raised {e!r}"[:300])
    rep["ok"] = ok
    return rep, orig


def _to_vocab_parallel(layer, shard) -> None:
    """Make `layer` the image's VocabParallelEmbedding of this rank: class, TP state, quant method, and `shard` (this
    rank's rows as the image's weight_loader would have loaded them) as its weight. The full table is freed. Restores every
    attribute v0.32.0's VocabParallelEmbedding.__init__ sets differently for disable_tp=True (tp_rank / tp_size /
    shard_indices, the per-partition sizes, output_size_per_partition / output_partition_sizes, use_fused_embedding);
    reduce_results / parallel_group are the constructor defaults of both layers and are set again for certainty."""
    si, size, rank = layer._p032_tp_indices, layer._p032_size, layer._p032_rank
    layer.__dict__.pop("_p032_view", None)         # the replicated view holds the full table
    layer.__class__ = VocabParallelEmbedding
    layer.disable_tp = False
    layer.tp_rank, layer.tp_size = rank, size
    layer.shard_indices = si
    layer.num_embeddings_per_partition = layer.num_embeddings_padded // size
    layer.output_size_per_partition = layer.num_embeddings_per_partition
    layer.output_partition_sizes = [layer.num_embeddings_per_partition]
    layer.num_org_embeddings_per_partition = si.org_vocab_end_index - si.org_vocab_start_index
    layer.num_added_embeddings_per_partition = si.added_vocab_end_index - si.added_vocab_start_index
    layer.quant_method = UnquantizedEmbeddingMethod()
    layer.use_fused_embedding = _vanilla_use_fused(size)
    layer.reduce_results = True
    layer.parallel_group = None
    w = Parameter(shard, requires_grad=False)
    set_weight_attrs(w, {"input_dim": 1, "output_dim": 0})
    set_weight_attrs(w, {"weight_loader": layer.weight_loader})
    layer.weight = w
    layer._p032_mode = MODE_VOCAB_PARALLEL


def _fallback(layer, world, orig, why: str) -> None:
    rows = layer._p032_rows
    if orig is None:      # nothing was canonicalised yet (the clone precedes it): the table holds the image's bytes
        with torch.no_grad():
            orig = layer.weight.narrow(0, layer._p032_rank * rows, rows).clone()
    _to_vocab_parallel(layer, orig)
    del orig
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    _inactive(getattr(world, "rank", layer._p032_rank), getattr(world, "size", layer._p032_size), why,
              f"embed_tokens is the image's VocabParallelEmbedding again ({rows}-row shard, all-reduce kept)")


def boot(layer) -> None:
    """process_weights_after_loading hook (every rank, once per load)."""
    if layer._p032_booted:
        if layer._p032_mode == MODE_REPLICATED:      # a later weight reload rewrote the full table
            with torch.no_grad():
                _canonicalize_zeros(layer.weight)
            logger.info("%s embed_tokens reloaded on rank %d: its zeros are +0.0 again", TAG, layer._p032_rank)
        return
    layer._p032_booted = True
    t0 = time.perf_counter()
    try:
        world = _make_world()
    except Exception as e:  # noqa: BLE001 - no TP group: nothing collective can run, identically on every rank
        _fallback(layer, None, None, f"no TP group ({e!r})"[:400])
        return
    # ---- 1. local preflight, then gloo collective 1
    pre = {"rank": world.rank, "fp": None, "content": None, "why": None}
    scan = plan = None
    try:
        pre["why"] = _envelope_refusal(layer, world) or _layout_refusal(layer)
        if pre["why"] is None:
            scan = _scan_table(layer.weight)
            if scan["nonfinite"]:
                pre["why"] = f"{scan['nonfinite']} non-finite values in the embedding table"
        if pre["why"] is None:
            plan = build_plan(layer.org_vocab_size, layer._p032_rows, world.size, scan["special"])
            pre["fp"] = fingerprint(layer, world, _plan_sig(plan))
            pre["content"] = (f"digest {scan['digest']}, {scan['negzero']} -0.0, {scan['subnormal']} subnormal, "
                              f"{scan['special_rows']} special rows")
    except Exception as e:  # noqa: BLE001
        pre["why"] = f"preflight raised {e!r}"[:400]
    allp = _exchange(world, pre, "preflight")
    t1 = time.perf_counter()                  # t1 - t0: waiting for the slowest rank's weight loading
    refusals = [f"rank {p['rank']}: {p['why']}" for p in allp if p["why"] is not None]
    fps = sorted({str(p["fp"]) for p in allp})
    contents = sorted({str(p["content"]) for p in allp})
    if refusals or len(fps) != 1 or len(contents) != 1:
        if refusals:
            why = "; ".join(refusals)
        elif len(contents) != 1:
            why = ("the embedding table differs across ranks (checkpoint copies?): "
                   + " | ".join(f"rank {p['rank']}: {p['content']}" for p in allp))
        else:
            why = f"rank fingerprint mismatch {fps} (module / model.py / env / shapes / plan)"
        _fallback(layer, world, None, why[:1500])
        return
    # ---- 2. GPU self-test (every planned all-reduce on every rank), then gloo collective 2
    report, orig = _selftest(layer, world, plan)   # raises only if an all-reduce raised (a broken TP group)
    allv = _exchange(world, report, "verdict")
    bad = [v for v in allv if not v["ok"]]
    if bad:
        why = "boot self-test FAILED - " + " | ".join(f"rank {v['rank']}: " + "; ".join(v["fails"][:2])
                                                      for v in bad)
        _fallback(layer, world, orig, why[:1500])
        return
    # ---- 3. commit (identical decision on every rank: a function of allv only)
    layer._p032_mode = MODE_REPLICATED
    layer._p032_report = {"scan": scan, "calls": len(plan), "fingerprint": fps[0]}
    del orig
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    W = layer.weight
    n_probe = sum(1 for c in plan if c[0] == "probe")
    n_all = len(plan) - n_probe
    logger.info(
        "%s replicated input embedding ACTIVE (rank %d/%d; %d x %d %s = %.2f GiB per rank, +%.2f GiB vs the "
        "vocab-parallel shard; no all-reduce in embed_tokens, the MTP drafter shares the module; rev %d, fingerprint "
        "%s): boot self-test passed on all %d ranks in %.2f s after a %.1f s preflight wait (all %d ids in %d calls of "
        "<= %d tokens and %d probe "
        "calls of %d-%d tokens incl. %d out-of-range ids bitwise equal to the image's vocab-parallel lookup + all-reduce; "
        "CUDA-graph replay of %d x %d tokens bitwise; %d negative zeros set to +0.0, %d subnormals)",
        TAG, world.rank, world.size, layer.org_vocab_size, layer.embedding_dim,
        str(W.dtype).replace("torch.", ""), W.numel() * W.element_size() / 2**30,
        W.numel() * W.element_size() * (world.size - 1) / world.size / 2**30, REVISION, fps[0], len(allv),
        time.perf_counter() - t1, t1 - t0, layer.org_vocab_size, n_all, CHUNK_IDS, n_probe, min(PROBE_SIZES),
        max(PROBE_SIZES), len(oob_ids(layer.org_vocab_size)), GRAPH_REPLAYS, GRAPH_N, scan["negzero"],
        scan["subnormal"])


# ---------------------------------------------------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------------------------------------------------
def _set_world_factory_for_test(factory) -> None:
    global _WORLD_FACTORY
    _WORLD_FACTORY = factory
