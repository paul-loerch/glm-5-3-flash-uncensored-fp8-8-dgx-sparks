# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# [patch-020 gb10-sparse-mla-triton-prefill] — GB10 (sm_121) sparse-MLA prefill on Triton, REVISION 4
# (the vLLM v0.32.0 port). Installed as vllm/_gb10_sparse_mla_prefill.py by
# patches/020-gb10-sparse-mla-triton-prefill.sh. One opt-in gate:
#
#   VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1  (NUMERICS CHANGE) — route prefill rows [nd, T) of the
#       FlashInfer-SM90-on-SM12x sparse-MLA MQA path (vllm/v1/attention/backends/mla/
#       flashinfer_mla_sparse_sm90.py, nd == num_decode_tokens) to a Triton fused sparse-attention
#       kernel; decode rows [0, nd) keep the FlashInfer FA2 wrapper (planned for nd rows). Pure-decode
#       steps (nd == T) route nothing to Triton and run upstream's own pack_indices + wrapper.run, so they
#       are bit-identical to stock v0.32.0 and CUDA-graph safe; pure-prefill builds (nd == 0) skip the FA2
#       plan (no row reads it) and route everything to Triton; mixed steps split. The kernel reads the
#       E4M3 latent cache directly (fp8 -> fp16, exact), dequantises K AND V with the per-tensor k_scale
#       (as FA2's ckv_scale and the other vLLM MLA backends do), loops to the device-side valid_counts
#       (clamped to the index width), accumulates in fp32, and rounds P to fp16. Measured on GB10
#       2026-10-01 (revision 3, image Triton 3.7.1, old FlashInfer 0.6.18 FA2): Triton vs an fp64
#       reference of the dequantised attention 1.28e-3 mean-rel (p99 3.5e-3), FA2 2.24e-3 / 1.0e-2 on the
#       same rows, Triton vs FA2 1.7e-3; prefill +13.2 % (4k) ... +17.8 % (64k) tok/s, decode and
#       decode-KL unchanged. THE KERNEL IS BYTE-IDENTICAL TO REVISION 3; it has NOT been re-run on the
#       v0.32.0 image (new FlashInfer 0.7.0.post1 FA2, new Triton): the boot self-test below and
#       p020_gpu_selftest.py --kernel / --fa2 / --split are the acceptance. The result for a row depends
#       on the prefill-row count P through the split count (S = 2 iff P <= 48); with the served reorder
#       threshold (128) every prefill request has > 128 rows, so S = 1. Gate with decode-kl / needles.
#
# WHAT CHANGED vs revision 3 (old image, vLLM 487ecf187):
#   * Part A (VLLM_GB10_SPARSE_MLA_NOSYNC) is GONE: vLLM #58684 (v0.31.0) plans every build sync-free from
#     the host upper bound and clamps each FA2 work's kv_end on the device; patch 021 (the same idea) is
#     dropped for the same reason. The env var is ignored (the .sh prints a note when it is set).
#   * The process-wide _SM90_STATE / _get_sm90_state are gone: each metadata builder owns an _SM90State and
#     tags every metadata with it (metadata.state). plan_step therefore plans through builder.state, and
#     forward_split runs FA2 through the state the metadata carries (never a global).
#   * Upstream packs the converted slots with state.pack_indices(topk_slots) (valid prefix at the planned
#     kv_indptr offsets) instead of copying a clamped full-width block; _fa2_run now packs only the rows it
#     runs. Upstream's plan reuse (+32 slack) and device clamp apply unchanged to the decode prefix.
#   * The KV cache arrives as flat_kv_row_view's [num_rows, head_size] view (block stride may exceed the
#     block size): the Triton kernel indexes it with its own strides and bounds slots by num_rows.
#
# PROVENANCE (see also PATCHES.md):
#   * The Triton kernel is a port of vLLM's ROCm-aiter sparse-MLA prefill kernel
#     (vllm/v1/attention/ops/rocm_aiter_mla_sparse.py `_sparse_attn_prefill_ragged_kernel`; Apache-2.0,
#     "Copyright contributors to the vLLM project"). Changes (Apache-2.0 4(b)): the ragged per-row bound is
#     the device-side valid_counts (a compacted-prefix count) rather than a CSR indptr; K/V come from ONE
#     fp8 E4M3 latent gather (NoPE, K == V, D=512) converted to fp16 with k_scale applied to the scores and
#     v_scale to the output, not a bf16 [skv,d] tensor; q gets an exact per-head power-of-two pre-scale so
#     no bf16 |q| can overflow the fp16 operand; slot offsets are int64; the (row, head-tile, split) grid
#     keeps the ROCm kernel's head-tile axis (BLOCK_H=16); BLOCK_N=32; single-split rows are written as bf16
#     straight into the caller's output view.
#   * The fixed-order fp32 two-split + LSE merge follows vLLM PR #47629's split-KV/merge idea
#     (TRITON_MLA_SPARSE, Apache-2.0; not merged upstream); re-implemented here (no code copied).
#   * kindlingai/glm-5.3-flash-gx10 image/patches (gb10_sparse_mla.py) has NO root licence and was
#     NOT read for code — only the public design of the problem was used; this file is independent.
#
# Everything is fail-closed and inert unless the env gate is "1" (read at runtime, so a reused container with
# the edit applied but the env unset behaves exactly like stock vLLM). With the gate set, the patched backend
# refuses to fall back to the stock path: an import failure raises at backend import, startup_check() (from
# the impl's __init__) validates the config and runs an on-device kernel self-test, and every broken
# invariant raises Patch020Error. The hooks live ONLY in FlashInferMLASparseSM90{Builder,Impl}: if the engine
# selects another sparse-MLA backend (e.g. FLASHINFER_MLA_SPARSE_SM120 because the SM121 bake did not enable
# the SM90 backend on capability 12) none of this code runs — no "[patch-020] ACTIVE" line appears in the
# rank logs. Set the flag on EVERY rank (M_DOCKER_EXTRA): a per-rank mismatch would desync attention numerics.

import logging
import os

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

# The GLM-5.3-Flash NoPE latent dimension (kv_lora_rank); qk_rope_head_dim == 0 on this model.
_D_LATENT = 512
# Frozen kernel config, chosen by analysis — NO GPU sweep has been run. CPU-side
# compile for sm_121a (venv Triton 3.8.0 + ptxas-blackwell 13.3; the image has Triton 3.7.1): 49,408 B
# shared memory, 255 registers; S=1 0 spills, S=2 8 B spill (S=2 does not occur in this recipe). No
# runtime autotuning while serving. BLOCK_H is the head tile (the 8 heads per rank are padded to 16;
# more heads use more head tiles).
_BLOCK_H = 16
_BLOCK_N = 32
_NUM_WARPS = 4
_NUM_STAGES = 3
# Two-split threshold: use S=2 over the valid prefix only when the prefill row count is too small to
# fill ~two waves of resident programs. Frozen, host-side (no sync). NOTE: a row's output depends on
# S (~1e-4), i.e. on P; with the served reorder threshold 128 a prefill request has > 128 rows, so S=2
# never occurs in this recipe (kept for generality, covered by the interpreter tests).
_TWO_SPLIT_MAX_ROWS = 48
_GATE_TRITON = "VLLM_GB10_SPARSE_MLA_TRITON_PREFILL"
# Attribute plan_step leaves on the builder's _SM90State: how many rows (from 0) the FA2 wrapper was planned
# for on the latest build. forward_split refuses to run FA2 on a different row count.
_PLAN_ROWS_ATTR = "_p020_plan_rows"


class Patch020Error(RuntimeError):
    """Fail-closed refusal: a patch-020 gate is set but something it relies on does not hold."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise Patch020Error(f"[patch-020] {msg}")


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def triton_prefill_enabled() -> bool:
    return _flag(_GATE_TRITON)


# --------------------------------------------------------------------------------------------------
# Builder side: planning. The patched FlashInferMLASparseSM90Builder.build() calls plan_step() instead of
# its own planning lines whenever the gate is set (it returns right after, having tagged metadata.state).
# --------------------------------------------------------------------------------------------------
def plan_step(builder, cam, metadata) -> None:
    """Plan the FA2 wrapper for this step: ONLY the decode prefix [0, nd), nd = num_decode_tokens.

    The prefill rows [nd, T) run on the Triton kernel in forward_split and read no FA2 plan, so:

    * nd == 0 -> no plan at all (no zero-row FlashInfer plan, no host bound, no clamp launch). The
      wrapper keeps the previous step's plan, which stays valid for upstream's reuse test: the cached
      _planned_num_tokens / _lens_cpu / _saved_kv_end all describe it, and the next plan() call restores
      kv_end from _saved_kv_end before clamping.
    * nd > 0  -> exactly upstream's plan (builder._kv_lens_host + state.plan with the same sync-free upper
      bound, +32 slack, plan reuse and device clamp, or the exact path when no host bound exists) over the
      first n = min(nd, rows) rows. For a pure-decode step (nd == rows) the statements and arguments are
      those of the stock build(), so the plan is bit-identical.

    The same slice upstream's build() takes when the prefill rows go through the dense-MHA path
    (`_use_sparse_mha`), so this is not a new planning mode for the wrapper.
    """
    nd = getattr(metadata, "num_decode_tokens", None)
    _require(nd is not None, "attention metadata carries no num_decode_tokens")
    nd = int(nd)
    state = builder.state
    if nd == 0:
        setattr(state, _PLAN_ROWS_ATTR, 0)
        return
    num_rows, kv_lens, exact = builder._kv_lens_host(cam)
    n = min(nd, int(num_rows))
    if n < int(num_rows):
        kv_lens = kv_lens[:n]
    if exact:
        state.plan(n, kv_lens, None, None)
    else:
        state.plan(n, kv_lens, cam, metadata.req_id_per_token)
    setattr(state, _PLAN_ROWS_ATTR, n)


# --------------------------------------------------------------------------------------------------
# Triton sparse-MLA prefill kernel (NoPE, D=512, E4M3 latent cache, device-side valid_counts).
# --------------------------------------------------------------------------------------------------
@triton.jit
def _sparse_mla_prefill_kernel(
    q_ptr,  # bf16 [P, H, D] (strided view of the absorbed q_nope; not copied)
    kv_ptr,  # fp8_e4m3 [num_slots, D] (latent cache, K == V)
    slots_ptr,  # int32 [P, W] global slot indices, compacted prefix, -1 past valid_count
    vcount_ptr,  # int32 [P] valid slot count per row (loop bound, clamped to [0, W])
    out_ptr,  # NUM_SPLITS == 1: bf16 [P, H, D] final output; else fp32 [S, P, H, D] partials
    lse_ptr,  # NUM_SPLITS > 1: fp32 [S, P, H] natural-log LSE per split (unused when S == 1)
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    slots_stride_t,
    out_stride_s,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    lse_stride_s,
    lse_stride_t,
    qk_scale,  # fp32: sm_scale * k_scale (K dequant folded into the scores)
    v_scale,  # fp32: V dequant scale, applied to the normalised output (== k_scale for MLA)
    num_heads,
    num_slots,
    topk_width,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_tile = tl.program_id(1)
    split_id = tl.program_id(2)

    head_offsets = head_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    neg_large = -3.4028234663852886e38

    # q (bf16) -> fp16 operand with an EXACT per-head power-of-two pre-scale: a head whose max|q| is
    # >= 2**15 is scaled by 2**-e, e = floor(log2(max|q|)) - 14, so max|q| * 2**-e < 2**15 < 65504,
    # and 2**e is restored in fp32 on the scores. Heads with max|q| < 2**15 (e = 0) are converted
    # exactly as before (bf16 -> fp16 is exact there). Any finite bf16 q is therefore safe; the scores
    # can overflow fp32 only where FA2's fp32 accumulation would too. (Pre-multiplying by sm_scale =
    # 1/16 alone would only move the overflow from |q| > 65504 to |q| > ~2**20.)
    q_f32 = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    q_amax = tl.max(tl.abs(q_f32), axis=1)
    q_exp = (q_amax.to(tl.int32, bitcast=True) >> 23) & 0xFF  # biased exponent (0 for 0/subnormal)
    q_shift = tl.maximum(q_exp - (127 + 14), 0)  # <= 114, so 2**-e and 2**e stay normal fp32
    q_down = ((127 - q_shift) << 23).to(tl.float32, bitcast=True)  # 2**-e
    q_up = ((127 + q_shift) << 23).to(tl.float32, bitcast=True)  # 2**e
    q = (q_f32 * q_down[:, None]).to(tl.float16)
    head_scale = qk_scale * q_up  # [BLOCK_H]

    valid = tl.load(vcount_ptr + query_idx)
    valid = tl.minimum(tl.maximum(valid, 0), topk_width)
    # Split the valid prefix [0, valid) into NUM_SPLITS contiguous ranges (fixed order for the merge).
    span = (valid + NUM_SPLITS - 1) // NUM_SPLITS
    kv_lo = split_id * span
    kv_hi = tl.minimum(kv_lo + span, valid)

    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    slots_row = slots_ptr + query_idx * slots_stride_t
    for k_start in tl.range(kv_lo, kv_hi, BLOCK_N):
        k_pos = k_start + tl.arange(0, BLOCK_N)
        in_range = k_pos < kv_hi
        slot = tl.load(slots_row + k_pos, mask=in_range, other=-1)
        ok = in_range & (slot >= 0) & (slot < num_slots)
        # int64 slot offset: num_slots * D may exceed 2**31.
        slot64 = tl.where(ok, slot, 0).to(tl.int64)
        kv = tl.load(
            kv_ptr + slot64[:, None] * kv_stride_n + dim_offsets[None, :] * kv_stride_d,
            mask=ok[:, None],
            other=0.0,
        ).to(tl.float16)

        live = head_mask[:, None] & ok[None, :]
        scores = tl.dot(q, tl.trans(kv)) * head_scale[:, None]
        scores = tl.where(live, scores, neg_large)

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(live, tl.exp(scores - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # P rounds to fp16 before the PV product (FA2 rounds it to bf16); fp32 accumulation.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), kv)
        m_i = m_new

    # NORMALISED output o = v_scale * acc / l (empty split -> 0).
    denom = tl.maximum(l_i, 1.0e-30)
    o = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0) * v_scale
    o_ptrs = (
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d
    )
    if NUM_SPLITS == 1:
        # Single split: the final answer, rounded (RTNE) to bf16 straight into the caller's view.
        tl.store(o_ptrs, o.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=head_mask[:, None])
    else:
        tl.store(o_ptrs + split_id * out_stride_s, o, mask=head_mask[:, None])
        # LSE in natural log; empty split -> the sentinel so the merge ignores it.
        lse = tl.where(l_i > 0.0, m_i + tl.log(denom), neg_large)
        tl.store(
            lse_ptr + split_id * lse_stride_s + query_idx * lse_stride_t + head_offsets,
            lse,
            mask=head_mask,
        )


def _choose_num_splits(num_rows: int) -> int:
    # Frozen host-side rule (no sync): few rows -> two splits over the valid prefix to fill the SMs.
    return 2 if 0 < num_rows <= _TWO_SPLIT_MAX_ROWS else 1


def sparse_mla_prefill_triton(
    q: torch.Tensor,  # bf16 [P, H, D] absorbed q_nope for prefill rows (any strides)
    latent_cache: torch.Tensor,  # fp8_e4m3 [num_slots, D]
    topk_slots: torch.Tensor,  # int32 [P, W] global slots, compacted prefix
    valid_counts: torch.Tensor,  # int32 [P]
    sm_scale: float,
    k_scale: float = 1.0,
    v_scale: float | None = None,  # None -> k_scale (MLA: K == V == the latent)
    out: torch.Tensor | None = None,  # optional bf16 [P, H, D] destination view
) -> torch.Tensor:
    """Fused sparse-MLA attention for prefill rows. Returns bf16 [P, H, D] (``out`` if given)."""
    _require(q.dtype == torch.bfloat16 and q.ndim == 3, f"q must be bf16 [P,H,D], got {q.dtype} {tuple(q.shape)}")
    _require(
        latent_cache.dtype == torch.float8_e4m3fn and latent_cache.ndim == 2,
        "latent cache must be an fp8_e4m3 [num_slots, D] view",
    )
    P, H, D = q.shape
    _require(latent_cache.shape[1] == D and D == triton.next_power_of_2(D), f"latent dim {D} unsupported")
    _require(
        topk_slots.dtype == torch.int32 and topk_slots.ndim == 2 and topk_slots.shape[0] == P
        and topk_slots.stride(1) == 1,
        "topk_slots must be int32 [P, W] with unit column stride",
    )
    _require(
        valid_counts.dtype == torch.int32 and valid_counts.ndim == 1 and valid_counts.shape[0] == P,
        "valid_counts must be int32 [P]",
    )
    if v_scale is None:
        v_scale = k_scale
    if out is None:
        out = torch.empty((P, H, D), dtype=torch.bfloat16, device=q.device)
    _require(out.dtype == torch.bfloat16 and tuple(out.shape) == (P, H, D), "out must be bf16 [P, H, D]")
    if P == 0:
        return out

    num_slots, W = latent_cache.shape[0], topk_slots.shape[1]
    num_splits = _choose_num_splits(P)
    if num_splits == 1:
        dst, lse = out, out  # lse_ptr unused by the NUM_SPLITS == 1 variant
        dst_s, lse_s = 0, (0, 0)
    else:
        dst = torch.empty((num_splits, P, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((num_splits, P, H), dtype=torch.float32, device=q.device)
        dst_s, lse_s = dst.stride(0), (lse.stride(0), lse.stride(1))
    dst_t, dst_h, dst_d = dst.stride()[-3:]

    grid = (P, triton.cdiv(H, _BLOCK_H), num_splits)
    _sparse_mla_prefill_kernel[grid](
        q,
        latent_cache,
        topk_slots,
        valid_counts,
        dst,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        latent_cache.stride(0),
        latent_cache.stride(1),
        topk_slots.stride(0),
        dst_s,
        dst_t,
        dst_h,
        dst_d,
        lse_s[0],
        lse_s[1],
        float(sm_scale) * float(k_scale),
        float(v_scale),
        H,
        num_slots,
        W,
        NUM_SPLITS=num_splits,
        BLOCK_H=_BLOCK_H,
        BLOCK_D=D,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if num_splits > 1:
        out.copy_(_merge_splits(dst, lse))
    return out


def _merge_splits(out_parts: torch.Tensor, lse_parts: torch.Tensor) -> torch.Tensor:
    """Fixed-order fp32 LSE merge of S split partials. out_parts [S,P,H,D], lse_parts [S,P,H] (natural
    log; an empty split's lse is a large negative sentinel). Returns fp32 [P,H,D]. Host-side and
    CPU-testable (the kernel writes the partials; this combines them)."""
    m = lse_parts.max(dim=0).values  # [P, H]
    w = torch.exp(lse_parts - m.unsqueeze(0))  # [S, P, H]
    denom = w.sum(dim=0).clamp_min(1e-30)  # [P, H]
    merged = (w.unsqueeze(-1) * out_parts).sum(dim=0) / denom.unsqueeze(-1)
    # A row whose every split is empty (all sentinel) has acc == 0; keep it 0.
    all_empty = (lse_parts <= -3.0e38).all(dim=0).unsqueeze(-1)
    return torch.where(all_empty, torch.zeros_like(merged), merged)


# --------------------------------------------------------------------------------------------------
# Impl side: split forward_mqa into decode [0, nd) -> FA2 and prefill [nd, T) -> Triton.
# --------------------------------------------------------------------------------------------------
def _fa2_run(impl, state, q_nope, q_pe, topk_slots, kv_cache_rows, layer, n):
    """Run the FlashInfer FA2 wrapper over the first n rows with upstream's own statements (pack the
    valid slot prefixes at the planned offsets, then wrapper.run). The CALLER guarantees the wrapper is
    planned for exactly these n rows. n == all rows takes the unsliced tensors (the stock call)."""
    if n != q_nope.shape[0]:
        q_nope, q_pe, topk_slots = q_nope[:n], q_pe[:n], topk_slots[:n]
    state.pack_indices(topk_slots)
    flat = (
        kv_cache_rows.view(torch.float8_e4m3fn) if impl.use_fp8_kv_cache else kv_cache_rows
    ).reshape(-1, 1, impl.head_size)
    ckv = flat[..., : impl.kv_lora_rank]
    kpe = flat[..., impl.kv_lora_rank :]
    scale_kwargs = (
        {"ckv_scale": float(layer._k_scale_float or 1.0), "kpe_scale": 1.0}
        if impl.use_fp8_kv_cache
        else {}
    )
    return state.wrapper.run(q_nope, q_pe, ckv, kpe, **scale_kwargs)


def _check_impl_for_triton(impl) -> None:
    _require(
        bool(impl.use_fp8_kv_cache),
        "VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1 needs --kv-cache-dtype fp8/fp8_e4m3 (the kernel reads "
        "the E4M3 latent cache)",
    )
    _require(
        getattr(impl, "qk_rope_head_dim", None) == 0,
        "VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1 is NoPE-only (qk_rope_head_dim == 0): the kernel "
        "ignores q_pe/k_pe",
    )
    _require(
        int(impl.kv_lora_rank) == _D_LATENT and int(impl.head_size) == _D_LATENT,
        f"VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1 needs kv_lora_rank == head_size == {_D_LATENT}",
    )


_FIRST_USE_LOGGED = [False]


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - no CUDA build / not initialised: not capturing
        return False


def forward_split(
    impl, state, q_nope, q_pe, topk_slots, valid_counts, kv_cache_rows, layer, attn_metadata
):
    """Decode rows [0, nd) via FA2, prefill rows [nd, T) via the Triton kernel. bf16 [T, H, D].

    kv_cache_rows is the [num_rows, head_size] row view upstream's forward_mqa already built with
    flat_kv_row_view (the slots in topk_slots index it); state is the _SM90State the metadata carries."""
    num_tokens = q_nope.shape[0]
    nd = int(getattr(attn_metadata, "num_decode_tokens", 0) or 0)
    nd = max(0, min(nd, num_tokens))

    if nd == num_tokens:
        # Pure decode: nothing for Triton -> upstream's pack + FA2 run, unchanged / graph-safe.
        return _fa2_run(impl, state, q_nope, q_pe, topk_slots, kv_cache_rows, layer, num_tokens)

    _check_impl_for_triton(impl)
    _require(
        valid_counts is not None,
        "the Triton prefill path needs the device valid_counts (return_valid_counts=True)",
    )
    _require(
        not _capturing(),
        "a mixed / prefill batch reached forward_mqa inside CUDA graph capture; the decode/prefill row "
        "split is host metadata and must never be baked into a graph",
    )
    planned = getattr(state, _PLAN_ROWS_ATTR, None)
    _require(
        planned == nd,
        f"the FA2 plan covers {planned} rows but forward_mqa runs {nd} decode rows on FA2 (the builder "
        "hook did not plan this metadata?)",
    )
    out = q_nope.new_empty((num_tokens, impl.num_heads, impl.kv_lora_rank))
    if nd > 0:
        out[:nd] = _fa2_run(impl, state, q_nope, q_pe, topk_slots, kv_cache_rows, layer, nd)

    # Prefill rows -> Triton, written straight into out[nd:]. The latent cache is the E4M3 ckv
    # [num_rows, 512] view; K and V are both dequantised with the per-tensor k_scale (FA2 passes the
    # same value as ckv_scale for the whole latent).
    latent = kv_cache_rows.view(torch.float8_e4m3fn).reshape(-1, impl.head_size)[:, : impl.kv_lora_rank]
    k_scale = float(layer._k_scale_float or 1.0)
    if not _FIRST_USE_LOGGED[0]:
        _FIRST_USE_LOGGED[0] = True
        logger.info(
            "[patch-020] first Triton prefill batch: %d prefill rows on Triton, %d decode rows on FA2, "
            "index width %d, k_scale %s",
            num_tokens - nd,
            nd,
            topk_slots.shape[1],
            k_scale,
        )
    sparse_mla_prefill_triton(
        q_nope[nd:],
        latent,
        topk_slots[nd:],
        valid_counts[nd:],
        sm_scale=float(impl.scale),
        k_scale=k_scale,
        v_scale=k_scale,
        out=out[nd:],
    )
    return out


# --------------------------------------------------------------------------------------------------
# Start-up (fail closed): called from FlashInferMLASparseSM90Impl.__init__ when the gate is set.
# --------------------------------------------------------------------------------------------------
_STARTUP_OK: set = set()


def _torch_reference(q, latent, slots, valid, sm_scale, k_scale, v_scale):
    """fp64 sparse attention over the DEQUANTISED latent (K = fp8 * k_scale, V = fp8 * v_scale),
    in row chunks of <= ~256 MiB of gathered fp64 KV. Returns fp64 [P, H, D] on q's device."""
    P, H, D = q.shape
    W = slots.shape[1]
    lat64 = latent.to(torch.float64)
    out = torch.zeros((P, H, D), dtype=torch.float64, device=q.device)
    step = max(1, (256 << 20) // max(1, W * D * 8))
    cols = torch.arange(W, device=q.device)[None, :]
    for lo in range(0, P, step):
        hi = min(P, lo + step)
        kv = lat64[slots[lo:hi].clamp(min=0).long()]  # [rows, W, D]
        s = torch.einsum("phd,pwd->phw", q[lo:hi].to(torch.float64), kv) * (sm_scale * k_scale)
        live = cols < valid[lo:hi].clamp(0, W).to(torch.int64)[:, None]
        s = s.masked_fill(~live[:, None, :], float("-inf"))
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)  # valid == 0 rows -> 0
        out[lo:hi] = torch.einsum("phw,pwd->phd", p, kv) * v_scale
    return out


def kernel_self_test(device, num_heads: int, sm_scale: float, k_scale: float = 0.5) -> dict:
    """Compile (warm) BOTH split variants on `device` and check them against an fp64 reference on
    tiny inputs, including |q| lanes far beyond fp16 range and a non-unit k_scale (V-side scaling).
    Raises Patch020Error on any non-finite output or an error above the (loose) sanity tolerance."""
    # Runs from Impl.__init__ inside the model loader's `with torch.device(cuda:N)` + bf16 default-dtype
    # context: build every input explicitly on the CPU in fp32 (a CPU generator on a CUDA default device
    # raises), then move it.
    g = torch.Generator(device="cpu").manual_seed(20261001)
    D, num_slots, W, hot = _D_LATENT, 192, 96, 7
    cpu = torch.device("cpu")
    lat = torch.randn(num_slots, D, generator=g, dtype=torch.float32, device=cpu) * 0.1
    lat[:, 0] = lat[:, 0].clamp(-0.2, 0.2)
    lat[hot, 0] = 1.0  # slot 7 is the unique max on dim 0 -> the huge-q heads are well-conditioned
    lat8 = lat.to(torch.float8_e4m3fn).to(device)
    report = {}
    for P in (_TWO_SPLIT_MAX_ROWS + 16, 3):  # S = 1 and S = 2
        qh = torch.randn(P, num_heads, D, generator=g, dtype=torch.float32, device=cpu) * 2.0**-4
        qh[0, 0, 0] = 2.0**17  # > fp16 max without the pre-scale
        qh[P - 1, num_heads - 1, 0] = 2.0**40  # far beyond fp16 range
        valid = torch.randint(1, W + 1, (P,), generator=g, dtype=torch.int32, device=cpu)
        valid[0] = W
        slots = torch.full((P, W), -1, dtype=torch.int32, device=cpu)
        for r in range(P):
            v = int(valid[r])
            perm = torch.randperm(num_slots, generator=g, device=cpu)[:v]
            if not bool((perm == hot).any()):
                perm[-1] = hot
            slots[r, :v] = perm.to(torch.int32)
        q = qh.to(torch.bfloat16).to(device)
        slots, valid = slots.to(device), valid.to(device)
        out = sparse_mla_prefill_triton(q, lat8, slots, valid, sm_scale, k_scale=k_scale)
        ref = _torch_reference(q, lat8, slots, valid, sm_scale, k_scale, k_scale)
        o = out.to(torch.float64)
        finite = bool(torch.isfinite(o).all())
        max_err = float((o - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
        mean_rel = float(((o - ref).abs() / ref.abs().clamp_min(1e-3)).mean())
        report[f"P{P}_S{_choose_num_splits(P)}"] = {
            "finite": finite,
            "max_err_rel_to_max": max_err,
            "mean_rel": mean_rel,
            "max_abs_q": float(q.float().abs().max()),
        }
        _require(
            finite and max_err <= 2e-2 and mean_rel <= 1e-2,
            f"Triton sparse-MLA kernel self-test failed on {device}: {report}",
        )
    return report


def _warm_unaligned_variant(device, num_heads: int, sm_scale: float) -> None:
    """Pre-compile the one other Triton specialisation production reaches: Triton specialises pointer arguments
    on 16-byte alignment, and in a mixed step the kernel gets valid_counts[nd:] (4*nd bytes into an aligned buffer,
    so unaligned unless nd % 4 == 0). Without this the first mixed step with such an nd would JIT-compile it (a
    one-off stall of seconds) inside a live request. q / out / slots rows [nd:] stay 16-byte aligned (their row
    strides are multiples of 16 bytes). Inputs are built explicitly on the CPU in fp32 for the same reason as in
    kernel_self_test (the model loader's device / dtype context)."""
    g = torch.Generator(device="cpu").manual_seed(20261023)
    D, num_slots, W, P = _D_LATENT, 192, 96, _TWO_SPLIT_MAX_ROWS + 16  # P > _TWO_SPLIT_MAX_ROWS: the S == 1 variant
    cpu = torch.device("cpu")
    lat8 = (torch.randn(num_slots, D, generator=g, dtype=torch.float32, device=cpu) * 0.1).to(torch.float8_e4m3fn).to(device)
    q = torch.randn(P, num_heads, D, generator=g, dtype=torch.float32, device=cpu).to(torch.bfloat16).to(device)
    slots = torch.randint(0, num_slots, (P, W), generator=g, dtype=torch.int32, device=cpu).to(device)
    valid = torch.full((P + 1,), W, dtype=torch.int32, device=cpu).to(device)
    sparse_mla_prefill_triton(q, lat8, slots, valid[1:], sm_scale, k_scale=1.0)  # valid[1:] is 4 bytes past a 16-byte boundary
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def startup_check(impl, device) -> None:
    """Fail closed at engine start-up: refuse an unsupported config (bf16 cache, rope MLA, other dims)
    and run kernel_self_test on the rank's device, which also JIT-compiles both kernel variants before
    serving (so no compile lands in the first measured prefill), then pre-compile the unaligned-valid_counts
    specialisation (best effort: a failure there only logs a warning). Logs the self-test report and one
    "[patch-020] ACTIVE" line per process and (config, device)."""
    key = (
        int(impl.num_heads),
        int(impl.kv_lora_rank),
        getattr(impl, "qk_rope_head_dim", None),
        bool(impl.use_fp8_kv_cache),
        str(device),
    )
    if key in _STARTUP_OK:
        return
    _check_impl_for_triton(impl)
    dev = torch.device(device)
    if dev.type == "cuda":
        with torch.cuda.device(dev):
            report = kernel_self_test(dev, int(impl.num_heads), float(impl.scale))
            _warm_best_effort(dev, impl)
    else:
        report = kernel_self_test(dev, int(impl.num_heads), float(impl.scale))
        _warm_best_effort(dev, impl)
    logger.info("[patch-020] Triton sparse-MLA prefill self-test OK on %s: %s", dev, report)
    logger.info(
        "[patch-020] ACTIVE rev 4 (VLLM_GB10_SPARSE_MLA_TRITON_PREFILL=1): FLASHINFER_MLA_SPARSE_SM90 prefill "
        "rows [nd, T) -> Triton sparse-MLA kernel, decode rows [0, nd) -> FlashInfer FA2 (heads=%d, D=%d, "
        "fp8 KV, NoPE); sync-free planning is upstream's (#58684), VLLM_GB10_SPARSE_MLA_NOSYNC is ignored",
        int(impl.num_heads),
        int(impl.kv_lora_rank),
    )
    _STARTUP_OK.add(key)


def _warm_best_effort(dev, impl) -> None:
    try:
        _warm_unaligned_variant(dev, int(impl.num_heads), float(impl.scale))
    except Exception as e:  # noqa: BLE001 - only costs a one-off compile inside the first mixed step
        logger.warning(
            "[patch-020] could not pre-compile the unaligned valid_counts variant (%r); the first mixed step compiles it", e
        )
