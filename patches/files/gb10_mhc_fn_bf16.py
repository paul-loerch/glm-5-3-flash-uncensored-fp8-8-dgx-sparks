# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [patch-019 gb10-mhc-fn-bf16] installed as vllm/_gb10_mhc_fn_bf16.py by patches/019-gb10-mhc-fn-bf16.sh (v0.32.0 port).
"""GB10: the fused mHC post + pre-norm decode kernel reads a bf16 copy of hc_*_fn (opt-in, lossless).

Used only by the patch-019 edits of vllm/model_executor/kernels/mhc/tilelang.py (mhc_fused_post_pre_tilelang) and
tilelang_kernels.py (MhcFusedTileLangKernel.register_warmup), and only when VLLM_GB10_MHC_FN_BF16=1. For
num_tokens <= 32 that op fuses the mHC post-mapping and the pre-norm GEMM into one TileLang kernel (mhc_fused_tilelang,
launched through the JIT class MhcFusedTileLangKernel; grid [M, 24 / tile_n, n_splits]; mhc_pre_big_fuse_with_norm
follows) whose CTAs read all of fn = hc_attn_fn / hc_ffn_fn [24, 4 * 4096] as fp32 (1.5 MiB per call, 89 calls per
target forward). The checkpoint stores these weights in BF16 and vLLM widens them into fp32 Parameters
(default_weight_loader: param.data.copy_(bf16)), so half of those bytes are zero low mantissa bits.
mhc_fused_bf16w_tilelang below is the same kernel reading a bf16 copy (0.75 MiB per call).

Since vLLM v0.30.0 (#50178) the mHC TileLang kernels are class-based JIT kernels with a warmup registry
(vllm/model_executor/warmup/jit_warmup*.py). This module therefore defines the bf16 variant as a kernel CLASS:
MhcFusedBf16wTileLangKernel(MhcFusedTileLangKernel) overrides only kernel() (-> the bf16-weight JIT function) and
warmup_inputs() (weight_t as a bf16 warmup tensor); dispatch / get_warmup_keys / __call__ are inherited, so its compile
keys are exactly the fp32 kernel's, the runtime launch goes through the same kernel_launcher, and the registry warms it
(patch 019's MhcFusedTileLangKernel.register_warmup override registers this instance whenever the model registers the
fp32 one; both are compiled before CUDA-graph capture).

Why the outputs are bit-identical (not just close):
  1. every fp32 element of a checkpoint-bf16 weight has zero low 16 bits, so fn.to(bfloat16) is exact and
     widening it back gives the identical fp32 bits. _to_bf16_exact() checks this bitwise for EVERY weight at
     conversion and refuses (raises) otherwise;
  2. in the kernel the bf16 weight enters `weight * new_r` where new_r is fp32: TileLang/TIR promotes the bf16
     operand to fp32 (an exact widening; the same promotion this kernel already applies to x_in and residual_in),
     so each FMA gets the same fp32 operands in the same order (same unrolled n/j loops, same warp and cross-warp
     reductions, same pass_configs [tilelang_jit] and PDL setting) -> the same yp_out / rp_out / residual_out bits;
  3. selftest() runs both kernels on the first converted weight at M = 1, 7, 8, 15, 16, 24, 32 (both
     (tile_n, n_splits, n_thr) configs the op uses and their boundaries) twice: the raw JIT functions with NaN-poisoned
     outputs (bitwise equal and fully written) and the production classes (MhcFusedTileLangKernel vs
     MhcFusedBf16wTileLangKernel; bitwise equal and equal to the raw result), and raises unless every output matches.

CUDA graphs: the copy must exist before capture (a conversion inside capture would be recorded and replayed every
step, and the exactness check needs a host sync). weight_for() therefore converts on the first EAGER call of each
weight at ANY M: the pre-capture profile run (M = max_num_batched_tokens) calls the op for every fused weight, and
the V2 runner also runs an eager warm-up pass before each capture. A weight first seen inside a capture gets None
(the original fp32 kernel in that graph: exact, no gain) and a warning. Copies are never freed (graphs keep their
pointers). Not supported: modifying the fp32 hc_*_fn in place after the first forward (weight reload / RL weight
sync) - the copy would be stale; this recipe never does it.

JIT warmup: with kernel_config.enable_jit_warmup the model registers MhcFusedTileLangKernel for warmup under the
runner's active JitWarmupRegistry; register_warmup() below is called from that registration (patch 019's override in
tilelang_kernels.py), so this class's keys (the same 2 for num_tokens <= 32: tile_n 2 / 6, n_thr 128) are compiled in
kernel_warmup, before capture. Without the registry (enable_jit_warmup off) the first conversion's self-test compiles
them eagerly in the profile run, also before capture. TileLang caches them under TILELANG_CACHE_DIR like every kernel.

Memory: 24 * 16384 * 2 B = 768 KiB per weight; GLM-5.3-Flash has 45 mHC layers -> 89 fused weights (layer-0 attn
uses mhc_pre_tilelang) = 66.75 MiB per rank (mHC weights are replicated on every TP rank).

Provenance / licence:
  * mhc_fused_bf16w_tilelang is a port of vLLM's mhc_fused_tilelang from vllm/model_executor/kernels/mhc/
    tilelang_kernels.py as shipped in vLLM v0.32.0 (file sha256 61a384a0..., Apache-2.0, Copyright contributors to the
    vLLM project). Changes (Apache-2.0 s.4(b)): renamed; `weight_t` annotated T.bfloat16 instead of T.float32;
    docstring and comments. p019_cpu_check.py proves by AST comparison that nothing else differs (the v0.32.0 kernel
    body is byte-identical to the one the 487ecf187 image shipped and patch 019 first ported). The class
    MhcFusedBf16wTileLangKernel subclasses vLLM's MhcFusedTileLangKernel.
  * The idea (a cached bf16 copy of hc_*_fn for this kernel, because the checkpoint stores them in bf16) comes from
    github.com/kindlingai/glm-5.3-flash-gx10 @ 3ed8fea "Read the fused mHC decode kernel's projection weights in
    bf16 (#47)" (experimental/fixes/mhc_tilelang.py `_fn_bf16`, experimental/fixes/tilelang_kernels.py; both files
    carry Apache-2.0 / vLLM-contributors headers). No code was copied from that repository: the cache key, the
    exactness check, the self-test, the eager-only conversion and the capture fallback are written for this recipe.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc import tilelang_kernels as _tk

# The same objects the upstream fp32 kernel is compiled with (T, tilelang, tilelang_jit [pass_configs], ENABLE_PDL), so
# both kernels get identical compile/pass configuration and PDL launch semantics.
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    ENABLE_PDL,
    MhcFusedTileLangKernel,
    mhc_fused_post_pre_split_config,
    mhc_fused_tilelang,
)
from vllm.model_executor.warmup.jit_warmup_tilelang_helper import make_tilelang_warmup_tensor
from vllm.tilelang_utils import T, tilelang, tilelang_jit

logger = init_logger(__name__)

ENV = "VLLM_GB10_MHC_FN_BF16"
TAG = "[patch-019]"
# both (tile_n, n_splits, n_thr) configs of mhc_fused_post_pre_split_config and their boundaries (M < 16: 2/8/128;
# 16 <= M <= 32: 6/8/128; M > 32: not fused)
SELFTEST_M = (1, 7, 8, 15, 16, 24, 32)


@tilelang_jit
def mhc_fused_bf16w_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
) -> tilelang.JITKernel:
    """Fused mhc post-mapping + pre-norm GEMM FMA, weight_t read as bf16 (patch 019)"""
    m = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    h_per_split = h // split_k
    n_tiles = n_out // tile_n

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    # [patch-019] the only change to the vLLM kernel: T.float32 -> T.bfloat16
    weight_t: T.Tensor((n_out, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

    h_iters = h_per_split // n_thr
    num_warps = n_thr // 32

    with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_n, i_nt, i_ks):
        tid = T.get_thread_binding()
        warp_id = tid // 32
        lane = tid % 32

        s_warp = T.alloc_shared((num_warps, tile_n + 1), T.float32)
        s_post = T.alloc_shared((hc,), T.float32)
        s_comb = T.alloc_shared((hc, hc), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        new_r = T.alloc_local((hc,), T.float32)

        T.clear(acc)
        T.clear(sqr)
        h_split_start = i_ks * h_per_split

        if ENABLE_PDL:
            T.pdl_sync()

        T.copy(post_mix[i_n, 0], s_post)
        T.copy(comb_mix[i_n, 0, 0], s_comb)

        for j in T.unroll(hc):
            pm[j] = s_post[j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = s_comb[k, j]

        # Each thread owns h_iters elements of the k-split's h slice.
        for it in T.serial(h_iters):
            h_idx = h_split_start + it * n_thr + tid

            # Compute new residual from layer output and past residual
            for j in T.unroll(hc):
                new_r[j] = pm[j] * x_in[i_n, h_idx]
                for k in T.unroll(hc):
                    new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]

            # populate residual_out and compute sqr sum
            if i_nt == 0:
                for j in T.unroll(hc):
                    residual_out[i_n, j, h_idx] = new_r[j]
                    sqr[0] += new_r[j] * new_r[j]

            # Per-thread FMA into acc[n]. [patch-019] bf16 weight * fp32 new_r: the weight is promoted to
            # fp32 (exact), like x_in / residual_in above, so every FMA sees the fp32 kernel's operands.
            for n in T.unroll(tile_n):
                for j in T.unroll(hc):
                    acc[n] += weight_t[i_nt * tile_n + n, j, h_idx] * new_r[j]

        for n in T.unroll(tile_n):
            acc[n] = T.warp_reduce_sum(acc[n])
        if i_nt == 0:
            sqr[0] = T.warp_reduce_sum(sqr[0])

        # Cross-warp reduce via shared mem
        if lane == 0:
            for n in T.unroll(tile_n):
                s_warp[warp_id, n] = acc[n]
            if i_nt == 0:
                s_warp[warp_id, tile_n] = sqr[0]
        T.sync_threads()

        # Warp 0 does the final cross-warp sum and writes outputs
        if warp_id == 0:
            if lane < tile_n:
                v = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v += s_warp[w, lane]
                yp_out[i_ks, i_n, i_nt * tile_n + lane] = v

            if i_nt == 0 and lane == 0:
                v2 = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v2 += s_warp[w, tile_n]
                rp_out[i_ks, i_n] = v2

        if ENABLE_PDL:
            T.pdl_trigger()


class MhcFusedBf16wTileLangKernel(MhcFusedTileLangKernel):
    """MhcFusedTileLangKernel launching mhc_fused_bf16w_tilelang (bf16 weight_t). dispatch(), get_warmup_keys() and
    __call__() are inherited: same compile keys, same launch spec; only the JIT function and the warmup weight differ."""

    @staticmethod
    def kernel() -> Any:
        return mhc_fused_bf16w_tilelang

    def warmup_inputs(self, compile_key):
        inputs = super().warmup_inputs(compile_key)
        w = inputs["weight_t"]
        inputs["weight_t"] = make_tilelang_warmup_tensor(torch.bfloat16, *w.shape)
        return inputs


MHC_FUSED_BF16W_KERNEL = MhcFusedBf16wTileLangKernel()


def register_warmup(*args: Any, **kwargs: Any) -> None:
    """Register the bf16 variant with the active JIT warmup registry (called by the fp32 class's register_warmup)."""
    MHC_FUSED_BF16W_KERNEL.register_warmup(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# bf16 copies of the fused-op weights: made once per weight on its first eager call, kept for the process lifetime.
# ---------------------------------------------------------------------------------------------------------------
_CACHE: dict[tuple, torch.Tensor] = {}
# The fp32 source of every copy is kept referenced: its storage can never be freed and reused by another tensor
# under the same data_ptr key (so a key hit always means the same, unmodified-by-allocation weight).
_SRC: dict[tuple, torch.Tensor] = {}
_MISSED: set[tuple] = set()
_STATE = {"selftest": False, "converted": 0, "bytes": 0, "capture_misses": 0, "capture_logged": False}


def _key(fn: torch.Tensor) -> tuple:
    return (fn.data_ptr(), fn.device.type, fn.device.index, tuple(fn.shape), tuple(fn.stride()), fn.dtype)


def _fused_config(num_tokens: int, hidden_size: int, hc_mult: int = 4) -> tuple[int, int, int]:
    """(tile_n, n_splits, n_thr) exactly as mhc_fused_post_pre_tilelang picks them (upstream's own function: one
    source of truth for dispatch and launch). Raises if the op would not fuse num_tokens."""
    cfg = mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)
    if cfg is None:
        raise RuntimeError(f"{TAG} the fused mHC kernel does not cover num_tokens={num_tokens} at hidden_size="
                           f"{hidden_size}; refusing")
    return cfg


def _validate(fn: torch.Tensor, hc_mult: int, hidden_size: int) -> None:
    want = (hc_mult * (hc_mult + 2), hc_mult * hidden_size)
    if (fn.dtype != torch.float32 or tuple(fn.shape) != want or not fn.is_contiguous()
            or fn.device.type != "cuda"):
        raise RuntimeError(
            f"{TAG} {ENV}=1: unexpected mHC weight {tuple(fn.shape)} {fn.dtype} "
            f"contiguous={fn.is_contiguous()} on {fn.device} (want fp32 {want} contiguous on cuda); "
            f"unset {ENV}")


def _to_bf16_exact(fn: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(bf16 copy, number of elements whose fp32 bits do not survive fp32 -> bf16 -> fp32)."""
    with torch.no_grad():
        src = fn.detach()
        w = src.to(torch.bfloat16).contiguous()
        back = w.to(torch.float32)
        if torch.equal(back.view(torch.int32), src.view(torch.int32)):
            return w, 0
        return w, int((back.view(torch.int32) != src.view(torch.int32)).sum())


def selftest(fn: torch.Tensor, w: torch.Tensor, hc_mult: int, hidden_size: int) -> list[dict]:
    """Run the upstream fp32 kernel on fn and the bf16 kernel on w at SELFTEST_M; raise unless bitwise equal."""
    dev, n_out = fn.device, fn.shape[0]
    fp32_cls = _tk._MHC_FUSED_TILELANG_KERNEL
    g = torch.Generator(device=dev)
    g.manual_seed(20260930)
    rows = []
    for m in SELFTEST_M:
        tile_n, n_splits, n_thr = _fused_config(m, hidden_size, hc_mult)
        comb = torch.softmax(torch.randn(m, hc_mult, hc_mult, generator=g, device=dev), dim=-1)
        post = torch.sigmoid(torch.randn(m, hc_mult, generator=g, device=dev)) * 2.0
        res = (torch.randn(m, hc_mult, hidden_size, generator=g, device=dev) * 4.0).to(torch.bfloat16)
        x = torch.randn(m, hidden_size, generator=g, device=dev).to(torch.bfloat16)
        # (1) the raw JIT functions with NaN-poisoned outputs: bitwise equal AND every output element written
        raw = []
        for kernel, weight in ((mhc_fused_tilelang, fn), (mhc_fused_bf16w_tilelang, w)):
            yp = torch.full((n_splits, m, n_out), float("nan"), dtype=torch.float32, device=dev)
            rp = torch.full((n_splits, m), float("nan"), dtype=torch.float32, device=dev)
            ro = torch.full((m, hc_mult, hidden_size), float("nan"), dtype=torch.bfloat16, device=dev)
            kernel(comb, res, post, x, weight.view(n_out, hc_mult, hidden_size), yp, rp, ro,
                   hc_mult, hidden_size, n_out, n_thr=n_thr, tile_n=tile_n)
            raw.append((yp, rp, ro))
        # (2) the production classes (the call path the op takes): upstream's instance vs this variant
        cls = []
        for inst, weight in ((fp32_cls, fn), (MHC_FUSED_BF16W_KERNEL, w)):
            cls.append(inst(comb, res, post, x, weight.view(n_out, hc_mult, hidden_size), hc_mult, hidden_size,
                            n_out))

        def same(a, b):
            return (torch.equal(a[0].view(torch.int32), b[0].view(torch.int32))
                    and torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))
                    and torch.equal(a[2].view(torch.int16), b[2].view(torch.int16)))

        equal_raw = same(raw[0], raw[1])
        equal_cls = same(cls[0], cls[1])
        cls_is_raw = same(cls[0], raw[0])  # the class path launches the same kernel with the same config
        written = not (torch.isnan(raw[1][0]).any() or torch.isnan(raw[1][1]).any() or torch.isnan(raw[1][2]).any())
        rows.append({"M": m, "tile_n": tile_n, "n_splits": n_splits, "n_thr": n_thr, "equal": bool(equal_raw),
                     "equal_class": bool(equal_cls), "class_is_raw": bool(cls_is_raw), "written": bool(written)})
        if not (equal_raw and equal_cls and cls_is_raw and written):
            raise RuntimeError(
                f"{TAG} {ENV}=1: SELF-TEST FAILED at M={m} (tile_n {tile_n}, n_splits {n_splits}, n_thr {n_thr}): "
                f"bf16-weight kernel {'differs from' if not equal_raw else 'matches'} the fp32 kernel (raw JIT), "
                f"class path {'differs' if not equal_cls else 'matches'}"
                f"{'' if cls_is_raw else ', class path != raw launch'}"
                f"{'' if written else ' and left outputs unwritten'}; refusing to serve with it - unset {ENV}")
    return rows


def _convert(fn: torch.Tensor, hc_mult: int, hidden_size: int) -> torch.Tensor:
    _validate(fn, hc_mult, hidden_size)
    w, n_bad = _to_bf16_exact(fn)
    if n_bad:
        raise RuntimeError(
            f"{TAG} {ENV}=1 needs hc_*_fn that are exactly representable in bf16 (this checkpoint stores them as "
            f"BF16), but a weight {tuple(fn.shape)} has {n_bad} elements that are not: its bf16 copy would change "
            f"the outputs; unset {ENV}")
    if not _STATE["selftest"]:
        rows = selftest(fn, w, hc_mult, hidden_size)
        _STATE["selftest"] = True
        logger.info(
            "%s mHC fn bf16 ACTIVE: fused decode kernel (M <= %d) reads bf16 copies of hc_*_fn; self-test "
            "bit-exact vs the fp32 kernel (raw JIT and class path) at %s; each copy is verified exact "
            "(768 KiB per weight)",
            TAG, _tk._FUSED_POST_PRE_MAX_TOKENS,
            ", ".join(f"M={r['M']} (tile_n {r['tile_n']}, splits {r['n_splits']}, thr {r['n_thr']})" for r in rows))
    _STATE["converted"] += 1
    _STATE["bytes"] += w.numel() * w.element_size()
    return w


def weight_for(fn: torch.Tensor, hc_mult: int, hidden_size: int) -> torch.Tensor | None:
    """bf16 copy of fn for mhc_fused_bf16w_tilelang, or None: keep the upstream fp32 kernel for this call."""
    key = _key(fn)
    w = _CACHE.get(key)
    if w is not None:
        if not _STATE["capture_logged"] and torch.cuda.is_current_stream_capturing():
            _STATE["capture_logged"] = True
            logger.info(
                "%s %d bf16 hc_*_fn copies (%.2f MiB) serve the fused mHC decode kernel (M <= %d) in CUDA graphs; "
                "weights first seen inside a capture (fp32 kernel kept there): %d",
                TAG, _STATE["converted"], _STATE["bytes"] / 2**20, _tk._FUSED_POST_PRE_MAX_TOKENS,
                _STATE["capture_misses"])
        return w
    if torch.cuda.is_current_stream_capturing():
        # Converting here would be recorded in the graph (re-run every replay) and the exactness check needs a
        # host sync: keep the original fp32 kernel for this graph (bit-identical, no gain).
        if key not in _MISSED:
            _MISSED.add(key)
            _STATE["capture_misses"] += 1
            logger.warning(
                "%s mHC weight %s first seen inside a CUDA-graph capture: this graph keeps the fp32 fused kernel "
                "(exact, no speed-up). Expected the eager profile run to convert every weight first.",
                TAG, tuple(fn.shape))
        return None
    w = _convert(fn, hc_mult, hidden_size)
    _CACHE[key] = w
    _SRC[key] = fn
    return w


def stats() -> dict:
    return {"selftest": _STATE["selftest"], "converted": _STATE["converted"], "bytes": _STATE["bytes"],
            "capture_misses": _STATE["capture_misses"], "cached": len(_CACHE)}


def _reset_for_test() -> None:
    """Tests only: forget every copy (graphs captured before this call still hold their pointers)."""
    _CACHE.clear()
    _SRC.clear()
    _MISSED.clear()
    _STATE.update(selftest=False, converted=0, bytes=0, capture_misses=0, capture_logged=False)
