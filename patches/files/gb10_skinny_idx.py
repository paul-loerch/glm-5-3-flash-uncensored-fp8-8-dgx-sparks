"""GB10 decode-M kernels for the GLM-5.3-Flash DSA indexer's three narrow projections (recipe patch 014, opt-in
VLLM_GB10_SKINNY_IDX=1). Installed as vllm/_gb10_skinny_idx.py and imported by models/glm5next/common/attention.py
(models/glm5next/nvidia/attention.py on the day-0 image).

vLLM v0.32.0 port: this file is unchanged except for this paragraph. Checked against the v0.32.0 sources it calls: the CuTe
DSL kernels (model_executor/kernels/linear/cute_dsl/ll_bf16.py: LLBf16Gemm.CompileKey / compile / _compiled_cache[(M, K, bs)]
and the compiled call (x, w, out, N, stream), module-level is_available() and _stream(); skinny_gemm.py:
SkinnyGemmConfig(num_rows, block_size, outputs_per_block, k_unroll, vector_width), ShapeDynamicSkinnyGemm.is_available(),
the (dtype, config, has_residual) cache key, and the new row_stride_ok() check that a [M, 4096] contiguous bf16 input with
16-byte alignment, as use() requires, satisfies). The dot-product kernel's PDL variants (v0.32.0 adds a weight-prefetch
specialisation used only for M == 1 in ll_bf16_gemm(), which this module does not call) only matter when
current_platform.is_arch_support_pdl() is true; the SM12x image bake switches PDL off, which keeps these kernels identical to
the day-0 ones. v0.32.0's Indexer sets `self._wp_fp32 = None` in __init__ (the lazily created attribute of the day-0 image):
selftest() already reads it with getattr(..., None).

WHY: Indexer.forward runs three narrow projections of hidden_states [M,4096] per DSA layer (11 target layers + the MTP
layer): wk_weights_proj [160,4096] bf16, the fp32 head gate torch.mm(x.float(), _wp_fp32[4096,32]) and the kpool gate
[128,4096] bf16. At decode M cuBLAS launches 1-16 CTAs for them (the head gate is gemmSN_NN on ONE CTA at M=3). In a
decode profile they cost 132 us per call at M=3 and 1.62 ms per k=2 decode step.

ROUTES (per M = hidden_states.shape[0]; measured by idx-microbench.py in a rank container on node1, 2026-09-28:
DRAM-cold CUDA-graph timing with real layer-3 weights, idxmb.json):
  M      wk_weights_proj [160]        head gate fp32 [32]                  kpool gate [128]           us/call (today)
  1-2    skinny bs256 opb1 ku1 vw8    ll_bf16 dotprod bs256                skinny bs256 opb1 ku1 vw8  12.4 (31.4) at M=1
  3      skinny bs128 opb1 ku2 vw8    ll_bf16 dotprod bs256                skinny bs256 opb1 ku1 vw8  12.7 (130.6)
  4-8    cuBLAS, unchanged            ll_bf16 dotprod bs128                cuBLAS, unchanged          ~21-22 (93)
  9-16   cuBLAS, unchanged            Triton split-K mma16 BN16 BK64 S32   cuBLAS, unchanged          ~24 (89) at M=16
  M > 16, M == 0, non-bf16, non-contiguous or misaligned input: the original path, untouched.
M=1 and M=3 are the bench's P1 "no-repack" composite. M=2 was not measured; it uses the M=1 configs and the load-time
self-test must pass for it like every other M. The head gate reads the contiguous row slice W160[128:160] (no fp32 twin).
From M=4 on, today's cuBLAS bf16 kernel differs from any exact-order kernel by more than 1 bf16 ulp in cancellation
cases, so the bench's eligibility rule keeps wk / gate on today's call (bit-identical) and only the head gate changes.
The bench's faster M>=4 option (one cuBLAS GEMM over a repacked [wk | gate] weight, ~5 us/call less) returns [M,256]
row strides and would break the output contract below, so it is not used.

CONTRACT (unchanged downstream): k is the view [:, :128] of a [M,160] bf16 buffer (row stride 160, as
_fused_indexer_k_norm has always seen it); weights is fp32 [M,32] contiguous; gate_score is bf16 [M,128] contiguous.

NUMERICS: every kernel accumulates in fp32 with a fixed reduction order and no atomics, so results are deterministic and
identical on all ranks (the indexer is replicated). Head gate: bf16 x bf16 products are exact in fp32, only the summation
order changes. The load-time self-test applies the bench's rules at every M in 1..16 on 2 normal + 2 heavy-tailed input
draws: fp32 head gate max|y - exact| / sum|x w| <= max(1e-6, 2x today's); bf16 outputs within max(1, today's) bf16 ulp of
the fp64 exact result and within 1 bf16 ulp of today's; 3 runs bitwise identical; output layout equal to today's.

CUDA GRAPHS: cudagraph_num_of_warmups is 0 in this build, so a kernel launched for the first time inside capture would
JIT-compile there. use() therefore prepares each Indexer at its first call outside capture (vLLM's eager profile run
reaches every target and MTP indexer before capture): the first such call compiles every CuTe / Triton variant for
M = 1..16, then each layer runs the self-test on its own weights. A first call inside capture, or project() on an
unprepared layer, raises (fail-closed). CuTe kernels launch on vLLM's tracked stream, Triton and torch on the current
stream; both are the capture stream during capture. PDL stays off (SM12x).
"""

from __future__ import annotations

import os
import threading
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger("vllm._gb10_skinny_idx")

ENV = "VLLM_GB10_SKINNY_IDX"
HIDDEN, HEAD_DIM, N_HEAD = 4096, 128, 32  # config.json: hidden_size, index_head_dim, index_n_heads
N_WK = HEAD_DIM + N_HEAD  # wk_weights_proj rows: wk [0:128] | weights_proj [128:160]
MAX_M = 16
TOL32 = 1e-6  # bench --tol32

# Per-M routes: (wk_weights_proj, head gate, kpool gate).
#   skinny config = (block_size, outputs_per_block, k_unroll, vector_width) of vLLM's SkinnyGemmConfig; None = today's
#   cuBLAS call. head gate = ("ll", bs): in-tree ll_bf16 dot-product kernel, forced bs | ("triton", (MODE, BLOCK_N,
#   BLOCK_K, SPLIT_K, num_warps)): the bench's split-K GEMV, MODE 2 = bf16 tl.dot with fp32 accumulation.
_SK_BS256 = (256, 1, 1, 8)
_SK_BS128_KU2 = (128, 1, 2, 8)
_TRITON_HG = (2, 16, 64, 32, 4)
ROUTES: dict[int, tuple] = {
    1: (_SK_BS256, ("ll", 256), _SK_BS256),
    2: (_SK_BS256, ("ll", 256), _SK_BS256),
    3: (_SK_BS128_KU2, ("ll", 256), _SK_BS256),
    **{m: (None, ("ll", 128), None) for m in range(4, 9)},
    **{m: (None, ("triton", _TRITON_HG), None) for m in range(9, MAX_M + 1)},
}

_LOCK = threading.RLock()
_S: dict = {"compiled": False, "announced": False, "sk": None, "ll": None, "sk_cfg": {}, "ll_kern": {},
            "dummy_bf16": None, "compile_s": None, "warned": set()}


# ----------------------------------------------------------------------------------------------------------------------
# Triton split-K GEMV, verbatim from idx-microbench.py (fp32 accumulation, deterministic two-pass fixed-order reduce)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _gemv_splitk_kernel(x_ptr, w_ptr, part_ptr, o16_ptr, o32_ptr,
                        M, N, K, N16, stride_xm, stride_wn, x_goff, w_goff, k_per_split,
                        M_PAD: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        SPLIT_K: tl.constexpr, MODE: tl.constexpr):
    # out[g, m, n] = sum_k x[m, g*x_goff + k] * w[g*w_goff/stride + n, k]   (fp32 accumulate)
    # MODE 0: FMA broadcast-multiply + tree sum; 1: tl.dot fp32 "ieee" (FMA); 2: tl.dot bf16 (tensor cores)
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_g = tl.program_id(2)
    x_ptr += pid_g * x_goff
    w_ptr += pid_g * w_goff
    offs_m = tl.arange(0, M_PAD)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    k0 = pid_k * k_per_split
    for kk in range(0, k_per_split, BLOCK_K):
        offs_k = k0 + kk + offs_kb
        k_mask = offs_k < K
        xv = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wv = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0, eviction_policy="evict_first")
        if MODE == 0:
            acc += tl.sum(xv.to(tl.float32)[:, None, :] * wv.to(tl.float32)[None, :, :], axis=2)
        elif MODE == 1:
            acc += tl.dot(xv.to(tl.float32), tl.trans(wv.to(tl.float32)), input_precision="ieee")
        else:
            acc += tl.dot(xv, tl.trans(wv))
    o_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        n32 = N - N16
        tl.store(o16_ptr + pid_g * M * N16 + offs_m[:, None] * N16 + offs_n[None, :],
                 acc.to(tl.bfloat16), mask=o_mask & (offs_n < N16)[None, :])
        tl.store(o32_ptr + pid_g * M * n32 + offs_m[:, None] * n32 + (offs_n - N16)[None, :],
                 acc, mask=o_mask & (offs_n >= N16)[None, :])
    else:
        tl.store(part_ptr + (pid_g * SPLIT_K + pid_k) * M * N + offs_m[:, None] * N + offs_n[None, :],
                 acc, mask=o_mask)


@triton.jit
def _splitk_reduce_kernel(part_ptr, o16_ptr, o32_ptr, M, N, N16,
                          SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    # fixed, sequential split order -> deterministic, identical on every rank
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)
    mn = M * N
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < mn
    base = part_ptr + pid_g * SPLIT_K * mn
    acc = tl.load(base + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(base + s * mn + offs, mask=mask, other=0.0)
    m = offs // N
    n = offs - m * N
    n32 = N - N16
    tl.store(o16_ptr + pid_g * M * N16 + m * N16 + n, acc.to(tl.bfloat16), mask=mask & (n < N16))
    tl.store(o32_ptr + pid_g * M * n32 + m * n32 + (n - N16), acc, mask=mask & (n >= N16))


# ----------------------------------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------------------------------
def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _name(layer) -> str:
    return getattr(layer, "prefix", None) or f"Indexer@{id(layer):x}"


def _skinny(x: torch.Tensor, w: torch.Tensor, route: tuple) -> torch.Tensor:
    """bf16 [M,K] x [N,K]^T -> bf16 [M,N] contiguous (vLLM's shape-dynamic skinny GEMM, config forced per M)."""
    cfg = _S["sk_cfg"][(x.shape[0], route)]
    sk = _S["sk"].shape_dynamic_skinny_gemm
    if (x.dtype, cfg, False) not in sk._compiled:  # would JIT-compile here (inside capture, possibly)
        raise RuntimeError(f"[patch-014] skinny GEMM {cfg} not compiled (fail-closed)")
    return sk(x, w, cfg)


def _head_gate(x: torch.Tensor, wp: torch.Tensor, route: tuple) -> torch.Tensor:
    """bf16 x [M,4096] . bf16 wp [32,4096]^T -> fp32 [M,32] contiguous."""
    m = x.shape[0]
    kind, arg = route
    out = torch.empty((m, N_HEAD), dtype=torch.float32, device=x.device)
    if kind == "ll":
        _S["ll_kern"][(m, arg)](x, wp, out, N_HEAD, _S["ll"]._stream())
        return out
    mode, bn, bk, split, warps = arg
    part = torch.empty((split, m, N_HEAD), dtype=torch.float32, device=x.device)
    dummy = _S["dummy_bf16"]  # o16 is never written (N16 = 0)
    _gemv_splitk_kernel[(triton.cdiv(N_HEAD, bn), split, 1)](
        x, wp, part, dummy, out, m, N_HEAD, HIDDEN, 0, x.stride(0), wp.stride(0), 0, 0, HIDDEN // split,
        M_PAD=16, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, MODE=mode, num_warps=warps)
    _splitk_reduce_kernel[(triton.cdiv(m * N_HEAD, 256), 1)](
        part, dummy, out, m, N_HEAD, 0, SPLIT_K=split, BLOCK=256, num_warps=4)
    return out


def _compute(layer, x: torch.Tensor):
    wk_r, hg_r, g_r = ROUTES[x.shape[0]]
    w_all = layer.wk_weights_proj.weight.detach()
    if wk_r is None:
        kw, _ = layer.wk_weights_proj(x)  # today's call, bit-identical
    else:
        kw = _skinny(x, w_all, wk_r)  # [M,160]: same buffer layout as today's kw
    weights = _head_gate(x, w_all[HEAD_DIM:], hg_r)  # contiguous row slice, no fp32 twin
    if g_r is None:
        gate_score = F.linear(x, layer.index_kpool_compress_gate)  # today's call, bit-identical
    else:
        gate_score = _skinny(x, layer.index_kpool_compress_gate.detach(), g_r)
    return kw[:, :HEAD_DIM], weights, gate_score


def _original(layer, x: torch.Tensor, wp32: torch.Tensor):
    """Today's Indexer.forward lines (attention.py), for the self-test."""
    kw, _ = layer.wk_weights_proj(x)
    return kw[:, :HEAD_DIM], torch.mm(x.float(), wp32), F.linear(x, layer.index_kpool_compress_gate)


# ----------------------------------------------------------------------------------------------------------------------
# warm-up (compile every variant) and self-test
# ----------------------------------------------------------------------------------------------------------------------
def _ensure_compiled(device: torch.device) -> None:
    if _S["compiled"]:
        return
    if _capturing():
        raise RuntimeError("[patch-014] kernels are not compiled and a CUDA graph is being captured (fail-closed)")
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"):
        raise RuntimeError(f"[patch-014] {ENV}=1 conflicts with VLLM_BATCH_INVARIANT (per-M kernels); unset one of them")
    from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16 as LL
    from vllm.model_executor.kernels.linear.cute_dsl import skinny_gemm as SK

    if not SK.ShapeDynamicSkinnyGemm.is_available() or not LL.is_available():
        raise RuntimeError("[patch-014] CuTe DSL (cutlass) is not importable: cannot build the skinny indexer kernels")
    t0 = time.perf_counter()
    x = torch.zeros(MAX_M, HIDDEN, dtype=torch.bfloat16, device=device)
    w_wk = torch.zeros(N_WK, HIDDEN, dtype=torch.bfloat16, device=device)
    w_g = torch.zeros(HEAD_DIM, HIDDEN, dtype=torch.bfloat16, device=device)
    sk_cfg, ll_kern, where = {}, {}, "?"
    try:
        for m, (wk_r, hg_r, g_r) in ROUTES.items():
            for route, w in ((wk_r, w_wk), (g_r, w_g)):
                if route is not None:
                    where = f"skinny M={m} {route}"
                    cfg = SK.SkinnyGemmConfig(m, *route)
                    sk_cfg[(m, route)] = cfg
                    SK.shape_dynamic_skinny_gemm(x[:m], w, cfg)  # compiles (M, cfg) once (N is symbolic), launches
            kind, arg = hg_r
            if kind == "ll":
                where = f"ll_bf16 dotprod M={m} bs{arg}"
                LL.ll_bf16_gemm_kernel.compile(LL.LLBf16Gemm.CompileKey(backend="dotprod", M=m, K=HIDDEN, bs=arg))
                ll_kern[(m, arg)] = LL.ll_bf16_gemm_kernel._compiled_cache[(m, HIDDEN, arg)]
        _S.update(sk=SK, ll=LL, sk_cfg=sk_cfg, ll_kern=ll_kern,
                  dummy_bf16=torch.zeros(16, dtype=torch.bfloat16, device=device))
        for m, (_, hg_r, _) in ROUTES.items():  # first launch of every head-gate variant (Triton JIT specialisations)
            where = f"head gate M={m} {hg_r}"
            _head_gate(x[:m], w_wk[HEAD_DIM:], hg_r)
        where = "synchronize"
        torch.cuda.synchronize(device)
    except Exception as e:
        raise RuntimeError(f"[patch-014] kernel build failed at {where}: {type(e).__name__}: {e} (unset {ENV} to boot "
                           "without patch 014)") from e
    _S["compile_s"] = time.perf_counter() - t0
    _S["compiled"] = True


def _check_layer(layer) -> None:
    problems = []
    if getattr(layer, "head_dim", None) != HEAD_DIM or getattr(layer, "n_head", None) != N_HEAD:
        problems.append(f"head_dim/n_head {getattr(layer, 'head_dim', None)}/{getattr(layer, 'n_head', None)} "
                        f"!= {HEAD_DIM}/{N_HEAD}")
    if getattr(layer.wk_weights_proj, "bias", None) is not None:
        problems.append("wk_weights_proj has a bias")
    for nm, t, shape in (("wk_weights_proj.weight", layer.wk_weights_proj.weight, (N_WK, HIDDEN)),
                         ("index_kpool_compress_gate", layer.index_kpool_compress_gate, (HEAD_DIM, HIDDEN))):
        if (tuple(t.shape) != shape or t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous()
                or t.data_ptr() % 16):
            problems.append(f"{nm}: {tuple(t.shape)} {t.dtype} {t.device} contiguous={t.is_contiguous()} "
                            f"ptr%16={t.data_ptr() % 16} (want {shape} bf16 cuda contiguous aligned)")
    if problems:
        raise RuntimeError(f"[patch-014] {_name(layer)} does not match the measured shapes: " + "; ".join(problems))


def _ulp_of(v: torch.Tensor, bits: int) -> torch.Tensor:
    """spacing of the floating grid at |v| (float64 tensor): bf16 bits=8, fp32 bits=24 (as in the bench)."""
    _, e = torch.frexp(v)
    u = torch.ldexp(torch.ones_like(v), e - bits)
    return torch.where(v > 0, u, torch.full_like(v, 2.0 ** (-126 - bits + 1)))


def compare(y: torch.Tensor, c: torch.Tensor, ex: torch.Tensor, s: torch.Tensor, kind: str, tol32: float = TOL32):
    """The bench's eligibility rule. y: candidate, c: today's output, ex: fp64 exact, s: fp64 sum|x||w| ([M,N])."""
    y64, c64 = y.double(), c.double()
    d_cur = (y64 - c64).abs()
    d_ex = (y64 - ex).abs()
    s = s.clamp_min(1e-30)
    d_today = (c64 - ex).abs()
    r = {"max_abs_vs_cur": d_cur.max().item(),
         "norm_err_vs_exact": (d_ex / s).max().item(),
         "today_norm_err_vs_exact": (d_today / s).max().item()}
    if kind == "bf16":
        den = _ulp_of(ex.abs(), 8) + 1e-6 * s
        r["ulp_ratio_vs_exact"] = (d_ex / den).max().item()
        r["today_ulp_ratio_vs_exact"] = (d_today / den).max().item()
        r["ulp_ratio_vs_cur"] = (d_cur / (_ulp_of(torch.maximum(c64.abs(), y64.abs()), 8) + 1e-6 * s)).max().item()
        r["frac_elems_differ_vs_cur"] = (y != c).double().mean().item()
        r["ok"] = (r["ulp_ratio_vs_exact"] <= max(1.0, r["today_ulp_ratio_vs_exact"])
                   and r["ulp_ratio_vs_cur"] <= 1.0)
    else:
        r["ok"] = r["norm_err_vs_exact"] <= max(tol32, 2.0 * r["today_norm_err_vs_exact"])
    return r


def make_draws(device: torch.device, n: int = 4, seed: int = 20260928) -> list[torch.Tensor]:
    """n bf16 [16,4096] inputs, generated on the CPU (identical on every rank): first half N(0,1), second half with
    heavy-tailed outlier channels (x * exp(1.5 * N(0,1)) per channel), as in the bench."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for d in range(n):
        x = torch.randn(MAX_M, HIDDEN, generator=g)
        if d >= n // 2:
            x = x * torch.exp(torch.randn(1, HIDDEN, generator=g) * 1.5)
        out.append(x.to(torch.bfloat16).to(device).contiguous())
    return out


def _layout_problems(new, ref) -> list[str]:
    out = []
    for nm, a, b in zip(("k", "weights", "gate_score"), new, ref):
        if (a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride()
                or a.storage_offset() != b.storage_offset() or a.device != b.device):
            out.append(f"{nm} {tuple(a.shape)} {a.dtype} stride {a.stride()} +{a.storage_offset()} vs today "
                       f"{tuple(b.shape)} {b.dtype} stride {b.stride()} +{b.storage_offset()}")
    return out


def selftest(layer, ms=None, draws: int = 4, det_runs: int = 3, seed: int = 20260928, tol32: float = TOL32) -> dict:
    """Compare the per-M kernels with today's path on this layer's weights. Raises RuntimeError on any failure."""
    ms = list(ms) if ms is not None else list(range(1, MAX_M + 1))
    w_all = layer.wk_weights_proj.weight.detach()
    gate = layer.index_kpool_compress_gate.detach()
    dev = w_all.device
    wp32 = getattr(layer, "_wp_fp32", None)
    if wp32 is None:  # exactly what Indexer.forward builds lazily (not stored: no side effect on the layer)
        wp32 = w_all[HEAD_DIM:, :].t().contiguous().float()
    w64 = (w_all[:HEAD_DIM].double(), w_all[HEAD_DIM:].double(), gate.double())
    w64abs = tuple(w.abs() for w in w64)
    kinds = ("bf16", "fp32", "bf16")
    xs = make_draws(dev, draws, seed)
    fails, worst = [], {"hg_err": 0.0, "hg_err_today": 0.0, "bf16_ulp_vs_today": 0.0, "bf16_frac_differ": 0.0}
    for m in ms:
        for d, xa in enumerate(xs):
            x = xa[:m]
            new = _compute(layer, x)
            ref = _original(layer, x, wp32)
            lp = _layout_problems(new, ref)
            if lp:
                fails.append(f"M={m}: layout " + "; ".join(lp))
            x64 = x.double()
            ax = x64.abs()
            for j in range(3):
                r = compare(new[j], ref[j], x64 @ w64[j].t(), ax @ w64abs[j].t(), kinds[j], tol32)
                if kinds[j] == "fp32":
                    worst["hg_err"] = max(worst["hg_err"], r["norm_err_vs_exact"])
                    worst["hg_err_today"] = max(worst["hg_err_today"], r["today_norm_err_vs_exact"])
                else:
                    worst["bf16_ulp_vs_today"] = max(worst["bf16_ulp_vs_today"], r["ulp_ratio_vs_cur"])
                    worst["bf16_frac_differ"] = max(worst["bf16_frac_differ"], r["frac_elems_differ_vs_cur"])
                if not r["ok"]:
                    fails.append(f"M={m} draw {d} {('k', 'weights', 'gate_score')[j]}: " + ", ".join(
                        f"{k}={v:.3g}" for k, v in r.items() if isinstance(v, float)))
            if d == 0:
                for _ in range(det_runs - 1):
                    again = _compute(layer, x)
                    if not all(torch.equal(a, b) for a, b in zip(new, again)):
                        fails.append(f"M={m}: not bitwise deterministic")
                        break
    torch.cuda.synchronize(dev)
    if fails:
        raise RuntimeError(f"[patch-014] self-test FAILED on {_name(layer)} ({len(fails)} problems; unset {ENV} to boot "
                           "without patch 014): " + " | ".join(fails[:12]))
    worst.update(ms=f"{ms[0]}..{ms[-1]}" if ms == list(range(ms[0], ms[-1] + 1)) else str(ms), draws=draws)
    return worst


def route_table() -> str:
    def fmt(route, what):
        if route is None:
            return "cuBLAS F.linear (unchanged)"
        if what == "sk":
            return "skinny bs%d opb%d ku%d vw%d" % route
        kind, arg = route
        if kind == "ll":
            return f"ll_bf16 dotprod bs{arg}"
        return "triton %s BN%d BK%d S%d w%d" % ((("fma", "dot32", "mma16")[arg[0]],) + tuple(arg[1:]))

    rows, start, ms = [], None, sorted(ROUTES)
    for i, m in enumerate(ms):
        start = m if start is None else start
        if i + 1 < len(ms) and ROUTES[ms[i + 1]] == ROUTES[m]:
            continue
        wk_r, hg_r, g_r = ROUTES[m]
        label = str(m) if start == m else f"{start}-{m}"
        rows.append(f"  M={label:<5s} wk_weights_proj: {fmt(wk_r, 'sk'):<27s} head gate: {fmt(hg_r, 'hg'):<30s} "
                    f"kpool gate: {fmt(g_r, 'sk')}")
        start = None
    return "\n".join(rows)


def _prepare(layer, device: torch.device) -> None:
    with _LOCK:
        if getattr(layer, "_p014_ready", False):
            return
        if _capturing():
            raise RuntimeError(f"[patch-014] {_name(layer)}: first call is inside CUDA-graph capture; the kernels and "
                               f"the self-test must run in an eager pass before capture (fail-closed). Unset {ENV} to "
                               "boot without patch 014.")
        _check_layer(layer)
        _ensure_compiled(device)
        t0 = time.perf_counter()
        rep = selftest(layer)
        layer._p014_ready = True
        if not _S["announced"]:
            _S["announced"] = True
            logger.info("[patch-014] skinny indexer ACTIVE (%s=1): decode-sized Indexer projections (M <= %d, bf16, "
                        "contiguous) use per-M kernels, everything else keeps the original path. Kernels compiled "
                        "in %.1f s.\n%s", ENV, MAX_M, _S["compile_s"] or 0.0, route_table())
        logger.info("[patch-014] self-test OK %s (%.2f s): M=%s x %d draws, deterministic, layout = today's; head gate "
                    "err/exact %.2e (today %.2e); k/gate max %.2f bf16 ulp vs today, <= %.2f%% elements differ",
                    _name(layer), time.perf_counter() - t0, rep["ms"], rep["draws"], rep["hg_err"],
                    rep["hg_err_today"], rep["bf16_ulp_vs_today"], 100.0 * rep["bf16_frac_differ"])


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _S["warned"]:
        _S["warned"].add(key)
        logger.warning(msg, *args)


# ----------------------------------------------------------------------------------------------------------------------
# entry points used by the patched Indexer.forward
# ----------------------------------------------------------------------------------------------------------------------
def use(layer, x: torch.Tensor) -> bool:
    """True if this call takes the per-M path. The first call of each Indexer outside capture compiles the kernels
    (once per process) and self-tests the layer; a first call inside capture raises."""
    if torch.compiler.is_compiling():  # never traced by dynamo in this build (CompilationMode.NONE); stay out if it is
        return False
    if not getattr(layer, "_p014_ready", False):
        _prepare(layer, layer.wk_weights_proj.weight.device)
    if x.dim() != 2 or not 1 <= x.shape[0] <= MAX_M:
        return False
    if (x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[1] != HIDDEN or not x.is_contiguous()
            or x.data_ptr() % 16):
        _warn_once(f"{x.dtype}/{x.is_contiguous()}/{x.shape[1]}/{x.data_ptr() % 16}",
                   "[patch-014] original path for M=%d: dtype=%s contiguous=%s hidden=%d ptr%%16=%d "
                   "(the per-M kernels need bf16, contiguous, 16-byte aligned [M,%d])",
                   x.shape[0], x.dtype, x.is_contiguous(), x.shape[1], x.data_ptr() % 16, HIDDEN)
        return False
    return True


def project(layer, x: torch.Tensor):
    """(k, weights, gate_score) with today's shapes, dtypes and strides. Call only after use() returned True."""
    if not (_S["compiled"] and getattr(layer, "_p014_ready", False)):
        raise RuntimeError(f"[patch-014] project() on {_name(layer)} before its warm-up/self-test (fail-closed)")
    return _compute(layer, x)
