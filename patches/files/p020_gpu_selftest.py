#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# [patch-020] Triton kernel tests, REVISION 4 (vLLM v0.32.0 port). Four modes; NONE has been run on the v0.32.0
# image yet (revision 3's --interp / --kernel / --fa2 were run on 2026-10-01; the kernel is byte-identical):
#
#   # 1. CPU, no GPU (a Linux box with torch + triton; TRITON_INTERPRET=1 — an INTERPRETER run, not a GPU run):
#   TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 p020_gpu_selftest.py --interp
#
#   # 2. GPU kernel tests (WRITTEN, NOT RUN: cluster rule). One idle GB10, inside a rank container of the
#   #    v0.32.0 image (its Triton + fp8), no model/engine. Logs max|q| of every case:
#   docker exec <rank container> python3 /opt/recipe-patches/files/p020_gpu_selftest.py --kernel --out /tmp/p020-kernel.json
#
#   # 3. Triton vs FlashInfer FA2 (WRITTEN, NOT RUN), same place: builds the image's own _SM90State (v0.32.0:
#   #    builder-owned, ctor with index_topk / index_kpool), plans ALL P rows with the exact per-row lengths (the
#   #    valid counts) through plan(n, lens, None, None), packs with pack_indices, runs FA2 over those rows, and
#   #    compares Triton and FA2 against an fp64 reference. Standalone — never wired into forward_split:
#   docker exec <rank container> python3 /opt/recipe-patches/files/p020_gpu_selftest.py --fa2 --out /tmp/p020-fa2.json
#
#   # 4. (NEW) --split (WRITTEN, NOT RUN), same place: the production hooks end to end on the REAL _SM90State and
#   #    FlashInfer FA2 + the Triton kernel: plan_step (decode-prefix plan, none when nd == 0) then forward_split
#   #    for pure-decode / mixed / pure-prefill row splits; decode rows vs fp64 (the FA2 path), prefill rows vs
#   #    fp64 (Triton), and the pure-decode output BITWISE equal to the stock call sequence (plan all rows +
#   #    pack_indices + wrapper.run):
#   docker exec <rank container> python3 /opt/recipe-patches/files/p020_gpu_selftest.py --split --out /tmp/p020-split.json
#
# Checks (tolerances from the kernel's design analysis; relative errors use |ref| clamped at 1e-3):
#   * vs an fp64 reference of the DEQUANTISED attention (K = fp8*k_scale, V = fp8*v_scale): mean rel
#     < 4 bf16 ulp (the interpreter measures ~1.3-1.5e-3); k_scale in {0.5, 2}: ||out||/||ref|| = 1;
#   * |q| beyond fp16 (7e4, 2**20, 2**40, 2**100 lanes, and a whole head at ~2**26): finite, same bound;
#   * head tiles: H in {1, 8, 20, 32}; strided q / out views == contiguous, bit for bit;
#   * valid > W is clamped to W; valid == 0 -> 0; repeated launches bit-identical;
#   * S=1 vs S=2 on the same rows: EACH within the fp64 bound; they differ (~1e-4 per row), so a row's
#     result depends on the prefill-row count P (S = 2 iff P <= 48; never in the served config);
#   * the start-up self-test (kernel_self_test) the engine runs at boot;
#   * --fa2: Triton mean rel vs fp64 <= 1.25 x FA2's + 1e-4, and Triton vs FA2 mean rel <= 3 bf16 ulp;
#   * --split: see above.
import argparse
import json
import os
import sys
import types

BF16_ULP = 2.0**-8


def _load():
    import importlib.util
    from pathlib import Path

    # In a rank container `vllm.triton_utils` is the real module; standalone (CPU / --interp on a box
    # without vllm installed) shim it to bare triton so the kernel still imports.
    try:
        import vllm.triton_utils  # noqa: F401
    except Exception:
        import triton
        from triton import language as tl

        vp = sys.modules.get("vllm") or types.ModuleType("vllm")
        vp.__path__ = getattr(vp, "__path__", [])
        tu = types.ModuleType("vllm.triton_utils")
        tu.tl, tu.triton = tl, triton
        sys.modules["vllm"] = vp
        sys.modules["vllm.triton_utils"] = tu
    p = Path(__file__).resolve().parent / "gb10_sparse_mla_prefill.py"
    spec = importlib.util.spec_from_file_location("gb10_sparse_mla_prefill", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_inputs(P, H=8, D=512, num_slots=4096, W=2176, vmax=2051, seed=0, device="cuda"):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(P, H, D, generator=g) * (2.0**-4)).to(torch.bfloat16)
    latent = (torch.randn(num_slots, D, generator=g) * 0.1).to(torch.float8_e4m3fn)
    valid = torch.randint(1, min(vmax, W) + 1, (P,), generator=g, dtype=torch.int32)
    slots = torch.full((P, W), -1, dtype=torch.int32)
    for r in range(P):
        v = int(valid[r])
        slots[r, :v] = torch.randperm(num_slots, generator=g)[:v].to(torch.int32)
    return q.to(device), latent.to(device), slots.to(device), valid.to(device)


def _rel(out, ref):
    import torch

    r = (out.to(torch.float64).cpu() - ref.cpu()).abs() / ref.cpu().abs().clamp_min(1e-3)
    return float(r.mean()), float(r.flatten().kthvalue(max(1, int(0.99 * r.numel()))).values)


def run_kernel_tests(device, out_path=None, quick=False):
    import torch

    m = _load()
    ref_fn = m._torch_reference
    tol = 4 * BF16_ULP
    shapes = dict(num_slots=1024, W=256, vmax=256) if quick else {}
    res = {"device": str(device), "interpreter": os.environ.get("TRITON_INTERPRET") == "1"}
    if res["interpreter"]:
        # Measured 2026-10-01: the interpreter casts fp32 -> bf16 by TRUNCATION, the GPU (cvt.rn) by
        # RTNE. The S=1 path rounds in-kernel, so its interpreter error is ~2x the GPU's (2.7e-3 vs
        # 1.4e-3); the kernel math itself is bit-exact to the pre-fix kernel (S=1 == RTZ(pre-fix fp32),
        # S=2 identical) for |q| < 2**15 and k_scale == 1.
        res["note"] = "interpreter fp32->bf16 cast truncates; S=1 errors here ~2x the GPU's RTNE"
    sm = 1.0 / 16

    def check(name, q, lat, slots, valid, ks=1.0, **extra):
        out = m.sparse_mla_prefill_triton(q, lat, slots, valid, sm, k_scale=ks)
        ref = ref_fn(q, lat, slots, valid, sm, ks, ks)
        mean_rel, p99 = _rel(out, ref)
        res[name] = {"mean_rel": mean_rel, "p99_rel": p99, "max_abs_q": float(q.float().abs().max()),
                     "num_splits": m._choose_num_splits(q.shape[0]),
                     "finite": bool(torch.isfinite(out.float()).all()), **extra}
        assert res[name]["finite"], f"{name}: non-finite output"
        assert mean_rel < tol, f"{name}: mean rel {mean_rel:.2e} >= {tol:.2e}"
        return out, ref

    # 1. accuracy vs fp64 over both split regimes + determinism
    for P in ((1, 3, 49, 64) if quick else (1, 3, 16, 48, 49, 64, 200)):
        q, lat, slots, valid = _make_inputs(P, device=device, seed=P, **shapes)
        out, _ = check(f"P{P}", q, lat, slots, valid)
        again = m.sparse_mla_prefill_triton(q, lat, slots, valid, sm)
        assert torch.equal(out, again), f"P={P} nondeterministic"
    # 2. k_scale != 1 (K AND V dequant)
    q, lat, slots, valid = _make_inputs(3, device=device, seed=11, **shapes)
    for ks in (0.5, 2.0):
        out, ref = check(f"k_scale_{ks}", q, lat, slots, valid, ks=ks)
        ratio = float(out.to(torch.float64).cpu().norm() / ref.cpu().norm())
        res[f"k_scale_{ks}"]["norm_ratio"] = ratio
        assert abs(ratio - 1.0) < 1e-2, f"k_scale={ks}: ||out||/||ref|| = {ratio:.4f}"
    # 3. |q| beyond the fp16 range (bf16 holds up to ~3.4e38): the per-head power-of-two pre-scale keeps
    #    every output finite (the FA2 bf16 path returns NaN on these inputs - GB10, 2026-10-01). Accuracy
    #    holds for heads whose lanes span < ~2**24: a LONE 2**40 lane flushes the head's ordinary lanes
    #    (~0.06) to fp16 zero, so keys that tie on that lane (fp8 has few levels) are weighted equally
    #    instead of by the small lanes (measured rel 1.8 on that one head; 2**20: 1.9e-2). Real q here is
    #    ~0.3 (P-tests' max_abs_q), so only finiteness + the other heads are asserted.
    q, lat, slots, valid = _make_inputs(4, device=device, seed=12, **shapes)
    q = q.clone()
    q[0, 0, 0], q[1, 1, 3], q[2, 2, 7], q[3, 3, 9] = 7.0e4, 2.0**20, -(2.0**40), 2.0**100
    q[0, 5] = q[0, 5] * (2.0**30)  # a whole head at ~2**26
    out = m.sparse_mla_prefill_triton(q, lat, slots, valid, sm)
    ref = ref_fn(q, lat, slots, valid, sm, 1.0, 1.0)
    assert bool(torch.isfinite(out.float()).all()), "big_q: non-finite output"
    r = (out.to(torch.float64).cpu() - ref.cpu()).abs() / ref.cpu().abs().clamp_min(1e-3)
    per = r.mean(dim=-1)  # [P, H]
    edge = torch.zeros_like(per, dtype=torch.bool)
    edge[1, 1] = edge[2, 2] = True  # the lone 2**20 and 2**40 lanes
    res["big_q"] = {"finite": True, "max_abs_q": float(q.float().abs().max()),
                    "mean_rel_other_heads": float(per[~edge].mean()), "max_rel_other_heads": float(per[~edge].max()),
                    "rel_lone_2p20_head": float(per[1, 1]), "rel_lone_2p40_head": float(per[2, 2])}
    assert float(per[~edge].max()) < tol, f"big_q: a well-scaled head is off: {res['big_q']}"
    # 4. head tiles (BLOCK_H = 16): 1 head, 2 tiles (partial / full)
    for H in (1, 20, 32):
        qh, lh, sh, vh = _make_inputs(3, H=H, device=device, seed=20 + H, **shapes)
        check(f"H{H}", qh, lh, sh, vh)
    # 5. strided q (the MLA transpose view) and a strided out view == contiguous, bit for bit (S=2, S=1)
    for P in (5, 49):
        q, lat, slots, valid = _make_inputs(P, device=device, seed=13, **shapes)
        base = m.sparse_mla_prefill_triton(q, lat, slots, valid, sm)
        q_t = q.transpose(0, 1).contiguous().transpose(0, 1)  # strides (D, P*D, 1)
        big = torch.full((P + 4, q.shape[1], q.shape[2]), 5.0, dtype=torch.bfloat16, device=device)
        m.sparse_mla_prefill_triton(q_t, lat, slots, valid, sm, out=big[3:P + 3])
        assert torch.equal(big[3:P + 3], base), f"P={P}: strided q / out view differs from contiguous"
        assert bool((big[:3] == 5.0).all() and (big[P + 3:] == 5.0).all()), "wrote outside the out view"
    res["strided_views"] = "bit-identical (S=2 and S=1)"
    # 5b. an unaligned valid_counts pointer (valid[1:] of a buffer: the mixed-step case, 4*nd bytes past an aligned
    #     base) is a different Triton specialisation (patch 020 pre-compiles it at start-up): bit-identical results
    for P in (49, 64):
        q, lat, slots, valid = _make_inputs(P, device=device, seed=15, **shapes)
        base = m.sparse_mla_prefill_triton(q, lat, slots, valid, sm)
        padded = torch.cat([torch.zeros(1, dtype=valid.dtype, device=valid.device), valid])
        assert padded[1:].data_ptr() % 16 != 0, "test setup: the view must be unaligned"
        unal = m.sparse_mla_prefill_triton(q, lat, slots, padded[1:], sm)
        assert torch.equal(base, unal), f"P={P}: unaligned valid_counts differs from the aligned launch"
    res["unaligned_valid_counts"] = "bit-identical to the aligned variant (P=49, 64)"
    # 6. clamps: valid > W behaves as W; valid == 0 -> 0
    q, lat, slots, valid = _make_inputs(2, device=device, seed=14, **shapes)
    W = slots.shape[1]
    full = torch.randperm(lat.shape[0])[:W].to(torch.int32).to(device)
    slots = slots.clone()
    slots[0] = full
    v_hi = valid.clone()
    v_hi[0], v_hi[1] = W + 50, 0
    v_eq = v_hi.clone()
    v_eq[0] = W
    o_hi = m.sparse_mla_prefill_triton(q, lat, slots, v_hi, sm)
    o_eq = m.sparse_mla_prefill_triton(q, lat, slots, v_eq, sm)
    assert torch.equal(o_hi, o_eq) and bool((o_hi[1] == 0).all()), "valid clamp / empty row"
    res["clamps"] = "valid>W == W; valid==0 -> 0"
    # 7. S=1 vs S=2 on the same rows: each within the fp64 bound; they are NOT bit-identical
    q, lat, slots, valid = _make_inputs(3, device=device, seed=99, **shapes)
    ref = ref_fn(q, lat, slots, valid, sm, 1.0, 1.0)
    orig = m._choose_num_splits
    try:
        m._choose_num_splits = lambda n: 1
        o1 = m.sparse_mla_prefill_triton(q, lat, slots, valid, sm)
        m._choose_num_splits = lambda n: 2
        o2 = m.sparse_mla_prefill_triton(q, lat, slots, valid, sm)
    finally:
        m._choose_num_splits = orig
    r1, r2 = _rel(o1, ref)[0], _rel(o2, ref)[0]
    res["S1_vs_S2"] = {"S1_mean_rel": r1, "S2_mean_rel": r2,
                       "mutual_max_abs": float((o1.float() - o2.float()).abs().max())}
    assert r1 < tol and r2 < tol, f"split-vs-fp64 rel {r1:.2e}/{r2:.2e} too large"
    # 8. the start-up self-test the engine runs at boot (both split variants, huge-q lanes, k_scale 0.5)
    res["startup_self_test"] = m.kernel_self_test(device, 8, sm)

    where = ("Triton CPU INTERPRETER - not a GPU run" if res["interpreter"] else f"GPU {device}")
    print(f"[020] kernel tests PASS ({where}):", json.dumps(res, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    return res


def compare_triton_vs_fa2(m, state, q, lat8, slots, valid, sm_scale, k_scale):
    """Plan the FA2 wrapper for ALL P rows with the exact per-row lengths (= valid_counts, host copy; the
    exact-plan contract of v0.32.0's _SM90State.plan(n, lens, None, None)), THEN pack and run FA2 over those P
    rows and run Triton on the same rows; compare both against fp64. Returns the metrics dict."""
    import torch

    P, H, D = q.shape
    state.plan(P, valid.to("cpu", torch.int32), None, None)  # plan ALL rows before FA2 reads them
    impl = types.SimpleNamespace(use_fp8_kv_cache=True, head_size=D, kv_lora_rank=D)
    layer = types.SimpleNamespace(_k_scale_float=k_scale)
    q_pe = q.new_empty((P, H, 0))
    fa2 = m._fa2_run(impl, state, q, q_pe, slots, lat8.view(torch.uint8), layer, P)  # pack_indices handles the -1 tail
    tri = m.sparse_mla_prefill_triton(q, lat8, slots, valid, sm_scale, k_scale=k_scale)
    ref = m._torch_reference(q, lat8, slots, valid, sm_scale, k_scale, k_scale)
    t_rel, t_p99 = _rel(tri, ref)
    f_rel, f_p99 = _rel(fa2, ref)
    d = (tri.float().cpu() - fa2.float().cpu()).abs() / fa2.float().cpu().abs().clamp_min(1e-3)
    return {"triton_mean_rel": t_rel, "triton_p99_rel": t_p99, "fa2_mean_rel": f_rel, "fa2_p99_rel": f_p99,
            "tri_vs_fa2_mean_rel": float(d.mean()), "tri_vs_fa2_max_rel": float(d.max()),
            "max_abs_q": float(q.float().abs().max())}


def _assert_fa2_tolerances(name, r):
    assert r["triton_mean_rel"] <= 1.25 * r["fa2_mean_rel"] + 1e-4, f"{name}: Triton less accurate than FA2: {r}"
    assert r["tri_vs_fa2_mean_rel"] <= 3 * BF16_ULP, f"{name}: Triton vs FA2 > 3 bf16 ulp: {r}"


def _new_state(sm90, dev, H, P, W, sm_scale):
    """The v0.32.0 builder-owned _SM90State (one wrapper + reserved buffers); exact plans need no cam."""
    import torch

    return sm90._SM90State(dev, H, torch.float8_e4m3fn, max_tokens=P, topk_width=W, kv_lora_rank=512,
                           qk_rope_head_dim=0, sm_scale=sm_scale, index_topk=2048, index_kpool=4)


def run_fa2_compare(out_path=None):
    """GPU only (image container): Triton vs the image's FlashInfer FA2 wrapper."""
    import torch

    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm90 as sm90

    m = _load()
    dev = torch.device("cuda")
    sm = 1.0 / 16
    res = {}
    for P, ks in ((129, 1.0), (512, 1.0), (512, 0.5), (3840, 1.0)):
        q, lat, slots, valid = _make_inputs(P, device=dev, seed=1000 + P)
        state = _new_state(sm90, dev, q.shape[1], P, slots.shape[1], sm)
        r = compare_triton_vs_fa2(m, state, q, lat, slots, valid, sm, ks)
        res[f"P{P}_k{ks}"] = r
        _assert_fa2_tolerances(f"P{P}", r)
    print("[020] Triton vs FA2 PASS (GPU):", json.dumps(res, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    return res


class _ShimBuilder:
    """What plan_step needs from the builder: .state and _kv_lens_host(cam) -> (rows, lens, exact). Exact
    lengths (the valid counts), so the plan is the exact path (state.plan(n, lens, None, None))."""

    def __init__(self, state, valid_cpu):
        self.state, self._valid = state, valid_cpu

    def _kv_lens_host(self, cam):
        return int(self._valid.shape[0]), self._valid.clone(), True


def run_split(out_path=None):
    """GPU only: plan_step + forward_split on the real _SM90State / FlashInfer FA2 / Triton, for pure-decode,
    mixed and pure-prefill splits (the production hooks end to end, minus the vLLM metadata plumbing)."""
    import torch

    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm90 as sm90

    m = _load()
    dev = torch.device("cuda")
    sm, W = 1.0 / 16, 2176
    tol = 4 * BF16_ULP
    res = {}
    for T, nd, ks in ((260, 0, 1.0), (260, 4, 1.0), (136, 8, 0.5), (131, 128, 1.0), (8, 8, 1.0), (4, 4, 0.5)):
        q, lat, slots, valid = _make_inputs(T, device=dev, seed=2000 + T + nd, W=W)
        H, D = q.shape[1], q.shape[2]
        state = _new_state(sm90, dev, H, T, W, sm)
        valid_cpu = valid.to("cpu", torch.int32)
        md = types.SimpleNamespace(num_decode_tokens=nd, req_id_per_token=None)
        m.plan_step(_ShimBuilder(state, valid_cpu), None, md)
        impl = types.SimpleNamespace(use_fp8_kv_cache=True, head_size=D, kv_lora_rank=D, num_heads=H, scale=sm,
                                     qk_rope_head_dim=0)
        layer = types.SimpleNamespace(_k_scale_float=ks)
        q_pe = q.new_empty((T, H, 0))
        rows = lat.view(torch.uint8)
        out = m.forward_split(impl, state, q, q_pe, slots.clone(), valid, rows, layer, md)
        torch.cuda.synchronize()
        ref = m._torch_reference(q, lat, slots, valid, sm, ks, ks)
        r = {"T": T, "nd": nd, "k_scale": ks, "finite": bool(torch.isfinite(out.float()).all())}
        if nd > 0:
            r["decode_mean_rel"], r["decode_p99_rel"] = _rel(out[:nd], ref[:nd])
            assert r["decode_mean_rel"] < tol, f"T{T}/nd{nd}: decode rows (FA2) {r['decode_mean_rel']:.2e} >= {tol:.2e}"
        if nd < T:
            r["prefill_mean_rel"], r["prefill_p99_rel"] = _rel(out[nd:], ref[nd:])
            assert r["prefill_mean_rel"] < tol, f"T{T}/nd{nd}: prefill rows (Triton) {r['prefill_mean_rel']:.2e} >= {tol:.2e}"
        assert r["finite"], f"T{T}/nd{nd}: non-finite output"
        if nd == T:
            # pure decode == the stock call sequence: plan ALL rows (exact), pack, run — BITWISE
            stock_state = _new_state(sm90, dev, H, T, W, sm)
            stock_state.plan(T, valid_cpu, None, None)
            stock_state.pack_indices(slots)
            flat = rows.view(torch.float8_e4m3fn).reshape(-1, 1, D)
            want = stock_state.wrapper.run(q, q_pe, flat[..., :D], flat[..., D:], ckv_scale=ks, kpe_scale=1.0)
            r["pure_decode_bitwise_vs_stock"] = bool(torch.equal(out, want))
            assert r["pure_decode_bitwise_vs_stock"], f"T{T}: pure-decode output differs from the stock call sequence"
        res[f"T{T}_nd{nd}"] = r
    # a mixed batch the builder hook did not plan for must be refused
    state = _new_state(sm90, dev, 8, 16, W, sm)
    q, lat, slots, valid = _make_inputs(16, device=dev, seed=77, W=W)
    md = types.SimpleNamespace(num_decode_tokens=4, req_id_per_token=None)
    impl = types.SimpleNamespace(use_fp8_kv_cache=True, head_size=512, kv_lora_rank=512, num_heads=8, scale=sm,
                                 qk_rope_head_dim=0)
    try:
        m.forward_split(impl, state, q, q.new_empty((16, 8, 0)), slots, valid, lat.view(torch.uint8),
                        types.SimpleNamespace(_k_scale_float=1.0), md)
        raise AssertionError("an unplanned mixed batch must be refused")
    except m.Patch020Error:
        res["unplanned_refused"] = True
    print("[020] split (plan_step + forward_split) PASS (GPU):", json.dumps(res, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    return res


class _FakeFA2State:
    """CPU stand-in for _SM90State: plan() records rows/lengths (v0.32.0 signature); pack_indices() lays the
    valid prefixes out row-major; wrapper.run() REFUSES any row count other than the planned one and computes
    exact attention over each row's PLANNED length (as FA2 bakes kv_len into its schedule at plan time)."""

    def __init__(self, P_max, W, sm_scale):
        import torch

        self.kv_indices = torch.zeros(P_max * W, dtype=torch.int32)
        self.W, self.sm_scale, self.planned = W, sm_scale, None
        self.wrapper = types.SimpleNamespace(run=self._run)

    def plan(self, n, lens, cam=None, req_id_per_token=None):
        self.planned = (int(n), lens.clone())

    def pack_indices(self, topk_slots):
        n, w = topk_slots.shape
        self.kv_indices[: n * w].copy_(topk_slots.reshape(-1).clamp(min=0).to(self.kv_indices.dtype))

    def _run(self, q_nope, q_pe, ckv, kpe, ckv_scale=1.0, kpe_scale=1.0):
        import torch

        assert self.planned is not None, "FA2 run without a plan"
        n, lens = self.planned
        assert q_nope.shape[0] == n, f"FA2 run over {q_nope.shape[0]} rows but {n} planned"
        out = torch.zeros(q_nope.shape, dtype=torch.float64)
        for r in range(n):
            idx = self.kv_indices[r * self.W: r * self.W + int(lens[r])].long()
            kv = ckv[idx, 0, :].to(torch.float64) * ckv_scale
            s = (q_nope[r].to(torch.float64) @ kv.t()) * self.sm_scale
            out[r] = torch.softmax(s, dim=-1) @ kv
        return out.to(torch.bfloat16)


def run_fa2_orchestration_cpu():
    """CPU: compare_triton_vs_fa2 plans every row with the exact lengths before FA2 reads them."""
    m = _load()
    P, sm, ks = 6, 1.0 / 16, 0.5
    q, lat, slots, valid = _make_inputs(P, device="cpu", seed=7, num_slots=512, W=128, vmax=128)
    st = _FakeFA2State(P, slots.shape[1], sm)
    st.plan(2, valid[:2].clone(), None, None)  # a stale decode-prefix plan, as a live TRITON step leaves it
    r = compare_triton_vs_fa2(m, st, q, lat, slots, valid, sm, ks)
    assert st.planned[0] == P and bool((st.planned[1] == valid).all()), "must replan all rows"
    assert bool((slots >= -1).all()) and int((slots == -1).sum()) > 0, "caller's slots were clamped"
    # the fake FA2 is exact fp64 + bf16 rounding (more accurate than FA2's bf16 P), so only the
    # Triton-vs-FA2 and vs-fp64 bounds apply here, not "Triton <= 1.25 x FA2 error"
    assert r["tri_vs_fa2_mean_rel"] <= 3 * BF16_ULP and r["triton_mean_rel"] < 4 * BF16_ULP, r
    print("[020] FA2-compare orchestration (CPU fake wrapper): replans all rows before FA2 reads them,",
          "Triton vs fake-FA2 within 3 bf16 ulp:", json.dumps(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="store_true", help="kernel tests on the GPU (container/GB10)")
    ap.add_argument("--fa2", action="store_true", help="Triton vs FA2 on the GPU (container/GB10)")
    ap.add_argument("--split", action="store_true", help="plan_step + forward_split end to end on the GPU (container/GB10)")
    ap.add_argument("--interp", action="store_true", help="kernel tests under TRITON_INTERPRET=1 (CPU)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.interp:
        if os.environ.get("TRITON_INTERPRET") != "1" or "triton" in sys.modules:
            os.environ["TRITON_INTERPRET"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.execv(sys.executable, [sys.executable] + sys.argv)  # interpreter must be set pre-import
        run_kernel_tests(device="cpu", out_path=args.out, quick=True)
        run_fa2_orchestration_cpu()
    elif args.kernel:
        run_kernel_tests(device="cuda", out_path=args.out)
    elif args.fa2:
        run_fa2_compare(out_path=args.out)
    elif args.split:
        run_split(out_path=args.out)
    else:
        print("nothing to do: pass --interp (CPU), --kernel, --fa2 or --split (GPU)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
