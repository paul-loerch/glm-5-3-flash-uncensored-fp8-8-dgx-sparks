#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-020 gb10-sparse-mla-triton-prefill] OPTIONAL differential check with REAL torch (CPU wheel; no GPU, no
triton, no vLLM install). The mandatory suite (p020_cpu_check.py) runs on python3 + numpy over a numpy-backed fake
torch; this script re-runs the pieces whose semantics depend on torch itself on a real CPU torch, so the fake is
validated where it matters:

  R1  (and R1b for the unaligned-valid_counts pre-compile) the model loader's context (`with torch.device("meta")` + a bf16 default dtype) around kernel_self_test: every
      input must be built explicitly on the CPU in fp32 (revision 2 of this patch would have failed every GB10 boot
      on exactly this) — real torch raises or builds meta tensors if one creation forgets its device/dtype;
  R2  _merge_splits and _torch_reference against numpy;
  R3  forward_split on real tensors: the strided q view (no copy), out[nd:] in place, the [num_rows, 512] E4M3 row
      view of a block-strided cache, valid_counts / slot slices, the FA2 call arguments;
  R4  the REAL upstream FlashInferMLASparseSM90Builder._kv_lens_host / _row_contexts of the v0.32.0 file executed on
      real torch tensors and on the fake torch, same inputs, identical outputs (6,000 V2 batches);
  R5  the gate-off patched build() == stock build() on real torch (plan call log).

  python3 -m venv /tmp/tv && /tmp/tv/bin/pip install torch==2.13.0 numpy
  /tmp/tv/bin/python files/p020_realtorch_check.py --vllm-root <v0.32.0 tree>/vllm
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch as real_torch  # noqa: E402  (real torch FIRST; the testlib then installs its stubs)

import p020_cpu_check as C  # noqa: E402
import p020_testlib as T  # noqa: E402

RES = {"pass": 0, "fail": 0}


def check(name, ok, info=""):
    RES["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info and not ok else ""), flush=True)


def use_real_torch():
    sys.modules["torch"] = real_torch
    sys.modules["torch.cuda"] = real_torch.cuda


def load_module_real():
    """The vendored module with REAL torch and a fake triton."""
    env = T.install_stubs()  # vllm.* stubs + fake triton (and the fake torch)
    use_real_torch()
    mod = T.load_file(env, "p020_mod_real", C.MODULE_PATH, install_as="vllm._gb10_sparse_mla_prefill")
    return env, mod


def r1_loader_context():
    env, mod = load_module_real()
    torch = real_torch
    seen = []

    def fake_kernel(q, latent, slots, valid, sm_scale, k_scale=1.0, v_scale=None, out=None):
        seen.append((q.device.type, q.dtype, latent.device.type, latent.dtype, slots.device.type, valid.device.type))
        return mod._torch_reference(q, latent, slots, valid, sm_scale, k_scale, k_scale).to(torch.bfloat16)

    mod.sparse_mla_prefill_triton = fake_kernel
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            rep = mod.kernel_self_test("cpu", 8, 1.0 / 16)
    finally:
        torch.set_default_dtype(old)
    ok = len(seen) == 2 and all(s == ("cpu", torch.bfloat16, "cpu", torch.float8_e4m3fn, "cpu", "cpu") for s in seen) \
        and all(v["finite"] for v in rep.values())
    check("R1 kernel_self_test under a REAL `torch.device('meta')` + bf16 default dtype: inputs on the CPU, fp8 latent, "
          "2 cases finite", ok, str(seen))


def r1b_warm_unaligned_real():
    env, mod = load_module_real()
    torch = real_torch
    seen = []

    def fake_kernel(q, latent, slots, valid, sm_scale, k_scale=1.0, v_scale=None, out=None):
        seen.append((q.data_ptr() % 16, slots.data_ptr() % 16, valid.data_ptr() % 16, q.device.type, valid.device.type,
                     q.dtype, latent.dtype, slots.dtype, valid.dtype, tuple(valid.shape), valid.is_contiguous()))
        return q

    mod.sparse_mla_prefill_triton = fake_kernel
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            mod._warm_unaligned_variant("cpu", 8, 1.0 / 16)
    finally:
        torch.set_default_dtype(old)
    ok = (len(seen) == 1 and seen[0][:3] == (0, 0, 4) and seen[0][3:5] == ("cpu", "cpu")
          and seen[0][5:9] == (torch.bfloat16, torch.float8_e4m3fn, torch.int32, torch.int32)
          and seen[0][9] == (64,) and seen[0][10])
    check("R1b _warm_unaligned_variant under a REAL meta device + bf16 default dtype: one launch, valid_counts pointer % 16 == 4 "
          "(real torch slice), q / slots aligned", ok, str(seen))


def r2_merge_reference():
    env, mod = load_module_real()
    torch = real_torch
    g = torch.Generator().manual_seed(0)
    P, H, D, N = 5, 8, 64, 130
    q = torch.randn(P, H, D, generator=g, dtype=torch.float32)
    k = torch.randn(N, D, generator=g, dtype=torch.float32)
    scores = torch.einsum("phd,nd->phn", q, k) / 16
    ref = torch.einsum("phn,nd->phd", torch.softmax(scores, -1), k)

    def part(lo, hi):
        s = scores[..., lo:hi]
        m = s.max(-1).values
        e = torch.exp(s - m.unsqueeze(-1))
        l = e.sum(-1)
        o = torch.einsum("phn,nd->phd", e, k[lo:hi]) / l.clamp_min(1e-30).unsqueeze(-1)
        return o, torch.where(l > 0, m + torch.log(l), torch.full_like(m, -3.4e38))

    a0, l0 = part(0, 64)
    a1, l1 = part(64, N)
    merged = mod._merge_splits(torch.stack([a0, a1]), torch.stack([l0, l1]))
    err = float((merged - ref).abs().max())
    z = mod._merge_splits(torch.zeros(2, 1, H, D), torch.full((2, 1, H), -3.4e38))
    check(f"R2 _merge_splits vs direct softmax on real torch: max abs err {err:.1e}", err < 1e-4 and bool((z == 0).all()))
    # _torch_reference vs numpy
    num_slots, W = 40, 16
    lat = (torch.randn(num_slots, 512, generator=g) * 0.1).to(torch.float8_e4m3fn)
    qq = torch.randn(3, 8, 512, generator=g).to(torch.bfloat16)
    slots = torch.randint(0, num_slots, (3, W), generator=g, dtype=torch.int32)
    valid = torch.tensor([16, 5, 0], dtype=torch.int32)
    out = mod._torch_reference(qq, lat, slots, valid, 1 / 16, 0.5, 0.5)
    l64 = lat.to(torch.float64).numpy()
    q64 = qq.to(torch.float64).numpy()
    want = np.zeros((3, 8, 512))
    for r in range(3):
        v = int(valid[r])
        if v == 0:
            continue
        kv = l64[slots[r, :v].long().numpy()]
        s = (q64[r] @ kv.T) * (1 / 16 * 0.5)
        p = np.exp(s - s.max(-1, keepdims=True))
        want[r] = (p / p.sum(-1, keepdims=True)) @ kv * 0.5
    check("R2b _torch_reference vs numpy (valid=0 row -> 0)", float(np.abs(out.numpy() - want).max()) < 1e-9)


def r3_forward_split_real_tensors():
    env, mod = load_module_real()
    torch = real_torch
    H, D, Tn, nd = 8, 512, 9, 3
    seen = {}

    def fa2(impl, state, q_nope, q_pe, topk_slots, kv_cache_rows, layer, n):
        seen["fa2"] = (n, q_nope.data_ptr(), topk_slots.data_ptr(), kv_cache_rows.shape, kv_cache_rows.dtype)
        return torch.full((n, H, D), 7.0, dtype=torch.bfloat16)

    def tri(q, latent, slots, vc, sm_scale, k_scale=1.0, v_scale=None, out=None):
        seen["tri"] = dict(q_ptr=q.data_ptr(), q_stride=q.stride(), lat_shape=tuple(latent.shape), lat_stride=latent.stride(),
                           lat_dtype=latent.dtype, lat_ptr=latent.data_ptr(), slots_shape=tuple(slots.shape),
                           vc_shape=tuple(vc.shape), out_ptr=out.data_ptr(), ks=k_scale, vs=v_scale)
        out.fill_(3.0)
        return out

    mod._fa2_run, mod.sparse_mla_prefill_triton = fa2, tri
    base = torch.randn(H, Tn, D).to(torch.bfloat16)
    qn = base.transpose(0, 1)  # the MLA bmm(out=...transpose) view: strides (D, Tn*D, 1)
    qp = torch.zeros(Tn, H, 0, dtype=torch.bfloat16)
    slots = torch.randint(0, 100, (Tn, 64), dtype=torch.int32)
    vc = torch.full((Tn,), 7, dtype=torch.int32)
    # a block-strided fp8 cache and its flat row view exactly as upstream's flat_kv_row_view builds it
    nb, bs, stride_rows = 6, 64, 80
    storage = torch.zeros(nb * stride_rows * D, dtype=torch.uint8)
    cache = storage.view(torch.float8_e4m3fn).as_strided((nb, bs, D), (stride_rows * D, D, 1))
    rows = cache.as_strided(((nb - 1) * stride_rows + bs, D), (D, 1))
    state = types.SimpleNamespace(_p020_plan_rows=nd)
    md = types.SimpleNamespace(num_decode_tokens=nd)
    impl = types.SimpleNamespace(num_heads=H, kv_lora_rank=D, head_size=D, qk_rope_head_dim=0, scale=1 / 16, use_fp8_kv_cache=True)
    out = mod.forward_split(impl, state, qn, qp, slots, vc, rows, types.SimpleNamespace(_k_scale_float=0.25), md)
    t = seen["tri"]
    ok = (out.shape == (Tn, H, D) and bool((out[:nd] == 7).all()) and bool((out[nd:] == 3).all())
          and t["q_ptr"] == qn[nd:].data_ptr() and t["q_stride"] == qn.stride() and t["out_ptr"] == out[nd:].data_ptr()
          and t["lat_shape"] == ((nb - 1) * stride_rows + bs, D) and t["lat_stride"] == (D, 1)
          and t["lat_dtype"] == torch.float8_e4m3fn and t["lat_ptr"] == cache.data_ptr()
          and t["slots_shape"] == (Tn - nd, 64) and t["vc_shape"] == (Tn - nd,) and t["ks"] == t["vs"] == 0.25
          and seen["fa2"][0] == nd)
    check("R3 forward_split on real tensors: strided q view (no copy), out[nd:] in place, E4M3 row view of a block-strided "
          "cache (stride 80 rows/block), slices", ok, str(t))


def _batches(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        r = T.random_batch(rng)
        if r:
            out.append(r)
    return out


def r4_kv_lens_host_differential(vllm_root: Path, n: int):
    stock_text = (vllm_root / C.SM_REL).read_text()
    # fake-torch world
    env = T.install_stubs()
    fake_mod = T.load_source(env, "sm90_fake", stock_text, "stock_fake.py")
    # real-torch world
    use_real_torch()
    real_mod = T.load_source(env, "sm90_real", stock_text, "stock_real.py")
    bad = 0
    for reqs in _batches(n, 4242):
        for host in (True, False):
            for adaptive in (False, True):
                outs = []
                for world, mod in (("fake", fake_mod), ("real", real_mod)):
                    b = object.__new__(mod.FlashInferMLASparseSM90Builder)
                    b.state = T.FakeState()
                    b._adaptive_verification = adaptive
                    if world == "fake":
                        cam = T.make_cam(env, reqs, with_host_bound=host)
                        rows, lens, exact = b._kv_lens_host(cam)
                        outs.append((int(rows), [int(x) for x in lens.tolist()], exact))
                    else:
                        qsl = [0]
                        for r in reqs:
                            qsl.append(qsl[-1] + r["q"])
                        cam = types.SimpleNamespace(
                            num_reqs=len(reqs), query_start_loc_cpu=real_torch.tensor(qsl, dtype=real_torch.int32),
                            seq_lens=real_torch.tensor([r["gpu_nc"] + r["q"] for r in reqs], dtype=real_torch.int32),
                            seq_lens_cpu_upper_bound=(real_torch.tensor([r["cpu_nc"] + r["q"] for r in reqs], dtype=real_torch.int32)
                                                      if host else None))
                        rows, lens, exact = b._kv_lens_host(cam)
                        outs.append((int(rows), [int(x) for x in lens.tolist()], exact))
                bad += outs[0] != outs[1]
    check(f"R4 the REAL upstream _kv_lens_host: fake torch == real torch on {n * 4} calls ({n} V2 batches x host bound/exact x "
          "adaptive)", bad == 0, f"{bad} differ")


def r5_build_parity_real_torch(vllm_root: Path, n: int):
    stock_text = (vllm_root / C.SM_REL).read_text()
    ap = C.load_applier()
    patched_text = ap.apply(stock_text)
    env = T.install_stubs()
    use_real_torch()
    stock = T.load_source(env, "sm90_stock_r", stock_text, "stock_r.py")
    mod = T.load_file(env, "p020_mod_real5", C.MODULE_PATH, install_as="vllm._gb10_sparse_mla_prefill")
    patched = T.load_source(env, "sm90_patched_r", patched_text, "patched_r.py")

    class RState:
        index_topk, index_kpool, max_valid = 2048, 4, 2051

        def __init__(self):
            self.log = []

        def plan(self, n_, lens, cam, req):
            self.log.append(("plan", int(n_), [int(x) for x in lens.tolist()], cam is not None))

    bad = 0
    for reqs in _batches(n, 77):
        qsl = [0]
        for r in reqs:
            qsl.append(qsl[-1] + r["q"])
        rid = real_torch.repeat_interleave(real_torch.arange(len(reqs), dtype=real_torch.int32),
                                           real_torch.tensor([r["q"] for r in reqs], dtype=real_torch.int64))
        cam = types.SimpleNamespace(
            num_reqs=len(reqs), num_actual_tokens=qsl[-1], max_query_len=max(r["q"] for r in reqs),
            max_seq_len=max(r["gpu_nc"] + r["q"] for r in reqs),
            query_start_loc_cpu=real_torch.tensor(qsl, dtype=real_torch.int32), query_start_loc=None,
            seq_lens=real_torch.tensor([r["gpu_nc"] + r["q"] for r in reqs], dtype=real_torch.int32),
            seq_lens_cpu_upper_bound=real_torch.tensor([r["cpu_nc"] + r["q"] for r in reqs], dtype=real_torch.int32),
            slot_mapping=None, block_table_tensor=None, req_id_per_token=rid)
        for mha in (False, True):
            logs = []
            for m in (stock, patched):
                b = object.__new__(m.FlashInferMLASparseSM90Builder)
                b.state = RState()
                b._adaptive_verification = False
                b._attention_layer = T.FakeLayer(mha)
                C.setgate(False)
                md = b.build(0, cam)
                logs.append((b.state.log, md.state is b.state, md.num_decode_tokens))
            bad += logs[0] != logs[1]
    C.setgate(False)
    check(f"R5 gate off: patched build() == stock build() on REAL torch tensors ({n * 2} builds)", bad == 0, f"{bad} differ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True, type=Path)
    ap.add_argument("--batches", type=int, default=300)
    args = ap.parse_args()
    print("torch", real_torch.__version__)
    r1_loader_context()
    r1b_warm_unaligned_real()
    r2_merge_reference()
    r3_forward_split_real_tensors()
    r4_kv_lens_host_differential(args.vllm_root, args.batches)
    r5_build_parity_real_torch(args.vllm_root, args.batches)
    print(f"[020] real-torch differential checks: {RES['pass']} passed, {RES['fail']} failed")
    return 0 if RES["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
