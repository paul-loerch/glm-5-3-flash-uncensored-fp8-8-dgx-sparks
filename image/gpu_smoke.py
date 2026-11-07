#!/usr/bin/env python3
"""gpu_smoke.py -- optional GPU check of the SM121 bake.  Run INSIDE the baked image on ONE GB10 node:

    docker run --rm --gpus all --entrypoint python3 <image> -I /opt/glm53-sm121/gpu_smoke.py

(add -v <dir>:/root/.cache/flashinfer to keep the JIT build).  The first run JIT-compiles the patched FlashInfer FA2
FP8 NoPE batch-MLA module with nvcc for sm_121a (several minutes); later runs reuse it.  It does not need the model.

STATUS: written on a machine without a GPU, torch or FlashInfer -- it has NOT been executed.  Its reference maths and
expected arrays were derived by reading the v0.32.0 / FlashInfer 0.7.0.post1 sources.  If it fails with an exception in the
plumbing (argument shapes, dtypes) rather than a numeric mismatch, fix the script, not the bake; a numeric mismatch or the
named FlashInfer / CUDA errors below are real.

Checks (each prints PASS / FAIL; exit status 1 if any FAIL):
  device    compute capability 12.x, the device name vLLM builds MoE file names from
  pdl       vLLM's platform says PDL is off
  backend   the SM90 NoPE sparse-MLA backend validates on this device for fp8_e4m3 KV
  kpool     the Triton pool-expand kernel drops pool ids >= pool_len (pid 8, 9 of 8 complete pools)
  moe       vLLM finds the tuned E=288,N=256 table for this device
  fa2_fp8   FlashInfer FA2 batch MLA, FP8 E4M3 KV, kpe=0, vs an fp32 PyTorch reference.  Failure modes it exposes:
            "FP8 kv_data_type for MLA requires an SM90 (Hopper) device"  -> fi_fp8_gate missing
            "cudaFuncSetAttribute ... invalid argument" / "too much shared data" -> fi_tile_cap missing
            module not named *_fp8tilecap in the JIT cache                   -> fi_uri missing
"""
from __future__ import annotations

import sys
import traceback

FAILS = 0


def report(ok: bool, name: str, detail: str = "") -> None:
    global FAILS
    FAILS += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)


def run(name, fn):
    try:
        detail = fn()
        report(True, name, detail or "")
    except AssertionError as exc:
        report(False, name, str(exc))
    except Exception as exc:  # noqa: BLE001
        report(False, name, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def check_device():
    import torch

    cap = torch.cuda.get_device_capability(0)
    prop = torch.cuda.get_device_properties(0)
    from vllm.utils.platform_utils import get_device_name_as_file_name

    name = get_device_name_as_file_name()
    smem = getattr(prop, "shared_memory_per_multiprocessor", None)
    assert cap[0] == 12, f"compute capability {cap}: this bake is for SM12x"
    return f"{prop.name} sm_{cap[0]}{cap[1]}, smem/SM={smem}, vLLM file-name device = {name!r}"


def check_pdl():
    from vllm.platforms import current_platform

    assert current_platform.is_arch_support_pdl() is False, "PDL still reported as supported on this device"
    return "current_platform.is_arch_support_pdl() is False"


def check_backend():
    import torch
    from vllm.platforms import current_platform
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import FlashInferMLASparseSM90Backend as B

    cap = current_platform.get_device_capability()
    reasons = B.validate_configuration(
        head_size=512, dtype=torch.bfloat16, kv_cache_dtype="fp8_e4m3", block_size=256, use_mla=True, has_sink=False,
        use_sparse=True, use_mm_prefix=False, use_per_head_quant_scales=False, device_capability=cap,
        attn_type=AttentionType.DECODER)
    assert reasons == [], f"SM90 backend refuses this device/config: {reasons}"
    return f"FLASHINFER_MLA_SPARSE_SM90 valid on {tuple(cap)} (fp8_e4m3, block 256)"


def check_kpool():
    import torch
    from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail

    # pool size 4, seq_len 33 -> pool_len 8, tail = 1 token (index 32); history ids 8 and 9 are >= pool_len
    pool_ids = torch.tensor([[0, 1, 5, 7, 8, 9]], dtype=torch.int64, device="cuda")
    seq_len = torch.tensor([33], dtype=torch.int32, device="cuda")
    got = expand_pools_and_append_tail(pool_ids, seq_len, 4).cpu().tolist()[0]
    want = [0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 28, 29, 30, 31, -1, -1, -1, -1, -1, -1, -1, -1, 32, -1, -1]
    assert got == want, f"got {got}\n     want {want}\n     (stock kernel would give 32..39 for ids 8, 9: tokens past the sequence)"
    return "pool ids 8 and 9 (>= pool_len 8) expand to -1; tail token 32 appended once"


def check_moe():
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name, get_moe_configs

    get_moe_configs.cache_clear()
    cfg = get_moe_configs(288, 256, "fp8_w8a8", 128, 128)
    assert cfg, f"no tuned table found for {get_config_file_name(288, 256, 'fp8_w8a8', [128, 128])}"
    return f"{get_config_file_name(288, 256, 'fp8_w8a8', [128, 128])}: {len(cfg)} batch sizes"


def check_fa2_fp8():
    import torch
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    torch.manual_seed(0)
    dev = torch.device("cuda")
    H, CKV, KPE, N = 8, 512, 0, 4096          # 8 heads per rank at TP=8; NoPE; 4096 KV slots
    lens = [1, 17, 200, 2176, 2176]           # valid top-k entries per query row (2176 = 2048 + 128 table width)
    T = len(lens)
    sm_scale = 256 ** -0.5                    # GLM-5.3-Flash qk head dim 256
    ckv_bf16 = torch.randn(N, CKV, dtype=torch.bfloat16, device=dev)
    scale = float(ckv_bf16.abs().max()) / 448.0
    ckv_fp8 = (ckv_bf16.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    ckv_ref = ckv_fp8.float() * scale         # what the kernel dequantises to (exact in fp32)
    q_nope = (0.5 * torch.randn(T, H, CKV, device=dev)).to(torch.bfloat16)
    q_pe = torch.empty(T, H, KPE, dtype=torch.bfloat16, device=dev)
    idx = [torch.randperm(N, device=dev)[:n].to(torch.int32) for n in lens]
    kv_indices = torch.cat(idx).cpu()
    kv_indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    qo_indptr = torch.arange(T + 1, dtype=torch.int32)
    kv_len_arr = torch.tensor(lens, dtype=torch.int32)

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev), backend="fa2")
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, H, CKV, KPE, 1, False, sm_scale,
                 torch.bfloat16, torch.float8_e4m3fn)
    out = wrapper.run(q_nope, q_pe, ckv_fp8.view(N, 1, CKV), torch.empty(N, 1, KPE, dtype=torch.float8_e4m3fn, device=dev),
                      ckv_scale=scale, kpe_scale=1.0)
    torch.cuda.synchronize()

    ref = torch.empty(T, H, CKV, dtype=torch.float32, device=dev)
    for t in range(T):
        k = ckv_ref[idx[t].long()]                              # [L, CKV]
        p = torch.softmax((q_nope[t].float() @ k.T) * sm_scale, dim=-1)
        ref[t] = p @ k
    err = (out.float() - ref).norm() / ref.norm()
    assert bool(torch.isfinite(out.float()).all()), "non-finite output"
    assert float(err) < 2e-2, f"relative L2 error {float(err):.3e} vs fp32 reference (expect ~3e-3 for bf16 output)"
    return f"FA2 FP8-KV NoPE MLA, rows {lens}: relative L2 error {float(err):.2e}"


def main() -> int:
    for name, fn in [("device", check_device), ("pdl", check_pdl), ("backend", check_backend), ("kpool", check_kpool),
                     ("moe", check_moe), ("fa2_fp8", check_fa2_fp8)]:
        run(name, fn)
    print("GPU SMOKE " + ("FAILED" if FAILS else "OK"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
