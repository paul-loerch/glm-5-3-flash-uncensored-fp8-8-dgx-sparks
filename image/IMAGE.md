# The serving image

The recipe runs the **official vLLM v0.32.0 image** with a small set of build-time source edits for the GB10 GPU
(compute capability 12.1, "sm_121") in the DGX Spark. Everything else that the recipe changes happens at container
start and is described in [../PATCHES.md](../PATCHES.md).

| | |
|---|---|
| Base | `vllm/vllm-openai:v0.32.0-aarch64@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc` |
| Inside the base | vLLM commit `5df63a7203cd`, CUDA 13.0.2, FlashInfer 0.7.0.post1, NCCL 2.30.7, Python 3.12, arm64 |
| Result | `glm53-flash-sm121:v032-1` (the tag `config/recipe.env` expects) |
| Build inputs | `Dockerfile`, `glm53_sm121_v032.py` (the edits), `verify_bake.py`, `gpu_smoke.py`, `moe/` (one tuned table), `LICENSES/` |

Do not use the `-cu129` variant of the base: it is built for CUDA 12.9 and has no kernels for sm_121.

## Build it

On the head node (NODE1), from the repository root:

```bash
image/build.sh                 # docker build + the static check of every edit
image/build.sh --distribute    # ... then copy the image to NODE2..NODE8 and compare the image IDs (uses cluster.env)
image/build.sh --gpu-smoke     # optional: also run the GPU smoke test on this node
```

The build needs no network beyond pulling the base by digest, no GPU and no pip. The image is about 10 GB; copying it to
seven nodes takes a few minutes. `launch/up.sh` refuses to start unless all eight nodes report the same
image ID.

## What the edits do, and why

vLLM v0.32.0 supports GLM-5.3-Flash, but on GB10 its defaults pick kernels and settings that either fail or are slower.
The edits are applied once while the image is built. Every edit must match its anchor exactly once in a file whose
sha256 is pinned; otherwise the build fails and nothing is written. `verify_bake.py` re-checks every edit afterwards.

| Group | Edit | Problem on GB10 | Change | Numerics |
|---|---|---|---|---|
| attn | `pri12`, `sm90cap`, `sm90fa2` | For this sparse-MLA model the stock selection on capability 12 is the SM120 backend, which also rewrites the FP8 KV cache to a 656 B/token/layer format (512 B with this recipe) and has no dense prefill for short prompts | Allow the SM90 NoPE sparse-MLA backend on capability 12, running through FlashInfer's FA2 kernels (FA3 is Hopper-only) | Different kernel from stock v0.32.0; the same attention arithmetic as the recipe's previous base |
| attn | `fi_fp8_gate` | FlashInfer's plan-time validator rejects FP8 KV for MLA on anything but SM90; this is a test-coverage limit, not a kernel limit | Allow it for the FA2 backend on SM12x | none (validation only) |
| attn | `fi_tile_cap` | The FA2 MLA kernel picks its KV tile from the shared memory size; with FP8 KV the chosen configuration needs 117,312 B per block, above GB10's 101,376 B limit | Cap the FP8 KV tile so it fits (16-wide tile, 91,680 B) | Tile size affects the softmax accumulation order; the same configuration as the previous base |
| attn | `fi_uri` | FlashInfer names JIT modules by parameters only, so a prebuilt stock module could be served instead of the edited one | Give the edited module its own name | none |
| pdl | `pdl` | Programmatic dependent launch is enabled for every capability >= 9; on GB10 it is a source of nondeterministic corruption in the linear-attention state kernels | Turn it off on SM12x | Launch scheduling only |
| topk_neg1 | `topk_neg1_prefill`, `topk_neg1_decode` | Uninitialised pool top-k slots could feed bogus indices into the attention gathers | Start those buffers at -1 | Bit-exact whenever every slot is written |
| kpool_pid | `kpool_pid` | A pool id at or past the last complete pool expands to tokens that do not exist | Keep only ids in [0, pool_len) | Bit-exact for valid ids; invalid ids are excluded |
| moe | `moe_table` | No tuned fused-MoE table exists for this model's expert shape (E=288, N=256) on GB10, so a default heuristic is used | Install a tuned table | Same K-blocking and scale order; tile sizes, warps and stages change |

The image also sets a few environment defaults the recipe relies on: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`VLLM_ENGINE_READY_TIMEOUT_S=3600` (an 8-node cold boot with JIT builds takes longer than the default), the CUDA arch lists
for `12.1a`, and the vLLM usage-statistics opt-out (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`).

To bisect one edit group, rebuild with `--build-arg BAKE_ARGS="--skip <group>"` (groups: `attn`, `pdl`, `topk_neg1`,
`kpool_pid`, `moe`). `attn` is all-or-nothing.

## How to tell that the image is right

- Build: `verify_bake.py --static` prints one PASS line per edit (it runs inside `build.sh`).
- Boot log of rank 0: `Using FLASHINFER_MLA_SPARSE_SM90 attention backend` and
  `Using configuration from .../E=288,N=256,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json`, and no
  `Using fp8_ds_mla KV cache format` line. `launch/readiness.txt` requires the attention-backend line and fails on the
  `fp8_ds_mla` line; the MoE-table line is only reported (a build with `--skip moe` logs "Using default MoE config"
  instead).
- Optional, on one idle node: `image/build.sh --gpu-smoke` runs the FA2 FP8 MLA kernel, the PDL switch and the pool
  expansion on the GPU.

## Testing the bake on a laptop

`tests/test_bake.py` applies the edits to copies of the pristine sources and checks them (Python 3 and the standard
library; no GPU, no torch). It needs two inputs that are not in this repository:

```bash
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm vllm-v0.32.0
pip download flashinfer-python==0.7.0.post1 --no-deps -d image/ref     # sha256 c7adf826...0716, pinned in the test
BAKE_TEST_VLLM_TREE=vllm-v0.32.0/vllm python3 -I image/tests/test_bake.py
```

## Credits

The original seven SM121 steps are the work of MiaAI-Lab (`glm53-flash_SM121.py` in
`MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark`, MIT; the notice is in `LICENSES/MiaAI-Lab.MIT.txt`). The anchors were
re-derived for vLLM v0.32.0 and FlashInfer 0.7.0.post1 and cross-checked against the public port in
`SpaceshipCreative/glm-5.3-flash-spark`, which also contributed the FlashInfer retargets used here (the FP8 gate, the
tile cap and the separate JIT name) and the PDL variant.
