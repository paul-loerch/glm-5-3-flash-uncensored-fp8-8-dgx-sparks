#
# SM121 bake for GLM-5.3-Flash on vLLM v0.32.0 (arm64, CUDA 13.0, FlashInfer 0.7.0.post1) -- GB10 / DGX Spark.
#
#   build   docker build -t glm53-flash-sm121:v032-1 -f Dockerfile .        (context = this directory, see .dockerignore)
#   check   docker run --rm --entrypoint python3 glm53-flash-sm121:v032-1 -I /opt/glm53-sm121/verify_bake.py
#   gpu     docker run --rm --gpus all --entrypoint python3 glm53-flash-sm121:v032-1 -I /opt/glm53-sm121/gpu_smoke.py    (one GB10 node, optional)
#   offline docker build --build-arg BASE_IMAGE=<local tag of the base> ...   (a `docker load`ed image has no registry digest)
#
# What it does: python3 glm53_sm121_v032.py edits vLLM and FlashInfer SOURCE files in site-packages (anchors must match
# exactly once, sha256 pins, nothing is written if anything is off) and installs the tuned Triton MoE table.  No network,
# no pip, no GPU is needed at build time (no `# syntax=` line on purpose: it would make BuildKit pull a frontend image).  The edits and why they are needed: IMAGE.md next to this file.
#
# Credit: the original SM121 bake is MiaAI-Lab's (MIT, LICENSES/MiaAI-Lab.MIT.txt); see the header of the script.

ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0-aarch64@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
FROM ${BASE_IMAGE}

# Extra arguments for the bake script, e.g. BAKE_ARGS="--skip pdl" to bisect one edit group
# (groups: attn, pdl, topk_neg1, kpool_pid, moe) or "--allow-pin-mismatch" after reviewing a diff against another base.
ARG BAKE_ARGS=""

COPY glm53_sm121_v032.py verify_bake.py gpu_smoke.py /opt/glm53-sm121/
COPY LICENSES/ /opt/glm53-sm121/LICENSES/
# A directory, not the file: the table's name contains "[128,128]", which COPY would read as a glob.
COPY moe/ /opt/glm53-sm121/moe/

# SKIP_VERIFY=1 is an escape hatch for the post-bake check only (e.g. if the check itself trips over something in the environment).
ARG SKIP_VERIFY=0

RUN python3 -I /opt/glm53-sm121/glm53_sm121_v032.py ${BAKE_ARGS} \
 && if [ "${SKIP_VERIFY}" = "1" ]; then echo "verify_bake skipped (SKIP_VERIFY=1)"; \
    else python3 -I /opt/glm53-sm121/verify_bake.py --static; fi

# Serving defaults the recipe relies on without passing them itself.  All are overridable with `docker run -e`.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1

LABEL org.opencontainers.image.title="vLLM v0.32.0 + SM121 bake (GLM-5.3-Flash, GB10)" \
      org.opencontainers.image.description="Official vllm-openai v0.32.0 aarch64 plus build-time source edits that run GLM-5.3-Flash on sm_121" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.32.0-aarch64" \
      org.opencontainers.image.base.digest="sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc" \
      sm121.bake="glm53_sm121_v032 1"
