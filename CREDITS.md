# Credits

This recipe stands on other people's work. Thank you.

## Software

- **vLLM** and its contributors: the engine, and the GLM-5.3-Flash support this recipe builds on. Upstream work this
  release relies on directly includes GLM-5.3-Flash support (#53906), the TP-sharded indexer prefill (#54951), FlashKDA
  chunked prefill (#55737), the kpool and indexer fixes that replaced several of our earlier patches (see
  [PATCHES.md](PATCHES.md)), and the tool-call strictness work (#56403, #61328).
- **FlashInfer**: the FA2 MLA kernels the GB10 attention path runs on.
- **MiaAI-Lab**: the original SM121 bake for GLM-5.3-Flash on DGX Spark (MIT), whose steps the build-time edits in
  `image/` derive from, and their published GLM-5.3-Flash Spark recipes.
- **SpaceshipCreative** (`glm-5.3-flash-spark`, Apache-2.0): the FlashInfer 0.7.0.post1 retargets and the PDL variant
  we cross-checked the bake against.
- **Anemll** and MiaAI-Lab's DSpark recipe: the GB10 spin-wait setting that patch 001 carries.
- **kindlingai** (`glm-5.3-flash-gx10`): the idea of running the fused mHC decode kernel on bf16 weights (patch 019; no
  code was copied), and with **knapcio** (`GLM-5.3-Flash-4x-DGX-Spark-TP4`) independent work on FA2 with an FP8 KV cache
  on GB10.

## Community recipes and benchmarks

The benchmark conventions here (state the prompt class, salt every cold prompt, publish raw results) follow the
GLM-5.3-Flash DGX Spark community. We read, among others: alexellis/rigmark and alexellis/glm-5.3-flash-4x-dgx-spark-switchless,
knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4, jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless,
Wpnx330/GLM-5.3-Flash-FP8-4x-DGX-Spark, himorishige/glm53-flash-2x-dgx-spark-recipe, eugr/spark-vllm-docker and
FujitsuPolycom/sparkring.

## What is original here

- Running GLM-5.3-Flash FP8 at **tensor parallel 8 across eight Sparks** with the full 1,048,576-token context, and
  the multi-node findings that made it stable: NCCL channel limits on the RoCE fabric, file-descriptor limits at eight
  ranks, the custom all-reduce rendezvous stall, link-flap handling so a short outage does not poison a running engine
  (see [docs/](docs/)).
- The runtime patches for GB10 at TP=8 (spin-wait and blocking sync, one-shot all-reduce, mHC kernels and sequence
  parallelism, skinny indexer kernels, sparse-MLA Triton prefill, draft-path compression, replicated embedding,
  FP8 W8A16 MLA projections, the depth-gated indexer sharding and its compact exchange), documented in
  [PATCHES.md](PATCHES.md).
- The measurement that v0.32's default grammar-constrained tool calls cost most of the MTP decode speed on agent
  workloads on this setup, and the tool-parser plugin that restores the previous default.
