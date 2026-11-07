# GLM-5.3-Flash FP8 on eight DGX Sparks

Serve **GLM-5.3-Flash in FP8** with **vLLM v0.32.0** across **eight NVIDIA DGX Spark** nodes: tensor parallel 8 over a
100 GbE RoCE fabric, MTP speculative decoding, and the model's full **1,048,576-token** context window. The engine is the
official vLLM image by digest plus a few build-time GB10 edits, 14 documented runtime patches and one tool-parser plugin.

[Results](#results-at-a-glance) · [What it is](#what-it-is) · [Requirements](#requirements) · [Quick start](#quick-start) ·
[Configuration](#configuration) · [Operations](#operations) · [How it works](#how-it-works) · [Patches](#patches) ·
[Known limitations](#known-limitations) · [Layout](#repository-layout)

## Results at a glance

Measured on 23-24 October 2026 on eight DGX Sparks with exactly the configuration in this repository (tensor parallel 8,
100 GbE, GPU clocks capped at 2,200 MHz, no other traffic). Speeds are generated tokens per second, measured with the tools
in [bench/](bench/); every prompt was new to the server, so nothing came from the prefix cache. Re-checked on 7 November 2026
from a fresh boot of exactly this repository (the speed benchmarks, not the quality checks): throughput and prefill within 2 % of
the numbers below (one user 73.8 / 76.0 tok/s, 8 users 162 tok/s, prefill 3,212-3,438 tok/s, a 1M-token prompt in 324 s).

| | |
|---|---|
| **One user** (while generating) | chat **~75** · coding agent **~77** tok/s |
| **Several users at once** (total, chat) | 2 users **94** · 4 users **125** · 8 users **164** tok/s, first token after 0.46 s (median, 8 users) |
| **Reading a prompt** (prefill) | **about 3,400 tok/s** for prompts of 8k-512k tokens, 3,260 tok/s at 1M |
| **Long prompts**, wait for the first token | 8k **2.4 s** · 32k **9.4 s** · 128k **38 s** · 512k **2.6 min** · 1M **5.3 min** |
| **Long prompts from 8 users at once** | eight 32k-token prompts sent together: all answers complete after **101 s**, 174 tok/s while all decode |
| **Context window** | **1,048,576 tokens** per request |
| **KV cache** | **2,571,268 tokens** in total: 2 requests at the full 1M context, or 8 at 256k |
| **Concurrency** | **up to 8 requests** at a time; further requests wait in a queue |
| **Quality** | tool calls **24/24** valid JSON (20/24 an exact match), long-context recall **12/12**, GSM8K **49/50**, HumanEval+ **20/20** |

## What it is

| | |
|---|---|
| Model | [`orcarouter/GLM-5.3-Flash-Uncensored-FP8`](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) at revision `3cec42d6` (full hash in step 2 of the quick start): an FP8 checkpoint of GLM-5.3-Flash (305.8 GiB), derived from `zai-org/GLM-5.3-Flash` |
| Hardware | 8 × DGX Spark (GB10, 128 GB unified memory each), one tensor-parallel rank per node, 100 GbE RoCE through one switch |
| Engine | vLLM v0.32.0, official `vllm/vllm-openai:v0.32.0-aarch64` image by digest, plus five build-time GB10 edits, 14 runtime patches that are applied when each container starts, and one tool-parser plugin |
| Decoding | MTP speculative decoding (2 draft tokens for one user, 1 for 2-8 users), FP8 KV cache, prefix caching |
| Memory plan | about 41 GiB of weights and a pinned 16 GiB KV pool per node; 45-48 GiB of unified memory stays free |
| License | Apache-2.0 for this repository; the model and the benchmark datasets have their own terms ([NOTICE](NOTICE)) |

**About the model.** Every number here was measured with the checkpoint above. It is an *abliterated* ("uncensored")
derivative: refusal behaviour was removed by its publisher. The official `zai-org/GLM-5.3-Flash-FP8` checkpoint has the
same architecture and should run with this recipe unchanged, but it has not been tested. See [Responsible use](#responsible-use).

## Requirements

- **Eight DGX Spark nodes** running DGX OS, with Docker and the NVIDIA container toolkit (both ship with DGX OS).
- **One 100 GbE link per node** to one switch, with jumbo frames (MTU 9000) end to end and RoCE working. Check every node
  against every other with `ib_write_bw` before the first launch: a single marginal cable or switch port slows the whole
  cluster to its pace.
- **About 330 GB (306 GiB) free on each node's NVMe** for the weights, plus about 10 GB for the image and a few GB of JIT
  caches: plan for 350 GB. Each rank reads its weights from its own node's disk, not from a network share.
- **Nothing else on the nodes.** About 45-48 GiB stay free per node while serving; another workload, a large copy or a
  second model on a node breaks that.
- **Key-based ssh** from the first node (NODE1) to the other seven.
- **Recommended: the GPU clock capped at 2200 MHz on every node** (`sudo nvidia-smi -lgc 0,2200`; lost on reboot). At
  TP=8 the hottest node sets the pace, GB10 units have been reported to power off at 88-92 °C, and every number here was
  taken at this cap.
- **Recommended host settings** (identical on all eight nodes): a `nofile` limit of at least 524288 for the account that
  runs the launcher (the containers get `--ulimit nofile=524288:524288`; eight NCCL ranks fail with Docker's default
  1,024), automatic upgrades off, and NetworkManager told to keep the fabric addresses through a short carrier loss
  ([Known limitations](#known-limitations)).

## Quick start

Run everything on NODE1, from a clone of this repository at the same path on every node (or copy `patches/` to that
path on the other nodes, step 4).

**1. Describe your cluster.**

```bash
cp cluster.env.example cluster.env
$EDITOR cluster.env        # node names, fabric addresses, interface and RoCE device, model path
```

**2. Put the weights on every node,** at the path `MODEL_DIR` names:

```bash
pipx install huggingface_hub        # or: uv tool install huggingface_hub; provides the `hf` command
. ./cluster.env
hf download orcarouter/GLM-5.3-Flash-Uncensored-FP8 \
    --revision 3cec42d6ed14ec197e328c09650c17fd3660c26a --local-dir "$MODEL_DIR"
for n in "$NODE2" "$NODE3" "$NODE4" "$NODE5" "$NODE6" "$NODE7" "$NODE8"; do
  rsync -a --mkpath "$MODEL_DIR/" "$n:$MODEL_DIR/"
done
```

**3. Build the image and copy it to all nodes** (about 10 GB):

```bash
image/build.sh --distribute
```

**4. Make the patch directory identical on every node** (the launcher checks this and refuses otherwise):

```bash
for n in "$NODE2" "$NODE3" "$NODE4" "$NODE5" "$NODE6" "$NODE7" "$NODE8"; do
  rsync -a --delete --mkpath patches/ "$n:$PWD/patches/"
done
```

**5. Start it.** The copy in step 2 leaves the weights in every node's page cache, and on GB10's shared memory that
cache counts against the GPU, so release it before the first start: set `DROP_CACHES=1` in `cluster.env` (it needs the
one-line sudoers rule described there), or run once by hand:

```bash
for n in "$NODE1" "$NODE2" "$NODE3" "$NODE4" "$NODE5" "$NODE6" "$NODE7" "$NODE8"; do
  ssh -t "$n" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
done
```

A boot takes 9-15 minutes: about 6.5 for the weights, about 2 of engine start-up, the rest kernel compilation and CUDA
graphs. A first boot with empty kernel caches took 9.7 minutes in our last check (7 November 2026).

```bash
launch/up.sh --dry-run     # read-only preflight on all eight nodes, prints what it would start
launch/up.sh --wait        # start the eight ranks and wait until the API answers
```

**6. Use it.** The API is OpenAI-compatible, on NODE1 at `API_PORT` (default 8000):

```bash
curl -s http://NODE1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm53-uncen-fp8-tp8",
  "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}]
}'
```

The model thinks first: its reasoning arrives in a separate `reasoning` field, then the answer in `content`. Leave it
enough `max_tokens`, or add `"chat_template_kwargs": {"enable_thinking": false}` for a short reply. The API has no key
and speaks plain HTTP: put access control and TLS in front of it before anyone but you can reach it.

Stop it with `launch/down.sh` (it refuses while requests are running; `--force` after a crash). Check it with
`launch/status.sh`.

## Is it working?

- `launch/up.sh --wait` only returns success when rank 0's log shows every expected line: all patches applied, the GB10
  attention backend, the KV cache size, the tool-parser plugin. The rules are in `launch/readiness.txt`.
- `launch/status.sh` shows every rank container, the model and its 1,048,576-token limit, and each GPU's power and
  temperature.
- `bench/run-bench.sh --quick` runs a short version of the published benchmarks, after you have prepared the benchmark
  data ([bench/README.md](bench/README.md), step 1).

## Configuration

`cluster.env` says where the recipe runs; `config/recipe.env` says what is served. The settings you are most likely to
change:

| Setting | In `config/recipe.env` | Default | Note |
|---|---|---|---|
| Context length | `--max-model-len` | 1048576 | Lower it if you never need the full window; it does not change speed by itself |
| Concurrent requests | `--max-num-seqs` | 8 | The KV pool and CUDA graphs are sized for at most 8 |
| KV cache per node | `--kv-cache-memory-bytes` | 17179869184 (16 GiB) | Holds 2,571,268 tokens; a full 1,048,576-token request uses about 41 % of it |
| Speculative decoding | `--speculative-config` | MTP, 2 / 1 draft tokens | Tuned for this setup: 2 drafts for one user, 1 from two users up |
| A runtime patch | the `VLLM_GB10_*` variables in `M_DOCKER_EXTRA` | all on (001 has no switch) | Remove a variable to turn its patch off; for 035 also delete its `require` rule in `launch/readiness.txt` |
| Page cache | `DROP_CACHES` in `cluster.env` | 1 recommended | Needed after the weights were copied or read |

Every patch except 001 is switched by its own environment variable in `M_DOCKER_EXTRA` ([PATCHES.md](PATCHES.md) names
each one); a patch whose switch is absent is inert. Set the same variables on all eight ranks: the launcher gives the
file to every rank unchanged.

## Operations

- **Tear down only with `launch/down.sh`.** Never `docker kill`, `rm` or `stop` a rank, and never kill an NCCL job, while
  the engine may be inside a collective: the other ranks are stranded and a GPU can be left spinning until its node is
  rebooted. `down.sh` refuses while requests are running (and when it cannot read the metrics), removes the workers first
  and rank 0 last, and waits until every GPU is idle; `--verify` only checks, `--force` is for after a crash.
- **After a crash, a wedge or a lost node:** save the rank logs first, let the engine fail on its own if it may still be
  inside a collective (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` is 600 s), then `launch/down.sh --force` and `--verify`. A GPU
  that stays busy with no container is an orphaned context: reboot that node. Find the cause (memory, link events,
  temperature, a node that rebooted and lost its clock cap) before you start again.
- **One serving stack at a time,** and no large copies or model loads on a node while a rank is starting or running: on
  GB10's shared memory a multi-hundred-GB copy alone can push free memory down to a few GiB.
- **Never run `systemctl daemon-reload` on a node with running GPU containers:** it cuts their GPU access.
- **Re-apply the clock cap after every reboot;** nothing checks it for you. Before rebooting a node, stop the engine
  everywhere, make sure no firmware capsule is staged (a capsule on a headless reboot can hang the node), and have someone
  able to reach the machine: DGX Spark has no BMC. After a reboot, check the link (MTU 9000, RoCE active) and its quality:
  a reboot can leave a marginal link flapping.
- **Kernel, driver, NCCL, image and firmware are one version:** identical on all eight nodes, changed on all eight
  together (canary node first), then measured again. The NVIDIA driver ships as precompiled modules for exactly one kernel.
- **Start from empty swap and an empty page cache;** `DROP_CACHES=1` does the cache.
- **Run stop, start and benchmark as separate commands** and check each exit status: chained commands hide failures.

## How it works

- **Eight small GPUs on one switch.** Each DGX Spark is a GB10 with 128 GB of memory shared with the CPU. The model is cut
  into eight, so every layer ends in network all-reduces; a decode step is limited by the memory bandwidth of each node and
  by the latency of those all-reduces over 100 GbE. The recipe sends decode-size all-reduces in one network step with a
  fixed rank-order sum (patch 012), uses four NCCL channels for the larger prefill messages, replicates the input embedding
  so that no all-reduce opens a step (032), and runs the narrow decode projections through skinny or FP8 W8A16 kernels (014,
  033).
- **Speculative decoding.** The checkpoint's MTP layer drafts 2 tokens for one user and 1 from two users up; verification is
  exact, so drafting changes speed, not the output distribution. The draft's `lm_head` runs in FP8 (029).
- **Prefill.** The model's mHC residual stream is sharded over the ranks in prefill, so the block all-reduces become
  reduce-scatter plus all-gather (025), its sparse MLA runs a Triton prefill kernel (020), and the sharded indexer exchanges
  512 pool ids per row instead of 2,051 expanded ids (035, +11-12 % prefill from 128k tokens).
- **Tool calls without a grammar.** vLLM v0.32 decodes every request that carries tools under a grammar, which on this
  setup took agent turns from 79.9 to 9.2 tok/s. The tool-parser plugin restores the earlier default: a grammar only when
  the client asks for one.
- **Exact or gated.** Every change is either bit-exact, exact for the output distribution (drafting), or it changes the
  numerics slightly (a reduction order or kernel) and passed the quality checks above. [PATCHES.md](PATCHES.md) says which
  is which.

## Patches

Everything is in [PATCHES.md](PATCHES.md): for each change the problem, the change, the switch, the numerics, the measured
effect and its upstream status. Effects marked "previous base" were measured on the earlier vLLM base this recipe was
ported from; the port to v0.32.0 was then measured as a whole (it beat the previous base on every number above).

| ID | What it does | Measured effect | Exact? |
|---|---|---|---|
| B1-B5 | build time: SM90 sparse-MLA backend on SM12x through FlashInfer FA2 with the FP8 KV tile capped for GB10, PDL off on SM12x, two indexer index guards, a tuned MoE table for 288 experts | enablers and guards; the MoE table chat decode +1-2 % | B1 another kernel; the rest exact |
| 001 | inter-process waits sleep after 2 ms instead of spinning for 1 s | scheduler-thread CPU 40 → 9 % | yes |
| 003 | the CUDA context sleeps in host waits | worker CPU 96-100 → 8-20 % | yes |
| 011 | mHC pre-norm `block_m=2` kernel from 128 tokens | prefills of 256-1,000 tokens 4-7 % faster | yes |
| 012 | decode-size all-reduces (≤ 64 KiB) in one network step | decode +11-12 % | order |
| 014 | skinny kernels for three narrow indexer projections in decode | decode +2-6 % | last bit |
| 019 | the fused mHC decode kernel reads bf16 copies of its weights | −0.21 ms per forward | yes |
| 020 | Triton kernel for sparse-MLA prefill rows | cold prefill +13-18 % from 4k tokens | prefill order |
| 022 | mHC pre-norm GEMM split by rows over the ranks in prefill | cold prefill +8.5 % at 32k (idle while 025 is on) | yes |
| 025 | sequence-parallel mHC residual stream in prefill | cold prefill +15 % at 32k and 512k | order |
| 029 | FP8 copy of the draft's `lm_head` | decode +1.5 % | drafts only |
| 031 | depth gate for the TP-sharded indexer prefill | 1M first token 572 → 413 s (own-code version) | yes |
| 032 | replicated input embedding | decode +4-5 % | yes |
| 033 | MLA projections as FP8 W8A16 (Marlin) for up to 16 tokens | decode +2.7 % (mean) | minimal |
| 035 | the sharded indexer exchanges 512 pool ids per row instead of 2,051 | prefill +11-12 % at 128k-1M (1M first token 358 → 319 s) | yes, checked at run time |
| T1 | tool-call grammar only when the client asks for one | 16 recorded agent turns: 9.2 → 79.9 tok/s | yes |

## Known limitations

- **Tool calls that force a grammar are slow.** Requests with `tool_choice` set to `required` or a named function, tools
  marked `strict: true`, or `parallel_tool_calls: false` still decode under a grammar, at a fraction of normal speed.
- **Text only.** The vision encoder is switched off (`--language-model-only`).
- **Not tested on this release: several prompts above about 64k tokens prefilling while other users decode.** On the
  previous base that mix could wedge the engine. Single prompts up to 1,040,000 tokens and eight concurrent 32k prompts
  were tested and ran clean.
- **At most 8 concurrent requests.** Concurrent long prompts share one 8,192-token prefill budget per step, so their
  time to first token grows with the number of users.
- **A network outage of 6 s or more breaks a running engine** even after the link recovers: NetworkManager drops the
  fabric address and the RoCE GIDs move, and NCCL keeps its old queue pairs. Restart the engine. To avoid most of these,
  tell NetworkManager to ignore carrier loss on the fabric interfaces (a drop-in with `ignore-carrier=<interface>` under
  `[main]` in `/etc/NetworkManager/conf.d/`, identical on every node).
- **Temperature 0 is not bit-reproducible** across boots: vLLM is not batch-invariant by default.
- **Boots are slow** (9-15 minutes), and every boot needs the same image ID and patch directory on all
  eight nodes. The patches pin the files of the image they were written against: a different vLLM image makes them
  refuse to start.

## Repository layout

```
cluster.env.example   your nodes, addresses and paths (copy to cluster.env)
config/recipe.env     what is served: image, vLLM arguments, patch switches
recipe.lock.yaml      every version and pin behind the published numbers
launch/               up.sh, down.sh, status.sh and the readiness rules
image/                Dockerfile, the build-time GB10 edits and build.sh (IMAGE.md)
patches/              runtime patches applied at container start, the tool-parser plugin, their checks
bench/                benchmark tools and dataset preparation (bench/README.md)
tests/                offline test of the launcher (stubbed ssh, docker and nvidia-smi)
tools/check-public.sh scan for private addresses, keys and paths before you publish a fork
PATCHES.md            every change to vLLM, why, its measured effect and its upstream status
CREDITS.md, NOTICE, LICENSE
```

## Credits and license

This recipe builds on vLLM, FlashInfer, MiaAI-Lab's SM121 work and the GLM-5.3-Flash DGX Spark community; see
[CREDITS.md](CREDITS.md). The code in this repository is Apache-2.0 ([LICENSE](LICENSE)); third-party terms are in
[NOTICE](NOTICE).

## Responsible use

The checkpoint used here has had its refusal behaviour removed, so it will answer requests the original model declines.
You are responsible for how it is used: follow the law and the model licenses, and add your own safeguards (access control,
usage policies, logging) before you let anyone else use the endpoint. The recipe starts the API without an API key (vLLM's `--api-key`
can add one); do not expose it beyond a network you control.
