# Benchmarks

The tools behind the numbers published for this recipe: GLM-5.3-Flash FP8 on vLLM, tensor parallel over eight DGX
Spark nodes, served through the OpenAI-compatible API. They send requests to the API and read its `/metrics`; they run
on any machine that can reach the API and need nothing installed on the nodes. The prompts, seeds, cells, `max_tokens`,
sampling and formulas are the ones that produced our tables. Only plumbing differs: the server address and the model
name come from arguments and `GET /v1/models`, paths are arguments, and the node watchdog is optional.

```text
bench/
  run-bench.sh        the whole sequence in the order we run it: drift, context curve, decode, serve, canary -> summary.md
  tools/
    drift-probe.py        cold vs warm prefix-cache drift
    ctx-curve.py          cold prefill, TTFT and decode vs context length and users (optional node guard)
    decode-bench.py       single-user decode speed, short chat prompts
    decode-bench-w2.py    single-user decode speed, recorded coding-agent turns
    decode-bench-w2-tc.py the same turns with tool_choice auto or none
    conc-sweep.py         closed-loop chat concurrency sweep
    canary-leak-test.py   cross-request leakage / corruption canary
    arm-measure.sh        decode-bench-w2 + decode-bench + conc-sweep as one controlled "arm"
    summarize.py          builds summary.md from a run directory
  data/               scripts that build the inputs from public datasets (no data is shipped), see data/README.md
  results/            default output location of run-bench.sh
```

## Quick start

```bash
# 1. Build the inputs once. They are downloaded from Hugging Face to your machine (details: data/README.md).
pip install huggingface_hub transformers jinja2 pyarrow numpy
python3 bench/data/fetch_sources.py
python3 bench/data/prep_w2_agent.py
python3 bench/data/prep_w3_docqa.py

# 2. Smoke test, about 25 minutes. Its numbers are not comparable with a full run.
bench/run-bench.sh --quick

# 3. The full sequence, about 2.5 hours. State the clock cap you run at; it is recorded in the summary.
#    Add --guard (ssh to the nodes, see "The optional node guard") to reproduce our context-curve numbers.
bench/run-bench.sh --clock-cap "2200 MHz"
```

**Data.** The inputs are built on your machine from public Hugging Face datasets
(`nebius/SWE-rebench-openhands-trajectories`, `zai-org/LongBench-v2`, `princeton-nlp/SWE-bench_Lite_bm25_27K`, and
optionally `allenai/WildChat-4.8M`) at pinned revisions. This repository ships no data: its scripts download and convert
the datasets on your machine, nothing is redistributed, and the prepared files are git-ignored. Licenses: see each dataset card. WildChat contains real user prompts:
only its user turns are used, and they are not redistributed here. Details: [`data/README.md`](data/README.md).

Needs: the API up and **quiet** (no other clients; nothing else sending requests), `python3` 3.9 or newer and `curl` on the
machine that runs the benchmark. The tools use the standard library only. The data scripts need the packages above. Set
`BASE_URL` (or pass `--base-url`) if the API is not at `http://127.0.0.1:8000`; if you changed `API_PORT` in `cluster.env`,
this is the place to follow it. For the full context curve the served `max_model_len` must be at least 1,040,512 tokens;
cells that do not fit are left out and listed in `summary.md`. Run the benchmark from another machine than the nodes if you
can, or pin it to the efficiency cores, and keep everything else off the nodes while it runs ([README, Operations](../README.md#operations)).

## What each benchmark measures, and why

| Tool | Measures | Why it matters |
|---|---|---|
| `drift-probe.py` | Whether a request whose prompt prefix comes from the prefix cache produces the same tokens as the same request computed cold | The model is hybrid (linear attention plus sparse attention): restoring recurrent state at a cached block boundary can go wrong silently, and the output is then wrong without any error |
| `ctx-curve.py` | TTFT, prefill tok/s, decode tok/s and decode step time for one user at 8k to 1,040,000 prompt tokens, and 2, 4, 8 users at 32k; all prefills cold | How the recipe behaves at long context, and where it stops being comfortable |
| `decode-bench.py` | Single-user decode tok/s on 6 short chat prompts, with speculative-decoding acceptance | The decode speed a chat user sees |
| `decode-bench-w2.py` | The same on 16 recorded coding-agent turns with 7k-49k tokens of context and tool definitions | The decode speed an agent sees; the same turns on every run, so two configurations pair turn by turn |
| `decode-bench-w2-tc.py` | The agent turns with `tool_choice` auto and none | Separates the cost of vLLM's tool-call enforcement (structural tag) from everything else |
| `conc-sweep.py` | Aggregate tok/s, per-request rate and TTFT with 1-8 closed-loop chat users, 120 s per level | How throughput and latency trade off as users are added |
| `vllm bench serve` | The standard vLLM load tool on random prompts, 8192 in / 1024 out, concurrency 1, 2, 4, 8 | A reference that others can reproduce with a stock tool. It is synthetic; see the pitfalls |
| `canary-leak-test.py` | 8 long-lived concurrent agent-shaped sessions with unique canary strings for 80 minutes: does a response ever contain another session's string, degenerate text, or a warm answer that differs from the cold one | Prefix cache + speculative decoding + hybrid attention have public reports of rare, silent cross-request corruption. A load-dependent bug does not show in a one-shot test |

### Prompt classes and settings

| Tool | Prompts | Sampling | Thinking | Output length |
|---|---|---|---|---|
| `drift-probe.py` | document text (W3), prefixes of 9001, 17333, 23011 tokens + 211 | temperature 0, top-5 logprobs | not applicable (completions on token ids) | 48 tokens |
| `ctx-curve.py` | document text (W3, then W2 traces beyond 0.9 M tokens), one distinct slice per user | 0.6 / top_p 0.95, seed 1000 + user | off | `--max-tokens` (512), `ignore_eos` |
| `decode-bench.py` | 6 fixed chat prompts | 1.0 / 0.95, seed 1000 x rep + prompt | off | 400, `ignore_eos` |
| `decode-bench-w2.py` | 16 recorded agent turns (first 8 sessions, an early and a late turn each) | 1.0 / 0.95, seed 7000 + 100 x rep + turn | template default (on) | 256, `ignore_eos` |
| `conc-sweep.py` | 8 fixed chat prompts | 1.0 / 0.95, fixed seeds | off | up to 300, ends naturally |
| `vllm bench serve` | random token ids | not set by the client (server default) | not applicable | 1024, `--ignore-eos` |
| `canary-leak-test.py` | synthetic sessions with unique canaries, 8k-100k tokens | temperature 0 and 1 | off (run-bench) | up to 768 |

## Running one benchmark

Every tool prints its usage with `--help`. They read `BASE_URL` (default `http://127.0.0.1:8000`) and, for the data,
`BENCH_DATA_DIR` (default `bench/data/prepared`); each also has `--base-url`.

```bash
export BASE_URL=http://node1:8000
mkdir -p out

bench/tools/drift-probe.py       out/drift.json --docs 4
bench/tools/ctx-curve.py         out/ctx-curve.json --cells 8192x1,32768x1,131072x1 --reps 1 --max-tokens 512
bench/tools/ctx-curve.py         out/ctx-conc.json  --cells 32768x2,32768x4,32768x8 --reps 1 --max-tokens 512
bench/tools/decode-bench.py      out/decode-chat.json 2
bench/tools/decode-bench-w2.py   out/decode-agent.json 2
bench/tools/decode-bench-w2-tc.py out/tc-auto.json auto        # then: ... out/tc-none.json none
bench/tools/conc-sweep.py        out/conc.json 1,2,3,4,6,8 120
bench/tools/canary-leak-test.py  "$BASE_URL" --sessions 8 --time-budget 4800 --out-dir out/canary \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
bench/tools/arm-measure.sh       --out out --label decode        # decode-bench-w2 + decode-bench + conc-sweep
```

Send a few requests first (see "Warm-up" below). `ctx-curve.py` also has `--build-only` (build the prompts, print their sizes,
send no chat request). The canary has `--dry-run`, `--mock` and `--selftest`, which need no server.

### `vllm bench serve`

`run-bench.sh` runs `vllm bench serve` for you with `--vllm-bench auto|local|docker|skip`. Run it by hand like this, for
concurrency `C` and `N = max(8, 4 x C)` prompts (the model name is the one `GET /v1/models` lists):

```bash
vllm bench serve --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions \
  --model "$SERVED_MODEL" --tokenizer "$MODEL_DIR" --trust-remote-code \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
  --num-prompts "$N" --max-concurrency "$C" --seed "$((1234 + C))" \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir out --result-filename "vllm-bench-c$C.json"
```

Two ways to get a `vllm` CLI, both supported by `run-bench.sh`:

* **Local install.** `pip install vllm` on a Linux machine (only the client side of the CLI is used). `run-bench.sh` uses
  `vllm` from `PATH`, or `VLLM_BIN`. The tokenizer must be a directory on that machine that holds the checkpoint's
  tokenizer files (copy just those files if you do not want 300 GB of weights); `run-bench.sh` takes `MODEL_DIR` from
  `cluster.env`, or `--tokenizer DIR`.
* **Container** (the mode we used). Use the recipe's own vLLM image, which contains the CLI:
  `run-bench.sh --vllm-bench docker --image IMAGE`
  (the default image is `M_IMAGE` from `config/recipe.env`). It runs `docker run --rm --network host --entrypoint vllm ...`
  with the tokenizer directory mounted read-only at the same path and the output directory at `/out`, so run it on a Linux
  host that can reach the API (typically node1). The result files are owned by root.

The usage-statistics opt-outs (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`) are set for the client.

## Running everything: `run-bench.sh`

```text
bench/run-bench.sh [--out DIR] [--quick] [--clock-cap TEXT] [--guard] [--skip LIST | --only LIST] [more, see --help]
```

Steps and what `--quick` changes (steps are named for `--skip` / `--only`):

| Step | Full | `--quick` | Time, full |
|---|---|---|---|
| health check | `GET /health`, engine version, model, `max_model_len` | same | seconds |
| `drift` | `drift-probe.py --docs 4` | `--docs 1` | about 5 min |
| cool-down hint | printed; `--cooldown SECONDS` also waits | same | 0 |
| `ctx` | a discarded warm-up cell (32k, 64 tokens); one user at 8192, 32768, 131072, 262144, 524288, 1040000 tokens; then 32768 x 2, 4, 8 users; `--max-tokens 512`, one repetition | warm-up, 8192 and 32768 x 1, then 32768 x 2 | about 25 min, plus cool-down waits with `--guard` |
| `decode` | `arm-measure.sh`: agent turns x 2, chat x 2, sweep 1,2,3,4,6,8 users x 120 s | x 1, sweep 1,2,4 users x 30 s | about 25 min |
| `serve` | `vllm bench serve` at concurrency 1, 2, 4, 8 | concurrency 1 and 4 | about 15 min |
| `canary` | 8 sessions, 4800 s (plus up to 900 s for requests in flight) | 4 sessions, 300 s | 80-95 min |
| summary | `summary.md` | same, marked as a quick run | seconds |

The full run takes about 2.5 hours (about 3 hours with `--guard`), the quick run about 25 minutes. The estimates come from
the rates we measured for this recipe (prefill about 3,400 tok/s up to 512k tokens and 3,260 tok/s at 1M, decode about
75 tok/s for one user); a slower configuration takes longer, and the 1M cell alone takes about 5.5 minutes of prefill.

The order is the one we use: the drift probe and the discarded 32k cell warm the engine up, the long cells follow, and the
canary runs last because it takes the longest and does not measure speed. After every step that needs the engine,
`run-bench.sh` checks `/health` and stops the run if it is gone, still writing `summary.md` for what finished; Ctrl-C does
the same. Exit status: 0 all steps ran, 1 a step failed, was skipped for lack of a tool, or the engine was not healthy,
2 a usage or setup error, 130 interrupted. `ctx-curve.py` exits with 3 when a request stalls ("possible wedge"): check the
engine ([README, Operations](../README.md#operations)) before anything else. The canary exits with 0 pass, 1 fail,
2 warn or underpowered, 3 error; `run-bench.sh` treats 2 as a warning.

Output directory (`--out DIR`, default `bench/results/run-<UTC time>`, must not exist or be empty):

```text
summary.md           one table per measurement, each followed by its disclosure line
run-info.json        profile, date, engine version, model, max_model_len, clock cap, guard state, cells left out
steps.tsv            step, result, exit status, seconds
drift.json  ctx-curve.json  ctx-conc.json  decode-agent.json  decode-chat.json  decode-conc.json  vllm-bench-c*.json
canary/              report.json, summary.txt, requests.jsonl, flagged/
raw-local/           console log and the discarded warm-up; git-ignored, may contain host names
```

Review the files before you publish them. The JSON files written by our tools hold no URLs, host names or local paths,
except the model name that `GET /v1/models` reports (the checkpoint path when the server was started without
`--served-model-name`); it appears in `run-info.json`, `summary.md`, `ctx-*.json` and the canary report.
`vllm-bench-c*.json` is written by vLLM and records the model name and the tokenizer path you passed.

## The optional node guard

Long cells heat the nodes, and the memory margin gets thin at 512k to 1M tokens (the launcher itself refuses to start below
`MEM_AVAILABLE_MIN_GIB`, 40 GiB, see `cluster.env.example`). In our measurements the context curve ran with a watchdog that
read every node; it is available here but **off by default**, because it needs ssh access to the nodes. Without it, `ctx-curve.py` and `run-bench.sh` say so at the start and nothing waits for the nodes to cool. To
reproduce our context-curve numbers, turn it on:

```bash
bench/run-bench.sh --guard --clock-cap "2200 MHz"
bench/tools/ctx-curve.py out/ctx.json --cells 262144x1 --guard          # or on its own
bench/tools/ctx-curve.py --guard --dry-run                              # read every node twice, send nothing
```

What it does (`ctx-curve.py --guard`):

* Reads each node every 5 s over one persistent ssh connection: the hottest ACPI thermal zone, GPU temperature and power
  draw, and `MemAvailable`. Needs passwordless ssh to every node and `nvidia-smi` on the nodes.
* A node at zone >= `--zstop` (default 87 C; `run-bench.sh` uses 88) or GPU >= `--gstop` (83 C), or with `MemAvailable`
  below `--memstop` GiB (default off; `run-bench.sh` uses 41, one GiB above the launcher's 40 GiB minimum), on two readings in
  a row trips the watchdog: the cell is aborted and the run stops (exit 2).
* Before every cell, all nodes must be freshly read at <= `--cool` (zone, default 65 C) and <= `--gcool` (GPU, default 60 C),
  for at most `--cool-max` seconds (600), otherwise the run stops. This wait is part of the method: back-to-back cells on
  warm nodes measure the heat.
* The node list is `--nodes "n1 n2 ..."`, else `$NODES`, else `NODES` or `NODE1`..`NODE8` in `cluster.env`
  (`$CLUSTER_ENV` or `--cluster-env` points elsewhere; the default is `cluster.env` in the repository root). `SSH_USER` and
  `SSH_OPTS` come from the same file. A node whose name equals this machine's hostname is read directly. Results name nodes
  by position (`node1` is the first entry), never by host name.
* If a node cannot be read, `--guard` refuses to start (exit 3) instead of running unprotected.

`--guard` in `run-bench.sh` also makes `arm-measure.sh` take a RoCE / link counter snapshot of every node before and after
the decode runs (`--fabric`; `ROCE_HCA` and `FABRIC_IFACE` from `cluster.env`): `packet_seq_err`, `out_of_sequence`,
`local_ack_timeout_err` and `link_down_events_phy`. Any ack timeout or link-down growth, or more than about 100 new
`out_of_sequence` in a run, is reported as a warning: a run that adds packet loss is not a clean run
([README, Operations](../README.md#operations)).

If you have no ssh access, follow the cool-down advice by hand: let every node return to idle (in
our measurements the hottest board sensor at 44-47 C and the GPU under 10 W; `launch/status.sh` shows the GPU's power and
temperature, and `cat /sys/class/thermal/thermal_zone*/temp` on a node shows the board sensors) before the long
cells, or use `--cooldown SECONDS`.

## Reading the outputs

**`ctx-curve.json`, `ctx-conc.json`** (`cells[]`, rewritten after each cell). Per cell: `N` (prompt tokens per user, nominal)
and `U` (simultaneous users). In `reqs[]`, per user: `prompt_tokens` (exact, from the server's usage), `ttft_s` (send to
first streamed token; includes queueing), `prefill_tok_s` = `prompt_tokens` / `ttft_s`, `decode_tok_s` =
(completion tokens after the first chunk) / (last token time - first token time), using the server's exact token counts, and
`tl`, the token timeline. For one user: `mean_accept_len` = 1 + accepted / drafts and `step_ms` = 1000 x decode seconds /
drafts, both from the server's speculative-decoding counters, so the step time is free of acceptance noise. For several
users: `window` is the interval in which all users are decoding, with `per_user_tok_s` and `agg_tok_s` (their sum) inside
it; if the users overlap for less than 5 s only a note is recorded. `metrics_delta.prefix_cache_hits_total` must be about 0
(the prompts are cold), `num_preemptions_total` is preemptions during the cell, `kv_usage_max` is a fraction of the KV
pool. With the guard: `fleet_peak` (zone C, GPU C, power W per node), `mem_avail_min_gib`, `waited_cool_s`. `aborted` is
the reason a cell stopped, or null.

**`decode-chat.json`.** `decode_tok_s_median` / `_mean` / `_all` use 399 / (last chunk - first chunk) per request;
`AL` = 1 + accepted / drafts and `accept_rate` = accepted / proposed draft tokens, over the whole run.

**`decode-agent.json`.** `rows[]` has `turn`, `ctx` (recorded prompt tokens), `decode_tok_s` = 255 / (last delta - first
delta) and the per-request `AL`; then `median`, `mean`, `AL`, `n`.

**`decode-conc.json`.** One object per level: `agg_tok_s` from the server's `vllm:generation_tokens_total` over the
window (system throughput), `per_user_tps_p50` (median per-request **chunk** rate, see pitfalls), `ttft_p50`, `ttft_p90`,
`running_mean`, `preemptions`, `kv_usage_end`, `n_req`.

**`drift.json`.** One entry per document and split point, then `summary`. Warm should look like the noise floor
(`cold_vs_cold`): identical, or a late and weak divergence. An early, strong divergence on the warm side only means the
cache restored the wrong state. `warm_cache_hits` must equal `n`; if it is 0 the warm request never used the cache and the
test says nothing.

**`vllm-bench-c*.json`.** Written by vLLM: `output_throughput`, `request_throughput`, `mean_/median_/p50_/p90_/p99_` of
`ttft`, `tpot`, `itl`, `e2el` in ms, and (when the server exposes the counters) `spec_decode_acceptance_length`. TPOT
excludes the first token; ITL is the gap between streamed chunks.

**`canary/report.json`, `summary.txt`.** `verdict` PASS, WARN, FAIL or ERROR. A LEAK is another session's marker, access
code or invented word in a response; a PHANTOM is a canary-shaped string nobody was given (a corrupted copy). With zero
events in n requests the 95 % upper bound on the per-request rate is about 3 / n, so 600 requests certify under 0.5 %.
Cold replays (fresh `cache_salt`) are the control group for the warm answers.

## Disclosure rules

Under every published table we state these eight things. `summary.md` prints them under each table, filled from the run
(date, engine and clock cap from `run-info.json`; the rest from the tool's output and fixed settings). Fill in what the
tools cannot know.

1. **Date** of the run (UTC).
2. **Engine version**: the vLLM version (`GET /version`), the image, and which patches were applied. A different engine is a
   different table.
3. **Prompt class**: real chat prompts, recorded agent turns, document text, or random tokens. Never mix them in one number.
4. **n**: requests (and users, repetitions) behind each number. A median of 3 is not a p90.
5. **Temperature** (and top_p) and whether the output length was forced (`ignore_eos`) or ended naturally.
6. **Thinking state**: on, off, or the template default, and the reasoning effort if you set it.
7. **GPU clock cap**: we run all eight nodes at 2,200 MHz (`sudo nvidia-smi -lgc 0,2200`, re-applied after every reboot;
   [README, Operations](../README.md#operations)). It is not detected; pass `--clock-cap`. Compare only runs at the same cap.
8. **Definition of concurrency**: what "N users" means. `conc-sweep.py`: N client threads, each with exactly one request in
   flight at all times, no think time. `ctx-curve.py`: U requests submitted together, one per user. `vllm bench serve`:
   `--max-concurrency`, unlimited request rate. The canary: concurrent sessions, each sequential, plus forks and replays
   within a cap of 8 in flight.

Also say whether the prefill was cold or warm, report aborted and failed cells instead of dropping them, and say whether
the node guard was on. Example footer:

> *<date> UTC, vLLM <version>, chat prompts (6 fixed), n = 12 per number, temperature 1.0 / top_p 0.95, thinking off,
> 2,200 MHz, one request at a time, `ignore_eos` with 400 tokens.*

## Pitfalls

* **Thinking tokens.** The model thinks by default. A client that waits for `content` sees a TTFT that includes the whole
  reasoning, and with a small `max_tokens` the answer may never start (`finish_reason` length, empty content; the canary calls
  this BUDGET). Reasoning arrives in a separate delta field (`reasoning` in vLLM v0.32; older versions used
  `reasoning_content`) that counts as output but not as `content`. The chat tools here turn thinking off
  (`chat_template_kwargs.enable_thinking=false`), so TTFT is the first answer token; `decode-bench.py` and `conc-sweep.py`
  look only at `content` and `reasoning_content` and are meant for thinking off. The agent tools leave the template default
  and count reasoning and tool-call deltas. Whatever you choose, state it.
* **Random-token prompts.** `vllm bench serve --dataset-name random` sends random token ids, `--ignore-eos` forces 1024
  tokens of continuation, and every request carries an 8,192-token prefill. MTP still accepts a fair share of drafts
  (acceptance length 2.17 for one user in our run, against 2.49 on chat prompts; the result file shows
  `spec_decode_acceptance_length` when the server exposes the counters), but the output rate stays below what a chat or
  agent user sees. It is a reference for comparing with other posts; quote the chat and agent tools for user speed.
* **`cache_salt` for cold prefill.** Prefix caching is on, and a repeated prompt skips most of its prefill: in our measurements
  it roughly doubled one prefill reading. `ctx-curve.py` sends a fresh random `cache_salt` with every request and reports
  `prefix_cache_hits_total` per cell (about 0 when it worked); `drift-probe.py` and the canary's cold replays do the same. Do
  not switch prefix caching off to get cold numbers: that measures a different engine. `vllm bench serve` sets no salt:
  with one seed for every concurrency level it regenerates the same prompts, and later runs hit prefixes cached by earlier
  ones (TTFT then looks far too good). `run-bench.sh` therefore uses seed 1234 + concurrency. Compare the server's
  prefix-cache hit rate over the run before you trust a TTFT from it.
* **Warm-up.** The first requests after a start pay for one-time kernel compilation and graph work, and a first boot has
  empty JIT caches. `run-bench.sh` runs the drift probe and a discarded 32k cell first. If you run a tool on its own, send a
  few requests first and discard them.
* **Aggregate vs per-user.** `agg_tok_s` is what the whole system generates, summed over users; per user it is lower. Use
  aggregate / users for a token-based per-user figure. `conc-sweep.py`'s `per_user_tps_p50` counts streamed chunks, not tokens;
  with speculative decoding a chunk usually carries more than one token, so it reads low by about the mean acceptance
  length. `vllm bench serve`'s `output_throughput` divides by the whole duration including ramp-up and tail, so with few
  prompts it understates steady state. Never quote an aggregate as "speed".
* **Forced lengths.** `ignore_eos` makes every request do the same work, which is what a speed comparison needs, but real
  answers end earlier and at different points. Only `conc-sweep.py` lets answers end naturally.
* **Noise.** Identical runs differed by up to about 7 % on some metrics and about 1 % on the controlled decode runs. Run each
  configuration at least twice and treat smaller differences as noise. Compare only runs at the same clock cap, on the same
  engine, after a cool-down.
* **Quiet endpoint.** Any other client, health check that generates tokens, or model load on the nodes changes the numbers.
