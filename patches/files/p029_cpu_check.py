#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-029 gb10-draft-compress] CPU checks (Mac / any CPU: python3 >= 3.10 + numpy; NO torch, NO vLLM install).
vLLM v0.32.0 port (module rev 3).

  python3 -I -B files/p029_cpu_check.py --vllm-root <v0.32.0 source tree>/vllm [--work DIR] [--quick] [--mutants]

--vllm-root must hold the v0.32.0 ORIGINAL files (sha-checked against the .sh pins): models/glm5next/common/mtp.py,
v1/worker/gpu/spec_decode/mtp/speculator.py, v1/worker/gpu/spec_decode/eagle/utils.py, v1/worker/gpu/spec_decode/speculator.py
(contract only: sample_draft), model_executor/layers/logits_processor.py,
model_executor/layers/quantization/utils/marlin_utils{,_fp8}.py.
The REAL files are executed in a stub vLLM namespace over files/p029_testlib.py's numpy-backed fake torch. v0.32.0 specifics
the stubs model: marlin_utils registers torch.ops.vllm.marlin_gemm through direct_register_custom_op (the stub binds the
REAL _marlin_gemm wrapper to torch.ops.vllm, which calls the faked torch.ops._C.marlin_gemm positionally with b_q_type_id),
prepare_fp8_layer_for_marlin creates no workspace (the FP8 head must allocate and pass its own: the vllm.v1.worker.workspace
module is deliberately NOT stubbed, so a workspace=None call would fail its import), ops.gptq_marlin_repack has no perm,
SharedHead(defer_lm_head=True) starts without a head and the REAL load_eagle_model shares the target's lm_head into it,
speculative_config.apply_draft_overrides, hf_text_config.
Readiness rules (launch/readiness.txt) are read from $PATCH_READINESS_TXT, else <patches>/../launch/readiness.txt, else
files/testdata/readiness.txt (a copy of launch/readiness.txt).

  A  applier + pins: the v0.32.0 originals == the .sh pins; apply / check / revert on both targets == the pinned
     results (revert byte-exact); the hooks sit in the right functions (AST); refusals (marker present, anchor
     missing / duplicated, revert of a base, a hook moved to the wrong function); the module gate block alone (unset,
     0, 1, ' 1 ', true, 2; gate set + module missing -> ImportError); no earlier patch (NNN < 029) references either
     target as a write target (static enumeration of every .sh and vendored applier).
  B  module units on the fake torch: env parsing; per-row e4m3 quantisation (scale = bf16(amax/448) first, exact
     dequant, zero rows, the 448 code, chunking invariance); the FP8 head through the REAL marlin_utils_fp8 helpers
     (prepare_fp8_layer_for_marlin(size_k_first=False) + apply_fp8_marlin_linear, N padded 200 -> 256) == the exact
     dequant GEMM; eligibility; the self-test row helpers; _cuda_timer call counts; the H-checks on a small shard
     (clean -> pass; top-1/top-2 swapped on the random rows -> the H3 floor fails); peaked rows on a vocab with
     dominating / near-tie rows (skipped, not failed); peaked rows on an anisotropic shard (B9: the centred lift
     still builds them).
  C  8-rank threaded end-to-end on the REAL patched mtp.py + mtp/speculator.py + eagle/utils.py (lm_head sharing) +
     logits_processor.py (all-gather) + marlin utils: clean boot -> ACTIVE on every rank with identical collective
     schedules; draft forward (eh split = exact gather, within the bound of the replicated eh_proj, the one-shot
     really issued for M <= 8, the image path bitwise for M > 8) and draft logits (FP8 == the exact dequant GEMM,
     gathered in rank order, bf16 [M, vocab], the image path bitwise for M > HEAD_MAX_M or a swapped head); the
     shared target lm_head untouched; gate unset == the unpatched image files bitwise; parts=head / parts=eh; and the
     FAIL-CLOSED scenarios (one / all ranks without the one-shot, a corrupted FP8 head on one rank, a rank-local
     exception inside the eh collective phase, a one-shot that corrupts one rank's output, a fingerprint mismatch,
     failed barriers, a verdict exchange that fails locally on one rank, a slow FP8 GEMM on one rank,
     VLLM_BATCH_INVARIANT, an fp32 head_dtype, a non-GLM draft, an unexpected exception) -> the SAME decision on
     every rank, no hang, no schedule mismatch, nothing raised into the engine; C6: when prepare() calls
     torch.cuda.empty_cache(), a refused FP8 copy is already unreferenced, an accepted one alive once per rank.
     Rev 2: C1 also counts the CUDA-graph E5 (4 exchanges; 240 replayed one-shots between the capture and the verdict
     exchange; 4 graphs per rank captured on a side stream in thread_local mode, replayed 12 times, reset; the
     thread's stream restored) and applies the boot-gate scripts' patterns + launch/readiness.txt fail rules to the
     emulated boot log. B10: eh-gate parsing, the log defanging, the guard while capturing / replaying, the capture
     helper (and what a capture forbids), the 032 embedding check, the replay order and medians.
  C7 (rev 2) E5 timed inside CUDA graphs (testlib: per-rank streams, capture / replay, a virtual GPU clock with a cost
     model per replay): the rev-1 eager numbers (0.437 vs 0.295 ms) with graphs faster -> eh ACTIVE; eager faster but
     graphs slower -> eh refused; slower on one rank only; 0.94x (outside the 0.9 bound); a 3 ms host-callback round
     trip (the leading one-shot); capture failures (one rank / all ranks / a host sync inside the split path) -> no
     rank replays; a replay that fails to launch -> its one-shots issued eagerly (same schedule); the capture exchange
     failing on every rank; exception texts that quote gate / readiness texts (defanged); eh gate numerics (ACTIVE on
     numerics, no capture, still refused on an E3 failure), ' Numerics ', an invalid gate, a gate that differs on one
     rank (fingerprint); patch 032's replicated embedding (eh refused at the preflight; PARTS=head clean).
  Verification of rev 2 (2026-10-03): the testlib now keeps torch's caching-allocator capture bookkeeping (an
     invalidated / failed-to-begin capture leaves its pool registered, empty_cache() then asserts) and every C
     scenario asserts that vLLM's next empty_cache() works on every rank after prepare(); B10 adds both torch
     orderings, the heal as a no-op, a failed capture begin, the E5 decision on exact clock values (inclusive 0.9
     bound, outliers vs the median, zero times) and _safe against EVERY readiness.txt fail / require rule; C7 adds the
     other ordering, no double release, a rank whose G_split records fewer one-shots than budgeted, replay-time
     outliers on one rank, a split at 0.88x.
  D  real-size numerics (one 19,360 x 4,096 shard of t-distributed rows, the H-checks of the boot self-test)
     [skipped with --quick].
  E  (--mutants) 45 planted bugs in a copy of the module (rev 1's 25 that still apply + 15 for rev 2 + 5 from its
     verification); sections B/C must FAIL on each.
Exit 0 = ALL PASS.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
SH = PDIR / "029-gb10-draft-compress.sh"
MODULE_PATH = Path(os.environ.get("P029_MODULE_OVERRIDE") or (HERE / "p029_draft_compress.py"))
APPLIER_PATH = HERE / "p029_apply.py"
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import p029_testlib as T  # noqa: E402

REL = {
    "mtp": "models/glm5next/common/mtp.py",
    "spec": "v1/worker/gpu/spec_decode/mtp/speculator.py",
    "eagle": "v1/worker/gpu/spec_decode/eagle/utils.py",
    "dspec": "v1/worker/gpu/spec_decode/speculator.py",
    "lp": "model_executor/layers/logits_processor.py",
    "mu": "model_executor/layers/quantization/utils/marlin_utils.py",
    "mu8": "model_executor/layers/quantization/utils/marlin_utils_fp8.py",
}
PIN_OF = {"mtp": "MTP_ORIG", "spec": "SPEC_ORIG", "eagle": "EAGLE_ORIG", "dspec": "DSPEC_ORIG", "lp": "LP_ORIG",
          "mu": "MU_ORIG", "mu8": "MU8_ORIG"}
SMS = 48                  # num_compute_units() of the stub (GB10: 48 SMs)
MARLIN_BLOCKS_PER_SM = 4  # marlin_utils.MARLIN_MAX_BLOCKS_PER_SM (v0.32.0)
RESULTS = {"pass": 0, "fail": 0, "failed": []}
QUIET = {"on": False}


def check(name, ok, info=""):
    ok = bool(ok)
    if not QUIET["on"] or not ok:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    RESULTS["pass" if ok else "fail"] += 1
    if not ok:
        RESULTS["failed"].append(name)
    return ok


def raises(fn, exc=Exception, match=None):
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return match is None or re.search(match, str(e)) is not None
    return False


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(name):
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", SH.read_text(), re.M)
    if not m:
        raise SystemExit(f"no pin {name} in {SH}")
    return m.group(1)


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# =====================================================================================================================
# stub vLLM namespace (over the fake torch); the REAL files are exec'd into it
# =====================================================================================================================
class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, rec):
        self.records.append((threading.current_thread().name, rec.levelname, rec.getMessage()))


LOGS = _ListHandler()


def _mod(name, pkg=False, **attrs):
    m = types.ModuleType(name)
    if pkg:
        m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    return m


def _exec_real(name, path_or_text, package, filename=None):
    if isinstance(path_or_text, Path):
        src, fn = path_or_text.read_text(), str(path_or_text)
    else:
        src, fn = path_or_text, filename or name
    m = types.ModuleType(name)
    m.__file__ = fn
    m.__package__ = package
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    exec(compile(src, fn, "exec"), m.__dict__)
    return m


TLS = threading.local()


def build_ns(vroot: Path, mtp_text: str, spec_text: str, module_path: Path | None, env: dict | None = None,
             head_dtype_name="bfloat16"):
    """Fresh fake torch + stub vLLM + the real files. env is applied to os.environ BEFORE the real files are exec'd
    (the mtp.py gate is read at import). v0.32.0: the REAL marlin_utils.py (incl. its torch.ops.vllm.marlin_gemm custom-op
    wrapper _marlin_gemm), marlin_utils_fp8.py, logits_processor.py, eagle/utils.py, glm5next/common/mtp.py and
    mtp/speculator.py run over the stubs below; only torch.ops._C.* (the CUDA kernels) is faked."""
    for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.") or k == "torch" or k.startswith("torch.")]:
        del sys.modules[k]
    for k in ("VLLM_GB10_DRAFT_COMPRESS", "VLLM_GB10_DRAFT_COMPRESS_PARTS", "VLLM_BATCH_INVARIANT",
              "VLLM_GB10_DRAFT_COMPRESS_EH_GATE", "VLLM_GB10_REPL_EMBED"):
        os.environ.pop(k, None)
    for k, v in (env or {}).items():
        os.environ[k] = v
    tm = T.make_fake_torch()
    T.install_fake_marlin_ops_v032(tm)
    T.GRAPH_COST_FN = graph_cost
    sys.modules["torch"] = tm
    for sub in ("nn", "nn.functional", "cuda", "distributed"):
        obj = tm
        for part in sub.split("."):
            obj = getattr(obj, part)
        sys.modules["torch." + sub] = obj
    torch = tm
    bf16 = torch.bfloat16
    lg = logging.getLogger("p029test")
    lg.setLevel(logging.INFO)
    if LOGS not in lg.handlers:
        lg.addHandler(LOGS)
    lg.propagate = False

    _mod("vllm", pkg=True)

    class _VLogger:
        """vLLM's logger surface (info / warning / debug + the *_once variants)."""

        def __init__(self, base):
            self._b, self._seen = base, set()

        def __getattr__(self, k):
            if k.endswith("_once"):
                lvl = k[:-5]

                def once(msg, *a, **kw):
                    key = (lvl, msg, a)
                    if key not in self._seen:
                        self._seen.add(key)
                        getattr(self._b, lvl)(msg, *a)
                return once
            return getattr(self._b, k)
    vlg = _VLogger(lg)
    _mod("vllm.logger", init_logger=lambda name: vlg)

    class VllmConfig:
        pass

    cur_cfg = NS(model_config=NS(head_dtype=getattr(torch, head_dtype_name)))

    def replace(obj, **kw):
        o = NS(**vars(obj))
        for k, v in kw.items():
            setattr(o, k, v)
        return o
    _mod("vllm.config", VllmConfig=VllmConfig, get_current_vllm_config=lambda: cur_cfg, replace=replace)

    def tp_all_gather(t, dim=-1):
        return T.current_world().all_gather_tensor(t, dim)

    def tp_gather(t, dst=0, dim=-1):
        raise RuntimeError("tensor_model_parallel_gather must not be used (use_all_gather is True)")
    _mod("vllm.distributed", pkg=True, tensor_model_parallel_all_gather=tp_all_gather,
         tensor_model_parallel_gather=tp_gather)
    _mod("vllm.distributed.parallel_state", get_pp_group=lambda: NS(world_size=1),
         get_tp_group=lambda: (_ for _ in ()).throw(RuntimeError("get_tp_group is not used in these tests")))
    _mod("vllm.distributed.utils", verify_group_size_divides_partition=lambda *a, **k: None)
    plat = NS(use_all_gather=lambda: True, device_type="cuda", has_device_capability=lambda c: True,
              is_cuda=lambda: True, is_rocm=lambda: False, is_cpu=lambda: False, is_cuda_alike=lambda: True,
              is_xpu=lambda: False, is_device_capability=lambda c: False, is_device_capability_family=lambda c: True)
    _mod("vllm.platforms", current_platform=plat)
    _mod("vllm.envs", VLLM_MARLIN_USE_ATOMIC_ADD=False, VLLM_MARLIN_INPUT_DTYPE=None, VLLM_BATCH_INVARIANT=False)

    class ScalarType:
        def __init__(self, name):
            self.name = name
            self.id = T.SCALAR_IDS[name]        # v0.32.0: the custom op takes ScalarType.id

        def __repr__(self):
            return self.name
    _mod("vllm.scalar_type", ScalarType=ScalarType,
         scalar_types=NS(float8_e4m3fn=ScalarType("float8_e4m3fn"), uint4b8=ScalarType("uint4b8"),
                         uint8b128=ScalarType("uint8b128"), float4_e2m1f=ScalarType("float4_e2m1f")))
    _mod("vllm.utils", pkg=True)
    _mod("vllm.utils.math_utils", round_up=lambda x, m: (x + m - 1) // m * m)
    _mod("vllm.utils.platform_utils", num_compute_units=lambda idx=None: SMS)

    def direct_register_custom_op(op_name, op_func, mutates_args=None, fake_impl=None, target_lib=None,
                                  dispatch_key=None, tags=()):
        """torch.library registration: torch.ops.vllm.<op_name>(...) runs op_func (marlin_utils._marlin_gemm)."""
        setattr(torch.ops.vllm, op_name, op_func)
    _mod("vllm.utils.torch_utils", direct_register_custom_op=direct_register_custom_op,
         current_stream=lambda: torch.cuda.current_stream())
    _mod("vllm.utils.flashinfer", has_flashinfer=lambda: False)

    def _cops_repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit=False):
        return torch.ops._C.gptq_marlin_repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit)

    def _cops_gemm(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, workspace, b_q_type, size_m,
                   size_n, size_k, use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False):
        return torch.ops._C.marlin_gemm(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, workspace,
                                        b_q_type.id, size_m, size_n, size_k, use_atomic_add, use_fp32_reduce,
                                        is_zp_float)
    _mod("vllm._custom_ops", marlin_gemm=_cops_gemm, gptq_marlin_repack=_cops_repack)
    _mod("vllm.model_executor", pkg=True)

    def replace_parameter(layer, name, new, prefer_copy=False):
        setattr(layer, name, torch.nn.Parameter(new, requires_grad=False))
    _mod("vllm.model_executor.utils", replace_parameter=replace_parameter)

    class PluggableLayer(torch.nn.Module):
        @classmethod
        def register(cls, name):
            def deco(c):
                c.name = name
                return c
            return deco
    _mod("vllm.model_executor.custom_op", PluggableLayer=PluggableLayer)
    _mod("vllm.model_executor.layers", pkg=True)

    class UnquantizedEmbeddingMethod:
        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    class VocabParallelEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, prefix="", tp_rank=0, tp_size=1, **kw):
            super().__init__()
            self.org_vocab_size = num_embeddings
            self.tp_size, self.tp_rank = tp_size, tp_rank
            self.num_embeddings_per_partition = num_embeddings // tp_size
            self.weight = torch.nn.Parameter(torch.zeros(self.num_embeddings_per_partition, embedding_dim,
                                                         dtype=bf16, device="cuda:0"), requires_grad=False)
            self.quant_method = UnquantizedEmbeddingMethod()

        def forward(self, ids):
            return self.weight[ids._a]

    class ParallelLMHead(VocabParallelEmbedding):
        def __init__(self, *a, quant_config=None, prefix="", bias=False, **kw):
            super().__init__(*a, prefix=prefix, **kw)
            self.bias = None
    _mod("vllm.model_executor.layers.vocab_parallel_embedding", UnquantizedEmbeddingMethod=UnquantizedEmbeddingMethod,
         VocabParallelEmbedding=VocabParallelEmbedding, ParallelLMHead=ParallelLMHead)

    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden, eps=1e-5):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden, dtype=bf16, device="cuda:0"), requires_grad=False)
            self.variance_epsilon = eps

        def _n(self, x):
            xf = x.float()
            return (xf / ((xf * xf).mean(dim=-1, keepdim=True) + self.variance_epsilon).sqrt()
                    * self.weight.float()).to(bf16)

        def forward(self, x, residual=None):
            if residual is None:
                return self._n(x)
            s = (x.float() + residual.float()).to(bf16)
            return self._n(s), s
    _mod("vllm.model_executor.layers.layernorm", RMSNorm=RMSNorm)

    class RoutedExperts:
        pass

    _mod("vllm.model_executor.layers.fused_moe", pkg=True, RoutedExperts=RoutedExperts,
         fused_moe_make_expert_params_mapping=lambda *a, **k: [])
    _mod("vllm.model_executor.layers.fused_moe.config", FusedMoEConfig=type("FusedMoEConfig", (), {}))
    _mod("vllm.model_executor.layers.fused_moe.utils",
         is_model_fused_shared_expert_compatible=lambda layers, moe_cls, attr: False)

    class UnquantizedLinearMethod:
        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)
    _mod("vllm.model_executor.layers.linear", LinearBase=type("LinearBase", (torch.nn.Module,), {}),
         UnquantizedLinearMethod=UnquantizedLinearMethod)
    _mod("vllm.model_executor.layers.quantization", pkg=True)
    _mod("vllm.model_executor.layers.quantization.input_quant_fp8", QuantFP8=type("QuantFP8", (), {}))
    _mod("vllm.model_executor.layers.quantization.utils", pkg=True)
    _mod("vllm.model_executor.layers.quantization.utils.int8_utils", per_token_quant_int8=lambda x: None)
    _mod("vllm.model_executor.layers.quantization.utils.quant_utils", GroupShape=NS(PER_TOKEN="per_token"),
         pack_cols=lambda *a: None, unpack_cols=lambda *a: None)
    mu = _exec_real("vllm.model_executor.layers.quantization.utils.marlin_utils", vroot / REL["mu"],
                    "vllm.model_executor.layers.quantization.utils")
    mu8 = _exec_real("vllm.model_executor.layers.quantization.utils.marlin_utils_fp8", vroot / REL["mu8"],
                     "vllm.model_executor.layers.quantization.utils")
    lpm = _exec_real("vllm.model_executor.layers.logits_processor", vroot / REL["lp"], "vllm.model_executor.layers")

    _mod("vllm.model_executor.model_loader", pkg=True,
         get_model=lambda vllm_config=None, model_config=None: TLS.draft_model)
    _mod("vllm.model_executor.model_loader.utils", get_draft_load_config=lambda vc: vc.load_config)
    _mod("vllm.model_executor.model_loader.weight_utils", default_weight_loader=lambda p, w: None,
         maybe_remap_kv_scale_name=lambda n, p: n)
    _mod("vllm.model_executor.models", pkg=True)

    class SharedHead(torch.nn.Module):
        """v0.32.0 deepseek_mtp.SharedHead: defer_lm_head=True -> head is None until load_eagle_model shares the target's."""

        def __init__(self, config, prefix="", quant_config=None, *, defer_lm_head=False):
            super().__init__()
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = None if defer_lm_head else ParallelLMHead(config.vocab_size, config.hidden_size,
                                                                  prefix=prefix + ".head")
    _mod("vllm.model_executor.models.deepseek_mtp", SharedHead=SharedHead)

    class DeepseekV2MixtureOfExperts:
        def extract_moe_parameters(self, example):
            self.example_moe = example
    _mod("vllm.model_executor.models.deepseek_v2", DeepseekV2MixtureOfExperts=DeepseekV2MixtureOfExperts)
    _mod("vllm.model_executor.models.utils", maybe_prefix=lambda p, n: f"{p}.{n}" if p else n,
         PPMissingLayer=type("PPMissingLayer", (), {}))
    _mod("vllm.sequence", IntermediateTensors=dict)
    _mod("vllm.models", pkg=True)
    _mod("vllm.models.glm5next", pkg=True)
    _mod("vllm.models.glm5next.nvidia", pkg=True)
    _mod("vllm.models.glm5next.common", pkg=True)

    class Glm5NextMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = NS(name="experts")

    class Glm5NextMLAAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mla_attn = NS(skip_topk=False, topk_indices_buffer=None)

    class Glm5NextDecoderLayer(torch.nn.Module):
        def __init__(self, vllm_config=None, config=None, layer_idx=0, prefix="", topk_indices_buffer=None,
                     is_mtp_layer=False):
            super().__init__()
            self.mlp = Glm5NextMoE()
            self.self_attn = Glm5NextMLAAttention()
            self.captured = []

        def forward(self, positions, hidden_states, residual=None):
            self.captured.append(hidden_states)
            return (hidden_states * 0.5), hidden_states, None, None
    _mod("vllm.models.glm5next.common.model", Glm5NextDecoderLayer=Glm5NextDecoderLayer, Glm5NextMoE=Glm5NextMoE,
         Glm5NextMLAAttention=Glm5NextMLAAttention, _fused_shared_expert_name=lambda n, e: n,
         _num_fused_shared_experts=lambda n, enabled: 0,
         _try_load_fp8_attn_proj=lambda *a, **k: False, _try_load_fp8_indexer_wk=lambda *a, **k: False,
         get_spec_layer_idx_from_weight_name=lambda c, n: None)
    _mod("vllm.models.glm5next.nvidia.ops", pkg=True)

    def fused_eh_norm(positions, inputs_embeds, previous_hidden, enorm_w, hnorm_w, eps):
        e = inputs_embeds.float()
        mask = (positions == 0)
        e = torch.where(mask.unsqueeze(1), torch.zeros_like(e), e) if bool(mask.any()) else e

        def rms(x, w):
            return x / ((x * x).mean(dim=-1, keepdim=True) + eps).sqrt() * w.float()
        return torch.cat([rms(e, enorm_w), rms(previous_hidden.float(), hnorm_w)], dim=1).to(bf16)
    _mod("vllm.models.glm5next.nvidia.ops.fused_eh_norm", fused_eh_norm=fused_eh_norm)
    _mod("vllm.lora", pkg=True)
    _mod("vllm.lora.layers", pkg=True)
    _mod("vllm.lora.layers.base", BaseLayerWithLoRA=type("BaseLayerWithLoRA", (), {}))
    _mod("vllm.compilation", pkg=True)
    _mod("vllm.compilation.backends", set_model_tag=lambda tag: contextlib.nullcontext())
    for p in ("vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.spec_decode",
              "vllm.v1.worker.gpu.spec_decode.target_dependent_ar", "vllm.v1.worker.gpu.spec_decode.eagle",
              "vllm.v1.worker.gpu.spec_decode.mtp"):
        _mod(p, pkg=True)
    _mod("vllm.v1.worker.gpu.spec_decode.utils", get_pp_safe_draft_load_config=lambda lc: lc)

    class TargetDependentARSpeculator:
        pass
    _mod("vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator",
         TargetDependentARSpeculator=TargetDependentARSpeculator)
    eagle = _exec_real("vllm.v1.worker.gpu.spec_decode.eagle.utils", vroot / REL["eagle"],
                       "vllm.v1.worker.gpu.spec_decode.eagle")
    mod = None
    if module_path is not None:
        mod = _exec_real("vllm._gb10_draft_compress", Path(module_path), "vllm")
        mod.__file__ = str(module_path)
        mod._WORLD_FACTORY = T.current_world
    mtp = _exec_real("vllm.models.glm5next.common.mtp", mtp_text, "vllm.models.glm5next.common",
                     str(vroot / REL["mtp"]))
    spec = _exec_real("vllm.v1.worker.gpu.spec_decode.mtp.speculator", spec_text,
                      "vllm.v1.worker.gpu.spec_decode.mtp", str(vroot / REL["spec"]))
    return NS(torch=torch, mu=mu, mu8=mu8, lp=lpm, eagle=eagle, mtp=mtp, spec=spec, mod=mod,
              ParallelLMHead=ParallelLMHead, VocabParallelEmbedding=VocabParallelEmbedding, cur_cfg=cur_cfg)


# =====================================================================================================================
# model construction per rank (small dims: H 256, eh [256, 512], vocab 1,600 = 8 x 200; Marlin pads 200 -> 256)
# =====================================================================================================================
DIMS = NS(H=256, V=1600, W=8, ONESHOT_MAX=2048)


def global_weights(seed=7, H=None, V=None):
    H = H or DIMS.H
    V = V or DIMS.V
    rng = np.random.default_rng(seed)
    row_sd = 0.02 * np.exp(rng.normal(0, 0.3, size=(V, 1)))
    lm = (rng.standard_t(6, size=(V, H)) * row_sd / np.sqrt(1.5)).astype(np.float32)
    eh = (rng.standard_normal((H, 2 * H)) * 0.03).astype(np.float32)
    emb = (rng.standard_normal((V, H)) * 0.5).astype(np.float32)
    norms = [np.exp(rng.normal(0, 0.2, size=H)).astype(np.float32) for _ in range(3)]
    return NS(lm=lm, eh=eh, emb=emb, enorm=norms[0], hnorm=norms[1], hnorm_head=norms[2])


def make_cfg(ns, H, V):
    hf = NS(hidden_size=H, rms_norm_eps=1e-5, vocab_size=V, index_topk=8, index_kpool=1, num_hidden_layers=3,
            num_nextn_predict_layers=1, n_group=1, n_routed_experts=4, n_shared_experts=1, mla_use_nope=False,
            qk_rope_head_dim=0, model_type="glm5_next_mtp")
    # v0.32.0: hf_text_config (transformers' Glm5NextTextConfig), apply_draft_overrides, parallel_config / load_config
    return NS(speculative_config=NS(draft_model_config=NS(hf_config=hf, hf_text_config=hf), method="mtp",
                                    apply_draft_overrides=lambda vc: vc),
              model_config=NS(hf_config=hf, hf_text_config=hf, head_dtype=ns.cur_cfg.model_config.head_dtype),
              quant_config=None, scheduler_config=NS(max_num_batched_tokens=16),
              parallel_config=NS(prefill_context_parallel_size=1), load_config=NS(load_format="auto"))


def build_rank(ns, rank, G, W=None, draft_cls=None):
    """-> (speculator, draft model, target namespace). The draft model is built by the REAL mtp.py (its SharedHead has
    no head yet: defer_lm_head=True); the target's lm_head shard (rank's rows of G.lm) is shared into it by the REAL
    load_eagle_model."""
    torch = ns.torch
    W = W or DIMS.W
    V, H = G.lm.shape
    torch.set_default_dtype(torch.bfloat16)
    cfg = make_cfg(ns, H, V)
    draft = (draft_cls or ns.mtp.Glm5NextMTP)(vllm_config=cfg, prefix="")
    ly = draft.model._mtp_layers[0]
    ly.eh_proj.weight.copy_(torch.tensor(G.eh))
    ly.enorm.weight.copy_(torch.tensor(G.enorm))
    ly.hnorm.weight.copy_(torch.tensor(G.hnorm))
    ly.shared_head.norm.weight.copy_(torch.tensor(G.hnorm_head))
    vp = V // W
    head = ns.ParallelLMHead(V, H, tp_rank=rank, tp_size=W)
    head.weight.copy_(torch.tensor(G.lm[rank * vp:(rank + 1) * vp]))
    emb = ns.VocabParallelEmbedding(V, H, tp_rank=rank, tp_size=W)    # the base's vocab-parallel embed_tokens
    emb.weight = torch.nn.Parameter(torch.tensor(G.emb).to(torch.bfloat16).to("cuda:0"), requires_grad=False)
    target = NS(lm_head=head, model=NS(embed_tokens=emb))
    TLS.draft_model = draft
    sp = object.__new__(ns.spec.MTPSpeculator)
    sp.vllm_config = cfg
    return sp, draft, target


DEFAULT_COSTS = {"bf16": 1.0, "fp8": 0.4, "split": 0.1, "full": 0.3}
# rev 2: what one replay of a captured E5 graph costs on the fake GPU (ms, real-size equivalents: microbenchmark
# eh_proj M=3 291 us, 1/8 row block 40 us + pad 2.4 us, one-shot ~47 us; NCCL's per-replay host-callback round trip
# 0.5-0.9 ms (profile) is paid by the FIRST one-shot of a replay). Per-rank overrides: world.graph_cost.
GRAPH_COSTS = {"rtt": 0.6, "oneshot": 0.047, "full": 0.291, "block": 0.040, "pad": 0.0024}


def graph_cost(ops, world) -> float:
    c = dict(GRAPH_COSTS)
    c.update(getattr(world, "graph_cost", None) or {})
    t, first = 0.0, True
    for op in ops:
        if op[0] == "oneshot":
            t += c["oneshot"] + (c["rtt"] if first else 0.0)
            first = False
        elif op[0] == "linear":
            rows = int(op[1][0])
            t += c["full"] if rows == DIMS.H else (c["block"] if rows == DIMS.H // DIMS.W else 0.0)
        elif op[0] == "pad":
            t += c["pad"]
    seq = getattr(world, "graph_cost_seq", None)     # (verification) per-replay noise: seq(kind, n, ms) -> ms
    if seq is not None:
        kind = "split" if sum(1 for op in ops if op[0] == "oneshot") > 1 else "full"
        cnt = world.__dict__.setdefault("_graph_replay_n", {})
        n = cnt.get(kind, 0)
        cnt[kind] = n + 1
        t = float(seq(kind, n, t))
    return t


def _timed_kind(fn) -> str:
    """Which path the boot self-test is timing (from the lambda's code): the image BF16 head, the FP8 head, the eh
    split (collective) or the replicated eh_proj."""
    names, free = fn.__code__.co_names, fn.__code__.co_freevars
    if "_bf16_head_apply" in names:
        return "bf16"
    if "eh_proj" in names:
        return "full"
    if "fh" in free:
        return "fp8"
    if "sp" in free:
        return "split"
    return "other"


def fake_timer_factory(costs):
    """Calls fn warmup + iters times (so collective call counts are real) and returns a per-rank modelled cost."""
    def timer(fn, warmup, iters):
        for _ in range(warmup + iters):
            fn()
        w = T.current_world()
        table = dict(DEFAULT_COSTS)
        table.update((costs or {}).get(w.rank if w is not None else 0, {}))
        return table.get(_timed_kind(fn), 1.0)
    return timer


def run_boot(ns, G, *, worlds_cfg=None, times=None, rv_timeout=4.0, inject=None, draft_cls=None):
    """Boot the speculator on 8 threads; returns dict with per-rank summaries, models, rendezvous, errors."""
    W = DIMS.W
    rv = T.Rendezvous(W, timeout=rv_timeout)
    ns.mod._TIMER = fake_timer_factory(times)
    LOGS.records.clear()
    out = {}

    def body(r):
        cfgw = (worlds_cfg or {}).get(r, {})
        w = T.ThreadWorld(rv, r, W, oneshot_max=cfgw.get("oneshot_max", DIMS.ONESHOT_MAX))
        for k, v in cfgw.items():
            if k != "oneshot_max":
                setattr(w, k, v)
        T.set_current_world(w)
        sp, draft, target = build_rank(ns, r, G, draft_cls=draft_cls)
        if inject:
            inject(r, ns, draft, target, w)
        ret = sp.load_draft_model(target, set())
        out[r] = NS(sp=sp, draft=draft, target=target, world=w, ret=ret)
        stream_after[r] = ns.torch.cuda.current_stream().id
        capture_after[r] = ns.torch.cuda.is_current_stream_capturing()
        # (verification) what vLLM does next (DeviceMemoryProfiler.__exit__ -> torch.cuda.empty_cache()) must work:
        # no capture left registered with the caching allocator, no pool use count below zero. The testlib's own
        # empty_cache(), not ns.torch.cuda.empty_cache: C6 replaces that one to sample the FP8 copies at prepare()'s call
        try:
            T.empty_cache()
            alloc_after[r] = T.allocator_clean()
        except Exception:  # noqa: BLE001
            alloc_after[r] = False
        return None

    stream_after, capture_after, alloc_after = {}, {}, {}
    res, err = T.run_ranks(W, body, timeout=120)
    return NS(ranks=out, rv=rv, err=err, stream_after=stream_after, capture_after=capture_after,
              alloc_after=alloc_after)


def mu_marlin_gemm_impl(ns):
    """The function direct_register_custom_op bound to torch.ops.vllm.marlin_gemm: the REAL marlin_utils._marlin_gemm."""
    return ns.mu._marlin_gemm


def state_of(rb):
    """per rank: (head active?, eh active?)"""
    st = {}
    for r, o in rb.ranks.items():
        pred = o.draft.model
        st[r] = (getattr(pred, "_p029_head", None) is not None,
                 all(getattr(ly, "_p029_eh", None) is not None for ly in pred._mtp_layers))
    return st


def consistent(rb):
    st = state_of(rb)
    return len(set(st.values())) == 1 and len(st) == DIMS.W, st


def schedules_equal(rb):
    s = rb.rv.schedules
    return all(x == s[0] for x in s), [len(x) for x in s]


def errs(rb):
    return [(r, e[0]) for r, e in enumerate(rb.err) if e is not None]


def _segments(s):
    """Collective-schedule landmarks (rev 2): one-shots before the graph-capture exchange (E1-E4 + eager E5), one-shots
    between it and the verdict exchange (graph replays), exchanges from the verdict on."""
    idx = [i for i, e in enumerate(s) if e[0] == "all_gather_object"]
    cap, ver = (idx[1], idx[2]) if len(idx) >= 3 else (None, None)
    pre = s[:cap] if cap is not None else s
    mid = s[cap + 1:ver] if cap is not None else []
    last_os = max((i for i, e in enumerate(s) if e[0] == "oneshot"), default=-1)
    first_ag = min((i for i, e in enumerate(s) if e[0] == "all_gather"), default=len(s))
    return {"pre_cap_os": sum(1 for e in pre if e[0] == "oneshot"),
            "cap_ver_os": sum(1 for e in mid if e[0] == "oneshot"),
            "head_after_eh": first_ag > last_os,
            "post_ver": (len(idx) - 2) if ver is not None else 0, "exchanges": len(idx)}


def without_replays(s):
    """The clean schedule minus the graph replays (every rank decided not to replay after the capture exchange)."""
    idx = [i for i, e in enumerate(s) if e[0] == "all_gather_object"]
    if len(idx) < 3:
        return list(s)
    a, b = idx[1], idx[2]
    return list(s[:a + 1]) + [e for e in s[a + 1:b] if e[0] != "oneshot"] + list(s[b:])


def without_capture_exchange(s):
    """The clean schedule minus the capture exchange (its barrier failed before issuing) and the replays."""
    w = without_replays(s)
    idx = [i for i, e in enumerate(w) if e[0] == "all_gather_object"]
    a = idx[1]
    return w[:a - 1] + w[a + 1:]          # drop ("barrier",), ("all_gather_object",) of the capture exchange


# the gate texts other scripts count (the boot-gate scripts; launch/readiness.txt fail rules): rev 2 log lines must
# never reproduce one of them by quoting
NIGHT_ACTIVE = re.compile(r"\[patch-029\] draft-path compression ACTIVE \(rank (\d+)/")
NIGHT_BAD = re.compile(r"\[patch-029\].*(INACTIVE|not active|exchange failed|unexpected)|\[029\] FATAL")
_READINESS_FALLBACK = [r"FilteredTopK fallback requires", r"\[recipe-entrypoint\] FATAL", r"pe_dim must be 64 for fp8_ds_mla",
                       r"Unknown SF transformation", r"invalid choice",
                       r"Free memory on device .* less than desired GPU memory utilization",
                       r"Engine core initialization failed", r"died unexpectedly", r"EngineDeadError",
                       r"no kernel image", r"cudaFuncSetAttribute.*invalid argument", r"OutOfMemoryError",
                       r"NV_ERR_NO_MEMORY", r"NCCL (WARN|error)", r"Too many open files",
                       r"Using FLASHINFER_MLA_SPARSE_SM120 attention backend"]


def readiness_path() -> Path:
    """$PATCH_READINESS_TXT, else the recipe's launch/readiness.txt next to the patches dir, else the shipped test copy."""
    cands = [os.environ.get("PATCH_READINESS_TXT") or "", str(PDIR.parent / "launch" / "readiness.txt"),
             str(HERE / "testdata" / "readiness.txt")]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return Path(cands[-1])


def readiness_fail_patterns():
    p = readiness_path()
    pats = []
    try:
        for line in p.read_text().splitlines():
            s = line.strip()
            if s.startswith("fail") and len(s.split(None, 1)) == 2:
                pats.append(s.split(None, 1)[1])
    except OSError:
        pass
    return [re.compile(x) for x in (pats or _READINESS_FALLBACK)]


READINESS_FAIL = readiness_fail_patterns()


def readiness_rules(kinds=("fail", "require")):
    """(verification) every fail AND require rule of launch/readiness.txt, as read now."""
    out = []
    try:
        for line in readiness_path().read_text().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0] in kinds:
                out.append((parts[0], re.compile(parts[1])))
    except OSError:
        pass
    return out


READINESS_GATES = readiness_rules()
# one text matching each launch/readiness.txt fail / require rule (B10: every rule read from the file must be matched
# by one of these - a new rule fails B10 until its text is added here - and none may still match after _safe())
GATE_TEXTS = [
    "[recipe-entrypoint] all patches applied", "busy_loop_s = 0.002", "[002] OK persistent_topk gate present",
    "Graph capturing finished in 12 secs", "world_size=8 rank=0 local_rank=0 distributed_init_method=env:// backend=nccl",
    "Using FLASHINFER_MLA_SPARSE_SM90 attention backend", "Using TRITON Fp8 MoE backend", "Model loading took 39.31 GiB",
    "Setting attention block size to 1280 tokens", "reserved 8.0 GiB memory for KV Cache",
    "GPU KV cache size: 513,263 tokens", "init engine (profile, create kv cache, warmup model) took 120.5 seconds",
    "Starting vLLM server on http://0.0.0.0:8000", "Application startup complete",
    "FilteredTopK fallback requires k <= 2048", "[recipe-entrypoint] FATAL: 029-gb10-draft-compress.sh failed",
    "pe_dim must be 64 for fp8_ds_mla", "Unknown SF transformation", "error: argument x: invalid choice: 'y'",
    "Free memory on device cuda:0 (1.2/119.7 GiB) on startup is less than desired GPU memory utilization",
    "Engine core initialization failed", "Worker proc died unexpectedly", "vllm.v1.engine.exceptions.EngineDeadError",
    "no kernel image is available for execution on the device",
    "cudaFuncSetAttribute(kernel, attr, 1024) failed: invalid argument", "torch.OutOfMemoryError: CUDA out of memory",
    "NV_ERR_NO_MEMORY", "NCCL WARN Cuda failure", "NCCL error: unhandled cuda error", "Too many open files",
    "Using FLASHINFER_MLA_SPARSE_SM120 attention backend",
    # the v0.32.0 recipe's readiness file adds these rules
    "Initial free memory 118.1 GiB, reserved 16.0 GiB memory for KV Cache as specified by --kv-cache-memory-bytes",
    "Using fp8_ds_mla KV cache format for FLASHINFER_MLA_SPARSE_SM120 backend",
    "Failed to initialize FlashInfer Allreduce: unsupported topology",
    "Unknown vLLM environment variable detected: VLLM_EAGLE_KEEP_LAST_BLOCK",
]


def gate_counts(records):
    msgs = [m for (_, _, m) in records]
    ranks = sorted(int(mo.group(1)) for m in msgs for mo in [NIGHT_ACTIVE.search(m)] if mo)
    bad = sum(1 for m in msgs if NIGHT_BAD.search(m))
    rf = [(pt.pattern, m[:90]) for m in msgs if "[patch-029]" in m for pt in READINESS_FAIL if pt.search(m)]
    act_inactive = sum(1 for m in msgs if NIGHT_ACTIVE.search(m) and "INACTIVE" in m)
    return {"active_ranks": ranks, "bad": bad, "readiness_fail": rf, "active_quoting_inactive": act_inactive}


def run_ranks_fn(ns, rb, fn):
    """Run fn(rank, o) on all 8 rank threads with each rank's world (post-boot runtime calls)."""
    W = DIMS.W
    rv = T.Rendezvous(W, timeout=4.0)
    for o in rb.ranks.values():
        o.world.rv, o.world.seq = rv, 0

    def body(r):
        T.set_current_world(rb.ranks[r].world)
        return fn(r, rb.ranks[r])
    res, err = T.run_ranks(W, body, timeout=60)
    return res, err, rv


# =====================================================================================================================
# A: applier + pins
# =====================================================================================================================
def section_a(vroot: Path, work: Path, ap):
    print("== A: applier + pins on the v0.32.0 source")
    texts = {}
    for k, rel in REL.items():
        p = vroot / rel
        ok = p.is_file() and sha_file(p) == pin(PIN_OF[k])
        check(f"A1 {rel} == pinned v0.32.0 original {PIN_OF[k]}", ok, sha_file(p)[:16] if p.is_file() else "missing")
        texts[k] = p.read_text()
    for kind, pin_res in (("mtp", "MTP_P029"), ("spec", "SPEC_P029")):
        base = texts[kind]
        out = ap.apply(kind, base)
        check(f"A2 apply({kind}) == pinned {pin_res}", sha_text(out) == pin(pin_res), sha_text(out)[:16])
        check(f"A2 revert(apply({kind})) == v0.32.0 original byte-exact", ap.revert(kind, out) == base)
        check(f"A2 {kind}: patched structure valid", not raises(lambda: ap.check_structure(kind, out, True)))
        check(f"A3 {kind}: apply on a patched file refused", raises(lambda: ap.apply(kind, out), ap.Refused))
        check(f"A3 {kind}: revert of the base refused", raises(lambda: ap.revert(kind, base), ap.Refused))
        # CLI on a temp copy
        d = work / f"cli_{kind}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / Path(REL[kind]).name
        f.write_text(base)
        os.chmod(f, 0o640)
        r = subprocess.run([sys.executable, str(APPLIER_PATH), "check", kind, str(f)], capture_output=True, text=True)
        check(f"A2 CLI check {kind} prints the pinned result sha", r.returncode == 0 and r.stdout.strip() == pin(pin_res))
        r = subprocess.run([sys.executable, str(APPLIER_PATH), "apply", kind, str(f)], capture_output=True, text=True)
        check(f"A2 CLI apply {kind}: result sha pinned, mode kept",
              r.returncode == 0 and sha_file(f) == pin(pin_res) and (os.stat(f).st_mode & 0o777) == 0o640)
        r = subprocess.run([sys.executable, str(APPLIER_PATH), "apply", kind, str(f)], capture_output=True, text=True)
        check(f"A3 CLI apply {kind} twice -> exit 1, file unchanged", r.returncode == 1 and sha_file(f) == pin(pin_res))
        r = subprocess.run([sys.executable, str(APPLIER_PATH), "revert", kind, str(f)], capture_output=True, text=True)
        check(f"A2 CLI revert {kind} -> v0.32.0 original", r.returncode == 0 and f.read_text() == base)
        r = subprocess.run([sys.executable, str(APPLIER_PATH), "apply", "nope", str(f)], capture_output=True, text=True)
        check(f"A3 CLI unknown kind -> exit 2", r.returncode == 2)
        check(f"A3 {kind}: wrong kind refused", raises(lambda: ap.apply("spec" if kind == "mtp" else "mtp", base)))
    m = texts["mtp"]
    check("A3 mtp: eh anchor missing refused",
          raises(lambda: ap.apply("mtp", m.replace("self.eh_proj(eh_input)", "self.eh_proj( eh_input)")), ap.Refused))
    check("A3 mtp: eh anchor duplicated refused",
          raises(lambda: ap.apply("mtp", m + "\n# hidden_states = self.eh_proj(eh_input)\n"
                                  "        hidden_states = self.eh_proj(eh_input)\n"), ap.Refused))
    check("A3 mtp: logits anchor missing refused",
          raises(lambda: ap.apply("mtp", m.replace("return self.logits_processor(mtp_layer.shared_head.head",
                                                   "return self.logits_processor(  mtp_layer.shared_head.head")),
                 ap.Refused))
    check("A3 mtp: (v0.32.0) module-gate anchor (end of the `from .model import (...)` block + the class line) missing "
          "refused", raises(lambda: ap.apply("mtp", m.replace("class Glm5NextMultiTokenPredictorLayer(nn.Module):",
                                                              "class Glm5NextMultiTokenPredictorLayer( nn.Module):")),
                            ap.Refused))
    check("A3 mtp: (v0.32.0) module-gate anchor duplicated refused",
          raises(lambda: ap.apply("mtp", m + "\n" + ap.MTP_GATE_OLD), ap.Refused))
    check("A3 mtp: (v0.32.0) the day-0 gate anchor (`from .ops.fused_eh_norm import fused_eh_norm`) is not in the v0.32.0 "
          "file, and the new anchor occurs exactly once",
          "from .ops.fused_eh_norm import fused_eh_norm" not in m and m.count(ap.MTP_GATE_OLD) == 1
          and m.count(ap.MTP_EH_OLD) == 1 and m.count(ap.MTP_LOGITS_OLD) == 1 and texts["spec"].count(ap.SPEC_OLD) == 1)
    s = texts["spec"]
    check("A3 spec: anchor missing refused",
          raises(lambda: ap.apply("spec", s.replace("load_eagle_model(target_model, self.vllm_config)",
                                                    "load_eagle_model(target_model, self.vllm_config, )")),
                 ap.Refused))
    # AST placement refusals on hand-made patched texts
    good = ap.apply("mtp", m)
    moved = good.replace("            return _p029.draft_logits(self, mtp_layer, hidden_states)\n",
                         "            return self.logits_processor(mtp_layer.shared_head.head, hidden_states)\n")
    moved = moved.replace("        return self.logits_processor.get_top_tokens(",
                          "        if _P029_ON:\n            return _p029.draft_logits(self, mtp_layer, hidden_states)\n"
                          "        return self.logits_processor.get_top_tokens(")
    check("A3 mtp: draft_logits hook in get_top_tokens (wrong function) refused",
          raises(lambda: ap.check_structure("mtp", moved, True), ap.Refused))
    gs = ap.apply("spec", s)
    bad_spec = gs.replace("            _p029.prepare(draft_model, self.vllm_config)\n", "            pass\n")
    bad_spec = bad_spec.replace("        return draft_model\n",
                                "        _p029.prepare(draft_model, self.vllm_config)\n        return draft_model\n")
    check("A3 spec: prepare hook moved after the topk setup refused",
          raises(lambda: ap.check_structure("spec", bad_spec, True), ap.Refused))
    # gate block alone
    gate_src = ap.MTP_GATE_NEW.split("\n\n\nclass")[0].replace("    get_spec_layer_idx_from_weight_name,\n)\n", "")

    def run_gate(val, with_module):
        g = {}
        saved = {k: sys.modules.get(k) for k in ("vllm", "vllm._gb10_draft_compress")}
        try:
            for k in list(saved):
                sys.modules.pop(k, None)
            vm = types.ModuleType("vllm")
            vm.__path__ = []
            sys.modules["vllm"] = vm
            if with_module:
                mm = types.ModuleType("vllm._gb10_draft_compress")
                sys.modules["vllm._gb10_draft_compress"] = mm
            if val is None:
                os.environ.pop("VLLM_GB10_DRAFT_COMPRESS", None)
            else:
                os.environ["VLLM_GB10_DRAFT_COMPRESS"] = val
            exec(compile(gate_src, "gate", "exec"), g)
            return g.get("_P029_ON"), "_p029" in g
        finally:
            os.environ.pop("VLLM_GB10_DRAFT_COMPRESS", None)
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    check("A4 gate unset -> _P029_ON False, module not imported", run_gate(None, False) == (False, False))
    check("A4 gate 0 -> off", run_gate("0", False) == (False, False))
    check("A4 gate 'true' / '2' -> off (only 1; the .sh refuses other values)",
          run_gate("true", False) == (False, False) and run_gate("2", False) == (False, False))
    check("A4 gate ' 1 ' -> on, module imported", run_gate(" 1 ", True) == (True, True))
    check("A4 gate 1 + module missing -> ImportError (fail closed)",
          raises(lambda: run_gate("1", False), ImportError))
    # A5 earlier patches: none writes mtp.py or mtp/speculator.py, none touches the MTP draft objects at runtime
    targets = ("models/glm5next/common/mtp.py", "models/glm5next/nvidia/mtp.py", "spec_decode/mtp/speculator.py",
               "spec_decode/mtp/", "glm5next/common/mtp", "glm5next/nvidia/mtp")
    symbols = ("MTPSpeculator", "load_draft_model", "Glm5NextMultiTokenPredictor", "eh_proj", "shared_head",
               "load_eagle_model", "Glm5NextMTP")
    test_like = ("cpu_check", "testlib", "sh_check", "e2e_check", "gpu_selftest", "probe", "bench", "test_", "rankcheck")
    hits, scanned = [], []
    earlier = sorted(p for p in PDIR.glob("[0-9][0-9][0-9]-*.sh") if int(p.name[:3]) < 29)
    def _later(name):   # pNNN_* files of patch 029 and of LATER patches (they run after 029: not a base of it)
        m = re.match(r"p(\d{3})_", name)
        return bool(m) and int(m.group(1)) >= 29
    files = list(earlier) + sorted(p for p in (HERE).glob("*.py") if not _later(p.name)
                                   and not any(x in p.name for x in test_like))
    files += sorted(p for p in HERE.glob("*/*.py") if not any(x in p.name for x in test_like))
    for p in files:
        scanned.append(p.name)
        for line in p.read_text(errors="replace").splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            if any(x in s for x in targets) or any(re.search(rf"\b{x}\b", s) for x in symbols):
                hits.append((p.name, s[:120]))
    check(f"A5 no earlier patch ({len(earlier)} .sh {earlier[0].name[:3]}..{earlier[-1].name[:3]}, "
          f"{len(files) - len(earlier)} vendored runtime/applier files) references mtp.py / spec_decode/mtp/ or the "
          "MTP draft objects outside comments -> each target has one base, the v0.32.0 original", not hits,
          str(hits[:3]))
    print("     enumerated: " + ", ".join(p.name[:3] for p in earlier))
    # A7 the vendored files the .sh pins (a stale pin after an edit of the applier / module / patch 012's module is caught here)
    check("A7 .sh pins APPLIER_SHA / MOD_SHA == the vendored applier / module (the module under test: "
          f"{MODULE_PATH.name})", sha_file(APPLIER_PATH) == pin("APPLIER_SHA") and sha_file(MODULE_PATH) == pin("MOD_SHA"))
    os_mod = HERE / "gb10_oneshot_ar.py"
    check("A7 .sh pins OS_MOD_SHA == files/gb10_oneshot_ar.py (patch 012's module, which the eh part's one-shot comes from)",
          os_mod.is_file() and sha_file(os_mod) == pin("OS_MOD_SHA"), sha_file(os_mod)[:16] if os_mod.is_file() else "missing")
    # A6 the container runs Python 3.12 (this Mac a newer one): everything shipped must parse at that feature level
    for label, src in (("module", MODULE_PATH.read_text()), ("applier", APPLIER_PATH.read_text()),
                       ("patched mtp.py", ap.apply("mtp", texts["mtp"])),
                       ("patched mtp/speculator.py", ap.apply("spec", texts["spec"]))):
        try:
            ast.parse(src, feature_version=(3, 12))
            ok = True
        except SyntaxError:
            ok = False
        check(f"A6 {label} parses at ast feature_version 3.12", ok)
    return texts


# =====================================================================================================================
# B: module units
# =====================================================================================================================
def section_b(vroot, texts, ap):
    print("== B: module units (fake torch + the REAL marlin_utils(_fp8).py)")
    mtp_p = ap.apply("mtp", texts["mtp"])
    spec_p = ap.apply("spec", texts["spec"])
    ns = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env={"VLLM_GB10_DRAFT_COMPRESS": "1"})
    torch, mod = ns.torch, ns.mod
    check("B1 parse_parts default", mod.parse_parts(None) == ("eh", "head") and mod.parse_parts("") == ("eh", "head"))
    check("B1 parse_parts subsets", mod.parse_parts("head") == ("head",) and mod.parse_parts(" EH , head ") == ("eh", "head"))
    check("B1 parse_parts invalid -> ValueError", raises(lambda: mod.parse_parts("all"), ValueError)
          and raises(lambda: mod.parse_parts("head,x"), ValueError) and raises(lambda: mod.parse_parts(","), ValueError))
    check("B1 gate_on only for 1", mod.gate_on("1") and mod.gate_on(" 1") and not mod.gate_on("0")
          and not mod.gate_on("") and not mod.gate_on("true"))
    # B2 quantisation
    rng = np.random.default_rng(3)
    Wn = (rng.standard_t(5, size=(300, 256)) * 0.02).astype(np.float32)
    Wn[5] = 0.0
    W = torch.tensor(Wn).to(torch.bfloat16).to("cuda:0")
    q, s = mod.quantize_rows_fp8(W, chunk=64)
    q2, s2 = mod.quantize_rows_fp8(W, chunk=4096)
    check("B2 dtypes / shapes", q.dtype is torch.float8_e4m3fn and s.dtype is torch.bfloat16
          and tuple(q.shape) == (300, 256) and tuple(s.shape) == (300,))
    check("B2 chunking invariant (bitwise)", np.array_equal(q._a, q2._a) and np.array_equal(s._a, s2._a))
    Wf = W.float()._a
    amax = np.abs(Wf).max(axis=1)
    s_ref = T.bf16_bits_to_f32(T.f32_to_bf16_bits(np.maximum(amax / 448.0, 2.0 ** -100)))
    check("B2 scale = bf16(amax/448) per row", np.array_equal(s.float()._a, s_ref))
    deq = q.float()._a * s.float()._a[:, None]
    nz = np.abs(Wf) >= 2.0 ** -6 * s_ref[:, None]
    relerr = np.abs(deq - Wf)[nz] / np.abs(Wf)[nz]
    check("B2 normal-range relative error <= 2^-4 (e4m3 half-ulp)", relerr.max() <= 2.0 ** -4 + 1e-6, f"{relerr.max():.4f}")
    check("B2 zero row -> q = 0, finite scale", np.all(deq[5] == 0) and np.isfinite(s.float()._a[5]))
    check("B2 no NaN codes, the row max hits the 448 code",
          not np.any(q._a & 0x7F == 0x7F) and np.all(np.abs(q.float()._a).max(axis=1)[np.arange(300) != 5] == 448.0))
    # B3 the FP8 head through the REAL marlin helpers
    head = ns.ParallelLMHead(1600, 256, tp_rank=0, tp_size=8)
    head.weight.copy_(torch.tensor(Wn[:200]))
    T.MarlinLog.calls.clear()
    fh = mod._Fp8Head(head)
    check("B3 Marlin repack called with the padded N (200 -> 256), K 256",
          ("repack", 256, 256) in T.MarlinLog.calls, str(T.MarlinLog.calls[:2]))
    check("B3 Marlin weight [K/16, 4*Np] int32, scales [1, Np] bf16, int32 workspace",
          tuple(fh.w.shape) == (16, 1024) and fh.w.dtype is torch.int32 and tuple(fh.ws.shape) == (1, 256)
          and fh.ws.dtype is torch.bfloat16 and fh.workspace.dtype is torch.int32)
    Lw = torch.nn.Module()
    qw, sw = mod.quantize_rows_fp8(head.weight)
    Lw.weight, Lw.weight_scale = torch.nn.Parameter(qw, requires_grad=False), torch.nn.Parameter(sw, requires_grad=False)
    Lw.orig_dtype, Lw.output_size_per_partition, Lw.input_size_per_partition = torch.bfloat16, 200, 256
    Lw.weight_block_size = None
    ns.mu8.prepare_fp8_layer_for_marlin(Lw, size_k_first=False)
    check("B3 (v0.32.0) the REAL prepare_fp8_layer_for_marlin leaves no layer.workspace; the FP8 head owns its Marlin "
          f"lock buffer instead: int32, num_SMs x MARLIN_MAX_BLOCKS_PER_SM = {SMS * MARLIN_BLOCKS_PER_SM} zeros, "
          "allocated at load",
          not hasattr(Lw, "workspace") and int(fh.workspace.numel()) == SMS * MARLIN_BLOCKS_PER_SM
          and not np.any(fh.workspace._a))
    T.MarlinLog.workspaces.clear()
    for M in (1, 2, 3, 5, 64):
        h = torch.tensor(rng.standard_normal((M, 256)).astype(np.float32)).to(torch.bfloat16).to("cuda:0")
        out = fh.apply(fh, h)
        ref = (h.float()._a @ (fh.q.float()._a * fh.s.float()._a[:, None]).T)
        refb = T.bf16_bits_to_f32(T.f32_to_bf16_bits(ref.astype(np.float32)))
        check(f"B3 FP8 head M={M}: shape [M, 200] bf16, == exact dequant GEMM (fake kernel)",
              tuple(out.shape) == (M, 200) and out.dtype is torch.bfloat16
              and np.max(np.abs(out.float()._a - refb)) <= 2.0 ** -7 * np.max(np.abs(refb)))
    gemms = [c for c in T.MarlinLog.calls if c[0] == "gemm"]
    check("B3 marlin_gemm: fp32 reduce on, atomic add off, padded N 256",
          gemms and all(c[4] is False and c[5] is True and c[2] == 256 for c in gemms))
    check("B3 (v0.32.0) the REAL marlin_utils._marlin_gemm (torch.ops.vllm.marlin_gemm) got the head's OWN workspace "
          "object on all 5 calls (never None -> no use of the workspace manager's persistent locks, which would be "
          "allocated inside a CUDA-graph capture)",
          len(T.MarlinLog.workspaces) == 5 and all(w is fh.workspace for w in T.MarlinLog.workspaces)
          and torch.ops.vllm.marlin_gemm is mu_marlin_gemm_impl(ns))
    h_tw = torch.tensor(rng.standard_normal((3, 256)).astype(np.float32)).to(torch.bfloat16).to("cuda:0")
    check("B3 (v0.32.0) tripwire: apply_fp8_marlin_linear(workspace=None) reaches marlin_utils.get_marlin_workspace -> "
          "vllm.v1.worker.workspace, which these stubs deliberately do not provide -> ImportError (so a head that "
          "stopped passing its workspace fails this suite)",
          raises(lambda: ns.mu8.apply_fp8_marlin_linear(input=h_tw, weight=fh.w, weight_scale=fh.ws, workspace=None,
                                                        size_n=200, size_k=256, bias=None), ImportError))
    hnc = torch.tensor(rng.standard_normal((4, 512)).astype(np.float32)).to(torch.bfloat16).to("cuda:0")[:, ::2]
    try:
        nc_ok = tuple(fh.apply(fh, hnc).shape) == (4, 200)
    except Exception as e:  # noqa: BLE001
        nc_ok = False
    check("B3 non-contiguous input handled (made contiguous before marlin_gemm)", nc_ok)
    lp = ns.lp.LogitsProcessor(1600)
    hb = torch.zeros(3, 256, dtype=torch.bfloat16, device="cuda:0")
    check("B4 head eligible: bf16 [3, 256] cuda", fh.eligible(lp, hb))
    check("B4 head not eligible: M > HEAD_MAX_M, fp32, wrong K, cpu, M = 0",
          not fh.eligible(lp, torch.zeros(65, 256, dtype=torch.bfloat16, device="cuda:0"))
          and not fh.eligible(lp, torch.zeros(3, 256, dtype=torch.float32, device="cuda:0"))
          and not fh.eligible(lp, torch.zeros(3, 128, dtype=torch.bfloat16, device="cuda:0"))
          and not fh.eligible(lp, torch.zeros(3, 256, dtype=torch.bfloat16, device="cpu"))
          and not fh.eligible(lp, torch.zeros(0, 256, dtype=torch.bfloat16, device="cuda:0")))
    lp.head_dtype = torch.float32
    check("B4 head not eligible with an fp32 head_dtype", not fh.eligible(lp, hb))
    # B5 helpers
    a = torch.tensor(np.array([[1.0, -0.0]], np.float32)).to(torch.bfloat16)
    b = torch.tensor(np.array([[1.0, 0.0]], np.float32)).to(torch.bfloat16)
    check("B5 _value_mismatches treats -0 == +0; _mismatches is bitwise",
          mod._value_mismatches(a, b) == 0 and mod._mismatches(a, b) == 1)
    nanp = torch.tensor(np.array([[np.nan]], np.float32)).to(torch.bfloat16)
    check("B5 _finite / NaN never equal", not mod._finite(nanp) and mod._value_mismatches(nanp, nanp) == 1)
    calls = []
    t = mod._cuda_timer(lambda: calls.append(1), 3, 10)
    check("B5 _cuda_timer: warmup + iters calls, float ms", len(calls) == 13 and isinstance(t, float))
    # B6 _EhSplit units (an in-process one-shot stand-in: no collective)
    class _OS:
        max_elems, rank, world_size = 2048, 3, 8

        def should_use(self, x):
            return x.dtype is torch.bfloat16 and x.is_contiguous() and 0 < x.numel() <= self.max_elems

        def __call__(self, x):
            return x
    ehl = torch.nn.Linear(512, 256, bias=False)
    ehl.weight.copy_(torch.tensor(rng.standard_normal((256, 512)).astype(np.float32) * 0.03))
    sp = mod._EhSplit(ehl, 3, 8, _OS())
    check("B6 eh split geometry: rank 3 -> rows [96, 128), pads 96 / 128, max_tokens 8, the slice is a view of the "
          "weight rows", (sp.n, sp.off, sp.left, sp.right, sp.max_tokens) == (32, 96, 96, 128, 8)
          and np.array_equal(sp.w._a, ehl.weight._a[96:128]))
    xx = torch.tensor(rng.standard_normal((3, 512)).astype(np.float32)).to(torch.bfloat16).to("cuda:0")
    yy = sp.pad(sp.block(xx))
    full = ehl(xx)
    check("B6 eh pad(block(x)): [3, 256], own columns == the replicated eh_proj's, zeros elsewhere",
          tuple(yy.shape) == (3, 256) and np.array_equal(yy._a[:, 96:128], full._a[:, 96:128])
          and not np.any(yy.float()._a[:, :96]) and not np.any(yy.float()._a[:, 128:]))
    check("B6 eh eligible: bf16 [1..8, 512] cuda only",
          sp.eligible(xx) and sp.eligible(xx[:1]) and not sp.eligible(torch.zeros(9, 512, dtype=torch.bfloat16, device="cuda:0"))
          and not sp.eligible(xx.float()) and not sp.eligible(torch.zeros(3, 256, dtype=torch.bfloat16, device="cuda:0"))
          and not sp.eligible(xx.to("cpu")) and not sp.eligible(torch.zeros(0, 512, dtype=torch.bfloat16, device="cuda:0")))
    check("B6 eh split refuses a TP size that does not divide H",
          raises(lambda: mod._EhSplit(ehl, 0, 6, _OS()), ValueError))
    lyr = NS(eh_proj=ehl, _p029_eh=None)
    check("B6 eh_proj_forward: no split installed -> the replicated eh_proj, bitwise",
          np.array_equal(mod.eh_proj_forward(lyr, xx)._a, full._a))
    r = mod._Rows()
    r.check("x", True)
    r.check("y", False, "d")
    check("B5 _Rows", not r.ok and r.fail == ["y: d"])
    # B7 H-checks on the small shard: clean -> all pass; an FP8 output whose top-1 / top-2 entries are swapped on the
    # random rows only must fail H3's floor (whatever H1 / H2 say about such a small perturbation)
    saved_timer = mod._TIMER
    mod._TIMER = lambda fn, w, i: (0.45 if "fh" in fn.__code__.co_freevars else 1.0)
    try:
        rows_ok = mod._Rows()
        mod._head_local_checks(fh, head, None, rows_ok)
        real_apply = fh.apply

        def swapped(layer, x, bias=None):
            out = real_apply(layer, x, bias)
            if int(x.shape[0]) == mod.HEAD_TEST_RANDOM:
                v = out.float()._a
                order = np.argsort(-v, axis=1, kind="stable")
                i1, i2 = order[:, 0], order[:, 1]
                rr = np.arange(v.shape[0])
                b1, b2 = out._a[rr, i1].copy(), out._a[rr, i2].copy()
                out._a[rr, i1], out._a[rr, i2] = b2, b1
            return out
        fh.apply = swapped
        rows_sw = mod._Rows()
        mod._head_local_checks(fh, head, None, rows_sw)
    finally:
        fh.__dict__.pop("apply", None)
        mod._TIMER = saved_timer
    check("B7 H-checks on the small shard: clean -> all pass (H1-H5); top-1/top-2 swapped on the random rows -> "
          "'H3 top-1 random rows' fails", rows_ok.ok and any(f.startswith("H3 top-1 random rows") for f in rows_sw.fail),
          f"clean {rows_ok.fail[:2]} {rows_ok.info.get('top1_random')} swapped {rows_sw.fail[:3]}")
    # B8 peaked rows on a vocab with correlated rows (as real unembeddings have: small-norm rare tokens sharing a
    # direction): of the 64 candidates, 24 get a DOMINATING row u = 1.5 w_v (a lift along w_v raises u more: v can
    # never win) and 8 a NEAR-TIE row u = 0.98 w_v (v wins by a hair): both kinds are skipped, the other 32 are built
    # and must all agree -> H3 passes (no false refusal); the near-tie rows are NOT counted as peaked
    Nv, Kv = 256, 256
    Wc = (rng.standard_t(5, size=(Nv, Kv)) * 0.02).astype(np.float32)
    cand = [int((i * 7919 + 13) % Nv) for i in range(mod.HEAD_PEAK_CANDIDATES)]
    free = [u for u in range(Nv) if u not in set(cand)]
    for j, i in enumerate(range(0, 24)):
        Wc[free[j]] = 1.5 * Wc[cand[i]]
    for j, i in enumerate(range(24, 32)):
        Wc[free[24 + j]] = 0.98 * Wc[cand[i]] + rng.standard_normal(Kv).astype(np.float32) * 1e-4
    headc = ns.ParallelLMHead(Nv * 8, Kv, tp_rank=0, tp_size=8)
    headc.weight.copy_(torch.tensor(Wc))
    fhc = mod._Fp8Head(headc)
    mod._TIMER = lambda fn, w, i: (0.45 if "fh" in fn.__code__.co_freevars else 1.0)
    try:
        rows_c = mod._Rows()
        mod._head_local_checks(fhc, headc, None, rows_c)
    finally:
        mod._TIMER = saved_timer
    check("B8 peaked rows with dominating / near-tie vocab rows: those 32 candidates skipped, the other 32 built and "
          "agreeing -> H-checks pass", rows_c.ok and rows_c.info.get("top1_peaked") == "32/32",
          f"{rows_c.fail[:2]} peaked {rows_c.info.get('top1_peaked')}")
    # B9 (review fix) peaked rows on an ANISOTROPIC shard: every row shares a mean direction (mean row cosine ~0.6;
    # trained unembeddings are anisotropic). The centred lift (along w_v - the mean row) builds the candidates and
    # they all agree -> H3 passes; a lift along w_v itself raises every row through the shared direction and builds
    # none of them (a false refusal on real weights: the mutant 'peaked rows: uncentred lift' below)
    Na, Ka = 256, 256
    rng_a = np.random.default_rng(11)
    Ea = (rng_a.standard_t(5, size=(Na, Ka)) * 0.02).astype(np.float32)
    mu = rng_a.standard_normal(Ka).astype(np.float32)
    mu /= np.linalg.norm(mu)
    typ = float(np.median(np.linalg.norm(Ea, axis=1)))
    Wa = Ea + typ * np.sqrt(0.6 / 0.4) * mu[None, :]
    an = Wa.astype(np.float64) / np.linalg.norm(Wa.astype(np.float64), axis=1, keepdims=True)
    mcos = float((np.einsum("ik,jk->ij", an, an).sum() - Na) / (Na * (Na - 1)))
    heada = ns.ParallelLMHead(Na * 8, Ka, tp_rank=0, tp_size=8)
    heada.weight.copy_(torch.tensor(Wa))
    fha = mod._Fp8Head(heada)
    mod._TIMER = lambda fn, w, i: (0.45 if "fh" in fn.__code__.co_freevars else 1.0)
    try:
        rows_a = mod._Rows()
        mod._head_local_checks(fha, heada, None, rows_a)
    finally:
        mod._TIMER = saved_timer
    built = str(rows_a.info.get("top1_peaked", "0/0")).split("/")
    check("B9 peaked rows on an anisotropic shard (mean row cosine ~0.6): the centred lift builds >= 48 of the 64 "
          "candidates, all agreeing -> H-checks pass", rows_a.ok and int(built[-1] or 0) >= 48 and 0.55 <= mcos <= 0.65,
          f"mean cos {mcos:.3f} {rows_a.fail[:2]} peaked {rows_a.info.get('top1_peaked')}")
    # B10 (rev 2) units: eh gate parsing, the log defanging, the guard while capturing / replaying, the capture
    # helper, the 032 embedding check, the fingerprint, the replay order and medians
    check("B10 parse_eh_gate: unset / '' -> graph, ' Numerics ' -> numerics, 'GRAPH' -> graph",
          mod.parse_eh_gate(None) == "graph" and mod.parse_eh_gate("") == "graph"
          and mod.parse_eh_gate(" Numerics ") == "numerics" and mod.parse_eh_gate("GRAPH") == "graph")
    check("B10 parse_eh_gate: eager / 1 / graph,numerics / off -> ValueError",
          all(raises(lambda v=v: mod.parse_eh_gate(v), ValueError) for v in ("eager", "1", "graph,numerics", "off")))
    quoted = ("CUDA OutOfMemoryError; NCCL error x; NCCL WARN y; no kernel image; cudaFuncSetAttribute(f) invalid "
              "argument; NV_ERR_NO_MEMORY; EngineDeadError; Engine core initialization failed; died unexpectedly; "
              "invalid choice; Too many open files; [recipe-entrypoint] FATAL; [029] FATAL; [patch-029] x ACTIVE "
              "INACTIVE [OK] patch 029 in place; Free memory on device 3 less than desired GPU memory utilization; "
              "FilteredTopK fallback requires")
    sd = mod._safe(quoted)
    plain = "E5 eh M=3 CUDA graphs: split <= 0.9 x replicated: 1.362 vs 2.975 ms per graph"
    check("B10 _safe: no readiness fail rule, no INACTIVE / ACTIVE / [OK] / [029] / [patch-029] left; plain reasons "
          "unchanged", not any(pt.search(sd) for pt in READINESS_FAIL)
          and not any(x in sd for x in ("INACTIVE", "ACTIVE", "[OK]", "[029]", "[patch-029]"))
          and mod._safe(plain) == plain, sd)
    uncovered = [pt.pattern for _, pt in READINESS_GATES if not any(pt.search(t) for t in GATE_TEXTS)]
    leaks = [(k, pt.pattern, t) for t in GATE_TEXTS for k, pt in READINESS_GATES if pt.search(mod._safe(t))]
    check("B10 (verification) _safe covers EVERY launch/readiness.txt fail AND require rule (read at run time, one "
          "quoted text each): none matches after _safe; idempotent",
          len(READINESS_GATES) >= 30 and not uncovered and not leaks
          and all(mod._safe(mod._safe(t)) == mod._safe(t) for t in GATE_TEXTS),
          f"{len(READINESS_GATES)} rules, uncovered {uncovered[:3]} leaks {leaks[:3]}")

    class _OS2(_OS):
        def __init__(self):
            self.log = []

        def __call__(self, x):
            self.log.append(T.capturing_graph() is not None)
            return x
    inner = _OS2()
    guard = mod._OneshotGuard(inner, 256, torch.device("cuda:0"))
    guard.expect(3, 0)
    s_side = torch.cuda.Stream()
    z = torch.zeros((3, 256), dtype=torch.bfloat16, device="cuda:0")
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0, stream=s_side, capture_error_mode="thread_local"):
        guard(z)
        guard(z)
    check("B10 guard while capturing: calls pass to the one-shot (recorded) without consuming the budget",
          guard.captured == 2 and guard.issued == 0 and guard.budget == 0 and inner.log == [True, True])

    def bad_capture():
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=s_side):
            guard(torch.zeros((4, 256), dtype=torch.bfloat16, device="cuda:0"))
    check("B10 guard while capturing: a malformed input raises (= a failed capture), nothing passed on, capture closed",
          raises(bad_capture, RuntimeError, "captured one-shot input") and inner.log == [True, True]
          and not torch.cuda.is_current_stream_capturing())
    guard.expect(3, 10)
    guard.replayed(9)
    guard.replayed(1)
    check("B10 guard.replayed(n) counts a launched replay against the plan; over the plan it raises",
          guard.budget == 0 and guard.issued == 10 and raises(lambda: guard.replayed(1), RuntimeError, "exceeds"))
    gs = mod._capture([lambda: None], s_side)
    check("B10 _capture: one graph on the given non-default stream, thread_local mode",
          gs.captured and gs.capture_stream == s_side and gs.capture_mode == "thread_local")
    before = torch.cuda.current_stream()

    def boom():
        raise RuntimeError("inside the capture")
    check("B10 _capture: an exception inside re-raises, the graph is reset, no capture left open, stream restored",
          raises(lambda: mod._capture([boom], s_side), RuntimeError, "inside the capture")
          and T.CUDAGraph.REGISTRY[-1].was_reset and not torch.cuda.is_current_stream_capturing()
          and torch.cuda.current_stream() == before)
    # (verification) a prohibited call INVALIDATES the capture: torch's capture_end then raises at cudaStreamEndCapture
    # (chained to the original error) and - in torch's ordering - leaves the pool registered with the caching
    # allocator; _capture must leave the allocator usable (no capture registered, empty_cache() works), in both orderings
    # and without ever releasing a pool twice
    def _invalidated(fn):
        ev0 = len(T.POOL_EVENTS)
        ok = raises(lambda: mod._capture([fn], s_side), RuntimeError, "previous error during capture")
        T.set_stream(before)               # torch.cuda.graph skips its stream exit when capture_end raises
        return ok and T.allocator_clean() and not raises(torch.cuda.empty_cache) and len(T.POOL_EVENTS) == ev0
    stuck_ok = _invalidated(lambda: z.float().sum().item()) and _invalidated(torch.cuda.synchronize)
    T.STUCK_ON_INVALIDATED[0] = False
    unstuck_ok = _invalidated(lambda: z.float().sum().item()) and _invalidated(torch.cuda.synchronize)
    T.STUCK_ON_INVALIDATED[0] = True
    check("B10 prohibited while capturing (as CUDA): .item() / torch.cuda.synchronize() invalidate the capture, "
          "capture_end raises; _capture leaves no pool registered (empty_cache() works) whether torch's capture_end "
          "leaves it registered or not, never a double release; a capture on the default stream is refused",
          stuck_ok and unstuck_ok
          and raises(lambda: mod._capture([lambda: None], T.DEFAULT_STREAM), RuntimeError, "non-default")
          and T.allocator_clean(), f"stuck {stuck_ok} unstuck {unstuck_ok}")
    pool_g = mod._capture([lambda: None], s_side)
    pid = pool_g.pool_id
    mod._reset_graph(pool_g)
    ev0 = len(T.POOL_EVENTS)
    check("B10 _unregister_pool after a clean capture + reset: 'not registered', nothing released twice",
          mod._unregister_pool(pid, None) == "not registered" and len(T.POOL_EVENTS) == ev0 and T.allocator_clean()
          and mod._unregister_pool(None, None) == "no pool handle")
    def _begin_fails(g):
        raise RuntimeError("injected begin")
    T.set_current_world(NS(rank=0, capture_hook=_begin_fails))
    try:
        begin_ok = raises(lambda: mod._capture([lambda: None], s_side), RuntimeError, "injected begin")
    finally:
        T.set_current_world(None)
    T.set_stream(before)
    check("B10 a failed capture BEGIN (cudaStreamBeginCapture; torch registered the pool before it): _capture "
          "unregisters it, empty_cache() works", begin_ok and T.allocator_clean() and not raises(torch.cuda.empty_cache))

    # (verification) the E5 decision itself on exact clock values: the 0.9 bound is inclusive, median of 10 timed
    # replays (upper median) after 2 untimed, robust to < 5 outliers, a degenerate replicated time refuses
    class _GS:
        def __init__(self, seq):
            self.seq, self.i = list(seq), 0

        def replay(self):
            T._vclock_advance(self.seq[min(self.i, len(self.seq) - 1)])
            self.i += 1

    def e5(split_seq, full_seq):
        rows_t = mod._Rows()
        gt = mod._OneshotGuard(_OS2(), 256, torch.device("cuda:0"))
        mod._eh_graph_timing([(0, 3, _GS(split_seq), _GS(full_seq), None, None)], gt, rows_t)
        return rows_t.ok, gt.budget, rows_t.info.get("eh_graph_ms")
    at_bound = e5([2.25] * 12, [2.5] * 12)                          # 2.25 == 0.9 x 2.5 exactly (binary fractions)
    over = e5([2.25 + 2.0 ** -20] * 12, [2.5] * 12)
    out4 = e5([9.0, 9.0] + [40.0] * 4 + [1.25] * 6, [2.5] * 12)      # 2 warm-ups + 4 timed outliers
    out5 = e5([1.25, 1.25] + [40.0] * 5 + [1.25] * 5, [2.5] * 12)    # 5 of 10 timed: the upper median moves
    zero = e5([0.0] * 12, [0.0] * 12)
    check("B10 E5 decision: split == 0.9 x replicated passes (inclusive), a hair over fails; 4 of 10 timed replays "
          "40 ms slower still pass (median), 5 fail; zero times fail; the plan's 120 one-shots consumed each time",
          at_bound[0] and not over[0] and out4[0] and not out5[0] and not zero[0]
          and all(r[1] == 0 for r in (at_bound, over, out4, out5, zero)),
          f"{at_bound} {over} {out4} {out5} {zero}")
    w8 = NS(size=8)
    check("B10 eh embed check: a vocab-parallel embed_tokens over TP 8 passes; tp_size 1 / disable_tp / none -> "
          "refused, naming patch 032 and the PARTS=head remedy",
          mod._embed_refusal(NS(embed_tokens=NS(tp_size=8, disable_tp=False)), w8) is None
          and "patch 032" in (mod._embed_refusal(NS(embed_tokens=NS(tp_size=1)), w8) or "")
          and "VLLM_GB10_DRAFT_COMPRESS_PARTS=head" in (mod._embed_refusal(NS(embed_tokens=NS(tp_size=1)), w8) or "")
          and mod._embed_refusal(NS(embed_tokens=NS(tp_size=8, disable_tp=True)), w8) is not None
          and mod._embed_refusal(NS(), w8) is not None)
    check("B10 fingerprint covers the eh gate", mod.fingerprint(("eh", "head"), 8, {"a": 1}, "graph")
          != mod.fingerprint(("eh", "head"), 8, {"a": 1}, "numerics"))

    class _G:
        def __init__(self, ms, log, name):
            self.ms, self.log, self.name = ms, log, name

        def replay(self):
            self.log.append(self.name)
            T._vclock_advance(self.ms)
    order = []
    gq = mod._OneshotGuard(_OS2(), 256, torch.device("cuda:0"))
    gq.expect(1, 12 * (9 + 1))
    ts, tf = mod._replay_timed(((_G(1.5, order, "s"), 9), (_G(3.0, order, "f"), 1)), gq)
    check("B10 _replay_timed: 2 untimed + 10 timed interleaved rounds (split, full), CUDA-event times, the plan's "
          "one-shots consumed exactly", order == ["s", "f"] * 12 and ts == [1.5] * 10 and tf == [3.0] * 10
          and gq.budget == 0 and gq.issued == 120, f"{order[:4]} {ts[:2]} {tf[:2]} {gq.budget}")
    check("B10 _median (upper median of the timed replays)", mod._median([3, 1, 2]) == 2
          and mod._median([4, 1, 3, 2]) == 3 and math.isnan(mod._median([])))
    return ns


def _fn_src(text: str, qualname: str) -> str:
    tree = ast.parse(text)
    body, node = tree.body, None
    for part in qualname.split("."):
        node = next((n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part), None)
        if node is None:
            return ""
        body = node.body
    return ast.get_source_segment(text, node) or ""


def section_b11(vroot, texts, ap):
    """v0.32.0 contracts read from the REAL sources (the sha pins already freeze them; these say WHAT 029 relies on)."""
    print("== B11: v0.32.0 contracts of the hooked draft-logits path, read from the REAL sources")
    sd = _fn_src(texts["dspec"], "DraftModelSpeculator.sample_draft")
    cdl = _fn_src(texts["dspec"], "DraftModelSpeculator.compute_draft_logits")
    gdt = _fn_src(texts["dspec"], "DraftModelSpeculator.get_draft_top_tokens")
    vla = _fn_src(texts["dspec"], "DraftModelSpeculator._validate_local_argmax_reduction")
    check("B11 compute_draft_logits is `return self.model.compute_logits(hidden_states)`: the method patch 029 hooks "
          "(Glm5NextMTP.compute_logits -> Glm5NextMultiTokenPredictor.compute_logits)",
          "return self.model.compute_logits(hidden_states)" in cdl, cdl[-120:])
    check("B11 sample_draft: logits = compute_draft_logits(...) feeds gumbel_sample(logits_cache=draft_logits) when the "
          "draft-logits cache exists (draft_sample_method=probabilistic: the served config), and logits.argmax otherwise",
          "logits = self.compute_draft_logits(hidden_states, spec_step_idx)" in sd and "logits_cache=draft_logits" in sd
          and "sampled = logits.argmax(dim=-1)" in sd)
    check("B11 LIMITATION: get_top_tokens (BF16 head, not hooked) is reached only when draft_logits is None AND "
          "use_local_argmax_reduction (greedy drafts with the local-argmax reduction)",
          "if draft_logits is None and self.use_local_argmax_reduction:" in sd
          and "return self.get_draft_top_tokens(hidden_states, spec_step_idx)" in sd
          and "return self.model.get_top_tokens(hidden_states)" in gdt)
    check("B11 use_local_argmax_reduction is refused together with draft_sample_method=probabilistic",
          "draft_sample_method == \"probabilistic\"" in vla and "raise ValueError" in vla)
    check("B11 the draft-logits cache is allocated only for probabilistic drafting, in head_dtype (= the dtype of the "
          "logits the hook returns: bf16 for the served model)",
          "if self.speculative_config.draft_sample_method == \"probabilistic\":" in texts["dspec"]
          and "return vllm_config.model_config.head_dtype, 0.0" in texts["dspec"])
    lp = texts["lp"]
    check("B11 LogitsProcessor._apply_head calls lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias) "
          "when head_dtype is None or equals the activations' dtype (what the FP8 stand-in head implements)",
          "return lm_head.quant_method.apply(\n                lm_head, hidden_states, bias=embedding_bias\n            )"
          in _fn_src(lp, "LogitsProcessor._apply_head"))
    gl = _fn_src(lp, "LogitsProcessor._get_logits")
    check("B11 LogitsProcessor._get_logits: _apply_head, then (unless skip_gather) the TP gather keyed on lm_head.tp_size, "
          "then the vocab slice [..., :org_vocab_size]", "self._apply_head(lm_head, hidden_states, embedding_bias)" in gl
          and "if lm_head.tp_size > 1:" in gl and "logits[..., : self.org_vocab_size]" in gl)
    eg = _fn_src(texts["eagle"], "load_eagle_model")
    check("B11 load_eagle_model shares the target lm_head into every MTP layer's shared_head.head (the module mirrors "
          "exactly that module)", "sh.head = target_lm_head" in eg and "del sh.head" in eg and "layers = getattr(draft_inner, \"layers\", None)" in eg)
    check("B11 the v0.32.0 MTP layer builds its SharedHead with defer_lm_head=True (head arrives from load_eagle_model, so "
          "prepare() must run after it: the hook sits right after load_eagle_model)",
          "defer_lm_head=True" in texts["mtp"] and ap.SPEC_OLD in texts["spec"])
    ld = _fn_src(texts["spec"], "MTPSpeculator.load_draft_model")
    check("B11 MTPSpeculator.load_draft_model starts with the load_eagle_model assignment (the hook's anchor) and returns "
          "the draft model", ld.splitlines()[1:3] and "draft_model = load_eagle_model(target_model, self.vllm_config)" in ld
          and "return draft_model" in ld)


# =====================================================================================================================
# C: 8-rank end-to-end
# =====================================================================================================================
def section_c(vroot, texts, ap, G):
    print("== C: 8-rank threaded end-to-end on the REAL patched files")
    mtp_p = ap.apply("mtp", texts["mtp"])
    spec_p = ap.apply("spec", texts["spec"])
    on = {"VLLM_GB10_DRAFT_COMPRESS": "1"}

    def fresh(env=None, **kw):
        return build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env={**on, **(env or {})}, **kw)

    # ---- C1 clean boot
    ns = fresh()
    graphs_before = len(T.CUDAGraph.REGISTRY)
    rb = run_boot(ns, G)
    st = state_of(rb)
    check("C1 clean boot: no thread error", not errs(rb), str(errs(rb)[:1]))
    check("C1 clean boot: head + eh ACTIVE on all 8 ranks", all(v == (True, True) for v in st.values()), str(st))
    eqs, lens = schedules_equal(rb)
    check("C1 identical collective schedules, no hang / mismatch", eqs and not rb.rv.hangs and not rb.rv.mismatches,
          f"lens {lens}")
    sched = rb.rv.schedules[0]
    n_os = sum(1 for s in sched if s[0] == "oneshot")
    n_ag = sum(1 for s in sched if s[0] == "all_gather")
    n_ex = sum(1 for s in sched if s[0] == "all_gather_object")
    mod = ns.mod
    n_rep = 2 * (mod.EH_GRAPH_WARMUP + mod.EH_GRAPH_ITERS) * (mod.EH_GRAPH_CALLS + 2)
    check("C1 planned collectives (rev 2): 4 eh checks + 2 x 13 eager-timed one-shots + 2 x 12 x (9 + 1) graph-replayed "
          "one-shots, 3 x 2 logits all-gathers, 4 exchanges (preflight, graph capture, verdict, commit)",
          n_os == 4 + 2 * 13 + n_rep and n_rep == 240 and n_ag == 6 and n_ex == 4,
          f"oneshot {n_os} all_gather {n_ag} exchanges {n_ex}")
    seg = _segments(sched)
    check("C1 order: E1-E4 + eager one-shots before the capture exchange, all 240 replayed one-shots between the capture "
          "and the verdict exchange, the head's logits all-gathers after every eh one-shot, verdict + commit last",
          seg["pre_cap_os"] == 30 and seg["cap_ver_os"] == 240 and seg["head_after_eh"] and seg["post_ver"] == 2,
          str(seg))
    replayed = [o.world._oneshot.replayed for o in rb.ranks.values()]
    captured = [o.world._oneshot.captured for o in rb.ranks.values()]
    check("C1 every rank captured 2 M x (9 + 1) one-shots (nothing communicated while capturing) and replayed 240",
          replayed == [240] * 8 and captured == [20] * 8, f"replayed {replayed} captured {captured}")
    gr = [g for g in T.CUDAGraph.REGISTRY[graphs_before:]]
    check("C1 boot graphs: 4 per rank (G_split, G_full x M = 1, 3), captured on a side (non-default) stream in "
          "thread_local mode, every one replayed 12 times and reset before prepare() returned",
          len(gr) == 32 and all(g.capture_mode == "thread_local" and g.capture_stream is not None
                                and g.capture_stream.id != 0 and g.was_reset for g in gr)
          and sorted(g.replays for g in gr) == [12] * 32, f"{len(gr)} graphs")
    check("C1 the thread's current stream is the default again after prepare(), no capture left open",
          all(st == 0 for st in rb.stream_after.values()) and not any(rb.capture_after.values()),
          f"{rb.stream_after} {rb.capture_after}")
    check("C1 (verification) the caching allocator is usable after prepare() on every rank: no boot-graph pool left "
          "registered, every pool released once (vLLM's next empty_cache() works)",
          len(rb.alloc_after) == 8 and all(rb.alloc_after.values())
          and all(g.pool_id is not None and not g.capture_ended for g in gr), f"{rb.alloc_after}")
    act = [m for (_, lvl, m) in LOGS.records if "ACTIVE (rank" in m and "INACTIVE" not in m]
    check("C1 one '[patch-029] draft-path compression ACTIVE' line per rank", len(act) == 8, f"{len(act)}")
    gate = gate_counts(LOGS.records)
    check("C1 the boot-gate scripts' patterns on the boot log: 'ACTIVE (rank' exactly once per rank (8), the "
          "INACTIVE / not active / exchange failed / unexpected / [029] FATAL pattern 0 times, no readiness fail pattern",
          gate["active_ranks"] == list(range(8)) and gate["bad"] == 0 and gate["readiness_fail"] == [], str(gate))
    a0 = next((m for m in act if "(rank 0/8" in m), "")
    check("C1 the ACTIVE line reports the CUDA-graph E5 numbers [M, split, replicated, saved per call] and the eager "
          "timing as information, module rev 3",
          "E5 CUDA graphs (one-shot + 8 calls, median of 10 replays, bound 0.9) [M, split, replicated, saved per call]"
          " ms [[1, 1.3622, 2.975, 0.2016], [3, 1.3622, 2.975, 0.2016]]" in a0
          and "eager [M, split, replicated] ms [[1, 0.1, 0.3], [3, 0.1, 0.3]] (information)" in a0
          and "module rev 3" in a0, a0[-420:])
    check("C1 the shared draft head IS the target's lm_head (real load_eagle_model)",
          all(o.draft.model._mtp_layers[0].shared_head.head is o.target.lm_head for o in rb.ranks.values()))
    wbytes = {r: o.target.lm_head.weight._a.copy() for r, o in rb.ranks.items()}
    vp = DIMS.V // DIMS.W
    check("C1 target lm_head weights untouched (== the rank's bf16 rows), quant_method unchanged",
          all(np.array_equal(T.f32_to_bf16_bits(G.lm[r * vp:(r + 1) * vp]), wbytes[r]) for r in rb.ranks)
          and all(type(o.target.lm_head.quant_method).__name__ == "UnquantizedEmbeddingMethod"
                  for o in rb.ranks.values()))

    # ---- C2 runtime: draft forward (eh hook) and draft logits on 8 ranks
    torch = ns.torch

    def fwd(r, o, M, use_split=True):
        ly = o.draft.model._mtp_layers[0]
        saved = ly._p029_eh
        if not use_split:
            ly._p029_eh = None
        ids = torch.tensor(np.arange(1, M + 1) % DIMS.V, dtype=torch.int64, device="cuda:0")
        pos = torch.tensor(np.arange(5, 5 + M), dtype=torch.int64, device="cuda:0")
        prev = torch.tensor(np.random.default_rng(100 + M).standard_normal((M, DIMS.H)).astype(np.float32)).to(
            torch.bfloat16).to("cuda:0")
        ly.mtp_block.captured.clear()
        os0 = o.world._oneshot.calls if o.world._oneshot else 0
        out = o.draft(ids, pos, prev)
        ly._p029_eh = saved
        return ly.mtp_block.captured[-1], (o.world._oneshot.calls if o.world._oneshot else 0) - os0, out

    for M in (1, 3, 4, 8, 9, 16):
        res, er, rv = run_ranks_fn(ns, rb, lambda r, o: fwd(r, o, M))
        res2, er2, rv2 = run_ranks_fn(ns, rb, lambda r, o: fwd(r, o, M, use_split=False))
        if any(e is not None for e in er + er2):
            check(f"C2 eh forward M={M} ran", False, str([e for e in er + er2 if e][:1]))
            continue
        split = [x[0] for x in res]
        img = [x[0] for x in res2]
        same_all = all(np.array_equal(split[0]._a, s._a) for s in split)
        if M <= 8:
            x = split[0].float()._a
            ref = img[0].float()._a
            d = np.abs(x - ref)
            bound = (np.abs(x) + np.abs(ref)) * 2.0 ** -8 + 1e-6 + 2.0 ** -10 * 1.0
            check(f"C2 eh M={M}: split path taken (1 one-shot per rank), identical on 8 ranks, within the bound of "
                  "the replicated eh_proj", all(x[1] == 1 for x in res) and same_all and np.all(d <= bound),
                  f"oneshots {[x[1] for x in res]} max|d| {d.max():.3g}")
            check(f"C2 eh M={M}: no collective mismatch / hang", not rv.hangs and not rv.mismatches)
        else:
            check(f"C2 eh M={M} > one-shot limit: image path bitwise, no collective",
                  all(x[1] == 0 for x in res) and all(np.array_equal(a._a, b._a) for a, b in zip(split, img))
                  and not rv.schedules[0])

    def logits(r, o, M, noncontig=False, swap_head=False):
        pred = o.draft.model
        rng = np.random.default_rng(500 + M)
        hh = torch.tensor(rng.standard_normal((M, 2 * DIMS.H if noncontig else DIMS.H)).astype(np.float32)).to(
            torch.bfloat16).to("cuda:0")
        if noncontig:
            hh = hh[:, ::2]
        ly = pred._mtp_layers[0]
        saved_head = ly.shared_head.head
        if swap_head:
            ly.shared_head.head = ns.ParallelLMHead(DIMS.V, DIMS.H, tp_rank=r, tp_size=DIMS.W)
            ly.shared_head.head.weight.copy_(saved_head.weight)
        a = pred.compute_logits(hh)
        ly.shared_head.head = saved_head
        fh = pred._p029_head
        pred._p029_head = None
        b = pred.compute_logits(hh)
        pred._p029_head = fh
        return a, b, hh

    for M, nc in ((1, False), (3, True), (8, False), (64, False), (65, False)):
        res, er, rv = run_ranks_fn(ns, rb, lambda r, o: logits(r, o, M, nc))
        if any(e is not None for e in er):
            check(f"C2 draft logits M={M} ran", False, str([e for e in er if e][:1]))
            continue
        a0, b0, h0 = res[0]
        same = all(np.array_equal(res[r][0]._a, a0._a) for r in range(8))
        if M <= 64:
            # reference: exact dequant of every rank's e4m3 shard, concatenated in rank order
            qs = []
            for r in range(8):
                Wr = torch.tensor(G.lm[r * vp:(r + 1) * vp]).to(torch.bfloat16).to("cuda:0")
                q, s = ns.mod.quantize_rows_fp8(Wr)
                qs.append(q.float()._a * s.float()._a[:, None])
            deq = np.concatenate(qs, 0)
            ref = T.bf16_bits_to_f32(T.f32_to_bf16_bits((h0.contiguous().float()._a @ deq.T).astype(np.float32)))
            dd = np.abs(a0.float()._a - ref).max() / max(np.abs(ref).max(), 1e-9)
            rel = np.sqrt(((a0.float()._a - b0.float()._a) ** 2).sum() / (b0.float()._a ** 2).sum())
            check(f"C2 draft logits M={M}{' (non-contiguous h)' if nc else ''}: FP8 path, [M, vocab] bf16, identical "
                  "on 8 ranks, == exact dequant of the 8 shards in rank order, relRMSE vs BF16 <= 0.05",
                  tuple(a0.shape) == (M, DIMS.V) and a0.dtype is torch.bfloat16 and same and dd <= 2.0 ** -7
                  and rel <= 0.05 and not np.array_equal(a0._a, b0._a), f"dd {dd:.2g} rel {rel:.4f}")
            cache = a0.to(torch.bfloat16)
            check(f"C2 draft logits M={M}: the bf16 draft-logits cache stores exactly the sampled-from values",
                  np.array_equal(cache._a, a0._a))
        else:
            check(f"C2 draft logits M={M} > HEAD_MAX_M: image BF16 path bitwise", np.array_equal(a0._a, b0._a) and same)
    res, er, rv = run_ranks_fn(ns, rb, lambda r, o: logits(r, o, 3, swap_head=True))
    check("C2 draft logits with a swapped head module (not the one the FP8 copy mirrors): image path bitwise",
          not any(er) and all(np.array_equal(x[0]._a, x[1]._a) for x in res))

    def greedy(r, o):
        pred = o.draft.model
        hh = torch.tensor(np.random.default_rng(9).standard_normal((2, DIMS.H)).astype(np.float32)).to(
            torch.bfloat16).to("cuda:0")
        return pred.compute_logits(hh)
    # target logits untouched: the target's own LogitsProcessor on the shared head == plain BF16 math
    def target_logits(r, o):
        lp_t = ns.lp.LogitsProcessor(DIMS.V)
        hh = torch.tensor(np.random.default_rng(11).standard_normal((3, DIMS.H)).astype(np.float32)).to(
            torch.bfloat16).to("cuda:0")
        return lp_t(o.target.lm_head, hh), hh
    res, er, rv = run_ranks_fn(ns, rb, target_logits)
    a0, hh = res[0]
    ref = T.bf16_bits_to_f32(T.f32_to_bf16_bits(
        (hh.float()._a @ T.bf16_bits_to_f32(T.f32_to_bf16_bits(G.lm)).T).astype(np.float32)))
    check("C2 the TARGET path (its LogitsProcessor on the shared lm_head) is exactly the BF16 GEMM after prepare",
          not any(er) and np.array_equal(a0.float()._a, ref))

    # ---- C3 gate unset: patched files == unpatched image files, bitwise; no prepare, no attributes
    def boot_plain(nsx):
        W = DIMS.W
        rv = T.Rendezvous(W)
        out = {}

        def body(r):
            w = T.ThreadWorld(rv, r, W, oneshot_max=DIMS.ONESHOT_MAX)
            T.set_current_world(w)
            sp, draft, target = build_rank(nsx, r, G)
            sp.load_draft_model(target, set())
            out[r] = NS(sp=sp, draft=draft, target=target, world=w)
        _, er = T.run_ranks(W, body)
        return NS(ranks=out, rv=rv, err=er)
    got = {}
    for label, mt, st_, modp in (("patched", mtp_p, spec_p, MODULE_PATH), ("image", texts["mtp"], texts["spec"], None)):
        nsx = build_ns(vroot, mt, st_, modp, env={})
        rbx = boot_plain(nsx)
        if label == "patched":
            check("C3 gate unset: the patched mtp.py does not import the module, no collective at boot, no _p029 "
                  "attributes", not getattr(nsx.mtp, "_P029_ON", True) and not hasattr(nsx.mtp, "_p029")
                  and not any(rbx.rv.schedules) and not any(errs(rbx))
                  and all("_p029_head" not in vars(o.draft.model) for o in rbx.ranks.values()))
        rf, e1, _ = run_ranks_fn(nsx, rbx, lambda r, o: _fwd_plain(nsx, o, 3))
        rl, e2, _ = run_ranks_fn(nsx, rbx, lambda r, o: _logits_plain(nsx, o, 3))
        got[label] = (rf, rl, e1 + e2)
    (pf, pl, pe), (imf, iml, ie) = got["patched"], got["image"]
    check("C3 gate unset: draft forward + draft logits of the patched files == the image files, bitwise",
          not any(pe) and not any(ie) and all(np.array_equal(a._a, b._a) for a, b in zip(pf, imf))
          and all(np.array_equal(a._a, b._a) for a, b in zip(pl, iml)))

    # ---- C4 parts
    for parts, want in (("head", (True, False)), ("eh", (False, True))):
        nsx = fresh({"VLLM_GB10_DRAFT_COMPRESS_PARTS": parts})
        rbx = run_boot(nsx, G)
        ok, stx = consistent(rbx)
        n_os = sum(1 for s in rbx.rv.schedules[0] if s[0] == "oneshot")
        check(f"C4 parts={parts}: {want} on every rank, no eh collectives when eh is off",
              ok and stx[0] == want and not errs(rbx) and (n_os == 0) == (not want[1]), f"{stx[0]} oneshots {n_os}")

    # ---- C5 fail-closed scenarios
    def scen(name, want, *, env=None, worlds_cfg=None, times=None, inject=None, head_dtype="bfloat16",
             draft_cls=None, same_schedule_as_clean=False, allow_bounded_timeout=False, schedule_like=None,
             tag="C5"):
        """schedule_like: 'clean' (== the clean boot's collective schedule), 'no_replays' (== the clean one minus the
        graph replays: every rank decided after the capture exchange not to replay), 'no_capture_exchange', or a
        predicate on rank 0's schedule."""
        nsx = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env={**on, **(env or {})}, head_dtype_name=head_dtype)
        rbx = run_boot(nsx, G, worlds_cfg=worlds_cfg, times=times, inject=inject, draft_cls=draft_cls,
                       rv_timeout=2.0)
        ok, stx = consistent(rbx)
        eqs, lens = schedules_equal(rbx)
        hang_ok = not rbx.rv.hangs if not allow_bounded_timeout else all(h[1] == ("barrier",) for h in rbx.rv.hangs)
        like = "clean" if same_schedule_as_clean else schedule_like
        s0 = rbx.rv.schedules[0]
        if like is None:
            extra, lbl = True, ""
        elif like == "clean":
            extra, lbl = s0 == sched, ", schedule == the clean boot's"
        elif like == "no_replays":
            extra, lbl = s0 == without_replays(sched), ", schedule == the clean boot's minus the graph replays"
        elif like == "no_capture_exchange":
            extra, lbl = s0 == without_capture_exchange(sched), ", schedule == the clean boot's minus the capture exchange and the replays"
        else:
            extra, lbl = bool(like(s0)), ", schedule as planned"
        inact = [m for (_, lvl, m) in LOGS.records if "INACTIVE" in m or "not active" in m]
        alloc_ok = len(rbx.alloc_after) == DIMS.W and all(rbx.alloc_after.values())
        check(f"{tag} {name}: {want} on all 8 ranks, nothing raised, no hang/mismatch, caching allocator usable" + lbl,
              ok and stx[0] == want and not errs(rbx) and hang_ok and not rbx.rv.mismatches
              and (eqs or allow_bounded_timeout) and extra and alloc_ok,
              f"state {stx.get(0)} errs {errs(rbx)[:1]} hangs {rbx.rv.hangs[:1]} mism {rbx.rv.mismatches[:1]} "
              f"lens {lens} (clean {len(sched)}) alloc {rbx.alloc_after} why {[x[-300:] for x in inact[:1]]}")
        rbx.logs = list(LOGS.records)
        return rbx

    scen("one rank without patch 012's one-shot", (True, False), worlds_cfg={3: {"oneshot_max": None}})
    scen("no rank with the one-shot", (True, False), worlds_cfg={r: {"oneshot_max": None} for r in range(8)})

    def corrupt_marlin(rank_bad):
        def inj(r, nsx, draft, target, w):
            if r == rank_bad:        # installed once (by the bad rank's thread); only that rank's outputs change
                orig = nsx.torch.ops._C.marlin_gemm

                def bad(*a, **kw):          # v0.32.0: _marlin_gemm calls torch.ops._C.marlin_gemm positionally
                    out = orig(*a, **kw)
                    return out * 1.5 if T.current_world().rank == rank_bad else out
                nsx.torch.ops._C.marlin_gemm = bad
        return inj
    scen("rank 5's FP8 GEMM corrupt (x1.5)", (False, True), inject=corrupt_marlin(5))

    def eh_block_raises(rank_bad):
        def inj(r, nsx, draft, target, w):
            if r == rank_bad:
                orig_init = nsx.mod._EhSplit.block

                def blk(self, x):
                    if T.current_world().rank == rank_bad:
                        raise RuntimeError("injected local failure in the eh block GEMM")
                    return orig_init(self, x)
                nsx.mod._EhSplit.block = blk
        return inj
    scen("rank 2's eh block raises inside the collective phase (E1-E4 zeros keep the schedule; its capture of the "
         "split path fails too -> nobody replays)", (True, False), inject=eh_block_raises(2), schedule_like="no_replays")

    def oneshot_corrupt(rank_bad):
        def inj(r, nsx, draft, target, w):
            if r == rank_bad:
                class Bad(T.FakeOneshot):
                    def __call__(self, x):
                        out = T.FakeOneshot.__call__(self, x)
                        return out * 1.0078125
                b = Bad(w, DIMS.ONESHOT_MAX)
                w._oneshot = b
        return inj
    scen("rank 6's one-shot output differs from the other ranks'", (True, False), inject=oneshot_corrupt(6),
         same_schedule_as_clean=True)
    # fingerprint mismatch: rank 4 sees a different module sha (as if a different module revision were installed)

    def fp_mismatch(r, nsx, draft, target, w):
        if r == 4:
            w.fp_override = True
    nsfp = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env=on)
    orig_fp = nsfp.mod.fingerprint

    def fp(*a, **k):
        f = orig_fp(*a, **k)
        return f[:-1] + "x" if getattr(T.current_world(), "fp_override", False) else f
    nsfp.mod.fingerprint = fp
    rbx = run_boot(nsfp, G, inject=fp_mismatch, rv_timeout=2.0)
    ok, stx = consistent(rbx)
    check("C5 rank-fingerprint mismatch (rank 4): off on all ranks, only the 2 exchanges issued, no hang",
          ok and stx[0] == (False, False) and not rbx.rv.hangs and not rbx.rv.mismatches
          and [s[0] for s in rbx.rv.schedules[0]] == ["barrier", "all_gather_object"], str(rbx.rv.schedules[0][:4]))
    scen("every rank's preflight barrier fails", (False, False), worlds_cfg={r: {"fail_barrier": True} for r in range(8)})
    scen("one rank's preflight barrier fails (others time out, bounded)", (False, False),
         worlds_cfg={1: {"fail_barrier": True}}, allow_bounded_timeout=True)

    def verdict_fail(r, nsx, draft, target, w):
        if r == 7:
            orig = w.all_gather_object
            cnt = {"n": 0}

            def ago(obj):
                cnt["n"] += 1
                out = orig(obj)
                if cnt["n"] == 3:      # rev 2: preflight, graph capture, VERDICT, commit
                    raise RuntimeError("injected local failure after the verdict all-gather")
                return out
            w.all_gather_object = ago
    scen("rank 7's verdict exchange fails locally after the collective (commit round)", (False, False),
         inject=verdict_fail)
    scen("rank 2's FP8 head not faster (0.95x)", (False, True), times={2: {"fp8": 0.95}})
    scen("rank 0's EAGER split 1.03x of the replicated eh_proj (rev 1 refused eh here; information only since rev 2, the "
         "CUDA-graph timing decides): eh ACTIVE", (True, True), times={0: {"split": 0.31}}, schedule_like="clean")
    scen("rank 0's EAGER split 0.97x of the replicated eh_proj: eh ACTIVE", (True, True), times={0: {"split": 0.29}},
         schedule_like="clean")
    scen("VLLM_BATCH_INVARIANT=1", (False, False), env={"VLLM_BATCH_INVARIANT": "1"})
    scen("head_dtype float32", (False, True), head_dtype="float32")

    nsn = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env=on)

    class OtherMTP(nsn.mtp.Glm5NextMTP):
        pass
    rbx = run_boot(nsn, G, draft_cls=OtherMTP, rv_timeout=2.0)
    ok, stx = consistent(rbx)
    check("C5 a draft model that is not glm5next Glm5NextMTP: off on all ranks, nothing raised",
          ok and stx[0] == (False, False) and not errs(rbx) and not rbx.rv.hangs)
    # (v0.32.0) the class' module path moved nvidia/ -> common/: the same class name from the OLD path is refused
    nso = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env=on)

    class Glm5NextMTP(nso.mtp.Glm5NextMTP):
        pass
    Glm5NextMTP.__module__ = "vllm.models.glm5next.nvidia.mtp"
    rbx = run_boot(nso, G, draft_cls=Glm5NextMTP, rv_timeout=2.0)
    ok, stx = consistent(rbx)
    check("C5 (v0.32.0) a Glm5NextMTP from the old module path vllm.models.glm5next.nvidia.mtp: off on all ranks, "
          "nothing raised, no hang", ok and stx[0] == (False, False) and not errs(rbx) and not rbx.rv.hangs, str(stx.get(0)))

    def drop_vllm_op(r, nsx, draft, target, w):
        if hasattr(nsx.torch.ops.vllm, "marlin_gemm"):
            delattr(nsx.torch.ops.vllm, "marlin_gemm")      # idempotent; any rank seeing it gone refuses the head for all
    rbx = scen("(v0.32.0) torch.ops.vllm.marlin_gemm is not registered (marlin_utils not imported / op renamed): the head "
               "is refused on every rank (reason names the op), the eh part is unaffected", (False, True),
               inject=drop_vllm_op)
    check("C5 (v0.32.0) the refusal names torch.ops.vllm.marlin_gemm",
          any("torch.ops.vllm.marlin_gemm missing" in m for (_, _, m) in rbx.logs))

    nsu = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env=on)
    orig_ce = nsu.mod._common_envelope

    def ce(world, parts):
        if world.rank == 3:
            raise RuntimeError("injected unexpected exception before the preflight")
        return orig_ce(world, parts)
    nsu.mod._common_envelope = ce
    rbx = run_boot(nsu, G, rv_timeout=2.0)
    ok, stx = consistent(rbx)
    check("C5 an unexpected exception on rank 3 in the local phase: prepare() never raises; rank 3 still joins the "
          "preflight (as a refusal) -> off everywhere, no rank waits out the barrier, only the 2 exchanges issued",
          ok and stx[0] == (False, False) and not errs(rbx) and not rbx.rv.hangs and not rbx.rv.mismatches
          and [s[0] for s in rbx.rv.schedules[0]] == ["barrier", "all_gather_object"],
          f"{stx} {rbx.rv.hangs[:1]} {errs(rbx)[:1]} {rbx.rv.schedules[0][:3]}")
    def nondet_m3(rank_bad):
        def inj(r, nsx, draft, target, w):
            if r == rank_bad:        # rank_bad's FP8 GEMM varies between calls, only at M = 3 (= the collective phase)
                orig = nsx.torch.ops._C.marlin_gemm
                cnt = {"n": 0}

                def bad(*a, **kw):          # size_m is positional #10 of torch.ops._C.marlin_gemm (v0.32.0)
                    out = orig(*a, **kw)
                    if T.current_world().rank == rank_bad and a[10] == 3:
                        cnt["n"] += 1
                        return out * (1.0 + 0.0078125 * cnt["n"])
                    return out
                nsx.torch.ops._C.marlin_gemm = bad
        return inj
    scen("rank 5's FP8 GEMM nondeterministic at M=3 only (fails G2 on rank 5 alone, collective phase)", (False, True),
         inject=nondet_m3(5))

    def wrong_rows(rank_bad):
        def inj(r, nsx, draft, target, w):
            if r == rank_bad:        # rank_bad's eh slice holds the next rank's rows but keeps its own placement
                orig = nsx.mod._EhSplit.__init__

                def init(self, eh_proj, rank, world_size, oneshot):
                    orig(self, eh_proj, rank, world_size, oneshot)
                    if rank == rank_bad:
                        o2 = (self.off + self.n) % self.H
                        self.w = eh_proj.weight[o2:o2 + self.n].contiguous()
                nsx.mod._EhSplit.__init__ = init
        return inj
    scen("rank 3's eh slice holds the next rank's rows (E1 + shas pass; only E3 vs replicated catches it)",
         (True, False), inject=wrong_rows(3), same_schedule_as_clean=True)

    def bitflip_other_block(rank_bad):
        def inj(r, nsx, draft, target, w):
            if r == rank_bad:        # rank_bad's one-shot output: 1 ulp off in rank 0's block (its own block intact)
                class Flip(T.FakeOneshot):
                    def __call__(self, x):
                        out = T.FakeOneshot.__call__(self, x)
                        out._a[0, 0] ^= 1
                        return out
                w._oneshot = Flip(w, DIMS.ONESHOT_MAX)
        return inj
    scen("rank 6's one-shot flips 1 bit in rank 0's block (own checks pass; only the cross-rank sha catches it)",
         (True, False), inject=bitflip_other_block(6), same_schedule_as_clean=True)
    nsw = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env=on)

    def no_world():
        raise RuntimeError("injected: the TP group is not initialised")
    nsw.mod._WORLD_FACTORY = no_world
    rbx = run_boot(nsw, G, rv_timeout=2.0)
    ok, stx = consistent(rbx)
    check("C5 the TP world cannot be created: prepare()'s last-resort guard -> off on all ranks, nothing raised into "
          "load_draft_model, no collective", ok and stx[0] == (False, False) and not errs(rbx)
          and not any(rbx.rv.schedules), f"{stx.get(0)} {errs(rbx)[:1]}")
    # chunked quantisation path in the live boot
    nsc = fresh()
    nsc.mod.QUANT_CHUNK_ROWS = 64
    rbx = run_boot(nsc, G)
    ok, stx = consistent(rbx)
    check("C5 QUANT_CHUNK_ROWS=64 (multi-chunk build + reference GEMMs): ACTIVE everywhere", ok and stx[0] == (True, True))

    # ---- C7 (rev 2) E5 timed inside CUDA graphs; the numerics-only fallback; patch 032's replicated embedding;
    # log lines that never quote other gates' texts
    print("== C7: rev 2 - E5 in CUDA graphs, eh gate numerics, 032 interplay, gate texts")
    prev_eager = {r: {"split": 0.437, "full": 0.295} for r in range(8)}
    rbx = scen("the rev-1 eager numbers on every rank (split 0.437 vs replicated 0.295 ms: rev 1 refused eh here) and the "
               "CUDA-graph timing faster (1.362 vs 2.975 ms per graph) -> eh ACTIVE", (True, True), times=prev_eager,
               schedule_like="clean", tag="C7")
    a0 = [m for (_, _, m) in rbx.logs if "(rank 0/8" in m and NIGHT_ACTIVE.search(m)]
    g7 = gate_counts(rbx.logs)
    check("C7 rev-1-numbers case: the ACTIVE line keeps the eager numbers as information, no 'not active', the gate patterns "
          "count 8 ACTIVE / 0 bad", len(a0) == 1 and "eager [M, split, replicated] ms [[1, 0.437, 0.295], [3, 0.437, "
          "0.295]] (information)" in a0[0] and "not active" not in a0[0] and g7["active_ranks"] == list(range(8))
          and g7["bad"] == 0, (a0[0][-300:] if a0 else "") + str(g7))
    rbx = scen("eager FASTER (split 0.20 vs 0.295 ms) but the CUDA-graph timing SLOWER (one-shot 0.4 ms inside the "
               "graphs: 4.54 vs 3.33 ms per graph) -> eh refused on every rank, head ACTIVE", (True, False),
               times={r: {"split": 0.2, "full": 0.295} for r in range(8)},
               worlds_cfg={r: {"graph_cost": {"oneshot": 0.4}} for r in range(8)}, schedule_like="clean", tag="C7")
    na = [m for (_, _, m) in rbx.logs if "(rank 0/8" in m and NIGHT_ACTIVE.search(m)]
    check("C7 graph-slower case: rank 0's ACTIVE line names the failing CUDA-graph row in its 'not active' suffix",
          len(na) == 1 and "; not active: eh self-test failed on rank(s) [0, 1, 2, 3, 4, 5, 6, 7]: E5 eh M=1 CUDA "
          "graphs: split <= 0.9 x replicated: 4.539 vs 3.328 ms per graph" in na[0], na[0][-300:] if na else "")
    scen("CUDA-graph timing slower on rank 6 only -> eh refused on every rank", (True, False),
         worlds_cfg={6: {"graph_cost": {"oneshot": 0.4}}}, schedule_like="clean", tag="C7")
    scen("CUDA-graph split 0.94x of the replicated graph (not slower, but outside the 0.9 bound) -> eh refused",
         (True, False), worlds_cfg={r: {"graph_cost": {"block": 0.22}} for r in range(8)}, schedule_like="clean",
         tag="C7")
    scen("NCCL host-callback round trip of 3 ms at the first one-shot of every replay (both graphs lead with one) -> "
         "eh ACTIVE (3.76 vs 5.38 ms)", (True, True), worlds_cfg={r: {"graph_cost": {"rtt": 3.0}} for r in range(8)},
         schedule_like="clean", tag="C7")

    def cap_fail(ranks, msg="injected: cudaStreamBeginCapture failed"):
        def hook(g):
            raise RuntimeError(msg)
        return {r: {"capture_hook": hook} for r in ranks}
    scen("rank 3's CUDA-graph capture fails -> capture exchange: no rank replays, eh refused everywhere, head ACTIVE",
         (True, False), worlds_cfg=cap_fail([3]), schedule_like="no_replays", tag="C7")
    scen("every rank's capture fails -> no replays, eh refused, head ACTIVE", (True, False),
         worlds_cfg=cap_fail(range(8)), schedule_like="no_replays", tag="C7")

    def replay_fail(rank, at):
        cnt = {"n": 0}

        def hook(g):
            cnt["n"] += 1
            if cnt["n"] == at:
                raise RuntimeError("injected: cudaGraphLaunch failed")
        return {rank: {"replay_hook": hook}}
    rbx = scen("rank 5's 4th graph replay fails to launch -> the one-shots of its unlaunched replays are issued eagerly "
               "(zeros): schedule == the clean boot's, no hang, eh refused everywhere", (True, False),
               worlds_cfg=replay_fail(5, 4), schedule_like="clean", tag="C7")
    rep5 = rbx.ranks[5].world._oneshot
    check("C7 replay failure: rank 5 replayed fewer one-shots and made up the difference eagerly (same total)",
          rep5.replayed < 240 and rbx.ranks[0].world._oneshot.replayed == 240
          and rep5.calls - 30 == 240 - rep5.replayed, f"replayed {rep5.replayed} eager {rep5.calls}")

    def nth_barrier_fails(n, ranks):
        def inj(r, nsx, draft, target, w):
            if r in ranks:
                orig = w.barrier_bounded
                cnt = {"n": 0}

                def bb(timeout_s):
                    cnt["n"] += 1
                    if cnt["n"] == n:
                        raise RuntimeError("injected monitored_barrier timeout (graph-capture exchange)")
                    return orig(timeout_s)
                w.barrier_bounded = bb
        return inj
    scen("every rank's graph-capture exchange barrier fails -> nobody replays, eh refused, head ACTIVE",
         (True, False), inject=nth_barrier_fails(2, set(range(8))), schedule_like="no_capture_exchange", tag="C7")

    def sync_in_pad(r, nsx, draft, target, w):
        if r == 0:                        # one class patch: every rank's split path syncs the host
            orig = nsx.mod._EhSplit.pad

            def padx(self, y):
                float(y.float().abs().amax().item())      # fine eagerly, forbidden while capturing
                return orig(self, y)
            nsx.mod._EhSplit.pad = padx
    scen("the split path syncs the host (.item()): fine eagerly (E1-E4 pass), forbidden in a capture -> every rank's "
         "capture fails, eh refused, head ACTIVE, no hang", (True, False), inject=sync_in_pad,
         schedule_like="no_replays", tag="C7")
    quoting = ("CUDA OutOfMemoryError; NCCL error: unhandled cuda error; no kernel image; died unexpectedly; "
               "[029] FATAL; INACTIVE; [OK] patch 029 in place")
    rbx = scen("rank 2's capture fails with an exception text quoting readiness / gate texts -> eh refused", (True, False),
               worlds_cfg=cap_fail([2], quoting), schedule_like="no_replays", tag="C7")
    gq = gate_counts(rbx.logs)
    check("C7 quoted exception text is defanged: no [patch-029] line matches a launch/readiness.txt fail rule, no ACTIVE "
          "line contains INACTIVE or '[OK] patch 029 in place', the refusal still shows as 'not active' (8 ranks)",
          gq["readiness_fail"] == [] and gq["active_quoting_inactive"] == 0 and gq["active_ranks"] == list(range(8))
          and not any("[OK] patch 029 in place" in m for (_, _, m) in rbx.logs)
          and sum(1 for (_, _, m) in rbx.logs if NIGHT_ACTIVE.search(m) and "; not active: " in m) == 8, str(gq))

    # ---- (verification of rev 2) the capture-failure scenarios above ran with torch's ordering (an invalidated or
    # failed-to-begin capture leaves its pool registered with the caching allocator; every scen() asserts that vLLM's
    # next empty_cache() works on every rank); here the other ordering, never a double release, a rank whose G_split
    # records fewer one-shots than its replays are budgeted for, replay-time outliers, a split inside the bound
    T.STUCK_ON_INVALIDATED[0] = False
    scen("the split path syncs the host inside the capture, on a torch whose capture_end unregisters the pool before it "
         "raises -> the same refusal on every rank", (True, False), inject=sync_in_pad, schedule_like="no_replays",
         tag="C7")
    scen("every rank's capture fails to begin, same torch -> no replays, eh refused", (True, False),
         worlds_cfg=cap_fail(range(8)), schedule_like="no_replays", tag="C7")
    T.STUCK_ON_INVALIDATED[0] = True
    check("C7 no caching-allocator pool released twice in any scenario so far (releasePool never below zero)",
          not T.POOL_EVENTS, str(T.POOL_EVENTS[:3]))

    def no_oneshot_recorded(rank):
        def inj(r, nsx, draft, target, w):
            if r == 0:                    # one class patch; only `rank`'s split path skips the one-shot while capturing
                orig = nsx.mod._EhSplit.__call__

                def call(self, x):
                    if T.current_world().rank == rank and T.capturing_graph() is not None:
                        return self.pad(self.block(x))
                    return orig(self, x)
                nsx.mod._EhSplit.__call__ = call
        return inj
    scen("rank 6's split path records no one-shot while capturing (its G_split: 1 instead of 9) -> rank 6 reports a "
         "failed capture (recorded vs planned), nobody replays: no mismatch, no hang, eh refused", (True, False),
         inject=no_oneshot_recorded(6), schedule_like="no_replays", tag="C7")

    def outliers(rank, kind, timed, factor):
        def seq(k, n, ms):                # n counts this rank's replays of that graph kind: 12 per M (2 + 10 timed)
            return ms * factor if (k == kind and (n % 12) - 2 in timed) else ms
        return {rank: {"graph_cost_seq": seq}}
    scen("rank 3: 4 of its 10 timed G_split replays 10x slower (preemption noise) -> the median holds, eh ACTIVE",
         (True, True), worlds_cfg=outliers(3, "split", {0, 3, 6, 9}, 10.0), schedule_like="clean", tag="C7")
    scen("rank 3: 6 of its 10 timed G_split replays 10x slower -> its median fails, eh refused on every rank",
         (True, False), worlds_cfg=outliers(3, "split", {0, 1, 2, 3, 4, 5}, 10.0), schedule_like="clean", tag="C7")
    scen("rank 5: 4 of its 10 timed G_full replays 3x faster -> the median holds, eh ACTIVE", (True, True),
         worlds_cfg=outliers(5, "full", {1, 2, 7, 8}, 1.0 / 3.0), schedule_like="clean", tag="C7")
    scen("CUDA-graph split 0.88x of the replicated graph (inside the 0.9 bound) -> eh ACTIVE", (True, True),
         worlds_cfg={r: {"graph_cost": {"block": 0.197}} for r in range(8)}, schedule_like="clean", tag="C7")

    num = {"VLLM_GB10_DRAFT_COMPRESS_EH_GATE": "numerics"}

    def plan_numerics(s):
        seg = _segments(s)
        return (sum(1 for e in s if e[0] == "oneshot") == 30 and seg["exchanges"] == 3
                and sum(1 for e in s if e[0] == "all_gather") == 6)
    rbx = scen("eh gate 'numerics' (operator fallback) with the rev-1 eager numbers -> eh ACTIVE on its numerics; no "
               "capture, no capture exchange (30 one-shots, 3 exchanges)", (True, True), env=num, times=prev_eager,
               schedule_like=plan_numerics, tag="C7")
    an = [m for (_, _, m) in rbx.logs if "(rank 0/8" in m and NIGHT_ACTIVE.search(m)]
    check("C7 numerics gate: the ACTIVE line says the speed gate is off and keeps the eager numbers, no graph made",
          len(an) == 1 and "E5 speed gate off (VLLM_GB10_DRAFT_COMPRESS_EH_GATE=numerics)" in an[0]
          and "[[1, 0.437, 0.295], [3, 0.437, 0.295]]" in an[0]
          and all(o.world._oneshot.captured == 0 for o in rbx.ranks.values()), an[0][-300:] if an else "")
    scen("eh gate 'numerics' + rank 3's eh slice holding the next rank's rows -> eh refused (E3 still gates)",
         (True, False), env=num, inject=wrong_rows(3), tag="C7")
    scen("eh gate ' Numerics ' (case / blanks as the .sh normalises them) -> numerics mode, eh ACTIVE", (True, True),
         env={"VLLM_GB10_DRAFT_COMPRESS_EH_GATE": " Numerics "}, times=prev_eager, schedule_like=plan_numerics, tag="C7")
    rbx = scen("eh gate 'eager' (invalid; the .sh refuses it before any write) on every rank -> eh refused at the "
               "preflight, head ACTIVE", (True, False), env={"VLLM_GB10_DRAFT_COMPRESS_EH_GATE": "eager"}, tag="C7")
    check("C7 invalid eh gate: the reason names the variable",
          any("VLLM_GB10_DRAFT_COMPRESS_EH_GATE='eager'" in m for (_, _, m) in rbx.logs))
    nsg = build_ns(vroot, mtp_p, spec_p, MODULE_PATH, env=on)
    orig_peg = nsg.mod.parse_eh_gate
    nsg.mod.parse_eh_gate = lambda raw: "numerics" if T.current_world().rank == 4 else orig_peg(raw)
    rbx = run_boot(nsg, G, rv_timeout=2.0)
    ok, stx = consistent(rbx)
    check("C7 eh gate 'numerics' on rank 4 only (a mixed M_DOCKER_EXTRA): fingerprint mismatch -> both parts off on "
          "every rank, only the preflight issued, no hang", ok and stx[0] == (False, False) and not errs(rbx)
          and not rbx.rv.hangs and not rbx.rv.mismatches
          and [s[0] for s in rbx.rv.schedules[0]] == ["barrier", "all_gather_object"], str(rbx.rv.schedules[0][:4]))

    def repl_embed(r, nsx, draft, target, w):
        class ReplicatedEmbedding(nsx.VocabParallelEmbedding):
            pass
        ReplicatedEmbedding.__module__ = "vllm._gb10_repl_embed"
        e = ReplicatedEmbedding(DIMS.V, DIMS.H, tp_rank=0, tp_size=1)
        e.disable_tp = True
        e.weight = target.model.embed_tokens.weight
        target.model.embed_tokens = e

    def plan_head_only(s):
        return (not any(e[0] == "oneshot" for e in s) and _segments(s)["exchanges"] == 3
                and sum(1 for e in s if e[0] == "all_gather") == 6)
    rbx = scen("patch 032's replicated embedding (tp_size 1, no all-reduce) shared into the draft -> eh refused at the "
               "preflight on every rank (no eh collective, no capture exchange), head ACTIVE", (True, False),
               inject=repl_embed, schedule_like=plan_head_only, tag="C7")
    check("C7 032 interplay: the reason names patch 032's replicated embedding and the PARTS=head remedy",
          any("ReplicatedEmbedding, tp_size 1" in m and "patch 032" in m
              and "VLLM_GB10_DRAFT_COMPRESS_PARTS=head" in m for (_, _, m) in rbx.logs))
    scen("patch 032's embedding with VLLM_GB10_DRAFT_COMPRESS_PARTS=head (what 032's .sh requires) -> head ACTIVE, eh "
         "not selected, no 'not active'", (True, False), env={"VLLM_GB10_DRAFT_COMPRESS_PARTS": "head"},
         inject=repl_embed, schedule_like=plan_head_only, tag="C7")
    hh = [m for (_, _, m) in LOGS.records if NIGHT_ACTIVE.search(m)]
    check("C7 PARTS=head under 032: 8 ACTIVE lines without a 'not active' suffix (the gate scripts pass)",
          len(hh) == 8 and not any("not active" in m for m in hh) and gate_counts(LOGS.records)["bad"] == 0)

    # ---- C6 memory: when prepare() calls torch.cuda.empty_cache(), a refused FP8 copy is already unreferenced (its
    # 79 MB per rank go back to the driver); an accepted one is alive exactly once per rank (the installed head)
    import gc
    import weakref

    def alive_at_empty_cache(inject):
        nsm = fresh()
        made = []
        orig_init = nsm.mod._Fp8Head.__init__

        def init(self, head):
            orig_init(self, head)
            made.append((T.current_world().rank, weakref.ref(self)))
        nsm.mod._Fp8Head.__init__ = init
        seen = {}

        def empty_cache():
            gc.collect()
            r = T.current_world().rank
            seen[r] = sum(1 for rr, w in list(made) if rr == r and w() is not None)
        nsm.torch.cuda.empty_cache = empty_cache
        rbm = run_boot(nsm, G, inject=inject, rv_timeout=2.0)
        return consistent(rbm), seen, errs(rbm)
    (okc, stc), seen_c, erc = alive_at_empty_cache(None)
    (okr, str_), seen_r, err_ = alive_at_empty_cache(corrupt_marlin(5))
    check("C6 FP8 copy at empty_cache(): accepted -> 1 alive per rank (installed); refused -> 0 (released first)",
          okc and stc[0] == (True, True) and seen_c == {r: 1 for r in range(8)} and not erc
          and okr and str_[0] == (False, True) and seen_r == {r: 0 for r in range(8)} and not err_,
          f"accepted {seen_c} refused {seen_r}")
    return rb


def _fwd_plain(ns, o, M):
    torch = ns.torch
    ly = o.draft.model._mtp_layers[0]
    ids = torch.tensor(np.arange(1, M + 1) % DIMS.V, dtype=torch.int64, device="cuda:0")
    pos = torch.tensor(np.arange(5, 5 + M), dtype=torch.int64, device="cuda:0")
    prev = torch.tensor(np.random.default_rng(100 + M).standard_normal((M, DIMS.H)).astype(np.float32)).to(
        torch.bfloat16).to("cuda:0")
    ly.mtp_block.captured.clear()
    o.draft(ids, pos, prev)
    return ly.mtp_block.captured[-1]


def _logits_plain(ns, o, M):
    torch = ns.torch
    hh = torch.tensor(np.random.default_rng(500 + M).standard_normal((M, DIMS.H)).astype(np.float32)).to(
        torch.bfloat16).to("cuda:0")
    return o.draft.model.compute_logits(hh)


# =====================================================================================================================
# D: real-size numerics of the boot H-checks (one shard)
# =====================================================================================================================
def section_d(vroot, texts, ap):
    print("== D: real-size numerics (19,360 x 4,096 shard, the boot self-test's H-checks)")
    ns = build_ns(vroot, ap.apply("mtp", texts["mtp"]), ap.apply("spec", texts["spec"]), MODULE_PATH,
                  env={"VLLM_GB10_DRAFT_COMPRESS": "1"})
    torch, mod = ns.torch, ns.mod
    rng = np.random.default_rng(42)
    N, K = 19360, 4096
    row_sd = 0.02 * np.exp(rng.normal(0, 0.3, size=(N, 1)))
    Wn = (rng.standard_t(6, size=(N, K)) * row_sd / np.sqrt(1.5)).astype(np.float32)
    head = ns.ParallelLMHead(N * 8, K, tp_rank=0, tp_size=8)
    head.weight.copy_(torch.tensor(Wn))
    norm_w = torch.tensor(np.exp(rng.normal(0, 0.2, size=K)).astype(np.float32)).to(torch.bfloat16).to("cuda:0")
    mod._TIMER = lambda fn, w, i: (fn(), 1.0)[1]
    seq = iter([1.0, 0.45])
    mod._TIMER = lambda fn, w, i: next(seq)
    fh = mod._Fp8Head(head)
    rows = mod._Rows()
    mod._head_local_checks(fh, head, norm_w, rows)
    info = rows.info
    check("D1 real-size H-checks pass (kernel rel, relRMSE, bound, top-1 random / peaked, determinism, speed)",
          rows.ok, f"{rows.fail[:3]} {info}")
    check("D1 numbers as simulated: relRMSE 0.02-0.035, bound ratio < 0.2, top-1 random >= 0.85",
          0.02 <= info.get("rel", 0) <= 0.035 and info.get("bound_ratio", 9) < 0.2
          and info.get("top1_random", 0) >= 0.85, str(info))
    check("D1 FP8 copy size 79.4 MB (+ scales) vs 158.6 MB bf16", abs(fh.bytes / 1e6 - 79.47) < 0.2, f"{fh.bytes / 1e6:.2f}")
    # negative controls on the real shape: a 2x scale error and a row-permuted weight are caught
    fh2 = mod._Fp8Head(head)
    fh2.ws = fh2.ws * 2.0
    rows2 = mod._Rows()
    seq = iter([1.0, 0.45])
    mod._head_local_checks(fh2, head, norm_w, rows2)
    check("D2 negative control: a 2x scale error fails H1/H2", not rows2.ok and any("H1" in f or "H2" in f for f in rows2.fail),
          str(rows2.fail[:2]))


# =====================================================================================================================
# E: mutants
# =====================================================================================================================
MUTANTS = [
    ("eh weight rows of the next rank (own placement)", "self.w = w[self.off:self.off + self.n].contiguous()",
     "self.w = w[(self.off + self.n) % self.H:(self.off + self.n) % self.H + self.n].contiguous()"),
    ("eh pad sides swapped", "return torch.nn.functional.pad(y, (self.left, self.right))",
     "return torch.nn.functional.pad(y, (self.right, self.left))"),
    ("eh one-shot skipped", "return self.oneshot(self.pad(self.block(x)))", "return self.pad(self.block(x))"),
    ("eh eligible ignores the one-shot limit", "and 0 < int(x.shape[0]) <= self.max_tokens and x.is_cuda)",
     "and 0 < int(x.shape[0]) and x.is_cuda)"),
    ("eh eligible never", "        return bool(x.dim() == 2 and x.dtype == torch.bfloat16 and int(x.shape[1]) == self.K",
     "        return False and bool(x.dim() == 2 and x.dtype == torch.bfloat16 and int(x.shape[1]) == self.K"),
    ("head scales reversed", "self.q, self.s = quantize_rows_fp8(w)",
     "self.q, self.s = quantize_rows_fp8(w); self.s = self.s.flip(0) if hasattr(self.s, 'flip') else "
     "self.s[::-1].contiguous()"),
    ("head used beyond HEAD_MAX_M", "and 0 < int(h.shape[0]) <= HEAD_MAX_M and h.is_cuda",
     "and 0 < int(h.shape[0]) and h.is_cuda"),
    ("draft_logits ignores the head identity", "if fh is not None and fh.head is head and fh.eligible(lp, hidden_states):",
     "if fh is not None and fh.eligible(lp, hidden_states):"),
    ("head apply without .contiguous()", "input=x.contiguous(), weight=self.w", "input=x, weight=self.w"),
    ("decide from this rank's row only", "    head = plan.head and all(r.get(\"head_ok\") for r in ver_rows)\n",
     "    head = plan.head and ver_rows[0].get(\"head_ok\")\n"),
    ("commit round dropped",
     "        if com_rows is None or len({(bool(r.get(\"head_on\")), bool(r.get(\"eh_on\"))) for r in com_rows}) != 1:",
     "        if False:"),
    ("self-test guard never issues the missing planned one-shots", "    def flush(self) -> None:\n        while self.budget > 0:",
     "    def flush(self) -> None:\n        while False:"),
    ("plan ignores the fingerprint", "    if len(fps) != 1:\n        return _Plan(False, False, (), (), (), (f\"rank fingerprint",
     "    if False:\n        return _Plan(False, False, (), (), (), (f\"rank fingerprint"),
    ("head self-test rows never fail", "        if not ok:\n            self.fail.append",
     "        if False:\n            self.fail.append"),
    ("H5 perf gate dropped", "rows.check(\"H5 FP8 faster at M=1\", t_fp <= PERF_RATIO_MAX * t_bf",
     "rows.check(\"H5 FP8 faster at M=1\", True"),
    ("prepare() lets exceptions escape", "    except Exception as e:  # noqa: BLE001 - last resort: the image path",
     "    except ZeroDivisionError as e:  # noqa: BLE001 - last resort: the image path"),
    ("batch-invariant refusal dropped", "    if str(os.environ.get(ENV_BATCH_INVARIANT, \"0\")).strip() not in (\"\", \"0\"):",
     "    if False:"),
    ("eh cross-rank sha check dropped", "    if eh and len({tuple(r.get(\"eh_sha\") or ()) for r in ver_rows}) != 1:",
     "    if False:"),
    ("eh eligible ignores the input dtype",
     "        return bool(x.dim() == 2 and x.dtype == torch.bfloat16 and int(x.shape[1]) == self.K",
     "        return bool(x.dim() == 2 and int(x.shape[1]) == self.K"),
    ("E3 bound check disabled", "rows.check(f\"E3 eh M={m} vs replicated\", ratio <= BOUND_RATIO_MAX,",
     "rows.check(f\"E3 eh M={m} vs replicated\", True,"),
    ("refused FP8 copy still referenced at empty_cache()",
     "        fh = loc.fh = None                    # drop the refused FP8 copy BEFORE empty_cache (79 MB per rank)\n",
     "        fh = None\n"),
    ("H3 top-1 floor never fails", "rows.check(\"H3 top-1 random rows\", t1 >= HEAD_TOP1_RANDOM_MIN,",
     "rows.check(\"H3 top-1 random rows\", True,"),
    ("peaked rows: no lead check (near-ties counted)", "        if top - float(row.amax().item()) >= 0.5 * margin:\n",
     "        if True:\n"),
    ("peaked rows: the old 3/4-of-the-candidates rule", "    rows.check(\"H3 peaked rows built\", len(valid) >= HEAD_TEST_PEAKED,",
     "    rows.check(\"H3 peaked rows built\", len(valid) >= (3 * npk) // 4,"),
    ("peaked rows: uncentred lift (along w_v itself)", "        dv = wv - wmean\n", "        dv = wv\n"),
    # ---- rev 2: E5 in CUDA graphs, the numerics fallback, the 032 guard, the log defanging
    ("E5 CUDA-graph gate dropped",
     "rows.check(f\"E5 eh M={m} CUDA graphs: split <= {EH_GRAPH_RATIO_MAX} x replicated\", ok,",
     "rows.check(f\"E5 eh M={m} CUDA graphs: split <= {EH_GRAPH_RATIO_MAX} x replicated\", True,"),
    ("E5 gates on the eager timing again (rev 1)",
     "                rows.info.setdefault(\"eh_eager_ms\", []).append([m, round(t_sp, 4), round(t_full, 4)])\n",
     "                rows.check(f\"E5 eh M={m} split faster\", t_sp <= 1.0 * t_full, f\"{t_sp:.3f} vs {t_full:.3f}\")\n"
     "                rows.info.setdefault(\"eh_eager_ms\", []).append([m, round(t_sp, 4), round(t_full, 4)])\n"),
    ("E5 CUDA-graph bound 0.9 -> 1.0", "EH_GRAPH_RATIO_MAX = 0.9 ", "EH_GRAPH_RATIO_MAX = 1.0 "),
    ("G_full without the leading one-shot",
     "rec[3] = _capture([lambda: guard(lead)] + [lambda: ly.eh_proj(x)] * EH_GRAPH_CALLS, side, dev)",
     "rec[3] = _capture([lambda: ly.eh_proj(x)] * EH_GRAPH_CALLS, side, dev)"),
    ("G_split without the leading one-shot",
     "rec[2] = _capture([lambda: guard(lead)] + [lambda: sp(x)] * EH_GRAPH_CALLS, side, dev)",
     "rec[2] = _capture([lambda: sp(x)] * EH_GRAPH_CALLS, side, dev)"),
    ("replays although a rank failed its capture",
     "            if ok_all:\n                _eh_graph_timing(graphs, guard, rows)",
     "            if True:\n                _eh_graph_timing(graphs, guard, rows)"),
    ("a failed replay launch not made up eagerly",
     "            rows.check(f\"E5 eh M={m} CUDA-graph timing\", False, f\"{type(e).__name__}: {str(e)[:200]}\")\n"
     "        guard.flush()",
     "            rows.check(f\"E5 eh M={m} CUDA-graph timing\", False, f\"{type(e).__name__}: {str(e)[:200]}\")\n"
     "        guard.budget = 0"),
    ("guard without a capture mode", "        if _capturing():\n            if not self._planned(x):",
     "        if False:\n            if not self._planned(x):"),
    ("032 embed check dropped", "    r = _embed_refusal(pred, world)\n    if r:\n        why.append(r)",
     "    r = None"),
    ("boot graphs never reset",
     "        for rec in graphs:\n            _reset_graph(rec[2])\n            _reset_graph(rec[3])\n        graphs = []\n"
     "        if entered:",
     "        graphs = []\n        if entered:"),
    ("numerics gate still captures and exchanges", "        if plan.eh_gate == \"graph\":\n            if not why_local:",
     "        if plan.eh_gate in (\"graph\", \"numerics\"):\n            if not why_local:"),
    ("the side stream left current", "        if entered:\n            try:\n                ctx.__exit__(None, None, None)",
     "        if False:\n            try:\n                ctx.__exit__(None, None, None)"),
    ("capture in global mode", "CAPTURE_ERROR_MODE = \"thread_local\"", "CAPTURE_ERROR_MODE = \"global\""),
    ("log lines quote raw exception text", "    for pat, rep in _DEFANG:\n", "    for pat, rep in ():\n"),
    ("plan ignores the eh gate mode", "                 eh_gate if eh_gate in EH_GATES else EH_GATES[0])",
     "                 EH_GATES[0])"),
    # ---- verification of rev 2: the caching allocator after a failed capture, the recorded one-shot count, _safe
    # against every readiness rule, the inclusive bound
    ("a failed capture leaves its pool registered (no unregister)",
     "        _reset_graph(g)\n        _unregister_pool(pool, dev)\n        raise",
     "        _reset_graph(g)\n        raise"),
    ("unregister releases a pool capture_end already unregistered (double release)",
     "    except Exception:  # noqa: BLE001 - not registered: capture_end unregistered it, nothing to repair\n"
     "        return \"not registered\"",
     "    except Exception:  # noqa: BLE001 - not registered: capture_end unregistered it, nothing to repair\n"
     "        idx = 0"),
    ("recorded one-shot count not checked",
     "                if (c1 - c0, guard.captured - c1) != (EH_GRAPH_CALLS + 1, 1):",
     "                if False:"),
    ("_safe misses the 'Using ... backend' rules", "    (re.compile(r\"\\bUsing\\b\"), \"using\"),\n", ""),
    ("E5 bound exclusive", "and t_s <= EH_GRAPH_RATIO_MAX * t_f)", "and t_s < EH_GRAPH_RATIO_MAX * t_f)"),
    # ---- rev 3 (v0.32.0 port): the Marlin workspace, the op check, the class' module path
    ("FP8 head passes workspace=None (the workspace manager's locks)",
     "input=x.contiguous(), weight=self.w, weight_scale=self.ws, workspace=self.workspace,",
     "input=x.contiguous(), weight=self.w, weight_scale=self.ws, workspace=None,"),
    ("FP8 head workspace sized one block per SM (the day-0 default)",
     "int(getattr(MU, \"MARLIN_MAX_BLOCKS_PER_SM\", 4))", "1"),
    ("FP8 head workspace not allocated (reuses a stale attribute)", "        self.workspace = _make_workspace(w.device)\n",
     "        self.workspace = getattr(L, \"workspace\", None)\n"),
    ("torch.ops.vllm.marlin_gemm presence not checked",
     "        if not hasattr(torch.ops.vllm, \"marlin_gemm\"):      # v0.32.0: apply_fp8_marlin_linear -> torch.ops.vllm.marlin_gemm\n",
     "        if False:\n"),
    ("_find_mtp accepts the old module path", "            \"models.glm5next.common.mtp\"):", "            \"mtp\"):"),
]


def section_e(vroot, work, texts, ap, G):
    print("== E: planted mutants (each must make sections B/C fail)")
    src = MODULE_PATH.read_text()
    caught = 0
    for name, old, new in MUTANTS:
        if src.count(old) != 1:
            check(f"E mutant '{name}': anchor found once in the module", False, f"{src.count(old)}x")
            continue
        mp = work / "mutants" / (re.sub(r"[^a-z0-9]+", "_", name) + ".py")
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(src.replace(old, new, 1))
        env = dict(os.environ, P029_MODULE_OVERRIDE=str(mp), P029_QUIET="1")
        r = subprocess.run([sys.executable, str(Path(__file__)), "--vllm-root", str(vroot), "--work",
                            str(work / "mut_run"), "--quick", "--no-a"], env=env, capture_output=True, text=True,
                           timeout=1800)
        fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAIL ")]
        crash = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()][-1:] if not fails else []
        ok = r.returncode != 0
        caught += ok
        check(f"E mutant '{name}' caught", ok,
              fails[0][5:110] if fails else (f"suite crashed: {crash[0][:100]}" if crash else f"rc {r.returncode}"))
    print(f"MUTANTS caught {caught}/{len(MUTANTS)}")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--vllm-root", required=True)
    ap_.add_argument("--work", default=None)
    ap_.add_argument("--quick", action="store_true")
    ap_.add_argument("--mutants", action="store_true")
    ap_.add_argument("--no-a", action="store_true")
    args = ap_.parse_args()
    QUIET["on"] = os.environ.get("P029_QUIET") == "1"
    vroot = Path(args.vllm_root)
    work = Path(args.work or tempfile.mkdtemp(prefix="p029-cpu-"))
    work.mkdir(parents=True, exist_ok=True)
    ap = load_file_module("p029_apply", APPLIER_PATH)
    texts = {k: (vroot / rel).read_text() for k, rel in REL.items()}
    if not args.no_a:
        section_a(vroot, work, ap)
    G = global_weights()
    section_b(vroot, texts, ap)
    section_b11(vroot, texts, ap)
    section_c(vroot, texts, ap, G)
    if not args.quick:
        section_d(vroot, texts, ap)
    if args.mutants:
        section_e(vroot, work, texts, ap, G)
    print(f"\n{RESULTS['pass']} passed, {RESULTS['fail']} failed")
    if RESULTS["failed"]:
        print("FAILED: " + " | ".join(RESULTS["failed"][:30]))
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())
