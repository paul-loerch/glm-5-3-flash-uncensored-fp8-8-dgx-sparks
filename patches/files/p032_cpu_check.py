#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-032 gb10-repl-embed] CPU checks, vLLM v0.32.0 port (Mac / any CPU: python3 >= 3.10 + numpy; NO torch, NO vLLM install).

  python3 -I -B files/p032_cpu_check.py --vllm-root <v0.32.0 source copy>/vllm [--work DIR] [--mutants]
  python3 -I -B files/p032_cpu_check.py --vllm-root <...>/vllm --print-pins      (prints every pin line, writes nothing)
  python3 -I -B files/p032_cpu_check.py --vllm-root <...>/vllm --write-pins      (rewrites the pin lines of the .sh in place)

--vllm-root must hold the v0.32.0 files (sha-checked against the .sh pins): models/glm5next/common/{model,mtp}.py,
model_executor/layers/vocab_parallel_embedding.py, v1/worker/gpu/spec_decode/eagle/utils.py,
model_executor/model_loader/{utils,base_loader,weight_tying}.py, plus (unpinned, exec'd / extracted)
model_executor/utils.py, model_executor/layers/quantization/base_config.py and model_executor/custom_op.py.
The REAL files run over files/p032_testlib.py's numpy-backed fake torch; the fused `ops.vocab_parallel_embedding` kernel is
emulated from csrc/vocab_parallel_embedding_kernels.cu (v0.32.0: int64 range compares, a byte copy of the owned row,
`vec_t()` = all-zero bytes for every other id).

  A  pins / bases / applier / static enumeration: the v0.32.0 originals == the .sh pins; the model.py bases (the original,
     and the 025 result when files/p025_apply.py exists, else an explicit SKIP) ; p032_apply apply / check / revert ==
     the pinned results, revert byte-exact; AST placement and refusals; the gate block alone; the vendored shas in the .sh;
     the chain: among the .sh files < 032 only 025 may write model.py, mtp.py only by 029 (== MTP_P029), nothing
     writes a 032 dependency; the 033 .sh (when present) pins this patch's results as its bases.
  B  module units on the REAL vocab_parallel_embedding.py (fused kernel branch AND masked fallback branch), 8 threaded
     ranks: construction envelope; the REAL weight_loader loads the FULL table; exactness (every id + out-of-range ids, a
     table with -0.0 / +0.0 / subnormals: the unpatched lookup + all-reduce (one-shot and ring arithmetic) == canon(table
     row) bitwise, and != the raw row exactly at the -0.0 elements, which is why the boot canonicalises); boot ACTIVE with
     identical collective schedules; fail-closed scenarios -> INACTIVE on all 8 ranks, no hang, the layer == the
     unpatched layer (class and EVERY constructor attribute, weight bytes incl. -0.0, attrs) and its forward == the
     unpatched one; pending mode == the unpatched one; plan properties at the real size (154,880 x 4,096, TP 8);
     log-format hygiene; v0.32.0 specifics (fused-kernel decision, the REAL loader's new steps, word_embeddings_untied,
     supports_pre_processed_weights).
  C  the REAL patched model.py (each base) + REAL common/mtp.py + REAL eagle/utils.py + REAL process_weights_after_loading,
     8 threaded ranks: gate unset == the unpatched layer; gate 1 -> ACTIVE, the REAL Glm5NextModel.forward output == the
     unpatched embedding bitwise with no all-reduce, the REAL load_eagle_model (maybe_share_target_embed) hands the draft
     the TARGET's (replicated) embed_tokens, the REAL Glm5NextMTP.forward embeds with it, the lm_head stays a
     vocab-parallel ParallelLMHead shared into shared_head.head; CUDA-graph replay (tape emulation) of the REAL forward;
     a forced boot failure -> the unpatched path in both the target and the MTP; gate set + module missing -> ImportError.
  D  (--mutants) planted bugs in a copy of the module; each must make B / C fail.
Exit 0 = ALL PASS (SKIP lines are counted separately and are never passes).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
SH = PDIR / "032-gb10-repl-embed.sh"
MODULE_PATH = HERE / "p032_gb10_repl_embed.py"
APPLIER_PATH = HERE / "p032_apply.py"
P025_APPLIER = HERE / "p025_apply.py"          # absent until the 025 port exists
P029_APPLIER = HERE / "p029_apply.py"          # patch 029's applier (the MTP_P029 pin is derived from it)
READINESS_FIXTURES = (HERE / "testdata" / "readiness.txt", PDIR.parent / "launch" / "readiness.txt")
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import p032_testlib as T  # noqa: E402

REL = {
    "model": "models/glm5next/common/model.py",
    "mtp": "models/glm5next/common/mtp.py",
    "vpe": "model_executor/layers/vocab_parallel_embedding.py",
    "eagle": "v1/worker/gpu/spec_decode/eagle/utils.py",
    "lutil": "model_executor/model_loader/utils.py",
    "bload": "model_executor/model_loader/base_loader.py",
    "wtie": "model_executor/model_loader/weight_tying.py",
    "mutils": "model_executor/utils.py",
    "bcfg": "model_executor/layers/quantization/base_config.py",
    "cop": "model_executor/custom_op.py",
}
PIN_OF = {"model": "MODEL_ORIG", "mtp": "MTP_ORIG", "vpe": "VPE_ORIG", "eagle": "EAGLE_ORIG", "lutil": "LUTIL_ORIG",
          "bload": "BLOAD_ORIG", "wtie": "WTIE_ORIG"}
RESULTS = {"pass": 0, "fail": 0, "skip": 0, "failed": [], "skipped": []}
QUIET = {"on": False}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def check(name, ok, info=""):
    ok = bool(ok)
    RESULTS["pass" if ok else "fail"] += 1
    if not ok:
        RESULTS["failed"].append(name)
    if not QUIET["on"] or not ok:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return ok


def skip(name, why):
    RESULTS["skip"] += 1
    RESULTS["skipped"].append(name)
    if not QUIET["on"]:
        print(f"SKIP {name}  [{why}]", flush=True)


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin_raw(name, sh=None):
    m = re.search(rf"^{name}=(\S+)$", (sh or SH).read_text(), re.M)
    if not m:
        raise SystemExit(f"no pin {name} in {sh or SH}")
    return m.group(1)


def pin(name, sh=None):
    """The 64-hex pin, or None while the line is a placeholder (pending-025-port / pending-029-port)."""
    v = pin_raw(name, sh)
    return v if HEX64.match(v) else None


def load_file_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# =====================================================================================================================
# the stub vLLM namespace; the REAL files are exec'd into it
# =====================================================================================================================
TLS = threading.local()
ENV_KEYS = ("VLLM_GB10_REPL_EMBED", "VLLM_GB10_MHC_SP", "VLLM_BATCH_INVARIANT")


def amod(name, **attrs):
    """A stub package whose missing attributes auto-stub (so `from X import anything` works)."""
    m = T._AutoModule(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    return m


FUSED = {"calls": 0}


def make_fused_kernel(tm):
    """`ops.vocab_parallel_embedding` as csrc/vocab_parallel_embedding_kernels.cu (v0.32.0) computes it: ids compared as int64
    against the org / added ranges, the owned row copied byte for byte, every other row all-zero bytes (+0.0 in bf16)."""
    def vocab_parallel_embedding(input_ids, weight, org_start, org_end, num_org_vocab_padding, added_start, added_end):
        if weight.dim() != 2 or input_ids.dim() != 1:
            raise RuntimeError("vocab_parallel_embedding: weight must be 2D, input_ids 1D")
        if not (input_ids.is_contiguous() and weight.is_contiguous()):
            raise RuntimeError("vocab_parallel_embedding requires contiguous tensors")
        if input_ids.dtype.name not in ("int32", "int64"):
            raise RuntimeError("input_ids must be int32 or int64")
        FUSED["calls"] += 1
        T._counter("fused_embedding")
        added_row_offset = org_end - org_start + num_org_vocab_padding - added_start

        def fn(ids, w):
            i = np.asarray(ids).astype(np.int64)
            in_org = (i >= org_start) & (i < org_end)
            in_added = (i >= added_start) & (i < added_end)
            row = np.where(in_org, i - org_start, i + added_row_offset)
            valid = in_org | in_added
            out = np.zeros((i.shape[0], w.shape[1]), dtype=w.dtype)
            out[valid] = w[row[valid]]
            return out
        return T.Tensor._op(fn, [input_ids, weight], weight.dtype, weight.device)
    return vocab_parallel_embedding


def build_ns(vroot: Path, model_text: str | None, module_path: Path | None, env: dict | None = None,
             with_model: bool = True, cuda: bool = True, extra_mtp_stub: bool = True):
    """Fresh fake torch + stub vLLM + the REAL files. env is applied BEFORE model.py is exec'd (its gate is read at
    import). module_path None = the patch module is NOT installed. cuda=False makes current_platform.is_cuda() False (the
    masked-gather branch of VocabParallelEmbedding.forward)."""
    for k in [k for k in sys.modules if k in ("vllm", "transformers") or k.startswith(("vllm.", "transformers."))]:
        del sys.modules[k]
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in (env or {}).items():
        os.environ[k] = v
    tm = T.install_fake_torch()
    tm.set_default_dtype(tm.bfloat16)
    T.LOGS.clear()
    FUSED["calls"] = 0
    logger = T.make_logger()
    mod = T.mod
    mod("vllm", pkg=True)
    mod("vllm.logger", init_logger=lambda name: logger)
    mod("vllm.envs", VLLM_BATCH_INVARIANT=False, VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=False,
        VLLM_LOG_MODEL_INSPECTION=False)

    class VllmConfig:
        pass

    class ModelConfig:
        pass

    class ParallelConfig:
        pass

    class LoadConfig:
        pass

    def replace(obj, **kw):
        o = NS(**vars(obj))
        for k, v in kw.items():
            setattr(o, k, v)
        return o
    mod("vllm.config", VllmConfig=VllmConfig, ModelConfig=ModelConfig, ParallelConfig=ParallelConfig,
        LoadConfig=LoadConfig, get_current_vllm_config_or_none=lambda: getattr(TLS, "vcfg", None),
        get_current_vllm_config=lambda: TLS.vcfg, replace=replace,
        set_current_vllm_config=lambda c: contextlib.nullcontext(), get_cached_compilation_config=lambda: NS())

    def divide(a, b):
        assert a % b == 0, f"{a} is not divisible by {b}"
        return a // b

    def w():
        return T.current_world()

    def pp_group():
        ww = w()
        return NS(world_size=ww.pp_size, is_first_rank=True, is_last_rank=True, rank_in_group=0)

    def tp_group():
        ww = w()
        return NS(rank_in_group=ww.rank, world_size=ww.size, cpu_group=("gloo", ww.rank), unique_name="tp:0",
                  all_reduce=ww.all_reduce)

    def tp_all_reduce(t):
        return w().all_reduce(t)

    def tp_all_gather(*a, **k):
        raise RuntimeError("tensor_model_parallel_all_gather is not used here")

    class GroupCoordinator:
        pass
    mod("vllm.distributed", pkg=True, divide=divide, get_tensor_model_parallel_rank=lambda: w().rank,
        get_tensor_model_parallel_world_size=lambda: w().size, tensor_model_parallel_all_reduce=tp_all_reduce,
        tensor_model_parallel_all_gather=tp_all_gather, get_pp_group=pp_group, get_tp_group=tp_group,
        get_ep_group=lambda: NS(world_size=1), GroupCoordinator=GroupCoordinator)
    mod("vllm.distributed.parallel_state", get_tp_group=tp_group, get_pp_group=pp_group)
    plat = NS(device_type="cuda", is_cpu=lambda: False, is_cuda_alike=lambda: True, simple_compile_backend="inductor",
              use_sync_weight_loader=lambda: False, is_integrated_gpu=lambda i=None: True, is_cuda=lambda: cuda,
              is_xpu=lambda: False, is_rocm=lambda: False)
    mod("vllm.platforms", current_platform=plat)
    mod("vllm.utils", pkg=True)
    mod("vllm.utils.torch_utils", is_torch_equal_or_newer=lambda v: True,
        get_accelerator_view_from_cpu_tensor=lambda t: t)
    mod("vllm.utils.mem_utils", release_device_memory_under_pressure=lambda d: T._counter("release_pressure"))
    mod("vllm.utils.platform_utils", is_pin_memory_available=lambda: False)
    mod("vllm._custom_ops", vocab_parallel_embedding=make_fused_kernel(tm))
    mod("vllm.model_executor", pkg=True)
    mutils = T.exec_real("vllm.model_executor.utils", (vroot / REL["mutils"]).read_text(), str(vroot / REL["mutils"]),
                         "vllm.model_executor")
    cop = mod("vllm.model_executor.custom_op", logger=logger, nn=tm.nn, torch=tm)
    exec(compile(T.extract(vroot / REL["cop"], ["op_registry", "op_registry_oot", "PluggableLayer"]),
                 str(vroot / REL["cop"]), "exec"), cop.__dict__)

    class BasevLLMParameter(tm.Parameter):
        pass
    mod("vllm.model_executor.parameter", BasevLLMParameter=BasevLLMParameter)
    mod("vllm.model_executor.determinism", pkg=True)
    mod("vllm.model_executor.determinism.batch_invariant", linear_batch_invariant=lambda x, w_, b=None: tm.nn.functional
        .linear(x, w_, b))
    mod("vllm.model_executor.layers", pkg=True)
    mod("vllm.model_executor.layers.utils", dispatch_unquantized_gemm=lambda *a: (lambda layer, x, w_, b=None:
                                                                                  tm.nn.functional.linear(x, w_, b)))
    qpkg = mod("vllm.model_executor.layers.quantization", pkg=True)
    bcfg = mod("vllm.model_executor.layers.quantization.base_config", torch=tm, nn=tm.nn)
    import abc
    import inspect
    bcfg.ABC, bcfg.abstractmethod, bcfg.inspect = abc.ABC, abc.abstractmethod, inspect

    class QuantizationConfig:
        pass
    bcfg.QuantizationConfig = QuantizationConfig
    qpkg.QuantizationConfig = QuantizationConfig
    exec(compile(T.extract(vroot / REL["bcfg"], ["QuantizeMethodBase", "method_has_implemented_embedding"]),
                 str(vroot / REL["bcfg"]), "exec"), bcfg.__dict__)
    bcfg.resolve_quant_method = lambda qc, layer, prefix: qc.get_quant_method(layer, prefix=prefix)
    mod("vllm.model_executor.layers.quantization.utils", pkg=True)
    mod("vllm.model_executor.layers.quantization.utils.fp8_utils", is_fp8=lambda x: x is tm.float8_e4m3fn)
    vpe = T.exec_real("vllm.model_executor.layers.vocab_parallel_embedding", (vroot / REL["vpe"]).read_text(),
                      str(vroot / REL["vpe"]), "vllm.model_executor.layers")

    class RMSNorm(tm.nn.Module):
        def __init__(self, hidden, eps=1e-5, **kw):
            super().__init__()
            self.variance_epsilon = eps
            self.weight = tm.nn.Parameter(tm.ones(hidden, dtype=tm.bfloat16, device="cuda:0"), requires_grad=False)

        def forward(self, x, residual=None):
            return x if residual is None else (x, residual)
    mod("vllm.model_executor.layers.layernorm", RMSNorm=RMSNorm)

    class LogitsProcessor(tm.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
    mod("vllm.model_executor.layers.logits_processor", LogitsProcessor=LogitsProcessor)
    mod("vllm.model_executor.layers.fused_moe", pkg=True)
    mod("vllm.model_executor.layers.fused_moe.utils", is_model_fused_shared_expert_compatible=lambda *a, **k: False,
        resolve_layer_fused_shared_expert=lambda *a, **k: False)
    mod("vllm.model_executor.model_loader", pkg=True, get_model=lambda vllm_config=None, model_config=None: TLS.draft)

    def default_weight_loader(param, loaded_weight, **kw):
        param.data.copy_(loaded_weight)
    mod("vllm.model_executor.model_loader.weight_utils", default_weight_loader=default_weight_loader,
        maybe_remap_kv_scale_name=lambda n, p: n)
    wtie = T.exec_real("vllm.model_executor.model_loader.weight_tying", (vroot / REL["wtie"]).read_text(),
                       str(vroot / REL["wtie"]), "vllm.model_executor.model_loader")

    class OnlineLinearBase:
        pass

    class OnlineMoEMethodBase:
        pass
    mod("vllm.model_executor.layers.quantization.online", pkg=True)
    mod("vllm.model_executor.layers.quantization.online.fp8", OnlineLinearBase=OnlineLinearBase)
    mod("vllm.model_executor.layers.quantization.online.moe_base", OnlineMoEMethodBase=OnlineMoEMethodBase)
    lutil = mod("vllm.model_executor.model_loader.utils", torch=tm, nn=tm.nn, envs=sys.modules["vllm.envs"],
                QuantizeMethodBase=bcfg.QuantizeMethodBase, is_deferred_attention_layer=lambda m: False,
                ModelConfig=ModelConfig, nullcontext=contextlib.nullcontext,
                maybe_retie_word_embeddings=wtie.maybe_retie_word_embeddings,
                is_weights_pre_processed=mutils.is_weights_pre_processed,
                release_device_memory_under_pressure=sys.modules["vllm.utils.mem_utils"]
                .release_device_memory_under_pressure, is_pin_memory_available=lambda: False,
                get_accelerator_view_from_cpu_tensor=lambda t: t, set_torchao_reload_attrs=lambda *a: None,
                contextmanager=contextlib.contextmanager, get_draft_load_config=lambda vc: vc.load_config)
    exec(compile(T.extract(vroot / REL["lutil"], ["process_weights_after_loading", "device_loading_context"]),
                 str(vroot / REL["lutil"]), "exec"), lutil.__dict__)
    mod("vllm.model_executor.models", pkg=True)

    def make_layers(n, fn, prefix=""):
        return 0, 0, tm.nn.ModuleList([])

    class PPMissingLayer(tm.nn.Module):
        pass
    mod("vllm.model_executor.models.utils", make_layers=make_layers,
        maybe_prefix=lambda p, n: n if not p else f"{p}.{n}", PPMissingLayer=PPMissingLayer,
        is_pp_missing_parameter=lambda n, m: False, AutoWeightsLoader=T._Stub("AutoWeightsLoader"),
        init_vllm_registered_model=T._Stub("init_vllm_registered_model"),
        sequence_parallel_chunk=T._Stub("sequence_parallel_chunk"))

    class EagleModelMixin:
        aux_hidden_state_layers: tuple = ()

    def _empty(n):
        return type(n, (), {})
    mod("vllm.model_executor.models.interfaces", HasInnerState=_empty("HasInnerState"), IsHybrid=_empty("IsHybrid"),
        MixtureOfExperts=_empty("MixtureOfExperts"), SupportsPP=_empty("SupportsPP"),
        SupportsEagle3=_empty("SupportsEagle3"), EagleModelMixin=EagleModelMixin, SupportsQuant=_empty("SupportsQuant"))

    class DeepseekV2MixtureOfExperts:
        def extract_moe_parameters(self, example):
            self.example_moe = example
    mod("vllm.model_executor.models.deepseek_v2", DeepseekV2MixtureOfExperts=DeepseekV2MixtureOfExperts,
        _get_moe_router_dtype=lambda *a, **k: tm.float32)

    class SharedHead(tm.nn.Module):
        """deepseek_mtp.SharedHead (v0.32.0): defer_lm_head=True leaves `head` None until load_eagle_model shares the
        target's lm_head into it."""

        def __init__(self, config, prefix="", quant_config=None, *, defer_lm_head=False):
            super().__init__()
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = None if defer_lm_head else vpe.ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.shared_head.head")
    mod("vllm.model_executor.models.deepseek_mtp", SharedHead=SharedHead)

    class Glm4vForConditionalGeneration(tm.nn.Module):
        pass
    mod("vllm.model_executor.models.glm4_1v", Glm4vForConditionalGeneration=Glm4vForConditionalGeneration,
        Glm4vDummyInputsBuilder=_empty("Glm4vDummyInputsBuilder"))
    mod("vllm.sequence", IntermediateTensors=dict)
    mod("vllm.lora", pkg=True)
    mod("vllm.lora.layers", pkg=True)
    mod("vllm.lora.layers.base", BaseLayerWithLoRA=_empty("BaseLayerWithLoRA"))
    mod("vllm.compilation", pkg=True)
    mod("vllm.compilation.backends", set_model_tag=lambda tag: contextlib.nullcontext())
    for p in ("vllm.models", "vllm.models.glm5next", "vllm.models.glm5next.common", "vllm.models.glm5next.nvidia",
              "vllm.models.glm5next.nvidia.ops", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
              "vllm.v1.worker.gpu.spec_decode", "vllm.v1.worker.gpu.spec_decode.eagle"):
        mod(p, pkg=True)
    mod("vllm.v1.worker.gpu.spec_decode.utils", get_pp_safe_draft_load_config=lambda lc: lc)
    # transformers (not a vllm.* import: the auto-stub finder does not cover it)
    mod("transformers", pkg=True, Glm5NextTextConfig=type("Glm5NextTextConfig", (), {}))

    def fused_eh_norm(positions, inputs_embeds, previous_hidden, enorm_w, hnorm_w, eps):
        TLS.eh_inputs = getattr(TLS, "eh_inputs", []) + [inputs_embeds]
        return tm.cat([inputs_embeds, previous_hidden], dim=1)
    mod("vllm.models.glm5next.nvidia.ops.fused_eh_norm", fused_eh_norm=fused_eh_norm)
    pmod = None
    if module_path is not None:
        pmod = T.exec_real("vllm._gb10_repl_embed", Path(module_path).read_text(), str(module_path), "vllm")
    model = mtp = eagle = None
    if with_model:
        T.enable_autostubs()
        try:
            model = T.exec_real("vllm.models.glm5next.common.model", model_text, str(vroot / REL["model"]),
                                "vllm.models.glm5next.common")
        finally:
            T.disable_autostubs()

        class StubMoE(tm.nn.Module):
            pass

        class StubMLAAttention(tm.nn.Module):
            def __init__(self):
                super().__init__()
                self.mla_attn = NS(topk_indices_buffer=None, skip_topk=False)

        class StubDecoderLayer(tm.nn.Module):
            def __init__(self, vllm_config=None, config=None, layer_idx=0, prefix="", topk_indices_buffer=None,
                         is_mtp_layer=False):
                super().__init__()
                self.mlp = StubMoE()
                self.self_attn = StubMLAAttention()

            def forward(self, positions, hidden_states, residual=None):
                return hidden_states, hidden_states, None, None
        real_layer, real_attn = model.Glm5NextDecoderLayer, model.Glm5NextMLAAttention
        model.Glm5NextDecoderLayer, model.Glm5NextMLAAttention = StubDecoderLayer, StubMLAAttention
        mtp = T.exec_real("vllm.models.glm5next.common.mtp", (vroot / REL["mtp"]).read_text(), str(vroot / REL["mtp"]),
                          "vllm.models.glm5next.common")
        model.Glm5NextDecoderLayer, model.Glm5NextMLAAttention = real_layer, real_attn
        eagle = T.exec_real("vllm.v1.worker.gpu.spec_decode.eagle.utils", (vroot / REL["eagle"]).read_text(),
                            str(vroot / REL["eagle"]), "vllm.v1.worker.gpu.spec_decode.eagle")
    return NS(torch=tm, vpe=vpe, mod=pmod, model=model, mtp=mtp, eagle=eagle, lutil=lutil, mutils=mutils,
              bcfg=bcfg, cop=cop, wtie=wtie, logger=logger, plat=plat)


# =====================================================================================================================
# test data / per-rank helpers
# =====================================================================================================================
W = 8


def make_ckpt(V, H, seed=11, negzero=True, subnormal=True, nan=False):
    rng = np.random.default_rng(seed)
    bits = T.f32_to_bf16_bits((rng.standard_normal((V, H)) * 0.02).astype(np.float32))
    if negzero:
        bits[V - 60, :] = 0x8000                 # an all -0.0 row (a never-trained token)
        bits[V - 59, : H // 2] = 0x0000          # +0.0 zeros
        bits[3, 5] = 0x8000
        bits[V // 2 + 1, H - 1] = 0x8000         # in another rank's shard
        bits[(V // W) * 5 + 2, 1] = 0x8000
    if subnormal:
        bits[77 % V, 2] = 0x0001                 # smallest positive bf16 subnormal
        bits[(V // W) * 6 + 7, H // 3] = 0x8003  # a negative subnormal
    if nan:
        bits[5, 5] = 0x7FC0
    return bits


def cpu_tensor(ns, bits):
    return ns.torch.Tensor(np.array(bits, copy=True), ns.torch.bfloat16, ns.torch.device("cpu"))


def canon_bits(bits):
    b = np.array(bits, copy=True)
    b[b == 0x8000] = 0x0000
    return b


def mk_cfg(V, H, tie=False):
    return NS(vocab_size=V, hidden_size=H, tie_word_embeddings=tie, num_attention_heads=W, num_hidden_layers=2,
              num_nextn_predict_layers=1, index_topk=16, index_kpool=4, index_kpool_always_select_tail=True,
              rms_norm_eps=1e-5, n_routed_experts=None, n_shared_experts=None, n_group=1, mla_use_nope=False,
              qk_rope_head_dim=0, layer_types=None, pad_token_id=V - 60, eos_token_id=[V - 60, V - 53])


def mk_vcfg(cfg, lora=None, untied=False):
    mc = NS(hf_config=cfg, hf_text_config=cfg, quantization="fp8", dtype="bfloat16", model="p032-test",
            word_embeddings_untied_by_checkpoint=untied)
    return NS(model_config=mc, lora_config=lora, quant_config=None, load_config=NS(),
              scheduler_config=NS(max_num_batched_tokens=16), parallel_config=NS(use_sequence_parallel_moe=False),
              speculative_config=NS(draft_model_config=mc, kv_cache_dtype=None, apply_draft_overrides=lambda vc: vc))


def ids_all_plus_oob(V):
    return list(range(V)) + [-1, -2, V, V + 1, -V, (1 << 31) - 1, -(1 << 31)]


def run8(fn, timeout=120.0):
    return T.run_ranks(W, fn, timeout=timeout)


def setup_rank(rv, r, H, vcfg, ftz=False, oneshot_tokens=8):
    w = T.World(rv, r, W, oneshot_max=oneshot_tokens * H, ftz=ftz)
    T.set_current_world(w)
    TLS.vcfg = vcfg
    TLS.eh_inputs = []
    return w


PREFIX = "model.embed_tokens"


def image_layer(ns, V, H, bits, prefix=PREFIX):
    """The unpatched layer: the REAL v0.32.0 VocabParallelEmbedding built at TP=8 on this rank's thread."""
    tm = ns.torch
    with tm.device("cuda"):
        img = ns.vpe.VocabParallelEmbedding(V, H, prefix=prefix)
    img.weight.weight_loader(img.weight, cpu_tensor(ns, bits))
    return img


def image_lookup(img, ids, ns, chunk=4096):
    """The unpatched forward (REAL VocabParallelEmbedding.forward: fused lookup + world all-reduce) in chunks."""
    tm = ns.torch
    outs = []
    for s in range(0, len(ids), chunk):
        t = tm.tensor(ids[s:s + chunk], dtype=tm.int32, device="cuda:0")
        outs.append(np.array(img(t)._a, copy=True))
    return np.concatenate(outs, axis=0)


def local_lookup(layer, ids, ns, chunk=4096):
    tm = ns.torch
    outs = []
    for s in range(0, len(ids), chunk):
        t = tm.tensor(ids[s:s + chunk], dtype=tm.int32, device="cuda:0")
        outs.append(np.array(layer(t)._a, copy=True))
    return np.concatenate(outs, axis=0)


def build_layer(ns, V, H, bits, cfg, load=True, prefix=PREFIX):
    tm = ns.torch
    with tm.device("cuda"):
        cls = ns.mod.embedding_class(cfg)
        lay = cls(V, H, prefix=prefix)
    if load:
        lay.weight.weight_loader(lay.weight, cpu_tensor(ns, bits))
    return lay


def post_load(ns, layer, untied=False):
    h = ns.torch.nn.Module()
    h.embed_tokens = layer
    ns.lutil.process_weights_after_loading(h, NS(quantization="fp8", word_embeddings_untied_by_checkpoint=untied),
                                           ns.torch.device("cuda"))
    return h.embed_tokens


def mode_of(ns, layer):
    if type(layer) is ns.vpe.VocabParallelEmbedding:
        return "image"
    return getattr(layer, "_p032_mode", "?")


def image_state(layer):
    """EVERY public instance attribute v0.32.0's VocabParallelEmbedding.__init__ leaves on the layer (not just the TP
    ones), the weight's loader attrs and shape: a restored layer must equal a freshly built unpatched layer."""
    st = {}
    for k, v in layer.__dict__.items():
        if k.startswith("_") or k == "training":
            continue
        if k == "shard_indices":
            st[k] = dict(vars(v))
        elif k == "quant_method":
            st[k] = type(v).__name__
        else:
            st[k] = v
    st["weight_shape"] = tuple(layer.weight.shape)
    st["weight_attrs"] = (getattr(layer.weight, "input_dim", None), getattr(layer.weight, "output_dim", None),
                          getattr(getattr(layer.weight, "weight_loader", None), "__func__", None))
    return st


# =====================================================================================================================
# A. pins / bases / applier / static enumeration
# =====================================================================================================================
def _mkroot(work: Path, vroot: Path, name: str) -> Path:
    r = Path(tempfile.mkdtemp(prefix=f"{name}.", dir=work))
    (r / REL["model"]).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vroot / REL["model"], r / REL["model"])
    return r


def p025_base(vroot: Path, work: Path):
    """-> (text, why): the model.py patch 025's applier hands over (None + the reason while the 025 port is absent)."""
    if not P025_APPLIER.exists():
        return None, f"{P025_APPLIER.name} absent (the 025 port does not exist yet)"
    r = _mkroot(work, vroot, "p025")
    p = r / REL["model"]
    rc = subprocess.run([sys.executable, "-I", "-B", str(P025_APPLIER), "apply", str(p)], capture_output=True, text=True)
    if rc.returncode != 0:
        return None, f"p025_apply.py apply refused ({(rc.stderr or rc.stdout).strip()[-160:]})"
    return p.read_text(), ""


def p029_mtp_result(vroot: Path):
    """-> (sha, why): the mtp.py sha patch 029's applier produces from the v0.32.0 original."""
    if not P029_APPLIER.exists():
        return None, f"{P029_APPLIER.name} absent"
    rc = subprocess.run([sys.executable, "-I", "-B", str(P029_APPLIER), "check", "mtp", str(vroot / REL["mtp"])],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        return None, f"p029_apply.py check mtp refused ({(rc.stderr or rc.stdout).strip()[-160:]})"
    return rc.stdout.strip(), ""


def compute_pins(vroot: Path, work: Path, ap) -> dict:
    """Every pin line of the .sh, recomputed from the files on disk (--print-pins / --write-pins)."""
    out = {}
    for k, p in PIN_OF.items():
        out[p] = sha_file(vroot / REL[k])
    orig = (vroot / REL["model"]).read_text()
    out["MODEL_ORIG_P032"] = sha_text(ap.apply(orig))
    t25, why25 = p025_base(vroot, work)
    if t25 is not None:
        out["MODEL_P025"] = sha_text(t25)
        out["MODEL_P025_P032"] = sha_text(ap.apply(t25))
    else:
        out["MODEL_P025"] = out["MODEL_P025_P032"] = "pending-025-port"
    s29, _ = p029_mtp_result(vroot)
    out["MTP_P029"] = s29 or "pending-029-port"
    out["APPLIER_SHA"] = sha_file(APPLIER_PATH)
    out["MOD_SHA"] = sha_file(MODULE_PATH)
    return out


def pins_txt_lines(pins: dict) -> list[str]:
    """pins.txt form (`<sha256>  <relative path>  <patch>[:kind]`, the style of the other patches' pins files)."""
    rows = [("MODEL_ORIG", REL["model"], "032"), ("MODEL_ORIG_P032", REL["model"], "032:result"),
            ("MODEL_P025", REL["model"], "032:base(025 result)"), ("MODEL_P025_P032", REL["model"], "032:result(025 base)"),
            ("VPE_ORIG", REL["vpe"], "032:contract"), ("EAGLE_ORIG", REL["eagle"], "032:contract"),
            ("MTP_ORIG", REL["mtp"], "032:contract"), ("MTP_P029", REL["mtp"], "032:contract(029 result)"),
            ("LUTIL_ORIG", REL["lutil"], "032:contract"), ("BLOAD_ORIG", REL["bload"], "032:contract"),
            ("WTIE_ORIG", REL["wtie"], "032:contract"), ("APPLIER_SHA", "files/p032_apply.py", "032:vendored"),
            ("MOD_SHA", "files/p032_gb10_repl_embed.py", "032:vendored")]
    return [f"{pins[k]}  {path}  {label}" for k, path, label in rows if k in pins]


def write_pins(pins: dict) -> None:
    t = SH.read_text()
    for k, v in pins.items():
        t, n = re.subn(rf"^{k}=\S+$", f"{k}={v}", t, flags=re.M)
        if n != 1:
            raise SystemExit(f"pin line {k} found {n} times in {SH}")
    SH.write_text(t)


def earlier_writers(needle: str, upto: int = 32):
    """Names of the .sh files numbered < upto and of the vendored python files they own that mention `needle`."""
    shs = sorted(p for p in PDIR.glob("[0-9][0-9][0-9]-*.sh") if int(p.name[:3]) < upto)
    texts = {p.name: p.read_text(errors="replace") for p in PDIR.glob("[0-9][0-9][0-9]-*.sh")}
    sh_hits = sorted(p.name for p in shs if needle in texts[p.name])
    owned = set()
    for py in HERE.glob("*.py"):
        if py.name.startswith(("p032_", "p033_")) or re.search(r"(cpu_check|testlib|gpu_selftest|e2e_check|_probe|"
                                                                r"_test\.py|rankcheck|bench)", py.name):
            continue
        if any(py.name in texts[s.name] for s in shs):
            owned.add(py)
    py_hits = sorted(p.name for p in owned if needle in p.read_text(errors="replace"))
    return sh_hits, py_hits


def section_a(vroot: Path, work: Path, ap) -> dict:
    print("== A. pins / bases / applier / static enumeration")
    for k, p in PIN_OF.items():
        check(f"A1 --vllm-root {REL[k]} == the .sh pin {p}", pin(p) is not None and sha_file(vroot / REL[k]) == pin(p))
    # the mtp.py pin of patch 029's result: recomputed from the REAL 029 applier when it can run on this tree
    s29, why29 = p029_mtp_result(vroot)
    if s29 is not None:
        check("A1 MTP_P029 == what files/p029_apply.py check mtp prints for the v0.32.0 mtp.py (stale pin otherwise: run "
              "--print-pins)", pin("MTP_P029") == s29, f"pin {pin_raw('MTP_P029')[:16]} applier {s29[:16]}")
    elif pin("MTP_P029") is None:
        skip("A1 MTP_P029 pin", f"placeholder pin and {why29}")
    else:
        check("A1 MTP_P029 pin is a sha but the 029 applier cannot reproduce it", False, why29)
    texts = {"MODEL_ORIG": (vroot / REL["model"]).read_text()}
    t25, why25 = p025_base(vroot, work)
    if t25 is not None:
        texts["MODEL_P025"] = t25
        check("A2 the 025 result (REAL files/p025_apply.py on the v0.32.0 model.py) == the MODEL_P025 pin (placeholder "
              "or stale: run --print-pins)", pin("MODEL_P025") == sha_text(t25), pin_raw("MODEL_P025")[:20])
    else:
        skip("A2 MODEL_P025 / MODEL_P025_P032 (025 bases)", why25)
    check("A2 MODEL_ORIG is the base of the v0.32.0 tree", sha_text(texts["MODEL_ORIG"]) == pin("MODEL_ORIG"))
    # A3: static enumeration of the chain
    sh_hits, py_hits = earlier_writers("glm5next/common/model.py")
    check("A3 .sh files < 032 naming glm5next/common/model.py are a subset of {025}", set(sh_hits) <= {
        "025-gb10-mhc-prefill-sp.sh"}, str(sh_hits))
    check("A3 vendored python of patches < 032 naming common/model.py are 025's files only (applier / module)",
          set(py_hits) <= {"p025_apply.py", "gb10_prefill_sp.py"}, str(py_hits))
    if not QUIET["on"]:
        print("INFO A3 the 025 .sh " + ("is" if "025-gb10-mhc-prefill-sp.sh" in sh_hits else "is NOT")
              + " in this patches/ directory (model.py writers < 032 found: " + str(sh_hits) + ")", flush=True)
    for needle in ("vocab_parallel_embedding", "model_loader/utils.py", "base_loader.py", "weight_tying",
                   "_gb10_repl_embed"):
        sh_n, py_n = earlier_writers(needle)
        check(f"A3 no .sh < 032 / vendored applier of one names {needle}", not sh_n and not py_n, str(sh_n + py_n))
    sh_n, _ = earlier_writers("spec_decode/eagle/utils.py")
    check("A3 eagle/utils.py: at most 029 names it among the .sh < 032 (read-only pin)", set(sh_n) <= {
        "029-gb10-draft-compress.sh"}, str(sh_n))
    sh_n, _ = earlier_writers("common/mtp.py")
    check("A3 mtp.py: at most 029 names it among the .sh < 032 (029 writes it; 032 accepts its result)",
          set(sh_n) <= {"029-gb10-draft-compress.sh"}, str(sh_n))
    mtp_texts = {"v0.32.0 original": (vroot / REL["mtp"]).read_text()}
    if P029_APPLIER.exists():
        r29 = _mkroot(work, vroot, "mtp029")
        (r29 / REL["mtp"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vroot / REL["mtp"], r29 / REL["mtp"])
        rc29 = subprocess.run([sys.executable, "-I", "-B", str(P029_APPLIER), "apply", "mtp", str(r29 / REL["mtp"])],
                              capture_output=True, text=True)
        if rc29.returncode == 0:
            mtp_texts["029 result"] = (r29 / REL["mtp"]).read_text()
    for what, mt in mtp_texts.items():
        check(f"A3 common/mtp.py ({what}) never declares has_own_embed_tokens / SupportsEagle* nor calls "
              "process_eagle_weight: the draft takes the target's embed_tokens (maybe_share_target_embed -> _should_share "
              "True), so no mtp.py edit is needed",
              not re.search(r"has_own_embed_tokens|process_eagle_weight|SupportsEagle", mt))
    mtp_cls = [ln for ln in mtp_texts["v0.32.0 original"].splitlines() if ln.startswith("class Glm5NextMTP")]
    check("A3 class Glm5NextMTP derives from nn.Module, DeepseekV2MixtureOfExperts only (no EAGLE interface with "
          "has_own_embed_tokens)", mtp_cls == ["class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):"], str(mtp_cls))
    check("A3 the 032 .sh pins mtp.py as original-or-029 and names no mtp.py WRITE (applier edits model.py only)",
          "MTP_P029" in SH.read_text() and "p032_apply.py" in SH.read_text() and "mtp" not in " ".join(
              e[0] for e in ap.EDITS))
    later = [p for p in PDIR.glob("[0-9][0-9][0-9]-*.sh") if p.name.startswith("033-")]
    if later:
        t33 = later[0].read_text()
        m1 = re.search(r"^MODEL_ORIG_P032=(\S+)$", t33, re.M)
        m2 = re.search(r"^MODEL_P025_P032=(\S+)$", t33, re.M)
        v1 = m1.group(1) if m1 else None
        v2 = m2.group(1) if m2 else None
        if v1 is None or not HEX64.match(v1):
            skip("A3 patch 033's .sh pins this patch's results as its bases",
                 f"033 has not adopted them yet (MODEL_ORIG_P032={v1}); refresh with --print-pins")
        else:
            check("A3 patch 033's .sh pins this patch's results as its bases (MODEL_ORIG_P032 / MODEL_P025_P032 equal)",
                  v1 == pin_raw("MODEL_ORIG_P032") and (v2 is None or v2 == pin_raw("MODEL_P025_P032")),
                  f"033 has {v1[:16]} / {str(v2)[:16]}")
    else:
        skip("A3 patch 033's pins of this patch's results", "033-gb10-mla-w8a16.sh absent")
    # A4: applier on every base
    for b, t in texts.items():
        out = ap.apply(t)
        check(f"A4 apply({b}) == pinned {b}_P032", pin(b + "_P032") is not None and sha_text(out) == pin(b + "_P032"),
              sha_text(out)[:16])
        check(f"A4 revert({b} + 032) == {b} byte for byte", ap.revert(out) == t)
        check(f"A4 {b} + 032: {ap.N_MARKERS} markers, compiles", out.count(ap.MARKER) == ap.N_MARKERS)
        try:
            ast.parse(out, feature_version=(3, 12))
            ok312 = True
        except SyntaxError:
            ok312 = False
        check(f"A4 {b} + 032 parses as Python 3.12", ok312)
        try:
            ap.apply(out)
            check(f"A4 apply on {b} + 032 refused (marker present)", False)
        except ap.Refused:
            check(f"A4 apply on {b} + 032 refused (marker present)", True)
    t0 = texts["MODEL_ORIG"]
    # CLI
    p = Path(tempfile.mkdtemp(prefix="a4cli.", dir=work)) / "model.py"
    p.write_text(t0)
    rc = subprocess.run([sys.executable, "-I", "-B", str(APPLIER_PATH), "check", str(p)], capture_output=True, text=True)
    check("A4 CLI check prints the pinned result", rc.returncode == 0 and rc.stdout.strip() == pin("MODEL_ORIG_P032"))
    rc = subprocess.run([sys.executable, "-I", "-B", str(APPLIER_PATH), "apply", str(p)], capture_output=True, text=True)
    check("A4 CLI apply writes the pinned result", rc.returncode == 0 and sha_file(p) == pin("MODEL_ORIG_P032"))
    rc = subprocess.run([sys.executable, "-I", "-B", str(APPLIER_PATH), "revert", str(p)], capture_output=True, text=True)
    check("A4 CLI revert restores the original", rc.returncode == 0 and sha_file(p) == pin("MODEL_ORIG"))

    # refusals
    def refused(text, what):
        try:
            ap.apply(text)
            return check(f"A4 refused: {what}", False)
        except (ap.Refused, SyntaxError):
            return check(f"A4 refused: {what}", True)
    refused(t0.replace("logger = init_logger(__name__)\n", "logger = init_logger(__name__)  # x\n", 1),
            "gate anchor missing")
    refused(t0 + "\nlogger = init_logger(__name__)\n", "gate anchor duplicated")
    refused(t0.replace("            self.embed_tokens = VocabParallelEmbedding(\n",
                       "            self.embed_tokens = VocabParallelEmbedding( \n", 1), "ctor anchor missing")
    refused(t0 + "\n\nclass _X:\n    def __init__(self, config, prefix):\n" + ap.CTOR_OLD, "ctor anchor duplicated")
    t_moved = t0.replace(ap.CTOR_OLD, "        if get_pp_group().is_first_rank:\n            self.embed_tokens = None\n", 1)
    t_moved += "\n\nclass _X:\n    def __init__(self, config, prefix):\n" + ap.CTOR_OLD
    refused(t_moved, "ctor anchor outside Glm5NextModel.__init__")
    refused(t0.replace("class Glm5NextModel(nn.Module, EagleModelMixin):",
                       "class Glm5NextModel(nn.Module, EagleModelMixin):\n    _p032 = 1\n", 1), "base already mentions _p032")
    out = ap.apply(t0)
    for bad, what in ((out.replace('== "1":\n    from vllm import _gb10_repl_embed', '!= "0":\n    from vllm import '
                                   '_gb10_repl_embed', 1), "gate expression changed"),
                      (out.replace("_p032.embedding_class(config)", "_p032.ReplicatedEmbedding", 1),
                       "class choice changed")):
        try:
            ap.check_structure(bad, patched=True)
            check(f"A4 check_structure refuses: {what}", False)
        except ap.Refused:
            check(f"A4 check_structure refuses: {what}", True)
    # A5: the gate block alone
    tree = ast.parse(out)
    k = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
         and n.targets[0].id == "logger"][0]
    gate = ast.unparse(ast.Module(body=tree.body[k + 1:k + 3], type_ignores=[]))
    for val, want_mod in ((None, False), ("0", False), ("", False), ("1", True), (" 1", False), ("1 ", False),
                          ("true", False), ("2", False)):
        g = {}
        env = {} if val is None else {"VLLM_GB10_REPL_EMBED": val}
        saved = {kk: os.environ.get(kk) for kk in ENV_KEYS}
        try:
            os.environ.pop("VLLM_GB10_REPL_EMBED", None)
            os.environ.update(env)
            fake_vllm = NS(_gb10_repl_embed="MODULE")
            sys.modules["vllm"] = fake_vllm
            sys.modules["vllm._gb10_repl_embed"] = "MODULE"
            exec(compile(gate, "gate", "exec"), g)
            ok = (g["_p032"] == "MODULE") if want_mod else (g["_p032"] is None)
        except Exception as e:  # noqa: BLE001
            ok, g = False, {"err": repr(e)}
        finally:
            for kk, vv in saved.items():
                if vv is None:
                    os.environ.pop(kk, None)
                else:
                    os.environ[kk] = vv
            sys.modules.pop("vllm._gb10_repl_embed", None)
            sys.modules.pop("vllm", None)
        check(f"A5 gate VLLM_GB10_REPL_EMBED={val!r}: {'imports the module' if want_mod else '_p032 None'}", ok)
    # gate set + module missing -> ImportError
    os.environ["VLLM_GB10_REPL_EMBED"] = "1"
    try:
        sys.modules["vllm"] = type(sys)("vllm")
        sys.modules["vllm"].__path__ = []
        exec(compile(gate, "gate", "exec"), {})
        check("A5 gate 1 + module missing -> ImportError (fail closed)", False)
    except ImportError:
        check("A5 gate 1 + module missing -> ImportError (fail closed)", True)
    finally:
        os.environ.pop("VLLM_GB10_REPL_EMBED", None)
        sys.modules.pop("vllm", None)
    # A6: vendored shas in the .sh
    check("A6 APPLIER_SHA / MOD_SHA == the vendored files", pin("APPLIER_SHA") == sha_file(APPLIER_PATH)
          and pin("MOD_SHA") == sha_file(MODULE_PATH))
    for f in (MODULE_PATH, APPLIER_PATH, Path(__file__)):
        try:
            ast.parse(f.read_text(), feature_version=(3, 12))
            ok312 = True
        except SyntaxError:
            ok312 = False
        check(f"A6 {f.name} parses as Python 3.12 (the image's interpreter)", ok312)
    return texts


# =====================================================================================================================
# B. module units (REAL vocab_parallel_embedding.py, 8 threaded ranks)
# =====================================================================================================================
BRANCH = {"masked": {}, "fused": {}}          # per rank thread: calls of get_masked_input_and_mask / the fused kernel


def _count_branches(ns):
    """Count, per rank thread, the REAL forward's two lookup branches (the masked helper of vocab_parallel_embedding.py and
    the fused kernel stand-in)."""
    real_masked = ns.vpe.get_masked_input_and_mask

    def masked(*a, **k):
        n = threading.current_thread().name
        BRANCH["masked"][n] = BRANCH["masked"].get(n, 0) + 1
        return real_masked(*a, **k)
    ns.vpe.get_masked_input_and_mask = masked
    real_fused = ns.vpe.ops.vocab_parallel_embedding

    def fused(*a, **k):
        n = threading.current_thread().name
        BRANCH["fused"][n] = BRANCH["fused"].get(n, 0) + 1
        return real_fused(*a, **k)
    ns.vpe.ops.vocab_parallel_embedding = fused


def boot_scenario(vroot, module_path, V, H, bits_per_rank, *, env=None, cfg=None, inject=None, ftz=False,
                  chunk=None, load_times=1, oneshot_tokens=8, post=None, prepare=None, timeout=60.0,
                  img_loaded=True, cuda=True, untied=False):
    """Builds the replicated layer on 8 ranks, loads it, runs the REAL process_weights_after_loading (the boot) and
    compares the outcome with the unpatched layer on every rank. inject(r, ns, layer, world) runs before the boot (its
    world hooks and T.INJ flags are cleared after it); prepare(ns) runs once before the threads start."""
    ns = build_ns(vroot, None, module_path, env=env or {"VLLM_GB10_REPL_EMBED": "1"}, with_model=False, cuda=cuda)
    BRANCH["masked"].clear()
    BRANCH["fused"].clear()
    _count_branches(ns)
    if chunk is not None:
        ns.mod.CHUNK_IDS = chunk
    if prepare is not None:
        prepare(ns)
    cfg = cfg or mk_cfg(V, H)
    vcfg = mk_vcfg(cfg, untied=untied)
    rv = T.Rendezvous(W, timeout=6.0)
    ids = ids_all_plus_oob(V)

    def body(r):
        w = setup_rank(rv, r, H, vcfg, ftz=ftz, oneshot_tokens=oneshot_tokens)
        bits = bits_per_rank(r)
        lay = build_layer(ns, V, H, bits, cfg, load=False)
        for _ in range(load_times):
            lay.weight.weight_loader(lay.weight, cpu_tensor(ns, bits))
        if inject is not None:
            inject(r, ns, lay, w)
        lay = post_load(ns, lay, untied=untied)
        w.ar_hook = w.ar_out_hook = None
        w.ftz = False
        for k in list(vars(T.INJ)):
            if k != "orig_refs":
                delattr(T.INJ, k)
        ar_boot = len(w.ar_calls)
        mode = mode_of(ns, lay)
        if img_loaded:
            img = image_layer(ns, V, H, bits)
        else:      # the unpatched layer of a load that never delivered the weight: the same torch.empty bytes
            with ns.torch.device("cuda"):
                img = ns.vpe.VocabParallelEmbedding(V, H, prefix=PREFIX)
        ref = image_lookup(img, ids, ns)
        me = threading.current_thread().name
        n0 = len(w.ar_calls)
        f0, m0 = BRANCH["fused"].get(me, 0), BRANCH["masked"].get(me, 0)
        got = local_lookup(lay, ids, ns)
        n_fwd = len(w.ar_calls) - n0
        res = {"mode": mode, "ar_boot": ar_boot, "fwd_ars": n_fwd, "equal": bool(np.array_equal(got, ref)),
               "type": type(lay).__name__, "state_ok": None, "bytes_ok": None,
               "fwd_fused": BRANCH["fused"].get(me, 0) - f0, "fwd_masked": BRANCH["masked"].get(me, 0) - m0,
               "img_fused": img.use_fused_embedding}
        if mode == "image":
            res["state_ok"] = image_state(lay) == image_state(img)
            res["bytes_ok"] = bool(np.array_equal(lay.weight._a, img.weight._a))
        if mode == "replicated":
            res["negzero_left"] = int(np.count_nonzero(lay.weight._a == 0x8000))
            res["table_ok"] = bool(np.array_equal(lay.weight._a, canon_bits(bits)))
        if post is not None:
            res.update(post(r, ns, lay, w, img) or {})
        return res
    out = run8(body, timeout=timeout)
    errs = [repr(x) for ok, x in out if not ok]
    rs = [x for ok, x in out if ok]
    act = T.LOGS.lines(contains="replicated input embedding ACTIVE (")
    ina = T.LOGS.lines(contains="replicated input embedding INACTIVE (")
    fps = sorted({m_.group(1) for m_ in (re.search(r"fingerprint ([0-9a-f]{16})", m) for m in act) if m_})
    return NS(ns=ns, rs=rs, errs=errs, act=act, ina=ina, fps=fps, log=list(rv.log),
              modes=sorted({x["mode"] for x in rs}), hang=any(isinstance(x, T.Hang) for ok, x in out if not ok))


def all_image(sc):
    return (not sc.errs and len(sc.rs) == W and sc.modes == ["image"] and all(x["equal"] and x["state_ok"]
            and x["bytes_ok"] and x["fwd_ars"] >= 1 for x in sc.rs) and len(sc.ina) == W and not sc.act)


def section_b(vroot: Path, module_path: Path, quick=False):
    print("== B. module units (REAL v0.32.0 vocab_parallel_embedding.py, 8 threaded ranks)")
    V, H = 512, 16
    bits = make_ckpt(V, H)
    same = lambda r: bits  # noqa: E731
    imod = _instrumented(module_path)
    res = {}
    if not quick:
        # B1 construction envelope
        ns = build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
        rv = T.Rendezvous(W)

        def cons(r):
            out = {}
            for name, tp, pp, lora, tie, VV, dt, untied, want in (
                    ("served", 8, 1, None, False, V, "bf16", False, "ReplicatedEmbedding"),
                    ("TP 4", 4, 1, None, False, V, "bf16", False, "VocabParallelEmbedding"),
                    ("PP 2", 8, 2, None, False, V, "bf16", False, "VocabParallelEmbedding"),
                    ("LoRA", 8, 1, NS(), False, V, "bf16", False, "VocabParallelEmbedding"),
                    ("tied", 8, 1, None, True, V, "bf16", False, "VocabParallelEmbedding"),
                    ("untied by the checkpoint", 8, 1, None, False, V, "bf16", True, "VocabParallelEmbedding"),
                    ("vocab 520", 8, 1, None, False, 520, "bf16", False, "VocabParallelEmbedding"),
                    ("fp32", 8, 1, None, False, V, "fp32", False, "VocabParallelEmbedding")):
                ww = T.World(rv, r, tp, oneshot_max=8 * H, pp_size=pp)
                T.set_current_world(ww)
                cfg = mk_cfg(VV, H, tie=tie)
                TLS.vcfg = mk_vcfg(cfg, lora=lora, untied=untied)
                out[name] = (ns.mod.embedding_class(cfg).__name__ if dt == "bf16" else _with_dtype(
                    ns, ns.torch.float32, lambda c=cfg: ns.mod.embedding_class(c).__name__), want)
            return out
        o = run8(cons)
        for name, (got, want) in (o[0][1].items() if o[0][0] else []):
            check(f"B1 embedding_class: {name} -> {want}", all(ok and x[name][0] == want for ok, x in o), got)
        check("B1 construction ran on 8 ranks", all(ok for ok, _ in o), str([x for ok, x in o if not ok][:1]))
        check("B1 each refusal logged as an INACTIVE line (7 per rank)",
              len(T.LOGS.lines(contains="replicated input embedding INACTIVE (")) == 7 * W)
        check("B1 the untied-by-checkpoint refusal names the lm_head re-tie",
              any("re-tie" in m for m in T.LOGS.lines(contains="replicated input embedding INACTIVE (")))
        # B2 weight loading through the REAL weight_loader
        rv = T.Rendezvous(W)

        def loadchk(r):
            setup_rank(rv, r, H, mk_vcfg(mk_cfg(V, H)))
            lay = build_layer(ns, V, H, bits, mk_cfg(V, H))
            img = image_layer(ns, V, H, bits)
            P = V // W
            return (type(lay).__name__ == "ReplicatedEmbedding" and tuple(lay.weight.shape) == (V, H)
                    and np.array_equal(lay.weight._a, bits) and lay._p032_full_loads == 1 and lay.tp_size == 1
                    and np.array_equal(img.weight._a, bits[r * P:(r + 1) * P]) and tuple(img.weight.shape) == (P, H))
        o = run8(loadchk)
        check("B2 the REAL weight_loader fills the FULL [V, H] table on every rank (unpatched layer: rows rP..(r+1)P)",
              all(ok and x for ok, x in o), str([x for ok, x in o if not ok][:1]))
        # B3 exactness of the unpatched path itself: lookup + all-reduce == canon(row); != the raw row at -0.0
        for arith, oneshot_tokens in (("one-shot (patch 012)", 1 << 20), ("NCCL-like ring", 0)):
            rv = T.Rendezvous(W)
            ids = ids_all_plus_oob(V)

            def ex(r, oneshot_tokens=oneshot_tokens, ids=ids, rv=rv):
                setup_rank(rv, r, H, mk_vcfg(mk_cfg(V, H)), oneshot_tokens=oneshot_tokens)
                img = image_layer(ns, V, H, bits)
                return image_lookup(img, ids, ns, chunk=8 if oneshot_tokens else 4096)
            o = run8(ex)
            raw = np.concatenate([bits, np.zeros((7, H), np.uint16)])
            canon = canon_bits(raw)
            check(f"B3 {arith}: the unpatched fused lookup + all-reduce == canon(row) bitwise for all {V} ids + 7 "
                  "out-of-range ids", all(ok and np.array_equal(x, canon) for ok, x in o))
            diff = o[0][1] != raw
            check(f"B3 {arith}: it differs from the raw row exactly at the -0.0 elements ({int(diff.sum())})",
                  np.array_equal(diff, raw == 0x8000) and diff.sum() > H)
        rv = T.Rendezvous(W)

        def ftzc(r):
            setup_rank(rv, r, H, mk_vcfg(mk_cfg(V, H)), oneshot_tokens=0, ftz=True)
            img = image_layer(ns, V, H, bits)
            return image_lookup(img, list(range(V)), ns)
        o = run8(ftzc)
        check("B3 a flushing (FTZ) reduction would change the subnormal elements (what the boot must catch)",
              o[0][0] and not np.array_equal(o[0][1], canon_bits(bits)))
    # B4 clean boot
    sc = boot_scenario(vroot, module_path, V, H, same)
    res["clean"] = sc
    check("B4 clean boot: ACTIVE on all 8 ranks, one fingerprint, no INACTIVE",
          not sc.errs and sc.modes == ["replicated"] and len(sc.act) == W and len(sc.fps) == 1 and not sc.ina,
          f"{sc.errs[:1]} {sc.modes} {sc.ina[:1]}")
    check("B4 replicated forward == the unpatched forward BITWISE (all ids + out-of-range) on every rank, 0 all-reduces",
          len(sc.rs) == W and all(x["equal"] and x["fwd_ars"] == 0 for x in sc.rs))
    check("B4 the replicated forward runs the FUSED kernel branch (as the unpatched layer at TP=8 does): one fused call "
          "per lookup chunk, the masked-gather helper never",
          len(sc.rs) == W and all(x["img_fused"] and x["fwd_fused"] >= 1 and x["fwd_masked"] == 0 for x in sc.rs),
          str([(x["fwd_fused"], x["fwd_masked"]) for x in sc.rs][:2]))
    check("B4 the table is the canonicalised checkpoint (no -0.0 left, every other bit untouched)",
          len(sc.rs) == W and all(x.get("table_ok") and x.get("negzero_left") == 0 for x in sc.rs))
    check("B4 identical collective schedule on all ranks; every planned all-reduce issued",
          len(sc.rs) == W and len({x["ar_boot"] for x in sc.rs}) == 1 and sc.rs[0]["ar_boot"] > 100,
          str(sorted({x["ar_boot"] for x in sc.rs})))
    nneg = int(np.count_nonzero(bits == 0x8000))
    check("B4 the ACTIVE line reports the -0.0 / subnormal counts",
          len(sc.act) == W and all(f"{nneg} negative zeros set to +0.0, 2 subnormals" in m for m in sc.act))
    if not quick:
        sc = boot_scenario(vroot, module_path, V, H, same, cuda=False)
        check("B4 CPU/non-CUDA platform (use_fused_embedding False: the masked-gather branch of the REAL forward): ACTIVE, "
              "bitwise, the masked helper runs and the fused kernel never",
              sc.modes == ["replicated"] and len(sc.rs) == W and all(
                  x["equal"] and not x["img_fused"] and x["fwd_fused"] == 0 and x["fwd_masked"] >= 1 for x in sc.rs)
              and len(sc.act) == W, f"{sc.errs[:1]} {sc.modes}")
        sc = boot_scenario(vroot, module_path, V, H, same, chunk=128)
        check("B4 multi-chunk exhaustive check (CHUNK_IDS 128 -> 4 calls): ACTIVE, bitwise",
              sc.modes == ["replicated"] and len(sc.rs) == W and all(x["equal"] for x in sc.rs) and len(sc.act) == W)
        sc = boot_scenario(vroot, module_path, V, H, lambda r: make_ckpt(V, H, negzero=False, subnormal=False))
        check("B4 a table without -0.0 / subnormals: ACTIVE, bitwise", sc.modes == ["replicated"]
              and len(sc.rs) == W and all(x["equal"] for x in sc.rs))
        sc = boot_scenario(vroot, module_path, V, H, same, oneshot_tokens=0)
        check("B4 every all-reduce on the ring (no one-shot): ACTIVE, bitwise", sc.modes == ["replicated"]
              and len(sc.rs) == W and all(x["equal"] for x in sc.rs))
    # B5 fail-closed scenarios
    bad3 = bits.copy()
    bad3[(V // W) * 2 + 9, 4] ^= 0x0010

    def corrupt_rank3(r):
        return bad3 if r == 3 else bits

    def blind_digest(ns_):
        real = ns_.mod._scan_table

        def fake(Wt, _real=real):
            d = _real(Wt)
            d["digest"] = "blinded"
            return d
        ns_.mod._scan_table = fake

    def on_rank(rank, **flags):
        def inj(r, ns_, lay, w):
            if r == rank:
                for k, v in flags.items():
                    setattr(T.INJ, k, v)
        return inj

    def oneshot_corrupt(r, ns_, lay, w):
        if r == 5:
            def hook(seq, out):
                if out.size <= 8 * H:
                    out = out.copy()
                    out.reshape(-1)[0] ^= np.uint16(1)
                return out
            w.ar_out_hook = hook

    scen = [
        ("a corrupt row on rank 3 (the content digest differs)", dict(bits_per_rank=corrupt_rank3), "differs across"),
        ("a corrupt row on rank 3, digest blinded (the exhaustive check alone)",
         dict(bits_per_rank=corrupt_rank3, prepare=blind_digest), "calls differ"),
        ("a NaN in the table", dict(bits_per_rank=lambda r: make_ckpt(V, H, nan=True)), "non-finite"),
        ("a flushing (FTZ) ring all-reduce + subnormal rows", dict(bits_per_rank=same, ftz=True, oneshot_tokens=0),
         "calls differ"),
        ("rank 5 receives a corrupted one-shot result", dict(bits_per_rank=same, inject=oneshot_corrupt),
         "calls differ"),
        ("a rank-local exception in the 7th lookup on rank 2 (dummies keep the schedule)",
         dict(bits_per_rank=same, inject=on_rank(2, lookup_fail_at=7)), "injected kernel failure"),
        ("CUDA-graph capture raises on rank 6", dict(bits_per_rank=same, inject=on_rank(6, graph_fail=True)),
         "CUDA-graph check raised"),
        ("CUDA-graph replay corrupts an element on rank 1",
         dict(bits_per_rank=same, inject=on_rank(1, corrupt_replay=True)), "CUDA-graph replay differs"),
        ("a fingerprint mismatch on rank 4", dict(bits_per_rank=same, inject=on_rank(4, fp_override="d" * 16)),
         "fingerprint mismatch"),
        ("embed_tokens.weight never loaded (compared with an unpatched layer that was never loaded either)",
         dict(bits_per_rank=same, load_times=0, img_loaded=False), "want exactly 1"),
        ("embed_tokens.weight loaded twice", dict(bits_per_rank=same, load_times=2), "want exactly 1"),
        ("the preflight scan raises on rank 7", dict(bits_per_rank=same, inject=on_rank(7, scan_raise=True)),
         "preflight raised"),
        ("canonicalisation leaves a -0.0 on rank 3", dict(bits_per_rank=same, inject=on_rank(3, negzero_left=True)),
         "negative zeros left"),
        ("the clone of the image shard raises on rank 0", dict(bits_per_rank=same, inject=on_rank(0, clone_raise=True)),
         "clone / canonicalise raised"),
        ("the masked-gather branch (non-CUDA platform) with a corrupt row on rank 3",
         dict(bits_per_rank=corrupt_rank3, cuda=False), "differs across"),
    ]
    for name, kw, why in (scen if not quick else scen[:6]):
        sc = boot_scenario(vroot, imod, V, H, **kw)
        ok = all_image(sc) and not sc.hang and all(why in m for m in sc.ina)
        check(f"B5 {name} -> INACTIVE on all 8 ranks, no hang, layer == the unpatched layer, forward == unpatched", ok,
              f"errs {sc.errs[:1]} modes {sc.modes} hang {sc.hang} ina {sc.ina[:1]}")
        res["scen:" + name] = ok
    if quick:
        return res
    # the envelope refusal itself (a rank mismatch cannot be injected without breaking the harness's world)
    ns = build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
    lay_ns = NS(_p032_size=8, _p032_rank=3)
    check("B5 _envelope_refusal: TP 4 / rank mismatch / PP 2 refused, the served envelope accepted",
          bool(ns.mod._envelope_refusal(lay_ns, NS(size=4, rank=3, pp_size=1)))
          and "construction-time" in ns.mod._envelope_refusal(lay_ns, NS(size=8, rank=2, pp_size=1))
          and "pipeline" in ns.mod._envelope_refusal(lay_ns, NS(size=8, rank=3, pp_size=2))
          and ns.mod._envelope_refusal(lay_ns, NS(size=8, rank=3, pp_size=1)) is None)

    # B6 fallback / commit details
    def keep_ref(r, ns_, lay, w):
        TLS.table_ref = weakref.ref(lay.weight)
        if r == 4:
            T.INJ.fp_override = "d" * 16

    def post_ref(r, ns_, lay, w, img):
        return {"table_freed": TLS.table_ref() is None,
                "no_view": "_p032_view" not in lay.__dict__,
                "negzero_kept": int(np.count_nonzero(lay.weight._a == 0x8000)) == int(np.count_nonzero(
                    img.weight._a == 0x8000)) and int(np.count_nonzero(img.weight._a == 0x8000)) >= 0}
    sc = boot_scenario(vroot, imod, V, H, same, inject=keep_ref, post=post_ref)
    check("B6 fallback (early: a preflight fingerprint mismatch): the full table is released and the shard keeps the "
          "unpatched layer's -0.0 bits",
          len(sc.rs) == W and all(x["table_freed"] and x["no_view"] and x["negzero_kept"] and x["bytes_ok"]
                                  for x in sc.rs), str(sc.errs[:1]))

    def keep_ref_late(r, ns_, lay, w):
        TLS.table_ref = weakref.ref(lay.weight)
        if r == 1:
            T.INJ.corrupt_replay = True
    sc = boot_scenario(vroot, imod, V, H, same, inject=keep_ref_late, post=post_ref)
    check("B6 fallback (late: a CUDA-graph replay mismatch on rank 1, after every rank cached its replicated view): the "
          "full table is released (the cached view is dropped, it would keep the table alive) and the shard keeps the "
          "unpatched layer's -0.0 bits",
          len(sc.rs) == W and all(x["table_freed"] and x["no_view"] and x["negzero_kept"] and x["bytes_ok"]
                                  and x["state_ok"] for x in sc.rs) and len(sc.ina) == W, str(sc.errs[:1]))
    check("B6 fallback: class VocabParallelEmbedding and EVERY constructor attribute (TP state, shard indices, per-"
          "partition sizes, output_size_per_partition, output_partition_sizes, use_fused_embedding, reduce_results, "
          "parallel_group, weight attrs) == a freshly built unpatched layer's",
          len(sc.rs) == W and all(x["type"] == "VocabParallelEmbedding" and x["state_ok"] for x in sc.rs))

    def post_active(r, ns_, lay, w, img):
        refs = getattr(T.INJ, "orig_refs", [])
        return {"orig_freed": len(refs) == 1 and refs[0]() is None}
    sc = boot_scenario(vroot, imod, V, H, same, post=post_active)
    check("B6 ACTIVE: the cloned unpatched shard is released after the commit",
          len(sc.rs) == W and all(x["orig_freed"] for x in sc.rs) and sc.modes == ["replicated"])
    # B7 pending mode (boot never ran)
    ns = build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
    rv = T.Rendezvous(W)
    ids = ids_all_plus_oob(V)

    def pend(r):
        w = setup_rank(rv, r, H, mk_vcfg(mk_cfg(V, H)))
        lay = build_layer(ns, V, H, bits, mk_cfg(V, H))
        img = image_layer(ns, V, H, bits)
        ref = image_lookup(img, ids, ns)
        n0 = len(w.ar_calls)
        got = local_lookup(lay, ids, ns)
        return np.array_equal(got, ref), len(w.ar_calls) - n0
    o = run8(pend)
    check("B7 pending mode (no process_weights_after_loading): forward == the unpatched one, all-reduce issued",
          all(ok and x[0] and x[1] >= 1 for ok, x in o))
    check("B7 pending mode warns once per rank", len(T.LOGS.lines(contains="boot self-test never ran")) == W)

    # B8 reload: a second process_weights_after_loading re-canonicalises locally
    def reload_post(r, ns_, lay, w, img):
        n0 = w.seq
        lay.weight.weight_loader(lay.weight, cpu_tensor(ns_, bits))
        reloaded_neg = int(np.count_nonzero(lay.weight._a == 0x8000))
        post_load(ns_, lay)
        return {"reload_ok": reloaded_neg > 0 and int(np.count_nonzero(lay.weight._a == 0x8000)) == 0
                and w.seq == n0 and mode_of(ns_, lay) == "replicated"}
    sc = boot_scenario(vroot, module_path, V, H, same, post=reload_post)
    check("B8 a weight reload + process_weights_after_loading re-canonicalises locally (no collective)",
          len(sc.rs) == W and all(x.get("reload_ok") for x in sc.rs))
    # B9 plan properties at the real size
    ns = build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
    m = ns.mod
    Vr, Pr = 154880, 19360
    special = [3, 154820, 77]
    plan = m.build_plan(Vr, Pr, 8, special)
    alls = [c for c in plan if c[0] == "all"]
    cover = sorted(i for c in alls for i in range(c[1], c[2])) == list(range(Vr))
    check("B9 real size: 19 exhaustive calls of <= 8,192 ids cover [0, 154880) exactly once",
          len(alls) == 19 and cover and max(c[2] - c[1] for c in alls) == 8192)
    probe = m.probe_ids(Vr, Pr, 8, special)
    check("B9 probe list: shard boundaries, special rows, 7 out-of-range ids, 32 random, no duplicates",
          all(x in probe for x in [0, Vr - 1] + [r * Pr - 1 for r in range(1, 8)] + [r * Pr for r in range(1, 8)]
              + special + m.oob_ids(Vr)) and len(probe) == len(set(probe)))
    sizes = sorted({c[1] for c in plan if c[0] == "probe"})
    check("B9 probe sizes 1..8 (one-shot at H=4096), 16, 48, 3840", sizes == [1, 2, 3, 4, 5, 6, 7, 8, 16, 48, 3840])
    cov = all(set(probe) <= {i for c in plan if c[0] == "probe" and c[1] == n for i in c[2]} for n in sizes)
    check("B9 every probe size covers the whole probe list", cov)
    check("B9 >= 4 probe sets of GRAPH_N tokens for the CUDA-graph check",
          sum(1 for c in plan if c[0] == "probe" and c[1] == m.GRAPH_N) >= m.GRAPH_REPLAYS)
    check("B9 the plan is a pure function of (V, rows, size, special rows)",
          m.build_plan(Vr, Pr, 8, special) == plan and m._plan_sig(plan) == m._plan_sig(m.build_plan(Vr, Pr, 8,
                                                                                                      special)))
    one = sum(1 for c in plan if c[0] == "probe" and c[1] <= 8)
    check("B9 boot cost at the real size: <= 300 all-reduces (19 x 64 MiB, the rest <= 3,840 tokens)",
          len(plan) <= 300, f"{len(plan)} calls, {one} of them one-shot sized")
    # B11 an end-to-end boot at the REAL vocabulary (154,880 rows, 8 ranks, H = 8: 19 exhaustive calls + 171 probes)
    Vr, Hr = 154880, 8
    bits_r = make_ckpt(Vr, Hr, seed=3)
    sc = boot_scenario(vroot, module_path, Vr, Hr, lambda r: bits_r, timeout=600.0)
    n_all = sum(1 for m in sc.act if "all 154880 ids in 19 calls of <= 8192 tokens" in m)
    check("B11 real vocabulary (154,880 x 8): ACTIVE on 8 ranks, 19 exhaustive calls, forward == unpatched bitwise for "
          "every id + 7 out-of-range, 0 all-reduces", not sc.errs and sc.modes == ["replicated"] and n_all == W
          and all(x["equal"] and x["fwd_ars"] == 0 and x["table_ok"] for x in sc.rs),
          f"{sc.errs[:1]} {sc.modes} {sc.act[:1]}")
    b16 = bits_r.astype(np.int32)
    special_r = [int(i) for i in np.nonzero(((b16 == 0x8000) | (((b16 & 0x7F80) == 0) & ((b16 & 0x7FFF) != 0)))
                                            .any(axis=1))[0]]
    want_calls = len(m.build_plan(Vr, Vr // W, W, special_r[:m.MAX_SPECIAL_ROWS]))
    check(f"B11 real vocabulary: {want_calls} planned all-reduces per rank (19 exhaustive + the probes over "
          f"{len(special_r)} special rows), identical on all ranks",
          bool(sc.rs) and len({x["ar_boot"] for x in sc.rs}) == 1 and sc.rs[0]["ar_boot"] == want_calls,
          str(sorted({x["ar_boot"] for x in sc.rs})))
    # B10 log hygiene: no gate text quoted by another line (the logger format strings, read from the AST)
    tree = ast.parse(Path(module_path).read_text())
    fmts = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "logger" and n.args
            and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)]
    act_fmt = [s for s in fmts if "replicated input embedding ACTIVE" in s]
    ina_fmt = [s for s in fmts if "replicated input embedding INACTIVE" in s]
    check("B10 exactly one ACTIVE and one INACTIVE logger format; ACTIVE is followed by '(rank', neither contains the "
          "other's key", len(act_fmt) == 1 and len(ina_fmt) == 1 and "INACTIVE" not in act_fmt[0]
          and "embedding ACTIVE (rank" in act_fmt[0] and "embedding ACTIVE" not in ina_fmt[0], str(len(fmts)))
    other = [s for s in fmts if s not in act_fmt + ina_fmt and re.search(r"embedding (IN)?ACTIVE", s)]
    never = [s for s in fmts if "never ran" in s]
    check("B10 no other logger format carries a gate phrase; 'never ran' only in the pending-mode warning",
          not other and len(never) == 1 and "ACTIVE" not in never[0], str(other))
    sh = SH.read_text()
    oks = re.findall(r'echo "(\[OK\][^"]*)"', sh)
    check("B10 the .sh [OK] lines quote no gate text (no ACTIVE / INACTIVE / [patch-032] / 'replicated input "
          "embedding' / 'never ran')", len(oks) == 2 and not any(
              re.search(r"ACTIVE|\[patch-032\]|replicated input embedding|never ran", s) for s in oks), str(oks))
    outs = [s for ln in sh.splitlines() if not ln.lstrip().startswith("#")
            for s in re.findall(r'(?:echo|fail) "([^"]*)"', ln)]
    check("B10 no line the .sh prints (echo / fail) carries gate text", len(outs) > 20 and not any(
        re.search(r"ACTIVE|\[patch-032\]|replicated input embedding|never ran", s) for s in outs),
        str([s for s in outs if re.search(r"ACTIVE|\[patch-032\]|replicated input embedding|never ran", s)]))
    section_b12(vroot, module_path, bits)
    return res


def section_b12(vroot: Path, module_path: Path, bits):
    """v0.32.0 specifics: the fused-kernel decision, the REAL loader's new steps, supports_pre_processed_weights."""
    print("== B12 v0.32.0 specifics")
    V, H = 512, 16
    # (c) the view's fused decision is the unpatched constructor's, on both platform settings
    for cuda in (True, False):
        ns = build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False, cuda=cuda)
        rv = T.Rendezvous(W)

        def fz(r, ns=ns):
            setup_rank(rv, r, H, mk_vcfg(mk_cfg(V, H)))
            img = image_layer(ns, V, H, bits)
            with ns.torch.device("cuda"):
                head = ns.vpe.ParallelLMHead(V, H, prefix="lm_head")
            return (ns.mod._vanilla_use_fused(W) == img.use_fused_embedding == head.use_fused_embedding,
                    img.use_fused_embedding)
        o = run8(fz)
        check(f"B12 _vanilla_use_fused(8) == the REAL constructor's use_fused_embedding (is_cuda={cuda}: "
              f"{o[0][1][1] if o[0][0] else '?'})", all(ok and x[0] and x[1] == cuda for ok, x in o), str(o[0][1])[:100])
    # (e) maybe_retie_word_embeddings never touches the replicated table (flag True, equal-count modules)
    ns = build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
    rv = T.Rendezvous(W, timeout=6.0)
    cfg = mk_cfg(V, H)

    def retie(r):
        setup_rank(rv, r, H, mk_vcfg(cfg))
        tm = ns.torch
        lay = build_layer(ns, V, H, bits, cfg)
        with tm.device("cuda"):
            head = ns.vpe.ParallelLMHead(V, H, prefix="lm_head")
        head.weight.weight_loader(head.weight, cpu_tensor(ns, bits))
        h = tm.nn.Module()
        h.embed_tokens = lay
        h.lm_head = head
        w_before, hw_before = lay.weight, head.weight
        untied = ns.wtie._get_untied_lm_head(h)
        ns.lutil.process_weights_after_loading(h, NS(quantization="fp8", word_embeddings_untied_by_checkpoint=True,
                                                     model="x"), tm.device("cuda"))
        return (untied is None, h.lm_head is head and head.weight is hw_before, mode_of(ns, h.embed_tokens))
    o = run8(retie)
    check("B12 maybe_retie_word_embeddings (flag True) finds no untied lm_head for the replicated table (shape differs) "
          "and re-ties nothing; the lm_head object and weight are untouched; the boot still ran",
          all(ok and x[0] and x[1] and x[2] in ("replicated", "image") for ok, x in o), str(o[0][1])[:100])
    # (f) supports_pre_processed_weights = False: the REAL loader rejects the method under weights_already_processed()
    rv = T.Rendezvous(W, timeout=6.0)

    def preproc(r):
        setup_rank(rv, r, H, mk_vcfg(cfg))
        tm = ns.torch
        lay = build_layer(ns, V, H, bits, cfg)
        vanilla = image_layer(ns, V, H, bits)
        h = tm.nn.Module()
        h.embed_tokens = lay
        h2 = tm.nn.Module()
        h2.embed_tokens = vanilla
        errs = []
        with ns.mutils.weights_already_processed():
            ns.lutil.process_weights_after_loading(h2, NS(quantization="fp8", word_embeddings_untied_by_checkpoint=False),
                                                   tm.device("cuda"))
            try:
                ns.lutil.process_weights_after_loading(h, NS(quantization="fp8", word_embeddings_untied_by_checkpoint=False),
                                                       tm.device("cuda"))
            except RuntimeError as e:
                errs.append(str(e))
        return errs, lay._p032_booted, ns.mod._P032EmbeddingMethod.supports_pre_processed_weights
    o = run8(preproc)
    check("B12 under weights_already_processed() the REAL loader raises 'does not support pre-processed weights' for "
          "the replicated layer (fail closed, the boot not run) and accepts the unpatched layer",
          all(ok and len(x[0]) == 1 and "does not support pre-processed weights" in x[0][0] and not x[1] and
              x[2] is False for ok, x in o), str(o[0][1])[:140])
    # (g) the replicated view is rebuilt when the weight Parameter object is replaced (never stale)
    sc = boot_scenario(vroot, module_path, V, H, lambda r: bits)
    rv = T.Rendezvous(W, timeout=6.0)

    def stale(r):
        setup_rank(rv, r, H, mk_vcfg(cfg))
        lay = post_load(ns, build_layer(ns, V, H, bits, cfg))
        v1 = lay._p032_replicated_view()
        same = lay._p032_replicated_view() is v1
        old = lay.weight
        lay.weight = ns.torch.nn.Parameter(old.data.clone(), requires_grad=False)
        v2 = lay._p032_replicated_view()
        return same and v2 is not v1 and v2.weight is lay.weight
    o = run8(stale)
    check("B12 the cached replicated view is reused while the weight Parameter is the same and rebuilt when it is replaced",
          sc.modes == ["replicated"] and all(ok and x for ok, x in o), str(o[0][1])[:100])


def _with_dtype(ns, dt, fn):
    old = ns.torch.get_default_dtype()
    ns.torch.set_default_dtype(dt)
    try:
        return fn()
    finally:
        ns.torch.set_default_dtype(old)


WORK = {"dir": None}


def _instrumented(module_path: Path) -> Path:
    """The module + test-only fault hooks (p032_testlib.INJ, per rank thread), appended so the production code above
    them is byte-identical; the hooks rebind module globals, which every call site looks up at call time."""
    out = Path(tempfile.mkdtemp(prefix="instr.", dir=WORK["dir"])) / "p032_gb10_repl_embed_instr.py"
    extra = """

# ---- test-only instrumentation (p032_cpu_check.py) ----
import p032_testlib as _T
_real_lookup = _lookup
def _lookup(view, input_):
    n = getattr(_T.INJ, "lookup_fail_at", None)
    if n is not None:
        _T.INJ.lookups = getattr(_T.INJ, "lookups", 0) + 1
        if _T.INJ.lookups == n:
            raise RuntimeError("injected kernel failure")
    return _real_lookup(view, input_)
_real_graph_check = _graph_check
def _graph_check(layer, refs):
    if getattr(_T.INJ, "graph_fail", False):
        raise RuntimeError("injected capture failure")
    return _real_graph_check(layer, refs)
_real_fingerprint = fingerprint
def fingerprint(layer, world, plan_sig):
    o = getattr(_T.INJ, "fp_override", None)
    return o if o is not None else _real_fingerprint(layer, world, plan_sig)
_real_scan = _scan_table
def _scan_table(W):
    if getattr(_T.INJ, "scan_raise", False):
        raise RuntimeError("injected scan failure")
    return _real_scan(W)
_real_count_negzero = _count_negzero
def _count_negzero(W):
    if getattr(_T.INJ, "negzero_left", False):
        return 1
    return _real_count_negzero(W)
_real_canon = _canonicalize_zeros
def _canonicalize_zeros(W):
    if getattr(_T.INJ, "clone_raise", False):
        raise RuntimeError("injected clone failure")
    return _real_canon(W)
_real_selftest = _selftest
def _selftest(layer, world, plan):
    rep, orig = _real_selftest(layer, world, plan)
    import weakref as _w
    if orig is not None:
        _T.INJ.orig_refs = getattr(_T.INJ, "orig_refs", []) + [_w.ref(orig)]
    return rep, orig
"""
    out.write_text(Path(module_path).read_text() + extra)
    return out


# =====================================================================================================================
# C. the REAL patched model.py + mtp.py + eagle/utils.py + process_weights_after_loading (8 threaded ranks)
# =====================================================================================================================
def model_scenario(vroot, model_text, module_path, env, V=512, H=16, bits=None, inject=None, graph=False, untied=False):
    ns = build_ns(vroot, model_text, module_path, env=env)
    tm = ns.torch
    cfg = mk_cfg(V, H)
    vcfg = mk_vcfg(cfg, untied=untied)
    bits = make_ckpt(V, H) if bits is None else bits
    rng = np.random.default_rng(5)
    lm_bits = T.f32_to_bf16_bits((rng.standard_normal((V, H)) * 0.02).astype(np.float32))
    rv = T.Rendezvous(W, timeout=6.0)
    ids = [0, 5, V - 60, V - 1, 3, V // 2 + 1, -1, V, 77, 200]

    def body(r):
        w = setup_rank(rv, r, H, vcfg)
        if inject is not None:
            inject(r, ns, w)
        with tm.device("cuda"):
            target = ns.model.Glm5NextForCausalLM(vllm_config=vcfg, prefix="")
        loaded = target.model.load_weights(iter([("embed_tokens.weight", cpu_tensor(ns, bits))]))
        target.lm_head.weight.weight_loader(target.lm_head.weight, cpu_tensor(ns, lm_bits))
        ns.lutil.process_weights_after_loading(target, vcfg.model_config, tm.device("cuda"))
        emb = target.model.embed_tokens
        img = image_layer(ns, V, H, bits)
        ref = image_lookup(img, ids, ns)
        t_ids = tm.tensor(ids, dtype=tm.int32, device="cuda:0")
        pos = tm.arange(len(ids), dtype=tm.int64, device="cuda:0")
        n0 = len(w.ar_calls)
        hs = target.model(t_ids, pos, None)
        n_fwd = len(w.ar_calls) - n0
        out = {"cls": type(emb).__name__, "mode": mode_of(ns, emb), "loaded": "embed_tokens.weight" in loaded,
               "fwd_eq": bool(np.array_equal(hs._a, ref)), "fwd_ars": n_fwd,
               "lm_head": (type(target.lm_head).__name__, tuple(target.lm_head.weight.shape))}
        # the drafter: REAL Glm5NextMTP + REAL load_eagle_model (-> maybe_share_target_embed)
        with tm.device("cuda"):
            draft = ns.mtp.Glm5NextMTP(vllm_config=vcfg, prefix="")
        own = draft.model.embed_tokens
        TLS.draft = draft
        d = ns.eagle.load_eagle_model(target, vcfg)
        layer = d.model._mtp_layers[0]
        out["shared"] = d.model.embed_tokens is emb and own is not emb
        out["own_dropped"] = all(m is not own for m in d.modules())
        out["head_shared"] = layer.shared_head.head is target.lm_head and d.lm_head is target.lm_head
        TLS.eh_inputs = []
        prev = tm.zeros(len(ids), H, dtype=tm.bfloat16, device="cuda:0")
        n0 = len(w.ar_calls)
        d(t_ids, pos, prev)
        out["draft_ars"] = len(w.ar_calls) - n0
        out["draft_eq"] = len(TLS.eh_inputs) == 1 and bool(np.array_equal(TLS.eh_inputs[0]._a, ref))
        if graph:
            static = tm.zeros(4, dtype=tm.int32, device="cuda:0")
            p4 = tm.arange(4, dtype=tm.int64, device="cuda:0")
            target.model(static, p4, None)
            g = tm.cuda.CUDAGraph()
            if out["mode"] == "replicated":
                with tm.cuda.graph(g):
                    o4 = target.model(static, p4, None)
                good = True
                for k in range(3):
                    sel = ids[k * 3:k * 3 + 4] if len(ids[k * 3:k * 3 + 4]) == 4 else ids[:4]
                    static.copy_(tm.tensor(sel, dtype=tm.int32, device="cuda:0"))
                    g.replay()
                    good &= bool(np.array_equal(o4._a, ref[[ids.index(s) for s in sel]]))
                out["graph_eq"] = good and g.replays == 3
            else:
                out["graph_eq"] = None
        return out
    o = run8(body)
    errs = [repr(x) for ok, x in o if not ok]
    rs = [x for ok, x in o if ok]
    return NS(rs=rs, errs=errs, act=T.LOGS.lines(contains="replicated input embedding ACTIVE ("),
              ina=T.LOGS.lines(contains="replicated input embedding INACTIVE ("), ns=ns)


def section_c(vroot: Path, module_path: Path, texts: dict, ap, quick=False):
    print("== C. the REAL patched model.py + common/mtp.py + eagle/utils.py + process_weights_after_loading")
    V, H = 512, 16
    out = {}
    for b in (list(texts) if not quick else ["MODEL_ORIG"]):
        pt = ap.apply(texts[b])
        sc = model_scenario(vroot, pt, module_path, {"VLLM_GB10_REPL_EMBED": "1"}, graph=(b == "MODEL_ORIG"))
        ok = (not sc.errs and len(sc.rs) == W and all(x["mode"] == "replicated" and x["fwd_eq"] and x["fwd_ars"] == 0
                                                       and x["loaded"] for x in sc.rs) and len(sc.act) == W)
        check(f"C2 {b} + 032, gate 1: ACTIVE on 8 ranks; REAL Glm5NextModel.forward == the unpatched embedding "
              "bitwise, 0 all-reduces", ok, f"{sc.errs[:1]} {[x.get('mode') for x in sc.rs][:2]}")
        check(f"C2 {b} + 032: REAL load_eagle_model -> maybe_share_target_embed hands the draft the TARGET's replicated "
              "embed_tokens (common/mtp.py's own copy dropped); REAL Glm5NextMTP.forward embeds with it bitwise, 0 "
              "all-reduces", all(x["shared"] and x["own_dropped"] and x["draft_eq"] and x["draft_ars"] == 0
                                 for x in sc.rs))
        check(f"C2 {b} + 032: lm_head stays a vocab-parallel ParallelLMHead [{V // W}, {H}] shared into "
              "shared_head.head", all(x["lm_head"] == ("ParallelLMHead", (V // W, H)) and x["head_shared"]
                                      for x in sc.rs))
        if b == "MODEL_ORIG":
            check("C5 CUDA-graph capture (tape emulation) of the REAL Glm5NextModel.forward, 3 replays with new ids == "
                  "the unpatched embedding", all(x["graph_eq"] for x in sc.rs))
        out[b] = ok
        sc0 = model_scenario(vroot, pt, module_path, {})
        check(f"C1 {b} + 032, gate unset: embed_tokens is the unpatched VocabParallelEmbedding, forward == unpatched, "
              "all-reduce issued; no patch-032 log line",
              not sc0.errs and all(x["cls"] == "VocabParallelEmbedding" and x["fwd_eq"] and x["fwd_ars"] == 1
                                   for x in sc0.rs) and not sc0.act and not sc0.ina
              and not T.LOGS.lines(contains="[patch-032]"), f"{sc0.errs[:1]}")
        if quick:
            break
    pt = ap.apply(texts["MODEL_ORIG"])

    def fp_bad(r, ns_, w):
        if r == 2:
            T.INJ.fp_override = "0123456789abcdef"
    sc = model_scenario(vroot, pt, _instrumented(module_path), {"VLLM_GB10_REPL_EMBED": "1"}, inject=fp_bad)
    check("C3 a forced boot failure (fingerprint on rank 2): INACTIVE on 8 ranks, the REAL forward == unpatched with its "
          "all-reduce, the draft shares the (unpatched) layer",
          not sc.errs and all(x["mode"] == "image" and x["fwd_eq"] and x["fwd_ars"] == 1 and x["shared"]
                              and x["draft_eq"] and x["draft_ars"] == 1 for x in sc.rs) and len(sc.ina) == W,
          f"{sc.errs[:1]}")
    try:
        build_ns(vroot, pt, None, env={"VLLM_GB10_REPL_EMBED": "1"})
        check("C4 gate 1 + module missing -> ImportError at model.py import", False)
    except ImportError:
        check("C4 gate 1 + module missing -> ImportError at model.py import", True)
    ns = build_ns(vroot, pt, None, env={})
    check("C4 gate unset + module missing: model.py imports, _p032 None", ns.model._p032 is None)
    # C6: the config the checkpoint untied (word_embeddings_untied_by_checkpoint): the unpatched layer is kept
    sc = model_scenario(vroot, pt, module_path, {"VLLM_GB10_REPL_EMBED": "1"}, untied=True)
    check("C6 word_embeddings_untied_by_checkpoint: embed_tokens stays the unpatched VocabParallelEmbedding on 8 ranks "
          "(one INACTIVE line each, forward == unpatched with its all-reduce, draft shares it)",
          not sc.errs and all(x["cls"] == "VocabParallelEmbedding" and x["fwd_eq"] and x["fwd_ars"] == 1
                              and x["shared"] for x in sc.rs) and len(sc.ina) == W and not sc.act, f"{sc.errs[:1]}")
    return out


# =====================================================================================================================
# D. mutants
# =====================================================================================================================
MUTANTS = [
    ("replicated forward without masking (plain gather)",
     "            return _lookup(self._p032_replicated_view(), input_)\n",
     "            return self.quant_method.embedding(self, input_.long())\n"),
    ("replicated view keeps the all-reduce (reduce=True)",
     "                                                    self.embedding_dim, reduce=False)\n",
     "                                                    self.embedding_dim, reduce=True)\n"),
    ("no zero canonicalisation", "        x.masked_fill_(x.view(torch.int16) == BF16_NEG_ZERO, 0.0)\n", "        pass\n"),
    ("subnormals flushed with the zeros", "        x.masked_fill_(x.view(torch.int16) == BF16_NEG_ZERO, 0.0)\n",
     "        x.masked_fill_((x.view(torch.int16) & BF16_EXP) == 0, 0.0)\n"),
    ("negative zeros not scanned", "        nz = xi == BF16_NEG_ZERO\n", "        nz = xi == 12345\n"),
    ("canonicalise before cloning the image shard",
     "            orig = layer.weight.narrow(0, start, rows).clone()     # this rank's image shard bytes "
     "(uncanonicalised)\n            _canonicalize_zeros(layer.weight)\n",
     "            _canonicalize_zeros(layer.weight)\n            orig = layer.weight.narrow(0, start, rows).clone()\n"),
    ("fallback shard offset by one rank",
     "    _to_vocab_parallel(layer, orig)\n    del orig\n",
     "    _to_vocab_parallel(layer, layer.weight.narrow(0, ((layer._p032_rank + 1) % layer._p032_size) * rows, "
     "rows).clone() if layer._p032_mode != 'x' else orig)\n    del orig\n"),
    ("fallback keeps the replicated TP state", "    layer.tp_rank, layer.tp_size = rank, size\n",
     "    layer.tp_rank, layer.tp_size = 0, 1\n"),
    ("fallback forgets use_fused_embedding", "    layer.use_fused_embedding = _vanilla_use_fused(size)\n",
     "    pass\n"),
    ("fallback forgets output_partition_sizes",
     "    layer.output_partition_sizes = [layer.num_embeddings_per_partition]\n", "    pass\n"),
    ("fallback keeps the cached replicated view (full table leaks)",
     "    layer.__dict__.pop(\"_p032_view\", None)         # the replicated view holds the full table\n", "    pass\n"),
    ("reference without the all-reduce", "            ref = world.all_reduce(contrib)",
     "            world.all_reduce(contrib)\n            ref = contrib"),
    ("local decision (no verdict exchange)", '    bad = [v for v in allv if not v["ok"]]',
     '    bad = [v for v in allv if not v["ok"] and v["rank"] == world.rank]'),
    ("exhaustive check dropped (probes only)",
     '    plan = [("all", c0, min(V, c0 + CHUNK_IDS)) for c0 in range(0, V, CHUNK_IDS)]',
     '    plan = []'),
    ("rank-dependent probe ids", "    x = 0x9E3779B9\n",
     "    import threading as _th\n    x = 0x9E3779B9 + (hash(_th.current_thread().name) % 7)\n"),
    ("no dummies after a local failure", "            if contrib is None:      # keep the collective schedule: zero "
     "rows of the planned shape\n", "            if contrib is None:\n                break\n            if False:\n"),
    ("graph check never fails", "                if gbad:\n", "                if False:\n"),
    ("pending mode skips the all-reduce",
     "                           self.embedding_dim, reduce=True)\n",
     "                           self.embedding_dim, reduce=False)\n"),
    ("load count not checked", "    if layer._p032_full_loads != 1:\n", "    if False:\n"),
    ("LoRA not refused", '    if getattr(vc, "lora_config", None) is not None:\n', "    if False:\n"),
    ("untied-by-checkpoint not refused",
     '    if getattr(getattr(vc, "model_config", None), "word_embeddings_untied_by_checkpoint", False):\n',
     "    if False:\n"),
    ("pre-processed weights claimed supported", "    supports_pre_processed_weights = False\n",
     "    supports_pre_processed_weights = True\n"),
    ("content mismatch ignored", "    if refusals or len(fps) != 1 or len(contents) != 1:\n",
     "    if refusals or len(fps) != 1:\n"),
    ("fingerprint mismatch ignored", "    if refusals or len(fps) != 1 or len(contents) != 1:\n",
     "    if refusals or len(contents) != 1:\n"),
    ("non-finite table accepted", '            if scan["nonfinite"]:\n', "            if False:\n"),
    ("ACTIVE although a rank failed", "    if bad:\n        why = \"boot self-test FAILED",
     "    if False:\n        why = \"boot self-test FAILED"),
    ("fused decision hard-wired off in the view", "    return bool(size > 1 and current_platform.is_cuda())\n",
     "    return False\n"),
]


def section_d(vroot: Path, module_path: Path, texts: dict, ap, work: Path):
    print("== D. mutants (each must make section B / C fail)")
    src = module_path.read_text()
    caught = 0
    for name, old, new in MUTANTS:
        if src.count(old) != 1:
            check(f"D mutant '{name}' anchor found {src.count(old)}x", False)
            continue
        mp = Path(tempfile.mkdtemp(prefix="mut.", dir=work)) / "p032_gb10_repl_embed.py"
        mp.write_text(src.replace(old, new, 1))
        snap = {"pass": RESULTS["pass"], "fail": RESULTS["fail"], "failed": list(RESULTS["failed"]),
                "skip": RESULTS["skip"], "skipped": list(RESULTS["skipped"])}
        QUIET["on"] = True
        raised = None
        try:
            section_b(vroot, mp, quick=True)
            if RESULTS["fail"] == snap["fail"]:
                section_c(vroot, mp, texts, ap, quick=True)
            if RESULTS["fail"] == snap["fail"]:
                section_b(vroot, mp, quick=False)
        except Exception as e:  # noqa: BLE001
            raised = e
        finally:
            QUIET["on"] = False
        got = RESULTS["fail"] - snap["fail"] + (1 if raised is not None else 0)
        RESULTS.update(snap)
        caught += got > 0
        check(f"D mutant caught: {name}", got > 0, f"{got} failing checks" + (f", raised {raised!r:.80}" if raised
                                                                              else ""))
    check(f"D all {len(MUTANTS)} mutants caught", caught == len(MUTANTS))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--vllm-root", required=True)
    ap_.add_argument("--work", default=None)
    ap_.add_argument("--mutants", action="store_true")
    ap_.add_argument("--print-pins", action="store_true", help="print every pin line (sh form + pins.txt form), write nothing")
    ap_.add_argument("--write-pins", action="store_true", help="rewrite the pin lines of the .sh in place")
    a = ap_.parse_args()
    vroot = Path(a.vllm_root)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="p032-cpu-"))
    work.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run.", dir=work))
    WORK["dir"] = str(work)
    ap = load_file_module("p032_apply", APPLIER_PATH)
    if a.print_pins or a.write_pins:
        pins = compute_pins(vroot, work, ap)
        for k in ("MODEL_ORIG", "MODEL_P025", "MODEL_ORIG_P032", "MODEL_P025_P032", "VPE_ORIG", "EAGLE_ORIG", "MTP_ORIG",
                  "MTP_P029", "LUTIL_ORIG", "BLOAD_ORIG", "WTIE_ORIG", "APPLIER_SHA", "MOD_SHA"):
            if k in pins:
                print(f"{k}={pins[k]}")
        print("# pins.txt form:")
        for ln in pins_txt_lines(pins):
            print(ln)
        if a.write_pins:
            write_pins(pins)
            print(f"# wrote {SH}")
        sys.exit(0)
    texts = section_a(vroot, work, ap)
    section_b(vroot, MODULE_PATH)
    section_c(vroot, MODULE_PATH, texts, ap)
    if a.mutants:
        section_d(vroot, MODULE_PATH, texts, ap, work)
    print(f"== {RESULTS['pass']} PASS, {RESULTS['fail']} FAIL, {RESULTS['skip']} SKIP"
          + (f": {RESULTS['failed']}" if RESULTS["fail"] else "")
          + (f"  (skipped: {RESULTS['skipped']})" if RESULTS["skip"] else ""))
    sys.exit(0 if RESULTS["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
