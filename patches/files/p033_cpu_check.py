#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-033 gb10-mla-w8a16] CPU checks (Mac / any CPU: python3 >= 3.10 + numpy; NO torch, NO vLLM install).
vLLM v0.32.0 port (module REVISION 2).

  python3 -I -B files/p033_cpu_check.py --vllm-root <v0.32.0 tree>/vllm [--work DIR] [--quick] [--mutants] [--only ABCDE]
                                        [--list-mutants]
  python3 -I -B files/p033_cpu_check.py --vllm-root <...>/vllm --print-pins     (prints the model.py base/result pin lines)
  python3 -I -B files/p033_cpu_check.py --vllm-root <...>/vllm --dump-bases DIR  (writes the bases for p033_sh_check.sh)

--vllm-root is the v0.32.0 `vllm/` package directory (every python file of the tag, sha-checked against the .sh pins):
models/glm5next/common/{model,mtp}.py, model_executor/layers/linear.py, model_executor/layers/quantization/utils/
marlin_utils{,_fp8}.py, _custom_ops.py, model_executor/model_loader/{utils,base_loader}.py, v1/worker/gpu/spec_decode/eagle/utils.py,
plus (read, not pinned) deepseek_v2.py, layers/mla.py, parameter.py, fused-moe stubs. The model.py bases are rebuilt here from the
original with the REAL appliers of the patches that edit it before 033 (p032_apply.py; p025_apply.py once it exists: until then the
+025 rows are explicit SKIP lines, never passes) and compared with the pins.

The REAL files are executed over files/p029_testlib.py's numpy-backed fake torch (read-only; extended below only by
monkeypatching, never by editing the library): marlin_utils.py (including its torch.ops.vllm.marlin_gemm registration through
direct_register_custom_op and the python `_marlin_gemm` wrapper that forwards to torch.ops._C.marlin_gemm), marlin_utils_fp8.py,
the REAL UnquantizedLinearMethod (extracted from linear.py) and the REAL _try_load_fp8_attn_proj / _dequant_fp8_block / the install
statement (extracted from the PATCHED model.py). The fake C++ ops (MF2) model a group-128 W8A16 kernel with the v0.32.0
signatures (gptq_marlin_repack WITHOUT perm; marlin_gemm with b_q_type_id and no g_idx/perm/is_k_full): GPTQ int32 [K/4, N] ->
Marlin [K/16, 4N] through a stand-in byte permutation, scales un-permuted with the REAL get_scale_perms()[0], divided by 2^120
(the folded exponent bias) and applied per 128-row group, weights rounded to bf16 like the kernel's dequant, fp32 accumulation.
THE REAL KERNEL'S TILE PERMUTATION IS NOT MODELLED: the first GPU boot's logged C1-C3 numbers are the layout acceptance.
model_loader/utils.py::process_weights_after_loading is a STAND-IN loop in the threaded boots (the real function drags in the
online-quantisation modules); its per-module contract is verified statically against the REAL v0.32.0 source in A7.

  A  applier + pins: v0.32.0 originals == the .sh pins; the bases == the pins; apply / check / revert on every base == the pinned
     results (revert byte-exact); idempotence; py_compile of every result; AST placement of the hooks; refusals (unknown base
     with every file untouched, marker present, anchors missing / duplicated, hooks in the wrong function); the gate block alone;
     the gate-off path of the REAL patched loader equals the unpatched one bitwise; static enumeration of the chain (the
     model.py writers); A7 the real process_weights_after_loading / base_loader contract.
  B  module units on the fake torch: env parsing, defanging, capture sharding vs an independent loader oracle for every
     TP rank, storage retention, V1-V4 (zero-scale blocks, NaN codes, scale >= 128), the Marlin build against an fp32
     reference implementation (the own-workspace allocation included), the dispatch rule in every state, the self-test failing on
     transposed / mis-scaled / mis-ordered weights, the verdict function, the speed gate; R1: every test namespace carries a
     torch.backends.cuda.matmul whose allow_tf32 raises RuntimeError on any access (a process that used the new TF32 API), so the
     float64 reference GEMMs of the self-test are proven flag-independent.
  C  8-rank threaded boots on the REAL patched loader code and the REAL v0.32.0 Marlin helpers: clean boot -> ACTIVE on
     every rank (one line, one fingerprint, identical schedules); the probe state (MF1: restored on every path, exactly
     one hit per real module call); per-type fallback (SF2); fail-closed scenarios (corrupted / missing / mis-sharded
     capture on one rank, scale >= 128, exchange / barrier failures, fingerprint mismatch, bypassed seams, build
     failure, slow Marlin, capture failure, MTP failures, install refusals) -> the SAME decision on every rank, no hang,
     no schedule mismatch; first small-M call lines (SF8); an MTP-off boot (target group decides alone); R2: a second load
     after the decision returns the module to the image GEMM but keeps its Marlin tensors alive (captured graphs hold raw
     pointers to them) with one restart warning per rank.
  D  real-size numerics (E3 per-rank shapes, synthetic checkpoint) [skipped with --quick].
  E  (--mutants) planted bugs: the seven build-path mutants of MF2 (size_k_first flipped, scales not transposed,
     repeat_interleave on the wrong axis, missing fp8_fused_exponent_bias_into_scales, q_a/kv_a row order swapped, o_proj
     shard on dim 0, q_b scale rows off by one block), MF1's "dispatch requires ACTIVE only", and further module mutants.
Exit 0 = ALL PASS (SKIP lines do not count as passes).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import logging
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
SH = PDIR / "033-gb10-mla-w8a16.sh"
MODULE_PATH = Path(os.environ.get("P033_MODULE_OVERRIDE") or (HERE / "p033_mla_w8a16.py"))
APPLIER_PATH = HERE / "p033_apply.py"
sys.dont_write_bytecode = True                # no .pyc next to the (read-only) vendored libraries
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import p029_testlib as T  # noqa: E402  (read-only)

REL = {
    "model": "models/glm5next/common/model.py",
    "mtp": "models/glm5next/common/mtp.py",
    "linear": "model_executor/layers/linear.py",
    "mu8": "model_executor/layers/quantization/utils/marlin_utils_fp8.py",
    "mu": "model_executor/layers/quantization/utils/marlin_utils.py",
    "cops": "_custom_ops.py",
    "lutil": "model_executor/model_loader/utils.py",
    "bload": "model_executor/model_loader/base_loader.py",
    "eagle": "v1/worker/gpu/spec_decode/eagle/utils.py",
}
PIN_OF = {"model": "MODEL_ORIG", "mtp": "MTP_ORIG", "linear": "LINEAR_ORIG", "mu8": "MU8_ORIG", "mu": "MU_ORIG",
          "cops": "COPS_ORIG", "lutil": "LUTIL_ORIG", "bload": "BLOAD_ORIG", "eagle": "EAGLE_ORIG"}
BASES = ["MODEL_ORIG", "MODEL_P025", "MODEL_ORIG_P032", "MODEL_P025_P032"]
SERVED = "MODEL_ORIG_P032"          # the base whose patched text the module / loader tests run on (025 does not touch a hook)
RESULTS = {"pass": 0, "fail": 0, "skip": 0, "failed": []}
QUIET = {"on": False}
MUT = {"mu8": None}          # mutated text of the image's marlin_utils_fp8.py (mutant runs)


def check(name, ok, info=""):
    ok = bool(ok)
    if not QUIET["on"] or not ok:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    RESULTS["pass" if ok else "fail"] += 1
    if not ok:
        RESULTS["failed"].append(name)
    return ok


def skip(name, why):
    print(f"SKIP {name}  [{why}]", flush=True)
    RESULTS["skip"] += 1


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


def pin_raw(name):
    m = re.search(rf"^{name}=(\S+)$", SH.read_text(), re.M)
    if not m:
        raise SystemExit(f"no pin {name} in {SH}")
    return m.group(1)


def pending(name):
    """True while the pin waits for another task's port (value `pending-<NNN>-port`): such a pin can never equal a sha."""
    return pin_raw(name).startswith("pending-")


def pin(name):
    v = pin_raw(name)
    if not re.fullmatch(r"[0-9a-f]{64}", v):
        raise SystemExit(f"pin {name} in {SH} is {v!r}: not a sha256 (pending another port?)")
    return v


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# =====================================================================================================================
# extensions of the fake torch (monkeypatched onto the read-only library's classes; idempotent)
# =====================================================================================================================
def _patch_fake_torch():
    Tn = T.Tensor
    if getattr(Tn, "_p033_patched", False):
        return

    def repeat_interleave(self, repeats, dim=None):
        if dim is None:
            return self._new(np.repeat(self._a.reshape(-1), int(repeats)))
        return self._new(np.repeat(self._a, int(repeats), axis=dim))

    def untyped_storage(self):
        base = self._a
        while base.base is not None and isinstance(base.base, np.ndarray):
            base = base.base
        return NS(data_ptr=lambda: base.__array_interface__["data"][0], nbytes=lambda: int(base.nbytes))

    def narrow(self, dim, start, length):
        sl = [slice(None)] * self._a.ndim
        sl[dim] = slice(int(start), int(start) + int(length))
        return self._new(self._a[tuple(sl)])

    Tn.repeat_interleave = repeat_interleave
    Tn.untyped_storage = untyped_storage
    Tn.narrow = narrow
    Tn._p033_patched = True


_patch_fake_torch()


# =====================================================================================================================
# fake Marlin ops, group-128 (MF2). Checked against the shapes the REAL marlin_utils(_fp8).py hand them.
# =====================================================================================================================
class MarlinLog:
    calls: list = []


def make_marlin_fakes(get_scale_perms, inject=None, fp8_type_id=None):
    """-> (gptq_marlin_repack, marlin_gemm): fakes of the two C++ ops with their v0.32.0 signatures. `get_scale_perms` is the
    REAL marlin_utils function; `inject` (optional) is a per-rank-thread hook(kind) that may raise (build-failure scenarios);
    `fp8_type_id` is the id of scalar_types.float8_e4m3fn in the stub namespace."""

    def repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit=False):
        if inject is not None:
            inject("repack")
        assert num_bits == 8, num_bits
        assert b_q_weight.dtype is T.int32, b_q_weight.dtype
        assert tuple(b_q_weight.shape) == (size_k // 4, size_n), (tuple(b_q_weight.shape), size_k, size_n)
        assert (size_n % 64 == 0 and size_k % 128 == 0) or (size_n % 128 == 0 and size_k % 64 == 0), (size_n, size_k)
        MarlinLog.calls.append(("repack", size_k, size_n))
        # GPTQ int32 [K/4, N] (element [k4, n] holds W[n, 4*k4 + 0..3], little endian) -> Marlin int32 [K/16, 4N]:
        # stand-in tile permutation (k4 = 4*kt + g): [K/16, g, N, j] -> [K/16, N, g, j], bytes viewed as int32
        by = np.ascontiguousarray(b_q_weight._a).view(np.uint8).reshape(size_k // 16, 4, size_n, 4)
        out = np.ascontiguousarray(by.transpose(0, 2, 1, 3)).view(np.int32).reshape(size_k // 16, size_n * 4)
        return T.Tensor(out, T.int32, b_q_weight.device)

    def gemm(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, workspace, b_q_type_id,
             size_m, size_n, size_k, use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False):
        if inject is not None:
            inject("gemm")
        assert fp8_type_id is not None and b_q_type_id == fp8_type_id, (b_q_type_id, fp8_type_id)
        assert c is None
        assert a.dtype in (T.bfloat16, T.float16), a.dtype
        assert a.is_contiguous(), "marlin_gemm: A must be contiguous"
        assert tuple(a.shape) == (size_m, size_k), (tuple(a.shape), size_m, size_k)
        assert tuple(b_q_weight.shape) == (size_k // 16, size_n * 4), (tuple(b_q_weight.shape), size_k, size_n)
        assert b_scales.dtype is a.dtype, (b_scales.dtype, a.dtype)
        assert tuple(b_scales.shape) == (size_k // 128, size_n), ("group-128 scales [K/128, N]", tuple(b_scales.shape))
        assert b_bias is None and a_scales is None and b_zeros is None and global_scale is None
        assert workspace is not None and workspace.dtype is T.int32 and workspace.numel() > 0, "an explicit lock buffer is required"
        assert not use_atomic_add, "the atomic-add variant is not what E3 measured"
        assert use_fp32_reduce is True, "fp32 global reduce (USE_FP32_REDUCE_DEFAULT) is what E3 measured"
        # packed bytes -> W[n, k]
        by = np.ascontiguousarray(b_q_weight._a).view(np.uint8).reshape(size_k // 16, size_n, 4, 4)
        gptq = np.ascontiguousarray(by.transpose(0, 2, 1, 3)).reshape(size_k // 4, size_n, 4)
        w8 = np.ascontiguousarray(gptq.transpose(1, 0, 2)).reshape(size_n, size_k)
        wf = T.fp8_bits_to_f32(w8)
        # scales: undo the REAL scale permutation, the folded 2^120, expand per 128-row group
        sp = np.array(get_scale_perms()[0])
        s2 = b_scales._f().reshape(-1, len(sp))[:, np.argsort(sp)].reshape(size_k // 128, size_n)
        s = s2.astype(np.float64) / (2.0 ** 120)
        s_full = np.repeat(s, 128, axis=0).T                                        # [N, K]
        with np.errstate(all="ignore"):
            w_eff = T.bf16_bits_to_f32(T.f32_to_bf16_bits((wf.astype(np.float64) * s_full).astype(np.float32)))
        with np.errstate(all="ignore"):             # mutated scale folds overflow to inf / nan on purpose
            acc = a._f().astype(np.float32) @ w_eff.T.astype(np.float32)
        MarlinLog.calls.append(("gemm", size_m, size_n, size_k))
        g = T.capturing_graph()
        if g is not None:
            g.ops.append(("marlin", (size_n, size_k)))
        return T.Tensor(T._encode(acc, a.dtype), a.dtype, a.device)

    return repack, gemm


# =====================================================================================================================
# stub vLLM namespace (over the fake torch); the REAL files are executed into it
# =====================================================================================================================
class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self._l = threading.Lock()

    def emit(self, rec):
        with self._l:
            self.records.append((threading.current_thread().name, rec.levelname, rec.getMessage()))

    def lines(self, rank=None, contains=None, level=None):
        with self._l:
            return [m for (t, lv, m) in self.records if (rank is None or t == f"rank{rank}")
                    and (contains is None or contains in m) and (level is None or lv == level)]

    def clear(self):
        with self._l:
            self.records.clear()


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


def _exec_text(name, text, package, filename):
    m = types.ModuleType(name)
    m.__file__ = filename
    m.__package__ = package
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    exec(compile(text, filename, "exec"), m.__dict__)
    return m


def extract_defs(text: str, names: list[str]) -> str:
    """Source of the named top-level defs / classes / assignments, in file order."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    out, found = [], set()
    for node in tree.body:
        nm = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            out.append("".join(lines[start - 1: node.end_lineno]) + "\n")
            found.add(nm)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"definitions not found: {sorted(missing)}")
    return "".join(out)


def install_stmt_source(model_text: str) -> str | None:
    """The `if _p033 is not None ...:` statement of Glm5NextDecoderLayer.__init__ (the REAL patched text), as a function
    body source, or None when the text carries no hook."""
    tree = ast.parse(model_text)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Glm5NextDecoderLayer":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    for st in fn.body:
                        if isinstance(st, ast.If) and "_p033" in ast.unparse(st):
                            return ast.unparse(st)
    return None


TLS = threading.local()
COST = {"bf16_bps": 200e9, "marlin_bps": 200e9}        # bytes/s of the virtual GPU clock; tests override


def graph_cost(ops, world) -> float:
    scale = getattr(world, "cost_scale", None) or {}
    ms = 0.0
    for op in ops:
        if op[0] == "linear":
            n, k = op[1]
            ms += n * k * 2 / (COST["bf16_bps"] * scale.get("bf16", 1.0)) * 1e3
        elif op[0] == "marlin":
            n, k = op[1]
            ms += n * k * 1.02 / (COST["marlin_bps"] * scale.get("marlin", 1.0)) * 1e3
    return ms


def build_ns(vroot: Path, *, model_text: str | None = None, module_path: Path | None = None, mu8_text: str | None = None,
             env: dict | None = None, inject=None, with_marlin=True, cap=(12, 1)):
    """Fresh fake torch + stub vLLM + the REAL files. env is applied to os.environ BEFORE anything is executed."""
    for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.") or k == "torch" or k.startswith("torch.")]:
        del sys.modules[k]
    for k in ("VLLM_GB10_MLA_W8A16", "VLLM_GB10_MLA_W8A16_MAX_M", "VLLM_GB10_MLA_W8A16_SPEED_GATE", "VLLM_BATCH_INVARIANT",
              "VLLM_GB10_REPL_EMBED", "VLLM_GB10_DRAFT_COMPRESS"):
        os.environ.pop(k, None)
    for k, v in (env or {}).items():
        os.environ[k] = v
    tm = T.make_fake_torch()
    T.set_default_dtype(T.bfloat16)                      # vLLM builds the model under set_default_torch_dtype(bf16)
    tm.cuda.get_device_capability = lambda d=None: cap

    class _Tf32Trap:
        """torch.backends.cuda.matmul of a process that has used the NEW TF32 API (fp32_precision): any READ or WRITE of the
        legacy allow_tf32 flag raises RuntimeError (reproduced on torch 2.14, finding R1). Patch 033 must never touch it: with
        the trap installed in EVERY test namespace, a regression to a flag-dependent reference GEMM refuses the boot."""
        @property
        def allow_tf32(self):
            raise RuntimeError("mixed legacy / new TF32 API (test trap): 033 must not touch torch.backends.*.allow_tf32")

        @allow_tf32.setter
        def allow_tf32(self, v):
            raise RuntimeError("mixed legacy / new TF32 API (test trap): 033 must not touch torch.backends.*.allow_tf32")
    tm.backends = NS(cuda=NS(matmul=_Tf32Trap()))
    sys.modules["torch"] = tm
    for sub in ("nn", "nn.functional", "cuda", "distributed"):
        obj = tm
        for part in sub.split("."):
            obj = getattr(obj, part)
        sys.modules["torch." + sub] = obj
    torch = tm
    bf16 = torch.bfloat16
    T.GRAPH_COST_FN = graph_cost
    lg = logging.getLogger("p033test")
    lg.setLevel(logging.INFO)
    if LOGS not in lg.handlers:
        lg.addHandler(LOGS)
    lg.propagate = False
    _mod("vllm", pkg=True)

    class _VLogger:
        def __init__(self, base):
            self._b = base

        def __getattr__(self, k):
            return getattr(self._b, k[:-5] if k.endswith("_once") else k)
    vlg = _VLogger(lg)
    _mod("vllm.logger", init_logger=lambda name: vlg)

    class VllmConfig:
        pass
    _mod("vllm.config", VllmConfig=VllmConfig, get_current_vllm_config=lambda: NS())

    def tp_rank():
        return T.current_world().rank

    def tp_size():
        return T.current_world().size

    _mod("vllm.distributed", pkg=True, get_tensor_model_parallel_rank=tp_rank,
         get_tensor_model_parallel_world_size=tp_size, get_pp_group=lambda: NS(world_size=T.current_world().pp_size))
    _mod("vllm.distributed.parallel_state", get_pp_group=lambda: NS(world_size=T.current_world().pp_size),
         get_tp_group=lambda: (_ for _ in ()).throw(RuntimeError("get_tp_group is not used in these tests")))
    _mod("vllm.distributed.utils", verify_group_size_divides_partition=lambda *a, **k: None)
    plat = NS(use_all_gather=lambda: True, device_type="cuda", has_device_capability=lambda c: True,
              is_cuda=lambda: True, is_rocm=lambda: False, is_cpu=lambda: False, is_xpu=lambda: False,
              is_cuda_alike=lambda: True, is_device_capability=lambda c: False, is_device_capability_family=lambda c: True)
    _mod("vllm.platforms", current_platform=plat)
    envs = _mod("vllm.envs", VLLM_MARLIN_USE_ATOMIC_ADD=False, VLLM_MARLIN_INPUT_DTYPE=None, VLLM_BATCH_INVARIANT=False,
                VLLM_XPU_FORCE_N_CONTIG_WEIGHT=False)

    class ScalarType:
        _next = [1000]

        def __init__(self, name):
            self.name = name
            self.id = ScalarType._next[0]            # v0.32.0 passes scalar_types.float8_e4m3fn.id to torch.ops.vllm.marlin_gemm
            ScalarType._next[0] += 1
    _mod("vllm.scalar_type", ScalarType=ScalarType,
         scalar_types=NS(float8_e4m3fn=ScalarType("float8_e4m3fn"), uint4b8=ScalarType("uint4b8"),
                         uint8b128=ScalarType("uint8b128"), float4_e2m1f=ScalarType("float4_e2m1f")))
    _mod("vllm.utils", pkg=True)
    _mod("vllm.utils.math_utils", round_up=lambda x, m: (x + m - 1) // m * m)
    _mod("vllm.utils.platform_utils", num_compute_units=lambda idx=None: 48)
    tm.ops.vllm = NS()

    def direct_register_custom_op(op_name, op_func, mutates_args=None, fake_impl=None, target_lib=None, dispatch_key=None,
                                  tags=()):
        """vllm.utils.torch_utils.direct_register_custom_op: the op becomes torch.ops.vllm.<op_name> (called with keyword
        arguments by the REAL marlin_utils_fp8.py); here it forwards to the REAL python wrapper (`_marlin_gemm`)."""
        setattr(tm.ops.vllm, op_name, lambda *a, **k: op_func(*a, **k))
    _mod("vllm.utils.torch_utils", direct_register_custom_op=direct_register_custom_op)
    _mod("vllm.model_executor", pkg=True)

    def replace_parameter(layer, name, new, prefer_copy=False):
        setattr(layer, name, torch.nn.Parameter(new, requires_grad=False))
    _mod("vllm.model_executor.utils", replace_parameter=replace_parameter,
         set_weight_attrs=lambda *a, **k: None)
    _mod("vllm.model_executor.layers", pkg=True)

    # ---- linear.py: the REAL UnquantizedLinearMethod (+ stubs for what it names)
    class QuantizeMethodBase:
        def process_weights_after_loading(self, layer):
            return None

    lin_ns = {"torch": torch, "QuantizeMethodBase": QuantizeMethodBase, "abstractmethod": lambda f: f, "envs": envs,
              "current_platform": plat, "linear_batch_invariant": lambda x, w, b: torch.nn.functional.linear(x, w, b),
              # v0.32.0: UnquantizedLinearMethod.__init__ resolves its GEMM once from the construction-time vLLM config
              "get_current_vllm_config_or_none": lambda: None,
              "dispatch_unquantized_gemm": lambda backend="auto": (lambda layer, x, w, b: torch.nn.functional.linear(x, w, b)),
              "set_weight_attrs": lambda *a, **k: None, "ModelWeightParameter": object,
              "Parameter": torch.nn.Parameter, "Any": object}
    lin_text = (vroot / REL["linear"]).read_text()
    lin_src = extract_defs(lin_text, ["LinearMethodBase", "UnquantizedLinearMethod"])
    lm = types.ModuleType("vllm.model_executor.layers.linear")
    lm.__dict__.update(lin_ns)
    sys.modules[lm.__name__] = lm
    setattr(sys.modules["vllm.model_executor.layers"], "linear", lm)
    exec(compile(lin_src, str(vroot / REL["linear"]), "exec"), lm.__dict__)
    lm.LinearBase = type("LinearBase", (torch.nn.Module,), {})
    _mod("vllm.model_executor.layers.fused_moe", pkg=True, RoutedExperts=type("RoutedExperts", (), {}))
    _mod("vllm.model_executor.layers.fused_moe.config", FusedMoEConfig=type("FusedMoEConfig", (), {}))
    _mod("vllm.model_executor.layers.quantization", pkg=True)
    _mod("vllm.model_executor.layers.quantization.input_quant_fp8", QuantFP8=type("QuantFP8", (), {}))
    _mod("vllm.model_executor.layers.quantization.utils", pkg=True)
    _mod("vllm.model_executor.layers.quantization.utils.int8_utils", per_token_quant_int8=lambda x: None)
    _mod("vllm.model_executor.layers.quantization.utils.quant_utils", GroupShape=NS(PER_TOKEN="per_token"),
         pack_cols=lambda *a: None, unpack_cols=lambda *a: None)
    ops = NS()

    def _cops_repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit=False):
        """vllm._custom_ops.gptq_marlin_repack (v0.32.0: no `perm`)."""
        return ops.gptq_marlin_repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit)
    _mod("vllm._custom_ops", gptq_marlin_repack=_cops_repack)
    mu = _exec_text("vllm.model_executor.layers.quantization.utils.marlin_utils", (vroot / REL["mu"]).read_text(),
                    "vllm.model_executor.layers.quantization.utils", str(vroot / REL["mu"]))
    mu8_src = mu8_text if mu8_text is not None else (MUT["mu8"] if MUT["mu8"] is not None
                                                     else (vroot / REL["mu8"]).read_text())
    mu8 = _exec_text("vllm.model_executor.layers.quantization.utils.marlin_utils_fp8", mu8_src,
                     "vllm.model_executor.layers.quantization.utils", str(vroot / REL["mu8"]))
    if with_marlin:
        fp8_id = sys.modules["vllm.scalar_type"].scalar_types.float8_e4m3fn.id
        repack, gemm = make_marlin_fakes(mu.get_scale_perms, inject, fp8_id)
        tm.ops._C.marlin_gemm, tm.ops._C.gptq_marlin_repack = gemm, repack      # the C++ ops the REAL wrappers forward to
        ops.gptq_marlin_repack = repack
    else:
        ops.gptq_marlin_repack = None
        del tm.ops.vllm.marlin_gemm                                               # "no Marlin op registered"
    # ---- the patch module
    mod = None
    if module_path is not None:
        mod = _exec_text("vllm._gb10_mla_w8a16", Path(module_path).read_text(), "vllm", str(module_path))
        mod._WORLD_FACTORY = T.current_world
        mod._set_ctx_key_for_test(lambda: getattr(T.current_world(), "rank", 0))
    # ---- the REAL loader code of the (patched) model.py
    fns = None
    inst = None
    if model_text is not None:
        g = {"torch": torch, "_p033": mod, "__name__": "model_under_test"}
        exec(compile(extract_defs(model_text, ["_dequant_fp8_block", "_FP8_ATTN_PROJS", "_try_load_fp8_attn_proj"]),
                     "model.py(extracted)", "exec"), g)
        fns = NS(try_load=g["_try_load_fp8_attn_proj"], dequant=g["_dequant_fp8_block"], g=g)
        isrc = install_stmt_source(model_text)
        if isrc is not None:
            body = "\n".join("    " + ln for ln in isrc.splitlines())
            ig = {"_p033": mod}
            exec(compile(f"def _stmt(self, vllm_config, layer_idx, is_mtp_layer):\n{body}\n", "model.py(E2)", "exec"), ig)
            inst = ig["_stmt"]
    return NS(torch=torch, mu=mu, mu8=mu8, mod=mod, UQM=lm.UnquantizedLinearMethod, fns=fns, install_stmt=inst,
              plat=plat, envs=envs, ops=ops)


# =====================================================================================================================
# synthetic checkpoint, per-rank model, the REAL loader code, the PWAL stand-in, the rank runner
# =====================================================================================================================
SMALL = NS(name="small", hidden=256, q_lora=128, kv=128, qb_rows=128, o_cols=128, tp=8)
REAL = NS(name="real", hidden=4096, q_lora=1536, kv=512, qb_rows=2048, o_cols=2048, tp=8)
TARGET_LAYERS = (3, 7)
MTP_LAYER = 45


def quant_blocks(w: np.ndarray, rng, zero_blocks=()):
    n, k = w.shape
    blk = w.reshape(n // 128, 128, k // 128, 128)
    amax = np.abs(blk).max(axis=(1, 3))
    sc = (amax / 448.0).astype(np.float32)
    for (bi, bj) in zero_blocks:
        sc[bi, bj] = 0.0
    safe = np.where(sc == 0, 1.0, sc)
    q = (w / np.repeat(np.repeat(safe, 128, 0), 128, 1)).astype(np.float32)
    return T.f32_to_fp8_bits(q), sc


def make_ckpt(dims, layer_ids=TARGET_LAYERS + (MTP_LAYER,), seed=1, zero_blocks=None, only_rank=None):
    """{layer: {key: (fp8 bytes uint8 [N,K], float32 block scales)}} for the FULL (unsharded) checkpoint tensors. With
    only_rank (real-size tests) only that rank's shard of q_b / o_proj holds data (the rest is an inert filler)."""
    rng = np.random.default_rng(seed)
    out = {}
    for L in layer_ids:
        d = {}
        shapes = {"q_a": (dims.q_lora, dims.hidden), "kv_a": (dims.kv, dims.hidden),
                  "q_b": (dims.qb_rows * dims.tp, dims.q_lora), "o_proj": (dims.hidden, dims.o_cols * dims.tp)}
        for key, (n, k) in shapes.items():
            zb = (zero_blocks or {}).get((L, key), ())
            if only_rank is not None and key in ("q_b", "o_proj"):
                w8 = np.zeros((n, k), np.uint8)
                sc = np.ones((n // 128, k // 128), np.float32)
                r = only_rank
                if key == "q_b":
                    P = dims.qb_rows
                    w = (rng.standard_normal((P, k)) * 0.03).astype(np.float32)
                    a, b = quant_blocks(w, rng, zb)
                    w8[r * P:(r + 1) * P], sc[r * P // 128:(r + 1) * P // 128] = a, b
                else:
                    P = dims.o_cols
                    w = (rng.standard_normal((n, P)) * 0.03).astype(np.float32)
                    a, b = quant_blocks(w, rng, zb)
                    w8[:, r * P:(r + 1) * P], sc[:, r * P // 128:(r + 1) * P // 128] = a, b
                d[key] = (w8, sc)
            else:
                w = (rng.standard_normal((n, k)) * 0.03).astype(np.float32)
                d[key] = quant_blocks(w, rng, zb)
        out[L] = d
    return out


def ck_tensors(ck, key):
    w8, sc = ck[key]
    return (T.Tensor(w8.copy(), T.float8_e4m3fn, "cuda:0"), T.Tensor(sc.copy(), T.float32, "cuda:0"))


def dequant_bf16(w8: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """model.py _dequant_fp8_block, numpy: bf16 bits of fp32(w) * fp32(block scale)."""
    s_full = np.repeat(np.repeat(sc.astype(np.float32), 128, 0), 128, 1)[: w8.shape[0], : w8.shape[1]]
    return T.f32_to_bf16_bits(T.fp8_bits_to_f32(w8) * s_full)


def oracle_weight(ck, typ, rank, dims) -> np.ndarray:
    """The BF16 weight (uint16 bits) the image loader leaves in this rank's module: independent re-implementation of the
    narrow rules (replicated fused rows q_a then kv_a; column-parallel rows; row-parallel columns)."""
    if typ == "qkv_a":
        return np.concatenate([dequant_bf16(*ck["q_a"]), dequant_bf16(*ck["kv_a"])], axis=0)
    full = dequant_bf16(*ck["q_b" if typ == "q_b" else "o_proj"])
    if typ == "q_b":
        return full[rank * dims.qb_rows:(rank + 1) * dims.qb_rows]
    return full[:, rank * dims.o_cols:(rank + 1) * dims.o_cols]


def make_linear_cls(ns):
    torch = ns.torch

    class Lin(torch.nn.Module):
        """LinearBase stand-in: weight Parameter, the REAL UnquantizedLinearMethod, the vLLM v2 loaders' narrow rules."""

        def __init__(self, kind, n, k, tp_rank, tp_size, n_q=0, bypass=False):
            super().__init__()
            self.kind, self.tp_rank, self.tp_size, self.n_q, self.bypass = kind, tp_rank, tp_size, n_q, bypass
            self.weight = torch.nn.Parameter(torch.zeros(n, k, dtype=torch.bfloat16, device="cuda:0"), requires_grad=False)
            self.weight.weight_loader = self.weight_loader
            self.quant_method = ns.UQM()
            self.reduce_results = kind == "row"
            self.bias = None
            self.calls = 0

        def weight_loader(self, param, loaded, shard_id=None):
            if self.kind == "fused":
                r0 = 0 if shard_id == 0 else self.n_q
                param[r0:r0 + loaded.shape[0]].copy_(loaded)
            elif self.kind == "col":
                p = param.shape[0]
                param.copy_(loaded.narrow(0, self.tp_rank * p, p))
            else:
                p = param.shape[1]
                param.copy_(loaded.narrow(1, self.tp_rank * p, p))

        def forward(self, x):
            self.calls += 1
            if self.bypass == "marlin":                     # a dedicated path with the SAME bits that never reaches apply
                out = ns.mod._marlin(self.quant_method._p, x)
            elif self.bypass:                               # a dedicated kernel that never reaches quant_method.apply
                out = torch.nn.functional.linear(x, self.weight)
            else:
                out = self.quant_method.apply(self, x, None)
            if self.kind == "row" and self.reduce_results and self.tp_size > 1:
                w = T.current_world()
                vals = w._collective(("all_reduce", tuple(out.shape)), out._a.copy())
                acc = np.zeros(out._a.shape, dtype=np.float32)
                for v in vals:
                    acc = (acc + T.bf16_bits_to_f32(v)).astype(np.float32)
                out = T.Tensor(T.f32_to_bf16_bits(acc), T.bfloat16, out.device)
            return out, None

    class Attn(torch.nn.Module):
        def __init__(self, dims, rank, bypass=(), bypass_marlin=()):
            super().__init__()
            self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim = dims.q_lora, dims.kv, 0

            def by(t):
                return "marlin" if t in bypass_marlin else (t in bypass)
            self.fused_qkv_a_proj = Lin("fused", dims.q_lora + dims.kv, dims.hidden, 0, 1, n_q=dims.q_lora, bypass=by("qkv_a"))
            self.q_b_proj = Lin("col", dims.qb_rows, dims.q_lora, rank, dims.tp, bypass=by("q_b"))
            self.o_proj = Lin("row", dims.hidden, dims.o_cols, rank, dims.tp, bypass=by("o_proj"))

    class Layer(torch.nn.Module):
        def __init__(self, dims, rank, bypass=(), bypass_marlin=()):
            super().__init__()
            self.layer_kind = "mla"
            self.self_attn = Attn(dims, rank, bypass, bypass_marlin)

    class Model(torch.nn.Module):
        pass

    return NS(Lin=Lin, Attn=Attn, Layer=Layer, Model=Model)


def make_vc(expect_mtp=True, lora=False, ubatch=False):
    return NS(speculative_config=NS(method="mtp", num_speculative_tokens=2) if expect_mtp else None,
              lora_config=NS() if lora else None, parallel_config=NS(use_ubatching=bool(ubatch)))


class Hooks:
    """Per-scenario knobs; every callable runs inside the rank thread."""

    def __init__(self, **kw):
        self.after_construct = None       # fn(rank, rb, phase)
        self.after_load = None            # fn(rank, rb, phase)   (before process_weights_after_loading)
        self.before_pwal = None
        self.after_pwal = None            # fn(rank, rb, phase)
        self.world = None                 # fn(rank, world)
        self.pp = lambda r: 1
        self.ckpt_for = None              # fn(rank, ckpt) -> ckpt
        self.skip_load = None             # fn(rank, phase, name) -> bool
        self.bypass = ()
        self.bypass_marlin = ()
        self.vc = make_vc()
        self.mtp = True
        self.layers = TARGET_LAYERS
        self.load_twice = False
        self.tp_rank_of = lambda r: r
        self.attn_mut = None              # fn(rank, attn) before install
        self.__dict__.update(kw)


def build_phase(ns, dims, rank, layer_ids, mtp, hooks):
    cls = make_linear_cls(ns)
    model = cls.Model()
    layers = {}
    for L in layer_ids:
        layer = cls.Layer(dims, hooks.tp_rank_of(rank), hooks.bypass, hooks.bypass_marlin)
        setattr(model, f"layer_{L}", layer)
        layers[L] = layer
        if hooks.attn_mut:
            hooks.attn_mut(rank, layer.self_attn)
        if ns.install_stmt is not None:
            ns.install_stmt(layer, hooks.vc, L, mtp)
    pd = {}
    for L, layer in layers.items():
        mid = "mtp_block." if mtp else ""
        base = f"model.layers.{L}.{mid}self_attn"
        pd[f"{base}.fused_qkv_a_proj.weight"] = layer.self_attn.fused_qkv_a_proj.weight
        pd[f"{base}.q_b_proj.weight"] = layer.self_attn.q_b_proj.weight
        pd[f"{base}.o_proj.weight"] = layer.self_attn.o_proj.weight
    return NS(model=model, layers=layers, params_dict=pd, mtp=mtp)


def load_phase(ns, rb, ckpt, rank, hooks, phase):
    buf, loaded = {}, set()
    for L in rb.layers:
        mid = "mtp_block." if rb.mtp else ""
        base = f"model.layers.{L}.{mid}self_attn"
        for key, suffix in (("q_a", ".q_a_proj"), ("kv_a", ".kv_a_proj_with_mqa"), ("q_b", ".q_b_proj"),
                            ("o_proj", ".o_proj")):
            w8, sc = ck_tensors(ckpt[L], key)
            for name, t in ((f"{base}{suffix}.weight", w8), (f"{base}{suffix}.weight_scale_inv", sc)):
                if hooks.skip_load is not None and hooks.skip_load(rank, phase, name):
                    continue
                handled = ns.fns.try_load(name, t, buf, rb.params_dict, loaded, 0)
                if not handled:
                    raise AssertionError(f"the REAL loader did not take {name}")
    return loaded


def pwal_standin(rb):
    """STAND-IN for model_loader/utils.py::process_weights_after_loading (that file is not on the Mac): every module of
    the model in named_modules() order, quant_method.process_weights_after_loading(module)."""
    for _name, m in rb.model.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(m)


def boot_rank(ns, dims, ckpt, rank, hooks):
    rbs = []
    ck = hooks.ckpt_for(rank, ckpt) if hooks.ckpt_for else ckpt
    phases = [("target", hooks.layers, False)] + ([("mtp", (MTP_LAYER,), True)] if hooks.mtp else [])
    for phase, lids, is_mtp in phases:
        rb = build_phase(ns, dims, rank, lids, is_mtp, hooks)
        rbs.append(rb)
        if hooks.after_construct:
            hooks.after_construct(rank, rb, phase)
        load_phase(ns, rb, ck, rank, hooks, phase)
        if hooks.load_twice:
            load_phase(ns, rb, ck, rank, hooks, phase)
        if hooks.after_load:
            hooks.after_load(rank, rb, phase)
        pwal_standin(rb)
        if hooks.after_pwal:
            hooks.after_pwal(rank, rb, phase)
    return rbs


def run_boot(ns, dims, ckpt, *, size=8, hooks=None, rv_timeout=6.0):
    hooks = hooks or Hooks()
    LOGS.clear()
    T.CUDAGraph.REGISTRY.clear()
    rv = T.Rendezvous(size, timeout=rv_timeout)
    worlds = [None] * size

    def fn(r):
        threading.current_thread().name = f"rank{r}"
        w = T.ThreadWorld(rv, r, size, oneshot_max=None, pp_size=hooks.pp(r))
        worlds[r] = w
        T.set_current_world(w)
        if hooks.world:
            hooks.world(r, w)
        return boot_rank(ns, dims, ckpt, r, hooks)

    res, err = T.run_ranks(size, fn, timeout=180.0)
    return NS(res=res, err=err, rv=rv, worlds=worlds, size=size, hooks=hooks)


# ---- reading the outcome -------------------------------------------------------------------------------------------
def final_lines(b, r):
    return [m for m in LOGS.lines(rank=r, contains="[patch-033]") if " ACTIVE (rank" in m or " INACTIVE (fail closed)" in m
            or "MLA FP8 W8A16 (Marlin) ACTIVE" in m]


def decision(b, r):
    ls = LOGS.lines(rank=r, contains="MLA FP8 W8A16")
    act = [m for m in ls if "(Marlin) ACTIVE (rank" in m]
    ina = [m for m in ls if "INACTIVE (fail closed)" in m]
    return "ACTIVE" if act and not ina else ("INACTIVE" if ina and not act else ("NONE" if not ls else "MIXED"))


def decisions(b):
    return [decision(b, r) for r in range(b.size)]


def all_modules(b):
    out = []
    for r in range(b.size):
        for rb in (b.res[r] or []):
            for L, layer in rb.layers.items():
                a = layer.self_attn
                out.append((r, L, "qkv_a", a.fused_qkv_a_proj))
                out.append((r, L, "q_b", a.q_b_proj))
                out.append((r, L, "o_proj", a.o_proj))
    return out


def healthy(b):
    """No exception on any rank, no hang, no schedule mismatch; every rank issued the same collective schedule."""
    errs = [e for e in b.err if e is not None]
    sch = [tuple(map(str, s)) for s in b.rv.schedules]
    return (not errs and not b.rv.hangs and not b.rv.mismatches and len(set(sch)) == 1), (
        f"errs {[e[1].strip().splitlines()[-1] for e in errs][:2]} hangs {b.rv.hangs[:1]} mism {b.rv.mismatches[:1]}")


def fingerprints(b):
    out = set()
    for r in range(b.size):
        for m in final_lines(b, r):
            mo = re.search(r"fingerprint ([0-9a-f]{16})", m)
            if mo:
                out.add(mo.group(1))
    return out


def new_ns(vroot, base_text, *, env=None, module_path=None, **kw):
    return build_ns(vroot, model_text=base_text, module_path=module_path or MODULE_PATH,
                    env=dict({"VLLM_GB10_MLA_W8A16": "1"}, **(env or {})), **kw)


# =====================================================================================================================
# A: applier, pins, bases
# =====================================================================================================================
BASE_DIRS = {"MODEL_ORIG": "orig", "MODEL_P025": "p025", "MODEL_ORIG_P032": "orig_p032", "MODEL_P025_P032": "p025_p032"}
APPLIER_OF = {"p025": HERE / "p025_apply.py", "p032": HERE / "p032_apply.py", "p029": HERE / "p029_apply.py"}


def _run(cmd, env=None, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


def _apply_cli(applier: Path, text: str, work: Path, tag: str, *args) -> str:
    """The text after `applier apply <file>` ('' when the applier is absent or refuses the base)."""
    if not applier.exists():
        return ""
    d = Path(tempfile.mkdtemp(prefix=f"b_{tag}.", dir=work))
    f = d / "model.py"
    f.write_text(text)
    rc = _run([sys.executable, str(applier), "apply", *args, str(f)])
    return f.read_text() if rc.returncode == 0 else ""


def derive_bases(vroot: Path, work: Path) -> dict:
    """The model.py bases the entrypoint can hand over to 033 (v0.32.0 original; + 025; + 032; + 025 + 032), by running the REAL
    appliers of the earlier patches (p025_apply.py once it exists; p032_apply.py). '' = not derivable (applier absent/refusing)."""
    orig = (vroot / REL["model"]).read_text()
    out = {"MODEL_ORIG": orig}
    out["MODEL_P025"] = _apply_cli(APPLIER_OF["p025"], orig, work, "p025")
    out["MODEL_ORIG_P032"] = _apply_cli(APPLIER_OF["p032"], orig, work, "p032a")
    out["MODEL_P025_P032"] = _apply_cli(APPLIER_OF["p032"], out["MODEL_P025"], work, "p032b") if out["MODEL_P025"] else ""
    return out


def read_bases(bases_dir: Path) -> dict:
    return {var: (bases_dir / dn / "model.py").read_text() for var, dn in BASE_DIRS.items()
            if (bases_dir / dn / "model.py").exists()}


def mtp_p029_result(vroot: Path, work: Path) -> str:
    """sha256 of mtp.py after patch 029 ('' when 029's applier is absent or refuses)."""
    ap29 = APPLIER_OF["p029"]
    if not ap29.exists():
        return ""
    d = Path(tempfile.mkdtemp(prefix="mtp29.", dir=work))
    f = d / "mtp.py"
    f.write_text((vroot / REL["mtp"]).read_text())
    rc = _run([sys.executable, str(ap29), "check", "mtp", str(f)])
    return rc.stdout.strip() if rc.returncode == 0 and re.fullmatch(r"[0-9a-f]{64}", rc.stdout.strip()) else ""


def print_pins(vroot: Path, work: Path, ap) -> int:
    """The model.py base / result pin lines for the .sh, as the CURRENT appliers produce them."""
    texts = derive_bases(vroot, work)
    lines = []
    for var in BASES:
        s = sha_text(texts[var]) if texts[var] else None
        why = "pending-025-port" if var.startswith("MODEL_P025") else "pending-032-port"
        lines.append((var, s or why))
    for var in BASES:
        res_name = {"MODEL_ORIG": "MODEL_ORIG_P033", "MODEL_P025": "MODEL_P025_P033", "MODEL_ORIG_P032": "MODEL_ORIG_P032_P033",
                    "MODEL_P025_P032": "MODEL_P025_P032_P033"}[var]
        why = "pending-025-port" if var.startswith("MODEL_P025") else "pending-032-port"
        lines.append((res_name, sha_text(ap.apply(texts[var])) if texts[var] else why))
    m29 = mtp_p029_result(vroot, work)
    lines.append(("MTP_P029", m29 or "pending-029-port"))
    for name, val in lines:
        print(f"{name}={val}")
    return 0


def section_a(vroot: Path, work: Path, bases_dir: Path | None):
    print("== A. pins / bases / applier / static enumeration")
    ap = load_file_module("p033_apply", APPLIER_PATH)
    # A1: pins of the v0.32.0 originals
    for k, p_ in PIN_OF.items():
        f = vroot / REL[k]
        check(f"A1 --vllm-root {REL[k]} == the .sh pin {p_}", sha_file(f) == pin(p_))

    # pin provenance: the pins this patch copies from earlier patches are those patches' pins
    def pin_of(sh, name):
        f = PDIR / sh
        if not f.exists():
            return None
        m = re.search(rf"^{name}=([0-9a-f]{{64}})$", f.read_text(), re.M)
        return m.group(1) if m else None
    for mine, (sh, theirs) in {"LUTIL_ORIG": ("032-gb10-repl-embed.sh", "LUTIL_ORIG"),
                               "BLOAD_ORIG": ("032-gb10-repl-embed.sh", "BLOAD_ORIG"),
                               "EAGLE_ORIG": ("029-gb10-draft-compress.sh", "EAGLE_ORIG"),
                               "MU8_ORIG": ("029-gb10-draft-compress.sh", "MU8_ORIG"),
                               "MU_ORIG": ("029-gb10-draft-compress.sh", "MU_ORIG"),
                               "LINEAR_ORIG": ("025-gb10-mhc-prefill-sp.sh", "LINEAR_ORIG"),
                               "MTP_ORIG": ("029-gb10-draft-compress.sh", "MTP_ORIG"),
                               "MTP_P029": ("029-gb10-draft-compress.sh", "MTP_P029")}.items():
        other = pin_of(sh, theirs)
        if other is None:
            skip(f"A1 pin {mine} == {sh} {theirs}", f"{sh} absent or its pin is not a sha yet")
        elif pending(mine):
            check(f"A1 pin {mine} is still pending although {sh} carries {other[:12]}...: run --print-pins and paste", False)
        else:
            check(f"A1 pin {mine} == {sh} {theirs}", pin(mine) == other)
    for var in ("MODEL_ORIG_P032", "MODEL_P025_P032", "MODEL_P025"):
        other = pin_of("032-gb10-repl-embed.sh", {"MODEL_ORIG_P032": "MODEL_ORIG_P032", "MODEL_P025_P032": "MODEL_P025_P032",
                                                  "MODEL_P025": "MODEL_P025"}[var])
        if other is None:
            skip(f"A1 base pin {var} == 032's own pin", "032's pin absent or pending")
        elif pending(var):
            check(f"A1 base pin {var} is still pending although 032 carries {other[:12]}...: paste --print-pins", False)
        else:
            check(f"A1 base pin {var} == 032's own pin", pin(var) == other)
    # A2: the bases
    texts = read_bases(bases_dir) if bases_dir else derive_bases(vroot, work)
    for var in BASES:
        text = texts.get(var, "")
        if not text:
            why = ("patch 025 is not ported yet (no files/p025_apply.py)" if "P025" in var and not APPLIER_OF["p025"].exists()
                   else "the earlier patch's applier is absent or refuses the base")
            skip(f"A2 base {var} == pin", why)
            continue
        if pending(var):
            check(f"A2 base {var} == pin", False, f"the pin is {pin_raw(var)} but the base is derivable ({sha_text(text)[:16]}): paste --print-pins")
        else:
            check(f"A2 base {var} == pin", sha_text(text) == pin(var), sha_text(text)[:16])
    avail = [v for v in BASES if texts.get(v)]
    check("A2 the derivable bases are distinct", len({sha_text(texts[v]) for v in avail}) == len(avail) and "MODEL_ORIG" in avail)
    texts = {v: texts.get(v, "") for v in BASES}
    # A3: static enumeration of the chain (every earlier patch and vendored python file)
    shs = sorted(p_ for p_ in PDIR.glob("[0-9][0-9][0-9]-*.sh") if int(p_.name[:3]) < 33)
    pys = sorted(p_ for p_ in HERE.glob("*.py") if not re.search(r"(cpu_check|testlib|gpu_selftest|e2e_check|_probe|_test|"
                                                                 r"verify|bench|rankcheck|^p033_|psmall_lib)", p_.name))

    def code_of(p_):
        """a .sh without its comment lines (headers name files they only talk about); python files are used whole"""
        txt = p_.read_text(errors="replace")
        if p_.suffix == ".sh":
            txt = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
        return txt

    def mentions(paths, needle):
        return sorted(p_.name for p_ in paths if needle in code_of(p_))
    m_sh = mentions(shs, "glm5next/common/model.py")
    allowed = {"025-gb10-mhc-prefill-sp.sh", "032-gb10-repl-embed.sh"}
    check("A3 .sh files 001-032 naming glm5next/common/model.py are within {025, 032} and include 032", set(m_sh) <= allowed
          and "032-gb10-repl-embed.sh" in m_sh, str(m_sh))
    if "025-gb10-mhc-prefill-sp.sh" not in m_sh:
        skip("A3 025 names common/model.py", "patch 025 is not ported yet")
    m_py = mentions(pys, "common/model.py")
    check("A3 vendored python naming common/model.py is within {p025_apply.py, gb10_prefill_sp.py, p032_apply.py, "
          "p032_gb10_repl_embed.py}", set(m_py) <= {"gb10_prefill_sp.py", "p025_apply.py", "p032_apply.py",
                                                     "p032_gb10_repl_embed.py"}, str(m_py))
    check("A3 mtp.py: only 029 writes it (032 reads it)", mentions(shs, "common/mtp.py") == [
        "029-gb10-draft-compress.sh", "032-gb10-repl-embed.sh"] or set(mentions(shs, "common/mtp.py")) <= {
        "029-gb10-draft-compress.sh", "032-gb10-repl-embed.sh"}, str(mentions(shs, "common/mtp.py")))
    check("A3 linear.py: at most 025 names it (a read-only pin)", set(mentions(shs, "model_executor/layers/linear.py")) <= {
        "025-gb10-mhc-prefill-sp.sh"}, str(mentions(shs, "model_executor/layers/linear.py")))
    check("A3 marlin_utils*: at most 029 names them (read-only pins)", set(mentions(shs, "marlin_utils")) <= {
        "029-gb10-draft-compress.sh"}, str(mentions(shs, "marlin_utils")))
    check("A3 model_loader/utils.py + base_loader.py: at most 032 names them (read-only pins)",
          set(mentions(shs, "model_loader/utils.py")) <= {"032-gb10-repl-embed.sh"}
          and set(mentions(shs, "base_loader.py")) <= {"032-gb10-repl-embed.sh"})
    check("A3 no patch 001-032 names _try_load_fp8_attn_proj, _gb10_mla_w8a16 or the 033 gate",
          not mentions(shs + pys, "_try_load_fp8_attn_proj") and not mentions(shs + pys, "_gb10_mla_w8a16")
          and not mentions(shs + pys, "VLLM_GB10_MLA_W8A16"), str(mentions(shs + pys, "_try_load_fp8_attn_proj")))
    check("A3 _custom_ops.py: no patch 001-032 edits it", not [n for n in mentions(shs, "_custom_ops")
                                                                  if n not in ("029-gb10-draft-compress.sh",)],
          str(mentions(shs, "_custom_ops")))
    check("A3 attention.py: only 014 (and its module) names it", set(mentions(shs, "attention.py")) <= {"014-gb10-skinny-indexer.sh"},
          str(mentions(shs, "attention.py")))
    # A4: the applier on every available base
    for b in BASES:
        t_ = texts[b]
        if not t_:
            skip(f"A4 apply({b}) == pinned {b}_P033", "base not derivable (025 / 032 port pending)")
            continue
        out = ap.apply(t_)
        res = {"MODEL_ORIG": "MODEL_ORIG_P033", "MODEL_P025": "MODEL_P025_P033", "MODEL_ORIG_P032": "MODEL_ORIG_P032_P033",
               "MODEL_P025_P032": "MODEL_P025_P032_P033"}[b]
        if pending(res):
            check(f"A4 apply({b}) == pinned {res}", False, f"pin is {pin_raw(res)} but the result is {sha_text(out)[:16]}: paste --print-pins")
        else:
            check(f"A4 apply({b}) == pinned {res}", sha_text(out) == pin(res), sha_text(out)[:16])
        check(f"A4 revert({b} + 033) == {b} byte for byte", ap.revert(out) == t_)
        check(f"A4 {b} + 033: {ap.N_MARKERS} markers", out.count(ap.MARKER) == ap.N_MARKERS)
        try:
            compile(out, "model.py", "exec")
            ast.parse(out, feature_version=(3, 12))
            pc = work / f"pc_{b}.py"
            pc.write_text(out)
            py_compile.compile(str(pc), cfile=str(work / f"pc_{b}.pyc"), doraise=True)
            check(f"A4 py_compile({b} + 033), python 3.12 grammar", True)
        except Exception as e:  # noqa: BLE001
            check(f"A4 py_compile({b} + 033)", False, str(e)[:100])
        check(f"A4 apply on {b} + 033 refused (marker present: a second apply never double-edits)",
              raises(lambda: ap.apply(out), ap.Refused, "already carries"))
    t0 = texts["MODEL_ORIG"]
    p = Path(tempfile.mkdtemp(prefix="a4cli.", dir=work)) / "model.py"
    p.write_text(t0)
    rc = _run([sys.executable, str(APPLIER_PATH), "check", str(p)])
    check("A4 CLI check prints the pinned result", rc.returncode == 0 and rc.stdout.strip() == pin("MODEL_ORIG_P033"))
    check("A4 CLI check leaves the file untouched", sha_file(p) == pin("MODEL_ORIG"))
    rc = _run([sys.executable, str(APPLIER_PATH), "apply", str(p)])
    check("A4 CLI apply writes the pinned result", rc.returncode == 0 and sha_file(p) == pin("MODEL_ORIG_P033"))
    rc2 = _run([sys.executable, str(APPLIER_PATH), "apply", str(p)])
    check("A4 CLI apply on the result: refused, file byte-identical (idempotent no-op)", rc2.returncode == 1
          and sha_file(p) == pin("MODEL_ORIG_P033"))
    rc = _run([sys.executable, str(APPLIER_PATH), "revert", str(p)])
    check("A4 CLI revert restores the v0.32.0 original", rc.returncode == 0 and sha_file(p) == pin("MODEL_ORIG"))
    check("A4 no temp file left behind by the CLI", not list(p.parent.glob("*.p033.tmp")))
    # A7: the REAL v0.32.0 loader contract the stand-in loop of the threaded boots (section C) stands for
    lt = ast.parse((vroot / REL["lutil"]).read_text())
    pw = [n for n in lt.body if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading"]
    loops = [n for n in ast.walk(pw[0]) if isinstance(n, ast.For) and ast.unparse(n.iter) == "model.named_modules()"] if pw else []
    first = loops[0] if loops else None
    qcalls = [c for c in ast.walk(first) if isinstance(c, ast.Call) and ast.unparse(c.func) == "quant_method.process_weights_after_loading"
              ] if first is not None else []
    check("A7 model_loader/utils.py: process_weights_after_loading(model, model_config, target_device) exists once",
          len(pw) == 1 and [a.arg for a in pw[0].args.args] == ["model", "model_config", "target_device"])
    check("A7 ... its first `for ... in model.named_modules()` loop calls quant_method.process_weights_after_loading(module) exactly "
          "once per module with a QuantizeMethodBase quant_method, in named_modules() order, without break/continue",
          first is not None and len(qcalls) == 1 and [ast.unparse(a) for a in qcalls[0].args] == ["module"]
          and "isinstance(quant_method, QuantizeMethodBase)" in ast.unparse(first)
          and not [n for n in ast.walk(first) if isinstance(n, (ast.Break, ast.Continue))])
    bl = ast.parse((vroot / REL["bload"]).read_text())
    lm = [n for c in bl.body if isinstance(c, ast.ClassDef) and c.name == "BaseModelLoader" for n in c.body
          if isinstance(n, ast.FunctionDef) and n.name == "load_model"]
    calls = [ast.unparse(c) for c in ast.walk(lm[0]) if isinstance(c, ast.Call)] if lm else []
    order = [i for i, ln in enumerate(c for c in ast.walk(lm[0]) if isinstance(c, ast.Call)) if False]
    srcl = ast.unparse(lm[0]) if lm else ""
    check("A7 base_loader.py: BaseModelLoader.load_model builds the model, calls self.load_weights(model, model_config) and THEN "
          "process_weights_after_loading(model, model_config, target_device), once",
          len(lm) == 1 and srcl.count("process_weights_after_loading(model, model_config, target_device)") == 1
          and srcl.count("self.load_weights(model, model_config)") == 1
          and srcl.index("self.load_weights(model, model_config)") < srcl.index("process_weights_after_loading(model, model_config, target_device)"))
    el = ast.parse((vroot / REL["eagle"]).read_text())
    lfn = [n for n in el.body if isinstance(n, ast.FunctionDef) and n.name == "load_eagle_model"]
    check("A7 eagle/utils.py: load_eagle_model loads the draft through get_model(...) (the normal loader, so the MTP layer's "
          "process_weights_after_loading runs 033's boot hook too)", len(lfn) == 1 and "get_model(" in ast.unparse(lfn[0]))
    # refusals: every file untouched
    def refused(text, what):
        q = Path(tempfile.mkdtemp(prefix="a4ref.", dir=work)) / "model.py"
        q.write_text(text)
        before = sha_file(q)
        rc = _run([sys.executable, str(APPLIER_PATH), "apply", str(q)])
        return check(f"A4 refused (exit 1, file untouched, no temp): {what}",
                     rc.returncode == 1 and sha_file(q) == before and not list(q.parent.glob("*.p033.tmp")),
                     rc.stderr.strip()[-110:])
    refused(t0.replace("logger = init_logger(__name__)\n", "logger = init_logger(__name__)  # x\n", 1), "gate anchor missing")
    refused(t0 + "\nlogger = init_logger(__name__)\n", "gate anchor duplicated")
    refused(t0.replace(ap.INST_OLD, ap.INST_OLD.replace("MTP layers", "MTP  layers"), 1), "install anchor missing")
    refused(t0 + "\n\nclass _X:\n    def __init__(self):\n" + ap.INST_OLD, "install anchor duplicated")
    refused(t0.replace(ap.CAP_OLD, ap.CAP_OLD.replace("//", "/ /"), 1), "capture anchor missing")
    refused(t0 + "\n\ndef _y():\n" + ap.CAP_OLD, "capture anchor duplicated / outside _try_load_fp8_attn_proj")
    moved = t0.replace(ap.INST_OLD, "        pass\n", 1) + "\n\nclass _X:\n    def __init__(self):\n" + ap.INST_OLD + "        pass\n"
    refused(moved, "install anchor outside Glm5NextDecoderLayer.__init__")
    moved = (t0.replace(ap.CAP_OLD, "    pass\n", 1) + "\n\ndef _z(buf, layer_prefix, key, weight_fp8, scale_inv):\n" + ap.CAP_OLD)
    refused(moved, "capture anchor outside _try_load_fp8_attn_proj")
    refused(t0.replace("class Glm5NextDecoderLayer(nn.Module):", "class Glm5NextDecoderLayer(nn.Module):\n    _p033 = 1\n", 1),
            "base already mentions _p033")
    refused(t0.replace("\nlogger = init_logger(__name__)\n", "\nlogger = init_logger(__name__)\nlogger = 1\n", 1), "two loggers")
    out = ap.apply(t0)
    for bad, what in ((out.replace('.strip() == "1":\n    from vllm import _gb10_mla_w8a16', '== "0":\n    from vllm import '
                                   '_gb10_mla_w8a16', 1), "gate expression changed"),
                      (out.replace('self.layer_kind == "mla"', 'self.layer_kind != "kda"', 1), "install guard changed"),
                      (out.replace("_p033.install(self.self_attn,", "_p033.install(self,", 1), "install call changed"),
                      (out.replace("    if _p033 is not None:\n        _p033.capture(", "    if True:\n        _p033.capture(", 1),
                       "capture guard changed"),
                      (out.replace("        _p033.capture(params_dict[target_w], key,", "        _p033.capture(params_dict[target_w], 'x',", 1),
                       "capture call changed"),
                      (out + "\n_p033.install(1, 2, 3, 4)\n", "a third _p033 use")):
        check(f"A4 check_structure refuses: {what}", raises(lambda: ap.check_structure(bad, patched=True), ap.Refused))
    # capture hook moved after the dequant
    mv = out.replace("    if _p033 is not None:\n        _p033.capture(params_dict[target_w], key, weight_fp8, scale_inv, kv_a_pad_size)\n"
                     "    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)\n",
                     "    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)\n"
                     "    if _p033 is not None:\n        _p033.capture(params_dict[target_w], key, weight_fp8, scale_inv, kv_a_pad_size)\n", 1)
    check("A4 check_structure refuses: capture hook after the BF16 dequant", raises(lambda: ap.check_structure(mv, True), ap.Refused))
    # A5: the gate block alone
    tree = ast.parse(out)
    k = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
         and n.targets[0].id == "logger"][0]
    gate = ast.unparse(ast.Module(body=tree.body[k + 1:k + 3], type_ignores=[]))
    for val, want_mod in ((None, False), ("0", False), ("", False), ("1", True), (" 1 ", True), ("true", False), ("2", False)):
        g = {}
        saved = {kk: os.environ.get(kk) for kk in ("VLLM_GB10_MLA_W8A16",)}
        try:
            os.environ.pop("VLLM_GB10_MLA_W8A16", None)
            if val is not None:
                os.environ["VLLM_GB10_MLA_W8A16"] = val
            sys.modules["vllm"] = NS(_gb10_mla_w8a16="MODULE")
            sys.modules["vllm._gb10_mla_w8a16"] = "MODULE"
            exec(compile(gate, "gate", "exec"), g)
            ok = (g["_p033"] == "MODULE") if want_mod else (g["_p033"] is None)
        except Exception as e:  # noqa: BLE001
            ok = False
        finally:
            for kk, vv in saved.items():
                if vv is None:
                    os.environ.pop(kk, None)
                else:
                    os.environ[kk] = vv
            sys.modules.pop("vllm._gb10_mla_w8a16", None)
            sys.modules.pop("vllm", None)
        check(f"A5 gate VLLM_GB10_MLA_W8A16={val!r}: {'imports the module' if want_mod else '_p033 is None'}", ok)
    os.environ["VLLM_GB10_MLA_W8A16"] = "1"
    try:
        sys.modules["vllm"] = type(sys)("vllm")
        sys.modules["vllm"].__path__ = []
        exec(compile(gate, "gate", "exec"), {})
        check("A5 gate 1 + module missing -> ImportError (fail closed)", False)
    except ImportError:
        check("A5 gate 1 + module missing -> ImportError (fail closed)", True)
    finally:
        os.environ.pop("VLLM_GB10_MLA_W8A16", None)
        sys.modules.pop("vllm", None)
    return ap, texts




def section_a_gate_off(vroot, texts, ap, dims):
    """The gate-off path of the REAL patched loader/construct code == the unpatched one, bitwise, on every base."""
    print("== A6. gate off == unpatched (the REAL extracted loader + install statement, every derivable base)")
    ck = make_ckpt(dims, (3,), seed=11)
    results = {}
    for b in BASES:
        if not texts.get(b):
            skip(f"A6 {b}: gate unset == unpatched", "base not derivable (025 / 032 port pending)")
            continue
        for tag, text in (("unpatched", texts[b]), ("patched", ap.apply(texts[b]))):
            ns = build_ns(vroot, model_text=text, module_path=None, env={})
            h = Hooks(layers=(3,), mtp=False)
            T.set_current_world(T.ThreadWorld(T.Rendezvous(1), 0, 8, oneshot_max=None))
            rb = build_phase(ns, dims, 0, (3,), False, h)
            h2 = ns.fns.g.get("_p033")
            load_phase(ns, rb, ck, 0, h, "target")
            a = rb.layers[3].self_attn
            results[(b, tag)] = tuple(w.weight._a.tobytes() for w in (a.fused_qkv_a_proj, a.q_b_proj, a.o_proj)) + (
                type(a.fused_qkv_a_proj.quant_method).__name__, h2 is None, ns.install_stmt is not None)
        r_un, r_pa = results[(b, "unpatched")], results[(b, "patched")]
        check(f"A6 {b}: gate unset: loaded BF16 weights identical, quant methods untouched, _p033 None",
              r_un[:3] == r_pa[:3] and r_pa[3] == "UnquantizedLinearMethod" and r_pa[4] is True)


# =====================================================================================================================
# B: module units (single thread, fake torch)
# =====================================================================================================================
_READINESS = []
READINESS_SRC = None
for _cand in (os.environ.get("PATCH_READINESS_TXT"), PDIR.parent / "launch" / "readiness.txt", HERE / "testdata" / "readiness.txt"):
    if _cand and Path(_cand).is_file():
        READINESS_SRC = Path(_cand)
        break
if READINESS_SRC is not None:
    for _ln in READINESS_SRC.read_text().splitlines():
        _parts = _ln.strip().split(None, 1)
        if len(_parts) == 2 and _parts[0] in ("fail", "require"):
            _READINESS.append((_parts[0], re.compile(_parts[1])))
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


def unit(vroot, text, dims=SMALL, rank=2, layers=(3,), env=None, seed=5, zero_blocks=None, **kw):
    """One rank's installed + loaded modules on the main thread (no process_weights_after_loading, no collective)."""
    ns = new_ns(vroot, text, env=env, **kw)
    w = T.ThreadWorld(T.Rendezvous(8, timeout=2.0), rank, 8, oneshot_max=None)
    T.set_current_world(w)
    h = Hooks(layers=layers, mtp=False)
    ck = make_ckpt(dims, layers, seed=seed, zero_blocks=zero_blocks)
    rb = build_phase(ns, dims, rank, layers, False, h)
    load_phase(ns, rb, ck, rank, h, "target")
    st = ns.mod._state()
    return NS(ns=ns, w=w, rb=rb, st=st, g=st.groups["target"], ck=ck, dims=dims, rank=rank, h=h)


def proj_of(u, typ, layer=None):
    return next(p for p in u.g.projs if p.typ == typ and (layer is None or p.layer_idx == layer))


def build_all(u):
    for p in u.g.projs:
        u.ns.mod._build(p, p.module)


def np_ref(p, x_np, bf16_scale=False):
    """numpy fp32 reference x @ (fp8 * scale)^T from the checkpoint-side bytes captured for this projection."""
    w = T.fp8_bits_to_f32(p.entry.w8._a)
    s = p.entry.sc._a
    if bf16_scale:
        s = T.bf16_bits_to_f32(T.f32_to_bf16_bits(s))
    s_full = np.repeat(np.repeat(s, 128, 0), 128, 1)
    return x_np.astype(np.float32) @ (w * s_full).T.astype(np.float32)


def relrmse(a, ref):
    d = a.astype(np.float64) - ref.astype(np.float64)
    return float(np.sqrt((d * d).sum() / max((ref.astype(np.float64) ** 2).sum(), 1e-30)))


class Res(list):
    def __call__(self, name, ok, info=""):
        self.append((name, bool(ok), info))
        return bool(ok)


def unit_selftest(cx):
    """B6: the boot self-test FAILS on wrong layouts / scales and PASSES on the unmodified build."""
    R = Res()
    vroot, text = cx.vroot, cx.text
    kw = dict(module_path=cx.module_path, mu8_text=cx.mu8_text)
    def selftest_fails(mutate, want):
        uu = unit(vroot, text, **kw,  rank=2, layers=(3,))
        build_all(uu)
        p = proj_of(uu, "qkv_a")                       # 256 x 256: square, so a transposed capture passes every shape assert
        mutate(uu, p)
        rows = uu.ns.mod._Rows()
        with uu.ns.mod._probing(uu.g.projs):
            uu.ns.mod._check_proj(p, rows, {}, {}, {})
        names = " ".join(rows.fail)
        return any(w in names for w in want), rows.fail[:1]
    def rebuild(uu, p, w8=None, sc=None, mutate_marlin=None):
        MU8 = uu.ns.mu8
        H = uu.ns.torch.nn.Module()
        H.weight = uu.ns.torch.nn.Parameter(w8 if w8 is not None else p.entry.w8, requires_grad=False)
        H.weight_scale_inv = uu.ns.torch.nn.Parameter(sc if sc is not None else p.entry.sc, requires_grad=False)
        H.orig_dtype = uu.ns.torch.bfloat16
        H.output_size_per_partition, H.input_size_per_partition = p.n, p.k
        H.weight_block_size = [128, 128]
        MU8.prepare_fp8_layer_for_marlin(H, size_k_first=False)
        p.w, p.s = H.weight.data, H.weight_scale_inv.data
        p.ws = uu.ns.mod._make_workspace(p.w.device)          # v0.32.0: prepare no longer attaches a workspace
    def transposed(uu, p):
        w8 = T.Tensor(np.ascontiguousarray(p.entry.w8._a.T), T.float8_e4m3fn, "cuda:0")
        sc = T.Tensor(np.ascontiguousarray(p.entry.sc._a.T), T.float32, "cuda:0")
        rebuild(uu, p, w8, sc)
    def scales_x2(uu, p):
        p.s = T.Tensor(p.s._a.copy(), T.bfloat16, p.s.device)
        p.s._a[:] = T.f32_to_bf16_bits(T.bf16_bits_to_f32(p.s._a) * 2.0)
    def scales_permuted(uu, p):
        s = p.s._a.copy()
        s[:, :] = s[:, ::-1]
        p.s = T.Tensor(np.ascontiguousarray(s), T.bfloat16, p.s.device)
    def scale_rows_reversed(uu, p):
        sc = T.Tensor(np.ascontiguousarray(p.entry.sc._a[::-1]), T.float32, "cuda:0")
        rebuild(uu, p, None, sc)
    def unfolded(uu, p):
        p.s = T.Tensor(T.f32_to_bf16_bits(T.bf16_bits_to_f32(p.s._a) / (2.0 ** 120)), T.bfloat16, p.s.device)
    def other_layer(uu, p):
        q = proj_of(uu, "q_b")
        p.w = q.w
    for name, fn, want in (("transposed weight + scales", transposed, ("C1", "C3")), ("scales x 2", scales_x2, ("C1", "C3")),
                           ("scale columns permuted", scales_permuted, ("C1", "C3")),
                           ("scale block rows reversed", scale_rows_reversed, ("C1", "C3")),
                           ("2^120 exponent bias not folded", unfolded, ("C1", "C3"))):
        ok, first = selftest_fails(fn, want)
        R(f"B6 the self-test FAILS on: {name}", ok, str(first)[:80])
    uu = unit(vroot, text, **kw,  rank=2, layers=(3,))
    build_all(uu)
    p = proj_of(uu, "qkv_a")
    rows = uu.ns.mod._Rows()
    with uu.ns.mod._probing(uu.g.projs):
        uu.ns.mod._check_proj(p, rows, {}, {}, {})
    R("B6 the self-test PASSES on the unmodified build (all of C1-C8 for qkv_a)", rows.ok, str(rows.fail[:2]))
    # R1: the fp32 references are float64 GEMMs rounded once and never read or write a TF32 flag (the trap raises on any access)
    mm = uu.ns.torch.backends.cuda.matmul
    R("R1 the test namespace models a process where reading torch.backends.cuda.matmul.allow_tf32 raises RuntimeError",
      raises(lambda: getattr(mm, "allow_tf32", None), RuntimeError) and raises(lambda: setattr(mm, "allow_tf32", False), RuntimeError))
    X = uu.ns.mod._synth("unit", 5, p.k, 77, p.module.weight.device)
    try:
        r32, rdq = uu.ns.mod._refs(p, X)
        e1 = relrmse(r32._f(), np_ref(p, X._f()))
        e2 = relrmse(rdq._f(), np_ref(p, X._f(), True))
        R("R1 _refs under the TF32 trap == the numpy references (x @ (fp8 x fp32 scale)^T and x @ (fp8 x bf16(scale))^T) to "
          "<= 1e-6, fp32 outputs", e1 <= 1e-6 and e2 <= 1e-6 and r32.dtype == T.float32 and rdq.dtype == T.float32,
          f"{e1:.2e} {e2:.2e}")
    except Exception as e:  # noqa: BLE001
        R("R1 _refs does not touch the TF32 flag", False, f"{type(e).__name__}: {str(e)[:80]}")
    return R


def unit_dispatch(cx):
    """B7: the dispatch rule in every state."""
    R = Res()
    vroot, text = cx.vroot, cx.text
    kw = dict(module_path=cx.module_path, mu8_text=cx.mu8_text)
    # ---- B7 the dispatch rule in every state
    u = unit(vroot, text, **kw,  rank=2, layers=(3,))
    build_all(u)
    mod = u.ns.mod
    for typ in ("qkv_a", "q_b", "o_proj"):
        p = proj_of(u, typ)
        K = p.k
        rules = []
        xs = {M: mod._synth("unit", M, K, 90 + M, p.module.weight.device) for M in (1, 3, 16, 17, 40)}
        for state in (mod.PENDING, mod.PROBE, mod.ACTIVE):
            p.mode, p.hits = state, 0
            for M, x in xs.items():
                y = p.method.apply(p.module, x)
                want_marlin = state != mod.PENDING and M <= 16
                ref = mod._marlin(p, x) if want_marlin else mod._bf16(p, x)
                rules.append(((state, M), mod._mismatches(y, ref) == 0))
            exp_hits = 3 if state == mod.PROBE else 0
            rules.append(((state, "hits"), p.hits == exp_hits))
            p.hits = 0
            bias = T.zeros(p.n, dtype=T.bfloat16, device=p.module.weight.device)
            x3 = xs[3]
            for nm, f in (("bias", lambda: p.method.apply(p.module, x3, bias)), ("3-D", lambda: p.method.apply(p.module, x3.reshape(1, 3, K))),
                          ("fp32", lambda: p.method.apply(p.module, x3.to(T.float32)))):
                ref = {"bias": lambda: mod._bf16(p, x3, bias), "3-D": lambda: mod._bf16(p, x3.reshape(1, 3, K)),
                       "fp32": lambda: mod._bf16(p, x3.to(T.float32))}[nm]
                ok, why = mod._same_outcome(f, ref)
                rules.append(((state, nm), ok and p.hits == 0))
            big = mod._synth("unit", 3, K + 64, 91, p.module.weight.device)[:, :K]
            ys = p.method.apply(p.module, big)
            yc = p.method.apply(p.module, big.contiguous())
            want_m = state != mod.PENDING
            rules.append(((state, "strided"), mod._mismatches(ys, yc) == 0 and p.hits == (2 if state == mod.PROBE else 0)
                          and (not want_m or mod._mismatches(ys, mod._marlin(p, big.contiguous())) == 0)))
        p.mode, p.hits = mod.PENDING, 0
        badr = [k for k, ok in rules if not ok]
        R(f"B7 {typ}: dispatch rule in pending / probe / active (M 1,3,16 Marlin; 17,40 / bias / 3-D / fp32 = BF16 "
              f"bitwise; strided == contiguous; hits only in probe)", not badr, str(badr[:3]))
    return R


def section_b(vroot, texts, ap):
    print("== B. module units on the fake torch")
    text = ap.apply(texts[SERVED])
    # ---- B1 env parsing
    ns = new_ns(vroot, text)
    m = ns.mod
    saved_gate = os.environ.pop("VLLM_GB10_MLA_W8A16", None)
    check("B1 gate_on() with the variable unset is False", m.gate_on() is False)
    for raw, want in (("0", False), ("", False), ("1", True), (" 1 ", True), ("true", False), ("2", False)):
        check(f"B1 gate_on({raw!r}) == {want}", m.gate_on(raw) == want)
    if saved_gate is not None:
        os.environ["VLLM_GB10_MLA_W8A16"] = saved_gate
    for raw, want in ((None, 16), ("", 16), ("8", 8), (" 8 ", 8), ("64", 64), ("1", 1)):
        check(f"B1 parse_max_m({raw!r}) == {want}", m.parse_max_m(raw) == want)
    check("B1 parse_max_m('1 6') == 16 (all whitespace removed, as the .sh's tr -d does)", m.parse_max_m("1 6") == 16)
    for raw in ("0", "65", "08", "x", "1.5", "-3"):
        check(f"B1 parse_max_m({raw!r}) refused", raises(lambda: m.parse_max_m(raw), ValueError))
    for raw, want in ((None, "graph"), ("", "graph"), (" Info ", "info"), ("GRAPH", "graph")):
        check(f"B1 parse_speed_gate({raw!r}) == {want}", m.parse_speed_gate(raw) == want)
    check("B1 parse_speed_gate('numerics') refused", raises(lambda: m.parse_speed_gate("numerics"), ValueError))
    # ---- B2 defanging against every readiness fail / require rule
    check("B2 launch/readiness.txt was read", len(_READINESS) > 20, f"{len(_READINESS)} rules from {READINESS_SRC}")
    for kind, rx in _READINESS:
        check(f"B2 {kind} rule '{rx.pattern[:46]}' is covered by a GATE_TEXTS entry",
              any(rx.search(g) for g in GATE_TEXTS))
    leaks = [(rx.pattern, g) for g in GATE_TEXTS for _, rx in _READINESS if rx.search(m._safe(g))]
    check("B2 _safe() defangs every gate text (none matches any fail / require rule afterwards)", not leaks, str(leaks[:2]))
    check("B2 _safe keeps the information readable", "inactive" in m._safe("INACTIVE x") and "(ok)" in m._safe("[OK] y")
          and "(patch-033)" in m._safe("[patch-033] z"))
    for meth, n, want in (("mtp", 2, True), ("deepseek_mtp", 1, True), ("glm4_moe_mtp", 3, True), ("MTP", 2, True),
                          ("eagle", 2, False), ("mtp", 0, False), (None, 2, False)):
        vc = NS(speculative_config=None if meth is None else NS(method=meth, num_speculative_tokens=n))
        check(f"B1 _expect_mtp(method={meth!r}, tokens={n}) == {want}", m._expect_mtp(vc) is want)
    check("B1 _expect_mtp(no speculative_config attribute) is False", m._expect_mtp(NS()) is False)
    # ---- B3 capture sharding: every rank, every type, vs the independent loader oracle
    bad = []
    for r in range(8):
        u = unit(vroot, text, rank=r, layers=(3, 7))
        for L in (3, 7):
            for typ in ("qkv_a", "q_b", "o_proj"):
                p = proj_of(u, typ, L)
                w_or = oracle_weight(u.ck[L], typ, r, u.dims)
                if not np.array_equal(p.module.weight._a, w_or):
                    bad.append((r, L, typ, "loader oracle"))
                if typ == "qkv_a":
                    e_w = np.concatenate([u.ck[L]["q_a"][0], u.ck[L]["kv_a"][0]], 0)
                    e_s = np.concatenate([u.ck[L]["q_a"][1], u.ck[L]["kv_a"][1]], 0)
                elif typ == "q_b":
                    P = u.dims.qb_rows
                    e_w, e_s = u.ck[L]["q_b"][0][r * P:(r + 1) * P], u.ck[L]["q_b"][1][r * P // 128:(r + 1) * P // 128]
                else:
                    P = u.dims.o_cols
                    e_w, e_s = u.ck[L]["o_proj"][0][:, r * P:(r + 1) * P], u.ck[L]["o_proj"][1][:, r * P // 128:(r + 1) * P // 128]
                if not (np.array_equal(p.entry.w8._a, e_w) and np.array_equal(p.entry.sc._a, e_s)):
                    bad.append((r, L, typ, "captured shard"))
                if p.entry.w8.untyped_storage().nbytes() != e_w.size or p.entry.sc.untyped_storage().nbytes() != 4 * e_s.size:
                    bad.append((r, L, typ, "storage"))
    check("B3 capture == the independent shard rules for 8 ranks x 2 layers x 3 types (and == what the BF16 loader "
          "leaves in the module; storage == the shard, never a view of the checkpoint)", not bad, str(bad[:3]))
    u = unit(vroot, text, rank=3)
    q = proj_of(u, "q_b")
    # SF5: a capture from VIEWS of larger tensors owns its storage; later changes of the source do not leak in
    big_w = T.Tensor(np.zeros((1024 * 2, 128), np.uint8), T.float8_e4m3fn, "cuda:0")
    big_s = T.Tensor(np.ones((8 * 2, 1), np.float32), T.float32, "cuda:0")
    big_w._a[:] = 0x38
    u2 = unit(vroot, text, rank=3)
    q2 = proj_of(u2, "q_b")
    q2.entry.w8 = q2.entry.sc = None
    q2.entry.have.clear()
    view_w, view_s = big_w[0:1024], big_s[0:8]
    u2.ns.mod.capture(q2.module.weight, "q_b", view_w, view_s, 0)
    cap = q2.entry.bad is None and q2.entry.w8 is not None
    before = q2.entry.w8._a.copy()
    big_w._a[:] = 0x40
    check("B3 SF5: capture from a view copies into its own storage (source change does not leak; nbytes == numel)",
          cap and np.array_equal(q2.entry.w8._a, before) and q2.entry.w8.untyped_storage().nbytes() == q2.entry.w8.numel())
    # a CPU-resident checkpoint tensor (the default safetensors iterator) is copied onto the parameter's device
    u3 = unit(vroot, text, rank=3)
    q3 = proj_of(u3, "q_b")
    q3.entry.w8 = q3.entry.sc = None
    q3.entry.have.clear()
    cw = T.Tensor(u3.ck[3]["q_b"][0].copy(), T.float8_e4m3fn, "cpu")
    cs = T.Tensor(u3.ck[3]["q_b"][1].copy(), T.float32, "cpu")
    u3.ns.mod.capture(q3.module.weight, "q_b", cw, cs, 0)
    check("B3 capture from CPU-resident checkpoint tensors lands on the parameter's CUDA device, bit-exact",
          q3.entry.bad is None and q3.entry.w8.device.type == "cuda" and q3.entry.sc.device.type == "cuda"
          and np.array_equal(q3.entry.w8._a, u3.ck[3]["q_b"][0][3 * 128:4 * 128]))
    # error paths (never raise; recorded on the entry)
    def cap_err(**mut):
        uu = unit(vroot, text, rank=3)
        p = proj_of(uu, "q_b")
        p.entry.w8 = p.entry.sc = None
        p.entry.have.clear()
        w8, sc = ck_tensors(uu.ck[3], "q_b")
        key, pad = "q_b", 0
        if "dtype" in mut:
            w8 = T.Tensor(w8._a, T.int8, "cuda:0")
        if "shape" in mut:
            w8 = T.Tensor(w8._a[:-128], T.float8_e4m3fn, "cuda:0")
            sc = T.Tensor(sc._a[:-1], T.float32, "cuda:0")
        if "pad" in mut:
            pad = 64
        if "key" in mut:
            key = "o_proj"
        if "partial" in mut:
            sc = T.Tensor(sc._a[:, :-1], T.float32, "cuda:0")
        uu.ns.mod.capture(p.module.weight, key, w8, sc, pad)
        return p.entry.bad
    for what in ("dtype", "shape", "pad", "key", "partial"):
        b = cap_err(**{what: 1})
        check(f"B3 capture error path '{what}' is recorded on the entry (no raise)", b is not None, str(b)[:70])
    uu = unit(vroot, text, rank=3)
    other = T.Tensor(np.zeros((4, 4), np.float32), T.float32, "cuda:0")
    w8, sc = ck_tensors(uu.ck[3], "q_b")
    check("B3 capture() of an unregistered parameter (a KDA o_proj) is ignored without raising",
          uu.ns.mod.capture(other, "o_proj", w8, sc, 0) is None and len(uu.st.by_param) == 3)
    # ---- B4 V1-V4 and the build
    u = unit(vroot, text, rank=2, zero_blocks={(3, "q_b"): [(0, 0)], (3, "o_proj"): [(1, 0)]})
    for p in u.g.projs:
        try:
            u.ns.mod._build(p, p.module)
            ok, msg = True, ""
        except Exception as e:  # noqa: BLE001
            ok, msg = False, str(e)
        check(f"B4 clean build {p.typ} (zero-scale blocks present in q_b / o_proj, SF4)", ok, msg[:90])
        check(f"B4 {p.typ}: Marlin weight [K/16, 4N] int32, scales [K/128, N] bf16, workspace int32",
              ok and tuple(p.w.shape) == (p.k // 16, 4 * p.n) and p.w.dtype is T.int32
              and tuple(p.s.shape) == (p.k // 128, p.n) and p.s.dtype is T.bfloat16 and p.ws.dtype is T.int32)
    def break_and_build(mutate, typ="q_b", what=""):
        uu = unit(vroot, text, rank=2)
        p = proj_of(uu, typ)
        mutate(p)
        try:
            uu.ns.mod._build(p, p.module)
            return None
        except Exception as e:  # noqa: BLE001
            return str(e)
    def flip_served(p):
        p.module.weight._a[0, 0] ^= 0x0100
    def flip_captured(p):
        p.entry.w8._a[1, 1] ^= 0x01
    def scale_big(p):
        p.entry.sc._a[0, 0] = 300.0
        p.module.weight._a[:128, :128] = dequant_bf16(p.entry.w8._a[:128, :128], p.entry.sc._a[:1, :1])
    def scale_nan(p):
        p.entry.sc._a[0, 0] = np.nan
    def scale_neg(p):
        p.entry.sc._a[0, 0] = -1e-3
    def nan_code(p):
        p.entry.w8._a[2, 2] = 0x7F
    def stale_scale(p):
        p.entry.sc._a *= np.float32(1.0 + 2.0 ** -7)
    def wrong_param(p):
        p.entry.param = T.Parameter(T.zeros(2, 2))
    def missing(p):
        p.entry.have.discard("q_b")
    def storage(p):
        p.entry.w8 = T.Tensor(np.zeros((p.n + 128, p.k), np.uint8), T.float8_e4m3fn, "cuda:0")[0:p.n]
    for name, fn, want in (("served weight differs by one bit", flip_served, "V2"), ("captured byte differs", flip_captured, "V2"),
                           ("scale >= 128", scale_big, "V4"), ("NaN scale", scale_nan, "V4"),
                           ("negative scale", scale_neg, "V4"), ("NaN FP8 code", nan_code, "V4"),
                           ("stale scale (bf16-ulp off)", stale_scale, "V2"), ("different Parameter object", wrong_param, "register"),
                           ("capture parts missing", missing, "captured"), ("view of a larger tensor kept", storage, "storage")):
        msg = break_and_build(fn)
        check(f"B4 build refused: {name}", msg is not None and want in msg, str(msg)[:90])
    # tiny (but non-zero) scale: V3 lemma excluded, served ~ 0 required
    def tiny(p):
        p.entry.sc._a[0, 0] = np.float32(1e-35)
        p.module.weight._a[:128, :128] = dequant_bf16(p.entry.w8._a[:128, :128], p.entry.sc._a[:1, :1])
    check("B4 a tiny non-zero scale block (1e-35) builds (V3 excluded, served ~ 0)", break_and_build(tiny) is None)
    # ---- B5 the Marlin build against an fp32 reference implementation (all types, several ranks)
    worst = 0.0
    bad = []
    for r in (0, 3, 7):
        u = unit(vroot, text, rank=r, layers=(7,))
        build_all(u)
        for p in u.g.projs:
            for M in (1, 3, 16):
                x = u.ns.mod._synth("unit", M, p.k, 77 + M, p.module.weight.device)
                y = u.ns.mod._marlin(p, x)
                ref = np_ref(p, x._f())
                refdq = np_ref(p, x._f(), bf16_scale=True)
                e1, e3 = relrmse(y._f(), ref), relrmse(y._f(), refdq)
                worst = max(worst, e1)
                if not (e1 <= 0.006 and e3 <= 0.0045 and tuple(y.shape) == (M, p.n) and y.dtype is T.bfloat16):
                    bad.append((r, p.typ, M, round(e1, 5), round(e3, 5)))
    check("B5 Marlin build (REAL prepare_fp8_layer_for_marlin over the group-128 fake kernel) == the fp32 reference "
          "x @ (fp8 x scale)^T: relRMSE <= 0.006 (vs bf16-scale dequant <= 0.0045) for 3 ranks x 3 types x M in {1,3,16}",
          not bad, f"worst vs fp32 {worst:.5f} {bad[:2]}")
    # ---- B6 / B7 (shared with the mutant runs)
    cx = NS(vroot=vroot, text=text, module_path=None, mu8_text=None)
    for name, ok, info in unit_selftest(cx) + unit_dispatch(cx):
        check(name, ok, info)
    # ---- B8 the verdict function (pure)
    mod = m
    rowsA = []
    for r in range(8):
        rowsA.append({"rank": r, "group": "target", "fp": "f1", "ok": True, "why": "", "type_ok": {"qkv_a": True, "q_b": True, "o_proj": True},
                      "type_why": {}, "qkv_digest": "d", "out_digest": "o", "worst": {"q_b": {"c1": 0.003, "c2": 0.004, "c3": 0.002,
                                                                                          "c4": 0.002, "c5": 0.99999}}, "bytes": 10, "speed": None})
    def vd(mut):
        rr = [dict(x, type_ok=dict(x["type_ok"]), type_why={}) for x in rowsA]
        mut(rr)
        return mod._group_verdict(rr, 8, "target")
    check("B8 verdict: clean rows -> ok, all three types", vd(lambda rr: None)["ok"] and vd(lambda rr: None)["types"] == list(mod.TYPES))
    check("B8 verdict: rows None -> refused", not mod._group_verdict(None, 8, "target")["ok"])
    check("B8 verdict: a missing rank -> refused", not mod._group_verdict(rowsA[:7], 8, "target")["ok"])
    check("B8 verdict: one rank not ok -> refused (message names it)", "rank(s) [4]" in " ".join(
        vd(lambda rr: rr[4].update(ok=False, why="C1 x"))["why"]))
    check("B8 verdict: fingerprint mismatch -> refused", not vd(lambda rr: rr[2].update(fp="f2"))["ok"])
    check("B8 verdict: replicated qkv_a FP8 digest differs -> refused", not vd(lambda rr: rr[6].update(qkv_digest="x"))["ok"])
    check("B8 verdict: replicated qkv_a output digest differs -> refused", not vd(lambda rr: rr[6].update(out_digest="x"))["ok"])
    v = vd(lambda rr: rr[1]["type_ok"].update(qkv_a=False))
    check("B8 verdict: one rank's qkv_a seam failure drops qkv_a alone (q_b, o_proj stay)", v["ok"] and v["types"] == ["q_b", "o_proj"]
          and any("qkv_a not active" in w for w in v["why"]))
    check("B8 verdict: every type's seam failing -> refused", not vd(lambda rr: [x["type_ok"].update(qkv_a=False, q_b=False, o_proj=False) for x in rr])["ok"])
    # ---- B9 the speed gate on the virtual clock
    def speed(cost=None, mode="graph", hook=None, scale=None):
        uu = unit(vroot, text, rank=2, layers=(3, 7), env={"VLLM_GB10_MLA_W8A16_SPEED_GATE": mode})
        build_all(uu)
        saved = dict(COST)
        COST.update(cost or {})
        uu.w.capture_hook = hook
        uu.w.cost_scale = scale
        rows = uu.ns.mod._Rows()
        try:
            info = uu.ns.mod._speed_gate(uu.st, uu.g, rows)
        finally:
            COST.clear()
            COST.update(saved)
        return info, rows, T.allocator_clean()
    info, rows, clean = speed()
    check("B9 speed gate: Marlin ~0.51 x BF16 on the virtual clock passes (E3: 0.53)", rows.ok and 0.45 < info["ratio"] < 0.6
          and info["saved_ms"] > 0, str(info))
    info, rows, clean = speed(cost={"marlin_bps": 100e9})
    check("B9 speed gate: Marlin at 1.02 x BF16 -> the S1 row FAILS (graph mode)", not rows.ok and "S1" in rows.fail[0], str(rows.fail[:1]))
    info, rows, clean = speed(cost={"marlin_bps": 100e9}, mode="info")
    check("B9 speed gate: the same slow Marlin in info mode is logged, not enforced", rows.ok and info["ratio"] > 0.85)
    def raiser(g):
        raise RuntimeError("injected cudaStreamBeginCapture failure")
    info, rows, clean = speed(hook=raiser)
    check("B9 speed gate: a failed capture -> S1 row fails (graph mode), the allocator is left clean",
          not rows.ok and "ran" in rows.fail[0] and clean and "error" in info, str(rows.fail[:1]))
    info, rows, clean = speed(hook=raiser, mode="info")
    check("B9 speed gate: a failed capture in info mode is only logged, allocator clean", rows.ok and clean and "error" in info)
    # ---- B10 atomic-add refusal, memory tally
    u = unit(vroot, text, rank=2, layers=(3,))
    build_all(u)
    p = proj_of(u, "q_b")
    check("B10 should_use_atomic_add_reduce is False for the test shapes -> no refusal", u.ns.mod._atomic_add_refusal(p) is None)
    u.ns.envs.VLLM_MARLIN_USE_ATOMIC_ADD = True
    fake = NS(n=1024, k=2048, max_m=16, module=p.module)
    check("B10 an atomic-add-eligible shape (N 1024, K 2048, env on) is refused", "atomic" in str(u.ns.mod._atomic_add_refusal(fake)))
    check("B10 the memory tally of a built module is its storage bytes", p.built_bytes == (p.w.untyped_storage().nbytes()
          + p.s.untyped_storage().nbytes() + p.ws.untyped_storage().nbytes()) and p.built_bytes > 0)


# =====================================================================================================================
# C: 8-rank threaded boots
# =====================================================================================================================
def boot(cx, hooks=None, env=None, ckpt=None, dims=SMALL, size=8, rv_timeout=6.0, inject=None, **ns_kw):
    ns = new_ns(cx.vroot, cx.text, env=env, module_path=cx.module_path, mu8_text=cx.mu8_text, inject=inject, **ns_kw)
    ck = ckpt if ckpt is not None else make_ckpt(dims)
    b = run_boot(ns, dims, ck, size=size, hooks=hooks, rv_timeout=rv_timeout)
    return b, ns, ck


def proj_in(rb, L, typ):
    a = rb.layers[L].self_attn
    return {"qkv_a": a.fused_qkv_a_proj, "q_b": a.q_b_proj, "o_proj": a.o_proj}[typ].quant_method._p


def shard_of(ck_layer, typ, rank, dims):
    """(fp8 bytes, block scales) of this rank's module from the checkpoint (independent of the patch)."""
    if typ == "qkv_a":
        return (np.concatenate([ck_layer["q_a"][0], ck_layer["kv_a"][0]], 0),
                np.concatenate([ck_layer["q_a"][1], ck_layer["kv_a"][1]], 0))
    if typ == "q_b":
        P = dims.qb_rows
        return ck_layer["q_b"][0][rank * P:(rank + 1) * P], ck_layer["q_b"][1][rank * P // 128:(rank + 1) * P // 128]
    P = dims.o_cols
    return ck_layer["o_proj"][0][:, rank * P:(rank + 1) * P], ck_layer["o_proj"][1][:, rank * P // 128:(rank + 1) * P // 128]


def restored(b, ns, ck, dims, check_weights=True, skip_restore=None):
    bad = []
    for r, L, typ, m in all_modules(b):
        if typ == skip_restore:                       # a module the patch never touched (its own foreign method stays)
            if type(m.quant_method).__name__ == "_Method":
                bad.append((r, L, typ, "touched"))
            continue
        if type(m.quant_method) is not ns.UQM:
            bad.append((r, L, typ, "quant_method " + type(m.quant_method).__name__))
        elif check_weights and not np.array_equal(m._parameters["weight"]._a, oracle_weight(ck[L], typ, r, dims)):
            bad.append((r, L, typ, "weight"))
    return not bad, str(bad[:2])


def expect_refused(R, b, ns, ck, dims, name, why, check_weights=True, allow_hang=False, skip_restore=None):
    ds = decisions(b)
    h, hi = healthy(b)
    R(f"C5 {name}: INACTIVE on all {b.size} ranks, exactly one decision line each",
      ds == ["INACTIVE"] * b.size and all(len(final_lines(b, r)) == 1 for r in range(b.size)), str(ds))
    R(f"C5 {name}: no exception on any rank" + ("" if allow_hang else ", no hang, no schedule mismatch, equal schedules"),
      (not [e for e in b.err if e is not None]) if allow_hang else h, hi)
    if why:
        R(f"C5 {name}: the reason names '{why}'",
          any(why in m for r in range(b.size) for m in final_lines(b, r)), str(final_lines(b, 0))[:200])
    ok, info = restored(b, ns, ck, dims, check_weights, skip_restore)
    R(f"C5 {name}: every module is the image's UnquantizedLinearMethod again" + (", BF16 weights intact" if check_weights else ""),
      ok, info)
    return b


def sc_clean(cx):
    R = Res()
    b, ns, ck = boot(cx)
    h, hi = healthy(b)
    R("C1 clean 8-rank boot: no exception, no hang, no schedule mismatch, equal collective schedules on all ranks", h, hi)
    ds = decisions(b)
    R("C1 every rank logs exactly one decision line and it is the ACTIVE one",
      ds == ["ACTIVE"] * 8 and all(len(final_lines(b, r)) == 1 for r in range(8)), str(ds))
    R("C1 one fingerprint on all ranks", len(fingerprints(b)) == 1, str(fingerprints(b)))
    R("C1 schedule = 3 exchanges (target group, MTP group, commit) x (bounded barrier + gather), nothing else",
      [s[0] for s in b.rv.schedules[0]] == ["barrier", "all_gather_object"] * 3 and len(b.rv.schedules[0]) == 6,
      str([s[0] for s in b.rv.schedules[0]]))
    line = final_lines(b, 5)[0] if final_lines(b, 5) else ""
    for frag in ("(rank 5/8)", "decode M<=16 on 2 target + 1 MTP MLA layers", "active types {qkv_a, q_b, o_proj}",
                 "(9 projections", "BF16 weights kept for M>16", "worst relRMSE vs fp32 [qkv_a ", "cos_min [", "deterministic",
                 "seam hit exactly once on every module", "qkv_a digests equal on 8 ranks", "speed gate (graph)",
                 " s target + ", " s MTP on this rank; module rev 2, fingerprint "):
        R(f"C1 the ACTIVE line carries '{frag}'", frag in line, line[:160])
    info = [m for r in range(8) for m in LOGS.lines(rank=r, contains="waiting for the MTP draft group")]
    R("C1 one INFO line per rank after the target group (waiting for the MTP group; deps recorded; no ACTIVE/INACTIVE words)",
      len(info) == 8 and all("deepseek_v2.py" in m and "ACTIVE" not in m and re.search(r"self-test \d+\.\d s\);", m)
                             for m in info), str(info[:1]))
    pl = [m for r in range(8) for m in LOGS.lines(rank=r, contains="[patch-033]")]
    R("C1 no WARNING / ERROR line from [patch-033] in a clean boot",
      not [1 for (t, lv, m) in LOGS.records if "[patch-033]" in m and lv in ("WARNING", "ERROR")])
    leak = [(rx.pattern, m[:60]) for m in pl for kind, rx in _READINESS if kind == "fail" and rx.search(m)]
    R("C1 no [patch-033] line matches a launch/readiness.txt fail rule", not leak, str(leak[:1]))
    R("C1 the [patch-033] ACTIVE lines are exactly 8 (one per rank), 0 INACTIVE",
      sum(1 for m in pl if "(Marlin) ACTIVE (rank" in m) == 8 and not [1 for m in pl if "INACTIVE" in m])
    mods = all_modules(b)
    R("C1 all 9 modules x 8 ranks: _Method installed, a UnquantizedLinearMethod subclass, mode active, no hit counted, "
      "raw FP8 released, BF16 weights bitwise the loader's",
      len(mods) == 72 and all(type(m.quant_method).__name__ == "_Method" and isinstance(m.quant_method, ns.UQM)
                              and m.quant_method._p.mode == "active" and m.quant_method._p.hits == 0
                              and m.quant_method._p.entry.w8 is None
                              and np.array_equal(m._parameters["weight"]._a, oracle_weight(ck[L], typ, r, SMALL))
                              for r, L, typ, m in mods))
    R("C1 the caching allocator is clean on the main thread after the boot", T.allocator_clean())
    # ---- post-boot behaviour on the REAL modules
    mod = ns.mod
    tot = []
    for r in (0, 5):
        T.set_current_world(b.worlds[r])
        for rb in b.res[r]:
            for L, layer in rb.layers.items():
                for typ, m in (("qkv_a", layer.self_attn.fused_qkv_a_proj), ("q_b", layer.self_attn.q_b_proj),
                               ("o_proj", layer.self_attn.o_proj)):
                    p = m.quant_method._p
                    w8, sc = shard_of(ck[L], typ, r, SMALL)
                    s_full = np.repeat(np.repeat(sc, 128, 0), 128, 1)
                    wf = T.fp8_bits_to_f32(w8) * s_full
                    for M in (1, 3, 16):
                        x = mod._synth("unit", M, p.k, 500 + M, m.weight.device)
                        old = m.reduce_results
                        m.reduce_results = False
                        try:
                            y = m(x)[0]
                        finally:
                            m.reduce_results = old
                        tot.append(("marlin", r, L, typ, M, mod._mismatches(y, mod._marlin(p, x)) == 0
                                    and relrmse(y._f(), x._f().astype(np.float32) @ wf.T.astype(np.float32)) <= 0.006))
                    for M in (17, 40):
                        x = mod._synth("unit", M, p.k, 600 + M, m.weight.device)
                        y = m.quant_method.apply(m, x)
                        tot.append(("bf16", r, L, typ, M, mod._mismatches(y, mod._bf16(p, x)) == 0 and p.hits == 0))
    badt = [t for t in tot if not t[-1]]
    R("C2 after the boot the REAL module calls: M in {1,3,16} == the Marlin result and <= 0.006 vs the checkpoint fp32 math; "
      "M in {17,40} == the BF16 path bitwise (2 ranks x 9 modules)", not badt and len(tot) == 90, str(badt[:2]))
    # SF8: first small-M call lines
    T.set_current_world(b.worlds[5])
    L5 = [m for m in LOGS.lines(contains="first small-M call on rank 5/8")]
    R("C2 SF8: exactly one 'first small-M call ... -> marlin' line per projection type on rank 5 (3 total) after 18 calls",
      len(L5) == 3 and all("-> marlin (M=" in m for m in L5) and {m.split(": ")[1].split(" ")[0] for m in L5} == set(TYPES_), str(L5))
    T.set_current_world(b.worlds[2])
    lay = next(iter(b.res[2][0].layers.values())).self_attn
    n0 = len(LOGS.lines(contains="first small-M call on rank 2/8"))
    x3 = mod._synth("unit", 3, lay.q_b_proj.weight.shape[1], 3, lay.q_b_proj.weight.device).reshape(1, 3, lay.q_b_proj.weight.shape[1])
    lay.q_b_proj(x3)
    lay.q_b_proj(x3)
    L2 = LOGS.lines(contains="first small-M call on rank 2/8")
    R("C2 SF8: a 3-D small-M call (the unpinned wrapper's possible shape) falls to BF16 and is logged once as such",
      len(L2) == n0 + 1 and "q_b -> bf16 (dim 3)" in L2[-1], str(L2[-1:]))
    # a second load after the decision: this module goes back to the image GEMM, but its Marlin tensors stay alive (R2: CUDA
    # graphs captured while it was active hold raw pointers to them) and one loud warning says to restart the server
    T.set_current_world(b.worlds[5])
    mm = b.res[5][0].layers[3].self_attn.q_b_proj
    mm2 = b.res[5][0].layers[3].self_attn.o_proj
    qm, qm2 = mm.quant_method, mm2.quant_method
    bufs = [t for q in (qm, qm2) for t in (q._p.w, q._p.s, q._p.ws)]
    R("R2 before the reload both projections hold their Marlin weight / scales / workspace", all(t is not None for t in bufs))
    n0 = len(LOGS.lines(contains="loaded again after the boot decision"))
    qm.process_weights_after_loading(mm)
    qm2.process_weights_after_loading(mm2)
    st5 = mod._state()
    W = LOGS.lines(contains="loaded again after the boot decision")[n0:]
    R("C2 a second process_weights_after_loading after the decision returns that module to the image GEMM (warned)",
      type(mm.quant_method) is ns.UQM and qm._p.mode == "off" and bool(W))
    R("R2 the reload warning says the captured graphs still read the OLD FP8 weights and to restart; one line per rank, not "
      "per module", len(W) == 1 and "CUDA graphs captured earlier still read the OLD" in W[0] and "restart the server" in W[0]
      and "ACTIVE" not in W[0], str(W[:1])[:160])
    R("R2 the Marlin weight / scales / workspace of both reloaded projections are NOT freed: the state keeps the very same "
      "tensor objects (graphs hold raw pointers to them), while the projections themselves dropped them",
      len(st5.retired) == 2 and all(any(t is u for tup in st5.retired for u in tup) for t in bufs)
      and qm._p.w is None and qm2._p.ws is None)
    qm.process_weights_after_loading(mm)                 # a third load changes nothing more
    R("R2 a further load of an already retired projection is a no-op", len(st5.retired) == 2
      and len(LOGS.lines(contains="loaded again after the boot decision")[n0:]) == 1)
    return R


TYPES_ = ("qkv_a", "q_b", "o_proj")


def sc_probe(cx):
    """MF1: probe state restored on every path; exactly one hit per real module call; pending == BF16."""
    R = Res()
    u = unit(cx.vroot, cx.text, layers=(3, 7), module_path=cx.module_path, mu8_text=cx.mu8_text)
    build_all(u)
    mod = u.ns.mod
    R("MF1 states are pending / probe / active / off", mod.MODES == ("pending", "probe", "active", "off"))
    R("MF1 every built module starts in pending", all(p.mode == mod.PENDING for p in u.g.projs))
    # (b) exactly one hit per real module call, only in the probe state
    ok = True
    info = ""
    with mod._probing(u.g.projs):
        for p in u.g.projs:
            x = mod._synth("unit", 3, p.k, 9, p.module.weight.device)
            old = p.module.reduce_results
            p.module.reduce_results = False
            h0 = p.hits
            out = p.module(x)[0]
            p.module.reduce_results = old
            if p.hits != h0 + 1 or mod._mismatches(out, mod._marlin(p, x)) != 0:
                ok, info = False, f"{p.label}: hits {p.hits - h0}"
    R("MF1 (b) in the probe state each REAL module call (fused qkv_a, q_b, o_proj) counts exactly one hit and returns the "
      "Marlin result", ok, info)
    R("MF1 the probe state is left on exit: pending, hits reset", all(p.mode == mod.PENDING and p.hits == 0 for p in u.g.projs))
    p = proj_of(u, "q_b")
    x = mod._synth("unit", 3, p.k, 9, p.module.weight.device)
    y = p.module(x)[0]
    R("MF1 in pending a module call is the BF16 path bitwise and counts no hit",
      mod._mismatches(y, mod._bf16(p, x)) == 0 and p.hits == 0)

    class Boom(BaseException):
        pass
    orig = mod._check_proj
    calls = [0]

    def bomb(p_, *a, **k):
        calls[0] += 1
        if calls[0] == 2:
            raise Boom()
        return orig(p_, *a, **k)
    mod._check_proj = bomb
    try:
        mod._selftest_group(u.st, u.g, u.w)
        raised = False
    except Boom:
        raised = True
    finally:
        mod._check_proj = orig
    R("MF1 (a) a BaseException raised mid-probe propagates AND leaves every module in pending (try/finally), hits reset",
      raised and all(p.mode == mod.PENDING and p.hits == 0 for p in u.g.projs))
    calls[0] = 0

    def bomb2(p_, *a, **k):
        calls[0] += 1
        if calls[0] == 2:
            raise ValueError("injected")
        return orig(p_, *a, **k)
    mod._check_proj = bomb2
    try:
        row = mod._selftest_group(u.st, u.g, u.w)
    finally:
        mod._check_proj = orig
    R("MF1 (a) an Exception mid-probe fails that module's row (the self-test goes on) and leaves every module pending",
      not row["ok"] and "C0" in row["why"] and all(p.mode == mod.PENDING and p.hits == 0 for p in u.g.projs), row["why"][:80])
    ms = {m for _d, m in mod._case_list(16)}
    R("SF7 the self-test M list covers every capture size 1,2,4,6,8,16 plus the eager verify size 3 (and MAX_M)",
      {1, 2, 3, 4, 6, 8, 16} <= ms and {m for _d, m in mod._case_list(8)} == {1, 2, 3, 4, 6, 8}, str(sorted(ms)))
    src = (cx.module_path or MODULE_PATH).read_text()
    R("MF1 the module has the state set and the finally",
      all(s in src for s in ('PENDING = "pending"', 'PROBE = "probe"', 'ACTIVE = "active"', 'OFF = "off"', "    finally:\n        for p in projs:\n            p.mode, p.hits = PENDING, 0")))
    return R


def unit_defang(cx):
    R = Res()
    ns = new_ns(cx.vroot, cx.text, module_path=cx.module_path, mu8_text=cx.mu8_text)
    m = ns.mod
    leaks = [(rx.pattern, g) for g in GATE_TEXTS for _, rx in _READINESS if rx.search(m._safe(g))]
    R("B2 _safe() defangs every gate text (none matches any readiness fail / require rule afterwards)", not leaks, str(leaks[:2]))
    return R


def sc_seams(cx):
    R = Res()
    for byp, active, gone in ((("qkv_a",), ["q_b", "o_proj"], "qkv_a"), (("q_b",), ["qkv_a", "o_proj"], "q_b"),
                              (("o_proj",), ["qkv_a", "q_b"], "o_proj")):
        b, ns, ck = boot(cx, hooks=Hooks(bypass=byp))
        h, hi = healthy(b)
        ds = decisions(b)
        R(f"C4 SF2: the {gone} module call bypasses quant_method.apply -> {gone} falls back ALONE: ACTIVE on 8 ranks, "
          f"healthy, one line each", ds == ["ACTIVE"] * 8 and h and all(len(final_lines(b, r)) == 1 for r in range(8)),
          f"{ds} {hi}")
        ln = final_lines(b, 3)[0] if final_lines(b, 3) else ""
        R(f"C4 the ACTIVE line shows the active set {{{', '.join(active)}}} and '; not active: {gone} not active'",
          f"active types {{{', '.join(active)}}}" in ln and f"; not active: {gone} not active" in ln, ln[-220:])
        ok = True
        why = ""
        for r, L, typ, m in all_modules(b):
            if typ == gone and type(m.quant_method) is not ns.UQM:
                ok, why = False, f"{r}/{L}/{typ} not restored"
            if typ != gone and (type(m.quant_method).__name__ != "_Method" or m.quant_method._p.mode != "active"):
                ok, why = False, f"{r}/{L}/{typ} not active"
        R(f"C4 {gone} modules are the image's method again, the other types stay Marlin-active", ok, why)
    b, ns, ck = boot(cx, hooks=Hooks(bypass_marlin=("qkv_a",)))
    ln = final_lines(b, 3)[0] if final_lines(b, 3) else ""
    R("C4 SF2: a dedicated qkv_a path returning the SAME bits without reaching quant_method.apply is caught by the hit "
      "count alone (qkv_a falls back, q_b / o_proj stay)", decisions(b) == ["ACTIVE"] * 8 and "active types {q_b, o_proj}" in ln
      and "reached quant_method.apply 0 times" in ln, ln[-200:])
    b, ns, ck = boot(cx, hooks=Hooks(bypass=TYPES_))
    expect_refused(R, b, ns, ck, SMALL, "all three module calls bypass quant_method.apply", "seam check")
    return R


def sc_failclosed(cx, subset=None):
    R = Res()
    dims = SMALL

    def corrupt(rank_t, typ, phase_t="target", L=3, mode="w8"):
        def f(r, rb, phase):
            if r == rank_t and phase == phase_t:
                p = proj_in(rb, L, typ)
                if mode == "w8":
                    p.entry.w8._a[0, 0] ^= 1
                elif mode == "tp":
                    p.entry.tp_rank = (p.entry.tp_rank + 1) % 8
        return f

    ck_ok = make_ckpt(dims)
    ck_300 = make_ckpt(dims)
    ck_300[3]["q_b"][1][0, 0] = 300.0
    ck_other = make_ckpt(dims, seed=99)
    saved_cost = dict(COST)

    def inject_repack(kind):
        w = T.current_world()
        if kind == "repack" and getattr(w, "rank", -1) == 2:
            raise RuntimeError("injected repack failure")

    def raiser(g):
        raise RuntimeError("injected cudaStreamBeginCapture failure")
    cleans = {}
    cases = []

    def case(name, why, hooks=None, env=None, ckpt=None, size=8, rv=6.0, allow_hang=False, weights=True, inject=None,
             cost=None, ns_kw=None, want="INACTIVE", after=None, skip_restore=None):
        cases.append(dict(name=name, why=why, hooks=hooks, env=env, ckpt=ckpt, size=size, rv=rv, allow_hang=allow_hang,
                          weights=weights, inject=inject, cost=cost, ns_kw=ns_kw or {}, want=want, after=after,
                          skip_restore=skip_restore))
    case("captured FP8 byte flipped on rank 3 (q_b of layer 3)", "V2", Hooks(after_load=corrupt(3, "q_b")))
    case("one checkpoint tensor never reaches the loader on rank 5 (o_proj scale)", "checkpoint parts captured",
         Hooks(skip_load=lambda r, ph, n: r == 5 and ph == "target" and n.endswith("layers.3.self_attn.o_proj.weight_scale_inv")),
         weights=False)
    case("mis-sharded capture on rank 2 (tp rank off by one in the entry)", "V2", Hooks(after_construct=corrupt(2, "q_b", mode="tp")))
    case("a block scale of 300 in the checkpoint (>= 128)", "V4", ckpt=ck_300)
    case("replicated qkv_a checkpoint differs on rank 3", "differs across ranks", Hooks(ckpt_for=lambda r, c: ck_other if r == 3 else c),
         weights=False)
    case("gather exchange fails on rank 4 only (after taking part)", "exchange failed on this rank",
         Hooks(world=lambda r, w: setattr(w, "fail_gather_object", r == 4)))
    def late_gather_fail(r, w):
        if r == 4:
            orig, n = w.all_gather_object, [0]

            def wrapped(obj):
                n[0] += 1
                out = orig(obj)                              # takes part in the collective, then fails locally
                if n[0] == 2:
                    raise RuntimeError("injected gather failure on the MTP group exchange")
                return out
            w.all_gather_object = wrapped
    case("MTP-group gather exchange fails on rank 4 only (the commit round must catch the divergence)",
         "commit round failed or the ranks' decisions differ", Hooks(world=late_gather_fail))
    case("bounded barrier fails on EVERY rank", "exchange failed on this rank",
         Hooks(world=lambda r, w: setattr(w, "fail_barrier", True)))
    case("bounded barrier fails on rank 6 only (the others wait for it, bounded)", "", Hooks(world=lambda r, w: setattr(
        w, "fail_barrier", r == 6)), rv=1.0, allow_hang=True)
    case("fingerprint differs on rank 5 (speed-gate mode)", "fingerprint mismatch",
         Hooks(after_construct=lambda r, rb, ph: setattr(cx_mod_state(rb, r), "speed", "info") if (r == 5 and ph == "target") else None))
    case("Marlin build raises on rank 2", "build", inject=inject_repack)
    case("slow Marlin (1.02 x BF16 in graphs) on every rank", "S1", cost={"marlin_bps": 100e9})
    case("slow Marlin on rank 6 only", "rank(s) [6]", Hooks(world=lambda r, w: setattr(w, "cost_scale", {"marlin": 0.3} if r == 6 else None)))
    case("CUDA-graph capture fails on rank 1 (speed gate)", "S1 CUDA-graph sweep ran",
         Hooks(world=lambda r, w: setattr(w, "capture_hook", raiser if r == 1 else None),
               after_pwal=lambda r, rb, ph: cleans.__setitem__((r, ph), T.allocator_clean())))
    case("MTP draft layer capture corrupted on rank 3 (the target group was fine)", "mtp group self-test failed on rank(s) [3]",
         Hooks(after_load=corrupt(3, "q_b", phase_t="mtp", L=45)))
    case("target group corrupted on rank 3: the MTP exchange and the commit round still follow", "target group self-test failed",
         Hooks(after_load=corrupt(3, "qkv_a")))
    case("TP world of 4 ranks (validated for 8 only)", "TP=8 only", size=4)
    case("pipeline parallel size 2", "pipeline parallel size 1", Hooks(pp=lambda r: 2))
    case("LoRA enabled", "LoRA", Hooks(vc=make_vc(lora=True)))
    case("dual-batch overlap / ubatching enabled (v0.32.0: one private lock buffer per projection)", "ubatching",
         Hooks(vc=make_vc(ubatch=True)))
    case("VLLM_BATCH_INVARIANT=1", "VLLM_BATCH_INVARIANT", env={"VLLM_BATCH_INVARIANT": "1"})
    case("default dtype float32", "model dtype", Hooks(world=lambda r, w: T.set_default_dtype(T.float32)), after="dtype")
    case("MAX_M = 65", "VLLM_GB10_MLA_W8A16_MAX_M", env={"VLLM_GB10_MLA_W8A16_MAX_M": "65"})
    case("speed gate mode 'numerics'", "VLLM_GB10_MLA_W8A16_SPEED_GATE", env={"VLLM_GB10_MLA_W8A16_SPEED_GATE": "numerics"})
    case("q_lora_rank is None", "q_lora_rank", Hooks(attn_mut=lambda r, a: setattr(a, "q_lora_rank", None)))
    case("o_proj without reduce_results", "reduce_results", Hooks(attn_mut=lambda r, a: delattr(a.o_proj, "reduce_results")))
    case("a bias on q_b", "has a bias", Hooks(attn_mut=lambda r, a: setattr(a.q_b_proj, "bias", T.zeros(4))))
    case("q_b weight is float32", "not a 2-D bf16 CUDA tensor", Hooks(attn_mut=fp32_weight), weights=False)
    case("a quantised quant_method on o_proj", "expected exactly UnquantizedLinearMethod",
         Hooks(attn_mut=lambda r, a: setattr(a.o_proj, "quant_method", type("Fp8LinearMethod", (), {})())), skip_restore="o_proj")
    case("no Marlin op registered", "marlin_gemm missing", ns_kw={"with_marlin": False})
    case("CUDA capability 9.0", "capability", ns_kw={"cap": (9, 0)})
    for c in cases:
        if subset is not None and not any(s in c["name"] for s in subset):
            continue
        COST.clear()
        COST.update(saved_cost)
        if c["cost"]:
            COST.update(c["cost"])
        T.set_default_dtype(T.bfloat16)
        try:
            b, ns, ck = boot(cx, hooks=c["hooks"], env=c["env"], ckpt=c["ckpt"] or ck_ok, size=c["size"], rv_timeout=c["rv"],
                             inject=c["inject"], **c["ns_kw"])
        finally:
            COST.clear()
            COST.update(saved_cost)
            T.set_default_dtype(T.bfloat16)
        expect_refused(R, b, ns, ck_ok if c["ckpt"] is None else c["ckpt"], dims, c["name"], c["why"],
                       check_weights=c["weights"], allow_hang=c["allow_hang"], skip_restore=c["skip_restore"])
        if c["after"] is None and "capture fails on rank 1" in c["name"]:
            R(f"C5 {c['name']}: the caching allocator is clean on every rank afterwards (the failed capture's pool was "
              "unregistered)", all(cleans.get((r, ph), False) for r in range(8) for ph in ("target",)), str(cleans))
    return R


def fp32_weight(r, a):
    w = T.Parameter(T.zeros(128, 128, dtype=T.float32, device="cuda:0"), requires_grad=False)
    w.weight_loader = a.q_b_proj.weight_loader
    a.q_b_proj.weight = w


def cx_mod_state(rb, r):
    """The p033 module state of the calling rank thread (via any installed method's module)."""
    mod = sys.modules["vllm._gb10_mla_w8a16"]
    return mod._state()


def sc_zero(cx):
    R = Res()
    # SF4: zero-scale blocks in the checkpoint -> still ACTIVE (V3 masks them, an all-zero row has cosine 1)
    zb = {(3, "q_b"): [(0, 0)], (7, "o_proj"): [(1, 0)], (3, "q_a"): [(0, 1)]}
    ck = make_ckpt(SMALL, zero_blocks=zb)
    b, ns, _ = boot(cx, ckpt=ck)
    R("C6 SF4 zero-scale blocks in q_a, q_b and o_proj checkpoints: ACTIVE on 8 ranks, healthy",
      decisions(b) == ["ACTIVE"] * 8 and healthy(b)[0], f"{decisions(b)} {healthy(b)[1]}")
    return R


def sc_misc(cx):
    R = Res()
    # load_weights entered twice before the decision (AutoWeightsLoader groups): still ACTIVE
    b, ns, _ = boot(cx, hooks=Hooks(load_twice=True))
    R("C6 the loader hook entered twice per checkpoint (before the decision): ACTIVE, healthy",
      decisions(b) == ["ACTIVE"] * 8 and healthy(b)[0])
    # MAX_M = 8: the dispatch bound follows the knob (ACTIVE; M=8 Marlin, M=9 the BF16 path bitwise)
    b, ns, ck = boot(cx, env={"VLLM_GB10_MLA_W8A16_MAX_M": "8"})
    T.set_current_world(b.worlds[3])
    mm = b.res[3][0].layers[3].self_attn.q_b_proj
    p = mm.quant_method._p
    x8, x9 = (ns.mod._synth("unit", n, p.k, 70 + n, mm.weight.device) for n in (8, 9))
    R("C6 MAX_M=8: ACTIVE on 8 ranks; M=8 is Marlin, M=9 the BF16 path bitwise; the ACTIVE line says M<=8",
      decisions(b) == ["ACTIVE"] * 8 and healthy(b)[0] and p.max_m == 8
      and ns.mod._mismatches(mm.quant_method.apply(mm, x8), ns.mod._marlin(p, x8)) == 0
      and ns.mod._mismatches(mm.quant_method.apply(mm, x9), ns.mod._bf16(p, x9)) == 0
      and "decode M<=8 on" in final_lines(b, 3)[0])
    # speed gate in info mode with a slow Marlin: logged, not enforced
    saved = dict(COST)
    COST["marlin_bps"] = 100e9
    try:
        b, ns, ck = boot(cx, env={"VLLM_GB10_MLA_W8A16_SPEED_GATE": "info"})
    finally:
        COST.clear()
        COST.update(saved)
    ln = final_lines(b, 2)[0] if final_lines(b, 2) else ""
    R("C6 speed gate 'info' with a slow Marlin: ACTIVE, the ratio is logged ('speed gate (info)', ratio 1.02), not enforced",
      decisions(b) == ["ACTIVE"] * 8 and "speed gate (info)" in ln and "ratio 1.02" in ln, ln[-160:])
    # MTP not expected, built later: the target group decides alone; the MTP layer stays BF16 (INFO)
    b, ns, ck = boot(cx, hooks=Hooks(vc=make_vc(expect_mtp=False)))
    ls = [m for r in range(8) for m in LOGS.lines(rank=r, contains="arrived after the boot decision")]
    R("C6 no MTP expected, MTP layer built later: ACTIVE for the target, 2 exchanges, the MTP layer is left as the image's BF16 GEMMs",
      decisions(b) == ["ACTIVE"] * 8 and healthy(b)[0] and len(b.rv.schedules[0]) == 4 and len(ls) == 8
      and all(type(m.quant_method) is ns.UQM for r, L, t, m in all_modules(b) if L == 45)
      and all(type(m.quant_method).__name__ == "_Method" for r, L, t, m in all_modules(b) if L != 45),
      f"{decisions(b)} sched {len(b.rv.schedules[0])} info {len(ls)}")
    # MTP off (no speculative config, no MTP layer: the MTP-off boot of the tier-1 gate): the target group decides alone
    b, ns, ck = boot(cx, hooks=Hooks(vc=make_vc(expect_mtp=False), mtp=False))
    ln = final_lines(b, 4)[0] if final_lines(b, 4) else ""
    R("C6 MTP off (no speculative config, no MTP layer): ACTIVE on all 8 ranks after the target group alone, 2 exchanges "
      "(target group, commit), '0 MTP MLA layers', no INFO 'waiting for the MTP draft group' line",
      decisions(b) == ["ACTIVE"] * 8 and healthy(b)[0] and len(b.rv.schedules[0]) == 4 and "on 2 target + 0 MTP MLA layers" in ln
      and not LOGS.lines(contains="waiting for the MTP draft group")
      and all(m.quant_method._p.mode == "active" for r, L, t, m in all_modules(b)),
      f"{decisions(b)} sched {len(b.rv.schedules[0])} {ln[:110]}")
    # MTP expected but never built: silent pending, BF16 behaviour, one warning at the first decode-sized call
    b, ns, ck = boot(cx, hooks=Hooks(mtp=False))
    R("C6 MTP expected but never built: no decision line on any rank, modules pending (BF16), no hang",
      decisions(b) == ["NONE"] * 8 and healthy(b)[0] and all(m.quant_method._p.mode == "pending" for r, L, t, m in all_modules(b)),
      str(decisions(b)))
    T.set_current_world(b.worlds[1])
    mm = b.res[1][0].layers[3].self_attn.q_b_proj
    x = ns.mod._synth("unit", 3, 128, 3, mm.weight.device)
    y = mm(x)[0]
    R("C6 ...and a decode-sized call there is the BF16 GEMM bitwise, with one warning that the decision never completed",
      ns.mod._mismatches(y, ns.mod._bf16(mm.quant_method._p, x)) == 0
      and len(LOGS.lines(contains="before the boot decision completed on rank 1/8")) == 1)
    # install after finalisation is a no-op, a second boot in the same process state is not needed (state per rank)
    return R


def run_section_c(cx):
    print("== C. 8-rank threaded boots (REAL patched loader code, REAL v0.32.0 Marlin helpers, group-128 fake kernel)")
    print("   [stand-in] process_weights_after_loading is a STAND-IN loop (model.named_modules() order, "
          "quant_method.process_weights_after_loading(module)); the REAL function's per-module contract is checked statically in A7")
    for fn in (sc_clean, sc_probe, sc_seams, sc_failclosed, sc_zero, sc_misc):
        for name, ok, info in fn(cx):
            check(name, ok, info)


# =====================================================================================================================
# D: real-size numerics
# =====================================================================================================================
def section_d(vroot, texts, ap):
    print("== D. real-size numerics (E3 per-rank shapes, synthetic checkpoint)")
    text = ap.apply(texts[SERVED])
    L = 23
    rank = 5
    ck = {L: make_ckpt(REAL, (L,), seed=3, only_rank=rank)[L]}
    ns = new_ns(vroot, text)
    w = T.ThreadWorld(T.Rendezvous(8, timeout=2.0), rank, 8, oneshot_max=None)
    T.set_current_world(w)
    h = Hooks(layers=(L,), mtp=False)
    rb = build_phase(ns, REAL, rank, (L,), False, h)
    load_phase(ns, rb, ck, rank, h, "target")
    st = ns.mod._state()
    g = st.groups["target"]
    for p in g.projs:
        ns.mod._build(p, p.module)
    rows = ns.mod._Rows()
    worst, seam, dig = {}, {}, {}
    with ns.mod._probing(g.projs):
        for p in g.projs:
            ns.mod._check_proj(p, rows, worst, seam, dig)
    check("D1 all of C1-C8 pass on the three real per-rank shapes (2048x4096, 2048x1536, 4096x2048)", rows.ok and not seam,
          str(rows.fail[:2]))
    for t in TYPES_:
        d = worst.get(t, {})
        check(f"D2 {t}: worst relRMSE vs fp32 {d.get('c1')} (E3 real worst 0.0035, bound 0.006), vs BF16 path {d.get('c2')} "
              f"(0.0042 / 0.008), vs own dequant {d.get('c3')} (0.0025 / 0.0045), BF16 vs fp32 {d.get('c4')} (0.0025 / 0.0045), "
              f"min cos {d.get('c5')}", d and d["c1"] <= 0.006 and d["c2"] <= 0.008 and d["c3"] <= 0.0045
              and d["c4"] <= 0.0045 and d["c5"] >= 0.99998)
    ok_mem = sum(p.built_bytes for p in g.projs) == 8519680 + 3194880 + 8519680 + 3 * 768
    check("D3 Marlin bytes of one layer = 8,519,680 + 3,194,880 + 8,519,680 (+ 3 own workspaces of 4 x 48 int32 = 768 B) = "
          "20,234,240 + 2,304", ok_mem, str(sum(p.built_bytes for p in g.projs)))
    check("D3 memory: 12 layers x 20,234,240 B = 0.2261 GiB", abs(12 * 20234240 / 2 ** 30 - 0.2261) < 5e-4)


# =====================================================================================================================
# E: mutants
# =====================================================================================================================
BUILD_MUTANTS = [
    # (name, target file, old, new)
    ("MF2-1 size_k_first flipped (prepare called with size_k_first=True)", "module",
     "MU8.prepare_fp8_layer_for_marlin(H, size_k_first=False)", "MU8.prepare_fp8_layer_for_marlin(H, size_k_first=True)"),
    ("MF2-2 scales not transposed", "mu8", "        if not size_k_first:\n            scales = scales.T.contiguous()\n",
     "        if False:\n            scales = scales.T.contiguous()\n"),
    ("MF2-3 repeat_interleave on the wrong axis", "mu8", "        scales = scales.repeat_interleave(block_n, 1)\n",
     "        scales = scales.repeat_interleave(block_n, 0)\n"),
    ("MF2-4 fp8_fused_exponent_bias_into_scales missing", "mu8",
     "        marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)\n    if hasattr(layer, \"weight_scale\"):",
     "        pass\n    if hasattr(layer, \"weight_scale\"):"),
    ("MF2-5 q_a / kv_a row order swapped in the fused capture", "module",
     'rows = {"q_a": (0, ent.n_q), "kv_a": (ent.n_q, ent.n)}', 'rows = {"q_a": (ent.n - ent.n_q, ent.n), "kv_a": (0, ent.n - ent.n_q)}'),
    ("MF2-6 o_proj shard taken on dim 0", "module",
     "src_w, src_s = weight_fp8[:, c0:c0 + K], scale_inv[:, t0:t0 + K // BLOCK]",
     "src_w, src_s = weight_fp8[c0:c0 + N, :K], scale_inv[t0:t0 + N // BLOCK, :K // BLOCK]"),
    ("MF2-7 q_b scale rows off by one block", "module", "s0 = ent.tp_rank * (N // BLOCK)", "s0 = ent.tp_rank * (N // BLOCK) + 1"),
    ("MF1 dispatch requires ACTIVE only (the probe state would run the BF16 path)", "module",
     "        if mode == ACTIVE or mode == PROBE:\n", "        if mode == ACTIVE:\n"),
]
GENERIC_MUTANTS = [
    ("dispatch ignores the MAX_M upper bound", "module", "and 0 < x.shape[0] <= p.max_m:", "and 0 < x.shape[0]:"),
    ("dispatch ignores the input dtype", "module", "x.dim() == 2 and x.dtype == torch.bfloat16 and 0 <", "x.dim() == 2 and 0 <"),
    ("dispatch ignores the bias", "module", "if bias is None and x.dim() == 2 and", "if x.dim() == 2 and"),
    ("dispatch ignores the rank of the input", "module", "if bias is None and x.dim() == 2 and", "if bias is None and"),
    ("hits counted in every state", "module", "                if mode == PROBE:\n                    p.hits += 1\n                else:\n                    _first_call(p, \"marlin\", x)\n",
     "                p.hits += 1\n                _first_call(p, \"marlin\", x)\n"),
    ("Marlin input not made contiguous", "module", "input=x if x.is_contiguous() else x.contiguous(), weight=p.w",
     "input=x, weight=p.w"),
    ("the probe state is left as active", "module", "            p.mode, p.hits = PENDING, 0\n\n\ndef _case_list",
     "            p.mode, p.hits = ACTIVE, 0\n\n\ndef _case_list"),
    ("V2 comparison disabled", "module", "        bad2 += _count(_i16(served) != _i16(w[r0:r1]))\n", "        bad2 += 0\n"),
    ("V4 scale bound disabled", "module", "float(sc.amax().item()) >= SCALE_MAX", "False"),
    ("commit round dropped", "module", "    if com is None or len({(tuple(r.get(\"types\", ())), str(r.get(\"fp\"))) for r in com}) != 1:",
     "    if False:"),
    ("commit rows omit the decided types (a diverging rank goes unnoticed)", "module",
     'com = _exchange(world, {"rank": world.rank, "types": list(types), "fp": fp_final}, "commit")',
     'com = _exchange(world, {"rank": world.rank, "types": [], "fp": "x"}, "commit")'),
    ("exchange failure not refused", "module", "        return None\n    if not isinstance(out, list)", "        return [{}] * world.size\n    if not isinstance(out, list)"),
    ("per-type fallback restores nothing", "module", "    drop = [p for p in projs if p not in keep]\n    _restore(st, drop)\n",
     "    drop = [p for p in projs if p not in keep]\n    for p in drop:\n        p.mode = OFF\n"),
    ("seam check accepts any number of hits", "module", "        if hits != 1:\n", "        if False:\n"),
    ("speed gate not enforced", "module", "        if st.speed == \"graph\":\n            rows.check(f\"S1 CUDA-graph sweep: Marlin", "        if False:\n            rows.check(f\"S1 CUDA-graph sweep: Marlin"),
    ("C1 and C3 bounds disabled (the two layout checks are mutually redundant: both off)", "module",
     ['rows.check(f"C1 {tag}: Marlin vs fp32 relRMSE <= {C1_MAX}", c1 <= C1_MAX,',
      'rows.check(f"C3 {tag}: Marlin vs its own dequant relRMSE <= {C3_MAX}", c3 <= C3_MAX,'],
     ['rows.check(f"C1 {tag}: Marlin vs fp32 relRMSE <= {C1_MAX}", True,',
      'rows.check(f"C3 {tag}: Marlin vs its own dequant relRMSE <= {C3_MAX}", True,']),
    ("V3 not masked for zero-scale blocks (SF4)", "module", "        bad3 += _count((_u8(enc) != _u8(w8)) & ~small)\n",
     "        bad3 += _count(_u8(enc) != _u8(w8))\n"),
    ("C5 min cosine of an all-zero row is NaN (SF4 guard removed)", "module",
     "torch.where(den > 0.0, (yf * af).sum(dim=1) / den.clamp_min(1e-30), torch.ones_like(den))",
     "(yf * af).sum(dim=1) / den"),
    ("raw FP8 kept after the group verdict", "module", "    for p in g.projs:                                   # the raw FP8 is only needed by the self-test\n        p.entry.w8 = p.entry.sc = None\n",
     "    for p in g.projs:\n        pass\n"),
    ("capture copies nothing (a view of the checkpoint is kept)", "module", "    dst_w.copy_(src_w)                      # explicit copies into our own storage (SF5: no view pins the checkpoint)\n",
     "    ent.w8 = src_w if ent.typ == 'q_b' else ent.w8\n    dst_w.copy_(src_w)\n"),
    ("the real module call is not probed (hit check dropped)", "module", "        hits = p.hits - h0 - 1                                  # the direct call above counted one\n",
     "        hits = 1\n"),
    ("defanging disabled", "module", "def _safe(text) -> str:\n    s = str(text)\n", "def _safe(text) -> str:\n    return str(text)\n    s = str(text)\n"),
    ("self-test M list loses M=6", "module", "SELFTEST_M = (1, 2, 3, 4, 6, 8, 16)", "SELFTEST_M = (1, 2, 3, 4, 8, 16)"),
    ("the reference GEMMs read the TF32 flag again (R1: RuntimeError in a process that used the new TF32 API)", "module",
     "    ent = p.entry\n    R = int(X.shape[0])\n    Xd = X.to(torch.float64)\n",
     "    ent = p.entry\n    R = int(X.shape[0])\n    _tf32 = getattr(torch.backends.cuda.matmul, 'allow_tf32', None)\n    Xd = X.to(torch.float64)\n"),
    ("a reload after the decision frees the Marlin tensors that captured graphs point to (R2)", "module",
     "    st.retired.append((p.w, p.s, p.ws))\n    _restore(st, [p])\n", "    _restore(st, [p])\n"),
]


def mutate(src, old, new):
    pairs = list(zip(old, new)) if isinstance(old, list) else [(old, new)]
    for o, n in pairs:
        if src.count(o) != 1:
            return None
        src = src.replace(o, n, 1)
    return src


def run_mutant(vroot, text, work, name, tgt, old, new, required):
    msrc = MODULE_PATH.read_text()
    mu8src = (vroot / REL["mu8"]).read_text()
    src = msrc if tgt == "module" else mu8src
    out = mutate(src, old, new)
    if out is None:
        check(f"E mutant '{name}': anchor found exactly once", False, f"{src.count(old)}x")
        return
    d = work / "mutants"
    d.mkdir(parents=True, exist_ok=True)
    mp = None
    if tgt == "module":
        mp = d / (re.sub(r"[^A-Za-z0-9]+", "_", name) + ".py")
        mp.write_text(out)
    cx = NS(vroot=vroot, text=text, module_path=mp, mu8_text=out if tgt == "mu8" else None)
    if required:                                  # a build-path mutant must make the CLEAN boot refuse on every rank
        b, ns, ck = boot(cx)
        ds = decisions(b)
        h, hi = healthy(b)
        ln = (final_lines(b, 0) or ["-"])[0]
        ok = ds == ["INACTIVE"] * 8 and h
        shown = re.search(r"\): (.{0,110})", ln)
        print(f"MUTANT {name}: " + ("FAIL-as-required (boot refused on all 8 ranks, identical decisions, no hang)"
                                      if ok else f"NOT CAUGHT {ds} {hi}"))
        check(f"E mutant '{name}' makes the boot FAIL on all 8 ranks with identical decisions and no hang", ok,
              (shown.group(1) if shown else ln)[:110])
        return
    failed = []
    for fn in (unit_defang, unit_dispatch, sc_probe, unit_selftest, sc_clean, sc_seams_min, sc_failclosed_min, sc_speed_min,
               sc_zero):
        try:
            res = fn(cx)
        except BaseException as e:  # noqa: BLE001 - a crash is a catch
            failed.append((fn.__name__, f"crash {type(e).__name__}: {str(e)[:60]}"))
            break
        failed += [(n, i) for n, ok, i in res if not ok]
        if failed:
            break
    caught = bool(failed)
    print(f"MUTANT {name}: " + ("caught" if caught else "NOT CAUGHT"))
    check(f"E mutant '{name}' caught by the suite", caught, (failed[0][0][:70] + " | " + str(failed[0][1])[:50]) if failed else "")


def sc_seams_min(cx):
    R = Res()
    b, ns, ck = boot(cx, hooks=Hooks(bypass_marlin=("qkv_a",)))
    ln = final_lines(b, 3)[0] if final_lines(b, 3) else ""
    R("mini: qkv_a bypass -> ACTIVE {q_b, o_proj}, qkv_a restored", decisions(b) == ["ACTIVE"] * 8
      and "active types {q_b, o_proj}" in ln and all(type(m.quant_method) is ns.UQM for r, L, t, m in all_modules(b) if t == "qkv_a")
      and healthy(b)[0], ln[-120:])
    return R


def sc_failclosed_min(cx):
    return sc_failclosed(cx, subset=("captured FP8 byte flipped", "gather exchange fails on rank 4", "MTP draft layer capture",
                                     "q_lora_rank is None", "a block scale of 300", "MTP-group gather"))


def sc_speed_min(cx):
    return sc_failclosed(cx, subset=("slow Marlin (1.02",))


def section_e(vroot, texts, ap, work):
    print("== E. planted mutants")
    text = ap.apply(texts[SERVED])
    print("MUTANTS (build path, MF1 / MF2): " + "; ".join(n for n, *_ in BUILD_MUTANTS))
    for name, tgt, old, new in BUILD_MUTANTS:
        run_mutant(vroot, text, work, name, tgt, old, new, required=True)
    for name, tgt, old, new in GENERIC_MUTANTS:
        run_mutant(vroot, text, work, name, tgt, old, new, required=False)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--vllm-root", required=True, help="the v0.32.0 `vllm/` package directory (every python file of the tag)")
    ap_.add_argument("--bases", default=None, help="a directory of precomputed bases (orig/, p025/, orig_p032/, p025_p032/, each with model.py)")
    ap_.add_argument("--work", default=None)
    ap_.add_argument("--quick", action="store_true", help="skip D (real-size numerics)")
    ap_.add_argument("--mutants", action="store_true", help="run E (planted mutants); also lists them")
    ap_.add_argument("--only", default=None, help="subset of sections, e.g. ABC")
    ap_.add_argument("--list-mutants", action="store_true")
    ap_.add_argument("--print-pins", action="store_true", help="print the model.py base / result / MTP_P029 pin lines the current appliers produce")
    ap_.add_argument("--dump-bases", default=None, help="derive the model.py bases, write them to DIR/<name>/model.py, exit")
    args = ap_.parse_args()
    if args.print_pins:
        work = Path(args.work or tempfile.mkdtemp(prefix="p033-pins-"))
        work.mkdir(parents=True, exist_ok=True)
        return print_pins(Path(args.vllm_root), work, load_file_module("p033_apply", APPLIER_PATH))
    if args.dump_bases:
        work = Path(args.work or tempfile.mkdtemp(prefix="p033-bases-"))
        work.mkdir(parents=True, exist_ok=True)
        texts = derive_bases(Path(args.vllm_root), work)
        for v, dn in BASE_DIRS.items():
            if texts.get(v):
                (Path(args.dump_bases) / dn).mkdir(parents=True, exist_ok=True)
                (Path(args.dump_bases) / dn / "model.py").write_text(texts[v])
        bad = [v for v in BASES if texts.get(v) and not pending(v) and sha_text(texts[v]) != pin(v)]
        print(f"{sum(1 for v in BASES if texts.get(v))} of {len(BASES)} bases derivable" + (f"; MISMATCH {bad}" if bad else ""))
        return 1 if bad else 0
    if args.list_mutants:
        for n, tgt, _o, _n in BUILD_MUTANTS + GENERIC_MUTANTS:
            print(f"{tgt:7s} {n}")
        return 0
    QUIET["on"] = os.environ.get("P033_QUIET") == "1"
    vroot = Path(args.vllm_root)
    work = Path(args.work or tempfile.mkdtemp(prefix="p033-cpu-"))
    work.mkdir(parents=True, exist_ok=True)
    only = args.only or "ABCDE"
    ap, texts = section_a(vroot, work, Path(args.bases) if args.bases else None)
    global SERVED
    if not texts.get(SERVED):
        print(f"NOTE: the served base {SERVED} is not derivable (patch 032's port is pending): the module / loader tests below run on "
              "MODEL_ORIG + 033 (033's three hooks do not overlap 032's edits; the stacked result is proven by the stack check)")
        SERVED = "MODEL_ORIG"
    section_a_gate_off(vroot, texts, ap, SMALL)
    text = ap.apply(texts[SERVED])
    if "B" in only:
        section_b(vroot, texts, ap)
    if "C" in only:
        run_section_c(NS(vroot=vroot, text=text, module_path=None, mu8_text=None))
    if "D" in only and not args.quick:
        section_d(vroot, texts, ap)
    if ("E" in only and args.mutants) or args.only == "E":
        section_e(vroot, texts, ap, work)
    print(f"\n{RESULTS['pass']} passed, {RESULTS['fail']} failed, {RESULTS['skip']} skipped (SKIP = not verifiable on the Mac / port pending)")
    if RESULTS["failed"]:
        print("FAILED: " + " | ".join(RESULTS["failed"][:30]))
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())
