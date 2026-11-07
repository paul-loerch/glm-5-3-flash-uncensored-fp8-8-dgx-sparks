# SPDX-License-Identifier: Apache-2.0
"""Shared CPU test library of the mHC patch family (011 / 019 / 022 / 025) for the v0.32.0 port.

Needs torch (CPU) but no GPU, no TileLang and no vLLM install. It builds a stub `vllm` namespace around the REAL files
of a v0.32.0 source tree:
  * REAL: vllm/model_executor/warmup/jit_warmup.py (VllmJitKernel, the AST-traced dispatch, WarmupIntRange,
    JitWarmupRegistry), vllm/model_executor/warmup/jit_warmup_tilelang_helper.py (VllmTileLangJitKernel, kernel_launcher,
    make_tilelang_warmup_tensor), vllm/model_executor/kernels/mhc/{tilelang_kernels,tilelang,warmup}.py (optionally the
    PATCHED texts) - exec'd from files on disk so inspect.getsource() (used by the dispatch tracer) works;
  * STUBS: vllm.platforms.current_platform, vllm.tilelang_utils (T / tilelang / tilelang_jit -> FakeJit), the deep_gemm
    module, math / torch utils, logging. Every @tilelang_jit kernel becomes a FakeJit that records its launches and
    compiles and can run a CPU implementation (row-local pre-norm GEMM, post, big_fuse, fused) so end-to-end outputs can
    be compared bitwise.

Several environments can exist at once; Env.activate() points sys.modules at one of them (lazy imports inside the mHC
functions resolve through sys.modules at call time).
"""
from __future__ import annotations

import contextlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch

MHC_REL = "model_executor/kernels/mhc"
SRC_RELS = {
    "jit_warmup": "model_executor/warmup/jit_warmup.py",
    "helper": "model_executor/warmup/jit_warmup_tilelang_helper.py",
    "tk": f"{MHC_REL}/tilelang_kernels.py",
    "tl": f"{MHC_REL}/tilelang.py",
    "mhc_warmup": f"{MHC_REL}/warmup.py",
}


# ----------------------------------------------------------------------------------------------------------------------
# fake TileLang
# ----------------------------------------------------------------------------------------------------------------------
class Anything:
    """Stands in for tilelang.language (T) and tilelang attributes; kernel bodies are never executed here."""

    def __getattr__(self, name):
        return Anything()

    def __call__(self, *a, **k):
        return Anything()


class FakeJit:
    """A @tilelang_jit kernel: records launches / compiles, optionally runs a CPU implementation."""

    def __init__(self, fn, env):
        self.fn, self.env = fn, env
        self.__name__ = self.name = fn.__name__
        self.calls: list[tuple] = []
        self.compiles: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        impl = self.env.impls.get(self.name)
        return impl(*args, **kwargs) if impl is not None else None

    def compile(self, *args, **kwargs):
        self.compiles.append((args, kwargs))
        return self

    def reset(self):
        self.calls.clear()
        self.compiles.clear()


# ----------------------------------------------------------------------------------------------------------------------
# CPU implementations of the kernels (row-local by construction; only equality of two paths matters)
# ----------------------------------------------------------------------------------------------------------------------
def tree_sum_last(v):
    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        assert v.shape[-1] == 2 * h, "power-of-two K expected"
        v = v[..., :h] + v[..., h:]
    return v[..., 0]


def rowlocal_prenorm(x, fn):
    """out[t, c] = sum_k x[t,k] fn[c,k] (fixed binary tree over k), sq[t] = sum_k x[t,k]^2: depends on row t only."""
    n, _ = x.shape
    o = torch.empty(n, fn.shape[0], dtype=torch.float32)
    s = torch.empty(n, dtype=torch.float32)
    for r0 in range(0, n, 32):
        xc = x[r0:r0 + 32].float()
        o[r0:r0 + 32] = tree_sum_last(xc.unsqueeze(1) * fn.unsqueeze(0))
        s[r0:r0 + 32] = tree_sum_last(xc * xc)
    return o, s


def impl_prenorm(x, fn, out, sqrsum, hidden_size, hc_mult, n_out, n_thr, tile_n, last):
    """hc_prenorm_gemm_tilelang / hc_prenorm_gemm_block_m_tilelang (n_splits == 1 / block_m): row-local."""
    assert out.shape[0] == 1 and sqrsum.shape[0] == 1
    o, s = rowlocal_prenorm(x, fn)
    out[0].copy_(o)
    sqrsum[0].copy_(s)


def impl_post(a, b, c, d, x, hc, hidden):
    """mhc_post_tilelang(comb, residual, post, layer_out, out, hc, hidden)."""
    T = b.shape[0]
    for j in range(hc):
        acc = c[:, j:j + 1].float() * d.float()
        for k in range(hc):
            acc = acc + a[:, k, j:j + 1].float() * b[:, k].float()
        x[:, j] = acc.to(x.dtype)


def impl_fused(comb, res, post, x_in, w, yp, rp, ro, hc, hidden, n_out, n_thr=256, h_blk=256, tile_n=1, split_k=1,
               **_):
    """mhc_fused_tilelang / mhc_fused_bf16w_tilelang: post then the pre-norm FMA, split-k over the hidden size. The
    weight is widened with .float() (an exact promotion for bf16), so fp32 and bf16-exact weights give equal bits."""
    T = res.shape[0]
    new = torch.empty(T, hc, hidden, dtype=torch.float32)
    for j in range(hc):
        acc = post[:, j:j + 1].float() * x_in.float()
        for k in range(hc):
            acc = acc + comb[:, k, j:j + 1].float() * res[:, k].float()
        new[:, j] = acc
    ro.copy_(new.to(ro.dtype))
    n_splits = yp.shape[0]
    per = hidden // n_splits
    wf = w.float()
    for s_ in range(n_splits):
        sl = slice(s_ * per, (s_ + 1) * per)
        nr = new[:, :, sl].reshape(T, -1)                          # [T, hc * per]
        for c in range(n_out):
            yp[s_, :, c] = tree_sum_last((nr * wf[c][:, sl].reshape(1, -1)))
        rp[s_] = tree_sum_last(nr * nr)


def impl_noop(*_a, **_k):
    return None


def impl_big_fuse(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, post_mix, comb_mix, layer_input, *rest,
                  **kw):
    """A deterministic stand-in for mhc_pre_big_fuse*: a pure function of its inputs (per row)."""
    T = residual.shape[0]
    mixes = gemm_out_mul.sum(0)
    rms = torch.rsqrt(gemm_out_sqrsum.sum(0).unsqueeze(-1) / 16384.0 + 1e-6)
    m = mixes * rms
    post_mix.copy_(torch.sigmoid(m[:, 4:8]))
    comb_mix.copy_(m[:, 8:24])
    layer_input.copy_(((residual.float().sum(dim=-2)) * m[:, :1]).to(layer_input.dtype))


DEFAULT_IMPLS = {
    "hc_prenorm_gemm_tilelang": impl_prenorm,
    "hc_prenorm_gemm_block_m_tilelang": impl_prenorm,
    "mhc_post_tilelang": impl_post,
    "mhc_fused_tilelang": impl_fused,
    "mhc_fused_bf16w_tilelang": impl_fused,
    "mhc_pre_big_fuse_tilelang": impl_big_fuse,
    "mhc_pre_big_fuse_with_norm_tilelang": impl_big_fuse,
    "mhc_pre_big_fuse_broadcast_with_norm_tilelang": impl_noop,
    "hc_head_fuse_tilelang": impl_noop,
}


# ----------------------------------------------------------------------------------------------------------------------
# the environment
# ----------------------------------------------------------------------------------------------------------------------
class Capturing:
    on = False


class Env:
    def __init__(self):
        self.modules: dict[str, types.ModuleType] = {}
        self.impls = dict(DEFAULT_IMPLS)
        self.jits: dict[str, FakeJit] = {}
        self.custom_ops: dict[str, object] = {}
        self.deep_gemm_supported = False
        self.deep_gemm_calls: list[tuple] = []
        self.tk = self.tl = self.mhc_warmup = self.jit_warmup = self.helper = None
        self.loggers: list[logging.Logger] = []

    def jit(self, name: str) -> FakeJit:
        return self.jits[name]

    def reset_jits(self):
        for j in self.jits.values():
            j.reset()

    @contextlib.contextmanager
    def activate(self):
        saved = {k: sys.modules.get(k) for k in self.modules}
        stale = [k for k in list(sys.modules) if (k == "vllm" or k.startswith("vllm.")) and k not in self.modules]
        removed = {k: sys.modules.pop(k) for k in stale}
        sys.modules.update(self.modules)
        try:
            yield self
        finally:
            for k in list(self.modules):
                if k in saved and saved[k] is not None:
                    sys.modules[k] = saved[k]
                else:
                    sys.modules.pop(k, None)
            sys.modules.update(removed)

    # -- extra modules (e.g. the vendored 019 module) are loaded INTO this env --
    def load_module(self, dotted: str, path: Path, attach_to_vllm: bool = True):
        with self.activate():
            spec = importlib.util.spec_from_file_location(dotted, str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mod
            self.modules[dotted] = mod
            spec.loader.exec_module(mod)
            if attach_to_vllm and dotted.startswith("vllm."):
                setattr(self.modules["vllm"], dotted.split(".")[-1], mod)
        return mod


def _pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    return m


def build_env(tree: Path, tk_path: Path | None = None, tl_path: Path | None = None, *, deep_gemm: bool = False,
              load_tl: bool = True, platform_cuda: bool = True, extra=()) -> Env:
    """Exec the real jit_warmup / helper / tilelang_kernels / (tilelang) / mhc.warmup into a stub namespace.
    tk_path / tl_path override the tree's files (patched copies)."""
    tree = Path(tree)
    env = Env()
    env.deep_gemm_supported = deep_gemm
    mods = env.modules
    for n in ("vllm", "vllm.model_executor", "vllm.model_executor.kernels", "vllm.model_executor.kernels.mhc",
              "vllm.model_executor.warmup", "vllm.utils", "vllm.distributed"):
        mods[n] = _pkg(n)
    lg = types.ModuleType("vllm.logger")
    lg.init_logger = lambda n: logging.getLogger(n)
    mods["vllm.logger"] = lg
    mods["vllm"].logger = lg

    plat = types.SimpleNamespace(is_cuda=lambda: platform_cuda, is_rocm=lambda: False,
                                 is_arch_support_pdl=lambda: False,
                                 is_device_capability_family=lambda x: False)
    pm = types.ModuleType("vllm.platforms")
    pm.current_platform = plat
    mods["vllm.platforms"] = pm

    tl_stub = Anything()
    tlu = types.ModuleType("vllm.tilelang_utils")
    tlu.T, tlu.tilelang = tl_stub, tl_stub
    tlu.tilelang_jit = lambda fn: env.jits.setdefault(fn.__name__, FakeJit(fn, env))
    mods["vllm.tilelang_utils"] = tlu

    mu = types.ModuleType("vllm.utils.math_utils")
    mu.cdiv = lambda a, b: -(-a // b)
    mods["vllm.utils.math_utils"] = mu
    tu = types.ModuleType("vllm.utils.torch_utils")

    def direct_register_custom_op(**kw):
        env.custom_ops[kw["op_name"]] = kw["op_func"]

    tu.direct_register_custom_op = direct_register_custom_op
    mods["vllm.utils.torch_utils"] = tu
    dg = types.ModuleType("vllm.utils.deep_gemm")
    dg.is_deep_gemm_supported = lambda: env.deep_gemm_supported

    def tf32_hc_prenorm_gemm(x, fn, out, sqrsum, n_splits):
        env.deep_gemm_calls.append((tuple(x.shape), n_splits))
        o, s = rowlocal_prenorm(x, fn)
        out[0].copy_(o)
        sqrsum[0].copy_(s)

    dg.tf32_hc_prenorm_gemm = tf32_hc_prenorm_gemm
    mods["vllm.utils.deep_gemm"] = dg
    dist = mods["vllm.distributed"]
    dist.is_global_first_rank = lambda: False
    if importlib.util.find_spec("tqdm") is None:
        tq = types.ModuleType("tqdm")

        class _Tqdm:
            def __init__(self, it=(), **kw):
                self.it = it

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                return iter(self.it)

            def set_postfix_str(self, *a, **k):
                pass

        tq.tqdm = _Tqdm
        mods["tqdm"] = tq

    def load_real(dotted: str, path: Path):
        spec = importlib.util.spec_from_file_location(dotted, str(path))
        mod = importlib.util.module_from_spec(spec)
        mods[dotted] = mod
        with env.activate():
            sys.modules[dotted] = mod
            spec.loader.exec_module(mod)
        parent, _, leaf = dotted.rpartition(".")
        setattr(mods[parent], leaf, mod)
        return mod

    env.jit_warmup = load_real("vllm.model_executor.warmup.jit_warmup", tree / SRC_RELS["jit_warmup"])
    env.helper = load_real("vllm.model_executor.warmup.jit_warmup_tilelang_helper", tree / SRC_RELS["helper"])
    torch.cuda.get_device_properties = lambda i=0: types.SimpleNamespace(multi_processor_count=48)
    for dotted, path in extra:                      # e.g. the vendored 022 module: tilelang.py imports it at import time
        load_real(dotted, Path(path))
    env.tk = load_real("vllm.model_executor.kernels.mhc.tilelang_kernels", Path(tk_path or tree / SRC_RELS["tk"]))
    env.mhc_warmup = load_real("vllm.model_executor.kernels.mhc.warmup", tree / SRC_RELS["mhc_warmup"])
    if load_tl:
        env.tl = load_real("vllm.model_executor.kernels.mhc.tilelang", Path(tl_path or tree / SRC_RELS["tl"]))
    return env


# ----------------------------------------------------------------------------------------------------------------------
# misc helpers shared by the checks
# ----------------------------------------------------------------------------------------------------------------------
def fake_vllm_config(max_tokens: int = 8192):
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=max_tokens),
        kernel_config=types.SimpleNamespace(enable_jit_warmup=True))


def run_registry_warmup(env: Env, register, max_tokens: int = 8192):
    """Build the REAL JitWarmupRegistry, run `register()` under activate() (like the model construction does) and warm
    up. Returns the registry. Raises whatever the real warmup raises (e.g. 'different compile key')."""
    reg = env.jit_warmup.JitWarmupRegistry(fake_vllm_config(max_tokens))
    with reg.activate():
        register()
    reg.warmup()
    return reg


def tensors_equal_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return bool(torch.equal(a.contiguous().view(iv), b.contiguous().view(iv)))


def tir_probe(tree, module=None, tilelang_path=None, out_dir=None):
    """Run pmhc_tir_probe.py (TileLang 0.1.12 frontend -> TIR facts) in a subprocess. Returns (status, info dict):
    'ok' | 'fail' | 'skip' (tilelang not importable: set TILELANG_PROBE_PATH or pass tilelang_path; see the probe's docstring)."""
    import json
    import os
    import subprocess

    here = Path(__file__).resolve().parent
    path = tilelang_path or os.environ.get("TILELANG_PROBE_PATH")
    if not path:
        cand = here.parent.parent / "mhc" / "tlpkgs"
        path = str(cand) if cand.is_dir() else None
    cmd = [sys.executable, str(here / "pmhc_tir_probe.py"), "--tree", str(tree)]
    if module:
        cmd += ["--module", str(module)]
    if path:
        cmd += ["--tilelang-path", str(path)]
    if out_dir:
        cmd += ["--out-dir", str(out_dir)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    try:
        info = json.loads(p.stdout)
    except Exception:  # noqa: BLE001
        info = {"stdout": p.stdout[-300:], "stderr": p.stderr[-600:]}
    return ({0: "ok", 3: "skip"}.get(p.returncode, "fail")), info
