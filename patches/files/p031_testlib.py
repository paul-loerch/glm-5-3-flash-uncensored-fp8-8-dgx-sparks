# SPDX-License-Identifier: Apache-2.0
"""[patch-031 gb10-idx-shard-gate] CPU test library (no torch, no GPU, no vLLM install; python3 + numpy).

  make_torch()      a numpy-backed fake `torch` with exactly the tensor ops balanced_prefill_row_shard uses (arange, repeat_interleave,
                    cumsum, searchsorted, Tensor.to / .sum / .numel / .tolist, int64) -- results are the real functions' on int64
  Harness           executes the REAL top-level nodes of vllm/v1/attention/backends/mla/indexer.py (the pristine v0.32.0 file or the
                    patched text) in a namespace with the fake torch, a capturing logger and a stand-in `vllm.distributed`:
                      MIN_TP_SHARD_ROWS_PER_RANK, balanced_prefill_row_shard, tp_prefill_row_sharding_supported and, patched,
                      _p031_config, _P031_ON/_P031_MIN_CTX/_P031_MIN_ROWS, _P031_STATE, _p031_row_gate, _p031_announce,
                      _p031_note_engaged
                    plus the REAL CUDAGraphMode enum extracted from config/compilation.py. The env gate is read exactly as at import.
  ref_split()       an independent pure-Python (numpy-free) re-implementation of upstream's work-balanced contiguous split
"""
from __future__ import annotations

import ast
import contextlib
import enum
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# fake torch
# ---------------------------------------------------------------------------------------------------------------------


class FDtype:
    def __init__(self, name, npd):
        self.name, self.np = name, np.dtype(npd)

    def __repr__(self):
        return f"torch.{self.name}"


class FT(np.ndarray):
    """numpy array with the few torch.Tensor methods the code under test calls."""

    def to(self, dtype):
        return self.astype(dtype.np).view(FT)

    def numel(self):
        return int(self.size)


def _ft(a):
    return np.asarray(a).view(FT)


def make_torch():
    t = types.SimpleNamespace()
    t.int64 = FDtype("int64", np.int64)
    t.int32 = FDtype("int32", np.int32)
    t.Tensor = FT

    def tensor(data, dtype=None):
        return _ft(np.asarray(data, dtype=(dtype.np if dtype is not None else None)))

    def arange(*a):
        return _ft(np.arange(*a, dtype=np.int64))

    t.tensor, t.arange = tensor, arange
    t.repeat_interleave = lambda x, r: _ft(np.repeat(np.asarray(x), np.asarray(r)))
    t.cumsum = lambda x, dim: _ft(np.cumsum(np.asarray(x), axis=dim))
    t.searchsorted = lambda seq, vals: _ft(np.searchsorted(np.asarray(seq), np.asarray(vals), side="left"))
    return t


# ---------------------------------------------------------------------------------------------------------------------
# logging capture
# ---------------------------------------------------------------------------------------------------------------------
class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))

    def lines(self, level=None):
        return [m for lv, m in self.records if level is None or lv == level]


# ---------------------------------------------------------------------------------------------------------------------
# extracting the real nodes
# ---------------------------------------------------------------------------------------------------------------------
def _segment(text: str, node) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1: node.end_lineno])


def _cudagraph_enum(tree_root: Path):
    """The REAL CUDAGraphMode enum (config/compilation.py), so mixed_mode() is upstream's."""
    src = (tree_root / "config" / "compilation.py").read_text()
    mod = ast.parse(src)
    node = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "CUDAGraphMode")
    ns = {"enum": enum}
    exec(compile(_segment(src, node), "CUDAGraphMode", "exec"), ns)  # noqa: S102 - trusted local tree
    return ns["CUDAGraphMode"]


WANT_NAMES = ("MIN_TP_SHARD_ROWS_PER_RANK", "balanced_prefill_row_shard", "tp_prefill_row_sharding_supported",
              "_p031_config", "_P031_STATE", "_p031_row_gate", "_p031_announce", "_p031_note_engaged")


class Harness:
    """One 'import' of indexer.py's relevant nodes with the given environment variables."""

    def __init__(self, text: str, tree_root: Path, env: dict[str, str] | None = None, dist_rank: int | None = 3,
                 cuda: bool = True, flags: dict | None = None):
        self.text = text
        self.tree_root = Path(tree_root)
        self.torch = make_torch()
        self.cap = Capture()
        self.logger = logging.getLogger(f"p031test.{id(self)}")
        self.logger.handlers = [self.cap]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.CUDAGraphMode = _cudagraph_enum(self.tree_root)
        self.flags = {"VLLM_DISABLE_PYNCCL": False, "VLLM_USE_NCCL_SYMM_MEM": False, "VLLM_BATCH_INVARIANT": False}
        self.flags.update(flags or {})
        envs = types.SimpleNamespace(**self.flags)
        platform = types.SimpleNamespace(is_cuda=lambda: cuda)
        ns = {"torch": self.torch, "logger": self.logger, "envs": envs, "current_platform": platform,
              "CUDAGraphMode": self.CUDAGraphMode, "VllmConfig": object, "__name__": "indexer_under_test"}
        mod = ast.parse(text)
        wanted = []
        for n in mod.body:
            if isinstance(n, (ast.FunctionDef,)) and n.name in WANT_NAMES:
                wanted.append(n)
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id in WANT_NAMES:
                wanted.append(n)
            elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) \
                    and [e.id for e in n.targets[0].elts if isinstance(e, ast.Name)] == ["_P031_ON", "_P031_MIN_CTX", "_P031_MIN_ROWS"]:
                wanted.append(n)
        # a stand-in vllm.distributed for _p031_announce's local import
        saved_mods = {k: sys.modules.get(k) for k in ("vllm", "vllm.distributed")}
        if dist_rank is not None:
            vllm = types.ModuleType("vllm")
            dist = types.ModuleType("vllm.distributed")
            dist.get_tensor_model_parallel_rank = lambda: dist_rank
            vllm.distributed = dist
            sys.modules["vllm"], sys.modules["vllm.distributed"] = vllm, dist
        else:  # the import must fail
            sys.modules["vllm"] = None  # type: ignore[assignment]
            sys.modules["vllm.distributed"] = None  # type: ignore[assignment]
        self._saved_mods = saved_mods
        old_env = {k: os.environ.get(k) for k in os.environ if k.startswith("VLLM_GB10_IDX_SHARD")}
        for k in old_env:
            del os.environ[k]
        os.environ.update(env or {})
        self._env_keys = list((env or {}).keys())
        self._old_env = old_env
        try:
            for n in wanted:  # in file order = the order the interpreter would run them
                exec(compile(_segment(text, n), f"indexer.py:{n.lineno}", "exec"), ns)  # noqa: S102
        except BaseException:
            self.close()  # an import-time ValueError (bad gate value) must not leave the stand-in vllm modules behind
            raise
        finally:
            for k in self._env_keys:
                os.environ.pop(k, None)
            os.environ.update({k: v for k, v in old_env.items() if v is not None})
        self.ns = ns
        self.names = {n.name if isinstance(n, ast.FunctionDef) else getattr(n.targets[0], "id", "?") for n in wanted}

    def close(self):
        for k, v in self._saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- calls --------------------------------------------------------------------------------------------------------
    def shard(self, seq_lens, query_lens, tp, ratio=4):
        t = self.torch
        return self.ns["balanced_prefill_row_shard"](t.tensor(seq_lens, t.int64), t.tensor(query_lens, t.int64), ratio, tp)

    def cfg(self):
        return self.ns.get("_P031_ON"), self.ns.get("_P031_MIN_CTX"), self.ns.get("_P031_MIN_ROWS")

    def supported_cfg(self, tp=8, dcp=1, pcp=False, mode="FULL_AND_PIECEWISE"):
        m = None if mode is None else getattr(self.CUDAGraphMode, mode)
        comp = types.SimpleNamespace(cudagraph_mode=m)
        return types.SimpleNamespace(compilation_config=comp), dcp, pcp, tp


# ---------------------------------------------------------------------------------------------------------------------
# independent reference of upstream's split (pure Python, no numpy / torch)
# ---------------------------------------------------------------------------------------------------------------------
def ref_split(seq_lens, query_lens, ratio, tp):
    """Contiguous TP row counts balanced by indexer MQA work (a literal re-statement of the docstring, not of the code):
    row i of request r (0-based within the request) scores (seq_len_r - q_r + 1 + i) // ratio pooled keys; cut the running
    cost at total*k/tp (first row whose cumulative cost reaches the target), keep >= 1 row per rank."""
    cost, per_row = [], []
    run = 0
    for s, q in zip(seq_lens, query_lens):
        first = s - q + 1
        for i in range(q):
            c = (first + i) // ratio
            run += c
            cost.append(run)
            per_row.append(c)
    n = len(cost)
    total = cost[-1] if cost else 0
    if total <= 0:
        return None
    import bisect
    bounds = [0]
    for k in range(1, tp):
        bounds.append(bisect.bisect_left(cost, k * total // tp))
    bounds.append(n)
    for i in range(1, tp):
        bounds[i] = max(bounds[i], bounds[i - 1] + 1)
    for i in range(tp - 1, 0, -1):
        bounds[i] = min(bounds[i], bounds[i + 1] - 1)
    return [bounds[i + 1] - bounds[i] for i in range(tp)], per_row


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: v for k, v in kw.items() if v is not None})
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
