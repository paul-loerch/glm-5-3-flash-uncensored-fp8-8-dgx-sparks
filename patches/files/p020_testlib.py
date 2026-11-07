# SPDX-License-Identifier: Apache-2.0
"""[patch-020 gb10-sparse-mla-triton-prefill] CPU test library, REVISION 4 (Mac / any CPU: python3 >= 3.10
+ numpy; NO torch, NO triton, NO vLLM install). Not installed into vLLM.

  * a numpy-backed fake ``torch`` with exactly the surface the vendored module (gb10_sparse_mla_prefill.py),
    the REAL upstream functions run by the checks (FlashInferMLASparseSM90Builder._kv_lens_host /
    _row_contexts / build, Impl.forward_mqa / __init__) and the reference helpers need: bf16 stored as
    bf16-rounded float32, float8_e4m3fn as uint8 bit patterns (RNE; |x| >= 464 -> NaN like c10), advanced
    indexing, views and strides (as_strided / data_ptr identity), Generator / randn / randint / randperm,
    einsum / softmax / nan_to_num / masked_fill, default-device and default-dtype contexts with a STRICT mode
    that raises when a tensor is created without an explicit device or dtype (the model loader runs
    Impl.__init__ inside ``with torch.device(cuda:N)`` + a bf16 default dtype);
  * a fake ``triton`` (jit returns a launchable stub, cdiv, next_power_of_2) and ``tl`` (constexpr);
  * a STUBBED ``vllm`` namespace providing only the names flashinfer_mla_sparse_sm90.py imports, into which
    the REAL stock / patched backend source is exec'd, so the REAL _SM90 builder / impl code runs;
  * fakes at the FlashInfer seam: FakeState (records plan / pack_indices / wrapper.run, carries the index
    parameters _kv_lens_host reads) and a batch generator that produces V2-style sorted decode/prefill
    batches with a host upper bound (optimistic on draft rows) next to the exact device lengths.

The Triton kernel itself is NOT executed here: p020_gpu_selftest.py (--interp on a Linux box with triton,
--kernel / --fa2 / --split on a GB10) tests it; the checks pin its source text by sha instead (it is byte-identical to revision 3).
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import logging
import math
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------------------------------
# bit-level helpers (bf16 / fp8 e4m3fn)
# ---------------------------------------------------------------------------------------------------------------------
def round_bf16(a: np.ndarray) -> np.ndarray:
    """float32 array -> the nearest bf16 value (RNE), as float32."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint32) << 16
    out = r.view(np.float32).copy()
    nan = np.isnan(a)
    if nan.any():
        out[nan] = np.nan
    return out


def _fp8_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0**-6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


FP8_TABLE = _fp8_table()
FP8_POS = FP8_TABLE[:127].astype(np.float64)  # bytes 0x00..0x7E: 0 .. 448, increasing


def f32_to_fp8_bits(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    x = np.abs(a).astype(np.float64)
    e = np.floor(np.log2(np.where(x > 0, x, 1.0)))
    e = np.maximum(e, -6.0)
    sp = 2.0 ** (e - 3)
    q = np.round(x / sp) * sp  # half-to-even
    idx = np.searchsorted(FP8_POS, q)
    idx = np.clip(idx, 0, 126).astype(np.uint8)
    out = idx | np.where(np.signbit(a), np.uint8(0x80), np.uint8(0))
    bad = np.isnan(a) | (x >= 464.0)  # c10: |x| >= 464 -> NaN (a plain cast does not saturate)
    return np.where(bad, np.uint8(0x7F), out).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# fake torch
# ---------------------------------------------------------------------------------------------------------------------
class _DType:
    def __init__(self, name, store, kind, rank):
        self.name, self.store, self.kind, self.rank = name, np.dtype(store), kind, rank

    def __repr__(self):
        return f"torch.{self.name}"

    @property
    def is_floating_point(self):
        return self.kind in ("f", "bf16", "fp8")


float64 = _DType("float64", np.float64, "f", 70)
float32 = _DType("float32", np.float32, "f", 60)
float16 = _DType("float16", np.float16, "f", 50)
bfloat16 = _DType("bfloat16", np.float32, "bf16", 55)  # stored bf16-rounded
float8_e4m3fn = _DType("float8_e4m3fn", np.uint8, "fp8", 10)
int64 = _DType("int64", np.int64, "i", 40)
int32 = _DType("int32", np.int32, "i", 30)
int16 = _DType("int16", np.int16, "i", 25)
uint8 = _DType("uint8", np.uint8, "u", 20)
bool_ = _DType("bool", np.bool_, "b", 0)
_ALL = {d.name: d for d in (float64, float32, float16, bfloat16, float8_e4m3fn, int64, int32, int16, uint8, bool_)}


class _Device:
    def __init__(self, spec="cpu", index=None):
        if isinstance(spec, _Device):
            spec, index = spec.type, spec.index
        s = str(spec)
        if ":" in s:
            s, i = s.split(":")
            index = int(i)
        self.type, self.index = s, index

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    __repr__ = __str__

    def __eq__(self, o):
        return isinstance(o, _Device) and (self.type, self.index) == (o.type, o.index)

    def __hash__(self):
        return hash((self.type, self.index))

    def __enter__(self):
        _STATE["device_stack"].append(self)
        return self

    def __exit__(self, *a):
        _STATE["device_stack"].pop()
        return False


_STATE = {"device_stack": [], "default_dtype": float32, "strict": False, "capturing": False}


def _default_device() -> _Device:
    return _STATE["device_stack"][-1] if _STATE["device_stack"] else _Device("cpu")


def _resolve(device, dtype, float_result: bool):
    if device is None:
        if _STATE["strict"]:
            raise RuntimeError(f"[fake torch] tensor created without an explicit device inside "
                               f"`with torch.device({_default_device()})`")
        device = _default_device()
    device = _Device(device)
    if float_result and dtype is None:
        if _STATE["strict"] and _STATE["default_dtype"] is not float32:
            raise RuntimeError("[fake torch] float tensor created without an explicit dtype under a "
                               f"{_STATE['default_dtype']} default dtype")
        dtype = _STATE["default_dtype"]
    return device, dtype


def _cast_store(a: np.ndarray, dt: _DType) -> np.ndarray:
    """numpy values (any dtype) -> the storage array for `dt` (bf16 rounding, fp8 encode)."""
    if dt.kind == "bf16":
        return round_bf16(np.asarray(a, dtype=np.float32))
    if dt.kind == "fp8":
        return f32_to_fp8_bits(np.asarray(a, dtype=np.float32))
    return np.asarray(a).astype(dt.store, copy=False)


def _values(t: "Tensor") -> np.ndarray:
    """a tensor's mathematical values as a numpy array (fp8 decoded)."""
    if t.dtype.kind == "fp8":
        return FP8_TABLE[t.arr]
    return t.arr


def _promote(a, b):
    ta, tb = isinstance(a, Tensor), isinstance(b, Tensor)
    if ta and tb:
        da, db = a.dtype, b.dtype
        if da is db:
            return da
        if da.is_floating_point and not db.is_floating_point:
            return da
        if db.is_floating_point and not da.is_floating_point:
            return db
        if {da.kind, db.kind} == {"f", "bf16"}:
            return float32 if max(da.rank, db.rank) >= 60 else float32
        return da if da.rank >= db.rank else db
    t, s = (a, b) if ta else (b, a)
    if isinstance(s, bool):
        return t.dtype
    if isinstance(s, float) and not t.dtype.is_floating_point:
        return _STATE["default_dtype"]
    return t.dtype


class Tensor:
    def __init__(self, arr: np.ndarray, dt: _DType, device=None):
        self.arr = arr
        self.dtype = dt
        self.device = _Device(device) if device is not None else _Device("cpu")

    # --- metadata
    @property
    def shape(self):
        return tuple(self.arr.shape)

    @property
    def ndim(self):
        return self.arr.ndim

    def dim(self):
        return self.arr.ndim

    def numel(self):
        return int(self.arr.size)

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]

    def stride(self, i=None):
        st = tuple(s // self.arr.itemsize for s in self.arr.strides)
        return st if i is None else st[i]

    def data_ptr(self):
        return int(self.arr.__array_interface__["data"][0])

    def is_contiguous(self):
        return bool(self.arr.flags["C_CONTIGUOUS"])

    def __len__(self):
        return self.arr.shape[0]

    def __repr__(self):
        return f"FakeTensor({self.dtype}, shape={self.shape}, device={self.device})"

    # --- conversions
    def _wrap(self, arr, dt=None, device=None):
        return Tensor(arr, dt or self.dtype, device or self.device)

    def to(self, *args, **kw):
        dt, dev = kw.get("dtype"), kw.get("device")
        for a in args:
            if isinstance(a, _DType):
                dt = a
            elif isinstance(a, (_Device, str)):
                dev = _Device(a)
        if dt is None or dt is self.dtype:
            out = self if dev is None or dev == self.device else self._wrap(self.arr.copy(), device=dev)
            return out
        v = _values(self)
        return Tensor(_cast_store(v, dt), dt, dev or self.device)

    def float(self):
        return self.to(float32)

    def long(self):
        return self.to(int64)

    def int(self):
        return self.to(int32)

    def cpu(self):
        return self if self.device.type == "cpu" else self._wrap(self.arr, device=_Device("cpu"))

    def contiguous(self):
        return self if self.is_contiguous() else self._wrap(np.ascontiguousarray(self.arr))

    def clone(self):
        return self._wrap(self.arr.copy())

    def tolist(self):
        return _values(self).tolist()

    def item(self):
        return _values(self).item()

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def __bool__(self):
        return bool(self.item())

    def __index__(self):
        return int(self.item())

    # --- views
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], _DType):
            dt = shape[0]
            if dt.store.itemsize != self.arr.itemsize or dt.kind == "bf16" or self.dtype.kind == "bf16":
                raise RuntimeError("[fake torch] view(dtype) supports same-size non-bf16 reinterpretation only")
            return Tensor(self.arr.view(dt.store), dt, self.device)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new = self.arr.reshape(shape)
        if not np.shares_memory(new, self.arr):
            raise RuntimeError("[fake torch] view() would copy")
        return self._wrap(new)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._wrap(self.arr.reshape(shape))

    def unsqueeze(self, d):
        return self._wrap(np.expand_dims(self.arr, d))

    def transpose(self, a, b):
        return self._wrap(np.swapaxes(self.arr, a, b))

    def as_strided(self, size, stride, storage_offset=0):
        isz = self.arr.itemsize
        base = self.arr.reshape(-1) if self.arr.flags["C_CONTIGUOUS"] else None
        assert base is not None, "[fake torch] as_strided needs a contiguous base"
        out = np.lib.stride_tricks.as_strided(base[storage_offset:], shape=tuple(size),
                                              strides=tuple(s * isz for s in stride))
        return self._wrap(out)

    def new_empty(self, size, dtype=None, device=None):
        return empty(tuple(size), dtype=dtype or self.dtype, device=device or self.device)

    def new_zeros(self, size, dtype=None, device=None):
        return zeros(tuple(size), dtype=dtype or self.dtype, device=device or self.device)

    # --- indexing
    @staticmethod
    def _idx(i):
        if isinstance(i, Tensor):
            return i.arr
        if isinstance(i, tuple):
            return tuple(Tensor._idx(x) for x in i)
        return i

    def __getitem__(self, i):
        return self._wrap(self.arr[Tensor._idx(i)])

    def __setitem__(self, i, v):
        if isinstance(v, Tensor):
            val = _cast_store(_values(v), self.dtype) if v.dtype is not self.dtype else v.arr
        else:
            val = _cast_store(np.asarray(v), self.dtype)
        self.arr[Tensor._idx(i)] = val

    # --- in-place
    def copy_(self, src, non_blocking=False):
        self[...] = src
        return self

    def fill_(self, v):
        self.arr[...] = _cast_store(np.asarray(v), self.dtype)
        return self

    def zero_(self):
        return self.fill_(0)

    def clamp_(self, min=None, max=None):
        v = _values(self)
        self.arr[...] = _cast_store(np.clip(v, min, max), self.dtype)
        return self

    # --- arithmetic
    def _bin(self, o, fn, force_float=False, bool_out=False):
        a = _values(self)
        b = _values(o) if isinstance(o, Tensor) else o
        with np.errstate(all="ignore"):
            r = fn(a, b)
        if bool_out:
            return Tensor(np.asarray(r, dtype=np.bool_), bool_, self.device)
        dt = _promote(self, o)
        if force_float and not dt.is_floating_point:
            dt = _STATE["default_dtype"]
        return Tensor(_cast_store(r, dt), dt, self.device)

    def __add__(self, o): return self._bin(o, np.add)
    def __radd__(self, o): return self._bin(o, lambda a, b: b + a)
    def __sub__(self, o): return self._bin(o, np.subtract)
    def __rsub__(self, o): return self._bin(o, lambda a, b: b - a)
    def __mul__(self, o): return self._bin(o, np.multiply)
    def __rmul__(self, o): return self._bin(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._bin(o, np.true_divide, force_float=True)
    def __rtruediv__(self, o): return self._bin(o, lambda a, b: b / a, force_float=True)
    def __floordiv__(self, o): return self._bin(o, np.floor_divide)
    def __mod__(self, o): return self._bin(o, np.mod)
    def __neg__(self): return Tensor(_cast_store(-_values(self), self.dtype), self.dtype, self.device)
    def __invert__(self):
        if self.dtype.kind == "b":
            return Tensor(~self.arr, bool_, self.device)
        return Tensor(~self.arr, self.dtype, self.device)
    def __and__(self, o): return self._bin(o, np.logical_and if self.dtype.kind == "b" else np.bitwise_and)
    def __or__(self, o): return self._bin(o, np.logical_or if self.dtype.kind == "b" else np.bitwise_or)
    def __lt__(self, o): return self._bin(o, np.less, bool_out=True)
    def __le__(self, o): return self._bin(o, np.less_equal, bool_out=True)
    def __gt__(self, o): return self._bin(o, np.greater, bool_out=True)
    def __ge__(self, o): return self._bin(o, np.greater_equal, bool_out=True)
    def __eq__(self, o): return self._bin(o, np.equal, bool_out=True)
    def __ne__(self, o): return self._bin(o, np.not_equal, bool_out=True)
    __hash__ = None  # type: ignore[assignment]

    def abs(self):
        return Tensor(_cast_store(np.abs(_values(self)), self.dtype), self.dtype, self.device)

    def exp(self):
        return exp(self)

    def clamp(self, min=None, max=None):
        return clamp(self, min, max)

    def clamp_min(self, m):
        return clamp(self, m, None)

    def masked_fill(self, mask, value):
        out = _values(self).copy()
        out = np.where(mask.arr, value, out)
        return Tensor(_cast_store(out, self.dtype), self.dtype, self.device)

    # --- reductions
    def _reduce(self, fn, dim=None, keepdim=False):
        v = _values(self)
        if dim is None:
            r = fn(v)
            return Tensor(_cast_store(np.asarray(r), self.dtype if fn is not np.sum else self.dtype),
                          self.dtype, self.device)
        r = fn(v, axis=dim, keepdims=keepdim)
        return Tensor(_cast_store(r, self.dtype), self.dtype, self.device)

    def max(self, dim=None, keepdim=False):
        if dim is None:
            return self._reduce(np.max)
        v = _values(self)
        return SimpleNamespace(values=Tensor(_cast_store(v.max(axis=dim, keepdims=keepdim), self.dtype), self.dtype, self.device),
                               indices=Tensor(v.argmax(axis=dim).astype(np.int64), int64, self.device))

    def min(self, dim=None, keepdim=False):
        if dim is None:
            return self._reduce(np.min)
        v = _values(self)
        return SimpleNamespace(values=Tensor(_cast_store(v.min(axis=dim, keepdims=keepdim), self.dtype), self.dtype, self.device),
                               indices=Tensor(v.argmin(axis=dim).astype(np.int64), int64, self.device))

    def sum(self, dim=None, keepdim=False):
        v = _values(self)
        dt = self.dtype if self.dtype.is_floating_point else int64
        r = v.sum(axis=dim, keepdims=keepdim) if dim is not None else v.sum()
        return Tensor(_cast_store(np.asarray(r), dt), dt, self.device)

    def mean(self, dim=None, keepdim=False):
        v = _values(self).astype(np.float64)
        r = v.mean(axis=dim, keepdims=keepdim) if dim is not None else v.mean()
        dt = self.dtype if self.dtype.is_floating_point else _STATE["default_dtype"]
        return Tensor(_cast_store(np.asarray(r), dt), dt, self.device)

    def all(self, dim=None):
        v = _values(self).astype(bool)
        r = v.all(axis=dim) if dim is not None else v.all()
        return Tensor(np.asarray(r, dtype=np.bool_), bool_, self.device)

    def any(self, dim=None):
        v = _values(self).astype(bool)
        r = v.any(axis=dim) if dim is not None else v.any()
        return Tensor(np.asarray(r, dtype=np.bool_), bool_, self.device)


# --- creation functions ---------------------------------------------------------------------------------------------
def _shape(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(size[0])
    return tuple(size)


def empty(*size, dtype=None, device=None):
    dev, dt = _resolve(device, dtype, float_result=dtype is None)
    dt = dt or _STATE["default_dtype"]
    shp = _shape(size)
    if dt.is_floating_point and dt.kind != "fp8":
        arr = np.full(shp, np.nan, dtype=dt.store)  # uninitialised memory must not be read as zeros
    else:
        arr = np.full(shp, -7777 if dt.kind in ("i",) else 0, dtype=dt.store)
    return Tensor(arr, dt, dev)


def zeros(*size, dtype=None, device=None):
    dev, dt = _resolve(device, dtype, float_result=dtype is None)
    dt = dt or _STATE["default_dtype"]
    return Tensor(np.zeros(_shape(size), dtype=dt.store), dt, dev)


def ones(*size, dtype=None, device=None):
    dev, dt = _resolve(device, dtype, float_result=dtype is None)
    dt = dt or _STATE["default_dtype"]
    return Tensor(_cast_store(np.ones(_shape(size)), dt), dt, dev)


def full(size, fill_value, dtype=None, device=None):
    if dtype is None:
        dtype = int64 if isinstance(fill_value, int) and not isinstance(fill_value, bool) else (
            bool_ if isinstance(fill_value, bool) else None)
    dev, dt = _resolve(device, dtype, float_result=dtype is None)
    dt = dt or _STATE["default_dtype"]
    return Tensor(_cast_store(np.full(tuple(size), fill_value), dt), dt, dev)


def zeros_like(t, dtype=None):
    return Tensor(np.zeros(t.shape, dtype=(dtype or t.dtype).store), dtype or t.dtype, t.device)


def arange(*args, dtype=None, device=None):
    dev, dt = _resolve(device, dtype or int64, float_result=False)
    return Tensor(np.arange(*args).astype(dt.store), dt, dev)


def tensor(data, dtype=None, device=None):
    arr = np.asarray(data)
    if dtype is None:
        dtype = float32 if arr.dtype.kind == "f" else (bool_ if arr.dtype.kind == "b" else int64)
    dev, dt = _resolve(device, dtype, float_result=False)
    return Tensor(_cast_store(arr, dt), dt, dev)


def clamp(t, min=None, max=None):
    v = _values(t)
    r = np.clip(v, min, max)
    dt = t.dtype
    return Tensor(_cast_store(r, dt), dt, t.device)


def where(cond, a, b):
    av = _values(a) if isinstance(a, Tensor) else a
    bv = _values(b) if isinstance(b, Tensor) else b
    dt = _promote(a, b) if (isinstance(a, Tensor) or isinstance(b, Tensor)) else float32
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        dt = _promote(a, b)
    dev = cond.device
    return Tensor(_cast_store(np.where(cond.arr, av, bv), dt), dt, dev)


def repeat_interleave(t, repeats):
    r = repeats.arr if isinstance(repeats, Tensor) else repeats
    return Tensor(np.repeat(t.arr, r), t.dtype, t.device)


def exp(t):
    return Tensor(_cast_store(np.exp(_values(t).astype(np.float64)), t.dtype), t.dtype, t.device)


def softmax(t, dim=-1):
    v = _values(t).astype(np.float64)
    with np.errstate(all="ignore"):
        e = np.exp(v - v.max(axis=dim, keepdims=True))
        r = e / e.sum(axis=dim, keepdims=True)
    return Tensor(_cast_store(r, t.dtype), t.dtype, t.device)


def einsum(eq, *ops):
    r = np.einsum(eq, *[_values(o).astype(np.float64) for o in ops])
    dt = ops[0].dtype
    return Tensor(_cast_store(r, dt), dt, ops[0].device)


def isfinite(t):
    return Tensor(np.isfinite(_values(t)), bool_, t.device)


def nan_to_num(t, nan=0.0):
    return Tensor(_cast_store(np.nan_to_num(_values(t), nan=nan), t.dtype), t.dtype, t.device)


class Generator:
    def __init__(self, device="cpu"):
        self.device = _Device(device)
        self.rng = np.random.default_rng(0)

    def manual_seed(self, s):
        self.rng = np.random.default_rng(int(s))
        return self


def randn(*size, generator=None, dtype=None, device=None):
    dev, dt = _resolve(device, dtype, float_result=dtype is None)
    dt = dt or _STATE["default_dtype"]
    assert generator is not None and generator.device == dev, "[fake torch] generator/device mismatch"
    return Tensor(_cast_store(generator.rng.standard_normal(_shape(size)), dt), dt, dev)


def randint(low, high, size, generator=None, dtype=None, device=None):
    dev, dt = _resolve(device, dtype or int64, float_result=False)
    assert generator is not None and generator.device == dev, "[fake torch] generator/device mismatch"
    return Tensor(generator.rng.integers(low, high, size=tuple(size)).astype(dt.store), dt, dev)


def randperm(n, generator=None, device=None):
    dev, dt = _resolve(device, int64, float_result=False)
    assert generator is not None and generator.device == dev, "[fake torch] generator/device mismatch"
    return Tensor(generator.rng.permutation(n).astype(np.int64), int64, dev)


def get_default_dtype():
    return _STATE["default_dtype"]


def set_default_dtype(dt):
    _STATE["default_dtype"] = dt


def make_torch() -> types.ModuleType:
    m = types.ModuleType("torch")
    for d in _ALL.values():
        setattr(m, d.name, d)
    m.dtype, m.Tensor, m.device, m.Generator = _DType, Tensor, _Device, Generator
    m.empty, m.zeros, m.ones, m.full, m.zeros_like, m.arange, m.tensor = empty, zeros, ones, full, zeros_like, arange, tensor
    m.clamp, m.where, m.repeat_interleave, m.exp, m.softmax, m.einsum = clamp, where, repeat_interleave, exp, softmax, einsum
    m.isfinite, m.nan_to_num, m.randn, m.randint, m.randperm = isfinite, nan_to_num, randn, randint, randperm
    m.get_default_dtype, m.set_default_dtype = get_default_dtype, set_default_dtype
    cuda = types.ModuleType("torch.cuda")
    cuda.is_current_stream_capturing = lambda: bool(_STATE["capturing"])
    cuda.device = lambda dev: contextlib.nullcontext()
    m.cuda = cuda
    m.__is_fake_torch__ = True
    return m


# ---------------------------------------------------------------------------------------------------------------------
# fake triton
# ---------------------------------------------------------------------------------------------------------------------
class FakeKernel:
    """What @triton.jit returns: indexable by a grid, launch is recorded and (optionally) emulated."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "kernel")
        self.launches: list = []
        self.emulator = None

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.launches.append((grid, args, kw))
            if self.emulator is not None:
                return self.emulator(grid, *args, **kw)
            raise RuntimeError(f"[fake triton] kernel {self.__name__} launched without an emulator")

        return launch


def make_triton():
    triton = types.ModuleType("triton")

    def jit(fn=None, **kw):
        if fn is None:
            return lambda f: FakeKernel(f)
        return FakeKernel(fn)

    triton.jit = jit
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (int(n) - 1).bit_length()
    tl = types.ModuleType("triton.language")
    tl.constexpr = type("constexpr", (), {})
    triton.language = tl
    return triton, tl


# ---------------------------------------------------------------------------------------------------------------------
# stubbed vllm namespace + loaders
# ---------------------------------------------------------------------------------------------------------------------
_STUB_MODULES = [
    "torch", "torch.cuda", "triton", "triton.language", "vllm", "vllm.config", "vllm.config.cache",
    "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.attention",
    "vllm.model_executor.layers.attention.sparse_mla_attention", "vllm.platforms", "vllm.platforms.interface",
    "vllm.triton_utils", "vllm.utils", "vllm.utils.flashinfer", "vllm.v1", "vllm.v1.attention",
    "vllm.v1.attention.backend", "vllm.v1.attention.backends", "vllm.v1.attention.backends.mla",
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse", "vllm.v1.attention.backends.mla.sparse_utils",
    "vllm.v1.kv_cache_interface", "vllm._gb10_sparse_mla_prefill",
]


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    return m


@dataclasses.dataclass(kw_only=True)
class StubSparseMetadata:
    """FlashInferMLASparseMetadata stand-in: the SparseMLACommonMetadata fields the SM90 code reads."""

    num_reqs: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    num_actual_tokens: int = 0
    query_start_loc: object = None
    slot_mapping: object = None
    block_table: object = None
    req_id_per_token: object = None
    seq_lens: object = None
    block_size: int = 64
    topk_tokens: int = 2048
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill: object = None


def split_decodes_and_prefills(cam, threshold: int):
    """vllm/v1/attention/backends/utils.py:803 (treat_short_extends_as_decodes=True, not uniform)."""
    num_reqs, num_tokens = int(cam.num_reqs), int(cam.num_actual_tokens)
    if int(cam.max_query_len) <= threshold:
        return num_reqs, 0, num_tokens
    qsl = cam.query_start_loc_cpu
    ql = (qsl[1:] - qsl[:-1]).tolist()  # tensor-agnostic (fake or real torch)
    if int(ql[0]) > threshold:
        return 0, num_reqs, 0
    first = int(np.argmax(np.asarray(ql) > threshold))
    return first, num_reqs - first, int(qsl[first])


class StubSparseBuilder:
    """FlashInferMLASparseMetadataBuilder stand-in: real build() semantics for what the SM90 hooks read."""

    metadata_cls = StubSparseMetadata
    reorder_batch_threshold = 128  # {8 heads/rank: 128}
    super_builds = 0

    def __init__(self, kv_cache_spec=None, layer_names=None, vllm_config=None, device=None):
        pass

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        nd, npf, ndt = split_decodes_and_prefills(cam, self.reorder_batch_threshold)
        self.super_builds = self.super_builds + 1
        n = int(cam.num_actual_tokens)
        return self.metadata_cls(
            num_reqs=int(cam.num_reqs), max_query_len=int(cam.max_query_len), max_seq_len=int(cam.max_seq_len),
            num_actual_tokens=n, query_start_loc=cam.query_start_loc, slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor, req_id_per_token=cam.req_id_per_token, seq_lens=cam.seq_lens,
            num_decodes=nd, num_prefills=npf, num_decode_tokens=ndt,
        )


class StubSparseImpl:
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype,
                 logits_soft_cap, attn_type, kv_sharing_target_layer_name, indexer=None,
                 topk_indices_buffer=None, **mla_args):
        self.num_heads, self.head_size, self.scale = num_heads, head_size, float(scale)
        self.num_kv_heads, self.kv_cache_dtype = num_kv_heads, kv_cache_dtype
        self.indexer, self.topk_indices_buffer = indexer, topk_indices_buffer
        self.kv_lora_rank = mla_args["kv_lora_rank"]
        self.qk_rope_head_dim = mla_args["qk_rope_head_dim"]


def stub_flat_kv_row_view(kv_cache, block_size):
    """vllm.v1.attention.backends.mla.sparse_utils.flat_kv_row_view on the fake tensor."""
    num_blocks, _, head_dim = kv_cache.shape
    assert kv_cache.stride(0) % head_dim == 0
    block_stride_rows = kv_cache.stride(0) // head_dim
    num_rows = (num_blocks - 1) * block_stride_rows + block_size
    return kv_cache.as_strided((num_rows, head_dim), (head_dim, 1)), block_stride_rows


def stub_convert(req_id, block_table, token_indices, BLOCK_SIZE=64, BLOCK_STRIDE_ROWS=None,
                 NUM_TOPK_TOKENS=2048, return_valid_counts=False, **_):
    """triton_convert_req_index_to_global_index stand-in: the harness writes GLOBAL slot ids with a -1 tail
    into the top-k buffer, so the conversion is a copy that moves the valid ones to the front (the real
    kernel's COMPACT_TO_FRONT) and counts them."""
    src = token_indices.arr
    out = np.full_like(src, -1)
    cnt = np.zeros(src.shape[0], dtype=np.int32)
    for r in range(src.shape[0]):
        v = src[r][src[r] >= 0]
        out[r, : len(v)] = v
        cnt[r] = len(v)
    slots = Tensor(out, int32, token_indices.device)
    if return_valid_counts:
        return slots, Tensor(cnt, int32, token_indices.device)
    return slots


@dataclasses.dataclass
class Env:
    """One loaded namespace: the fake torch / triton and the exec'd backend module(s)."""

    torch: types.ModuleType
    triton: types.ModuleType
    tl: types.ModuleType
    modules: dict


def install_stubs() -> Env:
    """(Re)install the fake torch / triton / vllm stubs into sys.modules. Idempotent per call."""
    for n in _STUB_MODULES:
        sys.modules.pop(n, None)
    _STATE["device_stack"].clear()
    _STATE.update(default_dtype=float32, strict=False, capturing=False)
    t = make_torch()
    sys.modules["torch"] = t
    sys.modules["torch.cuda"] = t.cuda
    triton, tl = make_triton()
    sys.modules["triton"], sys.modules["triton.language"] = triton, tl
    _mod("vllm")
    _mod("vllm.config", VllmConfig=object, get_current_vllm_config=lambda: None)
    _mod("vllm.config.cache", CacheDType=str)
    _mod("vllm.model_executor")
    _mod("vllm.model_executor.layers")
    _mod("vllm.model_executor.layers.attention")
    _mod("vllm.model_executor.layers.attention.sparse_mla_attention", SparseMLACommonImpl=StubSparseImpl)
    _mod("vllm.platforms")
    _mod("vllm.platforms.interface", DeviceCapability=type("DeviceCapability", (), {}))
    _mod("vllm.triton_utils", tl=tl, triton=triton)
    _mod("vllm.utils")
    _mod("vllm.utils.flashinfer", has_flashinfer_sm90_nope_mla=lambda: True)
    _mod("vllm.v1")
    _mod("vllm.v1.attention")
    _mod("vllm.v1.attention.backend", AttentionBackend=type("AttentionBackend", (), {}),
         AttentionLayer=type("AttentionLayer", (), {}), CommonAttentionMetadata=type("CommonAttentionMetadata", (), {}),
         MLAAttentionImpl=type("MLAAttentionImpl", (), {}), MultipleOf=type("MultipleOf", (), {}))
    _mod("vllm.v1.attention.backends")
    _mod("vllm.v1.attention.backends.mla")
    _mod("vllm.v1.attention.backends.mla.flashinfer_mla_sparse", FlashInferMLASparseMetadata=StubSparseMetadata,
         FlashInferMLASparseMetadataBuilder=StubSparseBuilder)
    _mod("vllm.v1.attention.backends.mla.sparse_utils", align_blocks_to_rows=lambda spec, rows=1: spec,
         flat_kv_row_view=stub_flat_kv_row_view, triton_convert_req_index_to_global_index=stub_convert)
    _mod("vllm.v1.kv_cache_interface", AttentionSpec=type("AttentionSpec", (), {}))
    return Env(t, triton, tl, {})


def load_file(env: Env, name: str, path: Path, install_as: str | None = None) -> types.ModuleType:
    """Exec a source file as module `name` (kept in env.modules); optionally also register it in sys.modules
    as `install_as` (the patched backend does `from vllm import _gb10_sparse_mla_prefill`)."""
    src = Path(path).read_text()
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod  # dataclasses (3.14) resolves string annotations through sys.modules[cls.__module__]
    if install_as:
        sys.modules[install_as] = mod
        parent, _, child = install_as.rpartition(".")
        setattr(sys.modules[parent], child, mod)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    env.modules[name] = mod
    return mod


def load_source(env: Env, name: str, source: str, filename: str = "<memory>") -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    exec(compile(source, filename, "exec"), mod.__dict__)
    env.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------------------------------------------------
# fakes at the FlashInfer seam + batch generator
# ---------------------------------------------------------------------------------------------------------------------
class FakeWrapper:
    def __init__(self, log):
        self.log = log

    def run(self, q_nope, q_pe, ckv, kpe, **kw):
        self.log.append(("run", q_nope.shape, q_nope.data_ptr(), q_pe.shape, ckv.shape, str(ckv.dtype),
                         ckv.data_ptr(), kpe.shape, tuple(sorted(kw.items()))))
        return Tensor(np.full(q_nope.shape, 7.0, dtype=np.float32), bfloat16, q_nope.device)


class FakeState:
    """_SM90State stand-in: records plan / pack_indices / wrapper.run; carries what _kv_lens_host reads."""

    def __init__(self, index_topk=2048, index_kpool=4, width=2176):
        self.index_topk = index_topk
        self.index_kpool = max(index_kpool, 1)
        self.max_valid = min(index_topk + self.index_kpool - 1, width)
        self.log: list = []
        self.wrapper = FakeWrapper(self.log)

    def plan(self, n, kv_lens, cam, req_id_per_token):
        self.log.append(("plan", int(n), [int(x) for x in kv_lens.tolist()], cam is not None,
                         None if req_id_per_token is None else req_id_per_token.data_ptr()))

    def pack_indices(self, topk_slots):
        # content hash instead of the pointer: the converted slots are a fresh allocation on every call
        h = hashlib.sha1(np.ascontiguousarray(topk_slots.arr).tobytes()).hexdigest()[:12]
        self.log.append(("pack", topk_slots.shape, h))


class FakeLayer:
    """MLAAttention stand-in as the builder sees it (`_use_sparse_mha`) and the impl sees it (_k_scale_float)."""

    def __init__(self, use_sparse_mha=False, k_scale=0.5):
        self.use_sparse_mha = use_sparse_mha
        self._k_scale_float = k_scale

    def _use_sparse_mha(self, metadata):
        return self.use_sparse_mha


THRESH = 128
K_MAX = 3


def make_cam(env: Env, reqs: list[dict], with_host_bound=True, req_dev="cpu"):
    """reqs: [{q, gpu_nc, cpu_nc}] in batch order (decode/draft rows first, as the V2 runner sorts them).
    Returns a CommonAttentionMetadata stand-in with the exact device lengths (gpu) and the host upper bound
    (cpu_nc >= gpu_nc: optimistic on draft rows)."""
    torch = env.torch
    qsl = [0]
    for r in reqs:
        qsl.append(qsl[-1] + r["q"])
    cam = SimpleNamespace()
    cam.num_reqs = len(reqs)
    cam.query_start_loc_cpu = torch.tensor(qsl, dtype=torch.int32)
    cam.query_start_loc = cam.query_start_loc_cpu
    cam.num_actual_tokens = qsl[-1]
    cam.max_query_len = max(r["q"] for r in reqs)
    cam.max_seq_len = max(r["gpu_nc"] + r["q"] for r in reqs)
    cam.seq_lens = torch.tensor([r["gpu_nc"] + r["q"] for r in reqs], dtype=torch.int32)
    cam.seq_lens_cpu_upper_bound = (
        torch.tensor([r["cpu_nc"] + r["q"] for r in reqs], dtype=torch.int32) if with_host_bound else None
    )
    cam.slot_mapping = None
    cam.block_table_tensor = None
    cam.req_id_per_token = torch.repeat_interleave(
        torch.arange(len(reqs), dtype=torch.int32), torch.tensor([r["q"] for r in reqs], dtype=torch.int64))
    return cam


def random_batch(rng: random.Random, pure_prefill: bool | None = None, max_model_len: int = 262144) -> list[dict]:
    """A V2-style sorted batch: draft/decode rows first, then prefill rows (query_len > THRESH)."""
    n = rng.randint(1, 8)
    pp = rng.random() < 0.3 if pure_prefill is None else pure_prefill
    n_dec = 0 if pp else rng.randint(0, n)
    reqs = []
    budget = 8192
    for i in range(n):
        gpu_nc = rng.randint(0, 200000)
        if i < n_dec:
            kind = rng.choice(["draft", "plain", "short"])
            if kind == "draft":
                d = rng.randint(1, K_MAX)
                q, cpu_nc = 1 + d, gpu_nc + rng.randint(0, d)
            elif kind == "plain":
                q, cpu_nc = 1, gpu_nc
            else:
                q, cpu_nc = rng.randint(2, THRESH), gpu_nc
        else:
            q, cpu_nc = rng.randint(THRESH + 1, 4096), gpu_nc
        if q > budget:
            break
        budget -= q
        if cpu_nc + q >= max_model_len:
            q = max_model_len - cpu_nc - 1
            if q < 1:
                continue
        reqs.append(dict(q=q, gpu_nc=gpu_nc, cpu_nc=cpu_nc))
    return reqs


def exact_lens(reqs: list[dict], topk=2048, kpool=4) -> list[int]:
    """Independent oracle: the exact valid count of every row (device truth), from first principles."""
    out = []
    for r in reqs:
        sl = r["gpu_nc"] + r["q"]
        for j in range(r["q"]):
            ctx = sl - r["q"] + j + 1
            out.append(ctx if ctx <= topk else topk + ctx % kpool)
    return out


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def attach(self, name: str):
        lg = logging.getLogger(name)
        lg.addHandler(self)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        return self
