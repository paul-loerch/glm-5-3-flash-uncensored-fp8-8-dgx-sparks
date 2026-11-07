# SPDX-License-Identifier: Apache-2.0
"""[patch-032 gb10-repl-embed] test library (Mac / CPU: python3 >= 3.10 + numpy; NO torch, NO vLLM install).

  * a numpy-backed fake `torch` with just the surface the REAL image files (vocab_parallel_embedding.py, model.py,
    mtp.py, spec_decode/eagle/utils.py, model_executor/utils.py, the loader's process_weights_after_loading) and the
    patch module use: bf16 stored as uint16 bit patterns (round-to-nearest-even, -0.0 / subnormals / NaN kept),
    byte-level .view(dtype), views that share memory (slices, narrow, view, unsqueeze), F.embedding with a bounds check
    that raises like a device-side assert, nn.Module / Parameter / ModuleList / ModuleDict / Linear;
  * a CUDA-graph emulation: while `torch.cuda.graph(g)` is open every op appends a replay closure to a tape, and
    g.replay() re-runs the tape INTO THE SAME output buffers from the current input buffers (what a replay does);
  * an N-thread rank world: every thread is one TP rank; collectives rendezvous by call index, every rank must issue the
    same (op, shape, dtype) at the same index (else ScheduleMismatch on all ranks), a missing rank is a HANG
    (TimeoutError after a few seconds). all_reduce arithmetic as the GPU computes it: patch 012's one-shot (bf16,
    <= oneshot_max elements: fp32 acc = +0.0, acc += x_r + 0.0 in rank order, one bf16 rounding) or an NCCL-like ring
    (bf16 partial sums rounded per hop, chunk-dependent start rank; optional flush-to-zero of subnormals);
  * a stub vLLM namespace into which the REAL files are exec'd (build_ns, in p032_cpu_check.py).

vLLM v0.32.0 port: this library is unchanged from the day-0 version (the fake torch already covers everything the v0.32.0
files need); the v0.32.0-specific stand-ins (the fused `ops.vocab_parallel_embedding` kernel, the new loader steps, the
transformers / deepseek_mtp / interfaces stubs) live in p032_cpu_check.build_ns.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.abc
import importlib.machinery
import logging
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# bf16 helpers
# ---------------------------------------------------------------------------------------------------------------------


def f32_to_bf16_bits(a) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint32)
    nan = np.isnan(a)
    r = np.where(nan, ((u >> 16) | 0x40).astype(np.uint32), r)
    return r.astype(np.uint16)


def bf16_bits_to_f32(b) -> np.ndarray:
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def ftz_f32(x: np.ndarray) -> np.ndarray:
    """Flush fp32 subnormals to a zero of the same sign."""
    x = np.asarray(x, dtype=np.float32)
    sub = (x != 0) & (np.abs(x) < np.float32(2.0 ** -126))
    return np.where(sub, np.copysign(np.float32(0.0), x), x).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# dtypes / devices
# ---------------------------------------------------------------------------------------------------------------------
class dtype:
    def __init__(self, name, store, itemsize, kind):
        self.name, self.store, self.itemsize, self.kind = name, np.dtype(store), itemsize, kind

    @property
    def is_floating_point(self):
        return self.kind in ("f", "bf16", "fp8")

    def __repr__(self):
        return f"torch.{self.name}"

    __str__ = __repr__


float32 = dtype("float32", np.float32, 4, "f")
float64 = dtype("float64", np.float64, 8, "f")
float16 = dtype("float16", np.float16, 2, "f")
bfloat16 = dtype("bfloat16", np.uint16, 2, "bf16")
float8_e4m3fn = dtype("float8_e4m3fn", np.uint8, 1, "fp8")
int8 = dtype("int8", np.int8, 1, "i")
uint8 = dtype("uint8", np.uint8, 1, "i")
int16 = dtype("int16", np.int16, 2, "i")
int32 = dtype("int32", np.int32, 4, "i")
int64 = dtype("int64", np.int64, 8, "i")
bool_ = dtype("bool", np.bool_, 1, "b")


class finfo:
    def __init__(self, dt):
        if dt is bfloat16:
            self.tiny, self.max, self.eps = 2.0 ** -126, float(bf16_bits_to_f32(np.uint16(0x7F7F))), 2.0 ** -7
        elif dt is float32:
            self.tiny, self.max, self.eps = 2.0 ** -126, float(np.finfo(np.float32).max), 2.0 ** -23
        elif dt is float16:
            self.tiny, self.max, self.eps = 2.0 ** -14, 65504.0, 2.0 ** -10
        else:
            raise TypeError(f"finfo: {dt}")
        self.min = -self.max
        self.smallest_normal = self.tiny


class device:
    def __init__(self, spec="cpu", index=None):
        if isinstance(spec, device):
            self.type, self.index = spec.type, spec.index
            return
        s = str(spec)
        if ":" in s:
            t, i = s.split(":")
            self.type, self.index = t, int(i)
        else:
            self.type, self.index = s, (index if index is not None else (0 if s == "cuda" else None))

    def __eq__(self, o):
        try:
            o = device(o) if not isinstance(o, device) else o
        except Exception:  # noqa: BLE001
            return False
        return self.type == o.type and (self.index or 0) == (o.index or 0)

    def __hash__(self):
        return hash((self.type, self.index or 0))

    def __repr__(self):
        return f"device(type='{self.type}'" + (f", index={self.index})" if self.index is not None else ")")

    def __str__(self):
        return self.type + (f":{self.index}" if self.index is not None else "")

    # `with torch.device("cuda"):` sets the default device of new tensors (thread-local, as torch's mode stack)
    def __enter__(self):
        _TLS.dev_stack = getattr(_TLS, "dev_stack", []) + [self]
        return self

    def __exit__(self, *e):
        _TLS.dev_stack = _TLS.dev_stack[:-1]
        return False


_TLS = threading.local()
INJ = threading.local()          # per-rank-thread fault injection flags read by the instrumented module copy
_DEFAULT = {"dtype": float32}


def _default_device():
    st = getattr(_TLS, "dev_stack", [])
    return st[-1] if st else device("cpu")


def get_default_dtype():
    return _DEFAULT["dtype"]


def set_default_dtype(d):
    _DEFAULT["dtype"] = d


class Size(tuple):
    def numel(self):
        return int(np.prod(self)) if len(self) else 1


# ---------------------------------------------------------------------------------------------------------------------
# CUDA-graph tape
# ---------------------------------------------------------------------------------------------------------------------
def _tape():
    return getattr(_TLS, "tape", None)


def _decode(a, dt):
    if dt is bfloat16:
        return bf16_bits_to_f32(a)
    if dt is float16:
        return np.asarray(a).astype(np.float32)
    return np.asarray(a)


def _encode(v, dt):
    v = np.asarray(v)
    if dt is bfloat16:
        return f32_to_bf16_bits(v.astype(np.float32))
    if dt.kind == "i" and v.dtype.kind == "f":
        v = np.trunc(v)
    return v.astype(dt.store)


def _result_dtype(a, b):
    """torch-like promotion for the ops these tests need. a: Tensor; b: Tensor or Python scalar."""
    if isinstance(b, Tensor):
        if a.dtype is b.dtype:
            return a.dtype
        if a.dtype.kind == "b":
            return b.dtype
        if b.dtype.kind == "b":
            return a.dtype
        if a.dtype.is_floating_point or b.dtype.is_floating_point:
            if a.dtype.is_floating_point and b.dtype.is_floating_point:
                return float32 if a.dtype.itemsize == b.dtype.itemsize else (a.dtype if a.dtype.itemsize >
                                                                               b.dtype.itemsize else b.dtype)
            return a.dtype if a.dtype.is_floating_point else b.dtype
        return a.dtype if a.dtype.itemsize >= b.dtype.itemsize else b.dtype
    if isinstance(b, bool):
        return a.dtype
    if isinstance(b, int):
        return int64 if a.dtype.kind == "b" else a.dtype
    if isinstance(b, float):
        return a.dtype if a.dtype.is_floating_point else float32
    raise TypeError(f"unsupported operand {type(b)}")


def _np_index(idx):
    if isinstance(idx, Tensor):
        return idx._a
    if isinstance(idx, tuple):
        return tuple(_np_index(i) for i in idx)
    return idx


class Tensor:
    __array_priority__ = 1000

    def __init__(self, a, dt, dev=None, requires_grad=False):
        self._a = a
        self.dtype = dt
        self.device = device(dev) if dev is not None else device("cpu")
        self.requires_grad = requires_grad

    # ---- properties
    @property
    def shape(self):
        return Size(self._a.shape)

    def size(self, d=None):
        return Size(self._a.shape) if d is None else self._a.shape[d]

    def dim(self):
        return self._a.ndim

    @property
    def ndim(self):
        return self._a.ndim

    def numel(self):
        return int(self._a.size)

    def element_size(self):
        return self.dtype.itemsize

    def is_contiguous(self):
        return bool(self._a.flags["C_CONTIGUOUS"])

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def data(self):
        return Tensor(self._a, self.dtype, self.device)

    @data.setter
    def data(self, v):
        self._a, self.dtype, self.device = v._a, v.dtype, v.device

    def data_ptr(self):
        return self._a.__array_interface__["data"][0]

    def untyped_storage(self):
        base = self._a
        while base.base is not None and isinstance(base.base, np.ndarray):
            base = base.base
        return NS(data_ptr=lambda: base.__array_interface__["data"][0], nbytes=lambda: base.nbytes)

    def __len__(self):
        return self._a.shape[0]

    def __repr__(self):
        return f"FakeTensor(shape={tuple(self._a.shape)}, dtype={self.dtype}, device={self.device})"

    # ---- op plumbing: every new result is recorded on an open graph tape
    def _wrap(self, a, dt):
        return Tensor(np.ascontiguousarray(a) if not a.flags["C_CONTIGUOUS"] else a, dt, self.device)

    @staticmethod
    def _op(fn, args, dt, dev):
        def run():
            return fn(*[x._a if isinstance(x, Tensor) else x for x in args])
        out = Tensor(np.array(run(), dtype=dt.store, copy=True), dt, dev)
        tp = _tape()
        if tp is not None:
            def replay(out=out):
                out._a[...] = run()
            tp.append((replay, out))
        return out

    def _inplace(self, fn):
        """fn(arr) mutates arr in place; recorded for replay."""
        fn(self._a)
        tp = _tape()
        if tp is not None:
            tp.append((lambda s=self: fn(s._a), self))
        return self

    # ---- conversion
    def to(self, *args, **kw):
        dt, dev = kw.get("dtype"), kw.get("device")
        for a in args:
            if isinstance(a, dtype):
                dt = a
            elif isinstance(a, (str, device)):
                dev = a
        out = self
        if dt is not None and dt is not self.dtype:
            src = self.dtype

            def conv(a, src=src, dt=dt):
                return _encode(_decode(a, src), dt)
            out = Tensor._op(conv, [self], dt, self.device)
        if dev is not None and device(dev) != out.device:
            out = Tensor(out._a.copy(), out.dtype, dev)
        return out

    def float(self):
        return self.to(float32)

    def bfloat16(self):
        return self.to(bfloat16)

    def long(self):
        return self.to(int64)

    def int(self):
        return self.to(int32)

    def bool(self):
        return self.to(bool_)

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self):
        return self.to(device="cuda:0")

    def tolist(self):
        return _decode(self._a, self.dtype).tolist()

    def item(self):
        if self._a.size != 1:
            raise RuntimeError("a Tensor with more than one element cannot be converted to Scalar")
        v = _decode(self._a, self.dtype).reshape(-1)[0]
        return v.item() if hasattr(v, "item") else v

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def __bool__(self):
        return bool(self.item())

    def __index__(self):
        return int(self.item())

    # ---- views (share memory) / copies
    def contiguous(self):
        return self if self.is_contiguous() else Tensor._op(lambda a: np.ascontiguousarray(a), [self], self.dtype,
                                                            self.device)

    def clone(self):
        return Tensor._op(lambda a: a.copy(), [self], self.dtype, self.device)

    def detach(self):
        return self

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], dtype):
            nd = shape[0]
            if not self._a.flags["C_CONTIGUOUS"]:
                raise RuntimeError("view(dtype) needs a contiguous tensor in this fake")
            if nd.itemsize != self.dtype.itemsize:
                raise RuntimeError("view(dtype) with a different element size is not supported in this fake")
            return Tensor(self._a.view(nd.store), nd, self.device)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, Size)):
            shape = tuple(shape[0])
        if not self._a.flags["C_CONTIGUOUS"]:
            raise RuntimeError("view size is not compatible with input tensor's size and stride")
        return Tensor(self._a.reshape(shape), self.dtype, self.device)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, Size)):
            shape = tuple(shape[0])
        if self._a.flags["C_CONTIGUOUS"]:
            return Tensor(self._a.reshape(shape), self.dtype, self.device)
        return Tensor._op(lambda a: a.reshape(shape), [self], self.dtype, self.device)

    def flatten(self):
        return self.reshape(-1)

    def unsqueeze(self, d):
        return Tensor(np.expand_dims(self._a, d), self.dtype, self.device)

    def narrow(self, d, start, length):
        sl = [slice(None)] * self._a.ndim
        sl[d] = slice(start, start + length)
        return Tensor(self._a[tuple(sl)], self.dtype, self.device)

    def __getitem__(self, idx):
        i = _np_index(idx)
        r = self._a[i]
        if isinstance(r, np.ndarray) and (np.may_share_memory(r, self._a) or r.size == 0):
            return Tensor(r, self.dtype, self.device)          # basic slicing: a view
        return Tensor._op(lambda a, i=i: a[i], [self], self.dtype, self.device)   # advanced indexing: a copy

    def __setitem__(self, idx, value):
        i = _np_index(idx)
        if isinstance(value, Tensor):
            v = value._a if value.dtype is self.dtype else _encode(_decode(value._a, value.dtype), self.dtype)
        else:
            v = _encode(np.asarray(value, dtype=np.float64 if self.dtype.is_floating_point else None), self.dtype)
        self._a[i] = v

    def copy_(self, src, non_blocking=False):
        sdt = src.dtype

        def f(a, src=src, sdt=sdt, me=self):
            v = src._a if sdt is me.dtype else _encode(_decode(src._a, sdt), me.dtype)
            a[...] = np.broadcast_to(v, a.shape)
        return self._inplace(f)

    def fill_(self, val):
        enc = _encode(np.asarray(val, dtype=np.float64), self.dtype) if self.dtype.is_floating_point else val

        def f(a, enc=enc):
            a[...] = enc
        return self._inplace(f)

    def zero_(self):
        return self.fill_(0)

    def masked_fill_(self, mask, value):
        enc = (_encode(np.asarray(value, dtype=np.float64), self.dtype) if self.dtype.is_floating_point
               else np.asarray(value).astype(self.dtype.store))

        def f(a, mask=mask, enc=enc):
            m = np.broadcast_to(mask._a, a.shape)
            a[m] = enc
        return self._inplace(f)

    def masked_fill(self, mask, value):
        return self.clone().masked_fill_(mask, value)

    # ---- elementwise
    def _bin(self, o, npfn, cmp=False, logical=False, reverse=False):
        dt = _result_dtype(self, o)
        odt = o.dtype if isinstance(o, Tensor) else None
        sdt = self.dtype

        def fn(a, b, sdt=sdt, odt=odt, dt=dt):
            if logical:
                r = npfn(a, b)
                return r
            if dt.kind in ("bf16",) or dt is float16 or sdt.kind == "bf16" or (odt is not None and odt.kind == "bf16"):
                x = _decode(a, sdt).astype(np.float32)
                y = _decode(b, odt).astype(np.float32) if odt is not None else np.float32(b)
            else:
                x = a.astype(dt.store) if not cmp else a
                y = (b.astype(dt.store) if not cmp else b) if odt is not None else b
            r = npfn(y, x) if reverse else npfn(x, y)
            if cmp:
                return np.asarray(r, dtype=np.bool_)
            return _encode(r, dt) if dt.kind == "bf16" else np.asarray(r).astype(dt.store)
        return Tensor._op(fn, [self, o], bool_ if cmp else dt, self.device)

    def __add__(self, o):
        return self._bin(o, np.add)

    __radd__ = __add__

    def __sub__(self, o):
        return self._bin(o, np.subtract)

    def __rsub__(self, o):
        return self._bin(o, np.subtract, reverse=True)

    def __mul__(self, o):
        if self.dtype.kind == "b" and isinstance(o, Tensor) and o.dtype.kind == "b":
            return self._bin(o, np.logical_and, logical=True)
        return self._bin(o, np.multiply)

    __rmul__ = __mul__

    def __mod__(self, o):
        return self._bin(o, np.mod)

    def __and__(self, o):
        return self._bin(o, np.logical_and if self.dtype.kind == "b" else np.bitwise_and, logical=True)

    def __or__(self, o):
        return self._bin(o, np.logical_or if self.dtype.kind == "b" else np.bitwise_or, logical=True)

    def __invert__(self):
        if self.dtype.kind != "b":
            raise TypeError("~ on a non-bool tensor is not supported in this fake")
        return Tensor._op(np.logical_not, [self], bool_, self.device)

    def __neg__(self):
        return self._bin(-1, np.multiply)

    def __eq__(self, o):
        return self._bin(o, np.equal, cmp=True)

    def __ne__(self, o):
        return self._bin(o, np.not_equal, cmp=True)

    def __lt__(self, o):
        return self._bin(o, np.less, cmp=True)

    def __le__(self, o):
        return self._bin(o, np.less_equal, cmp=True)

    def __gt__(self, o):
        return self._bin(o, np.greater, cmp=True)

    def __ge__(self, o):
        return self._bin(o, np.greater_equal, cmp=True)

    __hash__ = object.__hash__

    def __iadd__(self, o):
        r = self + o
        return self._inplace(lambda a, r=r: a.__setitem__(Ellipsis, r._a))

    def abs(self):
        if self.dtype is bfloat16:
            return Tensor._op(lambda a: a & np.uint16(0x7FFF), [self], bfloat16, self.device)
        return Tensor._op(np.abs, [self], self.dtype, self.device)

    def sum(self, dim=None, keepdim=False, dtype=None):
        odt = dtype or (int64 if self.dtype.kind in ("b", "i") else self.dtype)
        sdt = self.dtype

        def fn(a, sdt=sdt, odt=odt):
            if odt.kind in ("i", "b"):
                r = a.astype(np.int64).sum(axis=dim, keepdims=keepdim)
            else:
                r = _decode(a, sdt).astype(np.float64).sum(axis=dim, keepdims=keepdim)
            return _encode(r, odt) if odt.kind == "bf16" else np.asarray(r).astype(odt.store)
        return Tensor._op(fn, [self], odt, self.device)

    def any(self, dim=None):
        return Tensor._op(lambda a: np.asarray(a.astype(np.bool_).any(axis=dim)), [self], bool_, self.device)

    def all(self, dim=None):
        return Tensor._op(lambda a: np.asarray(a.astype(np.bool_).all(axis=dim)), [self], bool_, self.device)

    def nonzero(self):
        return Tensor(np.argwhere(self._a).astype(np.int64), int64, self.device)


class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True):
        super().__init__(data._a, data.dtype, data.device, requires_grad)


# ---------------------------------------------------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------------------------------------------------
def _size_args(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list, Size)):
        return tuple(size[0])
    return tuple(int(s) for s in size)


def _poison(dt, shape):
    if dt is bfloat16:
        return np.full(shape, 0x7FC1, dtype=np.uint16)
    if dt.kind == "i":
        return np.full(shape, 0x5A, dtype=dt.store)
    if dt.kind == "b":
        return np.ones(shape, dtype=np.bool_)
    return np.full(shape, np.nan, dtype=dt.store)


def _dev(d):
    return device(d) if d is not None else _default_device()


def empty(*size, dtype=None, device=None, **kw):
    dt = dtype or get_default_dtype()
    return Tensor(_poison(dt, _size_args(size)), dt, _dev(device))


def zeros(*size, dtype=None, device=None, **kw):
    dt = dtype or get_default_dtype()
    return Tensor(np.zeros(_size_args(size), dtype=dt.store), dt, _dev(device))


def ones(*size, dtype=None, device=None):
    dt = dtype or get_default_dtype()
    return Tensor(_encode(np.ones(_size_args(size)), dt), dt, _dev(device))


def full(size, value, dtype=None, device=None):
    dt = dtype or get_default_dtype()
    return Tensor(_encode(np.full(_size_args((size,)), value, dtype=np.float64), dt), dt, _dev(device))


def zeros_like(t, dtype=None):
    return zeros(*t.shape, dtype=dtype or t.dtype, device=t.device)


def empty_like(t, dtype=None):
    return empty(*t.shape, dtype=dtype or t.dtype, device=t.device)


def tensor(data, dtype=None, device=None):
    if isinstance(data, Tensor):
        return data.clone()
    arr = np.asarray(data)
    dt = dtype or (int64 if arr.dtype.kind in "iu" else bool_ if arr.dtype.kind == "b" else get_default_dtype())
    return Tensor(_encode(arr, dt), dt, _dev(device))


def arange(*args, dtype=None, device=None):
    dt = dtype or int64
    return Tensor(np.arange(*args).astype(dt.store), dt, _dev(device))


def cat(ts, dim=0):
    ts = list(ts)
    return Tensor._op(lambda *a: np.concatenate(a, axis=dim), ts, ts[0].dtype, ts[0].device)


def stack(ts, dim=0):
    ts = list(ts)
    return Tensor._op(lambda *a: np.stack(a, axis=dim), ts, ts[0].dtype, ts[0].device)


def where(cond, a, b):
    dt = a.dtype if isinstance(a, Tensor) else b.dtype
    av = a if isinstance(a, Tensor) else full(cond.shape, a, dtype=dt, device=cond.device)
    bv = b if isinstance(b, Tensor) else full(cond.shape, b, dtype=dt, device=cond.device)
    return Tensor._op(lambda c, x, y: np.where(c, x, y), [cond, av, bv], dt, cond.device)


def signbit(t):
    if t.dtype is bfloat16:
        return Tensor._op(lambda a: (a >> 15).astype(np.bool_), [t], bool_, t.device)
    return Tensor._op(np.signbit, [t], bool_, t.device)


def isfinite(t):
    return Tensor._op(lambda a, dt=t.dtype: np.isfinite(_decode(a, dt)), [t], bool_, t.device)


def equal(a, b):
    return a.shape == b.shape and a.dtype is b.dtype and bool(np.array_equal(a._a, b._a))


# ---------------------------------------------------------------------------------------------------------------------
# nn
# ---------------------------------------------------------------------------------------------------------------------
class Module:
    def __init__(self, *a, **k):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", False)

    def __setattr__(self, k, v):
        if "_parameters" not in self.__dict__:
            object.__setattr__(self, "_parameters", {})
            object.__setattr__(self, "_modules", {})
        params, mods = self.__dict__["_parameters"], self.__dict__["_modules"]
        if isinstance(v, Parameter):
            mods.pop(k, None)
            self.__dict__.pop(k, None)
            params[k] = v
        elif isinstance(v, Module):
            params.pop(k, None)
            self.__dict__.pop(k, None)
            mods[k] = v
        else:
            if k in params and v is not None and not isinstance(v, Parameter):
                raise TypeError(f"cannot assign {type(v).__name__} as parameter '{k}' (torch.nn.Parameter or None "
                                "expected)")
            if k in params and v is None:
                params[k] = None
                return
            if k in mods and v is None:
                mods[k] = None
                return
            object.__setattr__(self, k, v)

    def __getattr__(self, k):
        d = self.__dict__
        if "_parameters" in d and k in d["_parameters"]:
            return d["_parameters"][k]
        if "_modules" in d and k in d["_modules"]:
            return d["_modules"][k]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{k}'")

    def __delattr__(self, k):
        d = self.__dict__
        if k in d.get("_parameters", {}):
            del d["_parameters"][k]
        elif k in d.get("_modules", {}):
            del d["_modules"][k]
        else:
            object.__delattr__(self, k)

    def register_parameter(self, name, p):
        self.__dict__["_parameters"][name] = p

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        for mp, m in (self.named_modules(prefix) if recurse else [(prefix, self)]):
            for n, p in m.__dict__.get("_parameters", {}).items():
                if p is None or (remove_duplicate and id(p) in seen):
                    continue
                seen.add(id(p))
                yield (f"{mp}.{n}" if mp else n), p

    def parameters(self, recurse=True):
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_children(self):
        for n, m in self.__dict__.get("_modules", {}).items():
            if m is not None:
                yield n, m

    def children(self):
        for _, m in self.named_children():
            yield m

    def named_modules(self, prefix="", memo=None):
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for n, m in self.named_children():
            yield from m.named_modules(f"{prefix}.{n}" if prefix else n, memo)

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def eval(self):
        return self

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class ModuleList(Module):
    def __init__(self, mods=None):
        super().__init__()
        object.__setattr__(self, "_list", [])
        for m in (mods or []):
            self.append(m)

    def append(self, m):
        i = len(self._list)
        self._list.append(m)
        self.__dict__["_modules"][str(i)] = m

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ModuleList(self._list[i])
        return self._list[i]

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)


class ModuleDict(Module):
    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self.__dict__["_modules"][k] = v

    def __getitem__(self, k):
        return self.__dict__["_modules"][k]

    def values(self):
        return list(self.__dict__["_modules"].values())

    def items(self):
        return list(self.__dict__["_modules"].items())

    def keys(self):
        return list(self.__dict__["_modules"].keys())


def linear(x, w, b=None):
    xd, wd = x.dtype, w.dtype

    def fn(a, ww, xd=xd, wd=wd):
        r = _decode(a, xd).astype(np.float32) @ _decode(ww, wd).astype(np.float32).T
        return _encode(r, xd) if xd.kind == "bf16" else r.astype(xd.store)
    return Tensor._op(fn, [x, w], xd, x.device)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=None, device=None):
        super().__init__()
        dt = dtype or get_default_dtype()
        g = np.random.default_rng(in_features * 7 + out_features)
        self.weight = Parameter(Tensor(_encode(g.standard_normal((out_features, in_features)) * 0.05, dt), dt,
                                       _dev(device)), requires_grad=False)
        self.bias = None

    def forward(self, x):
        return linear(x, self.weight)


class _EmbedOOB(IndexError):
    pass


def embedding(input, weight, *a, **k):
    if input.dtype.kind != "i":
        raise TypeError("embedding indices must be integer")
    V = weight.shape[0]
    if input.numel() and (int(input._a.min()) < 0 or int(input._a.max()) >= V):
        raise _EmbedOOB(f"device-side assert: embedding index out of range [{int(input._a.min())}, "
                        f"{int(input._a.max())}] for {V} rows")
    return Tensor._op(lambda i, w: w[i], [input, weight], weight.dtype, weight.device)


def pad(x, p, mode="constant", value=0.0):
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------------
# cuda / graphs / grad modes
# ---------------------------------------------------------------------------------------------------------------------
class CUDAGraph:
    def __init__(self):
        self._tape = None
        self.replays = 0

    def replay(self):
        if self._tape is None:
            raise RuntimeError("replay of an empty / reset CUDA graph")
        self.replays += 1
        for f, _ in self._tape:
            f()
        if getattr(INJ, "corrupt_replay", False) and self._tape:
            tgt = self._tape[-1][1]._a.reshape(-1)
            tgt[0] ^= np.asarray(1, dtype=tgt.dtype)

    def reset(self):
        self._tape = None


class graph:
    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="global"):
        self.g = cuda_graph
        self.capture_error_mode = capture_error_mode

    def __enter__(self):
        if getattr(_TLS, "tape", None) is not None:
            raise RuntimeError("nested capture")
        _TLS.tape = []
        return self.g

    def __exit__(self, et, ev, tb):
        tape, _TLS.tape = _TLS.tape, None
        if et is None:
            self.g._tape = tape
        return False


class _GradMode:
    def __init__(self, *a, **k):
        pass

    def __call__(self, fn=None):
        if fn is None:
            return self
        def w(*a, **k):
            with type(self)():
                return fn(*a, **k)
        return w

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


class no_grad(_GradMode):
    pass


class inference_mode(_GradMode):
    pass


def _torch_compile(fn=None, **kw):
    """torch.compile stand-in: the decorated function runs eagerly (its ops are the fake's)."""
    if fn is None:
        return lambda f: f
    return fn


def make_fake_torch() -> types.ModuleType:
    tm = types.ModuleType("torch")
    for k, v in globals().items():
        if k.startswith("__"):
            continue
        setattr(tm, k, v)
    tm.bool = bool_
    tm.compile = _torch_compile
    tm.Tensor = Tensor
    tm.dtype = dtype
    tm.long = int64
    tm.int = int32
    tm.float = float32
    tm.half = float16
    nn = types.ModuleType("torch.nn")
    nn.Module, nn.ModuleList, nn.ModuleDict, nn.Linear, nn.Parameter = Module, ModuleList, ModuleDict, Linear, Parameter
    F = types.ModuleType("torch.nn.functional")
    F.embedding, F.linear, F.pad = embedding, linear, pad
    nn.functional = F
    nnp = types.ModuleType("torch.nn.parameter")
    nnp.Parameter = Parameter
    nn.parameter = nnp
    tm.nn = nn
    cuda = types.ModuleType("torch.cuda")
    cuda.CUDAGraph, cuda.graph = CUDAGraph, graph
    cuda.synchronize = lambda *a, **k: None
    cuda.empty_cache = lambda *a, **k: _counter("empty_cache")
    cuda.is_available = lambda: True
    cuda.is_current_stream_capturing = lambda: getattr(_TLS, "tape", None) is not None
    cuda.current_stream = lambda *a, **k: NS(wait_stream=lambda s: None)
    cuda.Stream = lambda *a, **k: NS(wait_stream=lambda s: None)
    cuda.stream = lambda s: contextlib.nullcontext()
    tm.cuda = cuda
    acc = types.ModuleType("torch.accelerator")
    acc.get_memory_info = lambda d=None: (1 << 40, 1 << 41)
    tm.accelerator = acc
    dist = types.ModuleType("torch.distributed")
    dist.all_gather_object = lambda out, obj, group=None: out.__setitem__(slice(None), current_world()
                                                                          .all_gather_object(obj))
    tm.distributed = dist
    return tm


COUNTERS: dict = {}
_CLOCK = threading.Lock()


def _counter(name):
    with _CLOCK:
        COUNTERS[name] = COUNTERS.get(name, 0) + 1


def install_fake_torch():
    for k in [k for k in sys.modules if k == "torch" or k.startswith("torch.")]:
        del sys.modules[k]
    tm = make_fake_torch()
    sys.modules["torch"] = tm
    for sub in ("nn", "nn.functional", "nn.parameter", "cuda", "distributed", "accelerator"):
        obj = tm
        for part in sub.split("."):
            obj = getattr(obj, part)
        sys.modules["torch." + sub] = obj
    return tm


# ---------------------------------------------------------------------------------------------------------------------
# rank world (threads)
# ---------------------------------------------------------------------------------------------------------------------
class ScheduleMismatch(RuntimeError):
    pass


class Hang(TimeoutError):
    pass


def timeout_scale() -> float:
    """Rank threads share one process, its GIL and the Mac's cores with whatever else runs there: a rendezvous that waits a few seconds for
    the other ranks must not turn a busy machine into a 'HANG'. $PATCH_TEST_TIMEOUT_SCALE (>= 1, default 1) multiplies every rendezvous
    and rank-thread timeout of this library; a suite runner sets 4 when it runs the suites in parallel."""
    try:
        return max(1.0, float(os.environ.get("PATCH_TEST_TIMEOUT_SCALE", "1")))
    except ValueError:
        return 1.0


class Rendezvous:
    def __init__(self, n: int, timeout: float = 8.0):
        self.n, self.timeout = n, timeout * timeout_scale()
        self.cv = threading.Condition()
        self.slots: dict = {}
        self.results: dict = {}
        self.log: list = []

    def exchange(self, rank: int, seq: int, sig, value, combine):
        with self.cv:
            slot = self.slots.setdefault(seq, {})
            slot[rank] = (sig, value)
            if len(slot) == self.n:
                sigs = {r: s for r, (s, _) in slot.items()}
                if len(set(sigs.values())) != 1:
                    self.results[seq] = ScheduleMismatch(f"collective #{seq}: {sigs}")
                else:
                    try:
                        self.results[seq] = combine([slot[r][1] for r in range(self.n)])
                    except Exception as e:  # noqa: BLE001
                        self.results[seq] = e
                self.log.append((seq, sigs[0]))
                self.cv.notify_all()
            ok = self.cv.wait_for(lambda: seq in self.results, timeout=self.timeout)
            if not ok:
                raise Hang(f"rank {rank}: collective #{seq} {sig} never completed (a rank did not join)")
            r = self.results[seq]
        if isinstance(r, Exception):
            raise r
        return r


def oneshot_sum(arrs: list) -> np.ndarray:
    """Patch 012's kernel: acc = +0.0; for r: acc += own_or_0 + got_or_0 (one is x_r, the other +0.0); bf16 once."""
    acc = np.zeros(arrs[0].shape, dtype=np.float32)
    for a in arrs:
        x = bf16_bits_to_f32(a).astype(np.float32)
        acc = (acc + (x + np.float32(0.0))).astype(np.float32)
    return f32_to_bf16_bits(acc)


def ring_sum(arrs: list, ftz: bool = False, nchunks: int = 4) -> np.ndarray:
    """NCCL-like bf16 ring: per chunk c the partial sum starts at rank (c+1) % W and each hop adds one rank's value in
    fp32 and rounds to bf16 (optionally flushing subnormals); every rank receives the same result."""
    W = len(arrs)
    flat = [np.asarray(a).reshape(-1) for a in arrs]
    n = flat[0].size
    out = np.empty(n, dtype=np.uint16)
    bounds = np.linspace(0, n, nchunks + 1).astype(int)
    for c in range(nchunks):
        lo, hi = bounds[c], bounds[c + 1]
        if lo == hi:
            continue
        s = (c + 1) % W
        f = (lambda v: ftz_f32(v)) if ftz else (lambda v: v)
        acc = f(bf16_bits_to_f32(flat[s][lo:hi]))
        for k in range(1, W):
            x = f(bf16_bits_to_f32(flat[(s + k) % W][lo:hi]))
            acc = bf16_bits_to_f32(f32_to_bf16_bits(f(acc + x)))
        out[lo:hi] = f32_to_bf16_bits(f(acc))
    return out.reshape(arrs[0].shape)


class World:
    """One rank's view: collectives through a shared Rendezvous."""

    kind = "thread"

    def __init__(self, rv: Rendezvous, rank: int, size: int, oneshot_max: int, pp_size: int = 1, ftz: bool = False):
        self.rv, self.rank, self.size, self.pp_size = rv, rank, size, pp_size
        self.oneshot_max, self.ftz = oneshot_max, ftz
        self.seq = 0
        self.ar_calls = []
        self.ag_calls = 0
        self.ar_hook = None          # fn(seq, x_array) -> array: corrupt this rank's contribution
        self.ar_out_hook = None      # fn(seq, out_array) -> array: corrupt what this rank receives

    def _next(self):
        s = self.seq
        self.seq += 1
        return s

    def all_reduce(self, t):
        if not isinstance(t, Tensor):
            raise TypeError("all_reduce needs a tensor")
        seq = self._next()
        x = np.array(t._a, copy=True)
        if self.ar_hook is not None:
            x = self.ar_hook(seq, x)
        oneshot = t.dtype is bfloat16 and t.is_contiguous() and 0 < t.numel() <= self.oneshot_max
        sig = ("all_reduce", tuple(t.shape), t.dtype.name, oneshot)
        self.ar_calls.append((tuple(t.shape), oneshot))
        if t.dtype is not bfloat16:
            raise TypeError("only bf16 all-reduce is emulated")
        res = self.rv.exchange(self.rank, seq, sig, x, (lambda arrs: oneshot_sum(arrs)) if oneshot
                               else (lambda arrs: ring_sum(arrs, ftz=self.ftz)))
        res = np.array(res, copy=True)
        if self.ar_out_hook is not None:
            res = self.ar_out_hook(seq, res)
        return Tensor(res, bfloat16, t.device)

    def all_gather_object(self, obj):
        seq = self._next()
        self.ag_calls += 1
        return list(self.rv.exchange(self.rank, seq, ("all_gather_object",), obj, lambda objs: list(objs)))


def current_world():
    return _TLS.world


def set_current_world(w):
    _TLS.world = w


def run_ranks(n: int, fn, timeout: float = 120.0):
    """fn(rank) in n threads (thread name 'rank<r>'); returns [(ok, result_or_exception)] per rank."""
    timeout = timeout * timeout_scale()
    out = [None] * n

    def body(r):
        try:
            out[r] = (True, fn(r))
        except BaseException as e:  # noqa: BLE001
            out[r] = (False, e)
    th = [threading.Thread(target=body, args=(r,), name=f"rank{r}", daemon=True) for r in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout)
    for r, t in enumerate(th):
        if t.is_alive():
            out[r] = (False, Hang(f"rank {r} thread still running after {timeout} s"))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# exec helpers (REAL files, whole or the named top-level definitions only)
# ---------------------------------------------------------------------------------------------------------------------
def mod(name, pkg=False, **attrs):
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


def exec_real(name, src: str, filename: str, package: str):
    m = types.ModuleType(name)
    m.__file__ = filename
    m.__package__ = package
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    exec(compile(src, filename, "exec"), m.__dict__)
    return m


def extract(path: Path, names: list[str]) -> str:
    """Source of the named top-level defs / classes / assignments of a REAL file, in file order."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out, found = [], set()
    for node in tree.body:
        nm = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nm = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            out.append("".join(lines[start - 1: node.end_lineno]) + "\n")
            found.add(nm)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"{path}: definitions not found: {sorted(missing)}")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------------
# auto-stubs for every other vllm.* import of model.py
# ---------------------------------------------------------------------------------------------------------------------
class _Stub:
    def __init__(self, name):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_kids", {})
        object.__setattr__(self, "_cls", None)

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        kids = object.__getattribute__(self, "_kids")
        if k not in kids:
            kids[k] = _Stub(f"{self._name}.{k}")
        return kids[k]

    def __call__(self, *a, **k):
        if len(a) == 1 and not k and isinstance(a[0], type):
            return a[0]                      # decorator application
        return _Stub(f"{self._name}()")

    def __mro_entries__(self, bases):
        if self._cls is None:
            object.__setattr__(self, "_cls", type(self._name.rsplit(".", 1)[-1], (), {}))
        return (self._cls,)

    def __or__(self, o):
        return self

    __ror__ = __or__

    def __getitem__(self, k):
        return self

    def __repr__(self):
        return f"<stub {self._name}>"


class _AutoModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        s = _Stub(f"{self.__name__}.{k}")
        setattr(self, k, s)
        return s


class _AutoFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIXES = ("vllm.",)
    EXCLUDE = ("vllm._gb10_repl_embed",)     # must stay a real ImportError when the patch module is not installed

    def find_spec(self, name, path, target=None):
        if name.startswith(self.PREFIXES) and name not in sys.modules and name not in self.EXCLUDE:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _AutoModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


_FINDER = _AutoFinder()


def enable_autostubs():
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


def disable_autostubs():
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)


# ---------------------------------------------------------------------------------------------------------------------
# logging capture
# ---------------------------------------------------------------------------------------------------------------------
class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self._rlock = threading.Lock()      # not self.lock: logging.Handler.handle() already holds that one

    def emit(self, rec):
        with self._rlock:
            self.records.append((threading.current_thread().name, rec.levelname, rec.getMessage()))

    def lines(self, rank=None, contains=None):
        with self._rlock:
            out = [m for (t, _, m) in self.records if (rank is None or t == f"rank{rank}")
                   and (contains is None or contains in m)]
        return out

    def clear(self):
        with self._rlock:
            self.records.clear()


LOGS = ListHandler()


def make_logger(name="p032test"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)
    if LOGS not in lg.handlers:
        lg.addHandler(LOGS)
    lg.propagate = False

    class _VLogger:
        def __init__(self, base):
            self._b = base

        def __getattr__(self, k):
            if k.endswith("_once"):
                return getattr(self._b, k[:-5])
            return getattr(self._b, k)
    return _VLogger(lg)
