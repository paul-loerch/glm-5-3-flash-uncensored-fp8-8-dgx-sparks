# SPDX-License-Identifier: Apache-2.0
"""[patch-035 gb10-idx-compact-xchg] CPU test library (no torch, no GPU, no vLLM install; python3 + numpy).

  FT / make_torch   a numpy-backed fake `torch` with exactly the tensor ops the REAL sparse_attn_indexer_kpool prefill branch and the
                    patch's helper block use (full, tensor, equal, Tensor.to / contiguous / view / stride / fill_ / new_empty, a
                    stand-in torch.distributed.all_reduce, torch.cuda.is_current_stream_capturing, torch.ops._C.top_k_per_row_prefill)
  expand_vec        a numpy model of `_expand_pools_and_append_tail_kernel` (models/glm5next/nvidia/ops/kpool_compress.py), written
                    from the Triton source line by line; expand_scalar is a second, literal per-element transliteration used to
                    cross-check it
  World             N ranks as N threads: all_gatherv / all_reduce with barrier semantics; every collective registers its kind, so a rank
                    that enters a different collective than its peers is a DesyncError (the hang the patch must never cause) instead
                    of a silent pass; a rank that dies aborts the barrier so the others fail fast
  Scenario          a random prefill step: requests (prior context, query rows), the work rows, the chunk metadata the real builder
                    would produce (token_start / token_end absolute rows, per-row cu_seqlen_ks / ke), the per-rank row-shard sizes
  run_world         executes the REAL top-level nodes of vllm/models/glm5next/nvidia/sparse_indexer.py (the original or the patched
                    text: the helper block and `sparse_attn_indexer_kpool` itself) plus the REAL PrefillRowShard and
                    _merge_dcp_topk_global of model_executor/layers/sparse_attn_indexer.py, on N rank threads, and returns every rank's
                    final top-k buffer, log records, kernel/collective trace
"""
from __future__ import annotations

import ast
import logging
import os
import sys
import threading
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

    device = "fake:0"

    def view(self, *a):
        if len(a) == 1 and isinstance(a[0], FDtype):
            return np.ndarray.view(self, a[0].np)
        return np.ndarray.view(self, *a)

    def to(self, dtype):
        return self.astype(dtype.np)

    def numel(self):
        return int(self.size)

    def contiguous(self):
        return np.ascontiguousarray(self).view(FT)

    def stride(self, i):
        return int(self.strides[i] // self.itemsize)

    def fill_(self, v):
        self[...] = v
        return self

    def new_empty(self, shape, dtype=None):
        return _ft(np.zeros(shape, dtype=(dtype.np if dtype is not None else self.dtype)))


def _ft(a):
    return np.asarray(a).view(FT)


T_INT8 = FDtype("int8", np.int8)
T_INT32 = FDtype("int32", np.int32)
T_INT64 = FDtype("int64", np.int64)
T_FLOAT32 = FDtype("float32", np.float32)
T_UINT8 = FDtype("uint8", np.uint8)
POISON = -7777


class _State:
    """Per-run switches the fake torch reads (CUDA graph capture, the top-k kernel)."""

    def __init__(self):
        self.capturing = False
        self.topk = None  # callable overriding the default deterministic fake top-k
        self.trace = None  # list the fake kernels append to


def make_torch(world_ref, state: _State):
    t = types.SimpleNamespace()
    t.int8, t.int32, t.int64, t.float32, t.uint8 = T_INT8, T_INT32, T_INT64, T_FLOAT32, T_UINT8
    t.Tensor = FT

    def tensor(data, dtype=None, device=None):
        return _ft(np.array(data, dtype=(dtype.np if dtype is not None else None)))

    def full(shape, fill, dtype=None, device=None):
        return _ft(np.full(shape, fill, dtype=(dtype.np if dtype is not None else None)))

    def empty(shape, dtype=None, device=None):  # poisoned: reading an unwritten element shows up
        return _ft(np.full(shape, POISON, dtype=(dtype.np if dtype is not None else np.int32)))

    def equal(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return bool(a.shape == b.shape and np.array_equal(a, b))

    t.tensor, t.full, t.empty, t.equal = tensor, full, empty, equal

    def all_reduce(tensor_, op=None, group=None):
        group.world.all_reduce(group.rank, tensor_, op)

    t.distributed = types.SimpleNamespace(all_reduce=all_reduce, ReduceOp=types.SimpleNamespace(MAX="MAX", MIN="MIN", SUM="SUM"))
    t.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: state.capturing)

    def top_k_per_row_prefill(logits, ks, ke, topk_dst, num_rows, stride0, stride1, select_k):
        if state.trace is not None:
            state.trace.append(("topk", int(num_rows), int(select_k)))
        (state.topk or fake_topk)(logits, ks, ke, topk_dst, num_rows, select_k)

    t.ops = types.SimpleNamespace(_C=types.SimpleNamespace(top_k_per_row_prefill=top_k_per_row_prefill))
    return t


def fake_topk(logits, ks, ke, topk_dst, num_rows, select_k):
    """Deterministic per-row top-k over the row's window [ks, ke): value descending, index ascending; ids relative to ks; -1 padded."""
    L = np.asarray(logits)
    ks, ke = np.asarray(ks), np.asarray(ke)
    for r in range(num_rows):
        a, b = int(ks[r]), int(ke[r])
        n = max(b - a, 0)
        out = np.full(select_k, -1, np.int32)
        if n > 0:
            vals = L[r, a:b].astype(np.float64)
            order = np.lexsort((np.arange(n), -vals))
            m = min(select_k, n)
            out[:m] = order[:m]
        topk_dst[r] = out


def make_racy_topk(seed):
    """Like the image kernel: the SET is exact but the ORDER (and tie members) vary from call to call."""
    rng = np.random.default_rng(seed)

    def racy(logits, ks, ke, topk_dst, num_rows, select_k):
        fake_topk(logits, ks, ke, topk_dst, num_rows, select_k)
        for r in range(num_rows):
            row = np.asarray(topk_dst[r]).copy()
            m = int((row >= 0).sum())
            perm = rng.permutation(m)
            row[:m] = row[:m][perm]
            topk_dst[r] = row

    return racy


# ---------------------------------------------------------------------------------------------------------------------
# the expand kernel, from the Triton source (kpool_compress.py::_expand_pools_and_append_tail_kernel)
# ---------------------------------------------------------------------------------------------------------------------
def _wrap32(x):
    """tl int32 cast of a Python int (two's complement wrap)."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def expand_scalar(pool_row, seq_len, pool_size):
    """One output row of the kernel, element by element (the kernel body for `row`, every column tile concatenated)."""
    n_groups = len(pool_row)
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    pool_len = seq_len // pool_size  # seq_len >= 1 in every real call (positions + 1)
    tail_start = pool_len * pool_size
    tail_count = seq_len - tail_start
    out = []
    for cols in range(out_cols):
        is_history = cols < topk
        g = cols // pool_size
        o = cols % pool_size
        pid = int(pool_row[g]) if is_history else -1  # tl.load(..., mask=mask & is_history, other=-1)
        hist_val = _wrap32(pid * pool_size + o)
        hist_out = hist_val if (pid >= 0 and pid < pool_len) else -1
        tail_off = cols - topk
        is_tail = tail_off >= 0 and tail_off < tail_count
        tail_val = _wrap32(tail_start + tail_off)
        tail_out = tail_val if is_tail else -1
        out.append(hist_out if is_history else tail_out)
    return out


def expand_vec(pool_ids, seq_lens, pool_size):
    """Vectorised model of the whole launch, grid (rows, n_tiles): out[rows, topk + pool_size - 1] int32."""
    pool_ids = np.asarray(pool_ids)
    seq_lens = np.asarray(seq_lens)
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    cols = np.arange(out_cols)
    seq_len = seq_lens.astype(np.int32)[:, None]
    pool_len = seq_len // pool_size
    tail_start = pool_len * pool_size
    tail_count = seq_len - tail_start
    is_history = cols[None, :] < topk
    g = cols // pool_size
    o = cols % pool_size
    gcl = np.minimum(g, n_groups - 1)
    pid = np.where(is_history, pool_ids[:, gcl], -1)
    hist_val = (pid.astype(np.int64) * pool_size + o[None, :]).astype(np.int32)
    hist_out = np.where((pid >= 0) & (pid < pool_len), hist_val, -1)
    tail_off = cols[None, :] - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail_val = (tail_start + tail_off).astype(np.int32)
    tail_out = np.where(is_tail, tail_val, -1)
    return np.where(is_history, hist_out, tail_out).astype(np.int32)


def expand_pools_to_tokens_np(group_ids, group_valid, topk, pool_size):
    """kpool_compress.expand_pools_to_tokens, identity path (the `positions is None` branch of the op)."""
    offsets = np.arange(pool_size, dtype=np.int64)
    token_ids = (np.asarray(group_ids).astype(np.int64)[:, :, None] * pool_size + offsets).reshape(group_ids.shape[0], topk)
    valid = np.repeat(np.asarray(group_valid)[:, :, None], pool_size, axis=2).reshape(group_ids.shape[0], topk)
    return np.where(valid, token_ids.astype(np.int32), -1).astype(np.int32)


# ---------------------------------------------------------------------------------------------------------------------
# the fake TP world: N rank threads
# ---------------------------------------------------------------------------------------------------------------------
class DesyncError(RuntimeError):
    pass


class World:
    def __init__(self, n, timeout=30.0):
        self.n = n
        self.bar = threading.Barrier(n, timeout=timeout)
        self.slots = [None] * n
        self.err = None
        self.trace = [[] for _ in range(n)]
        self.stats = {"gatherv_calls": 0, "gatherv_recv_bytes": 0, "gatherv_width": [], "allreduce_calls": 0}
        self.lock = threading.Lock()

    def abort(self):
        try:
            self.bar.abort()
        except Exception:  # noqa: BLE001
            pass

    def _round(self, rank, kind, payload, combine):
        self.slots[rank] = (kind, payload)
        self.trace[rank].append(kind)
        try:
            self.bar.wait()
            kinds = {s[0] for s in self.slots}
            result = None
            if len(kinds) != 1:
                self.err = DesyncError("ranks entered different collectives: " + "; ".join(sorted(map(str, kinds))))
            else:
                result = combine([s[1] for s in self.slots])
            self.bar.wait()
        except threading.BrokenBarrierError:
            raise DesyncError("collective never completed (a peer died or entered a different collective)") from None
        if self.err is not None:
            raise self.err
        return result

    def all_gatherv(self, rank, local, sizes):
        local = np.asarray(local)
        if sizes is None:
            sizes = [local.shape[0]] * self.n
        if local.shape[0] != sizes[rank]:
            raise AssertionError(f"{local.shape[0]} != {sizes[rank]}")  # the CudaCommunicator.all_gatherv assertion
        kind = ("gatherv", tuple(local.shape[1:]), str(local.dtype))

        def combine(payloads):
            ss = {p[1] for p in payloads}
            if len(ss) != 1:
                raise DesyncError(f"ranks disagree on sizes: {ss}")
            for r, (arr, sz) in enumerate(payloads):
                if arr.shape[0] != sz[r]:
                    raise DesyncError(f"rank {r} sent {arr.shape[0]} rows, sizes say {sz[r]}")
            return np.concatenate([p[0] for p in payloads], axis=0)

        out = self._round(rank, kind, (local.copy(), tuple(sizes)), combine)
        if rank == 0:
            with self.lock:
                self.stats["gatherv_calls"] += 1
                self.stats["gatherv_recv_bytes"] += int(out.nbytes)
                self.stats["gatherv_width"].append(int(local.shape[1]) if local.ndim > 1 else 1)
        return _ft(out.copy())

    def all_reduce(self, rank, tensor, op):
        arr = np.asarray(tensor)
        kind = ("allreduce", op, tuple(arr.shape), str(arr.dtype))

        def combine(payloads):
            stack = np.stack(payloads)
            return {"MAX": stack.max(0), "MIN": stack.min(0), "SUM": stack.sum(0)}[op]

        res = self._round(rank, kind, arr.copy(), combine)
        tensor[...] = res.astype(arr.dtype)
        if rank == 0:
            with self.lock:
                self.stats["allreduce_calls"] += 1


class FakeTPGroup:
    def __init__(self, world, rank):
        self.world, self.rank_in_group, self.world_size = world, rank, world.n
        self.cpu_group = types.SimpleNamespace(world=world, rank=rank)

    def all_gatherv(self, input_, dim=0, sizes=None):
        assert dim == 0
        return self.world.all_gatherv(self.rank_in_group, input_, sizes)


_TLS = threading.local()


def _install_fake_vllm():
    """sys.modules stand-ins for the imports done inside the code under test (vllm.distributed, vllm.utils.deep_gemm)."""
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.distributed", "vllm.utils", "vllm.utils.deep_gemm")}
    vllm = types.ModuleType("vllm")
    dist = types.ModuleType("vllm.distributed")
    dist.get_tp_group = lambda: _TLS.group
    dist.get_tensor_model_parallel_rank = lambda: _TLS.group.rank_in_group
    utils = types.ModuleType("vllm.utils")
    dg = types.ModuleType("vllm.utils.deep_gemm")
    dg.fp8_fp4_mqa_logits = lambda *a, **k: _TLS.mqa(*a, **k)
    vllm.distributed, vllm.utils, utils.deep_gemm = dist, utils, dg
    sys.modules.update({"vllm": vllm, "vllm.distributed": dist, "vllm.utils": utils, "vllm.utils.deep_gemm": dg})
    return saved


def _restore_vllm(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ---------------------------------------------------------------------------------------------------------------------
# logging capture
# ---------------------------------------------------------------------------------------------------------------------
class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))

    def lines(self, level=None, contains=None):
        return [m for lv, m in self.records if (level is None or lv == level) and (contains is None or contains in m)]


# ---------------------------------------------------------------------------------------------------------------------
# extracting the real nodes
# ---------------------------------------------------------------------------------------------------------------------
def _segment(text: str, node) -> str:
    lines = text.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(lines[first - 1: node.end_lineno])


HELPER_NAMES = ("_p035_config", "_P035_STATE", "_p035_rank0", "_p035_log0", "_p035_reduce_max", "_p035_decline_reason",
                "_P035Plan", "_p035_begin")
OP_NAMES = ("PrefillRowShard", "_merge_dcp_topk_global")


def _wanted(mod: ast.Module, names, tuple_targets=()):
    out = []
    for n in mod.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            out.append(n)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            out.append(n)
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) \
                and tuple(e.id for e in n.targets[0].elts if isinstance(e, ast.Name)) in tuple_targets:
            out.append(n)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------------------------------------------------
class Chunk:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Scenario:
    """A random prefill step. All arrays are plain numpy; run_world wraps them per rank."""

    def __init__(self, rng: np.random.Generator, tp: int, *, ratio=4, topk_tokens=64, nd=None, max_rows=300, sizes_mode="random",
                 cut_mode="random", max_prior=900, tiny_chunk=False, nreq=None, deep=True):
        self.ratio, self.topk_tokens, self.tp = ratio, topk_tokens, tp
        self.select_k = topk_tokens // ratio if ratio > 1 else topk_tokens
        self.width = topk_tokens + ratio - 1
        self.nd = int(rng.integers(0, 6)) if nd is None else nd
        nreq = int(rng.integers(1, 5)) if nreq is None else nreq
        reqs = []
        for i in range(nreq):
            q = int(rng.integers(1, max(2, max_rows // nreq + 1)))
            prior = int([0, int(rng.integers(0, max(max_prior, 1)))][int(rng.integers(0, 2))])
            reqs.append((prior, q))
        P = sum(q for _, q in reqs)
        if P < tp:  # at least one row per rank
            reqs[-1] = (reqs[-1][0], reqs[-1][1] + (tp - P))
        if deep and max(p + q for p, q in reqs) <= topk_tokens:  # a sharded step has a deep request (else the op's short-prefill branch runs)
            j = int(rng.integers(0, len(reqs)))
            reqs[j] = (topk_tokens + int(rng.integers(1, 200)), reqs[j][1])
        if tiny_chunk:  # a whole request below one pool => zero compressed keys => the real builder drops its chunk
            reqs.insert(int(rng.integers(0, len(reqs) + 1)), (0, int(rng.integers(1, ratio))))
            cut_mode = "requests"
        P = sum(q for _, q in reqs)
        self.reqs, self.P = reqs, P
        row_req, row_pos = [], []
        for i, (prior, q) in enumerate(reqs):
            row_req += [i] * q
            row_pos += list(range(prior, prior + q))
        self.row_req, self.row_pos = np.array(row_req), np.array(row_pos, dtype=np.int64)
        self.seq_lens = [prior + q for prior, q in reqs]
        self.max_prefill_seq_len = max(self.seq_lens)
        self.num_tokens = self.nd + P
        self.pad = int(rng.integers(0, 4))
        self.rows_alloc = self.num_tokens + self.pad
        self.buf_w = ((self.width + 127) // 128) * 128
        # --- row-shard sizes: each >= 1, sum P
        if sizes_mode == "equal":
            base = P // tp
            sizes = [base] * tp
            sizes[-1] += P - base * tp
        elif sizes_mode == "skewed":
            sizes = [1] * tp
            sizes[int(rng.integers(0, tp))] += P - tp
        else:
            cuts = np.sort(rng.choice(np.arange(1, P), size=tp - 1, replace=False)) if P > tp else np.arange(1, tp)
            sizes = list(np.diff(np.concatenate([[0], cuts, [P]])).astype(int))
        self.sizes = [int(s) for s in sizes]
        assert sum(self.sizes) == P and min(self.sizes) >= 1
        # --- chunk cuts over the prefill rows
        if cut_mode == "one":
            cuts = [0, P]
        elif cut_mode == "requests":
            cuts = [0]
            acc = 0
            for _, q in reqs:
                acc += q
                cuts.append(acc)
        else:
            ncut = int(rng.integers(0, 6))
            inner = sorted(set(int(c) for c in rng.integers(1, max(2, P), size=ncut))) if P > 1 else []
            cuts = [0] + [c for c in inner if 0 < c < P] + [P]
        self.cuts = sorted(set(cuts))
        self.chunks = self._build_chunks()
        self.dropped_rows = P - sum(c.token_end - c.token_start for c in self.chunks)
        # --- per-token arrays
        self.positions = np.concatenate([np.arange(self.nd, dtype=np.int64) + 5000, self.row_pos, np.zeros(self.pad, np.int64)])
        qq = np.zeros((self.rows_alloc, 2, 4), np.float32)
        qq[:, 0, 0] = np.arange(self.rows_alloc)
        self.q_quant = qq
        self.weights = np.ones((self.rows_alloc, 2), np.float32)
        self.total_seq_lens = max([c.total_seq_lens for c in self.chunks] + [1])

    def _build_chunks(self):
        chunks = []
        r = self.ratio
        for a, b in zip(self.cuts[:-1], self.cuts[1:]):
            rows = range(a, b)
            reqs_in = []
            for j in rows:
                if not reqs_in or reqs_in[-1] != self.row_req[j]:
                    reqs_in.append(int(self.row_req[j]))
            comp = {i: self.seq_lens[i] // r for i in reqs_in}
            total = sum(comp.values())
            if total == 0:
                continue  # build_prefill_chunk_metadata returns None: the chunk (and its rows) vanish from prefill_metadata.chunks
            base, acc = {}, 0
            for i in reqs_in:
                base[i] = acc
                acc += comp[i]
            ks = np.array([base[int(self.row_req[j])] for j in rows], np.int32)
            ke = np.array([base[int(self.row_req[j])] + (int(self.row_pos[j]) + 1) // r for j in rows], np.int32)
            first_prior = self.reqs[self.row_req[a]][0]
            mid_request = a > 0 and self.row_req[a] == self.row_req[a - 1]
            chunks.append(Chunk(token_start=self.nd + a, token_end=self.nd + b, cu_seqlen_ks=ks, cu_seqlen_ke=ke,
                                skip_kv_gather=bool(mid_request), local_total_seq_lens=total, max_local_total_seq_lens=total,
                                total_seq_lens=total, local_cu_seq_lens=np.array([0, total], np.int32), block_table=None))
        return chunks

    def tiles(self):
        pos = self.nd
        for c in self.chunks:
            if c.token_start != pos:
                return False
            pos = c.token_end
        return pos == self.nd + self.P


# ---------------------------------------------------------------------------------------------------------------------
# running the REAL code on N rank threads
# ---------------------------------------------------------------------------------------------------------------------
class RankResult:
    def __init__(self):
        self.bufs: list = []          # one final top-k buffer per call
        self.records: list = []       # (level, message)
        self.trace: list = []         # fake kernel launches in order
        self.exc: BaseException | None = None
        self.ns = None


class RunResult:
    def __init__(self, ranks, world, hung=False):
        self.ranks, self.world, self.hung = ranks, world, hung

    @property
    def ok(self):
        return not self.hung and all(r.exc is None for r in self.ranks)

    def errors(self):
        errs = [f"rank {i}: {type(r.exc).__name__}: {r.exc}" for i, r in enumerate(self.ranks) if r.exc is not None]
        return errs + (["a rank thread is still running (hang)"] if self.hung else [])


def _fake_ns(rank, world, state: _State, logger, text_nodes):
    torch = make_torch(world, state)
    holder = types.SimpleNamespace(attn=None)

    def record(name, *info):
        if state.trace is not None:
            state.trace.append((name,) + info)

    kpool_ops = types.SimpleNamespace()

    def expand_pools_and_append_tail(pool_ids, seq_lens, pool_size):
        record("expand", tuple(np.asarray(pool_ids).shape), str(np.asarray(pool_ids).dtype), tuple(np.asarray(seq_lens).shape))
        return _ft(expand_vec(pool_ids, seq_lens, pool_size))

    def expand_pools_to_tokens(group_ids, group_valid, topk, pool_size):
        record("expand_tokens", tuple(np.asarray(group_ids).shape))
        return _ft(expand_pools_to_tokens_np(group_ids, group_valid, topk, pool_size))

    kpool_ops.expand_pools_and_append_tail = expand_pools_and_append_tail
    kpool_ops.expand_pools_to_tokens = expand_pools_to_tokens
    ops = types.SimpleNamespace(cp_gather_indexer_k_quant_cache=lambda *a, **k: record("cp_gather"))

    def fp8_fp4_mqa_logits(q_pair, k_pair, weights, ks, ke, clean_logits=False):
        q, _ = q_pair
        kq, _ = k_pair
        ks, ke = np.asarray(ks), np.asarray(ke)
        record("mqa_logits", int(np.asarray(q).shape[0]), int(np.asarray(kq).shape[0]))
        ids = np.asarray(q)[:, 0, 0].astype(np.int64)
        cols = np.arange(np.asarray(kq).shape[0])[None, :]
        p = cols - ks[:, None]
        h = (ids[:, None] * 2654435761 + p * 40503 + 12345) % 1000003
        logits = (h.astype(np.float64) / 1000003.0).astype(np.float32)
        valid = (cols >= ks[:, None]) & (cols < ke[:, None])
        return _ft(np.where(valid, logits, np.float32(1e9)))

    def fill_causal_indices(buf, pos):
        record("fill_causal", tuple(buf.shape))
        buf[:, :] = np.asarray(pos)[:, None].astype(np.int32) + np.arange(buf.shape[1], dtype=np.int32)[None, :]

    class _Meta:  # DeepseekV32IndexerMetadata stand-in
        pass

    class _WSM:
        def get_simultaneous(self, *specs):
            return tuple(_ft(np.zeros(shape, dtype=dt.np)) for shape, dt in specs)

    ns = {
        "torch": torch, "logger": logger, "kpool_ops": kpool_ops, "ops": ops,
        "eager_break_during_capture": lambda f: f, "LayerNameType": str, "_resolve_layer_name": lambda x: x,
        "get_forward_context": lambda: types.SimpleNamespace(attn_metadata=holder.attn),
        "current_platform": types.SimpleNamespace(fp8_dtype=lambda: T_UINT8, is_cuda_alike=lambda: True, is_cuda=lambda: True),
        "_gather_workspace_shapes": lambda T, hd, dt, fp4: (((T, hd), dt), ((T, 4), T_UINT8)),
        "current_workspace_manager": lambda: _WSM(),
        "DeepseekV32IndexerMetadata": _Meta, "_Meta": _Meta,
        "_fill_causal_indices": fill_causal_indices,
        "get_tp_group": lambda: _TLS.group, "get_tensor_model_parallel_rank": lambda: _TLS.group.rank_in_group,
        "__name__": "sparse_indexer_under_test",
        "_holder": holder, "_state": state, "_mqa": fp8_fp4_mqa_logits,
    }
    return ns, holder, _Meta


def run_world(text, op_text, scenarios, tp, *, env=None, hooks=None, timeout=30.0, topk=None, capturing=False):
    """Run `sparse_attn_indexer_kpool` from `text` once per scenario (in order, same module state) on `tp` rank threads.

    hooks (all optional): wrap_expand(rank, fn, cur) -> fn   wrap a rank's expand kernel (cur = {"call": i, "P": rows of this call});  after_exec(rank, ns)   edit a rank's namespace
    after the module-level code ran;  index_kpool / topk_tokens / positions_none / no_row_shard / row_shard_sizes /
    max_prefill_seq_len / extra_tokens: layout switches that override what each Scenario says.  topk(rank) -> callable: a rank's
    top-k kernel;  capturing: CUDA graph capture state."""
    hooks = hooks or {}
    saved = _install_fake_vllm()
    world = World(tp, timeout=timeout)
    old_env = {k: os.environ.get(k) for k in os.environ if k.startswith("VLLM_GB10_IDX_COMPACT_XCHG")}
    for k in old_env:
        del os.environ[k]
    os.environ.update(env or {})
    results = [RankResult() for _ in range(tp)]
    namespaces = []
    try:
        mod = ast.parse(text)
        opmod = ast.parse(op_text)
        helper_nodes = _wanted(mod, HELPER_NAMES, tuple_targets=(("_P035_ON", "_P035_VERIFY"),))
        fn_nodes = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer_kpool"]
        op_nodes = _wanted(opmod, OP_NAMES)
        for rank in range(tp):
            logger = logging.getLogger(f"p035test.{id(world)}.{rank}")
            cap = Capture()
            logger.handlers, logger.propagate = [cap], False
            logger.setLevel(logging.DEBUG)
            state = _State()
            state.capturing = capturing
            state.trace = results[rank].trace
            if topk is not None:
                state.topk = topk(rank)
            ns, holder, Meta = _fake_ns(rank, world, state, logger, None)
            cur = {"call": -1, "P": -1}
            ns["_cur"] = cur
            if "wrap_expand" in hooks:
                ns["kpool_ops"].expand_pools_and_append_tail = hooks["wrap_expand"](rank, ns["kpool_ops"].expand_pools_and_append_tail, cur)
            _TLS.group = FakeTPGroup(world, rank)  # for the import-time code only (nothing in it needs the rank)
            for n in op_nodes:
                exec(compile(_segment(op_text, n), f"sparse_attn_indexer.py:{n.lineno}", "exec"), ns)  # noqa: S102
            for n in helper_nodes + fn_nodes:
                exec(compile(_segment(text, n), f"sparse_indexer.py:{n.lineno}", "exec"), ns)  # noqa: S102
            ns["_cap"] = cap
            if "after_exec" in hooks:
                hooks["after_exec"](rank, ns)
            results[rank].ns = ns
            namespaces.append((ns, holder, Meta, cap))
    finally:
        for k in [k for k in os.environ if k.startswith("VLLM_GB10_IDX_COMPACT_XCHG")]:
            del os.environ[k]
        os.environ.update({k: v for k, v in old_env.items() if v is not None})

    def body(rank):
        _TLS.group = FakeTPGroup(world, rank)
        ns, holder, Meta, cap = namespaces[rank]
        res = results[rank]
        _TLS.mqa = ns["_mqa"]
        try:
            for call_i, scen in enumerate(scenarios):
                ns["_cur"]["call"], ns["_cur"]["P"] = call_i, scen.P
                index_kpool = hooks.get("index_kpool", scen.ratio)
                topk_tokens = hooks.get("topk_tokens", scen.topk_tokens)
                md = Meta()
                md.slot_mapping = _ft(np.zeros(scen.num_tokens + hooks.get("extra_tokens", 0), np.int64))
                md.num_decodes, md.num_prefills = 0, 1
                md.num_decode_tokens = scen.nd
                prefill = types.SimpleNamespace(
                    chunks=[types.SimpleNamespace(
                        token_start=c.token_start, token_end=c.token_end, cu_seqlen_ks=_ft(c.cu_seqlen_ks), cu_seqlen_ke=_ft(c.cu_seqlen_ke),
                        skip_kv_gather=c.skip_kv_gather, local_total_seq_lens=c.local_total_seq_lens,
                        max_local_total_seq_lens=c.max_local_total_seq_lens, local_cu_seq_lens=_ft(c.local_cu_seq_lens),
                        block_table=None) for c in scen.chunks],
                    max_prefill_seq_len=hooks.get("max_prefill_seq_len", scen.max_prefill_seq_len),
                    row_shard_sizes=hooks.get("row_shard_sizes", scen.sizes) if not hooks.get("no_row_shard") else None)
                md.prefill, md.decode, md.max_seq_len = prefill, None, scen.max_prefill_seq_len
                holder.attn = {"layer.0": md}
                buf = _ft(np.full((scen.rows_alloc + 3, scen.buf_w), POISON, np.int32))
                positions = None if hooks.get("positions_none") else _ft(scen.positions)
                hidden = _ft(np.zeros((scen.rows_alloc, 1), np.float32))
                qq, ww = _ft(scen.q_quant), _ft(scen.weights)
                res.trace.append(("call", call_i))
                out = ns["sparse_attn_indexer_kpool"](
                    hidden, "layer.0", None, qq, None, None, ww, 128, "ue8m0", topk_tokens, 128, 1 << 20,
                    scen.total_seq_lens + 8, buf, True, False, None, None, index_kpool, positions, None, None, "auto", 0, 1, 1)
                assert out is buf
                res.bufs.append(np.asarray(buf).copy())
        except BaseException as e:  # noqa: BLE001
            res.exc = e
            world.abort()
        finally:
            res.records = list(cap.records)

    try:
        threads = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(tp)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout + 5)
        hung = any(th.is_alive() for th in threads)
    finally:
        _restore_vllm(saved)
    return RunResult(results, world, hung)
