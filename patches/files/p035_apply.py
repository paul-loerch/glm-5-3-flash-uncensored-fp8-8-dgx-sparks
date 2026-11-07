#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 035 (compact pool-id exchange for the TP-sharded indexer prefill) applier: five anchored edits to
vllm/models/glm5next/nvidia/sparse_indexer.py as it is in the SM121-baked v0.32.0 image (written here).

Why: upstream v0.32.0's PrefillRowShard (vLLM #54951, switched on by our patch 031's depth gate) all_gathers the EXPANDED token ids of
every sharded indexer call: [rows, topk_tokens + index_kpool - 1] int32 = 2051 columns. The expansion starts from select_k =
topk_tokens // index_kpool = 512 POOL ids per row, 4x fewer bytes. The old patch 031 exchanged the 512 pool ids and expanded
on every rank; this patch restores that on top of upstream's mechanism:

  per chunk     : top_k_per_row_prefill + _merge_dcp_topk_global exactly as now, then the pool ids pool_topk (int32 [rows, 512]) are
                  copied to their global rows of an int32 staging buffer pool_stage [sum(sizes), 512] instead of being expanded
  after the loop: ONE all_gatherv of this rank's contiguous pool_stage slice (width 512; the call PrefillRowShard.exchange_topk makes),
                  ONE expand_pools_and_append_tail over ALL gathered rows with positions[nd:end] + 1, ONE write into
                  topk_indices_buffer[nd:end, :2051]
  VERIFY        : the first N compact calls ALSO run the original path into a scratch buffer and compare (torch.equal); one TP
                  all_reduce(MAX) on the cpu_group makes every rank take the same decision

  apply(text) -> text     the baked v0.32.0 file -> + patch 035; fail-closed: each of the 5 anchors must match exactly once, every hook
                          must land inside its function (AST-checked), the result must carry N_MARKERS markers and compile
  revert(text) -> text    exact inverse

CLI (used by 035-gb10-idx-compact-xchg.sh):
  p035_apply.py apply <sparse_indexer.py>    in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p035_apply.py check <sparse_indexer.py>    dry run: print the sha256 the file would have after apply()
  p035_apply.py revert <sparse_indexer.py>   in-place inverse (tests / manual rollback)

Edits (line numbers of the baked file, sha256 3d97e02c...):
  A  :100  module level, directly before `@eager_break_during_capture def sparse_attn_indexer_kpool`: the helper block -- _p035_config()
           (env, read ONCE at import, validated: a typo raises), _P035_ON / _P035_VERIFY / _P035_STATE, _p035_rank0, _p035_log0,
           _p035_reduce_max, _p035_decline_reason, class _P035Plan (stage / finish / _verify), _p035_begin.
  B  :312  before `for chunk in prefill_metadata.chunks ...`: _p035_plan = _p035_begin(...) if _P035_ON and row_shard is not None else
           None, and _p035_out (where the ORIGINAL per-chunk expansion writes: the real buffer, or the scratch buffer in a VERIFY call).
  C  :400  in the chunk loop, directly before the `if index_kpool > 1:` expansion block: with a plan, stage the pool ids; unless the
           call is a VERIFY call, `continue` (the original expansion is skipped).
  D  :415  the expansion block's final write goes to _p035_out (== topk_indices_buffer unless a VERIFY call).
  E  :421  the exchange after the loop: with a plan _p035_plan.finish(topk_indices_buffer), else upstream's row_shard.exchange_topk.
Gate unset / 0: _P035_ON is False, _p035_plan is None and _p035_out is topk_indices_buffer; every original statement runs exactly as
before (files/p035_cpu_check.py F compares call traces, buffers and logs against the unpatched function).
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-035 gb10-idx-compact-xchg]"
N_MARKERS = 5
ENV = "VLLM_GB10_IDX_COMPACT_XCHG"
ENV_VERIFY = "VLLM_GB10_IDX_COMPACT_XCHG_VERIFY"
FUNC = "sparse_attn_indexer_kpool"
TAG = "[patch-035]"
HELPER_DEFS = ["_p035_config", "_p035_rank0", "_p035_log0", "_p035_reduce_max", "_p035_decline_reason", "_p035_begin"]
HELPER_ASSIGNS = ("_P035_ON", "_P035_VERIFY", "_P035_STATE")

# ---------------------------------------------------------------------------------------------------------------------
# Edit A: the helper block (module level, directly before the op function's decorator)
# ---------------------------------------------------------------------------------------------------------------------
A_OLD = "@eager_break_during_capture\ndef sparse_attn_indexer_kpool(\n"
BLOCK = (
    MARKER
    + """ opt-in VLLM_GB10_IDX_COMPACT_XCHG=1 (read once, at import).
# The TP-sharded indexer prefill (upstream PrefillRowShard) all_gathers the EXPANDED token ids: topk_tokens + index_kpool - 1 = 2051
# int32 per row, 4x the bytes of the select_k = topk_tokens // index_kpool = 512 POOL ids the expansion starts from. With the gate on,
# a sharded step in the served layout (row_shard, index_kpool > 1, positions) stages each chunk's pool_topk at its global rows,
# all_gathers the 512-wide pool rows once and expands ALL rows with ONE expand_pools_and_append_tail call.
# BITWISE equal to the original: that kernel is per-row integer code (row r of its output depends only on pool_ids[r] and
# seq_lens[r]); the original expands row r from this rank's pool_topk[r] with positions[r] + 1 and copies the expanded row through the
# gather, the compact path gathers the same pool row and expands it with the same positions[r] + 1. Host assertions (else the
# original exchange runs): step tokens == decode tokens + sum(sizes); select_k * index_kpool == topk_tokens (so the expanded width is
# topk_tokens + index_kpool - 1, also checked on the result); the chunks tile the sharded rows exactly (a row outside every chunk keeps
# the original's all -1 row, which the compact path would turn into a tail-only row).
# VERIFY (VLLM_GB10_IDX_COMPACT_XCHG_VERIFY, default 3, 0 = off): the first N compact calls ALSO run the original path (per-chunk
# expansion into a scratch buffer, then upstream's exchange_topk) and compare with torch.equal; the verdict is reduced with an
# all_reduce(MAX) on the TP cpu_group so EVERY rank decides alike: on a mismatch every rank logs an ERROR, uses the original rows and
# turns the compact path off for the process. Set both variables on EVERY rank (a rank on the original path would enter an
# all_gatherv of another width than its peers). Unset/0: nothing below is ever called.
def _p035_config():
    import os as _p035_os

    gate = _p035_os.environ.get("VLLM_GB10_IDX_COMPACT_XCHG", "0").strip()
    if gate in ("", "0"):
        return False, 0
    if gate != "1":
        raise ValueError(f"VLLM_GB10_IDX_COMPACT_XCHG={gate!r} must be unset/0 (inert) or 1")
    raw = _p035_os.environ.get("VLLM_GB10_IDX_COMPACT_XCHG_VERIFY", "").strip()
    if not raw:
        return True, 3
    if not (raw.isascii() and raw.isdigit()) or int(raw) > 1000:
        raise ValueError(f"VLLM_GB10_IDX_COMPACT_XCHG_VERIFY={raw!r} must be an integer in [0, 1000] (0 = off)")
    return True, int(raw)


_P035_ON, _P035_VERIFY = _p035_config()
_P035_STATE = {"calls": 0, "disabled": False, "logged": set()}


def _p035_rank0():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001 - a log line must never fail a step; log from every rank rather than from none
        return True


def _p035_log0(key, level, msg, *args):
    \"\"\"A rank-0 log line; with a key, at most once per process and key.\"\"\"
    if key is not None:
        if key in _P035_STATE["logged"]:
            return
        _P035_STATE["logged"].add(key)
    if _p035_rank0():
        getattr(logger, level)(msg, *args)


def _p035_reduce_max(flag):
    \"\"\"MAX of a 0/1 flag over the TP group: a CPU int32 tensor on the group's gloo cpu_group, which stays off the compute stream (the
    pattern of upstream's TP-uniform vote in moriio_connector.py). Every rank calls this once per VERIFY call, after the same gathers.\"\"\"
    from vllm.distributed import get_tp_group

    vote = torch.tensor([int(flag)], device="cpu", dtype=torch.int32)
    torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MAX, group=get_tp_group().cpu_group)
    return int(vote.item())


def _p035_decline_reason(
    row_shard, index_kpool, positions, topk_tokens, nd, num_tokens, topk_indices_buffer, prefill_metadata
):
    \"\"\"None = every host assumption of the compact exchange holds; else (category, detail). Pure host arithmetic on values that are
    identical on every TP rank, so the decision is too (an exception is a decline, identical on every rank).\"\"\"
    try:
        if index_kpool <= 1 or positions is None:
            return "layout", f"index_kpool={index_kpool}, positions {'missing' if positions is None else 'given'}: not the served layout"
        if torch.cuda.is_current_stream_capturing():
            return "capture", "CUDA graph capture is active"
        total = sum(row_shard.sizes)
        end = nd + total
        width = topk_tokens + index_kpool - 1
        if (topk_tokens // index_kpool) * index_kpool != topk_tokens:
            return "topk", f"topk_tokens {topk_tokens} is not a multiple of index_kpool {index_kpool}"
        if num_tokens != end:
            return "tokens", f"step tokens {num_tokens} != decode tokens {nd} + sharded rows {total}"
        if row_shard.num_decode_tokens != nd or not nd <= row_shard.start <= row_shard.stop <= end:
            return "slice", f"this rank's rows [{row_shard.start}, {row_shard.stop}) are not inside the sharded rows [{nd}, {end})"
        if positions.shape[0] < end:
            return "positions", f"positions has {positions.shape[0]} rows, {end} are needed"
        if topk_indices_buffer.shape[0] < end or topk_indices_buffer.shape[1] < width:
            return "buffer", f"the top-k buffer {tuple(topk_indices_buffer.shape)} is smaller than [{end}, {width}]"
        pos = nd
        for chunk in prefill_metadata.chunks:
            if chunk.token_start != pos or chunk.token_end < chunk.token_start:
                return "chunks", f"chunk rows [{chunk.token_start}, {chunk.token_end}) do not continue at row {pos}"
            pos = chunk.token_end
        if pos != end:
            return "chunks", f"the chunks end at row {pos}, the sharded rows at {end}"
        return None
    except Exception as e:  # noqa: BLE001 - identical on every rank
        return "exception", repr(e)


class _P035Plan:
    \"\"\"One compact-exchange call on one rank: the staged pool ids, the VERIFY scratch buffer and the finishing step.\"\"\"

    def __init__(self, row_shard, index_kpool, positions, nd, end, select_k, width, pool_stage, scratch, k, ctx):
        self.row_shard = row_shard
        self.index_kpool = index_kpool
        self.positions = positions
        self.nd = nd
        self.end = end
        self.select_k = select_k
        self.width = width
        self.pool_stage = pool_stage
        self.scratch = scratch
        self.verify = scratch is not None
        self.k = k
        self.ctx = ctx

    def stage(self, row_start, row_end, pool_topk):
        \"\"\"Keep one chunk's pool ids (int32 [rows, select_k], after top-k and the DCP merge) at their global rows.\"\"\"
        if row_start < self.nd or row_end > self.end:
            raise RuntimeError(f"{TAG_} chunk rows [{row_start}, {row_end}) lie outside the sharded rows [{self.nd}, {self.end})")
        self.pool_stage[row_start - self.nd : row_end - self.nd] = pool_topk

    def finish(self, topk_indices_buffer):
        \"\"\"Gather the pool rows, expand all rows once, write [nd, end) x [0, width); a VERIFY call compares with the original path.\"\"\"
        from vllm.distributed import get_tp_group

        nd, end, width, rs = self.nd, self.end, self.width, self.row_shard
        total = end - nd
        # (a) one all_gatherv of this rank's contiguous pool slice at width select_k (the group call PrefillRowShard.exchange_topk
        # makes). The local input must outlive the async PyNCCL gather's enqueued copy (upstream's comment): `local` is released only
        # after the expansion that consumes the gathered rows has been enqueued on the same stream.
        local = self.pool_stage[rs.start - nd : rs.stop - nd].contiguous()
        gathered = get_tp_group().all_gatherv(local, dim=0, sizes=rs.sizes)
        # (b) ONE expansion over all prefill rows; the original expands per chunk and rank and gathers the expanded rows instead.
        pool_ids = gathered.to(torch.int64)
        del gathered
        q_seq = self.positions[nd:end].to(torch.int32) + 1
        expanded = kpool_ops.expand_pools_and_append_tail(pool_ids, q_seq, self.index_kpool)
        del pool_ids, local
        self.pool_stage = None
        if tuple(expanded.shape) != (total, width):
            raise RuntimeError(f"{TAG_} expanded rows have shape {tuple(expanded.shape)}, expected {(total, width)}")
        if self.scratch is None:
            # (c) the same rows, columns and buffer the original exchange fills
            topk_indices_buffer[nd:end, : expanded.shape[-1]] = expanded
            return
        self._verify(topk_indices_buffer, expanded)

    def _verify(self, topk_indices_buffer, expanded):
        nd, end, width, rs, k = self.nd, self.end, self.width, self.row_shard, self.k
        scratch, self.scratch = self.scratch, None
        # the ORIGINAL path: the loop expanded this rank's rows chunk by chunk into `scratch`; now upstream's own exchange
        rs.exchange_topk(scratch, width)
        want = scratch[nd:end, :width]
        mine = 0 if torch.equal(expanded, want) else 1
        bad = _p035_reduce_max(mine)  # the SAME decision on every rank, or the next collectives would not line up
        ctx = self.ctx if self.ctx >= 0 else int(self.positions[nd:end].max().item()) + 1
        if bad:
            _P035_STATE["disabled"] = True
            topk_indices_buffer[nd:end, : expanded.shape[-1]] = want
            detail = "a peer rank differs"
            if mine:
                detail = "this rank differs"
                try:  # a diagnostic only: it must never fail the step (the decision above is already taken on every rank)
                    diff = (expanded != want).any(1)
                    detail = "this rank differs in %d of %d rows, first at row %d" % (
                        int(diff.sum().item()), end - nd, nd + int(diff.to(torch.int32).argmax().item()))
                except Exception:  # noqa: BLE001
                    pass
            logger.error(
                "{TAG_} verify %d/%d: MISMATCH between the compact and the original exchange (%s; deepest context %d): "
                "the ORIGINAL rows are used for this call and the compact exchange is DISABLED for the rest of this process",
                k, _P035_VERIFY, detail, ctx,
            )
        else:
            topk_indices_buffer[nd:end, : expanded.shape[-1]] = expanded
            _p035_log0(None, "info", "{TAG_} verify %d/%d: bitwise equal (%d rows, deepest context %d)", k, _P035_VERIFY, end - nd, ctx)


def _p035_begin(
    row_shard, index_kpool, positions, topk_tokens, num_decode_tokens, num_tokens, topk_indices_buffer, prefill_metadata
):
    \"\"\"The compact-exchange plan of this call, or None = the original exchange runs.

    Every input is host metadata or a shape that is identical on all TP ranks, so all ranks decide alike. Only the host checks are
    guarded; the allocations and collectives are not (a rank-local fallback there would desynchronize the group).\"\"\"
    if not _P035_ON or _P035_STATE["disabled"] or row_shard is None:
        return None
    why = _p035_decline_reason(
        row_shard, index_kpool, positions, topk_tokens, num_decode_tokens, num_tokens, topk_indices_buffer, prefill_metadata
    )
    if why is not None:
        _p035_log0(why[0], "warning", "{TAG_} compact exchange declined (%s): the original exchange runs", why[1])
        return None
    nd = num_decode_tokens
    end = nd + sum(row_shard.sizes)
    select_k = topk_tokens // index_kpool
    width = topk_tokens + index_kpool - 1
    _P035_STATE["calls"] += 1
    k = _P035_STATE["calls"]
    dev = topk_indices_buffer.device
    pool_stage = torch.full((end - nd, select_k), -1, dtype=torch.int32, device=dev)
    scratch = torch.full((end, width), -1, dtype=torch.int32, device=dev) if k <= _P035_VERIFY else None
    if k == 1:
        _p035_log0(None, "info", "{TAG_} compact exchange ACTIVE (width %d instead of %d)", select_k, width)
    return _P035Plan(
        row_shard, index_kpool, positions, nd, end, select_k, width, pool_stage, scratch, k, prefill_metadata.max_prefill_seq_len
    )"""
).replace("{TAG_}", TAG)
A_NEW = BLOCK + "\n\n\n" + A_OLD

# ---------------------------------------------------------------------------------------------------------------------
# Edit B: the plan, before the chunk loop
# ---------------------------------------------------------------------------------------------------------------------
B_OLD = "        for chunk in prefill_metadata.chunks if not short_prefill else ():\n"
B_NEW = (
    f"        {MARKER} gate on and a sharded step in the served layout: stage the POOL ids per chunk, exchange them (width\n"
    "        # select_k) and expand all rows once after the gather. _p035_out is where the ORIGINAL per-chunk expansion writes: the real\n"
    "        # buffer, or the scratch buffer in a VERIFY call. Gate off / no plan: the original statements, unchanged.\n"
    "        _p035_plan = (\n"
    "            _p035_begin(\n"
    "                row_shard,\n"
    "                index_kpool,\n"
    "                positions,\n"
    "                topk_tokens,\n"
    "                num_decode_tokens,\n"
    "                num_tokens,\n"
    "                topk_indices_buffer,\n"
    "                prefill_metadata,\n"
    "            )\n"
    "            if _P035_ON and row_shard is not None\n"
    "            else None\n"
    "        )\n"
    "        _p035_out = topk_indices_buffer if _p035_plan is None else _p035_plan.scratch\n"
) + B_OLD

# ---------------------------------------------------------------------------------------------------------------------
# Edit C: stage the chunk's pool ids (directly before the original expansion block)
# ---------------------------------------------------------------------------------------------------------------------
C_OLD = (
    "            if index_kpool > 1:\n"
    "                pool_ids = pool_topk.to(torch.int64)\n"
)
C_NEW = (
    f"            {MARKER} compact exchange: keep this chunk's POOL ids (int32 [rows, select_k]) at their global rows instead\n"
    "            # of expanding them here; a VERIFY call also runs the original expansion below (into the scratch buffer).\n"
    "            if _p035_plan is not None:\n"
    "                _p035_plan.stage(row_start, row_end, pool_topk)\n"
    "                if not _p035_plan.verify:\n"
    "                    continue\n"
) + C_OLD

# ---------------------------------------------------------------------------------------------------------------------
# Edit D: the original expansion block writes to _p035_out
# ---------------------------------------------------------------------------------------------------------------------
D_OLD = "                topk_indices_buffer[row_start:row_end, : expanded.shape[-1]] = expanded\n"
D_NEW = (
    f"                {MARKER} == topk_indices_buffer unless this is a VERIFY call (then the scratch buffer)\n"
    "                _p035_out[row_start:row_end, : expanded.shape[-1]] = expanded\n"
)

# ---------------------------------------------------------------------------------------------------------------------
# Edit E: the exchange after the loop
# ---------------------------------------------------------------------------------------------------------------------
E_OLD = "            row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)\n"
E_NEW = (
    f"            {MARKER} compact plan: gather the pool ids (width select_k) and expand all rows once (a VERIFY call\n"
    "            # compares with the original exchange); no plan: upstream's exchange, unchanged.\n"
    "            if _p035_plan is not None:\n"
    "                _p035_plan.finish(topk_indices_buffer)\n"
    "            else:\n"
    "                row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)\n"
)

EDITS = [
    ("helper block (before the op function)", A_OLD, A_NEW),
    ("plan (before the chunk loop)", B_OLD, B_NEW),
    ("stage + continue (chunk loop, before the expansion block)", C_OLD, C_NEW),
    ("expansion write target (_p035_out)", D_OLD, D_NEW),
    ("exchange (after the chunk loop)", E_OLD, E_NEW),
]


class Refused(ValueError):
    pass


def _fn(tree: ast.Module, name: str) -> ast.FunctionDef:
    hits = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(hits) != 1:
        raise Refused(f"module-level def {name} found {len(hits)} times (expected 1)")
    return hits[0]


def _span(text: str, node) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return sum(map(len, lines[: first - 1])), sum(map(len, lines[: node.end_lineno]))


def _same(node, src: str) -> bool:
    """node == the statement/expression `src` parsed by this very interpreter (so it is stable across Python versions)."""
    want = ast.parse(src).body[0]
    if isinstance(node, ast.expr) and isinstance(want, ast.Expr):
        want = want.value
    return ast.dump(node) == ast.dump(want)


def _has_prefill_if(fn: ast.FunctionDef) -> ast.If:
    ifs = [n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "has_prefill"]
    if len(ifs) != 1:
        raise Refused(f"expected exactly one `if has_prefill:` at function level in {FUNC}, found {len(ifs)}")
    return ifs[0]


def _chunk_loop(body: list) -> ast.For:
    loops = [n for n in body if isinstance(n, ast.For) and "prefill_metadata.chunks" in ast.unparse(n.iter)]
    if len(loops) != 1:
        raise Refused(f"expected exactly one chunk loop directly inside `if has_prefill:`, found {len(loops)}")
    return loops[0]


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    fn = _fn(tree, FUNC)
    if not patched:
        if "_p035" in text or "_P035" in text:
            raise Refused("base already mentions _p035")
        a, b = _span(text, fn)
        for name, old, _ in EDITS[1:]:
            if not a <= text.find(old) < b:
                raise Refused(f"the '{name}' anchor is not inside {FUNC}")
        if text.find(A_OLD) != a:
            raise Refused(f"the helper-block anchor is not the decorator + def line of {FUNC}")
        hp = _has_prefill_if(fn)
        loop = _chunk_loop(hp.body)
        if not any(n is loop for n in hp.body):
            raise Refused("chunk loop is not a direct child of `if has_prefill:`")
        # the expansion block: the LAST statement of the loop body, `if index_kpool > 1:` whose first statement is pool_ids = ...
        last = loop.body[-1]
        if not (isinstance(last, ast.If) and ast.unparse(last.test) == "index_kpool > 1"
                and _same(last.body[0], "pool_ids = pool_topk.to(torch.int64)")):
            raise Refused("the loop does not end with the `if index_kpool > 1: pool_ids = pool_topk.to(torch.int64)` expansion block")
        if not _same(last.body[-1], "topk_indices_buffer[row_start:row_end, : expanded.shape[-1]] = expanded"):
            raise Refused("the expansion block does not end with the write into topk_indices_buffer")
        # the exchange: `if row_shard is not None:` directly after the loop, whose last statement is the exchange call
        k = hp.body.index(loop)
        rs_if = hp.body[k + 1] if k + 1 < len(hp.body) else None
        if not (isinstance(rs_if, ast.If) and ast.unparse(rs_if.test) == "row_shard is not None"
                and _same(rs_if.body[-1], "row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)")):
            raise Refused("`if row_shard is not None:` + row_shard.exchange_topk(...) must directly follow the chunk loop")
        # the decorator + def sits at module level after _kpool_compress_insert (the helper block goes in between)
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        if "_kpool_compress_insert" not in names or names.index("_kpool_compress_insert") + 1 != names.index(FUNC):
            raise Refused(f"{FUNC} must directly follow _kpool_compress_insert")
        return
    # ---- patched
    body_names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for want in HELPER_DEFS + ["_P035Plan", FUNC]:
        if body_names.count(want) != 1:
            raise Refused(f"expected exactly one top-level def/class {want}, found {body_names.count(want)}")
    kf = body_names.index(FUNC)
    for want in HELPER_DEFS + ["_P035Plan"]:
        if body_names.index(want) > kf:
            raise Refused(f"{want} must be defined before {FUNC}")
    if body_names.index("_p035_config") > body_names.index("_p035_begin"):
        raise Refused("the helper block is out of order")
    cfg = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Tuple) and [e.id for e in t.elts if isinstance(e, ast.Name)] == ["_P035_ON", "_P035_VERIFY"]
        for t in n.targets)]
    if len(cfg) != 1 or ast.unparse(cfg[0].value) != "_p035_config()":
        raise Refused("expected `_P035_ON, _P035_VERIFY = _p035_config()` once at module level")
    state = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_P035_STATE"]
    if len(state) != 1:
        raise Refused("expected one module-level _P035_STATE")
    # class members
    plan_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_P035Plan")
    members = [n.name for n in plan_cls.body if isinstance(n, ast.FunctionDef)]
    if members != ["__init__", "stage", "finish", "_verify"]:
        raise Refused(f"_P035Plan members are {members}")
    # hook B: the two assignments directly before the chunk loop
    hp = _has_prefill_if(fn)
    loop = _chunk_loop(hp.body)
    k = hp.body.index(loop)
    if k < 2:
        raise Refused("the plan assignments must directly precede the chunk loop")
    plan_asg, out_asg = hp.body[k - 2], hp.body[k - 1]
    if not (isinstance(plan_asg, ast.Assign) and [ast.unparse(t) for t in plan_asg.targets] == ["_p035_plan"]):
        raise Refused("`_p035_plan = ...` must be the second statement before the chunk loop")
    v = plan_asg.value
    if not (isinstance(v, ast.IfExp) and ast.unparse(v.test) == "_P035_ON and row_shard is not None"
            and ast.unparse(v.orelse) == "None" and isinstance(v.body, ast.Call)
            and ast.unparse(v.body.func) == "_p035_begin"
            and [ast.unparse(a) for a in v.body.args] == [
                "row_shard", "index_kpool", "positions", "topk_tokens", "num_decode_tokens", "num_tokens",
                "topk_indices_buffer", "prefill_metadata"]):
        raise Refused("unexpected `_p035_plan = ...` expression")
    if not _same(out_asg, "_p035_out = topk_indices_buffer if _p035_plan is None else _p035_plan.scratch"):
        raise Refused("`_p035_out = ...` must directly precede the chunk loop")
    # hook C/D: inside the loop body
    lb = loop.body
    ci = [i for i, n in enumerate(lb) if isinstance(n, ast.If) and ast.unparse(n.test) == "_p035_plan is not None"]
    if len(ci) != 1 or ci[0] != len(lb) - 2:
        raise Refused("the staging `if _p035_plan is not None:` must be the second-to-last statement of the chunk loop")
    st = lb[ci[0]]
    if not (len(st.body) == 2 and not st.orelse
            and _same(st.body[0], "_p035_plan.stage(row_start, row_end, pool_topk)")
            and isinstance(st.body[1], ast.If) and ast.unparse(st.body[1].test) == "not _p035_plan.verify"
            and len(st.body[1].body) == 1 and isinstance(st.body[1].body[0], ast.Continue) and not st.body[1].orelse):
        raise Refused("the staging block must be `stage(...)` then `if not _p035_plan.verify: continue`")
    last = lb[-1]
    if not (isinstance(last, ast.If) and ast.unparse(last.test) == "index_kpool > 1"
            and _same(last.body[0], "pool_ids = pool_topk.to(torch.int64)")):
        raise Refused("the staging block must directly precede the original expansion block")
    if not _same(last.body[-1], "_p035_out[row_start:row_end, : expanded.shape[-1]] = expanded"):
        raise Refused("the expansion block must end with the write into _p035_out")
    # hook E: the exchange
    rs_if = hp.body[k + 1] if k + 1 < len(hp.body) else None
    if not (isinstance(rs_if, ast.If) and ast.unparse(rs_if.test) == "row_shard is not None"):
        raise Refused("`if row_shard is not None:` must directly follow the chunk loop")
    ex = rs_if.body[-1]
    if not (isinstance(ex, ast.If) and ast.unparse(ex.test) == "_p035_plan is not None" and len(ex.body) == 1
            and _same(ex.body[0], "_p035_plan.finish(topk_indices_buffer)") and len(ex.orelse) == 1
            and _same(ex.orelse[0], "row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)")):
        raise Refused("the exchange must be `if _p035_plan is not None: _p035_plan.finish(...) else: row_shard.exchange_topk(...)`")
    # every use of a _p035/_P035 name inside the op function sits inside one of the five hook statements
    in_hook = {id(x) for hook in (plan_asg, out_asg, st, last.body[-1], ex) for x in ast.walk(hook)}
    stray = [u.id for u in ast.walk(fn) if isinstance(u, ast.Name) and u.id.startswith(("_p035", "_P035")) and id(u) not in in_hook]
    if stray:
        raise Refused(f"{sorted(set(stray))} used in {FUNC} outside the five hook statements")
    # nothing else may use the helpers
    inside = set()
    for f in [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (n.name in HELPER_DEFS or n.name == "_P035Plan")]:
        inside.update(id(x) for x in ast.walk(f))
    users = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id.startswith(("_p035", "_P035")) and id(n) not in inside]
    allowed = {"_P035_ON", "_P035_VERIFY", "_P035_STATE", "_p035_config", "_p035_begin", "_p035_plan", "_p035_out"}
    bad = sorted({u.id for u in users if u.id not in allowed})
    if bad:
        raise Refused(f"unexpected uses of {bad} outside the helper block")
    outside_fn = [u for u in users if not any(u is x for x in ast.walk(fn)) and u.id not in ("_P035_ON", "_P035_VERIFY", "_P035_STATE", "_p035_config")]
    if outside_fn:
        raise Refused("a _p035 name is used at module level outside the helper block")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-035 marker")
    bad = [(name, text.count(old)) for name, old, _ in EDITS if text.count(old) != 1]
    if bad:
        raise Refused("; ".join(f"anchor '{n}' found {c}x (expected 1)" for n, c in bad))
    check_structure(text, patched=False)
    out = text
    for name, old, new in EDITS:
        if out.count(old) != 1:
            raise Refused(f"anchor '{name}' changed by an earlier edit")
        out = out.replace(old, new, 1)
    if out.count(MARKER) != N_MARKERS:
        raise Refused(f"expected {N_MARKERS} markers after the edits, found {out.count(MARKER)}")
    compile(out, "sparse_indexer.py", "exec")
    check_structure(out, patched=True)
    return out


def revert(text: str) -> str:
    out = text
    for name, old, new in reversed(EDITS):
        if out.count(new) != 1:
            raise Refused(f"patched block '{name}' found {out.count(new)}x (expected 1)")
        out = out.replace(new, old, 1)
    if MARKER in out:
        raise Refused("marker still present after revert")
    compile(out, "sparse_indexer.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p035_apply.py {apply|check|revert} <sparse_indexer.py>", file=sys.stderr)
        return 2
    p = Path(argv[2])
    try:
        src = p.read_text(encoding="utf-8")
        new = revert(src) if argv[1] == "revert" else apply(src)
    except (Refused, SyntaxError, OSError) as e:
        print(f"[FAIL] {p}: {e}", file=sys.stderr)
        return 1
    if argv[1] == "check":
        print(_sha(new.encode()))
        return 0
    tmp = p.with_name(p.name + ".p035.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if argv[1] == "revert" else "applied"
    print(f" [OK] {len(EDITS)} anchored edits {what} on {p.name} (helper block before {FUNC}; plan before the chunk loop; "
          f"stage + continue and the write target in the loop; exchange after the loop)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
