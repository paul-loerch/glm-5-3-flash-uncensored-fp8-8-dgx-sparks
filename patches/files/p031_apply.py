#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 031 (v0.32.0 port) applier: four anchored edits to vllm/v1/attention/backends/mla/indexer.py (written here).

Why this patch is small: vLLM v0.32.0 already contains the TP-sharded indexer prefill (#54951): the metadata builder splits the
prefill rows of a step into contiguous, work-balanced per-rank row counts (`balanced_prefill_row_shard`), every rank scores only
its slice (`PrefillRowShard.narrow` in the GLM5Next indexer op) and the group exchanges the final int32 top-k indices
(`PrefillRowShard.exchange_topk` = TP all_gatherv). Upstream's gate is `rows >= MIN_TP_SHARD_ROWS_PER_RANK (16,384) x TP` in ONE
step -- with --max-num-batched-tokens 8192 it never fires. This patch swaps ONLY that gate for the depth gate the old patch 031
measured: shard when the deepest prefill request of the step has context >= MIN_CTX (VLLM_GB10_IDX_SHARD_MIN_CTX,
default 32,768) and the step has >= MIN_ROWS indexed prefill rows (VLLM_GB10_IDX_SHARD_MIN_ROWS, default 256). Upstream main's
`attention_config.indexer_tp_shard_min_rows` (#61647, default 65,536 TOTAL rows, `rows >= max(min_rows, tp)`) is the row half of the
same idea; it has no depth half, so this stays an env-gated patch on v0.32.0.

  apply(text) -> text     the pristine v0.32.0 file -> + patch 031; fail-closed: each of the 4 anchors must match exactly once,
                          every hook must land inside its function (AST-checked), the result must carry N_MARKERS markers and
                          compile
  revert(text) -> text    exact inverse

CLI (used by 031-gb10-idx-shard-gate.sh):
  p031_apply.py apply <indexer.py>    in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p031_apply.py check <indexer.py>    dry run: print the sha256 the file would have after apply()
  p031_apply.py revert <indexer.py>   in-place inverse (tests / manual rollback)

Edits (line numbers of the pristine v0.32.0 file, sha256 aabcb7f2...):
  A  :206-208  after `MIN_TP_SHARD_ROWS_PER_RANK = 16_384`: the gate block -- _p031_config() (env, read ONCE at import, validated:
               a typo raises), _P031_ON / _P031_MIN_CTX / _P031_MIN_ROWS, _p031_row_gate(), _p031_announce(), _p031_note_engaged().
  B  :219      in balanced_prefill_row_shard: `num_rows < MIN_TP_SHARD_ROWS_PER_RANK * tp_size` -> `not _p031_row_gate(...)`.
  C  :1043     DeepseekV32IndexerMetadataBuilder.__init__, before `sm_count = num_compute_units(...)`: _p031_announce(...) = the
               one-per-rank boot line ("[patch-031] indexer prefill row-shard ACTIVE (rank r/8 ..." or "... INACTIVE (rank r/8): why").
  D  :242      balanced_prefill_row_shard's return: keep `sizes`, call _p031_note_engaged() (a no-op unless the gate is on: logs the
               FIRST sharded step once per process), return sizes.
Unset / 0 gate: _p031_row_gate() returns upstream's own decision (`rows >= 16,384 x tp`) and nothing is logged -- the function is
bit-for-bit upstream's (files/p031_cpu_check.py G1 compares 4,000 random inputs against the unpatched function).
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-031 gb10-idx-shard-gate]"
N_MARKERS = 4
ENV = "VLLM_GB10_IDX_SHARD"
FUNC = "balanced_prefill_row_shard"
BUILDER = "DeepseekV32IndexerMetadataBuilder"
TAG = "[patch-031]"

# ---------------------------------------------------------------------------------------------------------------------
# Edit A: module-level gate block (placed directly after MIN_TP_SHARD_ROWS_PER_RANK, before balanced_prefill_row_shard)
# ---------------------------------------------------------------------------------------------------------------------
A_OLD = (
    "# Below this many rows per rank, stay replicated (TP4 engages at 64K rows).\n"
    "MIN_TP_SHARD_ROWS_PER_RANK = 16_384\n"
)
A_NEW = A_OLD + f'''

{MARKER} opt-in {ENV}=1 (read once, at import): a DEPTH gate on upstream's TP-sharded indexer
# prefill. Upstream shards the prefill rows of a step across the TP group only from MIN_TP_SHARD_ROWS_PER_RANK x tp rows in ONE
# step, which never happens at --max-num-batched-tokens 8192. With the gate on, a step is sharded when its deepest prefill request
# has context >= MIN_CTX (VLLM_GB10_IDX_SHARD_MIN_CTX, default 32768 tokens, 4096..4194304) and the step has >= MIN_ROWS indexed
# prefill rows (VLLM_GB10_IDX_SHARD_MIN_ROWS, default 256, 64..8192). The split itself, PrefillRowShard and the all_gatherv
# exchange are upstream's. Unset/0: upstream's own gate, unchanged. Any other value, or a bad integer: ValueError at import.
def _p031_config():
    import os as _p031_os

    gate = _p031_os.environ.get("{ENV}", "0").strip()
    if gate in ("", "0"):
        return False, 32768, 256
    if gate != "1":
        raise ValueError(f"{ENV}={{gate!r}} must be unset/0 (inert) or 1")

    def _int(name, default, lo, hi):
        raw = _p031_os.environ.get(name, "").strip()
        if not raw:
            return default
        if not raw.isdigit() or not lo <= int(raw) <= hi:
            raise ValueError(f"{{name}}={{raw!r}} must be an integer in [{{lo}}, {{hi}}]")
        return int(raw)

    return (
        True,
        _int("VLLM_GB10_IDX_SHARD_MIN_CTX", 32768, 4096, 4194304),
        _int("VLLM_GB10_IDX_SHARD_MIN_ROWS", 256, 64, 8192),
    )


_P031_ON, _P031_MIN_CTX, _P031_MIN_ROWS = _p031_config()
_P031_STATE = {{"announced": False, "engaged": False, "failed": False}}


def _p031_row_gate(num_rows, seq_lens_cpu, tp_size):
    """True = shard this step's prefill rows. Gate off: upstream's decision. Never raises (a failure keeps the replicated path)."""
    if not _P031_ON:
        return num_rows >= MIN_TP_SHARD_ROWS_PER_RANK * tp_size
    try:
        if num_rows < max(_P031_MIN_ROWS, tp_size):
            return False
        return int(seq_lens_cpu.max()) >= _P031_MIN_CTX if seq_lens_cpu.numel() else False
    except Exception as e:  # noqa: BLE001 - the replicated path is always valid; the decision is identical on every rank
        if not _P031_STATE["failed"]:
            _P031_STATE["failed"] = True
            logger.warning("{TAG} depth gate failed (%r); the replicated indexer prefill stays", e)
        return False


def _p031_announce(vllm_config, dcp_world_size, use_pcp, tp_size):
    """One boot line per process (the worker of this rank): ACTIVE, or INACTIVE with the reason. Never raises."""
    if not _P031_ON or _P031_STATE["announced"]:
        return
    _P031_STATE["announced"] = True
    try:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            rank = get_tensor_model_parallel_rank()
        except Exception:  # noqa: BLE001
            rank = "?"
        if tp_prefill_row_sharding_supported(vllm_config, dcp_world_size, use_pcp, tp_size):
            logger.info(
                "{TAG} indexer prefill row-shard ACTIVE (rank %s/%d; context >= %d tokens and >= %d rows, eager prefill only; "
                "upstream v0.32 PrefillRowShard + TP all_gatherv exchange with a work-balanced contiguous split; upstream's "
                "gate of %d rows per rank is replaced)",
                rank, tp_size, _P031_MIN_CTX, _P031_MIN_ROWS, MIN_TP_SHARD_ROWS_PER_RANK,
            )
            return
        why = []
        if tp_size < 2:
            why.append("tensor_parallel_size < 2")
        if dcp_world_size != 1:
            why.append("decode context parallel")
        if use_pcp:
            why.append("prefill context parallel")
        if not current_platform.is_cuda():
            why.append("not CUDA")
        for name in ("VLLM_DISABLE_PYNCCL", "VLLM_USE_NCCL_SYMM_MEM", "VLLM_BATCH_INVARIANT"):
            if getattr(envs, name, False):
                why.append(name)
        mode = vllm_config.compilation_config.cudagraph_mode or CUDAGraphMode.NONE
        if mode.mixed_mode() == CUDAGraphMode.FULL:
            why.append("cudagraph mixed mode is FULL")
        logger.warning(
            "{TAG} indexer prefill row-shard INACTIVE (rank %s/%d): upstream tp_prefill_row_sharding_supported() is False (%s); "
            "the replicated indexer prefill stays",
            rank, tp_size, ", ".join(why) or "reason not identified",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("{TAG} indexer prefill row-shard INACTIVE (rank ?/%s): announce failed (%r)", tp_size, e)


def _p031_note_engaged(num_rows, seq_lens_cpu, sizes):
    """Log the first sharded step of this process once (rows, per-rank split, deepest context). Never raises."""
    if not _P031_ON or _P031_STATE["engaged"]:
        return
    _P031_STATE["engaged"] = True
    try:
        logger.info(
            "{TAG} first sharded indexer prefill step: %d rows -> per-rank rows %s, deepest context %d tokens",
            num_rows, list(sizes), int(seq_lens_cpu.max()),
        )
    except Exception:  # noqa: BLE001
        pass
'''

# ---------------------------------------------------------------------------------------------------------------------
# Edit B: the gate inside balanced_prefill_row_shard
# ---------------------------------------------------------------------------------------------------------------------
B_OLD = (
    "    num_rows = int(query_lens_cpu.sum())\n"
    "    if tp_size < 2 or num_rows < MIN_TP_SHARD_ROWS_PER_RANK * tp_size:\n"
    "        return None\n"
)
B_NEW = (
    "    num_rows = int(query_lens_cpu.sum())\n"
    f"    {MARKER} upstream: num_rows < MIN_TP_SHARD_ROWS_PER_RANK * tp_size. Gate on: context >= MIN_CTX\n"
    "    # and rows >= MIN_ROWS (_p031_row_gate); unset: exactly upstream's decision.\n"
    "    if tp_size < 2 or not _p031_row_gate(num_rows, seq_lens_cpu, tp_size):\n"
    "        return None\n"
)

# ---------------------------------------------------------------------------------------------------------------------
# Edit C: the boot line, DeepseekV32IndexerMetadataBuilder.__init__
# ---------------------------------------------------------------------------------------------------------------------
C_OLD = (
    "        sm_count = num_compute_units(self.device.index)\n"
    "        self.num_sms = sm_count\n"
)
C_NEW = (
    f"        {MARKER} gate on: one ACTIVE / INACTIVE boot line per rank (worker process).\n"
    "        _p031_announce(self.vllm_config, self.dcp_world_size, self.use_pcp, self.tp_size)\n"
    "\n" + C_OLD
)

# ---------------------------------------------------------------------------------------------------------------------
# Edit D: the first sharded step is logged once (a no-op unless the gate is on)
# ---------------------------------------------------------------------------------------------------------------------
D_OLD = "    return [bounds[i + 1] - bounds[i] for i in range(tp_size)]\n"
D_NEW = (
    "    sizes = [bounds[i + 1] - bounds[i] for i in range(tp_size)]\n"
    f"    {MARKER} gate on: log the first sharded step of this process once (no-op otherwise).\n"
    "    _p031_note_engaged(num_rows, seq_lens_cpu, sizes)\n"
    "    return sizes\n"
)

EDITS = [
    ("gate block (after MIN_TP_SHARD_ROWS_PER_RANK)", A_OLD, A_NEW),
    ("row gate (balanced_prefill_row_shard)", B_OLD, B_NEW),
    ("boot line (DeepseekV32IndexerMetadataBuilder.__init__)", C_OLD, C_NEW),
    ("first-step log (balanced_prefill_row_shard return)", D_OLD, D_NEW),
]


class Refused(ValueError):
    pass


def _fn(tree: ast.Module, name: str) -> ast.FunctionDef:
    hits = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(hits) != 1:
        raise Refused(f"module-level def {name} found {len(hits)} times (expected 1)")
    return hits[0]


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    cl = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    if len(cl) != 1:
        raise Refused(f"class {cls} found {len(cl)} times (expected 1)")
    ms = [n for n in cl[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(ms) != 1:
        raise Refused(f"{cls}.{name} found {len(ms)} times (expected 1)")
    return ms[0]


def _span(text: str, node) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[: node.lineno - 1])), sum(map(len, lines[: node.end_lineno]))


def _calls(node, name: str) -> list:
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name]


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    fn = _fn(tree, FUNC)
    init = _method(tree, BUILDER, "__init__")
    if not patched:
        if "_p031" in text or "_P031" in text:
            raise Refused("base already mentions _p031")
        a, b = _span(text, fn)
        for name, old in (("row-gate anchor", B_OLD), ("return anchor", D_OLD)):
            if not a <= text.find(old) < b:
                raise Refused(f"the {name} is not inside {FUNC}")
        a, b = _span(text, init)
        if not a <= text.find(C_OLD) < b:
            raise Refused(f"the sm_count anchor is not inside {BUILDER}.__init__")
        consts = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "MIN_TP_SHARD_ROWS_PER_RANK"]
        if len(consts) != 1:
            raise Refused("expected one module-level MIN_TP_SHARD_ROWS_PER_RANK")
        if tree.body.index(consts[0]) + 1 != tree.body.index(fn):
            raise Refused(f"MIN_TP_SHARD_ROWS_PER_RANK must directly precede def {FUNC}")
        return
    # ---- patched: the gate block sits between the constant and balanced_prefill_row_shard
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    need = ["_p031_config", "_p031_row_gate", "_p031_announce", "_p031_note_engaged", FUNC]
    for want in need:
        if names.count(want) != 1:
            raise Refused(f"expected exactly one def {want}, found {names.count(want)}")
    k0 = tree.body.index(_fn(tree, "_p031_config"))
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "MIN_TP_SHARD_ROWS_PER_RANK"]
    if len(consts) != 1 or tree.body.index(consts[0]) + 1 != k0:
        raise Refused("the gate block must directly follow MIN_TP_SHARD_ROWS_PER_RANK")
    if k0 > tree.body.index(fn):
        raise Refused(f"the gate block must precede def {FUNC}")
    cfg_assign = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Tuple) and [e.id for e in t.elts if isinstance(e, ast.Name)] == ["_P031_ON", "_P031_MIN_CTX", "_P031_MIN_ROWS"]
        for t in n.targets)]
    if len(cfg_assign) != 1 or ast.unparse(cfg_assign[0].value) != "_p031_config()":
        raise Refused("expected `_P031_ON, _P031_MIN_CTX, _P031_MIN_ROWS = _p031_config()` once at module level")
    # hook B: exactly one _p031_row_gate call, in the gate `if` of balanced_prefill_row_shard
    g = _calls(tree, "_p031_row_gate")
    if len(g) != 1 or not any(c is g[0] for c in ast.walk(fn)):
        raise Refused(f"expected exactly one _p031_row_gate call, inside {FUNC}")
    if ast.unparse(g[0]) != "_p031_row_gate(num_rows, seq_lens_cpu, tp_size)":
        raise Refused(f"unexpected gate call {ast.unparse(g[0])}")
    ifs = [n for n in fn.body if isinstance(n, ast.If) and any(c is g[0] for c in ast.walk(n.test))]
    if len(ifs) != 1 or ast.unparse(ifs[0].test) != "tp_size < 2 or not _p031_row_gate(num_rows, seq_lens_cpu, tp_size)":
        raise Refused("the gate call must be the second operand of `if tp_size < 2 or not ...:` at function level")
    if len(ifs[0].body) != 1 or not isinstance(ifs[0].body[0], ast.Return) or ast.unparse(ifs[0].body[0]) != "return None":
        raise Refused("the gate `if` must be followed by `return None`")
    # hook D: one _p031_note_engaged call, directly before the final `return sizes`
    d = _calls(tree, "_p031_note_engaged")
    if len(d) != 1 or not any(c is d[0] for c in ast.walk(fn)):
        raise Refused(f"expected exactly one _p031_note_engaged call, inside {FUNC}")
    if ast.unparse(d[0]) != "_p031_note_engaged(num_rows, seq_lens_cpu, sizes)":
        raise Refused(f"unexpected note call {ast.unparse(d[0])}")
    tail = fn.body[-3:]
    if not (len(tail) == 3 and ast.unparse(tail[0]) == "sizes = [bounds[i + 1] - bounds[i] for i in range(tp_size)]"
            and isinstance(tail[1], ast.Expr) and tail[1].value is d[0]
            and isinstance(tail[2], ast.Return) and ast.unparse(tail[2]) == "return sizes"):
        raise Refused("the function must end `sizes = [...]; _p031_note_engaged(...); return sizes`")
    # hook C: one _p031_announce call in the builder __init__, before `sm_count = ...`
    c = _calls(tree, "_p031_announce")
    if len(c) != 1 or not any(x is c[0] for x in ast.walk(init)):
        raise Refused(f"expected exactly one _p031_announce call, inside {BUILDER}.__init__")
    if ast.unparse(c[0]) != "_p031_announce(self.vllm_config, self.dcp_world_size, self.use_pcp, self.tp_size)":
        raise Refused(f"unexpected announce call {ast.unparse(c[0])}")
    sm = [n for n in init.body if isinstance(n, ast.Assign) and ast.unparse(n) == "sm_count = num_compute_units(self.device.index)"]
    ann = [n for n in init.body if isinstance(n, ast.Expr) and n.value is c[0]]
    if len(sm) != 1 or len(ann) != 1 or init.body.index(ann[0]) + 1 != init.body.index(sm[0]):
        raise Refused("_p031_announce must be the statement directly before `sm_count = ...` in __init__")
    # nothing else may use the helpers
    users = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id.startswith(("_p031", "_P031"))]
    inside = {id(x) for f in (_fn(tree, "_p031_config"), _fn(tree, "_p031_row_gate"), _fn(tree, "_p031_announce"),
                              _fn(tree, "_p031_note_engaged")) for x in ast.walk(f)}
    outside = [u for u in users if id(u) not in inside]
    allowed = {"_P031_ON", "_P031_MIN_CTX", "_P031_MIN_ROWS", "_P031_STATE", "_p031_config", "_p031_row_gate",
               "_p031_announce", "_p031_note_engaged"}
    if any(u.id not in allowed for u in outside):
        raise Refused("a _p031 name is used outside the four hooks")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-031 marker")
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
    compile(out, "indexer.py", "exec")
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
    compile(out, "indexer.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p031_apply.py {apply|check|revert} <indexer.py>", file=sys.stderr)
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
    tmp = p.with_name(p.name + ".p031.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    what = "reverted" if argv[1] == "revert" else "applied"
    print(f" [OK] {len(EDITS)} anchored edits {what} on {p.name} (gate block after MIN_TP_SHARD_ROWS_PER_RANK; row gate in {FUNC}; "
          f"boot line in {BUILDER}.__init__; first-step log)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
