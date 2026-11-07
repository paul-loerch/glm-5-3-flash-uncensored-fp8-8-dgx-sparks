#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-035 gb10-idx-compact-xchg] CPU checks (python3 + numpy; no torch, no GPU, no vLLM install).

  python3 files/p035_cpu_check.py [--files <dir with the image's sparse_indexer.py, kpool_compress.py, sparse_attn_indexer.py>]
                                  [--tree <v0.32.0 vllm/ package dir>] [--quick] [--soak N]

--files defaults to $IMAGE_FILES, else image-files (the three files exactly as the SM121-baked image carries them); --tree (only for the
provenance check P9) to $VLLM_TREE, else vllm-v0.32.0/vllm. Sections:
  S  applier / static: pin, 5 anchors exactly once, AST placement, markers, compile (also with feature_version 3.12), revert(apply(x)) ==
     x, idempotence of the transformation, only the 5 edit regions differ, refusal of every mutated / deleted / duplicated / moved anchor
     and of broken patched structures, the CLI (apply / check / revert, atomic write, mode kept)
  K  the expand kernel: a numpy model written from the Triton source (vectorised) against a literal per-element transliteration;
     per-row independence; the kernel SOURCE (baked kpool_compress.py) really has the per-row structure and lines the model assumes
  E  exactness on random layouts with the numpy model alone: "expand all gathered rows once" == "expand per chunk per rank, then
     gather" for random shards, random -1 rows, random sizes, multi-chunk layouts, decode offsets; plus the negative control that shows
     why the chunks-tile guard exists (rows outside every chunk)
  F  gate unset / 0 / empty: the REAL patched sparse_attn_indexer_kpool, run on 2-8 rank threads (numpy-backed fake torch, the REAL
     PrefillRowShard), behaves exactly like the unpatched function: buffers, kernel launches, collectives, log, helpers never called
  C  gate on: the compact path's final buffers are bitwise the original's on random layouts (ratio 2/4/8, top-k 32..2048, 2-8 ranks,
     decode offset, sub-chunks, size-1 ranks); one gather of width select_k, one expand over all rows, no per-chunk expansion
  V  VERIFY: counts, log lines (rank 0 only), cost; faults injected on one rank -> the same decision on every rank, the original rows,
     the compact path off for the rest of the process; a racy top-k order does not cause false mismatches
  D  declines (chunks do not tile, step tokens != decode + rows, top-k not a multiple of kpool, capture, index_kpool 1, positions None,
     short prefill, no row shard): the original path runs, once-per-reason warning on rank 0
  W  env parsing of the helper block
  M  mutants of the patched code: every one must make some check fail (the checks have teeth)
  P  pins of 035-gb10-idx-compact-xchg.sh / pins-035.txt, documented log lines, memory model, provenance of the baked files
Exit 1 on any [FAIL].
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p035_apply as A  # noqa: E402
import p035_testlib as TL  # noqa: E402

DEFAULT_FILES = Path(os.environ.get("IMAGE_FILES", "image-files"))
DEFAULT_TREE = Path(os.environ.get("VLLM_TREE", "vllm-v0.32.0/vllm"))
BAKE_SCRIPT = HERE.parent.parent / "image" / "glm53_sm121_v032.py"
SH = HERE.parent / "035-gb10-idx-compact-xchg.sh"
PINS = HERE.parent / "pins-035.txt"
GATE = "VLLM_GB10_IDX_COMPACT_XCHG"
VERIFY_ENV = "VLLM_GB10_IDX_COMPACT_XCHG_VERIFY"

N = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    if cond:
        N["pass"] += 1
        print(f"[PASS] {name}")
    else:
        N["fail"] += 1
        print(f"[FAIL] {name} {detail}")
    return bool(cond)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def env_on(verify=None):
    e = {GATE: "1"}
    if verify is not None:
        e[VERIFY_ENV] = str(verify)
    return e


class Ctx:
    """The three real files and a runner bound to them."""

    def __init__(self, files: Path):
        self.files = files
        self.base = (files / "sparse_indexer.py").read_text()
        self.op = (files / "sparse_attn_indexer.py").read_text()
        self.kp = (files / "kpool_compress.py").read_text()

    def run(self, text, scens, tp, **kw):
        return TL.run_world(text, self.op, scens, tp, **kw)


# ---------------------------------------------------------------------------------------------------------------------
# helpers on run results
# ---------------------------------------------------------------------------------------------------------------------
def same_bufs(ref, got):
    if len(ref.ranks) != len(got.ranks):
        return False
    for a, b in zip(ref.ranks, got.ranks):
        if len(a.bufs) != len(b.bufs):
            return False
        for x, y in zip(a.bufs, b.bufs):
            if x.shape != y.shape or not np.array_equal(x, y):
                return False
    return True


def ranks_agree(res):
    """Every rank ends with the same buffer after every call (the exchange gives all ranks all rows)."""
    base = res.ranks[0].bufs
    return all(len(r.bufs) == len(base) and all(np.array_equal(x, y) for x, y in zip(r.bufs, base)) for r in res.ranks)


def kernels(res, rank, name):
    return [e for e in res.ranks[rank].trace if e[0] == name]


def msgs(res, rank, level=None, contains=None):
    return [m for lv, m in res.ranks[rank].records if (level is None or lv == level) and (contains is None or contains in m)]


def all_msgs(res):
    return [m for r in range(len(res.ranks)) for _, m in res.ranks[r].records]


def mk(rng, tp, **kw):
    """A scenario whose chunks tile the sharded rows (retry until the random chunk builder did not drop a tiny chunk)."""
    for _ in range(50):
        s = TL.Scenario(rng, tp, **kw)
        if s.tiles():
            return s
    raise RuntimeError("could not build a tiling scenario")


def poison_begin(rank, ns):
    def boom(*a, **k):
        raise AssertionError("_p035_begin was called with the gate off")
    ns["_p035_begin"] = boom


# ---------------------------------------------------------------------------------------------------------------------
# S: applier / static
# ---------------------------------------------------------------------------------------------------------------------
def static_checks(ctx: Ctx, sh_text: str) -> str:
    base = ctx.base
    patched = A.apply(base)
    ok("S1 apply(baked v0.32.0 sparse_indexer.py) succeeds", True)
    ok("S2 patched text compiles", compile(patched, "sparse_indexer.py", "exec") is not None)
    try:
        ast.parse(patched, feature_version=(3, 12))
        ok("S2b patched text parses with the Python 3.12 grammar (the image's interpreter)", True)
    except SyntaxError as e:
        ok("S2b patched text parses with the Python 3.12 grammar", False, str(e))
    ok(f"S3 exactly {A.N_MARKERS} markers", patched.count(A.MARKER) == A.N_MARKERS)
    ok("S4 revert(apply(x)) == x byte for byte", A.revert(patched) == base)
    ok("S4b apply(revert(apply(x))) == apply(x) (the transformation is idempotent in the round trip)", A.apply(A.revert(patched)) == patched)
    for name, old, _ in A.EDITS:
        ok(f"S5 anchor '{name}' occurs exactly once in the baked file", base.count(old) == 1)
    try:
        A.apply(patched)
        ok("S6 apply on the patched result refused", False)
    except A.Refused:
        ok("S6 apply on the patched result refused (marker present)", True)

    def refuses(text, label):
        try:
            A.apply(text)
            ok(label, False, "applied")
        except (A.Refused, SyntaxError):
            ok(label, True)

    for name, old, _ in A.EDITS:
        i = base.index(old)
        mid = i + len(old) // 2
        refuses(base[:mid] + "X" + base[mid:], f"S7 mutated anchor refused: {name}")
        refuses(base.replace(old, "", 1), f"S7 deleted anchor refused: {name}")
        refuses(base + "\n" + old, f"S7 duplicated anchor refused: {name}")
    # anchors that still occur exactly once but moved out of their function / out of place
    moved = base.replace(A.E_OLD, "    pass\n", 1) + "\n\ndef _decoy(row_shard, topk_indices_buffer, topk_tokens, index_kpool):\n" + A.E_OLD
    refuses(moved, "S8a the exchange anchor moved out of sparse_attn_indexer_kpool is refused")
    moved = base.replace(A.D_OLD, "                pass\n", 1) + "\n\ndef _decoy():\n" + A.D_OLD
    refuses(moved, "S8b the expansion-write anchor moved out of the function is refused")
    moved = base.replace(A.B_OLD, "        pass\n", 1) + "\n\ndef _decoy(prefill_metadata, short_prefill):\n" + A.B_OLD + "        pass\n"
    refuses(moved, "S8c the chunk-loop anchor moved out of the function is refused")
    refuses(base.replace(A.A_OLD, "@some_other_decorator\n" + A.A_OLD, 1), "S8d a changed decorator stack above the op function is refused (the block must sit directly before it)")
    refuses("_P035_ON = 0\n" + base, "S8e a base that already mentions _P035 is refused")
    # structurally broken PATCHED files must be refused by the post-apply structure check
    def sub(old, new, count=1):
        assert patched.count(old) >= 1, old
        return patched.replace(old, new, count)

    broken = {
        "stage call removed": sub("                _p035_plan.stage(row_start, row_end, pool_topk)\n", "                pass\n"),
        "continue turned into pass": sub("                if not _p035_plan.verify:\n                    continue\n", "                if not _p035_plan.verify:\n                    pass\n"),
        "continue condition inverted": sub("if not _p035_plan.verify:", "if _p035_plan.verify:"),
        "_p035_out bound to the real buffer": sub("_p035_out = topk_indices_buffer if _p035_plan is None else _p035_plan.scratch", "_p035_out = topk_indices_buffer"),
        "expansion write back to the real buffer": sub("                _p035_out[row_start:row_end, : expanded.shape[-1]] = expanded\n", "                topk_indices_buffer[row_start:row_end, : expanded.shape[-1]] = expanded\n"),
        "finish call replaced": sub("                _p035_plan.finish(topk_indices_buffer)\n", "                pass\n"),
        "original exchange removed from the else branch": sub("            else:\n                row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)\n", "            else:\n                pass\n"),
        "plan gate without _P035_ON": sub("            if _P035_ON and row_shard is not None\n", "            if row_shard is not None\n"),
        "begin arguments reordered": sub("                index_kpool,\n                positions,\n", "                positions,\n                index_kpool,\n"),
        "second _p035_begin def": sub("def _p035_begin(", "def _p035_begin(a):\n    return None\n\n\ndef _p035_begin("),
        "config assignment replaced": sub("_P035_ON, _P035_VERIFY = _p035_config()", "_P035_ON, _P035_VERIFY = (False, 0)"),
        "a helper used inside the decode part": sub("    if has_decode:\n        decode_metadata = attn_metadata_narrowed.decode\n", "    if has_decode:\n        _p035_begin(None, 1, None, 1, 0, 0, None, None)\n        decode_metadata = attn_metadata_narrowed.decode\n"),
        "statement between the plan and the loop": sub("        _p035_out = topk_indices_buffer if _p035_plan is None else _p035_plan.scratch\n", "        _p035_out = topk_indices_buffer if _p035_plan is None else _p035_plan.scratch\n        _p035_dummy = 1\n"),
        "plan member renamed": sub("    def _verify(self, topk_indices_buffer, expanded):", "    def _verify2(self, topk_indices_buffer, expanded):"),
        "helper block after the op function": None,
    }
    i0 = patched.index(A.MARKER + " opt-in")
    i1 = patched.index(A.A_OLD)
    blk = patched[i0:i1]
    broken["helper block after the op function"] = patched[:i0] + patched[i1:] + "\n\n" + blk
    for label, text in broken.items():
        if text == patched:
            ok(f"S9 broken-structure mutant '{label}' really differs from the patched file", False)
            continue
        try:
            A.check_structure(text, patched=True)
            ok(f"S9 broken structure refused: {label}", False)
        except (A.Refused, SyntaxError):
            ok(f"S9 broken structure refused: {label}", True)
    ok("S10 check_structure accepts the genuine patched file", (A.check_structure(patched, patched=True) or True))
    # only the five edit regions differ from the baked file
    a_lines, b_lines = base.splitlines(), patched.splitlines()
    ops = [op for op in difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False).get_opcodes() if op[0] != "equal"]
    removed = sum(i2 - i1 for tag, i1, i2, j1, j2 in ops)
    ok("S11 the diff against the baked file is exactly 5 hunks (3 insertions, 2 one-line replacements), every one with a marker",
       len(ops) == 5 and [o[0] for o in ops].count("insert") == 3 and [o[0] for o in ops].count("replace") == 2
       and all(A.MARKER in "\n".join(b_lines[j1:j2]) for _, _, _, j1, j2 in ops) and removed == 2, str([(o[0], o[1], o[2]) for o in ops]))
    ok("S11b every original line of the baked file other than the 2 replaced ones is still in the patched file, in order",
       _is_subsequence(a_lines, b_lines, skip=2))
    # CLI
    cli = HERE / "p035_apply.py"
    with tempfile.TemporaryDirectory(prefix="p035cli.") as d:
        f = Path(d) / "sparse_indexer.py"
        f.write_text(base)
        os.chmod(f, 0o640)
        r = subprocess.run([sys.executable, str(cli), "check", str(f)], capture_output=True, text=True)
        ok("S12 CLI check prints the sha256 apply() would produce and leaves the file alone",
           r.returncode == 0 and r.stdout.strip() == sha(patched.encode()) and f.read_text() == base, r.stderr)
        r = subprocess.run([sys.executable, str(cli), "apply", str(f)], capture_output=True, text=True)
        ok("S12b CLI apply: exit 0, file == apply(), mode kept, no temp file left",
           r.returncode == 0 and f.read_text() == patched and (os.stat(f).st_mode & 0o7777) == 0o640
           and not list(Path(d).glob("*.p035.tmp")), r.stderr)
        r = subprocess.run([sys.executable, str(cli), "apply", str(f)], capture_output=True, text=True)
        ok("S12c CLI apply on the patched file is refused (exit 1) and leaves it alone", r.returncode == 1 and f.read_text() == patched)
        r = subprocess.run([sys.executable, str(cli), "revert", str(f)], capture_output=True, text=True)
        ok("S12d CLI revert restores the original bytes", r.returncode == 0 and f.read_text() == base, r.stderr)
        f.write_text(base.replace(A.E_OLD, A.E_OLD.replace("exchange_topk", "exchange_topk2"), 1))
        before = f.read_bytes()
        r = subprocess.run([sys.executable, str(cli), "apply", str(f)], capture_output=True, text=True)
        ok("S12e CLI apply on a mutated anchor: exit 1, '[FAIL]', file untouched", r.returncode == 1 and "[FAIL]" in r.stderr and f.read_bytes() == before)
        r = subprocess.run([sys.executable, str(cli)], capture_output=True, text=True)
        ok("S12f CLI without arguments: usage, exit 2", r.returncode == 2)
    return patched


def _is_subsequence(a, b, skip):
    """Every line of a appears in b in order, except `skip` lines of a (the two lines the patch replaces)."""
    j = miss = 0
    for line in a:
        k = j
        while k < len(b) and b[k] != line:
            k += 1
        if k == len(b):
            miss += 1
        else:
            j = k + 1
    return miss == skip


# ---------------------------------------------------------------------------------------------------------------------
# K: the expand kernel model
# ---------------------------------------------------------------------------------------------------------------------
def kernel_checks(ctx: Ctx, quick: bool) -> None:
    rng = np.random.default_rng(7)
    bad = n = 0
    for _ in range(300 if quick else 1500):
        ps = int(rng.choice([2, 4, 8]))
        ng = int(rng.integers(1, 40))
        rows = int(rng.integers(1, 10))
        seq = rng.integers(1, 3000, size=rows).astype(np.int32)
        pool = np.empty((rows, ng), np.int64)
        for r in range(rows):
            plen = int(seq[r]) // ps
            kind = rng.integers(0, 6, size=ng)
            vals = np.where(kind == 0, -1, np.where(kind == 1, rng.integers(0, max(plen, 1), size=ng),
                    np.where(kind == 2, plen + rng.integers(0, 60, size=ng), np.where(kind == 3, rng.integers(2**31, 2**40, size=ng),
                    np.where(kind == 4, -rng.integers(1, 2**33, size=ng), rng.integers(0, max(plen, 1), size=ng))))))
            pool[r] = vals
        vec = TL.expand_vec(pool, seq, ps)
        for r in range(rows):
            n += 1
            if not np.array_equal(vec[r], np.array(TL.expand_scalar(pool[r], int(seq[r]), ps), np.int32)):
                bad += 1
    ok(f"K1 numpy model == literal per-element transliteration of the Triton kernel on {n} rows (history, -1, past-pool_len, huge, negative ids)", bad == 0, f"{bad} differ")
    # per-row independence: any subset / permutation of the rows gives the same rows
    bad = 0
    for _ in range(300):
        ps = int(rng.choice([2, 4, 8]))
        ng = int(rng.integers(1, 30))
        rows = int(rng.integers(2, 40))
        seq = rng.integers(1, 2000, size=rows).astype(np.int32)
        pool = rng.integers(-3, 600, size=(rows, ng)).astype(np.int64)
        full = TL.expand_vec(pool, seq, ps)
        idx = rng.permutation(rows)[: int(rng.integers(1, rows + 1))]
        if not np.array_equal(TL.expand_vec(pool[idx], seq[idx], ps), full[idx]):
            bad += 1
        cut = int(rng.integers(1, rows))
        if not np.array_equal(np.concatenate([TL.expand_vec(pool[:cut], seq[:cut], ps), TL.expand_vec(pool[cut:], seq[cut:], ps)]), full):
            bad += 1
    ok("K2 per-row independence of the model: expand(subset) == subset of expand(all), expand(a)+expand(b) == expand(a+b) (600 splits)", bad == 0, f"{bad}")
    out = TL.expand_vec(np.zeros((3, 512), np.int64), np.array([5, 4096, 7], np.int32), 4)
    ok("K3 production dims: 512 pools, pool size 4 -> [rows, 2051] int32", out.shape == (3, 2051) and out.dtype == np.int32)
    seq = np.array([1, 2, 3, 4, 5, 8, 4097], np.int32)
    out = TL.expand_vec(np.full((len(seq), 8), 0, np.int64), seq, 4)
    tails = [list(out[i, 32:]) for i in range(len(seq))]
    ok("K4 the 3 tail columns hold tail_start.. (seq_len mod 4 entries) then -1", tails == [[0, -1, -1], [0, 1, -1], [0, 1, 2], [-1, -1, -1], [4, -1, -1], [-1, -1, -1], [4096, -1, -1]], str(tails))
    # the kernel SOURCE
    mod = ast.parse(ctx.kp)
    kern = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "_expand_pools_and_append_tail_kernel")
    wrap = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "expand_pools_and_append_tail")
    src = {ast.unparse(s) for s in kern.body}
    want = [
        "row = tl.program_id(0)", "tile = tl.program_id(1)", "seq_len = tl.load(seq_lens_ptr + row)",
        "pool_len = seq_len // POOL_SIZE", "tail_start = pool_len * POOL_SIZE", "tail_count = seq_len - tail_start",
        "is_history = cols < topk", "g = cols // POOL_SIZE", "o = cols % POOL_SIZE",
        "pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)",
        "hist_val = (pid * POOL_SIZE + o).to(tl.int32)",
        "hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)",
        "tail_off = cols - topk", "is_tail = (tail_off >= 0) & (tail_off < tail_count)",
        "tail_val = (tail_start + tail_off).to(tl.int32)", "tail_out = tl.where(is_tail, tail_val, -1)",
        "result = tl.where(is_history, hist_out, tail_out)", "tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)",
    ]
    missing = [w for w in want if w not in src]
    ok("K5 the baked kernel body contains every statement the numpy model transliterates", not missing, str(missing))
    used = sorted({ast.unparse(a) for a in ast.walk(kern) if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "tl"})
    allowed = {"tl.program_id", "tl.arange", "tl.load", "tl.where", "tl.store", "tl.int32", "tl.constexpr"}
    ok("K6 the kernel uses only program_id / arange / load / where / store / int32: no atomics, reductions, scans or dots", set(used) <= allowed, str(used))
    pids = [ast.unparse(c) for c in ast.walk(kern) if isinstance(c, ast.Call) and ast.unparse(c.func) == "tl.program_id"]
    loads = [ast.unparse(c.args[0]) for c in ast.walk(kern) if isinstance(c, ast.Call) and ast.unparse(c.func) in ("tl.load", "tl.store")]
    ok("K7 per-row structure of the SOURCE: 2 program ids (row, tile); every load/store address is `... + row * stride + ...` or `... + row`",
       sorted(pids) == ["tl.program_id(0)", "tl.program_id(1)"]
       and sorted(loads) == sorted(["seq_lens_ptr + row", "pool_ids_ptr + row * pid_s0 + g", "out_ptr + row * out_s0 + cols"]), str((pids, loads)))
    wsrc = {ast.unparse(s) for s in wrap.body}
    ok("K8 the wrapper computes rows/n_groups from pool_ids.shape, topk = n_groups * pool_size, out_cols = topk + pool_size - 1, one (rows, n_tiles) grid",
       "rows, n_groups = pool_ids.shape" in wsrc and "topk = n_groups * pool_size" in wsrc and "out_cols = topk + pool_size - 1" in wsrc
       and "out = torch.empty((rows, out_cols), dtype=torch.int32, device=pool_ids.device)" in wsrc
       and any(("_expand_pools_and_append_tail_kernel[(rows, n_tiles)]" in u or "_expand_pools_and_append_tail_kernel[rows, n_tiles]" in u)
               for u in (ast.unparse(s) for s in wrap.body)), str(sorted(x[:90] for x in wsrc)))


# ---------------------------------------------------------------------------------------------------------------------
# E: exactness on random layouts (numpy model only)
# ---------------------------------------------------------------------------------------------------------------------
def random_pools(rng, scen, ng):
    P = scen.P
    pools = np.full((P, ng), -1, np.int32)
    for j in range(P):
        plen = (int(scen.row_pos[j]) + 1) // scen.ratio
        kind = rng.random()
        if kind < 0.06:
            continue  # all -1 row
        if kind < 0.12:
            pools[j] = rng.integers(-5, plen + 60, size=ng)  # garbage incl. past pool_len
        else:
            m = int(rng.integers(0, min(ng, plen) + 1))
            pools[j, :m] = rng.choice(plen, size=m, replace=False) if m else []
    return pools


def original_rows(scen, pools, ratio):
    """What every rank holds after the ORIGINAL exchange: per rank, per chunk expansion of its rows, then the gather."""
    nd, P, tp = scen.nd, scen.P, len(scen.sizes)
    end = nd + P
    starts = np.concatenate([[0], np.cumsum(scen.sizes)]) + nd
    per_rank = []
    for r in range(tp):
        buf = np.full((end, scen.width), -1, np.int32)
        for c in scen.chunks:
            lo, hi = max(c.token_start, starts[r]), min(c.token_end, starts[r + 1])
            if lo >= hi:
                continue
            q_seq = scen.positions[lo:hi].astype(np.int32) + 1
            buf[lo:hi] = TL.expand_vec(pools[lo - nd:hi - nd].astype(np.int64), q_seq, ratio)
        per_rank.append(buf[starts[r]:starts[r + 1]])
    return np.concatenate(per_rank, axis=0)


def compact_rows(scen, pools, ratio):
    """The compact path: stage the pool rows per rank per chunk, gather the pool rows, expand ALL rows once."""
    nd, P, tp = scen.nd, scen.P, len(scen.sizes)
    starts = np.concatenate([[0], np.cumsum(scen.sizes)]) + nd
    gathered = []
    for r in range(tp):
        stage = np.full((P, pools.shape[1]), -1, np.int32)
        for c in scen.chunks:
            lo, hi = max(c.token_start, starts[r]), min(c.token_end, starts[r + 1])
            if lo >= hi:
                continue
            stage[lo - nd:hi - nd] = pools[lo - nd:hi - nd]
        gathered.append(stage[starts[r] - nd:starts[r + 1] - nd])
    g = np.concatenate(gathered, axis=0)
    return TL.expand_vec(g.astype(np.int64), scen.positions[nd:nd + P].astype(np.int32) + 1, ratio)


def layout_equivalence(n: int) -> None:
    rng = np.random.default_rng(3503)
    bad = rows_total = n_multi = n_one = n_skew = n_decode = 0
    for it in range(n):
        tp = int(rng.choice([2, 3, 4, 5, 8, 8, 16]))
        ratio = int(rng.choice([4, 4, 4, 2, 8]))
        ng = int(rng.integers(2, 48))
        scen = mk(rng, tp, ratio=ratio, topk_tokens=ng * ratio, sizes_mode=str(rng.choice(["random", "equal", "skewed", "random"])),
                  cut_mode=str(rng.choice(["random", "one", "requests", "random"])), max_rows=int(rng.choice([40, 120, 400])),
                  nd=int(rng.integers(0, 40)))
        pools = random_pools(rng, scen, ng)
        o, c = original_rows(scen, pools, ratio), compact_rows(scen, pools, ratio)
        rows_total += scen.P
        n_multi += len(scen.chunks) > 1
        n_one += len(scen.chunks) == 1
        n_skew += min(scen.sizes) == 1
        n_decode += scen.nd > 0
        if o.shape != c.shape or not np.array_equal(o, c):
            bad += 1
    ok(f"E1 expand-all-gathered == expand-per-chunk-per-rank-then-gather on {n} random layouts ({rows_total} rows; {n_multi} multi-chunk, {n_one} single-chunk, "
       f"{n_skew} with a 1-row rank, {n_decode} with decode rows before the prefill; random -1 / garbage pool rows; 2-16 ranks; kpool 2/4/8)",
       bad == 0 and n_multi > n // 4 and n_skew > 0 and n_decode > n // 2, f"{bad} layouts differ")
    # production dims, a few layouts
    bad = 0
    for _ in range(4):
        scen = mk(rng, 8, ratio=4, topk_tokens=2048, max_rows=900, max_prior=200000)
        pools = random_pools(rng, scen, 512)
        o, c = original_rows(scen, pools, 4), compact_rows(scen, pools, 4)
        bad += not np.array_equal(o, c) or o.shape[1] != 2051
    ok("E2 production dims (8 ranks, 512 pools, kpool 4, width 2051, contexts up to 200k): identical", bad == 0)
    # the negative control: rows outside every chunk
    differ = layouts = 0
    for _ in range(60):
        tp = int(rng.choice([2, 4, 8]))
        scen = TL.Scenario(rng, tp, ratio=4, topk_tokens=64, tiny_chunk=True, cut_mode="requests")
        if scen.tiles():
            continue
        layouts += 1
        pools = random_pools(rng, scen, 16)
        o, c = original_rows(scen, pools, 4), compact_rows(scen, pools, 4)
        # the dropped rows are exactly the rows where they differ; the original keeps them all -1
        covered = np.zeros(scen.P, bool)
        for ch in scen.chunks:
            covered[ch.token_start - scen.nd:ch.token_end - scen.nd] = True
        differ += (not np.array_equal(o, c)) and np.array_equal(o[covered], c[covered]) and bool((o[~covered] == -1).all()) and bool((c[~covered] >= 0).any(axis=1).all())
    ok(f"E3 negative control: when chunks leave rows uncovered the original keeps all -1 rows, the compact expansion gives tail-only rows ({layouts} layouts, all differ exactly there)",
       layouts > 10 and differ == layouts, f"{differ}/{layouts}")
    # bytes
    pool_b, tok_b = 512 * 4, 2051 * 4
    ok("E4 bytes per row: 2,048 (512 int32 pool ids) vs 8,204 (2,051 int32 token ids) = 4.006x", pool_b == 2048 and tok_b == 8204 and abs(tok_b / pool_b - 4.0059) < 1e-3)


# ---------------------------------------------------------------------------------------------------------------------
# suites on the REAL function (used for the genuine patched file and, with mini=True, for every mutant)
# ---------------------------------------------------------------------------------------------------------------------
def suite_gate_off(ctx, patched, chk, mini):
    rng = np.random.default_rng(3501)
    n = 3 if mini else 30
    mism = 0
    cases = 0
    hooks = {"after_exec": poison_begin}
    for i in range(n):
        tp = [2, 4, 8][i % 3]
        scen = TL.Scenario(rng, tp, nd=int(rng.integers(0, 6)), cut_mode=str(rng.choice(["random", "requests", "one"])))
        ref = ctx.run(ctx.base, [scen], tp, timeout=15)
        if not ref.ok:
            chk("F0 reference (unpatched) run completes", False, str(ref.errors()))
            return
        spellings = [{}] if mini else [{}, {GATE: "0"}, {GATE: ""}]
        for env in spellings:
            got = ctx.run(patched, [scen], tp, env=env, hooks=hooks, timeout=15)
            cases += 1
            same = (got.ok and same_bufs(ref, got) and [r.trace for r in ref.ranks] == [r.trace for r in got.ranks]
                    and ref.world.trace == got.world.trace and ref.world.stats == got.world.stats and not all_msgs(got))
            mism += not same
            if not same and mism == 1:
                chk("F1 first mismatch detail", False, f"ok={got.ok} errors={got.errors()} bufs={got.ok and same_bufs(ref, got)}")
    chk(f"F1 gate unset / 0 / empty: buffers, kernel launches, collectives and logs identical to the unpatched function, helpers never called ({cases} runs, 2-8 ranks)", mism == 0, f"{mism} differ")
    scen = TL.Scenario(rng, 4)
    r = ctx.run(patched, [scen], 4, env={})
    chk("F2 gate unset: _P035_ON is False and VERIFY 0 on every rank", r.ok and all(x.ns["_P035_ON"] is False and x.ns["_P035_VERIFY"] == 0 for x in r.ranks))


def suite_compact(ctx, patched, chk, mini):
    rng = np.random.default_rng(3502)
    cfgs = [dict(ratio=4, topk_tokens=64), dict(ratio=4, topk_tokens=128), dict(ratio=2, topk_tokens=32), dict(ratio=8, topk_tokens=64),
            dict(ratio=4, topk_tokens=2048, max_rows=150)]
    n = 8 if mini else 60
    bad_buf = bad_struct = bad_log = bad_bytes = 0
    used = {"multi": 0, "nd": 0, "one_row_rank": 0, "tps": set(), "cfgs": set()}
    first = None
    for i in range(n):
        cfg = dict(cfgs[i % len(cfgs)])
        if cfg["topk_tokens"] == 2048 and mini:
            cfg = dict(cfgs[0])
        tp = [2, 3, 4, 5, 8, 8][i % 6]
        scen = mk(rng, tp, sizes_mode=["random", "equal", "skewed"][i % 3], cut_mode=["random", "requests", "one", "random"][i % 4],
                  nd=[3, 0, 5, 1][i % 4], **cfg)
        ref = ctx.run(ctx.base, [scen], tp, timeout=15)
        got = ctx.run(patched, [scen], tp, env=env_on(0), timeout=15)
        if not (ref.ok and got.ok):
            chk("C0 runs complete", False, str(ref.errors() + got.errors()))
            return
        used["multi"] += len(scen.chunks) > 1
        used["nd"] += scen.nd > 0
        used["one_row_rank"] += min(scen.sizes) == 1
        used["tps"].add(tp)
        used["cfgs"].add((cfg["ratio"], cfg["topk_tokens"]))
        if not (same_bufs(ref, got) and ranks_agree(got)):
            bad_buf += 1
            first = first or f"buffers differ: tp={tp} cfg={cfg} P={scen.P} nd={scen.nd} sizes={scen.sizes} cuts={scen.cuts}"
        sk, w, P = scen.select_k, scen.width, scen.P
        for r in range(tp):
            ex = kernels(got, r, "expand")
            if not (len(ex) == 1 and ex[0][1] == (P, sk) and ex[0][2] == "int64" and ex[0][3] == (P,)):
                bad_struct += 1
                first = first or f"rank {r}: expand launches {ex}"
            if kernels(got, r, "mqa_logits") != kernels(ref, r, "mqa_logits") or kernels(got, r, "topk") != kernels(ref, r, "topk"):
                bad_struct += 1
                first = first or f"rank {r}: logits/top-k launches differ from the original"
            if kernels(got, r, "cp_gather") != kernels(ref, r, "cp_gather"):
                bad_struct += 1
        if not (got.world.stats["gatherv_calls"] == 1 and got.world.stats["gatherv_width"] == [sk] and got.world.stats["allreduce_calls"] == 0
                and ref.world.stats["gatherv_width"] == [w]):
            bad_struct += 1
            first = first or f"collectives {got.world.stats} vs {ref.world.stats}"
        if got.world.stats["gatherv_recv_bytes"] * w != ref.world.stats["gatherv_recv_bytes"] * sk:
            bad_bytes += 1
        a0 = msgs(got, 0)
        if not (a0 == [f"[patch-035] compact exchange ACTIVE (width {sk} instead of {w})"]) or any(all_msgs_rank for all_msgs_rank in [msgs(got, r) for r in range(1, tp)]):
            bad_log += 1
            first = first or f"log: {a0}"
    chk(f"C1 compact path (VERIFY=0): final buffers bitwise the original's, equal on every rank ({n} runs; ratios/top-k {sorted(used['cfgs'])}; ranks {sorted(used['tps'])}; "
        f"{used['multi']} multi-chunk, {used['nd']} with decode rows, {used['one_row_rank']} with a 1-row rank)", bad_buf == 0 and (mini or (used["multi"] > 10 and used["one_row_rank"] > 5)), first or "")
    chk("C2 per call: ONE all_gatherv of width select_k, ONE expand launch over all P rows with int64 pool ids and a [P] int32 q_seq, no per-chunk expansion, "
        "logits/top-k/gather launches as in the original, no all_reduce", bad_struct == 0, first or "")
    chk("C3 gathered bytes = original bytes x select_k / (topk + kpool - 1) exactly (the 4x of the production dims)", bad_bytes == 0)
    chk("C4 log: rank 0 prints exactly '[patch-035] compact exchange ACTIVE (width <select_k> instead of <topk + kpool - 1>)', the other ranks print nothing", bad_log == 0, first or "")
    # production dims: the numbers of the brief
    scen = mk(rng, 8, ratio=4, topk_tokens=2048, max_rows=120, max_prior=100000)
    ref = ctx.run(ctx.base, [scen], 8, timeout=20)
    got = ctx.run(patched, [scen], 8, env=env_on(0), timeout=20)
    ok_prod = ref.ok and got.ok and same_bufs(ref, got)
    chk("C5 production dims (top-k 2048, kpool 4, 8 ranks): identical buffers; gather width 512 instead of 2051; "
        "the compact gather moves 4.006x fewer bytes",
        ok_prod and got.world.stats["gatherv_width"] == [512] and ref.world.stats["gatherv_width"] == [2051]
        and abs(ref.world.stats["gatherv_recv_bytes"] / got.world.stats["gatherv_recv_bytes"] - 2051 / 512) < 1e-9
        and msgs(got, 0) == ["[patch-035] compact exchange ACTIVE (width 512 instead of 2051)"], str(ref.errors() + got.errors()))
    # several calls in one process (state persists): every call equal, ACTIVE once
    scens = [mk(rng, 8, nd=int(rng.integers(0, 6))) for _ in range(3 if mini else 6)]
    ref = ctx.run(ctx.base, scens, 8, timeout=20)
    got = ctx.run(patched, scens, 8, env=env_on(0), timeout=20)
    chk(f"C6 {len(scens)} consecutive calls in one process: every call's buffer equals the original's, ACTIVE logged once",
        ref.ok and got.ok and same_bufs(ref, got) and len(msgs(got, 0, contains="ACTIVE")) == 1 and got.world.stats["gatherv_calls"] == len(scens),
        str(ref.errors() + got.errors()))


def _fault_wrap(bad_rank, which, call_idx):
    """wrap_expand hook: flip one element of rank `bad_rank`'s first expansion of kind `which` in call `call_idx` (0-based);
    'compact' = the one launch over all P rows, 'chunk' = a per-chunk launch of the original path."""
    state = {"done": -1}

    def wrap(rank, fn, cur):
        if rank != bad_rank:
            return fn

        def f(pool_ids, seq_lens, pool_size):
            out = fn(pool_ids, seq_lens, pool_size)
            rows = np.asarray(pool_ids).shape[0]
            is_compact = rows == cur["P"]
            if cur["call"] == call_idx and state["done"] != call_idx and (which == "compact") == is_compact:
                state["done"] = call_idx
                o = np.asarray(out).copy()
                o[rows // 2, 0] ^= 1
                return TL._ft(o)
            return out
        return f
    return wrap


def suite_verify(ctx, patched, chk, mini):
    rng = np.random.default_rng(3504)
    tp = 8
    scens = [mk(rng, tp, nd=[2, 0, 4, 1, 3][i], cut_mode=["random", "requests", "one", "random", "random"][i]) for i in range(5)]
    ref = ctx.run(ctx.base, scens, tp, timeout=20)
    got = ctx.run(patched, scens, tp, env=env_on(), timeout=20)  # default VERIFY=3
    chk("V0 runs complete", ref.ok and got.ok, str(ref.errors() + got.errors()))
    if not (ref.ok and got.ok):
        return
    chk("V1 default VERIFY=3, 5 calls: final buffers bitwise the original's on every rank", same_bufs(ref, got) and ranks_agree(got))
    sk, w = scens[0].select_k, scens[0].width
    chk("V2 collectives: calls 1-3 = gather(select_k) + gather(original width) + 1 all_reduce each, calls 4-5 = one gather(select_k)",
        got.world.stats["gatherv_width"] == [sk, w] * 3 + [sk] * 2 and got.world.stats["allreduce_calls"] == 3,
        str(got.world.stats))
    exp = []
    for k, s in enumerate(scens[:3], 1):
        exp.append(f"[patch-035] verify {k}/3: bitwise equal ({s.P} rows, deepest context {s.max_prefill_seq_len})")
    r0 = msgs(got, 0)
    chk("V3 rank 0 logs ACTIVE once, then exactly 'verify k/3: bitwise equal (R rows, deepest context C)' for k = 1..3, nothing for calls 4-5",
        r0 == [f"[patch-035] compact exchange ACTIVE (width {sk} instead of {w})"] + exp, str(r0))
    chk("V4 the other ranks log nothing when everything is equal", all(not msgs(got, r) for r in range(1, tp)))
    # per-call kernel structure: verify calls also run the per-chunk (original) expansion
    # a rank only launches the chunks overlapping its slice; compare to the original's per-rank launches
    per_call_ref = [0] * 5
    cur = -1
    for e in ref.ranks[1].trace:
        if e[0] == "call":
            cur = e[1]
        elif e[0] == "expand":
            per_call_ref[cur] += 1
    per_call_got = [0] * 5
    cur = -1
    for e in got.ranks[1].trace:
        if e[0] == "call":
            cur = e[1]
        elif e[0] == "expand":
            per_call_got[cur] += 1
    chk("V5 per-call expand launches on a rank: calls 1-3 = the original's per-chunk launches + 1 compact one, calls 4-5 = exactly 1",
        per_call_got == [per_call_ref[0] + 1, per_call_ref[1] + 1, per_call_ref[2] + 1, 1, 1], f"{per_call_got} vs ref {per_call_ref}")
    # VERIFY = 0 / 1 / 5
    for v in (0, 1, 5):
        g = ctx.run(patched, scens, tp, env=env_on(v), timeout=20)
        nver = min(v, 5)
        lines = [m for m in msgs(g, 0) if "verify" in m]
        chk(f"V6 VERIFY={v}: {nver} verify line(s), {nver} all_reduce(s), {5 + nver} gathers, buffers equal",
            g.ok and same_bufs(ref, g) and len(lines) == nver and g.world.stats["allreduce_calls"] == nver and g.world.stats["gatherv_calls"] == 5 + nver
            and (v == 0) == all("verify" not in m for m in msgs(g, 0)), str(g.errors() or g.world.stats))
        if mini:
            break
    # a racy top-k order: both paths consume the SAME pool_topk, so there is no false mismatch; ranks still agree
    racy = ctx.run(patched, scens[:3], tp, env=env_on(), topk=lambda rk: TL.make_racy_topk(1000 + rk), timeout=20)
    chk("V7 racy per-rank top-k order (set exact, order varies): all 3 VERIFY calls still 'bitwise equal', all ranks end with the same buffer",
        racy.ok and len([m for m in msgs(racy, 0) if "bitwise equal" in m]) == 3 and ranks_agree(racy) and not any("MISMATCH" in m for m in all_msgs(racy)),
        str(racy.errors() or msgs(racy, 0)))
    # ---- fault injection: one rank's compact result is wrong
    for bad_rank in ([3] if mini else [0, 3, 7]):
        fscens = scens[:3]
        res = ctx.run(patched, fscens, tp, env=env_on(), hooks={"wrap_expand": _fault_wrap(bad_rank, "compact", 0)}, timeout=20)
        refx = ctx.run(ctx.base, fscens, tp, timeout=20)
        errs = [(r, m) for r in range(tp) for lv, m in res.ranks[r].records if lv == "ERROR"]
        chk(f"V8 fault: rank {bad_rank}'s compact result is wrong in call 1 -> ERROR on EVERY rank (one line each), 'this rank differs' only on rank {bad_rank}",
            res.ok and sorted(r for r, _ in errs) == list(range(tp)) and all("MISMATCH" in m and "verify 1/3" in m for _, m in errs)
            and [r for r, m in errs if "this rank differs" in m] == [bad_rank]
            and f"this rank differs in 1 of {fscens[0].P} rows, first at row {fscens[0].nd + fscens[0].P // 2}" in dict(errs)[bad_rank], str(res.errors() or errs[:2]))
        chk(f"V9 fault (rank {bad_rank}): the ORIGINAL rows are used for the failing call and every later call -> buffers bitwise the unpatched function's, on every rank",
            res.ok and same_bufs(refx, res) and ranks_agree(res))
        chk(f"V10 fault (rank {bad_rank}): the compact path is off on all ranks afterwards: calls 2-3 gather the original width only, no more all_reduce, no verify/ACTIVE lines",
            res.world.stats["gatherv_width"] == [fscens[0].select_k, fscens[0].width, fscens[0].width, fscens[0].width]
            and res.world.stats["allreduce_calls"] == 1 and all(x.ns["_P035_STATE"]["disabled"] for x in res.ranks)
            and len([m for m in msgs(res, 0) if "ACTIVE" in m]) == 1 and not [m for m in msgs(res, 0) if "bitwise equal" in m],
            str(res.world.stats))
    # fault in call 2 (call 1 verified fine)
    fscens = scens[:3]
    res = ctx.run(patched, fscens, tp, env=env_on(), hooks={"wrap_expand": _fault_wrap(5, "compact", 1)}, timeout=20)
    refx = ctx.run(ctx.base, fscens, tp, timeout=20)
    chk("V11 fault in call 2: 'verify 1/3: bitwise equal', then MISMATCH 2/3 on every rank, call 3 on the original path, buffers == unpatched",
        res.ok and same_bufs(refx, res) and any("verify 1/3: bitwise equal" in m for m in msgs(res, 0))
        and all(any("verify 2/3: MISMATCH" in m for m in msgs(res, r, "ERROR")) for r in range(tp))
        and res.world.stats["gatherv_width"] == [fscens[0].select_k, fscens[0].width] * 2 + [fscens[0].width], str(res.errors() or res.world.stats))
    if not mini:
        # a wrong ORIGINAL-path expansion on one rank is seen by everyone too (the gather hands the wrong rows to all ranks)
        res = ctx.run(patched, fscens, tp, env=env_on(), hooks={"wrap_expand": _fault_wrap(2, "chunk", 0)}, timeout=20)
        chk("V12 fault in the original-path expansion of rank 2: detected on all ranks (collective decision), disabled afterwards, no hang, all ranks end with the same buffer",
            res.ok and all(any("MISMATCH" in m for m in msgs(res, r, "ERROR")) for r in range(tp)) and ranks_agree(res)
            and res.world.stats["gatherv_width"] == [fscens[0].select_k, fscens[0].width, fscens[0].width, fscens[0].width], str(res.errors() or res.world.stats))
        # defense in depth: a natural (not injected) failure. Without the chunk-tiling guard the compact path would engage on a layout with
        # uncovered rows and give tail-only rows where the original keeps -1 rows: VERIFY must catch it on every rank and fall back
        guard_off = patched.replace("        if pos != end:\n", "        if False:\n").replace(
            "if chunk.token_start != pos or chunk.token_end < chunk.token_start:", "if False:")
        gaps = [s for s in (TL.Scenario(rng, tp, tiny_chunk=True) for _ in range(40)) if not s.tiles()][:3]
        res = ctx.run(guard_off, gaps, tp, env=env_on(), timeout=20)
        refg = ctx.run(ctx.base, gaps, tp, timeout=20)
        chk("V14 defense in depth: with the tiling guard removed, VERIFY sees the uncovered-rows difference in call 1 on every rank, uses the original rows "
            "(buffers == unpatched for all 3 calls) and switches the compact path off",
            guard_off != patched and res.ok and refg.ok and same_bufs(refg, res)
            and all(any("verify 1/3: MISMATCH" in m for m in msgs(res, r, "ERROR")) for r in range(tp))
            and res.world.stats["allreduce_calls"] == 1 and all(x.ns["_P035_STATE"]["disabled"] for x in res.ranks),
            str(res.errors() or res.world.stats))
        # a wrong pool row (a top-k bug) is NOT this patch's business: both paths see it, verify stays quiet
        def corrupt_topk(rk):
            def f(logits, ks, ke, topk_dst, num_rows, select_k):
                TL.fake_topk(logits, ks, ke, topk_dst, num_rows, select_k)
                if rk == 4 and num_rows:
                    topk_dst[0, 0] = 1
            return f
        res = ctx.run(patched, fscens[:2], tp, env=env_on(), topk=corrupt_topk, timeout=20)
        refc = ctx.run(ctx.base, fscens[:2], tp, topk=corrupt_topk, timeout=20)
        chk("V13 scope: a corrupted top-k row on one rank reaches both paths alike -> verify stays 'bitwise equal', result == the unpatched function with the same corruption",
            res.ok and refc.ok and same_bufs(refc, res) and not any("MISMATCH" in m for m in all_msgs(res)))


def suite_declines(ctx, patched, chk, mini):
    rng = np.random.default_rng(3505)
    tp = 8
    cases = []  # (label, scenario list, hooks, category fragment or None)
    for kind in ("gap", "tokens", "topk", "capture", "kpool1", "nopos", "short", "noshard"):
        if mini and kind in ("nopos", "short", "noshard"):
            continue
        if kind == "gap":
            scs = [TL.Scenario(rng, tp, tiny_chunk=True, ratio=4, topk_tokens=64) for _ in range(40)]
            scs = [s for s in scs if not s.tiles()][:3]
            cases.append(("gap", scs, {}, ("do not continue", "the chunks end at row")))
        elif kind == "tokens":
            cases.append(("tokens", [mk(rng, tp) for _ in range(3)], {"extra_tokens": 3}, "step tokens"))
        elif kind == "topk":
            cases.append(("topk", [mk(rng, tp) for _ in range(3)], {"topk_tokens": 66}, "not a multiple"))
        elif kind == "capture":
            cases.append(("capture", [mk(rng, tp) for _ in range(2)], {"_capturing": True}, "CUDA graph capture"))
        elif kind == "kpool1":
            cases.append(("kpool1", [mk(rng, tp) for _ in range(2)], {"index_kpool": 1}, "not the served layout"))
        elif kind == "nopos":
            cases.append(("nopos", [mk(rng, tp) for _ in range(2)], {"positions_none": True}, "not the served layout"))
        elif kind == "short":
            cases.append(("short", [TL.Scenario(rng, tp, deep=False, max_prior=0, max_rows=40) for _ in range(2)], {}, None))
        elif kind == "noshard":
            cases.append(("noshard", [mk(rng, tp) for _ in range(2)], {"no_row_shard": True}, None))
    for label, scs, hooks, frag in cases:
        if label == "short":
            scs = [s for s in scs if s.max_prefill_seq_len <= s.topk_tokens]
            if not scs:
                chk("D0 short-prefill scenario built", False)
                continue
        h = {k: v for k, v in hooks.items() if k != "_capturing"}
        cap = bool(hooks.get("_capturing"))
        ref = ctx.run(ctx.base, scs, tp, hooks=h, capturing=cap, timeout=20)
        got = ctx.run(patched, scs, tp, env=env_on(), hooks=h, capturing=cap, timeout=20)
        good = ref.ok and got.ok and same_bufs(ref, got) and ranks_agree(got)
        no_compact = got.world.stats["allreduce_calls"] == 0 and not msgs(got, 0, contains="ACTIVE") \
            and got.world.stats["gatherv_width"] == ref.world.stats["gatherv_width"] \
            and [kernels(got, 0, "expand")] == [kernels(ref, 0, "expand")]
        warns = msgs(got, 0, "WARNING")
        if frag is None:
            wok = not all_msgs(got)
            what = "silent (no sharded step -> nothing to decline)"
        else:
            frags = (frag,) if isinstance(frag, str) else frag
            wok = len(warns) == 1 and "compact exchange declined" in warns[0] and any(f in warns[0] for f in frags) \
                and warns[0].endswith("the original exchange runs") and all(not msgs(got, r) for r in range(1, tp))
            what = f"one rank-0 WARNING '{frags[0]}' for {len(scs)} calls (once per reason)"
        chk(f"D1 {label}: the original path runs (same buffers, collectives and expand launches as the unpatched function); {what}", good and no_compact and wok,
            str(ref.errors() + got.errors() or (warns, got.world.stats, ref.world.stats)))
    if mini:
        return
    # the gap really exists in the original: the dropped rows stay all -1
    scs = [s for s in (TL.Scenario(rng, tp, tiny_chunk=True) for _ in range(40)) if not s.tiles()][:2]
    ref = ctx.run(ctx.base, scs, tp, timeout=20)
    okgap = ref.ok
    for s, buf in zip(scs, ref.ranks[0].bufs):
        covered = np.zeros(s.P, bool)
        for c in s.chunks:
            covered[c.token_start - s.nd:c.token_end - s.nd] = True
        okgap &= bool((buf[s.nd:s.nd + s.P][~covered, :s.width] == -1).all()) and bool((~covered).any())
    chk("D2 the gap case is real: the original leaves the uncovered rows all -1 (the compact path would not)", okgap)
    # two different reasons -> two warnings; the same reason again -> still one
    mixed = [TL.Scenario(rng, tp, tiny_chunk=True) for _ in range(30)]
    mixed = [s for s in mixed if not s.tiles()][:2] + [mk(rng, tp)]
    got = ctx.run(patched, mixed, tp, env=env_on(), hooks={}, timeout=20)
    chk("D3 declined calls and a compact call in one process: one 'declined' warning (chunks) and one ACTIVE line, buffers == unpatched",
        got.ok and len(msgs(got, 0, "WARNING")) == 1 and len(msgs(got, 0, contains="ACTIVE")) == 1 and same_bufs(ctx.run(ctx.base, mixed, tp, timeout=20), got))


def suite_config(ctx, patched, chk, mini):
    def cfg(env):
        mod = ast.parse(patched)
        node = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "_p035_config")
        saved = {k: os.environ.get(k) for k in (GATE, VERIFY_ENV)}
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        try:
            ns = {}
            exec(compile(TL._segment(patched, node), "cfg", "exec"), ns)  # noqa: S102
            return ns["_p035_config"]()
        finally:
            for k in (GATE, VERIFY_ENV):
                os.environ.pop(k, None)
            os.environ.update({k: v for k, v in saved.items() if v is not None})

    table = [({}, (False, 0)), ({GATE: ""}, (False, 0)), ({GATE: "0"}, (False, 0)), ({GATE: " 0 "}, (False, 0)), ({GATE: "1"}, (True, 3)),
             ({GATE: " 1 "}, (True, 3)), ({GATE: "1", VERIFY_ENV: "0"}, (True, 0)), ({GATE: "1", VERIFY_ENV: "7"}, (True, 7)),
             ({GATE: "1", VERIFY_ENV: " 12 "}, (True, 12)), ({GATE: "1", VERIFY_ENV: "1000"}, (True, 1000)), ({GATE: "1", VERIFY_ENV: ""}, (True, 3)),
             ({VERIFY_ENV: "abc"}, (False, 0))]
    got = []
    for env, want in table:
        try:
            got.append((cfg(env), want))
        except Exception as e:  # noqa: BLE001
            got.append((repr(e), want))
    chk(f"W1 gate / VERIFY parsing ({len(table)} cases: unset, empty, 0, whitespace, 1, VERIFY 0/7/12/1000/empty, a bad VERIFY while the gate is off is ignored)", all(g == w for g, w in got), str([x for x in got if x[0] != x[1]]))
    bad_vals = [{GATE: v} for v in ("2", "yes", "true", "on", "-1", "1.0")] + [{GATE: "1", VERIFY_ENV: v} for v in ("abc", "-1", "1.5", "1001", "99999999999", "²", "1e3", "0x10")]
    n_raise = 0
    for env in bad_vals:
        try:
            cfg(env)
        except ValueError:
            n_raise += 1
        except Exception:  # noqa: BLE001
            pass
    chk(f"W2 a typo never silently disables or enables the patch: {len(bad_vals)} bad values raise ValueError at import", n_raise == len(bad_vals), f"{n_raise}/{len(bad_vals)}")


def behaviour(ctx, patched, mini, report=None):
    """Run the REAL-function suites; returns the names of failed checks (silent when report is None)."""
    bad = []

    def chk(name, cond, detail=""):
        if not cond:
            bad.append(name)
        if report is not None:
            report(name, cond, detail)

    for suite in (suite_gate_off, suite_compact, suite_verify, suite_declines, suite_config):
        try:
            suite(ctx, patched, chk, mini)
        except Exception as e:  # noqa: BLE001 - a mutant that breaks the module counts as caught
            bad.append(f"EXC:{suite.__name__}:{type(e).__name__}:{str(e)[:80]}")
            if report is not None:
                report(f"{suite.__name__} completed", False, repr(e))
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# M: mutants
# ---------------------------------------------------------------------------------------------------------------------
MUTANTS = [
    ("stage ignores the decode offset", "self.pool_stage[row_start - self.nd : row_end - self.nd] = pool_topk", "self.pool_stage[row_start : row_end] = pool_topk"),
    ("local slice ignores the decode offset", "local = self.pool_stage[rs.start - nd : rs.stop - nd].contiguous()", "local = self.pool_stage[rs.start : rs.stop].contiguous()"),
    ("local slice one row short", "local = self.pool_stage[rs.start - nd : rs.stop - nd].contiguous()", "local = self.pool_stage[rs.start - nd : rs.stop - nd - 1].contiguous()"),
    ("positions sliced from 0", "q_seq = self.positions[nd:end].to(torch.int32) + 1", "q_seq = self.positions[: end - nd].to(torch.int32) + 1"),
    ("q_seq without +1", "q_seq = self.positions[nd:end].to(torch.int32) + 1", "q_seq = self.positions[nd:end].to(torch.int32)"),
    ("last column not written", "topk_indices_buffer[nd:end, : expanded.shape[-1]] = expanded\n            return", "topk_indices_buffer[nd:end, : width - 1] = expanded[:, : width - 1]\n            return"),
    ("gather with sizes=None", "get_tp_group().all_gatherv(local, dim=0, sizes=rs.sizes)", "get_tp_group().all_gatherv(local, dim=0, sizes=None)"),
    ("continue removed (expand per chunk AND all rows)", "                if not _p035_plan.verify:\n                    continue\n", "                if not _p035_plan.verify:\n                    pass\n"),
    ("original expansion writes the real buffer in VERIFY", "                _p035_out[row_start:row_end, : expanded.shape[-1]] = expanded\n", "                topk_indices_buffer[row_start:row_end, : expanded.shape[-1]] = expanded\n"),
    ("verify flag always False", "self.verify = scratch is not None", "self.verify = False"),
    ("chunk-tiling check (start) dropped", "if chunk.token_start != pos or chunk.token_end < chunk.token_start:", "if False:"),
    ("chunk-tiling check (end) dropped", "        if pos != end:\n", "        if False:\n"),
    ("step-token check dropped", "        if num_tokens != end:\n", "        if False:\n"),
    ("top-k multiple check dropped", "        if (topk_tokens // index_kpool) * index_kpool != topk_tokens:\n", "        if False:\n"),
    ("layout check dropped", "        if index_kpool <= 1 or positions is None:\n", "        if False:\n"),
    ("capture check dropped", "        if torch.cuda.is_current_stream_capturing():\n", "        if False:\n"),
    ("gate on by default", "        return False, 0\n    if gate != \"1\":", "        return True, 3\n    if gate != \"1\":"),
    ("VERIFY default 0", "    if not raw:\n        return True, 3\n", "    if not raw:\n        return True, 0\n"),
    ("typo accepted", "    if gate != \"1\":\n        raise ValueError", "    if False:\n        raise ValueError"),
    ("VERIFY range unchecked", " or int(raw) > 1000:", ":"),
    ("verify count off by one", "if k <= _P035_VERIFY else None", "if k < _P035_VERIFY else None"),
    ("ACTIVE from every rank", "_p035_log0(None, \"info\", \"[patch-035] compact exchange ACTIVE", "logger.info(\"[patch-035] compact exchange ACTIVE"),
    ("ACTIVE every call", "    if k == 1:\n", "    if True:\n"),
    ("ACTIVE widths swapped", "compact exchange ACTIVE (width %d instead of %d)\", select_k, width)", "compact exchange ACTIVE (width %d instead of %d)\", width, select_k)"),
    ("verify line from every rank", "_p035_log0(None, \"info\", \"[patch-035] verify", "logger.info(\"[patch-035] verify"),
    ("rank 0 test wrong", "return get_tensor_model_parallel_rank() == 0", "return get_tensor_model_parallel_rank() == 1"),
    ("decline warning repeats", "        if key in _P035_STATE[\"logged\"]:\n            return\n", "        if False:\n            return\n"),
    ("mismatch decided per rank", "bad = _p035_reduce_max(mine)", "bad = mine"),
    ("reduce with MIN", "op=torch.distributed.ReduceOp.MAX", "op=torch.distributed.ReduceOp.MIN"),
    ("no comparison", "mine = 0 if torch.equal(expanded, want) else 1", "mine = 0"),
    ("mismatch keeps the compact rows", "            topk_indices_buffer[nd:end, : expanded.shape[-1]] = want\n", "            topk_indices_buffer[nd:end, : expanded.shape[-1]] = expanded\n"),
    ("mismatch does not disable", "            _P035_STATE[\"disabled\"] = True\n", "            pass\n"),
    ("disabled flag ignored", "if not _P035_ON or _P035_STATE[\"disabled\"] or row_shard is None:", "if not _P035_ON or row_shard is None:"),
    ("scratch too few rows", "torch.full((end, width), -1, dtype=torch.int32, device=dev)", "torch.full((end - nd, width), -1, dtype=torch.int32, device=dev)"),
    ("width off by one", "    width = topk_tokens + index_kpool - 1\n    _P035_STATE", "    width = topk_tokens + index_kpool\n    _P035_STATE"),
    ("stage rows wrong chunk", "_p035_plan.stage(row_start, row_end, pool_topk)", "_p035_plan.stage(row_start, row_end - 1, pool_topk)"),
    ("continue removed AND the original block writes the real buffer (only the launch trace can tell)",
     ["                if not _p035_plan.verify:\n                    continue\n", "_p035_out = topk_indices_buffer if _p035_plan is None else _p035_plan.scratch"],
     ["                if not _p035_plan.verify:\n                    pass\n", "_p035_out = topk_indices_buffer"]),
    ("verify compares only the first half of the rows", "mine = 0 if torch.equal(expanded, want) else 1", "mine = 0 if torch.equal(expanded[: (end - nd) // 2], want[: (end - nd) // 2]) else 1"),
    ("verify reads the scratch rows without the decode offset", "want = scratch[nd:end, :width]", "want = scratch[: end - nd, :width]"),
    ("verify log shows a wrong context", "ctx = self.ctx if self.ctx >= 0 else int(self.positions[nd:end].max().item()) + 1", "ctx = 0"),
    ("all_reduce only on the rank that mismatched", "bad = _p035_reduce_max(mine)", "bad = _p035_reduce_max(mine) if mine else 0"),
    ("original exchange removed (else)", "            else:\n                row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)\n", "            else:\n                pass\n"),
    ("plan built without the gate", "            if _P035_ON and row_shard is not None\n", "            if row_shard is not None\n"),
]


# ---------------------------------------------------------------------------------------------------------------------
# --soak N: extra random runs of the REAL function (not part of the default run)
# ---------------------------------------------------------------------------------------------------------------------
def soak(ctx: Ctx, patched: str, n: int) -> None:
    rng = np.random.default_rng(35035)
    cfgs = [dict(ratio=4, topk_tokens=64), dict(ratio=4, topk_tokens=128), dict(ratio=2, topk_tokens=32), dict(ratio=8, topk_tokens=64),
            dict(ratio=4, topk_tokens=256), dict(ratio=4, topk_tokens=2048, max_rows=120)]
    bad = calls = rows = racy_runs = verified = 0
    first = ""
    t0 = time.time()
    for i in range(n):
        tp = int(rng.choice([2, 3, 4, 5, 6, 7, 8, 8, 8]))
        cfg = dict(cfgs[int(rng.integers(0, len(cfgs)))])
        scens = [mk(rng, tp, sizes_mode=str(rng.choice(["random", "equal", "skewed", "random"])), cut_mode=str(rng.choice(["random", "requests", "one", "random"])),
                    nd=int(rng.integers(0, 12)), **cfg) for _ in range(int(rng.integers(1, 5)))]
        racy = rng.random() < 0.3
        env = env_on(int(rng.choice([0, 1, 3, 3, 3, 5])))
        kw = {"topk": (lambda rk, i=i: TL.make_racy_topk(1000 * i + rk))} if racy else {}
        got = ctx.run(patched, scens, tp, env=env, timeout=30, **kw)
        calls += len(scens)
        rows += sum(s.P for s in scens)
        if racy:
            racy_runs += 1
            good = got.ok and ranks_agree(got) and not any("MISMATCH" in m for m in all_msgs(got))
        else:
            ref = ctx.run(ctx.base, scens, tp, timeout=30)
            good = got.ok and ref.ok and same_bufs(ref, got) and ranks_agree(got) and not any("MISMATCH" in m for m in all_msgs(got))
        verified += len([m for m in msgs(got, 0) if "bitwise equal" in m]) if got.ok else 0
        if not good:
            bad += 1
            first = first or f"run {i}: tp={tp} cfg={cfg} calls={len(scens)} racy={racy} env={env} {got.errors()}"
    ok(f"SOAK {n} random multi-call runs of the real patched function (2-8 ranks, {calls} calls, {rows} prefill rows, {racy_runs} with a racy top-k order, "
       f"{verified} VERIFY comparisons): final buffers bitwise the unpatched function's (racy runs: equal on all ranks, no MISMATCH), {time.time() - t0:.0f} s", bad == 0, first)


# ---------------------------------------------------------------------------------------------------------------------
def pin(sh_text: str, name: str):
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", sh_text, re.M)
    return m.group(1) if m else None


def pin_checks(ctx: Ctx, patched: str, sh_text: str, tree: Path) -> None:
    ok("P1 TARGET_ORIG pin == sha256 of the baked sparse_indexer.py", pin(sh_text, "TARGET_ORIG") == sha(ctx.base.encode()) == "3d97e02cb49a21402f3814cee2b4895af661fb1d5c4b5f7080c6b0bbc6db32a5", str(pin(sh_text, "TARGET_ORIG")))
    ok("P2 TARGET_P035 pin == sha256 of apply(baked file)", pin(sh_text, "TARGET_P035") == sha(patched.encode()), str(pin(sh_text, "TARGET_P035")))
    ok("P3 APPLIER_SHA pin == sha256 of files/p035_apply.py", pin(sh_text, "APPLIER_SHA") == sha((HERE / "p035_apply.py").read_bytes()), str(pin(sh_text, "APPLIER_SHA")))
    ok("P4 OP_ORIG pin == sha256 of sparse_attn_indexer.py (PrefillRowShard, exchange_topk)", pin(sh_text, "OP_ORIG") == sha(ctx.op.encode()))
    ok("P5 KPOOL_BAKED pin == sha256 of the baked kpool_compress.py (the per-row expand kernel)", pin(sh_text, "KPOOL_BAKED") == sha(ctx.kp.encode()))
    ok("P5b the pristine-upstream sha in the .sh is only a hint (not an accepted state) and differs from the baked one",
       pin(sh_text, "TARGET_PRISTINE") == "cf659b6fdcfa1057ef6a61c3d31d302d2dd15dfea6f3c96743697664d87e3e44" != pin(sh_text, "TARGET_ORIG")
       and 'case "$tgt_now" in\n  "$TARGET_P035") mode=installed ;;\n  "$TARGET_ORIG") mode=apply ;;\n  "$TARGET_PRISTINE")\n    fail' in sh_text)
    ok("P5c the pristine kpool_compress.py sha in the .sh is a hint only (refused with a pointed message), differs from the baked one and equals the bake script's pristine pin",
       pin(sh_text, "KPOOL_PRISTINE") == "e5d1baa2f8fac4e490cc2f22a5ca05f67d4dea03a420efdae05f3a2bbeb6616f" != pin(sh_text, "KPOOL_BAKED")
       and '[ "$kp_now" != "$KPOOL_PRISTINE" ]' in sh_text)
    ok("P6 MARKERS in the .sh == N_MARKERS of the applier", re.search(r"^MARKERS=(\d+)$", sh_text, re.M).group(1) == str(A.N_MARKERS))
    ok("P7 the .sh's MARKER string == the applier's", f'MARKER="{A.MARKER}"' in sh_text)
    for needle in ("[patch-035] compact exchange ACTIVE (width 512 instead of 2051)", "[patch-035] verify 1/3: bitwise equal", "MISMATCH",
                   "compact exchange declined", "[state] patch 035 inert", "[OK] patch 035 in place", "VLLM_GB10_IDX_COMPACT_XCHG_VERIFY"):
        ok(f"P8 the .sh documents/prints {needle!r}", needle in sh_text)
    # pins-035.txt
    pins = PINS.read_text() if PINS.exists() else ""
    plines = [l.split() for l in pins.splitlines() if l.strip() and not l.startswith("#")]
    ok("P10 pins-035.txt: 3 image pins (target, 2 contracts) with the .sh's shas, three whitespace fields each (verify_pins.py format)",
       plines == [[pin(sh_text, "TARGET_ORIG"), "models/glm5next/nvidia/sparse_indexer.py", "035"],
                  [pin(sh_text, "OP_ORIG"), "model_executor/layers/sparse_attn_indexer.py", "035:contract"],
                  [pin(sh_text, "KPOOL_BAKED"), "models/glm5next/nvidia/ops/kpool_compress.py", "035:contract(baked)"]], str(plines))
    ok("P10b pins-035.txt records the patched result and the applier sha (comment lines) equal to the .sh's",
       f"# {pin(sh_text, 'TARGET_P035')}  models/glm5next/nvidia/sparse_indexer.py  035:result" in pins
       and f"# {pin(sh_text, 'APPLIER_SHA')}  files/p035_apply.py  035:vendored" in pins)
    # memory model (per row, at the served shapes: 512 pools, width 2051, top-k buffer width 2176; 8,192 rows = max-num-batched-tokens)
    rows, buf_w, w, sk, tpn = 8192, 2176, 2051, 512, 8
    reserve = 2 * rows * buf_w * 4
    stage, pool64, expanded, gathered = rows * sk * 4, rows * sk * 8, rows * w * 4, rows * sk * 4
    orig_ex = rows * w * 4 // tpn + rows * w * 4
    peak = max(stage + gathered, stage + gathered + pool64, stage + pool64 + expanded)  # finish(): del gathered after the cast; del pool_ids after the launch
    ok(f"P11 memory model at 8,192 rows: compact peak {peak / 1e6:.1f} MB ({peak // rows} B/row: pool_stage + int64 pool ids + expanded) <= the profiling reserve "
       f"{reserve / 1e6:.1f} MB ({reserve // rows} B/row); the old exchange needs {orig_ex / 1e6:.1f} MB ({orig_ex // rows} B/row)",
       peak <= reserve and peak // rows == 14348 and reserve // rows == 17408 and orig_ex // rows == 9229)
    v_peak = rows * w * 4 * 3 + rows * w * 4 // tpn  # scratch + expanded + the original exchange's gather + its local slice
    ok(f"P11b a VERIFY call peaks at {v_peak / 1e6:.0f} MB ({v_peak // rows} B/row), {(v_peak - reserve) / 1e6:.0f} MB above the reserve, 3 calls per process",
       v_peak // rows == 25637 and 66e6 < v_peak - reserve < 68e6)
    # provenance of the baked files: upstream tree + the bake script's edits == the image files
    if not BAKE_SCRIPT.exists() or not tree.exists():
        print("[SKIP] P9 provenance of the baked files (bake script or upstream tree not found)")
        return
    spec = importlib.util.spec_from_file_location("p035_bake_probe", BAKE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p035_bake_probe"] = mod
    spec.loader.exec_module(mod)
    for rel, want, label in (("models/glm5next/nvidia/sparse_indexer.py", ctx.base, "sparse_indexer.py"), ("models/glm5next/nvidia/ops/kpool_compress.py", ctx.kp, "kpool_compress.py")):
        txt = (tree / rel).read_text()
        for e in mod.EDITS:
            if e.pkg == "vllm" and e.rel == rel:
                if txt.count(e.old) != 1:
                    txt = None
                    break
                txt = txt.replace(e.old, e.new, 1)
        ok(f"P9 provenance: upstream v0.32.0 {label} + the bake script's edits == the image file given to this check", txt == want)
    ok("P9b the upstream sparse_attn_indexer.py in the tree == the image file (the bake does not touch it)", (tree / "model_executor/layers/sparse_attn_indexer.py").read_text() == ctx.op)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=str(DEFAULT_FILES))
    ap.add_argument("--tree", default=str(DEFAULT_TREE))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--soak", type=int, default=0, help="N extra random multi-call runs of the real function (default: none)")
    a = ap.parse_args()
    t0 = time.time()
    ctx = Ctx(Path(a.files))
    sh_text = SH.read_text()
    print(f"# p035_cpu_check: files {a.files}, target sha256 {sha(ctx.base.encode())}, numpy {np.__version__}, python {sys.version.split()[0]}")
    patched = static_checks(ctx, sh_text)
    kernel_checks(ctx, a.quick)
    layout_equivalence(300 if a.quick else 3000)

    def rep(name, cond, detail=""):
        ok(name, cond, detail)

    bad = behaviour(ctx, patched, mini=False, report=rep)
    ok("G* behaviour suites (F, C, V, D, W) on the genuine patched code: all checks passed", not bad, str(bad))
    pin_checks(ctx, patched, sh_text, Path(a.tree))
    if a.soak:
        soak(ctx, patched, a.soak)
    # ---- M: mutants
    caught = 0
    for label, old, new in MUTANTS:
        olds, news = (old if isinstance(old, list) else [old]), (new if isinstance(new, list) else [new])
        mutated, anchors_ok = patched, True
        for o, n in zip(olds, news):
            if mutated.count(o) != 1:
                ok(f"M mutant '{label}' anchor occurs once", False, f"{mutated.count(o)}x")
                anchors_ok = False
                break
            mutated = mutated.replace(o, n, 1)
        if not anchors_ok:
            continue
        try:
            compile(mutated, "mutant", "exec")
        except SyntaxError as e:
            ok(f"M mutant '{label}' compiles", False, str(e))
            continue
        bad = behaviour(ctx, mutated, mini=True)
        hit = bool(bad)
        caught += hit
        ok(f"M mutant caught: {label}" + (f"  (first failing check: {bad[0][:70]})" if bad else ""), hit)
    print(f"# mutants: {caught}/{len(MUTANTS)} caught")
    print(f"# p035_cpu_check: {N['pass']} passed, {N['fail']} failed ({time.time() - t0:.0f} s)")
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
