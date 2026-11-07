#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-031 gb10-idx-shard-gate] CPU checks (python3 + numpy; no torch, no GPU, no vLLM install).

  python3 files/p031_cpu_check.py [--tree <v0.32.0 vllm/ package dir>] [--quick]

The tree default is $VLLM_TREE, else vllm-v0.32.0/vllm (a path relative to the current directory). Sections:
  S  applier / static: pins, 4 anchors exactly once, AST placement, markers, compile, revert(apply(x)) == x, refusal of every
     mutated / duplicated / moved anchor and of 9 structurally broken patched files
  G  behaviour of the REAL patched functions (extracted by AST, run with a numpy fake torch): gate off == upstream bit for bit
     on 4,000 inputs; gate on == an independent reference of upstream's split behind the depth+rows gate; boundary table; env
     validation; failure containment; ACTIVE / INACTIVE boot line; first-sharded-step line; no log when off
  M  mutants of the patched code: every one must make some G check fail (the checks have teeth)
  C  call-site contracts read from the pristine v0.32.0 sources (argument order and meaning of seq_lens_cpu / query lens, the
     "context" definition, the op that narrows chunks and exchanges the top-k)
  P  the pins in 031-gb10-idx-shard-gate.sh (applier, base, result, the two read-only contracts)
Exit 1 on any [FAIL].
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p031_apply as A  # noqa: E402
import p031_testlib as TL  # noqa: E402

DEFAULT_TREE = Path(os.environ.get("VLLM_TREE", "vllm-v0.32.0/vllm"))
IDX_REL = "v1/attention/backends/mla/indexer.py"
OP_REL = "model_executor/layers/sparse_attn_indexer.py"
GLM_REL = "models/glm5next/nvidia/sparse_indexer.py"

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


BAKE_SCRIPT = HERE.parent.parent / "image" / "glm53_sm121_v032.py"


def baked_glm_text(tree: Path) -> str | None:
    """models/glm5next/nvidia/sparse_indexer.py after the SM121 bake (the image agent's script, edits of that file only)."""
    if not BAKE_SCRIPT.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("p031_bake_probe", BAKE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p031_bake_probe"] = mod
    spec.loader.exec_module(mod)
    txt = (tree / GLM_REL).read_text()
    for e in mod.EDITS:
        if e.pkg == "vllm" and e.rel == GLM_REL:
            if txt.count(e.old) != 1:
                return None
            txt = txt.replace(e.old, e.new, 1)
    return txt


# ---------------------------------------------------------------------------------------------------------------------
# random inputs
# ---------------------------------------------------------------------------------------------------------------------
def rand_case(rng: random.Random, big: bool = False):
    nreq = rng.randint(1, 6)
    qs, ss = [], []
    for _ in range(nreq):
        kind = rng.random()
        if big:
            q = rng.randint(30000, 90000)
        elif kind < 0.35:
            q = rng.randint(1, 300)
        elif kind < 0.85:
            q = rng.randint(1, 9000)
        else:
            q = rng.randint(9000, 40000)
        prior = rng.choice([0, 0, rng.randint(0, 3000), rng.randint(0, 70000), rng.randint(20000, 1_500_000)])
        qs.append(q)
        ss.append(q + prior)
    tp = rng.choice([1, 2, 4, 8, 8, 8])
    ratio = rng.choice([1, 4, 4, 4, 128])
    return ss, qs, tp, ratio


def expected_on(ss, qs, tp, ratio, min_ctx=32768, min_rows=256):
    rows = sum(qs)
    if tp < 2 or rows < max(min_rows, tp) or (max(ss) if ss else 0) < min_ctx:
        return None
    r = TL.ref_split(ss, qs, ratio, tp)
    return None if r is None else r[0]


# ---------------------------------------------------------------------------------------------------------------------
# G: behaviour checks (also used for the mutants: returns the list of failed check names, prints nothing)
# ---------------------------------------------------------------------------------------------------------------------
def behaviour(patched: str, base: str, tree: Path, quick: bool, report=None) -> list[str]:
    bad: list[str] = []

    def chk(name, cond, detail=""):
        if not cond:
            bad.append(name)
        if report is not None:
            report(name, cond, detail)

    try:
        _behaviour(patched, base, tree, quick, chk)
    except Exception as e:  # noqa: BLE001 - a mutant that breaks the module counts as caught
        bad.append(f"EXC:{type(e).__name__}:{str(e)[:80]}")
        if report is not None:
            report("behaviour suite completed", False, repr(e))
    return bad


def _behaviour(patched: str, base: str, tree: Path, quick: bool, chk) -> None:
    rng = random.Random(31)
    n_main = 600 if quick else 4000
    # ---- G1: gate off (unset) == upstream, bit for bit
    with TL.Harness(base, tree) as up, TL.Harness(patched, tree) as off0, TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "0"}) as off1, \
            TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": ""}) as off2:
        chk("G0 gate unset/0/empty: _P031_ON is False for all three", off0.cfg()[0] is False and off1.cfg()[0] is False and off2.cfg()[0] is False)
        mism = 0
        n_some = 0
        for i in range(n_main):
            ss, qs, tp, ratio = rand_case(rng, big=(i % 40 == 0))
            want = up.shard(ss, qs, tp, ratio)
            n_some += want is not None
            for h in (off0, off1, off2):
                if h.shard(ss, qs, tp, ratio) != want:
                    mism += 1
        chk(f"G1 gate off == upstream on {n_main} random inputs x 3 spellings ({n_some} engaged upstream's own gate)", mism == 0 and n_some > 0, f"{mism} mismatches, {n_some} engaged")
        chk("G1b gate off logs nothing", off0.cap.records == [] and off1.cap.records == [])
        # upstream threshold exactly: 16,384 x tp rows
        ss, qs = [200_000], [16384 * 8]
        chk("G1c gate off: 131,072 rows at tp 8 shard (upstream), 131,071 do not",
            off0.shard(ss, qs, 8) is not None and off0.shard([200_000], [16384 * 8 - 1], 8) is None)
        ann = off0.ns["_p031_announce"]
        cfg, dcp, pcp, tp = off0.supported_cfg()
        ann(cfg, dcp, pcp, tp)
        chk("G1d gate off: _p031_announce / _p031_note_engaged are silent", off0.cap.records == [])

    # ---- G2: gate on, defaults == independent reference behind the depth+rows gate
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}) as on:
        chk("G2a gate on: defaults (True, 32768, 256)", on.cfg() == (True, 32768, 256), str(on.cfg()))
        mism = 0
        n_engaged = n_none_depth = n_none_rows = 0
        for i in range(n_main):
            ss, qs, tp, ratio = rand_case(rng, big=(i % 50 == 0))
            want = expected_on(ss, qs, tp, ratio)
            got = on.shard(ss, qs, tp, ratio)
            if got != want:
                mism += 1
            n_engaged += want is not None
            if want is None and tp >= 2 and sum(qs) >= max(256, tp):
                n_none_depth += 1
            if want is None and tp >= 2 and sum(qs) < max(256, tp):
                n_none_rows += 1
        chk(f"G2b gate on == reference on {n_main} random inputs ({n_engaged} sharded, {n_none_depth} refused for depth, {n_none_rows} for rows)",
            mism == 0 and n_engaged > 100 and n_none_depth > 100 and n_none_rows > 20, f"{mism} mismatches")
        # properties of every engaged result
        bad_prop = 0
        for i in range(400 if not quick else 80):
            ss, qs, tp, ratio = rand_case(rng)
            got = on.shard(ss, qs, tp, ratio)
            if got is None:
                continue
            rows = sum(qs)
            if sum(got) != rows or len(got) != tp or min(got) < 1:
                bad_prop += 1
                continue
            _, per_row = TL.ref_split(ss, qs, ratio, tp)
            tot = sum(per_row)
            if rows >= 64 * tp and tot > 0:
                pos, mx = 0, 0
                for s in got:
                    mx = max(mx, sum(per_row[pos:pos + s]))
                    pos += s
                if mx - tot / tp > 2 * max(per_row) + tp:
                    bad_prop += 1
        chk("G3 engaged results: sum == rows, len == tp, every rank >= 1 row, work balanced within 2 rows' cost", bad_prop == 0, f"{bad_prop} bad")
        # ---- G4 boundary table (tp 8, one request, ratio 4)
        def one(ctx, rows, tp=8, h=on):
            return h.shard([ctx], [rows], tp)
        chk("G4a context 32,767 (rows 4096) -> replicated, 32,768 -> sharded",
            one(32767, 4096) is None and one(32768, 4096) is not None)
        chk("G4b rows 255 -> replicated, 256 -> sharded, 257 -> sharded (context 100k)",
            one(100_000, 255) is None and one(100_000, 256) is not None and one(100_000, 257) is not None)
        chk("G4c tp 1 -> replicated; tp 2 and tp 4 shard (4 pieces for tp 4)",
            one(100_000, 4096, tp=1) is None and one(100_000, 4096, tp=2) is not None and len(one(100_000, 4096, tp=4)) == 4)
        chk("G4d a multi-request step shards when ONE request is deep (the others shallow)",
            on.shard([40_000, 3_000, 2_000], [100, 1500, 1500], 8) is not None and on.shard([20_000, 3_000, 2_000], [100, 1500, 1500], 8) is None)
        chk("G4e empty step -> replicated, no exception", on.shard([], [], 8) is None)
        chk("G4f decisions are pure: the same input twice gives the same result", one(100_000, 4096) == one(100_000, 4096))
        chk("G4g the split sums to the step's rows for the operator's case (rows 7,680, ctx 262,144)",
            sum(one(262_144, 7680)) == 7680 and len(one(262_144, 7680)) == 8)
    # ---- G4h/G4i: overrides
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1", "VLLM_GB10_IDX_SHARD_MIN_CTX": "65536", "VLLM_GB10_IDX_SHARD_MIN_ROWS": "1024"}) as ov:
        chk("G4h overrides read: (True, 65536, 1024)", ov.cfg() == (True, 65536, 1024), str(ov.cfg()))
        chk("G4i overrides applied: ctx 65,535 / rows 1,023 refused; 65,536 and 1,024 accepted",
            ov.shard([65535], [4096], 8) is None and ov.shard([65536], [4096], 8) is not None
            and ov.shard([100_000], [1023], 8) is None and ov.shard([100_000], [1024], 8) is not None)
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": " 1 "}) as ws:
        chk("G4j whitespace around the gate value is tolerated", ws.cfg()[0] is True)
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1", "VLLM_GB10_IDX_SHARD_MIN_ROWS": "64"}) as lo:
        chk("G4k the row floor is max(MIN_ROWS, tp): tp 128 with 100 rows (MIN_ROWS 64) stays replicated, 128 rows shard one row each",
            lo.shard([100_000], [100], 128) is None and lo.shard([100_000], [128], 128) == [1] * 128)
    # ---- G5: env validation (a typo must not silently disable the gate)
    for k, v in (("VLLM_GB10_IDX_SHARD", "2"), ("VLLM_GB10_IDX_SHARD", "yes"), ("VLLM_GB10_IDX_SHARD", "true"),
                 ("VLLM_GB10_IDX_SHARD_MIN_CTX", "100"), ("VLLM_GB10_IDX_SHARD_MIN_CTX", "abc"),
                 ("VLLM_GB10_IDX_SHARD_MIN_CTX", "99999999999"), ("VLLM_GB10_IDX_SHARD_MIN_CTX", "-1"),
                 ("VLLM_GB10_IDX_SHARD_MIN_CTX", "1.5"), ("VLLM_GB10_IDX_SHARD_MIN_ROWS", "5"),
                 ("VLLM_GB10_IDX_SHARD_MIN_ROWS", "9000"), ("VLLM_GB10_IDX_SHARD_MIN_ROWS", "x")):
        e = {"VLLM_GB10_IDX_SHARD": "1"}
        e[k] = v
        raised = False
        try:
            TL.Harness(patched, tree, e).close()
        except ValueError:
            raised = True
        chk(f"G5 {k}={v!r} raises ValueError at import", raised)
    # the MIN_* values are ignored (and not validated) while the gate is off, like every other opt-in patch
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD_MIN_CTX": "abc"}) as inert:
        chk("G5b a bad MIN_CTX with the gate off is ignored (inert)", inert.cfg()[0] is False)
    # ---- G6: failure containment
    class Boom:
        def numel(self):
            return 3

        def max(self):
            raise RuntimeError("boom")

    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}) as on:
        g = on.ns["_p031_row_gate"]
        r1 = g(5000, Boom(), 8)
        r2 = g(5000, Boom(), 8)
        warns = [m for m in on.cap.lines("WARNING") if "[patch-031] depth gate failed" in m]
        chk("G6 a failing depth read keeps the replicated path, logs one warning, never raises", r1 is False and r2 is False and len(warns) == 1, str(on.cap.records))
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}) as on:
        empty = on.torch.tensor([], on.torch.int64)
        chk("G6b rows without any seq_lens (inconsistent input) -> replicated, silently (the numel guard, not the except)",
            on.ns["_p031_row_gate"](5000, empty, 8) is False and on.cap.records == [], str(on.cap.records))
    # ---- G7: ACTIVE / INACTIVE boot line
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}, dist_rank=5) as on:
        cfg, dcp, pcp, tp = on.supported_cfg()
        ann = on.ns["_p031_announce"]
        ann(cfg, dcp, pcp, tp)
        ann(cfg, dcp, pcp, tp)
        lines = on.cap.lines()
        chk("G7a supported: exactly one INFO line, 'ACTIVE (rank 5/8', no INACTIVE, second call silent",
            len(lines) == 1 and on.cap.records[0][0] == "INFO" and re.search(r"\[patch-031\] indexer prefill row-shard ACTIVE \(rank 5/8;", lines[0])
            and "INACTIVE" not in lines[0], str(on.cap.records))
        chk("G7b the ACTIVE line carries the gate values and mentions upstream's replaced gate",
            "context >= 32768 tokens and >= 256 rows" in lines[0] and "16384 rows per rank" in lines[0], lines[0] if lines else "")
    cases = [
        ("cudagraph mixed mode FULL", dict(mode="FULL"), {}, "cudagraph mixed mode is FULL"),
        ("DCP", dict(dcp=2), {}, "decode context parallel"),
        ("PCP", dict(pcp=True), {}, "prefill context parallel"),
        ("tp 1", dict(tp=1), {}, "tensor_parallel_size < 2"),
        ("symm mem", {}, {"VLLM_USE_NCCL_SYMM_MEM": True}, "VLLM_USE_NCCL_SYMM_MEM"),
        ("batch invariant", {}, {"VLLM_BATCH_INVARIANT": True}, "VLLM_BATCH_INVARIANT"),
        ("pynccl off", {}, {"VLLM_DISABLE_PYNCCL": True}, "VLLM_DISABLE_PYNCCL"),
    ]
    for name, kw, flags, why in cases:
        with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}, flags=flags) as on:
            cfg, dcp, pcp, tp = on.supported_cfg(**kw)
            on.ns["_p031_announce"](cfg, dcp, pcp, tp)
            recs = on.cap.records
            chk(f"G7c unsupported ({name}): one WARNING 'INACTIVE (rank 3/{tp}): ... ({why})' and no ACTIVE",
                len(recs) == 1 and recs[0][0] == "WARNING" and "row-shard INACTIVE (rank 3/" in recs[0][1] and why in recs[0][1]
                and not re.search(r"[^A-Za-z]ACTIVE", recs[0][1]), str(recs))
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}, dist_rank=None) as on:
        cfg, dcp, pcp, tp = on.supported_cfg()
        on.ns["_p031_announce"](cfg, dcp, pcp, tp)
        chk("G7d the rank import failing gives 'rank ?/8' and still ACTIVE, never raises",
            len(on.cap.records) == 1 and "ACTIVE (rank ?/8;" in on.cap.records[0][1], str(on.cap.records))
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}) as on:
        on.ns["_p031_announce"](object(), 1, False, 8)  # no compilation_config: must not raise
        chk("G7e an unexpected config object gives one INACTIVE 'announce failed' warning, never an exception",
            len(on.cap.records) == 1 and on.cap.records[0][0] == "WARNING" and "INACTIVE" in on.cap.records[0][1], str(on.cap.records))
    # ---- G8: first sharded step is logged once
    with TL.Harness(patched, tree, {"VLLM_GB10_IDX_SHARD": "1"}) as on:
        s1 = on.shard([262_144], [7680], 8)
        s2 = on.shard([270_000], [7680], 8)
        first = [m for m in on.cap.lines("INFO") if "first sharded indexer prefill step" in m]
        chk("G8 the first sharded step logs once: rows, per-rank split, deepest context; the second is silent",
            len(first) == 1 and "7680 rows" in first[0] and str(s1) in first[0] and "deepest context 262144 tokens" in first[0], str(on.cap.records))
        on.shard([1000], [1000], 8)
        chk("G8b a refused step logs nothing", len(on.cap.records) == 1)


# ---------------------------------------------------------------------------------------------------------------------
# S: applier / static
# ---------------------------------------------------------------------------------------------------------------------
def static_checks(base: str, tree: Path, sh_text: str, quick: bool) -> str:
    patched = A.apply(base)
    ok("S1 apply(pristine v0.32.0) succeeds", True)
    ok("S2 patched text compiles", compile(patched, "indexer.py", "exec") is not None)
    ok(f"S3 exactly {A.N_MARKERS} markers", patched.count(A.MARKER) == A.N_MARKERS)
    ok("S4 revert(apply(x)) == x byte for byte", A.revert(patched) == base)
    for name, old, _ in A.EDITS:
        ok(f"S5 anchor '{name}' occurs exactly once in the pristine file", base.count(old) == 1)
    for fn, what in ((A.apply, "apply twice"),):
        try:
            fn(patched)
            ok(f"S6 {what} refused", False)
        except A.Refused:
            ok(f"S6 {what} refused (marker present)", True)
    # mutated / deleted / duplicated anchors
    def refuses(text, label):
        try:
            A.apply(text)
            ok(label, False, "applied")
        except A.Refused:
            ok(label, True)
        except SyntaxError:
            ok(label, True)
    for name, old, _ in A.EDITS:
        i = base.index(old)
        mid = i + len(old) // 2
        refuses(base[:mid] + "X" + base[mid:], f"S7 mutated anchor refused: {name}")
        refuses(base.replace(old, "", 1), f"S7 deleted anchor refused: {name}")
        refuses(base + "\n" + old, f"S7 duplicated anchor refused: {name}")
    # anchors moved out of their function (still exactly once)
    c_old = A.C_OLD
    moved = base.replace(c_old, "", 1) + "\n\ndef _decoy():\n" + c_old
    refuses(moved, "S8a the sm_count anchor moved out of __init__ is refused")
    b_old = A.B_OLD
    moved = base.replace(b_old, "    pass\n", 1) + "\n\ndef _decoy(query_lens_cpu, tp_size):\n" + b_old + "    return 1\n"
    refuses(moved, "S8b the row-gate anchor moved out of balanced_prefill_row_shard is refused")
    # structurally broken PATCHED files: the post-apply structure check must refuse them
    broken = {
        "announce call removed": patched.replace("        _p031_announce(self.vllm_config, self.dcp_world_size, self.use_pcp, self.tp_size)\n", "        pass\n"),
        "row gate called with other arguments": patched.replace("_p031_row_gate(num_rows, seq_lens_cpu, tp_size)", "_p031_row_gate(num_rows, query_lens_cpu, tp_size)"),
        "gate `or` turned into `and`": patched.replace("if tp_size < 2 or not _p031_row_gate", "if tp_size < 2 and not _p031_row_gate"),
        "note call moved before sizes": patched.replace("    sizes = [bounds[i + 1] - bounds[i] for i in range(tp_size)]\n", "    _p031_note_engaged(num_rows, seq_lens_cpu, [])\n    sizes = [bounds[i + 1] - bounds[i] for i in range(tp_size)]\n"),
        "config assignment removed": patched.replace("_P031_ON, _P031_MIN_CTX, _P031_MIN_ROWS = _p031_config()", "_P031_ON, _P031_MIN_CTX, _P031_MIN_ROWS = (False, 32768, 256)"),
        "helper used elsewhere": patched.replace("    return sizes\n", "    _p031_announce(None, 1, False, 1)\n    return sizes\n", 1),
        "block detached from the constant": patched.replace("MIN_TP_SHARD_ROWS_PER_RANK = 16_384\n", "MIN_TP_SHARD_ROWS_PER_RANK = 16_384\nPAD = 1\n", 1),
        "second row-gate def": patched.replace("def _p031_announce(", "def _p031_row_gate(num_rows, seq_lens_cpu, tp_size):\n    return True\n\n\ndef _p031_announce(", 1),
        "announce after sm_count": patched.replace("        _p031_announce(self.vllm_config, self.dcp_world_size, self.use_pcp, self.tp_size)\n\n        sm_count = num_compute_units(self.device.index)\n", "        sm_count = num_compute_units(self.device.index)\n        _p031_announce(self.vllm_config, self.dcp_world_size, self.use_pcp, self.tp_size)\n", 1),
    }
    for label, text in broken.items():
        same = text == patched
        if same:
            ok(f"S9 broken-structure mutant '{label}' really differs from the patched file", False)
            continue
        try:
            A.check_structure(text, patched=True)
            ok(f"S9 broken structure refused: {label}", False)
        except A.Refused:
            ok(f"S9 broken structure refused: {label}", True)
    ok("S10 check_structure accepts the genuine patched file", (A.check_structure(patched, patched=True) or True))
    return patched


# ---------------------------------------------------------------------------------------------------------------------
# M: mutants of the patched code
# ---------------------------------------------------------------------------------------------------------------------
MUTANTS = [
    ("depth compare >= -> >", "return int(seq_lens_cpu.max()) >= _P031_MIN_CTX if", "return int(seq_lens_cpu.max()) > _P031_MIN_CTX if"),
    ("rows floor without tp", "if num_rows < max(_P031_MIN_ROWS, tp_size):", "if num_rows < _P031_MIN_ROWS:"),
    ("rows compare < -> <=", "if num_rows < max(_P031_MIN_ROWS, tp_size):", "if num_rows <= max(_P031_MIN_ROWS, tp_size):"),
    ("depth ignored", "return int(seq_lens_cpu.max()) >= _P031_MIN_CTX if seq_lens_cpu.numel() else False", "return True"),
    ("depth uses min instead of max", "return int(seq_lens_cpu.max()) >= _P031_MIN_CTX if seq_lens_cpu.numel() else False", "return int(seq_lens_cpu.min()) >= _P031_MIN_CTX if seq_lens_cpu.numel() else False"),
    ("off path loses upstream's gate", "return num_rows >= MIN_TP_SHARD_ROWS_PER_RANK * tp_size\n    try:", "return True\n    try:"),
    ("off path threshold x2", "return num_rows >= MIN_TP_SHARD_ROWS_PER_RANK * tp_size\n    try:", "return num_rows >= 2 * MIN_TP_SHARD_ROWS_PER_RANK * tp_size\n    try:"),
    ("gate on by default", 'return False, 32768, 256', 'return True, 32768, 256'),
    ("default MIN_CTX 16384", "_int(\"VLLM_GB10_IDX_SHARD_MIN_CTX\", 32768,", "_int(\"VLLM_GB10_IDX_SHARD_MIN_CTX\", 16384,"),
    ("default MIN_ROWS 128", "_int(\"VLLM_GB10_IDX_SHARD_MIN_ROWS\", 256,", "_int(\"VLLM_GB10_IDX_SHARD_MIN_ROWS\", 128,"),
    ("typo accepted", 'if gate != "1":\n        raise ValueError', 'if False:\n        raise ValueError'),
    ("range check dropped", "if not raw.isdigit() or not lo <= int(raw) <= hi:", "if not raw.isdigit():"),
    ("exception escapes the gate", "    except Exception as e:  # noqa: BLE001 - the replicated path", "    except ZeroDivisionError as e:  # noqa: BLE001 - the replicated path"),
    ("announce repeats", "    if not _P031_ON or _P031_STATE[\"announced\"]:\n        return\n    _P031_STATE[\"announced\"] = True", "    if not _P031_ON:\n        return"),
    ("announce says ACTIVE when unsupported", "if tp_prefill_row_sharding_supported(vllm_config, dcp_world_size, use_pcp, tp_size):", "if True:"),
    ("announce loses the rank", "rank, tp_size, _P031_MIN_CTX, _P031_MIN_ROWS, MIN_TP_SHARD_ROWS_PER_RANK,", "0, tp_size, _P031_MIN_CTX, _P031_MIN_ROWS, MIN_TP_SHARD_ROWS_PER_RANK,"),
    ("engaged log repeats", "    if not _P031_ON or _P031_STATE[\"engaged\"]:\n        return\n    _P031_STATE[\"engaged\"] = True", "    if not _P031_ON:\n        return"),
    ("engaged log when off", "    if not _P031_ON or _P031_STATE[\"engaged\"]:\n        return\n    _P031_STATE[\"engaged\"] = True", "    if _P031_STATE[\"engaged\"]:\n        return\n    _P031_STATE[\"engaged\"] = True"),
    ("sizes returned reversed", "    _p031_note_engaged(num_rows, seq_lens_cpu, sizes)\n    return sizes", "    _p031_note_engaged(num_rows, seq_lens_cpu, sizes)\n    return sizes[::-1]"),
    ("tp<2 check dropped", "if tp_size < 2 or not _p031_row_gate(num_rows, seq_lens_cpu, tp_size):", "if not _p031_row_gate(num_rows, seq_lens_cpu, tp_size):"),
    ("empty-step guard dropped", "if seq_lens_cpu.numel() else False", ""),
]


# ---------------------------------------------------------------------------------------------------------------------
# C: call-site contracts in the pristine sources
# ---------------------------------------------------------------------------------------------------------------------
def contract_checks(base: str, tree: Path) -> None:
    mod = ast.parse(base)
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == A.BUILDER)
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    calls = [c for c in ast.walk(build) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == A.FUNC]
    ok("C1 build() calls balanced_prefill_row_shard exactly once", len(calls) == 1)
    args = [ast.unparse(a) for a in calls[0].args] if calls else []
    ok("C2 argument order (prefill-only seq lens, prefill query lens, compress_ratio, tp_size)",
       args == ["seq_lens_cpu[num_decodes:]", "prefill_query_lens_cpu", "self.compress_ratio", "self.tp_size"], str(args))
    ok("C3 prefill_query_lens_cpu = diff of query_start_loc_cpu over the prefill requests (its sum = the step's prefill rows)",
       "prefill_query_lens_cpu = torch.diff(\n                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]\n            )" in base)
    ok("C4 'context' of the old patch (max_prefill_seq_len) is the same seq_lens_cpu[num_decodes:].max()",
       "int(seq_lens_cpu[num_decodes:].max().item())" in base)
    ok("C5 the call sits behind upstream's tp_prefill_row_sharding_supported() predicate",
       "if tp_prefill_row_sharding_supported(\n                self.vllm_config, self.dcp_world_size, self.use_pcp, self.tp_size\n            ):\n                row_shard_sizes = balanced_prefill_row_shard(" in base)
    ok("C6 MIN_TP_SHARD_ROWS_PER_RANK is used only at its definition and in the gate (nothing else depends on it)",
       len(re.findall(r"\bMIN_TP_SHARD_ROWS_PER_RANK\b", base)) == 2)
    ok("C7 balanced_prefill_row_shard has no other caller in the file", len(re.findall(r"\bbalanced_prefill_row_shard\(", base)) == 2)  # def + 1 call
    op = (tree / OP_REL).read_text()
    glms = [("pristine", (tree / GLM_REL).read_text())]
    baked = baked_glm_text(tree)
    if baked is not None:
        glms.append(("SM121-baked", baked))
    else:
        print("[SKIP] C8-C10 on the SM121-baked sparse_indexer.py: image/glm53_sm121_v032.py not found or its anchors do not match")
    for label, glm in glms:
        ok(f"C8 [{label}] the op narrows each chunk by PrefillRowShard.narrow and skips chunks outside this rank's slice",
           "narrowed = row_shard.narrow(chunk)" in glm and "if narrowed is None:" in glm and "continue" in glm)
        ok(f"C9 [{label}] the op exchanges `topk_tokens + index_kpool - 1` columns with exchange_topk after the chunk loop",
           "row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)" in glm)
        ok(f"C10 [{label}] short prefills (<= topk_tokens context) never shard (row_shard = None)",
           "row_shard = None\n        if not short_prefill:" in glm)
    ok("C11 exchange_topk is an all_gatherv over the TP group with the metadata's sizes",
       "get_tp_group().all_gatherv(" in op and "sizes=self.sizes" in op)
    ok("C12 PrefillRowShard.from_metadata reads prefill_metadata.row_shard_sizes (the builder's decision)",
       'sizes = getattr(prefill_metadata, "row_shard_sizes", None)' in op)
    cg = (tree / "config/compilation.py").read_text()
    ok("C13 CUDAGraphMode.mixed_mode() is the second tuple element for FULL_AND_PIECEWISE (so only a plain FULL disables the shard)",
       "FULL_AND_PIECEWISE = (FULL, PIECEWISE)" in cg and "return CUDAGraphMode(self.value[1]) if self.separate_routine() else self" in cg)


# ---------------------------------------------------------------------------------------------------------------------
def pin(sh_text: str, name: str) -> str | None:
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", sh_text, re.M)
    return m.group(1) if m else None


def pin_checks(base: str, patched: str, tree: Path, sh_text: str) -> None:
    ok("P1 IDX_ORIG pin == sha256 of the pristine v0.32.0 indexer.py", pin(sh_text, "IDX_ORIG") == sha(base.encode()), str(pin(sh_text, "IDX_ORIG")))
    ok("P2 IDX_P031 pin == sha256 of apply(pristine)", pin(sh_text, "IDX_P031") == sha(patched.encode()), str(pin(sh_text, "IDX_P031")))
    ok("P3 APPLIER_SHA pin == sha256 of files/p031_apply.py", pin(sh_text, "APPLIER_SHA") == sha((HERE / "p031_apply.py").read_bytes()), str(pin(sh_text, "APPLIER_SHA")))
    ok("P4 OP_ORIG pin == sha256 of sparse_attn_indexer.py", pin(sh_text, "OP_ORIG") == sha((tree / OP_REL).read_bytes()))
    ok("P5 GLMIDX_ORIG pin == sha256 of models/glm5next/nvidia/sparse_indexer.py", pin(sh_text, "GLMIDX_ORIG") == sha((tree / GLM_REL).read_bytes()))
    baked = baked_glm_text(tree)
    if baked is None:
        print("[SKIP] P5b GLMIDX_BAKED pin vs the bake script's result: image/glm53_sm121_v032.py not found")
    else:
        ok("P5b GLMIDX_BAKED pin == sha256 of sparse_indexer.py after the SM121 bake's topk_neg1 edits (image/glm53_sm121_v032.py)",
           pin(sh_text, "GLMIDX_BAKED") == sha(baked.encode()), f"{pin(sh_text, 'GLMIDX_BAKED')} vs {sha(baked.encode())}")
    ok("P6 MARKERS in the .sh == N_MARKERS of the applier", re.search(r"^MARKERS=(\d+)$", sh_text, re.M).group(1) == str(A.N_MARKERS))
    ok("P7 the .sh's MARKER string == the applier's", f'MARKER="{A.MARKER}"' in sh_text)
    for needle in ("[patch-031] indexer prefill row-shard ACTIVE", "INACTIVE", "first sharded indexer prefill step"):
        ok(f"P8 the .sh documents the log line {needle!r}", needle in sh_text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=str(DEFAULT_TREE))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    tree = Path(a.tree)
    base = (tree / IDX_REL).read_text()
    sh_text = (HERE.parent / "031-gb10-idx-shard-gate.sh").read_text()
    print(f"# p031_cpu_check: tree {tree}, base sha256 {sha(base.encode())}")
    patched = static_checks(base, tree, sh_text, a.quick)
    contract_checks(base, tree)
    pin_checks(base, patched, tree, sh_text)

    def rep(name, cond, detail=""):
        ok(name, cond, detail)

    bad = behaviour(patched, base, tree, a.quick, report=rep)
    ok("G* behaviour suite on the genuine patched code: all checks passed", not bad, str(bad))
    # ---- M: mutants
    caught = 0
    for label, old, new in MUTANTS:
        if patched.count(old) != 1:
            ok(f"M mutant '{label}' anchor occurs once", False, f"{patched.count(old)}x")
            continue
        mutated = patched.replace(old, new, 1)
        bad = behaviour(mutated, base, tree, True)
        hit = bool(bad)
        caught += hit
        ok(f"M mutant caught: {label}" + (f"  (first failing check: {bad[0]})" if bad else ""), hit)
    print(f"# mutants: {caught}/{len(MUTANTS)} caught")
    print(f"# p031_cpu_check: {N['pass']} passed, {N['fail']} failed")
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
