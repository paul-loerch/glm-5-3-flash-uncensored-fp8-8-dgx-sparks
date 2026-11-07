#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-032 gb10-repl-embed] independent verification suite (written by the 032 verifier, not the builder), vLLM v0.32.0 port.

  python3 -I -B files/p032_verify_cpu_check.py --vllm-root <v0.32.0 vllm> [--full-tree <v0.32.0 vllm>] [--work DIR]
                                               [--mutants]

Mac only (python3 + numpy; no torch, no GPU, no vLLM install). From the builder's files it reuses ONLY scaffolding: the
numpy fake torch / Rendezvous / thread World of files/p032_testlib.py and build_ns() of files/p032_cpu_check.py (the stub
vllm namespace into which the REAL v0.32.0 vocab_parallel_embedding.py, model.py, mtp.py, eagle/utils.py, PluggableLayer
and process_weights_after_loading are exec'd, with the fused ops.vocab_parallel_embedding kernel emulated from
csrc/vocab_parallel_embedding_kernels.cu). Every expected value and every pass / fail decision below is computed here:
  V0  exhaustive bf16 algebra in plain numpy (own bf16 codec): for ALL 65,536 bit patterns x at every owner position,
      patch 012's one-shot (fp32 acc, one rounding), every NCCL ring rotation (per-hop bf16 rounding) and a pairwise
      tree return canon(x) (x with 0x8000 -> 0x0000) for every finite x (the fused kernel writes the owner's row as a
      byte copy and every other row as all-zero bytes = +0.0); NaN / Inf are the only other non-identities (the boot
      refuses non-finite tables); with flush-to-zero exactly the subnormals change (what the boot must catch).
  V1  the REAL module on 8 threaded ranks, own scenarios:
      a  a table holding EVERY finite bf16 pattern: ACTIVE (one-shot + ring), forward == own numpy reference
         canon(table)[id] / zero rows for 7 out-of-range ids, bitwise, 0 all-reduces; the FTZ ring -> INACTIVE;
      b  adversarial reductions the boot must not assume away: sign-preserving (keeps -0.0), all-zero inputs -> -0.0
         (on a table with no zeros: only the out-of-range probes see it), a bit flip only in 16-token / 48-token /
         3,840-token all-reduces (NCCL protocol sizes that only the probes exercise) -> INACTIVE, layer == unpatched;
      c  a digest-invisible corruption (two elements swapped inside one non-probe row of the LAST exhaustive chunk, on
         one rank: same row sums, same counts) -> INACTIVE (the exhaustive bitwise check alone);
      d  systematic fault injection: every boot helper x {first, middle, last} call x 2 ranks raises once -> no hang,
         the same decision on all 8 ranks, INACTIVE, layer == unpatched (state + bytes), forward == unpatched;
      e  a CUDA-graph replay that corrupts one element on one rank -> INACTIVE on all 8.
  V2  the REAL model path (patched model.py, REAL Glm5NextModel.load_weights / process_weights_after_loading /
      Glm5NextMTP / load_eagle_model), checkpoint rows all distinct: ACTIVE table == canon(ckpt) on every rank; forced
      INACTIVE: rank r's weight == ckpt rows [rV/8, (r+1)V/8) (no offset bug) == the unpatched weight_loader bytes;
      lm_head untouched ([V/8, H], same bytes, shared into the draft); draft embed_tokens is the target's; gate unset
      -> unpatched class and no [patch-032] line.
  V3  the .sh: gate values; the 029 conflict versus the REAL 029 .sh (every case 029 accepts with eh in its parts is
      refused by 032); unset / 0 / '' on a full-tree clone touch NO file (size + mtime + inode snapshot).
  V4  log hygiene with the REAL grep: every line the .sh / applier / module print (captured from real runs) against the
      readiness.txt fail rules (testdata/readiness.txt, or the recipe's launch/readiness.txt when present) and 032's own
      count / must-be-0 patterns.
  V6  (--full-tree) the REAL entrypoint chain 001..031 (+ 032) on APFS clones for several env configurations: model.py
      always ends as one of the pinned 032 results (or a refusal), vocab_parallel_embedding.py / eagle/utils.py /
      model_loader utils + base_loader + weight_tying never change, mtp.py only by 029 (== MTP_P029), only 025 / 032 write
      model.py. (V5, the 'Model loading took' readiness-range rule of the day-0 recipe, is not part of the port.)
  M   (--mutants) planted bugs in the module and in the .sh, each must make a quick run fail.
Exit 0 = ALL PASS (SKIP lines are counted separately and are never passes). Never deletes anything; scratch output under
--work.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
RDIR = PDIR.parent
SH = PDIR / "032-gb10-repl-embed.sh"
MODULE = HERE / "p032_gb10_repl_embed.py"
APPLIER = HERE / "p032_apply.py"
P025_APPLIER = HERE / "p025_apply.py"
READINESS = [p for p in (Path(os.environ["PATCH_READINESS_TXT"]) if os.environ.get("PATCH_READINESS_TXT") else None,
                        HERE / "testdata" / "readiness.txt", RDIR / "launch" / "readiness.txt") if p is not None and p.exists()]
sys.path.insert(0, str(HERE))
RES = {"pass": 0, "fail": 0, "skip": 0, "failed": [], "skipped": []}
QUIET = {"on": False}
W = 8


def check(name, ok, info=""):
    ok = bool(ok)
    RES["pass" if ok else "fail"] += 1
    if not ok:
        RES["failed"].append(name)
    if not QUIET["on"] or not ok:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{str(info)[:400]}]" if info else ""), flush=True)
    return ok


def skip(name, why):
    RES["skip"] += 1
    RES["skipped"].append(name)
    if not QUIET["on"]:
        print(f"SKIP {name}  [{why}]", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(name, sh=SH):
    """The 64-hex pin, None while the line is a placeholder (pending-...)."""
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", Path(sh).read_text(), re.M)
    return m.group(1) if m else None


# =====================================================================================================================
# V0 exhaustive bf16 algebra (own codec, no testlib)
# =====================================================================================================================
def bf16_to_f32(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    """round-to-nearest-even; NaN -> quiet NaN with the top payload bits."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    return np.where(nan, ((u >> 16) | 0x40).astype(np.uint32), r).astype(np.uint16)


def ftz(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where((x != 0) & (np.abs(x) < np.float32(2.0 ** -126)), np.copysign(np.float32(0.0), x), x)


def v0_algebra():
    with np.errstate(invalid="ignore", over="ignore"):
        _v0_algebra()


def _v0_algebra():
    print("== V0 exhaustive bf16 algebra (every bit pattern, every owner position)")
    allp = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    finite = (allp & 0x7F80) != 0x7F80
    canon = allp.copy()
    canon[canon == 0x8000] = 0
    zero = np.zeros(65536, np.uint16)
    oneshot_ok = ring_ok = tree_ok = True
    for owner in range(W):
        rows = [allp if r == owner else zero for r in range(W)]
        acc = np.zeros(65536, np.float32)                       # patch 012: acc = +0.0; acc += own + got
        for r in range(W):
            own = bf16_to_f32(rows[r]) if r == owner else np.zeros(65536, np.float32)
            got = np.zeros(65536, np.float32) if r == owner else bf16_to_f32(rows[r])
            acc = (acc + (own + got)).astype(np.float32)
        out = f32_to_bf16(acc)
        oneshot_ok &= bool(np.array_equal(out[finite], canon[finite]))
        for s in range(W):                                      # ring: start at s, add one rank per hop, round
            acc = bf16_to_f32(rows[s])
            for k in range(1, W):
                acc = bf16_to_f32(f32_to_bf16(acc + bf16_to_f32(rows[(s + k) % W])))
            ring_ok &= bool(np.array_equal(f32_to_bf16(acc)[finite], canon[finite]))
        lv = [bf16_to_f32(x) for x in rows]                     # pairwise tree
        while len(lv) > 1:
            lv = [bf16_to_f32(f32_to_bf16(lv[i] + lv[i + 1])) for i in range(0, len(lv), 2)]
        tree_ok &= bool(np.array_equal(f32_to_bf16(lv[0])[finite], canon[finite]))
    check("V0 one-shot (fp32 acc, one rounding): all finite patterns x at all 8 owner positions -> canon(x)", oneshot_ok)
    check("V0 bf16 ring (every rotation, per-hop rounding) -> canon(x) for every finite pattern", ring_ok)
    check("V0 pairwise bf16 tree -> canon(x) for every finite pattern", tree_ok)
    changed = finite & (allp != canon)
    check("V0 the only finite pattern the reduction changes is 0x8000 (-0.0 -> +0.0)",
          np.array_equal(np.nonzero(changed)[0], [0x8000]))
    rows = [allp if r == 3 else zero for r in range(W)]
    acc = ftz(bf16_to_f32(rows[0]))
    for k in range(1, W):
        acc = bf16_to_f32(f32_to_bf16(ftz(acc + ftz(bf16_to_f32(rows[k])))))
    out = f32_to_bf16(ftz(acc))
    sub = finite & ((allp & 0x7F80) == 0) & ((allp & 0x7FFF) != 0)
    diff = finite & (out != canon)
    check("V0 a flush-to-zero ring changes exactly the 254 subnormal patterns (the boot must see them: special rows)",
          np.array_equal(diff, sub) and int(sub.sum()) == 254)


# =====================================================================================================================
# scaffolding (builder's build_ns + fake torch; own decisions)
# =====================================================================================================================
def scaffolding():
    import p032_cpu_check as C  # noqa: E402  (main() is guarded)
    import p032_testlib as T  # noqa: E402
    return C, T


def all_pattern_table(H=16):
    p = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    p[(p & 0x7F80) == 0x7F80] = 0x3F80              # NaN / Inf slots -> 1.0 (the boot refuses non-finite tables)
    return p.reshape(65536 // H, H)


def rand_table(V, H, seed, zeros=True):
    rng = np.random.default_rng(seed)
    b = np.asarray((rng.standard_normal((V, H)) * 0.02).astype(np.float32).view(np.uint32) >> 16, np.uint16)
    if zeros:
        b[2, 3] = 0x8000
        b[V - 5, :] = 0x8000
        b[V // 2 + 3, 1] = 0x0000
    else:
        b[(b & 0x7FFF) == 0] = 0x3C00              # no +/-0.0 at all
    return b


def own_ref(bits, ids):
    """canon(table)[id], zero rows for out-of-range ids - straight from the checkpoint, no module / image code."""
    V, H = bits.shape
    c = bits.copy()
    c[c == 0x8000] = 0
    out = np.zeros((len(ids), H), np.uint16)
    for i, t in enumerate(ids):
        if 0 <= t < V:
            out[i] = c[t]
    return out


def oob(V):
    return [-1, -2, V, V + 1, -V, (1 << 31) - 1, -(1 << 31)]


class Boot:
    """8 threaded ranks: construct via embedding_class, load with the REAL weight_loader, run the REAL
    process_weights_after_loading, then evaluate with own references."""

    def __init__(self, vroot, module_path, C, T):
        self.vroot, self.module_path, self.C, self.T = vroot, module_path, C, T

    def run(self, V, H, bits_of, *, oneshot_tokens=8, ftz=False, chunk=None, before=None, arith=None, ids=None,
            env=None, timeout=60.0):
        C, T = self.C, self.T
        ns = C.build_ns(self.vroot, None, self.module_path, env=env or {"VLLM_GB10_REPL_EMBED": "1"},
                        with_model=False)
        if chunk is not None:
            ns.mod.CHUNK_IDS = chunk
        cfg = C.mk_cfg(V, H)
        vcfg = C.mk_vcfg(cfg)
        rv = T.Rendezvous(W, timeout=5.0)
        ids = ids if ids is not None else list(range(V)) + oob(V)
        saved = (T.oneshot_sum, T.ring_sum)
        if arith is not None:
            T.oneshot_sum, T.ring_sum = arith, (lambda arrs, ftz=False, nchunks=4: arith(arrs))

        def body(r):
            w = C.setup_rank(rv, r, H, vcfg, ftz=ftz, oneshot_tokens=oneshot_tokens)
            bits = bits_of(r)
            with ns.torch.device("cuda"):
                lay = ns.mod.embedding_class(cfg)(V, H, prefix="model.embed_tokens")
            lay.weight.weight_loader(lay.weight, C.cpu_tensor(ns, bits))
            if before is not None:
                before(r, ns, lay, w)
            lay = C.post_load(ns, lay)
            me = threading.current_thread().name
            cnt = {n: c for (t, n), c in dict(getattr(ns.mod, "_VF_COUNT", {})).items() if t == me}
            n_boot = len(w.ar_calls)
            img = C.image_layer(ns, V, H, bits)
            ref_img = C.image_lookup(img, ids, ns)
            n0 = len(w.ar_calls)
            got = C.local_lookup(lay, ids, ns)
            return {"mode": C.mode_of(ns, lay), "got": got, "img": ref_img, "fwd_ars": len(w.ar_calls) - n0,
                    "n_boot": n_boot, "boot_counts": cnt, "state_eq": (C.image_state(lay) == C.image_state(img))
                    if C.mode_of(ns, lay) == "image" else None,
                    "bytes_eq": bool(np.array_equal(lay.weight._a, img.weight._a))
                    if C.mode_of(ns, lay) == "image" else None,
                    "table": np.array(lay.weight._a, copy=True) if C.mode_of(ns, lay) == "replicated" else None}
        try:
            out = C.run8(body, timeout=timeout)
        finally:
            T.oneshot_sum, T.ring_sum = saved
        errs = [repr(x) for ok, x in out if not ok]
        rs = [x for ok, x in out if ok]
        return NS(ns=ns, rs=rs, errs=errs, ids=ids, hang=any(isinstance(x, T.Hang) for ok, x in out if not ok),
                  act=T.LOGS.lines(contains="replicated input embedding ACTIVE ("),
                  ina=T.LOGS.lines(contains="replicated input embedding INACTIVE ("),
                  logs=[m for (_, _, m) in T.LOGS.records],
                  modes=sorted({x["mode"] for x in rs}))


def inactive_ok(sc):
    return (not sc.errs and not sc.hang and len(sc.rs) == W and sc.modes == ["image"] and len(sc.ina) == W
            and not sc.act and all(x["state_eq"] and x["bytes_eq"] and np.array_equal(x["got"], x["img"])
                                   and x["fwd_ars"] >= 1 for x in sc.rs))


def active_ok(sc, bits_of, want_ref=True):
    if sc.errs or sc.hang or len(sc.rs) != W or sc.modes != ["replicated"] or len(sc.act) != W or sc.ina:
        return False
    for r, x in enumerate(sc.rs):
        if not np.array_equal(x["got"], x["img"]) or x["fwd_ars"] != 0:
            return False
        if want_ref and not np.array_equal(x["got"], own_ref(bits_of(r), sc.ids)):
            return False
    return True


# =====================================================================================================================
# V1 module scenarios
# =====================================================================================================================
def adv_sign_preserving(arrs):
    """sum of the contributions that are not +0.0; if all are zeros, -0.0 survives (a reduction that keeps the owner's
    zero sign) - the boot must not assume NCCL / one-shot canonicalise -0.0."""
    xs = np.stack([bf16_to_f32(a) for a in arrs]).astype(np.float32)
    pz = np.stack([np.asarray(a) == 0 for a in arrs])
    s = np.where(pz, np.float32(0.0), xs).sum(axis=0, dtype=np.float32)
    anyneg = np.stack([np.asarray(a) == 0x8000 for a in arrs]).any(axis=0)
    allz = np.stack([(np.asarray(a) & 0x7FFF) == 0 for a in arrs]).all(axis=0)
    out = f32_to_bf16(s)
    out[allz & anyneg] = 0x8000
    return out.reshape(np.asarray(arrs[0]).shape)


def adv_allzero_negzero(arrs):
    """exact sum, but an element whose inputs are ALL +0.0 comes back as -0.0."""
    acc = np.zeros(np.asarray(arrs[0]).shape, np.float32)
    for a in arrs:
        acc = (acc + bf16_to_f32(a)).astype(np.float32)
    out = f32_to_bf16(acc)
    allpz = np.stack([np.asarray(a) == 0 for a in arrs]).all(axis=0)
    out[allpz] = 0x8000
    return out


def flip_on_tokens(n_tokens):
    def hook_factory(T):
        def arith(arrs):
            acc = np.zeros(np.asarray(arrs[0]).shape, np.float32)
            for a in arrs:
                acc = (acc + bf16_to_f32(a)).astype(np.float32)
            out = f32_to_bf16(acc)
            if out.ndim == 2 and out.shape[0] == n_tokens:
                out = out.copy()
                out[n_tokens // 2, 0] ^= np.uint16(0x0001)
            return out
        return arith
    return hook_factory


def v1_module(vroot, module_path, C, T, quick=False):
    print("== V1 the REAL module on 8 threaded ranks (own scenarios / references)")
    B = Boot(vroot, module_path, C, T)
    # a) every finite bf16 pattern
    tab = all_pattern_table(16)
    V, H = tab.shape
    same = lambda r: tab  # noqa: E731
    for name, kw in (("one-shot <= 8 tokens, ring above", {}), ("ring only", {"oneshot_tokens": 0})):
        sc = B.run(V, H, same, **kw)
        check(f"V1a every finite bf16 pattern ({V} x {H}), {name}: ACTIVE on 8 ranks, forward == own canon(table) "
              "reference bitwise (all ids + 7 out-of-range), 0 all-reduces",
              active_ok(sc, same), f"{sc.errs[:1]} {sc.modes} {sc.ina[:1]}")
        check(f"V1a ({name}) the table is canon(checkpoint): only the 0x8000 elements rewritten",
              len(sc.rs) == W and all(x["table"] is not None and np.array_equal(x["table"], own_ref(tab, list(range(V))))
                                      for x in sc.rs))
        if quick:
            break
    sc = B.run(V, H, same, oneshot_tokens=0, ftz=True)
    check("V1a every finite pattern + a flush-to-zero ring: INACTIVE on 8 ranks, layer == image, forward == image",
          inactive_ok(sc), f"{sc.errs[:1]} {sc.modes} {sc.ina[:1]}")
    # b) adversarial reductions
    V, H = 512, 16
    bz = rand_table(V, H, 7, zeros=True)
    bnz = rand_table(V, H, 8, zeros=False)
    cases = [("a sign-preserving reduction (keeps -0.0)", bz, adv_sign_preserving),
             ("all-zero inputs -> -0.0, table without any zero (only the out-of-range probes see it)", bnz,
              adv_allzero_negzero)]
    if not quick:
        cases += [(f"a bit flip only in {n}-token all-reduces", bz, flip_on_tokens(n)(T)) for n in (16, 48, 3840)]
    for name, bits, arith in cases:
        sc = B.run(V, H, lambda r, b=bits: b, arith=arith)
        check(f"V1b {name} -> INACTIVE on 8 ranks, layer == image (state + bytes), forward == image", inactive_ok(sc),
              f"{sc.errs[:1]} {sc.modes} {sc.ina[:1]}")
    # control: the same harness with the honest arithmetic goes ACTIVE (the scenarios above are not vacuous)
    sc = B.run(V, H, lambda r: bnz)
    check("V1b control: a table without zeros + honest arithmetic -> ACTIVE, own reference bitwise",
          active_ok(sc, lambda r: bnz), f"{sc.errs[:1]} {sc.modes} {sc.ina[:1]}")
    # c) digest-invisible corruption inside the last exhaustive chunk
    ns = C.build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
    probes = set(ns.mod.probe_ids(V, V // W, W, [2, V - 5, V // 2 + 3]))
    row = next(i for i in range(V - 1, 384, -1) if i not in probes and len(set(bz[i].tolist())) > 1)
    bad = bz.copy()
    j0, j1 = next((a, b) for a in range(H) for b in range(a + 1, H) if bad[row, a] != bad[row, b])
    bad[row, j0], bad[row, j1] = bad[row, j1], bad[row, j0]
    sc = B.run(V, H, lambda r: bad if r == 3 else bz, chunk=128)
    check(f"V1c two elements swapped in row {row} (last exhaustive chunk, not a probe id) on rank 3 only - same row "
          "sums, same counts, so the content exchange cannot see it -> INACTIVE on all 8 (the exhaustive check)",
          inactive_ok(sc) and all("calls differ" in m for m in sc.ina), f"{sc.errs[:1]} {sc.modes} {sc.ina[:1]}")
    # d) systematic fault injection
    v1d_faults(vroot, module_path, C, T, quick=quick)


FAULT_FNS = ("_envelope_refusal", "_layout_refusal", "_scan_table", "build_plan", "_plan_sig", "fingerprint",
             "_canonicalize_zeros", "_count_negzero", "_call_ids", "_image_view", "_lookup", "_graph_check")


def instrumented(module_path, work):
    """own instrumentation: wrap the module's helpers; _VF = {fn, rank, k} makes the k-th call of fn on rank raise
    once; counting always on (per thread)."""
    extra = f"""

# ---- p032_verify_cpu_check.py fault injection (test only) ----
import threading as _vth
_VF = {{"fn": None, "rank": None, "k": None}}
_VF_COUNT = {{}}
_VF_LOCK = _vth.Lock()
def _vf_wrap(_name, _f):
    def _w(*a, **k):
        t = _vth.current_thread().name
        with _VF_LOCK:
            c = _VF_COUNT.get((t, _name), 0) + 1
            _VF_COUNT[(t, _name)] = c
        if _VF["fn"] == _name and t == "rank%d" % (_VF["rank"] if _VF["rank"] is not None else -1) and c == _VF["k"]:
            raise RuntimeError("verifier fault %s #%d" % (_name, c))
        return _f(*a, **k)
    return _w
for _n in {FAULT_FNS!r}:
    globals()[_n] = _vf_wrap(_n, globals()[_n])
"""
    d = Path(tempfile.mkdtemp(prefix="vf.", dir=work))
    p = d / "p032_gb10_repl_embed.py"
    p.write_text(Path(module_path).read_text() + extra)
    return p


def v1d_faults(vroot, module_path, C, T, quick=False):
    ip = instrumented(module_path, Path(WORK["dir"]))
    B = Boot(vroot, ip, C, T)
    V, H = 512, 16
    bits = rand_table(V, H, 11, zeros=True)
    same = lambda r: bits  # noqa: E731
    sc = B.run(V, H, same, chunk=128)
    check("V1d instrumented module, no fault: ACTIVE on 8 ranks (the injector itself is inert)",
          active_ok(sc, same), f"{sc.errs[:1]} {sc.modes}")
    counts = sc.rs[3]["boot_counts"] if len(sc.rs) == W else {}
    missing = [n for n in FAULT_FNS if not counts.get(n)]
    check("V1d every wrapped helper runs during the boot (so every injection below really fires)", not missing,
          str(missing))
    plan = []
    for n in FAULT_FNS:
        c = counts.get(n, 0)
        for k in sorted({1, max(1, c // 2), c}) if c else []:
            for rank in ((3,) if quick else (0, 6)):
                plan.append((n, k, rank))
    if quick:
        plan = [p for p in plan if p[1] == 1][:6]
    bad = []
    for n, k, rank in plan:
        sc = B.run(V, H, same, chunk=128,
                   before=lambda r, ns, lay, w, n=n, k=k, rank=rank: ns.mod._VF.update(fn=n, rank=rank, k=k))
        if not (inactive_ok(sc) and any("verifier fault" in m for m in sc.ina)):
            bad.append((n, k, rank, sc.modes, sc.errs[:1], sc.hang, sc.ina[:1]))
    check(f"V1d {len(plan)} single faults (each boot helper x first / middle / last boot-time call x ranks): no hang, "
          "no escaping exception, INACTIVE on all 8 ranks with the fault in the reason, layer == image, forward == "
          "image", not bad and len(plan) >= 6, str(bad[:2]))
    # a corrupted CUDA-graph replay on one rank (testlib flag, read per rank thread)
    def corrupt(r, ns, lay, w):
        if r == 1:
            T.INJ.corrupt_replay = True

    sc = B.run(V, H, same, chunk=128, before=corrupt)
    for k in list(vars(T.INJ)):
        delattr(T.INJ, k)
    check("V1e a CUDA-graph replay that corrupts one element on rank 1 -> INACTIVE on all 8 ranks, layer == image",
          inactive_ok(sc) and all("CUDA-graph" in m for m in sc.ina), f"{sc.errs[:1]} {sc.modes} {sc.ina[:1]}")


# =====================================================================================================================
# V2 the REAL model path
# =====================================================================================================================
def v2_model(vroot, module_path, C, T, model_text, quick=False):
    print("== V2 REAL patched model.py (v0.32.0 original + 032) + load_weights + process_weights_after_loading + MTP + eagle")
    V, H = 512, 16
    rng = np.random.default_rng(21)
    ck = np.asarray((rng.standard_normal((V, H)) * 0.02).astype(np.float32).view(np.uint32) >> 16, np.uint16)
    ck[9, 0] = 0x8000
    ck[V - 1, :] = 0x8000
    assert len({ck[i].tobytes() for i in range(V)}) == V
    lm = np.asarray((rng.standard_normal((V, H)) * 0.02).astype(np.float32).view(np.uint32) >> 16, np.uint16)
    ids = [0, 9, 63, 64, V - 1, V // 2, 300, -1, V, -(1 << 31)]

    def scenario(env, mpath, force_fail=False):
        ns = C.build_ns(vroot, model_text, mpath, env=env)
        tm = ns.torch
        cfg = C.mk_cfg(V, H)
        vcfg = C.mk_vcfg(cfg)
        rv = T.Rendezvous(W, timeout=5.0)

        def body(r):
            w = C.setup_rank(rv, r, H, vcfg)
            if force_fail and r == 2:
                ns.mod._VF.update(fn="fingerprint", rank=2, k=1)
            with tm.device("cuda"):
                target = ns.model.Glm5NextForCausalLM(vllm_config=vcfg, prefix="")
            loaded = target.model.load_weights(iter([("embed_tokens.weight", C.cpu_tensor(ns, ck))]))
            target.lm_head.weight.weight_loader(target.lm_head.weight, C.cpu_tensor(ns, lm))
            lm_id = id(target.lm_head.weight)
            ns.lutil.process_weights_after_loading(target, vcfg.model_config, tm.device("cuda"))
            emb = target.model.embed_tokens
            t_ids = tm.tensor(ids, dtype=tm.int32, device="cuda:0")
            pos = tm.arange(len(ids), dtype=tm.int64, device="cuda:0")
            n0 = len(w.ar_calls)
            hs = target.model(t_ids, pos, None)
            fwd_ars = len(w.ar_calls) - n0
            P = V // W
            o = {"cls": type(emb).__name__, "mode": C.mode_of(ns, emb), "loaded": "embed_tokens.weight" in loaded,
                 "hs": np.array(hs._a, copy=True), "fwd_ars": fwd_ars, "wshape": tuple(emb.weight.shape),
                 "w": np.array(emb.weight._a, copy=True), "P": P,
                 "lm": (type(target.lm_head).__name__, tuple(target.lm_head.weight.shape),
                        bool(np.array_equal(target.lm_head.weight._a, lm[r * P:(r + 1) * P])),
                        id(target.lm_head.weight) == lm_id, type(target.lm_head.quant_method).__name__)}
            with tm.device("cuda"):
                draft = ns.mtp.Glm5NextMTP(vllm_config=vcfg, prefix="")
            C.TLS.draft = draft
            d = ns.eagle.load_eagle_model(target, vcfg)
            o["draft_shares"] = d.model.embed_tokens is emb
            o["head_shared"] = d.model._mtp_layers[0].shared_head.head is target.lm_head
            C.TLS.eh_inputs = []
            prev = tm.zeros(len(ids), H, dtype=tm.bfloat16, device="cuda:0")
            n0 = len(w.ar_calls)
            d(t_ids, pos, prev)
            o["draft_ars"] = len(w.ar_calls) - n0
            o["draft_emb"] = np.array(C.TLS.eh_inputs[0]._a, copy=True) if len(C.TLS.eh_inputs) == 1 else None
            return o
        out = C.run8(body)
        return NS(rs=[x for ok, x in out if ok], errs=[repr(x) for ok, x in out if not ok],
                  act=T.LOGS.lines(contains="replicated input embedding ACTIVE ("),
                  ina=T.LOGS.lines(contains="replicated input embedding INACTIVE ("),
                  any032=T.LOGS.lines(contains="[patch-032]"))
    ref = own_ref(ck, ids)
    canon = ck.copy()
    canon[canon == 0x8000] = 0
    sc = scenario({"VLLM_GB10_REPL_EMBED": "1"}, module_path)
    check("V2 gate 1: ACTIVE on 8 ranks, the REAL load_weights put the FULL checkpoint in (table == canon(ckpt)), "
          "REAL forward == own reference bitwise, 0 all-reduces",
          not sc.errs and len(sc.rs) == W and len(sc.act) == W and all(
              x["mode"] == "replicated" and x["loaded"] and x["wshape"] == (V, H) and np.array_equal(x["w"], canon)
              and np.array_equal(x["hs"], ref) and x["fwd_ars"] == 0 for x in sc.rs), f"{sc.errs[:1]}")
    check("V2 gate 1: lm_head untouched (ParallelLMHead [V/8, H], this rank's lm rows, same Parameter object, "
          "UnquantizedEmbeddingMethod) and shared into the draft's shared_head.head",
          len(sc.rs) == W and all(x["lm"] == ("ParallelLMHead", (V // W, H), True, True, "UnquantizedEmbeddingMethod")
                                  and x["head_shared"] for x in sc.rs), str(sc.rs[0]["lm"] if sc.rs else None))
    check("V2 gate 1: the draft's embed_tokens IS the target's replicated module; the REAL draft forward embeds with it "
          "(== own reference) and issues 0 all-reduces",
          len(sc.rs) == W and all(x["draft_shares"] and x["draft_ars"] == 0 and x["draft_emb"] is not None
                                  and np.array_equal(x["draft_emb"], ref) for x in sc.rs))
    ip = instrumented(module_path, Path(WORK["dir"]))
    sc = scenario({"VLLM_GB10_REPL_EMBED": "1"}, ip, force_fail=True)
    check("V2 forced INACTIVE (a preflight fault on rank 2): every rank's weight == ckpt rows [rV/8, (r+1)V/8) "
          "(no shard-offset bug, -0.0 bits kept), forward == own reference with 1 all-reduce, draft shares it",
          not sc.errs and len(sc.rs) == W and len(sc.ina) == W and all(
              x["mode"] == "image" and x["wshape"] == (V // W, H)
              and np.array_equal(x["w"], ck[r * (V // W):(r + 1) * (V // W)]) and np.array_equal(x["hs"], ref)
              and x["fwd_ars"] == 1 and x["draft_shares"] and x["draft_ars"] == 1 for r, x in enumerate(sc.rs)),
          f"{sc.errs[:1]}")
    for env, what in (({}, "unset"), ({"VLLM_GB10_REPL_EMBED": "0"}, "'0'"), ({"VLLM_GB10_REPL_EMBED": "true"},
                                                                              "'true'")):
        sc = scenario(env, module_path)
        check(f"V2 gate {what}: image VocabParallelEmbedding [V/8, H], forward == own reference with its all-reduce, "
              "no [patch-032] log line at all",
              not sc.errs and len(sc.rs) == W and not sc.any032 and all(
                  x["cls"] == "VocabParallelEmbedding" and x["wshape"] == (V // W, H) and np.array_equal(x["hs"], ref)
                  and x["fwd_ars"] == 1 for x in sc.rs), f"{sc.errs[:1]} {sc.any032[:1]}")
        if quick:
            break


# =====================================================================================================================
# V3 the .sh
# =====================================================================================================================
DEPS = ("models/glm5next/common/model.py", "models/glm5next/common/mtp.py",
        "model_executor/layers/vocab_parallel_embedding.py", "v1/worker/gpu/spec_decode/eagle/utils.py",
        "model_executor/model_loader/utils.py", "model_executor/model_loader/base_loader.py",
        "model_executor/model_loader/weight_tying.py")


def mkroot(work, vroot, tag, model_from=None):
    r = Path(tempfile.mkdtemp(prefix=f"{tag}.", dir=work))
    for rel in DEPS:
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vroot / rel, r / rel)
    if model_from is not None:
        shutil.copy2(model_from, r / DEPS[0])
    return r


def run_sh(script, root, env, log=None):
    e = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "PYTHONDONTWRITEBYTECODE": "1",
         "VLLM_ROOT": str(root), **env}
    p = subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True)
    if log is not None:
        log.extend((p.stdout + p.stderr).splitlines())
    return p.returncode, p.stdout + p.stderr


def tree_snap(root):
    d = {}
    for dp, _, fn in os.walk(root):
        for f in fn:
            st = os.lstat(os.path.join(dp, f))
            d[os.path.relpath(os.path.join(dp, f), root)] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return d


def base_text(vroot, work):
    """-> (text, path): the v0.32.0 model.py this suite patches (the original; V3 also exercises the 025 base)."""
    r = mkroot(work, vroot, "base")
    p = r / DEPS[0]
    return (p.read_text() if sha_file(p) == pin("MODEL_ORIG") else None), p


def p025_text(vroot, work):
    """-> (text, path) of the 025 base, or (None, None) while files/p025_apply.py is absent / refuses."""
    if not P025_APPLIER.exists():
        return None, None
    r = mkroot(work, vroot, "base025")
    p = r / DEPS[0]
    rc = subprocess.run([sys.executable, "-I", "-B", str(P025_APPLIER), "apply", str(p)], capture_output=True)
    return (p.read_text() if rc.returncode == 0 else None), p


def v3_sh(vroot, work, sh=SH, full_tree=None, lines=None, quick=False):
    print("== V3 the .sh (gate values, the 029 conflict vs the REAL 029 parse, inert = no file touched)")
    lines = lines if lines is not None else []
    for val, want in (("1", "apply"), ("", "skip"), ("0", "skip"), (None, "skip"), ("01", "refuse"), ("1\n", "refuse"),
                      ("on", "refuse"), ("TRUE", "refuse"), ("-1", "refuse"), ("1 ", "refuse"), ("I", "refuse")):
        r = mkroot(work, vroot, "g")
        before = tree_snap(r)
        env = {} if val is None else {"VLLM_GB10_REPL_EMBED": val}
        rc, out = run_sh(sh, r, env, lines)
        after = tree_snap(r)
        if want == "apply":
            ok = rc == 0 and sha_file(r / DEPS[0]) == pin("MODEL_ORIG_P032") and "[OK] patch 032 in place" in out
        elif want == "skip":
            ok = rc == 0 and before == after and "[OK] patch 032 skipped" in out
        else:
            ok = rc != 0 and before == after and "must be unset/0 (inert) or 1" in out
        check(f"V3 VLLM_GB10_REPL_EMBED={val!r}: {want}" + ("; no file touched (size / mtime / inode)"
                                                              if want != "apply" else ""), ok, "" if ok else out[-200:])
    # 029: run the REAL 029 .sh first (full tree), then 032 - every case 029 accepts with eh active must be refused
    if full_tree is not None:
        cases = [("1", None), ("1", "head"), ("1", "eh"), ("1", "head,eh"), ("1", "HEAD"), ("1", "Eh"), ("1", ""),
                 ("1", " "), ("1", "head, eh"), ("1", "head,head"), ("1", "eh,eh"), (" 1 ", "eh"), ("0", "eh"),
                 (None, "eh"), ("", "eh,head")]
        if quick:
            cases = cases[:4]
        bad, n_ref, n_acc = [], 0, 0
        for dc, parts in cases:
            fr = Path(tempfile.mkdtemp(prefix="v3dc.", dir=work))
            subprocess.run(["cp", "-cR", str(full_tree), str(fr / "vllm")], check=True)
            env = {}
            if dc is not None:
                env["VLLM_GB10_DRAFT_COMPRESS"] = dc
            if parts is not None:
                env["VLLM_GB10_DRAFT_COMPRESS_PARTS"] = parts
            rc29, out29 = run_sh(PDIR / "029-gb10-draft-compress.sh", fr / "vllm", env)
            m29 = re.search(r"\[OK\] patch 029 in place \((?:rev \d+; )?VLLM_GB10_DRAFT_COMPRESS=1, parts=([a-z,]+)", out29)
            eh_on = rc29 == 0 and m29 is not None and "eh" in m29.group(1).split(",")
            rc32, out32 = run_sh(sh, fr / "vllm", {**env, "VLLM_GB10_REPL_EMBED": "1"}, lines)
            refused = rc32 != 0 and "set VLLM_GB10_DRAFT_COMPRESS_PARTS=head" in out32
            if rc29 != 0:
                continue                          # the entrypoint aborts at 029 already: never reaches 032
            if eh_on != refused or (not refused and rc32 != 0):
                bad.append((dc, parts, rc29, eh_on, rc32, out32[-160:]))
            n_ref += refused
            n_acc += not refused and rc32 == 0
        check(f"V3 029 x 032 ({len(cases)} env cases, REAL 029 .sh then 032; {n_ref} refused, {n_acc} accepted): 032 "
              "refuses exactly the cases where 029 is in place with its eh part on", not bad and n_ref >= 1
              and n_acc >= 1, str(bad[:2]))
    # inert on a full-tree clone of a previous-base-like state: unset / 0 / '' touch nothing
    if full_tree is not None:
        for val in (None, "0", ""):
            fr = Path(tempfile.mkdtemp(prefix="v3inert.", dir=work))
            subprocess.run(["cp", "-cR", str(full_tree), str(fr / "vllm")], check=True)
            before = tree_snap(fr / "vllm")
            rc, out = run_sh(sh, fr / "vllm", {} if val is None else {"VLLM_GB10_REPL_EMBED": val}, lines)
            ok = rc == 0 and tree_snap(fr / "vllm") == before
            check(f"V3 full tree, VLLM_GB10_REPL_EMBED={val!r}: rc 0, not one of {len(before)} files touched", ok,
                  "" if ok else out[-160:])
            if quick:
                break
    # apply on each base (the original; the 025 result when 025's applier exists), idempotent re-run, reused-container
    # note, all lines captured
    for tag, (tb, pb), b in (("original", base_text(vroot, work), "MODEL_ORIG"),
                             ("025 result", p025_text(vroot, work), "MODEL_P025")):
        if tb is None or pin(b + "_P032") is None:
            skip(f"V3 {tag} base + gate 1", "base unavailable or its pin pair is a placeholder")
            continue
        r = mkroot(work, vroot, "prevbase", model_from=pb)
        rc, out = run_sh(sh, r, {"VLLM_GB10_REPL_EMBED": "1"}, lines)
        check(f"V3 {tag} base + gate 1: model.py == {b}_P032, module installed, exactly one '[OK] patch 032 in place' "
              "line", rc == 0 and sha_file(r / DEPS[0]) == pin(b + "_P032")
              and sha_file(r / "_gb10_repl_embed.py") == pin("MOD_SHA") and out.count("[OK] patch 032 in place") == 1,
              "" if rc == 0 else out[-200:])
        rc2, out2 = run_sh(sh, r, {"VLLM_GB10_REPL_EMBED": "1"}, lines)
        rc3, out3 = run_sh(sh, r, {}, lines)
        check(f"V3 {tag} re-run idempotent; gate unset on a patched model.py: note + skip, untouched",
              rc2 == 0 and rc3 == 0 and sha_file(r / DEPS[0]) == pin(b + "_P032") and "reused container" in out3)
    return lines


# =====================================================================================================================
# V4 log hygiene with the REAL grep
# =====================================================================================================================
def grep(pattern, lines, ere=True):
    p = subprocess.run(["grep", "-E" if ere else "-e", pattern] if ere else ["grep", "--", pattern],
                       input="\n".join(lines) + "\n", capture_output=True, text=True)
    return [ln for ln in p.stdout.splitlines() if ln]


def v4_logs(sh_lines, mod_lines_active, mod_lines_fail):
    print("== V4 log hygiene (REAL grep over captured lines)")
    if READINESS:
        rd = READINESS[0].read_text().splitlines()
        fails = [re.split(r"\s+", ln, maxsplit=1)[1] for ln in rd if ln.startswith("fail")]
        clean = [ln for ln in sh_lines if "FATAL" not in ln and "refus" not in ln.lower()] + mod_lines_active
        hit = {p_: grep(p_, clean) for p_ in fails}
        hit = {p_: h for p_, h in hit.items() if h}
        check(f"V4 no readiness.txt 'fail' rule ({len(fails)}, {READINESS[0].name} of {READINESS[0].parent.name}/) matches "
              f"any line of a clean 032 boot ({len(clean)} lines: .sh apply / skip / re-run, applier, built + ACTIVE)",
              not hit and len(fails) >= 5, str(hit)[:300])
    else:
        skip("V4 readiness.txt fail rules", "no readiness.txt fixture found")
    every = sh_lines + mod_lines_active + mod_lines_fail
    # other patches' count / must-be-0 patterns that 032 must never trigger: the [OK] / ACTIVE / FATAL shapes of the other
    # numbered patches and the generic gate texts
    others = [r"\[OK\] patch (0(0[0-9]|1[0-9]|2[0-9]|3[013-9]))( rev [0-9]+)? in place", r"NCCL WARN",
              r"\[patch-0(0[0-9]|1[0-9]|2[0-9]|3[013-9])\]", r"\[0(0[0-9]|1[0-9]|2[0-9]|3[013-9])\] FATAL"]
    hit = {p_: grep(p_, every) for p_ in others}
    hit = {p_: h for p_, h in hit.items() if h}
    check(f"V4 no gate pattern of the other numbered patches ({len(others)}) matches any line 032 can print",
          not hit, str(hit)[:300])
    cnt = r"\[patch-032\] replicated input embedding ACTIVE (rank"
    m0 = r"\[patch-032\] (replicated input embedding INACTIVE|boot self-test never ran|boot [a-z]+ exchange over the " \
         r"TP group failed)"
    c_hits = grep(cnt, every, ere=False)
    check("V4 032 count pattern (BRE, as documented) matches exactly the ACTIVE lines and nothing else",
          c_hits == [ln for ln in every if "replicated input embedding ACTIVE (rank" in ln] and len(c_hits) >= 1
          and all("INACTIVE" not in h for h in c_hits), str(c_hits[:2]))
    z_clean = grep(m0, sh_lines + mod_lines_active)
    z_fail = grep(m0, mod_lines_fail)
    check("V4 032 must-be-0 pattern: 0 hits on every .sh line + a clean boot; hits the INACTIVE / never-ran / "
          "exchange-failed lines", not z_clean and len(z_fail) >= 3 and all(
              any(k in h for h in z_fail) for k in ("INACTIVE", "never ran", "exchange over the TP group failed")),
          f"clean {z_clean[:1]} fail {len(z_fail)}")
    oks = [ln for ln in sh_lines if "[OK]" in ln]
    check("V4 no [OK] line quotes ACTIVE / INACTIVE / [patch-0NN] / 'replicated input embedding' / 'never ran'",
          oks and not [ln for ln in oks if re.search(r"ACTIVE|\[patch-0\d\d\]|replicated input embedding|never ran",
                                                     ln)], str(oks[:3]))


# =====================================================================================================================
# V6 entrypoint chain on the full tree
# =====================================================================================================================
def v6_chain(full_tree, work, sh_dir=PDIR, quick=False):
    print("== V6 the REAL entrypoint chain 001..031 (+032) on full-tree clones")
    KEY = {"model": DEPS[0], "vpe": DEPS[2], "eagle": DEPS[3], "mtp": DEPS[1], "lutil": DEPS[4], "bload": DEPS[5],
           "wtie": DEPS[6]}
    results = {r_ for r_ in (pin("MODEL_ORIG_P032"), pin("MODEL_P025_P032")) if r_}
    g32 = {"VLLM_GB10_REPL_EMBED": "1"}
    carried = {"VLLM_GB10_BLOCKING_SYNC": "1", "VLLM_GB10_ONESHOT_AR": "1", "VLLM_GB10_SKINNY_IDX": "1"}
    dchead = {"VLLM_GB10_DRAFT_COMPRESS": "1", "VLLM_GB10_DRAFT_COMPRESS_PARTS": "head"}
    mhc = {"VLLM_GB10_MHC_BM2_MIN_T": "128", "VLLM_GB10_MHC_FN_BF16": "1", "VLLM_GB10_MHC_PRS": "1"}
    cfgs = [("carried gates, 032 off", {**carried, **dchead}, "skip", "MODEL_ORIG"),
            ("carried gates + 032", {**carried, **dchead, **g32}, "apply", "MODEL_ORIG_P032"),
            ("032 alone", dict(g32), "apply", "MODEL_ORIG_P032"),
            ("carried + 029 head + mHC gates + 032", {**carried, **dchead, **mhc, **g32}, "apply", "MODEL_ORIG_P032"),
            ("carried + 025 (sp) + 032", {**carried, **dchead, **g32, "VLLM_GB10_MHC_SP": "1"}, "apply",
             "MODEL_P025_P032"),
            ("carried + 025 (sp) + mHC gates + 029 head + 032", {**carried, **dchead, **mhc, **g32,
                                                                 "VLLM_GB10_MHC_SP": "1"}, "apply", "MODEL_P025_P032"),
            ("029 default parts + 032", {**carried, "VLLM_GB10_DRAFT_COMPRESS": "1", **g32}, "refuse", None),
            ("032 gate '2'", {"VLLM_GB10_REPL_EMBED": "2"}, "refuse", None)]
    if quick:
        cfgs = cfgs[:2]
    for tag, env, want, wpin in cfgs:
        w = Path(tempfile.mkdtemp(prefix="v6.", dir=work))
        root = w / "vllm"
        subprocess.run(["cp", "-cR", str(full_tree), str(root)], check=True)
        pd = w / "patches"
        subprocess.run(["cp", "-R", str(sh_dir), str(pd)], check=True)
        writers, before, abort = {}, tree_snap(root), None
        for s in sorted(pd.glob("[0-9][0-9][0-9]-*.sh")):
            n = int(s.name[:3])
            if n > 32:
                continue
            e = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "PYTHONDONTWRITEBYTECODE": "1",
                 "VLLM_ROOT": str(root), **env}
            rc = subprocess.run(["bash", str(s)], env=e, capture_output=True).returncode
            after = tree_snap(root)
            for k in set(before) | set(after):
                if before.get(k) != after.get(k):
                    writers.setdefault(k, []).append(s.name[:3])
            before = after
            if rc != 0:
                abort = s.name[:3]
                break
        fin = {k: sha_file(root / v) for k, v in KEY.items()}
        mw = writers.get(DEPS[0], [])
        ro = all(fin[k] == pin(p_) for k, p_ in (("vpe", "VPE_ORIG"), ("eagle", "EAGLE_ORIG"), ("lutil", "LUTIL_ORIG"),
                                                 ("bload", "BLOAD_ORIG"), ("wtie", "WTIE_ORIG")))
        mtp_ok = fin["mtp"] in {x for x in (pin("MTP_ORIG"), pin("MTP_P029")) if x} and set(writers.get(DEPS[1], [])) <= {
            "029"}
        if want == "apply":
            if wpin == "MODEL_P025_P032" and not pin(wpin):
                skip(f"V6 {tag}", "025 pins are placeholders")
                continue
            ok = abort is None and fin["model"] == pin(wpin) and fin["model"] in results
        elif want == "skip":
            ok = abort is None and fin["model"] == pin(wpin) and not (root / "_gb10_repl_embed.py").exists()
        else:
            ok = abort in ("029", "032") and fin["model"] == pin("MODEL_ORIG") and not (root / "_gb10_repl_embed.py").exists()
        ok = ok and ro and mtp_ok and set(mw) <= {"025", "032"}
        check(f"V6 {tag}: {want}" + (f" -> {wpin}" if wpin else "") + "; read-only files untouched, model.py writers "
              f"{sorted(set(mw))}", ok, f"abort {abort} model {fin['model'][:16]}")


# =====================================================================================================================
# mutants
# =====================================================================================================================
MOD_MUTANTS = [
    ("bitwise compare replaced by a float compare (-0.0 == +0.0)",
     "            mm[k:k + 1].copy_((out.view(torch.int16) != ref.view(torch.int16)).sum().view(1))",
     "            mm[k:k + 1].copy_((out != ref).sum().view(1))"),
    ("exhaustive check skips the last chunk",
     '    plan = [("all", c0, min(V, c0 + CHUNK_IDS)) for c0 in range(0, V, CHUNK_IDS)]',
     '    plan = [("all", c0, min(V, c0 + CHUNK_IDS)) for c0 in range(0, V, CHUNK_IDS)][:-1]'),
    ("no out-of-range probe ids", "    ids += oob_ids(V)\n", "    pass\n"),
    ("fallback keeps the full-vocab shard indices", "    layer.shard_indices = si\n", "    pass\n"),
    ("pending mode takes the replicated path", "        if self._p032_mode == MODE_REPLICATED:\n",
     "        if self._p032_mode != MODE_VOCAB_PARALLEL:\n"),
    ("tied embeddings not refused", '    if getattr(config, "tie_word_embeddings", False):\n', "    if False:\n"),
    ("reference from the canonicalised table",
     "                    contrib = _lookup(_image_view(orig, si_tp, world.size, H, reduce=False), ids)",
     "                    contrib = _lookup(_image_view(layer.weight.narrow(0, start, rows), si_tp, world.size, H, "
     "reduce=False), ids)"),
    ("replicated view claims the masked branch while the reference runs the fused one",
     "    return bool(size > 1 and current_platform.is_cuda())\n", "    return False\n"),
    ("fallback forgets the per-partition output sizes",
     "    layer.output_size_per_partition = layer.num_embeddings_per_partition\n", "    pass\n"),
    ("verdict ignores the graph result",
     "                    fails.append(f\"CUDA-graph replay differs from the image results in {gbad} elements\")\n",
     "                    ok = True\n"),
    ("a failure after the exhaustive part skips the remaining collectives",
     "            ref = world.all_reduce(contrib)       # raises only if the TP group is broken -> propagates\n"
     "            if not ok:\n                continue\n",
     "            if not ok:\n                break\n            ref = world.all_reduce(contrib)\n"),
    ("fallback frees nothing (keeps the replicated weight)", "    layer.weight = w\n", "    pass\n"),
]
SH_MUTANTS = [
    ("029 parts check case-sensitive", "| tr 'A-Z' 'a-z' | tr -d '[:space:]')\"\n  [ -n \"$parts\" ]",
     "| tr -d '[:space:]')\"\n  [ -n \"$parts\" ]"),
    ("gate accepts anything starting with 1", "  1) ;;\n", "  1*) ;;\n"),
    ("unknown model.py base applied anyway", "    fail \"model.py sha256 $m_now is none of the two bases",
     "    mode=apply; expect=\"$m_now\"; base=unknown; : \"model.py sha256 $m_now is none of the two bases"),
    ("skip branch touches model.py", "    echo \"[OK] patch 032 skipped (VLLM_GB10_REPL_EMBED unset); files untouched\"\n",
     "    touch \"$MODEL\"\n    echo \"[OK] patch 032 skipped (VLLM_GB10_REPL_EMBED unset); files untouched\"\n"),
]


def run_mutants(vroot, work, full_tree, C, T, model_text):
    print("== M mutants (each must make a quick run fail)")
    src = MODULE.read_text()
    caught = 0
    for name, old, new in MOD_MUTANTS:
        if src.count(old) != 1:
            check(f"M module mutant '{name}': anchor found {src.count(old)}x", False)
            continue
        d = Path(tempfile.mkdtemp(prefix="mm.", dir=work))
        mp = d / "p032_gb10_repl_embed.py"
        mp.write_text(src.replace(old, new, 1))
        snap = (RES["pass"], RES["fail"], list(RES["failed"]))
        QUIET["on"] = True
        raised = None
        try:
            v1_module(vroot, mp, C, T, quick=True)
            if RES["fail"] == snap[1]:
                v2_model(vroot, mp, C, T, model_text, quick=True)
            if RES["fail"] == snap[1]:
                v1_module(vroot, mp, C, T, quick=False)
            if RES["fail"] == snap[1]:
                mutant_extra(vroot, mp, C, T)
        except Exception as e:  # noqa: BLE001
            raised = e
        finally:
            QUIET["on"] = False
        got = RES["fail"] - snap[1] + (raised is not None)
        RES["pass"], RES["fail"], RES["failed"] = snap
        caught += got > 0
        check(f"M module mutant caught: {name}", got > 0, f"{got} failing checks" + (f", raised {raised!r:.90}"
                                                                                    if raised else ""))
    ssrc = SH.read_text()
    for name, old, new in SH_MUTANTS:
        if ssrc.count(old) != 1:
            check(f"M .sh mutant '{name}': anchor found {ssrc.count(old)}x", False)
            continue
        d = Path(tempfile.mkdtemp(prefix="ms.", dir=work))
        pd = d / "patches"
        shutil.copytree(PDIR, pd, ignore=shutil.ignore_patterns("__pycache__", "_reverted"))
        (pd / SH.name).write_text(ssrc.replace(old, new, 1))
        snap = (RES["pass"], RES["fail"], list(RES["failed"]))
        QUIET["on"] = True
        try:
            v3_sh(vroot, work, sh=pd / SH.name, full_tree=full_tree, quick=False)
            sh_unknown_base(vroot, work, pd / SH.name)
        finally:
            QUIET["on"] = False
        got = RES["fail"] - snap[1]
        RES["pass"], RES["fail"], RES["failed"] = snap
        caught += got > 0
        check(f"M .sh mutant caught: {name}", got > 0, f"{got} failing checks")
    check(f"M all {len(MOD_MUTANTS) + len(SH_MUTANTS)} mutants caught", caught == len(MOD_MUTANTS) + len(SH_MUTANTS))


def sh_unknown_base(vroot, work, sh=SH):
    r = mkroot(work, vroot, "unk")
    p = r / DEPS[0]
    p.write_text(p.read_text() + "\n# a local edit\n")
    before = tree_snap(r)
    rc, out = run_sh(sh, r, {"VLLM_GB10_REPL_EMBED": "1"})
    ok = rc != 0 and tree_snap(r) == before and "none of the two bases" in out
    check("V3 an unknown model.py (not one of the two bases) is refused, nothing written", ok, "" if ok else out[-160:])


def mutant_extra(vroot, module_path, C, T):
    """construction envelope + pending mode, own checks (quick)."""
    ns = C.build_ns(vroot, None, module_path, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
    rv = T.Rendezvous(W, timeout=5.0)
    V, H = 512, 16
    bits = rand_table(V, H, 3, zeros=True)
    ids = list(range(V)) + oob(V)

    def body(r):
        C.setup_rank(rv, r, H, C.mk_vcfg(C.mk_cfg(V, H, tie=True)))
        tied = ns.mod.embedding_class(C.mk_cfg(V, H, tie=True)).__name__
        C.setup_rank(rv, r, H, C.mk_vcfg(C.mk_cfg(V, H)))
        with ns.torch.device("cuda"):
            lay = ns.mod.embedding_class(C.mk_cfg(V, H))(V, H, prefix="x")
        lay.weight.weight_loader(lay.weight, C.cpu_tensor(ns, bits))
        got = C.local_lookup(lay, ids, ns)            # never booted: pending mode
        return tied, bool(np.array_equal(got, own_ref(bits, ids)))
    o = C.run8(body)
    check("Mx tie_word_embeddings=True -> the image class", all(ok and x[0] == "VocabParallelEmbedding" for ok, x in o))
    check("Mx pending mode (boot never ran) == own reference (the image semantics: -0.0 canonicalised by the all-reduce)",
          all(ok and x[1] for ok, x in o))


# =====================================================================================================================
WORK = {"dir": None, "scratch": None}


def capture_module_lines(vroot, C, T):
    """real module log lines: a clean ACTIVE boot (+ built line), an INACTIVE boot, the never-ran warning, the reload
    line and the exchange-failure error."""
    B = Boot(vroot, MODULE, C, T)
    V, H = 512, 16
    bits = rand_table(V, H, 5, zeros=True)
    sc = B.run(V, H, lambda r: bits)
    active = list(sc.logs)
    sc2 = B.run(V, H, lambda r: bits, arith=adv_sign_preserving)
    fail = [m for m in sc2.logs if "INACTIVE" in m][:2]
    ns = C.build_ns(vroot, None, MODULE, env={"VLLM_GB10_REPL_EMBED": "1"}, with_model=False)
    rv = T.Rendezvous(1, timeout=2.0)
    T.LOGS.clear()

    def one(r):
        w = T.World(rv, 0, 1, oneshot_max=8 * H)
        T.set_current_world(w)
        C.TLS.vcfg = C.mk_vcfg(C.mk_cfg(V, H))
        with ns.torch.device("cuda"):
            lay = ns.mod.ReplicatedEmbedding(V, H, prefix="x")
        lay.weight.weight_loader(lay.weight, C.cpu_tensor(ns, bits))
        lay(ns.torch.tensor([1, 2], dtype=ns.torch.int32, device="cuda:0"))   # pending -> never-ran warning

        class Broken:
            rank = 0

            def all_gather_object(self, o):
                raise RuntimeError("gloo: connection reset")
        try:
            ns.mod._exchange(Broken(), {}, "preflight")
        except RuntimeError:
            pass
        lay._p032_booted, lay._p032_mode = True, "replicated"
        ns.mod.boot(lay)                                                        # reload line
        return True
    T.run_ranks(1, one)
    fail += [m for (_, _, m) in T.LOGS.records]
    return active, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--full-tree", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--mutants", action="store_true")
    a = ap.parse_args()
    vroot = Path(a.vllm_root)
    full = Path(a.full_tree) if a.full_tree else None
    base = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="p032-verify-"))
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run.", dir=base))
    WORK["dir"] = str(work)
    WORK["scratch"] = os.environ.get("P032_GATES_DIR")
    for k, p in (("model", "MODEL_ORIG"), ("vpe", "VPE_ORIG"), ("eagle", "EAGLE_ORIG"), ("mtp", "MTP_ORIG"),
                 ("lutil", "LUTIL_ORIG"), ("bload", "BLOAD_ORIG"), ("wtie", "WTIE_ORIG")):
        rel = {"model": DEPS[0], "mtp": DEPS[1], "vpe": DEPS[2], "eagle": DEPS[3], "lutil": DEPS[4],
               "bload": DEPS[5], "wtie": DEPS[6]}[k]
        check(f"pin: --vllm-root {rel} == {p}", sha_file(vroot / rel) == pin(p))
    check("pin: APPLIER_SHA / MOD_SHA == the vendored files", sha_file(APPLIER) == pin("APPLIER_SHA")
          and sha_file(MODULE) == pin("MOD_SHA"))
    v0_algebra()
    C, T = scaffolding()
    C.WORK["dir"] = str(work)
    import importlib.util
    spec = importlib.util.spec_from_file_location("p032_apply_v", str(APPLIER))
    apm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(apm)
    t0, _ = base_text(vroot, work)
    model_text = apm.apply(t0) if t0 is not None else None
    check("V2 setup: the v0.32.0 model.py -> applier -> MODEL_ORIG_P032", model_text is not None
          and hashlib.sha256(model_text.encode()).hexdigest() == pin("MODEL_ORIG_P032"))
    v1_module(vroot, MODULE, C, T)
    v2_model(vroot, MODULE, C, T, model_text)
    sh_lines = v3_sh(vroot, work, full_tree=full)
    sh_unknown_base(vroot, work)
    p = subprocess.run([sys.executable, str(APPLIER), "check", str(vroot / DEPS[0])], capture_output=True, text=True)
    sh_lines += (p.stdout + p.stderr).splitlines()
    act, fail = capture_module_lines(vroot, C, T)
    v4_logs(sh_lines, act, fail)
    if full is not None:
        v6_chain(full, work)
    if a.mutants:
        run_mutants(vroot, work, full, C, T, model_text)
    print(f"== {RES['pass']} PASS, {RES['fail']} FAIL, {RES['skip']} SKIP" + (f": {RES['failed']}" if RES["fail"] else "")
          + (f"  (skipped: {RES['skipped']})" if RES["skip"] else ""))
    sys.exit(0 if RES["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
