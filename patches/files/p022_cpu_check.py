#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-022 gb10-mhc-prefill-rowsplit] CPU checks (v0.32.0 port; no GPU, no TileLang, no vLLM install; torch on CPU).

  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 files/p022_cpu_check.py --vllm-root <v0.32.0 tree>/vllm [--work DIR]
  (needs a Python with torch)

--vllm-root holds the v0.32.0 ORIGINAL files (sha-checked against the .sh pins). Sections:
  A  pins + applier on the real v0.32.0 source: tilelang.py / tilelang_kernels.py hash to the pins (022 == 019 == 011); the two
     tilelang.py bases (original, 019 result - built by the REAL 019 and 011 .sh scripts on scratch roots) -> p022_apply
     apply / revert / check / CLI == the pinned results; AST scope (the result minus the 022 nodes == the base); refusals;
     019 / 022 commute; the module-level gate block alone (unset / set + module missing -> ImportError / set + module).
  B  gb10_mhc_prs.py logic with fake row-local kernels and an in-process world of 8 ranks (unchanged from rev 1 except that
     the hook now receives a JIT-class stand-in and asks its dispatch()): MIN_T parsing, the row plan for T = 1..20,000,
     split + gather + unpack == unsplit for even / uneven T (the padding of the tail shard poisoned), the exact kernel
     arguments and views, negative controls (a row-offset-, a num_tokens- and a CTA-pair-dependent kernel and a corrupted
     gather MUST be caught), routing (T < MIN_T, capture, unexpected layouts, a dispatch that does not pick block_m=2 ->
     upstream path), the boot (envelope, fingerprint, collective self-test, ACTIVE once) and its refusals.
  C  the REAL patched tilelang.py over the REAL JIT classes (pmhc_testlib: the real jit_warmup.py / helper /
     tilelang_kernels.py exec'd over fake TileLang with row-local CPU kernels): mhc_pre_tilelang (layer 0) and
     mhc_fused_post_pre_tilelang (num_tokens > 32) with the gate on are bitwise equal to the unpatched files and to the
     gate-off files for T = 1024 / 1027 / 2048 / 3840, on every tilelang.py x tilelang_kernels.py base; the block_m=2 JIT
     function ran on 8 shards (unpatched: once on all T rows); T below MIN_T, num_tokens <= 32, a dispatch that is not
     block_m=2 (011 off, 256 <= T < 1024) and the DeepGEMM branch never split; gate set + module missing -> ImportError.
Exit 0 = ALL PASS.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PDIR = HERE.parent
SH22 = PDIR / "022-gb10-mhc-prefill-rowsplit.sh"
SH19 = PDIR / "019-gb10-mhc-fn-bf16.sh"
SH11 = PDIR / "011-mhc-prenorm-bm2-threshold.sh"
TL_REL = "model_executor/kernels/mhc/tilelang.py"
TK_REL = "model_executor/kernels/mhc/tilelang_kernels.py"
BASES = ("TL_ORIG", "TL_P019")
RESULT = {"TL_ORIG": "TL_ORIG_P022", "TL_P019": "TL_P019_P022"}

RESULTS = {"pass": 0, "fail": 0, "failed": []}


def check(name, ok, info=""):
    ok = bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    RESULTS["pass" if ok else "fail"] += 1
    if not ok:
        RESULTS["failed"].append(name)
    return ok


def raises(fn, exc=Exception, match=None):
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return match is None or re.search(match, str(e)) is not None
    return False


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(name, sh=SH22):
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", Path(sh).read_text(), re.M)
    if not m:
        raise SystemExit(f"no pin {name} in {sh}")
    return m.group(1)


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v




def tree_sum_last(v):
    import torch

    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        assert v.shape[-1] == 2 * h, "power-of-two K expected"
        v = v[..., :h] + v[..., h:]
    return v[..., 0]


def rowlocal_prenorm(x, fn):
    import torch

    n, K = x.shape
    o = torch.empty(n, fn.shape[0], dtype=torch.float32)
    s = torch.empty(n, dtype=torch.float32)
    for r0 in range(0, n, 32):
        xc = x[r0:r0 + 32].float()
        o[r0:r0 + 32] = tree_sum_last(xc.unsqueeze(1) * fn.unsqueeze(0))
        s[r0:r0 + 32] = tree_sum_last(xc * xc)
    return o, s


class KernelLog:
    def __init__(self):
        self.calls = []

    def clear(self):
        self.calls.clear()


def check_bm2_args(x, fn, out, sqrsum, hidden_size, hc_mult, n_out, n_thr, tile_n, block_m):
    """The host-side ABI checks of the generated TileLang launcher (host_kernel.cu of the served cache)."""
    n, K = x.shape
    assert x.dim() == 2 and K == hc_mult * hidden_size and x.stride(1) == 1 and (x.stride(0) == K or n == 1), x.stride()
    assert tuple(fn.shape) == (n_out, K) and fn.is_contiguous()
    assert out.dim() == 3 and out.shape[0] == 1 and out.shape[1] == n and out.shape[2] == n_out
    assert out.stride(2) == 1 and (out.stride(1) == n_out or n == 1)
    assert sqrsum.dim() == 2 and sqrsum.shape[0] == 1 and sqrsum.shape[1] == n and (sqrsum.stride(1) == 1 or n == 1)
    assert (n_thr, tile_n, block_m) == (512, 12, 2), (n_thr, tile_n, block_m)


def make_bm2(log, variant="rowlocal"):
    """Fake hc_prenorm_gemm_block_m_tilelang. variants: rowlocal (correct), offset (adds the local row index),
    ntok (adds num_tokens), pair (adds the local row index % 2 = the CTA slot i_m)."""
    import torch

    def k(x, fn, out, sqrsum, hidden_size, hc_mult, n_out, n_thr, tile_n, block_m):
        check_bm2_args(x, fn, out, sqrsum, hidden_size, hc_mult, n_out, n_thr, tile_n, block_m)
        log.calls.append(("bm2", int(x.shape[0]), x.data_ptr(), int(x.stride(0)),
                          (sqrsum.data_ptr() - out.data_ptr()) % 256))
        o, s = rowlocal_prenorm(x, fn)
        n = x.shape[0]
        if variant == "offset":
            o = o + torch.arange(n, dtype=torch.float32).unsqueeze(1)
        elif variant == "ntok":
            s = s + float(n)
        elif variant == "pair":
            o = o + (torch.arange(n) % 2).to(torch.float32).unsqueeze(1) * 0.5
        out[0].copy_(o)
        sqrsum[0].copy_(s)

    return k




# ---------------------------------------------------------------------------------------------------------------------
# a stand-in for HcPrenormGemmTileLangKernel (section B): dispatch() + kernel(key), the two members the hook uses
# ---------------------------------------------------------------------------------------------------------------------
class KObj:
    def __init__(self, fn, bm2_min_t=0):
        self.fn, self.bm2_min_t, self.dispatches = fn, bm2_min_t, []

    def dispatch(self, *, num_tokens, hc_hidden_size, n_thr=512, tile_n=12, n_splits=1, **fields):
        self.dispatches.append((num_tokens, hc_hidden_size, n_thr, tile_n, n_splits, tuple(sorted(fields))))
        bm = n_splits == 1 and num_tokens >= self.bm2_min_t
        return types.SimpleNamespace(use_block_m=bm, block_m=2 if bm else 1, n_splits=n_splits, n_thr=n_thr,
                                     tile_n=tile_n, **fields)

    def kernel(self, key):
        return self.fn


def prs(M, kernel, x, fn, out, sqrsum, hidden_size, hc_mult, *_unused):
    """M.prenorm_rowsplit with the old call shape (kernel function or a KObj)."""
    kobj = kernel if isinstance(kernel, KObj) else KObj(kernel)
    return M.prenorm_rowsplit(kobj, x, fn, out, sqrsum, hidden_size, hc_mult)


def load_prs(log):
    for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
        del sys.modules[k]
    pkg = types.ModuleType("vllm")
    pkg.__path__ = []
    sys.modules["vllm"] = pkg
    m = load_file_module("vllm._gb10_mhc_prs", HERE / "gb10_mhc_prs.py")
    pkg._gb10_mhc_prs = m
    return m


def rand_inputs(T, K, n_out=24, seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, K, generator=g) * 3).to(torch.bfloat16)
    fn = torch.randn(n_out, K, generator=g)
    return x, fn


def unsplit(kernel, x, fn, hidden, hc):
    import torch

    T = x.shape[0]
    o = torch.full((1, T, fn.shape[0]), float("nan"))
    s = torch.full((1, T), float("nan"))
    kernel(x, fn, o, s, hidden, hc, fn.shape[0], 512, 12, 2)
    return o, s


def section_b(work):
    import torch

    print("== B. gb10_mhc_prs.py logic (fake kernels, 8 ranks in-process)")
    log = KernelLog()
    M = load_prs(log)
    hidden, hc = 128, 4
    K = hidden * hc
    # B1 MIN_T parsing
    ok = (M.parse_min_t(None) == 1024 and M.parse_min_t("") == 1024 and M.parse_min_t(" 2048 ") == 2048
          and M.parse_min_t("256") == 256 and M.parse_min_t("1048576") == 1048576)
    for bad in ("abc", "-5", "255", "1048577", "1e3", "12.5"):
        ok &= raises(lambda: M.parse_min_t(bad), ValueError)
    check("B1 MIN_T: default 1024, range [256, 2**20], non-integers refused", ok)
    with env(VLLM_GB10_MHC_PRS="1", VLLM_GB10_MHC_PRS_MIN_T="100"):
        check("B1 gate set + invalid MIN_T -> the module refuses at import",
              raises(lambda: load_file_module("vllm._p022_badmin", HERE / "gb10_mhc_prs.py"), ValueError))
    sys.modules.pop("vllm._p022_badmin", None)
    # B2 plan
    bad_plans = []
    for T in range(1, 20001):
        p = M.plan_rows(T, 8)
        try:
            M.check_plan(p)
            if T >= 256:
                assert all(p.rows(r) > 0 for r in range(8)), "empty shard"
            if T % 16 == 0:
                assert p.starts == tuple(r * T // 8 for r in range(8)), "not [r*T/8, (r+1)*T/8)"
            assert p.S == 2 * -(-(-(-T // 2)) // 8)
        except AssertionError as e:
            bad_plans.append((T, str(e)[:80]))
    check("B2 plan: partition, even starts, only the tail short, no empty shard >= 256, == r*T/8 when T % 16 == 0 "
          "(T = 1..20000)", not bad_plans, str(bad_plans[:3]))
    p = M.plan_rows(1027, 8)
    check("B2 plan T=1027: S=130, 7 x 130 + 117", p.S == 130 and [p.rows(r) for r in range(8)] == [130] * 7 + [117])
    p = M.plan_rows(1024, 8)
    check("B2 plan T=1024: 8 x 128 (the 011 threshold row count - irrelevant: shards always run the bm2 kernel)",
          p.S == 128 and all(p.rows(r) == 128 for r in range(8)))
    # B3 split == unsplit, args / views, padding poisoned
    bm2 = make_bm2(log)
    W = M.EmulatedWorld(8)

    class PoisonPad(M.EmulatedWorld):
        """Every send buffer's unused tail (rank q's padding and empty ranks) is NaN before the gather."""
        def gather_shards(self, make_send):
            parts = []
            for r in range(self.size):
                s = make_send(r)
                pl = self._plan
                n = pl.rows(r)
                sq_off, width = M.layout(pl.S, 24)
                assert s.numel() == width
                s.view(-1)[n * 24: sq_off] = float("nan")  # out padding + the alignment gap
                s.view(-1)[sq_off + n:] = float("nan")  # sqrsum padding
                parts.append(s)
            return torch.cat(parts)

    bad = []
    for T in (256, 257, 1000, 1024, 1025, 1027, 1039, 1040, 1280, 2047, 2048, 3840, 3853, 4096, 8191, 8192):
        x, fn = rand_inputs(T, K, seed=T)
        ro, rs = unsplit(bm2, x, fn, hidden, hc)
        w = PoisonPad(8)
        w._plan = M.plan_rows(T, 8)
        log.clear()
        o = torch.full((1, T, 24), float("nan"))
        s = torch.full((1, T), float("nan"))
        M.split_into(bm2, x, fn, o, s, hidden, hc, 512, 12, w)
        calls = list(log.calls)
        pl = w._plan
        okv = (M.mismatches(o, ro) == 0 and M.mismatches(s, rs) == 0 and torch.isfinite(o).all()
               and len(calls) == sum(1 for r in range(8) if pl.rows(r))
               and all(c[1] == pl.rows(r) and c[2] == x.data_ptr() + pl.starts[r] * K * 2 and c[3] == K and c[4] == 0
                       and pl.starts[r] % 2 == 0
                       for r, c in zip([r for r in range(8) if pl.rows(r)], calls)))
        if not okv:
            bad.append(T)
    check("B3 split + gather + unpack == unsplit bitwise; shard r = rows [r*S, ...) as a stride-K view of x starting at "
          "an even row (production: K = 16,384 -> 64 KiB-aligned), sqrsum view 256 B-aligned in the send buffer; "
          "padding never read (16 T incl. uneven)", not bad, str(bad))
    check("B3 layout: sqrsum offset = S*24 rounded up to 64 floats",
          all(M.layout(S, 24) == (-(-S * 24 // 64) * 64, -(-S * 24 // 64) * 64 + S) and M.layout(S, 24)[0] % 64 == 0
              for S in (2, 128, 130, 160, 256, 480, 482, 1024)))
    # B4 negative controls
    T = 1027
    x, fn = rand_inputs(T, K, seed=7)
    caught = {}
    for variant in ("offset", "ntok"):
        kv = make_bm2(log, variant)
        ro, rs = unsplit(kv, x, fn, hidden, hc)
        o, s = torch.empty(1, T, 24), torch.empty(1, T)
        M.split_into(kv, x, fn, o, s, hidden, hc, 512, 12, W)
        caught[variant] = M.mismatches(o, ro) + M.mismatches(s, rs)
    kp = make_bm2(log, "pair")
    ro, rs = unsplit(kp, x, fn, hidden, hc)
    o, s = torch.empty(1, T, 24), torch.empty(1, T)
    M.split_into(kp, x, fn, o, s, hidden, hc, 512, 12, W)
    caught["pair_even_starts"] = M.mismatches(o, ro) + M.mismatches(s, rs)
    odd = M.Plan(T, 8, 130, (0, 129, 259, 389, 519, 649, 779, 909), (129, 259, 389, 519, 649, 779, 909, 1027))
    o, s = torch.empty(1, T, 24), torch.empty(1, T)

    # an odd-start plan exists only for this control (production plans always start shards at even rows)
    def split_odd():
        S, n_out = odd.S, 24
        parts = []
        for r in range(8):
            b, e = odd.starts[r], odd.ends[r]
            snd = torch.zeros(S * 25)
            kp(x[b:e], fn, snd[: (e - b) * n_out].view(1, e - b, n_out), snd[S * n_out: S * n_out + e - b].view(1, e - b),
               hidden, hc, n_out, 512, 12, 2)
            parts.append(snd)
        recv = torch.stack(parts)
        for r in range(8):
            b, e = odd.starts[r], odd.ends[r]
            o[0, b:e] = recv[r, : (e - b) * n_out].view(e - b, n_out)
            s[0, b:e] = recv[r, S * n_out: S * n_out + e - b]
    split_odd()
    caught["pair_odd_starts"] = M.mismatches(o, ro) + M.mismatches(s, rs)

    class Corrupt(M.EmulatedWorld):
        def gather_shards(self, make_send):
            g = super().gather_shards(make_send)
            g.view(torch.int32)[3] ^= 1
            return g
    ro, rs = unsplit(bm2, x, fn, hidden, hc)
    o, s = torch.empty(1, T, 24), torch.empty(1, T)
    M.split_into(bm2, x, fn, o, s, hidden, hc, 512, 12, Corrupt(8))
    caught["corrupt_gather"] = M.mismatches(o, ro) + M.mismatches(s, rs)
    check("B4 negative controls caught: row-offset- and num_tokens-dependent kernels, a corrupted gather",
          caught["offset"] > 0 and caught["ntok"] > 0 and caught["corrupt_gather"] == 1, str(caught))
    check("B4 a CTA-pair-dependent kernel stays exact with the even shard starts, breaks with odd ones",
          caught["pair_even_starts"] == 0 and caught["pair_odd_starts"] > 0, str(caught))
    # B5 unpack edge cases (q full shards, tail, T == W*S)
    bad = []
    for T in (8, 9, 15, 16, 17, 30, 31, 33, 64, 65, 255, 256, 258):
        x, fn = rand_inputs(T, K, seed=100 + T)
        ro, rs = unsplit(bm2, x, fn, hidden, hc)
        o, s = torch.full((1, T, 24), float("nan")), torch.full((1, T), float("nan"))
        M.split_into(bm2, x, fn, o, s, hidden, hc, 512, 12, W)
        if M.mismatches(o, ro) or M.mismatches(s, rs):
            bad.append(T)
    check("B5 unpack exact for small / degenerate T (empty trailing ranks, T == W*S, odd T)", not bad, str(bad))

    # B6/B7 routing
    M._reset_for_test(min_t=1024)
    M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
    log.clear()
    x, fn = rand_inputs(1023, K, seed=3)
    o, s = torch.empty(1, 1023, 24), torch.empty(1, 1023)
    r = prs(M, bm2, x, fn, o, s, hidden, hc, 512, 12)
    check("B6 T < MIN_T -> False, nothing called, no boot", r is False and not log.calls and not M.stats()["active"]
          and M.stats()["small_skips"] == 1)
    x, fn = rand_inputs(1024, K, seed=4)
    o, s = torch.empty(1, 1024, 24), torch.empty(1, 1024)
    real_cap = M._capturing
    M._capturing = lambda t: True
    try:
        r = prs(M, bm2, x, fn, o, s, hidden, hc, 512, 12)
    finally:
        M._capturing = real_cap
    check("B6 during a CUDA-graph capture -> False, no kernel, no collective, no boot",
          r is False and not log.calls and not M.stats()["active"] and M.stats()["capture_skips"] == 1)
    bads = {
        "x non-contiguous": (x.t().contiguous().t(), fn, o, s),
        "x fp32": (x.float(), fn, o, s),
        "fn bf16": (x, fn.to(torch.bfloat16), o, s),
        "out wrong shape": (x, fn, torch.empty(1, 1024, 23), s),
        "sqrsum non-contiguous": (x, fn, o, torch.empty(1, 2048)[:, ::2]),
        "fn wrong width": (x, fn[:, :256].contiguous(), o, s),
    }
    res = {}
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    M.logger.addHandler(h)
    M.logger.setLevel(logging.INFO)
    for k, (a, b, c, d) in bads.items():
        res[k] = prs(M, bm2, a, b, c, d, hidden, hc, 512, 12)
    res["again"] = prs(M, bm2, *bads["x fp32"], hidden, hc, 512, 12)
    check("B7 unexpected layouts -> False (image path), warned once per reason, no boot",
          not any(res.values()) and not log.calls and stream.getvalue().count("row split skipped") == len(bads)
          and M.stats()["ineligible"] == len(bads) + 1, str(res))
    # B8 boot + steady state
    log.clear()
    stream.truncate(0)
    stream.seek(0)
    ro, rs = unsplit(bm2, x, fn, hidden, hc)
    log.clear()
    o, s = torch.full((1, 1024, 24), float("nan")), torch.full((1, 1024), float("nan"))
    r = prs(M, bm2, x, fn, o, s, hidden, hc, 512, 12)
    st = M.stats()
    n_cases = 1 + 2 * len({1024, 1027, 3840, 3853})
    check("B8 first eligible call: boot (envelope + fingerprint + self-test) -> True, the live output exact",
          r is True and st["active"] and M.mismatches(o, ro) == 0 and M.mismatches(s, rs) == 0
          and st["report"]["cases"] == n_cases and st["report"]["ranks"] == 8, str(st["report"]))
    check("B8 boot ran every case as unsplit + 2 splits x 8 shards", len(log.calls) == n_cases * 17, str(len(log.calls)))
    check("B8 ACTIVE logged once with the self-test summary",
          stream.getvalue().count("ACTIVE") == 1 and "bit-exact on all 8 ranks" in stream.getvalue())
    log.clear()
    T = 3853
    x2, fn2 = rand_inputs(T, K, seed=5)
    ro, rs = unsplit(bm2, x2, fn2, hidden, hc)
    log.clear()
    o, s = torch.empty(1, T, 24), torch.empty(1, T)
    r = prs(M, bm2, x2, fn2, o, s, hidden, hc, 512, 12)
    check("B8 steady state: one split (8 shard kernels), exact, no second boot / ACTIVE",
          r is True and len(log.calls) == 8 and M.mismatches(o, ro) == 0 and M.mismatches(s, rs) == 0
          and stream.getvalue().count("ACTIVE") == 1 and M.stats()["calls"] == 2)
    # B9 boot refusals
    def boot_with(world_factory, kernel=bm2, cfg=None, T=1024):
        M._reset_for_test(min_t=1024)
        M._set_world_factory_for_test(world_factory)
        real = M._vllm_config_or_none
        if cfg is not None:
            M._vllm_config_or_none = lambda: cfg
        try:
            xx, ff = rand_inputs(T, K, seed=9)
            oo, ss = torch.empty(1, T, 24), torch.empty(1, T)
            prs(M, kernel, xx, ff, oo, ss, hidden, hc, 512, 12)
        finally:
            M._vllm_config_or_none = real
    NS = types.SimpleNamespace

    class FPMismatch(M.EmulatedWorld):
        def all_gather_object(self, obj):
            out = super().all_gather_object(obj)
            if isinstance(obj, str):
                out[3] = "0123456789abcdef"
            return out

    class CrossMismatch(M.EmulatedWorld):
        def all_gather_tensor(self, t):
            g = super().all_gather_tensor(t).clone()
            g.view(torch.int32)[5 * t.numel()] ^= 1
            return g

    class OtherRankFails(M.EmulatedWorld):
        def all_gather_object(self, obj):
            out = super().all_gather_object(obj)
            if isinstance(obj, dict):
                out = [dict(o) for o in out]
                out[6] = {"rank": 6, "ok": False, "cases": [{"case": "live T=1024", "ok": False, "mismatch": 1}]}
            return out

    refusals = {
        "TP=4": (lambda: M.EmulatedWorld(4), bm2, None, r"TP=8 only"),
        "PP=2": (lambda: M.EmulatedWorld(8, pp_size=2), bm2, None, "pipeline parallel"),
        "SP-MoE": (lambda: M.EmulatedWorld(8), bm2, NS(parallel_config=NS(use_sequence_parallel_moe=True)),
                   "sequence-parallel MoE"),
        "PCP=2": (lambda: M.EmulatedWorld(8), bm2,
                  NS(parallel_config=NS(use_sequence_parallel_moe=False, prefill_context_parallel_size=2)),
                  "prefill_context_parallel_size"),
        "fingerprint": (lambda: FPMismatch(8), bm2, None, "fingerprint mismatch"),
        "offset kernel": (lambda: M.EmulatedWorld(8), make_bm2(log, "offset"), None, "BOOT SELF-TEST FAILED"),
        "ntok kernel": (lambda: M.EmulatedWorld(8), make_bm2(log, "ntok"), None, "BOOT SELF-TEST FAILED"),
        "inputs differ across ranks": (lambda: CrossMismatch(8), bm2, None, "BOOT SELF-TEST FAILED"),
        "another rank failed": (lambda: OtherRankFails(8), bm2, None, "rank 6"),
    }
    out = {}
    for k, (wf, kern, cfg, pat) in refusals.items():
        out[k] = raises(lambda: boot_with(wf, kern, cfg), RuntimeError, pat) and not M.stats()["active"]
    check("B9 boot refusals raise and leave the split off: " + ", ".join(refusals), all(out.values()), str(out))
    check("B9 an accepted vLLM config (TP group 8, no SP-MoE / PP / CP) boots",
          not raises(lambda: boot_with(lambda: M.EmulatedWorld(8), bm2, NS(parallel_config=NS(
              use_sequence_parallel_moe=False, pipeline_parallel_size=1, prefill_context_parallel_size=1,
              decode_context_parallel_size=1)))) and M.stats()["active"])
    # B10 synthetic generators
    a = M.synthetic_x(130, K, 5, "cpu")
    b = M.synthetic_x(130, K, 5, "cpu")
    c = M.synthetic_x(130, K, 6, "cpu")
    f = M.synthetic_fn(24, K, 7, "cpu")
    low16 = int(((f.view(torch.int32) & 0xFFFF) != 0).sum())
    check("B10 synthetic x: deterministic, seed-dependent, bf16 in +-64; fn fp32 with full mantissas",
          torch.equal(a.view(torch.int16), b.view(torch.int16)) and not torch.equal(a, c) and a.dtype == torch.bfloat16
          and float(a.float().abs().max()) <= 64 and f.dtype == torch.float32 and low16 > 0.9 * f.numel()
          and float(f.abs().max()) <= 0.5)
    # B11 fingerprint
    fp0 = M.fingerprint(1024, 8)
    with env(VLLM_GB10_MHC_BM2_MIN_T="256"):
        fp_bm2 = M.fingerprint(1024, 8)
    check("B11 fingerprint changes with MIN_T, TP size and patch 011's threshold",
          len({fp0, M.fingerprint(2048, 8), M.fingerprint(1024, 4), fp_bm2}) == 4)
    M._set_world_factory_for_test(None)
    M._reset_for_test()
    return M



# ---------------------------------------------------------------------------------------------------------------------
# A. pins + applier
# ---------------------------------------------------------------------------------------------------------------------
def build_bases(vroot, work):
    """The tilelang.py bases and the tilelang_kernels.py variants through the REAL 011 / 019 .sh scripts (bash, scratch roots)."""
    rels = [TL_REL, TK_REL, "model_executor/warmup/jit_warmup_tilelang_helper.py", "model_executor/warmup/jit_warmup.py",
            "tilelang_utils/__init__.py"]
    tl, tk = {}, {}
    for name, steps in (("ORIG", []), ("P011", ["011"]), ("P019", ["019"]), ("P011_P019", ["011", "019"])):
        r = work / "roots" / name
        shutil.rmtree(r, ignore_errors=True)
        for rel in rels:
            (r / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(vroot / rel, r / rel)
        for s in steps:
            sh, e = (SH11, {"VLLM_GB10_MHC_BM2_MIN_T": "128"}) if s == "011" else (SH19, {"VLLM_GB10_MHC_FN_BF16": "1"})
            p = subprocess.run(["bash", str(sh)], env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"),
                                                       "VLLM_ROOT": str(r), "PYTHONDONTWRITEBYTECODE": "1", **e},
                               capture_output=True, text=True, timeout=120)
            if p.returncode != 0:
                raise SystemExit(f"{sh.name} failed on {name}:\n{p.stdout}\n{p.stderr}")
        tk[name] = (r / TK_REL).read_text()
        if name in ("ORIG", "P019"):
            tl[name] = (r / TL_REL).read_text()
    return tl, tk


def strip_022(tree):
    """The AST with patch 022's nodes removed (module-level _P022_ON assign + `if _P022_ON:` import, the hook)."""
    def is22(n):
        names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        return "_P022_ON" in names or "_p022" in names
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or not is22(n)]
    for f in tree.body:
        if isinstance(f, ast.FunctionDef) and f.name == "_hc_prenorm_gemm_outputs":
            for n in f.body:
                if isinstance(n, ast.If):
                    n.orelse = [s for s in n.orelse if not is22(s)]
    return tree


def section_a(vroot, work, P22, P19, P11):
    print("== A. pins + applier on the real v0.32.0 source")
    check("A0 applier / module shas == the .sh pins",
          sha_file(HERE / "p022_apply.py") == pin("APPLIER_SHA") and sha_file(HERE / "gb10_mhc_prs.py") == pin("MOD_SHA"))
    check("A1 tilelang.py == TL_ORIG (022 pin == 019 pin)", sha_file(vroot / TL_REL) == pin("TL_ORIG") == pin("TL_ORIG", SH19))
    check("A1 tilelang_kernels.py == the 011 / 019 TK_ORIG and is in 022's known set",
          sha_file(vroot / TK_REL) == pin("TK_ORIG", SH11) == pin("TK_ORIG", SH19)
          and sha_file(vroot / TK_REL) in Path(SH22).read_text())
    tl, tk = build_bases(vroot, work)
    check("A2 TL_ORIG / TL_P019 built by the real 019 .sh == the pins (022 == 019)",
          sha_text(tl["ORIG"]) == pin("TL_ORIG") and sha_text(tl["P019"]) == pin("TL_P019") == pin("TL_P019", SH19))
    for v in ("ORIG", "P011", "P019", "P011_P019"):
        check(f"A2 tilelang_kernels.py {v} built by the real 011 / 019 .sh is in 022's known set",
              sha_text(tk[v]) in Path(SH22).read_text() and sha_text(tk[v]) == pin("TK_" + v, SH19), sha_text(tk[v])[:16])
    results = {}
    for b in BASES:
        t = tl[b[3:]]
        r = P22.apply(t)
        results[b] = r
        check(f"A3 {b}: apply == pinned {RESULT[b]}", sha_text(r) == pin(RESULT[b]), sha_text(r)[:16])
        check(f"A3 {b}: revert(apply) == base, byte for byte", P22.revert(r) == t)
        check(f"A3 {b}: markers == {P22.N_MARKERS}", r.count(P22.MARKER) == P22.N_MARKERS)
        check(f"A3 {b}: a second apply is refused", raises(lambda: P22.apply(r), P22.Refused, "already carries"))
        check(f"A3 {b}: AST scope - the result minus the 022 nodes == the base",
              ast.dump(strip_022(ast.parse(r))) == ast.dump(ast.parse(t)))
        f = work / f"cli_{b}.py"
        f.write_text(t)
        p = subprocess.run([sys.executable, str(HERE / "p022_apply.py"), "check", str(f)], capture_output=True, text=True)
        check(f"A3 {b}: CLI check prints the pinned sha", p.returncode == 0 and p.stdout.strip() == pin(RESULT[b]))
        p = subprocess.run([sys.executable, str(HERE / "p022_apply.py"), "apply", str(f)], capture_output=True, text=True)
        check(f"A3 {b}: CLI apply in place -> pinned sha", p.returncode == 0 and sha_file(f) == pin(RESULT[b])
              and not (work / f"cli_{b}.py.p022.tmp").exists())
        p = subprocess.run([sys.executable, str(HERE / "p022_apply.py"), "apply", str(f)], capture_output=True, text=True)
        check(f"A3 {b}: CLI apply on the result -> exit 1, file unchanged", p.returncode == 1 and sha_file(f) == pin(RESULT[b]))
        p = subprocess.run([sys.executable, str(HERE / "p022_apply.py"), "revert", str(f)], capture_output=True, text=True)
        check(f"A3 {b}: CLI revert -> the base byte for byte", p.returncode == 0 and f.read_text() == t)
    orig = tl["ORIG"]
    bad = orig.replace("            hidden_size,\n            hc_mult,\n        )\n    return out, sqrsum\n",
                       "            hidden_size,\n            hc_mult,\n        )\n    return out, sqrsum  \n", 1)
    check("A4 refuses: the kernel-call anchor changed (found 0x)", raises(lambda: P22.apply(bad), P22.Refused, "found 0x"))
    dup = orig + "\n\ndef _hc_prenorm_gemm_outputs(\n    x,\n):\n    pass\n"
    check("A4 refuses: duplicated gate anchor", raises(lambda: P22.apply(dup), P22.Refused, "found 2x"))
    ins = orig.replace("        from vllm.model_executor.kernels.mhc.tilelang_kernels import (\n            _HC_PRENORM_GEMM_TILELANG_KERNEL,\n        )\n\n        _HC_PRENORM_GEMM_TILELANG_KERNEL(\n",
                       "        from vllm.model_executor.kernels.mhc.tilelang_kernels import (\n            _HC_PRENORM_GEMM_TILELANG_KERNEL,\n        )\n        pass\n\n        _HC_PRENORM_GEMM_TILELANG_KERNEL(\n", 1)
    check("A4 refuses: a statement between the import and the kernel call (TileLang branch shape changed)",
          ins != orig and raises(lambda: P22.apply(ins), P22.Refused, "not \\[import, kernel call\\]"))
    check("A4 refuses: tilelang_kernels.py", raises(lambda: P22.apply(tk["ORIG"]), P22.Refused))
    check("A4 revert refuses the unpatched base", raises(lambda: P22.revert(orig), P22.Refused))
    check("A5 019 and 022 commute on tilelang.py (both orders, both bases)",
          P19.apply_tl(results["TL_ORIG"]) == P22.apply(P19.apply_tl(orig)) == P22.apply(tl["P019"])
          and sha_text(P22.apply(tl["P019"])) == pin("TL_P019_P022"))
    check("A5 019's revert on the 019+022 result == the 022 result", P19.revert_tl(P22.apply(tl["P019"])) == results["TL_ORIG"])
    import pmhc_testlib as L0  # noqa: E402  (HERE is on sys.path via section C's import below; add it first)
    st, info = L0.tir_probe(vroot)
    if st == "skip":
        print(f"SKIP A7 TIR row-independence facts: {info.get('skip')} (set TILELANG_PROBE_PATH; see pmhc_tir_probe.py)")
    else:
        facts = info.get("prenorm", {})
        check("A7 TIR (real TileLang 0.1.12 frontend): in the block_m=2 kernel num_tokens appears only in the row predicate, the grid and "
              "the output strides - no arithmetic reads it, the row offset or another row; same arithmetic as the per-token kernel",
              st == "ok" and bool(facts) and all(facts.values()), json.dumps({k: v for k, v in facts.items() if not v}))
    gate = P22.GATE_NEW[2:-len("\n\ndef _hc_prenorm_gemm_outputs(\n")]
    load_prs(KernelLog())
    ns = {}
    with env(VLLM_GB10_MHC_PRS=None):
        exec(gate, ns)
    check("A6 gate block, env unset: _P022_ON False, no module import", ns["_P022_ON"] is False and "_p022" not in ns)
    with env(VLLM_GB10_MHC_PRS="1"):
        for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
            del sys.modules[k]
        pkg = types.ModuleType("vllm")
        pkg.__path__ = []
        sys.modules["vllm"] = pkg
        check("A6 gate block, env set + module missing -> ImportError (fail closed)", raises(lambda: exec(gate, {}), ImportError))
    sentinel = types.ModuleType("vllm._gb10_mhc_prs")
    sys.modules["vllm._gb10_mhc_prs"] = sentinel
    pkg._gb10_mhc_prs = sentinel
    ns = {}
    with env(VLLM_GB10_MHC_PRS="1"):
        exec(gate, ns)
    check("A6 gate block, env set + module present -> _p022 is the module", ns["_P022_ON"] is True and ns["_p022"] is sentinel)
    with env(VLLM_GB10_MHC_PRS="true"):
        ns = {}
        exec(gate, ns)
    check("A6 gate block: only the exact value '1' enables it", ns["_P022_ON"] is False)
    return tl, tk, results



# ---------------------------------------------------------------------------------------------------------------------
# C. the real patched tilelang.py over the real JIT classes
# ---------------------------------------------------------------------------------------------------------------------
def block_inputs(T, hidden, hc, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, hidden, generator=g) * 2).to(torch.bfloat16)
    res = (torch.randn(T, hc, hidden, generator=g) * 4).to(torch.bfloat16)
    post = (torch.sigmoid(torch.randn(T, hc, 1, generator=g)) * 2).contiguous()
    comb = torch.softmax(torch.randn(T, hc, hc, generator=g), -1).contiguous()
    fn = torch.randn((2 + hc) * hc, hc * hidden, generator=g) * 0.05
    scale = torch.rand(3, generator=g) + 0.5
    base = torch.randn((2 + hc) * hc, generator=g) * 0.1
    nw = (torch.rand(hidden, generator=g) + 0.5).to(torch.bfloat16)
    return x, res, post, comb, fn, scale, base, nw


def run_block(envx, T, hidden, hc, seed, which):
    x, res, post, comb, fn, scale, base, nw = block_inputs(T, hidden, hc, seed)
    eps = (1e-5, 1e-6, 1e-6, 2.0, 20)
    with envx.activate():
        if which == "fused":
            return envx.tl.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, scale, base, *eps, 1, 1, nw, 1e-5)
        return envx.tl.mhc_pre_tilelang(res, fn, scale, base, *eps, 1, nw, 1e-5)


def bm2_rows(envx):
    return [int(a[0].shape[0]) for a, _k in envx.jit("hc_prenorm_gemm_block_m_tilelang").calls]


def section_c(root, work, tl, tk, results):
    import torch
    sys.path.insert(0, str(HERE))
    import pmhc_testlib as L

    print("== C. the real patched tilelang.py over the real JIT classes (2 tilelang.py x 4 tilelang_kernels.py bases)")
    hidden, hc = 128, 4
    prs_path = HERE / "gb10_mhc_prs.py"
    for b in BASES:
        for v in ("ORIG", "P011", "P019", "P011_P019"):
            is011 = "P011" in v
            tag = f"{b}/TK_{v}"
            paths = {}
            for nm, text in (("ref_tl", tl[b[3:]]), ("new_tl", results[b]), ("tk", tk[v])):
                paths[nm] = work / "c" / f"{b}_{v}_{nm}.py"
                paths[nm].parent.mkdir(parents=True, exist_ok=True)
                paths[nm].write_text(text)
            bm2env = "128" if is011 else None
            with env(VLLM_GB10_MHC_BM2_MIN_T=bm2env, VLLM_GB10_MHC_FN_BF16=None, VLLM_GB10_MHC_PRS=None):
                e_ref = L.build_env(root, tk_path=paths["tk"], tl_path=paths["ref_tl"])
                e_off = L.build_env(root, tk_path=paths["tk"], tl_path=paths["new_tl"])
            with env(VLLM_GB10_MHC_BM2_MIN_T=bm2env, VLLM_GB10_MHC_FN_BF16=None, VLLM_GB10_MHC_PRS="1"):
                e_on = L.build_env(root, tk_path=paths["tk"], tl_path=paths["new_tl"],
                                   extra=[("vllm._gb10_mhc_prs", prs_path)])
                try:
                    L.build_env(root, tk_path=paths["tk"], tl_path=paths["new_tl"])
                    miss = False
                except ImportError:
                    miss = True
            M = e_on.modules["vllm._gb10_mhc_prs"]
            M._reset_for_test(min_t=1024)
            M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
            check(f"C0 {tag}: gate on imports the module, gate off does not; gate on + module missing -> ImportError",
                  e_on.tl._P022_ON is True and e_on.tl._p022 is M and e_off.tl._P022_ON is False
                  and not hasattr(e_off.tl, "_p022") and miss)
            with env(VLLM_GB10_MHC_BM2_MIN_T=bm2env, VLLM_GB10_MHC_PRS="1"):
                run_block(e_on, 1024, hidden, hc, 1, "pre")      # boot (self-test) on the first eligible call
            check(f"C1 {tag}: boot on the first eligible call", M.stats()["active"] and M.stats()["report"]["ranks"] == 8)
            for T in (1024, 1027, 2048, 3840):
                for which in ("pre", "fused"):
                    for e in (e_ref, e_off, e_on):
                        e.reset_jits()
                    r_ref = run_block(e_ref, T, hidden, hc, T, which)
                    r_off = run_block(e_off, T, hidden, hc, T, which)
                    r_new = run_block(e_on, T, hidden, hc, T, which)
                    pl = M.plan_rows(T, 8)
                    shard_rows = bm2_rows(e_on)
                    check(f"C2 {tag} T={T} {which}: gate on == unpatched == gate off, all outputs bitwise; the block_m=2 "
                          "JIT function ran on 8 shards (unpatched: once on all T rows)",
                          all(L.tensors_equal_bits(u, w) and L.tensors_equal_bits(u, z) for u, w, z in zip(r_ref, r_new, r_off))
                          and shard_rows == [pl.rows(r) for r in range(8)] and bm2_rows(e_ref) == [T] and not bm2_rows(e_off) == []
                          and bm2_rows(e_off) == [T], f"shards {shard_rows}")
            # below MIN_T: no split (011 bases: block_m=2 on the full T; otherwise the per-token kernel)
            for e in (e_ref, e_on):
                e.reset_jits()
            r_ref = run_block(e_ref, 512, hidden, hc, 5, "fused")
            r_new = run_block(e_on, 512, hidden, hc, 5, "fused")
            kinds = (["bm2"] * len(bm2_rows(e_on))) + (["tok"] * len(e_on.jit("hc_prenorm_gemm_tilelang").calls))
            check(f"C3 {tag}: T=512 < MIN_T -> unchanged single call ({'bm2' if is011 else 'per-token'}), bitwise equal",
                  kinds == (["bm2"] if is011 else ["tok"]) and all(L.tensors_equal_bits(u, w) for u, w in zip(r_ref, r_new)),
                  str(kinds))
            # MIN_T 256: the hook only acts when the CLASS routes the call to block_m=2
            M._reset_for_test(min_t=256)
            run_block(e_on, 1024, hidden, hc, 1, "pre")          # re-boot at the new MIN_T
            for e in (e_ref, e_on):
                e.reset_jits()
            r_ref = run_block(e_ref, 512, hidden, hc, 5, "fused")
            r_new = run_block(e_on, 512, hidden, hc, 5, "fused")
            n_bm2 = len(bm2_rows(e_on))
            check(f"C4 {tag}: MIN_T=256, T=512 -> " + ("split (the class routes 512 to block_m=2 via 011)" if is011 else
                  "no split (upstream threshold 1024: the class does not route it to block_m=2) and counted not_bm2"),
                  (n_bm2 == 8 if is011 else n_bm2 == 0 and M.stats()["not_bm2"] >= 1)
                  and all(L.tensors_equal_bits(u, w) for u, w in zip(r_ref, r_new)), str(n_bm2))
            # num_tokens <= 32: the fused small kernel, the pre-norm hook is never reached
            for e in (e_ref, e_on):
                e.reset_jits()
            r_ref = run_block(e_ref, 8, hidden, hc, 6, "fused")
            r_new = run_block(e_on, 8, hidden, hc, 6, "fused")
            check(f"C5 {tag}: num_tokens=8 decode path -> the fused small kernel only (no pre-norm kernel), bitwise equal",
                  not bm2_rows(e_on) and not e_on.jit("hc_prenorm_gemm_tilelang").calls and e_on.jit("mhc_fused_tilelang").calls
                  and all(L.tensors_equal_bits(u, w) for u, w in zip(r_ref, r_new)))
            # DeepGEMM branch: the hook sits in the TileLang branch and is never reached
            M._reset_for_test(min_t=1024)
            M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
            e_on.deep_gemm_supported = True
            e_on.reset_jits()
            e_on.deep_gemm_calls.clear()
            r_dg = run_block(e_on, 2048, hidden, hc, 7, "fused")
            e_on.deep_gemm_supported = False
            check(f"C6 {tag}: is_deep_gemm_supported() -> the DeepGEMM branch runs, 022 never acts (no boot, no TileLang prenorm)",
                  e_on.deep_gemm_calls and not bm2_rows(e_on) and not M.stats()["active"] and M.stats()["calls"] == 0)
            M._reset_for_test()
            M._set_world_factory_for_test(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default="ABC")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch

    torch.set_num_threads(1)
    work = Path(a.work or tempfile.mkdtemp(prefix="p022-cpu-"))
    work.mkdir(parents=True, exist_ok=True)
    vroot = Path(a.vllm_root)
    P22 = load_file_module("p022_apply", HERE / "p022_apply.py")
    P19 = load_file_module("p019_apply", HERE / "p019_apply.py")
    P11 = load_file_module("p011_apply", HERE / "p011_apply.py")
    tl = tk = results = None
    if "A" in a.only or "C" in a.only:
        tl, tk, results = section_a(vroot, work, P22, P19, P11)
    if "B" in a.only:
        section_b(work)
    if "C" in a.only:
        M = section_b(work) if False else None
        section_c(vroot, work, tl, tk, results)
    print(f"p022 CPU checks: {RESULTS['pass']} passed, {RESULTS['fail']} failed"
          + (f" ({', '.join(RESULTS['failed'][:8])})" if RESULTS["failed"] else ""))
    print("ALL PASS" if RESULTS["fail"] == 0 else "SOME FAILED")
    return 0 if RESULTS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
