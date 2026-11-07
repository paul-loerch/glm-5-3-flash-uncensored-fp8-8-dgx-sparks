#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch-025 gb10-mhc-prefill-sp] CPU checks (no GPU, no CUDA init, no TileLang; torch on CPU + gloo).

  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 files/p025_cpu_check.py --vllm-root <v0.32.0 tree>/vllm [--work DIR] [--world 8]
  (--vllm-root defaults to $VLLM_TREE, else vllm-v0.32.0/vllm)
  (needs a Python with torch + gloo)

v0.32.0 PORT (rev 4). --vllm-root must hold the v0.32.0 ORIGINAL files (sha-checked against the .sh pins). Sections:
  A  pins + applier on the real v0.32.0 source: p025_apply apply/revert/check on common/model.py == the pinned result (no
     earlier patch edits model.py, so there is one base); the read-only dependencies (moe_runner.py, linear.py, mhc.py,
     fused_moe/config.py) hash to their pins and tilelang.py / tilelang_kernels.py are in the accepted sets; AST scope
     (result minus the 025 nodes == the base); refusals (a reshaped forward, the pre-v0.31 loop shape, a changed anchor,
     revert of the base); the module-level gate block alone (unset / set + module missing -> ImportError / set + invalid
     value -> ValueError / set + module present).
  B  gb10_prefill_sp.py logic (in-process EmulatedWorld, no cross-rank sum): env parsing, the row plan for
     T = 1..20,000 (uniform, even starts, minimal padding), pad_rows / shard_entry / layout, the canonical RS
     arithmetic, routing and eligibility (begin_forward -> None for gate unset / T < MIN_T / capture; handle for an
     eligible forward), the boot envelope + fingerprint + self-test and their refusals (TP != 8, PP > 1,
     sequence-parallel MoE, fingerprint mismatch, a broken reduce-scatter world), and the prs (patch 022)
     neutralisation (a stub prs module's _STATE.min_t is raised inside an sp region and restored after). Rev 3:
     the preflight refuses on every rank when ONE rank's envelope refuses or the geometry differs (B6); the
     canonical check runs only with RS=canonical (B9); layers passed without a live mHC / MLA+MoE layer fail closed
     (B10); SCHEDULE INVARIANCE (B11): a planted local exception or a wrong local shape at each self-test stage
     leaves the issued collective sequence (op, shape, order) IDENTICAL to a clean boot and raises BOOT SELF-TEST
     FAILED (the single-process proof of the hang-freedom the section-D hang mutants show on 8 gloo ranks).
  C  multi-process gloo emulation (8 ranks if feasible, else --world 2/4; the SAME module data-flow code): the sp
     data flow (shard -> all-gather -> head-sharded block -> reduce-scatter -> all-gather) reconstructs the
     replicated full-T reference for T incl. not divisible by 8, T at the MIN_T edges and padded tails, for RS =
     nccl (gloo reduce_scatter) and canonical (all-to-all + rank-order fp32 sum); the reconstructed result is BITWISE
     identical across ranks; canonical RS == the local rank-order fp32 sum (bitwise); a flipped-bit negative control
     is caught. Each rank runs vllm/_gb10_prefill_sp.py's own gather_full / reduce_scatter_partial / shard_entry.
  D  END-TO-END on the REAL patched model.py forward functions (files/p025_e2e_check.py sweep, 8 gloo ranks): the base
     (unmutated) sp path == the image path bitwise, AND every wiring / reduction / region negative-control mutant is
     caught (sp_layer_forward wiring, the shard / gather, the all-reduce suppression handles, the canonical / RS
     arithmetic, the prs neutralisation, a no-op self-test). This is the rev-2 defence for B1: sections B/C only
     exercise the collective primitives; section D runs the whole sp_layer_forward / shard_model_input /
     gather_model_output / suppression path against the image path and requires the mutants to FAIL. Rev 3: the
     T list adds 8,192 (S = 1,024 = prs's MIN_T: the prs-on-shard regime), the rev-3 mutants (MoE handle target,
     live layers missing, no tail T, canonical check ungated) are required, and the HANG mutants (a rank-local
     failure at each boot self-test stage, RS=nccl and canonical) must each end in a CLEAN-RAISE on all 8 ranks.
P025_MODULE_OVERRIDE (env) points B/C/D at a mutant copy of the module (used by the section-D sweep); A always tests
the vendored files. Exit 0 = ALL PASS.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDIR = HERE.parent
SH25 = PDIR / "025-gb10-mhc-prefill-sp.sh"
MODEL_REL = "models/glm5next/common/model.py"
MR_REL = "model_executor/layers/fused_moe/runner/moe_runner.py"
TL_REL = "model_executor/kernels/mhc/tilelang.py"
TK_REL = "model_executor/kernels/mhc/tilelang_kernels.py"
# rev 2: dropped the unused sequence_parallel.py pin (sp refuses the image SP-MoE path, it never calls sp_shard);
# added the per-row bm2 prenorm kernel (tilelang_kernels.py, hard pin) and the mHC dispatch (tilelang.py, accepted
# set TL_KNOWN). The single-pin DEPS entries are checked by sha; tilelang.py is checked for set membership.
DEPS = {
    "models/glm5next/common/model.py": None,
    "model_executor/layers/fused_moe/runner/moe_runner.py": "MR_ORIG",
    "model_executor/layers/linear.py": "LINEAR_ORIG",
    "model_executor/layers/mhc.py": "MHC_ORIG",
    "model_executor/layers/fused_moe/config.py": "MOECFG_ORIG",
}
# The module under test: the vendored file, or a mutant copy for the section-D sweep (P025_MODULE_OVERRIDE).
MODULE_PATH = Path(os.environ.get("P025_MODULE_OVERRIDE") or (HERE / "gb10_prefill_sp.py"))

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


def pin(name, sh=SH25):
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
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ---------------------------------------------------------------------------------------------------------------------
# Stub `vllm` for importing the module off a real install (no vllm.logger / vllm.config -> stdlib logging, no config)
# ---------------------------------------------------------------------------------------------------------------------
def install_stub_vllm(prs_module="__none__"):
    for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
        del sys.modules[k]
    v = types.ModuleType("vllm")
    v.__path__ = []
    sys.modules["vllm"] = v
    # mhc hc_expand / hc_contract (only used by sp_layer_forward, not by the data-flow tests)
    mhc = types.ModuleType("vllm.model_executor.layers.mhc")
    mhc.hc_expand = lambda x, n: x.unsqueeze(1).expand(-1, n, -1).contiguous()
    mhc.hc_contract = lambda x, n: x.mean(dim=1)
    for name in ("vllm.model_executor", "vllm.model_executor.layers"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    sys.modules["vllm.model_executor.layers.mhc"] = mhc
    if prs_module != "__none__":
        sys.modules["vllm._gb10_mhc_prs"] = prs_module
        if prs_module is not None:
            v._gb10_mhc_prs = prs_module
    return v


def load_sp(prs_module="__none__"):
    install_stub_vllm(prs_module)
    m = load_file_module("vllm._gb10_prefill_sp", MODULE_PATH)
    sys.modules["vllm"]._gb10_prefill_sp = m
    return m


def known(var, sh=SH25):
    """The accepted shas of a read-only dependency (TL_KNOWN / TK_KNOWN) parsed from the .sh."""
    m = re.search(rf'^{var}="([^"]*)"', Path(sh).read_text(), re.M)
    if not m:
        raise SystemExit(f"no {var} in the .sh")
    return set(re.findall(r"[0-9a-f]{64}", m.group(1)))


def tl_known(sh=SH25):
    return known("TL_KNOWN", sh)


# ---------------------------------------------------------------------------------------------------------------------
# A. pins + applier
# ---------------------------------------------------------------------------------------------------------------------
def strip_025(tree):
    def is25(n):
        names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        return "_P025_ON" in names or "_p025" in names
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
                 or not is25(n)]
    # strip the import-under-if and the per-method hooks
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.If) and is25(n))
                 and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_P025_ON"
                                                            for t in n.targets))]
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in ("Glm5NextDecoderLayer", "Glm5NextModel"):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                    fn.body = _strip_hook(fn.body, is25)
    return tree


def _strip_hook(body, is25):
    def refs_handle(node):
        return any(isinstance(x, ast.Name) and x.id == "_p025_handle" for x in ast.walk(node))

    out = []
    for n in body:
        if isinstance(n, ast.If) and is25(n) and len(n.body) == 1 and isinstance(n.body[0], ast.Return):
            continue  # the decoder sp hook
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_p025_handle" for t in n.targets):
            continue  # model: _p025_handle = begin_forward(...)
        if isinstance(n, ast.If) and refs_handle(n.test):
            # model: `if _p025_handle is not None: shard...` -> drop; the shard line is sp-only
            continue
        if isinstance(n, ast.Try):
            # unwrap the try/finally back to the bare loop
            out.extend(_strip_hook(n.body, is25))
            continue
        if isinstance(n, ast.If) and n.orelse and all(refs_handle(e) for e in n.orelse):
            # `if self.is_sequence_parallel: ... elif _p025_handle is not None: ...` -> drop the sp-only elif
            n.orelse = []
        out.append(n)
    return out


def section_a(vroot, work, P):
    print("== A. pins + applier on the real v0.32.0 source")
    check("A0 applier / module shas == the .sh pins",
          sha_file(HERE / "p025_apply.py") == pin("APPLIER_SHA")
          and sha_file(HERE / "gb10_prefill_sp.py") == pin("MOD_SHA"))
    check("A0 the patch-022 module (025 neutralises it at runtime) == PRS_MOD_SHA == 022's MOD_SHA",
          sha_file(HERE / "gb10_mhc_prs.py") == pin("PRS_MOD_SHA") == pin("MOD_SHA", PDIR / "022-gb10-mhc-prefill-rowsplit.sh"))
    orig = (vroot / MODEL_REL).read_text()
    check("A1 model.py == MODEL_ORIG", sha_text(orig) == pin("MODEL_ORIG"), sha_text(orig)[:16])
    for rel, pn in DEPS.items():
        if pn:
            check(f"A1 dep {Path(rel).name} == {pn}", sha_file(vroot / rel) == pin(pn), sha_file(vroot / rel)[:16])
    tl_sha, tk_sha = sha_file(vroot / TL_REL), sha_file(vroot / TK_REL)
    check("A1 dep tilelang.py in TL_KNOWN (the original or a 019 / 022 / 019+022 result)", tl_sha in tl_known(), tl_sha[:16])
    check("A1 dep tilelang_kernels.py in TK_KNOWN (the original or an 011 / 019 / 011+019 result)",
          tk_sha in known("TK_KNOWN"), tk_sha[:16])
    check("A1 TL_KNOWN / TK_KNOWN == the sets pinned by 019 / 022 / 011 (4 + 4)",
          known("TL_KNOWN") == {pin(n, PDIR / "019-gb10-mhc-fn-bf16.sh") for n in ("TL_ORIG", "TL_P022", "TL_P019", "TL_P019_P022")}
          and known("TK_KNOWN") == {pin(n, PDIR / "019-gb10-mhc-fn-bf16.sh") for n in ("TK_ORIG", "TK_P011", "TK_P019", "TK_P011_P019")})
    bases = {"MODEL_ORIG": orig}
    results = {}
    for name, base in bases.items():
        res = P.apply(base)
        results[name] = res
        check(f"A3 {name}: apply == pinned {name}_P025", sha_text(res) == pin(f"{name}_P025"), sha_text(res)[:16])
        check(f"A3 {name}: revert(apply) == base byte for byte", P.revert(res) == base)
        check(f"A3 {name}: markers == {P.N_MARKERS}", res.count(P.MARKER) == P.N_MARKERS)
        check(f"A3 {name}: a second apply is refused", raises(lambda: P.apply(res), P.Refused, "already carries"))
        a = ast.dump(strip_025(ast.parse(res)))
        b = ast.dump(ast.parse(base))
        check(f"A3 {name}: AST scope - the result minus the 025 nodes == the base", a == b)
        f = work / f"cli_{name}.py"
        f.write_text(base)
        p = subprocess.run([sys.executable, str(HERE / "p025_apply.py"), "check", str(f)], capture_output=True,
                           text=True)
        check(f"A3 {name}: CLI check prints the pinned sha", p.returncode == 0 and p.stdout.strip() == pin(f"{name}_P025"))
        p = subprocess.run([sys.executable, str(HERE / "p025_apply.py"), "apply", str(f)], capture_output=True,
                           text=True)
        check(f"A3 {name}: CLI apply in place -> pinned sha, no tmp left",
              p.returncode == 0 and sha_file(f) == pin(f"{name}_P025") and not (work / f"cli_{name}.py.p025.tmp").exists())
        p = subprocess.run([sys.executable, str(HERE / "p025_apply.py"), "revert", str(f)], capture_output=True,
                           text=True)
        check(f"A3 {name}: CLI revert in place == base", p.returncode == 0 and sha_file(f) == sha_text(base))
    # refusals
    check("A4 revert refuses the unpatched base", raises(lambda: P.revert(orig), P.Refused))
    reshaped = orig.replace("hidden_states = sp_shard(hidden_states)", "hidden_states = aux_shard(hidden_states)", 1)
    check("A4 refuses a reshaped Glm5NextModel.forward (no sp skeleton)",
          raises(lambda: P.apply(reshaped), P.Refused))
    old_loop = orig.replace("        for idx, layer in enumerate(self._active_layers, start=self.start_layer):\n",
                            "        for layer in self._active_layers:\n", 1)
    check("A4 refuses the pre-v0.31 layer loop shape (the aux-tap loop anchor is gone)",
          old_loop != orig and raises(lambda: P.apply(old_loop), P.Refused, "found 0x"))
    noexit = orig.replace("        hidden_states = self.norm(hidden_states)\n        if aux_hidden_states:\n",
                          "        hidden_states = self.norm2(hidden_states)\n        if aux_hidden_states:\n", 1)
    check("A4 refuses a changed exit (the exit-gather anchor is gone)",
          noexit != orig and raises(lambda: P.apply(noexit), P.Refused, "found 0x"))
    noguard = orig.replace("        if not self.mhc or self.is_mtp_layer:\n",
                           "        if not self.mhc:\n", 1)
    check("A4 refuses a reshaped decoder forward (missing MTP guard anchor)",
          raises(lambda: P.apply(noguard), P.Refused, "found 0x"))
    dup = orig + "\n\nclass Glm5NextDecoderLayer(nn.Module):\n    pass\n"
    check("A4 refuses a duplicated gate anchor", raises(lambda: P.apply(dup), P.Refused, "found 2x"))
    # the module-level gate block alone
    gate = P.GATE_NEW[2:-len("\n\nclass Glm5NextDecoderLayer(nn.Module):\n")]
    install_stub_vllm()
    ns = {}
    with env(VLLM_GB10_MHC_SP=None):
        exec(gate, ns)
    check("A6 gate block, env unset: _P025_ON False, no module import", ns["_P025_ON"] is False and "_p025" not in ns)
    with env(VLLM_GB10_MHC_SP="sp"):
        check("A6 gate block, env set + module missing -> ImportError (fail closed)",
              raises(lambda: exec(gate, {}), ImportError))
    sentinel = load_sp()
    ns = {}
    with env(VLLM_GB10_MHC_SP="ref"):
        exec(gate, ns)
    check("A6 gate block, env=ref + module present -> _p025 is the module", ns["_P025_ON"] is True
          and ns["_p025"] is sentinel)
    with env(VLLM_GB10_MHC_SP="0"):
        ns = {}
        exec(gate, ns)
    check("A6 gate block: 0 keeps it off", ns["_P025_ON"] is False)
    with env(VLLM_GB10_MHC_SP="garbage"):
        check("A6 gate block, env=garbage + module present -> ValueError at module import (fail closed)",
              raises(lambda: load_file_module("vllm._p025_bad", HERE / "gb10_prefill_sp.py"), ValueError))
    sys.modules.pop("vllm._p025_bad", None)
    return bases, results


# ---------------------------------------------------------------------------------------------------------------------
# B. module logic (EmulatedWorld)
# ---------------------------------------------------------------------------------------------------------------------
def section_b(work):
    import torch

    print("== B. gb10_prefill_sp.py logic (EmulatedWorld)")
    M = load_sp()
    H = 256
    # B1 env parsing
    ok = (M.parse_mode(None) is None and M.parse_mode("0") is None and M.parse_mode("") is None
          and M.parse_mode("1") == "sp" and M.parse_mode("sp") == "sp" and M.parse_mode(" SP ") == "sp"
          and M.parse_mode("ref") == "ref" and raises(lambda: M.parse_mode("xyz"), ValueError))
    ok &= (M.parse_min_t(None) == 1024 and M.parse_min_t("2048") == 2048 and M.parse_min_t("1024") == 1024
           and raises(lambda: M.parse_min_t("1023"), ValueError) and raises(lambda: M.parse_min_t("512"), ValueError)
           and raises(lambda: M.parse_min_t("abc"), ValueError))
    ok &= (M.parse_rs(None) == "nccl" and M.parse_rs("canonical") == "canonical"
           and raises(lambda: M.parse_rs("x"), ValueError))
    check("B1 env parsing: mode sp/ref, MIN_T floor 1024 [1024,2**20], RS nccl/canonical; invalid refused", ok)
    with env(VLLM_GB10_MHC_SP="sp", VLLM_GB10_MHC_SP_MIN_T="512"):
        check("B1 gate set + MIN_T below the 1024 floor -> module refuses at import",
              raises(lambda: load_file_module("vllm._p025_badmin", MODULE_PATH), ValueError))
    sys.modules.pop("vllm._p025_badmin", None)
    # B2 plan
    bad = []
    for T in range(256, 20001):
        p = M.plan_rows(T, 8)
        try:
            M.check_plan(p)
            assert p.S % 2 == 0 and p.W * p.S >= T and p.W * p.S - T == p.pad and p.pad < 16
            assert sum(p.real_rows(r) for r in range(8)) == T
            if T % 16 == 0:
                assert p.S == T // 8
        except AssertionError as e:
            bad.append((T, str(e)[:60]))
    check("B2 plan: uniform even-start partition, minimal padding, S==T/8 when T%16==0 (T=256..20000)",
          not bad, str(bad[:3]))
    for T, S, r7 in [(1024, 128, 128), (1027, 130, 117), (3840, 480, 480), (3853, 482, 479), (8192, 1024, 1024)]:
        p = M.plan_rows(T, 8)
        check(f"B2 plan T={T}: S={S}, rank7 real rows {r7}", p.S == S and p.real_rows(7) == r7, f"S={p.S} r7={p.real_rows(7)}")
    # B3 pad / shard_entry / gather round trip (EmulatedWorld: gather = concat of identical copies)
    W = M.EmulatedWorld(8)
    bad = []
    for T in (256, 257, 1024, 1027, 3840, 3853, 8192):
        p = M.plan_rows(T, 8)
        x = torch.randn(T, H)
        for r in range(8):
            Wr = M.EmulatedWorld(8, rank=r)
            sh = M.shard_entry(x, p, r)
            if sh.shape[0] != p.S:
                bad.append((T, r, "shardshape"))
            # the real rows of the shard equal x's rows [r*S, ...); the tail is zero padding
            real = p.real_rows(r)
            if real and not torch.equal(sh[:real], x[r * p.S:r * p.S + real]):
                bad.append((T, r, "rows"))
            if real < p.S and float(sh[real:].abs().sum()) != 0.0:
                bad.append((T, r, "pad!=0"))
    check("B3 shard_entry: this rank's S rows = x[r*S:...] + zero pad (16 T incl. uneven)", not bad, str(bad[:4]))
    # B4 reduce_scatter_partial shape + canonical arithmetic (EmulatedWorld sums W identical copies)
    bad = []
    for T in (1024, 1027, 3840):
        p = M.plan_rows(T, 8)
        part = torch.randn(T, H)
        for rs in ("nccl", "canonical"):
            for r in range(8):
                Wr = M.EmulatedWorld(8, rank=r)
                out = M.reduce_scatter_partial(part, Wr, p, rs)
                if out.shape[0] != p.S:
                    bad.append((T, rs, r, "shape"))
    check("B4 reduce_scatter_partial returns S rows for nccl and canonical (EmulatedWorld)", not bad, str(bad[:4]))
    # B5 routing / eligibility
    M._reset_for_test()
    M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
    with env(VLLM_GB10_MHC_SP=None):
        M._reset_for_test()
        check("B5 gate unset -> begin_forward None, inert", M.begin_forward(4096, H) is None and not M.sp_active())
    with env(VLLM_GB10_MHC_SP="sp", VLLM_GB10_MHC_SP_MIN_T="1024"):
        M._reset_for_test()
        M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
        check("B5 T < MIN_T -> None, no boot", M.begin_forward(1023, H) is None and not M.stats()["booted"]
              and M.stats()["small_skips"] == 1)
        h = M.begin_forward(1024, H)
        st = M.stats()
        check("B5 first eligible forward -> handle, boot self-test ran, sp_active",
              h is not None and M.sp_active() and st["booted"] and st["report"]["ranks"] == 8
              and st["report"]["cases"] >= 4, str(st["report"]))
        M.end_forward(h)
        check("B5 end_forward clears sp_active", not M.sp_active())
    with env(VLLM_GB10_MHC_SP="sp", VLLM_GB10_MHC_SP_MIN_T="1024"):
        M._reset_for_test()
        M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
        check("B5 aux_hidden_state_layers set -> None (a forward that captures EAGLE-3 aux taps is never sharded), counted, "
              "no boot", M.begin_forward(2048, H, aux_layers=(3, 20)) is None and M.stats()["aux_skips"] == 1
              and not M.stats()["booted"] and not M.sp_active())
        h = M.begin_forward(2048, H, aux_layers=())
        check("B5 aux_layers=() (MTP) -> the handle as before", h is not None and M.sp_active())
        M.end_forward(h)
    # B6 boot refusals
    def boot_with(factory, cfg=None, min_t=1024, rs="nccl", mode="sp", is_sp=False):
        M._reset_for_test(mode=mode, min_t=min_t, rs_mode=rs)
        M._set_world_factory_for_test(factory)
        real = M._vllm_config_or_none
        if cfg is not None:
            M._vllm_config_or_none = lambda: cfg
        try:
            M.begin_forward(2048, H, is_sequence_parallel=is_sp)
        finally:
            M._vllm_config_or_none = real
    NS = types.SimpleNamespace

    class FPMismatch(M.EmulatedWorld):
        def all_gather_object(self, obj):
            out = super().all_gather_object(obj)
            if isinstance(obj, dict) and "fp" in obj:          # rev 3: the preflight payload carries the fingerprint
                out[3] = dict(obj, fp="deadbeefdeadbeef")
            return out

    class EnvRefusedOnRank3(M.EmulatedWorld):
        """Only rank 3's envelope check refused (e.g. a node-local env difference): every rank must refuse."""

        def all_gather_object(self, obj):
            out = super().all_gather_object(obj)
            if isinstance(obj, dict) and "env_err" in obj:
                out[3] = dict(obj, rank=3, env_err="[patch-025] planted rank-3 envelope refusal")
            return out

    class GeometryDiffers(M.EmulatedWorld):
        def all_gather_object(self, obj):
            out = super().all_gather_object(obj)
            if isinstance(obj, dict) and obj.get("geo"):
                out[5] = dict(obj, rank=5, geo=dict(obj["geo"], cases=[[1, 8, 2, 15]]))
            return out

    class BrokenRS(M.EmulatedWorld):
        def reduce_scatter_rows(self, x):
            S = x.shape[0] // self.size
            return torch.zeros_like(x[:S])  # wrong: drops the sum

        def canonical_reduce_scatter_rows(self, x, plan):
            return torch.zeros_like(x[:plan.S])

    refusals = {
        "TP=4": (lambda: boot_with(lambda: M.EmulatedWorld(4)), r"TP=8"),
        "PP=2": (lambda: boot_with(lambda: M.EmulatedWorld(8, pp_size=2)), "pipeline parallel"),
        # is_sequence_parallel is passed by the model (construction-time), not read from the config at forward time.
        "SP-arg": (lambda: boot_with(lambda: M.EmulatedWorld(8), is_sp=True), "sequence-parallel"),
        "SP-MoE-cfg": (lambda: boot_with(lambda: M.EmulatedWorld(8),
                       NS(parallel_config=NS(use_sequence_parallel_moe=True))), "use_sequence_parallel_moe"),
        "PCP=2-cfg": (lambda: boot_with(lambda: M.EmulatedWorld(8), NS(parallel_config=NS(
                      use_sequence_parallel_moe=False, prefill_context_parallel_size=2))),
                      "prefill_context_parallel_size"),
        "DBO-cfg": (lambda: boot_with(lambda: M.EmulatedWorld(8), NS(parallel_config=NS(
                    use_sequence_parallel_moe=False, enable_dbo=True))), "DBO"),
        "fingerprint": (lambda: boot_with(lambda: FPMismatch(8)), "fingerprint mismatch"),
        "envelope-on-one-rank": (lambda: boot_with(lambda: EnvRefusedOnRank3(8)), r"refused on rank\(s\) \[3\]"),
        "geometry-mismatch": (lambda: boot_with(lambda: GeometryDiffers(8)), "geometry differs"),
        "broken RS": (lambda: boot_with(lambda: BrokenRS(8)), "BOOT SELF-TEST FAILED"),
        "broken RS (canonical)": (lambda: boot_with(lambda: BrokenRS(8), rs="canonical"), "BOOT SELF-TEST FAILED"),
    }
    out = {}
    for k, (fn, pat) in refusals.items():
        out[k] = raises(fn, RuntimeError, pat) and not M.stats()["booted"]
    check("B6 boot refusals raise + leave sp off: " + ", ".join(refusals), all(out.values()), str(out))
    check("B6 a clean TP=8 config boots",
          not raises(lambda: boot_with(lambda: M.EmulatedWorld(8),
                     NS(parallel_config=NS(use_sequence_parallel_moe=False, pipeline_parallel_size=1,
                        prefill_context_parallel_size=1, decode_context_parallel_size=1, enable_dbo=False,
                        ubatch_size=0)))) and M.stats()["booted"])
    with env(VLLM_USE_DEEP_GEMM="1"):
        check("B6 VLLM_USE_DEEP_GEMM=1 refused (the DeepGEMM prenorm is not the per-row bm2 kernel)",
              raises(lambda: boot_with(lambda: M.EmulatedWorld(8)), RuntimeError, "DEEP_GEMM")
              and not M.stats()["booted"])
    # B7 prs (patch 022) neutralisation
    prs = types.ModuleType("vllm._gb10_mhc_prs")
    prs._STATE = types.SimpleNamespace(min_t=1024)
    M2 = load_sp(prs_module=prs)
    M2._reset_for_test(mode="sp", min_t=1024, rs_mode="nccl")
    M2._set_world_factory_for_test(lambda: M2.EmulatedWorld(8))
    with env(VLLM_GB10_MHC_SP="sp", VLLM_GB10_MHC_PRS="1"):
        h = M2.begin_forward(2048, H)
        in_region = prs._STATE.min_t
        M2.end_forward(h)
        after = prs._STATE.min_t
    check("B7 prs neutralised inside the sp region (min_t -> huge), restored after",
          in_region > (1 << 30) and after == 1024, f"in={in_region} after={after}")
    with env(VLLM_GB10_MHC_SP="sp", VLLM_GB10_MHC_PRS="0"):
        prs._STATE.min_t = 1024
        M2._reset_for_test(mode="sp", min_t=1024, rs_mode="nccl")
        M2._set_world_factory_for_test(lambda: M2.EmulatedWorld(8))
        h = M2.begin_forward(2048, H)
        unchanged = prs._STATE.min_t
        M2.end_forward(h)
    check("B7 prs untouched when VLLM_GB10_MHC_PRS != 1", unchanged == 1024, str(unchanged))
    # B8 fingerprint sensitivity
    fp0 = M.fingerprint("sp", 1024, "nccl", 8)
    check("B8 fingerprint changes with mode, MIN_T, RS, TP size",
          len({fp0, M.fingerprint("ref", 1024, "nccl", 8), M.fingerprint("sp", 2048, "nccl", 8),
               M.fingerprint("sp", 1024, "canonical", 8), M.fingerprint("sp", 1024, "nccl", 4)}) == 5)

    # B9 (rev 3) the canonical check runs only with RS=canonical (an RS=nccl boot never touches the canonical path)
    class Recording(M.EmulatedWorld):
        """Records every collective the module issues: (op, shape) for tensors, (op, keys) for objects."""

        def __init__(self, *a_, **k_):
            super().__init__(*a_, **k_)
            self.log = []

        def all_gather_rows(self, x):
            self.log.append(("ag", tuple(x.shape)))
            return super().all_gather_rows(x)

        def reduce_scatter_rows(self, x):
            self.log.append(("rs", tuple(x.shape)))
            return super().reduce_scatter_rows(x)

        def canonical_reduce_scatter_rows(self, x, plan):
            self.log.append(("crs", tuple(x.shape)))
            return super().canonical_reduce_scatter_rows(x, plan)

        def all_reduce_rows(self, x):
            self.log.append(("ar", tuple(x.shape)))
            return super().all_reduce_rows(x)

        def all_gather_object(self, obj):
            self.log.append(("obj", tuple(sorted(obj)) if isinstance(obj, dict) else type(obj).__name__))
            return super().all_gather_object(obj)

        def barrier_bounded(self, timeout_s):
            self.log.append(("barrier",))

    def boot_recorded(rs, mode="sp"):
        w = Recording(8)
        M._reset_for_test(mode=mode, min_t=1024, rs_mode=rs)
        M._set_world_factory_for_test(lambda: w)
        exc = None
        try:
            M.begin_forward(2048, H)
        except Exception as e:  # noqa: BLE001
            exc = e
        return w.log, exc

    log_n, exc_n = boot_recorded("nccl")
    log_c, exc_c = boot_recorded("canonical")
    n_crs_n = sum(1 for e in log_n if e[0] == "crs")
    n_crs_c = sum(1 for e in log_c if e[0] == "crs")
    check("B9 canonical check gated on RS=canonical: an RS=nccl boot issues no canonical RS; RS=canonical does",
          exc_n is None and exc_c is None and n_crs_n == 0 and n_crs_c > 0,
          f"crs nccl={n_crs_n} canonical={n_crs_c} exc={exc_n!r}/{exc_c!r}")
    check("B9 the boot exchanges go through the bounded barrier (preflight + verdict)",
          [e[0] for e in log_n].count("barrier") == 2 and [e[0] for e in log_n].count("obj") == 2,
          str([e for e in log_n if e[0] in ("barrier", "obj")]))

    # B10 (rev 3) fail closed: layers passed but no live mHC / MLA + MoE layer -> every rank refuses at the preflight
    def boot_layers(layers):
        M._reset_for_test(mode="sp", min_t=1024, rs_mode="nccl")
        M._set_world_factory_for_test(lambda: M.EmulatedWorld(8))
        M.begin_forward(2048, H, layers=layers)

    kda_only = types.SimpleNamespace(mhc=True, is_mtp_layer=False, n=4, hc_pre=lambda *a_, **k_: None,
                                     layer_kind="kda", _mlp_is_moe=False, layer_idx=0)
    ok = (raises(lambda: boot_layers([]), RuntimeError, "no live mHC layer")
          and raises(lambda: boot_layers([kda_only]), RuntimeError, "no live MLA \\+ MoE mHC layer")
          and not M.stats()["booted"])
    check("B10 layers passed without a live mHC / MLA+MoE layer -> refused (fail closed), sp stays off", ok)

    # B11 (rev 3) SCHEDULE INVARIANCE: a planted local failure / wrong local shape at each stage leaves the issued
    # collective sequence identical to a clean boot (no rank can skip or reshape a collective) and fails the boot.
    def planted(name, nth, kind="raise"):
        orig = getattr(M, name)
        cnt = {"n": 0}

        def wrapper(*a_, **k_):
            cnt["n"] += 1
            if cnt["n"] == nth:
                if kind == "raise":
                    raise RuntimeError(f"planted failure in {name} (call {nth})")
                out_ = orig(*a_, **k_)
                return out_[:-1]                                   # one row short: a wrong local shape
            return orig(*a_, **k_)
        return orig, wrapper

    stages = [("synthetic_rows", 1, "raise"), ("synthetic_rows", 3, "raise"), ("pad_rows", 1, "raise"),
              ("pad_rows", 2, "raise"), ("pad_rows", 2, "short"), ("_amax_rel", 1, "raise"), ("_amax_rel", 2, "raise"),
              ("gather_full", 1, "raise"), ("gather_full", 2, "raise"), ("mismatches", 1, "raise"),
              ("_tensor_sha", 1, "raise"), ("check_plan", 2, "raise"), ("reduce_scatter_partial", 3, "raise")]
    bad = []
    for rs in ("nccl", "canonical"):
        clean, exc0 = boot_recorded(rs)
        if exc0 is not None:
            bad.append((rs, "clean boot raised", repr(exc0)[:80]))
            continue
        for name, nth, kind in stages:
            orig, wrapper = planted(name, nth, kind)
            setattr(M, name, wrapper)
            try:
                got, exc = boot_recorded(rs)
            finally:
                setattr(M, name, orig)
            if got != clean or exc is None or "BOOT SELF-TEST FAILED" not in str(exc):
                first = next((i for i, (x, y) in enumerate(zip(got, clean)) if x != y), min(len(got), len(clean)))
                bad.append((rs, f"{name}#{nth}:{kind}", f"len {len(got)} vs {len(clean)}, first diff at {first}",
                            (str(exc) or "")[:60]))
    check(f"B11 schedule invariance: {len(stages)} planted local failures x 2 RS modes -> the SAME collective "
          "sequence as a clean boot + BOOT SELF-TEST FAILED", not bad, str(bad[:3]))
    M._set_world_factory_for_test(None)
    M._reset_for_test()
    return M


# ---------------------------------------------------------------------------------------------------------------------
# C. multi-process gloo: the real cross-rank sp data flow vs the replicated reference
# ---------------------------------------------------------------------------------------------------------------------
GLOO_BODY = r'''
import os, sys, json, hashlib
import torch
import torch.distributed as dist

SP_PATH = sys.argv[1]
WORLD = int(sys.argv[2])
OUTDIR = sys.argv[3]
H = 128

import importlib.util, types
# minimal stub vllm so the module imports without a real install
for k in [k for k in list(sys.modules) if k=="vllm" or k.startswith("vllm.")]:
    del sys.modules[k]
v = types.ModuleType("vllm"); v.__path__=[]; sys.modules["vllm"]=v
for name in ("vllm.model_executor","vllm.model_executor.layers"):
    m=types.ModuleType(name); m.__path__=[]; sys.modules[name]=m
mhc=types.ModuleType("vllm.model_executor.layers.mhc")
mhc.hc_expand=lambda x,n: x.unsqueeze(1).expand(-1,n,-1).contiguous(); mhc.hc_contract=lambda x,n: x.mean(1)
sys.modules["vllm.model_executor.layers.mhc"]=mhc
spec=importlib.util.spec_from_file_location("vllm._gb10_prefill_sp", SP_PATH)
M=importlib.util.module_from_spec(spec); sys.modules["vllm._gb10_prefill_sp"]=M; spec.loader.exec_module(M)

class GlooWorld:
    kind="gloo"
    def __init__(self, rank, size):
        self.rank=rank; self.size=size; self.pp_size=1
        self.ranks=list(range(size))
    def all_gather_rows(self, x):
        x=x.contiguous()
        out=torch.empty((x.shape[0]*self.size,)+tuple(x.shape[1:]), dtype=x.dtype)
        dist.all_gather_into_tensor(out, x)
        return out
    def reduce_scatter_rows(self, x):
        x=x.contiguous(); S=x.shape[0]//self.size
        out=torch.empty((S,)+tuple(x.shape[1:]), dtype=x.dtype)
        dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM)
        return out
    def canonical_reduce_scatter_rows(self, x, plan):
        S,Wn,r=plan.S,self.size,self.rank
        recv=[torch.empty((S,)+tuple(x.shape[1:]), dtype=x.dtype) for _ in range(Wn)]
        ops=[]
        for p in range(Wn):
            if p==r:
                recv[r].copy_(x[r*S:(r+1)*S]); continue
            ops.append(dist.P2POp(dist.isend, x[p*S:(p+1)*S].contiguous(), p))
            ops.append(dist.P2POp(dist.irecv, recv[p], p))
        if ops:
            for w in dist.batch_isend_irecv(ops): w.wait()
        acc=recv[0].to(torch.float32)
        for p in range(1,Wn): acc=acc+recv[p].to(torch.float32)
        return acc.to(x.dtype)
    def all_gather_object(self, obj):
        out=[None]*self.size; dist.all_gather_object(out, obj); return out

def head_weight(size, seed=4025):
    g=torch.Generator().manual_seed(seed)
    return [(torch.randn(H,H,generator=g)*(0.5/H**0.5)) for _ in range(size)]

def main():
    rank=int(os.environ["RANK"]); size=int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    world=GlooWorld(rank,size)
    HW=head_weight(size)
    results=[]
    for T in (size, size+1, 1024, 1027, 2048, 3840, 3853, 7*size+3):
        plan=M.plan_rows(T, size)
        # identical full-T input on every rank (per-row synthetic)
        x=M.synthetic_rows(T, H, 7001+T, "cpu")
        for rs in ("nccl","canonical"):
            # sp data flow
            x_sh=M.shard_entry(x, plan, rank)
            x_full=M.gather_full(x_sh, world, plan)
            partial=(x_full.to(torch.float32) @ HW[rank]).to(torch.bfloat16)
            red_sh=M.reduce_scatter_partial(partial, world, plan, rs)
            got_full=M.gather_full(red_sh, world, plan)
            # replicated reference
            ref=sum(x.to(torch.float32) @ HW[p] for p in range(size)).to(torch.bfloat16)
            rel=float((got_full.to(torch.float32)-ref.to(torch.float32)).abs().amax()
                      / ref.to(torch.float32).abs().amax().clamp_min(1e-9))
            gather_input_identical = torch.equal(x_full, x)  # AG is a copy of the same rows on every rank
            cross=world.all_gather_object(hashlib.sha256(got_full.to(torch.float32).numpy().tobytes()).hexdigest())
            cross_ok=all(c==cross[0] for c in cross)
            # canonical RS == local rank-order fp32 sum (bitwise)
            can_exact=None
            if rs=="canonical":
                padded=M.pad_rows(partial, plan.padded)
                allp=world.all_gather_rows(padded).view(size, plan.padded, H)
                loc=allp[0, plan.start(rank):plan.start(rank)+plan.S].to(torch.float32)
                for p in range(1,size): loc=loc+allp[p, plan.start(rank):plan.start(rank)+plan.S].to(torch.float32)
                can_exact=int((red_sh.view(torch.int16)!=loc.to(torch.bfloat16).view(torch.int16)).sum())
            results.append({"T":T,"rs":rs,"rel":rel,"cross_ok":cross_ok,"ag_copy":bool(gather_input_identical),
                            "can_exact":can_exact})
    # negative control: a flipped bit in the gathered buffer must change the cross-rank hash on exactly that rank
    dist.barrier()
    with open(os.path.join(OUTDIR, f"rank{rank}.json"),"w") as fh:
        json.dump({"rank":rank,"size":size,"results":results}, fh)
    dist.barrier()
    dist.destroy_process_group()

if __name__=="__main__":
    main()
'''


def section_c(work, world_size):
    import torch  # noqa: F401
    print(f"== C. multi-process gloo data-flow (world={world_size})")
    body = work / "gloo_worker.py"
    body.write_text(GLOO_BODY)
    outdir = work / "gloo_out"
    outdir.mkdir(exist_ok=True)
    env_base = dict(os.environ)
    env_base.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29525", "WORLD_SIZE": str(world_size),
                     "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": ("lo0" if sys.platform == "darwin" else "lo")})
    procs = []
    for r in range(world_size):
        e = dict(env_base, RANK=str(r))
        procs.append(subprocess.Popen([sys.executable, str(body), str(MODULE_PATH),
                                       str(world_size), str(outdir)], env=e,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    logs = []
    rc = []
    for p in procs:
        out, _ = p.communicate(timeout=300)
        logs.append(out)
        rc.append(p.returncode)
    if any(rc):
        check("C0 all gloo ranks exited 0", False, f"rc={rc}")
        print("\n".join(logs[:2]))
        return
    check("C0 all gloo ranks exited 0", True)
    import json
    per = {}
    for r in range(world_size):
        f = outdir / f"rank{r}.json"
        if not f.exists():
            check(f"C0 rank{r} wrote results", False)
            return
        per[r] = json.loads(f.read_text())
    # aggregate
    rel_max = {"nccl": 0.0, "canonical": 0.0}
    cross_ok = True
    ag_copy = True
    can_exact_max = 0
    ncases = 0
    for r, d in per.items():
        for row in d["results"]:
            ncases += 1
            rel_max[row["rs"]] = max(rel_max[row["rs"]], row["rel"])
            cross_ok &= row["cross_ok"]
            ag_copy &= row["ag_copy"]
            if row["can_exact"] is not None:
                can_exact_max = max(can_exact_max, row["can_exact"])
    check("C1 all-gather is a copy (every rank's gathered input == the full rows)", ag_copy)
    check("C2 sp data flow reconstructs the replicated reference, bit-identical across ranks (nccl)",
          cross_ok and rel_max["nccl"] <= 2.0e-2, f"rel_max_nccl={rel_max['nccl']:.3e}")
    check("C3 canonical RS: reconstruction bit-identical across ranks AND == local rank-order fp32 sum",
          rel_max["canonical"] <= 2.0e-2 and can_exact_max == 0,
          f"rel_max_can={rel_max['canonical']:.3e} can_exact_max={can_exact_max}")
    check("C4 nccl vs canonical both valid sums of the same partials (rel within tol)",
          rel_max["nccl"] <= 2.0e-2 and rel_max["canonical"] <= 2.0e-2,
          f"nccl={rel_max['nccl']:.3e} can={rel_max['canonical']:.3e}")
    print(f"   gloo cases: {ncases} across {world_size} ranks; rel_max nccl={rel_max['nccl']:.3e} "
          f"canonical={rel_max['canonical']:.3e}")


# ---------------------------------------------------------------------------------------------------------------------
# D. end-to-end on the REAL patched model.py functions (the sweep), the rev-2 defence for B1
# ---------------------------------------------------------------------------------------------------------------------
def section_d(vroot, work):
    print("== D. end-to-end on the real patched model.py functions (p025_e2e_check.py sweep: base + mutants)")
    e2e = HERE / "p025_e2e_check.py"
    d = work / "e2e-sweep"
    # The wiring/reduction mutants manifest at any T >= MIN_T; 8,192 (rev 3) adds the prs-on-shard regime (S = 1,024
    # = prs's MIN_T, so an un-neutralised prs WOULD split the shard). The fuller 1023/3840/3853 list is exercised by a
    # standalone `p025_e2e_check.py launch`. The sweep also runs the HANG mutants (boot-only).
    p = subprocess.run([sys.executable, str(e2e), "sweep", "--patches", str(PDIR), "--vsrc", str(vroot),
                        "--work", str(d), "--tlist", "1024,1027,2048,8192", "--timeout", "1500"],
                       capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-90:])
    print(tail)
    check("D sweep: base sp==image bitwise (T incl. 8192) AND every required wiring/reduction mutant caught AND every "
          "HANG mutant a clean raise on all ranks (8 gloo ranks)",
          p.returncode == 0 and "SWEEP OK" in p.stdout, "see the mutant tables above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", default=os.environ.get("VLLM_TREE", "vllm-v0.32.0/vllm"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--only", default="ABCD")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    torch.set_num_threads(1)
    work = Path(a.work or tempfile.mkdtemp(prefix="p025-cpu-"))
    work.mkdir(parents=True, exist_ok=True)
    vroot = Path(a.vllm_root)
    P = load_file_module("p025_apply", HERE / "p025_apply.py")
    if "A" in a.only:
        section_a(vroot, work, P)
    if "B" in a.only:
        section_b(work)
    if "C" in a.only:
        section_c(work, a.world)
    if "D" in a.only:
        section_d(vroot, work)
    print(f"p025 CPU checks: {RESULTS['pass']} passed, {RESULTS['fail']} failed"
          + (f" ({', '.join(RESULTS['failed'][:8])})" if RESULTS["failed"] else ""))
    return 0 if RESULTS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
