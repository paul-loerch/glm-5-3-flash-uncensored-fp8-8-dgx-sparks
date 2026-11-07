#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch 003 gb10-blocking-sync] CPU checks (Mac / any CPU: python3 >= 3.10, stdlib only; NO torch, NO vLLM install).

  python3 -I -B files/p003_cpu_check.py [--vllm-root <v0.32.0 vllm package dir>]    (default: $VLLM_TREE, else vllm-v0.32.0/vllm)

  A  pins + applier on the REAL v0.32.0 v1/worker/gpu_worker.py: original == the .sh pin, apply() == the pinned result and
     adds exactly the 5 hook lines + marker, 3.12 grammar, revert() byte-exact, a second apply is refused, applier / module shas.
  B  refusals: anchor missing / duplicated / hooks moved out of Worker.init_device's cuda branch / marker present -> Refused.
  C  the REAL .sh end to end in a scratch VLLM_ROOT: banner, module installed, the [OK] patch 003 line, result == pinned
     sha, second run [skip], unknown base / tampered applier or module -> exit 1 with gpu_worker.py untouched.
  D  the REAL patched Worker.init_device cuda branch (extracted by AST, run over stubs): apply_before_context(visible id)
     runs BEFORE torch.accelerator.set_device_index and verify_after_context() AFTER it.
  E  the REAL module over a fake libcuda (ctypes.CDLL replaced): env unset = no CUDA call at all; env 1 = cuInit,
     cuDeviceGet, cuDevicePrimaryCtxSetFlags(BLOCKING_SYNC) and the verification (flags on the primary context / the live
     context; a context created before apply() gets cuCtxSetFlags; a driver without it, a failing call, a verify without
     apply -> RuntimeError: the worker refuses to run spinning).
Exit 0 = ALL PASS."""
from __future__ import annotations

import ast
import ctypes
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psmall_lib as L  # noqa: E402

SH = L.PDIR / "003-gb10-blocking-sync.sh"
APPLIER = HERE / "p003_apply.py"
MODULE = HERE / "gb10_blocking_sync.py"
REL = "v1/worker/gpu_worker.py"


class FakeCuda:
    """Just enough of libcuda for the module: byref()/c_* argument objects are filled in place."""

    SCHED = {"AUTO": 0, "SPIN": 1, "YIELD": 2, "BLOCKING": 4}

    def __init__(self, *, has_set_flags_v2=True, has_ctx_set_flags=True, fail=None, context_exists=False, ctx_sched=None):
        self.calls = []
        self.fail = fail or {}
        self.primary_flags = 0
        self.context_exists = context_exists
        self.ctx_flags = (ctx_sched if ctx_sched is not None else 1)
        self.hide = set()
        if not has_set_flags_v2:
            self.hide.add("cuDevicePrimaryCtxSetFlags_v2")
        if not has_ctx_set_flags:
            self.hide.add("cuCtxSetFlags")

    def __getattr__(self, name):
        if name.startswith("_") or name in self.hide or name not in (
                "cuInit", "cuDeviceGet", "cuDevicePrimaryCtxSetFlags_v2", "cuDevicePrimaryCtxSetFlags",
                "cuDevicePrimaryCtxGetState", "cuCtxGetFlags", "cuCtxSetFlags"):
            raise AttributeError(name)
        return lambda *a: self._call(name, *a)

    def _call(self, name, *a):
        self.calls.append(name)
        if name in self.fail:
            return self.fail[name]
        val = lambda x: getattr(x, "value", x)  # noqa: E731
        if name == "cuInit":
            return 0
        if name == "cuDeviceGet":
            a[0]._obj.value = val(a[1]) + 100       # the CUdevice handle differs from the ordinal on purpose
            self.dev_ordinal = val(a[1])
            return 0
        if name.startswith("cuDevicePrimaryCtxSetFlags"):
            self.primary_flags = val(a[1])
            if not self.context_exists:
                self.ctx_flags = val(a[1])           # the context, created later, will have these flags
            return 0
        if name == "cuDevicePrimaryCtxGetState":
            a[1]._obj.value = self.primary_flags
            a[2]._obj.value = 1
            return 0
        if name == "cuCtxGetFlags":
            a[0]._obj.value = self.ctx_flags
            return 0
        if name == "cuCtxSetFlags":
            self.ctx_flags = val(a[0])
            return 0
        raise AssertionError(name)


def load_module_with(fake: FakeCuda | None, env: dict):
    import os

    for k in ("VLLM_GB10_BLOCKING_SYNC",):
        os.environ.pop(k, None)
    os.environ.update(env)
    mod = L.load_module("vllm_gb10_blocking_sync_under_test", MODULE)
    loads = []

    def cdll(name):
        loads.append(name)
        if fake is None:
            raise AssertionError("libcuda must not be loaded")
        return fake

    mod.ctypes = types.SimpleNamespace(**{k: getattr(ctypes, k) for k in dir(ctypes) if not k.startswith("_")})
    mod.ctypes.CDLL = cdll
    return mod, loads


def main(argv) -> int:
    args = L.parse_args(argv)
    vroot = Path(args.vllm_root)
    ap = L.load_module("p003_apply", APPLIER)
    orig = (vroot / REL).read_text()
    check = L.check

    # ---------------------------------------------------------------- A
    check("A1 v0.32.0 gpu_worker.py == the WORKER_ORIG pin", L.sha_file(vroot / REL) == L.pin(SH, "WORKER_ORIG"))
    check("A2 applier == APPLIER_SHA, module == MOD_SHA", L.sha_file(APPLIER) == L.pin(SH, "APPLIER_SHA")
          and L.sha_file(MODULE) == L.pin(SH, "MOD_SHA"))
    new = ap.apply(orig)
    check("A3 apply() == the WORKER_P003 pin", L.sha_bytes(new.encode()) == L.pin(SH, "WORKER_P003"), L.sha_bytes(new.encode())[:16])
    ol, nl = orig.splitlines(), new.splitlines()
    added = [ln for ln in nl if ln not in ol or nl.count(ln) != ol.count(ln)]
    check("A4 exactly 5 lines added (marker, 2 imports, apply call, verify call)", len(nl) - len(ol) == 5
          and sum(1 for ln in added if "gb10_bs" in ln or "patch-003" in ln) == 5, str(len(nl) - len(ol)))
    check("A5 patched file compiles (also with the python 3.12 grammar)", L.compiles_312(new, REL))
    check("A6 revert(apply(x)) == x byte for byte", ap.revert(new) == orig)
    check("A7 a second apply() is refused", L.raises(lambda: ap.apply(new), ap.Refused))
    check("A8 the marker occurs exactly once", new.count(ap.MARKER) == 1)

    # ---------------------------------------------------------------- B
    anchor = ap.ANCHOR
    check("B1 anchor missing -> Refused", L.raises(lambda: ap.apply(orig.replace(anchor, "            torch.accelerator.set_device_index(dev)\n")),
                                                  ap.Refused))
    check("B2 anchor duplicated -> Refused", L.raises(lambda: ap.apply(orig + "\n#" + anchor), ap.Refused, "found 2x"))
    moved = orig.replace(ap.PREV, "            self.device = torch.device('cuda:0')\n")
    check("B3 self.device assignment altered -> Refused", L.raises(lambda: ap.apply(moved), ap.Refused))
    outside = orig.replace(anchor, "            pass\n").replace("    def init_device(self):", "    def init_device(self):\n        pass\n        pass", 1)
    check("B4 set_device_index removed from init_device -> Refused", L.raises(lambda: ap.apply(outside), ap.Refused))
    renamed = orig.replace("class Worker(", "class WorkerX(", 1)
    check("B5 class Worker renamed -> Refused (AST)", L.raises(lambda: ap.apply(renamed), ap.Refused))
    check("B6 marker already present -> Refused", L.raises(lambda: ap.apply(orig + "\n" + ap.MARKER + "\n"), ap.Refused, "marker"))
    broken = new.replace("_gb10_bs_verify()\n", "pass\n")
    check("B7 patched text without the verify call fails the structure check", L.raises(
        lambda: ap.check_structure(broken, patched=True), ap.Refused))
    check("B8 revert of an unpatched file is refused", L.raises(lambda: ap.revert(orig), ap.Refused))

    # ---------------------------------------------------------------- C
    roots = []

    def fresh(text=None):
        r = L.make_root(vroot, [REL])
        roots.append(r)
        if text is not None:
            (r / REL).write_text(text)
        return r

    root = fresh()
    r1 = L.run_sh(SH, root, {"VLLM_GB10_BLOCKING_SYNC": "1"})
    out = r1.stdout
    check("C1 .sh exit 0", r1.returncode == 0, r1.stderr[-300:])
    check("C2 banner, module install line, wiring line, verification, final [OK] line (gate shown)",
          "=== Patch 003: GB10 CUDA blocking-sync hook (opt-in VLLM_GB10_BLOCKING_SYNC=1) ===" in out
          and " [OK] installed _gb10_blocking_sync.py (sha256 " in out
          and " [OK] wired apply/verify around set_device_index in Worker.init_device" in out
          and "=== Verification ===" in out
          and "[OK] patch 003 in place (active only when VLLM_GB10_BLOCKING_SYNC=1; now: 1)" in out, out[-400:])
    check("C3 gpu_worker.py == the pinned result, module installed byte-identical",
          L.sha_file(root / REL) == L.pin(SH, "WORKER_P003") and L.sha_file(root / "_gb10_blocking_sync.py") == L.pin(SH, "MOD_SHA"))
    before = (root / REL).read_bytes()
    r2 = L.run_sh(SH, root)
    check("C4 second run (gate unset): [skip], exit 0, file untouched, final line says unset",
          r2.returncode == 0 and " [skip] gpu_worker.py already wired" in r2.stdout and (root / REL).read_bytes() == before
          and "now: unset)" in r2.stdout)
    root2 = fresh(orig + "\n# local change\n")
    b2 = (root2 / REL).read_bytes()
    r3 = L.run_sh(SH, root2)
    check("C5 unknown base -> exit 1, file untouched, module not installed", r3.returncode == 1 and (root2 / REL).read_bytes() == b2
          and not (root2 / "_gb10_blocking_sync.py").exists(), r3.stderr[-200:])
    root3 = fresh(new + "\n# edited after patching\n")
    b3 = (root3 / REL).read_bytes()
    r4 = L.run_sh(SH, root3)
    check("C6 stale patch-003 edit (marker present, unpinned sha) -> exit 1 'recreate the container'",
          r4.returncode == 1 and (root3 / REL).read_bytes() == b3 and "recreate the container" in r4.stderr)
    for what, fn in (("applier", "p003_apply.py"), ("module", "gb10_blocking_sync.py")):
        pdir = Path(tempfile.mkdtemp(prefix="smallchk_p_"))
        roots.append(pdir)
        shutil.copy2(SH, pdir / SH.name)
        (pdir / "files").mkdir()
        for f in ("p003_apply.py", "gb10_blocking_sync.py"):
            shutil.copy2(HERE / f, pdir / "files" / f)
        (pdir / "files" / fn).write_text((HERE / fn).read_text() + "\n# tampered\n")
        rr = fresh()
        bb = (rr / REL).read_bytes()
        r5 = L.run_sh(pdir / SH.name, rr)
        check(f"C7 tampered {what} -> exit 1 before any write", r5.returncode == 1 and (rr / REL).read_bytes() == bb
              and not (rr / "_gb10_blocking_sync.py").exists(), r5.stderr[-160:])
    r6 = L.run_sh(SH, Path(tempfile.mkdtemp(prefix="smallchk_")))
    check("C8 missing gpu_worker.py -> [FAIL] ... not found, exit 1", r6.returncode == 1 and "not found" in r6.stderr)

    # ---------------------------------------------------------------- D
    tree = ast.parse(new)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "init_device")
    blk = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and any(
        isinstance(s, ast.Expr) and ast.unparse(s.value).startswith("torch.accelerator.set_device_index") for s in n.body))
    i0 = next(i for i, s in enumerate(blk.body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "visible_device_index")
    i1 = next(i for i, s in enumerate(blk.body) if ast.unparse(s) == "_gb10_bs_verify()")
    src = "\n".join(ast.unparse(s) for s in blk.body[i0:i1 + 1])
    log = []
    stub_mod = types.ModuleType("vllm._gb10_blocking_sync")
    stub_mod.apply_before_context = lambda o: log.append(("apply", o))
    stub_mod.verify_after_context = lambda: log.append(("verify",))
    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm._gb10_blocking_sync")}
    sys.modules["vllm"], sys.modules["vllm._gb10_blocking_sync"] = vllm_pkg, stub_mod
    vllm_pkg._gb10_blocking_sync = stub_mod
    try:
        ns = {"current_platform": types.SimpleNamespace(logical_device_id_to_visible_device_id=lambda r: 100 + r),
              "torch": types.SimpleNamespace(device=lambda s: ("dev", s), accelerator=types.SimpleNamespace(
                  set_device_index=lambda d: log.append(("set_device_index", d))))}
        self_ = types.SimpleNamespace(local_rank=3)
        exec(compile("def f(self):\n" + "\n".join("    " + ln for ln in src.splitlines()) + "\n", "init_device(cuda branch)", "exec"), ns)
        ns["f"](self_)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    check("D1 the REAL patched cuda branch: apply(visible id) -> set_device_index -> verify, in this order",
          log == [("apply", 103), ("set_device_index", ("dev", "cuda:103")), ("verify",)], str(log))

    # ---------------------------------------------------------------- E
    m, loads = load_module_with(None, {})
    m.apply_before_context(0)
    m.verify_after_context()
    check("E1 gate unset: apply/verify are no-ops (libcuda never loaded)", loads == [])
    m, loads = load_module_with(None, {"VLLM_GB10_BLOCKING_SYNC": "0"})
    m.apply_before_context(0)
    m.verify_after_context()
    check("E2 gate 0: no-ops", loads == [])
    fake = FakeCuda()
    m, loads = load_module_with(fake, {"VLLM_GB10_BLOCKING_SYNC": "1"})
    m.apply_before_context(5)
    check("E3 gate 1: cuInit, cuDeviceGet(ordinal 5), primary-context flags = BLOCKING_SYNC (0x04), before the context exists",
          fake.calls == ["cuInit", "cuDeviceGet", "cuDevicePrimaryCtxSetFlags_v2"] and fake.dev_ordinal == 5
          and fake.primary_flags == 4, str(fake.calls))
    m.verify_after_context()
    check("E4 verify passes when the context was created with BLOCKING_SYNC", fake.calls[-2:] == ["cuDevicePrimaryCtxGetState", "cuCtxGetFlags"])
    fake = FakeCuda(context_exists=True, ctx_sched=1)
    m, _ = load_module_with(fake, {"VLLM_GB10_BLOCKING_SYNC": "1"})
    m.apply_before_context(0)
    m.verify_after_context()
    check("E5 a context that existed before apply() (SPIN) is repaired with cuCtxSetFlags and then verified",
          "cuCtxSetFlags" in fake.calls and fake.ctx_flags == 4)
    fake = FakeCuda(context_exists=True, ctx_sched=1, has_ctx_set_flags=False)
    m, _ = load_module_with(fake, {"VLLM_GB10_BLOCKING_SYNC": "1"})
    m.apply_before_context(0)
    check("E6 ... but without cuCtxSetFlags the worker refuses to run spinning (RuntimeError)",
          L.raises(m.verify_after_context, RuntimeError, "refusing to run spinning"))
    fake = FakeCuda(has_set_flags_v2=False)
    m, _ = load_module_with(fake, {"VLLM_GB10_BLOCKING_SYNC": "1"})
    m.apply_before_context(0)
    check("E7 the unversioned cuDevicePrimaryCtxSetFlags is used when the _v2 symbol is absent",
          "cuDevicePrimaryCtxSetFlags" in fake.calls and fake.primary_flags == 4)
    fake = FakeCuda(fail={"cuInit": 3})
    m, _ = load_module_with(fake, {"VLLM_GB10_BLOCKING_SYNC": "1"})
    check("E8 a failing driver call raises 'cuInit failed (CUresult 3)'", L.raises(lambda: m.apply_before_context(0), RuntimeError,
                                                                                  r"cuInit failed \(CUresult 3\)"))
    m, _ = load_module_with(FakeCuda(), {"VLLM_GB10_BLOCKING_SYNC": "1"})
    check("E9 verify without apply -> RuntimeError (patch wiring broken)", L.raises(m.verify_after_context, RuntimeError, "without apply"))
    fake = FakeCuda()
    m, _ = load_module_with(fake, {"VLLM_GB10_BLOCKING_SYNC": "1"})
    m.apply_before_context(0)
    fake.ctx_flags = 1                     # the driver ignored the primary-context flags: the live context spins
    fake.hide.add("cuCtxSetFlags")
    check("E10 a context that came up spinning although the flags were set -> refuses", L.raises(m.verify_after_context, RuntimeError))
    m, _ = load_module_with(FakeCuda(), {"VLLM_GB10_BLOCKING_SYNC": " 1 "})
    check("E11 the gate is exactly '1' after strip (' 1 ' on)", m.enabled())
    m, _ = load_module_with(FakeCuda(), {"VLLM_GB10_BLOCKING_SYNC": "true"})
    check("E12 'true' is not '1' (off)", not m.enabled())

    for d in roots:
        shutil.rmtree(d, ignore_errors=True)
    return L.finish("p003_cpu_check")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
