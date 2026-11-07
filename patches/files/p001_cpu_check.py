#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch 001 gb10-spin-wait] CPU checks (Mac / any CPU: python3 >= 3.10, stdlib only; NO torch, NO vLLM install).

  python3 -I -B files/p001_cpu_check.py [--vllm-root <v0.32.0 vllm package dir>]    (default: $VLLM_TREE, else vllm-v0.32.0/vllm)

  A  pins + applier on the REAL v0.32.0 shm_broadcast.py: original == the .sh pin, apply() == the pinned result and differs
     in exactly one line, 3.12 grammar, revert() byte-exact, idempotence (a second apply is refused), vendored applier sha.
  B  refusals of the applier: anchor missing / duplicated / in the wrong class / default not a constant -> Refused.
  C  the REAL .sh end to end in a scratch VLLM_ROOT: banner + [OK] lines (incl. the launch/readiness.txt require rule
     `busy_loop_s = 0\\.002`), result == pinned sha, second run = [skip] with the file untouched, unknown base / missing file
     / tampered applier -> non-zero exit with the file untouched.
  D  behaviour: the REAL patched SpinCondition (extracted by AST, run over stubs for zmq/time): a reader idle for > 2 ms goes to
     the poller (sleeps) instead of yielding; the unpatched class still yields for 1 s (the control).
Exit 0 = ALL PASS."""
from __future__ import annotations

import ast
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psmall_lib as L  # noqa: E402

SH = L.PDIR / "001-gb10-spin-wait.sh"
APPLIER = HERE / "p001_apply.py"
REL = "distributed/device_communicators/shm_broadcast.py"


def main(argv) -> int:
    args = L.parse_args(argv)
    vroot = Path(args.vllm_root)
    ap = L.load_module("p001_apply", APPLIER)
    orig_path = vroot / REL
    orig = orig_path.read_text()
    check = L.check

    # ---------------------------------------------------------------- A
    check("A1 v0.32.0 shm_broadcast.py == the SHM_ORIG pin", L.sha_file(orig_path) == L.pin(SH, "SHM_ORIG"))
    check("A2 vendored applier == the APPLIER_SHA pin", L.sha_file(APPLIER) == L.pin(SH, "APPLIER_SHA"))
    new = ap.apply(orig)
    check("A3 apply() == the SHM_P001 pin", L.sha_bytes(new.encode()) == L.pin(SH, "SHM_P001"), L.sha_bytes(new.encode())[:16])
    ol, nl = orig.splitlines(keepends=True), new.splitlines(keepends=True)
    diff = [(a, b) for a, b in zip(ol, nl) if a != b]
    check("A4 exactly one line differs: busy_loop_s: float = 1, -> 0.002,",
          len(ol) == len(nl) and diff == [("        busy_loop_s: float = 1,\n", "        busy_loop_s: float = 0.002,\n")], str(diff))
    check("A5 patched file compiles (also with the python 3.12 grammar)", L.compiles_312(new, REL))
    check("A6 revert(apply(x)) == x byte for byte", ap.revert(new) == orig)
    check("A7 a second apply() is refused (idempotence at the applier level)", L.raises(lambda: ap.apply(new), ap.Refused))
    check("A8 the module-level strings the old inline script used are gone from the patched text only where intended",
          new.count("busy_loop_s: float = 0.002") == 1 and "busy_loop_s: float = 1," not in new)

    # ---------------------------------------------------------------- B
    a = ap.ANCHOR_OLD
    check("B1 anchor missing -> Refused", L.raises(lambda: ap.apply(orig.replace(a, "        busy_loop_s: float = 2,\n")), ap.Refused, "found 0x"))
    check("B2 anchor duplicated -> Refused", L.raises(lambda: ap.apply(orig + "\n# " + a), ap.Refused, "found 2x"))
    check("B3 class renamed -> Refused (AST)", L.raises(lambda: ap.apply(orig.replace("class SpinCondition:", "class SpinConditionX:")),
                                                       ap.Refused, "class SpinCondition found 0"))
    check("B4 default not a constant -> Refused", L.raises(lambda: ap.apply(orig.replace(a, "        busy_loop_s: float = float(1),\n")),
                                                          ap.Refused))
    check("B5 consumer assignment gone -> Refused", L.raises(
        lambda: ap.apply(orig.replace("            self.busy_loop_s = busy_loop_s\n", "            self.busy_loop_s = 1\n")), ap.Refused))
    check("B6 revert of an unpatched file is refused", L.raises(lambda: ap.revert(orig), ap.Refused))

    # ---------------------------------------------------------------- C
    root = L.make_root(vroot, [REL])
    f = root / REL
    r1 = L.run_sh(SH, root)
    out = r1.stdout
    check("C1 .sh exit 0 on the v0.32.0 original", r1.returncode == 0, r1.stderr[-300:])
    check("C2 banner / [OK] / verification lines as in the day-0 port",
          "=== Hotfix: GB10 vLLM IPC spin-wait (CPU load / SoC heat, issue #79) ===" in out
          and " [OK] busy_loop_s 1 -> 0.002 (2ms grace before sleep)" in out
          and "=== Verification ===" in out
          and "[OK] busy_loop_s = 0.002 — IPC wait sleeps after 2ms instead of spinning 1s." in out, out[-300:])
    ready = re.compile(r"busy_loop_s = 0\.002")
    check("C3 the launch/readiness.txt rule `require busy_loop_s = 0\\.002` matches", bool(ready.search(out)))
    check("C4 file == the pinned result", L.sha_file(f) == L.pin(SH, "SHM_P001"))
    before = f.read_bytes()
    r2 = L.run_sh(SH, root)
    check("C5 second run: [skip], exit 0, file untouched", r2.returncode == 0 and " [skip] busy_loop_s already set to 0.002" in r2.stdout
          and f.read_bytes() == before, r2.stdout[-200:])
    check("C6 the second run still prints the require-rule line", bool(ready.search(r2.stdout)))
    # unknown base
    root2 = L.make_root(vroot, [REL])
    f2 = root2 / REL
    f2.write_text(orig + "\n# local change\n")
    b2 = f2.read_bytes()
    r3 = L.run_sh(SH, root2)
    check("C7 unknown base -> exit 1, file untouched, no [OK] verification line",
          r3.returncode == 1 and f2.read_bytes() == b2 and "[OK] busy_loop_s = 0.002" not in r3.stdout, r3.stderr[-200:])
    # already carries the value but under another sha (e.g. a different build): refused, not skipped
    f2.write_text(orig.replace(a, ap.ANCHOR_NEW) + "\n# other build\n")
    b2 = f2.read_bytes()
    r4 = L.run_sh(SH, root2)
    check("C8 0.002 under an unpinned sha -> exit 1 (fail closed, not [skip])", r4.returncode == 1 and f2.read_bytes() == b2)
    # missing file
    root3 = Path(tempfile.mkdtemp(prefix="smallchk_"))
    r5 = L.run_sh(SH, root3)
    check("C9 missing shm_broadcast.py -> ERROR line, exit 1", r5.returncode == 1 and "ERROR: vLLM shm_broadcast.py not found" in r5.stderr)
    # tampered applier
    pdir = Path(tempfile.mkdtemp(prefix="smallchk_p_"))
    shutil.copy2(SH, pdir / SH.name)
    (pdir / "files").mkdir()
    (pdir / "files" / "p001_apply.py").write_text(APPLIER.read_text() + "\n# tampered\n")
    root4 = L.make_root(vroot, [REL])
    b4 = (root4 / REL).read_bytes()
    r6 = L.run_sh(pdir / SH.name, root4)
    check("C10 tampered applier -> exit 1 before any write", r6.returncode == 1 and (root4 / REL).read_bytes() == b4
          and "APPLIER" not in r6.stdout and "sha256" in r6.stderr, r6.stderr[-200:])

    # ---------------------------------------------------------------- D
    def spin_class(text: str, label: str):
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpinCondition")
        src = ast.get_source_segment(text, cls)
        clock = {"t": 1000.0}

        class FakeSocket:
            def __init__(self, kind):
                self.kind, self.opts, self.sent, self.recv_calls = kind, {}, [], 0

            def setsockopt(self, k, v): self.opts[k] = v
            def setsockopt_string(self, k, v): self.opts[k] = v
            def connect(self, a): self.addr = a
            def bind(self, a): self.addr = a
            def send(self, b): self.sent.append(b)

            def recv(self, flags=0, copy=True):
                self.recv_calls += 1

        class FakePoller:
            def __init__(self): self.polled, self.reg, self.ret = [], [], {}
            def register(self, s, ev): self.reg.append(s)
            def poll(self, timeout=None):
                self.polled.append(timeout)
                return dict(self.ret)

        zmq = types.SimpleNamespace(CONFLATE=1, SNDHWM=2, NOBLOCK=3, PAIR=4, Poller=FakePoller, POLLIN=5)
        ctx = types.SimpleNamespace(socket=lambda kind: FakeSocket(kind))
        yields = []
        ns = {"zmq": zmq, "SUB": "SUB", "PUB": "PUB", "SUBSCRIBE": "SUBSCRIBE", "time": types.SimpleNamespace(monotonic=lambda: clock["t"]),
              "sched_yield": lambda: yields.append(1), "get_open_zmq_inproc_path": lambda: "inproc://x",
              "logger": types.SimpleNamespace(debug=lambda *a, **k: None)}
        exec(compile(src, label, "exec"), ns)
        return ns["SpinCondition"], ctx, clock, yields

    for label, text, expect_poll in (("patched", new, True), ("unpatched control", orig, False)):
        cls, ctx, clock, yields = spin_class(text, label)
        sc = cls(True, ctx, "ipc://notify")
        check(f"D1 {label}: reader busy_loop_s", sc.busy_loop_s == (0.002 if expect_poll else 1), str(sc.busy_loop_s))
        clock["t"] += 0.010                       # idle for 10 ms since the last read
        sc.wait(timeout_ms=5)
        polled = bool(sc.poller.polled)
        check(f"D2 {label}: a reader idle for 10 ms {'sleeps in the poller' if expect_poll else 'still yields (spins)'}",
              polled == expect_poll and (len(yields) == (0 if expect_poll else 1)), f"polled={sc.poller.polled} yields={len(yields)}")
        sc.record_read()
        clock["t"] += 0.001                       # 1 ms since the last read: still inside the 2 ms grace
        yields.clear()
        sc.wait(timeout_ms=5)
        check(f"D3 {label}: a reader idle for 1 ms yields (inside the grace)", len(yields) == 1)
        w = cls(False, ctx, "ipc://notify")
        check(f"D4 {label}: the writer side is unaffected (busy_loop_s 0, publishes)", w.busy_loop_s == 0 and w.poller is None)

    for d in (root, root2, root3, root4, pdir):
        shutil.rmtree(d, ignore_errors=True)
    return L.finish("p001_cpu_check")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
