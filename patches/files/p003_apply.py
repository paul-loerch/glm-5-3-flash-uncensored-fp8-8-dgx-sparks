#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 003 applier (vLLM v0.32.0 port): one anchored edit (two inserted call sites) in vllm/v1/worker/gpu_worker.py.

  apply(text) -> text     v0.32.0 original -> + patch 003; fail-closed: the anchor pair (the `self.device = torch.device(...)`
                          statement directly followed by `torch.accelerator.set_device_index(self.device)`) must match exactly
                          once, the inserted calls must land inside Worker.init_device in the `if cuda` branch right before /
                          right after the set_device_index statement (AST-checked), the result must compile
  revert(text) -> text    exact inverse (the tests rebuild the base from the patched file)

CLI (used by 003-gb10-blocking-sync.sh):
  p003_apply.py apply  <gpu_worker.py>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p003_apply.py check  <gpu_worker.py>   dry run: print the sha256 the file would have after apply()
  p003_apply.py revert <gpu_worker.py>   in-place inverse (tests / manual rollback)

The edit (v0.32.0 lines 500-501, Worker.init_device, `if self.device_config.device_type == "cuda":` branch):
    self.device = torch.device(f"cuda:{visible_device_index}")
    # [patch-003 gb10-blocking-sync] ...                              <- marker (1)
    from vllm._gb10_blocking_sync import apply_before_context as _gb10_bs_apply
    from vllm._gb10_blocking_sync import verify_after_context as _gb10_bs_verify
    _gb10_bs_apply(visible_device_index)
    torch.accelerator.set_device_index(self.device)                  <- creates the primary context
    _gb10_bs_verify()
Both calls are no-ops unless VLLM_GB10_BLOCKING_SYNC=1 (module-side gate). Identical text to the day-0 image port.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-003 gb10-blocking-sync]"
N_MARKERS = 1
CLASS, FUNC = "Worker", "init_device"

PREV = "            self.device = torch.device(f\"cuda:{visible_device_index}\")\n"
ANCHOR = "            torch.accelerator.set_device_index(self.device)\n"
NEW = (
    f"            {MARKER} opt-in: VLLM_GB10_BLOCKING_SYNC=1 -> CUDA context sleeps instead of spinning\n"
    "            from vllm._gb10_blocking_sync import apply_before_context as _gb10_bs_apply\n"
    "            from vllm._gb10_blocking_sync import verify_after_context as _gb10_bs_verify\n"
    "            _gb10_bs_apply(visible_device_index)\n"
    + ANCHOR +
    "            _gb10_bs_verify()\n"
)
EDITS = [("set_device_index anchor (Worker.init_device)", PREV + ANCHOR, PREV + NEW)]


class Refused(ValueError):
    pass


def _init_device(tree: ast.Module):
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS]
    if len(cls) != 1:
        raise Refused(f"class {CLASS} found {len(cls)} times (expected 1)")
    fns = [n for n in cls[0].body if isinstance(n, ast.FunctionDef) and n.name == FUNC]
    if len(fns) != 1:
        raise Refused(f"{CLASS}.{FUNC} found {len(fns)} times (expected 1)")
    return fns[0]


def _is_call(stmt, dotted: str) -> bool:
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    return ast.unparse(stmt.value.func) == dotted


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    fn = _init_device(tree)
    holders = [n for n in ast.walk(fn) if isinstance(n, ast.If)
               and any(_is_call(s, "torch.accelerator.set_device_index") for s in n.body)]
    if len(holders) != 1:
        raise Refused(f"torch.accelerator.set_device_index(...) statement found in {len(holders)} `if` blocks of "
                      f"{CLASS}.{FUNC} (expected 1)")
    if "cuda" not in ast.unparse(holders[0].test):
        raise Refused("the set_device_index block is not the cuda branch")
    body = holders[0].body
    k = next(i for i, s in enumerate(body) if _is_call(s, "torch.accelerator.set_device_index"))

    def _is_device_assign(s) -> bool:
        return isinstance(s, ast.Assign) and len(s.targets) == 1 and ast.unparse(s.targets[0]) == "self.device"

    # unpatched: `self.device = ...` directly before set_device_index; patched: the same assignment 4 statements earlier
    # (two imports and the apply call sit in between)
    if not _is_device_assign(body[k - (4 if patched else 1)] if k >= (4 if patched else 1) else None):
        raise Refused("the self.device assignment does not precede set_device_index (directly / before the hook block)")
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "vllm._gb10_blocking_sync"]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("_gb10_bs_apply", "_gb10_bs_verify")]
    if not patched:
        if imports or calls:
            raise Refused("base already mentions _gb10_blocking_sync")
        return
    if len(imports) != 2 or len(calls) != 2:
        raise Refused(f"expected 2 imports + 2 calls, found {len(imports)} + {len(calls)}")
    if not (k >= 4 and ast.unparse(body[k - 1]) == "_gb10_bs_apply(visible_device_index)"
            and isinstance(body[k - 2], ast.ImportFrom) and isinstance(body[k - 3], ast.ImportFrom)
            and k + 1 < len(body) and ast.unparse(body[k + 1]) == "_gb10_bs_verify()"):
        raise Refused("the hook calls are not directly around torch.accelerator.set_device_index(self.device)")
    if not any(isinstance(n, ast.Name) and n.id == "visible_device_index" for n in ast.walk(body[k - 1])):
        raise Refused("apply hook does not receive visible_device_index")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-003 marker")
    bad = [(name, text.count(old)) for name, old, _ in EDITS if text.count(old) != 1]
    if bad:
        raise Refused("; ".join(f"anchor '{n}' found {c}x (expected 1)" for n, c in bad))
    if text.count(ANCHOR) != 1:
        raise Refused(f"set_device_index anchor found {text.count(ANCHOR)}x (expected 1)")
    check_structure(text, patched=False)
    out = text
    for name, old, new in EDITS:
        if out.count(old) != 1:
            raise Refused(f"anchor '{name}' changed by an earlier edit")
        out = out.replace(old, new, 1)
    if out.count(MARKER) != N_MARKERS:
        raise Refused(f"expected {N_MARKERS} marker after the edit, found {out.count(MARKER)}")
    compile(out, "gpu_worker.py", "exec")
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
    compile(out, "gpu_worker.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p003_apply.py {apply|check|revert} <gpu_worker.py>", file=sys.stderr)
        return 2
    p = Path(argv[2])
    try:
        src = p.read_text(encoding="utf-8")
        new = revert(src) if argv[1] == "revert" else apply(src)
    except (Refused, SyntaxError, OSError, UnicodeDecodeError) as e:
        print(f"[FAIL] {p}: {e}", file=sys.stderr)
        return 1
    if argv[1] == "check":
        print(_sha(new.encode()))
        return 0
    tmp = p.with_name(p.name + ".p003.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    print(" [OK] wired apply/verify around set_device_index in Worker.init_device" if argv[1] == "apply"
          else " [OK] reverted the set_device_index hooks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
