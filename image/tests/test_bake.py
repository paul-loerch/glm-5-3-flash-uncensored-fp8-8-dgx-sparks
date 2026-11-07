#!/usr/bin/env python3
"""Mac-side tests for glm53_sm121_v032.py and verify_bake.py (python3 + stdlib; numpy optional; clang++ optional).

What is tested, against COPIES of the pristine sources (the originals are only read):
  * vLLM v0.32.0 tree  : the `vllm/` directory of git tag v0.32.0 (env BAKE_TEST_VLLM_TREE; checked against the pins)
  * FlashInfer 0.7.0.post1 : files extracted from the FlashInfer wheel (env BAKE_TEST_FI_WHEEL; PyPI, sha256 pinned below)

  python3 -I tests/test_bake.py            # all tests
  python3 -I tests/test_bake.py -v         # verbose
  BAKE_TEST_FULL=1 python3 -I tests/test_bake.py   # also bakes a copy of the WHOLE vLLM tree (48 MB) and writes ../PATCH.diff

No GPU, no torch.  What no Mac test can show (CUDA compile of the patched header, FA2 FP8 kernel on sm_121a,
KDA/mHC races without PDL) is listed in ../IMAGE.md.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent
BAKE = IMAGE / "glm53_sm121_v032.py"
VERIFY = IMAGE / "verify_bake.py"
MOE_SRC = IMAGE / "moe"
WORK = IMAGE / "work"
# the pristine vLLM v0.32.0 `vllm/` directory (git tag content); override with BAKE_TEST_VLLM_TREE
# Inputs (see IMAGE.md, "Testing the bake"): the `vllm/` directory of a v0.32.0 checkout and the pinned FlashInfer wheel
VLLM_TREE = Path(os.environ.get("BAKE_TEST_VLLM_TREE", "vllm-v0.32.0/vllm"))
FI_WHEEL = Path(os.environ.get("BAKE_TEST_FI_WHEEL", str(IMAGE / "ref/flashinfer_python-0.7.0.post1-py3-none-any.whl")))
FI_WHEEL_SHA256 = "c7adf826568d61fc1b7d3aadd4cae387a35a138bfc08e2deca2f34ecfa280716"  # == PyPI digest

VLLM_FILES = [
    "platforms/cuda.py",
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py",
    "models/glm5next/nvidia/sparse_indexer.py",
    "models/glm5next/nvidia/ops/kpool_compress.py",
    "model_executor/layers/fused_moe/fused_moe.py",
]
FI_FILES = [
    "mla/_batch_mla/_backends/_fa_common.py",
    "data/include/flashinfer/attention/mla.cuh",
    "jit/attention/modules.py",
    "_build_meta.py",
]


def load_bake():
    spec = importlib.util.spec_from_file_location("glm53_sm121_v032", BAKE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["glm53_sm121_v032"] = mod
    spec.loader.exec_module(mod)
    return mod


BK = load_bake()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd, **kw):
    return subprocess.run([sys.executable, "-I", *map(str, cmd)], capture_output=True, text=True, **kw)


class Fixture:
    """A throw-away pair of package roots (subset copies) plus helpers."""

    pristine_fi: Path | None = None

    @classmethod
    def fi_pristine(cls) -> Path:
        """FlashInfer files extracted once from the wheel into work/fixture-pristine/flashinfer."""
        if cls.pristine_fi is None:
            assert FI_WHEEL.is_file(), f"missing {FI_WHEEL} (download it from PyPI, see ../IMAGE.md)"
            assert sha(FI_WHEEL) == FI_WHEEL_SHA256, "wheel sha256 differs from the PyPI digest"
            dest = WORK / "fixture-pristine"
            if dest.exists():
                shutil.rmtree(dest)
            with zipfile.ZipFile(FI_WHEEL) as z:
                for rel in FI_FILES:
                    z.extract("flashinfer/" + rel, dest)
            cls.pristine_fi = dest / "flashinfer"
        return cls.pristine_fi

    def __init__(self):
        WORK.mkdir(exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="t-", dir=WORK))
        self.vllm = self.tmp / "vllm"
        self.fi = self.tmp / "flashinfer"
        for rel in VLLM_FILES:
            (self.vllm / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(VLLM_TREE / rel, self.vllm / rel)
        (self.vllm / "model_executor/layers/fused_moe/configs").mkdir(parents=True, exist_ok=True)
        for rel in FI_FILES:
            (self.fi / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.fi_pristine() / rel, self.fi / rel)
        # packages whose __init__ must never be executed by the bake (it uses the importlib finder only)
        for root in (self.vllm, self.fi):
            (root / "__init__.py").write_text("raise RuntimeError('the bake must not import this package')\n")
        self.manifest = self.tmp / "manifest.json"

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, pkg, rel) -> Path:
        return (self.vllm if pkg == "vllm" else self.fi) / rel

    def args(self, *extra):
        return [BAKE, "--vllm-root", self.vllm, "--flashinfer-root", self.fi, "--moe-src", MOE_SRC,
                "--manifest", self.manifest, *extra]

    def bake(self, *extra):
        return run(self.args(*extra))

    def verify(self, *extra):
        return run([VERIFY, "--vllm-root", self.vllm, "--flashinfer-root", self.fi, "--manifest", self.manifest, "--static", *extra])

    def snapshot(self) -> dict:
        snap = {}
        for root in (self.vllm, self.fi):
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.name != "__init__.py":
                    snap[str(p.relative_to(self.tmp))] = sha(p)
        return snap


class BakeTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.cleanup)

    # ---------------------------------------------------------------- sanity of the fixture itself
    def test_00_pristine_files_match_pins(self):
        for (pkg, rel), pin in BK.PINS.items():
            self.assertEqual(sha(self.fx.path(pkg, rel)), pin, f"{pkg}/{rel} is not the pinned pristine file")
        self.assertEqual(sha(MOE_SRC / BK.MOE_NAME), BK.MOE_SHA256)

    def test_01_every_anchor_matches_exactly_once_in_pristine(self):
        for e in BK.EDITS:
            text = self.fx.path(e.pkg, e.rel).read_text()
            self.assertEqual(text.count(e.old), 1, f"{e.id}: old anchor count")
            self.assertEqual(text.count(e.new), 0, f"{e.id}: new text must not be there yet")

    # ---------------------------------------------------------------- clean apply
    def test_10_clean_apply_compiles_and_verifies(self):
        before = self.fx.snapshot()
        r = self.fx.bake()
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("DONE: applied=10 already=0", r.stdout)
        after = self.fx.snapshot()
        changed = sorted(k for k in after if before.get(k) != after[k])
        self.assertEqual(len(changed), 8, changed)  # 7 edited files + the MoE table
        # every edited python file compiles
        for e in BK.EDITS:
            p = self.fx.path(e.pkg, e.rel)
            self.assertEqual(p.read_text().count(e.new), 1)
            self.assertEqual(p.read_text().count(e.old), 0)
            if e.rel.endswith(".py"):
                compile(p.read_text(), str(p), "exec")
        # the diff is printed
        self.assertIn("--- a/vllm/platforms/cuda.py", r.stdout)
        self.assertIn("+        return major >= 9 and major != 12", r.stdout)
        # manifest
        m = json.loads(self.fx.manifest.read_text())
        self.assertEqual(len(m["edits"]), 10)
        self.assertEqual(m["skipped_groups"], [])
        self.assertEqual(m["versions"]["flashinfer"], "0.7.0.post1")
        # moe table installed with the pinned content and mode
        moe = self.fx.vllm / BK.MOE_DEST_REL
        self.assertEqual(sha(moe), BK.MOE_SHA256)
        self.assertEqual(moe.stat().st_mode & 0o777, 0o644)
        v = self.fx.verify()
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertIn("VERIFY OK", v.stdout)
        self.assertNotIn("FAIL", v.stdout)

    def test_11_idempotent_rerun(self):
        self.assertEqual(self.fx.bake().returncode, 0)
        snap = self.fx.snapshot()
        r = self.fx.bake()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("DONE: applied=0 already=10", r.stdout)
        self.assertEqual(self.fx.snapshot(), snap, "second run changed something")

    def test_12_dry_run_writes_nothing(self):
        snap = self.fx.snapshot()
        r = self.fx.bake("--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("DRY RUN: would apply 10", r.stdout)
        self.assertEqual(self.fx.snapshot(), snap)
        self.assertFalse(self.fx.manifest.exists())

    def test_13_stale_pyc_removed_and_mode_kept(self):
        cuda = self.fx.vllm / "platforms/cuda.py"
        os.chmod(cuda, 0o640)
        cache = cuda.parent / "__pycache__"
        cache.mkdir()
        (cache / "cuda.cpython-312.pyc").write_bytes(b"stale")
        (cache / "cuda.cpython-312.opt-1.pyc").write_bytes(b"stale")
        (cache / "other.cpython-312.pyc").write_bytes(b"keep")
        self.assertEqual(self.fx.bake().returncode, 0)
        self.assertFalse((cache / "cuda.cpython-312.pyc").exists())
        self.assertFalse((cache / "cuda.cpython-312.opt-1.pyc").exists())
        self.assertTrue((cache / "other.cpython-312.pyc").exists())
        self.assertEqual(cuda.stat().st_mode & 0o777, 0o640)

    def test_14_does_not_import_the_packages_and_finds_them_via_env_or_finder(self):
        # --vllm-root / --flashinfer-root not given: env var path
        env = dict(os.environ, GLM53_VLLM_ROOT=str(self.fx.vllm), GLM53_FLASHINFER_ROOT=str(self.fx.fi))
        r = subprocess.run([sys.executable, "-I", str(BAKE), "--moe-src", str(MOE_SRC), "--manifest", str(self.fx.manifest), "--dry-run"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        # importlib finder: a sys.path entry holding the two packages; their __init__ raises if executed
        shutil.rmtree(self.fx.manifest, ignore_errors=True)
        site = self.fx.tmp / "site"
        site.mkdir()
        (site / "vllm").symlink_to(self.fx.vllm)
        (site / "flashinfer").symlink_to(self.fx.fi)
        code = (f"import sys; sys.path.insert(0, {str(site)!r}); sys.argv=['x','--moe-src',{str(MOE_SRC)!r},'--dry-run'];"
                f"import importlib.util as u; spec=u.spec_from_file_location('b', {str(BAKE)!r}); m=u.module_from_spec(spec);"
                f"sys.modules['b']=m; spec.loader.exec_module(m); sys.exit(m.main(['--moe-src',{str(MOE_SRC)!r},'--dry-run']))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DRY RUN: would apply 10", r.stdout)

    # ---------------------------------------------------------------- anchors must match exactly once, else fail closed
    def _mutate_absent(self, e):
        p = self.fx.path(e.pkg, e.rel)
        t = p.read_text()
        i = t.index(e.old)
        k = len(e.old) // 2
        while e.old[k] in "\n ":
            k += 1
        ch = "Q" if e.old[k] != "Q" else "R"
        p.write_text(t[:i] + e.old[:k] + ch + e.old[k + 1:] + t[i + len(e.old):])

    def _mutate_twice(self, e):
        p = self.fx.path(e.pkg, e.rel)
        p.write_text(p.read_text() + "\n" + e.old)

    def test_20_mutated_anchor_fails_and_writes_nothing(self):
        for e in BK.EDITS:
            for mutate in (self._mutate_absent, self._mutate_twice):
                with self.subTest(edit=e.id, mutation=mutate.__name__):
                    fx = Fixture()
                    self.addCleanup(fx.cleanup)
                    saved, self.fx = self.fx, fx
                    try:
                        mutate(e)
                        snap = fx.snapshot()
                        r = fx.bake("--allow-pin-mismatch")  # the mutation also changes the sha; anchors must still stop the bake
                        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                        self.assertIn(e.id, r.stderr)
                        self.assertIn("exactly once", r.stderr)
                        self.assertEqual(fx.snapshot(), snap, "files changed although the bake failed")
                        self.assertFalse(fx.manifest.exists())
                    finally:
                        self.fx = saved

    def test_21_pin_mismatch_fails_closed_unless_allowed(self):
        for pkg, rel in BK.PINS:
            with self.subTest(file=f"{pkg}/{rel}"):
                fx = Fixture()
                self.addCleanup(fx.cleanup)
                p = fx.path(pkg, rel)
                p.write_text(p.read_text() + ("\n// unrelated change\n" if rel.endswith(".cuh") else "\n# unrelated change\n"))
                snap = fx.snapshot()
                r = fx.bake()
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("sha256", r.stderr)
                self.assertIn(rel, r.stderr)
                self.assertEqual(fx.snapshot(), snap)
                r2 = fx.bake("--allow-pin-mismatch")
                self.assertEqual(r2.returncode, 0, r2.stderr)
                self.assertIn("WARNING", r2.stdout)

    def test_22_flashinfer_version_mismatch(self):
        meta = self.fx.fi / "_build_meta.py"
        meta.write_text(meta.read_text().replace("0.7.0.post1", "0.7.2"))
        r = self.fx.bake()
        self.assertEqual(r.returncode, 1)
        self.assertIn("0.7.2", r.stderr)

    def test_23_partial_state_is_an_error(self):
        self.assertEqual(self.fx.bake().returncode, 0)
        pdl = next(e for e in BK.EDITS if e.id == "pdl")
        p = self.fx.path(pdl.pkg, pdl.rel)
        p.write_text(p.read_text().replace(pdl.new, pdl.old))  # cuda.py keeps pri12 but loses pdl
        snap = self.fx.snapshot()
        r = self.fx.bake()
        self.assertEqual(r.returncode, 1)
        self.assertIn("partially patched", r.stderr)
        self.assertEqual(self.fx.snapshot(), snap)

    def test_24_syntax_error_in_patched_text_blocks_the_write(self):
        bad = [dataclasses.replace(e, new=e.new + "def broken(:\n") if e.id == "kpool_pid" else e for e in BK.EDITS]
        saved = BK.EDITS
        BK.EDITS = bad
        try:
            snap = self.fx.snapshot()
            rc = BK.main(["--vllm-root", str(self.fx.vllm), "--flashinfer-root", str(self.fx.fi), "--moe-src", str(MOE_SRC),
                          "--manifest", str(self.fx.manifest)])
        finally:
            BK.EDITS = saved
        self.assertEqual(rc, 1)
        self.assertEqual(self.fx.snapshot(), snap)

    def test_25_moe_conflict_blocks_everything(self):
        dest = self.fx.vllm / BK.MOE_DEST_REL
        dest.write_text("{}")
        snap = self.fx.snapshot()
        r = self.fx.bake()
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to overwrite", r.stderr)
        self.assertEqual(self.fx.snapshot(), snap)

    def test_26_moe_source_pin(self):
        bad = self.fx.tmp / "moe-bad"
        bad.mkdir()
        (bad / BK.MOE_NAME).write_text((MOE_SRC / BK.MOE_NAME).read_text().replace('"1"', '"1 "'))
        r = run([BAKE, "--vllm-root", self.fx.vllm, "--flashinfer-root", self.fx.fi, "--moe-src", bad, "--manifest", self.fx.manifest])
        self.assertEqual(r.returncode, 1)
        self.assertIn("pinned", r.stderr)

    # ---------------------------------------------------------------- groups
    def test_30_skip_groups(self):
        r = self.fx.bake("--skip", "pdl,moe")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("applied=9", r.stdout)  # 10 edits minus the pdl edit (moe is a data file, not an edit)
        cuda = (self.fx.vllm / "platforms/cuda.py").read_text()
        self.assertIn("major >= 9\n", cuda)           # pdl untouched
        self.assertIn("FLASHINFER_MLA_SPARSE_SM90,\n", cuda)  # pri12 applied
        self.assertFalse((self.fx.vllm / BK.MOE_DEST_REL).exists())
        m = json.loads(self.fx.manifest.read_text())
        self.assertEqual(m["skipped_groups"], ["moe", "pdl"])
        v = self.fx.verify()
        self.assertEqual(v.returncode, 0, v.stdout)
        self.assertIn("SKIP  sem.pdl", v.stdout)
        v2 = self.fx.verify("--require-all")
        self.assertEqual(v2.returncode, 1)

    def test_31_unknown_group(self):
        r = self.fx.bake("--skip", "nope")
        self.assertEqual(r.returncode, 2)

    def test_32_check_pins(self):
        r = self.fx.bake("--check-pins")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("pin OK"), 7)
        self.assertEqual(self.fx.bake().returncode, 0)
        r = self.fx.bake("--check-pins")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(r.stdout.count("BAKED"), 7)
        p = self.fx.vllm / "platforms/cuda.py"
        p.write_text(p.read_text().replace("# SM121 bake", "# something else"))
        r = self.fx.bake("--check-pins")
        self.assertEqual(r.returncode, 1)

    # ---------------------------------------------------------------- verify_bake must notice damage
    def test_40_verify_detects_each_reverted_edit(self):
        self.assertEqual(self.fx.bake().returncode, 0)
        for e in BK.EDITS:
            with self.subTest(edit=e.id):
                p = self.fx.path(e.pkg, e.rel)
                good = p.read_text()
                p.write_text(good.replace(e.new, e.old))
                v = self.fx.verify()
                self.assertEqual(v.returncode, 1, f"verify did not notice that {e.id} was reverted")
                self.assertIn(f"FAIL  text.{e.id}", v.stdout)
                p.write_text(good)
        self.assertEqual(self.fx.verify().returncode, 0)

    def test_41_verify_semantics_catch_a_wrong_replacement(self):
        """text matches the bake table, behaviour is wrong -> the sem.* checks (independent of the table) must fail."""
        wrong = {
            "pdl": ("major >= 9 and major != 12", "major >= 9"),
            "sm90cap": ("capability.major in (9, 12)", "capability.major in (9, 10)"),
            "kpool_pid": ("(pid >= 0) & (pid < pool_len)", "(pid >= 0) | (pid < pool_len)"),
            "fi_tile_cap": ("(CTA_TILE_KV < 32u ? CTA_TILE_KV : 32u)", "(CTA_TILE_KV > 32u ? CTA_TILE_KV : 32u)"),
            "fi_fp8_gate": ('not (major == 12 and backend == "fa2")', 'not (major == 12 and backend == "fa3")'),
        }
        for eid, (a, b) in wrong.items():
            with self.subTest(edit=eid):
                fx = Fixture()
                self.addCleanup(fx.cleanup)
                self.assertEqual(fx.bake().returncode, 0)
                e = next(x for x in BK.EDITS if x.id == eid)
                p = fx.path(e.pkg, e.rel)
                t = p.read_text()
                self.assertIn(a, t)
                p.write_text(t.replace(a, b))
                v = fx.verify()
                self.assertEqual(v.returncode, 1, v.stdout)
                self.assertRegex(v.stdout, rf"FAIL  text\.{eid}")   # the text no longer equals the bake table ...
                self.assertRegex(v.stdout, rf"FAIL  sem\.{eid}")    # ... and the behaviour check, which does not use the table, fails too

    def test_42_verify_moe_table_checks(self):
        self.assertEqual(self.fx.bake().returncode, 0)
        dest = self.fx.vllm / BK.MOE_DEST_REL
        good = dest.read_bytes()
        dest.write_bytes(good.replace(b'"BLOCK_SIZE_M": 16', b'"BLOCK_SIZE_M": 32', 1))
        v = self.fx.verify()
        self.assertEqual(v.returncode, 1)
        self.assertIn("FAIL  sem.moe_table", v.stdout)
        dest.unlink()
        self.assertEqual(self.fx.verify().returncode, 1)

    def test_43_verify_runtime_layer_skips_cleanly_without_torch(self):
        self.assertEqual(self.fx.bake().returncode, 0)
        v = run([VERIFY, "--vllm-root", self.fx.vllm, "--flashinfer-root", self.fx.fi, "--manifest", self.fx.manifest])
        # package __init__ raises (fixture) or torch is missing: either way Layer 2 must SKIP, not FAIL
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertIn("SKIP  rt", v.stdout)
        v = run([VERIFY, "--vllm-root", self.fx.vllm, "--flashinfer-root", self.fx.fi, "--manifest", self.fx.manifest, "--require-runtime"])
        self.assertEqual(v.returncode, 1)


    # ---------------------------------------------------------------- sha256 of every file after the bake (for the runtime patches that pin them)
    def _baked_pins_table(self) -> str:
        """sha256 of each edited file for every subset of its edit groups that can be applied (--skip the rest)."""
        import itertools

        groups = ["attn", "pdl", "topk_neg1", "kpool_pid"]
        file_groups: dict = {}
        for e in BK.EDITS:
            file_groups.setdefault((e.pkg, e.rel), set()).add(e.group)
        rows = {}
        for r in range(len(groups) + 1):
            for applied in itertools.combinations(groups, r):
                fx = Fixture()
                self.addCleanup(fx.cleanup)
                skip = ",".join(["moe"] + [g for g in groups if g not in applied])
                res = fx.bake("--skip", skip)
                self.assertEqual(res.returncode, 0, res.stderr)
                m = json.loads(fx.manifest.read_text())
                for (pkg, rel), gs in file_groups.items():
                    mine = tuple(sorted(g for g in applied if g in gs))
                    key = (pkg, rel, mine)
                    if mine and key not in rows:
                        rows[key] = m["files"][f"{pkg}/{rel}"]["sha256_after"]
        for (pkg, rel), pin in BK.PINS.items():
            rows[(pkg, rel, ())] = pin
        lines = []
        for (pkg, rel, mine), sha_ in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], len(kv[0][2]), kv[0][2])):
            lines.append(f"{sha_}  {pkg}/{rel}  applied={','.join(mine) or 'none'}")
        return "\n".join(lines) + "\n"

    def test_50_pins_after_bake_file_is_current(self):
        table = self._baked_pins_table()
        target = IMAGE / "pins-after-bake.txt"
        if os.environ.get("BAKE_WRITE_PINS"):
            target.write_text(table)
        self.assertTrue(target.exists(), "run once with BAKE_WRITE_PINS=1 to create pins-after-bake.txt")
        self.assertEqual(target.read_text(), table, "pins-after-bake.txt is stale (edit table changed); regenerate with BAKE_WRITE_PINS=1")


@unittest.skipUnless(shutil.which("clang++"), "clang++ not available")
class CppExpressionTest(unittest.TestCase):
    """The replaced statement in mla.cuh is valid C++17 and caps (not forces) the FP8 tile."""

    def test_tile_cap_expression(self):
        e = next(x for x in BK.EDITS if x.id == "fi_tile_cap")
        stmt = "\n".join(l for l in e.new.splitlines() if not l.strip().startswith("//"))
        src = f"""
#include <cstdint>
#include <type_traits>
struct __nv_fp8_e4m3 {{}};
template <typename DTypeKV, uint32_t CTA_TILE_KV>
constexpr uint32_t eff() {{
{stmt}
  return EFF_CTA_TILE_KV;
}}
// stock behaviour forced 32 for FP8 on every tier; the cap keeps 32 where it fits and uses the tier's tile below
static_assert(eff<__nv_fp8_e4m3, 64>() == 32, "tier >=221696 (H100/B200)");
static_assert(eff<__nv_fp8_e4m3, 32>() == 32, "tier >=147968");
static_assert(eff<__nv_fp8_e4m3, 16>() == 16, "GB10 tier");
static_assert(eff<float, 16>() == 16 && eff<float, 64>() == 64, "non-FP8 untouched");
int main() {{ return 0; }}
"""
        with tempfile.TemporaryDirectory(dir=WORK) as d:
            f = Path(d) / "t.cpp"
            f.write_text(src)
            r = subprocess.run(["clang++", "-std=c++17", "-fsyntax-only", str(f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


class OriginalsUntouchedTest(unittest.TestCase):
    def test_originals_unchanged(self):
        for (pkg, rel), pin in BK.PINS.items():
            if pkg == "vllm":
                self.assertEqual(sha(VLLM_TREE / rel), pin, f"original {rel} changed!")
        self.assertEqual(sha(FI_WHEEL), FI_WHEEL_SHA256)


@unittest.skipUnless(os.environ.get("BAKE_TEST_FULL"), "set BAKE_TEST_FULL=1 to bake a copy of the whole vLLM tree")
class FullTreeTest(unittest.TestCase):
    def test_full_tree_copy(self):
        WORK.mkdir(exist_ok=True)
        dest = WORK / "fixture-full"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(VLLM_TREE, dest / "vllm")
        fi = dest / "flashinfer"
        with zipfile.ZipFile(FI_WHEEL) as z:
            for rel in FI_FILES:
                z.extract("flashinfer/" + rel, dest)
        diff = IMAGE / "PATCH.diff"
        r = run([BAKE, "--vllm-root", dest / "vllm", "--flashinfer-root", fi, "--moe-src", MOE_SRC,
                 "--manifest", dest / "manifest.json", "--diff-out", diff])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        (dest / "bake.log").write_text(r.stdout)
        v = run([VERIFY, "--vllm-root", dest / "vllm", "--flashinfer-root", fi, "--manifest", dest / "manifest.json", "--static"])
        self.assertEqual(v.returncode, 0, v.stdout)
        # whole-tree compile of every .py the bake touched, and a diff against the untouched original tree
        n_changed = 0
        for p in (dest / "vllm").rglob("*"):
            if p.is_file():
                orig = VLLM_TREE / p.relative_to(dest / "vllm")
                if not orig.exists() or orig.read_bytes() != p.read_bytes():
                    n_changed += 1
        self.assertEqual(n_changed, 5, "expected 4 edited vLLM files + 1 installed MoE table")


if __name__ == "__main__":
    unittest.main()
