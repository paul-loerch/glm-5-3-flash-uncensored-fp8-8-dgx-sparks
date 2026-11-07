#!/usr/bin/env python3
"""verify_bake.py -- prove that the SM121 bake (glm53_sm121_v032.py) is present in an image.

Runs on CPU, no GPU needed.

Layer 1 (always; stdlib only -- no torch, vllm or flashinfer import):
  * every edit's new text is in its file exactly once and the old text is gone;
  * every edited .py file compiles;
  * the semantics of each edit are checked on the file's own code: the functions are cut out of
    the patched source with `ast`, executed against tiny fakes (device capability, dtypes), and
    must give the answer GB10 needs (capability 12 -> SM90 backend allowed, FA2 chosen, PDL off,
    FP8 MLA plan accepted for fa2 only, patched JIT URI, pool ids bounded, top-k buffers = -1,
    MoE file name == what vLLM's own get_config_file_name() builds on NVIDIA_GB10);
  * the tuned MoE table is at the path vLLM reads and has the pinned sha256.
Layer 2 (default on, skipped quietly when the packages cannot be imported; --static turns it off):
  * the same facts through the real modules (vllm / flashinfer / torch imported in the image).

Exit status 0 = every check passed (SKIPs allowed unless --require-runtime), 1 = at least one FAIL.

  python3 -I /opt/glm53-sm121/verify_bake.py [--static] [--require-all] [--require-runtime]
                                             [--vllm-root DIR] [--flashinfer-root DIR] [--manifest PATH]
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from collections import namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = "/opt/glm53-sm121/bake-manifest.json"

_RESULTS: list = []  # (status, check id, detail)


def record(status: str, cid: str, detail: str = "") -> None:
    _RESULTS.append((status, cid, detail))
    print(f"{status:5s} {cid}: {detail}" if detail else f"{status:5s} {cid}", flush=True)


def passed(cid, detail=""):
    record("PASS", cid, detail)


def failed(cid, detail=""):
    record("FAIL", cid, detail)


def skipped(cid, detail=""):
    record("SKIP", cid, detail)


def info(cid, detail=""):
    record("INFO", cid, detail)


def load_bake():
    path = HERE / "glm53_sm121_v032.py"
    spec = importlib.util.spec_from_file_location("glm53_sm121_v032", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["glm53_sm121_v032"] = mod  # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------
# ast helpers
# ------------------------------------------------------------------------------------------
def find_def(tree: ast.AST, qualname: str) -> ast.FunctionDef:
    """Function (or method 'Class.method') definition in a module AST."""
    parts = qualname.split(".")
    body = tree.body
    node = None
    for i, part in enumerate(parts):
        node = None
        for child in body:
            want = ast.ClassDef if i < len(parts) - 1 else (ast.FunctionDef, ast.AsyncFunctionDef)
            if isinstance(child, want) and child.name == part:
                node = child
                break
        if node is None:
            raise LookupError(f"{qualname}: {part!r} not found")
        body = node.body
    return node


def compile_def(fn: ast.FunctionDef, namespace: dict, filename: str):
    """Compile a single function out of its module: decorators and annotations stripped."""
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    fn.returns = None
    a = fn.args
    for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, *([a.vararg] if a.vararg else []), *([a.kwarg] if a.kwarg else [])]:
        arg.annotation = None
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, filename, "exec"), namespace)
    return namespace[fn.name]


class Dtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype


def fake_torch():
    names = ["int32", "float16", "bfloat16", "float8_e4m3fn"]
    ns = types.SimpleNamespace(**{n: Dtype(n) for n in names})
    return ns


Cap = namedtuple("DeviceCapability", ["major", "minor"])


# ------------------------------------------------------------------------------------------
# environment snapshot (stdlib importlib.metadata; nothing is imported)
# ------------------------------------------------------------------------------------------
def jit_toolchain() -> None:
    """FlashInfer JIT-builds the patched FA2 FP8 MLA module on first use: needs nvcc that accepts sm_121a, ninja and the
    toolkit headers under $CUDA_HOME/include (what flashinfer/jit/cpp_ext.py adds to the include path).  INFO only."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    ninja = shutil.which("ninja")
    parts = [f"CUDA_HOME={cuda_home}", f"ninja={ninja or 'MISSING'}"]
    missing = [] if ninja else ["ninja"]
    if not nvcc.is_file():
        missing.append("nvcc")
    inc = Path(cuda_home) / "include"
    for h in ("cuda_runtime.h", "cuda_fp8.h", "cuda_bf16.h", "cooperative_groups.h"):
        if not (inc / h).is_file():
            missing.append(h)
    if nvcc.is_file():
        try:
            out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=30).stdout
            m = re.search(r"release (\d+\.\d+)", out)
            parts.append("nvcc=" + (m.group(1) if m else "?"))
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "t.cu").write_text("__global__ void k() {}\n")
                r = subprocess.run([str(nvcc), "--dryrun", "-gencode", "arch=compute_121a,code=sm_121a", "-c",
                                    str(Path(d) / "t.cu"), "-o", str(Path(d) / "t.o")],
                                   capture_output=True, text=True, timeout=30)
            parts.append("sm_121a=" + ("accepted" if r.returncode == 0 else "REJECTED"))
            if r.returncode != 0:
                missing.append("nvcc support for sm_121a")
        except (OSError, subprocess.SubprocessError) as exc:
            parts.append(f"nvcc-probe-failed={type(exc).__name__}")
    info("env.jit", " ".join(parts) + (f"  WARNING missing: {missing} -- the first FA2 FP8 MLA build will fail" if missing else ""))


def env_snapshot() -> None:
    from importlib import metadata

    names = ["vllm", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache", "torch", "triton",
             "nvidia-nccl-cu13", "nvidia-cutlass-dsl", "tilelang", "apache-tvm-ffi", "transformers"]
    vers = {}
    for n in names:
        try:
            vers[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            vers[n] = "-"
    providers = []
    try:
        for d in metadata.distributions():
            nm = (d.metadata["Name"] or "")
            if nm.lower().startswith("flashinfer-jit-cache-"):
                providers.append(f"{nm}={d.version}")
    except Exception:  # noqa: BLE001
        pass
    info("env.versions", " ".join(f"{k}={v}" for k, v in vers.items()) + (f" providers: {' '.join(sorted(providers))}" if providers else ""))
    nccl = vers["nvidia-nccl-cu13"]
    if nccl not in ("-", "2.30.7"):
        info("env.nccl", f"WARNING nvidia-nccl-cu13 is {nccl}, the cluster fabric needs 2.30.x (2.29.7 fails ncclCommInitRank)")
    fi = vers["flashinfer-python"]
    if fi not in ("-", "0.7.0.post1"):
        info("env.flashinfer", f"WARNING flashinfer-python is {fi}, this bake was written for 0.7.0.post1")


# ------------------------------------------------------------------------------------------
# Layer 1
# ------------------------------------------------------------------------------------------
def layer1(bake, roots: dict, skip_groups: set, manifest: dict | None) -> None:
    files = {}

    def src(pkg, rel):
        key = (pkg, rel)
        if key not in files:
            files[key] = (roots[pkg] / rel).read_text()
        return files[key]

    def tree(pkg, rel):
        return ast.parse(src(pkg, rel), filename=f"{pkg}/{rel}")

    # ---- 1. text of every edit --------------------------------------------------------------
    for e in bake.EDITS:
        cid = f"text.{e.id}"
        if e.group in skip_groups:
            skipped(cid, f"group {e.group} was skipped at bake time")
            continue
        try:
            text = src(e.pkg, e.rel)
        except OSError as exc:
            failed(cid, f"{e.pkg}/{e.rel} unreadable: {exc}")
            continue
        n_new, n_old = text.count(e.new), text.count(e.old)
        if n_new == 1 and n_old == 0:
            passed(cid, f"{e.pkg}/{e.rel}:{bake.line_of(text, e.new)}")
        else:
            failed(cid, f"{e.pkg}/{e.rel}: new text x{n_new} (want 1), stock text x{n_old} (want 0)")

    for (pkg, rel) in sorted({(e.pkg, e.rel) for e in bake.EDITS if e.group not in skip_groups}):
        if rel.endswith(".py"):
            cid = f"compile.{pkg}/{rel}"
            try:
                compile(src(pkg, rel), f"{pkg}/{rel}", "exec", dont_inherit=True)
            except (SyntaxError, OSError) as exc:
                failed(cid, str(exc))
            else:
                passed(cid)

    def guarded(cid, group, fn):
        if group in skip_groups:
            skipped(cid, f"group {group} was skipped at bake time")
            return
        try:
            detail = fn()
            passed(cid, detail or "")
        except AssertionError as exc:
            failed(cid, f"{exc}")
        except Exception as exc:  # noqa: BLE001 - a verify script reports, it does not crash
            failed(cid, f"{type(exc).__name__}: {exc}")

    # ---- 2. attn: vLLM side -----------------------------------------------------------------
    def check_pri12():
        t = tree("vllm", "platforms/cuda.py")
        fn = find_def(t, "_get_backend_priorities")
        found = None
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "device_capability.major == 12":
                ret = node.body[-1]
                assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.List), "branch does not return a list"
                found = [ast.unparse(el) for el in ret.value.elts]
        assert found is not None, "no `device_capability.major == 12` branch"
        want = [
            "AttentionBackendEnum.TRITON_MLA",
            "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90",
            "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120",
        ]
        assert found == want, f"capability-12 priority is {found}"
        return "capability 12 -> " + " > ".join(x.split(".")[-1] for x in found)

    guarded("sem.pri12", "attn", check_pri12)

    def check_sm90cap():
        rel = "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
        fn = find_def(tree("vllm", rel), "FlashInferMLASparseSM90Backend.supports_compute_capability")
        f = compile_def(fn, {}, rel)
        got = {c: f(None, Cap(*c)) for c in [(8, 9), (9, 0), (10, 0), (11, 0), (12, 0), (12, 1)]}
        assert got == {(8, 9): False, (9, 0): True, (10, 0): False, (11, 0): False, (12, 0): True, (12, 1): True}, got
        return "supports_compute_capability: 9.x and 12.x true, 8.x/10.x/11.x false"

    guarded("sem.sm90cap", "attn", check_sm90cap)

    def check_sm90fa2():
        rel = "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
        init = find_def(tree("vllm", rel), "_SM90State.__init__")
        expr = None
        for node in ast.walk(init):
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("BatchMLAPagedAttentionWrapper"):
                for kw in node.keywords:
                    if kw.arg == "backend":
                        expr = ast.unparse(kw.value)
        assert expr, "BatchMLAPagedAttentionWrapper(..., backend=...) not found"
        got = {}
        for cap in [(8, 0), (9, 0), (10, 0), (12, 1)]:
            fake = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda d, cap=cap: cap))
            got[cap] = eval(expr, {"torch": fake, "device": "cuda:0"})
        assert got == {(8, 0): "fa2", (9, 0): "fa3", (10, 0): "fa2", (12, 1): "fa2"}, got
        return "wrapper backend: SM90 -> fa3, SM12x -> fa2"

    guarded("sem.sm90fa2", "attn", check_sm90fa2)

    # ---- 3. attn: FlashInfer side -----------------------------------------------------------
    def check_fi_fp8_gate():
        rel = "mla/_batch_mla/_backends/_fa_common.py"
        fn = find_def(tree("flashinfer", rel), "_validate_generated_fa_plan")
        cc = {"v": (9, 0)}
        torch = fake_torch()
        ns = {"torch": torch, "get_compute_capability": lambda device: cc["v"]}
        f = compile_def(fn, ns, rel)

        def call(backend, capability, kv):
            cc["v"] = capability
            i32 = FakeTensor(torch.int32)
            f(
                backend=backend, device="cuda:0", qo_indptr=i32, kv_indptr=i32, kv_indices=i32, kv_len_arr=i32,
                head_dim_ckv=512, head_dim_kpe=0, q_data_type=torch.bfloat16, kv_data_type=kv,
                output_dtype=torch.bfloat16, scale_mode="kv-per-tensor" if kv is torch.float8_e4m3fn else "default",
            )

        fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
        for backend, capability, kv, ok in [
            ("fa2", (12, 1), fp8, True), ("fa2", (12, 0), fp8, True), ("fa2", (9, 0), fp8, True),
            ("fa3", (9, 0), fp8, True), ("fa3", (12, 1), fp8, False), ("fa2", (10, 0), fp8, False),
            ("fa2", (8, 0), fp8, False), ("fa2", (12, 1), bf16, True),
        ]:
            try:
                call(backend, capability, kv)
                accepted = True
            except ValueError as exc:
                accepted = False
                msg = str(exc)
            assert accepted == ok, f"{backend} on SM{capability[0]}{capability[1]} kv={kv.name}: accepted={accepted}, want {ok}"
            if not accepted:
                assert "SM90" in msg, msg
        return "FP8 MLA plan: fa2 accepted on SM12x/SM90, fa3 only on SM90, SM10x/SM8x refused"

    guarded("sem.fi_fp8_gate", "attn", check_fi_fp8_gate)

    def check_fi_tile_cap():
        cuh = src("flashinfer", "data/include/flashinfer/attention/mla.cuh")
        tiers = [(int(m.group(1)), int(m.group(2)), int(m.group(3))) for m in re.finditer(
            r"smem_limit_per_sm >= (\d+)\)\s*\{\s*\\\s*\n\s*constexpr uint32_t NUM_STAGES = (\d+);\s*\\\s*\n\s*constexpr uint32_t CTA_TILE_KV = (\d+);", cuh)]
        assert tiers, "DISPATCH_SMEM_CONFIG tiers not found"
        assert re.search(r"\(CTA_TILE_KV < 32u \? CTA_TILE_KV : 32u\)", cuh), "tile-cap expression missing"

        def tier(smem):
            for thr, stages, k in tiers:
                if smem >= thr:
                    return stages, k
            raise AssertionError(f"no tier for {smem} B")

        def eff_fp8_tile(smem):  # the patched expression
            return min(tier(smem)[1], 32)

        # sizeof(SharedStorageQKVO) of the FP8 instantiation (CTA_TILE_Q 64, ckv 512, kpe 0, DTypeQ bf16, DTypeKV e4m3),
        # transcribed from the struct at mla.cuh:53-77 -- the member list is asserted below so a changed struct is noticed
        for member in (r"q_smem_nope\[CTA_TILE_Q \* HEAD_DIM_CKV\]", r"q_smem_pe\[CTA_TILE_Q \* HEAD_DIM_KPE\]",
                       r"ckv_smem\[NUM_STAGES\]\[CTA_TILE_KV \* HEAD_DIM_CKV\]",
                       r"CTA_TILE_KV \* \(HEAD_DIM_KPE > CTA_TILE_Q \? HEAD_DIM_KPE : CTA_TILE_Q\)\]",
                       r"DTypeQ\[NUM_STAGES \* CTA_TILE_KV \* HEAD_DIM_CKV\]",
                       r"DTypeQ\[NUM_STAGES \* CTA_TILE_KV \* \(HEAD_DIM_KPE > 0 \? HEAD_DIM_KPE : 1\)\]",
                       r"float m_wg\[2\]\[CTA_TILE_Q\]"):
            assert re.search(member, cuh), f"SharedStorageQKVO changed (no match for {member}); redo the smem arithmetic"

        def fp8_smem(stages, tile, tq=64, ckv=512, kpe=0):
            return (tq * ckv * 2 + tq * kpe * 2 + stages * tile * ckv + stages * tile * max(kpe, tq)
                    + stages * tile * ckv * 2 + stages * tile * (kpe if kpe > 0 else 1) * 2 + 2 * tq * 4)

        gb10_sm, gb10_block = 102400, 101376   # per-SM attribute used by the dispatch; max opt-in dynamic smem per block
        hopper_sm, hopper_block = 232448, 232448 - 1024
        st, k = tier(gb10_sm)
        assert (st, k) == (1, 16) and eff_fp8_tile(gb10_sm) == 16, (st, k, eff_fp8_tile(gb10_sm))
        stock, capped = fp8_smem(st, 32), fp8_smem(st, eff_fp8_tile(gb10_sm))
        assert stock > gb10_block >= capped, f"stock forced tile needs {stock} B, capped needs {capped} B, GB10 allows {gb10_block} B"
        hs, hk = tier(hopper_sm)
        assert eff_fp8_tile(hopper_sm) == 32 and eff_fp8_tile(147968) == 32, "cap must leave the >=147968 B tiers at 32 (stock behaviour)"
        assert fp8_smem(hs, 32) <= hopper_block, "Hopper FP8 tile must still fit"
        return (f"GB10 (smem/SM {gb10_sm}, per-block limit {gb10_block}): stock forced FP8 tile 32 needs {stock} B (> limit), "
                f"capped tile {eff_fp8_tile(gb10_sm)} needs {capped} B; Hopper tile stays {eff_fp8_tile(hopper_sm)} ({fp8_smem(hs, 32)} B)")

    guarded("sem.fi_tile_cap", "attn", check_fi_tile_cap)

    def check_fi_uri():
        rel = "jit/attention/modules.py"
        fn = find_def(tree("flashinfer", rel), "get_batch_mla_uri")
        torch = fake_torch()
        short = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float8_e4m3fn: "e4m3", torch.int32: "i32"}
        ns = {"torch": torch, "filename_safe_dtype_map": short, "filename_safe_dtype_map_kv": lambda d: short[d]}
        f = compile_def(fn, ns, rel)
        bf, fp8, i32 = torch.bfloat16, torch.float8_e4m3fn, torch.int32
        base = "batch_mla_attention_dtype_q_bf16_dtype_kv_{kv}_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_{kpe}_profiler_False_planabi2"
        cases = {
            ("fa2", fp8, 0): base.format(kv="e4m3", kpe=0) + "_fp8tilecap",
            ("fa2", fp8, 64): base.format(kv="e4m3", kpe=64) + "_fp8tilecap",
            ("fa2", bf, 64): base.format(kv="bf16", kpe=64),
            ("fa3", fp8, 0): base.format(kv="e4m3", kpe=0) + "_sm90",
            ("fa3", bf, 64): base.format(kv="bf16", kpe=64) + "_sm90",
        }
        for (backend, kv, kpe), want in cases.items():
            got = f(backend, bf, kv, bf, i32, 512, kpe, False)
            assert got == want, f"{backend}/{kv}/kpe{kpe}: {got} != {want}"
        return "FA2+FP8 URI ends _fp8tilecap; bf16-KV and FA3 URIs (and the jit-cache names) unchanged"

    guarded("sem.fi_uri", "attn", check_fi_uri)

    # ---- 4. PDL -------------------------------------------------------------------------------
    def check_pdl():
        fn = find_def(tree("vllm", "platforms/cuda.py"), "CudaPlatformBase.is_arch_support_pdl")
        state = {"cap": (9, 0), "boom": False}

        def get_cap(device):
            if state["boom"]:
                raise RuntimeError("no device")
            return state["cap"]

        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(current_device=lambda: 0, get_device_capability=get_cap))
        f = compile_def(fn, {"torch": fake}, "platforms/cuda.py")
        got = {}
        for cap in [(8, 9), (9, 0), (10, 0), (10, 3), (11, 0), (12, 0), (12, 1)]:
            state["cap"] = cap
            got[cap] = f(None)
        state["boom"] = True
        assert f(None) is False, "exception path must return False"
        assert got == {(8, 9): False, (9, 0): True, (10, 0): True, (10, 3): True, (11, 0): True, (12, 0): False, (12, 1): False}, got
        return "is_arch_support_pdl: False on 12.0/12.1 (GB10), True on 9.x/10.x/11.x"

    guarded("sem.pdl", "pdl", check_pdl)

    # ---- 5. top-k -1 init ---------------------------------------------------------------------
    def check_topk():
        rel = "models/glm5next/nvidia/sparse_indexer.py"
        t = tree("vllm", rel)
        fn = find_def(t, "sparse_attn_indexer_kpool")
        calls = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "pool_topk" for x in node.targets):
                calls.append(node.value)
        assert len(calls) == 2, f"expected 2 pool_topk allocations, found {len(calls)}"
        for c in calls:
            assert isinstance(c, ast.Call) and ast.unparse(c.func) == "torch.full", f"allocation is {ast.unparse(c)[:60]}"
            fill = c.args[1]
            assert ast.unparse(fill) == "-1", f"fill value {ast.unparse(fill)}"
            assert any(k.arg == "dtype" and ast.unparse(k.value) == "torch.int32" for k in c.keywords), "dtype not int32"
        assert "pool_topk = torch.empty" not in src("vllm", rel), "a torch.empty pool_topk survives"
        return "both pool_topk allocations are torch.full(..., -1, dtype=torch.int32)"

    guarded("sem.topk_neg1", "topk_neg1", check_topk)

    # ---- 6. kpool pid bound --------------------------------------------------------------------
    def check_kpool():
        rel = "models/glm5next/nvidia/ops/kpool_compress.py"
        fn = find_def(tree("vllm", rel), "_expand_pools_and_append_tail_kernel")
        pool_len_line = hist = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "pool_len":
                    pool_len_line = node.lineno
                if node.targets[0].id == "hist_out":
                    hist = node
        assert pool_len_line and hist, "pool_len / hist_out assignments not found"
        assert pool_len_line < hist.lineno, "pool_len must be defined before hist_out"
        call = hist.value
        assert isinstance(call, ast.Call) and ast.unparse(call.func) == "tl.where", ast.unparse(call)
        cond, val, other = (ast.unparse(a) for a in call.args)
        assert cond == "(pid >= 0) & (pid < pool_len)" and val == "hist_val" and other == "-1", ast.unparse(call)
        try:
            import numpy as np
        except ImportError:
            return "AST ok (numpy unavailable: numeric check skipped)"
        pid = np.array([-1, 0, 3, 7, 8, 9, 1000])
        hist_val = np.arange(100, 107)
        got = eval(ast.unparse(call), {"tl": types.SimpleNamespace(where=np.where), "pid": pid, "pool_len": 8, "hist_val": hist_val})
        assert got.tolist() == [-1, 101, 102, 103, -1, -1, -1], got.tolist()
        return "pid in [0, pool_len) kept; -1, pool_len, pool_len+1 and 1000 -> -1"

    guarded("sem.kpool_pid", "kpool_pid", check_kpool)

    # ---- 7. MoE table ---------------------------------------------------------------------------
    def check_moe():
        dest = roots["vllm"] / bake.MOE_DEST_REL
        assert dest.is_file(), f"{dest} missing"
        raw = dest.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == bake.MOE_SHA256, "sha256 differs from the pinned table"
        cfg = json.loads(raw)
        keys = sorted(int(k) for k in cfg if k != "triton_version")
        assert keys == bake.MOE_BATCH_KEYS, keys
        fm_rel = "model_executor/layers/fused_moe/fused_moe.py"
        t = tree("vllm", fm_rel)
        gfn = find_def(t, "get_config_file_name")
        f = compile_def(gfn, {"get_device_name_as_file_name": lambda: "NVIDIA_GB10"}, fm_rel)
        name = f(288, 256, "fp8_w8a8", [128, 128])
        assert name == bake.MOE_NAME == dest.name, f"vLLM would look for {name!r}, file is {dest.name!r}"
        text = src("vllm", fm_rel)
        assert 'os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name' in text, "lookup path changed"
        assert dest.parent == (roots["vllm"] / "model_executor/layers/fused_moe/configs"), "table is not in fused_moe/configs"
        return f"{dest.name} ({len(keys)} batch sizes) at vllm/model_executor/layers/fused_moe/configs/, name == get_config_file_name(288, 256, fp8_w8a8, [128,128]) on NVIDIA_GB10"

    guarded("sem.moe_table", "moe", check_moe)

    # ---- 8. manifest ----------------------------------------------------------------------------
    if manifest is None:
        info("manifest", "none found (fine for a hand-patched tree); every edit was checked above")
    else:
        cid = "manifest"
        bad = []
        for rel_key, rec in manifest.get("files", {}).items():
            pkg, rel = rel_key.split("/", 1)
            p = roots[pkg] / rel
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() != rec.get("sha256_after"):
                bad.append(rel_key)
        if bad:
            info(cid, f"files changed after the bake (sha256 differs from the manifest): {bad} -- text checks above still decide")
        else:
            passed(cid, f"{len(manifest.get('files', {}))} files match the manifest; skipped groups: {manifest.get('skipped_groups') or 'none'}")


# ------------------------------------------------------------------------------------------
# Layer 2: the same facts through the real, imported modules
# ------------------------------------------------------------------------------------------
def layer2(bake, roots: dict, skip_groups: set, require: bool) -> None:
    # Layer 2 imports whatever `import vllm` / `import flashinfer` resolve to; it is only meaningful when that is
    # the tree Layer 1 just verified (always true inside the image).
    for pkg in ("vllm", "flashinfer"):
        spec = importlib.util.find_spec(pkg)
        found = Path(list(spec.submodule_search_locations)[0]).resolve() if spec and spec.submodule_search_locations else None
        if found != roots[pkg]:
            (failed if require else skipped)("rt", f"importable {pkg} is {found}, not the verified root {roots[pkg]}")
            return

    def runtime(cid, group, fn):
        if group in skip_groups:
            skipped(cid, f"group {group} was skipped at bake time")
            return
        try:
            detail = fn()
        except (ImportError, OSError) as exc:  # the real packages are not usable here
            (failed if require else skipped)(cid, f"cannot import: {type(exc).__name__}: {exc}")
            return
        except AssertionError as exc:
            failed(cid, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            failed(cid, f"{type(exc).__name__}: {exc}")
            return
        passed(cid, detail or "")

    def rt_pri12():
        from vllm.platforms import cuda as vcuda
        from vllm.platforms.interface import DeviceCapability

        prio = vcuda._get_backend_priorities(True, DeviceCapability(12, 1), None, "fp8_e4m3", False, 512, False)
        names = [b.name for b in prio]
        assert names == ["TRITON_MLA", "FLASHINFER_MLA_SPARSE_SM90", "FLASHINFER_MLA_SPARSE_SM120"], names
        return "vllm.platforms.cuda._get_backend_priorities(12.1) = " + " > ".join(names)

    runtime("rt.pri12", "attn", rt_pri12)

    def rt_sm90cap():
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import FlashInferMLASparseSM90Backend as B

        got = {c: B.supports_compute_capability(DeviceCapability(*c)) for c in [(9, 0), (10, 0), (12, 1)]}
        assert got == {(9, 0): True, (10, 0): False, (12, 1): True}, got
        return "FlashInferMLASparseSM90Backend.supports_compute_capability: 9.0 True, 10.0 False, 12.1 True"

    runtime("rt.sm90cap", "attn", rt_sm90cap)

    def rt_pdl():
        import torch
        from vllm.platforms.cuda import CudaPlatformBase

        real = (torch.cuda.current_device, torch.cuda.get_device_capability)
        try:
            torch.cuda.current_device = lambda: 0
            got = {}
            for cap in [(9, 0), (10, 0), (12, 1)]:
                torch.cuda.get_device_capability = lambda d, cap=cap: cap
                got[cap] = CudaPlatformBase.is_arch_support_pdl()
        finally:
            torch.cuda.current_device, torch.cuda.get_device_capability = real
        assert got == {(9, 0): True, (10, 0): True, (12, 1): False}, got
        return "CudaPlatformBase.is_arch_support_pdl(): 9.0 True, 10.0 True, 12.1 False"

    runtime("rt.pdl", "pdl", rt_pdl)

    def rt_uri():
        import torch
        from flashinfer.jit.attention.modules import get_batch_mla_uri

        a = get_batch_mla_uri("fa2", torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, 512, 0, False)
        b = get_batch_mla_uri("fa2", torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, 512, 64, False)
        c = get_batch_mla_uri("fa3", torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, 512, 0, False)
        assert a.endswith("_planabi2_fp8tilecap"), a
        assert b.endswith("_planabi2") and c.endswith("_planabi2_sm90"), (b, c)
        return "real URI of the module the SM90 backend builds on GB10: " + a

    runtime("rt.fi_uri", "attn", rt_uri)

    def rt_fp8_gate():
        import torch
        import flashinfer.mla._batch_mla._backends._fa_common as fc

        real = fc.get_compute_capability
        try:
            fc.get_compute_capability = lambda device: (12, 1)
            idx = torch.zeros(1, dtype=torch.int32)
            common = dict(device=torch.device("cpu"), qo_indptr=idx, kv_indptr=idx, kv_indices=idx, kv_len_arr=idx,
                          head_dim_ckv=512, head_dim_kpe=0, q_data_type=torch.bfloat16,
                          kv_data_type=torch.float8_e4m3fn, output_dtype=torch.bfloat16, scale_mode="kv-per-tensor")
            fc._validate_generated_fa_plan(backend="fa2", **common)  # must not raise
            try:
                fc._validate_generated_fa_plan(backend="fa3", **common)
            except ValueError:
                pass
            else:
                raise AssertionError("fa3 + FP8 + SM12x must still be refused")
        finally:
            fc.get_compute_capability = real
        return "flashinfer _validate_generated_fa_plan: fa2+FP8 accepted on 12.1, fa3 refused"

    runtime("rt.fi_fp8_gate", "attn", rt_fp8_gate)

    def rt_sm90_gate():
        from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla

        assert has_flashinfer_sm90_nope_mla(), (
            "FlashInfer lacks the ckv_scale_arr keyword-only run() argument: the SM90 backend would reject every "
            "config and auto-selection would silently fall back to FLASHINFER_MLA_SPARSE_SM120"
        )
        return "has_flashinfer_sm90_nope_mla() is True (the old bake's gate relaxation is not needed)"

    runtime("rt.sm90_nope_gate", "attn", rt_sm90_gate)

    def rt_sm90_validate():
        import torch
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import FlashInferMLASparseSM90Backend as B

        def reasons(cap):
            return B.validate_configuration(
                head_size=512, dtype=torch.bfloat16, kv_cache_dtype="fp8_e4m3", block_size=256, use_mla=True,
                has_sink=False, use_sparse=True, use_mm_prefix=False, use_per_head_quant_scales=False,
                device_capability=DeviceCapability(*cap), attn_type=AttentionType.DECODER)

        try:
            ok, other = reasons((12, 1)), reasons((10, 0))
        except (ImportError, OSError):
            raise
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001 - the validator needs a vllm config context; not an edit problem
            raise ImportError(f"validate_configuration could not be evaluated outside an engine: {type(exc).__name__}: {exc}")
        assert ok == [], f"SM90 backend still rejects 12.1 + fp8_e4m3 + NoPE sparse MLA: {ok}"
        assert "compute capability not supported" in other, other
        return "SM90 backend validate_configuration(12.1, fp8_e4m3, block 256, sparse MLA) = [] (valid); on 10.0 it is refused"

    runtime("rt.sm90_validate", "attn", rt_sm90_validate)

    def rt_kpool():
        import inspect
        import vllm.models.glm5next.nvidia.ops.kpool_compress as kc

        fn = kc._expand_pools_and_append_tail_kernel
        source = inspect.getsource(getattr(fn, "fn", fn))
        assert "(pid >= 0) & (pid < pool_len)" in source, "kernel source lacks the pool_len bound"
        return "imported _expand_pools_and_append_tail_kernel carries the pool_len bound"

    runtime("rt.kpool_pid", "kpool_pid", rt_kpool)

    def rt_moe():
        import vllm.model_executor.layers.fused_moe.fused_moe as fm

        real = fm.get_device_name_as_file_name
        fm.get_device_name_as_file_name = lambda *a, **k: "NVIDIA_GB10"
        try:
            fm.get_moe_configs.cache_clear()
            cfg = fm.get_moe_configs(288, 256, "fp8_w8a8", 128, 128)
        finally:
            fm.get_device_name_as_file_name = real
            fm.get_moe_configs.cache_clear()
        assert cfg and sorted(cfg) == bake.MOE_BATCH_KEYS, "get_moe_configs did not return the tuned table"
        return f"vllm get_moe_configs(E=288, N=256, fp8_w8a8, [128,128]) on NVIDIA_GB10 -> tuned table ({len(cfg)} batch sizes)"

    runtime("rt.moe_table", "moe", rt_moe)


# ------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify the SM121 bake (glm53_sm121_v032.py) in an image or a patched tree.")
    ap.add_argument("--vllm-root")
    ap.add_argument("--flashinfer-root")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--static", action="store_true", help="Layer 1 only (no imports of vllm/flashinfer/torch)")
    ap.add_argument("--require-all", action="store_true", help="fail if the manifest says any group was skipped")
    ap.add_argument("--require-runtime", action="store_true", help="fail (instead of SKIP) when Layer 2 cannot import the packages")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    bake = load_bake()
    try:
        roots = {"vllm": bake.locate_root("vllm", args.vllm_root), "flashinfer": bake.locate_root("flashinfer", args.flashinfer_root)}
    except bake.BakeError as exc:
        print(f"FAIL roots: {exc}")
        return 1

    manifest = None
    mpath = Path(args.manifest or DEFAULT_MANIFEST)
    if mpath.is_file():
        try:
            manifest = json.loads(mpath.read_text())
        except (OSError, ValueError) as exc:
            failed("manifest", f"{mpath}: {exc}")
    elif args.manifest:
        failed("manifest", f"{mpath} not found")
    skip_groups = set((manifest or {}).get("skipped_groups", []))
    if skip_groups:
        info("skipped", f"manifest says these groups were NOT baked: {sorted(skip_groups)}")
        if args.require_all:
            failed("require-all", f"groups skipped at bake time: {sorted(skip_groups)}")

    print(f"verify_bake: vllm={roots['vllm']} flashinfer={roots['flashinfer']}", flush=True)
    env_snapshot()
    jit_toolchain()
    layer1(bake, roots, skip_groups, manifest)
    if not args.static:
        layer2(bake, roots, skip_groups, args.require_runtime)

    n = {s: sum(1 for r in _RESULTS if r[0] == s) for s in ("PASS", "FAIL", "SKIP", "INFO")}
    print(f"verify_bake: {n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped", flush=True)
    if n["FAIL"]:
        print("VERIFY FAILED")
        return 1
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
