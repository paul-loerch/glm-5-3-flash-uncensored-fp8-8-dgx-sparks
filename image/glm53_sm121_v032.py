#!/usr/bin/env python3
"""SM121 bake for GLM-5.3-Flash on vLLM v0.32.0 + FlashInfer 0.7.0.post1 (GB10, arm64, CUDA 13).

Build-time source edits, applied once while the image is built.  No network, no GPU, no
imports of vllm / flashinfer / torch: the files are located with importlib's finder (which does
not execute the package) and edited as text.

Base image the anchors and the sha256 pins were written against:
    vllm/vllm-openai:v0.32.0-aarch64@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
    (vLLM commit 5df63a7203cd, flashinfer-python 0.7.0.post1)

Rules (same contract as the old community bake this replaces):
  * every edit must match exactly once, or the build fails (exit 1) and NOTHING is written
    (plan first, compile-check every patched .py, then commit);
  * a file that already carries an edit is detected (new text present once, old text absent)
    and left alone, so a re-run is a no-op; a half-patched file is an error;
  * the pristine sha256 of every edited file is pinned (fail closed; --allow-pin-mismatch
    overrides when you have checked the diff yourself);
  * what changed is printed (unified diff per file) and a manifest is written for verify_bake.py.

Edits (group: ids)                                   why
  attn:      pri12, sm90cap, sm90fa2                 vLLM runs the FLASHINFER_MLA_SPARSE_SM90 NoPE backend on
                                                     capability 12 through FlashInfer FA2 (FA3 is Hopper-only)
  attn:      fi_fp8_gate, fi_tile_cap, fi_uri        FlashInfer: FA2 FP8-KV batch MLA allowed on SM12x, FP8 KV
                                                     tile capped to what ~100 KB smem/SM holds, own JIT URI
  pdl:       pdl                                     programmatic dependent launch off on SM12x
  topk_neg1: topk_neg1_prefill, topk_neg1_decode     kpool pool top-k buffers start at -1
  kpool_pid: kpool_pid                               kpool expand keeps only pool ids < pool_len
  moe:       moe_table (data file)                   tuned Triton fused-MoE table E=288,N=256 for NVIDIA_GB10

Obsolete / not carried from the old bake (see IMAGE.md): the ckv_scale_arr feature-gate relaxation of
the SM90 backend (FlashInfer 0.7.0.post1 has the symbol), the FlashInfer package pin / jit-cache removal /
NCCL and cutlass-dsl re-pins (the new base ships consistent versions), and the unused "sm120" packed-path mode.

Credits.  The seven original steps are the work of MiaAI-Lab (glm53-flash_SM121.py in
MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark, MIT, Copyright (c) 2026 MiaAI-Lab; notice in
LICENSES/MiaAI-Lab.MIT.txt); anchors here are re-derived for v0.32.0 / FlashInfer 0.7.0.post1 and
cross-checked against the public port in SpaceshipCreative/glm-5.3-flash-spark (FlashInfer patches
0001, 0003, 0004 and vLLM patches 0003, 0006, 0009, 0010; Apache-2.0 repo, MIT parts from MiaAI-Lab).
The FA2-FP8 idea for GB10 also circulates in kindlingai/glm-5.3-flash-gx10 and
knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4.

usage: python3 -I glm53_sm121_v032.py [--vllm-root DIR] [--flashinfer-root DIR] [--moe-src DIR]
                                      [--skip GROUP[,GROUP...]] [--dry-run] [--list]
                                      [--check-pins] [--print-pins] [--allow-pin-mismatch]
                                      [--manifest PATH] [--diff-out PATH]
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

LOG = "[glm53-sm121-v032]"
BAKE_VERSION = 1
EXPECTED_BASE = (
    "vllm/vllm-openai:v0.32.0-aarch64@"
    "sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc"
)
EXPECTED_FLASHINFER = "0.7.0.post1"

DEFAULT_MANIFEST = "/opt/glm53-sm121/bake-manifest.json"

GROUPS = {
    "attn": "SM90 NoPE sparse-MLA backend on SM12x through FlashInfer FA2 (+ FP8 KV)",
    "pdl": "programmatic dependent launch off on SM12x",
    "topk_neg1": "kpool pool top-k buffers initialised to -1",
    "kpool_pid": "kpool expand: pool id must be < pool_len",
    "moe": "tuned Triton fused-MoE table E=288,N=256 for NVIDIA_GB10",
}


@dataclasses.dataclass(frozen=True)
class Edit:
    id: str
    group: str
    pkg: str  # "vllm" or "flashinfer": the root the path is relative to
    rel: str  # path relative to the package root
    old: str
    new: str
    what: str


# --------------------------------------------------------------------------------------------
# Pristine sha256 of every edited file (vLLM v0.32.0 tag tree == image; FlashInfer 0.7.0.post1
# wheel).  Key: (pkg, rel).  The MoE table pin is separate (MOE_*).
# --------------------------------------------------------------------------------------------
PINS = {
    ("vllm", "platforms/cuda.py"):
        "18ae1dfc34848afcade72821892db50fd3a2e3a57b48829521cc263e3f877caa",
    ("vllm", "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"):
        "480f2060f82313f32e149af00e8e988d01e37b95e96e45a696c07b25b7a32327",
    ("vllm", "models/glm5next/nvidia/sparse_indexer.py"):
        "cf659b6fdcfa1057ef6a61c3d31d302d2dd15dfea6f3c96743697664d87e3e44",
    ("vllm", "models/glm5next/nvidia/ops/kpool_compress.py"):
        "e5d1baa2f8fac4e490cc2f22a5ca05f67d4dea03a420efdae05f3a2bbeb6616f",
    ("flashinfer", "mla/_batch_mla/_backends/_fa_common.py"):
        "af9b5d07a6534d730ed2fc91d7005b8466b1961c1c85c9ccb621cc28e0f2c2a1",
    ("flashinfer", "data/include/flashinfer/attention/mla.cuh"):
        "57ea43d1c475f3cea19e24e3f1b2e0ea1c569e19cba422b16e242aecb298c6e0",
    ("flashinfer", "jit/attention/modules.py"):
        "747c92bd33ecc40297bddfd4b9d388e86e1e70aac5d2d71a37bdfa752c19cca6",
}

MOE_NAME = "E=288,N=256,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json"
MOE_SHA256 = "843799dee3cc1aada0f8a2e7b8645beb03c5013f1767ef15315cd8cf411add37"
MOE_DEST_REL = "model_executor/layers/fused_moe/configs/" + MOE_NAME  # below the vllm root
MOE_BATCH_KEYS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192]

# --------------------------------------------------------------------------------------------
# The edits.  Strings start at column 0 inside the triple quotes (they are file text).
# --------------------------------------------------------------------------------------------
EDITS = [
    # ---- vLLM: backend selection ------------------------------------------------------------
    Edit(
        id="pri12",
        group="attn",
        pkg="vllm",
        rel="platforms/cuda.py",
        what="capability-12 MLA backend priority: FLASHINFER_MLA_SPARSE_SM90 ahead of _SM120",
        old="""\
        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
""",
        new="""\
        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                # SM121 bake: the NoPE sparse-MLA backend on FlashInfer FA2 (plain
                # fp8_e4m3 KV, dense-MHA prefill) ahead of the packed-cache SM120 path.
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
""",
    ),
    Edit(
        id="sm90cap",
        group="attn",
        pkg="vllm",
        rel="v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py",
        what="FlashInferMLASparseSM90Backend.supports_compute_capability: major 9 -> (9, 12)",
        old="""\
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9
""",
        new="""\
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # SM121 bake: SM12x (GB10) runs the portable FA2 kernel behind the same
        # wrapper (see _SM90State); FA3 stays Hopper-only.
        return capability.major in (9, 12)
""",
    ),
    Edit(
        id="sm90fa2",
        group="attn",
        pkg="vllm",
        rel="v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py",
        what="_SM90State: BatchMLAPagedAttentionWrapper backend fa3 on SM90, fa2 elsewhere",
        old="""\
            backend="fa3",
""",
        new="""\
            # SM121 bake: FA3 (wgmma) is Hopper-only; FA2 behind the same wrapper is
            # portable to SM12x and shares MLAPlan/MLAPlanInfo with FA3, so the
            # sync-free plan and the kv_end clamp in this file apply to both.
            backend=(
                "fa3" if torch.cuda.get_device_capability(device)[0] == 9 else "fa2"
            ),
""",
    ),
    # ---- FlashInfer: FA2 FP8-KV batch MLA on SM12x ------------------------------------------
    Edit(
        id="fi_fp8_gate",
        group="attn",
        pkg="flashinfer",
        rel="mla/_batch_mla/_backends/_fa_common.py",
        what="plan-time validator: FP8 KV for MLA allowed on SM12x with the fa2 backend",
        old="""\
        major, minor = get_compute_capability(device)
        if major != 9:
            raise ValueError(
                "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "
                f"got SM{major}{minor}."
            )
""",
        new="""\
        major, minor = get_compute_capability(device)
        # SM121 bake: the FA2 kernel's FP8 path also builds and runs on SM12x (GB10)
        # once its KV tile fits ~100 KB of shared memory (see mla.cuh); FA3 is Hopper-only.
        if major != 9 and not (major == 12 and backend == "fa2"):
            raise ValueError(
                "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "
                "or an SM12x device with the fa2 backend, "
                f"got SM{major}{minor}."
            )
""",
    ),
    Edit(
        id="fi_tile_cap",
        group="attn",
        pkg="flashinfer",
        rel="data/include/flashinfer/attention/mla.cuh",
        what="FA2 MLA kernel: cap (not force) the FP8 KV tile at 32 so it fits SM12x smem",
        old="""\
    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? 32 : CTA_TILE_KV;
""",
        new="""\
    // SM121 bake: FP8 caps the KV tile at 32 instead of forcing it; below 147968 B of smem per SM
    // (SM12x, ~100 KB) DISPATCH_SMEM_CONFIG picks 16 and the forced 32-wide tile does not fit.
    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3>
                                             ? (CTA_TILE_KV < 32u ? CTA_TILE_KV : 32u)
                                             : CTA_TILE_KV;
""",
    ),
    Edit(
        id="fi_uri",
        group="attn",
        pkg="flashinfer",
        rel="jit/attention/modules.py",
        what="give the FA2 FP8-KV batch-MLA JIT module its own URI (_fp8tilecap)",
        old="""\
        f"profiler_{use_profiler}_planabi2"
    ) + ("_sm90" if backend == "fa3" else "")
""",
        new="""\
        f"profiler_{use_profiler}_planabi2"
    ) + ("_sm90" if backend == "fa3" else "") + (
        # SM121 bake: this module is built from the patched mla.cuh (FP8 tile cap).  A distinct
        # name keeps any prebuilt (AOT / jit-cache) or stale JIT artifact of the stock source
        # from ever being picked up for it.
        "_fp8tilecap"
        if backend == "fa2" and dtype_kv == torch.float8_e4m3fn
        else ""
    )
""",
    ),
    # ---- vLLM: PDL ----------------------------------------------------------------------------
    Edit(
        id="pdl",
        group="pdl",
        pkg="vllm",
        rel="platforms/cuda.py",
        what="CudaPlatformBase.is_arch_support_pdl: major >= 9 -> off for major 12",
        old="""\
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
""",
        new="""\
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        # SM121 bake: programmatic dependent launch is unvalidated on SM12x (GB10) and races in
        # the KDA recurrent-state, mamba conv and mHC kernels there; keep it off for major 12.
        return major >= 9 and major != 12
""",
    ),
    # ---- vLLM: kpool top-k hardening ---------------------------------------------------------
    Edit(
        id="topk_neg1_prefill",
        group="topk_neg1",
        pkg="vllm",
        rel="models/glm5next/nvidia/sparse_indexer.py",
        what="indexer prefill: pool_topk starts at -1 instead of torch.empty",
        old="""\
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
""",
        new="""\
                # SM121 bake: -1 (invalid) so a slot the top-k kernel leaves unwritten can
                # never expand to a garbage token index.
                pool_topk = torch.full(
                    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device
                )
""",
    ),
    Edit(
        id="topk_neg1_decode",
        group="topk_neg1",
        pkg="vllm",
        rel="models/glm5next/nvidia/sparse_indexer.py",
        what="indexer decode: pool_topk starts at -1 instead of torch.empty",
        old="""\
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
""",
        new="""\
            # SM121 bake: -1 (invalid) so a slot the top-k kernel leaves unwritten can
            # never expand to a garbage token index.
            pool_topk = torch.full(
                (num_rows, select_k), -1, dtype=torch.int32, device=logits.device
            )
""",
    ),
    Edit(
        id="kpool_pid",
        group="kpool_pid",
        pkg="vllm",
        rel="models/glm5next/nvidia/ops/kpool_compress.py",
        what="_expand_pools_and_append_tail_kernel: history pool id must be in [0, pool_len)",
        old="""\
    hist_out = tl.where(pid >= 0, hist_val, -1)
""",
        new="""\
    # SM121 bake: only complete pools [0, pool_len) are history; an id past them (stale or
    # unwritten top-k slot) would index tokens beyond the sequence or repeat the tail.
    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)
""",
    ),
]


def _self_check_table() -> None:
    ids = set()
    for e in EDITS:
        assert e.id not in ids, f"duplicate edit id {e.id}"
        ids.add(e.id)
        assert e.group in GROUPS, e
        assert e.old and e.new and e.old != e.new, e.id
        # the three-state logic (pending / applied / error) needs these to be unambiguous
        assert e.old not in e.new, f"{e.id}: old text is a substring of new text"
        assert e.new not in e.old, f"{e.id}: new text is a substring of old text"
        assert e.old.endswith("\n") and e.new.endswith("\n"), e.id
        assert (e.pkg, e.rel) in PINS, f"{e.id}: no pin for {e.pkg}/{e.rel}"
    # edits sharing a file must not contain each other's text, or the exactly-once counts lie
    for a in EDITS:
        for b in EDITS:
            if a is not b and (a.pkg, a.rel) == (b.pkg, b.rel):
                for x, xn in ((a.old, f"{a.id}.old"), (a.new, f"{a.id}.new")):
                    for y, yn in ((b.old, f"{b.id}.old"), (b.new, f"{b.id}.new")):
                        assert x not in y, f"{xn} is a substring of {yn}"


_self_check_table()


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
class BakeError(Exception):
    pass


def log(msg: str) -> None:
    print(f"{LOG} {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text(path: Path) -> tuple[str, bytes]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), raw
    except UnicodeDecodeError as exc:  # pragma: no cover - the pinned files are ASCII
        raise BakeError(f"{path}: not valid UTF-8 ({exc})") from exc


def locate_root(pkg: str, override: str | None) -> Path:
    """Directory of an installed package without importing it."""
    if override:
        root = Path(override)
    else:
        env = os.environ.get({"vllm": "GLM53_VLLM_ROOT", "flashinfer": "GLM53_FLASHINFER_ROOT"}[pkg])
        if env:
            root = Path(env)
        else:
            spec = importlib.util.find_spec(pkg)
            if spec is None or not spec.submodule_search_locations:
                raise BakeError(f"cannot locate the {pkg} package (pass --{pkg}-root)")
            root = Path(list(spec.submodule_search_locations)[0])
    if not root.is_dir():
        raise BakeError(f"{pkg} root {root} is not a directory")
    return root.resolve()


def package_version(root: Path, pkg: str) -> str | None:
    """Version string from the generated version module, read as text."""
    candidates = {"vllm": ["_version.py"], "flashinfer": ["_build_meta.py"]}[pkg]
    for name in candidates:
        p = root / name
        if p.is_file():
            m = re.search(r"""(?:__version__|version)\s*(?::\s*\w+\s*)?=\s*['"]([^'"]+)['"]""", p.read_text())
            if m:
                return m.group(1)
    return None


def drop_stale_pyc(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for f in cache.glob(path.stem + ".*.pyc"):
            try:
                f.unlink()
            except OSError:
                pass


def atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        try:
            os.chmod(tmp, path.stat().st_mode & 0o7777)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def line_of(text: str, snippet: str) -> int:
    return text.count("\n", 0, text.index(snippet)) + 1


# --------------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FilePlan:
    pkg: str
    rel: str
    path: Path
    before_text: str
    before_sha: str
    edits: list  # selected Edit objects touching this file
    status: dict  # edit id -> "pending" | "applied"
    after_text: str | None = None
    after_sha: str | None = None
    lines: dict = dataclasses.field(default_factory=dict)  # edit id -> line number (pre-edit)

    @property
    def state(self) -> str:
        s = set(self.status.values())
        if s == {"pending"}:
            return "pristine"
        if s == {"applied"}:
            return "baked"
        return "partial"


def plan_files(roots: dict, selected: list, *, allow_pin_mismatch: bool) -> list[FilePlan]:
    by_file: dict = {}
    for e in selected:
        by_file.setdefault((e.pkg, e.rel), []).append(e)

    plans: list[FilePlan] = []
    errors: list[str] = []
    for (pkg, rel), edits in by_file.items():
        path = roots[pkg] / rel
        if not path.is_file():
            errors.append(f"{pkg}/{rel}: file not found ({path})")
            continue
        text, raw = read_text(path)
        sha = sha256_bytes(raw)
        status = {}
        lines = {}
        for e in edits:
            n_old, n_new = text.count(e.old), text.count(e.new)
            if n_old == 1 and n_new == 0:
                status[e.id] = "pending"
                lines[e.id] = line_of(text, e.old)
            elif n_old == 0 and n_new == 1:
                status[e.id] = "applied"
                lines[e.id] = line_of(text, e.new)
            else:
                errors.append(
                    f"{e.id}: anchor in {pkg}/{rel} must match exactly once "
                    f"(old text x{n_old}, already-applied text x{n_new}); the base is not what "
                    f"this bake was written for"
                )
        plan = FilePlan(pkg, rel, path, text, sha, edits, status, lines=lines)
        if len(status) == len(edits):
            if plan.state == "partial":
                done = [i for i, s in status.items() if s == "applied"]
                todo = [i for i, s in status.items() if s == "pending"]
                errors.append(
                    f"{pkg}/{rel} is partially patched (applied: {done}; pending: {todo}); "
                    "restore the pristine file or finish by hand"
                )
            elif plan.state == "pristine":
                pin = PINS[(pkg, rel)]
                if sha != pin:
                    msg = (
                        f"{pkg}/{rel}: sha256 {sha} != pinned pristine {pin} "
                        "(anchors still match; the file differs elsewhere)"
                    )
                    if allow_pin_mismatch:
                        log(f"WARNING {msg} -- continuing (--allow-pin-mismatch)")
                    else:
                        errors.append(msg)
        plans.append(plan)

    if errors:
        raise BakeError("\n  ".join(["refusing to bake:"] + errors))

    # compute the patched text (in memory) and compile-check every patched python file
    for plan in plans:
        text = plan.before_text
        for e in plan.edits:
            if plan.status[e.id] == "pending":
                text = text.replace(e.old, e.new, 1)
        if text != plan.before_text:
            for e in plan.edits:
                if plan.status[e.id] == "pending":
                    if text.count(e.new) != 1 or text.count(e.old) != 0:
                        raise BakeError(f"{e.id}: result check failed in {plan.pkg}/{plan.rel}")
            if plan.rel.endswith(".py"):
                try:
                    compile(text, f"{plan.pkg}/{plan.rel}", "exec", dont_inherit=True)
                except SyntaxError as exc:
                    raise BakeError(f"{plan.pkg}/{plan.rel}: patched text does not compile: {exc}") from exc
            plan.after_text = text
            plan.after_sha = sha256_bytes(text.encode("utf-8"))
        else:
            plan.after_text = text
            plan.after_sha = plan.before_sha
    return plans


def plan_moe(vllm_root: Path, moe_src: Path) -> dict:
    src = moe_src / MOE_NAME
    dest = vllm_root / MOE_DEST_REL
    if not src.is_file():
        raise BakeError(f"moe: source table not found: {src}")
    src_bytes = src.read_bytes()
    if sha256_bytes(src_bytes) != MOE_SHA256:
        raise BakeError(f"moe: {src} sha256 {sha256_bytes(src_bytes)} != pinned {MOE_SHA256}")
    cfg = json.loads(src_bytes)
    keys = sorted(int(k) for k in cfg if k != "triton_version")
    if keys != MOE_BATCH_KEYS:
        raise BakeError(f"moe: unexpected batch-size keys {keys}")
    if not dest.parent.is_dir():
        raise BakeError(f"moe: {dest.parent} does not exist (is this vLLM v0.32.0?)")
    if dest.exists():
        if sha256_bytes(dest.read_bytes()) == MOE_SHA256:
            state = "already"
        else:
            raise BakeError(f"moe: {dest} exists with different content; refusing to overwrite")
    else:
        state = "pending"
    return {"src": src, "dest": dest, "bytes": src_bytes, "state": state}


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="glm53_sm121_v032.py",
        description="SM121 bake for GLM-5.3-Flash on vLLM v0.32.0 + FlashInfer 0.7.0.post1.",
    )
    ap.add_argument("--vllm-root", help="path of the installed vllm package (default: located by importlib)")
    ap.add_argument("--flashinfer-root", help="path of the installed flashinfer package")
    ap.add_argument("--moe-src", default=str(Path(__file__).resolve().parent / "moe"),
                    help="directory holding the tuned MoE table (default: ./moe next to this script)")
    ap.add_argument("--skip", default="", help="comma-separated groups to leave out: " + ", ".join(GROUPS))
    ap.add_argument("--dry-run", action="store_true", help="plan, print the diff, write nothing")
    ap.add_argument("--list", action="store_true", help="list edits and groups, then exit")
    ap.add_argument("--check-pins", action="store_true",
                    help="only compare the target files with the pinned pristine sha256 (no writes)")
    ap.add_argument("--print-pins", action="store_true", help="print the pins in pins.txt format, then exit")
    ap.add_argument("--allow-pin-mismatch", action="store_true",
                    help="do not fail when a pristine file's sha256 differs from the pin (anchors must still match)")
    ap.add_argument("--manifest", default=None, help=f"manifest path (default {DEFAULT_MANIFEST}; written unless --dry-run)")
    ap.add_argument("--diff-out", default=None, help="also write the unified diff of all edited files here")
    return ap.parse_args(argv)


def do_list() -> int:
    for g, d in GROUPS.items():
        print(f"group {g:10s} {d}")
    print()
    for e in EDITS:
        print(f"  {e.id:18s} [{e.group:9s}] {e.pkg}/{e.rel}\n  {'':18s} {e.what}")
    print(f"  {'moe_table':18s} [moe      ] vllm/{MOE_DEST_REL}\n  {'':18s} install the tuned Triton MoE table (sha256 {MOE_SHA256[:12]}...)")
    return 0


def do_print_pins() -> int:
    ids: dict = {}
    for e in EDITS:
        ids.setdefault((e.pkg, e.rel), []).append(e.id)
    for (pkg, rel), sha in PINS.items():
        print(f"{sha}  {pkg}/{rel}  sm121-bake:{','.join(ids[(pkg, rel)])}")
    print(f"{MOE_SHA256}  vllm/{MOE_DEST_REL}  sm121-bake:moe_table")
    return 0


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        return do_list()
    if args.print_pins:
        return do_print_pins()

    skip = {g.strip() for g in args.skip.split(",") if g.strip()}
    unknown = skip - set(GROUPS)
    if unknown:
        print(f"{LOG} unknown group(s) in --skip: {sorted(unknown)}; known: {sorted(GROUPS)}", file=sys.stderr)
        return 2

    try:
        roots = {
            "vllm": locate_root("vllm", args.vllm_root),
            "flashinfer": locate_root("flashinfer", args.flashinfer_root),
        }
        versions = {pkg: package_version(root, pkg) for pkg, root in roots.items()}
        log(f"vllm {roots['vllm']} (version {versions['vllm'] or 'unknown'})")
        log(f"flashinfer {roots['flashinfer']} (version {versions['flashinfer'] or 'unknown'})")
        if versions["flashinfer"] and versions["flashinfer"] != EXPECTED_FLASHINFER:
            msg = f"flashinfer {versions['flashinfer']} != {EXPECTED_FLASHINFER} this bake was written for"
            if args.allow_pin_mismatch:
                log(f"WARNING {msg}")
            else:
                raise BakeError(msg)
        selected = [e for e in EDITS if e.group not in skip]

        if args.check_pins:
            bad = 0
            for (pkg, rel), pin in PINS.items():
                if not any(e.pkg == pkg and e.rel == rel for e in selected):
                    continue
                p = roots[pkg] / rel
                if not p.is_file():
                    log(f"PIN MISSING {pkg}/{rel}")
                    bad += 1
                    continue
                got = sha256_bytes(p.read_bytes())
                state = "OK" if got == pin else "DIFFERS"
                if got != pin:
                    text = p.read_text()
                    done = sum(1 for e in selected if e.pkg == pkg and e.rel == rel and text.count(e.new) == 1)
                    if done:
                        state = f"BAKED ({done} edit(s) already present)"
                    else:
                        bad += 1
                log(f"pin {state:36s} {pkg}/{rel} sha256={got}")
            return 1 if bad else 0

        plans = plan_files(roots, selected, allow_pin_mismatch=args.allow_pin_mismatch)
        moe = None
        if "moe" not in skip:
            moe = plan_moe(roots["vllm"], Path(args.moe_src))
    except BakeError as exc:
        print(f"{LOG} FAILED: {exc}", file=sys.stderr, flush=True)
        return 1

    # ---- report ---------------------------------------------------------------------------
    n_applied = n_already = 0
    diff_all: list = []
    for plan in plans:
        for e in plan.edits:
            st = plan.status[e.id]
            if st == "pending":
                n_applied += 1
                log(f"{'WOULD APPLY' if args.dry_run else 'APPLY'} {e.id:18s} {plan.pkg}/{plan.rel}:{plan.lines[e.id]}  {e.what}")
            else:
                n_already += 1
                log(f"already     {e.id:18s} {plan.pkg}/{plan.rel}:{plan.lines[e.id]}  {e.what}")
        if plan.after_text != plan.before_text:
            diff = list(difflib.unified_diff(
                plan.before_text.splitlines(True), plan.after_text.splitlines(True),
                fromfile=f"a/{plan.pkg}/{plan.rel}", tofile=f"b/{plan.pkg}/{plan.rel}", n=2))
            diff_all.extend(diff)
            for line in diff:
                print(line, end="" if line.endswith("\n") else "\n")
    if moe is not None:
        log(f"{'already     ' if moe['state'] == 'already' else ('WOULD INSTALL' if args.dry_run else 'INSTALL')} "
            f"moe_table          vllm/{MOE_DEST_REL}  (sha256 {MOE_SHA256[:12]}..., {len(MOE_BATCH_KEYS)} batch sizes)")
    if skip:
        log(f"skipped groups: {sorted(skip)}")
    if args.diff_out and diff_all:
        Path(args.diff_out).write_text("".join(diff_all))
        log(f"diff written to {args.diff_out}")

    if args.dry_run:
        log(f"DRY RUN: would apply {n_applied}, already present {n_already}; nothing written")
        return 0

    # ---- commit ---------------------------------------------------------------------------
    written = []
    try:
        for plan in plans:
            if plan.after_text != plan.before_text:
                atomic_write(plan.path, plan.after_text.encode("utf-8"))
                written.append(f"{plan.pkg}/{plan.rel}")
                if plan.rel.endswith(".py"):
                    drop_stale_pyc(plan.path)
        if moe is not None and moe["state"] == "pending":
            moe["dest"].write_bytes(moe["bytes"])
            os.chmod(moe["dest"], 0o644)
            written.append(f"vllm/{MOE_DEST_REL}")
        # post-commit self-verification
        for plan in plans:
            text, _ = read_text(plan.path)
            for e in plan.edits:
                if text.count(e.new) != 1 or text.count(e.old) != 0:
                    raise BakeError(f"post-write check failed for {e.id} in {plan.pkg}/{plan.rel}")
        if moe is not None and sha256_bytes(moe["dest"].read_bytes()) != MOE_SHA256:
            raise BakeError("post-write check failed for moe_table")
    except (OSError, BakeError) as exc:
        print(f"{LOG} FAILED while writing ({exc}); already written: {written}; re-run to resume "
              "(the bake is idempotent per file)", file=sys.stderr, flush=True)
        return 1

    manifest_path = Path(args.manifest or DEFAULT_MANIFEST)
    manifest = {
        "bake": "glm53-sm121-v032",
        "bake_version": BAKE_VERSION,
        "expected_base": EXPECTED_BASE,
        "versions": versions,
        "skipped_groups": sorted(skip),
        "edits": [
            {"id": e.id, "group": e.group, "file": f"{plan.pkg}/{plan.rel}",
             "line": plan.lines[e.id], "status": plan.status[e.id]}
            for plan in plans for e in plan.edits
        ],
        "files": {
            f"{plan.pkg}/{plan.rel}": {"sha256_before": plan.before_sha, "sha256_after": plan.after_sha}
            for plan in plans
        },
        "moe": None if moe is None else {"file": f"vllm/{MOE_DEST_REL}", "sha256": MOE_SHA256, "state": moe["state"]},
    }
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        log(f"manifest written to {manifest_path}")
    except OSError as exc:
        if args.manifest:
            print(f"{LOG} FAILED: cannot write manifest {manifest_path}: {exc}", file=sys.stderr)
            return 1
        log(f"WARNING cannot write default manifest {manifest_path}: {exc}")

    log(f"DONE: applied={n_applied} already={n_already} files_written={len(written)} "
        f"skipped_groups={sorted(skip) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
