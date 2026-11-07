#!/usr/bin/env bash
# [patch-020] p020_repin.sh <flashinfer_mla_sparse_sm90.py of the REAL image> [platforms/cuda.py of the real image]
# Run on any machine with bash + python3 (no torch) once the SM121 bake has produced the real image: it checks
# that the applier accepts the file (all five anchors match exactly once; a refusal exits non-zero), that revert
# is byte-exact, prints the backend probe, and prints the pins to paste into 020-gb10-sparse-mla-triton-prefill.sh
# (SM_REAL / SM_REAL_P020) and into pins.txt. Never modifies its inputs.
#   docker run --rm --entrypoint cat <image> /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py > /tmp/sm90.py
#   bash files/p020_repin.sh /tmp/sm90.py [/tmp/cuda.py]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
f="${1:-}"; cuda="${2:-}"
[ -f "$f" ] || { echo "usage: $0 <flashinfer_mla_sparse_sm90.py> [platforms/cuda.py]" >&2; exit 2; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cp "$f" "$tmp/sm90.py"
base="$(sha "$f")"
python3 "$DIR/p020_apply.py" apply "$tmp/sm90.py" || { echo "[repin] the applier REFUSED this file: an anchor drifted (see the message above); the hooks need a rework" >&2; exit 1; }
res="$(sha "$tmp/sm90.py")"
python3 -c "import sys; compile(open(sys.argv[1]).read(), 'patched', 'exec')" "$tmp/sm90.py" || { echo "[repin] the patched result does not compile" >&2; exit 1; }
python3 "$DIR/p020_apply.py" revert "$tmp/sm90.py"
[ "$(sha "$tmp/sm90.py")" = "$base" ] || { echo "[repin] revert is not byte-exact" >&2; exit 1; }
echo "[repin] probe (capability 12.1 accepted?, cuda.py lists the SM90 backend for capability 12?): $(python3 "$DIR/p020_apply.py" probe "$f" ${cuda:+"$cuda"})"
echo "[repin] paste into patches/020-gb10-sparse-mla-triton-prefill.sh:"
echo "SM_REAL=$base"
echo "SM_REAL_P020=$res"
echo "[repin] pins.txt lines:"
echo "$base  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py  020 (real baked base)"
echo "$res  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py  020 (patch-020 result on it)"
