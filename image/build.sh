#!/usr/bin/env bash
# image/build.sh [--tag TAG] [--distribute] [--gpu-smoke]
#
# Builds the serving image on this machine (run it on NODE1), checks the build-time edits, and optionally copies the
# image to the seven other nodes listed in ../cluster.env so that every node runs the identical image ID.
#
#   --tag TAG      image tag (default glm53-flash-sm121:v032-1, the tag config/recipe.env expects)
#   --distribute   docker save | ssh NODEn docker load, for NODE2..NODE8, then compare the image IDs
#   --gpu-smoke    also run gpu_smoke.py on this node's GPU (a few minutes the first time: JIT builds)
#
# Needs: docker; for --distribute, ../cluster.env (NODE1..NODE8, SSH_USER, SSH_OPTS) and docker on every node.
# The base image is pulled by digest (see the Dockerfile). Nothing else is downloaded at build time.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=glm53-flash-sm121:v032-1
DISTRIBUTE=0
SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift ;;
    --distribute) DISTRIBUTE=1 ;;
    --gpu-smoke) SMOKE=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

echo "== building $TAG"
docker build -t "$TAG" -f "$HERE/Dockerfile" "$HERE"

echo "== checking the build-time edits (no GPU needed)"
docker run --rm --entrypoint python3 "$TAG" -I /opt/glm53-sm121/verify_bake.py --static

if [ "$SMOKE" = 1 ]; then
  echo "== GPU smoke test on this node"
  docker run --rm --gpus all --entrypoint python3 "$TAG" -I /opt/glm53-sm121/gpu_smoke.py
fi

ID=$(docker image inspect "$TAG" --format '{{.Id}}')
echo "== built $TAG = $ID"
[ "$DISTRIBUTE" = 1 ] || exit 0

ENV_FILE=${CLUSTER_ENV:-$HERE/../cluster.env}
[ -r "$ENV_FILE" ] || { echo "no $ENV_FILE (copy cluster.env.example to cluster.env and fill it in)" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"
read -r -a SSH_ARGS <<< "${SSH_OPTS:-}"
target() { if [ -n "${SSH_USER:-}" ]; then echo "$SSH_USER@$1"; else echo "$1"; fi; }
fail=0
for i in 2 3 4 5 6 7 8; do
  var=NODE$i; node=${!var:-}
  [ -n "$node" ] && [ "$node" != CHANGE_ME ] || { echo "NODE$i is not set in $ENV_FILE" >&2; exit 2; }
  echo "== copying the image to $node (NODE$i)"
  docker save "$TAG" | ssh "${SSH_ARGS[@]}" "$(target "$node")" docker load
  got=$(ssh "${SSH_ARGS[@]}" "$(target "$node")" docker image inspect "$TAG" --format '{{.Id}}')
  if [ "$got" = "$ID" ]; then echo "   $node: $got"; else echo "   $node: $got != $ID" >&2; fail=1; fi
done
[ "$fail" = 0 ] && echo "== all eight nodes have $TAG = $ID" || { echo "image IDs differ: launch/up.sh will refuse" >&2; exit 1; }
