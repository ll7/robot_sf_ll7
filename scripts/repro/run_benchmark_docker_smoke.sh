#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE_TAG="${ROBOT_SF_REPRO_IMAGE:-robot-sf-benchmark-repro:py312.3-uv0.11.9}"
OUTPUT_ROOT="${ROBOT_SF_REPRO_OUTPUT_ROOT:-output/docker_repro/benchmark_bundle_smoke}"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"

docker build \
  -f docker/benchmark-repro.Dockerfile \
  -t "$IMAGE_TAG" \
  .

mkdir -p "$(dirname "$OUTPUT_ROOT")"

docker run --rm \
  -e ROBOT_SF_REPRO_GIT_COMMIT="$GIT_COMMIT" \
  -e ROBOT_SF_REPRO_OUTPUT_ROOT="$OUTPUT_ROOT" \
  -v "$ROOT_DIR/output/docker_repro:/workspace/robot_sf_ll7/output/docker_repro" \
  "$IMAGE_TAG"

echo "Inspect artifacts under $OUTPUT_ROOT"
