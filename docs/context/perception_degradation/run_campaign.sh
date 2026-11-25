#!/usr/bin/env bash
# Run every condition of one stage (smoke | pilot | main) from the repository root.
# Usage: docs/context/perception_degradation/run_campaign.sh <stage> <output-root>
# Requires: the execution commit checked out, `uv sync --frozen --extra training`, and the
# PPO checkpoint ppo_release_robot_b1002_last_20261001 (sha256 764a7d88...24bc) in
# output/model_cache/ (the registry's GitHub-release asset).
set -euo pipefail
stage="$1"
out_root="$2"
packet="docs/context/perception_degradation"
mkdir -p "$out_root"
{
  echo "commit=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain --untracked-files=no | wc -l)"
  echo "python=$(uv run python -V 2>&1)"
  echo "uv_lock_sha256=$(sha256sum uv.lock | cut -d' ' -f1)"
  echo "nproc=$(nproc)"
  echo "started=$(date -u +%FT%TZ)"
} > "$out_root/environment.txt"
for condition in oracle noise delay occlusion; do
  if grep -q "^$condition EXIT=0$" "$out_root/exits.txt" 2>/dev/null; then
    continue  # finished condition from an interrupted invocation
  fi
  set +e
  uv run python scripts/tools/run_camera_ready_benchmark.py \
    --config "$packet/inputs/campaign_${stage}_${condition}.yaml" \
    --output-root "$out_root" --label "$condition" --log-level WARNING \
    > "$out_root/${condition}.log" 2>&1
  rc=$?
  set -e
  echo "$condition EXIT=$rc" >> "$out_root/exits.txt"
done
echo "finished=$(date -u +%FT%TZ)" >> "$out_root/environment.txt"
