#!/bin/bash
# Single-node release-campaign submission (the citable-release path).
#
# The release runner (scripts/tools/run_benchmark_release.py) must execute as ONE
# process to produce a valid release bundle -- it cannot consume split per-arm
# aggregates (see docs/context/release_0_0_3_slurm_execution_packet.md). This
# template runs the whole campaign on one fat node with a FIXED --campaign-id.
# A fresh launch omits the optional sixth argument. Resubmitting the same campaign
# after an infrastructure interruption requires an immutable resume receipt; code,
# config, dependency, or checkpoint defects require a corrected commit and fresh id.
#
# Usage (from the repo checkout on the cluster, after `uv sync --all-extras` and
# a passing checkpoint staging preflight):
#   sbatch SLURM/submit_release_single_node.sbatch \
#     configs/benchmarks/releases/<release_manifest>.yaml <label> <fixed_campaign_id> \
#     output/release/checkpoints/staging_receipt.json \
#     output/benchmarks/camera_ready/<smoke_id>/release/release_result.json
# Infrastructure-only resume adds:
#     output/release/resume/<campaign_id>.json
#
# Field-tested: 14-arm x 30-seed x h600 campaign, epyc-gpu partition, LiCCA.
#SBATCH --job-name=rsf_release_single_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=output/slurm-release-%j.log
set -euo pipefail
MANIFEST="${1:?release manifest path required}"
LABEL="${2:?release label required}"
CAMPAIGN_ID="${3:?fixed campaign id required}"
CHECKPOINT_RECEIPT="${4:?staged-checkpoint receipt path required}"
RUNTIME_SMOKE_RECEIPT="${5:?exact-source runtime-smoke result path required}"
RESUME_RECEIPT="${6:-}"
cd "$SLURM_SUBMIT_DIR"
# Thread guards: the campaign parallelizes via its own worker pool (config `workers`);
# per-worker BLAS/OpenMP threading only oversubscribes the node.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
runner_args=(
  --manifest "$MANIFEST"
  --label "$LABEL"
  --campaign-id "$CAMPAIGN_ID"
  --checkpoint-receipt "$CHECKPOINT_RECEIPT"
  --runtime-smoke-receipt "$RUNTIME_SMOKE_RECEIPT"
)
if [[ -n "$RESUME_RECEIPT" ]]; then
  runner_args+=(--resume-receipt "$RESUME_RECEIPT")
fi
exec .venv/bin/python scripts/tools/run_benchmark_release.py "${runner_args[@]}"
