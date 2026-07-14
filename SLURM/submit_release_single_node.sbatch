#!/bin/bash
# Single-node release-campaign submission (the citable-release path).
#
# The release runner (scripts/tools/run_benchmark_release.py) must execute as ONE
# process to produce a valid release bundle -- it cannot consume split per-arm
# aggregates (see docs/context/release_0_0_3_slurm_execution_packet.md). This
# template runs the whole campaign on one fat node with a FIXED --campaign-id so
# restarts (walltime kill, node failure, --requeue) resume in place instead of
# starting over. Requires the --campaign-id release-runner passthrough.
#
# Usage (from the repo checkout on the cluster, after `uv sync --all-extras` and
# a passing checkpoint staging preflight):
#   sbatch SLURM/submit_release_single_node.sbatch \
#     configs/benchmarks/releases/<release_manifest>.yaml <label> <fixed_campaign_id>
#
# Field-tested: 14-arm x 30-seed x h600 campaign, epyc-gpu partition, LiCCA.
#SBATCH --job-name=rsf_release_single_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --output=output/slurm-release-%j.log
set -euo pipefail
MANIFEST="${1:?release manifest path required}"
LABEL="${2:?release label required}"
CAMPAIGN_ID="${3:?fixed campaign id required (enables resume across restarts)}"
cd "$SLURM_SUBMIT_DIR"
# Thread guards: the campaign parallelizes via its own worker pool (config `workers`);
# per-worker BLAS/OpenMP threading only oversubscribes the node.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python scripts/tools/run_benchmark_release.py \
  --manifest "$MANIFEST" \
  --label "$LABEL" \
  --campaign-id "$CAMPAIGN_ID"
