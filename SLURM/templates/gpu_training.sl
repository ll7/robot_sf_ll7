#!/usr/bin/env bash
# Public-safe Slurm template for a config-driven GPU training job.
#
# Copy this file into a cluster-specific private overlay and fill in the account,
# partition, QoS, module, and scratch policy there. Keep private hostnames and
# local filesystem details out of this public template.

#SBATCH --job-name=robot-sf-gpu-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-gpu-training.out

set -euo pipefail

PROJECT_ROOT="$(
  git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null \
    || echo "${SLURM_SUBMIT_DIR:-$(pwd)}"
)"
TRAIN_CONFIG="${ROBOT_SF_TRAIN_CONFIG:-}"

if [[ -z "${TRAIN_CONFIG}" ]]; then
  echo "ROBOT_SF_TRAIN_CONFIG is required." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

exec .venv/bin/python scripts/training/train_ppo.py --config "${TRAIN_CONFIG}"
