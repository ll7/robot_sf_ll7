#!/bin/bash
#SBATCH --job-name=ppo_grid_403
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=standard

# Optional GPU request (uncomment if you have GPU nodes)
#SBATCH --gres=gpu:1

set -euo pipefail

echo "Job started at $(date)"
echo "Hostname: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

source .venv/bin/activate

export SDL_VIDEODRIVER=dummy
export ROBOT_SF_ARTIFACT_ROOT="${ROBOT_SF_ARTIFACT_ROOT:-$REPO_ROOT/output}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Artifact root: $ROBOT_SF_ARTIFACT_ROOT"
echo "W&B mode: $WANDB_MODE"

uv run python scripts/training/train_expert_ppo.py \
  --config configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml

echo "Job finished at $(date)"
