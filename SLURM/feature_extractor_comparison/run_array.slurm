#!/bin/bash
#SBATCH --job-name=feature_extractor_array
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/feature_extractor_array_%A_%a.out
#SBATCH --error=slurm_logs/feature_extractor_array_%A_%a.err

# SLURM script for feature-extractor comparison jobs submitted as an array.
# Each array task trains exactly one extractor and writes to the same
# result directory layout used by the legacy per-job workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./extractors.sh
source "$SCRIPT_DIR/extractors.sh"

if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
  echo "This script must be submitted as a Slurm array job." >&2
  exit 1
fi

TASK_ID="$SLURM_ARRAY_TASK_ID"
if (( TASK_ID < 0 || TASK_ID >= ${#EXTRACTORS[@]} )); then
  echo "Array task id $TASK_ID is out of range for ${#EXTRACTORS[@]} extractors." >&2
  exit 1
fi

EXTRACTOR_NAME="${EXTRACTORS[$TASK_ID]}"

echo "Starting feature extractor array job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $TASK_ID"
echo "Extractor: $EXTRACTOR_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "============================================"

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"

mkdir -p slurm_logs

echo "Starting training with $EXTRACTOR_NAME extractor..."

OUTPUT_ROOT="results/single_extractor_${EXTRACTOR_NAME}"
CONFIG_PATH="$OUTPUT_ROOT/single_extractor_config.yaml"

mkdir -p "$OUTPUT_ROOT"

cat <<EOF > "$CONFIG_PATH"
run:
  run_id: "single-${EXTRACTOR_NAME}"
  worker_mode: "vectorized"
  num_envs: 8
  total_timesteps: 2000000
  eval_freq: 25000
  save_freq: 100000
  n_eval_episodes: 10
  device: "cuda"
  seed: 0
extractors:
  - name: "${EXTRACTOR_NAME}"
    preset: "${EXTRACTOR_NAME}"
    expected_resources: "gpu"
EOF

unset ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE

python scripts/multi_extractor_training.py \
  --config "$CONFIG_PATH" \
  --run-id "single-${EXTRACTOR_NAME}" \
  --output-root "$OUTPUT_ROOT"

echo "Training completed for $EXTRACTOR_NAME!"
echo "============================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
