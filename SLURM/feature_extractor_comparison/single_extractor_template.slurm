#!/bin/bash
#SBATCH --job-name=single_extractor_%EXTRACTOR_NAME%
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/single_extractor_%EXTRACTOR_NAME%_%j.out
#SBATCH --error=slurm_logs/single_extractor_%EXTRACTOR_NAME%_%j.err

# SLURM script for training a single feature extractor
# This template is used to submit individual extractor training jobs

echo "Starting single extractor training job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Extractor: %EXTRACTOR_NAME%"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "============================================"

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8
# module load gcc/9.3.0

# Set environment variables
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Create output directories
mkdir -p slurm_logs
mkdir -p results/single_extractor_%EXTRACTOR_NAME%

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source .venv/bin/activate

# Set display variables for headless operation
export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"

# Run single extractor training with the new CLI
echo "Starting training with %EXTRACTOR_NAME% extractor..."

OUTPUT_ROOT="results/single_extractor_%EXTRACTOR_NAME%"
CONFIG_PATH="$OUTPUT_ROOT/single_extractor_config.yaml"

mkdir -p "$OUTPUT_ROOT"

cat <<EOF > "$CONFIG_PATH"
run:
  run_id: "single-%EXTRACTOR_NAME%"
  worker_mode: "vectorized"
  num_envs: 8
  total_timesteps: 2000000
  eval_freq: 25000
  save_freq: 100000
  n_eval_episodes: 10
  device: "cuda"
  seed: 0
extractors:
  - name: "%EXTRACTOR_NAME%"
    preset: "%EXTRACTOR_NAME%"
    expected_resources: "gpu"
EOF

unset ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE

python scripts/multi_extractor_training.py \
  --config "$CONFIG_PATH" \
  --run-id "single-%EXTRACTOR_NAME%" \
  --output-root "$OUTPUT_ROOT"

STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "Training failed with exit code $STATUS"
  exit $STATUS
fi

echo "Training completed for %EXTRACTOR_NAME%!"

echo "============================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
