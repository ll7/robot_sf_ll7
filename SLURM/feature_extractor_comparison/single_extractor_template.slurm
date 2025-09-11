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

# Run single extractor training
echo "Starting training with %EXTRACTOR_NAME% extractor..."
python -c "
import sys
sys.path.append('.')
from scripts.multi_extractor_training import MultiExtractorTraining
from robot_sf.feature_extractors.config import FeatureExtractorPresets

# Get the specific extractor
extractor_map = {
    'dynamics_original': FeatureExtractorPresets.dynamics_original(),
    'dynamics_no_conv': FeatureExtractorPresets.dynamics_no_conv(),
    'mlp_small': FeatureExtractorPresets.mlp_small(),
    'mlp_large': FeatureExtractorPresets.mlp_large(),
    'attention_small': FeatureExtractorPresets.attention_small(),
    'attention_large': FeatureExtractorPresets.attention_large(),
    'lightweight_cnn': FeatureExtractorPresets.lightweight_cnn(),
}

extractor_name = '%EXTRACTOR_NAME%'
if extractor_name not in extractor_map:
    print(f'Unknown extractor: {extractor_name}')
    sys.exit(1)

# Create training manager with full parameters for single run
trainer = MultiExtractorTraining(
    output_dir=f'./results/single_extractor_{extractor_name}',
    n_envs=16,  # More environments for single run
    total_timesteps=2_000_000,  # More timesteps for single run
    eval_freq=25_000,
    save_freq=100_000,
    n_eval_episodes=10,
)

# Train only this extractor
results = trainer.run_comparison({extractor_name: extractor_map[extractor_name]})
print(f'Training completed for {extractor_name}!')
"

echo "============================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"