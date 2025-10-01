#!/bin/bash
#SBATCH --job-name=feature_extractor_comparison
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/feature_comparison_%j.out
#SBATCH --error=slurm_logs/feature_comparison_%j.err

# SLURM script for comprehensive feature extractor comparison
# This script runs the complete comparison with all extractors

echo "Starting feature extractor comparison job..."
echo "Job ID: $SLURM_JOB_ID"
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
mkdir -p results/feature_extractor_comparison

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment (adjust path as needed)
source .venv/bin/activate

# Install dependencies if needed (uncomment if first run)
# uv sync

# Set display variables for headless operation
export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"

# Run the complete comparison
echo "Starting multi-extractor training comparison..."
python scripts/multi_extractor_training.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Run analysis if results are available
    RESULTS_FILE="results/feature_extractor_comparison/complete_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "Running statistical analysis..."
        python scripts/analyze_feature_extractors.py "$RESULTS_FILE"
        
        if [ $? -eq 0 ]; then
            echo "Analysis completed successfully!"
        else
            echo "Analysis failed with exit code $?"
        fi
    else
        echo "Results file not found: $RESULTS_FILE"
    fi
else
    echo "Training failed with exit code $?"
fi

echo "============================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"