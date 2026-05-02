#!/bin/bash

# Submit the feature-extractor comparison as a Slurm job array.
# Each array task trains one extractor, which keeps the existing per-extractor
# result layout but avoids creating one standalone sbatch job per extractor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=./extractors.sh
source "$SCRIPT_DIR/extractors.sh"

ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"

echo "Submitting feature extractor job array..."
echo "Extractors: ${EXTRACTORS[*]}"
echo "Array concurrency cap: ${ARRAY_CONCURRENCY}"
echo "============================================"

mkdir -p slurm_logs

array_end=$(( ${#EXTRACTORS[@]} - 1 ))
job_output=$(sbatch --array="0-${array_end}%${ARRAY_CONCURRENCY}" "$SCRIPT_DIR/run_array.slurm")
array_job_id=$(echo "$job_output" | awk '{print $4}')

echo "Submitted array job: ${array_job_id}"
echo "Array tasks: 0-${array_end}"

echo "Submitting analysis job..."
analysis_job_output=$(sbatch --dependency=afterany:"$array_job_id" \
  "$SCRIPT_DIR/analyze_results.slurm")
analysis_job_id=$(echo "$analysis_job_output" | awk '{print $4}')

echo "Analysis job submitted with ID: $analysis_job_id"
echo "Analysis will run after the array completes"

echo "============================================"
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: slurm_logs/"
