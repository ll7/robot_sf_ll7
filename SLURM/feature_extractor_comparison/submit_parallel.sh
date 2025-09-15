#!/bin/bash

# Submit individual jobs for each feature extractor
# This allows parallel training and better resource utilization

EXTRACTORS=("dynamics_original" "dynamics_no_conv" "mlp_small" "mlp_large" "attention_small" "lightweight_cnn")

echo "Submitting individual feature extractor jobs..."
echo "Extractors: ${EXTRACTORS[@]}"
echo "============================================"

mkdir -p slurm_logs

JOB_IDS=()

for extractor in "${EXTRACTORS[@]}"; do
    echo "Submitting job for $extractor..."
    
    # Create extractor-specific SLURM script from template
    extractor_script="SLURM/feature_extractor_comparison/single_extractor_${extractor}.slurm"
    sed "s/%EXTRACTOR_NAME%/$extractor/g" \
        SLURM/feature_extractor_comparison/single_extractor_template.slurm > "$extractor_script"
    
    # Submit the job and capture job ID
    job_output=$(sbatch "$extractor_script")
    job_id=$(echo "$job_output" | awk '{print $4}')
    JOB_IDS+=("$job_id")
    
    echo "Submitted job $job_id for $extractor"
    
    # Clean up temporary script
    rm "$extractor_script"
done

echo "============================================"
echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[@]}"

# Submit analysis job that depends on all training jobs
echo "Submitting analysis job..."
dependency_list=$(IFS=:; echo "${JOB_IDS[*]}")
analysis_job_output=$(sbatch --dependency=afterany:$dependency_list \
    SLURM/feature_extractor_comparison/analyze_results.slurm)
analysis_job_id=$(echo "$analysis_job_output" | awk '{print $4}')

echo "Analysis job submitted with ID: $analysis_job_id"
echo "Analysis will run after all training jobs complete"

echo "============================================"
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: slurm_logs/"