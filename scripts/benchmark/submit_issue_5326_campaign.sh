#!/usr/bin/env bash
# SLURM submission wrapper for issue #5326 temporal-robustness objective comparison.
# This script submits the matched-budget campaign comparing temporal_robustness vs worst_case_snqi.
#
# Usage:
#   ./scripts/benchmark/submit_issue_5326_campaign.sh [--dry-run]
#
# Prerequisites:
#   - PR #5325 merged (temporal_robustness objective available)
#   - PR #5714 merged (CMA-ES sampler wired)
#   - SLURM access on cluster
#
# Campaign contract:
#   - Objectives: worst_case_snqi, temporal_robustness
#   - Budgets: 16, 32, 64 (Package-B grid)
#   - Seeds: 1101, 2202, 3303
#   - Samplers: random, coordinate, optuna
#   - Output: output/adversarial/issue_5326_objective_comparison/

set -euo pipefail

DRY_RUN="${1:-}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${REPO_ROOT}/configs/adversarial/issue_5326_objective_comparison.yaml"
OUTPUT_DIR="${REPO_ROOT}/output/adversarial/issue_5326_objective_comparison"
RUNNER="${REPO_ROOT}/scripts/tools/compare_adversarial_samplers.py"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
    echo "ERROR: Runner script not found: ${RUNNER}" >&2
    exit 1
fi

# SLURM parameters tuned for adversarial search campaigns
# Each job runs one objective-sampler-budget-seed combination
SLURM_TIME="4:00:00"
SLURM_MEM="8G"
SLURM_CPUS="4"
SLURM_PARTITION="gpu"

COMMAND=(
    "uv" "run" "python" "${RUNNER}"
    "--objective" "worst_case_snqi"
    "--objective" "temporal_robustness"
    "--package-b-budget-grid"
    "--seed" "1101" "--seed" "2202" "--seed" "3303"
    "--output-dir" "${OUTPUT_DIR}"
    "--out-json" "${OUTPUT_DIR}/report.json"
)

if [[ -n "${DRY_RUN}" ]]; then
    echo "=== DRY RUN MODE ==="
    echo "Config: ${CONFIG_PATH}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Command:"
    printf '  %s\n' "${COMMAND[@]}"
    echo ""
    echo "To submit, run without --dry-run:"
    echo "  ./scripts/benchmark/submit_issue_5326_campaign.sh"
    exit 0
fi

# Submit as SLURM job array
# Each array element handles one objective-sampler-budget-seed tuple
# For simplicity, submit as single job that runs full grid
sbatch <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=issue_5326_obj_cmp
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --output=${OUTPUT_DIR}/slurm-%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm-%j.err

set -euo pipefail
cd "${REPO_ROOT}"
source .venv/bin/activate
echo "Starting issue #5326 objective comparison campaign"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Started at: \$(date -Iseconds)"

"${COMMAND[@]}"

echo "Completed at: \$(date -Iseconds)"
SLURM_EOF

echo "Submitted SLURM job for issue #5326 campaign"
echo "Output directory: ${OUTPUT_DIR}"
echo "Monitor with: sbatch --qos=debug --wrap 'squeue -u \$USER -o \"%.10i %.9P %.20j %.8T %.10M %.8l %.6D %.20S\"'"
