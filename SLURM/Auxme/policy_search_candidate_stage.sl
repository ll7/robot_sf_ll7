#!/usr/bin/env bash
#SBATCH --job-name=rsf-polsearch
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm/%x_%A_%a.out
#SBATCH --error=output/slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

STAGE="${POLICY_SEARCH_STAGE:-nominal_sanity}"
CANDIDATES_FILE="${POLICY_SEARCH_CANDIDATES_FILE:?POLICY_SEARCH_CANDIDATES_FILE is required}"
RUN_ID="${POLICY_SEARCH_RUN_ID:-slurm_${SLURM_JOB_ID:-manual}}"
WORKERS="${POLICY_SEARCH_WORKERS:-2}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script expects a Slurm array task id." >&2
  exit 2
fi

CANDIDATE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CANDIDATES_FILE")"
if [[ -z "$CANDIDATE" ]]; then
  echo "No candidate found for array index ${SLURM_ARRAY_TASK_ID} in ${CANDIDATES_FILE}" >&2
  exit 2
fi

echo "[policy-search] repo=$REPO_ROOT"
echo "[policy-search] run_id=$RUN_ID stage=$STAGE candidate=$CANDIDATE workers=$WORKERS"
echo "[policy-search] candidates_file=$CANDIDATES_FILE"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate "$CANDIDATE" \
  --stage "$STAGE" \
  --allow-expensive-stage \
  --workers "$WORKERS" \
  --output-dir "output/policy_search/${CANDIDATE}/${STAGE}/${RUN_ID}"
