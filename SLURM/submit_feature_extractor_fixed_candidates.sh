#!/usr/bin/env bash
# Submit fixed feature-extractor validation candidates as one SLURM array.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CANDIDATE_FILE="configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml"
STUDY_NAME="feat_extractor_12m_hardening_$(date -u +%Y%m%dT%H%M%S)"
STORAGE=""
CONFIG=""
PARTITION=""
ACCOUNT=""
QOS=""
NICE=""
BEGIN=""
SLURM_TIME="13:00:00"
JOB_NAME="feat_12m_fixed"
ARRAY_CONCURRENCY=2
GPUS=1
CPUS=8
MEM="32G"
DISABLE_WANDB=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate-file) CANDIDATE_FILE="$2"; shift 2 ;;
    --study-name) STUDY_NAME="$2"; shift 2 ;;
    --storage) STORAGE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --nice) NICE="$2"; shift 2 ;;
    --begin) BEGIN="$2"; shift 2 ;;
    --time) SLURM_TIME="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --array-concurrency) ARRAY_CONCURRENCY="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --disable-wandb) DISABLE_WANDB="--disable-wandb"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$STORAGE" ]]; then
  SAFE_NAME="${STUDY_NAME//[^a-zA-Z0-9_-]/_}"
  mkdir -p output/optuna/feat_extractor
  STORAGE="sqlite:///output/optuna/feat_extractor/${SAFE_NAME}.db"
fi

COUNT=$(uv run python scripts/training/fixed_feature_extractor_candidates.py \
  --candidate-file "$CANDIDATE_FILE" \
  --candidate-index 0 \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --print-count)

if [[ "$COUNT" -lt 1 ]]; then
  echo "Candidate file has no candidates: $CANDIDATE_FILE" >&2
  exit 1
fi

mkdir -p output/slurm

PYTHON_CMD="uv run python scripts/training/fixed_feature_extractor_candidates.py"
PYTHON_CMD+=" --candidate-file $CANDIDATE_FILE"
PYTHON_CMD+=" --candidate-index \$SLURM_ARRAY_TASK_ID"
PYTHON_CMD+=" --study-name $STUDY_NAME"
PYTHON_CMD+=" --storage $STORAGE"
PYTHON_CMD+=" --log-level WARNING"
[[ -n "$CONFIG" ]] && PYTHON_CMD+=" --config $CONFIG"
[[ -n "$DISABLE_WANDB" ]] && PYTHON_CMD+=" $DISABLE_WANDB"

SBATCH_FLAGS=(
  "--time=$SLURM_TIME"
  "--cpus-per-task=$CPUS"
  "--mem=$MEM"
  "--gres=gpu:$GPUS"
  "--output=output/slurm/${JOB_NAME}_%A_%a.out"
  "--error=output/slurm/${JOB_NAME}_%A_%a.err"
)
[[ -n "$PARTITION" ]] && SBATCH_FLAGS+=("--partition=$PARTITION")
[[ -n "$ACCOUNT" ]] && SBATCH_FLAGS+=("--account=$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_FLAGS+=("--qos=$QOS")
[[ -n "$NICE" ]] && SBATCH_FLAGS+=("--nice=$NICE")
[[ -n "$BEGIN" ]] && SBATCH_FLAGS+=("--begin=$BEGIN")

ARRAY_SPEC="0-$((COUNT - 1))%${ARRAY_CONCURRENCY}"
CMD=(sbatch "--job-name=$JOB_NAME" "--array=$ARRAY_SPEC" "${SBATCH_FLAGS[@]}" --wrap "$PYTHON_CMD")

echo "Study:          $STUDY_NAME"
echo "Storage:        $STORAGE"
echo "Candidate file: $CANDIDATE_FILE"
echo "Array:          $ARRAY_SPEC"
echo "Time:           $SLURM_TIME"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] ${CMD[*]}"
else
  "${CMD[@]}"
  echo ""
  echo "Monitor: uv run python scripts/tools/inspect_optuna_db.py --db ${STORAGE#sqlite:///} --study-name $STUDY_NAME --show-params --top-n $COUNT"
fi
