#!/usr/bin/env bash
# Submit a feature extractor Optuna sweep to SLURM.
#
# Each trial runs as a Slurm array task that picks up one Optuna trial from the
# shared SQLite database.  This is the standard distributed-Optuna pattern:
# N tasks share one study and the TPE sampler handles deduplication.
#
# Usage:
#   bash SLURM/submit_feature_extractor_sweep.sh [--trials N] [options]
#
# Examples:
#   # Full 4M-step sweep, 20 trials, GPU partition
#   bash SLURM/submit_feature_extractor_sweep.sh \
#     --trials 20 \
#     --timesteps 4000000 \
#     --partition gpu \
#     --study-name feat_sweep_4m
#
#   # Dry-run: just print the sbatch commands without submitting
#   bash SLURM/submit_feature_extractor_sweep.sh --dry-run --trials 5
#
# Options:
#   --config PATH           Base training config (default: configs/training/ppo/feature_extractor_sweep_base.yaml)
#   --trials N              Number of trials / SLURM array tasks (default: 20)
#   --timesteps N           Steps per trial (default: 4000000)
#   --metric NAME           Metric to maximise (default: eval_episode_return)
#   --study-name NAME       Optuna study name (default: feat_sweep_<timestamp>)
#   --storage URL           SQLite URL (default: output/optuna/feat_extractor/<study>.db)
#   --seed N                Base Optuna sampler seed (default: 42)
#   --time HH:MM:SS         SLURM time limit per job (default: 08:00:00)
#   --job-name NAME         Slurm job name for the array (default: sweep_4m_array)
#   --gpus N                GPUs per job (default: 1)
#   --cpus N                CPUs per task (default: 8)
#   --mem MEM               Memory per job (default: 32G)
#   --partition NAME        SLURM partition (default: not set)
#   --account NAME          SLURM account (default: not set)
#   --qos NAME              SLURM QoS (default: not set)
#   --nice N                SLURM nice value for lower priority (default: unset)
#   --begin TIME            Optional sbatch begin time for delayed starts
#   --worker-offset N       Add N to each worker-index (default: 0)
#   --array-concurrency N   Max number of array tasks running at once (default: 2)
#   --fps-warn FLOAT        FPS threshold for slow-candidate warning (default: 100)
#   --exclude TYPE[,TYPE]   Comma-separated extractor types to skip
#   --disable-wandb         Disable W&B in all worker jobs
#   --dry-run               Print sbatch commands; do not submit
#   -h, --help              Show this help

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ---- defaults ----------------------------------------------------------------
CONFIG="configs/training/ppo/feature_extractor_sweep_base.yaml"
TRIALS=20
TIMESTEPS=4000000
METRIC="eval_episode_return"
STUDY_NAME="feat_sweep_$(date -u +%Y%m%dT%H%M%S)"
STORAGE=""
SEED=42
SLURM_TIME="08:00:00"
JOB_NAME="sweep_4m_array"
ARRAY_CONCURRENCY=2
GPUS=1
CPUS=8
MEM="32G"
PARTITION=""
ACCOUNT=""
QOS=""
NICE=""
BEGIN=""
WORKER_OFFSET=0
FPS_WARN=100
EXCLUDE=""
DISABLE_WANDB=""
DRY_RUN=0

# ---- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="$2";       shift 2 ;;
    --trials)        TRIALS="$2";       shift 2 ;;
    --timesteps)     TIMESTEPS="$2";    shift 2 ;;
    --metric)        METRIC="$2";       shift 2 ;;
    --study-name)    STUDY_NAME="$2";   shift 2 ;;
    --storage)       STORAGE="$2";      shift 2 ;;
    --seed)          SEED="$2";         shift 2 ;;
    --time)          SLURM_TIME="$2";   shift 2 ;;
    --job-name)      JOB_NAME="$2";     shift 2 ;;
    --array-concurrency) ARRAY_CONCURRENCY="$2"; shift 2 ;;
    --gpus)          GPUS="$2";         shift 2 ;;
    --cpus)          CPUS="$2";         shift 2 ;;
    --mem)           MEM="$2";          shift 2 ;;
    --partition)     PARTITION="$2";    shift 2 ;;
    --account)       ACCOUNT="$2";      shift 2 ;;
    --qos)           QOS="$2";          shift 2 ;;
    --nice)          NICE="$2";         shift 2 ;;
    --begin)         BEGIN="$2";        shift 2 ;;
    --worker-offset) WORKER_OFFSET="$2"; shift 2 ;;
    --fps-warn)      FPS_WARN="$2";     shift 2 ;;
    --exclude)       EXCLUDE="$2";      shift 2 ;;
    --disable-wandb) DISABLE_WANDB="--disable-wandb"; shift ;;
    --dry-run)       DRY_RUN=1;         shift ;;
    -h|--help)       head -55 "$0" | tail -50; exit 0 ;;
    *)               echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---- resolve storage ---------------------------------------------------------
if [[ -z "$STORAGE" ]]; then
  SAFE_NAME="${STUDY_NAME//[^a-zA-Z0-9_-]/_}"
  mkdir -p "output/optuna/feat_extractor"
  STORAGE="sqlite:///output/optuna/feat_extractor/${SAFE_NAME}.db"
fi

mkdir -p output/slurm

# ---- build the per-job command -----------------------------------------------
PYTHON_CMD="uv run python scripts/training/optuna_feature_extractor.py"
PYTHON_CMD+=" --config $CONFIG"
PYTHON_CMD+=" --trials 1"
PYTHON_CMD+=" --storage $STORAGE"
PYTHON_CMD+=" --study-name $STUDY_NAME"
PYTHON_CMD+=" --trial-timesteps $TIMESTEPS"
PYTHON_CMD+=" --metric $METRIC"
PYTHON_CMD+=" --fps-warn-threshold $FPS_WARN"
PYTHON_CMD+=" --log-level WARNING"
PYTHON_CMD+=" --seed $SEED"
[[ -n "$EXCLUDE" ]] && PYTHON_CMD+=" --extractor-exclude ${EXCLUDE//,/ }"
[[ -n "$DISABLE_WANDB" ]] && PYTHON_CMD+=" $DISABLE_WANDB"
PYTHON_CMD+=" --worker-index \$(( ${WORKER_OFFSET} + SLURM_ARRAY_TASK_ID ))"

# ---- sbatch flags ------------------------------------------------------------
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

# ---- submit ------------------------------------------------------------------
echo "Study:    $STUDY_NAME"
echo "Storage:  $STORAGE"
echo "Job name: $JOB_NAME"
echo "Trials:   $TRIALS  (array tasks; each runs $TIMESTEPS steps)"
echo "Config:   $CONFIG"
echo ""

if [[ $TRIALS -lt 1 ]]; then
  echo "Trials must be at least 1." >&2
  exit 1
fi

ARRAY_SPEC="0-$((TRIALS - 1))%${ARRAY_CONCURRENCY}"

CMD=(sbatch "--job-name=$JOB_NAME" "--array=$ARRAY_SPEC" "${SBATCH_FLAGS[@]}" --wrap "$PYTHON_CMD")
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] ${CMD[*]}"
else
  "${CMD[@]}"
fi

if [[ $DRY_RUN -eq 0 ]]; then
  echo ""
  echo "Submitted array job '$JOB_NAME' with tasks $ARRAY_SPEC."
  DB_PATH="${STORAGE#sqlite:///}"
  echo "Monitor: uv run python scripts/tools/inspect_optuna_db.py --db $DB_PATH"
fi
