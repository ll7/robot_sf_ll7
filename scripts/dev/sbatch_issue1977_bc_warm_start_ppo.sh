#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SBATCH_MAX_TIME_SCRIPT="$SCRIPT_DIR/sbatch_use_max_time.sh"
PARTITION_STATUS_SCRIPT="$SCRIPT_DIR/auxme_partition_status.sh"
SBATCH_SCRIPT="SLURM/Auxme/issue_1977_bc_warm_start_ppo.sl"
PPO_CONFIG="configs/training/ppo_imitation/ppo_finetune_issue_1977_bc_warm_start_rerun.yaml"
WANDB_ARTIFACT="ll7/robot_sf/issue_1108_bc_warm_start_job12472:v0"

PARTITION="a30"
QOS="a30-gpu"
TIME_LIMIT="1-12:00:00"
RUN_POLICY_ANALYSIS="1"
SHOW_STATUS=1
DRY_RUN=0
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/dev/sbatch_issue1977_bc_warm_start_ppo.sh [options]

Submit the issue #1977 clean BC warm-start PPO rerun. The job hydrates the
preserved #1108 W&B artifact and runs PPO fine-tuning from the preserved BC
checkpoint rather than recollecting trajectories or retraining BC.

Options:
  --config <path>          PPO fine-tune config (default: issue #1977 full run)
  --wandb-artifact <name>  W&B artifact to hydrate (default: preserved #1108 artifact)
  --partition <name>       Slurm partition (default: a30)
  --qos <name>             Slurm QoS (default: a30-gpu)
  --time <limit>           Wall time passed to sbatch (default: 1-12:00:00)
  --no-policy-analysis     Skip the post-training policy-analysis command
  --sbatch-arg <arg>       Extra sbatch argument
  --no-status              Skip Auxme partition status table
  --dry-run                Print the resolved sbatch command without submitting
  -h, --help               Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      PPO_CONFIG="$2"
      shift 2
      ;;
    --config=*)
      PPO_CONFIG="${1#*=}"
      shift
      ;;
    --wandb-artifact)
      WANDB_ARTIFACT="$2"
      shift 2
      ;;
    --wandb-artifact=*)
      WANDB_ARTIFACT="${1#*=}"
      shift
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --partition=*)
      PARTITION="${1#*=}"
      shift
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --qos=*)
      QOS="${1#*=}"
      shift
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --time=*)
      TIME_LIMIT="${1#*=}"
      shift
      ;;
    --no-policy-analysis)
      RUN_POLICY_ANALYSIS="0"
      shift
      ;;
    --sbatch-arg)
      EXTRA_SBATCH_ARGS+=("$2")
      shift 2
      ;;
    --no-status)
      SHOW_STATUS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      echo "Unexpected positional argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

for path in "$PPO_CONFIG" "$SBATCH_SCRIPT"; do
  [[ -f "$path" ]] || {
    echo "Required path not found: $path" >&2
    exit 2
  }
done

uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config "$PPO_CONFIG" \
  --dry-run >/dev/null

if [[ "$SHOW_STATUS" == "1" ]]; then
  "$PARTITION_STATUS_SCRIPT" || true
fi

echo "[issue1977-submit] branch=$(git branch --show-current)" >&2
echo "[issue1977-submit] commit=$(git rev-parse HEAD)" >&2
echo "[issue1977-submit] ppo_config=$PPO_CONFIG" >&2
echo "[issue1977-submit] wandb_artifact=$WANDB_ARTIFACT" >&2
echo "[issue1977-submit] run_policy_analysis=$RUN_POLICY_ANALYSIS" >&2

wrapper_args=(
  "--partition" "$PARTITION"
  "--qos" "$QOS"
  "--time" "$TIME_LIMIT"
  "--sbatch-arg" "--partition=$PARTITION"
  "--sbatch-arg" "--qos=$QOS"
  "--sbatch-arg" "--job-name=gse-1977-bcppo"
  "--sbatch-arg" "--export=ALL,ISSUE1977_PPO_CONFIG=$PPO_CONFIG,ISSUE1977_WANDB_ARTIFACT=$WANDB_ARTIFACT,ISSUE1977_RUN_POLICY_ANALYSIS=$RUN_POLICY_ANALYSIS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  wrapper_args+=("--dry-run")
fi
for arg in "${EXTRA_SBATCH_ARGS[@]}"; do
  wrapper_args+=("--sbatch-arg" "$arg")
done

"$SBATCH_MAX_TIME_SCRIPT" "${wrapper_args[@]}" "$SBATCH_SCRIPT"
