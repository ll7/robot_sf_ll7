#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION_STATUS_SCRIPT="$SCRIPT_DIR/auxme_partition_status.sh"
SBATCH_MAX_TIME_SCRIPT="$SCRIPT_DIR/sbatch_use_max_time.sh"

SLURM_SCRIPT=""
TRAIN_CONFIG=""
PARTITION=""
QOS=""
JOB_NAME=""
WANDB_POLICY="auto"
DRY_RUN=0
SHOW_STATUS=1
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/dev/sbatch_auxme_issue791.sh [options] <slurm-script>

Reliable submit path for issue-791 wrappers on Auxme.
It requires an explicit training config and can auto-pick partition/qos based on current pressure.

Options:
  --config <path>          Required training YAML (passed as ISSUE791_TRAIN_CONFIG)
  --partition <name>       Force partition (a30 or l40s)
  --qos <name>             Force qos (defaults to <partition>-gpu when omitted)
  --job-name <name>        Override Slurm job name
  --wandb-policy <value>   ISSUE791_WANDB_POLICY value (default: auto)
  --sbatch-arg <arg>       Extra sbatch argument forwarded via sbatch_use_max_time.sh
  --no-status              Skip partition status table before submit
  --dry-run                Show resolved command only
  -h, --help               Show help

Examples:
  scripts/dev/sbatch_auxme_issue791.sh \
    --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml \
    --job-name robot-sf-issue791-reward-curriculum \
    SLURM/Auxme/issue_791_reward_curriculum.sl

  scripts/dev/sbatch_auxme_issue791.sh \
    --config configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml \
    --partition l40s --qos l40s-gpu \
    SLURM/Auxme/issue_791_attention_head.sl
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --wandb-policy)
      WANDB_POLICY="$2"
      shift 2
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
    --)
      shift
      break
      ;;
    -* )
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SLURM_SCRIPT="$1"
      shift
      ;;
  esac
done

if [[ -z "$SLURM_SCRIPT" ]]; then
  echo "Missing slurm script path." >&2
  usage >&2
  exit 2
fi

if [[ -z "$TRAIN_CONFIG" ]]; then
  echo "--config is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Slurm script not found: $SLURM_SCRIPT" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_CONFIG" ]]; then
  echo "Training config not found: $TRAIN_CONFIG" >&2
  exit 2
fi

if [[ "$SHOW_STATUS" == "1" ]]; then
  "$PARTITION_STATUS_SCRIPT"
fi

if [[ -z "$PARTITION" || -z "$QOS" ]]; then
  recommendation="$($PARTITION_STATUS_SCRIPT --recommend)"
  rec_partition="$(echo "$recommendation" | sed -n 's/.*partition=\([^ ]*\).*/\1/p')"
  rec_qos="$(echo "$recommendation" | sed -n 's/.*qos=\([^ ]*\).*/\1/p')"
  if [[ -z "$PARTITION" ]]; then
    PARTITION="$rec_partition"
  fi
  if [[ -z "$QOS" ]]; then
    QOS="$rec_qos"
  fi
fi

if [[ -z "$QOS" ]]; then
  QOS="${PARTITION}-gpu"
fi

echo "[issue791-submit] selected partition=$PARTITION qos=$QOS config=$TRAIN_CONFIG wandb_policy=$WANDB_POLICY" >&2

wrapper_args=("--partition" "$PARTITION" "--qos" "$QOS")
if [[ "$DRY_RUN" == "1" ]]; then
  wrapper_args+=("--dry-run")
fi
if [[ -n "$JOB_NAME" ]]; then
  wrapper_args+=("--sbatch-arg" "--job-name=$JOB_NAME")
fi
for arg in "${EXTRA_SBATCH_ARGS[@]}"; do
  wrapper_args+=("--sbatch-arg" "$arg")
done

ISSUE791_TRAIN_CONFIG="$TRAIN_CONFIG" \
ISSUE791_WANDB_POLICY="$WANDB_POLICY" \
"$SBATCH_MAX_TIME_SCRIPT" "${wrapper_args[@]}" "$SLURM_SCRIPT"
