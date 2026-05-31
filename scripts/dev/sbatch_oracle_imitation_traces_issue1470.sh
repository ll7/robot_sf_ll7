#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SBATCH_MAX_TIME_SCRIPT="$SCRIPT_DIR/sbatch_use_max_time.sh"
PARTITION_STATUS_SCRIPT="$SCRIPT_DIR/auxme_partition_status.sh"
SBATCH_SCRIPT="SLURM/Auxme/issue_1470_oracle_imitation_traces.sl"

PARTITION="a30"
QOS="a30-gpu"
TIME_LIMIT="06:00:00"
SPLIT="train"
HORIZON="500"
WORKERS="1"
SHOW_STATUS=1
DRY_RUN=0
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/dev/sbatch_oracle_imitation_traces_issue1470.sh [options]

Prepare or submit the issue #1470 oracle-imitation source-candidate trace job.

Options:
  --partition <name>   Slurm partition (default: a30)
  --qos <name>         Slurm QoS (default: a30-gpu)
  --time <limit>       Explicit wall time passed to sbatch (default: 06:00:00)
  --split <name>       Launch-packet split: train, validation, evaluation (default: train)
  --horizon <n>        Episode horizon for trace collection (default: 500)
  --workers <n>        Map-runner workers inside the allocation (default: 1)
  --sbatch-arg <arg>   Extra sbatch argument
  --no-status          Skip partition status table
  --dry-run            Print resolved sbatch command without submitting
  -h, --help           Show help

Example:
  scripts/dev/sbatch_oracle_imitation_traces_issue1470.sh --dry-run --no-status
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --horizon)
      HORIZON="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
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

case "$SPLIT" in
  train|validation|evaluation)
    ;;
  *)
    echo "Unsupported split: $SPLIT" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"

uv run python scripts/validation/validate_oracle_imitation_launch_packet.py \
  --config configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml \
  --json >/dev/null

if [[ "$SHOW_STATUS" == "1" ]]; then
  "$PARTITION_STATUS_SCRIPT"
fi

wrapper_args=(
  "--time" "$TIME_LIMIT"
  "--partition" "$PARTITION"
  "--qos" "$QOS"
  "--sbatch-arg" "--partition=$PARTITION"
  "--sbatch-arg" "--qos=$QOS"
  "--sbatch-arg" "--job-name=rsf-1470-oracle-traces"
  "--sbatch-arg" "--export=ALL,ISSUE1470_SPLIT=$SPLIT,ISSUE1470_HORIZON=$HORIZON,ISSUE1470_WORKERS=$WORKERS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  wrapper_args+=("--dry-run")
fi
for arg in "${EXTRA_SBATCH_ARGS[@]}"; do
  wrapper_args+=("--sbatch-arg" "$arg")
done

"$SBATCH_MAX_TIME_SCRIPT" "${wrapper_args[@]}" "$SBATCH_SCRIPT"
