#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SBATCH_MAX_TIME_SCRIPT="$SCRIPT_DIR/sbatch_use_max_time.sh"
PARTITION_STATUS_SCRIPT="$SCRIPT_DIR/auxme_partition_status.sh"

PARTITION="a30"
QOS="a30-gpu"
TIME_LIMIT="12:00:00"
TRAIN_CONFIG="configs/training/ppo/ablations/expert_ppo_issue_1474_shielded_repair_collision20_5m.yaml"
LAUNCH_PACKET="configs/training/shielded_ppo_issue_1396_launch_packet.yaml"
DRY_RUN=0
SHOW_STATUS=1
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/dev/sbatch_shielded_ppo_repair_issue1474.sh [options]

Prepare or submit the issue #1474 shielded PPO repair training job.

Options:
  --partition <name>      Slurm partition (default: a30)
  --qos <name>            Slurm QoS (default: a30-gpu)
  --time <value>          Wall time passed to sbatch_use_max_time (default: 12:00:00)
  --train-config <path>   Training config to submit
  --launch-packet <path>  Shielded PPO launch packet to validate
  --sbatch-arg <arg>      Extra sbatch argument
  --no-status             Skip Auxme partition status table
  --dry-run               Print the resolved sbatch command without submitting
  -h, --help              Show help
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
    --train-config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --launch-packet)
      LAUNCH_PACKET="$2"
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
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

SLURM_SCRIPT="SLURM/Auxme/issue_1474_shielded_ppo_repair.sl"

for path in "$TRAIN_CONFIG" "$LAUNCH_PACKET" "$SLURM_SCRIPT"; do
  [[ -f "$path" ]] || {
    echo "Required path not found: $path" >&2
    exit 2
  }
done

uv run python scripts/validation/validate_shielded_ppo_launch_packet.py \
  --config "$LAUNCH_PACKET" \
  --json >/dev/null

uv run python - "$TRAIN_CONFIG" <<'PY'
import sys
from scripts.training.train_ppo import load_expert_training_config

config = load_expert_training_config(sys.argv[1])
weights = config.env_factory_kwargs.get("reward_kwargs", {}).get("weights", {})
if weights.get("collision") != -20.0:
    raise SystemExit("expected env_factory_kwargs.reward_kwargs.weights.collision == -20.0")
if config.resume_model_id != "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200":
    raise SystemExit(f"unexpected resume_model_id: {config.resume_model_id!r}")
PY

mkdir -p output/slurm

if [[ "$SHOW_STATUS" == "1" ]]; then
  "$PARTITION_STATUS_SCRIPT" || true
fi

echo "[issue1474-submit] branch=$(git branch --show-current)" >&2
echo "[issue1474-submit] commit=$(git rev-parse HEAD)" >&2
echo "[issue1474-submit] launch_packet=$LAUNCH_PACKET" >&2
echo "[issue1474-submit] train_config=$TRAIN_CONFIG" >&2

wrapper_args=(
  "--partition" "$PARTITION"
  "--qos" "$QOS"
  "--time" "$TIME_LIMIT"
  "--sbatch-arg" "--partition=$PARTITION"
  "--sbatch-arg" "--qos=$QOS"
  "--sbatch-arg" "--job-name=rsf-1474-shielded-ppo"
  "--sbatch-arg" "--export=ALL,ISSUE1474_TRAIN_CONFIG=$TRAIN_CONFIG,ISSUE1474_LAUNCH_PACKET=$LAUNCH_PACKET"
)

if [[ "$DRY_RUN" == "1" ]]; then
  wrapper_args+=("--dry-run")
fi
for arg in "${EXTRA_SBATCH_ARGS[@]}"; do
  wrapper_args+=("--sbatch-arg" "$arg")
done

"$SBATCH_MAX_TIME_SCRIPT" "${wrapper_args[@]}" "$SLURM_SCRIPT"
