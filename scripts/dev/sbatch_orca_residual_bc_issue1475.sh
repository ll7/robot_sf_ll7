#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SBATCH_MAX_TIME_SCRIPT="$SCRIPT_DIR/sbatch_use_max_time.sh"
PARTITION_STATUS_SCRIPT="$SCRIPT_DIR/auxme_partition_status.sh"

PARTITION="a30"
QOS="a30-gpu"
TIME_LIMIT="12:00:00"
EPISODES="3"
SEEDS="111:112:113"
RUN_NOMINAL="0"
DRY_RUN=0
SHOW_STATUS=1
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/dev/sbatch_orca_residual_bc_issue1475.sh [options]

Prepare or submit the bounded ORCA-residual BC smoke job for issue #1475.

Options:
  --partition <name>    Slurm partition (default: a30)
  --qos <name>          Slurm QoS (default: a30-gpu)
  --time <value>        Wall time passed to sbatch_use_max_time (default: 12:00:00)
  --episodes <n>        Bounded smoke collection episodes (default: 3)
  --seeds "<values>"    Colon-separated seed list (default: "111:112:113")
  --run-nominal         Run nominal_sanity only after the smoke gate passes
  --sbatch-arg <arg>    Extra sbatch argument
  --no-status           Skip Auxme partition status table
  --dry-run             Print the resolved sbatch command without submitting
  -h, --help            Show help
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
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --run-nominal)
      RUN_NOMINAL="1"
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
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

LINEAGE_CONFIG="configs/training/orca_residual/orca_residual_bc_issue_1428.yaml"
BC_CONFIG="configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml"
SLURM_SCRIPT="SLURM/Auxme/issue_1475_orca_residual_bc.sl"

for path in "$LINEAGE_CONFIG" "$BC_CONFIG" "$SLURM_SCRIPT"; do
  [[ -f "$path" ]] || {
    echo "Required path not found: $path" >&2
    exit 2
  }
done

uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config "$LINEAGE_CONFIG" \
  --json >/dev/null

if [[ "$SHOW_STATUS" == "1" ]]; then
  "$PARTITION_STATUS_SCRIPT" || true
fi

echo "[issue1475-submit] branch=$(git branch --show-current)" >&2
echo "[issue1475-submit] commit=$(git rev-parse HEAD)" >&2
echo "[issue1475-submit] lineage_config=$LINEAGE_CONFIG" >&2
echo "[issue1475-submit] bc_config=$BC_CONFIG" >&2
echo "[issue1475-submit] episodes=$EPISODES seeds=$SEEDS run_nominal=$RUN_NOMINAL" >&2

wrapper_args=(
  "--partition" "$PARTITION"
  "--qos" "$QOS"
  "--time" "$TIME_LIMIT"
  "--sbatch-arg" "--partition=$PARTITION"
  "--sbatch-arg" "--qos=$QOS"
  "--sbatch-arg" "--job-name=rsf-1475-orca-bc-smoke"
  "--sbatch-arg" "--export=ALL,ISSUE1475_EPISODES=$EPISODES,ISSUE1475_SEEDS=$SEEDS,ISSUE1475_RUN_NOMINAL=$RUN_NOMINAL"
)

if [[ "$DRY_RUN" == "1" ]]; then
  wrapper_args+=("--dry-run")
fi
for arg in "${EXTRA_SBATCH_ARGS[@]}"; do
  wrapper_args+=("--sbatch-arg" "$arg")
done

"$SBATCH_MAX_TIME_SCRIPT" "${wrapper_args[@]}" "$SLURM_SCRIPT"
