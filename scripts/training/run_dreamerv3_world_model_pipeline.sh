#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

COLLECT_CONFIG="configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_world_model_pretrain.yaml"
PRETRAIN_CONFIG="configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_world_model_pretrain.yaml"
FINETUNE_CONFIG="configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full_balanced_gpu_r8_warmstart.yaml"
DATASET_ID="${DREAMER_WM_DATASET_ID:-br08_world_model_ppo_teacher}"
DATASET_DIR="${DREAMER_WM_DATASET_DIR:-output/benchmarks/dreamer_world_model/${DATASET_ID}}"
TEACHER_MODE="${DREAMER_WM_TEACHER_MODE:-ppo}"
TEACHER_MODEL_ID="${DREAMER_WM_TEACHER_MODEL_ID:-}"
TEACHER_CHECKPOINT="${DREAMER_WM_TEACHER_CHECKPOINT:-}"
COLLECT_EPISODES="${DREAMER_WM_COLLECT_EPISODES:-200}"
PRETRAIN_RUN_ID_PREFIX="dreamerv3_br08_benchmark_socnav_grid_world_model_pretrain"

show_help() {
  cat <<'EOF'
Usage: scripts/training/run_dreamerv3_world_model_pipeline.sh <collect|pretrain|gate|finetune|all>

Environment variables:
  DREAMER_WM_TEACHER_MODEL_ID   Model registry id for PPO teacher.
  DREAMER_WM_TEACHER_CHECKPOINT Explicit PPO checkpoint path.
  DREAMER_WM_TEACHER_MODE       ppo|random|zero (default: ppo).
  DREAMER_WM_COLLECT_EPISODES   Episode count for offline dataset generation.
  DREAMER_WM_DATASET_ID         Dataset identifier.
  DREAMER_WM_DATASET_DIR        Dataset directory override.

Examples:
  DREAMER_WM_TEACHER_MODEL_ID=my_teacher scripts/training/run_dreamerv3_world_model_pipeline.sh all
  DREAMER_WM_TEACHER_CHECKPOINT=output/benchmarks/expert_policies/foo.zip \
    scripts/training/run_dreamerv3_world_model_pipeline.sh collect
EOF
}

latest_pretrain_summary() {
  python3 - <<'PY'
from pathlib import Path
root = Path("output/dreamerv3")
candidates = sorted(root.glob("dreamerv3_br08_benchmark_socnav_grid_world_model_pretrain_*"))
if not candidates:
    raise SystemExit(1)
print(candidates[-1] / "run_summary.json")
PY
}

latest_pretrain_checkpoint() {
  python3 - <<'PY'
import json
from pathlib import Path
root = Path("output/dreamerv3")
candidates = sorted(root.glob("dreamerv3_br08_benchmark_socnav_grid_world_model_pretrain_*"))
if not candidates:
    raise SystemExit(1)
summary = json.loads((candidates[-1] / "run_summary.json").read_text())
ckpt = summary.get("last_checkpoint_path")
if not ckpt:
    raise SystemExit(2)
print(ckpt)
PY
}

if [[ $# -lt 1 ]]; then
  show_help >&2
  exit 2
fi

MODE="$1"
shift || true

cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

export DISPLAY="${DISPLAY:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYGAME_HIDE_SUPPORT_PROMPT="${PYGAME_HIDE_SUPPORT_PROMPT:-1}"

TEACHER_ARGS=(--teacher-mode "$TEACHER_MODE")
if [[ -n "$TEACHER_MODEL_ID" ]]; then
  TEACHER_ARGS+=(--teacher-model-id "$TEACHER_MODEL_ID")
fi
if [[ -n "$TEACHER_CHECKPOINT" ]]; then
  TEACHER_ARGS+=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
fi

run_collect() {
  uv run --extra rllib python scripts/training/collect_dreamer_world_model_episodes.py \
    --dreamer-config "$COLLECT_CONFIG" \
    --dataset-id "$DATASET_ID" \
    --output-dir "$DATASET_DIR" \
    --episodes "$COLLECT_EPISODES" \
    "${TEACHER_ARGS[@]}"
}

run_pretrain() {
  uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
    --config "$PRETRAIN_CONFIG"
}

run_gate() {
  local summary_path
  summary_path="$(latest_pretrain_summary)"
  uv run --extra rllib python scripts/training/check_dreamer_world_model_pretrain.py \
    --run-summary "$summary_path"
}

run_finetune() {
  local checkpoint_path
  checkpoint_path="$(latest_pretrain_checkpoint)"
  uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
    --config "$FINETUNE_CONFIG" \
    --restore-checkpoint "$checkpoint_path" \
    --restore-component rl_module
}

case "$MODE" in
  collect)
    run_collect
    ;;
  pretrain)
    run_pretrain
    ;;
  gate)
    run_gate
    ;;
  finetune)
    run_finetune
    ;;
  all)
    run_collect
    run_pretrain
    run_gate
    run_finetune
    ;;
  -h|--help)
    show_help
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    show_help >&2
    exit 2
    ;;
esac
