#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

show_help() {
  cat <<'EOF'
Usage: scripts/training/run_dreamerv3_br08.sh <gate|full> [dreamer-args...]

Launch the BR-08 RLlib DreamerV3 training profiles with the repository virtualenv,
headless-safe defaults, and canonical config paths.

Examples:
  scripts/training/run_dreamerv3_br08.sh gate --dry-run
  scripts/training/run_dreamerv3_br08.sh full --log-level INFO
EOF
}

if [[ $# -lt 1 ]]; then
  show_help >&2
  exit 2
fi

profile="$1"
shift

case "$profile" in
  gate)
    config_path="configs/training/rllib_dreamerv3/drive_state_rays_br08_gate.yaml"
    ;;
  full)
    config_path="configs/training/rllib_dreamerv3/drive_state_rays_br08_full.yaml"
    ;;
  -h|--help)
    show_help
    exit 0
    ;;
  *)
    echo "Unknown profile: $profile" >&2
    show_help >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

export DISPLAY="${DISPLAY:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYGAME_HIDE_SUPPORT_PROMPT="${PYGAME_HIDE_SUPPORT_PROMPT:-1}"

exec uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config "$config_path" \
  "$@"