#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_ci_local.sh [--no-setup] [<phase> ...]

Run the canonical CI validation phases locally through the shared driver.
When no phases are provided, the local wrapper runs every phase advertised by:
  scripts/dev/ci_driver.sh --list-phases

Wrapper options:
  --no-setup     Skip dependency sync and artifact migration for repeat local runs
  --list-phases  List canonical CI phases without running setup
  -h, --help     Show this help message

Examples:
  scripts/dev/run_ci_local.sh
  scripts/dev/run_ci_local.sh --no-setup lint test
  scripts/dev/run_ci_local.sh lint test
  CI_DRIVER_EVENT_NAME=workflow_dispatch scripts/dev/run_ci_local.sh smoke
EOF
}

if [[ "$#" -gt 0 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

load_default_phases() {
  mapfile -t default_phases < <("$SCRIPT_DIR/ci_driver.sh" --list-phases)
  if [[ ${#default_phases[@]} -eq 0 ]]; then
    echo "Failed to load default CI phases from scripts/dev/ci_driver.sh --list-phases" >&2
    exit 2
  fi
  printf "%s\n" "${default_phases[@]}"
}

run_setup="1"
phases=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-setup)
      run_setup="0"
      shift
      ;;
    --list-phases)
      "$SCRIPT_DIR/ci_driver.sh" --list-phases
      exit 0
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      phases+=("$@")
      break
      ;;
    *)
      phases+=("$1")
      shift
      ;;
  esac
done

if [[ ${#phases[@]} -eq 0 ]]; then
  mapfile -t phases < <(load_default_phases)
fi

if [[ "$run_setup" == "1" ]]; then
  bash "$SCRIPT_DIR/ci_step_timer.sh" "Sync dependencies (locked)" uv sync --all-extras --frozen
  bash "$SCRIPT_DIR/ci_step_timer.sh" "Migrate legacy artifacts into canonical root" \
    uv run python scripts/tools/migrate_artifacts.py
else
  echo "Skipping run_ci_local setup (--no-setup)."
fi

"$SCRIPT_DIR/ci_driver.sh" "${phases[@]}"
