#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_ci_local.sh [<phase> ...]

Run the canonical CI validation phases locally through the shared driver.
When no phases are provided, the local wrapper runs every phase advertised by:
  scripts/dev/ci_driver.sh --list-phases

Examples:
  scripts/dev/run_ci_local.sh
  scripts/dev/run_ci_local.sh lint test
  CI_DRIVER_EVENT_NAME=workflow_dispatch scripts/dev/run_ci_local.sh smoke
EOF
}

load_default_phases() {
  mapfile -t default_phases < <("$SCRIPT_DIR/ci_driver.sh" --list-phases)
  if [[ ${#default_phases[@]} -eq 0 ]]; then
    echo "Failed to load default CI phases from scripts/dev/ci_driver.sh --list-phases" >&2
    exit 2
  fi
  printf "%s\n" "${default_phases[@]}"
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  show_help
  exit 0
fi

phases=("$@")
if [[ ${#phases[@]} -eq 0 ]]; then
  mapfile -t phases < <(load_default_phases)
fi

uv sync --all-extras --frozen
uv run python scripts/tools/migrate_artifacts.py

"$SCRIPT_DIR/ci_driver.sh" "${phases[@]}"
