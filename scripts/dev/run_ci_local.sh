#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_ci_local.sh [<phase> ...]

Run the canonical CI validation phases locally through the shared driver.
When no phases are provided, the local wrapper runs:
  lint typecheck test smoke artifact-policy

Examples:
  scripts/dev/run_ci_local.sh
  scripts/dev/run_ci_local.sh lint test
  CI_DRIVER_EVENT_NAME=workflow_dispatch scripts/dev/run_ci_local.sh smoke
EOF
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  show_help
  exit 0
fi

phases=("$@")
if [[ ${#phases[@]} -eq 0 ]]; then
  phases=("lint" "typecheck" "test" "smoke" "artifact-policy")
fi

uv sync --all-extras --frozen
uv run python scripts/tools/migrate_artifacts.py

"$SCRIPT_DIR/ci_driver.sh" "${phases[@]}"
