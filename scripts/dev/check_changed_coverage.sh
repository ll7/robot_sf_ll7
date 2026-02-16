#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

BASE_REF="${BASE_REF:-origin/main}"
MIN_COVERAGE="${MIN_COVERAGE:-80}"
GOAL_COVERAGE="${GOAL_COVERAGE:-100}"

uv run python scripts/coverage/check_changed_files_coverage.py \
  --base "$BASE_REF" \
  --min "$MIN_COVERAGE" \
  --goal "$GOAL_COVERAGE"
