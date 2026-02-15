#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
fi

BASE_REF="${BASE_REF:-origin/main}"
MIN_COVERAGE="${MIN_COVERAGE:-80}"
GOAL_COVERAGE="${GOAL_COVERAGE:-100}"

uv run python scripts/coverage/check_changed_files_coverage.py \
  --base "$BASE_REF" \
  --min "$MIN_COVERAGE" \
  --goal "$GOAL_COVERAGE"
