#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/scripts/dev"
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
fi

BASE_REF="${BASE_REF:-origin/main}"
MIN_COVERAGE="${MIN_COVERAGE:-80}"
GOAL_COVERAGE="${GOAL_COVERAGE:-100}"

"$SCRIPT_DIR/ruff_fix_format.sh"
"$SCRIPT_DIR/run_tests_parallel.sh" tests
BASE_REF="$BASE_REF" MIN_COVERAGE="$MIN_COVERAGE" GOAL_COVERAGE="$GOAL_COVERAGE" \
  "$SCRIPT_DIR/check_changed_coverage.sh"
BASE_REF="$BASE_REF" "$SCRIPT_DIR/check_docstring_todos_diff.sh"
