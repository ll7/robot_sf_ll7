#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

export BASE_REF="${BASE_REF:-origin/main}"
export MIN_COVERAGE="${MIN_COVERAGE:-80}"
export GOAL_COVERAGE="${GOAL_COVERAGE:-100}"

"$SCRIPT_DIR/ruff_fix_format.sh"
"$SCRIPT_DIR/run_tests_parallel.sh"
"$SCRIPT_DIR/check_changed_coverage.sh"
"$SCRIPT_DIR/check_docstring_todos_diff.sh"
