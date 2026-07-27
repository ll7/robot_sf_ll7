#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

BASE_REF="${BASE_REF:-origin/main}"

# Verify the base ref against its own committed baseline, then verify the branch
# baseline against the current working tree. The second leg makes a cleanup PR
# that removes placeholder docstrings fail until its baseline is regenerated,
# without comparing that regenerated baseline to the old base tree (issue #5894).
uv run python "$SCRIPT_DIR/../validation/check_docstring_todos.py" \
  --mode verify-baseline \
  --base "$BASE_REF" \
  --check-working-tree
