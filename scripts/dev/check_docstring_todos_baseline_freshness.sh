#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

BASE_REF="${BASE_REF:-origin/main}"

# Compare the committed baseline against both the base ref and the current
# working tree. The working-tree leg makes a cleanup PR that removes placeholder
# docstrings fail its own readiness gate until the baseline is regenerated, so
# drift can no longer slip through and break the next unrelated PR (issue #5894).
uv run python "$SCRIPT_DIR/../validation/check_docstring_todos.py" \
  --mode verify-baseline \
  --base "$BASE_REF" \
  --check-working-tree
