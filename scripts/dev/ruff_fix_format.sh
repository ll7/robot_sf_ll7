#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

# With explicit paths, format only the caller's intended files.  The PR
# readiness wrapper passes its committed Python diff here so an unformatted
# base-only file cannot be mutated as a side effect of validation.  Keeping the
# no-argument behavior whole-tree preserves the standalone developer command.
if [[ "$#" -gt 0 ]]; then
  format_targets=("$@")
else
  format_targets=(.)
fi

uv run ruff check --fix "${format_targets[@]}" --output-format concise || true
uv run ruff format "${format_targets[@]}"
uv run ruff check . --statistics
