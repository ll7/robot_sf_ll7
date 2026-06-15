#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_focused_tests.sh [pytest-args...]

Runs a focused pytest command and removes generated output/coverage files after
the test command succeeds, while preserving any tracked files under that tree.
Use this for narrow local validation where the coverage report is not the
artifact you intend to inspect or keep.

Examples:
  scripts/dev/run_focused_tests.sh tests/dev/test_issue_claim.py -q
  scripts/dev/run_focused_tests.sh tests/test_force_flags.py::test_force_flags_default_sync -q

Set FOCUSED_TEST_KEEP_COVERAGE=1 to preserve output/coverage after a successful
run.
EOF
}

if [[ "$#" -gt 0 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

if [[ "$#" -eq 0 ]]; then
  echo "No pytest target supplied. Pass focused pytest args or use --help." >&2
  exit 2
fi

uv run pytest "$@"

if [[ "${FOCUSED_TEST_KEEP_COVERAGE:-0}" == "1" ]]; then
  echo "Preserving output/coverage because FOCUSED_TEST_KEEP_COVERAGE=1." >&2
else
  python3 "$SCRIPT_DIR/clean_generated_output.py" output/coverage >/dev/null
fi
