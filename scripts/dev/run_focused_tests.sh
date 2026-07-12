#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_focused_tests.sh [pytest-args...]

Runs a focused pytest command with compact parent-thread output. Full pytest
stdout/stderr is written to a private agent-run artifact, and failures print only
the exit code, failing-test hints, bounded excerpts, and log path. The script
removes generated output/coverage files after the test command succeeds, while
preserving any tracked files under that tree.
Use this for narrow local validation where the coverage report is not the
artifact you intend to inspect or keep.

Examples:
  scripts/dev/run_focused_tests.sh tests/dev/test_issue_claim.py -q
  scripts/dev/run_focused_tests.sh tests/test_force_flags.py::test_force_flags_default_sync -q

Set FOCUSED_TEST_KEEP_COVERAGE=1 to preserve output/coverage after a successful
run.
Set FOCUSED_TEST_FULL_OUTPUT=1 to stream raw pytest output instead of the compact
artifact-backed mode.
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

if [[ "${FOCUSED_TEST_FULL_OUTPUT:-0}" == "1" ]]; then
  uv run pytest "$@"
else
  log_dir="$(resolve_agent_artifact_dir focused-tests)"
  mkdir -p "$log_dir"
  log_file="$log_dir/pytest-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
  set +e
  uv run pytest "$@" >"$log_file" 2>&1
  pytest_rc=$?
  set -e
  if [[ "$pytest_rc" -ne 0 ]]; then
    echo "Focused pytest failed: exit $pytest_rc"
    echo "Full log: $log_file"
    python3 - "$log_file" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()
interesting = [
    line
    for line in lines
    if re.search(r"(FAILED|ERROR|FAILURES|^\s*__+|::test_|short test summary)", line)
]
print("Failure summary:")
for line in interesting[:40]:
    print(line[:240])
if len(interesting) > 40:
    print(f"... {len(interesting) - 40} more matching lines omitted")
if not interesting:
    for line in lines[-40:]:
        print(line[:240])
PY
    exit "$pytest_rc"
  fi
  echo "Focused pytest passed. Full log: $log_file"
fi

if [[ "${FOCUSED_TEST_KEEP_COVERAGE:-0}" == "1" ]]; then
  echo "Preserving output/coverage because FOCUSED_TEST_KEEP_COVERAGE=1." >&2
else
  python3 "$SCRIPT_DIR/clean_generated_output.py" output/coverage >/dev/null
fi
