#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_tests_parallel.sh [wrapper-options] [pytest-args...]

Runs pytest in parallel (`-n auto`) with fail-fast defaults for local triage.

Wrapper options:
  --fast-fail      Stop on first failure (`-x`) [default]
  --no-fast-fail   Disable fail-fast
  --failed-first   Run previously failed tests first [default]
  --new-first      Run tests from new files first
  --no-ordering    Disable ordering hints
  -h, --help       Show this help message

Environment overrides:
  PYTEST_FAST_FAIL=1|0
  PYTEST_ORDER_MODE=failed-first|new-first|none

Examples:
  scripts/dev/run_tests_parallel.sh
  scripts/dev/run_tests_parallel.sh --new-first tests/benchmark
  scripts/dev/run_tests_parallel.sh --no-fast-fail tests
EOF
}

fast_fail="${PYTEST_FAST_FAIL:-1}"
order_mode="${PYTEST_ORDER_MODE:-failed-first}"

pytest_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast-fail)
      fast_fail="1"
      shift
      ;;
    --no-fast-fail)
      fast_fail="0"
      shift
      ;;
    --failed-first)
      order_mode="failed-first"
      shift
      ;;
    --new-first)
      order_mode="new-first"
      shift
      ;;
    --no-ordering)
      order_mode="none"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      pytest_args+=("$1")
      shift
      ;;
  esac
done

cmd=(uv run pytest -n auto)

if [[ "$fast_fail" == "1" ]]; then
  cmd+=("-x")
elif [[ "$fast_fail" != "0" ]]; then
  echo "Invalid PYTEST_FAST_FAIL value '$fast_fail' (expected 0 or 1)." >&2
  exit 2
fi

case "$order_mode" in
  failed-first)
    cmd+=("--failed-first")
    ;;
  new-first)
    cmd+=("--new-first")
    ;;
  none)
    ;;
  *)
    echo "Invalid PYTEST_ORDER_MODE value '$order_mode' (expected failed-first|new-first|none)." >&2
    exit 2
    ;;
esac

cmd+=("${pytest_args[@]}")
"${cmd[@]}"
