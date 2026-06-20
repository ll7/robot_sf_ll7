#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_xdist_race_validation.sh [wrapper-options] [pytest-args...]

Runs the full pytest suite through a high-concurrency xdist stress route, then
checks shared generated-output paths for likely race artifacts.

Wrapper options:
  --workers <int>|auto       xdist workers [default: 32]
  --timeout-seconds <secs>   Compact-validation timeout [default: 7200]
  --artifact-dir <path>      Compact logs and scan manifests
  -h, --help                 Show this help message

Environment overrides:
  XDIST_RACE_WORKERS=<int>|auto
  XDIST_RACE_TIMEOUT_SECONDS=<secs>
  XDIST_RACE_ARTIFACT_DIR=<path>
  XDIST_RACE_RUN_ID=<id>
  PYTEST_XDIST_DIST=load|worksteal|loadscope|loadfile|loadgroup

Examples:
  scripts/dev/run_xdist_race_validation.sh
  scripts/dev/run_xdist_race_validation.sh --workers 32 tests fast-pysf/tests
  scripts/dev/run_xdist_race_validation.sh --timeout-seconds 300 tests/dev
EOF
}

if [[ "$#" -gt 0 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

workers="${XDIST_RACE_WORKERS:-32}"
timeout_seconds="${XDIST_RACE_TIMEOUT_SECONDS:-7200}"
run_id="${XDIST_RACE_RUN_ID:-${GITHUB_RUN_ID:-local}-attempt-${GITHUB_RUN_ATTEMPT:-1}}"
artifact_dir="${XDIST_RACE_ARTIFACT_DIR:-output/validation/xdist-race/${run_id}}"
pytest_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      workers="${2:-}"
      shift 2
      ;;
    --timeout-seconds)
      timeout_seconds="${2:-}"
      shift 2
      ;;
    --artifact-dir)
      artifact_dir="${2:-}"
      shift 2
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

if [[ -z "$workers" || -z "$timeout_seconds" || -z "$artifact_dir" ]]; then
  echo "--workers, --timeout-seconds, and --artifact-dir require non-empty values." >&2
  exit 2
fi

if [[ "$workers" != "auto" && ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer or 'auto'." >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$artifact_dir" "output/tmp/xdist-race/${run_id}" output/coverage

baseline_json="$artifact_dir/baseline-artifacts.json"
scan_json="$artifact_dir/post-run-artifacts.json"
validation_status=0
scan_status=0

uv run python "$SCRIPT_DIR/check_xdist_race_artifacts.py" \
  --run-id "$run_id" \
  --write-manifest "$baseline_json" \
  output/tmp output/coverage >/dev/null

export PYTEST_NUM_WORKERS="$workers"
export PYTEST_XDIST_DIST="${PYTEST_XDIST_DIST:-worksteal}"
export PYTEST_FAST_FAIL="${PYTEST_FAST_FAIL:-0}"
export PYTEST_ORDER_MODE="${PYTEST_ORDER_MODE:-none}"
export COVERAGE_FILE="${COVERAGE_FILE:-output/coverage/.coverage.xdist-race-${run_id}}"

echo "xdist race run id: $run_id" >&2
echo "xdist race artifact dir: $artifact_dir" >&2
echo "xdist race workers: $PYTEST_NUM_WORKERS" >&2
echo "xdist race dist mode: $PYTEST_XDIST_DIST" >&2

set +e
uv run python "$SCRIPT_DIR/run_compact_validation.py" \
  --artifact-dir "$artifact_dir" \
  --timeout-seconds "$timeout_seconds" \
  -- "$SCRIPT_DIR/run_tests_parallel.sh" "${pytest_args[@]}"
validation_status=$?

uv run python "$SCRIPT_DIR/check_xdist_race_artifacts.py" \
  --run-id "$run_id" \
  --baseline-json "$baseline_json" \
  --write-manifest "$scan_json" \
  --json \
  output/tmp output/coverage
scan_status=$?
set -e

if [[ "$validation_status" -ne 0 ]]; then
  echo "xdist race validation failed with exit code $validation_status." >&2
fi
if [[ "$scan_status" -ne 0 ]]; then
  echo "xdist race artifact scan failed with exit code $scan_status." >&2
fi
if [[ "$validation_status" -ne 0 ]]; then
  exit "$validation_status"
fi
exit "$scan_status"
