#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  echo "Usage: scripts/dev/ci_step_timer.sh <label> <command> [args...]"
  echo "Optional environment variables:"
  echo "  CI_STEP_TIMEOUT_SECONDS  Run the command under a timeout of this many seconds."
  echo "                           Uses GNU timeout(1) or a portable Python fallback."
  exit 0
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/dev/ci_step_timer.sh <label> <command> [args...]" >&2
  exit 2
fi

label="$1"
shift

echo "::group::${label}"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_seconds="$(date +%s)"
echo "ci_step_timer step_start label=\"${label}\" started_at=${started_at}"

set +e
if [[ -n "${CI_STEP_TIMEOUT_SECONDS:-}" ]]; then
  if command -v timeout >/dev/null 2>&1; then
    timeout "${CI_STEP_TIMEOUT_SECONDS}" "$@"
    status=$?
  elif command -v python3 >/dev/null 2>&1; then
    script_path="${BASH_SOURCE[0]}"
    if [[ "$script_path" == */* ]]; then
      script_dir="$(cd -- "${script_path%/*}" && pwd)"
    else
      script_dir="$PWD"
    fi
    echo "ci_step_timer: GNU timeout unavailable; using portable Python timeout fallback" >&2
    python3 "$script_dir/run_with_timeout.py" "${CI_STEP_TIMEOUT_SECONDS}" -- "$@"
    status=$?
  else
    echo "::error::CI_STEP_TIMEOUT_SECONDS requires GNU timeout(1) or python3" >&2
    status=127
  fi
else
  "$@"
  status=$?
fi
set -e

completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
end_seconds="$(date +%s)"
duration=$((end_seconds - start_seconds))
echo "ci_step_timer step_end label=\"${label}\" status=${status} duration_seconds=${duration} completed_at=${completed_at}"
echo "::notice title=\"${label}\" timing::status=${status} duration_seconds=${duration}"
echo "::endgroup::"

exit "${status}"
