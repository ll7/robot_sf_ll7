#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  echo "Usage: scripts/dev/ci_step_timer.sh <label> <command> [args...]"
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
"$@"
status=$?

completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
end_seconds="$(date +%s)"
duration=$((end_seconds - start_seconds))
echo "ci_step_timer step_end label=\"${label}\" status=${status} duration_seconds=${duration} completed_at=${completed_at}"
echo "::notice title=\"${label}\" timing::status=${status} duration_seconds=${duration}"
echo "::endgroup::"

set -e
exit "${status}"
