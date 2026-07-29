#!/usr/bin/env bash
set -euo pipefail

run_with_python_timeout() {
  local timeout_seconds="$1"
  shift

  python3 -c '
import os
import signal
import subprocess
import sys

TIMEOUT_STATUS = 124
BACKEND_ERROR_STATUS = 125
TERMINATION_GRACE_SECONDS = 5


def main() -> int:
    try:
        timeout_seconds = float(sys.argv[1])
    except ValueError:
        print(
            f"ci_step_timer: invalid timeout in seconds: {sys.argv[1]!r}",
            file=sys.stderr,
        )
        return BACKEND_ERROR_STATUS

    if timeout_seconds <= 0:
        print("ci_step_timer: timeout must be greater than zero", file=sys.stderr)
        return BACKEND_ERROR_STATUS

    try:
        child = subprocess.Popen(sys.argv[2:], start_new_session=True)
    except FileNotFoundError:
        print(f"ci_step_timer: command not found: {sys.argv[2]}", file=sys.stderr)
        return 127
    except PermissionError:
        print(f"ci_step_timer: command is not executable: {sys.argv[2]}", file=sys.stderr)
        return 126
    forwarded_signal = [None]

    def forward_signal(signum: int, _frame: object) -> None:
        forwarded_signal[0] = signum
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward_signal)

    try:
        status = child.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        return TIMEOUT_STATUS

    if forwarded_signal[0] is not None:
        return 128 + forwarded_signal[0]
    if status < 0:
        return 128 - status
    return status


raise SystemExit(main())
' "${timeout_seconds}" "$@"
}

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  echo "Usage: scripts/dev/ci_step_timer.sh <label> <command> [args...]"
  echo "Optional environment variables:"
  echo "  CI_STEP_TIMEOUT_SECONDS  Run the command under a timeout of this many seconds."
  echo "                           Uses GNU timeout(1), or Python 3 when GNU timeout is absent."
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
    run_with_python_timeout "${CI_STEP_TIMEOUT_SECONDS}" "$@"
    status=$?
  else
    echo "::error::CI_STEP_TIMEOUT_SECONDS is set but no supported timeout backend is available" >&2
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
