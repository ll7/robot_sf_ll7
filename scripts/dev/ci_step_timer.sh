#!/usr/bin/env bash
set -euo pipefail

start_python_timeout() {
  local timeout_seconds="$1"
  shift

  python3 -c '
import math
import os
import signal
import subprocess
import sys
import time

TIMEOUT_STATUS = 124
BACKEND_ERROR_STATUS = 125
TERMINATION_GRACE_SECONDS = 5
POLL_INTERVAL_SECONDS = 0.01


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_process_group_exit(
    child: subprocess.Popen[bytes],
    process_group_id: int,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        child.poll()
        if not process_group_exists(process_group_id):
            child.wait()
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))


def signal_process_group(process_group_id: int, signum: int) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        pass


def terminate_process_group(
    child: subprocess.Popen[bytes],
    process_group_id: int,
    initial_signal: int,
    *,
    initial_signal_sent: bool = False,
) -> bool:
    if not initial_signal_sent:
        signal_process_group(process_group_id, initial_signal)
    if wait_for_process_group_exit(child, process_group_id, TERMINATION_GRACE_SECONDS):
        return True

    print(
        "ci_step_timer: process group ignored the termination grace; sending SIGKILL",
        file=sys.stderr,
    )
    signal_process_group(process_group_id, signal.SIGKILL)
    if wait_for_process_group_exit(child, process_group_id, TERMINATION_GRACE_SECONDS):
        return True

    print("ci_step_timer: process group remained alive after SIGKILL", file=sys.stderr)
    return False


def main() -> int:
    try:
        timeout_seconds = float(sys.argv[1])
    except ValueError:
        print(
            f"ci_step_timer: invalid timeout in seconds: {sys.argv[1]!r}",
            file=sys.stderr,
        )
        return BACKEND_ERROR_STATUS

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        print("ci_step_timer: timeout must be finite and greater than zero", file=sys.stderr)
        return BACKEND_ERROR_STATUS

    child_ref: list[subprocess.Popen[bytes] | None] = [None]
    forwarded_signal: list[int | None] = [None]

    def forward_signal(signum: int, _frame: object) -> None:
        if forwarded_signal[0] is None:
            forwarded_signal[0] = signum
        child = child_ref[0]
        if child is not None:
            signal_process_group(child.pid, signum)

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward_signal)
    try:
        os.kill(os.getppid(), signal.SIGUSR1)
    except ProcessLookupError:
        pass
    if forwarded_signal[0] is not None:
        return 128 + forwarded_signal[0]

    try:
        child = subprocess.Popen(sys.argv[2:], start_new_session=True)
    except FileNotFoundError:
        print(f"ci_step_timer: command not found: {sys.argv[2]}", file=sys.stderr)
        return 127
    except PermissionError:
        print(f"ci_step_timer: command is not executable: {sys.argv[2]}", file=sys.stderr)
        return 126
    child_ref[0] = child
    if forwarded_signal[0] is not None:
        signal_process_group(child.pid, forwarded_signal[0])

    deadline = time.monotonic() + timeout_seconds
    while True:
        if forwarded_signal[0] is not None:
            signum = forwarded_signal[0]
            cleaned = terminate_process_group(
                child,
                child.pid,
                signum,
                initial_signal_sent=True,
            )
            return 128 + signum if cleaned else BACKEND_ERROR_STATUS

        status = child.poll()
        if forwarded_signal[0] is not None:
            continue
        if status is not None:
            return 128 - status if status < 0 else status

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cleaned = terminate_process_group(child, child.pid, signal.SIGTERM)
            return TIMEOUT_STATUS if cleaned else BACKEND_ERROR_STATUS
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))


raise SystemExit(main())
' "${timeout_seconds}" "$@" &
  python_timeout_pid=$!
}

python_timeout_pid=""
python_timeout_signal=""
python_timeout_signal_name=""
python_timeout_signal_forwarded=""
python_timeout_ready=""
python_timeout_ready_status=""
python_timeout_traps_installed=""
python_timeout_wait_trap_statuses=""

forward_pending_python_timeout_signal() {
  if [[ -n "${python_timeout_ready}" && -n "${python_timeout_pid}" \
    && -n "${python_timeout_signal_name}" && -z "${python_timeout_signal_forwarded}" ]]; then
    if kill -s "${python_timeout_signal_name}" "${python_timeout_pid}" 2>/dev/null; then
      python_timeout_signal_forwarded=1
    fi
  fi
}

forward_python_timeout_signal() {
  python_timeout_wait_trap_statuses="${python_timeout_wait_trap_statuses} $((128 + $2))"
  if [[ -z "${python_timeout_signal}" ]]; then
    python_timeout_signal_name="$1"
    python_timeout_signal="$2"
    status=$((128 + python_timeout_signal))
  fi
  forward_pending_python_timeout_signal
}

mark_python_timeout_ready() {
  python_timeout_wait_trap_statuses="${python_timeout_wait_trap_statuses} ${python_timeout_ready_status}"
  python_timeout_ready=1
  forward_pending_python_timeout_signal
}

wait_for_python_timeout() {
  local wait_status
  while true; do
    python_timeout_wait_trap_statuses=""
    wait "${python_timeout_pid}" 2>/dev/null
    wait_status=$?
    case " ${python_timeout_wait_trap_statuses} " in
      *" ${wait_status} "*) continue ;;
      *) break ;;
    esac
  done

  if [[ -n "${python_timeout_signal}" ]]; then
    status=$((128 + python_timeout_signal))
  else
    status="${wait_status}"
  fi
  python_timeout_pid=""
}

timeout_value_is_nonfinite() {
  [[ "$1" =~ ^[[:space:]]*[+-]?([nN][aA][nN]|[iI][nN][fF]([iI][nN][iI][tT][yY])?)[[:space:]]*$ ]]
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
    if timeout_value_is_nonfinite "${CI_STEP_TIMEOUT_SECONDS}"; then
      echo "ci_step_timer: timeout must be finite and greater than zero" >&2
      status=125
    else
      timeout "${CI_STEP_TIMEOUT_SECONDS}" "$@"
      status=$?
    fi
  elif command -v python3 >/dev/null 2>&1; then
    python_timeout_ready_signal_number="$(kill -l USR1)"
    python_timeout_ready_status=$((128 + python_timeout_ready_signal_number))
    trap 'forward_python_timeout_signal HUP 1' HUP
    trap 'forward_python_timeout_signal INT 2' INT
    trap 'forward_python_timeout_signal TERM 15' TERM
    trap 'mark_python_timeout_ready' USR1
    python_timeout_traps_installed=1
    start_python_timeout "${CI_STEP_TIMEOUT_SECONDS}" "$@"
    forward_pending_python_timeout_signal
    wait_for_python_timeout
  else
    echo "::error::CI_STEP_TIMEOUT_SECONDS is set but no supported timeout backend is available" >&2
    status=127
  fi
else
  "$@"
  status=$?
fi

if [[ -n "${python_timeout_signal}" ]]; then
  status=$((128 + python_timeout_signal))
fi
completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
end_seconds="$(date +%s)"
duration=$((end_seconds - start_seconds))
echo "ci_step_timer step_end label=\"${label}\" status=${status} duration_seconds=${duration} completed_at=${completed_at}"
echo "::notice title=\"${label}\" timing::status=${status} duration_seconds=${duration}"
echo "::endgroup::"

if [[ -n "${python_timeout_traps_installed}" ]]; then
  trap - HUP INT TERM USR1
fi
exit "${status}"
