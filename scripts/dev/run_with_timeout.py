"""Run a command with a portable wall-clock timeout.

This helper backs ``ci_step_timer.sh`` on platforms such as GitHub's macOS
runners, where GNU ``timeout(1)`` is not installed. It starts the command in
its own process group so a timeout terminates child processes as well as the
immediate command.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_TIMEOUT_EXIT_CODE = 124
_COMMAND_NOT_FOUND_EXIT_CODE = 127
_COMMAND_NOT_EXECUTABLE_EXIT_CODE = 126
_TERMINATION_GRACE_SECONDS = 5.0


def _positive_timeout(value: str) -> float:
    """Parse a strictly positive timeout in seconds."""
    try:
        timeout_seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid timeout: {value!r}") from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return timeout_seconds


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate a timed-out process group, escalating after a short grace period."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_with_timeout(command: Sequence[str], timeout_seconds: float) -> int:
    """Run ``command`` for at most ``timeout_seconds`` and return a shell-style status."""
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except FileNotFoundError:
        print(f"run_with_timeout: command not found: {command[0]}", file=sys.stderr)
        return _COMMAND_NOT_FOUND_EXIT_CODE
    except PermissionError:
        print(f"run_with_timeout: command is not executable: {command[0]}", file=sys.stderr)
        return _COMMAND_NOT_EXECUTABLE_EXIT_CODE

    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"run_with_timeout: command timed out after {timeout_seconds:g} seconds",
            file=sys.stderr,
        )
        _terminate_process_group(process)
        return _TIMEOUT_EXIT_CODE

    if return_code < 0:
        return 128 - return_code
    return return_code


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeout_seconds", type=_positive_timeout)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command under the portable timeout."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after timeout_seconds")
    return run_with_timeout(command, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
