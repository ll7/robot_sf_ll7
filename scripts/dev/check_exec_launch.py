#!/usr/bin/env python3
"""Deterministic process launch preflight, failure attribution, and retry helper (issue #9321).

Provides:
1. Pre-spawn executable and working directory preflight checks.
2. Typed `exec_launch_receipt.v1` distinguishing process creation / launch
   failures ({phase: "preflight"|"launch"}) from command exit ({phase: "execution"}).
3. Single bounded retry for allowlisted read-only git and gh subcommands when
   process creation fails transiently.
4. Suggested retry command generation for non-allowlisted invocations.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA = "exec_launch_receipt.v1"

READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "diff",
        "log",
        "ls-remote",
        "rev-parse",
        "show",
        "status",
    }
)

READ_ONLY_GH_SUBCOMMANDS = frozenset(
    {
        "issue list",
        "issue view",
        "pr checks",
        "pr list",
        "pr view",
    }
)


def _base_cmd_name(cmd_path: str) -> str:
    """Return the base command name stripped of directory path."""
    return Path(cmd_path).name.lower()


def _is_git_read_only(command: Sequence[str]) -> bool:
    """Return True if git command runs an allowlisted read-only subcommand."""
    idx = 1
    while idx < len(command):
        arg = command[idx]
        if arg in ("-C", "--git-dir", "--work-tree"):
            idx += 2
            continue
        if arg.startswith(("-C", "--git-dir=", "--work-tree=")):
            idx += 1
            continue
        if arg.startswith("-"):
            idx += 1
            continue
        return arg in READ_ONLY_GIT_SUBCOMMANDS
    return False


def _is_gh_api_get(command: Sequence[str]) -> bool:
    """Return True if gh api invocation uses GET."""
    idx = 1
    method = "GET"
    while idx < len(command):
        arg = command[idx]
        if arg in ("-X", "--method") and idx + 1 < len(command):
            method = command[idx + 1].upper()
            break
        if arg.startswith(("-X", "--method=")):
            method = arg.split("=", 1)[-1].upper()
            break
        idx += 1
    return method == "GET"


def _is_gh_read_only(command: Sequence[str]) -> bool:
    """Return True if gh command runs an allowlisted read-only subcommand."""
    args = [a for a in command[1:] if not a.startswith("-")]
    if not args:
        return False

    if args[0] == "api":
        return _is_gh_api_get(command)

    if len(args) >= 2:
        two_word = f"{args[0]} {args[1]}"
        if two_word in READ_ONLY_GH_SUBCOMMANDS:
            return True

    return False


def is_retry_allowlisted(command: Sequence[str]) -> bool:
    """Return True if command is a recognized read-only git or gh invocation.

    Allowlisted commands are safe to retry automatically once when process
    creation fails (e.g. transient ENOENT / CreateProcess failure).
    """
    if not command:
        return False

    base = _base_cmd_name(command[0])
    if base == "git":
        return _is_git_read_only(command)
    if base == "gh":
        return _is_gh_read_only(command)
    return False


@dataclass(frozen=True, slots=True)
class ExecLaunchReceipt:
    """Typed execution launch receipt."""

    schema: str
    command: list[str]
    cwd: str
    resolved_executable: str | None
    phase: str  # "preflight" | "launch" | "execution"
    status: str  # "ok" | "launch_failed" | "failed"
    returncode: int | None
    error: str | None
    retried: bool
    suggested_retry: str | None
    stdout: str | None = None
    stderr: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt to JSON-serializable dictionary."""
        return asdict(self)


def check_launch_preflight(
    command: Sequence[str],
    cwd: str | Path | None = None,
) -> tuple[bool, str | None, str | None]:
    """Verify cwd and resolve executable before spawning.

    Returns:
        tuple of (passed, resolved_executable, error_message)
    """
    if not command:
        return False, None, "command sequence is empty"

    target_cwd = Path(cwd) if cwd is not None else Path.cwd()
    if not target_cwd.is_dir():
        return False, None, f"working directory does not exist or is not a directory: {target_cwd}"

    exe = command[0]
    resolved: str | None = None

    if os.path.sep in exe or (os.path.altsep and os.path.altsep in exe):
        # Explicit path
        exe_path = Path(exe)
        if not exe_path.is_absolute():
            exe_path = (target_cwd / exe_path).resolve()
        if exe_path.is_file() and os.access(exe_path, os.X_OK):
            resolved = str(exe_path)
        else:
            return False, None, f"executable path not found or not executable: {exe}"
    else:
        # Search PATH
        path_var = os.environ.get("PATH", "")
        resolved = shutil.which(exe, path=path_var)
        if resolved is None:
            return False, None, f"executable not found in PATH: {exe}"

    return True, resolved, None


def run_with_launch_check(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    text: bool = True,
    timeout: float | None = None,
    allow_retry: bool = True,
    retry_delay_s: float = 0.05,
) -> ExecLaunchReceipt:
    """Execute a command with pre-spawn validation, typed failure attribution, and retry.

    Args:
        command: Command and arguments sequence.
        cwd: Working directory (defaults to current directory).
        env: Environment mapping (defaults to current environment).
        capture_output: Whether to capture stdout and stderr in the receipt.
        text: Whether captured output is decoded as text.
        timeout: Optional execution timeout in seconds.
        allow_retry: Whether to attempt automatic single retry for allowlisted commands.
        retry_delay_s: Delay before retrying an allowlisted command.

    Returns:
        ExecLaunchReceipt describing the launch phase, status, and execution outcome.
    """
    cmd_list = list(command)
    target_cwd_str = str(Path(cwd).resolve()) if cwd is not None else str(Path.cwd().resolve())

    preflight_ok, resolved_exe, preflight_error = check_launch_preflight(cmd_list, cwd=cwd)
    if not preflight_ok:
        can_retry = allow_retry and is_retry_allowlisted(cmd_list)
        suggested = None if can_retry else shlex.join(cmd_list)
        return ExecLaunchReceipt(
            schema=SCHEMA,
            command=cmd_list,
            cwd=target_cwd_str,
            resolved_executable=resolved_exe,
            phase="preflight",
            status="launch_failed",
            returncode=None,
            error=preflight_error,
            retried=False,
            suggested_retry=suggested,
        )

    # Attempt process launch
    def _do_spawn() -> tuple[subprocess.Popen[Any] | None, str | None]:
        try:
            proc = subprocess.Popen(
                cmd_list,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=text,
            )
            return proc, None
        except OSError as exc:
            return None, f"process spawn failed: {type(exc).__name__}: {exc}"

    process, spawn_error = _do_spawn()
    retried = False

    if process is None:
        # Launch failed
        if allow_retry and is_retry_allowlisted(cmd_list):
            retried = True
            time.sleep(retry_delay_s)
            process, spawn_error = _do_spawn()

    if process is None:
        suggested = shlex.join(cmd_list)
        return ExecLaunchReceipt(
            schema=SCHEMA,
            command=cmd_list,
            cwd=target_cwd_str,
            resolved_executable=resolved_exe,
            phase="launch",
            status="launch_failed",
            returncode=None,
            error=spawn_error,
            retried=retried,
            suggested_retry=suggested,
        )

    # Process launched successfully -> execution phase
    stdout_data: str | None = None
    stderr_data: str | None = None
    returncode: int | None = None
    exec_error: str | None = None

    try:
        out, err = process.communicate(timeout=timeout)
        returncode = process.returncode
        if capture_output:
            stdout_data = (
                out
                if isinstance(out, str)
                else (out.decode("utf-8", errors="replace") if out else None)
            )
            stderr_data = (
                err
                if isinstance(err, str)
                else (err.decode("utf-8", errors="replace") if err else None)
            )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        returncode = process.returncode
        exec_error = f"command timed out after {timeout} seconds"
    except Exception as exc:  # noqa: BLE001
        process.kill()
        returncode = process.returncode
        exec_error = f"execution failed: {type(exc).__name__}: {exc}"

    status = "ok" if returncode == 0 else "failed"
    return ExecLaunchReceipt(
        schema=SCHEMA,
        command=cmd_list,
        cwd=target_cwd_str,
        resolved_executable=resolved_exe,
        phase="execution",
        status=status,
        returncode=returncode,
        error=exec_error,
        retried=retried,
        suggested_retry=None if status == "ok" else shlex.join(cmd_list),
        stdout=stdout_data,
        stderr=stderr_data,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for check_exec_launch."""
    parser = argparse.ArgumentParser(
        description="Preflight check, failure attribution, and retry helper for process execution."
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Target working directory for execution preflight and launch.",
    )
    parser.add_argument(
        "--receipt-file",
        type=Path,
        default=None,
        help="Path to write the execution receipt JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print execution receipt JSON to stdout.",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable automatic retry even for allowlisted commands.",
    )
    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="Capture stdout and stderr in the receipt.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command and arguments to execute (prefix with -- if needed).",
    )

    args = parser.parse_args(argv)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        parser.error("no command specified to execute")

    capture_output = args.capture_output or args.json
    receipt = run_with_launch_check(
        cmd,
        cwd=args.cwd,
        capture_output=capture_output,
        allow_retry=not args.no_retry,
    )

    receipt_dict = receipt.to_dict()

    if args.receipt_file:
        args.receipt_file.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_file.write_text(json.dumps(receipt_dict, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(receipt_dict, indent=2))
    elif receipt.status == "launch_failed":
        print(
            f"check_exec_launch: [{receipt.phase}] {receipt.error}",
            file=sys.stderr,
        )
        if receipt.suggested_retry:
            print(f"suggested retry: {receipt.suggested_retry}", file=sys.stderr)

    if receipt.phase in ("preflight", "launch"):
        return 127
    return receipt.returncode if receipt.returncode is not None else 1


if __name__ == "__main__":
    sys.exit(main())
