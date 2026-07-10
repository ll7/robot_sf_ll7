#!/usr/bin/env python3
"""Run a validation command with bounded stdout and private full-log artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "compact_validation_summary.v2"
DEFAULT_EXCERPT_LINES = 40
DEFAULT_EXCERPT_WIDTH = 240
TIMEOUT_EXIT_CODE = 124

FAILURE_PATTERNS = re.compile(
    r"(FAILED|ERROR|FAILURES|Traceback|AssertionError|Exception|short test summary|"
    r"^\s*__+|::test_|ruff|mypy|pyright|exit code)",
    re.IGNORECASE,
)
PYTEST_NODE_PATTERN = re.compile(r"([\w./-]+\.py::[^\s]+)")


def _repo_artifact_dir() -> Path:
    """Return the private artifact directory for compact validation logs."""
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()) / "codex-agent-runs" / "active" / "compact-validation"
    return Path("output") / "tmp" / "compact-validation"


def _now_slug() -> str:
    """Return a UTC timestamp safe for artifact file names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _quote_command(command: list[str]) -> str:
    """Return a shell-readable representation of *command*."""
    return " ".join(shlex.quote(part) for part in command)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> str:
    """Terminate a timed-out process and descendants started in its process group."""
    cleanup_status = "process_group_terminated_and_waited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return cleanup_status
    except OSError:
        process.terminate()
        cleanup_status = "direct_process_terminated_and_waited"
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        cleanup_status = cleanup_status.replace("terminated", "killed")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
            cleanup_status = "direct_process_killed_and_waited"
        process.wait()
    return cleanup_status


def _failure_lines(text: str, *, limit: int, width: int) -> tuple[list[str], bool]:
    """Extract bounded failure-oriented lines and whether output was truncated."""
    lines = text.splitlines()
    interesting = [line for line in lines if FAILURE_PATTERNS.search(line)]
    if not interesting:
        excerpt = lines[-limit:]
        truncated = len(lines) > limit
    else:
        excerpt = interesting[:limit]
        truncated = len(interesting) > limit
    return [line[:width] for line in excerpt], truncated


def _pytest_node_ids(text: str, *, limit: int = 40) -> list[str]:
    """Extract unique pytest node ids from command output."""
    node_ids: list[str] = []
    seen: set[str] = set()
    for match in PYTEST_NODE_PATTERN.finditer(text):
        node_id = match.group(1).rstrip(":,")
        if node_id not in seen:
            seen.add(node_id)
            node_ids.append(node_id)
        if len(node_ids) >= limit:
            break
    return node_ids


def run_compact_validation(
    command: list[str],
    *,
    artifact_dir: Path | None = None,
    excerpt_lines: int = DEFAULT_EXCERPT_LINES,
    excerpt_width: int = DEFAULT_EXCERPT_WIDTH,
    cwd: Path | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run *command*, save full output, and return a compact summary payload."""
    if not command:
        raise ValueError("command must not be empty")
    cwd = cwd or Path.cwd()
    artifact_dir = artifact_dir or _repo_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = f"validation-{_now_slug()}-{os.getpid()}"
    log_path = artifact_dir / f"{base}.log"
    summary_path = artifact_dir / f"{base}.summary.json"

    started = time.monotonic()
    timed_out = False
    timeout_message = ""
    cleanup_status = "not_needed"
    try:
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = TIMEOUT_EXIT_CODE
                cleanup_status = _terminate_process_group(process)
                timeout_message = f"Command timed out after {exc.timeout:g} seconds."
                log_file.write(f"\n{timeout_message}\n".encode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = TIMEOUT_EXIT_CODE
        cleanup_status = "timeout_without_process_handle"
        timeout_message = f"Command timed out after {exc.timeout:g} seconds."
        log_path.write_text(f"{timeout_message}\n", encoding="utf-8", errors="replace")
    elapsed = time.monotonic() - started
    combined = log_path.read_text(encoding="utf-8", errors="replace")
    if returncode == 0:
        excerpt, truncated = [], False
    else:
        excerpt, truncated = _failure_lines(combined, limit=excerpt_lines, width=excerpt_width)
    failing_node_ids = _pytest_node_ids(combined) if returncode != 0 else []
    summary: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "command": command,
        "command_display": _quote_command(command),
        "cwd": str(cwd),
        "exit_code": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "timeout_message": timeout_message,
        "cleanup_status": cleanup_status,
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "excerpt_line_count": len(excerpt),
        "excerpt_truncated": truncated,
        "failing_node_ids": failing_node_ids,
        "failure_excerpt": excerpt,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for full log and summary JSON; defaults under git common dir.",
    )
    parser.add_argument(
        "--excerpt-lines",
        type=int,
        default=DEFAULT_EXCERPT_LINES,
        help="Maximum failure-oriented lines to print.",
    )
    parser.add_argument(
        "--excerpt-width",
        type=int,
        default=DEFAULT_EXCERPT_WIDTH,
        help="Maximum characters per excerpt line.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Abort the wrapped command after this many seconds and emit a timeout summary.",
    )
    parser.add_argument("--json", action="store_true", help="Print the compact summary as JSON.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a validation command is required after --")
    if args.excerpt_lines < 1:
        parser.error("--excerpt-lines must be positive")
    if args.excerpt_width < 20:
        parser.error("--excerpt-width must be at least 20")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def _print_human_summary(summary: dict[str, Any]) -> None:
    """Print a bounded human-readable validation summary."""
    print(f"Command: {summary['command_display']}")
    print(f"Exit code: {summary['exit_code']}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")
    if summary.get("timed_out"):
        print(f"Timed out: {summary['timeout_seconds']} seconds")
    print(f"Full log: {summary['log_path']}")
    print(f"Summary JSON: {summary['summary_path']}")
    if summary["failing_node_ids"]:
        print("Failing node ids:")
        for node_id in summary["failing_node_ids"]:
            print(f"- {node_id}")
    if summary["failure_excerpt"]:
        print("Failure excerpt:")
        for line in summary["failure_excerpt"]:
            print(line)
    if summary["excerpt_truncated"]:
        print("... additional matching lines omitted; see full log")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    summary = run_compact_validation(
        args.command,
        artifact_dir=args.artifact_dir,
        excerpt_lines=args.excerpt_lines,
        excerpt_width=args.excerpt_width,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human_summary(summary)
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
