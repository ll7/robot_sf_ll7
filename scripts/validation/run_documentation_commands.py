#!/usr/bin/env python3
"""Execute explicitly tagged, allowlisted onboarding command blocks (issue #8726).

Only fenced blocks whose info string carries the ``exec-doc`` (temporary
working directory) or ``exec-doc-root`` (repository root, declared explicitly)
tag are executed. Everything else is left untouched.

Tagged blocks use a line-oriented argv grammar. Each non-empty line must be an
entry in the explicit allowlist and is executed with ``shell=False``. Shell
syntax, alternate interpreters, placeholders, network commands, destructive
commands, absolute paths, and working-directory escapes fail closed with a
stable reason code. A block stops at its first nonzero command.

Execution has a finite wall-clock budget, a process-group cleanup path, and a
bounded combined stdout/stderr capture. ``--format json`` emits a canonical
sorted-key receipt without wall-clock fields; output digests are reproducible
when the normalized command output is identical, not a blanket claim about
arbitrary runtime output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ONBOARDING_PAGES = (
    "README.md",
    "docs/adoption_path.md",
    "docs/user-guide.md",
)

PROFILES = {"onboarding": ONBOARDING_PAGES}

TAG_TMP = "exec-doc"
TAG_ROOT = "exec-doc-root"

DEFAULT_TIMEOUT_S = 120.0
MAX_TIMEOUT_S = 300.0
MAX_OUTPUT_BYTES = 1024 * 1024
PROCESS_GROUP_GRACE_S = 0.5
READ_CHUNK_BYTES = 64 * 1024

FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<fence>`{3,})(?P<info>.*)$")
FENCE_CLOSE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,})\s*$")

PLACEHOLDER_RE = re.compile(r"(<[^>\n]+>|\{[^}\n]+\}|\$\w+|\bTODO\b|\bTBD\b)")
NETWORK_RE = re.compile(
    r"\b(curl|wget|ssh|scp|ftp|pip\s+install|uv\s+(add|sync|pip)|apt(-get)?|brew|npm\s+i|git\s+clone|gh\s+(pr|issue|run|api))\b"
)
DESTRUCTIVE_RE = re.compile(
    r"(rm\s+(-[rf]+\s+)+|mkfs|:?\(\s*\)\s*\{|shutdown|reboot|dd\s+[^;]*of=|chmod\s+-R\s+/|chown\s+-R?\s+\S+\s+/)"
)
ABSOLUTE_PATH_RE = re.compile(r"(?:^|\s)(/(?!dev/null\b)[A-Za-z0-9_][^\s\"']*)")
ESCAPE_RE = re.compile(r"(?:^|\s)(\.\.|~)(?:/|\s|$)")
# These are rejected before argv admission so a command cannot smuggle shell
# composition or expansion through a token that happens to be allowlisted.
SHELL_CONTROL_RE = re.compile(r"[;&|<>`$()]")
SHELL_TEXT_ESCAPE_RE = re.compile(r"[\\'\"\r]")
INTERPRETER_RE = re.compile(
    r"\b(?:bash|dash|fish|ksh|perl|php|python(?:\d+(?:\.\d+)?)?|ruby|sh|zsh|node|deno|source)\b"
)
SEED_RE = re.compile(r"-?\d{1,10}\Z")
SLEEP_RE = re.compile(r"(?:0|[1-9]\d*)(?:\.\d*)?\Z")

LOG_PREFIX_RE = re.compile(
    r"^(WARNING: All log messages.*|I\d{4} \S+ \S+ .*port\.cc.*|I\d{4} \S+ \S+ .*cpu_feature_guard\.cc.*|"
    r"To enable the following instructions:.*|W\d{4} \S+ .*|E\d{4} \S+ .*)$"
)

# These small utilities are side-effect free and keep the process supervisor
# directly testable. The documentation-facing commands are listed separately
# below so ``uv run`` cannot be used to launch an arbitrary program.
SAFE_UTILITY_ARGV = frozenset(
    {
        ("cat", "/dev/null"),
        ("false",),
        ("pwd",),
        ("true",),
        ("yes",),
    }
)
APPROVED_UV_ARGV = frozenset(
    {
        ("uv", "run", "robot-sf", "--help"),
        ("uv", "run", "robot-sf", "--version"),
        ("uv", "run", "robot-sf", "doctor", "--skip-env-smoke", "--skip-quickstart-smoke"),
        ("uv", "run", "robot-sf", "examples", "list"),
        ("uv", "run", "robot-sf", "recipe", "list"),
        ("uv", "run", "robot-sf", "recipe", "explain", "first-demo"),
    }
)


def normalize_output(text: str, tmpdir: str, *, repo_root: Path = REPO_ROOT) -> str:
    """Replace machine-specific paths and drop known runtime log prefixes."""
    text = text.replace(tmpdir, "<tmp>")
    text = text.replace(str(repo_root), "<repo>")
    text = text.replace(os.path.expanduser("~"), "<home>")
    kept = [line for line in text.splitlines() if not LOG_PREFIX_RE.match(line)]
    return "\n".join(kept).strip() + "\n"


def digest(text: str) -> str:
    """Return the SHA-256 hex digest of normalized command output."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _screen_text(command: str) -> str | None:
    """Return a diagnostic rejection for obviously unsafe command text."""
    if PLACEHOLDER_RE.search(command):
        return "placeholder_token"
    if NETWORK_RE.search(command):
        return "network_access"
    if DESTRUCTIVE_RE.search(command):
        return "destructive_token"
    if ABSOLUTE_PATH_RE.search(command):
        return "absolute_path"
    if ESCAPE_RE.search(command):
        return "output_escape"
    if SHELL_CONTROL_RE.search(command):
        return "shell_syntax"
    if INTERPRETER_RE.search(command):
        return "shell_wrapper"
    if SHELL_TEXT_ESCAPE_RE.search(command):
        return "shell_syntax"
    return None


def _is_approved_argv(argv: list[str]) -> bool:
    """Return whether one parsed argv is in the narrow documentation grammar."""
    argv_tuple = tuple(argv)
    if argv_tuple in SAFE_UTILITY_ARGV or argv_tuple in APPROVED_UV_ARGV:
        return True
    if len(argv) == 2 and argv[0] == "sleep" and SLEEP_RE.fullmatch(argv[1]):
        return float(argv[1]) <= MAX_TIMEOUT_S
    if (
        len(argv) == 6
        and argv[:5] == ["uv", "run", "robot-sf", "demo", "--seed"]
        and SEED_RE.fullmatch(argv[5]) is not None
    ):
        return True
    return False


def _parse_command(command: str) -> tuple[list[list[str]], str | None]:
    """Parse a tagged block into allowlisted argv lines or a stable rejection."""
    rejection = _screen_text(command)
    if rejection is not None:
        return [], rejection
    argv_lines: list[list[str]] = []
    for raw_line in command.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        argv = line.split()
        if not argv:
            continue
        if not _is_approved_argv(argv):
            return [], "unsupported_command"
        argv_lines.append(argv)
    if not argv_lines:
        return [], "empty_command"
    return argv_lines, None


def screen_command(command: str) -> str | None:
    """Return a stable rejection reason, or ``None`` when argv admission succeeds."""
    _, rejection = _parse_command(command)
    return rejection


@dataclass(frozen=True)
class BlockResult:
    """One tagged-block execution receipt."""

    source: str
    line: int
    command: str
    cwd_mode: str
    exit_code: int | None
    stdout_digest: str | None
    stderr_digest: str | None
    duration_s: float | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON receipt record.

        Wall-clock duration is intentionally omitted because it is diagnostic rather than
        reproducibility evidence and would make otherwise identical receipts differ between runs.
        """
        return {
            "source": self.source,
            "line": self.line,
            "command": self.command,
            "cwd_mode": self.cwd_mode,
            "exit_code": self.exit_code,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _ProcessResult:
    """Bounded process output and its supervisor status."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    reason: str


def _process_group_exists(pid: int) -> bool:
    """Return whether a POSIX process group still exists."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate a child and its descendants, escalating after a short grace period."""
    if os.name != "posix":
        process.terminate()
        try:
            process.wait(timeout=PROCESS_GROUP_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        process.terminate()

    deadline = time.monotonic() + PROCESS_GROUP_GRACE_S
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.01)

    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
    process.wait()


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    """Close the supervisor-owned pipe endpoints."""
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _capture_ready_pipe(
    selector: selectors.BaseSelector,
    key: selectors.SelectorKey,
    remaining_output: int,
) -> bool:
    """Read one ready pipe and return whether the output budget was exceeded."""
    stream = key.fileobj
    try:
        data = os.read(stream.fileno(), READ_CHUNK_BYTES)
    except OSError:
        data = b""
    if not data:
        selector.unregister(stream)
        return False
    if len(data) > remaining_output:
        key.data.extend(data[: max(0, remaining_output)])
        return True
    key.data.extend(data)
    return False


def _capture_process(
    process: subprocess.Popen[bytes],
    *,
    timeout_s: float,
    max_output_bytes: int,
) -> _ProcessResult:
    """Capture a process's output until it completes, times out, or reaches its cap."""
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    streams = ((process.stdout, stdout), (process.stderr, stderr))
    for stream, buffer in streams:
        if stream is not None:
            selector.register(stream, selectors.EVENT_READ, buffer)
    deadline = time.monotonic() + timeout_s
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                return _ProcessResult(None, bytes(stdout), bytes(stderr), "timeout")
            events = selector.select(remaining)
            if not events:
                _terminate_process_group(process)
                return _ProcessResult(None, bytes(stdout), bytes(stderr), "timeout")
            remaining_output = max_output_bytes - len(stdout) - len(stderr)
            for key, _ in events:
                if _capture_ready_pipe(selector, key, remaining_output):
                    _terminate_process_group(process)
                    return _ProcessResult(None, bytes(stdout), bytes(stderr), "output_limit")
                remaining_output = max_output_bytes - len(stdout) - len(stderr)

        remaining = max(0.0, deadline - time.monotonic())
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return _ProcessResult(None, bytes(stdout), bytes(stderr), "timeout")
        return _ProcessResult(returncode, bytes(stdout), bytes(stderr), "completed")
    finally:
        selector.close()
        _close_process_pipes(process)


def _run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    max_output_bytes: int,
) -> _ProcessResult:
    """Run one argv with a process-group timeout and bounded combined output."""
    if not argv:
        return _ProcessResult(None, b"", b"", "process_error")
    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            close_fds=True,
            start_new_session=os.name == "posix",
        )
    except OSError:
        return _ProcessResult(None, b"", b"", "process_error")
    try:
        return _capture_process(
            process,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
        )
    except (OSError, ValueError):
        _terminate_process_group(process)
        return _ProcessResult(None, b"", b"", "process_error")


def _decode_output(data: bytes) -> str:
    """Decode bounded child output without allowing invalid bytes to abort a receipt."""
    return data.decode("utf-8", errors="replace")


def _fence_open(line: str) -> tuple[int, str] | None:
    """Return a backtick fence length and info string for an opening fence."""
    match = FENCE_OPEN_RE.match(line)
    if match is None or "`" in match.group("info"):
        return None
    return len(match.group("fence")), match.group("info").strip()


def _is_fence_close(line: str, opening_length: int) -> bool:
    """Return whether *line* closes a fence of at least *opening_length* ticks."""
    match = FENCE_CLOSE_RE.match(line)
    return match is not None and len(match.group("fence")) >= opening_length


def iter_tagged_blocks(page: Path) -> list[tuple[int, str, str]]:
    """Return ``(line, cwd_mode, command)`` for tagged fences in one page."""
    blocks: list[tuple[int, str, str]] = []
    lines = page.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        opener = _fence_open(lines[index])
        if opener is None:
            index += 1
            continue
        opening_length, info = opener
        tokens = info.split()
        modes = [mode for mode in (TAG_TMP, TAG_ROOT) if mode in tokens]
        mode = modes[0] if modes else ""
        if len(modes) > 1:
            raise ValueError(f"Multiple execution tags at line {index + 1}")
        start = index + 1
        end = start
        while end < len(lines) and not _is_fence_close(lines[end], opening_length):
            end += 1
        if end == len(lines):
            if mode:
                raise ValueError(f"Unterminated tagged fence at line {index + 1}")
            break
        if mode:
            command = "\n".join(lines[start:end]).strip()
            if command:
                blocks.append((start + 1, mode, command))
        index = end + 1
    return blocks


def _invalid_timeout(timeout_s: float) -> bool:
    """Return whether a caller supplied timeout is outside the finite bound."""
    return not math.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > MAX_TIMEOUT_S


def _timeout_arg(value: str) -> float:
    """Parse a CLI timeout while retaining a finite upper bound."""
    try:
        timeout_s = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number") from exc
    if _invalid_timeout(timeout_s):
        raise argparse.ArgumentTypeError(
            f"timeout must be greater than 0 and at most {MAX_TIMEOUT_S:g} seconds"
        )
    return timeout_s


def run_block(
    source: str,
    line: int,
    command: str,
    cwd_mode: str,
    timeout_s: float,
    *,
    repo_root: Path = REPO_ROOT,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> BlockResult:
    """Screen and execute one tagged block, returning its deterministic receipt."""
    commands, rejection = _parse_command(command)
    if cwd_mode not in {TAG_TMP, TAG_ROOT}:
        rejection = "invalid_cwd_mode"
    elif _invalid_timeout(timeout_s):
        rejection = "invalid_timeout"
    elif max_output_bytes <= 0:
        rejection = "invalid_output_limit"
    if rejection is not None:
        return BlockResult(source, line, command, cwd_mode, None, None, None, None, rejection)

    root = repo_root.resolve()
    with tempfile.TemporaryDirectory(prefix="doc-command-") as tmpdir:
        cwd = root if cwd_mode == TAG_ROOT else Path(tmpdir)
        started = time.monotonic()
        deadline = started + timeout_s
        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        captured_bytes = 0
        for argv in commands:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return BlockResult(
                    source, line, command, cwd_mode, None, None, None, timeout_s, "timeout"
                )
            process_result = _run_process(
                argv,
                cwd=cwd,
                timeout_s=remaining,
                max_output_bytes=max_output_bytes - captured_bytes,
            )
            if process_result.reason != "completed":
                return BlockResult(
                    source,
                    line,
                    command,
                    cwd_mode,
                    None,
                    None,
                    None,
                    timeout_s,
                    process_result.reason,
                )
            stdout_parts.append(process_result.stdout)
            stderr_parts.append(process_result.stderr)
            captured_bytes += len(process_result.stdout) + len(process_result.stderr)
            if process_result.returncode != 0:
                stdout = normalize_output(
                    _decode_output(b"".join(stdout_parts)), tmpdir, repo_root=root
                )
                stderr = normalize_output(
                    _decode_output(b"".join(stderr_parts)), tmpdir, repo_root=root
                )
                return BlockResult(
                    source,
                    line,
                    command,
                    cwd_mode,
                    process_result.returncode,
                    digest(stdout),
                    digest(stderr),
                    round(time.monotonic() - started, 3),
                    "nonzero_exit",
                )

        stdout = normalize_output(_decode_output(b"".join(stdout_parts)), tmpdir, repo_root=root)
        stderr = normalize_output(_decode_output(b"".join(stderr_parts)), tmpdir, repo_root=root)
        return BlockResult(
            source,
            line,
            command,
            cwd_mode,
            0,
            digest(stdout),
            digest(stderr),
            round(time.monotonic() - started, 3),
            "ok",
        )


def run_profile(
    profile: str, root: Path = REPO_ROOT, timeout_s: float = DEFAULT_TIMEOUT_S
) -> list[BlockResult]:
    """Execute tagged blocks across profile pages in deterministic order."""
    root = root.resolve()
    results: list[BlockResult] = []
    for page in PROFILES[profile]:
        source = root / page
        for line, mode, command in iter_tagged_blocks(source):
            results.append(
                run_block(
                    source.relative_to(root).as_posix(),
                    line,
                    command,
                    mode,
                    timeout_s,
                    repo_root=root,
                )
            )
    results.sort(key=lambda result: (result.source, result.line))
    return results


def main(argv: list[str] | None = None) -> int:
    """Run tagged onboarding command blocks; exit 1 unless all succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 unless all tagged blocks pass")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="onboarding")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--timeout", type=_timeout_arg, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    results = run_profile(args.profile, timeout_s=args.timeout)
    if args.format == "json":
        print(json.dumps([result.as_dict() for result in results], indent=2, sort_keys=True))
    else:
        for result in results:
            print(f"{result.source}:{result.line}: exit={result.exit_code} [{result.reason}]")
        failed = sum(1 for result in results if result.reason != "ok")
        print(f"{len(results) - failed}/{len(results)} tagged block(s) passed.")
    failed = sum(1 for result in results if result.reason != "ok")
    if args.check and (failed or not results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
