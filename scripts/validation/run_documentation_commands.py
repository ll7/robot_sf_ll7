#!/usr/bin/env python3
"""Execute explicitly tagged safe onboarding shell blocks (issue #8726).

Only fenced blocks whose info string carries the ``exec-doc`` (temporary
working directory) or ``exec-doc-root`` (repository root, declared explicitly)
tag are executed. Everything else is left untouched.

Before execution each tagged command is screened for placeholders, network
access, destructive tokens, absolute paths, and output escape; violations fail
closed with a stable reason code and nothing runs. Execution uses a bounded
timeout in a fresh temporary directory (or the repository root for
``exec-doc-root``), and stdout/stderr are normalized (temporary and repository
paths replaced, runtime log prefixes dropped) before digesting, so receipts are
deterministic across machines.

Exit status is ``0`` when every tagged block succeeds and ``1`` otherwise.
``--format json`` emits the byte-stable machine-readable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
FENCE_RE = re.compile(r"^\s*```(.*)$")

PLACEHOLDER_RE = re.compile(r"(<[^>\n]+>|\{[^}\n]+\}|\$\w+|\bTODO\b|\bTBD\b)")
NETWORK_RE = re.compile(
    r"\b(curl|wget|ssh|scp|ftp|pip\s+install|uv\s+(add|sync|pip)|apt(-get)?|brew|npm\s+i|git\s+clone|gh\s+(pr|issue|run|api))\b"
)
DESTRUCTIVE_RE = re.compile(
    r"(rm\s+(-[rf]+\s+)+|mkfs|:?\(\s*\)\s*\{|shutdown|reboot|dd\s+[^;]*of=|chmod\s+-R\s+/|chown\s+-R?\s+\S+\s+/)"
)
ABSOLUTE_PATH_RE = re.compile(r"(?:^|\s)(/(?!dev/null\b)[A-Za-z0-9_][^\s\"']*)")
ESCAPE_RE = re.compile(r"(?:^|\s)(\.\.|~)(?:/|\s|$)")
# Tagged blocks run in CI, so shell composition and alternate interpreters are not a safe
# execution boundary even when their nested text does not contain one of the blocked tokens.
SHELL_CONTROL_RE = re.compile(r"[;&|<>`$()]")
INTERPRETER_RE = re.compile(
    r"\b(?:bash|dash|fish|ksh|perl|php|python(?:\d+(?:\.\d+)?)?|ruby|sh|zsh|node|deno)\b"
)

LOG_PREFIX_RE = re.compile(
    r"^(WARNING: All log messages.*|I\d{4} \S+ \S+ .*port\.cc.*|I\d{4} \S+ \S+ .*cpu_feature_guard\.cc.*|"
    r"To enable the following instructions:.*|W\d{4} \S+ .*|E\d{4} \S+ .*)$"
)

DEFAULT_TIMEOUT_S = 120.0


def normalize_output(text: str, tmpdir: str) -> str:
    """Replace machine-specific paths and drop runtime log prefixes."""
    text = text.replace(tmpdir, "<tmp>")
    text = text.replace(str(REPO_ROOT), "<repo>")
    text = text.replace(os.path.expanduser("~"), "<home>")
    kept = [line for line in text.splitlines() if not LOG_PREFIX_RE.match(line)]
    return "\n".join(kept).strip() + "\n"


def digest(text: str) -> str:
    """Return the SHA-256 hex digest of normalized command output."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def screen_command(command: str) -> str | None:
    """Return a stable rejection reason, or None when the command may run."""
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
    return None


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


def iter_tagged_blocks(page: Path) -> list[tuple[int, str, str]]:
    """Return ``(line, cwd_mode, command)`` for tagged fences in one page."""
    blocks: list[tuple[int, str, str]] = []
    lines = page.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        match = FENCE_RE.match(lines[index])
        if match:
            tokens = match.group(1).split()
            mode = TAG_TMP if TAG_TMP in tokens else (TAG_ROOT if TAG_ROOT in tokens else "")
            if mode:
                start = index + 1
                end = start
                while end < len(lines) and not FENCE_RE.match(lines[end]):
                    end += 1
                if end == len(lines):
                    raise ValueError(f"Unterminated tagged fence at line {index + 1}")
                command = "\n".join(lines[start:end]).strip()
                if command:
                    blocks.append((start + 1, mode, command))
                index = end + 1
                continue
        index += 1
    return blocks


def run_block(source: str, line: int, command: str, cwd_mode: str, timeout_s: float) -> BlockResult:
    """Screen then execute one tagged block, returning its receipt."""
    rejection = screen_command(command)
    workdir = REPO_ROOT if cwd_mode == TAG_ROOT else None
    if rejection is not None:
        return BlockResult(source, line, command, cwd_mode, None, None, None, None, rejection)
    with tempfile.TemporaryDirectory(prefix="doc-command-") as tmpdir:
        cwd = str(REPO_ROOT) if workdir is not None else tmpdir
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return BlockResult(
                source, line, command, cwd_mode, None, None, None, timeout_s, "timeout"
            )
        duration = time.monotonic() - start
        stdout = normalize_output(proc.stdout, tmpdir)
        stderr = normalize_output(proc.stderr, tmpdir)
        reason = "ok" if proc.returncode == 0 else "nonzero_exit"
        return BlockResult(
            source,
            line,
            command,
            cwd_mode,
            proc.returncode,
            digest(stdout),
            digest(stderr),
            round(duration, 3),
            reason,
        )


def run_profile(
    profile: str, root: Path = REPO_ROOT, timeout_s: float = DEFAULT_TIMEOUT_S
) -> list[BlockResult]:
    """Execute tagged blocks across profile pages in deterministic order."""
    results: list[BlockResult] = []
    for page in PROFILES[profile]:
        source = root / page
        for line, mode, command in iter_tagged_blocks(source):
            results.append(run_block(str(source.relative_to(root)), line, command, mode, timeout_s))
    results.sort(key=lambda r: (r.source, r.line))
    return results


def main(argv: list[str] | None = None) -> int:
    """Run tagged onboarding command blocks; exit 1 unless all succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 unless all tagged blocks pass")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="onboarding")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    results = run_profile(args.profile, timeout_s=args.timeout)
    if args.format == "json":
        print(json.dumps([r.as_dict() for r in results], indent=2))
    else:
        for result in results:
            print(f"{result.source}:{result.line}: exit={result.exit_code} [{result.reason}]")
        failed = sum(1 for r in results if r.reason != "ok")
        print(f"{len(results) - failed}/{len(results)} tagged block(s) passed.")
    failed = sum(1 for r in results if r.reason != "ok")
    if args.check and (failed or not results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
