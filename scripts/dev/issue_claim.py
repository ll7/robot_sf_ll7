#!/usr/bin/env python3
"""Claim GitHub issues for cross-machine agent work with an atomic remote ref.

The claim is a stable remote branch named ``agent-claims/issue-<number>``. Creating that
ref through GitHub's create-ref API is atomic: if another machine already created it, the
API call fails and this agent should skip the issue.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

DEFAULT_REMOTE = "origin"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_SOURCE_REF = "origin/main"
CLAIM_PREFIX = "agent-claims"
ISSUE_RE = re.compile(r"^[1-9][0-9]*$")


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result with the command that produced it."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def validate_issue_number(value: str) -> int:
    """Return a valid positive GitHub issue number."""
    if not ISSUE_RE.match(value):
        raise argparse.ArgumentTypeError("issue number must be a positive integer")
    return int(value)


def claim_ref(issue_number: int, *, prefix: str = CLAIM_PREFIX) -> str:
    """Return the full Git ref used as the cross-machine issue claim."""
    return f"refs/heads/{prefix}/issue-{issue_number}"


def short_claim_ref(issue_number: int, *, prefix: str = CLAIM_PREFIX) -> str:
    """Return the branch-style claim ref without ``refs/heads/``."""
    return f"{prefix}/issue-{issue_number}"


def _run(command: list[str]) -> CommandResult:
    """Run a command without invoking a shell."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return CommandResult(
        command=tuple(command),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def build_status_command(issue_number: int, *, remote: str) -> list[str]:
    """Build the command that checks whether a claim ref exists."""
    return ["git", "ls-remote", "--heads", remote, claim_ref(issue_number)]


def build_resolve_source_command(*, source_ref: str) -> list[str]:
    """Build the command that resolves the source commit for a new claim."""
    return ["git", "rev-parse", "--verify", f"{source_ref}^{{commit}}"]


def build_acquire_command(issue_number: int, *, repo: str, sha: str) -> list[str]:
    """Build the atomic GitHub ref creation command."""
    return [
        "gh",
        "api",
        "-X",
        "POST",
        f"repos/{repo}/git/refs",
        "-f",
        f"ref={claim_ref(issue_number)}",
        "-f",
        f"sha={sha}",
    ]


def build_release_command(issue_number: int, *, remote: str) -> list[str]:
    """Build the command that deletes a remote claim ref."""
    return ["git", "push", remote, f":{claim_ref(issue_number)}"]


def _status_from_ls_remote(
    result: CommandResult, *, issue_number: int, remote: str
) -> dict[str, Any]:
    """Convert ``git ls-remote`` output into the command payload."""
    if result.returncode != 0:
        return {
            "schema": "issue_claim.v1",
            "action": "status",
            "ok": False,
            "claimed": None,
            "issue": issue_number,
            "remote": remote,
            "claim_ref": short_claim_ref(issue_number),
            "error": (result.stderr or result.stdout).strip(),
            "command": list(result.command),
        }

    target_ref = claim_ref(issue_number)
    sha = None
    for line in (result.stdout or "").strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == target_ref:
            sha = parts[0]
            break

    if sha is None:
        return {
            "schema": "issue_claim.v1",
            "action": "status",
            "ok": True,
            "claimed": False,
            "issue": issue_number,
            "remote": remote,
            "claim_ref": short_claim_ref(issue_number),
            "sha": None,
            "command": list(result.command),
        }

    return {
        "schema": "issue_claim.v1",
        "action": "status",
        "ok": True,
        "claimed": True,
        "issue": issue_number,
        "remote": remote,
        "claim_ref": short_claim_ref(issue_number),
        "sha": sha,
        "command": list(result.command),
    }


def status_issue(issue_number: int, *, remote: str) -> dict[str, Any]:
    """Return whether a remote claim currently exists."""
    return _status_from_ls_remote(
        _run(build_status_command(issue_number, remote=remote)),
        issue_number=issue_number,
        remote=remote,
    )


def acquire_issue(issue_number: int, *, repo: str, remote: str, source_ref: str) -> dict[str, Any]:
    """Try to create the claim ref and return a machine-readable result."""
    source_result = _run(build_resolve_source_command(source_ref=source_ref))
    if source_result.returncode != 0:
        return {
            "schema": "issue_claim.v1",
            "action": "acquire",
            "ok": False,
            "claimed": False,
            "issue": issue_number,
            "repo": repo,
            "remote": remote,
            "source_ref": source_ref,
            "claim_ref": short_claim_ref(issue_number),
            "command": list(source_result.command),
            "stdout": source_result.stdout.strip(),
            "stderr": source_result.stderr.strip(),
            "error": "source_ref_resolution_failed",
        }

    sha = source_result.stdout.strip()
    result = _run(build_acquire_command(issue_number, repo=repo, sha=sha))
    ok = result.returncode == 0
    return {
        "schema": "issue_claim.v1",
        "action": "acquire",
        "ok": ok,
        "claimed": ok,
        "issue": issue_number,
        "repo": repo,
        "remote": remote,
        "source_ref": source_ref,
        "sha": sha,
        "claim_ref": short_claim_ref(issue_number),
        "command": list(result.command),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "error": None
        if ok
        else (
            "claim_ref_already_exists_or_create_ref_failed; run status to inspect the current owner "
            "signal and skip this issue unless the claim is confirmed stale"
        ),
    }


def release_issue(issue_number: int, *, remote: str) -> dict[str, Any]:
    """Delete the remote claim ref."""
    status = status_issue(issue_number, remote=remote)
    if status["ok"] and not status["claimed"]:
        return {
            "schema": "issue_claim.v1",
            "action": "release",
            "ok": True,
            "claimed": False,
            "issue": issue_number,
            "remote": remote,
            "claim_ref": short_claim_ref(issue_number),
            "command": status["command"],
            "stdout": "Ref does not exist, nothing to release.",
            "stderr": "",
            "error": None,
        }

    result = _run(build_release_command(issue_number, remote=remote))
    ok = result.returncode == 0
    return {
        "schema": "issue_claim.v1",
        "action": "release",
        "ok": ok,
        "claimed": False if ok else None,
        "issue": issue_number,
        "remote": remote,
        "claim_ref": short_claim_ref(issue_number),
        "command": list(result.command),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "error": None
        if ok
        else "claim_ref_release_failed; inspect remote branch state before retrying",
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "status", "release"))
    parser.add_argument("issue", type=validate_issue_number)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository as OWNER/REPO.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote to use for the claim.")
    parser.add_argument(
        "--source-ref",
        default=DEFAULT_SOURCE_REF,
        help="Local ref to push when acquiring the claim. Defaults to origin/main.",
    )
    return parser


def _dump_json(payload: dict[str, Any]) -> None:
    """Print stable JSON to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    if args.action == "status":
        payload = status_issue(args.issue, remote=args.remote)
    elif args.action == "acquire":
        payload = acquire_issue(
            args.issue,
            repo=args.repo,
            remote=args.remote,
            source_ref=args.source_ref,
        )
    else:
        payload = release_issue(args.issue, remote=args.remote)

    _dump_json(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
