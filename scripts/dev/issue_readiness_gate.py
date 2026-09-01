#!/usr/bin/env python3
"""Gate ``state:ready`` on a fresh live admission check after issue creation.

The discovery write sequence is: create the issue without ``state:ready``, exact-read
the returned item through the canonical REST owner, evaluate a private prospective
``state:ready`` label when no state label exists, run the live ``goal_issue_admission``
check-only boundary, re-read for drift, and add ``state:ready`` only after a passing,
unclaimed, current check with a verified label readback. Any failed, stale, or
unavailable check leaves readiness absent and returns a stable JSON outcome naming the
canonical result. Retries are idempotent and never remove labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from typing import Any

from scripts.dev import gh_issue_rest, gh_pr_label_rest, goal_issue_admission
from scripts.dev.issue_implementability import READY_LABEL, preflight_body_file
from scripts.dev.issue_state_taxonomy import state_labels

DEFAULT_REMOTE = "origin"
DEFAULT_SOURCE_REF = "origin/main"
ISSUE_URL_RE = re.compile(r"/issues/(\d+)\s*$")

TERMINAL_OUTCOMES = frozenset(
    {
        "ready",
        "already_ready",
        "needs_spec",
        "blocked",
        "parent",
        "human_decision",
        "already_claimed",
        "state_conflict",
        "wrong_owner_repo",
        "needs_dependency",
        "stale_route_state",
    }
)


def _fingerprint(issue: dict[str, Any]) -> str:
    """Return a digest binding the read fields that readiness depends on."""
    payload = json.dumps(
        {
            "body": issue.get("body") or "",
            "title": issue.get("title") or "",
            "labels": list(issue.get("labels") or []),
            "assignees": list(issue.get("assignees") or []),
            "state": issue.get("state") or "",
            "updated_at": issue.get("updated_at") or "",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _result(
    outcome: str, *, issue: int | None, ready_added: bool, verified: bool, **extra: Any
) -> dict[str, Any]:
    """Build one stable gate result payload."""
    payload: dict[str, Any] = {
        "schema": "issue_readiness_gate.v1",
        "outcome": outcome,
        "issue": issue,
        "ready_label": READY_LABEL,
        "ready_added": ready_added,
        "verified": verified,
    }
    payload.update(extra)
    return payload


def _exact_read(
    number: int, *, repo: str, phase: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (issue, None) on success or (None, error_result) on failure."""
    issue = gh_issue_rest.fetch_issue(number, repo=repo)
    if issue.get("status") != "ok":
        return None, _result(
            "error",
            issue=number,
            ready_added=False,
            verified=False,
            phase=phase,
            error=issue.get("error", "exact read failed"),
        )
    return issue, None


def _evaluate_admission(
    number: int,
    *,
    repo: str,
    remote: str,
    source_ref: str,
    prospective_ready: bool = False,
) -> dict[str, Any] | None:
    """Run the live admission boundary and return a stop result when not ready."""
    admission = goal_issue_admission.admit_issue(
        number,
        repo=repo,
        remote=remote,
        source_ref=source_ref,
        check_only=True,
        prospective_ready=prospective_ready,
    )
    preflight = admission.get("preflight")
    if isinstance(preflight, dict):
        classification = str(preflight.get("classification") or "")
        ready = preflight.get("ready") is True
        reasons = preflight.get("reasons")
    else:
        classification = ""
        ready = False
        reasons = []
    if admission.get("ok") is True and ready:
        return None
    return _result(
        classification or "error",
        issue=number,
        ready_added=False,
        verified=False,
        reasons=list(reasons) if isinstance(reasons, list) else [str(reasons)],
    )


def _write_and_verify_readiness(number: int, *, repo: str) -> dict[str, Any]:
    """Add readiness and verify the post-write label set via one exact read."""
    label_write = gh_pr_label_rest.add_label(number, READY_LABEL, repo=repo)
    if label_write.get("status") != "ok":
        return _result(
            "error",
            issue=number,
            ready_added=False,
            verified=False,
            phase="label-write",
            error=str(label_write.get("error", "label write failed")),
        )
    final_read, error = _exact_read(number, repo=repo, phase="readback")
    if error is not None:
        return error
    assert final_read is not None
    if READY_LABEL not in final_read.get("labels", []):
        return _result(
            "error",
            issue=number,
            ready_added=True,
            verified=False,
            phase="readback",
            error="readiness label absent after verified write",
        )
    return _result("ready", issue=number, ready_added=True, verified=True)


def gate_issue(
    number: int,
    *,
    repo: str,
    remote: str = DEFAULT_REMOTE,
    source_ref: str = DEFAULT_SOURCE_REF,
) -> dict[str, Any]:
    """Run the live admission gate for one issue and conditionally add readiness."""
    first_read, error = _exact_read(number, repo=repo, phase="initial")
    if error is not None:
        return error
    assert first_read is not None
    if str(first_read.get("state", "")).upper() != "OPEN":
        return _result(
            "state_conflict",
            issue=number,
            ready_added=False,
            verified=False,
            reason=f"issue state is {first_read.get('state')!r}, not open",
        )
    fingerprint = _fingerprint(first_read)

    prospective_ready = not state_labels(set(first_read.get("labels", [])))
    admission_result = _evaluate_admission(
        number,
        repo=repo,
        remote=remote,
        source_ref=source_ref,
        prospective_ready=prospective_ready,
    )
    if admission_result is not None:
        return admission_result

    second_read, error = _exact_read(number, repo=repo, phase="pre-write")
    if error is not None:
        return error
    assert second_read is not None
    if _fingerprint(second_read) != fingerprint:
        return _result(
            "drift",
            issue=number,
            ready_added=False,
            verified=False,
            reason="issue drifted between admission and the readiness write",
        )
    if READY_LABEL in second_read.get("labels", []):
        return _result(
            "already_ready",
            issue=number,
            ready_added=False,
            verified=True,
            reason="readiness label already present",
        )

    return _write_and_verify_readiness(number, repo=repo)


def create_issue(
    *,
    title: str,
    body_file: str,
    labels: list[str],
    repo: str,
    remote: str = DEFAULT_REMOTE,
    source_ref: str = DEFAULT_SOURCE_REF,
) -> dict[str, Any]:
    """Create one issue without readiness, then run :func:`gate_issue` on it.

    ``state:ready`` is stripped from the initial label set; readiness may only be
    produced by a passing live admission check.
    """
    initial_labels = [label for label in labels if label != READY_LABEL]
    try:
        preflight = preflight_body_file(body_file)
    except OSError as exc:
        return _result(
            "error",
            issue=None,
            ready_added=False,
            verified=False,
            phase="create",
            error=f"body file unreadable: {exc}",
        )
    if preflight.get("ready") is not True:
        return _result(
            "preflight_rejected",
            issue=None,
            ready_added=False,
            verified=False,
            phase="create",
            missing_fields=list(preflight.get("missing_fields", [])),
            body_sha256=preflight.get("body_sha256", ""),
        )
    args = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--body-file",
        body_file,
    ]
    for label in initial_labels:
        args.extend(["--label", label])
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return _result(
            "error",
            issue=None,
            ready_added=False,
            verified=False,
            phase="create",
            error=(result.stderr or result.stdout).strip()[:500],
        )
    match = ISSUE_URL_RE.search(result.stdout.strip())
    if match is None:
        return _result(
            "error",
            issue=None,
            ready_added=False,
            verified=False,
            phase="create",
            error=f"could not parse created issue URL from: {result.stdout.strip()[:200]}",
        )
    number = int(match.group(1))
    outcome = gate_issue(number, repo=repo, remote=remote, source_ref=source_ref)
    return outcome


def _emit(payload: dict[str, Any], *, as_json: bool) -> int:
    """Print the result payload and return the process exit code."""
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            f"outcome={payload['outcome']} issue={payload.get('issue')} "
            f"ready_added={payload['ready_added']} verified={payload['verified']}"
        )
        if payload.get("reasons"):
            print("reasons: " + "; ".join(str(reason) for reason in payload["reasons"]))
        if payload.get("error"):
            print(f"error: {payload['error']}")
    if payload["outcome"] in TERMINAL_OUTCOMES:
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    """Run the readiness gate CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate_parser = subparsers.add_parser("gate", help="gate readiness for an existing issue")
    gate_parser.add_argument("number", type=int)
    gate_parser.add_argument("--repo", default=gh_issue_rest.DEFAULT_REPO)
    gate_parser.add_argument("--remote", default=DEFAULT_REMOTE)
    gate_parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    gate_parser.add_argument("--json", action="store_true")

    create_parser = subparsers.add_parser("create", help="create an issue, then gate readiness")
    create_parser.add_argument("--title", required=True)
    create_parser.add_argument("--body-file", required=True)
    create_parser.add_argument("--label", action="append", default=[])
    create_parser.add_argument("--repo", default=gh_issue_rest.DEFAULT_REPO)
    create_parser.add_argument("--remote", default=DEFAULT_REMOTE)
    create_parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    create_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "gate":
        payload = gate_issue(
            args.number,
            repo=args.repo,
            remote=args.remote,
            source_ref=args.source_ref,
        )
    else:
        payload = create_issue(
            title=args.title,
            body_file=args.body_file,
            labels=list(args.label),
            repo=args.repo,
            remote=args.remote,
            source_ref=args.source_ref,
        )
    return _emit(payload, as_json=bool(args.json))


if __name__ == "__main__":
    sys.exit(main())
