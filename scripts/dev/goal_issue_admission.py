#!/usr/bin/env python3
"""Gate an atomic issue claim on the live issue implementation contract."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from scripts.dev import issue_claim, issue_implementability

SCHEMA = "goal_issue_admission.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
DEFAULT_SOURCE_REF = "origin/main"


def admit_issue(
    issue_number: int,
    *,
    repo: str,
    remote: str,
    source_ref: str,
    check_only: bool,
) -> dict[str, Any]:
    """Evaluate one live issue and create its atomic claim only after a pass."""
    preflight = issue_implementability.live_issue_report(
        issue_number,
        repo=repo,
        remote=remote,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": issue_number,
        "repo": repo,
        "remote": remote,
        "source_ref": source_ref,
        "check_only": check_only,
        "preflight": preflight,
        "write_attempted": False,
        "claim": None,
        "ok": False,
    }
    if preflight.get("ready") is not True:
        payload["outcome"] = "not_admitted"
        return payload
    if check_only:
        payload["ok"] = True
        payload["outcome"] = "ready_check_only"
        return payload

    payload["write_attempted"] = True
    claim = issue_claim.acquire_issue(
        issue_number,
        repo=repo,
        remote=remote,
        source_ref=source_ref,
    )
    payload["claim"] = claim
    payload["ok"] = claim.get("ok") is True
    payload["outcome"] = "claim_acquired" if payload["ok"] else "claim_failed"
    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue", type=int, help="Positive GitHub issue number.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument(
        "--remote", default=DEFAULT_REMOTE, help="Git remote used by issue claims."
    )
    parser.add_argument(
        "--source-ref",
        default=DEFAULT_SOURCE_REF,
        help="Exact local source ref used by the atomic claim.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Evaluate admission without attempting an issue-claim write.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    if args.issue <= 0:
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "issue": args.issue,
            "ok": False,
            "outcome": "error",
            "write_attempted": False,
            "error": "issue number must be positive",
        }
    else:
        try:
            payload = admit_issue(
                args.issue,
                repo=args.repo,
                remote=args.remote,
                source_ref=args.source_ref,
                check_only=args.check_only,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            payload = {
                "schema": SCHEMA,
                "issue": args.issue,
                "ok": False,
                "outcome": "error",
                "write_attempted": False,
                "error": str(exc),
            }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("ok") is True:
        return 0
    if payload.get("outcome") == "not_admitted":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
