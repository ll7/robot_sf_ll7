#!/usr/bin/env python3
"""Gate an atomic issue claim on the live issue implementation contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.dev import issue_claim, issue_implementability

SCHEMA = "goal_issue_admission.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
DEFAULT_SOURCE_REF = "origin/main"


def _claim_outcome(claim: Any, *, admission_outcome: str) -> str:
    """Return the explicit atomic-claim outcome exposed to queue consumers."""
    if admission_outcome == "claim_acquired":
        return "acquired"
    if admission_outcome == "claim_failed":
        return "write_failed"
    if not isinstance(claim, dict):
        return "not_checked"
    if claim.get("ok") is not True:
        return "unavailable"
    if claim.get("claimed") is True:
        return "already_claimed"
    return "unclaimed"


def compact_admission(payload: dict[str, Any]) -> dict[str, Any]:
    """Project one admission result into the stable queue-snapshot contract.

    Queue snapshots are read-only route evidence.  They must expose the same
    preflight and claim verdicts as the live admission wrapper without copying
    the implementability rules into each snapshot producer.
    """
    preflight = payload.get("preflight")
    if not isinstance(preflight, dict):
        preflight = {}
    outcome = str(payload.get("outcome") or "error")
    claim = payload.get("claim")
    if not isinstance(claim, dict):
        claim = preflight.get("claim")
    if not isinstance(claim, dict):
        claim = None
    return {
        "schema": SCHEMA,
        "ok": payload.get("ok") is True,
        "outcome": outcome,
        "write_attempted": payload.get("write_attempted") is True,
        "source_ref": payload.get("source_ref", DEFAULT_SOURCE_REF),
        "classification": preflight.get("classification"),
        "admission_reason": preflight.get("admission_reason"),
        "reasons": list(preflight.get("reasons", [])),
        "execution_contract": preflight.get("execution_contract"),
        "ready": preflight.get("ready") is True,
        "write_allowed": preflight.get("write_allowed") is True,
        "claim": claim,
        "claim_outcome": _claim_outcome(claim, admission_outcome=outcome),
    }


def compact_preflight(
    preflight: dict[str, Any], *, source_ref: str = DEFAULT_SOURCE_REF
) -> dict[str, Any]:
    """Project a pure preflight into the same read-only admission shape.

    This is used when a snapshot already has the issue and claim payloads for
    an obviously non-ready row.  Snapshot callers route ready candidates
    through :func:`admit_issue` so future preflight extensions, including typed
    dependency packets, remain owned by the canonical live path.
    """
    ready = preflight.get("ready") is True
    payload = {
        "schema": SCHEMA,
        "ok": ready,
        "outcome": "ready_check_only" if ready else "not_admitted",
        "write_attempted": False,
        "source_ref": source_ref,
        "preflight": preflight,
        "claim": preflight.get("claim"),
    }
    return compact_admission(payload)


def _preflight_fingerprint(preflight: dict[str, Any]) -> dict[str, Any] | None:
    """Return the live issue inputs that must remain stable before a claim write."""
    issue = preflight.get("issue")
    contract = preflight.get("contract")
    if not isinstance(issue, dict) or not isinstance(contract, dict):
        return None
    body_sha256 = contract.get("body_sha256")
    state = issue.get("state")
    labels = issue.get("labels")
    assignees = issue.get("assignees")
    if (
        not isinstance(body_sha256, str)
        or not isinstance(state, str)
        or not isinstance(labels, list)
        or not isinstance(assignees, list)
    ):
        return None
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "state": state,
        "labels": list(labels),
        "assignees": list(assignees),
        "body_sha256": body_sha256,
    }


def admit_issue(
    issue_number: int,
    *,
    repo: str,
    remote: str,
    source_ref: str,
    check_only: bool,
    route_preflight: Mapping[str, Any] | None = None,
    prospective_ready: bool = False,
) -> dict[str, Any]:
    """Evaluate one live issue and create its atomic claim only after a pass."""
    preflight_kwargs: dict[str, Any] = {}
    if route_preflight is not None:
        preflight_kwargs["route_preflight"] = route_preflight
    if prospective_ready:
        preflight_kwargs["prospective_ready"] = True
    preflight = issue_implementability.live_issue_report(
        issue_number,
        repo=repo,
        remote=remote,
        **preflight_kwargs,
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
        "claim": preflight.get("claim"),
        "ok": False,
    }
    if preflight.get("ready") is not True:
        payload["outcome"] = "not_admitted"
        payload["claim_outcome"] = _claim_outcome(
            payload["claim"], admission_outcome=payload["outcome"]
        )
        return payload
    if check_only:
        payload["ok"] = True
        payload["outcome"] = "ready_check_only"
        payload["claim_outcome"] = _claim_outcome(
            payload["claim"], admission_outcome=payload["outcome"]
        )
        return payload

    revalidated = issue_implementability.live_issue_report(
        issue_number,
        repo=repo,
        remote=remote,
        **preflight_kwargs,
    )
    initial_fingerprint = _preflight_fingerprint(preflight)
    revalidated_fingerprint = _preflight_fingerprint(revalidated)
    inputs_match = (
        initial_fingerprint is not None
        and revalidated_fingerprint is not None
        and initial_fingerprint == revalidated_fingerprint
    )
    payload["initial_preflight"] = preflight
    payload["preflight"] = revalidated
    payload["claim"] = revalidated.get("claim")
    payload["revalidation"] = {
        "performed": True,
        "inputs_match": inputs_match,
    }
    if not inputs_match or revalidated.get("ready") is not True:
        payload["outcome"] = "not_admitted"
        payload["claim_outcome"] = _claim_outcome(
            payload["claim"], admission_outcome=payload["outcome"]
        )
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
    payload["claim_outcome"] = _claim_outcome(claim, admission_outcome=payload["outcome"])
    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue", type=int, help="Positive GitHub issue number.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote used by issue claims.")
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
    parser.add_argument(
        "--route-preflight-json",
        type=Path,
        help="Optional fresh route-plan JSON for an explicitly multi-repository issue.",
    )
    return parser


def _load_route_preflight(path: Path | None) -> Mapping[str, Any] | None:
    """Load one route-plan object without exposing provider or credential data."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("route preflight JSON must be an object")
    return payload


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
                route_preflight=_load_route_preflight(args.route_preflight_json),
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
