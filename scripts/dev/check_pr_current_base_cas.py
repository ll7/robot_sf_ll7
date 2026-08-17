#!/usr/bin/env python3
"""Fail-closed immediate current-main/head compare-and-swap preflight.

The risk-tiered stale-base policy permits an ordinary PR whose exact head was reviewed against an
older main to proceed only after this check observes the same expected head and current main SHA
immediately before the guarded merge. Base-sensitive PRs additionally require a fresh base and the
``base_sensitive`` test subset; use ``--require-fresh-base`` for that path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import Any

CAS_SCHEMA = "pr_current_base_cas.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"


@dataclass(frozen=True)
class CurrentBaseCASSnapshot:
    """Live pull-request and current-main state used by the pure evaluator."""

    observed_head_sha: str
    observed_main_sha: str
    base_sha: str
    base_ref: str
    state: str
    is_draft: bool


def _gh(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run one bounded GitHub CLI request."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_json(stdout: str) -> dict[str, Any] | None:
    """Parse an object response, returning ``None`` for malformed output."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _state_reasons(snapshot: CurrentBaseCASSnapshot) -> list[str]:
    reasons: list[str] = []
    if snapshot.state.upper() != "OPEN":
        reasons.append("pull_request_not_open")
    if snapshot.is_draft:
        reasons.append("pull_request_is_draft")
    if snapshot.base_ref != "main":
        reasons.append("pull_request_base_is_not_main")
    return reasons


def _sha_reasons(
    snapshot: CurrentBaseCASSnapshot,
    *,
    expected_head_sha: str,
    expected_main_sha: str,
) -> list[str]:
    reasons: list[str] = []
    if not expected_head_sha:
        reasons.append("expected_head_sha_missing")
    elif not snapshot.observed_head_sha:
        reasons.append("observed_head_sha_missing")
    elif snapshot.observed_head_sha != expected_head_sha:
        reasons.append("head_sha_changed")
    if not expected_main_sha:
        reasons.append("expected_main_sha_missing")
    elif not snapshot.observed_main_sha:
        reasons.append("observed_main_sha_missing")
    elif snapshot.observed_main_sha != expected_main_sha:
        reasons.append("main_sha_changed_during_preflight")
    return reasons


def _base_reasons(
    snapshot: CurrentBaseCASSnapshot,
    *,
    require_fresh_base: bool,
) -> tuple[list[str], str]:
    reasons: list[str] = []
    if not snapshot.base_sha:
        reasons.append("base_sha_missing")

    base_relation = "unknown"
    if snapshot.base_sha and snapshot.observed_main_sha:
        base_relation = (
            "fresh" if snapshot.base_sha == snapshot.observed_main_sha else "stale_allowed"
        )
        if require_fresh_base and base_relation != "fresh":
            reasons.append("base_sensitive_pr_base_is_stale")
    return reasons, base_relation


def evaluate_current_base_cas(
    snapshot: CurrentBaseCASSnapshot,
    *,
    expected_head_sha: str,
    expected_main_sha: str,
    require_fresh_base: bool = False,
) -> dict[str, Any]:
    """Evaluate an already-collected head/main snapshot without side effects."""
    base_reasons, base_relation = _base_reasons(
        snapshot,
        require_fresh_base=require_fresh_base,
    )
    reasons = (
        _state_reasons(snapshot)
        + _sha_reasons(
            snapshot,
            expected_head_sha=expected_head_sha,
            expected_main_sha=expected_main_sha,
        )
        + base_reasons
    )

    passed = not reasons
    return {
        "schema": CAS_SCHEMA,
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "reasons": reasons,
        "require_fresh_base": require_fresh_base,
        "base_relation": base_relation,
        "base_ref": snapshot.base_ref,
        "base_sha": snapshot.base_sha or None,
        "expected_head_sha": expected_head_sha or None,
        "observed_head_sha": snapshot.observed_head_sha or None,
        "expected_main_sha": expected_main_sha or None,
        "observed_main_sha": snapshot.observed_main_sha or None,
    }


def _fetch_live_snapshot(pr_number: str, *, repo: str) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch the pull request and current main SHA through REST-backed gh calls."""
    pull_result = _gh(["api", f"repos/{repo}/pulls/{pr_number}"])
    if pull_result.returncode != 0:
        return None, pull_result.stderr.strip() or "pull request lookup failed"
    pull = _parse_json(pull_result.stdout)
    if pull is None:
        return None, "pull request response was not a JSON object"

    branch_result = _gh(["api", f"repos/{repo}/branches/main", "--jq", ".commit.sha"])
    if branch_result.returncode != 0:
        return None, branch_result.stderr.strip() or "current main lookup failed"
    main_sha = branch_result.stdout.strip()
    if not main_sha:
        return None, "current main lookup returned an empty SHA"

    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    base = pull.get("base") if isinstance(pull.get("base"), dict) else {}
    return {
        "state": str(pull.get("state", "")),
        "is_draft": bool(pull.get("draft")),
        "observed_head_sha": str(head.get("sha", "")),
        "base_sha": str(base.get("sha", "")),
        "base_ref": str(base.get("ref", "")),
        "observed_main_sha": main_sha,
    }, None


def check_current_base_cas(
    pr_number: str,
    *,
    repo: str,
    expected_head_sha: str,
    expected_main_sha: str,
    require_fresh_base: bool = False,
) -> dict[str, Any]:
    """Fetch live state and evaluate the immediate compare-and-swap contract."""
    snapshot, error = _fetch_live_snapshot(pr_number, repo=repo)
    if snapshot is None:
        return {
            "schema": CAS_SCHEMA,
            "status": "error",
            "passed": False,
            "reasons": ["live_snapshot_unavailable"],
            "error": error or "live snapshot unavailable",
            "pr": pr_number,
        }
    result = evaluate_current_base_cas(
        CurrentBaseCASSnapshot(
            observed_head_sha=str(snapshot["observed_head_sha"]),
            observed_main_sha=str(snapshot["observed_main_sha"]),
            base_sha=str(snapshot["base_sha"]),
            base_ref=str(snapshot["base_ref"]),
            state=str(snapshot["state"]),
            is_draft=bool(snapshot["is_draft"]),
        ),
        expected_head_sha=expected_head_sha,
        expected_main_sha=expected_main_sha,
        require_fresh_base=require_fresh_base,
    )
    result["pr"] = pr_number
    result["repo"] = repo
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", nargs="?", help="GitHub pull request number")
    parser.add_argument("--pr", dest="pr_option", help="GitHub pull request number")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--require-fresh-base", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the live compare-and-swap preflight."""
    args = _parse_args(argv)
    if args.pr_number and args.pr_option and args.pr_number != args.pr_option:
        raise SystemExit("conflicting PR numbers")
    pr_number = args.pr_option or args.pr_number
    if not pr_number:
        raise SystemExit("PR number is required")
    try:
        result = check_current_base_cas(
            pr_number,
            repo=args.repo,
            expected_head_sha=args.expected_head_sha,
            expected_main_sha=args.expected_main_sha,
            require_fresh_base=args.require_fresh_base,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result = {
            "schema": CAS_SCHEMA,
            "status": "error",
            "passed": False,
            "reasons": ["live_snapshot_unavailable"],
            "error": str(exc),
            "pr": pr_number,
        }
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    if result.get("status") == "error":
        return 2
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
