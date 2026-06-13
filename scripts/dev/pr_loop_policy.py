#!/usr/bin/env python3
"""Machine-checkable state policy for autonomous PR loops.

Classifies PR state from compact snapshots and recommends one bounded action:
stop, continue, reroute, escalate, wait_ci, inspect_failed_ci, verify_artifacts,
refresh_snapshot, mark_ready_candidate, or no_action.

Every PolicyDecision also emits a high-level ``flow_decision`` — one of exactly
``stop``, ``continue``, ``reroute``, or ``escalate`` — for machine consumption.

Review state (e.g. CHANGES_REQUESTED, APPROVED, COMMENTED) from the snapshot
is incorporated: CHANGES_REQUESTED forces a non-continue flow decision.

Accepts a compact PR queue snapshot JSON (as emitted by snapshot_pr_queue.py)
or a single-PR mock, and emits JSON or concise text with the next action and
stop reason. Dry-run mode never calls gh or mutates state.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_MAX_ACTIONS = 5

VALID_ACTIONS = frozenset(
    {
        "stop",
        "continue",
        "reroute",
        "escalate",
        "wait_ci",
        "inspect_failed_ci",
        "verify_artifacts",
        "refresh_snapshot",
        "mark_ready_candidate",
        "no_action",
    }
)

VALID_FLOW_DECISIONS = frozenset({"stop", "continue", "reroute", "escalate"})

VALID_STATES = frozenset(
    {
        "pending_ci",
        "failed_ci",
        "missing_artifacts",
        "stale_worktree",
        "ready_to_merge",
        "no_action",
    }
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A deterministic policy recommendation for one PR."""

    pr: int
    action: str
    state: str
    flow_decision: str
    reason: str
    actions_remaining: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision as a plain dict."""
        return asdict(self)


def classify_pr_state(pr: dict[str, Any]) -> str:
    """Classify a single PR into a machine-checkable loop state.

    Pure function: no side effects, no GitHub calls.
    """
    if not isinstance(pr, dict):
        return "no_action"
    status = str(pr.get("status", ""))
    if status == "error":
        return "no_action"
    checks = pr.get("checks") or {}
    overall = str(checks.get("overall", ""))
    labels = pr.get("labels") or []
    if isinstance(labels, list):
        label_names = [str(label) for label in labels]
    else:
        label_names = []
    is_draft = bool(pr.get("draft", False))
    head_sha = str(pr.get("head_sha", ""))
    expected = str(pr.get("expected_head_sha", ""))
    artifacts = pr.get("artifacts")
    if is_draft:
        return "no_action"
    if overall == "failure":
        return "failed_ci"
    if overall == "pending":
        return "pending_ci"
    if expected and head_sha and head_sha != expected:
        return "stale_worktree"
    if artifacts is not None and not artifacts:
        return "missing_artifacts"
    if "merge-ready" in label_names and overall == "success":
        return "ready_to_merge"
    return "no_action"


def _review_state(pr: dict[str, Any]) -> str:
    """Extract review state string from a PR dict.

    Supports two formats:
      - scalar: ``review_state`` or ``review`` field with a single string value.
      - dict: ``reviews`` field mapping review conclusions to counts, e.g.
        ``{"CHANGES_REQUESTED": 1, "APPROVED": 1}``.

    Returns the uppercased review state or empty string if absent.
    Priority: CHANGES_REQUESTED > CHANGES_REQUESTED (via scalar) > any other dict
    key with count > 0 > scalar value > empty.
    """
    raw = pr.get("review_state") or pr.get("review") or ""
    reviews_dict = pr.get("reviews")
    if isinstance(reviews_dict, dict):
        if reviews_dict.get("CHANGES_REQUESTED", 0) > 0:
            return "CHANGES_REQUESTED"
        for key, count in reviews_dict.items():
            if isinstance(count, (int, float)) and count > 0:
                return str(key).upper()
    return str(raw).upper() if raw else ""


def _compute_flow_decision(
    state: str,
    *,
    review_state: str,
    budget_exhausted: bool,
) -> str:
    """Map classified state + review state to a high-level flow decision.

    Exactly one of: stop, continue, reroute, escalate.

    Deterministic rules:
      - budget_exhausted -> stop
      - CHANGES_REQUESTED -> escalate (review blocker overrides other routing)
      - pending_ci -> continue
      - ready_to_merge -> continue
      - failed_ci, missing_artifacts, stale_worktree -> reroute
      - no_action -> stop
    """
    if budget_exhausted:
        return "stop"
    if review_state == "CHANGES_REQUESTED":
        return "escalate"
    match state:
        case "pending_ci" | "ready_to_merge":
            return "continue"
        case "failed_ci" | "missing_artifacts" | "stale_worktree":
            return "reroute"
        case _:
            return "stop"


def recommend_action(
    state: str,
    *,
    pr_number: int,
    actions_remaining: int,
    has_merge_ready: bool = False,
    ci_success: bool = False,
    review_state: str = "",
) -> PolicyDecision:
    """Map a classified state to a deterministic next action.

    Pure function: no side effects.
    """
    budget_exhausted = actions_remaining <= 0
    flow_decision = _compute_flow_decision(
        state,
        review_state=review_state,
        budget_exhausted=budget_exhausted,
    )
    if budget_exhausted:
        return PolicyDecision(
            pr=pr_number,
            action="stop",
            state=state,
            flow_decision="stop",
            reason="loop budget exhausted",
            actions_remaining=0,
        )
    remaining = actions_remaining - 1
    if review_state == "CHANGES_REQUESTED":
        return PolicyDecision(
            pr=pr_number,
            action="escalate",
            state=state,
            flow_decision=flow_decision,
            reason="review changes requested; escalate before continuing loop automation",
            actions_remaining=remaining,
        )
    match state:
        case "pending_ci":
            return PolicyDecision(
                pr=pr_number,
                action="wait_ci",
                state=state,
                flow_decision=flow_decision,
                reason="CI checks still pending",
                actions_remaining=remaining,
            )
        case "failed_ci":
            return PolicyDecision(
                pr=pr_number,
                action="inspect_failed_ci",
                state=state,
                flow_decision=flow_decision,
                reason="CI checks failed; inspect failures before retry",
                actions_remaining=remaining,
            )
        case "missing_artifacts":
            return PolicyDecision(
                pr=pr_number,
                action="verify_artifacts",
                state=state,
                flow_decision=flow_decision,
                reason="required artifacts not present",
                actions_remaining=remaining,
            )
        case "stale_worktree":
            return PolicyDecision(
                pr=pr_number,
                action="refresh_snapshot",
                state=state,
                flow_decision=flow_decision,
                reason="PR head SHA does not match expected snapshot",
                actions_remaining=remaining,
            )
        case "ready_to_merge":
            return PolicyDecision(
                pr=pr_number,
                action="mark_ready_candidate",
                state=state,
                flow_decision=flow_decision,
                reason="CI green and merge-ready label present",
                actions_remaining=remaining,
            )
        case _:
            return PolicyDecision(
                pr=pr_number,
                action="no_action",
                state="no_action",
                flow_decision=flow_decision,
                reason="nothing actionable for this PR",
                actions_remaining=remaining,
            )


def _pr_number(pr: dict[str, Any]) -> int:
    """Extract PR number safely."""
    try:
        return int(pr.get("number", 0))
    except (TypeError, ValueError):
        return 0


def evaluate_queue(
    prs: list[dict[str, Any]],
    *,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    expected_head_shas: dict[int, str] | None = None,
    artifact_presence: dict[int, bool] | None = None,
) -> dict[str, Any]:
    """Evaluate a PR queue and emit per-PR decisions under a loop budget.

    Pure function: reads snapshot dicts, never calls external APIs.
    """
    decisions: list[dict[str, Any]] = []
    actions_used = 0
    expected_shas = expected_head_shas or {}
    artifacts = artifact_presence or {}
    for pr in prs:
        num = _pr_number(pr)
        enriched: dict[str, Any] = dict(pr)
        if num in expected_shas:
            enriched["expected_head_sha"] = expected_shas[num]
        if num in artifacts:
            enriched["artifacts"] = artifacts[num]
        state = classify_pr_state(enriched)
        review = _review_state(enriched)
        labels = enriched.get("labels") or []
        label_names = [str(label) for label in labels] if isinstance(labels, list) else []
        checks = enriched.get("checks") or {}
        has_merge = "merge-ready" in label_names
        ci_ok = str(checks.get("overall", "")) == "success"
        remaining = max_actions - actions_used
        if remaining <= 0:
            decisions.append(
                PolicyDecision(
                    pr=num,
                    action="stop",
                    state="no_action",
                    flow_decision="stop",
                    reason="loop budget exhausted",
                    actions_remaining=0,
                ).to_dict()
            )
            break
        decision = recommend_action(
            state,
            pr_number=num,
            actions_remaining=remaining,
            has_merge_ready=has_merge,
            ci_success=ci_ok,
            review_state=review,
        )
        decisions.append(decision.to_dict())
        actions_used += 1
    return {
        "schema": "pr_loop_policy.v1",
        "max_actions": max_actions,
        "actions_used": actions_used,
        "decisions": decisions,
    }


def format_text(result: dict[str, Any]) -> str:
    """Format a compact human-readable policy summary."""
    lines = [
        f"max_actions: {result['max_actions']}  actions_used: {result['actions_used']}",
    ]
    for d in result["decisions"]:
        lines.append(
            f"PR #{d['pr']}: {d['action']} (state={d['state']}, flow={d['flow_decision']}) "
            f"— {d['reason']} [remaining={d['actions_remaining']}]"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--snapshot",
        help="Path to a compact PR queue snapshot JSON file.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read snapshot JSON from stdin instead of a file.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=DEFAULT_MAX_ACTIONS,
        help=f"Loop budget: maximum actions before stop (default {DEFAULT_MAX_ACTIONS}).",
    )
    parser.add_argument(
        "--expected-sha",
        nargs="*",
        metavar="PR=SHA",
        help="Expected head SHAs as PR=SHA pairs for staleness detection.",
    )
    parser.add_argument(
        "--artifact-present",
        nargs="*",
        metavar="PR=true|false",
        help="Artifact presence as PR=true|false pairs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def _parse_pairs(pairs: list[str] | None, *, as_bool: bool = False) -> dict[int, Any]:
    """Parse PR=VALUE pairs into a dict."""
    result: dict[int, Any] = {}
    if not pairs:
        return result
    for pair in pairs:
        if "=" not in pair:
            continue
        key_str, _, value = pair.partition("=")
        try:
            key = int(key_str)
        except ValueError:
            continue
        if as_bool:
            result[key] = value.lower() in ("true", "1", "yes")
        else:
            result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    raw: dict[str, Any] | list[dict[str, Any]] | None = None
    if args.stdin:
        try:
            raw = json.load(sys.stdin)
        except json.JSONDecodeError as exc:
            print(f"invalid JSON on stdin: {exc}", file=sys.stderr)
            return 1
    elif args.snapshot:
        try:
            with open(args.snapshot) as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            print(f"file not found: {args.snapshot}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"invalid JSON in {args.snapshot}: {exc}", file=sys.stderr)
            return 1
    else:
        print("provide --snapshot or --stdin", file=sys.stderr)
        return 1
    if isinstance(raw, dict) and "prs" in raw:
        prs: list[dict[str, Any]] = raw["prs"]
    elif isinstance(raw, list):
        prs = raw
    else:
        print("snapshot must contain a 'prs' array or be a JSON array of PR dicts", file=sys.stderr)
        return 1
    expected_shas = _parse_pairs(args.expected_sha)
    artifact_presence = _parse_pairs(args.artifact_present, as_bool=True)
    result = evaluate_queue(
        prs,
        max_actions=args.max_actions,
        expected_head_shas=expected_shas,
        artifact_presence=artifact_presence,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
