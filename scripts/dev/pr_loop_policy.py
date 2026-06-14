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
from pathlib import Path
from typing import Any

from scripts.dev.route_efficiency_report import (
    EXPECTED_ARTIFACT_KEYS,
    has_validation_success,
    is_complete_artifact_set,
)

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
        "failed_validation",
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


def _extract_manifest_compact_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return compact artifact records from a routed-worker manifest."""
    compact = manifest.get("compact_artifacts")
    if isinstance(compact, dict):
        return compact
    attempts = manifest.get("attempted_routes")
    if not isinstance(attempts, list):
        return {}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("run_dir") == manifest.get("chosen_run_dir"):
            compact = attempt.get("compact_artifacts")
            return compact if isinstance(compact, dict) else {}
    return {}


def _compact_artifacts_present(compact_artifacts: dict[str, Any] | None) -> bool | None:
    """Return compact artifact completeness, or None when no manifest was supplied."""
    if compact_artifacts is None:
        return None
    return is_complete_artifact_set(compact_artifacts)


def _validation_failed(compact_artifacts: dict[str, Any] | None) -> bool:
    """Return True when the validation artifact is present and not successful."""
    if compact_artifacts is None:
        return False
    return has_validation_success(compact_artifacts) is not True


def _artifact_state(
    artifacts: Any,
    *,
    compact_artifacts: dict[str, Any] | None,
) -> str | None:
    """Return a PR-loop state derived from artifact evidence, if any."""
    compact_present = _compact_artifacts_present(compact_artifacts)
    if compact_present is False:
        return "missing_artifacts"
    if _validation_failed(compact_artifacts):
        return "failed_validation"
    if artifacts is not None and not artifacts:
        return "missing_artifacts"
    return None


def _compact_artifacts_from_pr(
    pr: dict[str, Any],
    compact_artifacts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return explicit compact artifacts or compact artifacts embedded in a PR dict."""
    if compact_artifacts is not None:
        return compact_artifacts
    compact_raw = pr.get("compact_artifacts")
    return compact_raw if isinstance(compact_raw, dict) else None


def classify_pr_state(
    pr: dict[str, Any],
    *,
    compact_artifacts: dict[str, Any] | None = None,
) -> str:
    """Classify a single PR into a machine-checkable loop state.

    Pure function: no side effects, no GitHub calls.
    """
    if not isinstance(pr, dict):
        return "no_action"
    compact_artifacts = _compact_artifacts_from_pr(pr, compact_artifacts)
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
    artifact_state = _artifact_state(artifacts, compact_artifacts=compact_artifacts)
    if artifact_state is not None:
        return artifact_state
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
      - failed_ci, failed_validation, missing_artifacts, stale_worktree -> reroute
      - no_action -> stop
    """
    if budget_exhausted:
        return "stop"
    if review_state == "CHANGES_REQUESTED":
        return "escalate"
    match state:
        case "pending_ci" | "ready_to_merge":
            return "continue"
        case "failed_ci" | "failed_validation" | "missing_artifacts" | "stale_worktree":
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
        case "failed_validation":
            return PolicyDecision(
                pr=pr_number,
                action="verify_artifacts",
                state=state,
                flow_decision=flow_decision,
                reason="validation artifact present but reports failure",
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
    compact_artifacts: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a PR queue and emit per-PR decisions under a loop budget.

    Pure function: reads snapshot dicts, never calls external APIs.
    """
    decisions: list[dict[str, Any]] = []
    actions_used = 0
    expected_shas = expected_head_shas or {}
    artifacts = artifact_presence or {}
    compact_by_pr = compact_artifacts or {}
    for pr in prs:
        num = _pr_number(pr)
        enriched: dict[str, Any] = dict(pr)
        if num in expected_shas:
            enriched["expected_head_sha"] = expected_shas[num]
        if num in artifacts:
            enriched["artifacts"] = artifacts[num]
        compact = compact_by_pr.get(num)
        if compact is not None:
            enriched["compact_artifacts"] = compact
        state = classify_pr_state(enriched, compact_artifacts=compact)
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
    parser.add_argument(
        "--manifest",
        nargs="*",
        metavar="PR=PATH",
        help="Routed-worker manifest paths as PR=PATH pairs; overrides --artifact-present.",
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


def _resolve_manifest_path(path_str: str, *, target_repo: Path) -> Path:
    """Resolve a manifest path inside the target repository."""
    path = Path(path_str)
    unresolved = path if path.is_absolute() else target_repo / path
    if unresolved.is_symlink():
        raise ValueError("manifest path must not be a symlink")
    resolved = unresolved.resolve(strict=False)
    if not resolved.is_relative_to(target_repo.resolve()):
        raise ValueError("manifest path must resolve inside target repository")
    return resolved


def _read_compact_artifact_text(
    artifact_path: str,
    *,
    target_repo: Path,
    max_chars: int = 4000,
) -> str | None:
    """Read a compact artifact text file from inside the target repository."""
    try:
        resolved = _resolve_manifest_path(artifact_path, target_repo=target_repo)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    try:
        return resolved.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return None


def _hydrate_validation_result(
    compact: dict[str, Any],
    *,
    target_repo: Path,
) -> dict[str, Any]:
    """Populate validation.result from the compact validation artifact file when absent."""
    validation = compact.get("validation")
    if not isinstance(validation, dict) or isinstance(validation.get("result"), str):
        return compact
    path = validation.get("path")
    if not isinstance(path, str):
        return compact
    result = _read_compact_artifact_text(path, target_repo=target_repo)
    if result is None:
        return compact
    hydrated = dict(compact)
    hydrated["validation"] = {**validation, "result": result}
    return hydrated


def load_manifest_artifacts(
    manifest_pairs: list[str] | None,
    *,
    target_repo: str | Path = ".",
) -> tuple[dict[int, bool], dict[int, dict[str, Any]], list[str]]:
    """Load routed-worker manifest pairs into PR artifact policy inputs."""
    artifact_presence: dict[int, bool] = {}
    compact_artifacts: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    repo_root = Path(target_repo).resolve()
    if not manifest_pairs:
        return artifact_presence, compact_artifacts, warnings

    for pair in manifest_pairs:
        if "=" not in pair:
            warnings.append(f"ignored malformed manifest pair: {pair}")
            continue
        key_str, _, path_str = pair.partition("=")
        try:
            pr_number = int(key_str)
        except ValueError:
            warnings.append(f"ignored manifest pair with non-integer PR: {pair}")
            continue
        try:
            manifest_path = _resolve_manifest_path(path_str, target_repo=repo_root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"ignored manifest for PR {pr_number}: {exc}")
            continue
        if not isinstance(manifest, dict):
            warnings.append(f"ignored manifest for PR {pr_number}: JSON root is not an object")
            continue
        compact = _hydrate_validation_result(
            _extract_manifest_compact_artifacts(manifest),
            target_repo=repo_root,
        )
        compact_artifacts[pr_number] = compact
        artifact_presence[pr_number] = is_complete_artifact_set(compact)
        missing = sorted(
            key
            for key in EXPECTED_ARTIFACT_KEYS
            if not isinstance(compact.get(key), dict) or compact[key].get("present") is not True
        )
        if missing:
            warnings.append(f"manifest for PR {pr_number} missing artifacts: {', '.join(missing)}")
    return artifact_presence, compact_artifacts, warnings


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
    manifest_presence, compact_artifacts, manifest_warnings = load_manifest_artifacts(
        args.manifest,
        target_repo=Path.cwd(),
    )
    artifact_presence.update(manifest_presence)
    for warning in manifest_warnings:
        print(warning, file=sys.stderr)
    result = evaluate_queue(
        prs,
        max_actions=args.max_actions,
        expected_head_shas=expected_shas,
        artifact_presence=artifact_presence,
        compact_artifacts=compact_artifacts,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
