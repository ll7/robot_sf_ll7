#!/usr/bin/env python3
"""Compute the repository-wide work-in-progress admission decision.

The PR queue snapshot and :mod:`pr_loop_policy` are the only PR-state authorities.  This
module adds the small missing coordination decision: how many implementation and
campaign/operations lanes are active, which rows are excluded (and why), and whether a
new lane may start.  It never creates a worktree, claim, PR, or scheduler job.

``wip_capacity.v1`` is intentionally usable with JSON fixtures so the failure path can be
verified without opening disposable GitHub PRs.  Live mode performs read-only GitHub and
remote-ref queries.  Unknown or truncated evidence never produces available capacity.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# File-path invocation (the documented preflight form) does not place the repository root on
# ``sys.path``. Add it before importing the existing ``scripts.dev`` authorities.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dev.pr_loop_policy import classify_pr_state  # noqa: E402

DEFAULT_REPO = "ll7/robot_sf_ll7"
SCHEMA_VERSION = "wip_capacity.v1"
POLICY_SCHEMA = "wip_capacity_policy.v1"
DEFAULT_POLICY_PATH = REPO_ROOT / "configs/workflow/wip_policy.json"
CLAIM_REF_RE = re.compile(r"^refs/heads/agent-claims/issue-(?P<issue>[1-9][0-9]*)$")
ISSUE_COVERAGE_RE = re.compile(
    r"(?i)\b(?:refs?|references?|close(?:s|d)?|fix(?:es|ed)?|"
    r"resolve(?:s|d)?|implement(?:s|ed)?)\s*:?\s*`?#(?P<issue>[1-9][0-9]*)\b`?"
)
ISSUE_TITLE_RE = re.compile(
    r"(?i)(?:\(#(?P<parent>[1-9][0-9]*)\)|\bissue\s*#?\s*(?P<issue>[1-9][0-9]*))"
)
SUPERSEDED_RE = re.compile(r"(?i)\bsuperseded(?:\s+by|$)")
COMMON_BASELINE_RE = re.compile(
    r"(?i)(?:common[- ]main|common[- ]baseline|current[- ]main|baseline dependency)"
)
VALID_MODES = frozenset({"policy", "enforce", "report-only"})
VALID_LANES = frozenset({"implementation", "campaign_operations"})
EXEMPTION_KINDS = frozenset(
    {"p0_red_main", "security_incident", "maintainer_override", "wip_reducing"}
)
COUNTED_DEFAULT = frozenset(
    {
        "active_writer",
        "pending_ci",
        "failed_ci",
        "failed_validation",
        "missing_artifacts",
        "stale_worktree",
        "unknown_review_threads",
        "pending_gate_verdict",
        "pending_pr_metadata",
        "ready_to_merge",
    }
)


def _utc_now(value: datetime | None = None) -> datetime:
    """Return a timezone-aware UTC instant for deterministic evaluation."""
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp, returning ``None`` for unknown evidence."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _utc_now(parsed)


def _load_json(path: Path) -> Any:
    """Load a JSON file without accepting a malformed partial payload."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load and validate the one canonical WIP policy contract."""
    policy_path = Path(path)
    raw = _load_json(policy_path)
    if not isinstance(raw, dict) or raw.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy must declare schema {POLICY_SCHEMA}")
    limits = raw.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("policy limits must be an object")
    normalized_limits: dict[str, int] = {}
    for lane in VALID_LANES:
        value = limits.get(lane)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"policy limit for {lane} must be a positive integer")
        normalized_limits[lane] = value
    default_mode = str(raw.get("default_mode", "")).strip().lower()
    if default_mode not in {"enforce", "report-only"}:
        raise ValueError("policy default_mode must be enforce or report-only")
    campaign_labels = raw.get("campaign_labels")
    if not isinstance(campaign_labels, list) or not all(
        isinstance(item, str) and item.strip() for item in campaign_labels
    ):
        raise ValueError("policy campaign_labels must be a non-empty string list")
    priority_labels = raw.get("implementation_priority_labels")
    if not isinstance(priority_labels, list) or not all(
        isinstance(item, str) and item.strip() for item in priority_labels
    ):
        raise ValueError("policy implementation_priority_labels must be a non-empty string list")
    counted_states = raw.get("counted_pr_states", sorted(COUNTED_DEFAULT))
    if not isinstance(counted_states, list) or not all(
        isinstance(item, str) for item in counted_states
    ):
        raise ValueError("policy counted_pr_states must be a string list")
    return {
        **raw,
        "default_mode": default_mode,
        "limits": normalized_limits,
        "campaign_labels": sorted(set(campaign_labels)),
        "implementation_priority_labels": sorted(set(priority_labels)),
        "count_unknown_priority": bool(raw.get("count_unknown_priority", True)),
        "counted_pr_states": sorted(set(counted_states)),
        "parked_pr_states": sorted(
            set(raw.get("parked_pr_states", ["author_decision", "blocked_preflight", "no_action"]))
        ),
    }


def _labels(row: dict[str, Any]) -> set[str]:
    """Normalize string or GitHub-shaped labels from a snapshot row."""
    raw = row.get("labels", [])
    if not isinstance(raw, list):
        return set()
    result: set[str] = set()
    for item in raw:
        if isinstance(item, str) and item:
            result.add(item)
        elif isinstance(item, dict) and item.get("name"):
            result.add(str(item["name"]))
    return result


def _author_login(row: dict[str, Any]) -> str:
    """Return a compact author login from either snapshot shape."""
    author = row.get("author")
    if isinstance(author, dict):
        return str(author.get("login", "") or author.get("name", ""))
    return str(row.get("author_login", author or ""))


def _text(row: dict[str, Any]) -> str:
    """Return title/body text used only for issue scope and supersession evidence."""
    return f"{row.get('title', '')}\n{row.get('body', '')}"


def issue_references(row: dict[str, Any]) -> list[int]:
    """Extract explicit issue references, preferring coverage verbs over incidental links."""
    projected = row.get("issue_references")
    if isinstance(projected, list) and all(
        isinstance(item, int) and item > 0 for item in projected
    ):
        return list(dict.fromkeys(projected))
    text = _text(row)
    covered = [int(match.group("issue")) for match in ISSUE_COVERAGE_RE.finditer(text)]
    if covered:
        return list(dict.fromkeys(covered))
    result: list[int] = []
    for match in ISSUE_TITLE_RE.finditer(str(row.get("title", ""))):
        value = match.group("parent") or match.group("issue")
        if value:
            result.append(int(value))
    return list(dict.fromkeys(result))


def scope_key(row: dict[str, Any]) -> str:
    """Return the stable one-issue ownership key or a PR/claim fallback."""
    references = issue_references(row)
    if len(references) == 1:
        return f"issue:{references[0]}"
    if len(references) > 1:
        return "issues:" + ",".join(str(item) for item in references)
    issue = row.get("issue")
    if isinstance(issue, int) and issue > 0:
        return f"issue:{issue}"
    try:
        number = int(row.get("number", 0))
    except (TypeError, ValueError):
        number = 0
    return f"pr:{number}" if number > 0 else "unknown"


def _is_superseded(row: dict[str, Any], labels: set[str]) -> bool:
    """Recognize explicit supersession without treating generic prose as terminal."""
    return "superseded" in labels or bool(SUPERSEDED_RE.search(str(row.get("title", ""))))


def _has_review_evidence(row: dict[str, Any]) -> bool:
    """Return whether the compact row contains a live review/reviewer signal."""
    reviews = row.get("reviews")
    if isinstance(reviews, dict):
        return any(isinstance(value, int) and value > 0 for value in reviews.values())
    if isinstance(reviews, list):
        return any(isinstance(item, dict) for item in reviews)
    snapshot = row.get("review_snapshot")
    return isinstance(snapshot, dict) and int(snapshot.get("total", 0) or 0) > 0


def _is_common_baseline_block(row: dict[str, Any], state: str) -> bool:
    """Identify a current-main/common-baseline blocker for an explicit exclusion reason."""
    if state == "stale_merge_base":
        return True
    preflight = row.get("preflight")
    if isinstance(preflight, dict):
        reasons = preflight.get("reasons", [])
        if isinstance(reasons, list) and any(
            COMMON_BASELINE_RE.search(str(item)) for item in reasons
        ):
            return True
    return bool(COMMON_BASELINE_RE.search(_text(row)))


def _priority_in_scope(labels: set[str], policy: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a non-campaign implementation row belongs to the P0/P1 policy."""
    configured = set(policy["implementation_priority_labels"])
    if labels & configured:
        return True, "priority_in_scope"
    if any(label.startswith("priority:") for label in labels):
        return False, "priority_outside_wip_policy"
    if policy["count_unknown_priority"]:
        return True, "priority_unlabelled_counted_fail_closed"
    return False, "priority_unknown_excluded_by_policy"


def _classify_pr(row: dict[str, Any], *, now: datetime) -> str:
    """Call the canonical PR classifier for one snapshot row."""
    return classify_pr_state(row, now=now)


def _pr_item(row: dict[str, Any], *, policy: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Classify one open PR into a counted or excluded diagnostic row."""
    labels = _labels(row)
    state = _classify_pr(row, now=now)
    number = row.get("number")
    item: dict[str, Any] = {
        "source": "pr",
        "number": number,
        "scope": scope_key(row),
        "title": str(row.get("title", "")),
        "labels": sorted(labels),
        "state": state,
        "category": "implementation",
        "counted": False,
    }
    if bool(row.get("draft", row.get("isDraft", False))):
        item.update({"category": "deferred_or_parked", "reason": "draft_pr"})
        return item
    if _is_superseded(row, labels):
        item.update({"category": "superseded", "reason": "explicit_superseded_state"})
        return item
    if state in {"author_decision", "blocked_preflight", "stale_merge_base", "no_action"}:
        reason = {
            "author_decision": "author_decision_parked",
            "blocked_preflight": "blocked_preflight",
            "stale_merge_base": "common_current_main_baseline_blocked",
            "no_action": "deferred_or_parked_no_action",
        }[state]
        if state == "stale_merge_base" and not _is_common_baseline_block(row, state):
            reason = "stale_merge_base_blocked"
        item.update(
            {
                "category": "blocked"
                if state in {"blocked_preflight", "stale_merge_base"}
                else "deferred_or_parked",
                "reason": reason,
            }
        )
        return item
    if state not in set(policy["counted_pr_states"]):
        item.update({"category": "deferred_or_parked", "reason": f"state_not_counted:{state}"})
        return item
    if "security" in labels:
        item["security"] = True
    if "dependabot" in _author_login(row).lower() or "dependencies" in labels:
        item.update({"category": "deferred_or_parked", "reason": "dependency_automation_lane"})
        return item
    if "campaign" in labels or labels & set(policy["campaign_labels"]):
        item.update(
            {
                "category": "campaign_operations",
                "counted": True,
                "reason": "campaign_or_operations_label",
            }
        )
        return item
    in_scope, priority_reason = _priority_in_scope(labels, policy)
    if not in_scope:
        item.update({"category": "deferred_or_parked", "reason": priority_reason})
        return item
    item.update(
        {
            "category": "review" if _has_review_evidence(row) else "implementation",
            "lane": "implementation",
            "counted": True,
            "reason": "review_in_progress" if _has_review_evidence(row) else priority_reason,
        }
    )
    return item


def _claim_item(claim: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    """Classify one claim ref, retaining explicit stale/expiry evidence."""
    issue = claim.get("issue")
    try:
        issue_number = int(issue)
    except (TypeError, ValueError):
        issue_number = 0
    item: dict[str, Any] = {
        "source": "claim",
        "number": issue_number or None,
        "scope": f"issue:{issue_number}" if issue_number > 0 else "unknown",
        "state": "active_writer",
        "category": "implementation",
        "counted": False,
        "claim_ref": claim.get("claim_ref", ""),
    }
    state = str(claim.get("state", claim.get("status", "active"))).lower()
    expired_at = _parse_timestamp(claim.get("expires_at"))
    stale = bool(claim.get("stale") or claim.get("expired")) or state in {"stale", "expired"}
    if expired_at is not None and expired_at <= now:
        stale = True
    if stale:
        item.update({"category": "stale_or_expired_claim", "reason": "stale_or_expired_claim"})
        return item
    if issue_number <= 0:
        item.update({"category": "blocked", "reason": "claim_issue_identity_unknown"})
        return item
    item.update({"counted": True, "reason": "active_issue_claim"})
    return item


def _snapshot_evidence(snapshot: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the queue snapshot and return rows plus fail-closed errors."""
    if not isinstance(snapshot, dict):
        return [], ["queue_snapshot_unavailable"]
    prs = snapshot.get("prs")
    errors: list[str] = []
    if snapshot.get("truncated") is True:
        errors.append("queue_snapshot_truncated")
    if not isinstance(prs, list):
        errors.append("queue_snapshot_prs_missing")
        return [], errors
    if any(not isinstance(row, dict) for row in prs):
        errors.append("queue_snapshot_prs_malformed")
    rows = [row for row in prs if isinstance(row, dict)]
    for row in rows:
        if row.get("status") == "error":
            errors.append(f"queue_snapshot_row_error:{row.get('number', 'unknown')}")
    return rows, list(dict.fromkeys(errors))


def _claims_evidence(claims: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate a claim snapshot; an absent list is unknown, not empty."""
    if not isinstance(claims, list):
        return [], ["claim_snapshot_unavailable"]
    errors = (
        ["claim_snapshot_row_malformed"] if any(not isinstance(row, dict) for row in claims) else []
    )
    return [row for row in claims if isinstance(row, dict)], errors


def _proposed_lane(proposed: dict[str, Any], policy: dict[str, Any]) -> str:
    """Derive a proposed lane from explicit lane or campaign labels."""
    explicit = str(proposed.get("lane", "")).strip()
    if explicit in VALID_LANES:
        return explicit
    labels = _labels(proposed)
    return (
        "campaign_operations"
        if labels & set(policy["campaign_labels"]) or "campaign" in labels
        else "implementation"
    )


def _audit_fields_present(exemption: dict[str, Any]) -> bool:
    """Require the audit envelope shared by every exemption kind."""
    required = ("actor", "reason", "scope", "issued_at", "expires_at")
    return all(isinstance(exemption.get(key), str) and exemption[key].strip() for key in required)


def validate_exemption(  # noqa: C901 - each exemption has an independent fail-closed predicate.
    exemption: dict[str, Any] | None,
    *,
    proposed: dict[str, Any] | None,
    now: datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one narrow, time-bounded exemption and return its audit record."""
    if exemption is None:
        return None, None
    if not isinstance(exemption, dict):
        return None, "exemption_malformed"
    kind = str(exemption.get("kind", "")).strip()
    if kind not in EXEMPTION_KINDS:
        return None, "exemption_kind_unknown"
    if not _audit_fields_present(exemption):
        return None, "exemption_audit_fields_missing"
    issued_at = _parse_timestamp(exemption.get("issued_at"))
    expires_at = _parse_timestamp(exemption.get("expires_at"))
    if issued_at is None or expires_at is None or issued_at > now or expires_at <= now:
        return None, "exemption_expired_or_timestamp_invalid"
    if proposed is None:
        return None, "exemption_requires_proposed_lane"
    proposed_scope = scope_key(proposed)
    scope = str(exemption["scope"])
    if scope not in {"repository", proposed_scope, str(proposed.get("issue", ""))}:
        return None, "exemption_scope_mismatch"
    labels = _labels(proposed)
    if kind == "p0_red_main" and not (
        "priority:0" in labels and ("red-main" in labels or bool(proposed.get("red_main")))
    ):
        return None, "p0_red_main_exemption_requires_red_main_priority_zero"
    if kind == "security_incident" and "security" not in labels:
        return None, "security_exemption_requires_security_label"
    if kind == "wip_reducing" and not bool(proposed.get("wip_reducing")):
        return None, "wip_reducing_exemption_requires_reducing_action"
    return {
        "kind": kind,
        "actor": str(exemption["actor"]),
        "reason": str(exemption["reason"]),
        "scope": scope,
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }, None


def _deduplicate_lanes(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Collapse one issue's PR/claim rows to one lane and expose coordination defects."""
    counted = [item for item in items if item.get("counted")]
    excluded = [item for item in items if not item.get("counted")]
    blockers: list[dict[str, Any]] = []
    occupied_scopes = {
        str(item.get("scope"))
        for item in items
        if item.get("scope") not in {None, "unknown"}
        and item.get("reason") not in {"explicit_superseded_state", "stale_or_expired_claim"}
    }
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for item in counted:
        by_scope.setdefault(str(item.get("scope")), []).append(item)
    kept: list[dict[str, Any]] = []
    for scope in sorted(by_scope):
        rows = sorted(
            by_scope[scope],
            key=lambda item: (0 if item.get("source") == "pr" else 1, str(item.get("number", ""))),
        )
        primary = rows[0]
        kept.append(primary)
        pr_rows = [row for row in rows if row.get("source") == "pr"]
        if len(pr_rows) > 1:
            blockers.append(
                {
                    "reason": "competing_pr_same_issue",
                    "scope": scope,
                    "items": [row.get("number") for row in pr_rows],
                }
            )
        for duplicate in rows[1:]:
            excluded.append(
                {
                    **duplicate,
                    "counted": False,
                    "category": "duplicate_coordination_defect",
                    "reason": "competing_pr_same_issue"
                    if duplicate.get("source") == "pr"
                    else "claim_covered_by_existing_lane",
                }
            )
    return kept, excluded, blockers, occupied_scopes


def evaluate_capacity(  # noqa: C901, PLR0912, PLR0915 - one auditable admission decision owns all predicates.
    snapshot: dict[str, Any],
    claims: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    proposed: dict[str, Any] | None = None,
    mode: str | None = None,
    now: datetime | None = None,
    evidence_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate WIP capacity from supplied evidence without external side effects."""
    evaluated_at = _utc_now(now)
    effective_mode = str(mode or policy.get("default_mode", "enforce")).lower()
    if effective_mode == "policy":
        effective_mode = str(policy.get("default_mode", "enforce")).lower()
    if effective_mode not in {"enforce", "report-only"}:
        effective_mode = "enforce"
    pr_rows, snapshot_errors = _snapshot_evidence(snapshot)
    claim_rows, claim_errors = _claims_evidence(claims)
    all_evidence_errors = list(
        dict.fromkeys((evidence_errors or []) + snapshot_errors + claim_errors)
    )
    items = [_pr_item(row, policy=policy, now=evaluated_at) for row in pr_rows]
    items.extend(_claim_item(row, now=evaluated_at) for row in claim_rows)
    counted, excluded, coordination_blockers, occupied_scopes = _deduplicate_lanes(items)
    counts = dict.fromkeys(VALID_LANES, 0)
    for item in counted:
        lane = str(item.get("category", "implementation"))
        if lane not in counts:
            lane = "implementation"
        counts[lane] += 1
    limits = {lane: int(policy["limits"][lane]) for lane in VALID_LANES}
    remaining = {lane: max(limits[lane] - counts[lane], 0) for lane in VALID_LANES}
    blockers: list[dict[str, Any]] = [
        {"reason": "capacity_evidence_unavailable", "detail": error}
        for error in all_evidence_errors
    ]
    blockers.extend(coordination_blockers)
    valid_exemption, exemption_error = validate_exemption(
        None if proposed is None else proposed.get("exemption"),
        proposed=proposed,
        now=evaluated_at,
    )
    if exemption_error:
        blockers.append({"reason": exemption_error})
    proposed_scope = scope_key(proposed) if proposed is not None else None
    active_claim_scopes = {
        str(item.get("scope")) for item in counted if item.get("source") == "claim"
    }
    active_pr_scopes = {
        str(item.get("scope"))
        for item in items
        if item.get("source") == "pr"
        and item.get("reason") not in {"explicit_superseded_state", "stale_or_expired_claim"}
        and item.get("scope") not in {None, "unknown"}
    }
    continuation_of_owned_claim = bool(
        proposed is not None
        and proposed_scope in active_claim_scopes
        and proposed_scope not in active_pr_scopes
    )
    if proposed is not None and proposed_scope in active_pr_scopes:
        blockers.append(
            {
                "reason": "issue_already_has_owner_or_pr",
                "scope": proposed_scope,
            }
        )
    elif (
        proposed is not None
        and proposed_scope in occupied_scopes
        and not continuation_of_owned_claim
    ):
        blockers.append(
            {
                "reason": "issue_already_has_owner_or_pr",
                "scope": proposed_scope,
            }
        )
    proposed_lane = _proposed_lane(proposed, policy) if proposed is not None else None
    if proposed is not None and proposed_lane not in VALID_LANES:
        blockers.append({"reason": "proposed_lane_unknown"})
    if (
        proposed is not None
        and not continuation_of_owned_claim
        and proposed_lane in VALID_LANES
        and counts[proposed_lane] >= limits[proposed_lane]
    ):
        blockers.append(
            {
                "reason": "wip_limit_full",
                "lane": proposed_lane,
                "count": counts[proposed_lane],
                "limit": limits[proposed_lane],
            }
        )
    if valid_exemption is not None:
        blockers = [
            blocker
            for blocker in blockers
            if blocker.get("reason") not in {"wip_limit_full", "issue_already_has_owner_or_pr"}
        ]
    hard_block = bool(blockers)
    if proposed is None:
        if hard_block and effective_mode == "enforce":
            decision = "block"
            next_action = "repair_capacity_evidence_or_coordination_defect"
        elif hard_block:
            decision = "report_only"
            next_action = "observe_blockers_before_enabling_enforcement"
        else:
            decision = "report_only" if effective_mode == "report-only" else "snapshot"
            next_action = "no_new_lane_requested"
    elif hard_block and effective_mode == "report-only":
        decision = "report_only"
        next_action = "observe_blockers_before_enabling_enforcement"
    elif hard_block:
        decision = "block"
        if any(item.get("reason") == "capacity_evidence_unavailable" for item in blockers):
            next_action = "refresh_complete_queue_and_claim_snapshot"
        elif any(item.get("reason") == "competing_pr_same_issue" for item in blockers):
            next_action = "resolve_competing_issue_owner"
        elif any(item.get("reason") == "issue_already_has_owner_or_pr" for item in blockers):
            next_action = "reuse_existing_issue_lane_or_release_terminal_claim"
        else:
            next_action = "wait_for_wip_reduction_or_audited_exemption"
    else:
        decision = "allow"
        next_action = "start_or_review_the_proposed_lane"
    available = not all_evidence_errors
    return {
        "schema": SCHEMA_VERSION,
        "repo": snapshot.get("repo", DEFAULT_REPO) if isinstance(snapshot, dict) else DEFAULT_REPO,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "mode": effective_mode,
        "decision": decision,
        "allowed": decision == "allow",
        "available_capacity_proven": available,
        "raw_open_pr_count_diagnostic": len(pr_rows),
        "policy": {
            "schema": policy.get("schema", POLICY_SCHEMA),
            "limits": limits,
            "counted_pr_states": policy.get("counted_pr_states", sorted(COUNTED_DEFAULT)),
            "campaign_labels": policy.get("campaign_labels", []),
            "implementation_priority_labels": policy.get("implementation_priority_labels", []),
        },
        "counts": counts,
        "remaining": remaining,
        "limits": limits,
        "counted_lanes": sorted(
            counted, key=lambda item: (str(item.get("category")), str(item.get("scope")))
        ),
        "excluded_items": sorted(
            excluded,
            key=lambda item: (
                str(item.get("category")),
                str(item.get("scope")),
                str(item.get("number")),
            ),
        ),
        "blockers": blockers,
        "coordination_blockers": coordination_blockers,
        "proposed": proposed,
        "proposed_lane": proposed_lane,
        "continuation_of_owned_claim": continuation_of_owned_claim,
        "exemption": valid_exemption,
        "next_action": next_action,
        "evidence": {
            "queue_snapshot": "complete" if not snapshot_errors else "unavailable",
            "claim_snapshot": "complete" if not claim_errors else "unavailable",
            "errors": all_evidence_errors,
        },
    }


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one read-only helper command without a shell."""
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _read_snapshot(*, repo: str, path: str | None, limit: int) -> tuple[dict[str, Any], list[str]]:
    """Read a fixture or obtain the canonical active PR snapshot."""
    if path:
        try:
            payload = _load_json(Path(path))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, [f"queue_snapshot_file_unavailable:{exc}"]
        return payload if isinstance(payload, dict) else {}, []
    try:
        from scripts.dev.snapshot_pr_queue import snapshot_active_prs

        return snapshot_active_prs(repo=repo, limit=limit), []
    except Exception as exc:  # noqa: BLE001 - live evidence must fail closed, not crash
        return {}, [f"queue_snapshot_live_query_failed:{exc}"]


def _read_claims(*, remote: str, path: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a fixture or the canonical read-only issue-claim ref snapshot."""
    if path:
        try:
            payload = _load_json(Path(path))
        except (OSError, json.JSONDecodeError) as exc:
            return [], [f"claim_snapshot_file_unavailable:{exc}"]
        if isinstance(payload, dict):
            payload = payload.get("claims")
        return payload if isinstance(payload, list) else [], [] if isinstance(payload, list) else [
            "claim_snapshot_file_malformed"
        ]
    try:
        from scripts.dev.issue_claim import build_claim_snapshot_command

        result = _run(build_claim_snapshot_command(remote=remote))
    except Exception as exc:  # noqa: BLE001 - evidence must fail closed
        return [], [f"claim_snapshot_live_query_failed:{exc}"]
    if result.returncode != 0:
        return [], ["claim_snapshot_live_query_failed"]
    claims: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        match = CLAIM_REF_RE.match(parts[1])
        if match:
            claims.append(
                {
                    "issue": int(match.group("issue")),
                    "claim_ref": parts[1].removeprefix("refs/heads/"),
                    "sha": parts[0],
                    "state": "active",
                }
            )
    return claims, []


def _read_proposed_issue(
    repo: str, issue: int, labels: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Read issue labels for lane/exemption classification without mutating GitHub."""
    command = ["gh", "api", f"repos/{repo}/issues/{issue}"]
    result = _run(command)
    if result.returncode != 0:
        if labels:
            return {"issue": issue, "labels": labels}, ["proposed_issue_metadata_unavailable"]
        return {"issue": issue, "labels": labels}, ["proposed_issue_metadata_unavailable"]
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"issue": issue, "labels": labels}, ["proposed_issue_metadata_malformed"]
    if not isinstance(payload, dict):
        return {"issue": issue, "labels": labels}, ["proposed_issue_metadata_malformed"]
    issue_labels = [
        str(item.get("name"))
        for item in payload.get("labels", [])
        if isinstance(item, dict) and item.get("name")
    ]
    return {
        "issue": issue,
        "title": payload.get("title", ""),
        "body": payload.get("body", ""),
        "labels": sorted(set(issue_labels + labels)),
    }, []


def _load_exemption(path: str | None) -> dict[str, Any] | None:
    """Load an optional audited exemption file."""
    if not path:
        return None
    payload = _load_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("exemption file must contain an object")
    return payload


def _format_text(payload: dict[str, Any]) -> str:
    """Format a compact human-readable admission result."""
    counts = payload.get("counts", {})
    lines = [
        f"decision={payload.get('decision')} mode={payload.get('mode')} "
        f"implementation={counts.get('implementation', 0)}/{payload.get('limits', {}).get('implementation', '?')} "
        f"campaign_operations={counts.get('campaign_operations', 0)}/{payload.get('limits', {}).get('campaign_operations', '?')}",
        f"next_action={payload.get('next_action')}",
    ]
    for blocker in payload.get("blockers", []):
        lines.append(f"blocker={blocker.get('reason', blocker)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the capacity CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="policy")
    parser.add_argument("--snapshot-file", help="Use a deterministic pr_queue_snapshot.v2 fixture.")
    parser.add_argument("--claims-file", help="Use a deterministic issue-claim fixture.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--proposed-issue", type=int)
    parser.add_argument("--proposed-lane", choices=sorted(VALID_LANES))
    parser.add_argument("--proposed-label", action="append", default=[])
    parser.add_argument("--exemption-file")
    parser.add_argument(
        "--json", action="store_true", help="Emit the complete machine-readable payload."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate capacity and return 2 for an enforced admission block."""
    args = build_parser().parse_args(argv)
    try:
        policy = load_policy(args.policy)
        snapshot, snapshot_errors = _read_snapshot(
            repo=args.repo, path=args.snapshot_file, limit=max(args.limit, 1)
        )
        claims, claim_errors = _read_claims(remote=args.remote, path=args.claims_file)
        proposed = None
        proposed_errors: list[str] = []
        if args.proposed_issue is not None:
            proposed, proposed_errors = _read_proposed_issue(
                args.repo, args.proposed_issue, list(args.proposed_label)
            )
            proposed["lane"] = args.proposed_lane
            if args.exemption_file:
                proposed["exemption"] = _load_exemption(args.exemption_file)
        payload = evaluate_capacity(
            snapshot,
            claims,
            policy,
            proposed=proposed,
            mode=args.mode,
            evidence_errors=snapshot_errors + claim_errors + proposed_errors,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema": SCHEMA_VERSION,
            "decision": "block",
            "allowed": False,
            "mode": "enforce" if args.mode != "report-only" else "report-only",
            "blockers": [{"reason": "capacity_evidence_unavailable", "detail": str(exc)}],
            "next_action": "repair_policy_or_refresh_capacity_evidence",
        }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_text(payload))
    if payload.get("decision") == "block" and payload.get("mode") == "enforce":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
