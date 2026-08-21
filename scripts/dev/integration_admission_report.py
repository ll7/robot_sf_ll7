#!/usr/bin/env python3
"""Classify one PR and a bounded queue snapshot for integration admission.

The report is an offline, deterministic routing aid.  It reads an existing
``pr_queue_snapshot.v2`` payload and never calls GitHub, writes Git refs, or
authorizes review, merge, or other external actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.dev.pr_loop_policy import (
    _base_freshness_provenance,
    _review_state,
    classify_pr_state,
)
from scripts.dev.snapshot_pr_queue import BLOCKING_LABELS, _base_freshness

SCHEMA_VERSION = "integration_admission_report.v1"
VALID_STATES = (
    "local_only",
    "preparation_pr",
    "integration_candidate",
    "integration_blocked",
    "review_active",
    "merge_candidate",
    "terminal",
    "unavailable",
    "invalid",
)

CHANGE_CLASSES = (
    "docs",
    "test_only",
    "tooling",
    "runtime",
    "benchmark",
    "evidence",
    "release",
    "mixed",
    "unknown",
    "unavailable",
)
SHARED_SURFACES = (
    "isolated",
    "component_shared",
    "repository_control_plane",
    "unknown",
    "unavailable",
)
CI_COSTS = ("low", "standard", "optional_matrix", "full", "unknown", "unavailable")
REVIEW_REQUIREMENTS = (
    "ordinary",
    "independent_exact_head",
    "domain",
    "author",
    "unknown",
    "unavailable",
)
EXTERNAL_ACTIONS = ("none", "network", "artifact", "compute", "release", "unknown", "unavailable")
BASE_SENSITIVITIES = ("ordinary", "current_base_required", "unknown", "unavailable")
OWNERSHIPS = ("local", "claimed", "maintainer", "external", "unassigned", "unknown", "unavailable")

_FRESH_AGE_SECONDS = 2 * 24 * 60 * 60
_AGING_AGE_SECONDS = 7 * 24 * 60 * 60
_CAPACITY_STATES = frozenset({"integration_candidate", "review_active", "merge_candidate"})


def _unique(values: list[str]) -> list[str]:
    """Return stable, lexicographically ordered non-empty strings."""
    return sorted({value for value in values if value})


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _labels(pr: dict[str, Any]) -> list[str]:
    raw = pr.get("labels", [])
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for label in raw:
        if isinstance(label, str):
            result.append(label)
        elif isinstance(label, dict) and isinstance(label.get("name"), str):
            result.append(label["name"])
    return sorted(set(result))


def _changed_paths(pr: dict[str, Any]) -> tuple[list[str], bool]:
    """Return changed paths and whether the producer supplied a path list."""
    for key in ("changed_files", "changed_paths", "files"):
        if key not in pr:
            continue
        raw = pr[key]
        if not isinstance(raw, list) or not all(isinstance(path, str) for path in raw):
            return [], True
        return sorted(set(raw)), True
    return [], False


def _path_class(path: str) -> str:
    normalized = path.removeprefix("./")
    if normalized.startswith(("docs/contracts/", "docs/evidence/")):
        return "evidence"
    if normalized.startswith(("docs/", "README", "CHANGELOG")):
        return "docs"
    if normalized.startswith(("tests/", "fast-pysf/tests/")):
        return "test_only"
    if normalized.startswith((".github/", "scripts/dev/", "scripts/validation/")):
        return "tooling"
    if normalized.startswith(("robot_sf/", "fast-pysf/")):
        return "runtime"
    if normalized.startswith(("configs/", "maps/", "model/", "scripts/benchmark/")):
        return "benchmark"
    if normalized.startswith(("release/", "packaging/")) or normalized in {
        "pyproject.toml",
        "uv.lock",
    }:
        return "release"
    return "mixed"


def _dimension(
    pr: dict[str, Any],
    key: str,
    allowed: tuple[str, ...],
    derived: str,
    invalid_codes: list[str],
) -> str:
    """Read an explicit dimension or use a conservative path-derived value."""
    if key not in pr:
        return derived
    value = pr[key]
    if not isinstance(value, str) or value not in allowed:
        invalid_codes.append(f"invalid_{key}")
        return "unavailable"
    return value


def _dimensions(pr: dict[str, Any], invalid_codes: list[str]) -> dict[str, str]:
    paths, paths_supplied = _changed_paths(pr)
    for key in ("changed_files", "changed_paths", "files"):
        if key in pr and (
            not isinstance(pr[key], list) or not all(isinstance(path, str) for path in pr[key])
        ):
            invalid_codes.append(f"invalid_{key}")
    classes = {_path_class(path) for path in paths}
    if not paths_supplied:
        change_class = "unavailable"
        shared_surface = "unavailable"
        path_derived_ci = "unavailable"
    elif not paths:
        change_class = "unknown"
        shared_surface = "unknown"
        path_derived_ci = "unknown"
    else:
        change_class = next(iter(classes)) if len(classes) == 1 else "mixed"
        shared_surface = (
            "component_shared"
            if {"runtime", "benchmark", "evidence", "mixed"} & classes
            else "repository_control_plane"
            if {"tooling", "release"} & classes
            else "isolated"
        )
        path_derived_ci = (
            "full"
            if {"runtime", "mixed"} & classes
            else "optional_matrix"
            if "benchmark" in classes
            else "standard"
            if {"tooling", "evidence", "release"} & classes
            else "low"
        )

    explicit_change = pr.get("change_class")
    if explicit_change is None and change_class in CHANGE_CLASSES:
        change = change_class
    else:
        change = _dimension(pr, "change_class", CHANGE_CLASSES, change_class, invalid_codes)
    return {
        "change_class": change,
        "shared_surface": _dimension(
            pr, "shared_surface", SHARED_SURFACES, shared_surface, invalid_codes
        ),
        "ci_cost": _dimension(pr, "ci_cost", CI_COSTS, path_derived_ci, invalid_codes),
        "review_requirement": _dimension(
            pr, "review_requirement", REVIEW_REQUIREMENTS, "ordinary", invalid_codes
        ),
        "external_action": _dimension(
            pr, "external_action", EXTERNAL_ACTIONS, "none", invalid_codes
        ),
        "base_sensitivity": _dimension(
            pr, "base_sensitivity", BASE_SENSITIVITIES, "ordinary", invalid_codes
        ),
        "ownership": _dimension(pr, "ownership", OWNERSHIPS, _derived_ownership(pr), invalid_codes),
    }


def _derived_ownership(pr: dict[str, Any]) -> str:
    explicit = pr.get("owner") or pr.get("assignee")
    if explicit:
        return "maintainer" if isinstance(explicit, str) and explicit.startswith("@") else "claimed"
    for entries in (pr.get("comments"), pr.get("reviews")):
        if isinstance(entries, list) and any(
            isinstance(entry, dict) and "review-claim:" in str(entry.get("body", ""))
            for entry in entries
        ):
            return "claimed"
    if pr.get("local_only") is True or str(pr.get("source", "")).lower() == "local":
        return "local"
    return "unassigned"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


def _age_freshness(
    pr: dict[str, Any], *, as_of: str | None, reason_codes: list[str]
) -> dict[str, Any]:
    opened_at = _as_text(pr.get("created_at") or pr.get("opened_at")) or None
    updated_at = _as_text(pr.get("updated_at") or pr.get("last_updated_at")) or None
    instant = _parse_time(as_of)
    anchor = _parse_time(updated_at or opened_at)
    if as_of is None:
        reason_codes.append("age_as_of_unavailable")
    elif instant is None:
        reason_codes.append("age_as_of_invalid")
    if anchor is None:
        reason_codes.append("age_timestamp_unavailable")
    elif (updated_at or opened_at) and _parse_time(updated_at or opened_at) is None:
        reason_codes.append("age_timestamp_invalid")
    age_seconds: int | None = None
    bucket = "unavailable"
    if instant is not None and anchor is not None:
        age_seconds = max(0, int((instant - anchor).total_seconds()))
        bucket = (
            "fresh"
            if age_seconds <= _FRESH_AGE_SECONDS
            else "aging"
            if age_seconds <= _AGING_AGE_SECONDS
            else "stale"
        )
    return {
        "as_of": as_of,
        "opened_at": opened_at,
        "updated_at": updated_at,
        "age_seconds": age_seconds,
        "freshness": bucket,
        "codes": _unique(reason_codes),
    }


def _baseline(pr: dict[str, Any], invalid_codes: list[str]) -> dict[str, Any]:
    raw = pr.get("base_freshness")
    if raw is not None and not isinstance(raw, dict):
        invalid_codes.append("invalid_base_freshness")
        raw = {}
    if isinstance(raw, dict) and raw:
        verdict = _as_text(raw.get("verdict"))
        base_sha = _as_text(raw.get("base_sha"))
        current_main_sha = _as_text(raw.get("current_main_sha"))
        source = "base_freshness"
        if not verdict:
            invalid_codes.append("base_freshness_verdict_unavailable")
    else:
        verdict, base_sha, current_main_sha = _base_freshness_provenance(pr)
        if not verdict:
            base_sha = _as_text(pr.get("base_sha"))
            current_main_sha = _as_text(pr.get("main_sha"))
            derived = _base_freshness(base_sha=base_sha, current_main_sha=current_main_sha)
            verdict = str(derived["verdict"])
        source = "legacy_base_fields" if base_sha or current_main_sha else "unavailable"
    if verdict not in {"fresh", "stale", "missing-base", "unavailable-current-main", ""}:
        invalid_codes.append("invalid_base_freshness_verdict")
    derived_verdict = _base_freshness(base_sha=base_sha, current_main_sha=current_main_sha)[
        "verdict"
    ]
    if verdict and verdict != derived_verdict:
        invalid_codes.append("inconsistent_base_freshness")
    return {
        "base_sha": base_sha or None,
        "current_main_sha": current_main_sha or None,
        "verdict": verdict or "unavailable",
        "available": bool(base_sha and current_main_sha),
        "source": source,
    }


def _identity(
    pr: dict[str, Any], baseline: dict[str, Any], reason_codes: list[str]
) -> dict[str, Any]:
    head_sha = _as_text(pr.get("head_sha"))
    expected = _as_text(pr.get("expected_head_sha"))
    if not head_sha:
        reason_codes.append("head_sha_unavailable")
    if not baseline["base_sha"]:
        reason_codes.append("base_sha_unavailable")
    if not baseline["current_main_sha"]:
        reason_codes.append("current_main_sha_unavailable")
    matches = None if not expected or not head_sha else head_sha == expected
    if matches is False:
        reason_codes.append("head_sha_mismatch")
    status = (
        "head_mismatch"
        if matches is False
        else "complete"
        if head_sha and baseline["base_sha"] and baseline["current_main_sha"]
        else "incomplete"
    )
    return {
        "pr_number": pr.get("number"),
        "base_sha": baseline["base_sha"],
        "head_sha": head_sha or None,
        "expected_head_sha": expected or None,
        "head_matches_expected": matches,
        "status": status,
    }


def _blockers(  # noqa: C901, PLR0912 - explicit fail-closed blocker dimensions
    pr: dict[str, Any],
    *,
    policy_state: str | None,
    baseline: dict[str, Any],
    identity: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    labels = _labels(pr)
    dimensions = _dimensions(pr, [])
    dependency: list[str] = []
    decision: list[str] = []
    review: list[str] = []
    external: list[str] = []
    for code in (
        pr.get("dependency_blockers", []) if isinstance(pr.get("dependency_blockers"), list) else []
    ):
        dependency.append(str(code))
    for code in (
        pr.get("decision_blockers", []) if isinstance(pr.get("decision_blockers"), list) else []
    ):
        decision.append(str(code))
    for code in (
        pr.get("external_blockers", []) if isinstance(pr.get("external_blockers"), list) else []
    ):
        external.append(str(code))
    for label in labels:
        if label in BLOCKING_LABELS:
            code = f"explicit_blocked:{label}"
            if label in {"decision-required", "state:hold"}:
                decision.append(code)
            elif label == "state:blocked-external-input":
                external.append(code)
            else:
                dependency.append(code)
    preflight = pr.get("preflight")
    if isinstance(preflight, dict):
        reasons = preflight.get("reasons", [])
        if isinstance(reasons, list):
            dependency.extend(str(reason) for reason in reasons)
        if str(preflight.get("status", "")).lower() == "blocked":
            dependency.append("preflight_blocked")
    checks = pr.get("checks") if isinstance(pr.get("checks"), dict) else {}
    if checks.get("overall") == "failure":
        dependency.append("ci_checks_failed")
    elif checks.get("overall") == "pending":
        dependency.append("ci_checks_pending")
    if baseline["verdict"] == "stale":
        dependency.append("base_sha_stale")
    elif baseline["verdict"] == "missing-base":
        dependency.append("base_sha_missing")
    elif baseline["verdict"] == "unavailable-current-main":
        dependency.append("current_main_sha_unavailable")
    if identity["head_matches_expected"] is False:
        dependency.append("head_sha_mismatch")
    review_state = _review_state(pr)
    if review_state == "CHANGES_REQUESTED":
        review.append("review_changes_requested")
    if policy_state == "unknown_review_threads":
        review.append("review_threads_unavailable")
    if policy_state == "active_writer":
        review.append("active_review_claim")
    if policy_state == "author_decision":
        decision.append("author_decision_required")
    if pr.get("reviewers") and not pr.get("review_approved"):
        review.append("requested_reviewers_outstanding")
    if dimensions["review_requirement"] == "domain" and not (
        pr.get("domain_approved") is True or "domain-approved" in labels
    ):
        review.append("domain_review_required")
    if dimensions["review_requirement"] == "author":
        decision.append("author_decision_required")
    if dimensions["external_action"] not in {"none", "unknown", "unavailable"}:
        external.append(f"external_action_required:{dimensions['external_action']}")
    result = {
        "dependency": {"blocked": bool(dependency), "codes": _unique(dependency)},
        "decision": {"blocked": bool(decision), "codes": _unique(decision)},
        "review": {"blocked": bool(review), "codes": _unique(review)},
        "external": {"blocked": bool(external), "codes": _unique(external)},
    }
    all_codes = _unique(dependency + decision + review + external)
    return result, all_codes


def _validate_pr(pr: Any) -> list[str]:
    if not isinstance(pr, dict):
        return ["pr_not_object"]
    invalid: list[str] = []
    if "number" in pr and (isinstance(pr["number"], bool) or not isinstance(pr["number"], int)):
        invalid.append("invalid_pr_number")
    if "labels" in pr and not isinstance(pr["labels"], list):
        invalid.append("invalid_labels")
    if "checks" in pr and not isinstance(pr["checks"], dict):
        invalid.append("invalid_checks")
    for key in ("head_sha", "expected_head_sha", "created_at", "updated_at"):
        if key in pr and pr[key] is not None and not isinstance(pr[key], str):
            invalid.append(f"invalid_{key}")
    return _unique(invalid)


def _select_pr(snapshot: Any, pr_number: int | None) -> tuple[Any, list[dict[str, Any]], list[str]]:
    if not isinstance(snapshot, dict):
        return None, [], ["snapshot_not_object"]
    rows = snapshot.get("prs")
    if not isinstance(rows, list):
        if pr_number is None and isinstance(snapshot.get("number"), int):
            return snapshot, [snapshot], []
        return None, [], ["snapshot_prs_unavailable"]
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if pr_number is None:
        if len(rows) != 1:
            return None, valid_rows, ["one_pr_required"]
        return rows[0], valid_rows, []
    matches = [row for row in rows if isinstance(row, dict) and row.get("number") == pr_number]
    if not matches:
        return None, valid_rows, ["pr_not_in_snapshot"]
    return matches[0], valid_rows, []


def _as_of(snapshot: dict[str, Any], requested: str | None) -> str | None:
    if requested is not None:
        return requested
    for key in ("captured_at", "generated_at", "as_of"):
        value = snapshot.get(key)
        if isinstance(value, str):
            return value
    return None


def _classify_pr(pr: Any, *, as_of: str | None) -> dict[str, Any]:  # noqa: C901, PLR0912
    invalid_codes = _validate_pr(pr)
    if invalid_codes:
        return {
            "state": "invalid",
            "reason_codes": invalid_codes,
            "invalidation_codes": invalid_codes,
        }
    assert isinstance(pr, dict)
    if pr.get("local_only") is True or str(pr.get("source", "")).lower() == "local":
        return {
            "state": "local_only",
            "reason_codes": ["local_source"],
            "invalidation_codes": [],
            "dimensions": _dimensions(pr, invalid_codes),
            "identity": {"status": "local", "pr_number": pr.get("number")},
            "current_main_baseline": {"verdict": "not_applicable", "available": False},
            "blockers": {
                key: {"blocked": False, "codes": []}
                for key in ("dependency", "decision", "review", "external")
            },
            "age_freshness": _age_freshness(pr, as_of=as_of, reason_codes=[]),
        }
    if str(pr.get("status", "")) == "error":
        return {
            "state": "unavailable",
            "reason_codes": ["snapshot_row_error"],
            "invalidation_codes": [],
        }
    lifecycle = str(pr.get("state", "")).upper()
    if lifecycle == "MERGED" or pr.get("merged_at"):
        return {
            "state": "terminal",
            "reason_codes": ["pr_merged"],
            "invalidation_codes": [],
        }
    if lifecycle == "CLOSED":
        return {
            "state": "terminal",
            "reason_codes": ["pr_closed"],
            "invalidation_codes": [],
        }
    if not _as_text(pr.get("head_sha")):
        return {
            "state": "unavailable",
            "reason_codes": ["head_sha_unavailable"],
            "invalidation_codes": [],
        }
    if not isinstance(pr.get("checks"), dict) or not _as_text(pr["checks"].get("overall")):
        return {
            "state": "unavailable",
            "reason_codes": ["ci_status_unavailable"],
            "invalidation_codes": [],
        }
    baseline = _baseline(pr, invalid_codes)
    identity_reason_codes: list[str] = []
    identity = _identity(pr, baseline, identity_reason_codes)
    dimensions = _dimensions(pr, invalid_codes)
    if "base_sensitivity" not in pr and baseline["verdict"] in {
        "unavailable",
        "missing-base",
        "unavailable-current-main",
    }:
        dimensions["base_sensitivity"] = "unavailable"
    if invalid_codes:
        return {
            "state": "invalid",
            "reason_codes": invalid_codes,
            "invalidation_codes": invalid_codes,
            "dimensions": dimensions,
        }
    policy_now = _parse_time(as_of) or datetime(1970, 1, 1, tzinfo=UTC)
    policy_state = classify_pr_state(pr, now=policy_now)
    blockers, blocker_codes = _blockers(
        pr, policy_state=policy_state, baseline=baseline, identity=identity
    )
    age_codes: list[str] = []
    age = _age_freshness(pr, as_of=as_of, reason_codes=age_codes)
    reason_codes = _unique(identity_reason_codes + blocker_codes + age_codes)
    hard_unavailable = {
        "head_sha_unavailable",
        "base_sha_unavailable",
        "current_main_sha_unavailable",
    }
    if hard_unavailable.intersection(reason_codes):
        state = "unavailable"
    elif pr.get("draft") is True:
        state = "preparation_pr"
        reason_codes.append("pr_is_draft")
    elif policy_state in {"active_writer", "author_decision"}:
        state = "review_active"
    elif blocker_codes:
        state = "integration_blocked"
    elif policy_state == "ready_to_merge":
        state = "merge_candidate"
    elif policy_state in {
        "pending_ci",
        "pending_gate_verdict",
        "pending_pr_metadata",
        "failed_ci",
        "stale_worktree",
        "stale_merge_base",
        "blocked_preflight",
        "unknown_review_threads",
    }:
        state = "integration_blocked"
    elif _review_state(pr) in {"CHANGES_REQUESTED", "COMMENTED", "APPROVED"}:
        state = "review_active"
    else:
        state = "integration_candidate"
    return {
        "state": state,
        "reason_codes": _unique(reason_codes),
        "invalidation_codes": _unique(
            blocker_codes if state in {"integration_blocked", "review_active"} else []
        ),
        "dimensions": dimensions,
        "identity": identity,
        "current_main_baseline": baseline,
        "blockers": blockers,
        "age_freshness": age,
        "policy_state": policy_state,
    }


def _queue_summary(
    rows: list[dict[str, Any]], *, snapshot: dict[str, Any], limit: int | None
) -> dict[str, Any]:
    bounded_rows = rows if limit is None else rows[: max(limit, 0)]
    states: dict[str, int] = dict.fromkeys(VALID_STATES, 0)
    reason_codes: list[str] = []
    invalidation_codes: list[str] = []
    for row in bounded_rows:
        result = _classify_pr(row, as_of=_as_of(snapshot, None))
        state = str(result.get("state", "invalid"))
        states[state if state in states else "invalid"] += 1
        reason_codes.extend(str(code) for code in result.get("reason_codes", []))
        invalidation_codes.extend(str(code) for code in result.get("invalidation_codes", []))
    lane_rows: list[dict[str, Any]] = []
    for row in bounded_rows:
        result = _classify_pr(row, as_of=_as_of(snapshot, None))
        if result.get("state") in _CAPACITY_STATES:
            lane_rows.append(result)

    def lane_counts(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in lane_rows:
            dimensions = result.get("dimensions")
            value = dimensions.get(key) if isinstance(dimensions, dict) else None
            category = value if isinstance(value, str) else "unavailable"
            counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items()))

    ci_counts = lane_counts("ci_cost")
    review_counts = lane_counts("review_requirement")
    external_counts = lane_counts("external_action")
    lane_demand = {
        "ci": {
            "candidates": len(lane_rows),
            "demand": len(lane_rows),
            "by_cost": ci_counts,
        },
        "review": {
            "candidates": len(lane_rows),
            "demand": len(lane_rows),
            "by_requirement": review_counts,
        },
        "external": {
            "candidates": len(lane_rows),
            "demand": sum(
                count for category, count in external_counts.items() if category != "none"
            ),
            "by_action": external_counts,
        },
    }
    truncated = bool(snapshot.get("truncated")) or (limit is not None and len(rows) > limit)
    if truncated:
        invalidation_codes.append("queue_snapshot_truncated")
    demand = sum(states[state] for state in _CAPACITY_STATES)
    capacity = snapshot.get("capacity")
    capacity_known = isinstance(capacity, int) and not isinstance(capacity, bool) and capacity >= 0
    return {
        "bounded": True,
        "requested_limit": limit,
        "observed_rows": len(bounded_rows),
        "source_rows": len(rows),
        "truncated": truncated,
        "state_counts": states,
        "capacity_demand": {
            "demand": demand,
            "capacity": capacity if capacity_known else None,
            "capacity_known": capacity_known,
            "pressure": (
                "unknown"
                if truncated or not capacity_known
                else "over_capacity"
                if demand > capacity
                else "within_capacity"
            ),
        },
        "lane_demand": lane_demand,
        "reason_codes": _unique(reason_codes),
        "invalidation_codes": _unique(invalidation_codes),
    }


def build_report(
    snapshot: Any,
    *,
    pr_number: int | None = None,
    as_of: str | None = None,
    max_queue_items: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic report from an offline queue snapshot."""
    selected, rows, selection_codes = _select_pr(snapshot, pr_number)
    snapshot_dict = snapshot if isinstance(snapshot, dict) else {}
    effective_as_of = _as_of(snapshot_dict, as_of)
    queue_snapshot = dict(snapshot_dict)
    if effective_as_of is not None:
        queue_snapshot["captured_at"] = effective_as_of
    queue = _queue_summary(rows, snapshot=queue_snapshot, limit=max_queue_items)
    if selection_codes:
        state = (
            "invalid"
            if any(code.endswith("required") or code.endswith("object") for code in selection_codes)
            else "unavailable"
        )
        pr_report: dict[str, Any] = {
            "state": state,
            "reason_codes": selection_codes,
            "invalidation_codes": selection_codes if state == "invalid" else [],
        }
    else:
        pr_report = _classify_pr(selected, as_of=effective_as_of)
        if isinstance(selected, dict):
            _ensure_classification_shape(pr_report, selected, as_of=effective_as_of)
    return {
        "schema": SCHEMA_VERSION,
        "report_only": True,
        "input": {
            "snapshot_schema": snapshot_dict.get("schema"),
            "repo": snapshot_dict.get("repo"),
            "as_of": effective_as_of,
        },
        "pr": pr_report,
        "queue": queue,
    }


def _read_input(path: Path | None, use_stdin: bool) -> Any:
    if use_stdin:
        return json.load(sys.stdin)
    if path is None:
        raise ValueError("provide --snapshot or --stdin")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_classification_shape(
    report: dict[str, Any], pr: dict[str, Any], *, as_of: str | None
) -> None:
    """Keep unavailable, invalid, and terminal rows shape-compatible with full rows."""
    report.setdefault(
        "dimensions",
        dict.fromkeys(
            (
                "change_class",
                "shared_surface",
                "ci_cost",
                "review_requirement",
                "external_action",
                "base_sensitivity",
                "ownership",
            ),
            "unavailable",
        ),
    )
    report.setdefault(
        "identity",
        {
            "pr_number": pr.get("number"),
            "base_sha": _as_text(pr.get("base_sha")) or None,
            "head_sha": _as_text(pr.get("head_sha")) or None,
            "expected_head_sha": _as_text(pr.get("expected_head_sha")) or None,
            "head_matches_expected": None,
            "status": "unavailable",
        },
    )
    report.setdefault(
        "current_main_baseline",
        {
            "base_sha": None,
            "current_main_sha": None,
            "verdict": "unavailable",
            "available": False,
            "source": "unavailable",
        },
    )
    report.setdefault(
        "blockers",
        {
            key: {"blocked": False, "codes": []}
            for key in ("dependency", "decision", "review", "external")
        },
    )
    report.setdefault("age_freshness", _age_freshness(pr, as_of=as_of, reason_codes=[]))


def build_parser() -> argparse.ArgumentParser:
    """Build the offline report CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Existing queue snapshot JSON file.")
    parser.add_argument("--stdin", action="store_true", help="Read the snapshot JSON from stdin.")
    parser.add_argument("--pr", dest="pr_number", type=int, help="PR number to classify.")
    parser.add_argument(
        "--as-of", help="Fixed ISO-8601 instant used for age/freshness classification."
    )
    parser.add_argument(
        "--max-queue-items", type=int, help="Bound the rows included in capacity demand."
    )
    parser.add_argument("--output", type=Path, help="Optional report output path.")
    parser.add_argument("--json", action="store_true", help="Emit indented JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the report-only offline classifier."""
    args = build_parser().parse_args(argv)
    if args.stdin and args.snapshot:
        print("--stdin cannot be combined with --snapshot", file=sys.stderr)
        return 2
    if not args.stdin and args.snapshot is None:
        print("provide --snapshot or --stdin", file=sys.stderr)
        return 2
    if args.max_queue_items is not None and args.max_queue_items < 0:
        print("--max-queue-items must be non-negative", file=sys.stderr)
        return 2
    try:
        payload = build_report(
            _read_input(args.snapshot, args.stdin),
            pr_number=args.pr_number,
            as_of=args.as_of,
            max_queue_items=args.max_queue_items,
        )
        encoded = json.dumps(payload, indent=2 if args.json else None, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"integration admission report failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
