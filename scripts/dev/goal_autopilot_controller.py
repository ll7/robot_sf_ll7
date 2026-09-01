#!/usr/bin/env python3
"""Arbitrate controller-wide goal-autopilot work and emit zero-work proof.

Lane workers report local exhaustion only.  This module is the parent-controller
boundary: it combines complete issue, pull-request, preparation, and discovery
evidence, chooses the next lane by deterministic precedence, and emits
``goal_autopilot_zero_work_proof.v1`` only when every terminal condition is
fresh and head-bound.

The input is deliberately evidence-shaped rather than GitHub-shaped.  Callers
are responsible for collecting live snapshots and for computing the freshness
digests.  A digest must change when its covered state changes:

``issue_state_digest``
    Issue labels and bodies used by the implementation/preparation scans.
``claim_state_digest``
    Atomic issue-claim state.
``pr_head_digest``
    Open PR heads and their review/merge state.
``preparation_audit_digest``
    The exact open-issue preparation audit.
``discovery_relevant_paths_digest``
    Paths and inputs covered by the discovery saturation decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONTROLLER_SCHEMA = "goal_autopilot_controller_decision.v1"
ZERO_WORK_PROOF_SCHEMA = "goal_autopilot_zero_work_proof.v1"
LANE_RESULT_SCHEMA = "goal_autopilot_lane_result.v1"
GLOBAL_ZERO_WORK = "genuine_zero_work"

LANE_EXHAUSTION_STATES = frozenset(
    {
        "implementation_queue_exhausted",
        "review_queue_exhausted",
        "merge_queue_exhausted",
        "preparation_queue_exhausted",
        "discovery_lane_saturated",
    }
)
LANE_NAMES = frozenset({"implementation", "review", "merge", "preparation", "discovery"})
FRESHNESS_FIELDS = (
    "issue_state_digest",
    "claim_state_digest",
    "pr_head_digest",
    "preparation_audit_digest",
    "discovery_relevant_paths_digest",
)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _stable_json(payload: object) -> str:
    """Return canonical JSON for deterministic digests and CLI output."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: object) -> str:
    """Return the SHA-256 digest of canonical JSON.

    This helper is intentionally public so snapshot producers can use the same
    serialization rule as the controller when building freshness evidence.
    """
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a JSON-shaped mapping without mutating caller-owned evidence."""
    return {str(key): value[key] for key in value}


def _lane(
    snapshot: Mapping[str, Any], name: str, errors: list[str], *aliases: str
) -> Mapping[str, Any]:
    """Return one lane mapping and record malformed/missing evidence."""
    value = snapshot.get(name)
    if value is None:
        value = next(
            (snapshot.get(alias) for alias in aliases if snapshot.get(alias) is not None), None
        )
    if not isinstance(value, Mapping):
        errors.append(f"{name}_evidence_missing_or_not_object")
        return {}
    return value


def _count(lane: Mapping[str, Any], *, lane_name: str, field: str, errors: list[str]) -> int | None:
    """Read one non-negative count without treating booleans as integers."""
    value = lane.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{lane_name}.{field}_must_be_non_negative_integer")
        return None
    return value


def _sha(value: Any, *, field: str, errors: list[str], length: int = 64) -> str | None:
    """Read a lowercase hexadecimal digest and record malformed values."""
    pattern = FULL_SHA_RE if length == 40 else SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        errors.append(f"{field}_must_be_{length}_character_lowercase_sha")
        return None
    return value


def _freshness(snapshot: Mapping[str, Any], *, errors: list[str]) -> dict[str, str] | None:
    """Validate and return the freshness block required for a terminal proof."""
    raw = snapshot.get("freshness")
    if not isinstance(raw, Mapping):
        errors.append("freshness_missing_or_not_object")
        return None
    values: dict[str, str] = {}
    for field in FRESHNESS_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            errors.append(f"freshness.{field}_must_be_64_character_lowercase_sha")
        else:
            values[field] = value
    return values if len(values) == len(FRESHNESS_FIELDS) else None


def _lane_origin_checks(
    lanes: Mapping[str, Mapping[str, Any]], *, origin_main_sha: str | None, errors: list[str]
) -> None:
    """Reject lane evidence collected against a different base revision."""
    if origin_main_sha is None:
        return
    for lane_name, lane in lanes.items():
        lane_sha = lane.get("origin_main_sha")
        if lane_sha is None:
            continue
        if lane_sha != origin_main_sha:
            errors.append(f"{lane_name}_origin_main_sha_drift")


def _status(value: Any, *, field: str, errors: list[str]) -> str | None:
    """Read a non-empty lane status."""
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field}_must_be_non_empty_string")
        return None
    return value.strip()


def _reconciliation_complete(preparation: Mapping[str, Any], count: int | None) -> bool:
    """Interpret the legacy zero-count form while preferring an explicit flag."""
    explicit = preparation.get("blocker_reconciliation_complete")
    if explicit is not None:
        return explicit is True and count == 0
    return count == 0


def _readiness_outcomes_complete(discovery: Mapping[str, Any]) -> bool:
    """Return whether every issue created by discovery has a recorded gate outcome."""
    explicit = discovery.get("readiness_outcomes_complete")
    if explicit is not None:
        if explicit is not True:
            return False
    created = discovery.get("created_issue_numbers")
    outcomes = discovery.get("readiness_outcomes")
    if not isinstance(created, list) or not isinstance(outcomes, list):
        return False
    if len(created) != len(outcomes):
        return False
    return all(
        isinstance(outcome, Mapping)
        and isinstance(outcome.get("outcome"), str)
        and bool(outcome["outcome"].strip())
        and isinstance(outcome.get("verified"), bool)
        for outcome in outcomes
    )


def _terminal_evidence(  # noqa: C901, PLR0912 - explicit fail-closed evidence normalization
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Collect normalized lane evidence and terminal-condition failures."""
    errors: list[str] = []
    origin_main_sha = _sha(
        snapshot.get("origin_main_sha"), field="origin_main_sha", errors=errors, length=40
    )
    implementation = _lane(snapshot, "implementation", errors, "issue_queue")
    pull_requests = _lane(snapshot, "pull_requests", errors, "pr_queue")
    preparation = _lane(snapshot, "preparation", errors)
    discovery = _lane(snapshot, "discovery", errors)
    lanes = {
        "implementation": implementation,
        "pull_requests": pull_requests,
        "preparation": preparation,
        "discovery": discovery,
    }
    _lane_origin_checks(lanes, origin_main_sha=origin_main_sha, errors=errors)

    claimable_count = _count(
        implementation, lane_name="implementation", field="claimable_count", errors=errors
    )
    merge_ready_count = _count(
        pull_requests, lane_name="pull_requests", field="merge_ready_count", errors=errors
    )
    review_eligible_count = _count(
        pull_requests, lane_name="pull_requests", field="review_eligible_count", errors=errors
    )
    recoverable_active_count = _count(
        pull_requests, lane_name="pull_requests", field="recoverable_active_count", errors=errors
    )
    open_count = _count(pull_requests, lane_name="pull_requests", field="open_count", errors=errors)
    promotable_count = _count(
        preparation, lane_name="preparation", field="promotable_count", errors=errors
    )
    formalizable_count = _count(
        preparation, lane_name="preparation", field="formalizable_count", errors=errors
    )
    blocker_reconciliation_count = _count(
        preparation,
        lane_name="preparation",
        field="blocker_reconciliation_count",
        errors=errors,
    )

    optional_counts: dict[str, int | None] = {}
    for field in ("decision_count", "blocker_count", "decomposition_count", "active_handoff_count"):
        value = preparation.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"preparation.{field}_must_be_non_negative_integer")
            optional_counts[field] = None
        else:
            optional_counts[field] = value

    queue_completeness = implementation.get("queue_completeness")
    if queue_completeness != "complete":
        errors.append("implementation_queue_not_complete")
    if implementation.get("zero_work_authoritative") is not True:
        errors.append("implementation_zero_work_not_authoritative")
    candidate_scope = implementation.get("candidate_scope")
    if candidate_scope != "state:ready":
        errors.append("implementation_candidate_scope_not_state_ready")
    admission_histogram = implementation.get("admission_reason_histogram")
    if not isinstance(admission_histogram, Mapping):
        errors.append("implementation_admission_reason_histogram_missing_or_not_object")
    else:
        for reason, value in admission_histogram.items():
            if not isinstance(reason, str) or not reason.strip():
                errors.append("implementation_admission_reason_histogram_key_invalid")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(
                    "implementation_admission_reason_histogram_value_must_be_non_negative_integer"
                )

    discovery_status = _status(discovery.get("status"), field="discovery.status", errors=errors)
    relevant_head_sha: str | None = None
    if discovery_status == "saturated":
        relevant_head_sha = _sha(
            discovery.get("relevant_head_sha"),
            field="discovery.relevant_head_sha",
            errors=errors,
            length=40,
        )
        if origin_main_sha is not None and relevant_head_sha is not None:
            if relevant_head_sha != origin_main_sha:
                errors.append("discovery_relevant_head_sha_drift")
        if not _readiness_outcomes_complete(discovery):
            errors.append("discovery_readiness_outcomes_incomplete")
        if not isinstance(discovery.get("created_issue_numbers"), list) or not isinstance(
            discovery.get("readiness_outcomes"), list
        ):
            errors.append("discovery_readiness_outcomes_missing_or_not_lists")

    freshness = _freshness(snapshot, errors=errors)
    _sha(preparation.get("audit_digest"), field="preparation.audit_digest", errors=errors)
    if discovery_status == "saturated" and (
        not isinstance(discovery.get("lane"), str) or not discovery.get("lane", "").strip()
    ):
        errors.append("discovery.lane_must_be_non_empty_string")

    normalized = {
        "origin_main_sha": origin_main_sha,
        "implementation": implementation,
        "pull_requests": pull_requests,
        "preparation": preparation,
        "discovery": discovery,
        "freshness": freshness,
        "counts": {
            "claimable_count": claimable_count,
            "merge_ready_count": merge_ready_count,
            "review_eligible_count": review_eligible_count,
            "recoverable_active_count": recoverable_active_count,
            "open_count": open_count,
            "promotable_count": promotable_count,
            "formalizable_count": formalizable_count,
            "blocker_reconciliation_count": blocker_reconciliation_count,
            **optional_counts,
        },
        "queue_completeness": queue_completeness,
        "discovery_status": discovery_status,
        "relevant_head_sha": relevant_head_sha,
        "reconciliation_complete": _reconciliation_complete(
            preparation, blocker_reconciliation_count
        ),
        "readiness_outcomes_complete": _readiness_outcomes_complete(discovery),
    }
    return (
        normalized,
        errors,
        {
            "origin_main_sha": origin_main_sha,
            "freshness": freshness,
        },
    )


def _positive(counts: Mapping[str, int | None], field: str) -> bool:
    """Return whether a known count contains actionable work."""
    value = counts.get(field)
    return isinstance(value, int) and value > 0


def _zero(counts: Mapping[str, int | None], field: str) -> bool:
    """Return whether a known count is exactly zero."""
    return counts.get(field) == 0


def _build_zero_work_proof(normalized: Mapping[str, Any]) -> dict[str, Any]:
    """Build the immutable receipt after terminal conditions have passed."""
    implementation = normalized["implementation"]
    preparation = normalized["preparation"]
    discovery = normalized["discovery"]
    freshness = normalized["freshness"]
    assert isinstance(freshness, Mapping)
    return {
        "schema": ZERO_WORK_PROOF_SCHEMA,
        "origin_main_sha": normalized["origin_main_sha"],
        "implementation": {
            "candidate_scope": "state:ready",
            "queue_completeness": implementation.get("queue_completeness"),
            "zero_work_authoritative": implementation.get("zero_work_authoritative"),
            "claimable_count": normalized["counts"]["claimable_count"],
            "admission_reason_histogram": dict(
                sorted((implementation.get("admission_reason_histogram") or {}).items())
            ),
        },
        "pull_requests": {
            "open_count": normalized["counts"]["open_count"],
            "recoverable_active_count": normalized["counts"]["recoverable_active_count"],
            "review_eligible_count": normalized["counts"]["review_eligible_count"],
            "merge_ready_count": normalized["counts"]["merge_ready_count"],
        },
        "preparation": {
            "audit_digest": preparation.get("audit_digest"),
            "promotable_count": normalized["counts"]["promotable_count"],
            "formalizable_count": normalized["counts"]["formalizable_count"],
            "blocker_reconciliation_count": normalized["counts"]["blocker_reconciliation_count"],
            "blocker_reconciliation_complete": True,
        },
        "discovery": {
            "lane": discovery.get("lane"),
            "relevant_head_sha": discovery.get("relevant_head_sha"),
            "status": discovery.get("status"),
            "created_issue_numbers": list(discovery.get("created_issue_numbers") or []),
            "readiness_outcomes": list(discovery.get("readiness_outcomes") or []),
            "readiness_outcomes_complete": True,
        },
        "freshness": dict(freshness),
        "stop_reason": GLOBAL_ZERO_WORK,
    }


def _lane_statuses(normalized: Mapping[str, Any]) -> dict[str, str]:
    """Return lane-local status names without ever producing a global terminal state."""
    counts = normalized["counts"]
    implementation_status = (
        "implementation_queue_exhausted"
        if normalized["queue_completeness"] == "complete"
        and normalized["implementation"].get("zero_work_authoritative") is True
        and _zero(counts, "claimable_count")
        else "implementation_queue_pending"
    )
    review_status = (
        "review_queue_exhausted"
        if _zero(counts, "review_eligible_count") and _zero(counts, "open_count")
        else "review_queue_pending"
    )
    merge_status = (
        "merge_queue_exhausted"
        if _zero(counts, "merge_ready_count") and _zero(counts, "open_count")
        else "merge_queue_pending"
    )
    preparation_status = (
        "preparation_queue_exhausted"
        if _zero(counts, "promotable_count")
        and _zero(counts, "formalizable_count")
        and normalized["reconciliation_complete"]
        else "preparation_queue_pending"
    )
    discovery_status = (
        "discovery_lane_saturated"
        if normalized["discovery_status"] == "saturated"
        and normalized["readiness_outcomes_complete"]
        else "discovery_lane_pending"
    )
    return {
        "implementation": implementation_status,
        "review": review_status,
        "merge": merge_status,
        "preparation": preparation_status,
        "discovery": discovery_status,
    }


def _choose_next_action(  # noqa: C901 - precedence is the controller contract
    normalized: Mapping[str, Any], errors: list[str]
) -> tuple[str | None, str | None]:
    """Choose the next controller lane using the documented precedence."""
    counts = normalized["counts"]
    if _positive(counts, "merge_ready_count"):
        return "merge", None
    if _positive(counts, "review_eligible_count"):
        return "review", None
    if _positive(counts, "recoverable_active_count"):
        return "recover_pr", None
    if _positive(counts, "open_count"):
        return "review", None
    if _positive(counts, "claimable_count"):
        return "implement", None
    if _positive(counts, "promotable_count"):
        return "gate_readiness", None
    if _positive(counts, "formalizable_count"):
        return "formalize_issue", None
    if _positive(counts, "decision_count"):
        return "prepare_decision", None
    if _positive(counts, "blocker_count") or not normalized["reconciliation_complete"]:
        return "reconcile_blockers", None
    if _positive(counts, "decomposition_count"):
        return "decompose_issue", None
    if _positive(counts, "active_handoff_count"):
        return "recover_pr", None

    if normalized["counts"]["claimable_count"] == 0 and (
        normalized["queue_completeness"] != "complete"
        or normalized["implementation"].get("zero_work_authoritative") is not True
    ):
        return "refresh_issue_queue", None
    if normalized["discovery_status"] is not None and normalized["discovery_status"] != "saturated":
        return "discover", None
    if (
        normalized["discovery_status"] == "saturated"
        and not normalized["readiness_outcomes_complete"]
    ):
        return "discover", None
    if errors:
        return "refresh_controller_evidence", None
    return None, GLOBAL_ZERO_WORK


def arbitrate_controller(
    snapshot: Mapping[str, Any],
    *,
    prior_zero_work_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the parent-only controller decision for one evidence snapshot.

    A lane worker's status is never accepted as a stop decision.  The returned
    ``global_zero_work`` flag is true only when the normalized evidence can be
    serialized into a complete, fresh zero-work receipt.
    """
    if not isinstance(snapshot, Mapping):
        raise ValueError("controller snapshot must be an object")
    normalized, errors, freshness_info = _terminal_evidence(snapshot)
    prior_validation: dict[str, Any] | None = None
    if prior_zero_work_proof is not None:
        current_freshness = freshness_info.get("freshness")
        current_origin = freshness_info.get("origin_main_sha")
        if not isinstance(current_origin, str) or not isinstance(current_freshness, Mapping):
            prior_validation = {
                "valid": False,
                "reasons": ["current_snapshot_has_no_valid_freshness"],
            }
        else:
            prior_validation = validate_zero_work_proof(
                prior_zero_work_proof,
                origin_main_sha=current_origin,
                freshness=current_freshness,
                snapshot=snapshot,
            )
        if prior_validation.get("valid") is not True:
            errors.extend(
                f"stale_zero_work_proof:{reason}" for reason in prior_validation.get("reasons", [])
            )

    next_action, stop_reason = _choose_next_action(normalized, errors)
    terminal_ready = (
        not errors
        and normalized["origin_main_sha"] is not None
        and normalized["queue_completeness"] == "complete"
        and normalized["implementation"].get("zero_work_authoritative") is True
        and _zero(normalized["counts"], "claimable_count")
        and _zero(normalized["counts"], "merge_ready_count")
        and _zero(normalized["counts"], "review_eligible_count")
        and _zero(normalized["counts"], "recoverable_active_count")
        and _zero(normalized["counts"], "open_count")
        and _zero(normalized["counts"], "promotable_count")
        and _zero(normalized["counts"], "formalizable_count")
        and all(
            _zero(normalized["counts"], field)
            for field in (
                "decision_count",
                "blocker_count",
                "decomposition_count",
                "active_handoff_count",
            )
        )
        and normalized["reconciliation_complete"]
        and normalized["discovery_status"] == "saturated"
        and normalized["relevant_head_sha"] == normalized["origin_main_sha"]
        and normalized["readiness_outcomes_complete"]
        and isinstance(normalized["freshness"], Mapping)
        and isinstance(normalized["preparation"].get("audit_digest"), str)
        and SHA256_RE.fullmatch(normalized["preparation"]["audit_digest"]) is not None
    )
    proof = _build_zero_work_proof(normalized) if terminal_ready else None
    if terminal_ready:
        next_action = None
        stop_reason = GLOBAL_ZERO_WORK
    elif stop_reason == GLOBAL_ZERO_WORK:
        next_action = "refresh_controller_evidence"
        stop_reason = None

    return {
        "schema": CONTROLLER_SCHEMA,
        "origin_main_sha": normalized["origin_main_sha"],
        "global_zero_work": terminal_ready,
        "next_action": next_action,
        "stop_reason": stop_reason,
        "lane_status": _lane_statuses(normalized),
        "reasons": sorted(set(errors)),
        "prior_zero_work_proof": prior_validation,
        "zero_work_proof": proof,
    }


def validate_zero_work_proof(  # noqa: C901, PLR0912, PLR0915 - validate every proof component fail-closed
    proof: Mapping[str, Any],
    *,
    origin_main_sha: str,
    freshness: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a receipt against current head and freshness evidence.

    The optional snapshot comparison catches changes to the proof's counts and
    lane verdicts in addition to digest drift.  Missing or malformed current
    evidence is invalid, never silently treated as unchanged.
    """
    reasons: list[str] = []
    if not isinstance(proof, Mapping):
        return {"valid": False, "reasons": ["proof_not_object"]}
    if proof.get("schema") != ZERO_WORK_PROOF_SCHEMA:
        reasons.append("proof_schema_mismatch")
    if not isinstance(origin_main_sha, str) or FULL_SHA_RE.fullmatch(origin_main_sha) is None:
        reasons.append("current_origin_main_sha_invalid")
    if proof.get("origin_main_sha") != origin_main_sha:
        reasons.append("origin_main_sha_drift")
    current_freshness: dict[str, Any] = {}
    if not isinstance(freshness, Mapping):
        reasons.append("current_freshness_not_object")
        freshness = {}
    for field in FRESHNESS_FIELDS:
        value = freshness.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            reasons.append(f"current_freshness_{field}_invalid")
        else:
            current_freshness[field] = value
    if proof.get("freshness") != current_freshness:
        reasons.append("freshness_digest_drift")

    implementation = proof.get("implementation")
    pull_requests = proof.get("pull_requests")
    preparation = proof.get("preparation")
    discovery = proof.get("discovery")
    for name, value in (
        ("implementation", implementation),
        ("pull_requests", pull_requests),
        ("preparation", preparation),
        ("discovery", discovery),
    ):
        if not isinstance(value, Mapping):
            reasons.append(f"proof_{name}_missing_or_not_object")

    if isinstance(implementation, Mapping):
        if implementation.get("candidate_scope") != "state:ready":
            reasons.append("proof_candidate_scope_mismatch")
        if implementation.get("queue_completeness") != "complete":
            reasons.append("proof_queue_incomplete")
        if implementation.get("zero_work_authoritative") is not True:
            reasons.append("proof_zero_work_not_authoritative")
        if implementation.get("claimable_count") != 0:
            reasons.append("proof_claimable_count_nonzero")
        histogram = implementation.get("admission_reason_histogram")
        if not isinstance(histogram, Mapping):
            reasons.append("proof_admission_reason_histogram_missing_or_not_object")
        elif any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in histogram.values()
        ):
            reasons.append("proof_admission_reason_histogram_invalid")
    if isinstance(pull_requests, Mapping):
        for field in (
            "open_count",
            "recoverable_active_count",
            "review_eligible_count",
            "merge_ready_count",
        ):
            if pull_requests.get(field) != 0:
                reasons.append(f"proof_{field}_nonzero")
    if isinstance(preparation, Mapping):
        for field in ("promotable_count", "formalizable_count", "blocker_reconciliation_count"):
            if preparation.get(field) != 0:
                reasons.append(f"proof_{field}_nonzero")
        if preparation.get("blocker_reconciliation_complete") is not True:
            reasons.append("proof_blocker_reconciliation_incomplete")
        if (
            not isinstance(preparation.get("audit_digest"), str)
            or SHA256_RE.fullmatch(preparation.get("audit_digest", "")) is None
        ):
            reasons.append("proof_audit_digest_invalid")
    if isinstance(discovery, Mapping):
        if discovery.get("status") != "saturated":
            reasons.append("proof_discovery_not_saturated")
        if discovery.get("relevant_head_sha") != origin_main_sha:
            reasons.append("proof_discovery_head_drift")
        if discovery.get("readiness_outcomes_complete") is not True:
            reasons.append("proof_discovery_readiness_incomplete")
        created = discovery.get("created_issue_numbers")
        outcomes = discovery.get("readiness_outcomes")
        if not isinstance(created, list) or not isinstance(outcomes, list):
            reasons.append("proof_discovery_readiness_outcomes_missing_or_not_lists")
        elif len(created) != len(outcomes) or not _readiness_outcomes_complete(discovery):
            reasons.append("proof_discovery_readiness_outcomes_invalid")
        if not isinstance(discovery.get("lane"), str) or not discovery.get("lane", "").strip():
            reasons.append("proof_discovery_lane_invalid")

    if snapshot is not None and not reasons:
        current_result = arbitrate_controller(snapshot)
        current_proof = current_result.get("zero_work_proof")
        if not isinstance(current_proof, Mapping):
            reasons.append("current_snapshot_has_no_zero_work_proof")
        elif current_proof != proof:
            reasons.append("proof_lane_evidence_drift")
    return {"valid": not reasons, "reasons": sorted(set(reasons))}


def lane_result(
    lane: str, status: str, *, evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a lane-local result and reject controller-wide terminal output."""
    if lane not in LANE_NAMES:
        raise ValueError(f"unknown lane: {lane!r}")
    if status == GLOBAL_ZERO_WORK:
        raise ValueError("genuine_zero_work is controller-only")
    if status not in LANE_EXHAUSTION_STATES:
        raise ValueError(f"invalid lane-local status: {status!r}")
    expected = {
        "implementation": "implementation_queue_exhausted",
        "review": "review_queue_exhausted",
        "merge": "merge_queue_exhausted",
        "preparation": "preparation_queue_exhausted",
        "discovery": "discovery_lane_saturated",
    }[lane]
    if status != expected:
        raise ValueError(f"status {status!r} does not belong to lane {lane!r}")
    return {
        "schema": LANE_RESULT_SCHEMA,
        "lane": lane,
        "status": status,
        "global_terminal": False,
        "evidence": _copy_mapping(evidence or {}),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the controller arbiter CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Controller evidence JSON")
    parser.add_argument(
        "--prior-proof",
        type=Path,
        help="Optional prior zero-work proof to validate against this snapshot",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON (the default)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic controller arbiter."""
    args = _build_parser().parse_args(argv)
    try:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        if not isinstance(snapshot, Mapping):
            raise ValueError("controller snapshot must be a JSON object")
        prior = None
        if args.prior_proof:
            prior_payload = json.loads(args.prior_proof.read_text(encoding="utf-8"))
            if not isinstance(prior_payload, Mapping):
                raise ValueError("prior zero-work proof must be a JSON object")
            prior = prior_payload
        print(json.dumps(arbitrate_controller(snapshot, prior_zero_work_proof=prior), indent=2))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"goal autopilot controller failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
