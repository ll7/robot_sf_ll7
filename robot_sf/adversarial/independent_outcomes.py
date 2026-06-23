"""Independent planner-outcome packet contract for issue #3275.

This module validates the small artifact shape that a future planner-execution
runner must produce before the proposal-vs-random report can move beyond
circular archive-nearness plumbing. It deliberately does not execute planners.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robot_sf.adversarial.disjoint_evaluation import (
    ranking_permutation_test,
    shuffled_outcome_null_test,
)


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for a JSON-like payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_independent_outcomes(path: Path | None) -> tuple[str, str, dict[str, Any] | None]:
    """Load an independent planner-outcome payload.

    Returns:
        ``(state, reason, payload)``. Missing payloads are not available; supplied
        but unreadable payloads are blocked because the run explicitly requested
        an independent outcome surface.
    """
    if path is None:
        return (
            "not_available",
            "No independent outcome path provided; held-out evidence remains fail-closed.",
            None,
        )
    if not path.exists():
        return "blocked", f"Independent outcome path {path} does not exist.", None
    if path.stat().st_size == 0:
        return "blocked", f"Independent outcome file {path} is empty.", None

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        return "blocked", f"Failed to load independent outcomes: {exc}.", None
    if not isinstance(payload, dict):
        return "blocked", "Independent outcome payload must be a JSON object.", None
    return "active", "Independent planner-execution outcomes loaded successfully.", payload


def _float_list(payload: dict[str, Any], key: str) -> list[float]:
    """Return a list of numeric values from ``payload[key]``."""
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a list")
    return [float(value) for value in values]


def _certification_available(payload: dict[str, Any], expected_count: int) -> bool:
    """Return whether every expected row has passed certification."""
    statuses = payload.get("certification_statuses")
    if not isinstance(statuses, list) or len(statuses) < expected_count:
        return False
    return all(status == "passed" for status in statuses)


def _metadata_allows_independent_outcomes(payload: dict[str, Any]) -> tuple[bool, str]:
    """Reject known circular, fallback, degraded, or unavailable packet metadata."""
    if payload.get("outcome_source") != "planner_execution":
        return False, "outcome_source must be planner_execution"
    if payload.get("objective") in {"archive_nearness", "objective_distance", "nearest_archive"}:
        return False, "archive-nearness outcomes are circular for issue #3275"
    row_statuses = payload.get("row_statuses", [])
    if not isinstance(row_statuses, list):
        return False, "row_statuses must be a list when provided"
    bad_statuses = {"fallback", "degraded", "not_available", "failed", "blocked"}
    observed_bad = sorted({status for status in row_statuses if status in bad_statuses})
    if observed_bad:
        return False, f"non-success planner rows present: {observed_bad}"
    return True, "ok"


def compute_metrics(values: list[float]) -> dict[str, Any]:
    """Compute the small outcome summary used by the comparison report."""
    return {
        "mean_objective": round(sum(values) / len(values), 4) if values else 0.0,
        "max_objective": round(max(values), 4) if values else 0.0,
        "failure_count": sum(1 for value in values if value >= 8.0),
    }


def build_independent_outcome_evaluation(
    payload: dict[str, Any] | None,
    *,
    budget: int,
    n_permutations: int,
    seed: int,
    expected_eval_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Summarize an independent-outcome packet and its null-test readiness."""
    if payload is None:
        return {
            "status": "not_available",
            "independent_outcomes_available": False,
            "certification_available": False,
            "null_tests_reject_null": False,
            "reason": "no_independent_outcome_payload",
        }

    metadata_ok, metadata_reason = _metadata_allows_independent_outcomes(payload)
    if not metadata_ok:
        return {
            "status": "blocked_invalid_independent_outcomes",
            "independent_outcomes_available": False,
            "certification_available": False,
            "null_tests_reject_null": False,
            "reason": metadata_reason,
            "payload_sha256": payload_sha256(payload),
        }
    if expected_eval_archive_sha256 is not None:
        observed_eval_hash = payload.get("eval_archive_sha256")
        if observed_eval_hash != expected_eval_archive_sha256:
            return {
                "status": "blocked_eval_archive_hash_mismatch",
                "independent_outcomes_available": False,
                "certification_available": False,
                "null_tests_reject_null": False,
                "reason": "independent outcome packet does not match the eval split hash",
                "expected_eval_archive_sha256": expected_eval_archive_sha256,
                "observed_eval_archive_sha256": observed_eval_hash,
                "payload_sha256": payload_sha256(payload),
            }

    try:
        proposal_outcomes = _float_list(payload, "proposal_outcomes")
        random_outcomes = _float_list(payload, "random_outcomes")
        ranked_outcomes = _float_list(payload, "ranked_outcomes")
    except (TypeError, ValueError) as exc:
        return {
            "status": "blocked_malformed_outcomes",
            "independent_outcomes_available": False,
            "certification_available": False,
            "null_tests_reject_null": False,
            "reason": str(exc),
            "payload_sha256": payload_sha256(payload),
        }

    expected_count = len(proposal_outcomes) + len(random_outcomes)
    certification_available = _certification_available(payload, expected_count)
    independent_available = bool(proposal_outcomes and random_outcomes and ranked_outcomes)
    if not independent_available:
        return {
            "status": "not_available_empty_outcomes",
            "independent_outcomes_available": False,
            "certification_available": certification_available,
            "null_tests_reject_null": False,
            "reason": "proposal, random, and ranked outcomes must all be non-empty",
            "payload_sha256": payload_sha256(payload),
        }

    shuffled = shuffled_outcome_null_test(
        proposal_outcomes,
        random_outcomes,
        n_permutations=n_permutations,
        seed=seed,
    )
    ranking = ranking_permutation_test(
        ranked_outcomes,
        selection_size=min(budget, len(ranked_outcomes)),
        n_permutations=n_permutations,
        seed=seed + 1,
    )
    nulls_reject = (
        shuffled.get("status") == "complete"
        and ranking.get("status") == "complete"
        and float(shuffled.get("p_value", 1.0)) <= 0.05
        and float(ranking.get("p_value", 1.0)) <= 0.05
    )
    return {
        "schema_version": payload.get("schema_version"),
        "status": "complete",
        "source": payload.get("source", "unspecified"),
        "objective": payload.get("objective", "unspecified"),
        "artifact": payload.get("artifact"),
        "eval_archive_sha256": payload.get("eval_archive_sha256"),
        "outcome_source": payload.get("outcome_source"),
        "independent_outcomes_available": True,
        "certification_available": certification_available,
        "null_tests_reject_null": nulls_reject,
        "proposal_metrics": compute_metrics(proposal_outcomes),
        "random_metrics": compute_metrics(random_outcomes),
        "null_tests": {
            "shuffled_archive_outcomes": shuffled,
            "proposal_ranking_permutation": ranking,
            "required_for_held_out_claim": True,
        },
        "payload_sha256": payload_sha256(payload),
    }
