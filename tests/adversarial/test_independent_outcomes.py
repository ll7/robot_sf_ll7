"""Tests for independent planner-outcome packet handling."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.adversarial.independent_outcomes import (
    build_independent_outcome_evaluation,
    load_independent_outcomes,
)


def _valid_payload() -> dict:
    """Build a small packet with separable proposal/random outcome signal."""
    return {
        "schema_version": "adversarial_independent_outcomes.v1",
        "source": "unit-test-fixture",
        "artifact": "docs/context/evidence/unit-test.json",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "proposal_outcomes": [10.0, 10.0, 10.0, 10.0],
        "random_outcomes": [0.0, 0.0, 0.0, 0.0],
        "ranked_outcomes": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        "certification_statuses": ["passed"] * 8,
        "row_statuses": ["success"] * 8,
    }


def test_load_independent_outcomes_missing_is_not_available() -> None:
    """Absence of a packet is a fail-closed not-available state."""
    state, reason, payload = load_independent_outcomes(None)
    assert state == "not_available"
    assert "No independent outcome path" in reason
    assert payload is None


def test_load_independent_outcomes_malformed_file_blocks(tmp_path: Path) -> None:
    """Supplied malformed payloads block instead of falling back."""
    path = tmp_path / "outcomes.json"
    path.write_text("{not json", encoding="utf-8")

    state, reason, payload = load_independent_outcomes(path)

    assert state == "blocked"
    assert "Failed to load independent outcomes" in reason
    assert payload is None


def test_build_independent_outcome_evaluation_rejects_circular_objective() -> None:
    """Archive-nearness outcomes are circular and cannot open the claim gate."""
    payload = _valid_payload()
    payload["objective"] = "archive_nearness"

    result = build_independent_outcome_evaluation(payload, budget=4, n_permutations=100, seed=0)

    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert result["independent_outcomes_available"] is False
    assert "archive-nearness" in result["reason"]


def test_build_independent_outcome_evaluation_requires_certification() -> None:
    """Independent outcomes without passed certification stay fail-closed."""
    payload = _valid_payload()
    payload["certification_statuses"][-1] = "not_available"

    result = build_independent_outcome_evaluation(payload, budget=4, n_permutations=100, seed=0)

    assert result["status"] == "complete"
    assert result["independent_outcomes_available"] is True
    assert result["certification_available"] is False


def test_build_independent_outcome_evaluation_complete_packet_rejects_nulls() -> None:
    """A strong packet can satisfy independent outcome, certification, and null-test gates."""
    result = build_independent_outcome_evaluation(
        _valid_payload(), budget=4, n_permutations=200, seed=0
    )

    assert result["status"] == "complete"
    assert result["independent_outcomes_available"] is True
    assert result["certification_available"] is True
    assert result["null_tests_reject_null"] is True
    assert result["null_tests"]["shuffled_archive_outcomes"]["status"] == "complete"
    assert result["null_tests"]["proposal_ranking_permutation"]["status"] == "complete"


def test_build_independent_outcome_evaluation_rejects_degraded_rows() -> None:
    """Fallback or degraded rows must not count as successful evidence."""
    payload = _valid_payload()
    payload["row_statuses"][-1] = "degraded"

    result = build_independent_outcome_evaluation(payload, budget=4, n_permutations=100, seed=0)

    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "degraded" in result["reason"]


def test_build_independent_outcome_evaluation_rejects_eval_hash_mismatch() -> None:
    """Outcome packets must match the held-out eval split they claim to score."""
    payload = _valid_payload()
    payload["eval_archive_sha256"] = "wrong"

    result = build_independent_outcome_evaluation(
        payload,
        budget=4,
        n_permutations=100,
        seed=0,
        expected_eval_archive_sha256="expected",
    )

    assert result["status"] == "blocked_eval_archive_hash_mismatch"
    assert result["independent_outcomes_available"] is False
    assert result["expected_eval_archive_sha256"] == "expected"
    assert result["observed_eval_archive_sha256"] == "wrong"


def test_load_independent_outcomes_reads_json_payload(tmp_path: Path) -> None:
    """Readable JSON objects are passed through for report integration."""
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps(_valid_payload()), encoding="utf-8")

    state, reason, payload = load_independent_outcomes(path)

    assert state == "active"
    assert "loaded successfully" in reason
    assert payload is not None
    assert payload["outcome_source"] == "planner_execution"
