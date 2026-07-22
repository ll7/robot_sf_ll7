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


def _v2_row(arm: str, rank: int, outcome: float, status: str = "native") -> dict:
    """Build a valid v2 row with complete lineage metadata."""
    return {
        "candidate_id": f"cand_{rank}",
        "manifest_sha256": "abc123sha",
        "selection_arm": arm,
        "rank": rank,
        "candidate_pool_seed": 42,
        "target_planner_id": "social_force",
        "planner_config_sha256": "cfg123sha",
        "scenario_family": "classic_group_crossing_medium",
        "scenario_seed": 100 + rank,
        "execution_commit": "ecf997d",
        "command_lineage": "robot_sf_bench ...",
        "execution_status": status,
        "termination_reason": "collision" if outcome >= 8.0 else "goal_reached",
        "independent_failure_outcome": outcome,
        "scenario_certification_status": "passed",
        "candidate_certification_status": "passed",
        "replay_lineage": "replay.jsonl",
        "record_hash": "rechash123",
        "exclusion_reason": None,
    }


def test_build_independent_outcome_evaluation_with_valid_v2_rows() -> None:
    """A valid v2 payload with complete row lineage is admitted and computes metrics."""
    rows = [
        _v2_row("proposal", 0, 10.0),
        _v2_row("proposal", 1, 10.0),
        _v2_row("proposal", 2, 10.0),
        _v2_row("proposal", 3, 10.0),
        _v2_row("random", 0, 0.0),
        _v2_row("random", 1, 0.0),
        _v2_row("random", 2, 0.0),
        _v2_row("random", 3, 0.0),
    ]
    payload = {
        "schema_version": "adversarial_independent_outcomes.v2",
        "source": "unit-test-fixture",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "rows": rows,
    }

    result = build_independent_outcome_evaluation(payload, budget=4, n_permutations=200, seed=0)
    assert result["status"] == "complete"
    assert result["independent_outcomes_available"] is True
    assert result["certification_available"] is True
    assert result["proposal_metrics"]["mean_objective"] == 10.0
    assert result["random_metrics"]["mean_objective"] == 0.0


def test_build_independent_outcome_evaluation_rejects_incomplete_v2_row_lineage() -> None:
    """Rows missing required lineage fields fail closed."""
    rows = [
        _v2_row("proposal", 0, 10.0),
        _v2_row("random", 0, 0.0),
    ]
    # Delete lineage field from row 0
    del rows[0]["manifest_sha256"]

    payload = {
        "schema_version": "adversarial_independent_outcomes.v2",
        "source": "unit-test-fixture",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "rows": rows,
    }

    result = build_independent_outcome_evaluation(payload, budget=2, n_permutations=100, seed=0)
    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "missing lineage fields" in result["reason"]


def test_build_independent_outcome_evaluation_rejects_degraded_v2_row() -> None:
    """Rows with fallback or degraded execution status fail closed."""
    rows = [
        _v2_row("proposal", 0, 10.0, status="degraded"),
        _v2_row("random", 0, 0.0),
    ]

    payload = {
        "schema_version": "adversarial_independent_outcomes.v2",
        "source": "unit-test-fixture",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "rows": rows,
    }

    result = build_independent_outcome_evaluation(payload, budget=2, n_permutations=100, seed=0)
    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "invalid execution_status" in result["reason"]
