"""Tests for independent planner-outcome packet handling."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.adversarial.independent_outcomes import (
    build_independent_outcome_evaluation,
    load_independent_outcomes,
    payload_sha256,
)

_PLANNER_CONFIG_SHA256 = "b" * 64
_RECORD_SHA256 = "c" * 64
_REPLAY_SIGNATURE = "d" * 64


def _legacy_payload() -> dict:
    """Build the legacy flat packet shape that must remain inadmissible."""
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
    payload = _legacy_payload()
    payload["objective"] = "archive_nearness"

    result = build_independent_outcome_evaluation(payload, budget=4, n_permutations=100, seed=0)

    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert result["independent_outcomes_available"] is False
    assert "archive-nearness" in result["reason"]


def test_build_independent_outcome_evaluation_rejects_legacy_flat_packet() -> None:
    """Legacy flat arrays cannot open the independent-outcome gate."""
    payload = _legacy_payload()
    payload["certification_statuses"][-1] = "not_available"

    result = build_independent_outcome_evaluation(payload, budget=4, n_permutations=100, seed=0)

    assert result["status"] == "blocked_legacy_independent_outcomes"
    assert result["independent_outcomes_available"] is False
    assert result["certification_available"] is False


def test_build_independent_outcome_evaluation_complete_packet_rejects_nulls() -> None:
    """A strong packet can satisfy independent outcome, certification, and null-test gates."""
    rows = [row for rank in range(4) for row in _candidate_rows("proposal", rank, 10.0)] + [
        row for rank in range(4) for row in _candidate_rows("random", rank + 4, 0.0)
    ]
    payload = _v2_payload(rows)
    result = build_independent_outcome_evaluation(
        payload,
        budget=4,
        n_permutations=200,
        seed=0,
        outcome_contract=_frozen_outcome_contract(rows),
    )

    assert result["status"] == "complete"
    assert result["independent_outcomes_available"] is True
    assert result["certification_available"] is True
    assert result["null_tests_reject_null"] is True
    assert result["null_tests"]["shuffled_archive_outcomes"]["status"] == "complete"
    assert result["null_tests"]["proposal_ranking_permutation"]["status"] == "complete"


def test_build_independent_outcome_evaluation_rejects_degraded_rows() -> None:
    """Fallback or degraded rows must not count as successful evidence."""
    payload = _legacy_payload()
    payload["row_statuses"][-1] = "degraded"

    result = build_independent_outcome_evaluation(payload, budget=4, n_permutations=100, seed=0)

    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "degraded" in result["reason"]


def test_build_independent_outcome_evaluation_rejects_eval_hash_mismatch() -> None:
    """Outcome packets must match the held-out eval split they claim to score."""
    payload = _legacy_payload()
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
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    state, reason, payload = load_independent_outcomes(path)

    assert state == "active"
    assert "loaded successfully" in reason
    assert payload is not None
    assert payload["outcome_source"] == "planner_execution"


def _v2_row(
    arm: str, rank: int, outcome: float, *, confirmation_index: int, status: str = "native"
) -> dict:
    """Build a valid v2 row with complete lineage metadata."""
    return {
        "candidate_id": f"{arm}_{rank}",
        "manifest_sha256": "pending",
        "selection_arm": arm,
        "rank": rank,
        "candidate_pool_seed": 42,
        "target_planner_id": "social_force",
        "planner_config_sha256": _PLANNER_CONFIG_SHA256,
        "scenario_family": "classic_cross_trap_medium",
        "scenario_seed": 100 + rank,
        "execution_seed": 10000 + rank * 10 + confirmation_index + (1000 if arm == "random" else 0),
        "execution_commit": "ecf997d",
        "command_lineage": "robot_sf_bench ...",
        "execution_status": status,
        "termination_reason": "collision" if outcome >= 8.0 else "goal_reached",
        "independent_failure_outcome": outcome,
        "scenario_certification_status": "passed",
        "candidate_certification_status": "passed",
        "replay_lineage": "replay.jsonl",
        "replay_signature": _REPLAY_SIGNATURE,
        "failure_attribution": {
            "status": "attributed",
            "primary_failure": "collision" if outcome >= 8.0 else "goal_reached",
            "details": {"termination_reason": "collision" if outcome >= 8.0 else "goal_reached"},
        },
        "record_hash": _RECORD_SHA256,
        "exclusion_reason": None,
    }


def _candidate_rows(arm: str, rank: int, outcome: float, status: str = "native") -> list[dict]:
    """Build the exact five confirmation rows for one frozen candidate."""
    return [
        _v2_row(arm, rank, outcome, confirmation_index=index, status=status) for index in range(5)
    ]


def _frozen_outcome_contract(rows: list[dict]) -> dict:
    """Build the minimal frozen contract that binds the supplied v2 rows."""
    candidates_by_id: dict[str, dict] = {}
    for row in rows:
        candidate = candidates_by_id.setdefault(
            row["candidate_id"],
            {
                "candidate_id": row["candidate_id"],
                "selection_arm": row["selection_arm"],
                "rank": row["rank"],
                "candidate_pool_seed": row["candidate_pool_seed"],
                "scenario_seed": row["scenario_seed"],
                "execution_seeds": [],
            },
        )
        candidate["execution_seeds"].append(row["execution_seed"])
    candidates = list(candidates_by_id.values())
    manifest_sha256 = payload_sha256({"candidates": candidates})
    for row in rows:
        row["manifest_sha256"] = manifest_sha256
    return {
        "target_planner": "social_force",
        "target_planner_config_sha256": _PLANNER_CONFIG_SHA256,
        "eval_scenario_family": "classic_cross_trap_medium",
        "study_parameters": {
            "candidate_budget_per_arm": len(candidates) // 2,
            "confirmation_seeds_per_candidate": 5,
        },
        "outcome_admission": {
            "schema_version": "adversarial_independent_outcomes.v2",
            "execution_status": "native",
            "independent_seed_confirmation": {
                "minimum_confirmed_count": 4,
                "confirmation_seeds_per_candidate": 5,
            },
            "candidate_manifest": {
                "status": "frozen",
                "sha256": manifest_sha256,
                "candidates": candidates,
            },
        },
    }


def _v2_payload(rows: list[dict]) -> dict:
    """Wrap v2 rows in the independent planner-outcome packet envelope."""
    return {
        "schema_version": "adversarial_independent_outcomes.v2",
        "source": "unit-test-fixture",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "rows": rows,
    }


def test_build_independent_outcome_evaluation_with_valid_v2_rows() -> None:
    """A valid v2 payload with complete row lineage is admitted and computes metrics."""
    rows = [
        *[row for rank in range(4) for row in _candidate_rows("proposal", rank, 10.0)],
        *[row for rank in range(4) for row in _candidate_rows("random", rank + 4, 0.0)],
    ]
    payload = _v2_payload(rows)

    result = build_independent_outcome_evaluation(
        payload,
        budget=4,
        n_permutations=200,
        seed=0,
        outcome_contract=_frozen_outcome_contract(rows),
    )
    assert result["status"] == "complete"
    assert result["independent_outcomes_available"] is True
    assert result["certification_available"] is True
    assert result["proposal_metrics"]["certified_failure_yield"] == 1.0
    assert result["random_metrics"]["certified_failure_yield"] == 0.0


def test_candidate_yield_uses_four_of_five_confirmations_once_per_candidate() -> None:
    """Three failure rows cannot inflate one candidate's certified binary yield."""
    proposal_rows = _candidate_rows("proposal", 0, 0.0)
    for row in proposal_rows[:3]:
        row["termination_reason"] = "collision"
        row["independent_failure_outcome"] = 10.0
        row["failure_attribution"] = {
            "status": "attributed",
            "primary_failure": "collision",
            "details": {"termination_reason": "collision"},
        }
    rows = [*proposal_rows, *_candidate_rows("random", 0, 0.0)]
    contract = _frozen_outcome_contract(rows)

    result = build_independent_outcome_evaluation(
        _v2_payload(rows), budget=1, n_permutations=20, seed=0, outcome_contract=contract
    )

    assert result["status"] == "complete"
    assert result["proposal_metrics"] == {
        "candidate_count": 1,
        "certified_failure_yield": 0.0,
        "certified_failure_count": 0,
    }


def test_candidate_outcomes_reject_duplicate_execution_rows() -> None:
    """A duplicate seed cannot replace a missing confirmation execution for one candidate."""
    rows = [*_candidate_rows("proposal", 0, 10.0), *_candidate_rows("random", 0, 0.0)]
    contract = _frozen_outcome_contract(rows)
    rows[1]["execution_seed"] = rows[0]["execution_seed"]

    result = build_independent_outcome_evaluation(
        _v2_payload(rows), budget=1, n_permutations=20, seed=0, outcome_contract=contract
    )

    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "exactly once" in result["reason"]


def test_candidate_outcomes_reject_manifest_content_or_replay_attribution_drift() -> None:
    """Candidate content, deterministic replay, and failure mechanism evidence are contract-bound."""
    rows = [*_candidate_rows("proposal", 0, 10.0), *_candidate_rows("random", 0, 0.0)]
    contract = _frozen_outcome_contract(rows)
    manifest = contract["outcome_admission"]["candidate_manifest"]
    manifest["candidates"][0]["scenario_seed"] = 999

    result = build_independent_outcome_evaluation(
        _v2_payload(rows), budget=1, n_permutations=20, seed=0, outcome_contract=contract
    )
    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "sha256 does not match its content" in result["reason"]

    rows = [*_candidate_rows("proposal", 0, 10.0), *_candidate_rows("random", 0, 0.0)]
    contract = _frozen_outcome_contract(rows)
    rows[1]["replay_signature"] = "e" * 64

    result = build_independent_outcome_evaluation(
        _v2_payload(rows), budget=1, n_permutations=20, seed=0, outcome_contract=contract
    )
    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "exact deterministic replay signature" in result["reason"]

    rows = [*_candidate_rows("proposal", 0, 10.0), *_candidate_rows("random", 0, 0.0)]
    contract = _frozen_outcome_contract(rows)
    rows[2]["failure_attribution"]["primary_failure"] = "timeout"
    result = build_independent_outcome_evaluation(
        _v2_payload(rows), budget=1, n_permutations=20, seed=0, outcome_contract=contract
    )
    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "unstable failure attribution" in result["reason"]


def test_build_independent_outcome_evaluation_rejects_incomplete_v2_row_lineage() -> None:
    """Rows missing required lineage fields fail closed."""
    rows = [
        *_candidate_rows("proposal", 0, 10.0),
        *_candidate_rows("random", 0, 0.0),
    ]
    contract = _frozen_outcome_contract(rows)
    # Delete lineage field from row 0.
    del rows[0]["manifest_sha256"]

    payload = _v2_payload(rows)

    result = build_independent_outcome_evaluation(
        payload, budget=2, n_permutations=100, seed=0, outcome_contract=contract
    )
    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "missing lineage fields" in result["reason"]


def test_build_independent_outcome_evaluation_requires_a_frozen_v2_contract() -> None:
    """Even structurally valid v2 rows cannot bypass the frozen study contract."""
    rows = [*_candidate_rows("proposal", 0, 10.0), *_candidate_rows("random", 0, 0.0)]

    result = build_independent_outcome_evaluation(
        _v2_payload(rows), budget=1, n_permutations=10, seed=0, outcome_contract=None
    )

    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "require a frozen study contract" in result["reason"]


def test_build_independent_outcome_evaluation_rejects_degraded_v2_row() -> None:
    """Rows with fallback or degraded execution status fail closed."""
    rows = [
        *_candidate_rows("proposal", 0, 10.0, status="degraded"),
        *_candidate_rows("random", 0, 0.0),
    ]

    payload = _v2_payload(rows)

    result = build_independent_outcome_evaluation(
        payload,
        budget=2,
        n_permutations=100,
        seed=0,
        outcome_contract=_frozen_outcome_contract(rows),
    )
    assert result["status"] == "blocked_invalid_independent_outcomes"
    assert "invalid execution_status" in result["reason"]


def test_build_independent_outcome_evaluation_rejects_contract_drift() -> None:
    """Wrong planner, config, family, candidate, seed, or success status fail closed."""
    rows = [*_candidate_rows("proposal", 0, 10.0), *_candidate_rows("random", 0, 0.0)]
    contract = _frozen_outcome_contract(rows)
    for key, value in (
        ("target_planner_id", "goal"),
        ("planner_config_sha256", "d" * 64),
        ("scenario_family", "classic_group_crossing_medium"),
        ("candidate_id", "not-in-manifest"),
        ("execution_seed", 9999),
        ("execution_status", "success"),
    ):
        candidate_rows = [dict(row) for row in rows]
        candidate_rows[0][key] = value
        result = build_independent_outcome_evaluation(
            _v2_payload(candidate_rows),
            budget=1,
            n_permutations=100,
            seed=0,
            outcome_contract=contract,
        )
        assert result["status"] == "blocked_invalid_independent_outcomes"
        assert result["independent_outcomes_available"] is False
