"""Tests for the v2 row-level independent planner-outcome contract (issue #3275)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_sf.adversarial.independent_outcomes import (
    OUTCOME_SCHEMA_VERSION,
    AdmissionSpec,
    build_independent_outcome_evaluation,
    load_independent_outcomes,
    payload_sha256,
)

_PLANNER = "social_force"
_PLANNER_CFG = "dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735"
_EVAL_FAMILY = "classic_cross_trap_medium"
_EXPECTED_MANIFEST_HASHES = {
    "c0": "manifest-c0",
    "cp0": "manifest-cp0",
    "cp1": "manifest-cp1",
    "cr0": "manifest-cr0",
    "shared": "manifest-shared",
    **{f"prop_cand_{index}": f"manifest-prop_cand_{index}" for index in range(64)},
    **{f"rand_cand_{index}": f"manifest-rand_cand_{index}" for index in range(64)},
}
_SPEC = AdmissionSpec(
    expected_target_planner_id=_PLANNER,
    expected_eval_family=_EVAL_FAMILY,
    confirmation_threshold="3_of_5",
    expected_target_planner_config_sha256=_PLANNER_CFG,
    expected_candidate_manifest_sha256_by_id=_EXPECTED_MANIFEST_HASHES,
)


def _replay() -> dict[str, Any]:
    """A passing deterministic-replay lineage block."""
    return {
        "exact_signature_match": True,
        "original_signature_sha256": "abc123",
        "replay_signature_sha256": "abc123",
    }


def _confirmation(*, confirmed: int = 3, attempts: int = 5) -> dict[str, Any]:
    """A passing 3-of-5 independent-seed confirmation block."""
    return {
        "confirmed_count": confirmed,
        "attempt_count": attempts,
        "stable_attribution": True,
    }


def _row(  # noqa: PLR0913
    *,
    row_id: str,
    manifest_id: str,
    arm: str,
    rank: int,
    failure: bool,
    scenario_seed: int = 99001,
    record_sha256: str = "rec-hash",
    admission_status: str = "admitted",
    exclusion_reason: str | None = None,
    execution_mode: str = "native",
    scenario_cert: str = "passed",
    candidate_cert: str = "passed",
    replay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one v2 outcome row with overridable admission-relevant fields."""
    return {
        "row_id": row_id,
        "candidate_manifest_id": manifest_id,
        "candidate_manifest_sha256": f"manifest-{manifest_id}",
        "selection_arm": arm,
        "selection_rank": rank,
        "candidate_pool_seed": 42,
        "candidate_pool_index": rank,
        "target_planner_id": _PLANNER,
        "target_planner_config_sha256": _PLANNER_CFG,
        "scenario_family": _EVAL_FAMILY,
        "scenario_seed": scenario_seed,
        "execution_commit": "ecf997d392a4f2c1a4fb5a56e8101acb030b7e2f",
        "execution_command": ["python", "-m", "robot_sf.run_eval"],
        "execution_config_lineage": {"config": "eval.yaml", "sha256": "cfg-hash"},
        "execution_mode": execution_mode,
        "termination_reason": "collision" if failure else "goal_reached",
        "independent_failure_outcome": failure,
        "scenario_certification_status": scenario_cert,
        "candidate_certification_status": candidate_cert,
        "replay_lineage": replay if replay is not None else _replay(),
        "confirmation_lineage": _confirmation(),
        "record_sha256": record_sha256,
        "admission_status": admission_status,
        "exclusion_reason": exclusion_reason,
    }


def _packet(rows: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    """Build a v2 packet wrapping ``rows``."""
    payload: dict[str, Any] = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "source": "unit-test-fixture",
        "artifact": "docs/context/evidence/unit-test.json",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "target_planner_id": _PLANNER,
        "target_planner_config_sha256": _PLANNER_CFG,
        "eval_archive_sha256": "eval-hash",
        "rows": rows,
    }
    payload.update(overrides)
    return payload


def _balanced_packet(*, proposal_failures: int, random_failures: int, per_arm: int = 4) -> dict:
    """Build a packet with ``per_arm`` candidates per arm and chosen failure counts."""
    rows: list[dict[str, Any]] = []
    for rank in range(per_arm):
        rows.append(
            _row(
                row_id=f"prop_{rank}",
                manifest_id=f"prop_cand_{rank}",
                arm="proposal",
                rank=rank + 1,
                failure=rank < proposal_failures,
            )
        )
    for rank in range(per_arm):
        rows.append(
            _row(
                row_id=f"rand_{rank}",
                manifest_id=f"rand_cand_{rank}",
                arm="random",
                rank=rank + 1,
                failure=rank < random_failures,
            )
        )
    return _packet(rows)


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


def test_v1_flat_array_packet_is_rejected_as_deprecated() -> None:
    """The deprecated flat-array v1 contract is no longer admitted."""
    legacy = {
        "schema_version": "adversarial_independent_outcomes.v1",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "proposal_outcomes": [10.0],
        "random_outcomes": [0.0],
    }
    result = build_independent_outcome_evaluation(
        legacy, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "deprecated" in result["reason"]
    assert result["independent_outcomes_available"] is False


def test_archive_nearness_objective_is_circular_and_blocked() -> None:
    """Archive-nearness objective cannot open the claim gate."""
    packet = _packet([], objective="archive_nearness")
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "circular" in result["reason"]


def test_target_planner_mismatch_blocks() -> None:
    """A packet for the wrong target planner fails closed."""
    packet = _packet([], target_planner_id="goal")
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "target_planner_id mismatch" in result["reason"]


def test_fallback_execution_mode_row_fails_closed() -> None:
    """A fallback execution row cannot be admitted (fail closed, not degraded)."""
    packet = _packet(
        [
            _row(
                row_id="r0",
                manifest_id="c0",
                arm="proposal",
                rank=1,
                failure=True,
                execution_mode="fallback",
            )
        ]
    )
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "execution_mode" in result["reason"]
    assert "native" in result["reason"]


def test_missing_required_field_fails_closed() -> None:
    """A row missing a required field fails closed with the field name."""
    row = _row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)
    del row["record_sha256"]
    packet = _packet([row])
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "record_sha256" in result["reason"]


def test_candidate_manifest_hash_requires_external_binding() -> None:
    """Rows cannot self-attest manifest lineage when no frozen binding is supplied."""
    packet = _packet([_row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)])
    unbound_spec = AdmissionSpec(
        expected_target_planner_id=_PLANNER,
        expected_eval_family=_EVAL_FAMILY,
        expected_target_planner_config_sha256=_PLANNER_CFG,
    )

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=unbound_spec
    )

    assert result["status"] == "blocked"
    assert "manifest_sha256 binding is unavailable" in result["reason"]


def test_candidate_manifest_hash_mismatch_fails_closed() -> None:
    """A row hash must match the separate frozen manifest-hash binding."""
    row = _row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)
    row["candidate_manifest_sha256"] = "wrong-manifest-hash"
    result = build_independent_outcome_evaluation(
        _packet([row]), budget_per_arm=1, minimally_important=0.20, admission_spec=_SPEC
    )

    assert result["status"] == "blocked"
    assert "candidate_manifest_sha256 mismatch" in result["reason"]


def test_replay_signature_mismatch_fails_closed() -> None:
    """A replay signature that differs from the original signature fails closed."""
    row = _row(
        row_id="r0",
        manifest_id="c0",
        arm="proposal",
        rank=1,
        failure=True,
        replay={
            "exact_signature_match": True,
            "original_signature_sha256": "abc",
            "replay_signature_sha256": "xyz",
        },
    )
    packet = _packet([row])
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "signature" in result["reason"]


def test_confirmation_below_threshold_fails_closed() -> None:
    """A candidate confirmed by only 2 of 5 seeds fails the frozen 3-of-5 threshold."""
    row = _row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)
    row["confirmation_lineage"] = _confirmation(confirmed=2, attempts=5)
    packet = _packet([row])
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "confirmation" in result["reason"]


def test_scenario_certification_not_passed_fails_closed() -> None:
    """A candidate whose scenario certification did not pass fails closed."""
    row = _row(
        row_id="r0",
        manifest_id="c0",
        arm="proposal",
        rank=1,
        failure=True,
        scenario_cert="stress_only",
    )
    packet = _packet([row])
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "scenario_certification_status" in result["reason"]


def test_wrong_eval_family_fails_closed() -> None:
    """A row from the fit family (not the held-out eval family) fails closed."""
    row = _row(
        row_id="r0",
        manifest_id="c0",
        arm="proposal",
        rank=1,
        failure=True,
    )
    row["scenario_family"] = "classic_group_crossing_medium"
    packet = _packet([row])
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "scenario_family" in result["reason"]


def test_excluded_row_with_reason_is_dropped_not_aggregated() -> None:
    """A predeclared excluded candidate (with a reason) is dropped, not aggregated."""
    rows = [
        _row(row_id="p0", manifest_id="cp0", arm="proposal", rank=1, failure=True),
        _row(
            row_id="p1",
            manifest_id="cp1",
            arm="proposal",
            rank=2,
            failure=False,
            admission_status="excluded",
            exclusion_reason="candidate_pool_collision_disjoint_by_candidate",
        ),
        _row(row_id="r0", manifest_id="cr0", arm="random", rank=1, failure=False),
    ]
    packet = _packet(rows)
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=2, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "complete"
    assert result["excluded_row_count"] == 1
    assert result["proposal"]["count"] == 1
    assert result["proposal"]["failures"] == 1


def test_unstable_attribution_across_seeds_fails_closed() -> None:
    """A candidate whose seeds disagree on the failure outcome fails closed."""
    rows = [
        _row(
            row_id="p0a", manifest_id="cp0", arm="proposal", rank=1, failure=True, scenario_seed=1
        ),
        _row(
            row_id="p0b", manifest_id="cp0", arm="proposal", rank=1, failure=False, scenario_seed=2
        ),
        _row(row_id="r0", manifest_id="cr0", arm="random", rank=1, failure=False),
    ]
    packet = _packet(rows)
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=2, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "blocked"
    assert "unstable attribution" in result["reason"]


def test_candidate_manifest_id_in_both_arms_fails_closed() -> None:
    """A candidate manifest cannot be aggregated into both disjoint arms."""
    packet = _packet(
        [
            _row(row_id="p0", manifest_id="shared", arm="proposal", rank=1, failure=True),
            _row(row_id="r0", manifest_id="shared", arm="random", rank=1, failure=False),
        ]
    )
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=_SPEC
    )

    assert result["status"] == "blocked"
    assert "both proposal and random arms" in result["reason"]


def test_complete_packet_decision_follows_execution() -> None:
    """A strong proposal-favors-execution packet yields a continue/underpowered decision."""
    # 4/4 proposal failures vs 0/4 random: large effect, but k=4 is underpowered
    # for delta=0.20 (min detectable at k=4 is 0.75), so the decision is inconclusive.
    packet = _balanced_packet(proposal_failures=4, random_failures=0, per_arm=4)
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=4, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "complete"
    assert result["independent_outcomes_available"] is True
    assert result["proposal_failure_yield"] == 1.0
    assert result["random_failure_yield"] == 0.0
    assert result["comparison"]["yield_improvement"] == 1.0
    assert result["comparison"]["null_rejected"] is True
    assert result["comparison"]["powered"] is False
    assert result["decision"]["status"] == "inconclusive"
    assert result["decision"]["reason"] == "underpowered_for_minimally_important_effect"


def test_underpowered_random_better_than_proposal_is_inconclusive() -> None:
    """An underpowered random-favoring result cannot produce a stop decision."""
    # 0/6 proposal vs 6/6 random is still underpowered for the frozen 0.20 effect.
    packet = _balanced_packet(proposal_failures=0, random_failures=6, per_arm=6)
    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=6, minimally_important=0.20, admission_spec=_SPEC
    )
    assert result["status"] == "complete"
    assert result["comparison"]["powered"] is False
    assert result["decision"]["status"] == "inconclusive"
    assert result["decision"]["reason"] == "underpowered_for_minimally_important_effect"


def test_opposite_sign_regressions_decision_follows_execution() -> None:
    """Decision follows execution independent of which arm it favors.

    The runner supplies archive-nearness only as a diagnostic; the decision is
    driven solely by the admitted execution rows. This test encodes the
    opposite-sign guarantee: the decision tracks execution, and the
    diagnostic archive-nearness namespace is never consulted here.
    """
    # k=30 is the smallest frozen-effect budget whose Fisher boundary is <= 0.20.
    # Case A: execution favors proposal.
    exec_proposal_better = _balanced_packet(proposal_failures=30, random_failures=0, per_arm=30)
    res_a = build_independent_outcome_evaluation(
        exec_proposal_better,
        budget_per_arm=30,
        minimally_important=0.20,
        admission_spec=_SPEC,
        n_permutations=10,
    )
    assert res_a["comparison"]["yield_improvement"] > 0.0

    # Case B: execution favors random (opposite sign).
    exec_random_better = _balanced_packet(proposal_failures=0, random_failures=30, per_arm=30)
    res_b = build_independent_outcome_evaluation(
        exec_random_better,
        budget_per_arm=30,
        minimally_important=0.20,
        admission_spec=_SPEC,
        n_permutations=10,
    )
    assert res_b["comparison"]["yield_improvement"] < 0.0
    assert res_b["decision"]["status"] == "stop"
    assert res_a["decision"]["status"] == "continue"
    assert res_a["comparison"]["yield_improvement"] * res_b["comparison"]["yield_improvement"] < 0.0


def test_eval_archive_hash_mismatch_fails_closed() -> None:
    """Outcome packets must match the held-out eval split they claim to score."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=2)
    result = build_independent_outcome_evaluation(
        packet,
        budget_per_arm=2,
        minimally_important=0.20,
        admission_spec=_SPEC,
        expected_eval_archive_sha256="expected-eval-hash",
    )
    assert result["status"] == "blocked"
    assert result["expected_eval_archive_sha256"] == "expected-eval-hash"
    assert result["observed_eval_archive_sha256"] == "eval-hash"


def test_load_independent_outcomes_reads_json_payload(tmp_path: Path) -> None:
    """Readable JSON objects are passed through for report integration."""
    path = tmp_path / "outcomes.json"
    path.write_text(
        json.dumps(_balanced_packet(proposal_failures=1, random_failures=0)), encoding="utf-8"
    )

    state, reason, payload = load_independent_outcomes(path)

    assert state == "active"
    assert "loaded successfully" in reason
    assert payload is not None
    assert payload["outcome_source"] == "planner_execution"
    assert payload_sha256(payload)
