"""Tests for the conservative scenario-admissibility adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial import (
    ADMISSIBLE_FEASIBILITY_UNKNOWN,
    EMPIRICALLY_FEASIBLE,
    GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY,
    PLANNER_SPECIFIC_FAILURE,
    STRUCTURALLY_INVALID,
    classify_scenario_admissibility,
    partition_candidates_by_admissibility,
    validate_scenario_admissibility,
)
from robot_sf.adversarial.feasibility_first import (
    SCENARIO_FEASIBILITY_CONTRACT_VERSION,
    SCENARIO_FEASIBILITY_PREDICATE_NAMES,
)
from robot_sf.scenario_certification.feasibility_oracle import (
    FEASIBILITY_ORACLE_SCHEMA,
    ISSUE_5574_REPORT_SCHEMA,
)
from robot_sf.scenario_certification.v1 import CERT_SCHEMA_VERSION


def _certificate(
    classification: str = "valid",
    *,
    eligibility: str = "eligible",
    scenario_id: str = "case-static",
    pedestrian_count: int = 0,
    route_reason: str | None = None,
) -> dict[str, Any]:
    reasons = (
        [route_reason]
        if route_reason is not None
        else (["route_requires_at_least_two_waypoints"] if classification == "invalid" else [])
    )
    return {
        "schema_version": CERT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "source": "test-fixture",
        "classification": classification,
        "benchmark_eligibility": eligibility,
        "reasons": reasons,
        "checks": {"route_count": 1, "settings": {"robot_radius_m": 0.4}},
        "route_certificates": [
            {
                "route_id": "route-0",
                "spawn_id": 0,
                "goal_id": 0,
                "classification": classification,
                "benchmark_eligibility": eligibility,
                "reasons": reasons,
                "checks": {"dynamic": {"single_pedestrian_count": pedestrian_count}},
                "evidence": {},
            }
        ],
        "evidence": {},
    }


def _predicates(case_id: str = "case-static", *, verdict: str = "valid") -> dict[str, Any]:
    return {
        "contract_version": SCENARIO_FEASIBILITY_CONTRACT_VERSION,
        "candidate_id": case_id,
        "predicates": [
            {
                "name": name,
                "verdict": verdict,
                "reason": "fixture evidence",
                "evidence": {"source": "fixture"} if verdict == "valid" else {},
            }
            for name in SCENARIO_FEASIBILITY_PREDICATE_NAMES
        ],
    }


def _oracle(
    *,
    scenario_id: str = "case-static",
    status: str = "feasible",
    geometric: bool = True,
    complete: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": FEASIBILITY_ORACLE_SCHEMA,
        "scenario_id": scenario_id,
        "envelope_radius_m": 0.4,
        "feasible": status == "feasible",
        "status": status,
        "claim_boundary": "diagnostic_only_not_benchmark_evidence",
        "geometric": {
            "route_geometrically_feasible": geometric,
            "classification": "hard_but_solvable" if geometric else "geometrically_infeasible",
            "benchmark_eligibility": "eligible" if geometric else "excluded",
        },
        "completion": {
            "route_completion_feasible": complete,
            "status": "passed" if complete else "failed",
        },
    }


def _oracle_report(oracle: dict[str, Any], *, scenario_id: str = "case-static") -> dict[str, Any]:
    return {
        "schema_version": ISSUE_5574_REPORT_SCHEMA,
        "scenario_ids": [scenario_id],
        "scenario_manifest": "configs/fixture.yaml",
        "rollout_algo": "goal",
        "cells": [
            {
                "schema_version": "envelope_sensitivity_axis.v1",
                "scenario_id": scenario_id,
                "category": oracle["status"],
                "nominal_envelope_radius_m": oracle["envelope_radius_m"],
                "nominal_verdict": oracle,
                "reduced_verdicts": [],
                "scenario_manifest": "configs/fixture.yaml",
                "rollout_algo": "goal",
                "rollout_seed": 19,
                "claim_boundary": "diagnostic_only_not_benchmark_evidence",
            }
        ],
    }


def _execution(
    planner_id: str,
    *,
    route_complete: bool,
    case_id: str = "case-static",
    scenario_id: str = "case-static",
    seed: int = 19,
    replay: bool = False,
) -> dict[str, Any]:
    record = {
        "case_id": case_id,
        "scenario_id": scenario_id,
        "scenario_variant": "original",
        "planner_id": planner_id,
        "run_status": "ok",
        "route_complete": route_complete,
        "seed": seed,
        "horizon_steps": 100,
        "scenario_sha256": "a" * 64,
        "robot_model_sha256": "b" * 64,
        "simulator_config_sha256": "c" * 64,
        "planner_config_sha256": "d" * 64 if planner_id == "target" else "c" * 64,
        "planner_checkpoint_sha256": (
            "not_applicable" if planner_id in {"goal", "social_force", "orca"} else "d" * 64
        ),
        "environment_sha256": "e" * 64,
        "source_commit": "f" * 40,
        "evidence_ref": f"artifacts/{planner_id}.json",
    }
    if planner_id == "replay" or replay:
        record.update(determinism_check_status="pass", resimulated=True)
    return record


def test_certificate_rejects_only_structural_and_geometric_exclusions() -> None:
    structural = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate("invalid", eligibility="excluded")
    )
    impossible = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("geometrically_infeasible", eligibility="excluded"),
    )
    kinematic = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("kinodynamically_infeasible", eligibility="excluded"),
    )
    invalid_geometry = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid", eligibility="excluded", route_reason="start_outside_map_bounds"
        ),
    )

    assert structural.verdict == STRUCTURALLY_INVALID
    assert structural.search_disposition == "reject"
    assert impossible.verdict == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    assert impossible.search_disposition == "reject"
    assert kinematic.verdict == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    assert invalid_geometry.verdict == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    assert impossible.assumptions["scenario_certificate"]["settings"] == {"robot_radius_m": 0.4}


def test_valid_certificate_and_valid_predicates_do_not_establish_feasibility() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        predicate_contract=_predicates(),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "valid_predicates_do_not_prove_solvability" in verdict.reason_codes


def test_certificate_conflicts_and_dynamic_overconstraint_remain_unknown() -> None:
    conflict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("valid", eligibility="excluded"),
        reference_execution=_execution("reference", route_complete=True),
    )
    dynamic = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("dynamically_overconstrained", eligibility="excluded"),
    )

    assert conflict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "conflicting_feasibility_evidence" in conflict.reason_codes
    assert dynamic.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert dynamic.search_disposition == "retain"


def test_unverifiable_invalid_certificate_remains_unknown() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid",
            eligibility="excluded",
            route_reason="planned_path_swept_envelope_unverifiable: geometry unavailable",
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes


def test_top_level_impossibility_does_not_reject_a_case_with_another_usable_route() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["checks"]["route_count"] = 2
    certificate["route_certificates"].append(
        {
            "route_id": "route-1",
            "spawn_id": 1,
            "goal_id": 0,
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": [],
            "checks": {"dynamic": {"single_pedestrian_count": 0}},
            "evidence": {},
        }
    )
    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_route_coverage_unresolved" in verdict.reason_codes


def test_declared_route_count_must_match_before_certificate_can_reject() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["checks"]["route_count"] = 2
    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert verdict.assumptions["scenario_certificate"]["route_inventory"] == {
        "declared_count": 2,
        "observed_count": 1,
        "complete": False,
    }
    assert "scenario_certificate_route_coverage_unresolved" in verdict.reason_codes


def test_oracle_exclusion_does_not_override_unresolved_mixed_route_certificate() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["checks"]["route_count"] = 2
    certificate["route_certificates"].append(
        {
            "route_id": "route-1",
            "spawn_id": 1,
            "goal_id": 0,
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": [],
            "checks": {"dynamic": {"single_pedestrian_count": 0}},
            "evidence": {},
        }
    )
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=certificate,
        feasibility_evidence=_oracle(
            status="infeasible_by_construction", geometric=False, complete=False
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in verdict.reason_codes


def test_oracle_exclusion_does_not_override_unresolved_invalid_certificate() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid",
            eligibility="excluded",
            route_reason="unrecognized invalidity reason",
        ),
        feasibility_evidence=_oracle(
            status="infeasible_by_construction", geometric=False, complete=False
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in verdict.reason_codes


def test_oracle_exclusion_requires_a_complete_matching_certificate() -> None:
    excluded = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    no_certificate = classify_scenario_admissibility("case-static", feasibility_evidence=excluded)
    incomplete = _certificate("geometrically_infeasible", eligibility="excluded")
    incomplete["checks"]["route_count"] = 2
    incomplete_certificate = classify_scenario_admissibility(
        "case-static", scenario_certificate=incomplete, feasibility_evidence=excluded
    )
    conflicting_positive = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        feasibility_evidence=excluded,
    )

    for verdict in (no_certificate, incomplete_certificate, conflicting_positive):
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
        assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in no_certificate.reason_codes
    assert (
        "oracle_geometric_exclusion_route_coverage_unresolved"
        in incomplete_certificate.reason_codes
    )
    assert "oracle_geometric_exclusion_certificate_conflict" in conflicting_positive.reason_codes


def test_incomplete_route_inventory_cannot_bind_oracle_success() -> None:
    certificate = _certificate()
    certificate["checks"]["route_count"] = 2
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=certificate,
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_success_not_bound_to_static_case_and_provenance" in verdict.reason_codes


@pytest.mark.parametrize("status", ["blocked", "time_truncated"])
def test_blocked_or_truncated_oracle_results_are_unknown(status: str) -> None:
    oracle = _oracle(status=status, complete=False)
    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=oracle
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"oracle_{status}_does_not_prove_impossibility" in verdict.reason_codes


def test_oracle_geometry_exclusion_without_certificate_remains_unknown() -> None:
    excluded = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    verdict = classify_scenario_admissibility("case-static", feasibility_evidence=excluded)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in verdict.reason_codes


def test_oracle_geometry_exclusion_is_named_with_complete_certificate() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    excluded = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=certificate, feasibility_evidence=excluded
    )

    assert verdict.verdict == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    assert verdict.search_disposition == "reject"
    assert verdict.assumptions["feasibility_oracle"]["envelope_radius_m"] == 0.4


def test_oracle_without_canonical_claim_boundary_remains_unknown() -> None:
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle.pop("claim_boundary")
    verdict = classify_scenario_admissibility("case-static", feasibility_evidence=oracle)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "feasibility_oracle_claim_boundary_missing_or_unsupported" in verdict.reason_codes


def test_committed_issue_5574_report_maps_excluded_and_unresolved_cells() -> None:
    report_path = Path(
        "docs/context/evidence/issue_5574_feasibility_oracle_2026-07-14/verdicts.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    excluded = classify_scenario_admissibility(
        "francis2023_narrow_doorway",
        scenario_id="francis2023_narrow_doorway",
        feasibility_evidence=report,
    )
    unresolved = classify_scenario_admissibility(
        "francis2023_blind_corner",
        scenario_id="francis2023_blind_corner",
        feasibility_evidence=report,
    )

    assert excluded.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert excluded.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in excluded.reason_codes
    assert excluded.assumptions["feasibility_oracle"]["envelope_radius_m"] == 1.0
    assert unresolved.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert unresolved.search_disposition == "retain"


def test_actor_free_oracle_success_requires_a_static_named_report() -> None:
    report = _oracle_report(_oracle())
    static = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=report
    )
    dynamic = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(pedestrian_count=1),
        feasibility_evidence=report,
    )
    unbound = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=_oracle()
    )

    assert static.verdict == EMPIRICALLY_FEASIBLE
    assert static.assumptions["feasibility_oracle"]["empirical_scope"] == (
        "named_actor_free_rollout_of_original_static_case"
    )
    assert dynamic.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert unbound.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN


def test_actor_free_oracle_success_does_not_resolve_unknown_invalid_certificate() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid",
            eligibility="excluded",
            route_reason="unrecognized invalidity reason",
        ),
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes


def test_actor_free_oracle_uses_explicit_scenario_binding_for_stable_case_id() -> None:
    verdict = classify_scenario_admissibility(
        "counterexample-0001",
        scenario_id="case-static",
        scenario_certificate=_certificate(),
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert verdict.case_id == "counterexample-0001"
    assert verdict.scenario_id == "case-static"


def test_malformed_certificate_cannot_bind_oracle_success() -> None:
    malformed = _certificate()
    malformed.pop("schema_version")
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=malformed,
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "scenario_certificate_missing_or_malformed" in verdict.reason_codes


def test_certificate_identity_mismatch_cannot_reject_a_candidate() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_id="case-static",
        scenario_certificate=_certificate("invalid", eligibility="excluded", scenario_id="other"),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_identity_mismatch" in verdict.reason_codes


def test_reference_success_is_empirical_but_target_failure_alone_is_unknown() -> None:
    reference = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        reference_execution=_execution("reference", route_complete=True),
    )
    target_only = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        target_execution=_execution("target", route_complete=False),
    )

    assert reference.verdict == EMPIRICALLY_FEASIBLE
    assert target_only.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert target_only.target_planner_outcome == "route_incomplete"
    assert "target_failure_alone_does_not_prove_infeasibility" in target_only.reason_codes


def test_matched_reference_target_failure_needs_reproducing_replay() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=_execution("target", route_complete=False, replay=True),
    )
    absent_replay = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
    )
    replay_success = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=_execution("target", route_complete=True, replay=True),
    )
    wrong_planner = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=_execution("other", route_complete=False, replay=True),
    )
    unmatched = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False, seed=20),
    )

    assert verdict.verdict == PLANNER_SPECIFIC_FAILURE
    assert verdict.search_disposition == "retain"
    assert "matched_reference_target_failure_reproduced_by_replay" in verdict.reason_codes
    assert absent_replay.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "planner_specific_failure_replay_missing_or_invalid" in absent_replay.reason_codes
    assert replay_success.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert (
        "planner_specific_failure_replay_did_not_reproduce_failure" in replay_success.reason_codes
    )
    assert wrong_planner.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "planner_specific_failure_replay_wrong_planner" in wrong_planner.reason_codes
    assert unmatched.verdict == EMPIRICALLY_FEASIBLE
    assert "matched_reference_success_target_planner_failure" not in unmatched.reason_codes


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("case_id", "other-case", "replay_execution_identity_mismatch"),
        ("scenario_id", "other-scenario", "planner_specific_failure_replay_case_mismatch"),
        ("scenario_sha256", "f" * 64, "planner_specific_failure_replay_case_mismatch"),
        ("robot_model_sha256", "f" * 64, "planner_specific_failure_replay_case_mismatch"),
        ("simulator_config_sha256", "f" * 64, "planner_specific_failure_replay_case_mismatch"),
        ("environment_sha256", "f" * 64, "planner_specific_failure_replay_case_mismatch"),
        ("source_commit", "e" * 40, "planner_specific_failure_replay_case_mismatch"),
        ("seed", 20, "planner_specific_failure_replay_case_mismatch"),
        ("horizon_steps", 101, "planner_specific_failure_replay_case_mismatch"),
        (
            "planner_config_sha256",
            "f" * 64,
            "planner_specific_failure_replay_configuration_mismatch",
        ),
        (
            "planner_checkpoint_sha256",
            "f" * 64,
            "planner_specific_failure_replay_configuration_mismatch",
        ),
    ],
)
def test_replay_must_match_target_case_and_execution_bindings(
    field: str, value: Any, reason: str
) -> None:
    replay = _execution("target", route_complete=False, replay=True)
    replay[field] = value
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=replay,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert reason in verdict.reason_codes


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"determinism_check_status": "fail"}, "replay_determinism_check_not_passed"),
        ({"resimulated": False}, "replay_did_not_resimulate_source_episode"),
    ],
)
def test_replay_success_requires_resimulation_and_determinism(
    change: dict[str, Any], reason: str
) -> None:
    replay = _execution("replay", route_complete=True)
    replay.update(change)
    verdict = classify_scenario_admissibility("case-static", replay_execution=replay)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert reason in verdict.reason_codes


def test_replay_completion_is_empirical_only_after_validated_resimulation() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        replay_execution=_execution("replay", route_complete=True),
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert verdict.evidence["replay_execution"]["resimulated"] is True


def test_execution_with_invalid_budget_is_not_used_as_feasibility_evidence() -> None:
    run = _execution("reference", route_complete=True, seed=-1)
    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_outcome_or_budget_invalid" in verdict.reason_codes


def test_execution_with_invalid_digest_is_not_used_as_feasibility_evidence() -> None:
    run = _execution("reference", route_complete=True)
    run["environment_sha256"] = "fixture-hash"
    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_provenance_incomplete" in verdict.reason_codes


def test_not_applicable_checkpoint_provenance_is_limited_to_classical_planners() -> None:
    classical = classify_scenario_admissibility(
        "case-static", reference_execution=_execution("goal", route_complete=True)
    )
    learned = _execution("ppo", route_complete=True)
    learned["planner_checkpoint_sha256"] = "not_applicable"
    rejected = classify_scenario_admissibility("case-static", reference_execution=learned)

    assert classical.verdict == EMPIRICALLY_FEASIBLE
    assert rejected.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_provenance_incomplete" in rejected.reason_codes


def test_not_applicable_checkpoint_sentinel_rejects_checkpoint_backed_sicnav() -> None:
    sicnav = _execution("sicnav", route_complete=True)
    sicnav["planner_checkpoint_sha256"] = "not_applicable"
    verdict = classify_scenario_admissibility("case-static", reference_execution=sicnav)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "reference_execution_provenance_incomplete" in verdict.reason_codes


@pytest.mark.parametrize("field", ["planner_config_sha256", "planner_checkpoint_sha256"])
def test_execution_with_missing_planner_provenance_stays_unknown(field: str) -> None:
    run = _execution("reference", route_complete=True)
    run.pop(field)
    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_provenance_incomplete" in verdict.reason_codes


def test_partition_rejects_only_explicit_exclusions_and_preserves_unknown() -> None:
    verdicts = [
        classify_scenario_admissibility(
            "invalid", scenario_certificate=_certificate("invalid", eligibility="excluded")
        ),
        classify_scenario_admissibility("unknown", scenario_certificate=_certificate()),
        classify_scenario_admissibility(
            "failure",
            scenario_certificate=_certificate(),
            reference_execution=_execution("reference", route_complete=True, case_id="failure"),
            target_execution=_execution("target", route_complete=False, case_id="failure"),
            replay_execution=_execution(
                "target", route_complete=False, case_id="failure", replay=True
            ),
        ),
    ]
    partition = partition_candidates_by_admissibility(verdicts)

    assert partition.retained_case_ids == ("unknown", "failure")
    assert partition.rejected[0]["case_id"] == "invalid"
    assert partition.by_verdict[ADMISSIBLE_FEASIBILITY_UNKNOWN] == ("unknown",)
    assert partition.by_verdict[PLANNER_SPECIFIC_FAILURE] == ("failure",)
    with pytest.raises(ValueError, match="unique"):
        partition_candidates_by_admissibility([verdicts[1], verdicts[1]])


def test_serialized_verdict_matches_contract_schema() -> None:
    payload = classify_scenario_admissibility("case-static").to_dict()
    validate_scenario_admissibility(payload)

    assert payload["schema_version"] == "scenario_admissibility.v1"
    assert payload["verdict"] == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert payload["target_planner_outcome"] == "not_evaluated"
    schema_path = Path("robot_sf/benchmark/schemas/scenario_admissibility.v1.json")
    json.loads(schema_path.read_text(encoding="utf-8"))
