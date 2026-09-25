"""Tests for the conservative scenario-admissibility adapter."""

# evidence-writer-exempt: synthetic scenario YAML is written only under pytest tmp_path to prove
# candidate artifact identity binding; no canonical evidence is published.

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import (
    ADMISSIBLE_FEASIBILITY_UNKNOWN,
    EMPIRICALLY_FEASIBLE,
    GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY,
    PLANNER_SPECIFIC_FAILURE,
    STRUCTURALLY_INVALID,
    partition_candidates_by_admissibility,
    validate_scenario_admissibility,
)
from robot_sf.adversarial import (
    classify_scenario_admissibility as _classify_scenario_admissibility,
)
from robot_sf.adversarial.feasibility_first import (
    SCENARIO_FEASIBILITY_CONTRACT_VERSION,
    SCENARIO_FEASIBILITY_PREDICATE_NAMES,
)
from robot_sf.adversarial.scenario_admissibility import _oracle_excludes
from robot_sf.scenario_certification import v1 as scenario_certification_v1
from robot_sf.scenario_certification.feasibility_oracle import (
    FEASIBILITY_ORACLE_SCHEMA,
    ISSUE_5574_REPORT_SCHEMA,
)
from robot_sf.scenario_certification.input_identity import (
    runtime_input_records_match,
    scenario_input_identity,
)
from robot_sf.scenario_certification.v1 import (
    CERT_SCHEMA_VERSION,
    ScenarioCertificate,
    certificate_to_dict,
    certify_scenario_file,
)

_SCENARIO_ARTIFACT = Path(__file__).resolve().parent / "fixtures/issue_9651/case_static.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_ARTIFACT_SHA256 = hashlib.sha256(_SCENARIO_ARTIFACT.read_bytes()).hexdigest()
_SCENARIO_EFFECTIVE_INPUT_SHA256 = scenario_input_identity(
    _SCENARIO_ARTIFACT, scenario_id="case-static"
).get("effective_input_sha256")


def classify_scenario_admissibility(case_id: str, **kwargs: Any) -> Any:
    """Use one explicit artifact and scenario identity for normalized test evidence."""
    kwargs.setdefault("scenario_artifact_path", _SCENARIO_ARTIFACT)
    kwargs.setdefault("scenario_id", "case-static")
    return _classify_scenario_admissibility(case_id, **kwargs)


def _certificate(
    classification: str = "valid",
    *,
    eligibility: str = "eligible",
    scenario_id: str = "case-static",
    pedestrian_count: int = 0,
    route_reason: str | None = None,
) -> dict[str, Any]:
    if route_reason is not None:
        reasons = [route_reason]
    elif classification == "invalid":
        reasons = ["route_requires_at_least_two_waypoints"]
    elif classification == "geometrically_infeasible":
        reasons = ["no_inflated_collision_free_path: empty_path"]
    elif classification == "kinodynamically_infeasible":
        reasons = ["route_turn_radius_below_bicycle_limit: 1.000 < 2.000"]
    else:
        reasons = []
    route_checks: dict[str, Any] = {"dynamic": {"single_pedestrian_count": pedestrian_count}}
    if classification == "invalid":
        route_checks.update(start=None, goal=None, waypoint_count=1)
    if classification == "geometrically_infeasible":
        route_checks["inflated_collision_free_path"] = False
        route_checks["planner"] = {"path_status": "no_path"}
    if classification == "kinodynamically_infeasible":
        route_checks["kinodynamic"] = {
            "robot_model": "BicycleDriveSettings",
            "command_limits_valid": True,
            "minimum_turning_radius_m": 2.0,
            "route_minimum_turn_radius_m": 1.0,
        }
    return {
        "schema_version": CERT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "source": _SCENARIO_ARTIFACT.as_posix(),
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
                "checks": route_checks,
                "evidence": {},
            }
        ],
        "evidence": {
            "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
            "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
            "effective_input_identity_stable": True,
            "runtime_input_identity_stable": True,
        },
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
        "scenario_manifest": _SCENARIO_ARTIFACT.as_posix(),
        "envelope_radius_m": 0.4,
        "feasible": status == "feasible",
        "status": status,
        "claim_boundary": "diagnostic_only_not_benchmark_evidence",
        "geometric": {
            "route_geometrically_feasible": geometric,
            "classification": "hard_but_solvable" if geometric else "geometrically_infeasible",
            "benchmark_eligibility": "eligible" if geometric else "excluded",
            "runtime_input_identity_stable": True,
        },
        "completion": {
            "route_completion_feasible": (
                True if complete else None if status == "blocked" else False
            ),
            "min_completion_steps": 20 if complete else None,
            "horizon_steps": 100,
            "completion_horizon_margin_steps": 80 if complete else None,
            "termination_reason": "success" if complete else None,
            "status": ("passed" if complete else "blocked" if status == "blocked" else "failed"),
            "blocker": (
                None
                if complete
                else "rollout_unavailable"
                if status == "blocked"
                else "route_geometrically_infeasible_no_traversal_path"
                if status == "infeasible_by_construction"
                else "rollout_incomplete"
            ),
            "fallback_or_degraded": False,
            "runtime_input_identity_stable": True,
            "fallback_marker": None,
            "observed_route_completion_feasible": True if complete else None,
            "rollout_blocker": None,
        },
        "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
        "source_artifact_identity_stable": True,
        "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
        "effective_input_identity_stable": True,
        "runtime_input_identity_stable": True,
    }


def _oracle_report(oracle: dict[str, Any], *, scenario_id: str = "case-static") -> dict[str, Any]:
    return {
        "schema_version": ISSUE_5574_REPORT_SCHEMA,
        "scenario_ids": [scenario_id],
        "scenario_manifest": _SCENARIO_ARTIFACT.as_posix(),
        "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
        "source_artifact_identity_stable": True,
        "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
        "effective_input_identity_stable": True,
        "rollout_algo": "goal",
        "cells": [
            {
                "schema_version": "envelope_sensitivity_axis.v1",
                "scenario_id": scenario_id,
                "category": oracle["status"],
                "nominal_envelope_radius_m": oracle["envelope_radius_m"],
                "nominal_verdict": oracle,
                "reduced_verdicts": [],
                "scenario_manifest": _SCENARIO_ARTIFACT.as_posix(),
                "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
                "source_artifact_identity_stable": True,
                "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
                "effective_input_identity_stable": True,
                "runtime_input_identity_stable": True,
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
        "episode_id": "episode-target",
        "run_status": "ok",
        "fallback_or_degraded": False,
        "route_complete": route_complete,
        "seed": seed,
        "horizon_steps": 100,
        "scenario_sha256": _SCENARIO_ARTIFACT_SHA256,
        "robot_model_sha256": "b" * 64,
        "simulator_config_sha256": "c" * 64,
        "planner_config_sha256": "d" * 64 if planner_id == "target" else "c" * 64,
        "planner_checkpoint_sha256": (
            "not_applicable" if planner_id in {"goal", "social_force", "orca"} else "d" * 64
        ),
        "environment_sha256": "e" * 64,
        "source_episodes_jsonl_sha256": "a" * 64,
        "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
        "effective_input_identity_stable": True,
        "source_commit": "f" * 40,
        "evidence_ref": f"artifacts/{planner_id}.json",
    }
    if planner_id == "replay" or replay:
        record.update(determinism_check_status="pass", resimulated=True)
    return record


def _referenced_scenario(tmp_path: Path, *, route_override: bool = True) -> Path:
    """Materialize a canonical scenario with separately hashed runtime resources."""
    scenario_path = tmp_path / "scenario.yaml"
    map_path = tmp_path / "map.svg"
    shutil.copyfile(_REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg", map_path)
    scenario: dict[str, Any] = {
        "name": "case-static",
        "map_file": map_path.name,
        "simulation_config": {"max_episode_steps": 200, "ped_density": 0.0},
        "robot_config": {},
        "metadata": {"archetype": "head_on_corridor"},
        "seeds": [19],
    }
    if route_override:
        scenario["route_overrides_file"] = "routes.yaml"
        (tmp_path / "routes.yaml").write_text("routes: []\n", encoding="utf-8")
    scenario_path.write_text(
        yaml.safe_dump({"scenarios": [scenario]}, sort_keys=False), encoding="utf-8"
    )
    return scenario_path


def _bind_certificate_to_scenario(path: Path) -> dict[str, Any]:
    """Attach raw and runtime-input identities to a schema-valid fixture certificate."""
    identity = scenario_input_identity(path, scenario_id="case-static")
    assert identity["status"] == "available"
    certificate = _certificate()
    certificate["source"] = path.as_posix()
    certificate["evidence"].update(
        source_artifact_sha256=identity["source_artifact_sha256"],
        effective_input_sha256=identity["effective_input_sha256"],
        effective_input_identity_stable=True,
    )
    return certificate


def _bind_oracle_to_scenario(path: Path) -> dict[str, Any]:
    """Attach raw and runtime-input identities to a fixture oracle report."""
    identity = scenario_input_identity(path, scenario_id="case-static")
    assert identity["status"] == "available"
    oracle = _oracle()
    oracle.update(
        scenario_manifest=path.as_posix(),
        source_artifact_sha256=identity["source_artifact_sha256"],
        effective_input_sha256=identity["effective_input_sha256"],
        effective_input_identity_stable=True,
    )
    return oracle


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
    assert invalid_geometry.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "scenario_certificate_invalidity_unresolved" in invalid_geometry.reason_codes
    assert impossible.assumptions["scenario_certificate"]["settings"] == {"robot_radius_m": 0.4}


def test_planner_exception_stays_unknown_and_retained() -> None:
    """A planner exception must not masquerade as a completed no-path result."""

    certificate = _certificate("unknown", eligibility="stress_only")
    reason = "inflated_path_planner_error: injected planner failure"
    certificate["reasons"] = [reason]
    route = certificate["route_certificates"][0]
    route["reasons"] = [reason]
    route["checks"].update(inflated_collision_free_path=None, planner={"path_status": "error"})

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


@pytest.mark.parametrize(
    ("reason", "path_status"),
    [
        ("no_inflated_collision_free_path: planner error", "no_path"),
        ("no_inflated_collision_free_path: empty_path", "error"),
    ],
)
def test_geometric_exclusion_requires_completed_no_path_evidence(
    reason: str, path_status: str
) -> None:
    """A reason label and a contradictory planner status cannot exclude a candidate."""

    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    route = certificate["route_certificates"][0]
    route["reasons"] = [reason]
    route["checks"]["planner"]["path_status"] = path_status
    certificate["reasons"] = [reason]

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


@pytest.mark.parametrize(
    "mutation", ["contradictory_waypoint_count", "wrong_endpoint", "top_reason"]
)
def test_structural_certificate_label_needs_matching_producer_check(mutation: str) -> None:
    certificate = _certificate("invalid", eligibility="excluded")
    route = certificate["route_certificates"][0]
    if mutation == "contradictory_waypoint_count":
        route["checks"].update(waypoint_count=2, start=[0.0, 0.0], goal=[1.0, 1.0])
    elif mutation == "wrong_endpoint":
        route["checks"].update(waypoint_count=2, start=[0.0, 0.0], goal=None)
    else:
        certificate["reasons"] = []

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes


def test_producer_shaped_nonfinite_endpoint_supports_structural_exclusion() -> None:
    certificate = _certificate(
        "invalid", eligibility="excluded", route_reason="start_point_not_finite"
    )
    certificate["route_certificates"][0]["checks"].update(
        waypoint_count=2, start=[None, 1.0], goal=[4.0, 5.0]
    )

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == STRUCTURALLY_INVALID
    assert verdict.search_disposition == "reject"


@pytest.mark.parametrize("reason", ["map_pool_empty", "no_applicable_robot_routes"])
def test_empty_route_certificate_requires_exact_supported_reason(reason: str) -> None:
    certificate = _certificate("invalid", eligibility="excluded")
    certificate.update(reasons=[reason], checks={"route_count": 0}, route_certificates=[])

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == STRUCTURALLY_INVALID
    assert verdict.search_disposition == "reject"


@pytest.mark.parametrize(
    "mutation",
    ["label_only", "missing_check", "contradictory_check", "missing_summary_reason"],
)
def test_geometric_certificate_label_needs_matching_producer_evidence(mutation: str) -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    route = certificate["route_certificates"][0]
    if mutation == "label_only":
        certificate["reasons"] = []
        route["reasons"] = []
        route["checks"].pop("inflated_collision_free_path")
    elif mutation == "missing_check":
        route["checks"].pop("inflated_collision_free_path")
    elif mutation == "contradictory_check":
        route["checks"]["inflated_collision_free_path"] = True
    else:
        certificate["reasons"] = []

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


def test_supported_geometric_collision_check_patterns_can_exclude() -> None:
    swept = _certificate("geometrically_infeasible", eligibility="excluded")
    swept_reason = "planned_path_swept_envelope_clips_obstacle: full_polyline_clearance_m=-0.25"
    swept["reasons"] = [swept_reason]
    swept_route = swept["route_certificates"][0]
    swept_route["reasons"] = [swept_reason]
    swept_route["checks"].update(
        {
            "inflated_collision_free_path": False,
            "swept_envelope": {
                "validated": True,
                "clips_obstacle": True,
                "clearance_m": -0.25,
                "vertex_clearance_m": -0.25,
                "clipped_vertex_count": 1,
                "planned_waypoint_count": 4,
            },
        }
    )

    simulator = _certificate("geometrically_infeasible", eligibility="excluded")
    simulator_reason = "planned_path_simulator_collision: first_collision_sample_index=3"
    simulator["reasons"] = [simulator_reason]
    simulator_route = simulator["route_certificates"][0]
    simulator_route["reasons"] = [simulator_reason]
    simulator_route["checks"].update(
        {
            "inflated_collision_free_path": False,
            "simulator_obstacle_collision": {
                "validated": True,
                "collides_obstacle": True,
                "runtime_component": "ContinuousOccupancy.is_obstacle_collision",
                "obstacle_source": "MapDefinition.obstacles_pysf_runtime_normalized",
                "sample_spacing_m": 0.05,
                "checked_sample_count": 4,
                "first_collision_sample_index": 3,
            },
        }
    )

    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=swept).verdict
        == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    )
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=simulator).verdict
        == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    )

    swept["route_certificates"][0]["checks"]["swept_envelope"]["clearance_m"] = -0.5
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=swept).verdict
        == ADMISSIBLE_FEASIBILITY_UNKNOWN
    )
    simulator["route_certificates"][0]["checks"]["simulator_obstacle_collision"][
        "first_collision_sample_index"
    ] = 2
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=simulator).verdict
        == ADMISSIBLE_FEASIBILITY_UNKNOWN
    )


def test_kinodynamic_certificate_label_needs_matching_bicycle_check_evidence() -> None:
    certificate = _certificate("kinodynamically_infeasible", eligibility="excluded")
    route = certificate["route_certificates"][0]
    certificate["reasons"] = []
    route["reasons"] = []
    route["checks"].pop("kinodynamic")

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"

    inconsistent = _certificate("kinodynamically_infeasible", eligibility="excluded")
    inconsistent["route_certificates"][0]["checks"]["kinodynamic"][
        "route_minimum_turn_radius_m"
    ] = 3.0
    unresolved = classify_scenario_admissibility("case-static", scenario_certificate=inconsistent)
    assert unresolved.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN


def test_supported_steering_limit_evidence_excludes_but_unknown_robot_type_does_not() -> None:
    steering = _certificate("kinodynamically_infeasible", eligibility="excluded")
    steering_reason = "bicycle_max_steer_non_positive"
    steering["reasons"] = [steering_reason]
    steering_route = steering["route_certificates"][0]
    steering_route["reasons"] = [steering_reason]
    steering_route["checks"]["kinodynamic"] = {
        "robot_model": "BicycleDriveSettings",
        "command_limits_valid": False,
    }

    unsupported = _certificate("kinodynamically_infeasible", eligibility="excluded")
    unsupported_reason = "unsupported_robot_config: CustomDriveSettings"
    unsupported["reasons"] = [unsupported_reason]
    unsupported_route = unsupported["route_certificates"][0]
    unsupported_route["reasons"] = [unsupported_reason]
    unsupported_route["checks"]["kinodynamic"] = {
        "robot_model": "CustomDriveSettings",
        "command_limits_valid": False,
    }

    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=steering).verdict
        == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    )
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=unsupported).verdict
        == ADMISSIBLE_FEASIBILITY_UNKNOWN
    )


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


@pytest.mark.parametrize("duplicate_identity", ["route_id", "spawn_goal_pair"])
def test_duplicate_route_identities_cannot_exclude_or_prove_actor_free_rollout(
    duplicate_identity: str,
) -> None:
    excluded_certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    positive_certificate = _certificate()
    for certificate in (excluded_certificate, positive_certificate):
        existing = certificate["route_certificates"][0]
        duplicate = {
            **existing,
            "route_id": (existing["route_id"] if duplicate_identity == "route_id" else "route-1"),
            "spawn_id": (
                existing["spawn_id"] + 1
                if duplicate_identity == "route_id"
                else existing["spawn_id"]
            ),
            "checks": dict(existing["checks"]),
            "evidence": dict(existing["evidence"]),
            "reasons": list(existing["reasons"]),
        }
        certificate["checks"]["route_count"] = 2
        certificate["route_certificates"].append(duplicate)

    excluded = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=excluded_certificate,
        feasibility_evidence=_oracle(
            status="infeasible_by_construction", geometric=False, complete=False
        ),
    )
    empirical = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=positive_certificate,
        feasibility_evidence=_oracle_report(_oracle()),
    )

    for verdict in (excluded, empirical):
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
        assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in excluded.reason_codes
    assert "oracle_success_not_bound_to_static_case_and_provenance" in empirical.reason_codes


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


@pytest.mark.parametrize("fallback_marker", [0, "false", 1])
def test_malformed_no_traversal_fallback_marker_cannot_exclude_case(
    fallback_marker: Any,
) -> None:
    """Only literal false or null can pass the no-traversal fallback gate."""
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle["completion"]["fallback_or_degraded"] = fallback_marker

    assert _oracle_excludes(oracle) is False


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
    report = json.loads(
        Path(
            "docs/context/evidence/issue_5574_feasibility_oracle_2026-07-14/verdicts.json"
        ).read_text(encoding="utf-8")
    )
    excluded = classify_scenario_admissibility(
        "francis2023_narrow_doorway",
        scenario_artifact_path=report["scenario_manifest"],
        scenario_id="francis2023_narrow_doorway",
        feasibility_evidence=report,
    )
    unresolved = classify_scenario_admissibility(
        "francis2023_blind_corner",
        scenario_artifact_path=report["scenario_manifest"],
        scenario_id="francis2023_blind_corner",
        feasibility_evidence=report,
    )

    assert excluded.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert excluded.search_disposition == "retain"
    assert "feasibility_oracle_report_producer_source_digest_missing_mismatch_or_unstable" in (
        excluded.reason_codes
    )
    assert excluded.assumptions["feasibility_oracle"] == {}
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


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("status", "blocked"),
        ("blocker", "rollout_was_blocked"),
        ("fallback_or_degraded", True),
        ("observed_route_completion_feasible", False),
        ("fallback_marker", "fallback_used=true"),
        ("rollout_blocker", "hidden-blocker"),
        ("termination_reason", "collision"),
        ("min_completion_steps", 101),
        ("completion_horizon_margin_steps", 81),
    ],
)
def test_contradictory_actor_free_completion_cannot_prove_feasibility(
    mutation: str, value: Any
) -> None:
    """Blocked, fallback, termination, and horizon conflicts retain the candidate as unknown."""
    report = _oracle_report(_oracle())
    completion = report["cells"][0]["nominal_verdict"]["completion"]
    completion[mutation] = value

    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=report
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_success_not_bound_to_static_case_and_provenance" in verdict.reason_codes


def test_contradictory_positive_completion_cannot_be_overridden_by_geometric_exclusion() -> None:
    """A geometrically excluded cell with contradictory positive completion stays unknown."""
    report = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    completion = report["completion"]
    completion.update(
        route_completion_feasible=True,
        min_completion_steps=20,
        completion_horizon_margin_steps=80,
        termination_reason="success",
        status="passed",
        blocker=None,
    )
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        feasibility_evidence=report,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_route_completion_feasible", True),
        ("rollout_blocker", "hidden-blocker"),
        ("fallback_marker", "fallback_used=true"),
    ],
)
def test_geometric_exclusion_rejects_contradictory_raw_completion_fields(
    field: str, value: Any
) -> None:
    """A no-traversal verdict requires the producer's raw rollout fields to be empty."""
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle["completion"][field] = value
    verdict = classify_scenario_admissibility("case-static", feasibility_evidence=oracle)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


def test_oracle_report_producer_digest_rejects_stale_same_path_report(tmp_path: Path) -> None:
    """A fresh certificate cannot make a stale report valid after its source path is reused."""
    scenario_path = tmp_path / "reused.yaml"
    source_a = b"scenarios:\n  - name: case-static\n    seeds: [19]\n"
    source_b = b"scenarios:\n  - name: case-static\n    seeds: [20]\n"
    scenario_path.write_bytes(source_a)
    digest_a = hashlib.sha256(source_a).hexdigest()
    digest_b = hashlib.sha256(source_b).hexdigest()
    report = _oracle_report(_oracle())
    report["scenario_manifest"] = scenario_path.as_posix()
    report["source_artifact_sha256"] = digest_a
    report["cells"][0]["scenario_manifest"] = scenario_path.as_posix()
    report["cells"][0]["source_artifact_sha256"] = digest_a
    scenario_path.write_bytes(source_b)
    fresh_certificate = _certificate()
    fresh_certificate["source"] = scenario_path.as_posix()
    fresh_certificate["evidence"]["source_artifact_sha256"] = digest_b

    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=scenario_path,
        scenario_id="case-static",
        scenario_certificate=fresh_certificate,
        feasibility_evidence=report,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "feasibility_oracle_report_producer_source_digest_missing_mismatch_or_unstable" in (
        verdict.reason_codes
    )


def test_producer_certificate_digest_is_bound_through_admissibility_adapter(
    tmp_path: Path,
) -> None:
    """A source-file edit after certification invalidates the producer's captured digest."""
    scenario_path = tmp_path / "case_static.yaml"
    map_path = _REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg"
    original_bytes = f"""scenarios:
  - name: case-static
    map_file: {map_path.as_posix()}
    simulation_config:
      max_episode_steps: 100
      ped_density: 0.0
    robot_config: {{}}
    metadata:
      archetype: head_on_corridor
    seeds: [19]
""".encode()
    scenario_path.write_bytes(original_bytes)
    certificate = certificate_to_dict(
        certify_scenario_file(scenario_path, scenario_id="case-static")[0]
    )
    original_digest = hashlib.sha256(original_bytes).hexdigest()
    assert certificate["evidence"]["source_artifact_sha256"] == original_digest

    scenario_path.write_bytes(original_bytes + b"# revised after certification\n")
    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=scenario_path,
        scenario_id="case-static",
        scenario_certificate=certificate,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_producer_source_digest_missing_or_mismatch" in (
        verdict.reason_codes
    )


@pytest.mark.parametrize("referenced_file", ["map.svg", "routes.yaml"])
def test_certificate_and_oracle_reject_stale_runtime_input_bytes(
    tmp_path: Path, referenced_file: str
) -> None:
    """Same manifest bytes cannot preserve evidence after a referenced input changes."""
    scenario_path = _referenced_scenario(tmp_path)
    certificate = _bind_certificate_to_scenario(scenario_path)
    oracle = _bind_oracle_to_scenario(scenario_path)
    original_identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    original_manifest_digest = original_identity["source_artifact_sha256"]
    execution = _execution("reference", route_complete=True)
    execution.update(
        scenario_sha256=original_identity["source_artifact_sha256"],
        effective_input_sha256=original_identity["effective_input_sha256"],
        effective_input_identity_stable=True,
    )
    referenced_path = tmp_path / referenced_file
    referenced_path.write_bytes(referenced_path.read_bytes() + b"\n# changed input\n")

    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=scenario_path,
        scenario_id="case-static",
        scenario_certificate=certificate,
        feasibility_evidence=oracle,
        reference_execution=execution,
    )

    assert hashlib.sha256(scenario_path.read_bytes()).hexdigest() == original_manifest_digest
    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_effective_input_identity_missing_mismatch_or_unstable" in (
        verdict.reason_codes
    )
    assert "feasibility_oracle_effective_input_identity_missing_mismatch_or_unstable" in (
        verdict.reason_codes
    )
    assert "reference_execution_effective_input_identity_missing_or_mismatch" in (
        verdict.reason_codes
    )


def test_certificate_producer_marks_referenced_input_change_unstable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A map edit during certification clears the producer's effective-input binding."""
    scenario_path = _referenced_scenario(tmp_path)
    map_path = tmp_path / "map.svg"

    def certify_then_mutate(
        scenario: dict[str, Any], *, scenario_path: Path, **_kwargs: Any
    ) -> ScenarioCertificate:
        map_path.write_bytes(map_path.read_bytes() + b"\n<!-- concurrent edit -->\n")
        return ScenarioCertificate(
            schema_version=CERT_SCHEMA_VERSION,
            scenario_id="case-static",
            source=scenario_path.as_posix(),
            classification="valid",
            benchmark_eligibility="eligible",
            reasons=[],
            checks={"route_count": 1},
            route_certificates=[],
        )

    monkeypatch.setattr(scenario_certification_v1, "certify_scenario", certify_then_mutate)

    certificate = certificate_to_dict(
        certify_scenario_file(scenario_path, scenario_id="case-static")[0]
    )

    assert certificate["evidence"]["effective_input_sha256"] is None
    assert certificate["evidence"]["effective_input_identity_stable"] is False


def test_certificate_producer_rejects_aba_include_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ABA include replacement cannot bind identity to bytes it did not parse."""
    scenario_path = tmp_path / "root.yaml"
    included_path = tmp_path / "included.yaml"
    scenario_path.write_text("includes: [included.yaml]\n", encoding="utf-8")
    original_bytes = b"scenarios:\n  - name: case-static\n    marker: restored-A\n    seeds: [19]\n"
    consumed_bytes = b"scenarios:\n  - name: case-static\n    marker: consumed-B\n    seeds: [19]\n"
    included_path.write_bytes(original_bytes)
    original_read_bytes = Path.read_bytes
    loaded_markers: list[str] = []
    swapped = False

    def read_with_aba(path: Path) -> bytes:
        nonlocal swapped
        if path == included_path and not swapped:
            swapped = True
            included_path.write_bytes(consumed_bytes)
            consumed = original_read_bytes(path)
            included_path.write_bytes(original_bytes)
            return consumed
        return original_read_bytes(path)

    def certify_fixture(scenario: dict[str, Any], *, scenario_path: Path, **_kwargs: Any):
        return ScenarioCertificate(
            schema_version=CERT_SCHEMA_VERSION,
            scenario_id="case-static",
            source=scenario_path.as_posix(),
            classification="valid",
            benchmark_eligibility="eligible",
            reasons=[],
            checks={"loaded_marker": scenario.get("marker")},
            route_certificates=[],
        )

    monkeypatch.setattr(Path, "read_bytes", read_with_aba)

    def inspect_loaded_scenario(
        scenario: dict[str, Any], *, scenario_path: Path, **kwargs: Any
    ) -> ScenarioCertificate:
        loaded_markers.append(str(scenario.get("marker")))
        return certify_fixture(scenario, scenario_path=scenario_path, **kwargs)

    monkeypatch.setattr(scenario_certification_v1, "certify_scenario", inspect_loaded_scenario)

    certificate = certificate_to_dict(
        certify_scenario_file(scenario_path, scenario_id="case-static")[0]
    )

    assert loaded_markers == ["consumed-B"]
    assert certificate["checks"]["loaded_marker"] == "consumed-B"
    assert included_path.read_bytes() == original_bytes
    assert certificate["evidence"]["effective_input_sha256"] is None
    assert certificate["evidence"]["effective_input_identity_stable"] is False


@pytest.mark.parametrize(
    "external_reference",
    (
        "map_id: fixture-map",
        "map_file: map.svg",
        "route_overrides_file: routes.yaml",
        "include: [included.yaml]",
        "includes: [included.yaml]",
        "scenario_files: [included.yaml]",
        "map_search_paths: [maps]",
    ),
)
def test_legacy_single_row_identity_rejects_external_references(
    tmp_path: Path, external_reference: str
) -> None:
    """Incomplete expansion cannot fall back to a root-only identity with references."""
    path = tmp_path / "legacy.yaml"
    path.write_text(f"name: legacy\n{external_reference}\n", encoding="utf-8")

    identity = scenario_input_identity(path, scenario_id="legacy")

    assert identity["status"] == "unavailable"
    assert identity["effective_input_sha256"] is None


def test_map_parser_cache_key_tracks_exact_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing map bytes at one path parses the new immutable source snapshot."""
    from robot_sf.nav import svg_map_parser
    from robot_sf.training import scenario_loader

    source_path = tmp_path / "map.svg"
    initial_bytes = (_REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg").read_bytes()
    updated_bytes = initial_bytes + b"\n<!-- cache-key-change -->\n"
    source_path.write_bytes(initial_bytes)
    parsed_sources: list[bytes] = []
    original_convert = svg_map_parser.convert_map

    def capture_source(
        path: str, *, geometry_contract: str = "legacy", source_bytes: bytes | None = None
    ) -> Any:
        assert source_bytes is not None
        parsed_sources.append(source_bytes)
        return original_convert(
            path, geometry_contract=geometry_contract, source_bytes=source_bytes
        )

    scenario_loader._load_map_definition.cache_clear()
    monkeypatch.setattr(svg_map_parser, "convert_map", capture_source)
    try:
        first = scenario_loader._load_map_definition(str(source_path))
        source_path.write_bytes(updated_bytes)
        second = scenario_loader._load_map_definition(str(source_path))
    finally:
        scenario_loader._load_map_definition.cache_clear()

    assert first is not None and second is not None
    assert parsed_sources == [initial_bytes, updated_bytes]
    assert first is not second


def test_default_map_pool_is_part_of_runtime_input_identity(tmp_path: Path) -> None:
    """An implicit default map pool is included and matches the bytes its loader consumed."""
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    scenario_path = tmp_path / "default-map.yaml"
    scenario = {"name": "default-map-case", "seeds": [19]}
    scenario_path.write_text(
        yaml.safe_dump({"scenarios": [scenario]}, sort_keys=False), encoding="utf-8"
    )
    identity = scenario_input_identity(scenario_path, scenario_id="default-map-case")
    consumed: list[dict[str, str]] = []

    build_robot_config_from_scenario(
        scenario,
        scenario_path=scenario_path,
        runtime_input_records=consumed,
    )

    assert identity["status"] == "available"
    assert identity["requires_effective_input_binding"] is True
    assert any(item["role"] == "default_map_pool" for item in identity["files"])
    assert runtime_input_records_match(identity, consumed, scenario_id="default-map-case") is True


def test_runtime_identity_rejects_bytes_consumed_during_aba_map_replacement(
    tmp_path: Path,
) -> None:
    """Parser-consumed map hashes catch replacement even when path bytes are restored."""
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    scenario_path = _referenced_scenario(tmp_path)
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"][0]
    original_map_bytes = (tmp_path / "map.svg").read_bytes()
    identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    consumed_variant = original_map_bytes + b"\n<!-- consumed-during-ABA -->\n"
    (tmp_path / "map.svg").write_bytes(consumed_variant)
    consumed: list[dict[str, str]] = []
    try:
        build_robot_config_from_scenario(
            scenario,
            scenario_path=scenario_path,
            runtime_input_records=consumed,
        )
    finally:
        (tmp_path / "map.svg").write_bytes(original_map_bytes)

    assert (
        scenario_input_identity(scenario_path, scenario_id="case-static")["effective_input_sha256"]
        == identity["effective_input_sha256"]
    )
    assert runtime_input_records_match(identity, consumed, scenario_id="case-static") is False


def test_runtime_identity_rejects_route_bytes_parsed_during_aba_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route YAML identity follows the exact byte snapshot passed to its parser."""
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    scenario_path = _referenced_scenario(tmp_path)
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"][0]
    route_path = tmp_path / "routes.yaml"
    original_route_bytes = route_path.read_bytes()
    consumed_variant = original_route_bytes + b"# consumed-during-ABA\n"
    identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    original_read_bytes = Path.read_bytes

    def read_route_during_aba(path: Path) -> bytes:
        if path.resolve() == route_path.resolve():
            route_path.write_bytes(consumed_variant)
            try:
                consumed = original_read_bytes(path)
            finally:
                route_path.write_bytes(original_route_bytes)
            return consumed
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_route_during_aba)
    consumed: list[dict[str, str]] = []
    build_robot_config_from_scenario(
        scenario,
        scenario_path=scenario_path,
        runtime_input_records=consumed,
    )

    assert original_read_bytes(route_path) == original_route_bytes
    assert any(
        item["role"] == "route_overrides_file"
        and item["sha256"] == hashlib.sha256(consumed_variant).hexdigest()
        for item in consumed
    )
    assert runtime_input_records_match(identity, consumed, scenario_id="case-static") is False


def test_certificate_runtime_identity_follows_exact_map_parser_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Certificate binding rejects map bytes consumed during an ABA replacement."""
    from robot_sf.training import scenario_loader

    scenario_path = _referenced_scenario(tmp_path, route_override=False)
    map_path = tmp_path / "map.svg"
    original_map_bytes = map_path.read_bytes()
    consumed_variant = original_map_bytes + b"\n<!-- consumed-during-cert-ABA -->\n"
    original_load = scenario_loader._load_map_definition_with_digest

    def load_variant_then_restore(
        path: str, *, geometry_contract: str = "legacy"
    ) -> tuple[Any, str | None]:
        if Path(path).resolve() == map_path.resolve():
            map_path.write_bytes(consumed_variant)
            try:
                return original_load(path, geometry_contract=geometry_contract)
            finally:
                map_path.write_bytes(original_map_bytes)
        return original_load(path, geometry_contract=geometry_contract)

    monkeypatch.setattr(
        scenario_loader, "_load_map_definition_with_digest", load_variant_then_restore
    )

    certificate = certificate_to_dict(
        certify_scenario_file(scenario_path, scenario_id="case-static")[0]
    )

    assert map_path.read_bytes() == original_map_bytes
    assert certificate["evidence"]["runtime_input_identity_stable"] is False
    assert certificate["evidence"]["effective_input_identity_stable"] is False


def test_map_registry_remap_changes_effective_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal map bytes selected through different suffixes retain parser identity."""
    from robot_sf.training import scenario_loader

    scenario_path = tmp_path / "candidate.yaml"
    registry_path = tmp_path / "registry.yaml"
    map_svg = tmp_path / "map.svg"
    map_yaml = tmp_path / "map.yaml"
    map_bytes = b"same map bytes\n"
    map_svg.write_bytes(map_bytes)
    map_yaml.write_bytes(map_bytes)
    scenario_path.write_text(
        "scenarios:\n  - name: case-static\n    map_id: fixture-map\n    seeds: [19]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", registry_path.as_posix())
    scenario_loader._load_map_registry.cache_clear()
    try:
        registry_path.write_text("maps:\n  fixture-map: map.svg\n", encoding="utf-8")
        svg_identity = scenario_input_identity(scenario_path, scenario_id="case-static")
        registry_path.write_text("maps:\n  fixture-map: map.yaml\n", encoding="utf-8")
        scenario_loader._load_map_registry.cache_clear()
        yaml_identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    finally:
        scenario_loader._load_map_registry.cache_clear()

    assert svg_identity["status"] == yaml_identity["status"] == "available"
    assert svg_identity["effective_input_sha256"] != yaml_identity["effective_input_sha256"]
    svg_map = next(item for item in svg_identity["files"] if item["role"] == "map_file")
    yaml_map = next(item for item in yaml_identity["files"] if item["role"] == "map_file")
    assert svg_map["sha256"] == yaml_map["sha256"]
    assert (svg_map["map_id"], svg_map["parser"], svg_map["path"]) == (
        "fixture-map",
        "svg",
        "map.svg",
    )
    assert (yaml_map["map_id"], yaml_map["parser"], yaml_map["path"]) == (
        "fixture-map",
        "legacy_serialized_map",
        "map.yaml",
    )


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


def test_rejection_and_execution_evidence_must_share_candidate_artifact_bytes(
    tmp_path: Path,
) -> None:
    other_artifact = tmp_path / "same-id-different-artifact.yaml"
    other_artifact.write_text("id: case-static\nchanged: true\n", encoding="utf-8")
    other_digest = hashlib.sha256(other_artifact.read_bytes()).hexdigest()
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["source"] = other_artifact.as_posix()
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle["scenario_manifest"] = other_artifact.as_posix()
    execution = _execution("reference", route_complete=True)
    execution["scenario_sha256"] = other_digest

    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=certificate,
        feasibility_evidence=oracle,
        reference_execution=execution,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert verdict.evidence["scenario_artifact_identity"]["sha256"] == (_SCENARIO_ARTIFACT_SHA256)
    assert "scenario_certificate_scenario_artifact_identity_mismatch" in verdict.reason_codes
    assert "feasibility_oracle_scenario_artifact_identity_mismatch" in verdict.reason_codes
    assert "reference_execution_scenario_artifact_identity_mismatch" in verdict.reason_codes


def test_oracle_report_and_selected_cell_cannot_disagree_on_artifact(
    tmp_path: Path,
) -> None:
    other_artifact = tmp_path / "other.yaml"
    other_artifact.write_text("id: case-static\nrevision: other\n", encoding="utf-8")
    report = _oracle_report(_oracle())
    report["cells"][0]["scenario_manifest"] = other_artifact.as_posix()

    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=report
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "feasibility_oracle_cell_scenario_artifact_identity_mismatch" in verdict.reason_codes


def test_missing_canonical_candidate_artifact_cannot_support_a_certificate_exclusion() -> None:
    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_id="case-static",
        scenario_certificate=_certificate("invalid", eligibility="excluded"),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_artifact_identity_missing_or_unavailable" in verdict.reason_codes
    assert "scenario_certificate_scenario_artifact_identity_unavailable" in verdict.reason_codes


def test_execution_scenario_hash_must_match_canonical_candidate_artifact() -> None:
    run = _execution("reference", route_complete=True)
    run["scenario_sha256"] = "f" * 64

    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_scenario_artifact_identity_mismatch" in verdict.reason_codes


def test_named_execution_requires_caller_bound_scenario_id() -> None:
    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=_SCENARIO_ARTIFACT,
        scenario_certificate=_certificate(),
        reference_execution=_execution(
            "reference", route_complete=True, scenario_id="different-scenario"
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert verdict.scenario_id is None
    assert "scenario_certificate_identity_unbound" in verdict.reason_codes
    assert "reference_execution_scenario_identity_unbound" in verdict.reason_codes


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
    assert absent_replay.verdict == EMPIRICALLY_FEASIBLE
    assert absent_replay.target_planner_outcome == "route_incomplete"
    assert "planner_specific_failure_replay_missing_or_invalid" in absent_replay.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in absent_replay.reason_codes
    assert replay_success.verdict == EMPIRICALLY_FEASIBLE
    assert replay_success.target_planner_outcome == "route_incomplete"
    assert (
        "planner_specific_failure_replay_did_not_reproduce_failure" in replay_success.reason_codes
    )
    assert wrong_planner.verdict == EMPIRICALLY_FEASIBLE
    assert wrong_planner.target_planner_outcome == "route_incomplete"
    assert "planner_specific_failure_replay_wrong_planner" in wrong_planner.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in wrong_planner.reason_codes
    assert unmatched.verdict == EMPIRICALLY_FEASIBLE
    assert "matched_reference_success_target_planner_failure" not in unmatched.reason_codes


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("case_id", "other-case", "replay_execution_identity_mismatch"),
        ("scenario_id", "other-scenario", "replay_execution_identity_mismatch"),
        (
            "scenario_sha256",
            "f" * 64,
            "replay_execution_scenario_artifact_identity_mismatch",
        ),
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
        (
            "episode_id",
            "unrelated-episode",
            "planner_specific_failure_replay_source_episode_mismatch",
        ),
        (
            "source_episodes_jsonl_sha256",
            "b" * 64,
            "planner_specific_failure_replay_source_episode_mismatch",
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

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert verdict.target_planner_outcome == "route_incomplete"
    assert reason in verdict.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


@pytest.mark.parametrize("checked_count", [True, 1.0], ids=["boolean", "float"])
def test_simulator_collision_requires_integer_sample_count(checked_count: Any) -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    reason = "planned_path_simulator_collision: first_collision_sample_index=0"
    certificate["reasons"] = [reason]
    route = certificate["route_certificates"][0]
    route["reasons"] = [reason]
    route["checks"].update(
        {
            "inflated_collision_free_path": False,
            "simulator_obstacle_collision": {
                "validated": True,
                "collides_obstacle": True,
                "runtime_component": "ContinuousOccupancy.is_obstacle_collision",
                "obstacle_source": "MapDefinition.obstacles_pysf_runtime_normalized",
                "sample_spacing_m": 0.05,
                "checked_sample_count": checked_count,
                "first_collision_sample_index": 0,
            },
        }
    )

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


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


@pytest.mark.parametrize("role", ["reference", "target", "replay"])
@pytest.mark.parametrize(
    "status_value",
    ["false", None],
    ids=["malformed", "null"],
)
def test_fallback_or_degraded_execution_cannot_establish_feasibility_or_planner_failure(
    role: str, status_value: Any
) -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    changed = {
        "reference": reference,
        "target": target,
        "replay": replay,
    }[role]
    changed["fallback_or_degraded"] = status_value
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    if role == "reference":
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"{role}_execution_fallback_status_missing_or_malformed" in verdict.reason_codes


@pytest.mark.parametrize("role", ["reference", "target", "replay"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_or_degraded", True),
        ("planner_runtime", {"readiness_status": "degraded"}),
        ("planner_runtime", {"fallback_used": True}),
        ("planner_runtime", {"availability_status": "fallback"}),
    ],
    ids=["summary-flag", "nested-degraded", "nested-fallback-flag", "nested-fallback-status"],
)
def test_canonical_fallback_and_degraded_signals_never_support_classification(
    role: str, field: str, value: Any
) -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    changed = {
        "reference": reference,
        "target": target,
        "replay": replay,
    }[role]
    changed[field] = value
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    if role == "reference":
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"{role}_execution_fallback_or_degraded" in verdict.reason_codes


@pytest.mark.parametrize("role", ["reference", "target", "replay"])
def test_missing_fallback_status_cannot_establish_outcomes(role: str) -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    {
        "reference": reference,
        "target": target,
        "replay": replay,
    }[role].pop("fallback_or_degraded")
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    if role == "reference":
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"{role}_execution_provenance_incomplete" in verdict.reason_codes


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
