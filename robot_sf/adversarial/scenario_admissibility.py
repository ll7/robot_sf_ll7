"""Conservative verdicts that keep implausible and uncertain scenarios distinct."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from robot_sf.adversarial.feasibility_first import ScenarioFeasibilityContract
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.scenario_certification.feasibility_diagnostics import DIAGNOSTIC_CLAIM_BOUNDARY

SCENARIO_ADMISSIBILITY_SCHEMA = "scenario_admissibility.v1"
FEASIBILITY_ORACLE_SCHEMA = "scenario_feasibility_oracle.v1"
ENVELOPE_SENSITIVITY_SCHEMA = "envelope_sensitivity_axis.v1"
ISSUE_5574_REPORT_SCHEMA = "issue_5574_feasibility_oracle_report.v1"

STRUCTURALLY_INVALID = "structurally_invalid"
GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY = "geometric_or_kinodynamic_impossibility"
ADMISSIBLE_FEASIBILITY_UNKNOWN = "admissible_feasibility_unknown"
EMPIRICALLY_FEASIBLE = "empirically_feasible"
PLANNER_SPECIFIC_FAILURE = "planner_specific_failure"
ADMISSIBILITY_VERDICTS = (
    STRUCTURALLY_INVALID,
    GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY,
    ADMISSIBLE_FEASIBILITY_UNKNOWN,
    EMPIRICALLY_FEASIBLE,
    PLANNER_SPECIFIC_FAILURE,
)
_EXCLUSIONS = {STRUCTURALLY_INVALID, GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY}
_CERT_PLAUSIBLE = "certificate_plausible"
_CERT_CONFLICT = "certificate_conflict"
_CERTIFICATE_STRUCTURAL_REASONS = {
    "map_pool_empty",
    "no_applicable_robot_routes",
    "route_requires_at_least_two_waypoints",
    "start_point_not_finite",
    "goal_point_not_finite",
}
_CERTIFICATE_GEOMETRIC_REASONS = {
    "start_outside_map_bounds",
    "goal_outside_map_bounds",
    "start_inside_static_obstacle",
    "goal_inside_static_obstacle",
}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_CHECKPOINT_FREE_CLASSICAL_PLANNERS = frozenset({"goal", "social_force", "orca"})
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark/schemas/scenario_admissibility.v1.json"
)


@dataclass(frozen=True, slots=True)
class ScenarioAdmissibilityVerdict:
    """One versioned result; planner outcome remains separate from feasibility."""

    case_id: str
    scenario_id: str | None
    verdict: str
    target_planner_outcome: str
    search_disposition: Literal["retain", "reject"]
    reason_codes: tuple[str, ...]
    assumptions: Mapping[str, Any]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize and validate the stable result contract."""
        result = _json_mapping(
            {
                "schema_version": SCENARIO_ADMISSIBILITY_SCHEMA,
                "case_id": self.case_id,
                "scenario_id": self.scenario_id,
                "verdict": self.verdict,
                "target_planner_outcome": self.target_planner_outcome,
                "search_disposition": self.search_disposition,
                "reason_codes": list(self.reason_codes),
                "assumptions": self.assumptions,
                "evidence": self.evidence,
            }
        )
        validate_scenario_admissibility(result)
        return result


@dataclass(frozen=True, slots=True)
class AdmissibilityPartition:
    """Search IDs retained after explicit exclusions, with reasoned exclusions."""

    retained_case_ids: tuple[str, ...]
    rejected: tuple[Mapping[str, Any], ...]
    by_verdict: Mapping[str, tuple[str, ...]]


def classify_scenario_admissibility(  # noqa: PLR0913 - explicit evidence bindings are API inputs.
    case_id: str,
    *,
    scenario_artifact_path: str | Path | None = None,
    scenario_id: str | None = None,
    scenario_certificate: Mapping[str, Any] | None = None,
    feasibility_evidence: Mapping[str, Any] | None = None,
    predicate_contract: Mapping[str, Any] | None = None,
    reference_execution: Mapping[str, Any] | None = None,
    target_execution: Mapping[str, Any] | None = None,
    replay_execution: Mapping[str, Any] | None = None,
) -> ScenarioAdmissibilityVerdict:
    """Map certificate, oracle, predicate, and named-run evidence without overclaiming.

    ``scenario_artifact_path`` identifies the candidate's canonical source artifact. Certificate
    and oracle source paths must hash to those exact bytes, and named execution
    ``scenario_sha256`` values must match that digest. Named execution mappings carry
    case/scenario/planner IDs, original variant, run status,
    route completion, seed/horizon, source commit, evidence reference, and hashes for the
    scenario, robot model, simulator config, and environment. Planner-specific failure needs
    matching bindings, distinct planner IDs, and a deterministic replay of the target failure.
    Replay evidence also needs a passing ``determinism_check_status`` and ``resimulated=True``.
    """
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be non-empty")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id.strip()):
        raise ValueError("scenario_id must be non-empty when provided")
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    artifact_identity = _file_artifact_identity(scenario_artifact_path)
    evidence["scenario_artifact_identity"] = artifact_identity
    artifact_sha256 = artifact_identity.get("sha256")
    if artifact_sha256 is None:
        reasons.append("scenario_artifact_identity_missing_or_unavailable")
    inputs = {
        name: _capture(name, value, evidence, reasons)
        for name, value in (
            ("scenario_certificate", scenario_certificate),
            ("feasibility_evidence", feasibility_evidence),
            ("predicate_contract", predicate_contract),
            ("reference_execution", reference_execution),
            ("target_execution", target_execution),
            ("replay_execution", replay_execution),
        )
    }

    cert = inputs["scenario_certificate"]
    cert_state, cert_valid, cert_assumptions = _certificate(
        cert, scenario_id, artifact_sha256, reasons
    )
    if scenario_id is None and cert_valid and cert is not None:
        scenario_id = cert["scenario_id"]
    predicate_state = _predicates(inputs["predicate_contract"], case_id, reasons)
    oracle_state, oracle_assumptions = _oracle(
        inputs["feasibility_evidence"],
        case_id,
        scenario_id,
        artifact_sha256,
        cert,
        cert_valid,
        reasons,
    )
    runs = {
        role: _execution(
            inputs[f"{role}_execution"], role, case_id, scenario_id, artifact_sha256, reasons
        )
        for role in ("reference", "target", "replay")
    }
    verdict = _resolve(cert_state, oracle_state, predicate_state, runs, reasons)
    target = runs["target"]
    target_outcome = (
        "not_evaluated"
        if target_execution is None
        else "unavailable"
        if target is None
        else "route_completed"
        if target["route_complete"]
        else "route_incomplete"
    )
    return ScenarioAdmissibilityVerdict(
        case_id=case_id,
        scenario_id=scenario_id,
        verdict=verdict,
        target_planner_outcome=target_outcome,
        search_disposition="reject" if verdict in _EXCLUSIONS else "retain",
        reason_codes=tuple(dict.fromkeys(reasons)) or ("feasibility_not_demonstrated",),
        assumptions={
            "scenario_certificate": cert_assumptions,
            "feasibility_oracle": oracle_assumptions,
            "predicate_contract": predicate_state,
        },
        evidence=evidence,
    )


def _capture(
    name: str, value: Any, evidence: dict[str, Any], reasons: list[str]
) -> dict[str, Any] | None:
    if value is None:
        evidence[name] = None
        return None
    try:
        copied = _json_mapping(value)
    except (TypeError, ValueError):
        evidence[name] = {"available": False, "error": "malformed_or_not_json_safe"}
        reasons.append(f"{name}_malformed")
        return None
    evidence[name] = copied
    return copied


def _file_artifact_identity(value: str | Path | None) -> dict[str, Any]:
    """Hash a caller-selected canonical scenario artifact without trusting declared digests."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return {"status": "unavailable", "path": None, "sha256": None}
    try:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            return {"status": "unavailable", "path": path.as_posix(), "sha256": None}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, RuntimeError, ValueError):
        return {"status": "unavailable", "path": str(value), "sha256": None}
    return {"status": "available", "path": path.as_posix(), "sha256": digest}


def _artifact_reference_matches(
    value: Any, expected_sha256: str | None, evidence_name: str, reasons: list[str]
) -> bool:
    """Require a source file reference to identify the canonical scenario bytes."""
    if expected_sha256 is None:
        reasons.append(f"{evidence_name}_scenario_artifact_identity_unavailable")
        return False
    identity = _file_artifact_identity(value if isinstance(value, (str, Path)) else None)
    if identity["sha256"] is None:
        reasons.append(f"{evidence_name}_scenario_artifact_identity_missing_or_unavailable")
        return False
    if identity["sha256"] != expected_sha256:
        reasons.append(f"{evidence_name}_scenario_artifact_identity_mismatch")
        return False
    return True


def _certificate(
    cert: Any, scenario_id: str | None, artifact_sha256: str | None, reasons: list[str]
) -> tuple[str | None, bool, dict[str, Any]]:
    if not _certificate_inputs_bound(cert, scenario_id, artifact_sha256, reasons):
        return None, False, {}
    classification = cert["classification"]
    eligibility = cert["benchmark_eligibility"]
    assumptions = {
        "classification": classification,
        "benchmark_eligibility": eligibility,
        "settings": cert.get("checks", {}).get("settings", {}),
        "route_certificates": cert.get("route_certificates", []),
        "route_inventory": _route_inventory_summary(cert),
    }
    if classification == "invalid" and eligibility == "excluded":
        invalidity = _invalid_certificate_category(cert)
        if invalidity is None:
            reasons.append("scenario_certificate_invalidity_unresolved")
            return None, True, assumptions
        reasons.append(
            "scenario_certificate_structurally_invalid"
            if invalidity == STRUCTURALLY_INVALID
            else "scenario_certificate_route_geometry_impossible"
        )
        return invalidity, True, assumptions
    if classification in {"geometrically_infeasible", "kinodynamically_infeasible"}:
        if eligibility == "excluded" and _all_routes_confirm_impossibility(cert, classification):
            reasons.append(f"scenario_certificate_{classification}")
            return GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY, True, assumptions
        if eligibility == "excluded":
            reasons.append("scenario_certificate_route_coverage_unresolved")
            return None, True, assumptions
        reasons.append("scenario_certificate_classification_eligibility_conflict")
        return _CERT_CONFLICT, True, assumptions
    if classification in {"valid", "knife_edge", "hard_but_solvable"}:
        if eligibility == "excluded":
            reasons.append("scenario_certificate_classification_eligibility_conflict")
            return _CERT_CONFLICT, True, assumptions
        reasons.append("scenario_certificate_does_not_prove_dynamic_feasibility")
        return _CERT_PLAUSIBLE, True, assumptions
    reasons.append("scenario_certificate_requires_more_evidence")
    return None, True, assumptions


def _certificate_inputs_bound(
    cert: Any, scenario_id: str | None, artifact_sha256: str | None, reasons: list[str]
) -> bool:
    """Validate certificate shape, named scenario, and selected candidate bytes."""
    if cert is None or list(
        Draft202012Validator(_load_schema("scenario_cert.v1.json")).iter_errors(cert)
    ):
        reasons.append("scenario_certificate_missing_or_malformed")
        return False
    if scenario_id is not None and cert["scenario_id"] != scenario_id:
        reasons.append("scenario_certificate_identity_mismatch")
        return False
    return _artifact_reference_matches(
        cert.get("source"), artifact_sha256, "scenario_certificate", reasons
    )


def _all_routes_confirm_impossibility(cert: Mapping[str, Any], classification: str) -> bool:
    """Require producer-shaped reason and check evidence for every excluded route."""
    routes = cert.get("route_certificates")
    if (
        not _route_inventory_complete(cert)
        or not isinstance(routes, Sequence)
        or isinstance(routes, (str, bytes))
        or not routes
    ):
        return False
    if not _certificate_reasons_match_routes(cert, routes):
        return False
    return all(
        isinstance(route, Mapping)
        and route.get("classification") == classification
        and route.get("benchmark_eligibility") == "excluded"
        and _route_has_impossibility_evidence(route, classification)
        for route in routes
    )


def _certificate_reasons_match_routes(cert: Mapping[str, Any], routes: Sequence[Any]) -> bool:
    """Require scenario-level reason aggregation to match all route-level reasons."""
    top_reasons = cert.get("reasons")
    if not isinstance(top_reasons, list) or not all(
        isinstance(reason, str) and reason for reason in top_reasons
    ):
        return False
    route_reasons: set[str] = set()
    for route in routes:
        reasons = route.get("reasons") if isinstance(route, Mapping) else None
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason for reason in reasons
        ):
            return False
        route_reasons.update(reasons)
    return top_reasons == sorted(route_reasons)


def _route_has_impossibility_evidence(route: Mapping[str, Any], classification: str) -> bool:
    """Validate a route exclusion against the checks emitted by scenario_cert.v1."""
    reasons = route.get("reasons")
    checks = route.get("checks")
    if (
        not isinstance(reasons, list)
        or len(reasons) != 1
        or not isinstance(reasons[0], str)
        or not isinstance(checks, Mapping)
    ):
        return False
    if classification == "geometrically_infeasible":
        return _route_has_geometric_evidence(reasons[0], checks)
    if classification == "kinodynamically_infeasible":
        return _route_has_kinodynamic_evidence(reasons[0], checks)
    return False


def _route_has_geometric_evidence(reason: str, checks: Mapping[str, Any]) -> bool:
    """Match one of the geometric failure records emitted by the canonical producer."""
    if checks.get("inflated_collision_free_path") is not False:
        return False
    if reason.startswith("no_inflated_collision_free_path:"):
        return bool(reason.partition(":")[2].strip())
    swept_match = re.fullmatch(
        r"planned_path_swept_envelope_clips_obstacle: full_polyline_clearance_m="
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        reason,
    )
    if swept_match is not None:
        swept = checks.get("swept_envelope")
        if not isinstance(swept, Mapping):
            return False
        clearance = _finite_number(swept.get("clearance_m"))
        vertex_clearance = _finite_number(swept.get("vertex_clearance_m"))
        clipped_vertices = swept.get("clipped_vertex_count")
        waypoint_count = swept.get("planned_waypoint_count")
        return (
            swept.get("validated") is True
            and swept.get("clips_obstacle") is True
            and clearance is not None
            and clearance < 0.0
            and vertex_clearance is not None
            and math.isclose(float(swept_match.group(1)), clearance, rel_tol=1e-6, abs_tol=1e-9)
            and _nonnegative_int(clipped_vertices)
            and _minimum_int(waypoint_count, 2)
        )
    simulator_match = re.fullmatch(
        r"planned_path_simulator_collision: first_collision_sample_index=(\d+)", reason
    )
    if simulator_match is not None:
        simulator = checks.get("simulator_obstacle_collision")
        if not isinstance(simulator, Mapping):
            return False
        sample_index = int(simulator_match.group(1))
        recorded_index = simulator.get("first_collision_sample_index")
        checked_count = simulator.get("checked_sample_count")
        sample_spacing = _finite_number(simulator.get("sample_spacing_m"))
        return (
            simulator.get("validated") is True
            and simulator.get("collides_obstacle") is True
            and simulator.get("runtime_component") == "ContinuousOccupancy.is_obstacle_collision"
            and simulator.get("obstacle_source")
            == "MapDefinition.obstacles_pysf_runtime_normalized"
            and sample_spacing is not None
            and sample_spacing > 0.0
            and isinstance(recorded_index, int)
            and not isinstance(recorded_index, bool)
            and recorded_index == sample_index
            and checked_count == sample_index + 1
        )
    return False


def _route_has_kinodynamic_evidence(reason: str, checks: Mapping[str, Any]) -> bool:
    """Match supported bicycle infeasibility reasons to their recorded check values."""
    kinodynamic = checks.get("kinodynamic")
    if not isinstance(kinodynamic, Mapping):
        return False
    if reason == "bicycle_max_steer_non_positive":
        return (
            kinodynamic.get("robot_model") == "BicycleDriveSettings"
            and kinodynamic.get("command_limits_valid") is False
        )
    match = re.fullmatch(
        r"route_turn_radius_below_bicycle_limit: (-?\d+\.\d{3}) < (-?\d+\.\d{3})",
        reason,
    )
    route_radius = _finite_number(kinodynamic.get("route_minimum_turn_radius_m"))
    minimum_radius = _finite_number(kinodynamic.get("minimum_turning_radius_m"))
    return (
        match is not None
        and kinodynamic.get("robot_model") == "BicycleDriveSettings"
        and kinodynamic.get("command_limits_valid") is True
        and route_radius is not None
        and minimum_radius is not None
        and route_radius >= 0.0
        and minimum_radius > 0.0
        and route_radius < minimum_radius
        and match.group(1) == f"{route_radius:.3f}"
        and match.group(2) == f"{minimum_radius:.3f}"
    )


def _finite_number(value: Any) -> float | None:
    """Return a finite non-boolean number as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except OverflowError:
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _minimum_int(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _route_inventory_complete(cert: Mapping[str, Any]) -> bool:
    """Require the declared route count and unique route identities to match the rows."""
    checks = cert.get("checks")
    routes = cert.get("route_certificates")
    if (
        not isinstance(checks, Mapping)
        or not isinstance(routes, Sequence)
        or isinstance(routes, (str, bytes))
    ):
        return False
    route_count = checks.get("route_count")
    if (
        isinstance(route_count, bool)
        or not isinstance(route_count, int)
        or route_count < 0
        or route_count != len(routes)
        or route_count == 0
    ):
        return False
    route_ids: set[str] = set()
    route_pairs: set[tuple[int, int]] = set()
    for route in routes:
        if not isinstance(route, Mapping):
            return False
        route_id, spawn_id, goal_id = (
            route.get("route_id"),
            route.get("spawn_id"),
            route.get("goal_id"),
        )
        if (
            not isinstance(route_id, str)
            or not route_id.strip()
            or isinstance(spawn_id, bool)
            or not isinstance(spawn_id, int)
            or isinstance(goal_id, bool)
            or not isinstance(goal_id, int)
        ):
            return False
        identity = (spawn_id, goal_id)
        if route_id in route_ids or identity in route_pairs:
            return False
        route_ids.add(route_id)
        route_pairs.add(identity)
    return True


def _route_inventory_summary(cert: Mapping[str, Any]) -> dict[str, Any]:
    """Expose route inventory completeness alongside the source certificate evidence."""
    checks = cert.get("checks")
    routes = cert.get("route_certificates")
    route_count = checks.get("route_count") if isinstance(checks, Mapping) else None
    observed_count = (
        len(routes)
        if isinstance(routes, Sequence) and not isinstance(routes, (str, bytes))
        else None
    )
    return {
        "declared_count": route_count,
        "observed_count": observed_count,
        "complete": _route_inventory_complete(cert),
    }


def _invalid_certificate_category(cert: Mapping[str, Any]) -> str | None:
    routes = cert.get("route_certificates")
    if not isinstance(routes, Sequence) or isinstance(routes, (str, bytes)):
        return None
    if not routes:
        reasons = cert.get("reasons")
        checks = cert.get("checks")
        route_count = checks.get("route_count") if isinstance(checks, Mapping) else None
        if (
            isinstance(route_count, int)
            and not isinstance(route_count, bool)
            and route_count == 0
            and isinstance(reasons, list)
            and reasons
            and all(
                reason in {"map_pool_empty", "no_applicable_robot_routes"} for reason in reasons
            )
        ):
            return STRUCTURALLY_INVALID
        return None
    if not _route_inventory_complete(cert):
        return None

    categories: set[str] = set()
    for route in routes:
        if not isinstance(route, Mapping):
            return None
        if (
            route.get("classification") != "invalid"
            or route.get("benchmark_eligibility") != "excluded"
        ):
            return None
        reasons = route.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            return None
        route_categories = {_invalid_reason_category(reason) for reason in reasons}
        if None in route_categories or len(route_categories) != 1:
            return None
        categories.update(route_categories)
    return next(iter(categories)) if len(categories) == 1 else None


def _invalid_reason_category(reason: Any) -> str | None:
    if not isinstance(reason, str):
        return None
    if (
        reason in _CERTIFICATE_STRUCTURAL_REASONS
        or re.fullmatch(r"waypoint_\d+_not_finite", reason)
        or reason.startswith("illegal_amv_infrastructure_traversal:")
    ):
        return STRUCTURALLY_INVALID
    if reason in _CERTIFICATE_GEOMETRIC_REASONS:
        return GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    return None


def _predicates(contract: Any, case_id: str, reasons: list[str]) -> str | None:
    if contract is None:
        reasons.append("predicate_contract_missing")
        return None
    try:
        parsed = ScenarioFeasibilityContract.from_mapping(contract)
    except (TypeError, ValueError):
        reasons.append("predicate_contract_malformed_or_unsupported")
        return "unavailable"
    if parsed.candidate_id != case_id:
        reasons.append("predicate_contract_identity_mismatch")
        return "unavailable"
    reasons.append(
        "valid_predicates_do_not_prove_solvability"
        if parsed.feasible
        else "nonvalid_predicates_do_not_prove_impossibility"
    )
    return "all_valid" if parsed.feasible else "nonvalid"


def _oracle(
    source: Any,
    case_id: str,
    scenario_id: str | None,
    artifact_sha256: str | None,
    cert: Any,
    cert_valid: bool,
    reasons: list[str],
) -> tuple[str | None, dict[str, Any]]:
    report = _select_oracle_cell(source, case_id, scenario_id, artifact_sha256, reasons)
    if report is None:
        return None, {}
    schema = report.get("schema_version")
    if schema == FEASIBILITY_ORACLE_SCHEMA:
        nominal = report
    elif schema == ENVELOPE_SENSITIVITY_SCHEMA:
        nominal = report.get("nominal_verdict")
    else:
        reasons.append("feasibility_oracle_schema_unsupported")
        return None, {}
    if (
        not isinstance(nominal, Mapping)
        or nominal.get("schema_version") != FEASIBILITY_ORACLE_SCHEMA
    ):
        reasons.append("feasibility_oracle_nominal_verdict_malformed")
        return None, {}
    source_id = report.get("scenario_id")
    nominal_id = nominal.get("scenario_id")
    if (
        not isinstance(source_id, str)
        or not source_id.strip()
        or source_id != nominal_id
        or source_id != (scenario_id or case_id)
    ):
        reasons.append("feasibility_oracle_identity_mismatch")
        return None, {}
    geo, completion = nominal.get("geometric"), nominal.get("completion")
    assumptions = {
        "scenario_id": source_id,
        "envelope_radius_m": nominal.get("envelope_radius_m"),
        "category": report.get("category"),
        "geometric": geo if isinstance(geo, Mapping) else {},
        "completion": completion if isinstance(completion, Mapping) else {},
        "rollout_algo": report.get("rollout_algo"),
        "rollout_seed": report.get("rollout_seed"),
        "scenario_manifest": report.get("scenario_manifest"),
        "claim_boundary": report.get("claim_boundary"),
    }
    if nominal.get("claim_boundary") != DIAGNOSTIC_CLAIM_BOUNDARY:
        reasons.append("feasibility_oracle_claim_boundary_missing_or_unsupported")
        return None, assumptions
    if _oracle_excludes(nominal):
        return _classify_oracle_exclusion(cert, cert_valid, reasons), assumptions
    if _oracle_proves_actor_free_rollout(nominal, report, geo, completion, cert, cert_valid):
        assumptions["empirical_scope"] = "named_actor_free_rollout_of_original_static_case"
        reasons.append("actor_free_reference_completed_original_static_case")
        return EMPIRICALLY_FEASIBLE, assumptions
    status = nominal.get("status")
    reasons.append(
        f"oracle_{status}_does_not_prove_impossibility"
        if status in {"blocked", "time_truncated"}
        else "oracle_success_not_bound_to_static_case_and_provenance"
    )
    return None, assumptions


def _classify_oracle_exclusion(cert: Any, cert_valid: bool, reasons: list[str]) -> str | None:
    """Only accept a geometric exclusion when complete scenario route coverage agrees."""
    if not cert_valid or _certificate_route_coverage_unresolved(cert):
        reasons.append("oracle_geometric_exclusion_route_coverage_unresolved")
        return None
    if not _certificate_supports_oracle_exclusion(cert):
        reasons.append("oracle_geometric_exclusion_certificate_conflict")
        return None
    reasons.append("oracle_geometric_exclusion_under_named_envelope")
    return GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY


def _oracle_proves_actor_free_rollout(
    nominal: Mapping[str, Any],
    report: Mapping[str, Any],
    geo: Any,
    completion: Any,
    cert: Any,
    cert_valid: bool,
) -> bool:
    """Check all bindings required for empirical actor-free rollout evidence."""
    algo, seed, manifest = (
        report.get("rollout_algo"),
        report.get("rollout_seed"),
        report.get("scenario_manifest"),
    )
    return (
        nominal.get("status") == "feasible"
        and nominal.get("feasible") is True
        and isinstance(geo, Mapping)
        and geo.get("route_geometrically_feasible") is True
        and isinstance(completion, Mapping)
        and completion.get("route_completion_feasible") is True
        and cert_valid
        and _certificate_supports_actor_free_rollout(cert)
        and isinstance(algo, str)
        and bool(algo.strip())
        and isinstance(seed, int)
        and not isinstance(seed, bool)
        and seed >= 0
        and isinstance(manifest, str)
        and bool(manifest.strip())
    )


def _select_oracle_cell(
    source: Any,
    case_id: str,
    scenario_id: str | None,
    artifact_sha256: str | None,
    reasons: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(source, Mapping):
        reasons.append("feasibility_oracle_missing_or_malformed")
        return None
    if source.get("schema_version") != ISSUE_5574_REPORT_SCHEMA:
        if not _artifact_reference_matches(
            source.get("scenario_manifest"), artifact_sha256, "feasibility_oracle", reasons
        ):
            return None
        return source
    cells = source.get("cells")
    expected = scenario_id or case_id
    matches = (
        [
            cell
            for cell in cells
            if isinstance(cell, Mapping) and cell.get("scenario_id") == expected
        ]
        if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes))
        else []
    )
    if len(matches) != 1:
        reasons.append("feasibility_oracle_cell_missing_or_ambiguous")
        return None
    cell_manifest = matches[0].get("scenario_manifest")
    if not _artifact_reference_matches(
        cell_manifest, artifact_sha256, "feasibility_oracle_cell", reasons
    ):
        return None
    report_manifest = source.get("scenario_manifest")
    if report_manifest is not None and not _artifact_reference_matches(
        report_manifest, artifact_sha256, "feasibility_oracle_report", reasons
    ):
        return None
    cell = {**dict(source), **dict(matches[0])}
    if cell.get("schema_version") == ISSUE_5574_REPORT_SCHEMA:
        cell["schema_version"] = ENVELOPE_SENSITIVITY_SCHEMA
    return cell


def _oracle_excludes(nominal: Mapping[str, Any]) -> bool:
    geo = nominal.get("geometric")
    radius = nominal.get("envelope_radius_m")
    return (
        isinstance(radius, (int, float))
        and not isinstance(radius, bool)
        and radius > 0
        and nominal.get("status") == "infeasible_by_construction"
        and nominal.get("feasible") is False
        and nominal.get("claim_boundary") == DIAGNOSTIC_CLAIM_BOUNDARY
        and isinstance(geo, Mapping)
        and geo.get("route_geometrically_feasible") is False
        and geo.get("benchmark_eligibility") == "excluded"
    )


def _certificate_supports_oracle_exclusion(cert: Any) -> bool:
    """Require a complete matching certificate before using an oracle exclusion globally."""
    if not isinstance(cert, Mapping) or not _route_inventory_complete(cert):
        return False
    classification = cert.get("classification")
    if classification in {"geometrically_infeasible", "kinodynamically_infeasible"}:
        return cert.get(
            "benchmark_eligibility"
        ) == "excluded" and _all_routes_confirm_impossibility(cert, str(classification))
    if classification == "invalid":
        return (
            cert.get("benchmark_eligibility") == "excluded"
            and _invalid_certificate_category(cert) == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
        )
    return False


def _certificate_route_coverage_unresolved(cert: Any) -> bool:
    """Return whether a certificate cannot establish complete route coverage."""
    if not isinstance(cert, Mapping) or not _route_inventory_complete(cert):
        return True
    classification = cert.get("classification")
    if classification in {"geometrically_infeasible", "kinodynamically_infeasible"}:
        return not _all_routes_confirm_impossibility(cert, str(classification))
    if classification == "invalid":
        return _invalid_certificate_category(cert) is None
    return False


def _static_certificate(cert: Any) -> bool:
    routes = cert.get("route_certificates") if isinstance(cert, Mapping) else None
    if not isinstance(routes, Sequence) or isinstance(routes, (str, bytes)) or not routes:
        return False
    for route in routes:
        checks = route.get("checks") if isinstance(route, Mapping) else None
        dynamic = checks.get("dynamic") if isinstance(checks, Mapping) else None
        count = dynamic.get("single_pedestrian_count") if isinstance(dynamic, Mapping) else None
        if not isinstance(count, int) or isinstance(count, bool) or count != 0:
            return False
    return True


def _certificate_supports_actor_free_rollout(cert: Any) -> bool:
    """Require a positive, static certificate before treating an oracle rollout as feasible."""
    if not isinstance(cert, Mapping):
        return False
    if cert.get("classification") not in {"valid", "knife_edge", "hard_but_solvable"}:
        return False
    if cert.get("benchmark_eligibility") not in {"eligible", "stress_only"}:
        return False
    routes = cert.get("route_certificates")
    if not isinstance(routes, Sequence) or isinstance(routes, (str, bytes)) or not routes:
        return False
    if any(
        not isinstance(route, Mapping)
        or route.get("classification") not in {"valid", "knife_edge", "hard_but_solvable"}
        or route.get("benchmark_eligibility") not in {"eligible", "stress_only"}
        for route in routes
    ):
        return False
    return _route_inventory_complete(cert) and _static_certificate(cert)


def _execution(
    source: Any,
    role: str,
    case_id: str,
    scenario_id: str | None,
    artifact_sha256: str | None,
    reasons: list[str],
) -> dict[str, Any] | None:
    if source is None:
        return None
    reason = _execution_problem(source, role, case_id, scenario_id, artifact_sha256)
    if reason is not None:
        reasons.append(reason)
        return None
    return dict(source)


def _execution_problem(
    source: Any,
    role: str,
    case_id: str,
    scenario_id: str | None,
    artifact_sha256: str | None,
) -> str | None:
    binding_problem = _execution_binding_problem(source, role, case_id, scenario_id)
    if binding_problem is not None:
        return binding_problem
    if source["scenario_variant"] != "original" or source["run_status"] != "ok":
        return f"{role}_execution_not_an_original_recorded_run"
    fallback_problem = _execution_fallback_problem(source, role)
    if fallback_problem is not None:
        return fallback_problem
    if not _execution_digest_fields_valid(source):
        return f"{role}_execution_provenance_incomplete"
    artifact_problem = _execution_artifact_problem(source, role, artifact_sha256)
    if artifact_problem is not None:
        return artifact_problem
    if not _execution_outcome_valid(source):
        return f"{role}_execution_outcome_or_budget_invalid"
    if role == "replay":
        replay_problem = _replay_validation_problem(source)
        if replay_problem is not None:
            return replay_problem
    return None


def _execution_binding_problem(
    source: Any, role: str, case_id: str, scenario_id: str | None
) -> str | None:
    """Validate normalized execution row shape and case/scenario identity."""
    required = {
        "case_id",
        "scenario_id",
        "scenario_variant",
        "planner_id",
        "run_status",
        "fallback_or_degraded",
        "route_complete",
        "seed",
        "horizon_steps",
        "scenario_sha256",
        "robot_model_sha256",
        "simulator_config_sha256",
        "planner_config_sha256",
        "planner_checkpoint_sha256",
        "environment_sha256",
        "source_commit",
        "evidence_ref",
    }
    if not isinstance(source, Mapping) or not required.issubset(source):
        return f"{role}_execution_provenance_incomplete"
    if source["case_id"] != case_id or (
        scenario_id is not None and source["scenario_id"] != scenario_id
    ):
        return f"{role}_execution_identity_mismatch"
    if not _execution_text_fields_valid(source):
        return f"{role}_execution_provenance_incomplete"
    return None


def _execution_artifact_problem(
    source: Mapping[str, Any], role: str, artifact_sha256: str | None
) -> str | None:
    """Bind named execution rows to the candidate's canonical scenario bytes."""
    if artifact_sha256 is None:
        return f"{role}_execution_scenario_artifact_identity_unavailable"
    if source["scenario_sha256"].lower() != artifact_sha256:
        return f"{role}_execution_scenario_artifact_identity_mismatch"
    return None


def _replay_validation_problem(source: Mapping[str, Any]) -> str | None:
    """Require replay determinism validation and simulator resimulation."""
    if source.get("determinism_check_status") != "pass":
        return "replay_determinism_check_not_passed"
    if source.get("resimulated") is not True:
        return "replay_did_not_resimulate_source_episode"
    return None


def _execution_fallback_problem(source: Mapping[str, Any], role: str) -> str | None:
    """Reject absent, malformed, or positive fallback/degraded execution metadata."""
    if not isinstance(source["fallback_or_degraded"], bool):
        return f"{role}_execution_fallback_status_missing_or_malformed"
    if source["fallback_or_degraded"] or runtime_fallback_or_degraded_marker(dict(source)):
        return f"{role}_execution_fallback_or_degraded"
    return None


def _execution_text_fields_valid(source: Mapping[str, Any]) -> bool:
    fields = ("scenario_id", "scenario_variant", "planner_id", "evidence_ref")
    return all(isinstance(source[key], str) and source[key].strip() for key in fields)


def _execution_digest_fields_valid(source: Mapping[str, Any]) -> bool:
    fields = (
        "scenario_sha256",
        "robot_model_sha256",
        "simulator_config_sha256",
        "planner_config_sha256",
        "environment_sha256",
    )
    checkpoint_hash = source["planner_checkpoint_sha256"]
    return (
        all(isinstance(source[key], str) and _SHA256.fullmatch(source[key]) for key in fields)
        and _checkpoint_digest_is_valid(source["planner_id"], checkpoint_hash)
        and isinstance(source["source_commit"], str)
        and bool(_GIT_COMMIT.fullmatch(source["source_commit"]))
    )


def _checkpoint_digest_is_valid(planner_id: str, checkpoint_hash: Any) -> bool:
    """Accept the no-checkpoint sentinel only for an explicit checkpoint-free planner."""
    if isinstance(checkpoint_hash, str) and _SHA256.fullmatch(checkpoint_hash):
        return True
    if checkpoint_hash != "not_applicable":
        return False
    metadata = enrich_algorithm_metadata(algo=planner_id, metadata={})
    if (
        metadata.get("baseline_category") != "classical"
        or metadata.get("canonical_algorithm") not in _CHECKPOINT_FREE_CLASSICAL_PLANNERS
    ):
        return False
    semantics = metadata.get("policy_semantics")
    upstream = metadata.get("upstream_reference")
    if isinstance(semantics, str) and "checkpoint" in semantics.lower():
        return False
    return not (
        isinstance(upstream, Mapping)
        and isinstance(upstream.get("default_checkpoint"), str)
        and bool(upstream["default_checkpoint"].strip())
    )


def _execution_outcome_valid(source: Mapping[str, Any]) -> bool:
    return (
        isinstance(source["seed"], int)
        and not isinstance(source["seed"], bool)
        and source["seed"] >= 0
        and isinstance(source["horizon_steps"], int)
        and not isinstance(source["horizon_steps"], bool)
        and source["horizon_steps"] > 0
        and isinstance(source["route_complete"], bool)
    )


def _resolve(
    cert: str | None,
    oracle: str | None,
    predicates: str | None,
    runs: Mapping[str, Mapping[str, Any] | None],
    reasons: list[str],
) -> str:
    reference, target, replay = runs["reference"], runs["target"], runs["replay"]
    exclusions = {item for item in (cert, oracle) if item in _EXCLUSIONS}
    completed = any(
        item is not None and item["route_complete"] for item in (reference, target, replay)
    )
    if (
        cert == _CERT_CONFLICT
        or len(exclusions) > 1
        or (
            exclusions
            and (completed or oracle == EMPIRICALLY_FEASIBLE or predicates == "all_valid")
        )
        or (cert == _CERT_PLAUSIBLE and oracle in _EXCLUSIONS)
    ):
        reasons.append("conflicting_feasibility_evidence")
        return ADMISSIBLE_FEASIBILITY_UNKNOWN
    matched_failure = (
        reference is not None
        and target is not None
        and reference["route_complete"]
        and not target["route_complete"]
        and reference["planner_id"] != target["planner_id"]
        and _same_case(reference, target)
    )
    if matched_failure:
        reasons.append("named_execution_completed_original_case")
        if replay is None:
            reasons.append("planner_specific_failure_replay_missing_or_invalid")
            reasons.append("planner_specific_failure_attribution_unconfirmed")
            return EMPIRICALLY_FEASIBLE
        if replay["planner_id"] != target["planner_id"]:
            reasons.append("planner_specific_failure_replay_wrong_planner")
            reasons.append("planner_specific_failure_attribution_unconfirmed")
            return EMPIRICALLY_FEASIBLE
        if not _same_case(replay, target):
            reasons.append("planner_specific_failure_replay_case_mismatch")
            reasons.append("planner_specific_failure_attribution_unconfirmed")
            return EMPIRICALLY_FEASIBLE
        if not _same_planner_configuration(replay, target):
            reasons.append("planner_specific_failure_replay_configuration_mismatch")
            reasons.append("planner_specific_failure_attribution_unconfirmed")
            return EMPIRICALLY_FEASIBLE
        if replay["route_complete"]:
            reasons.append("planner_specific_failure_replay_did_not_reproduce_failure")
            reasons.append("planner_specific_failure_attribution_unconfirmed")
            return EMPIRICALLY_FEASIBLE
        reasons.append("matched_reference_target_failure_reproduced_by_replay")
        return PLANNER_SPECIFIC_FAILURE
    if exclusions:
        return next(iter(exclusions))
    if oracle == EMPIRICALLY_FEASIBLE or completed:
        reasons.append("named_execution_completed_original_case") if completed else None
        return EMPIRICALLY_FEASIBLE
    reasons.append(
        "target_failure_alone_does_not_prove_infeasibility"
        if target is not None and not target["route_complete"]
        else "feasibility_not_demonstrated"
    )
    return ADMISSIBLE_FEASIBILITY_UNKNOWN


def _same_case(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "case_id",
        "scenario_id",
        "scenario_variant",
        "scenario_sha256",
        "robot_model_sha256",
        "simulator_config_sha256",
        "environment_sha256",
        "source_commit",
        "seed",
        "horizon_steps",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _same_planner_configuration(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Bind a target failure replay to the target planner config and checkpoint."""
    return all(
        left.get(key) == right.get(key)
        for key in ("planner_config_sha256", "planner_checkpoint_sha256")
    )


def partition_candidates_by_admissibility(
    verdicts: Sequence[ScenarioAdmissibilityVerdict],
) -> AdmissibilityPartition:
    """Retain unknown and failure cases; reject only explicit exclusions."""
    for item in verdicts:
        if not isinstance(item, ScenarioAdmissibilityVerdict):
            raise TypeError("verdicts must be ScenarioAdmissibilityVerdict records")
        item.to_dict()
    ids = [item.case_id for item in verdicts]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs must be unique")
    return AdmissibilityPartition(
        retained_case_ids=tuple(
            item.case_id for item in verdicts if item.search_disposition == "retain"
        ),
        rejected=tuple(
            {"case_id": item.case_id, "verdict": item.verdict, "reason_codes": item.reason_codes}
            for item in verdicts
            if item.search_disposition == "reject"
        ),
        by_verdict={
            status: tuple(item.case_id for item in verdicts if item.verdict == status)
            for status in ADMISSIBILITY_VERDICTS
        },
    )


def validate_scenario_admissibility(payload: Mapping[str, Any]) -> None:
    """Validate a serialized top-level verdict."""
    errors = list(
        Draft202012Validator(_load_schema("scenario_admissibility.v1.json")).iter_errors(payload)
    )
    if errors:
        raise ValueError("invalid scenario admissibility verdict: " + "; ".join(map(str, errors)))


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    result = json.loads(json.dumps(dict(value), allow_nan=False))
    if not isinstance(result, dict):
        raise TypeError("expected JSON object")
    return result


def _load_schema(name: str) -> dict[str, Any]:
    path = _SCHEMA_PATH.with_name(name)
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "ADMISSIBILITY_VERDICTS",
    "ADMISSIBLE_FEASIBILITY_UNKNOWN",
    "EMPIRICALLY_FEASIBLE",
    "GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY",
    "PLANNER_SPECIFIC_FAILURE",
    "SCENARIO_ADMISSIBILITY_SCHEMA",
    "STRUCTURALLY_INVALID",
    "AdmissibilityPartition",
    "ScenarioAdmissibilityVerdict",
    "classify_scenario_admissibility",
    "partition_candidates_by_admissibility",
    "validate_scenario_admissibility",
]
