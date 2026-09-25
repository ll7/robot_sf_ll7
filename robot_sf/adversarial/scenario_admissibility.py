"""Conservative verdicts that keep implausible and uncertain scenarios distinct."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator

from robot_sf._execution_context import execution_context_digest
from robot_sf.adversarial.feasibility_first import ScenarioFeasibilityContract
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.map_runner.map_runner_identity import (
    planner_independent_scenario_case_payload,
)
from robot_sf.benchmark.termination_reason import (
    TERMINATION_REASONS,
    outcome_contradictions,
    status_from_termination_reason,
)
from robot_sf.scenario_certification.feasibility_diagnostics import DIAGNOSTIC_CLAIM_BOUNDARY
from robot_sf.scenario_certification.input_identity import (
    runtime_input_records_match,
    scenario_input_identity,
)
from robot_sf.training.scenario_loader import load_scenarios_for_validation

SCENARIO_ADMISSIBILITY_SCHEMA = "scenario_admissibility.v1"
FEASIBILITY_ORACLE_SCHEMA = "scenario_feasibility_oracle.v1"
ENVELOPE_SENSITIVITY_SCHEMA = "envelope_sensitivity_axis.v1"
ISSUE_5574_REPORT_SCHEMA = "issue_5574_feasibility_oracle_report.v1"
TARGET_PLANNER_REPLAY_SCHEMA = "target_planner_replay_result.v1"

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
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_MAP_RUNNER_CONFIG_HASH = re.compile(r"^[0-9a-fA-F]{16}$")
_PRODUCER_RUN_ID = re.compile(r"^[0-9a-fA-F]{32}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_CHECKPOINT_FREE_CLASSICAL_PLANNERS = frozenset({"goal", "social_force", "orca"})
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark/schemas/scenario_admissibility.v1.json"
)
_EPISODE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark/schemas/episode.schema.v1.json"
)
_TARGET_PLANNER_REPLAY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark/schemas/target_planner_replay_result.v1.json"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RESULT_PROVENANCE_SCHEMA = "benchmark_result_provenance.v1"
_RESULT_PROVENANCE_INPUT_SCHEMA = "benchmark_result_provenance.input_binding.v2"


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


@dataclass(frozen=True, slots=True)
class _ReplayResultArtifact:
    """One separately hashed target-planner replay result artifact."""

    payload: Mapping[str, Any]
    path: Path
    sha256: str


def classify_scenario_admissibility(  # noqa: PLR0913 - explicit evidence bindings are API inputs.
    case_id: str,
    *,
    scenario_artifact_path: str | Path | None = None,
    evidence_root: str | Path | None = None,
    scenario_id: str | None = None,
    scenario_certificate: Mapping[str, Any] | None = None,
    feasibility_evidence: Mapping[str, Any] | None = None,
    predicate_contract: Mapping[str, Any] | None = None,
    reference_execution: Mapping[str, Any] | None = None,
    target_execution: Mapping[str, Any] | None = None,
    replay_execution: Mapping[str, Any] | None = None,
) -> ScenarioAdmissibilityVerdict:
    """Map certificate, oracle, predicate, and named-run evidence without overclaiming.

    ``scenario_artifact_path`` identifies the candidate's canonical source artifact. Relative
    execution ``evidence_ref`` values resolve under ``evidence_root``; absolute paths are accepted
    when they identify readable local artifacts. Reference and target refs identify canonical
    episode JSONL stores. Replay refs identify the canonical replay provenance sidecar, whose
    source episode-store path and digest are read and checked. In every case, the selected v1
    episode row must match the normalized row's episode, planner, scenario, seed, revision, and
    route outcome before it can establish an individual execution outcome. Cross-planner
    attribution additionally requires the adjacent benchmark producer manifest to bind the
    selected row, planner-config bytes, numerical context, simulator settings, and run identity.
    Caller-supplied row/result context extensions do not establish that binding.

    Certificate and oracle source paths must hash to those exact bytes, and named execution
    ``scenario_sha256`` values must match that digest. Named execution mappings carry
    case/scenario/planner IDs, original variant, run status,
    route completion, seed/horizon, source commit, evidence reference, and hashes for the
    scenario, robot model, simulator config, and environment. Every execution also carries
    ``episode_id`` and ``source_episodes_jsonl_sha256``. Planner-specific failure additionally
    requires a separate ``target_planner_replay_result.v1`` artifact, referenced and hashed by the
    replay sidecar. The result must bind the target planner/configuration, output producer manifest,
    source episode store, and terminal route outcome; its producer run ID must differ from the
    target run ID. A position-only visualization determinism check is diagnostic evidence and
    cannot establish a repeated target-planner failure.
    """
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be non-empty")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id.strip()):
        raise ValueError("scenario_id must be non-empty when provided")
    resolved_evidence_root = (
        Path(evidence_root).expanduser().resolve()
        if isinstance(evidence_root, (str, Path))
        else None
    )
    execution_scenario_id = scenario_id
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    artifact_identity = _file_artifact_identity(scenario_artifact_path)
    runtime_identity = (
        scenario_input_identity(scenario_artifact_path, scenario_id=scenario_id)
        if isinstance(scenario_artifact_path, (str, Path))
        else {}
    )
    artifact_identity.update(
        {
            "effective_input_sha256": runtime_identity.get("effective_input_sha256"),
            "requires_effective_input_binding": runtime_identity.get(
                "requires_effective_input_binding", True
            ),
            "effective_input_files": runtime_identity.get("files", []),
        }
    )
    evidence["scenario_artifact_identity"] = artifact_identity
    artifact_sha256 = artifact_identity.get("sha256")
    effective_input_sha256 = artifact_identity.get("effective_input_sha256")
    requires_effective_input_binding = artifact_identity.get(
        "requires_effective_input_binding", True
    )
    selected_scenario_row, selected_scenario_row_status = _selected_candidate_scenario_row(
        scenario_artifact_path,
        scenario_id=scenario_id,
        expected_sha256=artifact_sha256,
    )
    evidence["selected_scenario_row_binding"] = {
        "status": selected_scenario_row_status,
        "scenario_id": scenario_id,
        "source_artifact_sha256": artifact_sha256,
    }
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
    for role in ("reference", "target", "replay"):
        source = inputs[f"{role}_execution"]
        if isinstance(source, Mapping):
            inputs[f"{role}_execution"] = {
                **dict(source),
                "_selected_scenario_row": selected_scenario_row,
                "_selected_scenario_row_status": selected_scenario_row_status,
                "_candidate_runtime_input_identity": runtime_identity,
            }

    cert = inputs["scenario_certificate"]
    cert_state, cert_valid, cert_assumptions = _certificate(
        cert,
        scenario_id,
        artifact_sha256,
        effective_input_sha256,
        requires_effective_input_binding,
        reasons,
    )
    predicate_state = _predicates(inputs["predicate_contract"], case_id, reasons)
    oracle_state, oracle_assumptions = _oracle(
        inputs["feasibility_evidence"],
        case_id,
        scenario_id,
        artifact_sha256,
        effective_input_sha256,
        requires_effective_input_binding,
        cert,
        cert_valid,
        reasons,
    )
    execution_artifact_bindings: dict[str, Any] = {}
    runs = {
        role: _execution(
            inputs[f"{role}_execution"],
            role,
            case_id,
            execution_scenario_id,
            artifact_sha256,
            effective_input_sha256,
            requires_effective_input_binding,
            resolved_evidence_root,
            reasons,
            execution_artifact_bindings,
        )
        for role in ("reference", "target", "replay")
    }
    evidence["execution_artifact_bindings"] = execution_artifact_bindings
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
    cert: Any,
    scenario_id: str | None,
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
    reasons: list[str],
) -> tuple[str | None, bool, dict[str, Any]]:
    if not _certificate_inputs_bound(
        cert,
        scenario_id,
        artifact_sha256,
        effective_input_sha256,
        requires_effective_input_binding,
        reasons,
    ):
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
    cert: Any,
    scenario_id: str | None,
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
    reasons: list[str],
) -> bool:
    """Validate certificate shape, named scenario, and selected candidate bytes."""
    if cert is None or list(
        Draft202012Validator(_load_schema("scenario_cert.v1.json")).iter_errors(cert)
    ):
        reasons.append("scenario_certificate_missing_or_malformed")
        return False
    if scenario_id is None:
        reasons.append("scenario_certificate_identity_unbound")
        return False
    if scenario_id is not None and cert["scenario_id"] != scenario_id:
        reasons.append("scenario_certificate_identity_mismatch")
        return False
    if not _artifact_reference_matches(
        cert.get("source"), artifact_sha256, "scenario_certificate", reasons
    ):
        return False
    cert_evidence = cert.get("evidence")
    producer_digest = (
        cert_evidence.get("source_artifact_sha256") if isinstance(cert_evidence, Mapping) else None
    )
    if (
        artifact_sha256 is None
        or not isinstance(producer_digest, str)
        or _SHA256.fullmatch(producer_digest) is None
        or producer_digest.lower() != artifact_sha256.lower()
    ):
        reasons.append("scenario_certificate_producer_source_digest_missing_or_mismatch")
        return False
    if requires_effective_input_binding:
        effective_digest = (
            cert_evidence.get("effective_input_sha256")
            if isinstance(cert_evidence, Mapping)
            else None
        )
        if (
            effective_input_sha256 is None
            or not isinstance(effective_digest, str)
            or _SHA256.fullmatch(effective_digest) is None
            or effective_digest.lower() != effective_input_sha256.lower()
            or cert_evidence.get("effective_input_identity_stable") is not True
        ):
            reasons.append(
                "scenario_certificate_effective_input_identity_missing_mismatch_or_unstable"
            )
            return False
    return True


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
    if reason == "no_inflated_collision_free_path: empty_path":
        planner = checks.get("planner")
        return isinstance(planner, Mapping) and planner.get("path_status") == "no_path"
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
            and _nonnegative_int(checked_count)
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
        return _empty_route_inventory_category(cert)
    if not _route_inventory_complete(cert) or not _certificate_reasons_match_routes(cert, routes):
        return None
    route_categories = [_invalid_certificate_route_category(route) for route in routes]
    if None in route_categories or len(set(route_categories)) != 1:
        return None
    return route_categories[0]


def _empty_route_inventory_category(cert: Mapping[str, Any]) -> str | None:
    """Support only the v1 producer's explicit empty-route invalid certificates."""
    reasons = cert.get("reasons")
    checks = cert.get("checks")
    route_count = checks.get("route_count") if isinstance(checks, Mapping) else None
    return (
        STRUCTURALLY_INVALID
        if isinstance(route_count, int)
        and not isinstance(route_count, bool)
        and route_count == 0
        and isinstance(reasons, list)
        and len(reasons) == 1
        and reasons[0] in {"map_pool_empty", "no_applicable_robot_routes"}
        else None
    )


def _invalid_certificate_route_category(route: Any) -> str | None:
    """Return one corroborated invalid category for a single route certificate."""
    if not isinstance(route, Mapping):
        return None
    if route.get("classification") != "invalid" or route.get("benchmark_eligibility") != "excluded":
        return None
    reasons = route.get("reasons")
    checks = route.get("checks")
    if not isinstance(reasons, list) or not reasons or not isinstance(checks, Mapping):
        return None
    categories = {_invalid_reason_category(reason, checks) for reason in reasons}
    return next(iter(categories)) if None not in categories and len(categories) == 1 else None


def _invalid_reason_category(reason: Any, checks: Mapping[str, Any]) -> str | None:
    if not isinstance(reason, str):
        return None
    if reason == "route_requires_at_least_two_waypoints":
        waypoint_count = checks.get("waypoint_count")
        if (
            isinstance(waypoint_count, int)
            and not isinstance(waypoint_count, bool)
            and waypoint_count < 2
            and checks.get("start") is None
            and checks.get("goal") is None
        ):
            return STRUCTURALLY_INVALID
        return None
    if reason in {"start_point_not_finite", "goal_point_not_finite"}:
        endpoint = "start" if reason.startswith("start_") else "goal"
        return (
            STRUCTURALLY_INVALID
            if _producer_point_check_is_nonfinite(checks.get(endpoint))
            else None
        )
    waypoint_match = re.fullmatch(r"waypoint_(\d+)_not_finite", reason)
    if waypoint_match is not None:
        index = int(waypoint_match.group(1))
        waypoint_count = checks.get("waypoint_count")
        if (
            not isinstance(waypoint_count, int)
            or isinstance(waypoint_count, bool)
            or waypoint_count < 2
            or index >= waypoint_count
        ):
            return None
        endpoint = "start" if index == 0 else "goal" if index == waypoint_count - 1 else None
        if endpoint is not None and _producer_point_check_is_nonfinite(checks.get(endpoint)):
            return STRUCTURALLY_INVALID
    return None


def _producer_point_check_is_nonfinite(value: Any) -> bool:
    """Recognize only malformed endpoint shapes the v1 producer can serialize."""
    if not isinstance(value, list):
        return False
    if len(value) != 2:
        return True
    for coordinate in value:
        if coordinate is None:
            return True
        if (
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            and not math.isfinite(coordinate)
        ):
            return True
    return False


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


def _oracle(  # noqa: PLR0913 - oracle classification needs its bound source and certificate inputs.
    source: Any,
    case_id: str,
    scenario_id: str | None,
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
    cert: Any,
    cert_valid: bool,
    reasons: list[str],
) -> tuple[str | None, dict[str, Any]]:
    report = _select_oracle_cell(
        source,
        scenario_id,
        artifact_sha256,
        effective_input_sha256,
        requires_effective_input_binding,
        reasons,
    )
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
        or source_id != scenario_id
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
        "source_artifact_sha256": report.get("source_artifact_sha256"),
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
        and nominal.get("runtime_input_identity_stable") is True
        and isinstance(geo, Mapping)
        and geo.get("runtime_input_identity_stable") is True
        and geo.get("route_geometrically_feasible") is True
        and isinstance(completion, Mapping)
        and completion.get("runtime_input_identity_stable") is True
        and _completion_proves_actor_free_rollout(completion)
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


def _completion_proves_actor_free_rollout(completion: Mapping[str, Any]) -> bool:
    """Require one internally consistent, non-fallback completion record."""
    steps = completion.get("min_completion_steps")
    horizon = completion.get("horizon_steps")
    margin = completion.get("completion_horizon_margin_steps")
    termination = completion.get("termination_reason")
    return (
        completion.get("route_completion_feasible") is True
        and completion.get("status") == "passed"
        and completion.get("blocker") is None
        and completion.get("fallback_or_degraded") is False
        and completion.get("observed_route_completion_feasible") is True
        and completion.get("fallback_marker") is None
        and completion.get("rollout_blocker") is None
        and isinstance(steps, int)
        and not isinstance(steps, bool)
        and steps > 0
        and isinstance(horizon, int)
        and not isinstance(horizon, bool)
        and horizon > 0
        and steps <= horizon
        and isinstance(margin, int)
        and not isinstance(margin, bool)
        and margin == horizon - steps
        and termination
        in {
            "success",
            "goal_reached",
            "route_complete",
            "completed",
            "route_follow_reached_destination",
        }
    )


def _select_oracle_cell(
    source: Any,
    scenario_id: str | None,
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
    reasons: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(source, Mapping):
        reasons.append("feasibility_oracle_missing_or_malformed")
        return None
    if scenario_id is None:
        reasons.append("feasibility_oracle_scenario_identity_unbound")
        return None
    if source.get("schema_version") != ISSUE_5574_REPORT_SCHEMA:
        return _select_single_oracle(
            source,
            artifact_sha256,
            effective_input_sha256,
            requires_effective_input_binding,
            reasons,
        )
    return _select_issue_5574_oracle_cell(
        source,
        scenario_id,
        artifact_sha256,
        effective_input_sha256,
        requires_effective_input_binding,
        reasons,
    )


def _select_single_oracle(
    source: Mapping[str, Any],
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
    reasons: list[str],
) -> Mapping[str, Any] | None:
    """Select a single-cell oracle only when its captured source identity matches."""
    if not _captured_oracle_digest_matches(source, artifact_sha256, "feasibility_oracle", reasons):
        return None
    if not _artifact_reference_matches(
        source.get("scenario_manifest"), artifact_sha256, "feasibility_oracle", reasons
    ):
        return None
    if not _captured_effective_digest_matches(
        source,
        effective_input_sha256,
        requires_effective_input_binding,
        "feasibility_oracle",
        reasons,
    ):
        return None
    return source


def _select_issue_5574_oracle_cell(
    source: Mapping[str, Any],
    scenario_id: str,
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
    reasons: list[str],
) -> Mapping[str, Any] | None:
    """Select one report cell after binding both report and cell source identity."""
    cells = source.get("cells")
    matches = (
        [
            cell
            for cell in cells
            if isinstance(cell, Mapping) and cell.get("scenario_id") == scenario_id
        ]
        if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes))
        else []
    )
    if len(matches) != 1:
        reasons.append("feasibility_oracle_cell_missing_or_ambiguous")
        return None
    if not _captured_oracle_digest_matches(
        source, artifact_sha256, "feasibility_oracle_report", reasons
    ) or not _captured_oracle_digest_matches(
        matches[0], artifact_sha256, "feasibility_oracle_cell", reasons
    ):
        return None
    if not _captured_effective_digest_matches(
        matches[0],
        effective_input_sha256,
        requires_effective_input_binding,
        "feasibility_oracle_cell",
        reasons,
    ):
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


def _captured_oracle_digest_matches(
    source: Mapping[str, Any], expected_sha256: str | None, evidence_name: str, reasons: list[str]
) -> bool:
    """Require producer-time oracle identity and a stable read of its source manifest."""
    captured = source.get("source_artifact_sha256")
    if (
        expected_sha256 is None
        or not isinstance(captured, str)
        or _SHA256.fullmatch(captured) is None
        or captured.lower() != expected_sha256.lower()
        or source.get("source_artifact_identity_stable") is not True
    ):
        reasons.append(f"{evidence_name}_producer_source_digest_missing_mismatch_or_unstable")
        return False
    return True


def _captured_effective_digest_matches(
    source: Mapping[str, Any],
    expected_sha256: str | None,
    required: bool,
    evidence_name: str,
    reasons: list[str],
) -> bool:
    """Bind oracle evidence to referenced runtime inputs when the manifest depends on them."""
    if not required:
        return True
    captured = source.get("effective_input_sha256")
    if (
        expected_sha256 is None
        or not isinstance(captured, str)
        or _SHA256.fullmatch(captured) is None
        or captured.lower() != expected_sha256.lower()
        or source.get("effective_input_identity_stable") is not True
    ):
        reasons.append(f"{evidence_name}_effective_input_identity_missing_mismatch_or_unstable")
        return False
    return True


def _oracle_excludes(nominal: Mapping[str, Any]) -> bool:
    geo = nominal.get("geometric")
    completion = nominal.get("completion")
    radius = nominal.get("envelope_radius_m")
    return (
        isinstance(radius, (int, float))
        and not isinstance(radius, bool)
        and radius > 0
        and nominal.get("status") == "infeasible_by_construction"
        and nominal.get("feasible") is False
        and nominal.get("claim_boundary") == DIAGNOSTIC_CLAIM_BOUNDARY
        and isinstance(geo, Mapping)
        and geo.get("runtime_input_identity_stable") is True
        and geo.get("route_geometrically_feasible") is False
        and geo.get("benchmark_eligibility") == "excluded"
        and isinstance(completion, Mapping)
        and completion.get("runtime_input_identity_stable") is True
        and completion.get("route_completion_feasible") is False
        and completion.get("status") == "failed"
        and completion.get("blocker") == "route_geometrically_infeasible_no_traversal_path"
        and completion.get("min_completion_steps") is None
        and completion.get("completion_horizon_margin_steps") is None
        and completion.get("termination_reason") is None
        and _positive_int(completion.get("horizon_steps"))
        and "fallback_or_degraded" in completion
        and (
            completion["fallback_or_degraded"] is None
            or completion["fallback_or_degraded"] is False
        )
        and completion.get("observed_route_completion_feasible") is None
        and completion.get("fallback_marker") is None
        and completion.get("rollout_blocker") is None
    )


def _positive_int(value: Any) -> bool:
    """Return whether value is a positive integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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


def _execution(  # noqa: PLR0913 - explicit execution evidence bindings are passed through.
    source: Any,
    role: str,
    case_id: str,
    scenario_id: str | None,
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
    evidence_root: Path | None,
    reasons: list[str],
    artifact_bindings: dict[str, Any],
) -> dict[str, Any] | None:
    if source is None:
        return None
    reason = _execution_problem(
        source,
        role,
        case_id,
        scenario_id,
        artifact_sha256,
        effective_input_sha256,
        requires_effective_input_binding,
    )
    if reason is not None:
        reasons.append(reason)
        return None
    evidence_problem, binding = _execution_evidence_binding(
        source, role=role, evidence_root=evidence_root
    )
    artifact_bindings[role] = binding
    if evidence_problem is not None:
        reasons.append(evidence_problem)
        return None
    context_status = binding.get("run_context_binding", {}).get("status")
    if context_status != "valid":
        reasons.append(f"{role}_execution_run_context_{context_status or 'unavailable'}")
    run_context_binding = binding.get("run_context_binding", {})
    scenario_status = run_context_binding.get("scenario_matrix_binding_status")
    case_identity_status = run_context_binding.get("case_identity_binding_status")
    if scenario_status != "valid" or case_identity_status != "valid":
        identity_status = (
            "mismatch" if "mismatch" in {scenario_status, case_identity_status} else "unavailable"
        )
        reasons.append(f"{role}_execution_scenario_case_identity_{identity_status}")
        return None
    resource_closure_status = run_context_binding.get(
        "scenario_runtime_input_closure_binding_status"
    )
    if resource_closure_status not in {"valid", "not_required"}:
        reasons.append(f"{role}_execution_scenario_runtime_input_closure_{resource_closure_status}")
        return None
    if context_status == "mismatch":
        return None
    bound_source = dict(source)
    bound_source["_execution_context_binding_status"] = context_status
    producer_binding = binding.get("producer_provenance_binding")
    if not isinstance(producer_binding, Mapping):
        result_binding = binding.get("target_planner_replay_result_binding")
        producer_binding = (
            result_binding.get("producer_provenance_binding")
            if isinstance(result_binding, Mapping)
            else None
        )
    if isinstance(producer_binding, Mapping):
        bound_source["_producer_provenance_binding"] = dict(producer_binding)
    return bound_source


def _execution_evidence_binding(
    source: Mapping[str, Any], *, role: str, evidence_root: Path | None
) -> tuple[str | None, dict[str, Any]]:
    """Read and bind canonical episode-store and replay-sidecar bytes to one normalized run."""
    reference = source.get("evidence_ref")
    artifact_path = _resolve_evidence_path(reference, evidence_root=evidence_root)
    if artifact_path is None:
        return f"{role}_execution_evidence_ref_missing_or_unreadable", {
            "status": "unavailable",
            "evidence_ref": reference,
        }
    if role == "replay":
        return _replay_evidence_binding(
            source, artifact_path=artifact_path, evidence_root=evidence_root
        )
    return _episode_store_binding(
        source,
        role=role,
        evidence_ref=artifact_path,
        store_path=artifact_path,
    )


def _replay_evidence_binding(
    source: Mapping[str, Any], *, artifact_path: Path, evidence_root: Path | None
) -> tuple[str | None, dict[str, Any]]:
    """Load the replay sidecar and bind it through its referenced source episode store."""
    try:
        sidecar_bytes = artifact_path.read_bytes()
        sidecar_payload = json.loads(sidecar_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "replay_execution_replay_sidecar_missing_or_malformed", {
            "status": "unavailable",
            "evidence_ref": artifact_path.as_posix(),
        }
    if not isinstance(sidecar_payload, Mapping):
        return "replay_execution_replay_sidecar_missing_or_malformed", {
            "status": "unavailable",
            "evidence_ref": artifact_path.as_posix(),
        }
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    store_path = _resolve_evidence_path(
        sidecar_payload.get("source_episodes_jsonl_path"),
        evidence_root=evidence_root,
        relative_to=artifact_path.parent,
    )
    if store_path is None:
        return "replay_execution_episode_store_missing_or_unreadable", {
            "status": "unavailable",
            "evidence_ref": artifact_path.as_posix(),
            "replay_sidecar_sha256": sidecar_sha256,
        }
    result_path = _resolve_evidence_path(
        sidecar_payload.get("target_planner_replay_result_path"),
        evidence_root=evidence_root,
        relative_to=artifact_path.parent,
    )
    if result_path is None:
        return "replay_execution_target_planner_result_missing_or_unreadable", {
            "status": "unavailable",
            "evidence_ref": artifact_path.as_posix(),
            "replay_sidecar_sha256": sidecar_sha256,
        }
    try:
        result_bytes = result_path.read_bytes()
        result_payload = json.loads(result_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "replay_execution_target_planner_result_missing_or_malformed", {
            "status": "unavailable",
            "evidence_ref": result_path.as_posix(),
            "replay_sidecar_sha256": sidecar_sha256,
        }
    if not isinstance(result_payload, Mapping):
        return "replay_execution_target_planner_result_missing_or_malformed", {
            "status": "unavailable",
            "evidence_ref": result_path.as_posix(),
            "replay_sidecar_sha256": sidecar_sha256,
        }
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    sidecar_result_sha256 = sidecar_payload.get("target_planner_replay_result_sha256")
    if (
        not isinstance(sidecar_result_sha256, str)
        or _SHA256.fullmatch(sidecar_result_sha256) is None
        or sidecar_result_sha256.lower() != result_sha256
    ):
        return "replay_execution_target_planner_result_digest_mismatch", {
            "status": "mismatch",
            "evidence_ref": result_path.as_posix(),
            "replay_sidecar_sha256": sidecar_sha256,
            "target_planner_replay_result_sha256": result_sha256,
        }
    return _episode_store_binding(
        source,
        role="replay",
        evidence_ref=artifact_path,
        store_path=store_path,
        sidecar=sidecar_payload,
        sidecar_sha256=sidecar_sha256,
        replay_result_artifact=_ReplayResultArtifact(
            payload=result_payload,
            path=result_path,
            sha256=result_sha256,
        ),
    )


def _episode_store_binding(
    source: Mapping[str, Any],
    *,
    role: str,
    evidence_ref: Path,
    store_path: Path,
    sidecar: Mapping[str, Any] | None = None,
    sidecar_sha256: str | None = None,
    replay_result_artifact: _ReplayResultArtifact | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Validate store bytes and episode identity/outcome for a normalized execution row."""
    try:
        store_bytes = store_path.read_bytes()
    except OSError:
        return f"{role}_execution_episode_store_missing_or_unreadable", {
            "status": "unavailable",
            "evidence_ref": evidence_ref.as_posix(),
            "episode_store_path": store_path.as_posix(),
        }
    store_sha256 = hashlib.sha256(store_bytes).hexdigest()
    binding = {
        "status": "checked",
        "evidence_ref": evidence_ref.as_posix(),
        "episode_store_path": store_path.as_posix(),
        "episode_store_sha256": store_sha256,
        "episode_id": source.get("episode_id"),
        "route_complete": source.get("route_complete"),
    }
    if sidecar_sha256 is not None:
        binding["replay_sidecar_sha256"] = sidecar_sha256
    if replay_result_artifact is not None:
        binding["target_planner_replay_result_path"] = replay_result_artifact.path.as_posix()
        binding["target_planner_replay_result_sha256"] = replay_result_artifact.sha256
    if store_sha256.lower() != str(source.get("source_episodes_jsonl_sha256", "")).lower():
        binding["status"] = "mismatch"
        return f"{role}_execution_episode_store_digest_mismatch", binding

    episode, parse_reason = _episode_row_for_binding(
        store_bytes, episode_id=source.get("episode_id")
    )
    if episode is None:
        binding["status"] = "mismatch"
        return f"{role}_execution_{parse_reason}", binding
    row_reason = _episode_row_binding_problem(
        episode, source, compare_route_outcome=role != "replay"
    )
    if row_reason is not None:
        binding["status"] = "mismatch"
        return f"{role}_execution_{row_reason}", binding
    if sidecar is not None:
        auxiliary_reason, auxiliary_binding = _replay_auxiliary_binding(
            sidecar=sidecar,
            source=source,
            source_store_path=store_path,
            store_sha256=store_sha256,
            result_artifact=replay_result_artifact,
        )
        binding.update(auxiliary_binding)
        if auxiliary_reason is not None:
            binding["status"] = "mismatch"
            return f"replay_execution_{auxiliary_reason}", binding
    else:
        provenance_problem, producer_binding = _producer_provenance_binding(
            store_path=store_path,
            store_bytes=store_bytes,
            episode=episode,
            source=source,
            manifest_path=_episode_provenance_manifest_path(store_path),
        )
        binding["producer_provenance_binding"] = producer_binding
        binding["run_context_binding"] = producer_binding.get("run_context_binding", {})
        if provenance_problem is not None:
            binding["status"] = "mismatch"
            return f"{role}_execution_{provenance_problem}", binding
    binding["status"] = "valid"
    binding["row_identity"] = {
        "scenario_id": episode["scenario_id"],
        "planner_id": episode["algo"],
        "seed": episode["seed"],
        "source_commit": episode["git_hash"],
        "route_complete": episode["outcome"]["route_complete"],
        "termination_reason": episode["termination_reason"],
    }
    return None, binding


def _episode_provenance_manifest_path(store_path: Path) -> Path:
    """Return the canonical map-runner provenance sidecar path for an episode JSONL."""
    return store_path.with_name(store_path.name + ".provenance.json")


def _producer_provenance_binding(
    *,
    store_path: Path,
    store_bytes: bytes,
    episode: Mapping[str, Any],
    source: Mapping[str, Any],
    manifest_path: Path,
    expected_manifest_sha256: str | None = None,
    required: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    """Bind one episode row to the canonical benchmark producer manifest.

    This checks artifact and record consistency. The JSON manifest is not a signed execution
    attestation, so these checks cannot prove which external process wrote it.
    """
    expected_path = _episode_provenance_manifest_path(store_path).resolve()
    try:
        resolved_path = manifest_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        status = "unavailable"
        binding = {
            "status": status,
            "manifest_path": manifest_path.as_posix(),
            "run_context_binding": {"status": status},
        }
        return ("producer_provenance_unavailable" if required else None), binding
    if resolved_path != expected_path:
        return _producer_provenance_failure(
            "producer_provenance_path_not_adjacent",
            {
                "manifest_path": resolved_path.as_posix(),
                "expected_manifest_path": expected_path.as_posix(),
            },
        )
    read_problem, manifest, binding = _read_producer_manifest(
        resolved_path,
        store_bytes=store_bytes,
        expected_sha256=expected_manifest_sha256,
    )
    if read_problem is not None:
        return read_problem, binding
    if not isinstance(manifest, Mapping) or not _producer_manifest_identity_valid(
        manifest, episode
    ):
        return _producer_provenance_failure(
            "producer_provenance_identity_or_completeness_invalid", binding
        )
    row_problem, producer_row, row_index = _producer_episode_row(
        manifest, store_path=store_path, store_bytes=store_bytes, episode=episode
    )
    if row_problem is not None or producer_row is None:
        return _producer_provenance_failure(
            row_problem or "producer_provenance_episode_row_invalid", binding
        )
    settings_problem, settings_digest = _simulator_settings_digest(
        producer_row.get("simulator_settings")
    )
    if settings_problem is not None or settings_digest is None:
        return _producer_provenance_failure(
            settings_problem or "producer_provenance_simulator_settings_invalid", binding
        )
    binding.update(
        {
            "run_id": manifest["run"]["run_id"],
            "row_index": row_index,
            "row_config_hash": producer_row["config_hash"],
            "simulator_settings_sha256": settings_digest,
        }
    )
    context_binding = _producer_run_context_binding(manifest, source, episode=episode)
    binding.update(context_binding)
    binding["status"] = context_binding["run_context_binding"]["status"]
    return None, binding


def _producer_provenance_failure(
    reason: str, binding: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    binding["status"] = "mismatch"
    binding["run_context_binding"] = {"status": "mismatch"}
    return reason, binding


def _read_producer_manifest(
    path: Path,
    *,
    store_bytes: bytes,
    expected_sha256: str | None,
) -> tuple[str | None, Any, dict[str, Any]]:
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return (
            "producer_provenance_malformed",
            None,
            {
                "status": "mismatch",
                "manifest_path": path.as_posix(),
                "run_context_binding": {"status": "mismatch"},
            },
        )
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    binding: dict[str, Any] = {
        "status": "checked",
        "manifest_path": path.as_posix(),
        "manifest_sha256": digest,
        "episode_store_sha256": hashlib.sha256(store_bytes).hexdigest(),
    }
    if expected_sha256 is not None and (
        _SHA256.fullmatch(expected_sha256) is None or expected_sha256.lower() != digest
    ):
        return "producer_provenance_manifest_digest_mismatch", None, binding
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != _RESULT_PROVENANCE_SCHEMA
        or manifest.get("input_binding_schema_version") != _RESULT_PROVENANCE_INPUT_SCHEMA
    ):
        return "producer_provenance_schema_invalid", None, binding
    return None, manifest, binding


def _producer_manifest_identity_valid(
    manifest: Mapping[str, Any], episode: Mapping[str, Any]
) -> bool:
    run = manifest.get("run")
    campaign = manifest.get("campaign_identity")
    completeness = manifest.get("completeness")
    return (
        isinstance(run, Mapping)
        and isinstance(manifest.get("inputs"), Mapping)
        and isinstance(campaign, Mapping)
        and isinstance(completeness, Mapping)
        and isinstance(manifest.get("raw_artifacts"), list)
        and isinstance(manifest.get("rows"), list)
        and completeness.get("status") == "complete"
        and isinstance(run.get("run_id"), str)
        and bool(_PRODUCER_RUN_ID.fullmatch(run.get("run_id", "")))
        and isinstance(run.get("repo_commit"), str)
        and bool(_GIT_COMMIT.fullmatch(run.get("repo_commit", "")))
        and run.get("runner") == "map_runner.run_map_batch"
        and campaign.get("algorithm") == episode.get("algo")
        and run.get("repo_commit", "").lower() == str(episode.get("git_hash", "")).lower()
    )


def _producer_episode_row(
    manifest: Mapping[str, Any],
    *,
    store_path: Path,
    store_bytes: bytes,
    episode: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None, int | None]:
    artifacts = manifest.get("raw_artifacts")
    if not isinstance(artifacts, list):
        return "producer_provenance_episode_artifact_ambiguous", None, None
    matching_artifacts = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == "episodes_jsonl"
    ]
    if len(matching_artifacts) != 1:
        return "producer_provenance_episode_artifact_ambiguous", None, None
    artifact = matching_artifacts[0]
    artifact_path = _resolve_evidence_path(artifact.get("path"), evidence_root=_REPOSITORY_ROOT)
    if (
        artifact.get("artifact_status") != "available"
        or artifact_path is None
        or artifact_path.resolve() != store_path.resolve()
        or artifact.get("sha256") != hashlib.sha256(store_bytes).hexdigest()
    ):
        return "producer_provenance_episode_artifact_mismatch", None, None
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return "producer_provenance_episode_row_ambiguous", None, None
    matching_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("episode_id") == episode.get("episode_id")
    ]
    if len(matching_rows) != 1:
        return "producer_provenance_episode_row_ambiguous", None, None
    row = matching_rows[0]
    row_index = row.get("jsonl_line")
    try:
        lines = store_bytes.decode("utf-8").splitlines()
        line_episode = (
            json.loads(lines[row_index])
            if isinstance(row_index, int)
            and not isinstance(row_index, bool)
            and 0 <= row_index < len(lines)
            else None
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        line_episode = None
    if not _producer_episode_row_matches(
        row, artifact=artifact, episode=episode, line_episode=line_episode
    ):
        return "producer_provenance_episode_row_mismatch", None, None
    return None, row, row_index


def _producer_episode_row_matches(
    row: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    episode: Mapping[str, Any],
    line_episode: Any,
) -> bool:
    settings = row.get("simulator_settings")
    config_hash = row.get("config_hash")
    return (
        line_episode == dict(episode)
        and row.get("scenario_id") == episode.get("scenario_id")
        and row.get("seed") == episode.get("seed")
        and config_hash == episode.get("config_hash")
        and isinstance(config_hash, str)
        and bool(_MAP_RUNNER_CONFIG_HASH.fullmatch(config_hash))
        and row.get("repo_commit") == episode.get("git_hash")
        and row.get("raw_artifact") == artifact.get("path")
        and isinstance(settings, Mapping)
        and settings.get("horizon") == episode.get("horizon")
    )


def _simulator_settings_digest(settings: Any) -> tuple[str | None, str | None]:
    if not isinstance(settings, Mapping):
        return "producer_provenance_simulator_settings_invalid", None
    try:
        encoded = json.dumps(dict(settings), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return "producer_provenance_simulator_settings_invalid", None
    return None, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _producer_run_context_binding(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    run = manifest.get("run")
    inputs = manifest.get("inputs")
    context_status, context_digest, context_missing = _producer_execution_context(
        run.get("execution_context") if isinstance(run, Mapping) else None
    )
    scenario_status, scenario_digest, scenario_missing = _producer_scenario_input(
        inputs.get("scenario_matrix") if isinstance(inputs, Mapping) else None,
        source=source,
    )
    config_status, config_digest, config_missing = _producer_planner_config(
        inputs.get("algo_config") if isinstance(inputs, Mapping) else None,
        source=source,
    )
    case_identity_digest = _planner_independent_case_identity_digest(episode)
    self_identity_status = "valid" if case_identity_digest is not None else "unavailable"
    selected_case_status, candidate_case_digest, producer_case_digest = (
        _selected_scenario_case_binding(source, episode)
    )
    resource_closure_status = _producer_scenario_resource_closure_binding(source, episode)
    case_identity_status = _combine_binding_status(self_identity_status, selected_case_status)
    status = _combine_binding_status(
        _combine_binding_status(context_status, scenario_status),
        _combine_binding_status(
            config_status,
            _combine_binding_status(case_identity_status, resource_closure_status),
        ),
    )
    checkpoint_status = _producer_checkpoint_status(source)
    if checkpoint_status == "unavailable":
        status = _combine_binding_status(status, "unavailable")
    missing_fields = sorted(set(context_missing + scenario_missing + config_missing))
    if case_identity_digest is None:
        missing_fields.append("episode.scenario_params")
    if selected_case_status == "unavailable":
        missing_fields.append("candidate.selected_scenario_row")
    if resource_closure_status == "unavailable":
        missing_fields.append("producer.scenario_runtime_input_closure")
    if checkpoint_status == "unavailable":
        missing_fields.append("planner_checkpoint_sha256")
    return {
        "execution_context_sha256": context_digest,
        "scenario_matrix_sha256": scenario_digest,
        "planner_config_sha256": config_digest,
        "case_identity_sha256": case_identity_digest,
        "candidate_scenario_case_identity_sha256": candidate_case_digest,
        "producer_scenario_case_identity_sha256": producer_case_digest,
        "planner_checkpoint_status": checkpoint_status,
        "run_context_binding": {
            "status": status,
            "execution_context_binding_status": context_status,
            "scenario_matrix_binding_status": scenario_status,
            "planner_config_binding_status": config_status,
            "case_identity_binding_status": case_identity_status,
            "selected_scenario_row_binding_status": selected_case_status,
            "scenario_runtime_input_closure_binding_status": resource_closure_status,
            "missing_fields": sorted(set(missing_fields)),
            "fields": [
                "run_id",
                "execution_context_sha256",
                "scenario_matrix_sha256",
                "planner_config_sha256",
                "candidate_scenario_case_identity_sha256",
                "case_identity_sha256",
                "simulator_settings_sha256",
                "horizon",
            ],
        },
    }


def _selected_scenario_case_binding(
    source: Mapping[str, Any], episode: Mapping[str, Any]
) -> tuple[str, str | None, str | None]:
    """Compare the selected artifact row with the producer episode's scenario settings."""
    selected = source.get("_selected_scenario_row")
    if source.get("_selected_scenario_row_status") != "valid" or not isinstance(selected, Mapping):
        return "unavailable", None, None
    seed = episode.get("seed")
    params = episode.get("scenario_params")
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(params, Mapping):
        return "unavailable", None, None
    candidate_projection = planner_independent_scenario_case_payload(selected, seed=seed)
    producer_projection = planner_independent_scenario_case_payload(params, seed=seed)
    candidate_digest = _canonical_mapping_digest(candidate_projection)
    producer_digest = _canonical_mapping_digest(producer_projection)
    if candidate_digest is None or producer_digest is None:
        return "unavailable", candidate_digest, producer_digest
    return (
        "valid" if candidate_digest == producer_digest else "mismatch",
        candidate_digest,
        producer_digest,
    )


def _producer_scenario_resource_closure_binding(
    source: Mapping[str, Any], episode: Mapping[str, Any]
) -> str:
    """Require producer-row parser snapshots whenever the candidate has external inputs."""
    identity = source.get("_candidate_runtime_input_identity")
    if not isinstance(identity, Mapping):
        return "unavailable"
    if identity.get("requires_effective_input_binding") is not True:
        return "not_required"
    records = episode.get("runtime_input_records")
    scenario_id = episode.get("scenario_id")
    if not isinstance(records, list) or not isinstance(scenario_id, str) or not scenario_id.strip():
        return "unavailable"
    if not all(isinstance(record, Mapping) for record in records):
        return "mismatch"
    return (
        "valid"
        if runtime_input_records_match(
            identity,
            [dict(record) for record in records],
            scenario_id=scenario_id,
        )
        else "mismatch"
    )


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str | None:
    """Hash JSON-compatible scenario projections with deterministic encoding."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _selected_candidate_scenario_row(
    scenario_artifact_path: str | Path | None,
    *,
    scenario_id: str | None,
    expected_sha256: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Load exactly one selected canonical row, with the bounded legacy single-row form."""
    if not isinstance(scenario_artifact_path, (str, Path)):
        return None, "unavailable"
    try:
        root = Path(scenario_artifact_path).expanduser().resolve(strict=True)
        raw_bytes = root.read_bytes()
    except (OSError, RuntimeError, ValueError):
        return None, "unavailable"
    if (
        not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
        or hashlib.sha256(raw_bytes).hexdigest() != expected_sha256.lower()
    ):
        return None, "unavailable"

    report = load_scenarios_for_validation(root)
    if report.load_error is None and not report.entry_issues and not report.load_issues:
        rows = [
            row
            for row in report.scenarios
            if _scenario_row_identifier(row) is not None
            and (scenario_id is None or _scenario_row_identifier(row) == scenario_id)
        ]
        if len(rows) == 1:
            return dict(rows[0]), "valid"
        return None, "unavailable"

    # A narrow legacy form is retained for the existing one-row identity contract. It cannot
    # contain includes or resource references because those require loader provenance.
    try:
        legacy_row = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        return None, "unavailable"
    if not isinstance(legacy_row, Mapping) or "scenarios" in legacy_row:
        return None, "unavailable"
    external_reference_keys = {
        "includes",
        "include",
        "scenario_files",
        "map_id",
        "map_file",
        "map_search_paths",
        "route_overrides_file",
    }
    if any(
        _has_declared_external_reference(legacy_row.get(key)) for key in external_reference_keys
    ):
        return None, "unavailable"
    row_id = _scenario_row_identifier(legacy_row)
    if row_id is None or (scenario_id is not None and row_id != scenario_id):
        return None, "unavailable"
    return dict(legacy_row), "valid"


def _scenario_row_identifier(row: Mapping[str, Any]) -> str | None:
    """Use map-runner scenario ID precedence for one parsed row."""
    value = row.get("name") or row.get("scenario_id") or row.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _has_declared_external_reference(value: Any) -> bool:
    """Return whether a legacy row declares a non-empty external-resource reference."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value)
    return True


def _producer_scenario_input(
    scenario_input: Any, *, source: Mapping[str, Any]
) -> tuple[str, str | None, list[str]]:
    if (
        not isinstance(scenario_input, Mapping)
        or scenario_input.get("artifact_status") != "available"
    ):
        return "unavailable", None, ["inputs.scenario_matrix"]
    digest = scenario_input.get("sha256")
    path = _resolve_evidence_path(scenario_input.get("path"), evidence_root=_REPOSITORY_ROOT)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        return "mismatch", None, []
    if path is None:
        return "unavailable", None, ["inputs.scenario_matrix.bytes"]
    try:
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable", None, ["inputs.scenario_matrix.bytes"]
    if (
        actual_digest != digest.lower()
        or source.get("scenario_sha256", "").lower() != digest.lower()
    ):
        return "mismatch", actual_digest, []
    return "valid", actual_digest, []


def _planner_independent_case_identity_digest(episode: Mapping[str, Any]) -> str | None:
    """Hash map-runner scenario settings while excluding planner identity/configuration."""
    scenario_params = episode.get("scenario_params")
    if not isinstance(scenario_params, Mapping) or scenario_params.get("algo") != episode.get(
        "algo"
    ):
        return None
    params = dict(scenario_params)
    try:
        encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_config_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        if expected_config_hash != episode.get("config_hash"):
            return None
        params.pop("algo", None)
        params.pop("algo_config_hash", None)
        case_encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(case_encoded.encode("utf-8")).hexdigest()


def _producer_execution_context(context: Any) -> tuple[str, str | None, list[str]]:
    if not isinstance(context, Mapping):
        return "unavailable", None, ["run.execution_context"]
    digest = context.get("execution_context_sha256")
    payload = {
        key: value
        for key, value in context.items()
        if key not in {"hostname", "execution_context_sha256"}
    }
    try:
        computed_digest = execution_context_digest(payload)
    except (TypeError, ValueError):
        computed_digest = None
    valid_digest = isinstance(digest, str) and bool(_SHA256.fullmatch(digest))
    if not valid_digest or computed_digest != digest.lower():
        return "mismatch", digest if isinstance(digest, str) else None, []
    return "valid", digest, []


def _producer_planner_config(
    config: Any, *, source: Mapping[str, Any]
) -> tuple[str, str | None, list[str]]:
    if not isinstance(config, Mapping) or config.get("artifact_status") != "available":
        return "unavailable", None, ["inputs.algo_config"]
    digest = config.get("sha256")
    path = _resolve_evidence_path(config.get("path"), evidence_root=_REPOSITORY_ROOT)
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        return "mismatch", None, []
    if path is None:
        return "unavailable", None, ["inputs.algo_config.bytes"]
    try:
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable", None, ["inputs.algo_config.bytes"]
    if (
        actual_digest != digest.lower()
        or source.get("planner_config_sha256", "").lower() != digest.lower()
    ):
        return "mismatch", actual_digest, []
    return "valid", actual_digest, []


def _producer_checkpoint_status(source: Mapping[str, Any]) -> str:
    checkpoint = source.get("planner_checkpoint_sha256")
    if checkpoint == "not_applicable" and _checkpoint_digest_is_valid(
        source.get("planner_id", ""), checkpoint
    ):
        return "not_applicable"
    return "unavailable"


def _combine_binding_status(left: str, right: str) -> str:
    if "mismatch" in {left, right}:
        return "mismatch"
    if "unavailable" in {left, right}:
        return "unavailable"
    return "valid"


def _replay_auxiliary_binding(
    *,
    sidecar: Mapping[str, Any],
    source: Mapping[str, Any],
    source_store_path: Path,
    store_sha256: str,
    result_artifact: _ReplayResultArtifact | None,
) -> tuple[str | None, dict[str, Any]]:
    """Check replay sidecar and separate target-planner result against one source store."""
    sidecar_reason = _replay_sidecar_binding_problem(sidecar, source, store_sha256)
    if sidecar_reason is not None:
        return sidecar_reason, {"status": "mismatch"}
    if result_artifact is None:
        return "target_planner_result_missing_or_malformed", {"status": "unavailable"}
    result_reason, result_binding = _target_planner_replay_result_binding(
        result_artifact.payload,
        source=source,
        source_store_path=source_store_path,
        store_sha256=store_sha256,
        result_path=result_artifact.path,
    )
    return result_reason, {
        "target_planner_replay_result_path": result_artifact.path.as_posix(),
        "target_planner_replay_result_sha256": result_artifact.sha256,
        "target_planner_replay_result_binding": result_binding,
        "run_context_binding": result_binding.get("run_context_binding", {}),
        "producer_provenance_binding": result_binding.get("producer_provenance_binding", {}),
    }


def _resolve_evidence_path(
    value: Any,
    *,
    evidence_root: Path | None,
    relative_to: Path | None = None,
) -> Path | None:
    """Resolve a local evidence reference without allowing implicit CWD binding."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    root_relative = not path.is_absolute() and relative_to is None
    if not path.is_absolute():
        base = relative_to if relative_to is not None else evidence_root
        if base is None:
            return None
        path = base / path
    try:
        resolved = path.resolve(strict=True)
        if (
            root_relative
            and evidence_root is not None
            and not resolved.is_relative_to(evidence_root.resolve())
        ):
            return None
        if not resolved.is_file():
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _episode_row_for_binding(
    episode_store_bytes: bytes, *, episode_id: Any
) -> tuple[dict[str, Any] | None, str]:
    """Parse a JSONL store and select exactly one schema-valid source episode row."""
    if not isinstance(episode_id, str) or not episode_id.strip():
        return None, "episode_identity_mismatch"
    matches: list[dict[str, Any]] = []
    try:
        text = episode_store_bytes.decode("utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return None, "episode_store_malformed"
            if value.get("episode_id") == episode_id:
                matches.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "episode_store_malformed"
    if len(matches) != 1:
        return None, "episode_identity_missing_or_ambiguous"
    episode = matches[0]
    if list(_episode_schema_validator().iter_errors(episode)):
        return None, "episode_store_episode_schema_invalid"
    return episode, ""


def _load_episode_schema() -> dict[str, Any]:
    """Load the benchmark's canonical v1 episode JSON Schema."""
    return json.loads(_EPISODE_SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _episode_schema_validator() -> Draft202012Validator:
    """Reuse the canonical episode validator across candidate classifications."""
    return Draft202012Validator(_load_episode_schema())


@lru_cache(maxsize=1)
def _target_planner_replay_schema_validator() -> Draft202012Validator:
    """Reuse the versioned target-planner replay result validator."""
    schema = json.loads(_TARGET_PLANNER_REPLAY_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _episode_row_binding_problem(
    episode: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    expected_episode_id: Any = None,
    compare_route_outcome: bool = True,
) -> str | None:
    """Check source episode identity, clean runtime status, and route outcome against its row."""
    outcome = episode.get("outcome")
    integrity = episode.get("integrity")
    metadata = episode.get("algorithm_metadata")
    termination = episode.get("termination_reason")
    expected_id = source.get("episode_id") if expected_episode_id is None else expected_episode_id
    if (
        episode.get("episode_id") != expected_id
        or episode.get("scenario_id") != source.get("scenario_id")
        or episode.get("seed") != source.get("seed")
        or episode.get("algo") != source.get("planner_id")
        or episode.get("git_hash") != source.get("source_commit")
    ):
        return "episode_identity_mismatch"
    episode_horizon = episode.get("horizon")
    if (
        not isinstance(episode_horizon, int)
        or isinstance(episode_horizon, bool)
        or episode_horizon != source.get("horizon_steps")
    ):
        return "episode_horizon_mismatch"
    if (
        not isinstance(outcome, Mapping)
        or type(outcome.get("route_complete")) is not bool
        or (compare_route_outcome and outcome["route_complete"] is not source.get("route_complete"))
        or not isinstance(termination, str)
        or termination not in TERMINATION_REASONS
        or episode.get("status") != status_from_termination_reason(termination)
        or outcome_contradictions(
            termination_reason=termination,
            outcome=outcome,
            metrics=episode.get("metrics") if isinstance(episode.get("metrics"), Mapping) else None,
        )
    ):
        return "episode_outcome_mismatch"
    contradictions = integrity.get("contradictions") if isinstance(integrity, Mapping) else None
    if not isinstance(contradictions, list) or contradictions:
        return "episode_integrity_invalid_or_contradictory"
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("status") != "ok"
        or runtime_fallback_or_degraded_marker(dict(episode)) is not None
    ):
        return "episode_runtime_unavailable_or_degraded"
    return None


def _replay_sidecar_binding_problem(
    sidecar: Mapping[str, Any], source: Mapping[str, Any], store_sha256: str
) -> str | None:
    """Match canonical replay provenance fields to the selected episode and normalized run."""
    if (
        sidecar.get("episode_id") != source.get("episode_id")
        or sidecar.get("scenario_id") != source.get("scenario_id")
        or sidecar.get("seed") != source.get("seed")
        or sidecar.get("planner_key") != source.get("planner_id")
        or sidecar.get("repo_commit") != source.get("source_commit")
        or sidecar.get("determinism_check_status") != source.get("determinism_check_status")
        or sidecar.get("resimulated") is not source.get("resimulated")
        or sidecar.get("source_episodes_jsonl_sha256") != store_sha256
        or source.get("determinism_check_status") != "pass"
        or source.get("resimulated") is not True
        or not isinstance(sidecar.get("replay_command"), str)
        or not sidecar.get("replay_command", "").strip()
        or not isinstance(sidecar.get("target_planner_replay_result_path"), str)
        or not sidecar.get("target_planner_replay_result_path", "").strip()
        or not isinstance(sidecar.get("target_planner_replay_result_sha256"), str)
        or _SHA256.fullmatch(sidecar.get("target_planner_replay_result_sha256", "")) is None
    ):
        return "replay_sidecar_identity_or_outcome_mismatch"
    return None


def _target_planner_replay_result_binding(
    result: Mapping[str, Any] | None,
    *,
    source: Mapping[str, Any],
    source_store_path: Path,
    store_sha256: str,
    result_path: Path,
) -> tuple[str | None, dict[str, Any]]:
    """Bind a target-planner replay result to its separate canonical output episode."""
    binding: dict[str, Any] = {
        "schema_version": result.get("schema_version") if isinstance(result, Mapping) else None,
        "episode_id": result.get("episode_id") if isinstance(result, Mapping) else None,
        "route_complete": result.get("route_complete") if isinstance(result, Mapping) else None,
    }
    if not isinstance(result, Mapping):
        return "target_planner_result_missing_or_malformed", binding
    errors = list(_target_planner_replay_schema_validator().iter_errors(result))
    if errors:
        binding["status"] = "schema_invalid"
        return "target_planner_result_schema_invalid", binding
    identity_pairs = (
        ("episode_id", "episode_id"),
        ("scenario_id", "scenario_id"),
        ("seed", "seed"),
        ("planner_id", "planner_id"),
        ("source_commit", "source_commit"),
    )
    if any(
        result.get(result_key) != source.get(source_key)
        for result_key, source_key in identity_pairs
    ):
        binding["status"] = "identity_mismatch"
        return "target_planner_result_identity_mismatch", binding
    if result.get("source_episodes_jsonl_sha256", "").lower() != store_sha256.lower():
        binding["status"] = "source_digest_mismatch"
        return "target_planner_result_source_episode_mismatch", binding
    if result.get("route_complete") is not source.get("route_complete"):
        binding["status"] = "outcome_mismatch"
        return "target_planner_result_outcome_mismatch", binding
    if result.get("run_status") != "ok" or result.get("fallback_or_degraded") is not False:
        binding["status"] = "runtime_unavailable_or_degraded"
        return "target_planner_result_runtime_unavailable_or_degraded", binding
    replay_reason, replay_binding = _target_replay_episode_binding(
        result,
        source=source,
        source_store_path=source_store_path,
        result_path=result_path,
    )
    binding["replay_episode_binding"] = replay_binding
    binding["run_context_binding"] = replay_binding.get("run_context_binding", {})
    binding["producer_provenance_binding"] = replay_binding.get("producer_provenance_binding", {})
    if replay_reason is not None:
        binding["status"] = "replay_episode_invalid"
        return replay_reason, binding
    binding["status"] = "valid"
    return None, binding


def _target_replay_episode_binding(
    result: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    source_store_path: Path,
    result_path: Path,
) -> tuple[str | None, dict[str, Any]]:
    """Validate the distinct canonical episode store produced by target replay."""
    replay_store_path = _resolve_evidence_path(
        result.get("replay_episodes_jsonl_path"),
        evidence_root=None,
        relative_to=result_path.parent,
    )
    if replay_store_path is None:
        return "target_planner_result_replay_episode_store_missing_or_unreadable", {
            "status": "unavailable"
        }
    try:
        replay_store_bytes = replay_store_path.read_bytes()
    except OSError:
        return "target_planner_result_replay_episode_store_missing_or_unreadable", {
            "status": "unavailable",
            "replay_episode_store_path": replay_store_path.as_posix(),
        }
    replay_store_sha256 = hashlib.sha256(replay_store_bytes).hexdigest()
    binding = {
        "status": "checked",
        "replay_episode_store_path": replay_store_path.as_posix(),
        "replay_episode_store_sha256": replay_store_sha256,
        "replay_episode_id": result.get("replay_episode_id"),
    }
    if result.get("replay_episodes_jsonl_sha256", "").lower() != replay_store_sha256.lower():
        return "target_planner_result_replay_episode_store_digest_mismatch", binding
    if replay_store_path.resolve() == source_store_path.resolve():
        return "target_planner_result_replay_episode_store_not_distinct", binding
    replay_episode, parse_reason = _episode_row_for_binding(
        replay_store_bytes, episode_id=result.get("replay_episode_id")
    )
    if replay_episode is None:
        return "target_planner_result_replay_episode_" + parse_reason, binding
    row_reason = _episode_row_binding_problem(
        replay_episode,
        source,
        expected_episode_id=result.get("replay_episode_id"),
    )
    if row_reason is not None:
        return "target_planner_result_replay_" + row_reason, binding
    replay_outcome = replay_episode.get("outcome")
    if replay_outcome.get("route_complete") is not result.get(
        "route_complete"
    ) or replay_episode.get("termination_reason") != result.get("termination_reason"):
        return "target_planner_result_replay_episode_outcome_mismatch", binding
    provenance_path = _resolve_evidence_path(
        result.get("replay_provenance_manifest_path"),
        evidence_root=None,
        relative_to=result_path.parent,
    )
    if provenance_path is None:
        return "target_planner_result_producer_provenance_unavailable", {
            **binding,
            "producer_provenance_binding": {"status": "unavailable"},
            "run_context_binding": {"status": "unavailable"},
        }
    provenance_problem, producer_binding = _producer_provenance_binding(
        store_path=replay_store_path,
        store_bytes=replay_store_bytes,
        episode=replay_episode,
        source=source,
        manifest_path=provenance_path,
        expected_manifest_sha256=result.get("replay_provenance_manifest_sha256"),
        required=True,
    )
    binding["producer_provenance_binding"] = producer_binding
    binding["run_context_binding"] = producer_binding.get("run_context_binding", {})
    if provenance_problem is not None:
        return "target_planner_result_" + provenance_problem, binding
    binding["status"] = "valid"
    return None, binding


def _execution_problem(
    source: Any,
    role: str,
    case_id: str,
    scenario_id: str | None,
    artifact_sha256: str | None,
    effective_input_sha256: str | None,
    requires_effective_input_binding: bool,
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
    effective_digest = source.get("effective_input_sha256")
    if requires_effective_input_binding and (
        effective_input_sha256 is None
        or not isinstance(effective_digest, str)
        or _SHA256.fullmatch(effective_digest) is None
        or effective_digest.lower() != effective_input_sha256.lower()
        or source.get("effective_input_identity_stable") is not True
    ):
        return f"{role}_execution_effective_input_identity_missing_or_mismatch"
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
    # Every named run must resolve to one canonical source episode row. The replay role's
    # ``evidence_ref`` identifies its provenance sidecar; reference/target refs identify JSONL.
    required.update({"episode_id", "source_episodes_jsonl_sha256"})
    if not isinstance(source, Mapping) or not required.issubset(source):
        return f"{role}_execution_provenance_incomplete"
    if scenario_id is None:
        return f"{role}_execution_scenario_identity_unbound"
    if source["case_id"] != case_id or source["scenario_id"] != scenario_id:
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
    fields = (
        "scenario_id",
        "scenario_variant",
        "planner_id",
        "evidence_ref",
        "episode_id",
    )
    return all(isinstance(source[key], str) and source[key].strip() for key in fields)


def _execution_digest_fields_valid(source: Mapping[str, Any]) -> bool:
    fields = (
        "scenario_sha256",
        "robot_model_sha256",
        "simulator_config_sha256",
        "planner_config_sha256",
        "environment_sha256",
    )
    fields += ("source_episodes_jsonl_sha256",)
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
    potential_matched_failure = (
        reference is not None
        and target is not None
        and reference["route_complete"]
        and not target["route_complete"]
        and reference["planner_id"] != target["planner_id"]
    )
    if potential_matched_failure and not matched_failure:
        reasons.append("matched_reference_target_failure_run_context_unbound")
        reasons.append("planner_specific_failure_attribution_unconfirmed")
    if matched_failure:
        reasons.append("named_execution_completed_original_case")
        if replay is None:
            reasons.append("planner_specific_failure_replay_missing_or_invalid")
            reasons.append("planner_specific_failure_attribution_unconfirmed")
            return EMPIRICALLY_FEASIBLE
        replay_mismatch = _replay_binding_mismatch(replay, target)
        if replay_mismatch is not None:
            reasons.append(replay_mismatch)
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
    left_binding = left.get("_producer_provenance_binding")
    right_binding = right.get("_producer_provenance_binding")
    if not isinstance(left_binding, Mapping) or not isinstance(right_binding, Mapping):
        return False
    if (
        left_binding.get("status") != "valid"
        or right_binding.get("status") != "valid"
        or left_binding.get("run_context_binding", {}).get("status") != "valid"
        or right_binding.get("run_context_binding", {}).get("status") != "valid"
    ):
        return False
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
    return (
        all(left.get(key) == right.get(key) for key in keys)
        and left_binding.get("execution_context_sha256")
        == right_binding.get("execution_context_sha256")
        and left_binding.get("case_identity_sha256") is not None
        and left_binding.get("case_identity_sha256") == right_binding.get("case_identity_sha256")
        and left_binding.get("simulator_settings_sha256")
        == right_binding.get("simulator_settings_sha256")
    )


def _same_planner_configuration(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Bind a target failure replay to the target planner config and checkpoint."""
    left_binding = left.get("_producer_provenance_binding")
    right_binding = right.get("_producer_provenance_binding")
    if not isinstance(left_binding, Mapping) or not isinstance(right_binding, Mapping):
        return False
    return (
        left_binding.get("planner_config_sha256") is not None
        and left_binding.get("planner_config_sha256") == right_binding.get("planner_config_sha256")
        and left_binding.get("row_config_hash") == right_binding.get("row_config_hash")
        and left_binding.get("planner_checkpoint_status")
        == right_binding.get("planner_checkpoint_status")
        == "not_applicable"
    )


def _same_replay_source_episode(replay: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    """Bind canonical replay-sidecar identity and source artifact to the target row."""
    return (
        replay.get("episode_id") == target.get("episode_id")
        and isinstance(replay.get("source_episodes_jsonl_sha256"), str)
        and isinstance(target.get("source_episodes_jsonl_sha256"), str)
        and replay["source_episodes_jsonl_sha256"].lower()
        == target["source_episodes_jsonl_sha256"].lower()
    )


def _replay_binding_mismatch(replay: Mapping[str, Any], target: Mapping[str, Any]) -> str | None:
    """Return the first failed binding required to attribute a planner-specific failure."""
    if replay["planner_id"] != target["planner_id"]:
        return "planner_specific_failure_replay_wrong_planner"
    if (
        replay.get("_execution_context_binding_status") != "valid"
        or target.get("_execution_context_binding_status") != "valid"
    ):
        return "planner_specific_failure_replay_run_context_unbound"
    if not _same_case(replay, target):
        return "planner_specific_failure_replay_case_mismatch"
    if not _same_planner_configuration(replay, target):
        return "planner_specific_failure_replay_configuration_mismatch"
    if not _same_replay_source_episode(replay, target):
        return "planner_specific_failure_replay_source_episode_mismatch"
    replay_binding = replay.get("_producer_provenance_binding")
    target_binding = target.get("_producer_provenance_binding")
    if (
        not isinstance(replay_binding, Mapping)
        or not isinstance(target_binding, Mapping)
        or not isinstance(replay_binding.get("run_id"), str)
        or not isinstance(target_binding.get("run_id"), str)
    ):
        return "planner_specific_failure_replay_producer_run_id_unavailable"
    if replay_binding["run_id"] == target_binding["run_id"]:
        return "planner_specific_failure_replay_reused_target_producer_run"
    return None


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
