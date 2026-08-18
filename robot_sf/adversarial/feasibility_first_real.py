"""Bounded feasibility-first diagnostics on a real Robot SF manifest.

The fixture protocol in :mod:`robot_sf.adversarial.feasibility_first` deliberately stops
before the simulator.  This module is the next, still diagnostic-only, step for issue #7340:
it samples a fixed candidate pool with the canonical random adversarial sampler, sends each
candidate through the existing validation -> scenario-certification -> benchmark-runner path,
and compares uniform selection with hierarchical risk-feature selection on the same pool.

The report keeps rejected, unavailable, and degraded rows visible.  Native execution is
evidence that the local runtime accepted a candidate, not evidence for a benchmark, safety,
paper, or source-method claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.adversarial.certification import failed_status
from robot_sf.adversarial.config import SearchConfig
from robot_sf.adversarial.feasibility_first import (
    CHECK_NAMES,
    FeasibilityCandidate,
    FeasibilityCheck,
    HierarchicalScenarioValue,
    rank_feasible_candidates,
    sample_seeded_uniform,
)
from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.samplers import build_sampler
from robot_sf.adversarial.scenario_manifest import compute_control_hash
from robot_sf.adversarial.search import production_candidate_evaluator
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

SCHEMA_VERSION = "feasibility_first_real_manifest.v1"
EVIDENCE_TIER = "diagnostic-only"
CLAIM_BOUNDARY = (
    "diagnostic-only real-manifest comparison: native execution and observed safety events; "
    "no simulator, planner, safety, benchmark, paper, or source-method claim"
)
BASELINE_ID = "existing_adversarial_random_sampler.v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    Path(__file__).parent.parent
    / "benchmark"
    / "schemas"
    / ("feasibility_first_real_manifest.v1.json")
)
_CONFIG_KEYS = {
    "schema_version",
    "claim_boundary",
    "evidence_tier",
    "scenario_template",
    "search_space",
    "scenario_family",
    "policy",
    "candidate_pool_budget",
    "sample_budget",
    "sampling_seed",
    "criticality_threshold",
    "horizon",
    "dt",
    "workers",
    "benchmark_profile",
    "require_certification",
    "baseline",
    "domain_approval",
}


class RealManifestError(ValueError):
    """Raised when a real-manifest diagnostic cannot be interpreted safely."""


def _sha256_bytes(raw: bytes) -> str:
    """Return the SHA-256 digest for raw file content."""
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with stable key ordering."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(raw.encode("utf-8"))


def _repo_relative(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_input_path(value: object, *, config_path: Path, field: str) -> Path:
    """Resolve an input path against the repository and manifest directory."""
    if not isinstance(value, str) or not value.strip():
        raise RealManifestError(f"{field} must be a non-empty path")
    raw_path = Path(value)
    if raw_path.is_absolute():
        candidate_paths = (raw_path,)
    else:
        candidate_paths = (_REPO_ROOT / raw_path, config_path.parent / raw_path)
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate.resolve()
    raise RealManifestError(f"{field} does not resolve to a file: {value}")


def _load_config(config_path: Path) -> tuple[bytes, dict[str, Any]]:  # noqa: C901, PLR0912
    """Load and validate the config-first real-manifest declaration."""
    raw = config_path.read_bytes()
    payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise RealManifestError("real-manifest config must be a mapping")
    unknown = set(payload) - _CONFIG_KEYS
    missing = _CONFIG_KEYS - set(payload)
    if unknown:
        raise RealManifestError(f"real-manifest config has unknown fields: {sorted(unknown)}")
    if missing:
        raise RealManifestError(f"real-manifest config is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise RealManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    if payload["claim_boundary"] != CLAIM_BOUNDARY:
        raise RealManifestError("claim_boundary does not match the diagnostic contract")
    if payload["evidence_tier"] != EVIDENCE_TIER:
        raise RealManifestError("evidence_tier must be diagnostic-only")
    if payload["baseline"] != BASELINE_ID:
        raise RealManifestError(f"baseline must be {BASELINE_ID!r}")
    family = payload["scenario_family"]
    policy = payload["policy"]
    if not isinstance(family, str) or not family.strip():
        raise RealManifestError("scenario_family must be non-empty")
    if not isinstance(policy, str) or not policy.strip():
        raise RealManifestError("policy must be non-empty")
    approval = payload["domain_approval"]
    if not isinstance(approval, dict) or approval.get("status") != "required":
        raise RealManifestError("domain_approval.status must remain 'required'")
    for key in ("candidate_pool_budget", "sample_budget", "sampling_seed", "horizon", "workers"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RealManifestError(f"{key} must be a positive integer")
    if payload["sample_budget"] > payload["candidate_pool_budget"]:
        raise RealManifestError("sample_budget must not exceed candidate_pool_budget")
    for key in ("criticality_threshold", "dt"):
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RealManifestError(f"{key} must be finite")
    if not 0.0 <= float(payload["criticality_threshold"]) <= 1.0:
        raise RealManifestError("criticality_threshold must be between 0 and 1")
    if float(payload["dt"]) <= 0.0:
        raise RealManifestError("dt must be positive")
    if (
        not isinstance(payload["require_certification"], bool)
        or not payload["require_certification"]
    ):
        raise RealManifestError("require_certification must be true for a real-manifest probe")
    return raw, payload


def _route_certificates(certification_status: Any) -> list[dict[str, Any]]:
    """Extract route certificates from the canonical certification adapter."""
    details = getattr(certification_status, "details", {})
    if not isinstance(details, dict):
        return []
    raw_certificates = details.get("certificates")
    if not isinstance(raw_certificates, list):
        return []
    routes: list[dict[str, Any]] = []
    for certificate in raw_certificates:
        if not isinstance(certificate, dict):
            continue
        route_certificates = certificate.get("route_certificates")
        if isinstance(route_certificates, list):
            routes.extend(route for route in route_certificates if isinstance(route, dict))
    return routes


def _certificate_evidence(certification_status: Any) -> dict[str, Any]:
    """Return compact certificate evidence without copying planner internals."""
    routes = _route_certificates(certification_status)
    return {
        "schema_version": getattr(certification_status, "schema_version", "scenario_cert.v1"),
        "status": getattr(certification_status, "status", "unknown"),
        "reason": getattr(certification_status, "reason", "unknown"),
        "route_count": len(routes),
        "route_classifications": ",".join(str(route.get("classification")) for route in routes),
        "route_eligibility": ",".join(str(route.get("benchmark_eligibility")) for route in routes),
    }


def _route_check_failures(routes: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """Return route ids failing kinematic and geometry/traffic evidence."""
    kinematic_failures: list[str] = []
    geometry_failures: list[str] = []
    for route in routes:
        route_id = str(route.get("route_id", "unknown"))
        checks = route.get("checks") if isinstance(route.get("checks"), dict) else {}
        kinodynamic = checks.get("kinodynamic")
        if not isinstance(kinodynamic, dict) or kinodynamic.get("command_limits_valid") is not True:
            kinematic_failures.append(route_id)
        if checks.get("inflated_collision_free_path") is not True:
            geometry_failures.append(route_id)
        obstacle = checks.get("simulator_obstacle_collision")
        if not isinstance(obstacle, dict) or obstacle.get("validated") is not True:
            geometry_failures.append(route_id)
        elif obstacle.get("collides_obstacle") is True:
            geometry_failures.append(route_id)
        dynamic = checks.get("dynamic")
        if isinstance(dynamic, dict) and dynamic.get("static_blocking_pedestrian_ids"):
            geometry_failures.append(route_id)
    return kinematic_failures, geometry_failures


def _certificate_checks(
    certification_status: Any,
) -> tuple[FeasibilityCheck, FeasibilityCheck]:
    """Map scenario-cert route evidence to kinematic and geometry checks."""
    status = str(getattr(certification_status, "status", "unknown")).strip().lower()
    evidence = _certificate_evidence(certification_status)
    routes = _route_certificates(certification_status)
    if status in {"not_available", "unavailable"}:
        return (
            FeasibilityCheck(
                "kinematic_reachability",
                "unavailable",
                "scenario_cert.v1 was unavailable",
                evidence,
            ),
            FeasibilityCheck(
                "geometry_traffic",
                "unavailable",
                "scenario_cert.v1 was unavailable",
                evidence,
            ),
        )
    if status != "passed" or not routes:
        reason = str(getattr(certification_status, "reason", "scenario certification failed"))
        return (
            FeasibilityCheck("kinematic_reachability", "fail", reason, evidence),
            FeasibilityCheck("geometry_traffic", "fail", reason, evidence),
        )

    ineligible = [
        str(route.get("route_id", "unknown"))
        for route in routes
        if route.get("benchmark_eligibility") != "eligible"
    ]
    kinematic_failures, geometry_failures = _route_check_failures(routes)

    if ineligible:
        kinematic_failures.extend(ineligible)
        geometry_failures.extend(ineligible)
    if kinematic_failures:
        kinematic = FeasibilityCheck(
            "kinematic_reachability",
            "fail",
            "scenario-cert route eligibility or kinodynamic checks failed",
            {**evidence, "failed_routes": ",".join(sorted(set(kinematic_failures)))},
        )
    else:
        kinematic = FeasibilityCheck(
            "kinematic_reachability",
            "pass",
            "scenario-cert route passed kinematic checks",
            {
                **evidence,
                "passed_routes": ",".join(str(route.get("route_id")) for route in routes),
            },
        )
    if geometry_failures:
        geometry = FeasibilityCheck(
            "geometry_traffic",
            "fail",
            "scenario-cert geometry or traffic checks failed",
            {**evidence, "failed_routes": ",".join(sorted(set(geometry_failures)))},
        )
    else:
        geometry = FeasibilityCheck(
            "geometry_traffic",
            "pass",
            "scenario-cert geometry and traffic checks passed",
            {
                **evidence,
                "passed_routes": ",".join(str(route.get("route_id")) for route in routes),
            },
        )
    return kinematic, geometry


def _behavior_check(
    scenario_yaml_path: Path | None,
    *,
    config: SearchConfig,
) -> FeasibilityCheck:
    """Validate the real loader-backed robot/pedestrian binding."""
    evidence: dict[str, Any] = {"source": "scenario_loader.runtime_config"}
    if scenario_yaml_path is None or not scenario_yaml_path.is_file():
        return FeasibilityCheck(
            "behavioral_consistency",
            "unavailable",
            "materialized scenario was not available",
            evidence,
        )
    try:
        payload = yaml.safe_load(scenario_yaml_path.read_text(encoding="utf-8")) or {}
        scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
        scenario = scenarios[0] if isinstance(scenarios, list) and scenarios else None
        if not isinstance(scenario, dict):
            raise ValueError("materialized scenario has no first scenario")
        runtime_config = build_robot_config_from_scenario(
            scenario, scenario_path=scenario_yaml_path
        )
        map_def = next(iter(runtime_config.map_pool.map_defs.values()))
        pedestrian_id = config.search_space.pedestrian_id
        if not pedestrian_id:
            raise ValueError("search-space pedestrian.id is required")
        pedestrians = {ped.id: ped for ped in map_def.single_pedestrians}
        pedestrian = pedestrians.get(pedestrian_id)
        if pedestrian is None:
            raise ValueError(f"runtime map has no pedestrian {pedestrian_id!r}")
        metadata = scenario.get("metadata")
        candidate = (
            metadata.get("adversarial_candidate", {}) if isinstance(metadata, Mapping) else None
        )
        if not isinstance(metadata, Mapping) or not isinstance(candidate, dict):
            raise ValueError("candidate provenance metadata is missing or malformed")
        expected_speed = float(candidate["pedestrian_speed_mps"])
        expected_delay = float(candidate["spawn_time_s"])
        if pedestrian.speed_m_s is None or not math.isclose(
            float(pedestrian.speed_m_s), expected_speed, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("candidate pedestrian speed is not runtime-effective")
        if not math.isclose(
            float(pedestrian.start_delay_s), expected_delay, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("candidate pedestrian start timing is not runtime-effective")
        if not map_def.robot_routes or not map_def.ped_routes:
            raise ValueError("runtime map lacks robot or pedestrian routes")
        evidence.update(
            {
                "pedestrian_id": pedestrian_id,
                "pedestrian_route_mode": config.search_space.pedestrian_route_mode,
                "robot_route_count": len(map_def.robot_routes),
                "pedestrian_route_count": len(map_def.ped_routes),
                "speed_mps": float(pedestrian.speed_m_s),
                "start_delay_s": float(pedestrian.start_delay_s),
                "pedestrian_delay_runtime_binding": (
                    "not_bound"
                    if config.search_space.pedestrian_route_mode == "template"
                    else "wait_rule"
                ),
            }
        )
    except (KeyError, OSError, TypeError, ValueError, StopIteration) as exc:
        return FeasibilityCheck(
            "behavioral_consistency",
            "fail",
            f"scenario-loader behavioral binding failed: {exc}",
            {**evidence, "error": str(exc)},
        )
    return FeasibilityCheck(
        "behavioral_consistency",
        "pass",
        "scenario loader accepted the robot/pedestrian runtime binding",
        evidence,
    )


def _simulator_check(
    evaluation: Any, *, certification_status: Any
) -> tuple[FeasibilityCheck, dict[str, Any]]:
    """Classify the canonical benchmark-runner execution without hiding degradation."""
    runtime: dict[str, Any] = {
        "scenario_yaml_path": (
            _repo_relative(evaluation.scenario_yaml_path)
            if getattr(evaluation, "scenario_yaml_path", None)
            else None
        ),
        "bundle_path": (
            _repo_relative(evaluation.bundle_path)
            if getattr(evaluation, "bundle_path", None)
            else None
        ),
        "episode_record_path": (
            _repo_relative(evaluation.episode_record_path)
            if getattr(evaluation, "episode_record_path", None)
            else None
        ),
        "objective_value": getattr(evaluation, "objective_value", None),
        "error": getattr(evaluation, "error", None),
    }
    cert_status = str(getattr(certification_status, "status", "unknown")).strip().lower()
    attribution = getattr(evaluation, "failure_attribution", None)
    details = getattr(attribution, "details", {}) if attribution is not None else {}
    if not isinstance(details, dict):
        details = {}
    execution_mode = str(details.get("execution_mode", "unknown")).strip().lower()
    availability_status = str(details.get("availability_status", "unknown")).strip().lower()
    runtime.update(
        {
            "execution_mode": execution_mode,
            "availability_status": availability_status,
            "readiness_status": details.get("readiness_status"),
            "primary_failure": getattr(attribution, "primary_failure", None),
        }
    )
    if cert_status != "passed":
        return (
            FeasibilityCheck(
                "simulator_validity",
                "unavailable",
                "simulator was not run after certification rejection or unavailability",
                {"certification_status": cert_status},
            ),
            runtime,
        )
    record_path = getattr(evaluation, "episode_record_path", None)
    record_error: str | None = None
    if isinstance(record_path, Path):
        try:
            record = read_first_jsonl_record(record_path)
        except (OSError, TypeError, ValueError) as exc:
            record = None
            record_error = str(exc)
            runtime["record_error"] = record_error
    else:
        record = None
    if (
        record
        and execution_mode in {"native", "adapter", "mixed"}
        and availability_status == "available"
    ):
        runtime["record_status"] = record.get("status")
        runtime["termination_reason"] = record.get("termination_reason")
        return (
            FeasibilityCheck(
                "simulator_validity",
                "pass",
                "canonical benchmark runner produced an available episode record",
                {
                    "execution_mode": execution_mode,
                    "availability_status": availability_status,
                    "record_status": record.get("status"),
                    "termination_reason": record.get("termination_reason"),
                },
            ),
            runtime,
        )
    if execution_mode in {"fallback", "degraded", "unknown"} or availability_status != "available":
        return (
            FeasibilityCheck(
                "simulator_validity",
                "unavailable",
                "simulator execution was unavailable or degraded",
                {"execution_mode": execution_mode, "availability_status": availability_status},
            ),
            runtime,
        )
    return (
        FeasibilityCheck(
            "simulator_validity",
            "fail",
            "canonical benchmark runner did not produce a valid episode record",
            {
                "error": runtime.get("error") or "missing episode record",
                **({"record_error": record_error} if record_error else {}),
            },
        ),
        runtime,
    )


def _risk_value(
    checks: Sequence[FeasibilityCheck],
    certification_status: Any,
    *,
    candidate_id: str,
) -> tuple[HierarchicalScenarioValue, tuple[float, ...]]:
    """Derive bounded pre-simulation risk features from real certificate evidence."""
    routes = _route_certificates(certification_status)
    ratios: list[float] = []
    clearances: list[float] = []
    turns: list[float] = []
    for route in routes:
        route_checks = route.get("checks") if isinstance(route.get("checks"), dict) else {}
        ratio = route_checks.get("path_length_ratio")
        clearance = route_checks.get("minimum_static_clearance_m")
        turn_count = route_checks.get("planned_turn_count")
        if isinstance(ratio, (int, float)) and math.isfinite(ratio):
            ratios.append(float(ratio))
        if isinstance(clearance, (int, float)) and math.isfinite(clearance):
            clearances.append(float(clearance))
        if isinstance(turn_count, (int, float)) and math.isfinite(turn_count):
            turns.append(float(turn_count))
    ratio = max(ratios, default=1.0)
    clearance = min(clearances, default=3.0)
    turn_count = max(turns, default=0.0)
    kinematic_criticality = min(1.0, max(0.0, (ratio - 1.0) / 1.5))
    controllability_risk = min(1.0, max(0.0, turn_count / 8.0) + max(0.0, 1.0 - clearance / 3.0))
    digest_fraction = int(candidate_id[-8:], 16) / float(0xFFFFFFFF)
    diversity = min(1.0, max(0.0, digest_fraction))
    value = HierarchicalScenarioValue(
        kinematic_criticality=kinematic_criticality,
        controllability_risk=controllability_risk,
        diversity=diversity,
    )
    status_vector = tuple(float(check.status == "pass") for check in checks)
    return value, (
        value.kinematic_criticality,
        value.controllability_risk,
        value.diversity,
        *status_vector,
    )


def _method_summary(
    selected: Sequence[FeasibilityCandidate],
    *,
    budget: int,
    threshold: float,
    runtime_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize one method with explicit valid and safety denominators."""
    feasible = [candidate for candidate in selected if candidate.feasible]
    safety_events = Counter(
        str(runtime_by_id[candidate.candidate_id].get("primary_failure") or "unknown")
        for candidate in feasible
        if runtime_by_id.get(candidate.candidate_id, {}).get("execution_mode")
        in {"native", "adapter", "mixed"}
    )
    return {
        "status": "available",
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "selected_count": len(selected),
        "valid_scenario_rate": len(feasible) / float(budget),
        "discovery_yield": sum(
            candidate.value.kinematic_criticality >= threshold for candidate in feasible
        ),
        "unique_scenario_families": len({candidate.scenario_family for candidate in feasible}),
        "mean_diversity": (
            sum(candidate.value.diversity for candidate in feasible) / len(feasible)
            if feasible
            else None
        ),
        "rejected_count": len(selected) - len(feasible),
        "rejection_reasons": dict(
            sorted(
                Counter(
                    reason.split(":", maxsplit=2)[0]
                    for candidate in selected
                    for reason in candidate.rejection_reasons
                ).items()
            )
        ),
        "safety_event_severity": {
            "status": "available" if safety_events else "unavailable",
            "counts": dict(sorted(safety_events.items())),
            "denominator": len(feasible),
            "reason": "observed episode attribution; not a safety rate or guarantee",
        },
    }


def _unavailable_method(reason: str) -> dict[str, Any]:
    """Return a schema-compatible unavailable method summary."""
    return {
        "status": "unavailable",
        "reason": reason,
        "selected_candidate_ids": [],
        "selected_count": 0,
        "valid_scenario_rate": None,
        "discovery_yield": None,
        "unique_scenario_families": None,
        "mean_diversity": None,
        "rejected_count": None,
        "rejection_reasons": {},
        "safety_event_severity": {
            "status": "unavailable",
            "counts": {},
            "denominator": 0,
            "reason": reason,
        },
    }


def _candidate_record(
    candidate: FeasibilityCandidate,
    *,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Add execution provenance to a typed candidate record."""
    return {**candidate.to_dict(), "runtime": dict(runtime)}


def _config_search(
    manifest_path: Path,
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
) -> SearchConfig:
    """Build the canonical adversarial search config from the real manifest."""
    scenario_template = _resolve_input_path(
        payload["scenario_template"], config_path=manifest_path, field="scenario_template"
    )
    search_space = _resolve_input_path(
        payload["search_space"], config_path=manifest_path, field="search_space"
    )
    config = SearchConfig.from_files(
        policy=str(payload["policy"]),
        scenario_template=scenario_template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=output_dir,
        budget=int(payload["candidate_pool_budget"]),
        seed=int(payload["sampling_seed"]),
        horizon=int(payload["horizon"]),
        dt=float(payload["dt"]),
        workers=int(payload["workers"]),
        require_certification=True,
        benchmark_profile=str(payload["benchmark_profile"]),
    )
    config.validate()
    if not config.search_space.pedestrian_id:
        raise RealManifestError("real-manifest search space must declare pedestrian.id")
    return config


def _input_digests(manifest_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Collect content digests for the manifest inputs."""
    scenario_template = _resolve_input_path(
        payload["scenario_template"], config_path=manifest_path, field="scenario_template"
    )
    search_space = _resolve_input_path(
        payload["search_space"], config_path=manifest_path, field="search_space"
    )
    result: dict[str, Any] = {
        "manifest": {"path": _repo_relative(manifest_path), "sha256": _sha256_file(manifest_path)},
        "scenario_template": {
            "path": _repo_relative(scenario_template),
            "sha256": _sha256_file(scenario_template),
        },
        "search_space": {
            "path": _repo_relative(search_space),
            "sha256": _sha256_file(search_space),
        },
    }
    template_payload = yaml.safe_load(scenario_template.read_bytes()) or {}
    scenarios = template_payload.get("scenarios") if isinstance(template_payload, dict) else None
    first = scenarios[0] if isinstance(scenarios, list) and scenarios else {}
    map_file = first.get("map_file") if isinstance(first, dict) else None
    if isinstance(map_file, str):
        map_path = (scenario_template.parent / map_file).resolve()
        if map_path.is_file():
            result["map"] = {"path": _repo_relative(map_path), "sha256": _sha256_file(map_path)}
    return result


def run_real_manifest_diagnostic(
    config_path: Path,
    *,
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the bounded real-manifest diagnostic and optionally persist its report."""
    config_path = config_path.resolve()
    raw_config, payload = _load_config(config_path)
    actual_output_dir = (output_dir or _REPO_ROOT / "output/issue_7340_real_manifest").resolve()
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    search_config = _config_search(config_path, payload, output_dir=actual_output_dir)
    digests = _input_digests(config_path, payload)

    sampler = build_sampler(
        "random", search_config.search_space, seed=int(payload["sampling_seed"])
    )
    candidates: list[FeasibilityCandidate] = []
    runtime_by_id: dict[str, dict[str, Any]] = {}
    evaluator = production_candidate_evaluator()
    for index in range(int(payload["candidate_pool_budget"])):
        candidate = sampler.sample()
        candidate_id = f"{payload['scenario_family']}:{index:04d}:{compute_control_hash(candidate)}"
        try:
            evaluation = evaluator(search_config, candidate, index)
        except Exception as exc:  # noqa: BLE001 - diagnostic rows must remain visible.
            evaluation = None
            certification_status = failed_status(
                "candidate production pipeline raised", details={"error": repr(exc)}
            )
            scenario_path = None
            runtime = {"error": repr(exc), "execution_mode": "unknown"}
        else:
            certification_status = evaluation.certification_status
            scenario_path = evaluation.scenario_yaml_path
            simulator_check, runtime = _simulator_check(
                evaluation, certification_status=certification_status
            )
        if evaluation is None:
            simulator_check = FeasibilityCheck(
                "simulator_validity",
                "unavailable",
                "candidate production pipeline raised before simulator evidence",
                {"error": runtime.get("error")},
            )
        kinematic_check, geometry_check = _certificate_checks(certification_status)
        behavior_check = _behavior_check(scenario_path, config=search_config)
        checks = (kinematic_check, behavior_check, geometry_check, simulator_check)
        value, feature_vector = _risk_value(checks, certification_status, candidate_id=candidate_id)
        feasibility_candidate = FeasibilityCandidate(
            candidate_id=candidate_id,
            scenario_family=str(payload["scenario_family"]),
            scenario_seed=int(candidate.scenario_seed),
            control_hash=compute_control_hash(candidate),
            checks=checks,
            value=value,
            feature_vector=feature_vector,
            candidate_controls={
                **candidate.to_json(),
                "pedestrian_id": search_config.search_space.pedestrian_id,
                "pedestrian_route_mode": search_config.search_space.pedestrian_route_mode,
            },
        )
        candidates.append(feasibility_candidate)
        runtime_by_id[candidate_id] = runtime

    budget = int(payload["sample_budget"])
    sampling_seed = int(payload["sampling_seed"])
    uniform = sample_seeded_uniform(candidates, budget=budget, seed=sampling_seed)
    feasible = rank_feasible_candidates(candidates)
    risk_feedback = feasible[:budget] if len(feasible) >= budget else None
    methods = {
        "seeded_uniform": _method_summary(
            uniform,
            budget=budget,
            threshold=float(payload["criticality_threshold"]),
            runtime_by_id=runtime_by_id,
        ),
        "risk_feedback_hierarchical_value": (
            _method_summary(
                risk_feedback,
                budget=budget,
                threshold=float(payload["criticality_threshold"]),
                runtime_by_id=runtime_by_id,
            )
            if risk_feedback is not None
            else _unavailable_method(
                f"only {len(feasible)} candidates passed all four checks; sample budget is {budget}"
            )
        ),
    }
    executed = [
        runtime
        for runtime in runtime_by_id.values()
        if runtime.get("execution_mode") in {"native", "adapter", "mixed"}
        and runtime.get("availability_status") == "available"
    ]
    rejection_counts = Counter(
        reason.split(":", maxsplit=2)[0]
        for candidate in candidates
        for reason in candidate.rejection_reasons
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": EVIDENCE_TIER,
        "config_sha256": _sha256_bytes(raw_config),
        "input_digests": digests,
        "seed_manifest": {
            "sampling_seed": sampling_seed,
            "candidate_seeds": [candidate.scenario_seed for candidate in candidates],
            "candidate_ids_in_source_order": [candidate.candidate_id for candidate in candidates],
        },
        "feasibility": {
            "check_names": list(CHECK_NAMES),
            "total_candidates": len(candidates),
            "feasible_candidates": sum(candidate.feasible for candidate in candidates),
            "rejected_candidates": sum(not candidate.feasible for candidate in candidates),
            "unavailable_candidates": sum(
                any(check.status == "unavailable" for check in candidate.checks)
                for candidate in candidates
            ),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "invalid_candidates_excluded_from_safety_denominators": True,
        },
        "comparison": {
            "sample_budget": budget,
            "criticality_threshold": float(payload["criticality_threshold"]),
            "existing_adversarial_baseline": {
                "id": BASELINE_ID,
                "status": "executed_candidate_pool",
                "reason": (
                    "the canonical random sampler generated the fixed candidate pool; "
                    "this report is not an admitted budget-matched baseline campaign"
                ),
                "claim_eligible": False,
            },
            "selection_protocol": {
                "candidate_pool_fixed": True,
                "simulator_outcomes_used_for_selection": False,
                "risk_features_source": "scenario_cert.v1 route evidence",
                "safety_denominator": "feasible candidates with available episode evidence",
            },
            "methods": methods,
            "safety_event_severity": {
                "status": "available" if executed else "unavailable",
                "counts": dict(
                    sorted(
                        Counter(
                            str(runtime.get("primary_failure") or "unknown") for runtime in executed
                        ).items()
                    )
                ),
                "denominator": len(executed),
                "reason": "observed native/adapter episode attribution; not a safety rate",
            },
        },
        "candidates": [
            _candidate_record(candidate, runtime=runtime_by_id[candidate.candidate_id])
            for candidate in candidates
        ],
        "governance": {
            "simulator_executed": bool(executed),
            "benchmark_evidence": False,
            "campaign_approval_required": True,
            "domain_approval_status": "required",
            "adapted_from_source_method": True,
            "source_transfer_claim": False,
        },
        "config": {
            "path": _repo_relative(config_path),
            "schema_version": SCHEMA_VERSION,
            "scenario_template": _repo_relative(search_config.scenario_template),
            "search_space": _repo_relative(search_config.search_space_path),
            "candidate_pool_budget": int(payload["candidate_pool_budget"]),
            "sample_budget": budget,
            "policy": str(payload["policy"]),
            "horizon": int(payload["horizon"]),
            "dt": float(payload["dt"]),
            "pedestrian_id": search_config.search_space.pedestrian_id,
            "pedestrian_route_mode": search_config.search_space.pedestrian_route_mode,
        },
    }
    validate_real_report(report)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def load_real_report_schema() -> dict[str, Any]:
    """Load the committed real-manifest report schema."""
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_real_report(payload: Mapping[str, Any]) -> None:
    """Validate a real-manifest report against its versioned JSON Schema."""
    errors = sorted(Draft202012Validator(load_real_report_schema()).iter_errors(payload), key=str)
    if errors:
        raise RealManifestError("; ".join(error.message for error in errors))


__all__ = [
    "CLAIM_BOUNDARY",
    "EVIDENCE_TIER",
    "SCHEMA_VERSION",
    "RealManifestError",
    "load_real_report_schema",
    "run_real_manifest_diagnostic",
    "validate_real_report",
]
