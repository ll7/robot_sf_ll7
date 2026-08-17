"""Feasibility-first scenario search diagnostics.

This module provides a small, typed contract for rejecting scenario candidates before
they enter a risk-search denominator.  It composes the existing adversarial manifest
identity and validation surfaces, but deliberately does not execute a simulator or
claim that a candidate is safe.  The fixture runner is a reproducible research
diagnostic for issue #7315, not a benchmark campaign.

The four feasibility dimensions are intentionally separate:

* ``kinematic_reachability`` — can the route be reached under the declared horizon?
* ``behavioral_consistency`` — are the robot/virtual-road-user assumptions coherent?
* ``geometry_traffic`` — is the geometry/traffic arrangement admissible?
* ``simulator_validity`` — did the simulator or its validator accept the input?

Missing or contradictory evidence is represented as ``unavailable`` and is never
treated as a passing row.  Ranking uses a deterministic lexicographic value with
kinematic criticality, controllability/risk, and diversity components; it is not an
opaque weighted score.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.scenario_manifest import compute_control_hash

SCHEMA_VERSION = "feasibility_first_scenario_search.v1"
EVIDENCE_TIER = "diagnostic-only"
EXISTING_BASELINE_ID = "existing_adversarial_random_sampler.v1"
CLAIM_BOUNDARY = (
    "diagnostic-only fixture protocol: feasibility rejection and risk-feature ordering; "
    "no simulator, planner, safety, or benchmark claim"
)
CHECK_NAMES: tuple[str, ...] = (
    "kinematic_reachability",
    "behavioral_consistency",
    "geometry_traffic",
    "simulator_validity",
)
CHECK_STATUSES = {"pass", "fail", "unavailable"}
_REPORT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "schemas"
    / "feasibility_first_scenario_search.v1.json"
)


class FeasibilityFirstError(ValueError):
    """Raised when a feasibility-first record cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class FeasibilityCheck:
    """One independently auditable feasibility predicate result."""

    name: str
    status: Literal["pass", "fail", "unavailable"]
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject unknown dimensions and non-actionable check explanations."""
        if self.name not in CHECK_NAMES:
            raise FeasibilityFirstError(
                f"check name must be one of {CHECK_NAMES!r}, got {self.name!r}"
            )
        if self.status not in CHECK_STATUSES:
            raise FeasibilityFirstError(
                f"check {self.name!r} has unsupported status {self.status!r}"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise FeasibilityFirstError(f"check {self.name!r} requires a non-empty reason")
        if self.status == "pass" and not self.evidence:
            raise FeasibilityFirstError(f"check {self.name!r} requires evidence when passing")
        _validate_evidence_mapping(self.evidence, path=f"checks.{self.name}.evidence")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, expected_name: str) -> FeasibilityCheck:
        """Parse one strict check mapping and fail closed on unknown fields."""
        if not isinstance(payload, Mapping):
            raise FeasibilityFirstError(f"checks.{expected_name} must be a mapping")
        unknown = set(payload) - {"name", "status", "reason", "evidence"}
        if unknown:
            raise FeasibilityFirstError(
                f"checks.{expected_name} has unknown fields: {sorted(unknown)}"
            )
        name = payload.get("name", expected_name)
        if name != expected_name:
            raise FeasibilityFirstError(
                f"checks.{expected_name}.name must be {expected_name!r}, got {name!r}"
            )
        status = payload.get("status")
        reason = payload.get("reason")
        if "evidence" not in payload:
            raise FeasibilityFirstError(f"checks.{expected_name} requires an evidence mapping")
        evidence = payload["evidence"]
        if not isinstance(status, str) or not isinstance(reason, str):
            raise FeasibilityFirstError(f"checks.{expected_name} requires string status and reason")
        if not isinstance(evidence, Mapping):
            raise FeasibilityFirstError(f"checks.{expected_name}.evidence must be a mapping")
        return cls(name, status, reason, dict(evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe check record."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class HierarchicalScenarioValue:
    """Risk-search value components ordered without a hidden weighted sum."""

    kinematic_criticality: float
    controllability_risk: float
    diversity: float

    def __post_init__(self) -> None:
        """Require finite normalized components so ordering is reproducible."""
        for name, value in (
            ("kinematic_criticality", self.kinematic_criticality),
            ("controllability_risk", self.controllability_risk),
            ("diversity", self.diversity),
        ):
            _bounded_float(value, name, minimum=0.0, maximum=1.0)

    def sort_key(self) -> tuple[float, float, float]:
        """Return the descending lexicographic priority key."""
        return (
            float(self.kinematic_criticality),
            float(self.controllability_risk),
            float(self.diversity),
        )

    def to_dict(self) -> dict[str, float]:
        """Return normalized value components."""
        return {
            "kinematic_criticality": float(self.kinematic_criticality),
            "controllability_risk": float(self.controllability_risk),
            "diversity": float(self.diversity),
        }


@dataclass(frozen=True, slots=True)
class FeasibilityCandidate:
    """A candidate with split feasibility evidence and deterministic value."""

    candidate_id: str
    scenario_family: str
    scenario_seed: int
    control_hash: str
    checks: tuple[FeasibilityCheck, ...]
    value: HierarchicalScenarioValue
    feature_vector: tuple[float, ...]
    candidate_controls: dict[str, Any]

    def __post_init__(self) -> None:
        """Enforce identity, check coverage, and finite feature invariants."""
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise FeasibilityFirstError("candidate_id must be non-empty")
        if not isinstance(self.scenario_family, str) or not self.scenario_family.strip():
            raise FeasibilityFirstError(f"{self.candidate_id}: scenario_family must be non-empty")
        if isinstance(self.scenario_seed, bool) or not isinstance(self.scenario_seed, int):
            raise FeasibilityFirstError(f"{self.candidate_id}: scenario_seed must be an integer")
        if self.scenario_seed < 0:
            raise FeasibilityFirstError(f"{self.candidate_id}: scenario_seed must be non-negative")
        if not isinstance(self.control_hash, str) or not self.control_hash.strip():
            raise FeasibilityFirstError(f"{self.candidate_id}: control_hash must be non-empty")
        names = tuple(check.name for check in self.checks)
        if names != CHECK_NAMES:
            raise FeasibilityFirstError(
                f"{self.candidate_id}: checks must contain {CHECK_NAMES!r} in canonical order"
            )
        if not self.feature_vector:
            raise FeasibilityFirstError(f"{self.candidate_id}: feature_vector must not be empty")
        for index, value in enumerate(self.feature_vector):
            _finite_float(value, f"{self.candidate_id}.feature_vector[{index}]")
        if not isinstance(self.candidate_controls, Mapping):
            raise FeasibilityFirstError(
                f"{self.candidate_id}: candidate_controls must be a mapping"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeasibilityCandidate:  # noqa: C901
        """Parse a strict candidate record from JSON/YAML data."""
        if not isinstance(payload, Mapping):
            raise FeasibilityFirstError("candidate record must be a mapping")
        required = {
            "candidate_id",
            "scenario_family",
            "scenario_seed",
            "control_hash",
            "checks",
            "value",
            "feature_vector",
            "candidate_controls",
        }
        missing = required - set(payload)
        derived_fields = {"feasible", "rejection_reasons"}
        unknown = set(payload) - required - derived_fields
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if unknown:
                details.append(f"unknown {sorted(unknown)}")
            raise FeasibilityFirstError("candidate record fields: " + "; ".join(details))
        checks_raw = payload["checks"]
        if not isinstance(checks_raw, Sequence) or isinstance(checks_raw, str | bytes):
            raise FeasibilityFirstError("candidate checks must be a sequence")
        if len(checks_raw) != len(CHECK_NAMES):
            raise FeasibilityFirstError(f"candidate checks must contain {len(CHECK_NAMES)} entries")
        checks = tuple(
            FeasibilityCheck.from_mapping(item, expected_name=name)
            for name, item in zip(CHECK_NAMES, checks_raw, strict=True)
        )
        value_raw = payload["value"]
        if not isinstance(value_raw, Mapping):
            raise FeasibilityFirstError("candidate value must be a mapping")
        value_fields = {"kinematic_criticality", "controllability_risk", "diversity"}
        if set(value_raw) != value_fields:
            raise FeasibilityFirstError(
                f"candidate value fields must be exactly {sorted(value_fields)}"
            )
        feature_vector_raw = payload["feature_vector"]
        if not isinstance(feature_vector_raw, Sequence) or isinstance(
            feature_vector_raw, str | bytes
        ):
            raise FeasibilityFirstError("candidate feature_vector must be a sequence")
        controls = payload["candidate_controls"]
        if not isinstance(controls, Mapping):
            raise FeasibilityFirstError("candidate candidate_controls must be a mapping")
        candidate = cls(
            candidate_id=_required_text(payload["candidate_id"], "candidate_id"),
            scenario_family=_required_text(payload["scenario_family"], "scenario_family"),
            scenario_seed=_required_int(payload["scenario_seed"], "scenario_seed"),
            control_hash=_required_text(payload["control_hash"], "control_hash"),
            checks=checks,
            value=HierarchicalScenarioValue(
                kinematic_criticality=_required_number(
                    value_raw["kinematic_criticality"], "value.kinematic_criticality"
                ),
                controllability_risk=_required_number(
                    value_raw["controllability_risk"], "value.controllability_risk"
                ),
                diversity=_required_number(value_raw["diversity"], "value.diversity"),
            ),
            feature_vector=tuple(
                _required_number(value, f"feature_vector[{index}]")
                for index, value in enumerate(feature_vector_raw)
            ),
            candidate_controls=dict(controls),
        )
        if "feasible" in payload and payload["feasible"] != candidate.feasible:
            raise FeasibilityFirstError(
                f"{candidate.candidate_id}: derived feasible field contradicts checks"
            )
        if "rejection_reasons" in payload:
            reasons = payload["rejection_reasons"]
            if not isinstance(reasons, Sequence) or isinstance(reasons, str | bytes):
                raise FeasibilityFirstError(
                    f"{candidate.candidate_id}: rejection_reasons must be a sequence"
                )
            if list(reasons) != list(candidate.rejection_reasons):
                raise FeasibilityFirstError(
                    f"{candidate.candidate_id}: rejection_reasons contradict checks"
                )
        return candidate

    @property
    def feasible(self) -> bool:
        """Return true only when every declared predicate explicitly passes."""
        return all(check.status == "pass" for check in self.checks)

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        """Return stable reason codes for failed or unavailable dimensions."""
        return tuple(
            f"{check.name}:{check.status}:{check.reason}"
            for check in self.checks
            if check.status != "pass"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe candidate record."""
        return {
            "candidate_id": self.candidate_id,
            "scenario_family": self.scenario_family,
            "scenario_seed": int(self.scenario_seed),
            "control_hash": self.control_hash,
            "checks": [check.to_dict() for check in self.checks],
            "value": self.value.to_dict(),
            "feature_vector": [float(value) for value in self.feature_vector],
            "candidate_controls": dict(self.candidate_controls),
            "feasible": self.feasible,
            "rejection_reasons": list(self.rejection_reasons),
        }


def rank_feasible_candidates(
    candidates: Sequence[FeasibilityCandidate],
) -> list[FeasibilityCandidate]:
    """Return feasible candidates in deterministic hierarchical priority order."""
    return sorted(
        (candidate for candidate in candidates if candidate.feasible),
        key=lambda candidate: (
            -candidate.value.kinematic_criticality,
            -candidate.value.controllability_risk,
            -candidate.value.diversity,
            candidate.candidate_id,
        ),
    )


def sample_seeded_uniform(
    candidates: Sequence[FeasibilityCandidate],
    *,
    budget: int,
    seed: int,
) -> list[FeasibilityCandidate]:
    """Draw a deterministic uniform baseline from the complete candidate pool."""
    _validate_sampling_inputs(candidates, budget=budget, seed=seed)
    ordered = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    return random.Random(seed).sample(ordered, k=budget)


def sample_risk_feedback(
    candidates: Sequence[FeasibilityCandidate],
    *,
    budget: int,
) -> list[FeasibilityCandidate]:
    """Select the highest-valued feasible candidates before simulator execution.

    The name records the intended research direction: in a future campaign, the
    value components can be updated from observed risk feedback.  This bounded
    slice only consumes precomputed fixture features and therefore makes no
    closed-loop search claim.
    """
    _validate_sampling_inputs(candidates, budget=budget, seed=0)
    ranked = rank_feasible_candidates(candidates)
    if len(ranked) < budget:
        raise FeasibilityFirstError(
            f"risk-feedback sample needs {budget} feasible candidates; only {len(ranked)} exist"
        )
    return ranked[:budget]


def build_comparison_report(
    candidates: Sequence[FeasibilityCandidate],
    *,
    budget: int,
    seed: int,
    config_sha256: str,
    criticality_threshold: float,
) -> dict[str, Any]:
    """Build a deterministic diagnostic comparison with explicit unavailable metrics."""
    _validate_sampling_inputs(candidates, budget=budget, seed=seed)
    _bounded_float(criticality_threshold, "criticality_threshold", minimum=0.0, maximum=1.0)
    candidate_list = list(candidates)
    ids = [candidate.candidate_id for candidate in candidate_list]
    if len(ids) != len(set(ids)):
        raise FeasibilityFirstError("candidate_id values must be unique")
    hashes = [candidate.control_hash for candidate in candidate_list]
    if len(hashes) != len(set(hashes)):
        raise FeasibilityFirstError("control_hash values must be unique")

    uniform = sample_seeded_uniform(candidate_list, budget=budget, seed=seed)
    risk_feedback = sample_risk_feedback(candidate_list, budget=budget)
    rejection_counts = Counter(
        reason.split(":", maxsplit=2)[0]
        for candidate in candidate_list
        for reason in candidate.rejection_reasons
    )
    methods = {
        "seeded_uniform": _summarize_selection(
            uniform,
            budget=budget,
            criticality_threshold=criticality_threshold,
        ),
        "risk_feedback_hierarchical_value": _summarize_selection(
            risk_feedback,
            budget=budget,
            criticality_threshold=criticality_threshold,
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": EVIDENCE_TIER,
        "config_sha256": config_sha256,
        "seed_manifest": {
            "sampling_seed": int(seed),
            "candidate_seeds": [int(candidate.scenario_seed) for candidate in candidate_list],
            "candidate_ids_in_source_order": ids,
        },
        "feasibility": {
            "check_names": list(CHECK_NAMES),
            "total_candidates": len(candidate_list),
            "feasible_candidates": sum(candidate.feasible for candidate in candidate_list),
            "rejected_candidates": sum(not candidate.feasible for candidate in candidate_list),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "invalid_candidates_excluded_from_safety_denominators": True,
        },
        "comparison": {
            "sample_budget": int(budget),
            "criticality_threshold": float(criticality_threshold),
            "existing_adversarial_baseline": {
                "id": EXISTING_BASELINE_ID,
                "status": "not_executed",
                "reason": (
                    "fixture holds a candidate pool fixed; the existing adversarial sampler "
                    "requires a real scenario/search-space input"
                ),
                "claim_eligible": False,
            },
            "methods": methods,
            "safety_event_severity": {
                "status": "unavailable",
                "reason": "fixture diagnostic does not execute a simulator",
            },
        },
        "candidates": [candidate.to_dict() for candidate in candidate_list],
        "governance": {
            "simulator_executed": False,
            "benchmark_evidence": False,
            "campaign_approval_required": True,
            "adapted_from_source_method": True,
            "source_transfer_claim": False,
        },
    }


def validate_report(payload: Mapping[str, Any]) -> None:
    """Validate a report against the committed JSON Schema."""
    errors = sorted(Draft202012Validator(load_report_schema()).iter_errors(payload), key=str)
    if errors:
        raise FeasibilityFirstError("; ".join(error.message for error in errors))


def load_report_schema() -> dict[str, Any]:
    """Load the versioned report schema shipped with the package."""
    return json.loads(_REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))


def build_fixture_candidates() -> list[FeasibilityCandidate]:
    """Return the small deterministic fixture used by the diagnostic CLI."""
    fixture_rows = (
        ("crossing_low_margin", "crossing", 101, "pass", "pass", "pass", "pass", 0.55, 0.35, 0.30),
        ("crossing_high_risk", "crossing", 102, "pass", "pass", "pass", "pass", 0.92, 0.78, 0.35),
        ("doorway_controlled", "doorway", 201, "pass", "pass", "pass", "pass", 0.68, 0.88, 0.72),
        (
            "blind_corner_diverse",
            "blind_corner",
            301,
            "pass",
            "pass",
            "pass",
            "pass",
            0.80,
            0.62,
            0.94,
        ),
        (
            "bottleneck_geometry_reject",
            "bottleneck",
            401,
            "pass",
            "pass",
            "fail",
            "pass",
            0.97,
            0.91,
            0.81,
        ),
        (
            "overtake_behavior_reject",
            "overtake",
            501,
            "pass",
            "fail",
            "pass",
            "pass",
            0.86,
            0.74,
            0.66,
        ),
        (
            "crossing_simulator_unavailable",
            "crossing",
            103,
            "pass",
            "pass",
            "pass",
            "unavailable",
            0.89,
            0.80,
            0.41,
        ),
        (
            "invalid_kinematics",
            "constriction",
            601,
            "fail",
            "pass",
            "pass",
            "pass",
            0.99,
            0.95,
            0.58,
        ),
    )
    candidates: list[FeasibilityCandidate] = []
    for index, row in enumerate(fixture_rows):
        (
            candidate_id,
            family,
            seed,
            kinematic_status,
            behavior_status,
            geometry_status,
            simulator_status,
            kinematic_criticality,
            controllability_risk,
            diversity,
        ) = row
        candidate = _fixture_candidate_spec(index=index, seed=seed)
        candidates.append(
            FeasibilityCandidate(
                candidate_id=candidate_id,
                scenario_family=family,
                scenario_seed=seed,
                control_hash=compute_control_hash(candidate),
                checks=(
                    FeasibilityCheck(
                        "kinematic_reachability",
                        kinematic_status,
                        "route is reachable within declared horizon"
                        if kinematic_status == "pass"
                        else "required travel exceeds declared kinematic envelope",
                        {"horizon_s": 8.0, "required_time_s": 5.0 + index * 0.1},
                    ),
                    FeasibilityCheck(
                        "behavioral_consistency",
                        behavior_status,
                        "robot and virtual-road-user timing assumptions agree"
                        if behavior_status == "pass"
                        else "pedestrian timing and behavior fields are contradictory",
                        {"timing_overlap_s": max(0.0, 2.0 - index * 0.1)},
                    ),
                    FeasibilityCheck(
                        "geometry_traffic",
                        geometry_status,
                        "geometry and traffic arrangement is admissible"
                        if geometry_status == "pass"
                        else "bottleneck clearance conflicts with traffic envelope",
                        {"minimum_clearance_m": 0.62 - index * 0.01},
                    ),
                    FeasibilityCheck(
                        "simulator_validity",
                        simulator_status,
                        "fixture validator accepted the scenario payload"
                        if simulator_status == "pass"
                        else "no simulator validation artifact is available in fixture mode",
                        {"execution_mode": "fixture_only"},
                    ),
                ),
                value=HierarchicalScenarioValue(
                    kinematic_criticality=kinematic_criticality,
                    controllability_risk=controllability_risk,
                    diversity=diversity,
                ),
                feature_vector=(
                    float(kinematic_criticality),
                    float(controllability_risk),
                    float(diversity),
                    float(index),
                ),
                candidate_controls=candidate.to_json(),
            )
        )
    return candidates


def run_fixture_diagnostic(
    config_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the config-first fixture protocol and optionally persist its report."""
    raw_config = config_path.read_bytes()
    config = yaml.safe_load(raw_config)
    if not isinstance(config, Mapping):
        raise FeasibilityFirstError("config must be a mapping")
    _validate_config(config)
    report = build_comparison_report(
        build_fixture_candidates(),
        budget=_required_int(config["sample_budget"], "sample_budget"),
        seed=_required_int(config["sampling_seed"], "sampling_seed"),
        config_sha256=hashlib.sha256(raw_config).hexdigest(),
        criticality_threshold=_required_number(
            config["criticality_threshold"], "criticality_threshold"
        ),
    )
    report["config"] = {
        "path": config_path.as_posix(),
        "schema_version": config["schema_version"],
        "fixture": config["fixture"],
    }
    validate_report(report)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def _summarize_selection(
    selected: Sequence[FeasibilityCandidate],
    *,
    budget: int,
    criticality_threshold: float,
) -> dict[str, Any]:
    """Summarize one method without replacing unavailable outcomes with zeros."""
    feasible_count = sum(candidate.feasible for candidate in selected)
    critical_candidates = sum(
        candidate.value.kinematic_criticality >= criticality_threshold
        for candidate in selected
        if candidate.feasible
    )
    return {
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "selected_count": len(selected),
        "valid_scenario_rate": feasible_count / budget,
        "discovery_yield": critical_candidates,
        "unique_scenario_families": len(
            {candidate.scenario_family for candidate in selected if candidate.feasible}
        ),
        "mean_kinematic_criticality": _mean_or_none(
            candidate.value.kinematic_criticality for candidate in selected if candidate.feasible
        ),
        "mean_controllability_risk": _mean_or_none(
            candidate.value.controllability_risk for candidate in selected if candidate.feasible
        ),
        "mean_diversity": _mean_or_none(
            candidate.value.diversity for candidate in selected if candidate.feasible
        ),
        "rejected_count": len(selected) - feasible_count,
        "rejection_reasons": dict(
            sorted(
                Counter(
                    reason.split(":", maxsplit=2)[0]
                    for candidate in selected
                    for reason in candidate.rejection_reasons
                ).items()
            )
        ),
    }


def _fixture_candidate_spec(*, index: int, seed: int) -> CandidateSpec:
    """Create a stable candidate control payload for one fixture row."""
    return CandidateSpec(
        start=Pose2D(float(index), 0.0),
        goal=Pose2D(float(index) + 5.0, 0.0),
        spawn_time_s=float(index) * 0.1,
        pedestrian_speed_mps=1.0 + float(index) * 0.01,
        pedestrian_delay_s=float(index) * 0.05,
        scenario_seed=seed,
    )


def _validate_sampling_inputs(
    candidates: Sequence[FeasibilityCandidate],
    *,
    budget: int,
    seed: int,
) -> None:
    """Validate shared deterministic sampling inputs."""
    if not candidates:
        raise FeasibilityFirstError("candidate pool must not be empty")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise FeasibilityFirstError("sample budget must be a positive integer")
    if budget > len(candidates):
        raise FeasibilityFirstError(
            f"sample budget {budget} exceeds candidate pool size {len(candidates)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FeasibilityFirstError("sampling seed must be a non-negative integer")


def _validate_config(config: Mapping[str, Any]) -> None:
    """Validate the compact config contract used by the fixture CLI."""
    required = {
        "schema_version",
        "claim_boundary",
        "fixture",
        "baseline",
        "sampling_seed",
        "sample_budget",
        "criticality_threshold",
    }
    unknown = set(config) - required
    missing = required - set(config)
    if missing or unknown:
        raise FeasibilityFirstError(
            f"config fields invalid; missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise FeasibilityFirstError(f"config schema_version must be {SCHEMA_VERSION!r}")
    if config["claim_boundary"] != CLAIM_BOUNDARY:
        raise FeasibilityFirstError("config claim_boundary does not match the code contract")
    if not isinstance(config["fixture"], str) or not config["fixture"].strip():
        raise FeasibilityFirstError("config fixture must be non-empty")
    if config["baseline"] != EXISTING_BASELINE_ID:
        raise FeasibilityFirstError(f"config baseline must be {EXISTING_BASELINE_ID!r}")
    _required_int(config["sampling_seed"], "sampling_seed")
    _required_int(config["sample_budget"], "sample_budget")
    _required_number(config["criticality_threshold"], "criticality_threshold")


def _validate_evidence_mapping(evidence: Mapping[str, Any], *, path: str) -> None:
    """Reject non-JSON scalar evidence and non-finite numeric values."""
    if not isinstance(evidence, Mapping):
        raise FeasibilityFirstError(f"{path} must be a mapping")
    for key, value in evidence.items():
        if not isinstance(key, str) or not key.strip():
            raise FeasibilityFirstError(f"{path} keys must be non-empty strings")
        if isinstance(value, bool | str) or value is None:
            continue
        if isinstance(value, int | float):
            _finite_float(value, f"{path}.{key}")
            continue
        raise FeasibilityFirstError(f"{path}.{key} must be a JSON scalar")


def _mean_or_none(values: Sequence[float] | Any) -> float | None:
    """Return a finite mean or ``None`` when the evidence set is empty."""
    values_list = [float(value) for value in values]
    return sum(values_list) / len(values_list) if values_list else None


def _bounded_float(value: object, name: str, *, minimum: float, maximum: float) -> float:
    """Return a finite bounded float or raise a contract error."""
    numeric = _finite_float(value, name)
    if not minimum <= numeric <= maximum:
        raise FeasibilityFirstError(f"{name} must be between {minimum} and {maximum}")
    return numeric


def _finite_float(value: object, name: str) -> float:
    """Return a finite number, rejecting booleans and NaN/inf."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FeasibilityFirstError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FeasibilityFirstError(f"{name} must be finite")
    return numeric


def _required_number(value: object, name: str) -> float:
    """Return a finite number for a parsed field."""
    return _finite_float(value, name)


def _required_int(value: object, name: str) -> int:
    """Return a non-boolean integer for a parsed field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeasibilityFirstError(f"{name} must be an integer")
    return value


def _required_text(value: object, name: str) -> str:
    """Return a non-empty string for a parsed field."""
    if not isinstance(value, str) or not value.strip():
        raise FeasibilityFirstError(f"{name} must be a non-empty string")
    return value.strip()


__all__ = [
    "CHECK_NAMES",
    "CLAIM_BOUNDARY",
    "EVIDENCE_TIER",
    "EXISTING_BASELINE_ID",
    "SCHEMA_VERSION",
    "FeasibilityCandidate",
    "FeasibilityCheck",
    "FeasibilityFirstError",
    "HierarchicalScenarioValue",
    "build_comparison_report",
    "build_fixture_candidates",
    "load_report_schema",
    "rank_feasible_candidates",
    "run_fixture_diagnostic",
    "sample_risk_feedback",
    "sample_seeded_uniform",
    "validate_report",
]
