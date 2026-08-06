#!/usr/bin/env python3
"""Offline primitive-vs-RBF risk-ranker candidate comparison (issue #6768).

This script runs a deterministic offline comparison of the existing
:func:`generate_primitive_candidates` generator and the deterministic RBF
:func:`generate_rbf_candidates` generator (merged with #6676) on the same
held-out fixture/trace set, with equal candidate budgets and identical risk
estimator, ranking-weight, and hard-gate configuration. It never wires either
generator into a planner loop.

For each generator and fixture the report separately records:

- candidate count and finite/unique validity;
- trajectory-verifier and actuator-gate rejection counts;
- eligible-candidate count;
- selected-candidate identity and whether generator choice changes selection;
- decomposed risk / time / jerk / path-length / clearance components;
- a model-risk reliability diagnostic where a fixture declares a known contact
  outcome;
- generation time, ranking time, and total time;
- unavailable denominators and their reasons.

The script fails closed on calibration/evaluation split overlap, missing
fixture provenance, non-finite values, unequal candidate budgets, or a
generator/config hash mismatch. The JSON and Markdown reports are deterministic
for a pinned generation timestamp; wall-clock timing is reported separately and
caveated as measured local offline time.

Claim boundary: ``diagnostic_only``. This is not planner-improvement, calibrated
real-world probability, safety, or online-readiness evidence.

Example:
    uv run python scripts/analysis/compare_risk_ranker_generators_issue_6768.py \
        --config configs/analysis/issue_6768_risk_ranker_generator_comparison.yaml \
        --output <report>.json --output-md <report>.md
    uv run python scripts/analysis/compare_risk_ranker_generators_issue_6768.py \
        --check-config configs/analysis/issue_6768_risk_ranker_generator_comparison.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from robot_sf.benchmark.actuator_feasibility import (
    VERDICT_ACTUATOR_FEASIBLE,
    ActuatorLimitsConfig,
)
from robot_sf.benchmark.trajectory_verifier import DECISION_FALLBACK_BRAKE, TrajectoryVerifierConfig
from robot_sf.nav.predictive_types import PedestrianState
from robot_sf.planner.risk_aware_trajectory_ranker import (
    PrimitiveGeneratorConfig,
    RankingWeights,
    RBFGeneratorConfig,
    generate_primitive_candidates,
    generate_rbf_candidates,
    rank_trajectories,
)
from robot_sf.research.collision_risk import CandidateAction, RiskEstimatorConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from robot_sf.planner.risk_aware_trajectory_ranker import CandidateRanking

#: Testable monotonic clock so deterministic tests can pin wall-clock timing.
_perf_counter_ns = time.perf_counter_ns

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/analysis/issue_6768_risk_ranker_generator_comparison.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output/analysis/issue_6768_risk_ranker_generator_comparison.json"

REPORT_SCHEMA_VERSION = "issue_6768_risk_ranker_generator_comparison.v1"
CONFIG_SCHEMA_VERSION = "issue_6768_risk_ranker_generator_comparison_config.v1"
FIXTURE_SCHEMA_VERSION = "issue_6768_risk_ranker_generator_comparison_fixture.v1"
CLAIM_BOUNDARY = (
    "diagnostic_only: compares finite RBF candidate proposals with the deterministic "
    "primitive baseline on matched held-out fixtures with equal budgets and unchanged "
    "risk/hard-gate settings; it does not establish planner improvement, calibrated "
    "collision probability, safety, nominal benchmark evidence, or online readiness"
)
EVALUATION_SPLIT = "held_out"
CALIBRATION_SPLIT = "calibration"

#: Fixed decimal places for reported wall-clock milliseconds; timing is always
#: caveated as measured offline time and is not a performance claim.
_TIMING_DECIMALS = 3


class ComparisonError(ValueError):
    """Raised, fail-closed, when the matched comparison inputs are invalid."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a YAML mapping or fail closed with a useful field name."""
    if not isinstance(value, dict):
        raise ComparisonError(f"{name} must be a YAML mapping")
    return dict(value)


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    """Load a required YAML mapping, failing closed on read or parse errors."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ComparisonError(f"could not read {name}: {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ComparisonError(f"could not parse {name}: {path}: {exc}") from exc
    return _mapping(payload, name)


def _resolve_repo_path(value: Any, *, field: str) -> Path:
    """Resolve a config path relative to the repository root."""
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    """Prefer a repository-relative provenance path."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    """Return the full SHA-256 digest of a provenance input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    """Read and validate the exact repository commit used by the report."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    head = result.stdout.strip()
    if result.returncode != 0 or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ComparisonError("git rev-parse HEAD did not return an exact 40-character SHA")
    return head


def _git_status() -> list[str]:
    """Return short status lines without exposing environment secrets."""
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ["git status failed"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def _require_finite(value: Any, *, field: str) -> None:
    """Fail closed on any non-finite numeric value."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ComparisonError(f"{field} must be finite, got {value!r}")


def _finite_vector(value: Any, *, field: str) -> list[float]:
    """Validate a finite two-dimensional fixture vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ComparisonError(f"{field} must be a finite length-two vector")
    return [float(item) for item in array]


def _require_fixture_provenance(payload: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Fail closed when a fixture file lacks complete top-level provenance."""
    provenance = payload.get("provenance")
    provenance = _mapping(provenance, f"{name}.provenance")
    missing = [
        field
        for field in ("fixture_source", "generated_by", "disjoint_from")
        if not str(provenance.get(field) or "").strip()
    ]
    if missing:
        raise ComparisonError(f"{name} provenance is incomplete; missing fields: {missing}")
    return dict(provenance)


def _parse_known_contact_outcome(case_id: str, raw: Any) -> dict[str, Any]:
    """Validate and normalize a case's declared known contact outcome."""
    outcome = _mapping(raw, f"{case_id}.known_contact_outcome")
    if "contact_certain" not in outcome or not isinstance(outcome["contact_certain"], bool):
        raise ComparisonError(
            f"fixture case {case_id!r}.known_contact_outcome.contact_certain must be a boolean"
        )
    if not str(outcome.get("reason") or "").strip():
        raise ComparisonError(
            f"fixture case {case_id!r}.known_contact_outcome.reason must be non-empty"
        )
    candidate_role = str(outcome.get("candidate_role") or "straight")
    if not candidate_role:
        raise ComparisonError(
            f"fixture case {case_id!r}.known_contact_outcome.candidate_role must be non-empty"
        )
    return {
        "contact_certain": bool(outcome["contact_certain"]),
        "candidate_role": candidate_role,
        "reason": str(outcome["reason"]),
    }


def _normalize_fixture_case(raw_case: Any, *, index: int, seen_ids: set[str]) -> dict[str, Any]:
    """Validate and normalize one fixture case with per-case provenance."""
    case = _mapping(raw_case, f"fixture.cases[{index}]")
    case_id = str(case.get("case_id") or "")
    if not case_id or case_id in seen_ids:
        raise ComparisonError(f"fixture case ids must be non-empty and unique: {case_id!r}")
    split = str(case.get("split") or "")
    if split not in {EVALUATION_SPLIT, CALIBRATION_SPLIT}:
        raise ComparisonError(
            f"fixture case {case_id!r} split must be {EVALUATION_SPLIT!r} or "
            f"{CALIBRATION_SPLIT!r}, got {split!r}"
        )
    if case.get("status", "valid") != "valid":
        raise ComparisonError(f"fixture case {case_id!r} is not a valid evidence row")
    case_provenance = case.get("provenance")
    case_provenance = _mapping(case_provenance, f"fixture case {case_id}.provenance")
    if not str(case_provenance.get("source") or "").strip():
        raise ComparisonError(
            f"fixture case {case_id!r} provenance is incomplete; missing provenance.source"
        )

    pedestrians = case.get("pedestrians", [])
    if not isinstance(pedestrians, list):
        raise ComparisonError(f"fixture case {case_id!r}.pedestrians must be a list")
    normalized_pedestrians: list[dict[str, Any]] = []
    actor_ids: set[int] = set()
    for actor_index, raw_actor in enumerate(pedestrians):
        actor = _mapping(raw_actor, f"fixture case {case_id}.pedestrians[{actor_index}]")
        actor_id = int(actor.get("id"))
        if actor_id in actor_ids:
            raise ComparisonError(f"fixture case {case_id!r} repeats pedestrian id {actor_id}")
        actor_ids.add(actor_id)
        normalized_pedestrians.append(
            {
                "id": actor_id,
                "position": _finite_vector(
                    actor.get("position"),
                    field=f"{case_id}.pedestrians[{actor_index}].position",
                ),
                "velocity": _finite_vector(
                    actor.get("velocity"),
                    field=f"{case_id}.pedestrians[{actor_index}].velocity",
                ),
            }
        )

    known_outcome: dict[str, Any] | None = None
    if "known_contact_outcome" in case:
        known_outcome = _parse_known_contact_outcome(case_id, case["known_contact_outcome"])

    seen_ids.add(case_id)
    return {
        "case_id": case_id,
        "split": split,
        "start_position": _finite_vector(
            case.get("start_position"), field=f"{case_id}.start_position"
        ),
        "local_goal": _finite_vector(case.get("local_goal"), field=f"{case_id}.local_goal"),
        "pedestrians": normalized_pedestrians,
        "known_contact_outcome": known_outcome,
    }


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load and validate a committed fixture file with full provenance."""
    payload = _load_yaml(path, name="fixture")
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ComparisonError(f"fixture schema_version must be {FIXTURE_SCHEMA_VERSION}")
    if str(payload.get("split") or "") not in {EVALUATION_SPLIT, CALIBRATION_SPLIT}:
        raise ComparisonError(
            f"fixture split must be {EVALUATION_SPLIT!r} or {CALIBRATION_SPLIT!r}"
        )
    _require_fixture_provenance(payload, name=_display_path(path))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ComparisonError("fixture cases must be a non-empty list")

    seen_ids: set[str] = set()
    cases = [
        _normalize_fixture_case(raw_case, index=index, seen_ids=seen_ids)
        for index, raw_case in enumerate(raw_cases)
    ]
    declared_split = str(payload.get("split"))
    mismatched = [case["case_id"] for case in cases if case["split"] != declared_split]
    if mismatched:
        raise ComparisonError(
            f"fixture {_display_path(path)} declares split {declared_split!r} but cases "
            f"{mismatched} declare a different split"
        )
    return cases


def _verify_split_integrity(
    evaluation_cases: Sequence[Mapping[str, Any]],
    calibration_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed on calibration/evaluation split overlap and return the record."""
    evaluation_ids = [case["case_id"] for case in evaluation_cases]
    calibration_ids = [case["case_id"] for case in calibration_cases]
    overlap = sorted(set(evaluation_ids) & set(calibration_ids))
    if overlap:
        raise ComparisonError(
            "calibration and evaluation fixtures overlap on case ids: "
            f"{overlap}; split integrity requires disjoint fixtures"
        )
    for case in evaluation_cases:
        if case["split"] != EVALUATION_SPLIT:
            raise ComparisonError(
                f"evaluation fixture case {case['case_id']!r} is not in the held_out split"
            )
    for case in calibration_cases:
        if case["split"] != CALIBRATION_SPLIT:
            raise ComparisonError(
                f"calibration fixture case {case['case_id']!r} is not in the calibration split"
            )
    return {
        "evaluation_split": EVALUATION_SPLIT,
        "calibration_split": CALIBRATION_SPLIT,
        "evaluation_case_ids": evaluation_ids,
        "calibration_case_ids": calibration_ids,
        "case_ids_overlap": False,
        "disjoint": True,
    }


def _dataclass_from_config(config_class: Any, raw: Any, *, name: str) -> Any:
    """Construct one validated dataclass, normalizing YAML tuples."""
    values = _mapping(raw, name)
    if "lateral_offsets_m" in values:
        offsets = values["lateral_offsets_m"]
        if not isinstance(offsets, (list, tuple)):
            raise ComparisonError(f"{name}.lateral_offsets_m must be a list")
        values["lateral_offsets_m"] = tuple(float(item) for item in offsets)
    try:
        return config_class(**values)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"invalid {name}: {exc}") from exc


def _build_configs(
    payload: Mapping[str, Any],
) -> tuple[
    RiskEstimatorConfig,
    RankingWeights,
    TrajectoryVerifierConfig,
    ActuatorLimitsConfig,
    PrimitiveGeneratorConfig,
    RBFGeneratorConfig,
    int,
]:
    """Build the shared estimator, gate, ranking, and generator configs."""
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ComparisonError("config seed must be an integer")
    risk_values = _mapping(payload.get("risk_estimator"), "risk_estimator")
    if "seed" in risk_values and risk_values["seed"] != seed:
        raise ComparisonError("top-level seed and risk_estimator.seed must match")
    for field in ("horizon_steps", "dt_s", "n_samples", "velocity_std_m_s", "robot_radius_m"):
        if field in risk_values:
            _require_finite(risk_values[field], field=f"risk_estimator.{field}")
    if int(risk_values.get("horizon_steps", 0)) <= 0:
        raise ComparisonError("risk_estimator.horizon_steps must be positive")
    risk_values["seed"] = seed
    return (
        _dataclass_from_config(RiskEstimatorConfig, risk_values, name="risk_estimator"),
        _dataclass_from_config(
            RankingWeights, payload.get("ranking_weights", {}), name="ranking_weights"
        ),
        _dataclass_from_config(
            TrajectoryVerifierConfig, payload.get("verifier", {}), name="verifier"
        ),
        _dataclass_from_config(
            ActuatorLimitsConfig, payload.get("actuator_limits", {}), name="actuator_limits"
        ),
        _dataclass_from_config(
            PrimitiveGeneratorConfig,
            payload.get("primitive_generator", {}),
            name="primitive_generator",
        ),
        _dataclass_from_config(
            RBFGeneratorConfig, payload.get("rbf_generator", {}), name="rbf_generator"
        ),
        int(payload.get("candidate_budget", 0)),
    )


def _generator_config_hash(
    *,
    candidate_budget: int,
    risk_config: RiskEstimatorConfig,
    weights: RankingWeights,
    verifier_config: TrajectoryVerifierConfig,
    actuator_config: ActuatorLimitsConfig,
    primitive_config: PrimitiveGeneratorConfig,
    rbf_config: RBFGeneratorConfig,
) -> str:
    """Return a stable hash over every matched comparison input."""
    payload = {
        "candidate_budget": candidate_budget,
        "risk_estimator": asdict(risk_config),
        "ranking_weights": asdict(weights),
        "verifier": asdict(verifier_config),
        "actuator_limits": asdict(actuator_config),
        "primitive_generator": asdict(primitive_config),
        "rbf_generator": asdict(rbf_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pedestrians(case: Mapping[str, Any]) -> list[PedestrianState]:
    """Build the shared actor-prediction objects for one fixture case."""
    return [
        PedestrianState(
            id=int(actor["id"]),
            position=np.asarray(actor["position"], dtype=float),
            velocity=np.asarray(actor["velocity"], dtype=float),
        )
        for actor in case["pedestrians"]
    ]


def _case_digest(case: Mapping[str, Any]) -> str:
    """Return a stable digest of the matched start, goal, and actor inputs."""
    encoded = json.dumps(case, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validate_candidates(
    candidates: Sequence[CandidateAction],
    *,
    case: Mapping[str, Any],
    horizon_steps: int,
    budget: int,
) -> dict[str, Any]:
    """Validate the shared candidate contract and summarize finite/unique validity."""
    if len(candidates) != budget:
        raise ComparisonError(
            f"{case['case_id']}: generator produced {len(candidates)} candidates, "
            f"expected {budget}; the two generators must receive equal candidate budgets"
        )
    start = np.asarray(case["start_position"], dtype=float)
    action_ids = [candidate.action_id for candidate in candidates]
    invalid: list[str] = []
    finite_count = 0
    shape_count = 0
    start_count = 0
    stable_count = 0
    for candidate in candidates:
        if isinstance(candidate.action_id, str) and candidate.action_id:
            stable_count += 1
        else:
            invalid.append("missing_action_id")
        try:
            waypoints = candidate.as_array(horizon_steps=horizon_steps)
        except (TypeError, ValueError) as exc:
            invalid.append(f"{candidate.action_id}: {exc}")
            continue
        shape_count += int(waypoints.shape == (horizon_steps + 1, 2))
        finite_count += int(bool(np.all(np.isfinite(waypoints))))
        start_count += int(bool(np.allclose(waypoints[0], start, atol=1.0e-12)))
    duplicate_ids = sorted(
        action_id for action_id, count in Counter(action_ids).items() if count > 1
    )
    if duplicate_ids:
        invalid.append(f"duplicate_action_ids={duplicate_ids}")
    if len(invalid) > 0 or shape_count != budget or finite_count != budget or start_count != budget:
        raise ComparisonError(f"{case['case_id']}: invalid candidate contract: {invalid}")
    return {
        "expected_budget": budget,
        "candidate_count": len(candidates),
        "valid_count": budget,
        "invalid_count": 0,
        "finite_waypoint_sequences": finite_count,
        "shape_valid_sequences": shape_count,
        "start_state_valid_sequences": start_count,
        "stable_unique_action_ids": stable_count == budget and not duplicate_ids,
        "action_ids": action_ids,
        "status": "pass",
    }


def _rank_with_timing(
    candidates: Sequence[CandidateAction],
    pedestrians: Sequence[PedestrianState],
    *,
    risk_config: RiskEstimatorConfig,
    weights: RankingWeights,
    verifier_config: TrajectoryVerifierConfig,
    actuator_config: ActuatorLimitsConfig,
) -> tuple[list[CandidateRanking], float, list[dict[str, Any]]]:
    """Rank a matched set and measure isolated per-candidate wall time."""
    started = _perf_counter_ns()
    rankings = rank_trajectories(
        candidates,
        pedestrians,
        risk_config=risk_config,
        weights=weights,
        verifier_config=verifier_config,
        actuator_config=actuator_config,
    )
    total_ms = (_perf_counter_ns() - started) / 1.0e6
    per_candidate: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_started = _perf_counter_ns()
        rank_trajectories(
            [candidate],
            pedestrians,
            risk_config=risk_config,
            weights=weights,
            verifier_config=verifier_config,
            actuator_config=actuator_config,
        )
        per_candidate.append(
            {
                "action_id": candidate.action_id,
                "ranking_and_gate_ms": (_perf_counter_ns() - candidate_started) / 1.0e6,
            }
        )
    return rankings, total_ms, per_candidate


def _components_payload(record: CandidateRanking) -> dict[str, Any]:
    """Serialize the decomposed risk/time/jerk/length/clearance components."""
    components = record.components
    return {
        "calibrated_collision_risk": float(components.calibrated_collision_risk),
        "travel_time_s": float(components.travel_time_s),
        "integrated_jerk": float(components.integrated_jerk),
        "path_length_m": float(components.path_length_m),
        "clearance_penalty": float(components.clearance_penalty),
        "min_clearance_m": _clearance_json(components.min_clearance_m),
        "calibration_applied": bool(components.calibration_applied),
    }


def _clearance_json(min_clearance_m: float) -> float | None:
    """Return a JSON-safe clearance value, preserving the no-hazard sentinel.

    The estimator reports ``+inf`` minimum clearance when no pedestrian hazard is
    present; that sentinel is serialized as ``null`` (JSON has no Infinity). Any
    other non-finite clearance fails closed.
    """
    if math.isinf(min_clearance_m) and min_clearance_m > 0.0:
        return None
    if not math.isfinite(min_clearance_m):
        raise ComparisonError(
            f"min_clearance_m must be finite or the +inf no-hazard sentinel, got {min_clearance_m!r}"
        )
    return float(min_clearance_m)


def _candidate_payload(record: CandidateRanking) -> dict[str, Any]:
    """Serialize one full decomposed candidate record."""
    for field, value in (
        ("composite_score", record.composite_score),
        ("joint_contact_probability", record.joint_contact_probability),
        ("calibrated_collision_risk", record.components.calibrated_collision_risk),
        ("travel_time_s", record.components.travel_time_s),
        ("integrated_jerk", record.components.integrated_jerk),
        ("path_length_m", record.components.path_length_m),
        ("clearance_penalty", record.components.clearance_penalty),
    ):
        _require_finite(value, field=f"candidate {record.action_id} {field}")
    _clearance_json(record.components.min_clearance_m)
    uncertainty = record.estimate.uncertainty
    return {
        "action_id": record.action_id,
        "rank": int(record.rank),
        "eligible": bool(record.eligible),
        "composite_score": float(record.composite_score),
        "components": _components_payload(record),
        "hard_gate": {
            "verifier_decision": record.hard_gate.verifier_decision,
            "actuator_verdict": record.hard_gate.actuator_verdict,
            "violated_predicates": list(record.hard_gate.violated_predicates),
            "violated_limits": list(record.hard_gate.violated_limits),
            "ineligibility_reason": record.hard_gate.ineligibility_reason,
        },
        "risk_score_reliability": {
            "finite": math.isfinite(float(record.joint_contact_probability)),
            "in_range": 0.0 <= float(record.joint_contact_probability) <= 1.0,
            "mc_standard_error": float(uncertainty.mc_standard_error),
            "ci95_halfwidth": float(uncertainty.ci95_halfwidth),
            "abstained": bool(uncertainty.abstained),
            "abstention_reasons": list(uncertainty.abstention_reasons),
            "ood_actor_flags": list(uncertainty.ood_actor_flags),
        },
        "deterministic_contact_certain": bool(record.estimate.deterministic.contact_certain),
        "first_contact_step": int(record.estimate.deterministic.first_contact_step),
        "provenance_config_hash": record.provenance.config_hash,
    }


def _gate_summary(rankings: Sequence[CandidateRanking]) -> dict[str, Any]:
    """Summarize deterministic hard-gate outcomes without changing them."""
    verifier_rejections = [
        record
        for record in rankings
        if record.hard_gate.verifier_decision == DECISION_FALLBACK_BRAKE
    ]
    actuator_rejections = [
        record
        for record in rankings
        if record.hard_gate.actuator_verdict != VERDICT_ACTUATOR_FEASIBLE
    ]
    both_rejections = [
        record
        for record in rankings
        if not record.eligible
        and record.hard_gate.verifier_decision == DECISION_FALLBACK_BRAKE
        and record.hard_gate.actuator_verdict != VERDICT_ACTUATOR_FEASIBLE
    ]
    return {
        "candidate_count": len(rankings),
        "eligible_count": sum(record.eligible for record in rankings),
        "rejected_count": sum(not record.eligible for record in rankings),
        "verifier_fallback_brake_rejections": len(verifier_rejections),
        "actuator_non_feasible_rejections": len(actuator_rejections),
        "both_authoritative_gates_rejected": len(both_rejections),
        "verifier_decisions": dict(
            sorted(Counter(record.hard_gate.verifier_decision for record in rankings).items())
        ),
        "actuator_verdicts": dict(
            sorted(Counter(record.hard_gate.actuator_verdict for record in rankings).items())
        ),
        "authority": "verify_trajectory and evaluate_actuator_feasibility remain authoritative",
    }


def _selected_record(rankings: Sequence[CandidateRanking]) -> CandidateRanking | None:
    """Return the first eligible candidate, matching the ranker selection rule."""
    return next((record for record in rankings if record.eligible), None)


def _role(action_id: str) -> str:
    """Normalize generator-specific ids to a comparable candidate role."""
    return action_id.split("_", 1)[-1].rsplit("_", 1)[0]


def _selection_payload(
    rankings: Sequence[CandidateRanking], candidates: Sequence[CandidateAction]
) -> dict[str, Any]:
    """Serialize selected-candidate identity without treating ids as shared."""
    selected = _selected_record(rankings)
    if selected is None:
        return {
            "selected": False,
            "selected_action_id": None,
            "selected_slot": None,
            "selected_role": None,
        }
    return {
        "selected": True,
        "selected_action_id": selected.action_id,
        "selected_slot": next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.action_id == selected.action_id
        ),
        "selected_role": _role(selected.action_id),
        "selected_risk_score": float(selected.joint_contact_probability),
    }


def _reliability_diagnostic(
    rankings: Sequence[CandidateRanking],
    repeated: Sequence[CandidateRanking],
    *,
    declared_outcome: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the model-risk reliability diagnostic for one arm.

    Where a fixture declares a known contact outcome for a candidate role (by
    default the ``straight`` reference path), the model's deterministic
    ``contact_certain`` flag is compared against that declared outcome for the
    role-matched candidate(s). Finite/in-range/provenance/repeatability checks
    are always reported; calibration is never evaluated.
    """
    repeated_by_id = {record.action_id: record for record in repeated}
    per_candidate: list[dict[str, Any]] = []
    finite_count = 0
    in_range_count = 0
    provenance_count = 0
    repeatable_count = 0
    role_candidates: list[CandidateRanking] = []
    if declared_outcome is not None:
        role = str(declared_outcome["candidate_role"])
        role_candidates = [record for record in rankings if _role(record.action_id) == role]
    agreeing_count = 0
    disagreeing: list[str] = []
    for record in rankings:
        score = float(record.joint_contact_probability)
        finite = math.isfinite(score)
        finite_count += int(finite)
        in_range_count += int(finite and 0.0 <= score <= 1.0)
        provenance = record.provenance
        provenance_complete = all(
            bool(value)
            for value in (
                provenance.estimator_id,
                provenance.forecast_model,
                provenance.geometry_version,
                provenance.config_hash,
                provenance.action_id,
                provenance.action_representation,
            )
        )
        provenance_count += int(provenance_complete)
        repeatable_count += int(
            record.action_id in repeated_by_id
            and math.isclose(
                score,
                repeated_by_id[record.action_id].joint_contact_probability,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )
        model_contact_certain = bool(record.estimate.deterministic.contact_certain)
        if record in role_candidates:
            agrees = model_contact_certain == declared_outcome["contact_certain"]
            agreeing_count += int(agrees)
            if not agrees:
                disagreeing.append(record.action_id)
        per_candidate.append(
            {
                "action_id": record.action_id,
                "model_contact_certain": model_contact_certain,
                "risk_score": score,
                "finite": finite,
                "in_range": 0.0 <= score <= 1.0,
                "provenance_complete": provenance_complete,
                "repeatable": bool(
                    record.action_id in repeated_by_id
                    and math.isclose(
                        score,
                        repeated_by_id[record.action_id].joint_contact_probability,
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    )
                ),
            }
        )
    model_status = (
        "pass"
        if finite_count == len(rankings)
        and in_range_count == len(rankings)
        and provenance_count == len(rankings)
        and repeatable_count == len(rankings)
        else "inconclusive"
    )
    if declared_outcome is not None:
        outcome_status = "pass" if not disagreeing else "inconclusive"
        outcome_summary: dict[str, Any] = {
            "declared_outcome_present": True,
            "declared_contact_certain": bool(declared_outcome["contact_certain"]),
            "candidate_role": str(declared_outcome["candidate_role"]),
            "reason": str(declared_outcome.get("reason") or ""),
            "candidates_with_declared_outcome": len(role_candidates),
            "agreeing_candidates": agreeing_count,
            "disagreeing_candidates": disagreeing,
            "status": outcome_status,
        }
    else:
        outcome_summary = {
            "declared_outcome_present": False,
            "declared_contact_certain": None,
            "candidate_role": None,
            "reason": "no known contact outcome declared for this fixture",
            "candidates_with_declared_outcome": 0,
            "agreeing_candidates": 0,
            "disagreeing_candidates": [],
            "status": "not_applicable",
        }
    return {
        "model_score_checks": {
            "candidate_count": len(rankings),
            "finite_risk_scores": finite_count,
            "in_range_risk_scores": in_range_count,
            "complete_provenance_rows": provenance_count,
            "repeatable_risk_scores": repeatable_count,
            "calibration_status": "not_evaluated; model score reliability checks only",
            "status": model_status,
        },
        "declared_outcome_check": outcome_summary,
        "per_candidate": per_candidate,
    }


def _unavailable_denominators(
    *,
    reliability_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report every denominator that could not be computed, with its reason."""
    unavailable: list[dict[str, Any]] = []
    declared_cases = sum(
        int(row["declared_outcome_present"]) for row in reliability_rows if row is not None
    )
    if declared_cases == 0:
        unavailable.append(
            {
                "metric": "model_risk_declared_outcome_agreement",
                "denominator": None,
                "reason": "no held-out fixture declares a known contact outcome",
            }
        )
    no_eligible_primitive = sum(
        int(not row["deterministic_primitive"]["selected"]) for row in selection_rows
    )
    no_eligible_rbf = sum(int(not row["rbf"]["selected"]) for row in selection_rows)
    for generator_name, count in (
        ("deterministic_primitive", no_eligible_primitive),
        ("rbf", no_eligible_rbf),
    ):
        if count:
            unavailable.append(
                {
                    "metric": f"selection_{generator_name}",
                    "denominator": None,
                    "reason": f"{count} case(s) had no eligible candidate; selection identity "
                    "could not be compared",
                }
            )
    return unavailable


def _aggregate_counts(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """Sum integer fields across fixture cases."""
    result: dict[str, Any] = {key: sum(int(row[key]) for row in rows) for key in keys}
    result["case_count"] = len(rows)
    return result


def _verify_pinned_hash(config: Mapping[str, Any], generator_config_hash: str) -> str:
    """Fail closed when the config's pinned generator/config hash disagrees."""
    expected_hash = config.get("expected_generator_config_hash")
    if not isinstance(expected_hash, str) or not expected_hash.strip():
        raise ComparisonError(
            "config expected_generator_config_hash is required; generator/config hashing "
            "must be pinned to fail closed on mismatch"
        )
    if expected_hash.strip() != generator_config_hash:
        raise ComparisonError(
            "generator/config hash mismatch: config declares "
            f"{expected_hash.strip()}, computed {generator_config_hash}; regenerate the "
            "expected hash or the comparison fails closed"
        )
    return expected_hash.strip()


def _evaluate_case(
    case: Mapping[str, Any],
    generators: Mapping[str, tuple[Callable[..., list[CandidateAction]], Any]],
    *,
    risk_config: RiskEstimatorConfig,
    weights: RankingWeights,
    verifier_config: TrajectoryVerifierConfig,
    actuator_config: ActuatorLimitsConfig,
    candidate_budget: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate every generator arm for one fixture case.

    Returns the case payload (with per-arm decomposed candidates) and per-arm
    raw timing that the caller aggregates into the top-level timing sections.
    """
    pedestrians = _pedestrians(case)
    case_payload: dict[str, Any] = {
        "case_id": case["case_id"],
        "split": case["split"],
        "input_digest": _case_digest(case),
        "declared_contact_outcome": (
            dict(case["known_contact_outcome"]) if case["known_contact_outcome"] else None
        ),
        "arms": {},
    }
    arm_timing: dict[str, dict[str, Any]] = {}
    for generator_name, (generator, generator_config) in generators.items():
        started = _perf_counter_ns()
        candidates = generator(
            case["start_position"],
            case["local_goal"],
            horizon_steps=risk_config.horizon_steps,
            dt_s=risk_config.dt_s,
            config=generator_config,
        )
        generation_ms = (_perf_counter_ns() - started) / 1.0e6
        validity = _validate_candidates(
            candidates,
            case=case,
            horizon_steps=risk_config.horizon_steps,
            budget=candidate_budget,
        )
        rankings, ranking_ms, per_candidate_timing = _rank_with_timing(
            candidates,
            pedestrians,
            risk_config=risk_config,
            weights=weights,
            verifier_config=verifier_config,
            actuator_config=actuator_config,
        )
        repeated = rank_trajectories(
            candidates,
            pedestrians,
            risk_config=risk_config,
            weights=weights,
            verifier_config=verifier_config,
            actuator_config=actuator_config,
        )
        gates = _gate_summary(rankings)
        reliability = _reliability_diagnostic(
            rankings, repeated, declared_outcome=case["known_contact_outcome"]
        )
        selection = _selection_payload(rankings, candidates)
        case_payload["arms"][generator_name] = {
            "candidate_validity": validity,
            "hard_gate_rejection": gates,
            "selection": selection,
            "decomposed_candidates": [_candidate_payload(record) for record in rankings],
            "risk_score_reliability": reliability,
            "timing": {
                "generation_ms": round(generation_ms, _TIMING_DECIMALS),
                "ranking_and_gates_ms": round(ranking_ms, _TIMING_DECIMALS),
                "total_ms": round(generation_ms + ranking_ms, _TIMING_DECIMALS),
            },
        }
        arm_timing[generator_name] = {
            "generation_ms": generation_ms,
            "ranking_ms": ranking_ms,
            "candidate_count": len(candidates),
            "per_candidate": per_candidate_timing,
        }
    return case_payload, arm_timing


def _build_selection_rows(case_payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build per-case selection-comparison rows across the two generator arms."""
    selection_rows: list[dict[str, Any]] = []
    for case_payload in case_payloads:
        primitive_selection = case_payload["arms"]["deterministic_primitive"]["selection"]
        rbf_selection = case_payload["arms"]["rbf"]["selection"]
        selection_rows.append(
            {
                "case_id": case_payload["case_id"],
                "deterministic_primitive": primitive_selection,
                "rbf": rbf_selection,
                "selected_slot_changed": primitive_selection["selected_slot"]
                != rbf_selection["selected_slot"],
                "selected_role_changed": primitive_selection["selected_role"]
                != rbf_selection["selected_role"],
                "generator_choice_changes_selection": (
                    primitive_selection["selected"]
                    and rbf_selection["selected"]
                    and primitive_selection["selected_slot"] != rbf_selection["selected_slot"]
                ),
            }
        )
    return selection_rows


def _build_reliability_rows(case_payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build per-case reliability summary rows across the two generator arms."""
    reliability_rows: list[Mapping[str, Any]] = []
    for case_payload in case_payloads:
        primitive_reliability = case_payload["arms"]["deterministic_primitive"][
            "risk_score_reliability"
        ]
        reliability_rows.append(
            {
                "case_id": case_payload["case_id"],
                "declared_outcome_present": primitive_reliability["declared_outcome_check"][
                    "declared_outcome_present"
                ],
            }
        )
    return reliability_rows


def build_report(config_path: Path, *, generated_at_utc: str | None = None) -> dict[str, Any]:
    """Build the complete matched diagnostic report from a committed config."""
    config = _load_yaml(config_path, name="config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ComparisonError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION}, got {config.get('schema_version')!r}"
        )
    if config.get("evidence_status") != "diagnostic_only":
        raise ComparisonError("config evidence_status must be diagnostic_only")

    evaluation_path = _resolve_repo_path(config.get("fixture_path"), field="fixture_path")
    calibration_path = _resolve_repo_path(
        config.get("calibration_fixture_path"), field="calibration_fixture_path"
    )
    evaluation_cases = _load_fixture(evaluation_path)
    calibration_cases = _load_fixture(calibration_path)
    split_integrity = _verify_split_integrity(evaluation_cases, calibration_cases)

    (
        risk_config,
        weights,
        verifier_config,
        actuator_config,
        primitive_config,
        rbf_config,
        candidate_budget,
    ) = _build_configs(config)
    if candidate_budget < 3:
        raise ComparisonError("candidate_budget must be at least three")
    if len(evaluation_cases) < 2:
        raise ComparisonError("matched comparison requires at least two held-out fixture cases")

    generator_config_hash = _generator_config_hash(
        candidate_budget=candidate_budget,
        risk_config=risk_config,
        weights=weights,
        verifier_config=verifier_config,
        actuator_config=actuator_config,
        primitive_config=primitive_config,
        rbf_config=rbf_config,
    )
    expected_hash = _verify_pinned_hash(config, generator_config_hash)

    generators: dict[str, tuple[Callable[..., list[CandidateAction]], Any]] = {
        "deterministic_primitive": (generate_primitive_candidates, primitive_config),
        "rbf": (generate_rbf_candidates, rbf_config),
    }
    case_payloads: list[dict[str, Any]] = []
    per_generator_validity: dict[str, list[dict[str, Any]]] = {name: [] for name in generators}
    per_generator_gates: dict[str, list[dict[str, Any]]] = {name: [] for name in generators}
    per_generator_reliability: dict[str, list[dict[str, Any]]] = {name: [] for name in generators}
    timing_generation: dict[str, dict[str, Any]] = {
        name: {"total_ms": 0.0, "total_candidates": 0, "per_case": []} for name in generators
    }
    timing_ranking: dict[str, dict[str, Any]] = {
        name: {"total_ms": 0.0, "total_candidates": 0, "per_case": []} for name in generators
    }

    for case in evaluation_cases:
        case_payload, arm_timing = _evaluate_case(
            case,
            generators,
            risk_config=risk_config,
            weights=weights,
            verifier_config=verifier_config,
            actuator_config=actuator_config,
            candidate_budget=candidate_budget,
        )
        case_payloads.append(case_payload)
        for generator_name, timing in arm_timing.items():
            validity = case_payload["arms"][generator_name]["candidate_validity"]
            gates = case_payload["arms"][generator_name]["hard_gate_rejection"]
            reliability = case_payload["arms"][generator_name]["risk_score_reliability"]
            per_generator_validity[generator_name].append(validity)
            per_generator_gates[generator_name].append(gates)
            per_generator_reliability[generator_name].append(reliability)
            timing_generation[generator_name]["total_ms"] += timing["generation_ms"]
            timing_generation[generator_name]["total_candidates"] += timing["candidate_count"]
            timing_generation[generator_name]["per_case"].append(
                {
                    "case_id": case["case_id"],
                    "candidate_count": timing["candidate_count"],
                    "generation_ms": round(timing["generation_ms"], _TIMING_DECIMALS),
                    "generation_ms_per_candidate": round(
                        timing["generation_ms"] / timing["candidate_count"], _TIMING_DECIMALS
                    ),
                }
            )
            timing_ranking[generator_name]["total_ms"] += timing["ranking_ms"]
            timing_ranking[generator_name]["total_candidates"] += timing["candidate_count"]
            timing_ranking[generator_name]["per_case"].append(
                {
                    "case_id": case["case_id"],
                    "ranking_and_gates_ms": round(timing["ranking_ms"], _TIMING_DECIMALS),
                    "ranking_and_gates_ms_per_candidate": round(
                        timing["ranking_ms"] / timing["candidate_count"], _TIMING_DECIMALS
                    ),
                    "per_candidate": [
                        {
                            "action_id": row["action_id"],
                            "ranking_and_gate_ms": round(
                                row["ranking_and_gate_ms"], _TIMING_DECIMALS
                            ),
                        }
                        for row in timing["per_candidate"]
                    ],
                }
            )

    selection_rows = _build_selection_rows(case_payloads)
    reliability_rows = _build_reliability_rows(case_payloads)

    timing_total_ms: dict[str, float] = {}
    for generator_name in generators:
        timing_total_ms[generator_name] = round(
            timing_generation[generator_name]["total_ms"]
            + timing_ranking[generator_name]["total_ms"],
            _TIMING_DECIMALS,
        )

    unavailable = _unavailable_denominators(
        reliability_rows=reliability_rows, selection_rows=selection_rows
    )

    resolved_generated_at = generated_at_utc or str(config.get("pinned_generated_at_utc") or "")
    if resolved_generated_at:
        pinned = True
    else:
        pinned = False
        resolved_generated_at = dt.datetime.now(dt.UTC).isoformat()

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evidence_status": "diagnostic_only",
        "diagnostic_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "caveats": [
            "RBF is a deterministic radial-basis-function proposal, not a trained policy.",
            "Risk scores are constant-velocity model scores; calibration is not evaluated.",
            "Timing is measured local offline wall time and is not an online performance claim.",
            "Hard-gate rejections are reported as exclusions from selection, not success evidence.",
            "Planner-loop wiring, online adaptation, nominal benchmark execution, safety, and "
            "real-world claims are deferred.",
        ],
        "uncertainty": {
            "confidence": "diagnostic-only; no planner-improvement conclusion",
            "risk_score_interpretation": "finite/range/provenance/repeatability checks plus "
            "deterministic contact-outcome agreement only",
            "selection_difference_interpretation": "descriptive on a small held-out fixture split",
        },
        "provenance": {
            "command": " ".join(shlex.quote(argument) for argument in sys.argv),
            "config_path": _display_path(config_path),
            "config_sha256": _sha256_file(config_path),
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "fixture_path": _display_path(evaluation_path),
            "fixture_sha256": _sha256_file(evaluation_path),
            "calibration_fixture_path": _display_path(calibration_path),
            "calibration_fixture_sha256": _sha256_file(calibration_path),
            "seed": risk_config.seed,
            "generator_config_hash": generator_config_hash,
            "expected_generator_config_hash": expected_hash.strip(),
            "git_commit_sha": _git_head(),
            "git_status_short": _git_status(),
            "generated_at_utc": resolved_generated_at,
            "pinned_generated_at_utc": pinned,
        },
        "split_integrity": split_integrity,
        "matched_comparison": {
            "baseline": "deterministic_primitive",
            "candidate_generators": ["deterministic_primitive", "rbf"],
            "evaluation_case_count": len(evaluation_cases),
            "calibration_case_count": len(calibration_cases),
            "candidate_budget": candidate_budget,
            "same_start_states_local_goals_actor_predictions": True,
            "same_risk_estimator_config": True,
            "same_ranking_weights": True,
            "same_hard_gate_configs": True,
            "same_horizon_and_timestep": True,
            "hard_gates": ["verify_trajectory", "evaluate_actuator_feasibility"],
            "default_planner_behavior_changed": False,
            "planner_loop_wiring": "not_run; intentionally out of scope",
        },
        "fallback_degraded_exclusions": {
            "fallback_rows_excluded": 0,
            "degraded_rows_excluded": 0,
            "provenance_incomplete_rows_excluded": 0,
            "invalid_rows_excluded": 0,
            "non_finite_rows_excluded": 0,
            "policy": "invalid, degraded, or provenance-incomplete inputs fail closed; no "
            "fallback row contributes to a success denominator",
        },
        "candidate_validity": {
            "by_generator": {
                generator_name: _aggregate_counts(
                    per_generator_validity[generator_name],
                    (
                        "expected_budget",
                        "candidate_count",
                        "valid_count",
                        "invalid_count",
                        "finite_waypoint_sequences",
                        "shape_valid_sequences",
                        "start_state_valid_sequences",
                    ),
                )
                for generator_name in generators
            },
            "per_case": {
                case_payload["case_id"]: {
                    generator_name: case_payload["arms"][generator_name]["candidate_validity"]
                    for generator_name in generators
                }
                for case_payload in case_payloads
            },
        },
        "hard_gate_rejection": {
            "by_generator": {
                generator_name: _aggregate_counts(
                    per_generator_gates[generator_name],
                    (
                        "candidate_count",
                        "eligible_count",
                        "rejected_count",
                        "verifier_fallback_brake_rejections",
                        "actuator_non_feasible_rejections",
                        "both_authoritative_gates_rejected",
                    ),
                )
                for generator_name in generators
            },
            "per_case": {
                case_payload["case_id"]: {
                    generator_name: case_payload["arms"][generator_name]["hard_gate_rejection"]
                    for generator_name in generators
                }
                for case_payload in case_payloads
            },
        },
        "selection_differences": {
            "cases_with_selected_slot_change": sum(
                row["selected_slot_changed"] for row in selection_rows
            ),
            "cases_with_selected_role_change": sum(
                row["selected_role_changed"] for row in selection_rows
            ),
            "cases_where_generator_choice_changes_selection": sum(
                row["generator_choice_changes_selection"] for row in selection_rows
            ),
            "cases_with_no_eligible_deterministic": sum(
                not row["deterministic_primitive"]["selected"] for row in selection_rows
            ),
            "cases_with_no_eligible_rbf": sum(not row["rbf"]["selected"] for row in selection_rows),
            "per_case": selection_rows,
        },
        "model_risk_reliability": {
            "declared_outcome_cases": sum(
                int(bool(row["declared_outcome_present"])) for row in reliability_rows
            ),
            "calibration_status": "not_evaluated; model score reliability checks only",
            "by_generator": {
                generator_name: _aggregate_reliability(per_generator_reliability[generator_name])
                for generator_name in generators
            },
            "per_case": {
                case_payload["case_id"]: {
                    generator_name: case_payload["arms"][generator_name]["risk_score_reliability"]
                    for generator_name in generators
                }
                for case_payload in case_payloads
            },
        },
        "timing": {
            "measurement_boundary": "offline local wall-clock diagnostic; not an online performance claim",
            "generation": {
                generator_name: {
                    "total_ms": round(
                        timing_generation[generator_name]["total_ms"], _TIMING_DECIMALS
                    ),
                    "total_candidates": timing_generation[generator_name]["total_candidates"],
                    "per_case": timing_generation[generator_name]["per_case"],
                }
                for generator_name in generators
            },
            "ranking_and_hard_gates": {
                generator_name: {
                    "total_ms": round(timing_ranking[generator_name]["total_ms"], _TIMING_DECIMALS),
                    "total_candidates": timing_ranking[generator_name]["total_candidates"],
                    "per_case": timing_ranking[generator_name]["per_case"],
                }
                for generator_name in generators
            },
            "total": timing_total_ms,
        },
        "unavailable_denominators": unavailable,
        "cases": case_payloads,
    }
    return report


def _aggregate_reliability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate model-score reliability checks while retaining outcome checks."""
    checks = [row["model_score_checks"] for row in rows]
    keys = (
        "candidate_count",
        "finite_risk_scores",
        "in_range_risk_scores",
        "complete_provenance_rows",
        "repeatable_risk_scores",
    )
    result = _aggregate_counts(checks, keys)
    result["calibration_status"] = "not_evaluated; model score reliability checks only"
    result["status"] = (
        "pass"
        if result["finite_risk_scores"] == result["candidate_count"]
        and result["in_range_risk_scores"] == result["candidate_count"]
        and result["complete_provenance_rows"] == result["candidate_count"]
        and result["repeatable_risk_scores"] == result["candidate_count"]
        else "inconclusive"
    )
    declared_cases = [row["declared_outcome_check"] for row in rows]
    present = [row for row in declared_cases if row["declared_outcome_present"]]
    result["declared_outcome_cases"] = len(present)
    result["declared_outcome_agreeing_candidates"] = sum(
        int(row["agreeing_candidates"]) for row in present
    )
    result["declared_outcome_disagreeing_candidates"] = sorted(
        {action_id for row in present for action_id in row["disagreeing_candidates"]}
    )
    result["declared_outcome_status"] = (
        "not_applicable"
        if not present
        else ("pass" if not result["declared_outcome_disagreeing_candidates"] else "inconclusive")
    )
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable report with claim boundaries first."""
    selection = report["selection_differences"]
    reliability = report["model_risk_reliability"]["by_generator"]
    gates = report["hard_gate_rejection"]["by_generator"]
    validity = report["candidate_validity"]["by_generator"]
    provenance = report["provenance"]
    unavailable = report["unavailable_denominators"]

    lines = [
        "# Risk-ranker generator comparison (primitive vs RBF)",
        "",
        f"Claim boundary: `{report['claim_boundary']}`",
        f"Evidence status: `{report['evidence_status']}`.",
        "",
        "Caveats: fallback/degraded/provenance-incomplete rows are excluded; timing is local "
        "offline wall time; risk reliability checks are not calibration; planner-loop wiring "
        "is not run.",
        "",
        f"Uncertainty: {report['uncertainty']['confidence']}; selection differences are "
        "descriptive on the small held-out fixture split.",
        "",
        "## Matched comparison",
        "",
        f"- Baseline: `{report['matched_comparison']['baseline']}`",
        f"- Evaluation cases: `{report['matched_comparison']['evaluation_case_count']}`; "
        f"calibration cases: `{report['matched_comparison']['calibration_case_count']}`; "
        f"candidate budget: `{report['matched_comparison']['candidate_budget']}`.",
        f"- Calibration/evaluation case ids disjoint: `{report['split_integrity']['disjoint']}`.",
        f"- Seed: `{provenance['seed']}`; commit: `{provenance['git_commit_sha']}`.",
        f"- Fixture: `{provenance['fixture_path']}`.",
        "",
        "## Candidate validity",
        "",
    ]
    for name, row in validity.items():
        lines.append(
            f"- `{name}`: {row['valid_count']}/{row['candidate_count']} valid candidate rows; "
            f"finite={row['finite_waypoint_sequences']}, shape-valid={row['shape_valid_sequences']}."
        )
    lines.extend(["", "## Hard-gate rejection", ""])
    for name, row in gates.items():
        lines.append(
            f"- `{name}`: rejected={row['rejected_count']}; "
            f"verifier fallback-brake={row['verifier_fallback_brake_rejections']}; "
            f"actuator non-feasible={row['actuator_non_feasible_rejections']}."
        )
    lines.extend(
        [
            "",
            "## Selection differences",
            "",
            f"- Cases with selected-slot change: `{selection['cases_with_selected_slot_change']}`",
            f"- Cases with selected-role change: `{selection['cases_with_selected_role_change']}`",
            f"- Cases where generator choice changes selection: "
            f"`{selection['cases_where_generator_choice_changes_selection']}`",
            f"- Cases without an eligible deterministic selection: "
            f"`{selection['cases_with_no_eligible_deterministic']}`",
            f"- Cases without an eligible RBF selection: "
            f"`{selection['cases_with_no_eligible_rbf']}`",
            "",
            "## Model-risk reliability",
            "",
            f"- Fixtures with a declared known contact outcome: "
            f"`{report['model_risk_reliability']['declared_outcome_cases']}`.",
        ]
    )
    for name, row in reliability.items():
        lines.append(
            f"- `{name}`: model-score status=`{row['status']}`, "
            f"finite/in-range/provenance/repeatable="
            f"{row['finite_risk_scores']}/{row['in_range_risk_scores']}/"
            f"{row['complete_provenance_rows']}/{row['repeatable_risk_scores']}; "
            f"declared-outcome status=`{row['declared_outcome_status']}`, "
            f"agreeing={row['declared_outcome_agreeing_candidates']}."
        )
    lines.extend(
        [
            "",
            "## Timing",
            "",
            "Timing is measured offline wall-clock time and is not an online performance claim.",
        ]
    )
    for name, row in report["timing"]["generation"].items():
        lines.append(
            f"- `{name}` generation: total={row['total_ms']:.3f} ms; "
            f"candidates={row['total_candidates']}."
        )
    for name, row in report["timing"]["ranking_and_hard_gates"].items():
        lines.append(f"- `{name}` ranking+hard-gates: total={row['total_ms']:.3f} ms.")
    lines.extend(["", "## Unavailable denominators", ""])
    if unavailable:
        for entry in unavailable:
            lines.append(f"- `{entry['metric']}`: unavailable -- {entry['reason']}")
    else:
        lines.append("- None; every reported denominator was computed.")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Command: `{provenance['command']}`",
            f"- Config SHA-256: `{provenance['config_sha256']}`",
            f"- Fixture SHA-256: `{provenance['fixture_sha256']}`",
            f"- Generator/config hash: `{provenance['generator_config_hash']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def check_config(config_path: Path) -> None:
    """Validate a config fully without writing a report.

    Builds the report (which exercises every fail-closed gate) and returns
    normally on success, raising :class:`ComparisonError` on any failure.
    """
    build_report(config_path)
    print(f"check-config pass: {_display_path(config_path)}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the offline comparison CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-md", type=Path, help="Optional Markdown report output path")
    parser.add_argument(
        "--generated-at",
        type=str,
        help="Pinned UTC ISO-8601 generation timestamp for deterministic reports.",
    )
    parser.add_argument(
        "--check-config",
        nargs="?",
        const=True,
        type=Path,
        default=False,
        help="Validate the config (including fail-closed gates) and exit without writing. "
        "Optionally takes the config path directly.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and write JSON plus optional Markdown output."""
    args = _parse_args(argv)
    try:
        if args.check_config:
            check_config(
                (
                    args.check_config if isinstance(args.check_config, Path) else args.config
                ).resolve()
            )
            return 0
        report = build_report(args.config.resolve(), generated_at_utc=args.generated_at)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, TypeError, ValueError, ComparisonError, KeyError) as exc:
        print(f"issue #6768 generator comparison failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": "pass", "output": str(args.output), "schema_version": REPORT_SCHEMA_VERSION}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
