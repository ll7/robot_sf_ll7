#!/usr/bin/env python3
"""Run the issue #6676 matched RBF-vs-primitive ranker diagnostic.

The runner uses one committed held-out fixture split, one risk configuration,
one candidate budget, and the same deterministic hard-gate configuration for
both candidate generators. It is an offline diagnostic only: it does not call
``map_runner`` or any planner control loop and does not make calibration,
safety, or planner-improvement claims.
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from robot_sf.benchmark.actuator_feasibility import ActuatorLimitsConfig
from robot_sf.benchmark.trajectory_verifier import TrajectoryVerifierConfig
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmark/risk_aware_trajectory_ranker_comparison.yaml"
REPORT_SCHEMA_VERSION = "risk_aware_trajectory_ranker_comparison.v1"
FIXTURE_SCHEMA_VERSION = "risk_aware_trajectory_ranker_comparison_fixture.v1"
CLAIM_BOUNDARY = (
    "smoke/diagnostic only: compares finite RBF candidate proposals with the deterministic "
    "primitive baseline under matched held-out fixtures; it does not establish planner "
    "improvement, calibrated collision probability, safety, or nominal benchmark evidence"
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a YAML mapping or fail closed with a useful field name."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(value)


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    """Load a required YAML mapping."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {name}: {path}: {exc}") from exc
    return _mapping(payload, name)


def _resolve_repo_path(value: Any, *, field: str) -> Path:
    """Resolve a config path relative to the repository root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
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
        raise ValueError("git rev-parse HEAD did not return an exact 40-character SHA")
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


def _finite_vector(value: Any, *, field: str) -> list[float]:
    """Validate a finite two-dimensional fixture vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite length-two vector")
    return [float(item) for item in array]


def _normalize_fixture_case(raw_case: Any, *, index: int, seen_ids: set[str]) -> dict[str, Any]:
    """Validate and normalize one held-out fixture case."""
    case = _mapping(raw_case, f"fixture.cases[{index}]")
    case_id = str(case.get("case_id") or "")
    if not case_id or case_id in seen_ids:
        raise ValueError(f"fixture case ids must be non-empty and unique: {case_id!r}")
    if case.get("split") != "held_out":
        raise ValueError(f"fixture case {case_id!r} is not in the held_out split")
    if case.get("status", "valid") != "valid":
        raise ValueError(f"fixture case {case_id!r} is not a valid evidence row")
    pedestrians = case.get("pedestrians", [])
    if not isinstance(pedestrians, list):
        raise ValueError(f"fixture case {case_id!r}.pedestrians must be a list")
    normalized_pedestrians: list[dict[str, Any]] = []
    actor_ids: set[int] = set()
    for actor_index, raw_actor in enumerate(pedestrians):
        actor = _mapping(raw_actor, f"fixture case {case_id}.pedestrians[{actor_index}]")
        actor_id = int(actor.get("id"))
        if actor_id in actor_ids:
            raise ValueError(f"fixture case {case_id!r} repeats pedestrian id {actor_id}")
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
    seen_ids.add(case_id)
    return {
        "case_id": case_id,
        "split": "held_out",
        "start_position": _finite_vector(
            case.get("start_position"), field=f"{case_id}.start_position"
        ),
        "local_goal": _finite_vector(case.get("local_goal"), field=f"{case_id}.local_goal"),
        "pedestrians": normalized_pedestrians,
    }


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load and validate the committed held-out fixture rows."""
    payload = _load_yaml(path, name="fixture")
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(f"fixture schema_version must be {FIXTURE_SCHEMA_VERSION}")
    if payload.get("split") != "held_out":
        raise ValueError("fixture split must be held_out")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("fixture cases must be a non-empty list")

    seen_ids: set[str] = set()
    return [
        _normalize_fixture_case(raw_case, index=index, seen_ids=seen_ids)
        for index, raw_case in enumerate(raw_cases)
    ]


def _dataclass_from_config(config_class: Any, raw: Any, *, name: str) -> Any:
    """Construct one validated dataclass, normalizing YAML tuples."""
    values = _mapping(raw, name)
    if "lateral_offsets_m" in values:
        offsets = values["lateral_offsets_m"]
        if not isinstance(offsets, (list, tuple)):
            raise ValueError(f"{name}.lateral_offsets_m must be a list")
        values["lateral_offsets_m"] = tuple(float(value) for value in offsets)
    return config_class(**values)


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
        raise ValueError("config seed must be an integer")
    risk_values = _mapping(payload.get("risk_estimator"), "risk_estimator")
    if "seed" in risk_values and risk_values["seed"] != seed:
        raise ValueError("top-level seed and risk_estimator.seed must match")
    risk_values["seed"] = seed
    return (
        RiskEstimatorConfig(**risk_values),
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
    """Validate the shared CandidateAction contract and summarize it."""
    if len(candidates) != budget:
        raise ValueError(
            f"{case['case_id']}: generator produced {len(candidates)} candidates, expected {budget}"
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
        raise ValueError(f"{case['case_id']}: invalid candidate contract: {invalid}")
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
    started = time.perf_counter_ns()
    rankings = rank_trajectories(
        candidates,
        pedestrians,
        risk_config=risk_config,
        weights=weights,
        verifier_config=verifier_config,
        actuator_config=actuator_config,
    )
    total_ms = (time.perf_counter_ns() - started) / 1.0e6
    per_candidate: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_started = time.perf_counter_ns()
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
                "ranking_and_gate_ms": (time.perf_counter_ns() - candidate_started) / 1.0e6,
            }
        )
    return rankings, total_ms, per_candidate


def _gate_summary(rankings: Sequence[CandidateRanking]) -> dict[str, Any]:
    """Summarize deterministic hard-gate outcomes without changing them."""
    verifier_rejections = [
        record for record in rankings if record.hard_gate.verifier_decision == "fallback_brake"
    ]
    actuator_rejections = [
        record for record in rankings if record.hard_gate.actuator_verdict != "actuator_feasible"
    ]
    both_rejections = [
        record
        for record in rankings
        if not record.eligible
        and record.hard_gate.verifier_decision == "fallback_brake"
        and record.hard_gate.actuator_verdict != "actuator_feasible"
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


def _risk_reliability(
    rankings: Sequence[CandidateRanking], repeated: Sequence[CandidateRanking]
) -> dict[str, Any]:
    """Report finite/range/provenance/repeatability checks, not calibration."""
    finite_count = 0
    in_range_count = 0
    provenance_count = 0
    uncertainty_rows: list[dict[str, Any]] = []
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
        uncertainty = record.estimate.uncertainty
        uncertainty_rows.append(
            {
                "action_id": record.action_id,
                "risk_score": score,
                "mc_standard_error": float(uncertainty.mc_standard_error),
                "ci95_halfwidth": float(uncertainty.ci95_halfwidth),
                "abstained": bool(uncertainty.abstained),
                "ood_actor_flags": list(uncertainty.ood_actor_flags),
            }
        )
    repeated_by_id = {record.action_id: record for record in repeated}
    repeatable_count = sum(
        record.action_id in repeated_by_id
        and math.isclose(
            record.joint_contact_probability,
            repeated_by_id[record.action_id].joint_contact_probability,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for record in rankings
    )
    halfwidths = [row["ci95_halfwidth"] for row in uncertainty_rows]
    return {
        "candidate_count": len(rankings),
        "finite_risk_scores": finite_count,
        "in_range_risk_scores": in_range_count,
        "complete_provenance_rows": provenance_count,
        "repeatable_risk_scores": repeatable_count,
        "abstained_rows": sum(row["abstained"] for row in uncertainty_rows),
        "mean_ci95_halfwidth": float(np.mean(halfwidths)) if halfwidths else 0.0,
        "max_ci95_halfwidth": float(np.max(halfwidths)) if halfwidths else 0.0,
        "calibration_status": "not_evaluated; model score reliability checks only",
        "status": "pass"
        if finite_count == len(rankings)
        and in_range_count == len(rankings)
        and provenance_count == len(rankings)
        and repeatable_count == len(rankings)
        else "inconclusive",
        "per_candidate": uncertainty_rows,
    }


def _aggregate_counts(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """Sum integer fields across fixture cases."""
    result: dict[str, Any] = {key: sum(int(row[key]) for row in rows) for key in keys}
    result["case_count"] = len(rows)
    return result


def _aggregate_reliability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate reliability checks while retaining uncertainty summaries."""
    keys = (
        "candidate_count",
        "finite_risk_scores",
        "in_range_risk_scores",
        "complete_provenance_rows",
        "repeatable_risk_scores",
        "abstained_rows",
    )
    result = _aggregate_counts(rows, keys)
    halfwidths = [float(row["max_ci95_halfwidth"]) for row in rows]
    result["mean_case_max_ci95_halfwidth"] = float(np.mean(halfwidths)) if halfwidths else 0.0
    result["max_ci95_halfwidth"] = float(np.max(halfwidths)) if halfwidths else 0.0
    result["calibration_status"] = "not_evaluated; model score reliability checks only"
    result["status"] = (
        "pass"
        if result["finite_risk_scores"] == result["candidate_count"]
        and result["in_range_risk_scores"] == result["candidate_count"]
        and result["complete_provenance_rows"] == result["candidate_count"]
        and result["repeatable_risk_scores"] == result["candidate_count"]
        else "inconclusive"
    )
    return result


def build_report(config_path: Path) -> dict[str, Any]:
    """Build the complete matched diagnostic report from a committed config."""
    config = _load_yaml(config_path, name="config")
    if config.get("schema_version") != REPORT_SCHEMA_VERSION.replace(
        "comparison", "comparison_config"
    ):
        raise ValueError(
            "config schema_version must be risk_aware_trajectory_ranker_comparison_config.v1"
        )
    if config.get("evidence_status") != "smoke/diagnostic":
        raise ValueError("config evidence_status must be smoke/diagnostic")
    fixture_path = _resolve_repo_path(config.get("fixture_path"), field="fixture_path")
    cases = _load_fixture(fixture_path)
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
        raise ValueError("candidate_budget must be at least three")
    if len(cases) < 2:
        raise ValueError("matched comparison requires at least two held-out fixture cases")

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

    for case in cases:
        pedestrians = _pedestrians(case)
        case_payload: dict[str, Any] = {
            "case_id": case["case_id"],
            "split": case["split"],
            "input_digest": _case_digest(case),
            "arms": {},
        }
        for generator_name, (generator, generator_config) in generators.items():
            started = time.perf_counter_ns()
            candidates = generator(
                case["start_position"],
                case["local_goal"],
                horizon_steps=risk_config.horizon_steps,
                dt_s=risk_config.dt_s,
                config=generator_config,
            )
            generation_ms = (time.perf_counter_ns() - started) / 1.0e6
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
            reliability = _risk_reliability(rankings, repeated)
            selection = _selection_payload(rankings, candidates)
            per_generator_validity[generator_name].append(validity)
            per_generator_gates[generator_name].append(gates)
            per_generator_reliability[generator_name].append(reliability)
            timing_generation[generator_name]["total_ms"] += generation_ms
            timing_generation[generator_name]["total_candidates"] += len(candidates)
            timing_generation[generator_name]["per_case"].append(
                {
                    "case_id": case["case_id"],
                    "candidate_count": len(candidates),
                    "generation_ms": generation_ms,
                    "generation_ms_per_candidate": generation_ms / len(candidates),
                }
            )
            timing_ranking[generator_name]["total_ms"] += ranking_ms
            timing_ranking[generator_name]["total_candidates"] += len(candidates)
            timing_ranking[generator_name]["per_case"].append(
                {
                    "case_id": case["case_id"],
                    "ranking_and_gates_ms": ranking_ms,
                    "ranking_and_gates_ms_per_candidate": ranking_ms / len(candidates),
                    "per_candidate": per_candidate_timing,
                }
            )
            case_payload["arms"][generator_name] = {
                "candidate_validity": validity,
                "hard_gate_rejection": gates,
                "risk_score_reliability": reliability,
                "selection": selection,
                "ranking_action_ids": [record.action_id for record in rankings],
            }
        case_payloads.append(case_payload)

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
            }
        )

    config_schema_version = str(config["schema_version"])
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_status": "smoke/diagnostic",
        "caveats": [
            "RBF is a deterministic radial-basis-function proposal, not a trained policy.",
            "Risk scores are constant-velocity model scores; calibration is not evaluated.",
            "Timing is local offline wall time and is not an online deadline or speed claim.",
            "Hard-gate rejections are reported as exclusions from selection, not success evidence.",
            "Planner-loop wiring, online adaptation, nominal campaigns, safety, and real-world claims are deferred.",
        ],
        "uncertainty": {
            "confidence": "diagnostic-only; no planner-improvement conclusion",
            "risk_score_interpretation": "finite/range/provenance/repeatability checks only",
            "selection_difference_interpretation": "descriptive on a small held-out fixture split",
        },
        "provenance": {
            "command": " ".join(shlex.quote(argument) for argument in sys.argv),
            "config_path": _display_path(config_path),
            "config_sha256": _sha256_file(config_path),
            "config_schema_version": config_schema_version,
            "fixture_path": _display_path(fixture_path),
            "fixture_sha256": _sha256_file(fixture_path),
            "seed": risk_config.seed,
            "git_commit_sha": _git_head(),
            "git_status_short": _git_status(),
            "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        },
        "matched_comparison": {
            "baseline": "deterministic_primitive",
            "candidate_generators": ["deterministic_primitive", "rbf"],
            "split_policy": "all fixture rows are valid held_out rows",
            "case_count": len(cases),
            "candidate_budget": candidate_budget,
            "same_start_states_local_goals_actor_predictions": True,
            "same_risk_estimator_config": True,
            "same_ranking_weights": True,
            "same_hard_gate_configs": True,
            "hard_gates": ["verify_trajectory", "evaluate_actuator_feasibility"],
            "default_planner_behavior_changed": False,
            "planner_loop_wiring": "not_run; intentionally out of scope",
        },
        "fallback_degraded_exclusions": {
            "fallback_rows_excluded": 0,
            "degraded_rows_excluded": 0,
            "provenance_incomplete_rows_excluded": 0,
            "policy": "invalid or degraded inputs fail closed; no fallback row is evidence",
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
                case_id: case_payload["arms"]
                for case_id, case_payload in ((row["case_id"], row) for row in case_payloads)
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
            "cases_with_no_eligible_deterministic": sum(
                not row["deterministic_primitive"]["selected"] for row in selection_rows
            ),
            "cases_with_no_eligible_rbf": sum(not row["rbf"]["selected"] for row in selection_rows),
            "per_case": selection_rows,
        },
        "risk_score_reliability": {
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
            "generation": timing_generation,
            "ranking_and_hard_gates": timing_ranking,
            "per_candidate": {
                generator_name: [
                    {
                        "case_id": case["case_id"],
                        "candidates": case["arms"][generator_name]["candidate_validity"][
                            "action_ids"
                        ],
                        "ranking_and_gate_ms": next(
                            row["per_candidate"]
                            for row in timing_ranking[generator_name]["per_case"]
                            if row["case_id"] == case["case_id"]
                        ),
                    }
                    for case in case_payloads
                ]
                for generator_name in generators
            },
        },
        "cases": case_payloads,
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable report with claim boundaries first."""
    selection = report["selection_differences"]
    reliability = report["risk_score_reliability"]["by_generator"]
    gates = report["hard_gate_rejection"]["by_generator"]
    lines = [
        "# Risk-aware trajectory ranker comparison",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        f"Evidence status: `{report['evidence_status']}`.",
        "",
        "Caveats: fallback/degraded/provenance-incomplete rows are excluded; timing is local offline "
        "wall time; risk reliability checks are not calibration; planner-loop wiring is not run.",
        "",
        f"Uncertainty: {report['uncertainty']['confidence']}; selection differences are descriptive "
        "on the small held-out fixture split.",
        "",
        "## Matched comparison",
        "",
        f"- Baseline: `{report['matched_comparison']['baseline']}`",
        f"- Cases: `{report['matched_comparison']['case_count']}`; candidate budget: `{report['matched_comparison']['candidate_budget']}`",
        f"- Seed: `{report['provenance']['seed']}`; commit: `{report['provenance']['git_commit_sha']}`",
        f"- Fixture: `{report['provenance']['fixture_path']}`",
        "",
        "## Candidate validity",
        "",
    ]
    for name, row in report["candidate_validity"]["by_generator"].items():
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
            f"- Selected-slot changes: `{selection['cases_with_selected_slot_change']}`",
            f"- Selected-role changes: `{selection['cases_with_selected_role_change']}`",
            f"- Cases without an eligible deterministic selection: `{selection['cases_with_no_eligible_deterministic']}`",
            f"- Cases without an eligible RBF selection: `{selection['cases_with_no_eligible_rbf']}`",
            "",
            "## Risk-score reliability",
            "",
        ]
    )
    for name, row in reliability.items():
        lines.append(
            f"- `{name}`: status=`{row['status']}`, finite/in-range/provenance/repeatable="
            f"{row['finite_risk_scores']}/{row['in_range_risk_scores']}/"
            f"{row['complete_provenance_rows']}/{row['repeatable_risk_scores']}; "
            f"abstained={row['abstained_rows']}."
        )
    lines.extend(
        [
            "",
            "## Timing",
            "",
            "Timing is reported separately for generation, full ranking plus hard gates, and isolated per-candidate ranking.",
        ]
    )
    for name, row in report["timing"]["generation"].items():
        lines.append(
            f"- `{name}` generation: total={row['total_ms']:.3f} ms; "
            f"candidates={row['total_candidates']}."
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Command: `{report['provenance']['command']}`",
            f"- Config SHA-256: `{report['provenance']['config_sha256']}`",
            f"- Fixture SHA-256: `{report['provenance']['fixture_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the offline comparison CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="JSON report output path")
    parser.add_argument("--output-md", type=Path, help="Optional Markdown report output path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and write JSON plus optional Markdown output."""
    args = _parse_args(argv)
    try:
        report = build_report(args.config.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"risk-aware ranker comparison failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": "pass", "output": str(args.output), "schema_version": REPORT_SCHEMA_VERSION}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
