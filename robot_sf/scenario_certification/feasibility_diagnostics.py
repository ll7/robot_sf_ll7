"""Executable feasibility diagnostics for issue #3484 scenario families.

The diagnostics in this module are CPU-local probes, not benchmark evidence.  They
run route-clearance certification plus small rollout probes and then hand the
observed lane outcomes to the existing failure-cause classifier.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner import _run_map_episode
from robot_sf.scenario_certification.failure_cause import (
    DiagnosticLaneEvidence,
    FamilyDiagnostics,
    classify_failure_cause,
    diagnostic_lane_evidence_to_dict,
)
from robot_sf.scenario_certification.v1 import (
    GEOMETRICALLY_INFEASIBLE,
    HARD_BUT_SOLVABLE,
    KNIFE_EDGE,
    VALID,
    ScenarioCertificate,
    certificate_to_dict,
    certify_scenario,
)
from robot_sf.training.scenario_loader import load_scenarios

FEASIBILITY_DIAGNOSTICS_SCHEMA = "scenario_feasibility_diagnostics.v1"
FEASIBILITY_CLAIM_BOUNDARY_SYNTHESIS_SCHEMA = "scenario_feasibility_claim_boundary_synthesis.v1"
DIAGNOSTIC_CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_evidence"
DEFAULT_FAMILIES = ("bottleneck", "cross_trap", "head_on_corridor")
SOLVABLE_ROUTE_CLASSES = {VALID, KNIFE_EDGE, HARD_BUT_SOLVABLE}

EpisodeRunner = Callable[[Mapping[str, Any], int, int | None, str], Mapping[str, Any]]
Certifier = Callable[[Mapping[str, Any], Path], ScenarioCertificate]


@dataclass(frozen=True, slots=True)
class FeasibilityDiagnosticConfig:
    """Configuration for issue #3484 feasibility diagnostics.

    ``seeds_per_scenario`` is a single-seed selector for this diagnostic slice. Values above one
    are rejected instead of implying unsupported multi-seed rollout semantics.
    """

    scenario_path: Path
    families: tuple[str, ...] = DEFAULT_FAMILIES
    seeds_per_scenario: int | None = 1
    baseline_algo: str = "goal"
    oracle_algo: str = "goal"
    extended_time_multiplier: float = 2.0
    run_actor_free: bool = True
    run_oracle: bool = True
    run_extended_time: bool = False
    record_simulation_step_trace: bool = False


@dataclass(frozen=True, slots=True)
class LaneResult:
    """Observed result for one diagnostic lane."""

    passed: bool | None
    status: str
    blocker: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioDiagnosticOutcome:
    """Diagnostic outputs for one scenario row."""

    scenario_id: str
    family_id: str
    difficulty_level: str | None
    seed: int | None
    route_clearance: LaneResult
    actor_free: LaneResult
    oracle_trajectory: LaneResult
    extended_time: LaneResult


def run_feasibility_diagnostics(
    config: FeasibilityDiagnosticConfig,
    *,
    episode_runner: EpisodeRunner | None = None,
    certifier: Certifier | None = None,
) -> dict[str, Any]:
    """Run route-clearance and rollout diagnostics.

    Returns:
        Versioned diagnostic-only report with per-scenario rows and per-family verdicts.
    """

    selected_families = {family.casefold() for family in config.families}
    scenarios = [
        dict(scenario)
        for scenario in load_scenarios(config.scenario_path)
        if _scenario_family_id(scenario).casefold() in selected_families
    ]
    runner = episode_runner or _default_episode_runner(config)
    certify = certifier or _default_certifier
    outcomes = [
        _run_scenario_diagnostics(
            scenario,
            config=config,
            episode_runner=runner,
            certifier=certify,
        )
        for scenario in scenarios
    ]
    return _build_report(config=config, scenario_count=len(scenarios), outcomes=outcomes)


def build_difficulty_ramp_claim_boundary_synthesis(
    report: Mapping[str, Any],
    *,
    source_report: str,
    target_families: Sequence[str] = DEFAULT_FAMILIES,
) -> dict[str, Any]:
    """Synthesize retained diagnostics into a fail-closed claim boundary.

    The synthesis consumes an existing ``scenario_feasibility_diagnostics.v1`` report;
    it does not rerun scenarios or promote diagnostic proxy evidence into ranking evidence.

    Returns:
        Versioned summary of vehicle-infeasible, dynamic-blocked, rank-comparable, and
        still-unsupported family states.
    """

    unsupported_all_reason = _unsupported_report_reason(report)
    target = tuple(target_families)
    if unsupported_all_reason is not None:
        family_rows = [
            _unsupported_synthesis_row(family_id, unsupported_all_reason) for family_id in target
        ]
    else:
        verdicts = _family_verdicts_by_id(report)
        ramps = _difficulty_ramps_by_id(report)
        family_rows = [
            _synthesize_family_claim_boundary(
                family_id,
                verdict=verdicts.get(family_id),
                difficulty_ramp=ramps.get(family_id),
            )
            for family_id in target
        ]

    return {
        "schema_version": FEASIBILITY_CLAIM_BOUNDARY_SYNTHESIS_SCHEMA,
        "issue": "3484",
        "claim_boundary": "diagnostic_only_not_ranking_evidence",
        "source_report": source_report,
        "source_report_schema": report.get("schema_version"),
        "source_report_claim_boundary": report.get("claim_boundary"),
        "target_families": list(target),
        "route_infeasible_families": _families_in_category(family_rows, "route_infeasible"),
        "vehicle_infeasible_families": _families_in_category(family_rows, "vehicle_infeasible"),
        "dynamic_blocked_families": _families_in_category(family_rows, "dynamic_blocked"),
        "comparable_for_ranking_families": [
            row["family_id"] for row in family_rows if row["comparable_for_ranking"] is True
        ],
        "not_comparable_for_ranking_families": [
            row["family_id"] for row in family_rows if row["comparable_for_ranking"] is False
        ],
        "still_unsupported_families": _families_in_category(family_rows, "still_unsupported"),
        "family_claim_boundaries": family_rows,
    }


def make_actor_free_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return an in-memory scenario variant with dynamic pedestrian actors removed."""

    mutated = deepcopy(dict(scenario))
    sim_cfg = dict(mutated.get("simulation_config") or {})
    sim_cfg["ped_density"] = 0.0
    sim_cfg.pop("single_pedestrians", None)
    sim_cfg.pop("pedestrian_flows", None)
    sim_cfg.pop("social_groups", None)
    mutated["simulation_config"] = sim_cfg
    mutated["single_pedestrians"] = []
    mutated["social_groups"] = []
    metadata = dict(mutated.get("metadata") or {})
    metadata["diagnostic_variant"] = "actor_free"
    metadata["diagnostic_claim_boundary"] = DIAGNOSTIC_CLAIM_BOUNDARY
    mutated["metadata"] = metadata
    mutated["seeds"] = _first_seeds(scenario, 1)
    return mutated


def make_extended_time_scenario(
    scenario: Mapping[str, Any],
    *,
    multiplier: float,
) -> tuple[dict[str, Any], int | None]:
    """Return an in-memory scenario variant with a multiplied episode horizon."""

    mutated = deepcopy(dict(scenario))
    original_horizon = _scenario_horizon(mutated)
    extended_horizon = None
    if original_horizon is not None:
        extended_horizon = max(original_horizon + 1, round(original_horizon * multiplier))
        sim_cfg = dict(mutated.get("simulation_config") or {})
        sim_cfg["max_episode_steps"] = extended_horizon
        mutated["simulation_config"] = sim_cfg
    metadata = dict(mutated.get("metadata") or {})
    metadata["diagnostic_variant"] = "extended_time"
    metadata["extended_time_multiplier"] = multiplier
    metadata["diagnostic_claim_boundary"] = DIAGNOSTIC_CLAIM_BOUNDARY
    mutated["metadata"] = metadata
    return mutated, extended_horizon


def route_clearance_lane(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    certifier: Certifier | None = None,
) -> LaneResult:
    """Run static route-clearance certification and map it to a lane result.

    Returns:
        Lane result containing the pass/fail/blocked route-clearance observation.
    """

    certify = certifier or _default_certifier
    try:
        certificate = certify(scenario, scenario_path)
    except Exception as exc:  # noqa: BLE001 - diagnostics must fail closed.
        return LaneResult(
            passed=None,
            status="blocked",
            blocker=f"route_clearance_certifier_error: {exc}",
        )
    certificate_payload = certificate_to_dict(certificate)
    classification = str(certificate.classification)
    eligibility = str(certificate.benchmark_eligibility)
    if classification == GEOMETRICALLY_INFEASIBLE or eligibility == "excluded":
        passed = False
    elif classification in SOLVABLE_ROUTE_CLASSES:
        passed = True
    else:
        passed = None
    status = _status_from_passed(passed)
    blocker = None if passed is not None else f"unsupported_route_certificate: {classification}"
    return LaneResult(
        passed=passed,
        status=status,
        blocker=blocker,
        evidence={
            "certificate_schema_version": certificate_payload.get("schema_version"),
            "classification": classification,
            "benchmark_eligibility": eligibility,
            "reasons": certificate_payload.get("reasons", []),
            "certificate": certificate_payload,
        },
    )


def rollout_lane(
    scenario: Mapping[str, Any],
    *,
    seed: int,
    horizon: int | None,
    algo: str,
    episode_runner: EpisodeRunner,
) -> LaneResult:
    """Run one diagnostic rollout and map the episode record to a lane result.

    Returns:
        Lane result containing the rollout outcome and compact episode evidence.
    """

    try:
        record = dict(episode_runner(scenario, seed, horizon, algo))
    except Exception as exc:  # noqa: BLE001 - diagnostics must fail closed.
        return LaneResult(
            passed=None,
            status="blocked",
            blocker=f"rollout_error: {exc}",
            evidence={"algo": algo, "seed": seed, "horizon": horizon},
        )
    passed, reason = _episode_success(record)
    return LaneResult(
        passed=passed,
        status=_status_from_passed(passed),
        blocker=None if passed is not None else reason,
        evidence={
            "algo": algo,
            "seed": seed,
            "horizon": horizon,
            "termination_reason": record.get("termination_reason"),
            "success_reason": reason,
            "record_excerpt": _record_excerpt(record),
        },
    )


def _run_scenario_diagnostics(
    scenario: Mapping[str, Any],
    *,
    config: FeasibilityDiagnosticConfig,
    episode_runner: EpisodeRunner,
    certifier: Certifier,
) -> ScenarioDiagnosticOutcome:
    scenario_id = _scenario_id(scenario)
    family_id = _scenario_family_id(scenario)
    seed = _first_seed(scenario, limit=config.seeds_per_scenario)
    route = route_clearance_lane(
        scenario,
        scenario_path=config.scenario_path,
        certifier=certifier,
    )
    actor_free = LaneResult(None, "not_run", "disabled_by_config")
    if config.run_actor_free and seed is not None:
        actor_free_scenario = make_actor_free_scenario(scenario)
        actor_free = rollout_lane(
            actor_free_scenario,
            seed=seed,
            horizon=_scenario_horizon(actor_free_scenario),
            algo=config.baseline_algo,
            episode_runner=episode_runner,
        )
    elif seed is None:
        actor_free = LaneResult(None, "blocked", "scenario_has_no_seed")

    oracle = LaneResult(None, "not_run", "disabled_by_config")
    if config.run_oracle and seed is not None:
        oracle = rollout_lane(
            scenario,
            seed=seed,
            horizon=_scenario_horizon(scenario),
            algo=config.oracle_algo,
            episode_runner=episode_runner,
        )
    elif seed is None:
        oracle = LaneResult(None, "blocked", "scenario_has_no_seed")

    extended = LaneResult(None, "not_run", "not_in_this_slice")
    if config.run_extended_time and seed is not None:
        extended_scenario, extended_horizon = make_extended_time_scenario(
            scenario,
            multiplier=config.extended_time_multiplier,
        )
        extended = rollout_lane(
            extended_scenario,
            seed=seed,
            horizon=extended_horizon,
            algo=config.oracle_algo,
            episode_runner=episode_runner,
        )
    elif config.run_extended_time and seed is None:
        extended = LaneResult(None, "blocked", "scenario_has_no_seed")
    return ScenarioDiagnosticOutcome(
        scenario_id=scenario_id,
        family_id=family_id,
        difficulty_level=_scenario_difficulty_level(scenario),
        seed=seed,
        route_clearance=route,
        actor_free=actor_free,
        oracle_trajectory=oracle,
        extended_time=extended,
    )


def _build_report(
    *,
    config: FeasibilityDiagnosticConfig,
    scenario_count: int,
    outcomes: Sequence[ScenarioDiagnosticOutcome],
) -> dict[str, Any]:
    lane_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        lane_rows.extend(_scenario_lane_rows(outcome, source=config.scenario_path.as_posix()))
    family_verdicts = [
        _family_verdict(family_id, [row for row in outcomes if row.family_id == family_id])
        for family_id in sorted({row.family_id for row in outcomes})
    ]
    return {
        "schema_version": FEASIBILITY_DIAGNOSTICS_SCHEMA,
        "issue": "3484",
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "source": config.scenario_path.as_posix(),
        "families": list(config.families),
        "scenario_count": scenario_count,
        "scenario_rows": [_scenario_row_to_dict(row) for row in outcomes],
        "diagnostic_lane_rows": lane_rows,
        "family_verdicts": family_verdicts,
        "difficulty_ramp": _difficulty_ramp_summary(outcomes),
    }


def _unsupported_report_reason(report: Mapping[str, Any]) -> str | None:
    if report.get("schema_version") != FEASIBILITY_DIAGNOSTICS_SCHEMA:
        return "unsupported_source_report_schema"
    if report.get("claim_boundary") != DIAGNOSTIC_CLAIM_BOUNDARY:
        return "unsupported_source_claim_boundary"
    return None


def _family_verdicts_by_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("family_verdicts") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    return {
        str(row.get("family_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("family_id") is not None
    }


def _difficulty_ramps_by_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("difficulty_ramp") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    return {
        str(row.get("family_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("family_id") is not None
    }


def _synthesize_family_claim_boundary(
    family_id: str,
    *,
    verdict: Mapping[str, Any] | None,
    difficulty_ramp: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if verdict is None:
        return _unsupported_synthesis_row(family_id, "missing_family_verdict")
    if difficulty_ramp is None:
        return _unsupported_synthesis_row(family_id, "missing_difficulty_ramp")

    failure_verdict = verdict.get("failure_cause_verdict")
    if not isinstance(failure_verdict, Mapping):
        return _unsupported_synthesis_row(family_id, "missing_failure_cause_verdict")

    if failure_verdict.get("evidence_complete") is not True:
        return _unsupported_synthesis_row(family_id, "incomplete_diagnostic_evidence")

    ramp_levels = difficulty_ramp.get("levels")
    if not isinstance(ramp_levels, Sequence) or isinstance(ramp_levels, (str, bytes)):
        return _unsupported_synthesis_row(family_id, "missing_difficulty_ramp_levels")
    if len(ramp_levels) == 0:
        return _unsupported_synthesis_row(family_id, "empty_difficulty_ramp_levels")

    cause = str(failure_verdict.get("cause") or "")
    category = {
        "infeasible_route": "route_infeasible",
        "vehicle_infeasible": "vehicle_infeasible",
        "dynamic_blocking_or_deadlock": "dynamic_blocked",
        "planner_limited": "planner_limited",
        "time_limited": "time_limited",
    }.get(cause, "still_unsupported")
    if category == "still_unsupported":
        return _unsupported_synthesis_row(family_id, f"unsupported_verdict:{cause or 'missing'}")

    inputs = failure_verdict.get("inputs")
    if not isinstance(inputs, Mapping):
        return _unsupported_synthesis_row(family_id, "missing_failure_cause_inputs")

    comparable = failure_verdict.get("comparable_for_ranking") is True
    return {
        "family_id": family_id,
        "claim_state": category,
        "failure_cause": cause,
        "comparable_for_ranking": comparable,
        "evidence_complete": True,
        "route_feasible": inputs.get("route_feasible"),
        "actor_free_solved": inputs.get("actor_free_solved"),
        "extended_time_solved": inputs.get("extended_time_solved"),
        "oracle_solved": inputs.get("oracle_solved"),
        "difficulty_ramp": {
            "axis": difficulty_ramp.get("axis"),
            "level_count": len(ramp_levels),
            "first_actor_free_failure_level": difficulty_ramp.get("first_actor_free_failure_level"),
            "first_oracle_failure_level": difficulty_ramp.get("first_oracle_failure_level"),
        },
        "unsupported_reason": None,
    }


def _unsupported_synthesis_row(family_id: str, reason: str) -> dict[str, Any]:
    return {
        "family_id": family_id,
        "claim_state": "still_unsupported",
        "failure_cause": None,
        "comparable_for_ranking": False,
        "evidence_complete": False,
        "route_feasible": None,
        "actor_free_solved": None,
        "extended_time_solved": None,
        "oracle_solved": None,
        "difficulty_ramp": None,
        "unsupported_reason": reason,
    }


def _families_in_category(rows: Sequence[Mapping[str, Any]], category: str) -> list[str]:
    return [str(row["family_id"]) for row in rows if row.get("claim_state") == category]


def _family_verdict(
    family_id: str,
    outcomes: Sequence[ScenarioDiagnosticOutcome],
) -> dict[str, Any]:
    diagnostics = FamilyDiagnostics(
        route_feasible=_aggregate_lane([row.route_clearance.passed for row in outcomes]),
        actor_free_solved=_aggregate_lane([row.actor_free.passed for row in outcomes]),
        extended_time_solved=_aggregate_lane([row.extended_time.passed for row in outcomes]),
        oracle_solved=_aggregate_lane([row.oracle_trajectory.passed for row in outcomes]),
    )
    verdict = classify_failure_cause(diagnostics)
    verdict["diagnostic_only_family_aggregation"] = {
        "rule": "false_if_any_false_true_only_if_all_observed_true_null_if_unobserved",
        "scenario_count": len(outcomes),
    }
    return {
        "family_id": family_id,
        "route_feasible": diagnostics.route_feasible,
        "actor_free_solved": diagnostics.actor_free_solved,
        "extended_time_solved": diagnostics.extended_time_solved,
        "oracle_solved": diagnostics.oracle_solved,
        "failure_cause_verdict": verdict,
    }


def _scenario_lane_rows(outcome: ScenarioDiagnosticOutcome, *, source: str) -> list[dict[str, Any]]:
    rows = []
    for lane, result in (
        ("route_clearance", outcome.route_clearance),
        ("actor_free_run", outcome.actor_free),
        ("oracle_or_scripted_controller", outcome.oracle_trajectory),
        ("extended_time_run", outcome.extended_time),
    ):
        rows.append(
            diagnostic_lane_evidence_to_dict(
                DiagnosticLaneEvidence(
                    family_id=outcome.family_id,
                    lane=lane,
                    passed=result.passed,
                    source=source,
                    scenario_id=outcome.scenario_id,
                    evidence_ref=f"scenario_rows/{outcome.scenario_id}/{lane}",
                    notes=result.blocker or result.status,
                )
            )
        )
    return rows


def _scenario_row_to_dict(outcome: ScenarioDiagnosticOutcome) -> dict[str, Any]:
    return {
        "scenario_id": outcome.scenario_id,
        "family_id": outcome.family_id,
        "difficulty_level": outcome.difficulty_level,
        "seed": outcome.seed,
        "route_feasible": _lane_to_dict(outcome.route_clearance),
        "actor_free_solved": _lane_to_dict(outcome.actor_free),
        "oracle_solved": _lane_to_dict(outcome.oracle_trajectory),
        "extended_time_solved": _lane_to_dict(outcome.extended_time),
    }


def _difficulty_ramp_summary(
    outcomes: Sequence[ScenarioDiagnosticOutcome],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family_id in sorted({row.family_id for row in outcomes}):
        family_rows = sorted(
            [row for row in outcomes if row.family_id == family_id],
            key=lambda row: (
                _difficulty_sort_key(row.difficulty_level),
                row.scenario_id,
            ),
        )
        summaries.append(
            {
                "family_id": family_id,
                "axis": "scenario_variant_difficulty",
                "source": "scenario_metadata_or_name",
                "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
                "levels": [
                    {
                        "scenario_id": row.scenario_id,
                        "difficulty_level": row.difficulty_level,
                        "route_feasible": row.route_clearance.passed,
                        "actor_free_solved": row.actor_free.passed,
                        "oracle_solved": row.oracle_trajectory.passed,
                        "extended_time_solved": row.extended_time.passed,
                    }
                    for row in family_rows
                ],
                "first_actor_free_failure_level": _first_failure_level(
                    family_rows, lane="actor_free"
                ),
                "first_oracle_failure_level": _first_failure_level(family_rows, lane="oracle"),
            }
        )
    return summaries


def _first_failure_level(
    rows: Sequence[ScenarioDiagnosticOutcome],
    *,
    lane: str,
) -> str | None:
    for row in rows:
        result = row.actor_free if lane == "actor_free" else row.oracle_trajectory
        if result.passed is False:
            return row.difficulty_level or row.scenario_id
    return None


def _lane_to_dict(result: LaneResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "status": result.status,
        "blocker": result.blocker,
        "evidence": dict(result.evidence),
    }


def _aggregate_lane(values: Iterable[bool | None]) -> bool | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    if any(value is False for value in observed):
        return False
    return True


def _default_certifier(scenario: Mapping[str, Any], scenario_path: Path) -> ScenarioCertificate:
    """Adapter for the existing scenario certificate API.

    Returns:
        Scenario certificate emitted by the canonical certification module.
    """

    return certify_scenario(scenario, scenario_path=scenario_path)


def _default_episode_runner(config: FeasibilityDiagnosticConfig) -> EpisodeRunner:
    """Build a single-episode runner around the canonical map episode runner.

    Returns:
        Callable that executes one scenario/seed/algo diagnostic rollout.
    """

    def _run(
        scenario: Mapping[str, Any],
        seed: int,
        horizon: int | None,
        algo: str,
    ) -> Mapping[str, Any]:
        scenario_payload = deepcopy(dict(scenario))
        scenario_payload["seeds"] = [int(seed)]
        return _run_map_episode(
            scenario_payload,
            int(seed),
            horizon=horizon,
            dt=None,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo=algo,
            scenario_path=config.scenario_path,
            record_simulation_step_trace=config.record_simulation_step_trace,
        )

    return _run


def _episode_success(record: Mapping[str, Any]) -> tuple[bool | None, str]:
    success_fields = (
        record.get("success"),
        record.get("route_complete"),
        record.get("goal_reached"),
    )
    if any(value is True for value in success_fields):
        return True, "success_field_true"
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        if metrics.get("success") is True or metrics.get("route_complete") is True:
            return True, "metrics_success_true"
        if metrics.get("goal_reached") is True:
            return True, "metrics_goal_reached_true"
        if (
            metrics.get("success") is False
            or metrics.get("route_complete") is False
            or metrics.get("goal_reached") is False
        ):
            return False, "metrics_success_false"
    termination = str(record.get("termination_reason") or record.get("status") or "").lower()
    if termination in {"success", "goal_reached", "route_complete", "completed"}:
        return True, f"termination_reason={termination}"
    if termination in {"timeout", "collision", "failure", "failed", "error", "truncated"}:
        return False, f"termination_reason={termination}"
    if any(value is False for value in success_fields):
        return False, "success_field_false"
    return None, "unknown_episode_outcome"


def _record_excerpt(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "scenario_id",
        "seed",
        "termination_reason",
        "success",
        "route_complete",
        "goal_reached",
        "collisions",
        "collision",
        "time_to_goal",
    )
    return {key: record.get(key) for key in keys if key in record}


def _status_from_passed(passed: bool | None) -> str:
    if passed is True:
        return "passed"
    if passed is False:
        return "failed"
    return "not_run"


def _scenario_family_id(scenario: Mapping[str, Any]) -> str:
    metadata = scenario.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    for value in (
        metadata_map.get("scenario_family"),
        metadata_map.get("family"),
        metadata_map.get("archetype"),
        scenario.get("scenario_family"),
        scenario.get("family"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _scenario_difficulty_level(scenario: Mapping[str, Any]) -> str | None:
    metadata = scenario.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    for key in ("difficulty", "difficulty_level", "density_tier"):
        value = metadata_map.get(key)
        if value is not None:
            return str(value)
    scenario_id = _scenario_id(scenario)
    suffix = scenario_id.rsplit("_", 1)[-1]
    if suffix in {"low", "medium", "high"}:
        return suffix
    return None


def _difficulty_sort_key(level: str | None) -> tuple[int, str]:
    order = {"low": 0, "medium": 1, "high": 2}
    if level is None:
        return (99, "")
    return (order.get(level, 50), level)


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    for key in ("scenario_id", "name", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _first_seed(scenario: Mapping[str, Any], *, limit: int | None) -> int | None:
    if limit is not None and limit > 1:
        raise ValueError("seeds_per_scenario currently supports only 0, 1, or None")
    seeds = _first_seeds(scenario, 1 if limit is None else min(limit, 1))
    return seeds[0] if seeds else None


def _first_seeds(scenario: Mapping[str, Any], count: int | None) -> list[int]:
    raw = scenario.get("seeds")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        seeds = [int(seed) for seed in raw]
    else:
        seed = scenario.get("seed")
        seeds = [int(seed)] if seed is not None else []
    if count is None:
        return seeds
    return seeds[: max(count, 0)]


def _scenario_horizon(scenario: Mapping[str, Any]) -> int | None:
    sim_cfg = scenario.get("simulation_config")
    if not isinstance(sim_cfg, Mapping):
        return None
    value = sim_cfg.get("max_episode_steps")
    if value is None:
        return None
    return int(value)


__all__ = [
    "DEFAULT_FAMILIES",
    "DIAGNOSTIC_CLAIM_BOUNDARY",
    "FEASIBILITY_DIAGNOSTICS_SCHEMA",
    "FeasibilityDiagnosticConfig",
    "LaneResult",
    "ScenarioDiagnosticOutcome",
    "make_actor_free_scenario",
    "make_extended_time_scenario",
    "rollout_lane",
    "route_clearance_lane",
    "run_feasibility_diagnostics",
]
