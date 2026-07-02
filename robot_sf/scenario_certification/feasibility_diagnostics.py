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
DIAGNOSTIC_CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_evidence"
DEFAULT_FAMILIES = ("bottleneck", "cross_trap", "head_on_corridor")
SOLVABLE_ROUTE_CLASSES = {VALID, KNIFE_EDGE, HARD_BUT_SOLVABLE}

EpisodeRunner = Callable[[Mapping[str, Any], int, int | None, str], Mapping[str, Any]]
Certifier = Callable[[Mapping[str, Any], Path], ScenarioCertificate]


@dataclass(frozen=True, slots=True)
class FeasibilityDiagnosticConfig:
    """Configuration for issue #3484 feasibility diagnostics."""

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


def make_actor_free_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return an in-memory scenario variant with dynamic pedestrian actors removed."""

    mutated = deepcopy(dict(scenario))
    sim_cfg = dict(mutated.get("simulation_config") or {})
    sim_cfg["ped_density"] = 0.0
    sim_cfg["single_pedestrians"] = []
    sim_cfg["pedestrian_flows"] = []
    sim_cfg["social_groups"] = []
    mutated["simulation_config"] = sim_cfg
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
    return ScenarioDiagnosticOutcome(
        scenario_id=scenario_id,
        family_id=family_id,
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
    }


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
        "seed": outcome.seed,
        "route_feasible": _lane_to_dict(outcome.route_clearance),
        "actor_free_solved": _lane_to_dict(outcome.actor_free),
        "oracle_solved": _lane_to_dict(outcome.oracle_trajectory),
        "extended_time_solved": _lane_to_dict(outcome.extended_time),
    }


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


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    for key in ("scenario_id", "name", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _first_seed(scenario: Mapping[str, Any], *, limit: int | None) -> int | None:
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
