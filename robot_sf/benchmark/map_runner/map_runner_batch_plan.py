"""Batch-planning helpers for map-based benchmark runs."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.map_runner.map_runner_identity import _select_seeds
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    scenario_robot_kinematics_label,
)
from robot_sf.benchmark.runtime_seed_guard import check_simulation_seed

if TYPE_CHECKING:
    from pathlib import Path

MapBatchJob = tuple[dict[str, Any], int]


def resolve_batch_kinematics_tag(scenarios: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Return the batch-level kinematics tag and observed scenario tags."""
    scenario_kinematics = sorted({scenario_robot_kinematics_label(sc) for sc in scenarios})
    if not scenario_kinematics:
        return "unknown", scenario_kinematics
    if len(scenario_kinematics) == 1:
        return scenario_kinematics[0], scenario_kinematics
    return "mixed", scenario_kinematics


def map_scenario_identity(scenario: dict[str, Any]) -> str:
    """Return the scenario identifier emitted by the map runner."""
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def validate_scenario_names(scenarios: list[dict[str, Any]]) -> None:
    """Refuse ambiguous scenario names before fresh or resumed job dispatch."""
    seen: set[str] = set()
    for scenario in scenarios:
        name = map_scenario_identity(scenario)
        if name in seen:
            raise ValueError(f"campaign plan contains duplicate scenario name {name!r}")
        seen.add(name)


def _validate_behavior_sanity(scenario: dict[str, Any]) -> list[str]:
    """Check behavior metadata has the fields needed by pedestrian definitions.

    Returns:
        list[str]: Non-fatal behavior sanity errors.
    """
    errors: list[str] = []
    meta = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    behavior = str(meta.get("behavior") or "").strip().lower()
    single_peds = (
        scenario.get("single_pedestrians")
        if isinstance(scenario.get("single_pedestrians"), list)
        else []
    )

    if behavior in {"wait", "join", "leave", "follow", "lead", "accompany"}:
        if not single_peds:
            errors.append("behavior requires single_pedestrians entries")
        else:
            for ped in single_peds:
                if not isinstance(ped, dict):
                    continue
                role = str(ped.get("role") or "").strip().lower()
                if behavior == "wait" and not ped.get("wait_at"):
                    errors.append("wait behavior requires wait_at rules")
                    break
                if behavior in {"join", "leave", "follow", "lead", "accompany"} and not role:
                    errors.append("role behavior requires role field")
                    break

    return errors


def select_batch_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select executable map scenarios using the runner's admission filters.

    Returns:
        Supported scenarios whose pedestrian behavior metadata can execute.
    """
    validate_scenario_names(scenarios)
    filtered: list[dict[str, Any]] = []
    for scenario in scenarios:
        metadata = scenario.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if scenario.get("supported") is False or metadata.get("supported") is False:
            continue
        sanity_errors = _validate_behavior_sanity(scenario)
        if sanity_errors:
            logger.warning("Skipping scenario '{}': {}", scenario.get("name"), sanity_errors)
            continue
        filtered.append(scenario)
    return filtered


def _expand_seed_jobs(
    scenarios: list[dict[str, Any]],
    *,
    suite_seeds: dict[str, list[int]],
    suite_key: str,
) -> list[MapBatchJob]:
    """Return selected map jobs without admitting them for simulation execution."""
    jobs: list[MapBatchJob] = []
    seen: set[tuple[str, int]] = set()
    for scenario in select_batch_scenarios(scenarios):
        seeds = _select_seeds(scenario, suite_seeds=suite_seeds, suite_key=suite_key)
        for seed in seeds:
            try:
                if isinstance(seed, bool):
                    raise TypeError
                value = operator.index(seed)
            except TypeError as exc:
                raise ValueError("simulation seed must be an integer at map seed dispatch") from exc
            identity = (map_scenario_identity(scenario), int(value))
            if identity in seen:
                raise ValueError("campaign plan contains duplicate job identities")
            seen.add(identity)
            jobs.append((scenario, int(value)))
    return jobs


def build_job_identities(
    scenarios: list[dict[str, Any]],
    *,
    suite_seeds: dict[str, list[int]],
    suite_key: str,
) -> set[tuple[str, int]]:
    """Enumerate logical map jobs for metadata-only resume and integrity readers.

    This does not authorize execution; ``build_seed_jobs`` retains the simulator
    seed guard. Historical artifact inspection must not execute that guard.

    Returns:
        Unique scenario/seed pairs from the same expansion used for dispatch.
    """
    jobs = _expand_seed_jobs(scenarios, suite_seeds=suite_seeds, suite_key=suite_key)
    return {(map_scenario_identity(scenario), seed) for scenario, seed in jobs}


def build_seed_jobs(
    scenarios: list[dict[str, Any]],
    *,
    suite_seeds: dict[str, list[int]],
    suite_key: str,
) -> list[MapBatchJob]:
    """Expand selected scenarios into jobs and check simulation seed admission.

    Returns:
        Ordered ``(scenario, seed)`` jobs for worker dispatch.
    """
    jobs = _expand_seed_jobs(scenarios, suite_seeds=suite_seeds, suite_key=suite_key)
    for _, seed in jobs:
        check_simulation_seed(seed, boundary="map seed dispatch")
    return jobs


def build_worker_fixed_params(  # noqa: PLR0913
    *,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    algo: str,
    raw_policy_cfg: dict[str, Any],
    algo_config_path: str | None,
    scenario_path: Path,
    adapter_impact_eval: bool,
    experimental_ped_impact: bool,
    ped_impact_radius_m: float,
    ped_impact_window_steps: int,
    noise_spec: dict[str, Any],
    tracking_precision_spec: dict[str, Any],
    batch_observation_mode: str | None,
    observation_level: str | None,
    benchmark_track: str | None,
    track_schema_version: str | None,
    actuation_profile_metadata: dict[str, Any] | None,
    latency_profile_metadata: dict[str, Any] | None,
    latency_stress_metrics: dict[str, Any] | None,
    safety_wrapper: dict[str, Any] | None,
    record_planner_decision_trace: bool,
    record_simulation_step_trace: bool,
    cbf_safety_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the serialized parameter payload shared by all map workers.

    Returns:
        Fixed parameters copied into each serialized map-worker job.
    """
    return {
        "horizon": horizon,
        "dt": dt,
        "record_forces": record_forces,
        "snqi_weights": snqi_weights,
        "snqi_baseline": snqi_baseline,
        "algo": algo,
        "algo_config": raw_policy_cfg,
        "algo_config_path": algo_config_path,
        "scenario_path": str(scenario_path),
        "adapter_impact_eval": bool(adapter_impact_eval),
        "experimental_ped_impact": bool(experimental_ped_impact),
        "ped_impact_radius_m": float(ped_impact_radius_m),
        "ped_impact_window_steps": int(ped_impact_window_steps),
        "observation_noise": noise_spec,
        "tracking_precision": tracking_precision_spec,
        "observation_mode": batch_observation_mode,
        "observation_level": observation_level,
        "benchmark_track": benchmark_track,
        "track_schema_version": track_schema_version,
        "synthetic_actuation_profile": actuation_profile_metadata,
        "latency_stress_profile": latency_profile_metadata,
        "latency_stress_metrics": latency_stress_metrics,
        "safety_wrapper": safety_wrapper,
        "cbf_safety_filter": cbf_safety_filter,
        "record_planner_decision_trace": bool(record_planner_decision_trace),
        "record_simulation_step_trace": bool(record_simulation_step_trace),
    }
