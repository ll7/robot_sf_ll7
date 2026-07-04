"""Batch-planning helpers for map-based benchmark runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_actions import (
    scenario_robot_kinematics_label,
)
from robot_sf.benchmark.map_runner_identity import _select_seeds

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


def build_seed_jobs(
    scenarios: list[dict[str, Any]],
    *,
    suite_seeds: dict[str, list[int]],
    suite_key: str,
) -> list[MapBatchJob]:
    """Expand scenarios and configured seeds into map-runner jobs.

    Returns:
        Ordered ``(scenario, seed)`` jobs for worker dispatch.
    """
    jobs: list[MapBatchJob] = []
    for scenario in scenarios:
        seeds = _select_seeds(scenario, suite_seeds=suite_seeds, suite_key=suite_key)
        for seed in seeds:
            jobs.append((scenario, int(seed)))
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
