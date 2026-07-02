"""Serialized worker helpers for map-based benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _param_or_default(params: dict[str, Any], key: str, default: Any) -> Any:
    """Return a parameter value, treating explicit ``None`` as missing.

    Returns:
        The configured value when not ``None``; otherwise ``default``.
    """
    value = params.get(key)
    return default if value is None else value


def _required_path_param(params: dict[str, Any], key: str) -> Path:
    """Return a required path parameter with an actionable error for missing values.

    Returns:
        Path built from the configured value.
    """
    value = params.get(key)
    if value is None:
        raise ValueError(f"{key} is required in map-runner job params")
    return Path(value)


def execute_map_job(
    job: tuple[dict[str, Any], int, dict[str, Any]],
    *,
    run_map_episode: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Execute one serialized map-runner job with the supplied episode runner.

    Returns:
        dict[str, Any]: Episode record returned by ``run_map_episode``.
    """
    scenario, seed, params = job
    algo = str(_param_or_default(params, "algo", "goal"))
    record_forces = bool(_param_or_default(params, "record_forces", True))
    ped_impact_radius_m = float(_param_or_default(params, "ped_impact_radius_m", 2.0))
    ped_impact_window_steps = int(_param_or_default(params, "ped_impact_window_steps", 5))
    return run_map_episode(
        scenario,
        seed,
        horizon=params.get("horizon"),
        dt=params.get("dt"),
        record_forces=record_forces,
        snqi_weights=params.get("snqi_weights"),
        snqi_baseline=params.get("snqi_baseline"),
        algo=algo,
        algo_config=params.get("algo_config"),
        algo_config_path=params.get("algo_config_path"),
        scenario_path=_required_path_param(params, "scenario_path"),
        adapter_impact_eval=bool(params.get("adapter_impact_eval", False)),
        experimental_ped_impact=bool(params.get("experimental_ped_impact", False)),
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
        observation_mode=params.get("observation_mode"),
        observation_level=params.get("observation_level"),
        benchmark_track=params.get("benchmark_track"),
        track_schema_version=params.get("track_schema_version"),
        observation_noise=params.get("observation_noise"),
        tracking_precision=params.get("tracking_precision"),
        synthetic_actuation_profile=params.get("synthetic_actuation_profile"),
        latency_stress_profile=params.get("latency_stress_profile"),
        safety_wrapper=params.get("safety_wrapper"),
        cbf_safety_filter=params.get("cbf_safety_filter"),
        record_simulation_step_trace=bool(params.get("record_simulation_step_trace", False)),
    )
