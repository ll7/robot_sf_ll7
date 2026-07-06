"""Environment configuration helpers for benchmark map-runner episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.run_config_provenance import metric_affecting_run_config
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.gym_env.unified_config import RobotSimulationConfig


def build_env_config(
    scenario: dict[str, Any],
    *,
    scenario_path: Path,
) -> RobotSimulationConfig:
    """Build the benchmark environment config for one scenario.

    Returns:
        RobotSimulationConfig: Config with SocNav structured observations and grid enabled.
    """
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    config.observation_mode = ObservationMode.SOCNAV_STRUCT
    config.use_occupancy_grid = True
    config.include_grid_in_observation = True
    config.grid_config = GridConfig(
        # Benchmark default upgraded to higher-resolution occupancy grids.
        resolution=0.2,
        width=32.0,
        height=32.0,
        channels=[
            GridChannel.OBSTACLES,
            GridChannel.PEDESTRIANS,
            GridChannel.COMBINED,
        ],
        use_ego_frame=True,
        center_on_robot=True,
    )
    apply_pedestrian_reactivity_to_env_config(config, scenario=scenario)
    if "pedestrian_control_trace_labels" in scenario:
        config.sim_config.pedestrian_control_trace_labels = scenario[
            "pedestrian_control_trace_labels"
        ]
    return config


def representative_metric_affecting_config(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    scenario_path: Path,
) -> dict[str, Any]:
    """Return the metric-affecting run-config provenance block for a batch (issue #3701).

    Builds a representative environment config from the first scenario and
    extracts the metric-affecting toggles (LiDAR ``scan_noise`` and the
    collision-handling regime) so the benchmark run manifest is self-describing.
    The benchmark uses a uniform sensor configuration across a batch, so the
    first scenario is representative of the run.

    This helper is *fail-soft*: it returns a ``status: not_available`` block
    instead of raising, because this is descriptive provenance metadata and must
    never break an otherwise valid benchmark run. A run that omits the block on
    error is still honest; a run that crashes while annotating metadata is not.

    Returns:
        A JSON-safe metric-affecting run-config block, or a
        ``{"status": "not_available", "reason": ...}`` block when no scenario is
        available or config construction fails.
    """
    if not scenarios:
        return {"status": "not_available", "reason": "no scenarios"}
    try:
        config = build_env_config(dict(scenarios[0]), scenario_path=scenario_path)
        return metric_affecting_run_config(config)
    except Exception as exc:  # noqa: BLE001 - provenance must never break a benchmark run.
        logger.debug("metric-affecting run-config provenance unavailable: {}", exc)
        return {"status": "not_available", "reason": str(exc)}


def apply_pedestrian_reactivity_to_env_config(
    config: RobotSimulationConfig,
    *,
    scenario: Mapping[str, Any],
) -> None:
    """Apply the scenario's pedestrian-reactivity toggle to the env config.

    The optional scenario key ``peds_have_robot_repulsion`` selects the pedestrian-reactivity
    condition for the #3573 reactive-vs-replay ablation: ``True`` (default) keeps the robot->
    pedestrian social force (reactive social-force pedestrians), ``False`` disables it
    (open-loop / non-reactive replay — the robot-response term is off, so pedestrians do not
    yield to the robot). Absent the key, current behavior (reactive) is preserved.

    Sets ``sim_config.prf_config.is_active`` directly because the deprecated
    ``peds_have_robot_repulsion`` alias is only synced in ``__post_init__`` (already run).
    """
    reactive = scenario.get("peds_have_robot_repulsion")
    if reactive is None:
        return
    is_active = bool(reactive)
    config.sim_config.prf_config.is_active = is_active
    config.peds_have_robot_repulsion = is_active


def apply_active_observation_mode_to_env_config(
    config: RobotSimulationConfig,
    *,
    active_observation_mode: str,
) -> None:
    """Apply planner observation-mode requirements to the runtime environment config."""
    if active_observation_mode != "sensor_fusion_state":
        return
    config.observation_mode = ObservationMode.DEFAULT_GYM
    config.use_occupancy_grid = False
    config.include_grid_in_observation = False
    config.grid_config = None


_POLICY_ENV_OBSERVATION_OVERRIDE_KEYS = frozenset(
    {
        "predictive_foresight_enabled",
        "predictive_foresight_model_id",
        "predictive_foresight_checkpoint_path",
        "predictive_foresight_device",
        "predictive_foresight_max_agents",
        "predictive_foresight_horizon_steps",
        "predictive_foresight_rollout_dt",
        "predictive_foresight_ego_conditioning",
        "predictive_foresight_near_distance",
        "predictive_foresight_front_corridor_length",
        "predictive_foresight_front_corridor_half_width",
    }
)


def apply_policy_env_observation_overrides(
    config: RobotSimulationConfig,
    policy_cfg: Mapping[str, Any],
) -> None:
    """Apply candidate observation-contract env overrides before env construction."""

    raw_overrides = policy_cfg.get("env_overrides")
    overrides = raw_overrides if isinstance(raw_overrides, Mapping) else {}
    for key in _POLICY_ENV_OBSERVATION_OVERRIDE_KEYS:
        if key in overrides:
            setattr(config, key, overrides[key])


def validate_sensor_fusion_adapter_config(
    *,
    algo: str,
    active_observation_mode: str,
    algo_config: dict[str, Any],
) -> None:
    """Fail closed when a planner requests sensor-fusion input without an adapter."""
    if active_observation_mode != "sensor_fusion_state":
        return
    algo_key = str(algo).strip().lower()
    if algo_key == "safety_barrier" and not algo_config.get("lidar_occupancy_adapter"):
        raise ValueError(
            "safety_barrier with sensor_fusion_state/lidar_2d requires "
            "algo_config['lidar_occupancy_adapter']."
        )
