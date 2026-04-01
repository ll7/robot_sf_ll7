"""Shared helpers for applying training-config environment overrides.

Trajectory collection and imitation workflows need to recreate the same runtime
observation contract as their source PPO configuration. These helpers keep the
override application logic in one reusable place outside the main training CLI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings


def _coerce_grid_channels(values: Sequence[object]) -> list[GridChannel]:
    """Convert channel identifiers to GridChannel enums."""
    channels: list[GridChannel] = []
    for value in values:
        if isinstance(value, GridChannel):
            channels.append(value)
        else:
            channels.append(GridChannel(str(value)))
    return channels


def _apply_simple_overrides(env_config: Any, overrides: Mapping[str, object]) -> None:
    """Apply top-level environment overrides."""
    observation_mode = overrides.get("observation_mode")
    if observation_mode is not None:
        env_config.observation_mode = ObservationMode(str(observation_mode))

    for key in (
        "use_occupancy_grid",
        "include_grid_in_observation",
        "show_occupancy_grid",
        "use_planner",
        "planner_backend",
        "planner_clearance_margin",
        "peds_have_obstacle_forces",
        "peds_have_static_obstacle_forces",
        "peds_have_robot_repulsion",
        "map_id",
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
    ):
        if key in overrides:
            setattr(env_config, key, overrides[key])


def _resolve_robot_config_type(
    value: object,
) -> type[BicycleDriveSettings] | type[DifferentialDriveSettings] | None:
    """Resolve a robot config class from a string override."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"bicycle", "bicycle_drive", "bicycle_drive_settings"}:
            return BicycleDriveSettings
        if normalized in {
            "differential",
            "diff",
            "differential_drive",
            "differential_drive_settings",
        }:
            return DifferentialDriveSettings
    return None


def _apply_robot_overrides(env_config: Any, overrides: Mapping[str, object]) -> None:
    """Apply robot drivetrain overrides."""
    robot_overrides = overrides.get("robot_config")
    if robot_overrides is None:
        return
    if isinstance(robot_overrides, (BicycleDriveSettings, DifferentialDriveSettings)):
        env_config.robot_config = robot_overrides
        return
    if isinstance(robot_overrides, str):
        config_cls = _resolve_robot_config_type(robot_overrides)
        if config_cls is None:
            raise ValueError(f"robot_config override type '{robot_overrides}' is not supported")
        env_config.robot_config = config_cls()
        return
    if isinstance(robot_overrides, Mapping):
        payload = dict(robot_overrides)
        config_type = payload.pop("type", None)
        if config_type is not None:
            config_cls = _resolve_robot_config_type(config_type)
            if config_cls is None:
                raise ValueError(f"robot_config override type '{config_type}' is not supported")
            env_config.robot_config = config_cls()
        target = env_config.robot_config
        for key, value in payload.items():
            if not hasattr(target, key):
                raise ValueError(f"robot_config override has unknown field '{key}'")
            setattr(target, key, value)
        return
    raise ValueError("robot_config override must be a mapping, dataclass, or type string")


def _apply_grid_override(env_config: Any, overrides: Mapping[str, object]) -> None:
    """Apply occupancy grid overrides to the environment config."""
    grid_override = overrides.get("grid_config")
    if grid_override is None:
        return
    if isinstance(grid_override, GridConfig):
        env_config.grid_config = grid_override
        return
    if isinstance(grid_override, Mapping):
        payload = dict(grid_override)
        if "channels" in payload:
            payload["channels"] = _coerce_grid_channels(payload["channels"])
        env_config.grid_config = GridConfig(**payload)
        return
    raise ValueError("grid_config override must be a mapping or GridConfig instance")


def _apply_nested_overrides(config_obj: object, overrides: Mapping[str, object]) -> None:
    """Recursively apply overrides to nested config objects."""
    for key, value in overrides.items():
        if not hasattr(config_obj, key):
            continue
        current_attr = getattr(config_obj, key)
        if isinstance(value, Mapping) and hasattr(current_attr, "__dict__"):
            _apply_nested_overrides(current_attr, value)
        else:
            setattr(config_obj, key, value)


def _apply_sim_overrides(env_config: Any, overrides: Mapping[str, object]) -> None:
    """Apply sim_config overrides."""
    sim_overrides = overrides.get("sim_config")
    if not isinstance(sim_overrides, Mapping):
        return
    _apply_nested_overrides(env_config.sim_config, sim_overrides)


def apply_env_overrides(env_config: Any, overrides: Mapping[str, object]) -> None:
    """Apply training-config environment overrides to a config object."""
    if not overrides:
        return
    _apply_simple_overrides(env_config, overrides)
    _apply_robot_overrides(env_config, overrides)
    _apply_grid_override(env_config, overrides)
    _apply_sim_overrides(env_config, overrides)


def load_training_env_overrides(training_config_path: Path | None) -> dict[str, object]:
    """Load env_overrides from a training YAML config when available."""
    if training_config_path is None:
        return {}
    raw = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    return dict(raw.get("env_overrides", {}) or {})
