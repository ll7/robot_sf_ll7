"""Explicit interaction-law profiles independent of pedestrian integration models."""

from dataclasses import asdict

from pysocialforce.config import (
    GroupGazeForceV2Config,
    GroupRepulsiveForceV2Config,
    SimulatorConfig,
)

from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig, PedRobotForceV2Config

PEDESTRIAN_INTERACTION_V2 = "pedestrian_interaction_v2"


def normalize_pedestrian_force_profile(value: str | None) -> str | None:
    """Validate the optional selector; absence preserves legacy laws.

    Returns:
        The supported selector or None.
    """
    if value is None or value == PEDESTRIAN_INTERACTION_V2:
        return value
    raise ValueError(f"Unsupported pedestrian_force_profile: {value!r}")


def apply_pedestrian_force_profile(
    config: SimulatorConfig,
    robot_config: PedRobotForceConfig,
    profile: str | None,
) -> PedRobotForceConfig:
    """Select only group gaze/repulsion and robot avoidance; return the robot config.

    Existing force factors and group distance gates are preserved. A caller may
    supply ``PedRobotForceV2Config`` to vary the edge onset for sensitivity analysis.
    Absence does not replace or mutate any force config.

    Returns:
        The selected robot interaction config.
    """
    profile = normalize_pedestrian_force_profile(profile)
    if profile is None:
        return robot_config
    config.group_gaze_force_config = GroupGazeForceV2Config(
        **asdict(config.group_gaze_force_config)
    )
    config.group_repulsive_force_config = GroupRepulsiveForceV2Config(
        **asdict(config.group_repulsive_force_config)
    )
    if isinstance(robot_config, PedRobotForceV2Config):
        return robot_config
    return PedRobotForceV2Config(**asdict(robot_config))
