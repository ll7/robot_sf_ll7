"""Fail closed when a planner has no verified contract for a selected body."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class UnsupportedRobotFootprintError(ValueError):
    """A planner would silently replace the selected physical capsule with a disc."""


def require_supported_robot_footprint(planner: str, robot_config: Any) -> None:
    """Refuse all currently unverified capsule planner combinations.

    An execution command projection is not evidence of heading-aware body
    clearance or candidate/plant agreement. Manual plant diagnostics can still
    drive the robot directly; no planner gets an implicit disc substitution.
    """
    capsule = (
        robot_config.get("capsule")
        if isinstance(robot_config, Mapping)
        else getattr(robot_config, "capsule", None)
    )
    if capsule is not None:
        raise UnsupportedRobotFootprintError(
            f"Planner '{planner}' does not support the selected capsule/bicycle robot: "
            "heading-aware capsule clearance and executed bicycle candidate rollouts "
            "are not verified. Use a manual plant diagnostic or an explicitly selected "
            "disc profile; no covering-disc fallback is applied."
        )


def require_supported_env_footprint(planner: str, env: Any) -> None:
    """Check both authored settings and live robots before binding a planner."""
    for name in ("env_config", "config"):
        require_supported_robot_footprint(
            planner, getattr(getattr(env, name, None), "robot_config", None)
        )
    robots = getattr(getattr(env, "simulator", None), "robots", None)
    if isinstance(robots, (list, tuple)):
        for robot in robots:
            require_supported_robot_footprint(planner, getattr(robot, "config", None))
