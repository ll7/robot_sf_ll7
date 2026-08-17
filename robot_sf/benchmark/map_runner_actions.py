"""Compatibility import for the canonical map-runner action helpers."""

from __future__ import annotations

import sys

from robot_sf.benchmark.map_runner_policies import map_runner_actions as _canonical
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    DEFAULT_KINEMATICS,
    command_xy_payload,
    policy_command_to_env_action,
    robot_kinematics_label,
    robot_max_speed,
    scenario_robot_kinematics_label,
    stack_ped_positions,
    vel_and_acc,
)

__all__ = (
    "DEFAULT_KINEMATICS",
    "command_xy_payload",
    "policy_command_to_env_action",
    "robot_kinematics_label",
    "robot_max_speed",
    "scenario_robot_kinematics_label",
    "stack_ped_positions",
    "vel_and_acc",
)

sys.modules[__name__] = _canonical
