"""Action and kinematics helpers for map-based benchmark runs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.robot.action_adapters import holonomic_to_diff_drive_action

if TYPE_CHECKING:
    from robot_sf.gym_env.unified_config import RobotSimulationConfig


DEFAULT_KINEMATICS = "differential_drive"


def robot_kinematics_label(config: RobotSimulationConfig) -> str:
    """Derive the runtime robot kinematics label from simulation config.

    Returns:
        Canonical kinematics label used in benchmark metadata.
    """
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return DEFAULT_KINEMATICS
    cls_name = robot_cfg.__class__.__name__.lower()
    if "bicycle" in cls_name:
        return "bicycle_drive"
    if "differential" in cls_name:
        return "differential_drive"
    if "holonomic" in cls_name or "omni" in cls_name:
        return "holonomic"
    return cls_name or DEFAULT_KINEMATICS


def robot_max_speed(config: RobotSimulationConfig) -> float | None:
    """Extract a positive robot max-speed setting from simulation config if available.

    Returns:
        Configured positive max speed, or ``None`` when not available.
    """
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return None
    for attr in ("max_linear_speed", "max_velocity", "max_speed"):
        value = getattr(robot_cfg, attr, None)
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
    return None


def scenario_robot_kinematics_label(scenario: dict[str, Any]) -> str:
    """Derive the scenario-declared robot kinematics label from scenario metadata.

    Returns:
        Canonical kinematics label inferred from scenario robot configuration fields.
    """
    robot_cfg = scenario.get("robot_config")
    if not isinstance(robot_cfg, dict):
        return DEFAULT_KINEMATICS
    raw = str(robot_cfg.get("type") or robot_cfg.get("model") or "").strip().lower()
    if "bicycle" in raw:
        return "bicycle_drive"
    if "holonomic" in raw or "omni" in raw:
        return "holonomic"
    if "differential" in raw or raw == "":
        return DEFAULT_KINEMATICS
    return raw


def vel_and_acc(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite-difference velocity and acceleration arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: Velocity and acceleration with input shape.
    """
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def stack_ped_positions(traj: list[np.ndarray], *, fill_value: float = np.nan) -> np.ndarray:
    """Stack variable-count pedestrian position arrays into one padded tensor.

    Returns:
        np.ndarray: Array shaped ``(time, max_pedestrians, 2)``.
    """
    if not traj:
        return np.zeros((0, 0, 2), dtype=float)
    first_shape = traj[0].shape
    if all(arr.shape == first_shape for arr in traj):
        return np.stack(traj).astype(float, copy=False)
    max_k = max(p.shape[0] for p in traj)
    stacked = np.full((len(traj), max_k, 2), fill_value, dtype=float)
    for i, arr in enumerate(traj):
        if arr.size == 0:
            continue
        stacked[i, : arr.shape[0]] = arr
    return stacked


def command_xy_payload(command: tuple[float, float] | dict[str, Any]) -> np.ndarray:
    """Extract a world-frame XY payload from tuple or structured commands.

    Returns:
        np.ndarray: Two-element world-frame XY payload.
    """
    if isinstance(command, dict):
        return np.array(
            [float(command.get("vx", 0.0)), float(command.get("vy", 0.0))],
            dtype=float,
        )
    return np.array([float(command[0]), float(command[1])], dtype=float)


def policy_command_to_env_action(  # noqa: C901
    *,
    env: Any,
    config: RobotSimulationConfig,
    command: tuple[float, float] | dict[str, Any],
) -> np.ndarray:
    """Convert a policy command into the robot's native environment action space.

    Returns:
        np.ndarray: Action vector compatible with ``env.step``.
    """
    simulator = getattr(env, "simulator", None)
    sim_robots = getattr(simulator, "robots", None)
    if not isinstance(sim_robots, list) or not sim_robots:
        return command_xy_payload(command)
    robot = sim_robots[0]
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return command_xy_payload(command)

    if isinstance(command, dict):
        command_kind = str(command.get("command_kind", "")).strip().lower()
        if command_kind != "holonomic_vxy_world":
            raise ValueError(f"Unsupported structured policy command: {command}")
        velocity_world = np.array(
            [float(command.get("vx", 0.0)), float(command.get("vy", 0.0))],
            dtype=float,
        )
        max_linear_speed = float(
            getattr(robot_cfg, "max_linear_speed", getattr(robot_cfg, "max_speed", 0.0)) or 0.0
        )
        max_angular_speed = float(getattr(robot_cfg, "max_angular_speed", 0.0) or 0.0)

    cls_name = robot_cfg.__class__.__name__.lower()
    if isinstance(command, dict):
        if "holonomic" in cls_name:
            mode = str(getattr(robot_cfg, "command_mode", "vx_vy")).strip().lower()
            if mode == "vx_vy":
                return velocity_world
            command_vw = holonomic_to_diff_drive_action(
                velocity_world,
                robot.pose,
                max_linear_speed=max_linear_speed,
                max_angular_speed=max_angular_speed,
            )
            return np.asarray(command_vw, dtype=float)

        command_vw = holonomic_to_diff_drive_action(
            velocity_world,
            robot.pose,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
        )
        if "bicycle" in cls_name:
            adapter = PlannerActionAdapter(
                robot=robot,
                action_space=env.action_space,
                time_step=float(config.sim_config.time_per_step_in_secs),
            )
            return np.asarray(
                adapter.from_velocity_command(tuple(command_vw.tolist())), dtype=float
            )
        current_linear, current_angular = robot.current_speed
        d_linear = float(command_vw[0]) - float(current_linear)
        d_angular = float(command_vw[1]) - float(current_angular)
        return np.array([d_linear, d_angular], dtype=float)

    if "bicycle" in cls_name:
        adapter = PlannerActionAdapter(
            robot=robot,
            action_space=env.action_space,
            time_step=float(config.sim_config.time_per_step_in_secs),
        )
        return np.asarray(adapter.from_velocity_command(command), dtype=float)

    if "holonomic" in cls_name:
        mode = str(getattr(robot_cfg, "command_mode", "vx_vy")).strip().lower()
        linear, angular = float(command[0]), float(command[1])
        if mode == "vx_vy":
            # Preserve turning intent by projecting at midpoint heading over this step.
            step_dt = float(getattr(config.sim_config, "time_per_step_in_secs", 0.0) or 0.0)
            heading = float(robot.pose[1]) + (angular * max(step_dt, 0.0) * 0.5)
            vx = linear * math.cos(heading)
            vy = linear * math.sin(heading)
            return np.array([vx, vy], dtype=float)
        return np.array([linear, angular], dtype=float)

    current_linear, current_angular = robot.current_speed
    d_linear = float(command[0]) - float(current_linear)
    d_angular = float(command[1]) - float(current_angular)
    return np.array([d_linear, d_angular], dtype=float)
