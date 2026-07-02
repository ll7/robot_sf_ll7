"""Action-conversion helpers shared by map-runner learned-policy builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi as _normalize_heading
from robot_sf.planner.kinematics_model import KinematicsModel, resolve_benchmark_kinematics_model


def ppo_action_to_unicycle(
    action: dict[str, Any],
    obs: dict[str, Any],
    cfg: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    kinematics_model: KinematicsModel | None = None,
    project_command: bool = True,
) -> tuple[float, float, str]:
    """Convert PPO-style action dictionaries into map-runner unicycle commands.

    Returns:
        Tuple ``(linear_velocity, angular_velocity, conversion_mode)`` where
        conversion_mode is either ``"native"`` or ``"adapter"``.
    """
    model = kinematics_model or resolve_benchmark_kinematics_model(
        robot_kinematics=robot_kinematics,
        command_limits=cfg,
    )
    if "v" in action and "omega" in action:
        if project_command:
            v, omega = model.project((float(action["v"]), float(action["omega"])))
        else:
            v, omega = float(action["v"]), float(action["omega"])
        return v, omega, "native"

    if "vx" not in action or "vy" not in action:
        raise ValueError(f"Unsupported PPO action payload: {action}")

    vx = float(action["vx"])
    vy = float(action["vy"])
    speed = float(np.hypot(vx, vy))
    if speed < 1e-9:
        if project_command:
            v, omega = model.project((0.0, 0.0))
        else:
            v, omega = 0.0, 0.0
        return v, omega, "adapter"

    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])
    desired_heading = float(np.arctan2(vy, vx))
    heading_error = _normalize_heading(desired_heading - heading)
    omega_max = float(cfg.get("omega_max", cfg.get("max_angular_speed", 1.0)))
    omega_kp = float(cfg.get("omega_kp", cfg.get("heading_error_gain", 1.0)))
    angular_velocity = float(np.clip(omega_kp * heading_error, -omega_max, omega_max))
    if project_command:
        v, omega = model.project((float(speed), angular_velocity))
    else:
        v, omega = float(speed), angular_velocity
    return v, omega, "adapter"


def update_adapter_impact_metrics(
    meta: dict[str, Any],
    conversion_mode: str,
    *,
    count_native: bool | None = None,
) -> None:
    """Update native-vs-adapted step counters when adapter-impact probing is enabled."""
    impact = meta.get("adapter_impact")
    if not isinstance(impact, dict) or not bool(impact.get("requested", False)):
        return
    if count_native is None:
        count_native = conversion_mode == "native"
    if count_native:
        impact["native_steps"] = int(impact.get("native_steps", 0)) + 1
    else:
        impact["adapted_steps"] = int(impact.get("adapted_steps", 0)) + 1
    impact["status"] = "collecting"
