"""Action-conversion helpers shared by map-runner learned-policy builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi as _normalize_heading
from robot_sf.planner.kinematics_model import KinematicsModel, resolve_benchmark_kinematics_model


def _require_finite(name: str, value: float) -> float:
    """Reject non-finite command-conversion inputs before numerical operations.

    Returns:
        float: The validated finite value.
    """
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _ppo_action_bounds(cfg: Mapping[str, Any]) -> tuple[str, float, float]:
    """Resolve and validate the scalar bounds declared by a PPO config.

    Returns:
        Tuple of action-space name, maximum linear speed, and maximum angular speed.
    """
    action_space = str(cfg.get("action_space", "velocity")).strip().lower()
    if action_space not in {"velocity", "unicycle"}:
        raise ValueError(f"Unsupported PPO action_space: {action_space!r}")
    v_max = _require_finite("v_max", float(cfg.get("v_max", 2.0)))
    omega_max = _require_finite("omega_max", float(cfg.get("omega_max", 1.0)))
    if v_max < 0.0:
        raise ValueError(f"v_max must be non-negative, got {v_max!r}")
    if omega_max < 0.0:
        raise ValueError(f"omega_max must be non-negative, got {omega_max!r}")
    return action_space, v_max, omega_max


def _ppo_action_field(action: Mapping[str, Any], name: str, *, action_space: str) -> float:
    """Read one finite PPO action field with a stable error message.

    Returns:
        The validated finite field value.
    """
    if name not in action:
        raise ValueError(f"PPO {action_space} action is missing '{name}'")
    raw = action[name]
    if isinstance(raw, bool):
        raise ValueError(f"PPO action field '{name}' must be a finite number")
    return _require_finite(name, float(raw))


def validate_ppo_policy_action(
    action: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate one PPO action in its declared policy space before projection.

    PPO adapters historically projected commands at the benchmark boundary. That
    projection is useful for runtime robustness, but it can hide a policy that
    emitted an out-of-contract action. This validator is deliberately separate
    from projection so callers can fail closed before any clipping occurs.

    Returns:
        A JSON-compatible action-contract payload describing the checked bounds.

    Raises:
        ValueError: If the action/config is malformed or outside its policy bounds.
    """
    if not isinstance(action, Mapping):
        raise TypeError(f"PPO action must be a mapping, got {type(action).__name__}")
    action_space, v_max, omega_max = _ppo_action_bounds(cfg)

    if action_space == "unicycle":
        linear = _ppo_action_field(action, "v", action_space=action_space)
        angular = _ppo_action_field(action, "omega", action_space=action_space)
        if linear < -tolerance or linear > v_max + tolerance:
            raise ValueError(f"PPO action field 'v'={linear!r} is outside [0.0, {v_max}]")
        if angular < -omega_max - tolerance or angular > omega_max + tolerance:
            raise ValueError(
                f"PPO action field 'omega'={angular!r} is outside [-{omega_max}, {omega_max}]"
            )
        output_keys = ["v", "omega"]
        bounds = {
            "v": [0.0, float(v_max)],
            "omega": [-float(omega_max), float(omega_max)],
        }
    else:
        vx = _ppo_action_field(action, "vx", action_space=action_space)
        vy = _ppo_action_field(action, "vy", action_space=action_space)
        speed = float(np.hypot(vx, vy))
        if speed > v_max + tolerance:
            raise ValueError(f"PPO velocity action speed={speed!r} exceeds v_max={v_max}")
        output_keys = ["vx", "vy"]
        bounds = {
            "speed": [0.0, float(v_max)],
            "vx": [-float(v_max), float(v_max)],
            "vy": [-float(v_max), float(v_max)],
        }

    return {
        "status": "valid",
        "action_space": action_space,
        "output_keys": output_keys,
        "bounds": bounds,
        "source": "PPOPlannerConfig",
    }


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
        v = _require_finite("v", float(action["v"]))
        omega = _require_finite("omega", float(action["omega"]))
        if project_command:
            v, omega = model.project((v, omega))
        return v, omega, "native"

    if "vx" not in action or "vy" not in action:
        raise ValueError(f"Unsupported PPO action payload: {action}")

    vx = _require_finite("vx", float(action["vx"]))
    vy = _require_finite("vy", float(action["vy"]))
    speed = float(np.hypot(vx, vy))
    if speed < 1e-9:
        if project_command:
            v, omega = model.project((0.0, 0.0))
        else:
            v, omega = 0.0, 0.0
        return v, omega, "adapter"

    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    raw_heading = robot.get("heading")
    if raw_heading is None:
        raw_heading = [0.0]
    heading_arr = np.asarray(raw_heading, dtype=float).reshape(-1)
    heading = float(heading_arr[0]) if heading_arr.size > 0 else 0.0
    desired_heading = float(np.arctan2(vy, vx))
    heading_error = _normalize_heading(desired_heading - heading)
    omega_max = _require_finite(
        "omega_max", float(cfg.get("omega_max", cfg.get("max_angular_speed", 1.0)))
    )
    omega_kp = _require_finite(
        "omega_kp", float(cfg.get("omega_kp", cfg.get("heading_error_gain", 1.0)))
    )
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
