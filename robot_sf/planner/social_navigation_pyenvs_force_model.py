"""Thin upstream-backed force-model adapters for Social-Navigation-PyEnvs."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from robot_sf.planner.social_navigation_pyenvs_orca import (
    _as_array,
    _as_matrix,
    _current_heading,
    _normalize_angle,
    _robot_goal_observation_fields,
    _upstream_import_context,
)

ForceModelName = Literal["socialforce", "sfm_helbing"]

_POLICY_SPECS: dict[ForceModelName, tuple[str, str]] = {
    "socialforce": ("crowd_nav.policy_no_train.socialforce", "SocialForce"),
    "sfm_helbing": ("crowd_nav.policy_no_train.sfm_helbing", "SFMHelbing"),
}


@dataclass(frozen=True)
class SocialNavigationPyEnvsForceModelConfig:
    """Configuration for one upstream Social-Navigation-PyEnvs force-model adapter."""

    policy_name: ForceModelName
    repo_root: Path = Path("output/repos/Social-Navigation-PyEnvs")
    preferred_speed: float = 1.0
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0


def build_social_navigation_pyenvs_force_model_config(
    data: dict[str, Any] | None,
    *,
    default_policy_name: ForceModelName,
) -> SocialNavigationPyEnvsForceModelConfig:
    """Build adapter config from an algo-config mapping.

    Returns:
        SocialNavigationPyEnvsForceModelConfig: Parsed config with explicit upstream checkout
        path and selected upstream policy.
    """
    payload = data or {}
    repo_root_raw = payload.get("repo_root", "output/repos/Social-Navigation-PyEnvs")
    policy_name = str(payload.get("policy_name", default_policy_name)).strip().lower()
    if policy_name not in _POLICY_SPECS:
        raise ValueError(
            f"Unsupported Social-Navigation-PyEnvs force-model policy '{policy_name}'. "
            f"Expected one of {sorted(_POLICY_SPECS)}."
        )
    preferred_speed = float(payload.get("preferred_speed", 1.0))
    max_linear_speed = float(payload.get("max_linear_speed", 1.0))
    max_angular_speed = float(payload.get("max_angular_speed", 1.0))
    if preferred_speed < 0.0 or max_linear_speed < 0.0 or max_angular_speed < 0.0:
        raise ValueError(
            "preferred_speed, max_linear_speed, and max_angular_speed must be non-negative"
        )
    return SocialNavigationPyEnvsForceModelConfig(
        policy_name=policy_name,
        repo_root=Path(str(repo_root_raw)),
        preferred_speed=preferred_speed,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )


class SocialNavigationPyEnvsForceModelAdapter:
    """Thin adapter from Robot SF SocNav observations to upstream force-model inference."""

    projection_policy = "heading_safe_velocity_to_unicycle_vw"

    def __init__(self, config: SocialNavigationPyEnvsForceModelConfig) -> None:
        """Load one upstream non-trainable force-model policy from a checked-out repo."""
        self.config = config
        self.repo_root = self.config.repo_root.resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(
                "Social-Navigation-PyEnvs checkout not found: "
                f"{self.config.repo_root}. Clone the upstream repo under output/repos/ first."
            )
        module_name, class_name = _POLICY_SPECS[self.config.policy_name]
        with _upstream_import_context(self.repo_root):
            action_mod = importlib.import_module("crowd_nav.utils.action")
            state_mod = importlib.import_module("crowd_nav.utils.state")
            policy_mod = importlib.import_module(module_name)
        self._ActionXY = action_mod.ActionXY
        self._FullState = state_mod.FullState
        self._ObservableState = state_mod.ObservableState
        self._JointState = state_mod.JointState
        self._policy = getattr(policy_mod, class_name)()
        self.upstream_policy = f"{module_name}.{class_name}"

    def _joint_state(self, observation: dict[str, Any]) -> Any:
        """Translate Robot SF observation into the upstream JointState contract.

        Returns:
            Any: Upstream ``JointState`` instance for force-model inference.
        """
        robot_pos, goal_pos, heading, robot_velocity_xy, radius, velocity_source = (
            _robot_goal_observation_fields(observation)
        )
        if "robot" in observation:
            ped_state = observation.get("pedestrians", {})
            ped_positions = _as_matrix(ped_state.get("positions"), cols=2)
            ped_velocities = _as_matrix(ped_state.get("velocities"), cols=2)
            ped_count = (
                int(_as_array(ped_state.get("count"), pad=1)[0])
                if ped_state.get("count") is not None
                else ped_positions.shape[0]
            )
            ped_radius_value = float(_as_array(ped_state.get("radius"), pad=1, fill=0.3)[0])
        else:
            ped_positions = _as_matrix(observation.get("pedestrians_positions"), cols=2)
            ped_velocities = _as_matrix(observation.get("pedestrians_velocities"), cols=2)
            ped_count = (
                int(_as_array(observation.get("pedestrians_count"), pad=1)[0])
                if observation.get("pedestrians_count") is not None
                else ped_positions.shape[0]
            )
            ped_radius_value = float(
                _as_array(observation.get("pedestrians_radius"), pad=1, fill=0.3)[0]
            )

        vx = float(robot_velocity_xy[0])
        vy = float(robot_velocity_xy[1])
        self_state = self._FullState(
            float(robot_pos[0]),
            float(robot_pos[1]),
            vx,
            vy,
            radius,
            float(goal_pos[0]),
            float(goal_pos[1]),
            float(self.config.preferred_speed),
            heading,
        )
        ped_count = min(ped_count, ped_positions.shape[0], ped_velocities.shape[0])
        humans = [
            self._ObservableState(
                float(ped_positions[i, 0]),
                float(ped_positions[i, 1]),
                float(ped_velocities[i, 0]),
                float(ped_velocities[i, 1]),
                ped_radius_value,
            )
            for i in range(ped_count)
        ]
        joint_state = self._JointState(self_state, humans)
        joint_state.robot_sf_velocity_source = velocity_source
        return joint_state

    def act(
        self, observation: dict[str, Any], *, time_step: float
    ) -> tuple[float, float, dict[str, Any]]:
        """Return projected `(v, w)` command plus explicit projection metadata."""
        joint_state = self._joint_state(observation)
        self._policy.time_step = float(time_step)
        action = self._policy.predict(joint_state)
        if not isinstance(action, self._ActionXY):
            raise TypeError(f"Unexpected upstream action type: {type(action)}")

        vx = float(action.vx)
        vy = float(action.vy)
        speed = float(np.hypot(vx, vy))
        current_heading = _current_heading(observation)
        desired_heading = float(math.atan2(vy, vx)) if speed > 1e-8 else current_heading
        heading_error = _normalize_angle(desired_heading - current_heading)
        dt = max(float(time_step), 1e-6)
        linear = float(
            np.clip(
                speed * max(0.0, math.cos(heading_error)),
                0.0,
                float(self.config.max_linear_speed),
            )
        )
        angular = float(
            np.clip(
                heading_error / dt,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        return (
            linear,
            angular,
            {
                "upstream_action_xy": [vx, vy],
                "projected_command_vw": [linear, angular],
                "heading_error_rad": heading_error,
                "projection_policy": self.projection_policy,
                "self_velocity_source": getattr(joint_state, "robot_sf_velocity_source", "unknown"),
                "upstream_policy": self.upstream_policy,
            },
        )

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Map-runner adapter entrypoint that returns only the projected command.

        Returns:
            tuple[float, float]: Projected ``(linear, angular)`` command.
        """
        dt_source = observation.get("dt", 0.1)
        if "sim" in observation and isinstance(observation["sim"], dict):
            dt_source = observation["sim"].get("timestep", dt_source)
        dt = float(np.asarray(dt_source, dtype=float).reshape(-1)[0])
        linear, angular, _meta = self.act(observation, time_step=dt)
        return linear, angular


__all__ = [
    "SocialNavigationPyEnvsForceModelAdapter",
    "SocialNavigationPyEnvsForceModelConfig",
    "build_social_navigation_pyenvs_force_model_config",
]
