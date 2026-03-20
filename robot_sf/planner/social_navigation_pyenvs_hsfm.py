"""Thin upstream-backed headed-force-model adapter for Social-Navigation-PyEnvs."""

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
    _require_array,
    _upstream_import_context,
)

HSFMPolicyName = Literal["hsfm_new_guo", "hsfm_farina"]

_POLICY_SPECS: dict[HSFMPolicyName, tuple[str, str]] = {
    "hsfm_new_guo": ("crowd_nav.policy_no_train.hsfm_new_guo", "HSFMNewGuo"),
    "hsfm_farina": ("crowd_nav.policy_no_train.hsfm_farina", "HSFMFarina"),
}


@dataclass(frozen=True)
class SocialNavigationPyEnvsHSFMConfig:
    """Configuration for one upstream Social-Navigation-PyEnvs HSFM adapter."""

    policy_name: HSFMPolicyName
    repo_root: Path = Path("output/repos/Social-Navigation-PyEnvs")
    preferred_speed: float = 1.0
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0


def build_social_navigation_pyenvs_hsfm_config(
    data: dict[str, Any] | None,
    *,
    default_policy_name: HSFMPolicyName = "hsfm_new_guo",
) -> SocialNavigationPyEnvsHSFMConfig:
    """Build adapter config from an algo-config mapping.

    Returns:
        SocialNavigationPyEnvsHSFMConfig: Parsed config with explicit upstream checkout
        path and selected upstream headed-force-model policy.
    """
    payload = data or {}
    repo_root_raw = payload.get("repo_root", "output/repos/Social-Navigation-PyEnvs")
    policy_name = str(payload.get("policy_name", default_policy_name)).strip().lower()
    if policy_name not in _POLICY_SPECS:
        raise ValueError(
            f"Unsupported Social-Navigation-PyEnvs HSFM policy '{policy_name}'. "
            f"Expected one of {sorted(_POLICY_SPECS)}."
        )
    preferred_speed = float(payload.get("preferred_speed", 1.0))
    max_linear_speed = float(payload.get("max_linear_speed", 1.0))
    max_angular_speed = float(payload.get("max_angular_speed", 1.0))
    if preferred_speed < 0.0 or max_linear_speed < 0.0 or max_angular_speed < 0.0:
        raise ValueError(
            "preferred_speed, max_linear_speed, and max_angular_speed must be non-negative"
        )
    return SocialNavigationPyEnvsHSFMConfig(
        policy_name=policy_name,
        repo_root=Path(str(repo_root_raw)),
        preferred_speed=preferred_speed,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )


def _world_to_body(world_velocity_xy: np.ndarray, heading: float) -> np.ndarray:
    """Rotate a world-frame planar velocity into the robot body frame.

    Returns:
        np.ndarray: Body-frame planar velocity ``[bvx, bvy]``.
    """
    cos_h = float(math.cos(heading))
    sin_h = float(math.sin(heading))
    vx = float(world_velocity_xy[0])
    vy = float(world_velocity_xy[1])
    return np.array(
        [
            cos_h * vx + sin_h * vy,
            -sin_h * vx + cos_h * vy,
        ],
        dtype=float,
    )


def _robot_headed_observation_fields(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float, str]:
    """Extract required robot/goal state for upstream headed-force-model inference.

    Returns:
        tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float, str]: Robot position,
        goal position, heading, body-frame planar self velocity, angular velocity, robot
        radius, and velocity-source label.
    """
    if "robot" in observation:
        robot_state = observation.get("robot", {})
        goal_state = observation.get("goal", {})
        robot_pos = _require_array(robot_state.get("position"), size=2, field="robot.position")
        goal_pos = _require_array(goal_state.get("current"), size=2, field="goal.current")
        heading = float(
            _require_array(robot_state.get("heading"), size=1, field="robot.heading")[0]
        )
        radius = float(_as_array(robot_state.get("radius"), pad=1, fill=0.3)[0])
        velocity_xy = _require_array(
            robot_state.get("velocity_xy"), size=2, field="robot.velocity_xy"
        )
        body_velocity_xy = _world_to_body(velocity_xy, heading)
        angular_velocity = float(
            _require_array(
                robot_state.get("angular_velocity"),
                size=1,
                field="robot.angular_velocity",
            )[0]
        )
        return (
            robot_pos,
            goal_pos,
            heading,
            body_velocity_xy,
            angular_velocity,
            radius,
            "robot.velocity_xy+robot.angular_velocity",
        )

    robot_pos = _require_array(observation.get("robot_position"), size=2, field="robot_position")
    goal_pos = _require_array(observation.get("goal_current"), size=2, field="goal_current")
    heading = float(
        _require_array(observation.get("robot_heading"), size=1, field="robot_heading")[0]
    )
    radius = float(_as_array(observation.get("robot_radius"), pad=1, fill=0.3)[0])
    velocity_xy = _require_array(
        observation.get("robot_velocity_xy"), size=2, field="robot_velocity_xy"
    )
    body_velocity_xy = _world_to_body(velocity_xy, heading)
    angular_velocity = float(
        _require_array(
            observation.get("robot_angular_velocity"),
            size=1,
            field="robot_angular_velocity",
        )[0]
    )
    return (
        robot_pos,
        goal_pos,
        heading,
        body_velocity_xy,
        angular_velocity,
        radius,
        "robot_velocity_xy+robot_angular_velocity",
    )


class SocialNavigationPyEnvsHSFMAdapter:
    """Thin adapter from Robot SF SocNav observations to upstream HSFM inference."""

    projection_policy = "body_velocity_heading_safe_to_unicycle_vw"

    def __init__(self, config: SocialNavigationPyEnvsHSFMConfig) -> None:
        """Load one upstream non-trainable headed-force-model policy from a checked-out repo."""
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
        self._ActionXYW = action_mod.ActionXYW
        self._NewHeadedState = action_mod.NewHeadedState
        self._FullStateHeaded = state_mod.FullStateHeaded
        self._ObservableState = state_mod.ObservableState
        self._ObservableStateHeaded = state_mod.ObservableStateHeaded
        self._JointState = state_mod.JointState
        self._policy = getattr(policy_mod, class_name)()
        self.upstream_policy = f"{module_name}.{class_name}"

    def _joint_state(self, observation: dict[str, Any]) -> Any:
        """Translate Robot SF observation into the upstream headed JointState contract.

        Returns:
            Any: Upstream ``JointState`` instance for headed-force-model inference.
        """
        (
            robot_pos,
            goal_pos,
            heading,
            body_velocity_xy,
            angular_velocity,
            radius,
            velocity_source,
        ) = _robot_headed_observation_fields(observation)
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

        self_state = self._FullStateHeaded(
            float(robot_pos[0]),
            float(robot_pos[1]),
            float(body_velocity_xy[0]),
            float(body_velocity_xy[1]),
            radius,
            float(goal_pos[0]),
            float(goal_pos[1]),
            float(self.config.preferred_speed),
            heading,
            angular_velocity,
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
        dt = max(float(time_step), 1e-6)
        self._policy.time_step = dt
        action = self._policy.predict(joint_state)

        if isinstance(action, self._ActionXYW):
            body_vx = float(action.bvx)
            body_vy = float(action.bvy)
            angular_velocity = float(action.w)
            action_kind = "ActionXYW"
        elif isinstance(action, self._NewHeadedState):
            body_vx = float(action.bvx)
            body_vy = float(action.bvy)
            angular_velocity = float(action.w)
            action_kind = "NewHeadedState"
        else:
            raise TypeError(f"Unexpected upstream action type: {type(action)}")

        body_speed = float(np.hypot(body_vx, body_vy))
        body_heading_offset = float(math.atan2(body_vy, body_vx)) if body_speed > 1e-8 else 0.0
        linear = float(
            np.clip(
                body_speed * max(0.0, math.cos(body_heading_offset)),
                0.0,
                float(self.config.max_linear_speed),
            )
        )
        angular = float(
            np.clip(
                angular_velocity + (body_heading_offset / dt),
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        return (
            linear,
            angular,
            {
                "upstream_action_body_xyw": [body_vx, body_vy, angular_velocity],
                "upstream_action_kind": action_kind,
                "projected_command_vw": [linear, angular],
                "body_heading_offset_rad": body_heading_offset,
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
        try:
            dt_arr = np.asarray(0.1 if dt_source is None else dt_source, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            dt_arr = np.asarray([0.1], dtype=float)
        dt = float(dt_arr[0]) if dt_arr.size else 0.1
        if not math.isfinite(dt) or dt <= 0.0:
            dt = 0.1
        linear, angular, _meta = self.act(observation, time_step=dt)
        return linear, angular


__all__ = [
    "SocialNavigationPyEnvsHSFMAdapter",
    "SocialNavigationPyEnvsHSFMConfig",
    "build_social_navigation_pyenvs_hsfm_config",
]
