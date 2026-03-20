"""Thin upstream-backed ORCA adapter for Social-Navigation-PyEnvs."""

from __future__ import annotations

import importlib
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class SocialNavigationPyEnvsORCAConfig:
    """Configuration for the upstream Social-Navigation-PyEnvs ORCA adapter."""

    repo_root: Path = Path("output/repos/Social-Navigation-PyEnvs")
    preferred_speed: float = 1.0
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0


def build_social_navigation_pyenvs_orca_config(
    data: dict[str, Any] | None,
) -> SocialNavigationPyEnvsORCAConfig:
    """Build adapter config from an algo-config mapping.

    Returns:
        SocialNavigationPyEnvsORCAConfig: Parsed config with explicit upstream checkout path.
    """
    payload = data or {}
    repo_root_raw = payload.get("repo_root", "output/repos/Social-Navigation-PyEnvs")
    return SocialNavigationPyEnvsORCAConfig(
        repo_root=Path(str(repo_root_raw)),
        preferred_speed=float(payload.get("preferred_speed", 1.0)),
        max_linear_speed=float(payload.get("max_linear_speed", 1.0)),
        max_angular_speed=float(payload.get("max_angular_speed", 1.0)),
    )


def _normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi).

    Returns:
        float: Wrapped heading angle in radians.
    """
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _as_array(value: Any, *, pad: int, fill: float = 0.0) -> np.ndarray:
    """Return a 1D float array padded or cropped to the requested length."""
    arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    if arr.size >= pad:
        return arr[:pad]
    out = np.full((pad,), fill, dtype=float)
    out[: arr.size] = arr
    return out


def _require_array(value: Any, *, size: int, field: str) -> np.ndarray:
    """Return a required 1D float array or raise a clear contract error."""
    arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    if arr.size < size:
        raise ValueError(f"Missing or malformed required field: {field}")
    return arr[:size]


def _as_matrix(value: Any, *, cols: int) -> np.ndarray:
    """Return a 2D float array with the requested column count."""
    arr = np.asarray(value if value is not None else [], dtype=float)
    if arr.size == 0:
        return np.zeros((0, cols), dtype=float)
    return arr.reshape(-1, cols)


def _robot_goal_observation_fields(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Extract required robot/goal state from nested or flat observation variants.

    Returns:
        tuple[np.ndarray, np.ndarray, float, float, float]: Robot position, goal position,
        heading, speed, and robot radius.
    """
    if "robot" in observation:
        robot_state = observation.get("robot", {})
        goal_state = observation.get("goal", {})
        robot_pos = _require_array(robot_state.get("position"), size=2, field="robot.position")
        goal_pos = _require_array(goal_state.get("current"), size=2, field="goal.current")
        heading = float(
            _require_array(robot_state.get("heading"), size=1, field="robot.heading")[0]
        )
        speed = float(_require_array(robot_state.get("speed"), size=1, field="robot.speed")[0])
        radius = float(_as_array(robot_state.get("radius"), pad=1, fill=0.3)[0])
        return robot_pos, goal_pos, heading, speed, radius

    robot_pos = _require_array(observation.get("robot_position"), size=2, field="robot_position")
    goal_pos = _require_array(observation.get("goal_current"), size=2, field="goal_current")
    heading = float(
        _require_array(observation.get("robot_heading"), size=1, field="robot_heading")[0]
    )
    speed = float(_require_array(observation.get("robot_speed"), size=1, field="robot_speed")[0])
    radius = float(_as_array(observation.get("robot_radius"), pad=1, fill=0.3)[0])
    return robot_pos, goal_pos, heading, speed, radius


def _current_heading(observation: dict[str, Any]) -> float:
    """Extract the robot heading from nested or flat observation variants.

    Returns:
        float: Robot heading in radians.
    """
    if "robot" in observation:
        return float(
            _require_array(
                observation.get("robot", {}).get("heading"),
                size=1,
                field="robot.heading",
            )[0]
        )
    return float(_require_array(observation.get("robot_heading"), size=1, field="robot_heading")[0])


@contextmanager
def _upstream_import_context(repo_root: Path) -> Iterator[None]:
    """Temporarily prepend the upstream repo root and restore import state."""
    repo_str = str(repo_root)
    original_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "crowd_nav" or name.startswith("crowd_nav.")
    }
    sys.path.insert(0, repo_str)
    try:
        for name in list(sys.modules):
            if name == "crowd_nav" or name.startswith("crowd_nav."):
                sys.modules.pop(name, None)
        yield
    finally:
        for name in list(sys.modules):
            if name == "crowd_nav" or name.startswith("crowd_nav."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_path


class SocialNavigationPyEnvsORCAAdapter:
    """Thin adapter from Robot SF SocNav observations to upstream ORCA policy inference."""

    projection_policy = "heading_safe_velocity_to_unicycle_vw"
    upstream_policy = "crowd_nav.policy_no_train.orca.ORCA"

    def __init__(self, config: SocialNavigationPyEnvsORCAConfig | None = None) -> None:
        """Load upstream ORCA policy and state/action types from a checked-out repo."""
        self.config = config or SocialNavigationPyEnvsORCAConfig()
        self.repo_root = self.config.repo_root.resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(
                "Social-Navigation-PyEnvs checkout not found: "
                f"{self.config.repo_root}. Clone the upstream repo under output/repos/ first."
            )
        with _upstream_import_context(self.repo_root):
            action_mod = importlib.import_module("crowd_nav.utils.action")
            state_mod = importlib.import_module("crowd_nav.utils.state")
            policy_mod = importlib.import_module("crowd_nav.policy_no_train.orca")
        self._ActionXY = action_mod.ActionXY
        self._FullState = state_mod.FullState
        self._ObservableState = state_mod.ObservableState
        self._JointState = state_mod.JointState
        self._policy = policy_mod.ORCA()

    def _joint_state(self, observation: dict[str, Any]) -> Any:
        """Translate Robot SF observation into the upstream ORCA joint-state contract.

        Returns:
            Any: Upstream ``JointState`` instance for ORCA inference.
        """
        robot_pos, goal_pos, heading, speed, radius = _robot_goal_observation_fields(observation)
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

        vx = float(speed * math.cos(heading))
        vy = float(speed * math.sin(heading))
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
        return self._JointState(self_state, humans)

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
    "SocialNavigationPyEnvsORCAAdapter",
    "SocialNavigationPyEnvsORCAConfig",
    "_upstream_import_context",
    "build_social_navigation_pyenvs_orca_config",
]
