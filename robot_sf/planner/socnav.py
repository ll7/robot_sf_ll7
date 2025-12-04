"""
Lightweight adapters for running SocNavBench-style planners against the structured
observation mode.

These adapters are intentionally minimal to provide an in-process bridge while
full SocNavBench planners are integrated. They operate on the SocNav structured
observation emitted when `ObservationMode.SOCNAV_STRUCT` is enabled.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, pi
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SocNavPlannerConfig:
    """Simple config for SocNav-like planner adapters."""

    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0
    angular_gain: float = 1.0
    goal_tolerance: float = 0.25


class SamplingPlannerAdapter:
    """
    Minimal waypoint-to-velocity adapter inspired by SocNavBench sampling planner.

    This is a placeholder that consumes structured SocNav observations and returns
    differential-drive (v, w) commands. It is designed so that the internals can be
    swapped for the real SocNavBench sampling planner without changing callers.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None):
        """Initialize the adapter with optional planner configuration."""

        self.config = config or SocNavPlannerConfig()

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute a (v, w) command from the structured observation.

        Args:
            observation: SocNav structured observation Dict (robot, goal, pedestrians, map, sim).

        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        robot_state = observation["robot"]
        goal_state = observation["goal"]
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(robot_state["heading"][0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        distance = float(np.linalg.norm(to_goal))
        if distance < self.config.goal_tolerance:
            return 0.0, 0.0

        desired_heading = atan2(to_goal[1], to_goal[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)

        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )

        # Slow down when sharply turning
        linear_scale = max(0.0, 1.0 - abs(heading_error) / pi)
        linear = float(
            np.clip(distance * linear_scale, 0.0, self.config.max_linear_speed),
        )
        return linear, angular

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap angle to [-pi, pi].

        Returns:
            float: Wrapped angle in radians.
        """
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle


class SocNavPlannerPolicy:
    """Thin policy wrapper to plug planner adapters into Gym loops."""

    def __init__(self, adapter: SamplingPlannerAdapter | None = None):
        """Initialize the policy with a planner adapter."""

        self.adapter = adapter or SamplingPlannerAdapter()

    def act(self, observation: dict) -> tuple[float, float]:
        """Return (v, w) action for a SocNav structured observation."""
        return self.adapter.plan(observation)


class SocNavBenchSamplingAdapter(SamplingPlannerAdapter):
    """
    Adapter that attempts to delegate to the upstream SocNavBench SamplingPlanner.

    If upstream dependencies are unavailable, it falls back to the lightweight
    SamplingPlannerAdapter behavior.
    """

    def __init__(
        self,
        config: SocNavPlannerConfig | None = None,
        socnav_root: Path | None = None,
        planner_factory: Callable[[], Any] | None = None,
    ):
        """Initialize the adapter and optionally load the upstream planner."""

        super().__init__(config=config)
        self._planner = None
        # Allow passing an already-initialized planner for advanced use.
        if planner_factory is not None:
            self._planner = self._safe_call_factory(planner_factory)
        else:
            self._planner = self._load_upstream_planner(socnav_root)

    @staticmethod
    def _safe_call_factory(factory: Callable[[], Any]) -> Any | None:
        """
        Invoke a user-provided factory defensively.

        Returns:
            Planner instance from the factory or ``None`` on failure.
        """
        try:
            return factory()
        except Exception:  # pragma: no cover - defensive fallback  # noqa: BLE001
            return None

    def _load_upstream_planner(self, socnav_root: Path | None) -> Any | None:
        """
        Best-effort import of SocNavBench SamplingPlanner with defaults.

        Returns:
            SamplingPlanner | None: Upstream planner when dependencies resolve; otherwise ``None``.
        """
        root = socnav_root or Path(__file__).resolve().parents[2] / "output" / "SocNavBench"
        sys.path.append(str(root))
        try:
            import control_pipelines.control_pipeline_v0 as cp  # type: ignore  # noqa: PLC0415
            import objectives.goal_distance as gd  # type: ignore  # noqa: PLC0415
            import params.central_params as central  # type: ignore  # noqa: PLC0415
            import planners.sampling_planner as sp  # type: ignore  # noqa: PLC0415
            from dotmap import DotMap  # type: ignore  # noqa: PLC0415
        except Exception:  # pragma: no cover - optional dependency path  # noqa: BLE001
            return None

        try:
            socnav_params = central.create_socnav_params()
            robot_params = central.create_robot_params()

            # Minimal parameter DotMap inspired by upstream defaults
            p = DotMap()
            p.planner = DotMap()
            p.control_pipeline_params = DotMap()
            p.control_pipeline_params.pipeline = cp.ControlPipelineV0
            p.control_pipeline_params.system_dynamics_params = DotMap(
                system="dubins", dt=robot_params.delta_theta
            )
            p.control_pipeline_params.planning_horizon = 1.0
            p.dt = (
                socnav_params.camera_params.dt if hasattr(socnav_params, "camera_params") else 0.1
            )
            p.planning_horizon = 1
            obj_fn = gd.GoalDistance(p=None)
            return sp.SamplingPlanner(obj_fn=obj_fn, params=p)
        except Exception:  # pragma: no cover - optional dependency path  # noqa: BLE001
            return None

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute a (v, w) command, preferring the upstream SocNavBench planner when available.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        if self._planner is None:
            return super().plan(observation)

        try:
            robot_state = observation["robot"]
            goal_state = observation["goal"]
            pos = robot_state["position"]
            heading = robot_state["heading"][0]
            start_config = self._planner.opt_waypt.__class__.from_pos3([pos[0], pos[1], heading])
            goal = goal_state["current"]
            goal_config = self._planner.opt_waypt.__class__.from_pos3([goal[0], goal[1], 0.0])
            data = self._planner.optimize(start_config=start_config, goal_config=goal_config)
            traj = data.get("trajectory")
            if traj is None:
                return super().plan(observation)
            # Extract first control as heading to waypoint
            next_pos = traj.position_nk2()[0, 0]
            to_next = next_pos - pos
            desired_heading = atan2(to_next[1], to_next[0])
            heading_error = self._wrap_angle(desired_heading - heading)
            angular = float(
                np.clip(
                    self.config.angular_gain * heading_error,
                    -self.config.max_angular_speed,
                    self.config.max_angular_speed,
                ),
            )
            linear = float(
                np.clip(np.linalg.norm(to_next), 0.0, self.config.max_linear_speed),
            )
            return linear, angular
        except Exception:  # pragma: no cover - safety net  # noqa: BLE001
            return super().plan(observation)
