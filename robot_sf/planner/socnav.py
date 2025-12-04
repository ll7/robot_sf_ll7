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


class SocNavBenchComplexPolicy(SocNavPlannerPolicy):
    """
    Policy that prefers the upstream SocNavBench SamplingPlanner when available.

    Falls back to the lightweight adapter when upstream dependencies are missing.
    """

    def __init__(
        self,
        socnav_root: Path | None = None,
        adapter_config: SocNavPlannerConfig | None = None,
    ):
        """Initialize the policy, preferring the upstream SocNavBench planner when present."""

        adapter = SocNavBenchSamplingAdapter(config=adapter_config, socnav_root=socnav_root)
        super().__init__(adapter=adapter)


class SocialForcePlannerAdapter(SamplingPlannerAdapter):
    """Heuristic social-force style planner: goal attraction plus pedestrian repulsion."""

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using goal attraction and inverse-square pedestrian repulsion.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state = observation["robot"]
        goal_state = observation["goal"]
        ped_state = observation["pedestrians"]
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(robot_state["heading"][0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions = np.asarray(ped_state["positions"], dtype=float)
        ped_count = int(ped_state["count"][0])
        ped_positions = ped_positions[:ped_count]
        repulse = np.zeros(2, dtype=float)
        for ped in ped_positions:
            delta = robot_pos - ped
            dist = np.linalg.norm(delta) + 1e-6
            repulse += delta / dist**2

        combined = goal_vec + 0.8 * repulse
        if np.linalg.norm(combined) > 1e-6:
            combined = combined / np.linalg.norm(combined)

        desired_heading = atan2(combined[1], combined[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            ),
        )
        linear = float(
            np.clip(
                np.linalg.norm(to_goal),
                0.0,
                self.config.max_linear_speed * max(0.0, 1.0 - abs(heading_error) / pi),
            ),
        )
        return linear, angular


class ORCAPlannerAdapter(SamplingPlannerAdapter):
    """Simplified ORCA-inspired adapter using reciprocal-style avoidance."""

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using goal direction plus reciprocal-style avoidance.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state = observation["robot"]
        goal_state = observation["goal"]
        ped_state = observation["pedestrians"]
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(robot_state["heading"][0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions = np.asarray(ped_state["positions"], dtype=float)
        ped_count: int = int(ped_state["count"][0])
        ped_positions = ped_positions[:ped_count]

        avoidance = np.zeros(2, dtype=float)
        for ped in ped_positions:
            delta = ped - robot_pos
            dist = np.linalg.norm(delta) + 1e-6
            if dist < 5.0:
                avoidance -= delta / dist

        combined = goal_vec + 1.2 * avoidance
        if np.linalg.norm(combined) > 1e-6:
            combined = combined / np.linalg.norm(combined)

        desired_heading = atan2(combined[1], combined[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                1.5 * self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            ),
        )
        linear = float(
            np.clip(
                np.linalg.norm(to_goal),
                0.0,
                self.config.max_linear_speed * max(0.0, 1.0 - abs(heading_error) / pi),
            ),
        )
        return linear, angular


class SACADRLPlannerAdapter(SamplingPlannerAdapter):
    """Heuristic SA-CADRL-style adapter using nearest pedestrians to bias heading."""

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using nearest pedestrian bias similar to SA-CADRL heuristics.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state = observation["robot"]
        goal_state = observation["goal"]
        ped_state = observation["pedestrians"]
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(robot_state["heading"][0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions = np.asarray(ped_state["positions"], dtype=float)
        ped_count = int(ped_state["count"][0])
        ped_positions = ped_positions[:ped_count]
        if ped_positions.shape[0] > 0:
            dists = np.linalg.norm(ped_positions - robot_pos, axis=1)
            nearest_idx = np.argsort(dists)[:3]
            bias = np.zeros(2, dtype=float)
            for idx in nearest_idx:
                delta = robot_pos - ped_positions[idx]
                dist = dists[idx] + 1e-6
                bias += delta / dist**1.5
            combined = goal_vec + 0.6 * bias
        else:
            combined = goal_vec

        if np.linalg.norm(combined) > 1e-6:
            combined = combined / np.linalg.norm(combined)

        desired_heading = atan2(combined[1], combined[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            ),
        )
        linear = float(
            np.clip(
                np.linalg.norm(to_goal),
                0.0,
                self.config.max_linear_speed * max(0.0, 1.0 - abs(heading_error) / pi),
            ),
        )
        return linear, angular


def make_social_force_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for social-force-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SocialForcePlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SocialForcePlannerAdapter(config=config))


def make_orca_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for ORCA-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping ORCAPlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=ORCAPlannerAdapter(config=config))


def make_sacadrl_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for SA-CADRL-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SACADRLPlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SACADRLPlannerAdapter(config=config))


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
        root_str = str(Path(root).resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
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
            # NOTE: upstream returns a trajectory and controller matrices; for now we
            # consume only the immediate waypoint to preserve the (v, w) interface and
            # avoid binding to controller specifics. This keeps the adapter lightweight
            # while still aligning heading toward the planned path.
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
