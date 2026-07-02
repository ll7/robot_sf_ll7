"""Route-conditioned learned local planner adapter.

This module implements the first issue #4015 slice: a global route waypoint
provider rewrites the local goal seen by an existing learned local policy. It
does not add a new global planner algorithm or train a new RL policy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import atan2, cos, sin
from typing import Any, Protocol

import numpy as np

from robot_sf.baselines.ppo import PPOPlanner
from robot_sf.baselines.sac import SACPlanner
from robot_sf.planner.grid_route import GridRoutePlannerAdapter, build_grid_route_config
from robot_sf.planner.risk_dwa import _wrap_angle


@dataclass(frozen=True)
class HybridGlobalRLLocalConfig:
    """Configuration for route-conditioned learned local policy execution."""

    local_policy_algo: str = "sac"
    local_policy_config: dict[str, Any] = field(default_factory=dict)
    waypoint_provider: str = "grid_route"
    grid_route: dict[str, Any] = field(default_factory=dict)
    allow_goal_fallback: bool = False
    fail_closed_on_missing_waypoint: bool = True
    preserve_final_goal: bool = True
    waypoint_max_distance_from_robot: float = 3.0
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0


@dataclass(frozen=True)
class WaypointDecision:
    """Waypoint selection result used to condition the learned local policy."""

    status: str
    waypoint: tuple[float, float] | None
    source: str
    reason: str | None = None
    route_geometry: dict[str, Any] | None = None


class WaypointProvider(Protocol):
    """Provider interface for route-conditioning waypoints."""

    def waypoint(self, observation: dict[str, Any]) -> WaypointDecision:
        """Return the short-horizon waypoint for ``observation``."""


class GridRouteWaypointProvider:
    """Waypoint provider backed by the existing occupancy-grid route planner."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the provider from grid-route configuration."""

        self.adapter = GridRoutePlannerAdapter(config=build_grid_route_config(config))

    def waypoint(self, observation: dict[str, Any]) -> WaypointDecision:
        """Return grid-route waypoint diagnostics without issuing a command."""

        route_geometry = self.adapter.route_geometry(observation)
        waypoint = self.adapter.route_waypoint(observation)
        if waypoint is None:
            return WaypointDecision(
                status="missing",
                waypoint=None,
                source="grid_route",
                reason="route_waypoint_unavailable",
                route_geometry=route_geometry,
            )
        waypoint_arr = np.asarray(waypoint, dtype=float).reshape(-1)
        if waypoint_arr.size < 2 or not np.all(np.isfinite(waypoint_arr[:2])):
            return WaypointDecision(
                status="missing",
                waypoint=None,
                source="grid_route",
                reason="route_waypoint_nonfinite",
                route_geometry=route_geometry,
            )
        return WaypointDecision(
            status="ok",
            waypoint=(float(waypoint_arr[0]), float(waypoint_arr[1])),
            source="grid_route",
            route_geometry=route_geometry,
        )


class HybridGlobalRLLocalAdapter:
    """Adapter that feeds route-conditioned observations to an RL local policy."""

    def __init__(
        self,
        *,
        config: HybridGlobalRLLocalConfig | None = None,
        waypoint_provider: WaypointProvider | None = None,
        local_policy: Any | None = None,
    ):
        """Initialize the route-conditioned local planner adapter."""

        self.config = config or HybridGlobalRLLocalConfig()
        if self.config.waypoint_provider != "grid_route" and waypoint_provider is None:
            raise ValueError("Only waypoint_provider='grid_route' is supported by issue #4015 PR 1")
        self.waypoint_provider = waypoint_provider or GridRouteWaypointProvider(
            self.config.grid_route
        )
        self.local_policy = local_policy or self._build_local_policy()
        self._last_diagnostics: dict[str, Any] = {"status": "not_started"}

    def _build_local_policy(self) -> Any:
        """Instantiate the configured learned local policy wrapper.

        Returns:
            Existing SAC or PPO local policy wrapper.
        """

        algo = self.config.local_policy_algo.strip().lower()
        if algo == "sac":
            return SACPlanner(self.config.local_policy_config, seed=None)
        if algo == "ppo":
            return PPOPlanner(self.config.local_policy_config, seed=None)
        raise ValueError(f"Unsupported local_policy_algo {self.config.local_policy_algo!r}")

    def reset(self, *, seed: int | None = None) -> None:
        """Reset diagnostics and propagate reset to the local policy."""

        self._last_diagnostics = {"status": "reset"}
        reset = getattr(self.local_policy, "reset", None)
        if callable(reset):
            reset(seed=seed)

    def close(self) -> None:
        """Close the wrapped local policy if it exposes a close hook."""

        close = getattr(self.local_policy, "close", None)
        if callable(close):
            close()

    def bind_env(self, env: Any) -> None:
        """Bind benchmark environment to the wrapped policy when supported."""

        bind_env = getattr(self.local_policy, "bind_env", None)
        if callable(bind_env):
            bind_env(env)

    def diagnostics(self) -> dict[str, Any]:
        """Return last route-conditioning decision and local policy metadata."""

        diagnostics = dict(self._last_diagnostics)
        get_metadata = getattr(self.local_policy, "get_metadata", None)
        if callable(get_metadata):
            diagnostics["local_policy_metadata"] = get_metadata()
        return diagnostics

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Run the route-conditioned learned local planner.

        Returns:
            Bounded unicycle command as ``(linear_velocity, angular_velocity)``.
        """

        decision = self.waypoint_provider.waypoint(observation)
        if decision.waypoint is None:
            if not self.config.allow_goal_fallback and self.config.fail_closed_on_missing_waypoint:
                self._last_diagnostics = self._diagnostics_payload(
                    decision=decision,
                    conditioned=False,
                    action_conversion_mode="not_run",
                    fallback_status="fail_closed",
                )
                raise RuntimeError("hybrid_global_rl route waypoint unavailable")
            conditioned_observation = deepcopy(observation)
            fallback_status = "goal_fallback"
            waypoint = self._extract_final_goal(conditioned_observation)
        else:
            waypoint = decision.waypoint
            conditioned_observation = self._inject_waypoint_goal(observation, waypoint)
            fallback_status = "none"

        action = self.local_policy.step(conditioned_observation)
        linear, angular, conversion_mode = self._action_to_unicycle(action, conditioned_observation)
        self._last_diagnostics = self._diagnostics_payload(
            decision=decision,
            conditioned=decision.waypoint is not None,
            action_conversion_mode=conversion_mode,
            fallback_status=fallback_status,
            injected_waypoint=waypoint,
        )
        return linear, angular

    def _diagnostics_payload(
        self,
        *,
        decision: WaypointDecision,
        conditioned: bool,
        action_conversion_mode: str,
        fallback_status: str,
        injected_waypoint: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Build JSON-ready diagnostics for the last planner decision.

        Returns:
            Structured metadata payload suitable for episode diagnostics.
        """

        return {
            "status": "ok" if conditioned or fallback_status == "goal_fallback" else "failed",
            "waypoint_source": decision.source,
            "waypoint_status": decision.status,
            "waypoint_reason": decision.reason,
            "waypoint": list(injected_waypoint) if injected_waypoint is not None else None,
            "route_geometry": decision.route_geometry,
            "fallback_status": fallback_status,
            "action_conversion_mode": action_conversion_mode,
            "claim_boundary": "diagnostic-only route-conditioned local-policy adapter",
        }

    def _inject_waypoint_goal(
        self, observation: dict[str, Any], waypoint: tuple[float, float]
    ) -> dict[str, Any]:
        """Return observation copy with local goal rewritten to ``waypoint``."""

        conditioned = deepcopy(observation)
        waypoint_list = [float(waypoint[0]), float(waypoint[1])]
        if self.config.preserve_final_goal:
            conditioned.setdefault(
                "hybrid_global_rl_final_goal", self._extract_final_goal(observation)
            )

        goal = conditioned.get("goal")
        if goal is None:
            goal = {}
            conditioned["goal"] = goal
        if isinstance(goal, dict):
            goal["current"] = list(waypoint_list)
        if "goal_current" in conditioned:
            conditioned["goal_current"] = list(waypoint_list)
        if not isinstance(goal, dict) and "goal_current" not in conditioned:
            conditioned["goal_current"] = list(waypoint_list)
        return conditioned

    @staticmethod
    def _extract_final_goal(observation: dict[str, Any]) -> tuple[float, float] | None:
        """Extract the current final/local goal from known observation shapes.

        Returns:
            Two-dimensional goal coordinates when present and finite.
        """

        goal = observation.get("goal")
        candidate: Any | None = None
        if isinstance(goal, dict):
            candidate = goal.get("current")
            if candidate is None:
                candidate = goal.get("next")
        if candidate is None:
            candidate = observation.get("goal_current")
        if candidate is None:
            return None
        arr = np.asarray(candidate, dtype=float).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            return None
        return (float(arr[0]), float(arr[1]))

    def _action_to_unicycle(
        self, action: dict[str, Any], observation: dict[str, Any]
    ) -> tuple[float, float, str]:
        """Convert wrapped local-policy action to bounded unicycle command.

        Returns:
            Linear velocity, angular velocity, and conversion-mode label.
        """

        if "v" in action or "omega" in action:
            return (
                self._clip(float(action.get("v", 0.0)), self.config.max_linear_speed),
                self._clip(float(action.get("omega", 0.0)), self.config.max_angular_speed),
                "unicycle",
            )

        if "vx" in action or "vy" in action:
            vx = float(action.get("vx", 0.0))
            vy = float(action.get("vy", 0.0))
            heading = self._extract_heading(observation)
            forward = vx * cos(heading) + vy * sin(heading)
            desired_heading = atan2(vy, vx) if abs(vx) > 1e-12 or abs(vy) > 1e-12 else heading
            angular = _wrap_angle(desired_heading - heading)
            return (
                self._clip(forward, self.config.max_linear_speed),
                self._clip(angular, self.config.max_angular_speed),
                "world_velocity_to_unicycle",
            )

        raise ValueError("Local policy action must contain v/omega or vx/vy")

    @staticmethod
    def _extract_heading(observation: dict[str, Any]) -> float:
        """Extract robot heading from common observation shapes.

        Returns:
            Robot heading in radians, defaulting to zero when absent.
        """

        robot = observation.get("robot")
        if isinstance(robot, dict) and "heading" in robot:
            return float(robot["heading"])
        if "robot_heading" in observation:
            return float(observation["robot_heading"])
        return 0.0

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        """Clip scalar symmetrically by non-negative ``limit``.

        Returns:
            Clipped scalar.
        """

        return float(np.clip(value, -abs(float(limit)), abs(float(limit))))


def build_hybrid_global_rl_config(data: dict[str, Any] | None) -> HybridGlobalRLLocalConfig:
    """Build hybrid global/RL-local adapter config from a plain mapping.

    Returns:
        Typed adapter configuration.
    """

    payload = dict(data or {})
    return HybridGlobalRLLocalConfig(
        local_policy_algo=str(payload.get("local_policy_algo", "sac")),
        local_policy_config=dict(payload.get("local_policy_config", {})),
        waypoint_provider=str(payload.get("waypoint_provider", "grid_route")),
        grid_route=dict(payload.get("grid_route", {})),
        allow_goal_fallback=bool(payload.get("allow_goal_fallback", False)),
        fail_closed_on_missing_waypoint=bool(payload.get("fail_closed_on_missing_waypoint", True)),
        preserve_final_goal=bool(payload.get("preserve_final_goal", True)),
        waypoint_max_distance_from_robot=float(
            payload.get("waypoint_max_distance_from_robot", 3.0)
        ),
        max_linear_speed=float(payload.get("max_linear_speed", 1.0)),
        max_angular_speed=float(payload.get("max_angular_speed", 1.0)),
    )
