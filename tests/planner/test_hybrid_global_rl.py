# ruff: noqa: D103
"""Tests for route-conditioned learned local planner adapter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from robot_sf.planner.hybrid_global_rl import (
    HybridGlobalRLLocalAdapter,
    HybridGlobalRLLocalConfig,
    WaypointDecision,
)


class _WaypointProvider:
    def __init__(self, waypoint: tuple[float, float] | None):
        self._waypoint = waypoint

    def waypoint(self, observation: dict[str, Any]) -> WaypointDecision:
        return WaypointDecision(
            status="ok" if self._waypoint is not None else "missing",
            waypoint=self._waypoint,
            source="test_provider",
            reason=None if self._waypoint is not None else "test_missing",
            route_geometry={"route_path_cell_count": 2} if self._waypoint is not None else None,
        )


class _LocalPolicy:
    def __init__(self, action: dict[str, float] | None = None):
        self.action = action or {"v": 0.4, "omega": 0.1}
        self.seen: list[dict[str, Any]] = []
        self.reset_seed: int | None = None
        self.closed = False

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        self.seen.append(observation)
        return self.action

    def reset(self, *, seed: int | None = None) -> None:
        self.reset_seed = seed

    def close(self) -> None:
        self.closed = True

    def get_metadata(self) -> dict[str, Any]:
        return {"status": "ok"}


def test_route_waypoint_is_injected_into_nested_goal_current() -> None:
    local_policy = _LocalPolicy()
    adapter = HybridGlobalRLLocalAdapter(
        waypoint_provider=_WaypointProvider((1.5, 0.25)),
        local_policy=local_policy,
    )

    command = adapter.plan({"goal": {"current": [9.0, 0.0]}, "robot": {"heading": 0.0}})

    assert command == (0.4, 0.1)
    assert local_policy.seen[0]["goal"]["current"] == [1.5, 0.25]
    assert local_policy.seen[0]["hybrid_global_rl_final_goal"] == (9.0, 0.0)
    assert adapter.diagnostics()["waypoint"] == [1.5, 0.25]


def test_route_waypoint_handles_array_goal_without_ambiguous_truth_value() -> None:
    local_policy = _LocalPolicy()
    adapter = HybridGlobalRLLocalAdapter(
        waypoint_provider=_WaypointProvider((1.5, 0.25)),
        local_policy=local_policy,
    )

    adapter.plan(
        {
            "goal": {"current": np.asarray([9.0, 0.0], dtype=float)},
            "robot": {"heading": 0.0},
        }
    )

    assert local_policy.seen[0]["goal"]["current"] == [1.5, 0.25]
    assert local_policy.seen[0]["hybrid_global_rl_final_goal"] == (9.0, 0.0)


def test_route_waypoint_initializes_missing_goal_dict() -> None:
    local_policy = _LocalPolicy()
    adapter = HybridGlobalRLLocalAdapter(
        waypoint_provider=_WaypointProvider((1.5, 0.25)),
        local_policy=local_policy,
    )

    adapter.plan({"goal": None, "robot": {"heading": 0.0}})

    assert local_policy.seen[0]["goal"]["current"] == [1.5, 0.25]
    assert "goal_current" not in local_policy.seen[0]


def test_route_waypoint_is_injected_into_flat_goal_current() -> None:
    local_policy = _LocalPolicy()
    adapter = HybridGlobalRLLocalAdapter(
        waypoint_provider=_WaypointProvider((0.75, -0.5)),
        local_policy=local_policy,
    )

    adapter.plan({"goal_current": [3.0, 4.0], "robot_heading": 0.0})

    assert local_policy.seen[0]["goal_current"] == [0.75, -0.5]
    assert local_policy.seen[0]["hybrid_global_rl_final_goal"] == (3.0, 4.0)


def test_missing_waypoint_fails_closed_by_default() -> None:
    adapter = HybridGlobalRLLocalAdapter(
        waypoint_provider=_WaypointProvider(None),
        local_policy=_LocalPolicy(),
    )

    with pytest.raises(RuntimeError, match="route waypoint unavailable"):
        adapter.plan({"goal_current": [3.0, 4.0]})

    diagnostics = adapter.diagnostics()
    assert diagnostics["fallback_status"] == "fail_closed"
    assert diagnostics["waypoint_status"] == "missing"


def test_missing_waypoint_can_use_explicit_goal_fallback() -> None:
    local_policy = _LocalPolicy()
    adapter = HybridGlobalRLLocalAdapter(
        config=HybridGlobalRLLocalConfig(allow_goal_fallback=True),
        waypoint_provider=_WaypointProvider(None),
        local_policy=local_policy,
    )

    assert adapter.plan({"goal_current": [3.0, 4.0]}) == (0.4, 0.1)

    assert local_policy.seen[0]["goal_current"] == [3.0, 4.0]
    assert adapter.diagnostics()["fallback_status"] == "goal_fallback"


def test_world_velocity_action_converts_to_bounded_unicycle() -> None:
    local_policy = _LocalPolicy(action={"vx": 2.0, "vy": 2.0})
    adapter = HybridGlobalRLLocalAdapter(
        config=HybridGlobalRLLocalConfig(max_linear_speed=0.5, max_angular_speed=0.25),
        waypoint_provider=_WaypointProvider((1.0, 1.0)),
        local_policy=local_policy,
    )

    linear, angular = adapter.plan({"goal_current": [3.0, 4.0], "robot_heading": 0.0})

    assert linear == 0.5
    assert angular == 0.25
    assert adapter.diagnostics()["action_conversion_mode"] == "world_velocity_to_unicycle"


def test_reset_and_close_propagate_to_local_policy() -> None:
    local_policy = _LocalPolicy()
    adapter = HybridGlobalRLLocalAdapter(
        waypoint_provider=_WaypointProvider((1.0, 1.0)),
        local_policy=local_policy,
    )

    adapter.reset(seed=17)
    adapter.close()

    assert local_policy.reset_seed == 17
    assert local_policy.closed is True
    assert adapter.diagnostics()["status"] == "reset"


def test_unicycle_action_is_bounded() -> None:
    local_policy = _LocalPolicy(action={"v": 2.0, "omega": -2.0})
    adapter = HybridGlobalRLLocalAdapter(
        config=HybridGlobalRLLocalConfig(max_linear_speed=0.7, max_angular_speed=0.3),
        waypoint_provider=_WaypointProvider((1.0, 1.0)),
        local_policy=local_policy,
    )

    linear, angular = adapter.plan({"goal_current": [3.0, 4.0]})

    assert math.isclose(linear, 0.7)
    assert math.isclose(angular, -0.3)
