"""Tests for provenance-bound route-progress recording in expert datasets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.nav.navigation import RouteNavigator
from scripts.training.collect_expert_trajectories import (
    _record_episode,
    _remaining_route_length,
)


def test_remaining_route_length_follows_active_route_polyline() -> None:
    """The helper measures current pose to remaining waypoints, not goal distance."""
    navigator = RouteNavigator(
        waypoints=[(3.0, 0.0), (3.0, 4.0)],
        proximity_threshold=0.1,
        pos=(0.0, 0.0),
    )
    env = SimpleNamespace(simulator=SimpleNamespace(robot_navs=[navigator]))

    assert _remaining_route_length(env) == pytest.approx(7.0)

    navigator.pos = (3.0, 1.0)
    navigator.waypoint_id = 1
    assert _remaining_route_length(env) == pytest.approx(3.0)


def test_remaining_route_length_returns_zero_at_destination() -> None:
    """A reached final waypoint has no remaining route length."""
    navigator = RouteNavigator(
        waypoints=[(1.0, 0.0)],
        proximity_threshold=0.2,
        pos=(1.1, 0.0),
    )
    env = SimpleNamespace(simulator=SimpleNamespace(robot_navs=[navigator]))

    assert _remaining_route_length(env) == 0.0


def test_remaining_route_length_fails_closed_without_a_route() -> None:
    """Missing route provenance must not fall back to a proxy signal."""
    navigator = RouteNavigator(pos=(0.0, 0.0))
    env = SimpleNamespace(simulator=SimpleNamespace(robot_navs=[navigator]))

    with pytest.raises(RuntimeError, match="active robot route is empty"):
        _remaining_route_length(env)


def test_record_episode_emits_one_route_length_per_action_boundary() -> None:
    """Collected route lengths contain the initial boundary plus each action result."""

    navigator = RouteNavigator(
        waypoints=[(3.0, 0.0), (3.0, 4.0)],
        proximity_threshold=0.1,
        pos=(0.0, 0.0),
    )

    class _Env:
        action_space = SimpleNamespace(shape=(2,))

        def __init__(self) -> None:
            self.simulator = SimpleNamespace(robot_navs=[navigator])
            self.state = SimpleNamespace(max_sim_steps=3, nav=SimpleNamespace(pos=(0.0, 0.0)))
            self._stepped = False

        def reset(self):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            del action
            navigator.update_position((1.0, 0.0))
            self.state.nav.pos = navigator.pos
            self._stepped = True
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    record = _record_episode(_Env(), policy=None, dry_run=True)

    assert len(record["actions"]) == 1
    assert len(record["remaining_route_length"]) == 2
    assert record["remaining_route_length"] == pytest.approx([7.0, 6.0])
