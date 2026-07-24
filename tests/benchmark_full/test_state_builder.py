"""Unit tests for state_builder (build_minimal_states / iter_states).

Verifies correct timestep indexing, robot pose formatting, and iterator alias
parity using a stub ReplayEpisode with ReplayStep instances.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.replay import ReplayEpisode, ReplayStep
from robot_sf.benchmark.full_classic.state_builder import (
    _MinimalState,
    build_minimal_states,
    iter_states,
)


def _make_episode(*pose_tuples: tuple[float, float, float, float]) -> ReplayEpisode:
    steps = [ReplayStep(t=t, x=x, y=y, heading=h) for t, x, y, h in pose_tuples]
    return ReplayEpisode(episode_id="test", scenario_id="test", steps=steps)


def test_build_minimal_states_empty_episode():
    ep = _make_episode()
    states = list(build_minimal_states(ep))
    assert states == []


def test_build_minimal_states_correct_robot_pose():
    ep = _make_episode((0.0, 1.0, 2.0, 0.5), (1.0, 3.0, 4.0, 1.0))
    states = list(build_minimal_states(ep))
    assert len(states) == 2
    assert states[0].robot_pose == ((1.0, 2.0), 0.5)
    assert states[1].robot_pose == ((3.0, 4.0), 1.0)


def test_build_minimal_states_correct_timestep_index():
    ep = _make_episode((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.1), (2.0, 2.0, 2.0, 0.2))
    states = list(build_minimal_states(ep))
    assert [s.timestep for s in states] == [0, 1, 2]


def test_build_minimal_states_returns_minimal_state_type():
    ep = _make_episode((0.0, 5.0, 5.0, 1.57))
    states = list(build_minimal_states(ep))
    assert len(states) == 1
    assert isinstance(states[0], _MinimalState)


def test_iter_states_alias_parity():
    ep = _make_episode(
        (0.0, 10.0, 20.0, 0.0),
        (0.1, 11.0, 21.0, 0.1),
        (0.2, 12.0, 22.0, 0.2),
    )
    expected = list(build_minimal_states(ep))
    actual = list(iter_states(ep))
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a.timestep == e.timestep
        assert a.robot_pose == e.robot_pose


def test_iter_states_empty_parity():
    ep = _make_episode()
    assert list(iter_states(ep)) == []


def test_build_minimal_states_does_not_mutate_episode():
    poses = [(0.0, 1.0, 2.0, 0.5), (1.0, 3.0, 4.0, 1.0)]
    ep = _make_episode(*poses)
    original_len = len(ep.steps)
    list(build_minimal_states(ep))
    assert len(ep.steps) == original_len


def test_build_minimal_states_negative_coordinates():
    ep = _make_episode((0.0, -1.0, -2.0, -0.5))
    states = list(build_minimal_states(ep))
    assert len(states) == 1
    assert states[0].robot_pose == ((-1.0, -2.0), -0.5)
