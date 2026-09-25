"""Repeated seeded crowd and planner episodes must reproduce their traces."""

from __future__ import annotations

import pytest

from tests.metamorphic.support import (
    BASE_MAP,
    EPISODE_SEED,
    assert_trace_byte_identical,
    run_episode,
)
from tests.metamorphic.test_pedestrian_removal import (
    _crossing_scene,
    _run_social_force_episode,
)


def test_same_seed_replays_an_identical_crowd_trace() -> None:
    """Fresh crowd environments on one host agree in values and representation."""
    first = run_episode(BASE_MAP, seed=EPISODE_SEED)
    repeated = run_episode(BASE_MAP, seed=EPISODE_SEED)

    assert first.row_keys == repeated.row_keys
    assert_trace_byte_identical(first, repeated)


def test_same_seed_replays_planner_commands_actions_and_robot_state() -> None:
    """A bounded real RobotEnv run repeats the native planner's closed-loop trace."""
    scene = _crossing_scene(pedestrian_present=True)
    first = _run_social_force_episode(scene, seed=EPISODE_SEED, max_steps=8)
    repeated = _run_social_force_episode(scene, seed=EPISODE_SEED, max_steps=8)

    assert len(first.commands) == 8
    assert len(first.positions) == len(first.commands) + 1
    assert len(first.actions) == len(first.commands)
    assert first.fallback is False and first.fallback_count == 0
    assert repeated.fallback is False and repeated.fallback_count == 0
    assert first == repeated


@pytest.mark.slow
def test_same_seed_replays_a_complete_planner_episode() -> None:
    """A full successful robot episode repeats commands, actions, state and outcome."""
    scene = _crossing_scene(pedestrian_present=False)
    first = _run_social_force_episode(scene, seed=EPISODE_SEED, max_steps=140)
    repeated = _run_social_force_episode(scene, seed=EPISODE_SEED, max_steps=140)

    assert first.success and repeated.success
    assert len(first.commands) >= 100
    assert first.fallback is False and first.fallback_count == 0
    assert repeated.fallback is False and repeated.fallback_count == 0
    assert first == repeated
