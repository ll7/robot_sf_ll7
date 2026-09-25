"""Repeated seeded crowd and planner episodes must reproduce their traces.

Every replay case samples something from its seed: the robot start and goal come
from zones with real area, the crowd is drawn from a route and a crowded zone,
and the baseline planner adds seeded command noise. A negative control with a
different seed must change the trace, so a seed that is silently ignored (a fixed
or unseeded RNG) cannot pass as determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner import socnav
from tests.metamorphic.planner_arms import (
    RELEASE_ARMS,
    SAMPLED_PED_DENSITY,
    run_arm_episode,
    sampled_scene,
)
from tests.metamorphic.support import (
    BASE_MAP,
    EPISODE_SEED,
    assert_trace_byte_identical,
    run_episode,
)
from tests.metamorphic.test_pedestrian_removal import _run_social_force_episode

_OTHER_SEED = EPISODE_SEED + 1
_NOISE_STD = 0.1


def test_same_seed_replays_an_identical_crowd_trace() -> None:
    """Fresh crowd environments on one host agree in values and representation."""
    first = run_episode(BASE_MAP, seed=EPISODE_SEED)
    repeated = run_episode(BASE_MAP, seed=EPISODE_SEED)

    assert first.row_keys == repeated.row_keys
    assert_trace_byte_identical(first, repeated)


def _baseline(seed: int, *, planner_seed: int | None = None, max_steps: int = 30):
    return _run_social_force_episode(
        sampled_scene(),
        seed=seed,
        planner_seed=planner_seed,
        max_steps=max_steps,
        ped_density=SAMPLED_PED_DENSITY,
        noise_std=_NOISE_STD,
    )


def test_same_seed_replays_sampled_scene_with_noisy_baseline_planner() -> None:
    """Sampled start, sampled crowd and seeded planner noise replay exactly."""
    first = _baseline(EPISODE_SEED)
    repeated = _baseline(EPISODE_SEED)

    assert len(first.pedestrian_positions[0]) >= 3, "the crowd must be sampled"
    assert len(first.commands) == 30
    assert first.fallback is False and first.fallback_count == 0
    assert first == repeated


def test_a_different_seed_changes_the_sampled_scene_and_planner_noise() -> None:
    """Negative controls: the env seed moves the scene, the planner seed moves the noise."""
    base = _baseline(EPISODE_SEED)
    other_env = _baseline(_OTHER_SEED, planner_seed=EPISODE_SEED)
    other_planner = _baseline(EPISODE_SEED, planner_seed=_OTHER_SEED)

    assert base.positions[0] != other_env.positions[0], "env seed must move the robot start"
    assert base.pedestrian_positions[0] != other_env.pedestrian_positions[0], (
        "env seed must move the sampled crowd"
    )
    assert base.positions[0] == other_planner.positions[0]
    assert base.pedestrian_positions[0] == other_planner.pedestrian_positions[0]
    assert base.commands != other_planner.commands, "planner seed must move the command noise"


@pytest.mark.parametrize("arm", RELEASE_ARMS)
def test_release_arm_replays_a_sampled_episode_and_depends_on_its_seed(arm: str) -> None:
    """Release arms repeat commands, actions, poses and crowd for the same seed only."""
    if arm == "orca" and socnav.rvo2 is None:
        pytest.skip("rvo2 is required for the native ORCA release arm")

    def episode(seed: int):
        return run_arm_episode(
            arm, sampled_scene(), seed=seed, max_steps=40, ped_density=SAMPLED_PED_DENSITY
        )

    first = episode(EPISODE_SEED)
    repeated = episode(EPISODE_SEED)
    other = episode(_OTHER_SEED)

    assert first.status == "ok"
    assert len(first.pedestrian_positions[0]) >= 3, "the crowd must be sampled"
    assert first == repeated
    assert first.poses[0] != other.poses[0]
    assert first.pedestrian_positions[0] != other.pedestrian_positions[0]
    assert not np.array_equal(np.asarray(first.commands[:5]), np.asarray(other.commands[:5])), (
        "a different seed must change the closed-loop commands"
    )
