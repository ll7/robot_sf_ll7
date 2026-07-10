"""Tests for decoupled pedestrian desired speed (issue #4972).

Covers :class:`pysocialforce.scene.PedState` decoupling of the goal-driving
speed (``max_speeds``) from the spawn speed, plus the truncated-normal sampler.
"""

import numpy as np
import pytest
from pysocialforce.config import SceneConfig
from pysocialforce.scene import PedState, sample_truncated_normal_speeds


def _state(num_peds: int, speed: float = 0.5) -> np.ndarray:
    """Build a pedestrian state matrix with a fixed spawn speed heading +x."""
    state = np.zeros((num_peds, 7))
    state[:, 2] = speed  # vx = spawn speed
    state[:, 4] = 10.0  # goal x far away so peds keep moving
    return state


def test_legacy_default_couples_max_speed_to_spawn_speed():
    """Without desired-speed config, max_speeds = multiplier * initial_speed."""
    peds = PedState(_state(3, speed=0.5), [], SceneConfig())
    # Default multiplier is 1.3; spawn speed 0.5 -> 0.65 m/s (the slow regime).
    np.testing.assert_allclose(peds.max_speeds, np.full(3, 0.65))


def test_desired_speed_mean_decouples_from_spawn_speed():
    """Configured desired_speed_mean overrides the spawn-coupled derivation."""
    config = SceneConfig(desired_speed_mean=1.3, desired_speed_std=0.0, desired_speed_seed=7)
    peds = PedState(_state(4, speed=0.5), [], config)
    # Spawn speed is 0.5, but desired speed is decoupled to 1.3 (std=0 -> deterministic).
    np.testing.assert_allclose(peds.max_speeds, np.full(4, 1.3))
    # initial_speeds still reflect the spawn velocity (decoupling is one-way).
    np.testing.assert_allclose(peds.initial_speeds, np.full(4, 0.5))


def test_desired_speed_sampling_is_deterministic_with_seed():
    """Same seed must reproduce the sampled desired-speed distribution."""
    cfg = lambda: SceneConfig(  # noqa: E731
        desired_speed_mean=1.3, desired_speed_std=0.2, desired_speed_seed=42
    )
    peds_a = PedState(_state(50), [], cfg())
    peds_b = PedState(_state(50), [], cfg())
    np.testing.assert_array_equal(peds_a.max_speeds, peds_b.max_speeds)


def test_desired_speed_distribution_mean_near_configured():
    """The sampled distribution should center near the configured mean."""
    config = SceneConfig(desired_speed_mean=1.3, desired_speed_std=0.2, desired_speed_seed=1)
    peds = PedState(_state(2000), [], config)
    assert abs(float(np.mean(peds.max_speeds)) - 1.3) < 0.05
    # Truncation keeps speeds non-negative and below the configured high bound.
    assert np.all(peds.max_speeds >= 0.0)
    assert np.all(peds.max_speeds <= config.desired_speed_high)


def test_desired_speed_persists_across_integration_steps():
    """Explicit desired speeds must survive repeated _update_state calls (step())."""
    config = SceneConfig(desired_speed_mean=1.3, desired_speed_std=0.0, desired_speed_seed=0)
    peds = PedState(_state(2, speed=0.5), [], config)
    # Simulate a state refresh like PedState.step does via the state setter.
    new_state = peds.state.copy()
    new_state[:, 2] = 0.9  # velocity changes during stepping
    peds.state = new_state
    np.testing.assert_allclose(peds.max_speeds, np.full(2, 1.3))


def test_assign_desired_speeds_overrides_and_validates():
    """assign_desired_speeds sets max_speeds and rejects bad inputs."""
    peds = PedState(_state(3), [], SceneConfig())
    peds.assign_desired_speeds(np.array([0.8, 1.0, 1.2]))
    np.testing.assert_allclose(peds.max_speeds, [0.8, 1.0, 1.2])

    with pytest.raises(ValueError, match="one entry per pedestrian"):
        peds.assign_desired_speeds(np.array([0.8, 1.0]))  # wrong length

    with pytest.raises(ValueError, match="non-negative"):
        peds.assign_desired_speeds(np.array([-0.1, 1.0, 1.2]))  # negative


def test_clear_desired_speeds_restores_legacy_derivation():
    """clear_desired_speeds returns to multiplier * initial_speed."""
    peds = PedState(_state(3, speed=0.5), [], SceneConfig())
    peds.assign_desired_speeds(np.array([1.3, 1.3, 1.3]))
    np.testing.assert_allclose(peds.max_speeds, 1.3)
    peds.clear_desired_speeds()
    np.testing.assert_allclose(peds.max_speeds, np.full(3, 0.65))


def test_sample_truncated_normal_speeds_clips_and_handles_zero():
    """The sampler clips to bounds and returns empty for zero pedestrians."""
    speeds = sample_truncated_normal_speeds(1000, mean=1.0, std=0.3, high=1.2, seed=3)
    assert speeds.shape == (1000,)
    assert np.all(speeds >= 0.0)
    assert np.all(speeds <= 1.2)
    assert sample_truncated_normal_speeds(0, mean=1.0, std=0.2, high=3.0).shape == (0,)
    # std=0 collapses to the mean (clipped to the bound).
    np.testing.assert_allclose(
        sample_truncated_normal_speeds(5, mean=0.7, std=0.0, high=3.0, seed=None), 0.7
    )
