"""Factory/reset seed determinism tests.

Explicit factory and reset seeds should yield identical initial observation arrays.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

SEED = 1234


def _extract_numeric(obs):
    """Extract a deterministic numeric array from reset observation structure.

    Many envs return a tuple/list/dict structure. We attempt to locate a dict with
    'drive_state' key (common in this project) and return that array.
    Fallback: flatten numeric values found; if none, raise to skip.
    """
    if isinstance(obs, dict) and "drive_state" in obs:
        return np.asarray(obs["drive_state"], dtype=float)
    if isinstance(obs, list | tuple):
        for item in obs:
            if isinstance(item, dict) and "drive_state" in item:
                return np.asarray(item["drive_state"], dtype=float)
    # Fallback: try direct conversion
    try:
        return np.asarray(obs, dtype=float)
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Cannot extract numeric observation: {e}") from e


@pytest.mark.parametrize("debug", [False])
def test_seed_determinism(debug):
    """Explicit same-seed factory/reset calls should produce matching reset observations.

    Args:
        debug: Whether to create the environment in visual debug mode.
    """
    env1 = make_robot_env(config=RobotSimulationConfig(), debug=debug, seed=SEED)
    env2 = make_robot_env(config=RobotSimulationConfig(), debug=debug, seed=SEED)
    try:
        obs1 = env1.reset(seed=SEED)
        obs2 = env2.reset(seed=SEED)
        arr1 = _extract_numeric(obs1)
        arr2 = _extract_numeric(obs2)
        assert arr1.shape == arr2.shape

        # Focus on the robot/pedestrian rows and avoid coupling the test to future
        # observation extensions appended after the stable state rows.
        rows_to_compare = min(2, arr1.shape[0])
        np.testing.assert_allclose(
            arr1[:rows_to_compare],
            arr2[:rows_to_compare],
            rtol=1e-6,
            atol=1e-8,
        )
    finally:
        env1.close()
        env2.close()


def test_seeded_reset_does_not_leak_global_rng_state():
    """Seeded reset should not advance caller-owned Python or NumPy RNG state."""
    env = make_robot_env(config=RobotSimulationConfig(), debug=False, seed=SEED)
    outer_random_state = random.getstate()
    outer_numpy_state = np.random.get_state()
    try:
        random.seed(9876)
        np.random.seed(9876)
        expected_random_state = random.getstate()
        expected_numpy_state = np.random.get_state()

        env.reset(seed=SEED)
        actual = (random.random(), np.random.random())

        random.setstate(expected_random_state)
        np.random.set_state(expected_numpy_state)
        expected = (random.random(), np.random.random())

        assert actual == expected
    finally:
        random.setstate(outer_random_state)
        np.random.set_state(outer_numpy_state)
        env.close()
