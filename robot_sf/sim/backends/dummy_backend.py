"""Dummy simulator backend for testing and smoke tests.

Provides a minimal, deterministic simulator that returns constant positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.nav.map_config import MapDefinition


class DummySimulator:
    """Minimal simulator that returns constant positions for testing."""

    def __init__(self, seed: int = 0):
        """TODO docstring. Document this function.

        Args:
            seed: TODO docstring.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

    def reset_state(self):
        """Reset simulator to initial state."""
        self.timestep = 0
        self.rng = np.random.default_rng(self.seed)

    def step_once(self, _actions):
        """Advance one timestep (no-op for dummy)."""
        self.timestep += 1

    @property
    def robot_poses(self):
        """Return constant robot poses."""
        return [np.array([[1.0, 1.0], [0.0]])]  # position + orientation

    @property
    def ped_pos(self):
        """Return constant pedestrian positions."""
        return np.array([[2.0, 2.0], [3.0, 3.0]])

    @property
    def goal_pos(self):
        """Return constant goal positions."""
        return [np.array([5.0, 5.0])]

    @property
    def robots(self):
        """Return mock robot objects."""
        return [_MockRobot()]


class _MockRobot:
    """Mock robot for dummy simulator."""

    def parse_action(self, action):
        """Pass-through action parsing.

        Returns:
            The action parameter unchanged.
        """
        return action


def dummy_factory(env_config: EnvSettings, _map_def: MapDefinition, _peds: bool) -> DummySimulator:
    """Create a dummy simulator instance for testing.

    Parameters
    ----------
    env_config : EnvSettings
        Environment configuration (seed extracted if available)
    _map_def : MapDefinition
        Map definition (unused in dummy)
    _peds : bool
        Pedestrian interaction flag (unused in dummy)

    Returns
    -------
    DummySimulator
        A minimal simulator instance returning static positions and zero forces for smoke tests.
    """
    seed = getattr(env_config, "seed", 0)
    return DummySimulator(seed=seed)
