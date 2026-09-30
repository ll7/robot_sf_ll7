"""Run Gymnasium's unchanged checks using development episode seeds.

Gymnasium hard-codes 123 and 456 in its reset determinism check and defaults
its step determinism check to 123. The adapter changes only those checker inputs;
observations, RNG state, rewards, terminations, and info assertions stay intact.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.utils.env_checker import check_env


class _DevelopmentSeedCheckerEnv(gym.Wrapper):
    """Translate the upstream checker's two fixed seeds at this explicit call site."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Forward checker resets to the same underlying environment on dev seeds."""
        seed = {123: 1013, 456: 1014}.get(seed, seed)  # seed-holdout: synthetic-fixture
        return self.env.reset(seed=seed, options=options)


def check_env_on_dev_seeds(env: gym.Env, **kwargs: Any) -> None:
    """Keep all upstream checks while isolating their hard-coded episode seeds."""
    check_env(_DevelopmentSeedCheckerEnv(env), **kwargs)
