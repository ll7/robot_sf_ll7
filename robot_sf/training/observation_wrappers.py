"""Observation helpers shared by training scripts.

Provides utilities to ensure Stable-Baselines3 policies encounter the same flattened
observation space regardless of where the environment is instantiated.
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces as gym_spaces
from gymnasium.wrappers import FlattenObservation
from loguru import logger

__all__ = ["maybe_flatten_env_observations"]


def maybe_flatten_env_observations(env: Any, *, context: str = "training") -> Any:
    """Wrap ``env`` in ``FlattenObservation`` when it exposes a dict space.

    Stable-Baselines3 persists the observation space in its checkpoints. If BC or
    PPO fine-tuning runs with a flattened space, every subsequent consumer must
    recreate the same shape before calling ``PPO.load``. This helper provides a
    single place to apply the wrapper and emit consistent logging.
    """

    if isinstance(getattr(env, "observation_space", None), gym_spaces.Dict):
        logger.info(
            "Applying FlattenObservation wrapper for dict observation space during {}.",
            context,
        )
        return FlattenObservation(env)

    return env
