"""Registry of reusable training entry points.

This module exposes discoverable dotted paths for the imitation learning
training scripts added by the PPO expert feature. Downstream tooling (CLI
launchers, docs, or VS Code tasks) can import this mapping to avoid hard-coded
paths when presenting available workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPERT_TRAINING_ENTRYPOINT = "scripts.training.train_expert_ppo:main"
DREAMERV3_RLLIB_ENTRYPOINT = "scripts.training.train_dreamerv3_rllib:main"

TRAINING_ENTRYPOINTS: Mapping[str, str] = {
    "train_expert_ppo": EXPERT_TRAINING_ENTRYPOINT,
    "train_dreamerv3_rllib": DREAMERV3_RLLIB_ENTRYPOINT,
}

__all__ = [
    "DREAMERV3_RLLIB_ENTRYPOINT",
    "EXPERT_TRAINING_ENTRYPOINT",
    "TRAINING_ENTRYPOINTS",
]
