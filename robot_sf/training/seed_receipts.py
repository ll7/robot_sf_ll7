"""Worker-owned episode seed receipts, independent of parent-process hooks."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import gymnasium as gym

from robot_sf.common.artifact_paths import get_artifact_root

if TYPE_CHECKING:
    from pathlib import Path


class EpisodeSeedReceiptEnv(gym.Wrapper):
    """Persist the effective reset identity inside the process owning the environment."""

    def __init__(self, env: gym.Env, *, worker_index: int) -> None:
        """Create a worker-specific sink under the configured artifact root."""
        super().__init__(env)
        self.worker_index = worker_index
        self.episode_index = 0
        directory = get_artifact_root() / "training_seed_receipts"
        directory.mkdir(parents=True, exist_ok=True)
        self.receipt_path: Path = directory / f"worker-{worker_index}-pid-{os.getpid()}.jsonl"

    def reset(self, *, seed=None, options=None):
        """Write one receipt only after a successful, explicitly seeded reset.

        Returns:
            The wrapped environment's unchanged observation and info.
        """
        result = self.env.reset(seed=seed, options=options)
        effective_seed = getattr(self.env, "episode_seed", None)
        if isinstance(effective_seed, bool) or not isinstance(effective_seed, int):
            raise ValueError("training seed telemetry requires an explicit effective episode seed")
        if not 100000 <= effective_seed <= 199999:
            raise ValueError(
                "training seed telemetry refuses identities outside the training-only range"
            )
        record = {
            "schema": "robot-sf-training-episode-seed.v1",
            "worker_index": self.worker_index,
            "episode_index": self.episode_index,
            "seed": effective_seed,
            "scenario_id": getattr(self.env, "scenario_id", None),
        }
        with self.receipt_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.episode_index += 1
        return result
