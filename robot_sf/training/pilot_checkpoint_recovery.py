"""Validate checkpoint lineage and finite episode-stream offsets for opt-in pilots."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np


def remaining_targets(checkpoint_steps: int) -> tuple[int, ...]:
    """Return original rollout-rounded checkpoints without repeating completed chunks."""
    targets = tuple(1_003_520 * epoch for epoch in range(1, 6))
    if checkpoint_steps not in targets:
        raise ValueError("recovery requires an admitted completed checkpoint")
    return tuple(step for step in targets if step > checkpoint_steps)


def receipt_offsets(directory: Path, *, policy: str, master: int) -> dict[int, int]:
    """Verify all successful resets and skip every consumed seed on continuation.

    Counts include successful resets after the saved model, so discarded partial
    training never causes seed reuse. New environment and optimiser RNG states
    are not a bit-identical continuation of an uninterrupted run.
    Returns:
        The next per-worker episode ordinal.
    """
    records: dict[int, list[dict]] = {}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            worker = row["worker_index"]
            if isinstance(worker, bool) or not isinstance(worker, int) or not 0 <= worker < 20:
                raise ValueError("invalid recovery worker identity")
            records.setdefault(worker, []).append(row)
    if set(records) != set(range(20)):
        raise ValueError("recovery requires all workers")
    if policy not in {"A", "B"}:
        raise ValueError("unknown recovery policy")
    permutation = np.random.default_rng(master).permutation(100000) + 100000
    offsets = {}
    for worker, rows in records.items():
        rows.sort(key=lambda row: row["episode_index"])
        _validate_worker_rows(
            rows, worker=worker, policy=policy, master=master, permutation=permutation
        )
        offsets[worker] = len(rows)
    return offsets


def _validate_worker_rows(
    rows: list[dict], *, worker: int, policy: str, master: int, permutation: np.ndarray
) -> None:
    """Verify contiguous ordinals and the exact permitted seed schedule."""
    if [row["episode_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("recovery receipts must be unique and contiguous")
    for index, row in enumerate(rows):
        seed = row["seed"]
        if row["episode_index"] != index:
            raise ValueError("recovery receipts must be unique and contiguous")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 100000 <= seed <= 199999:
            raise ValueError("recovery seed outside training-only range")
        if policy == "A" and not master <= seed < master + 20:
            raise ValueError("recovery receipt outside fixed pool")
        if policy == "B" and (index >= 5000 or seed != int(permutation[worker + 20 * index])):
            raise ValueError("recovery receipt differs from master stream")
