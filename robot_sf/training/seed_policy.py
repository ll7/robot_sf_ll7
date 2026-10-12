"""Reproducible training-only episode identities and resolution-time split guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark import seed_bands


def _integer(value: object) -> int:
    """Validate a seed without truncation.

    Returns:
        The validated integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**31:
        raise ValueError("seed policy requires non-negative integer seeds below the RNG limit")
    return value


def _sealed(seed: int) -> bool:
    """Check the complete sealed interval.

    Returns:
        Whether the identity is protected.
    """
    return (
        min(seed_bands.EVAL_SEEDS_0_0_8) <= seed <= max(seed_bands.EVAL_SEEDS_0_0_8)
        or seed in seed_bands.RETIRED_EVAL_SEEDS_0_0_7
    )


@dataclass(frozen=True)
class TrainingSeedPolicy:
    """Inclusive range and one logged master seed, with no episode reuse."""

    seed_range: tuple[int, int]
    master_seed: int

    @classmethod
    def resolve(
        cls,
        seed_range: Sequence[int],
        master_seed: int,
        *,
        selection_seeds: Sequence[int] = (),
        evaluation_seeds: Sequence[int] = (),
    ) -> TrainingSeedPolicy:
        """Reject leakage before environment construction.

        Returns:
            The validated training policy.
        """
        if not isinstance(seed_range, (list, tuple)) or len(seed_range) != 2:
            raise ValueError("training_seed_range must be an inclusive [start, end] pair")
        low, high = (_integer(value) for value in seed_range)
        if low > high:
            raise ValueError("training_seed_range start must not exceed end")
        master = _integer(master_seed)
        if any(
            low <= end and start <= high
            for start, end in (
                (min(seed_bands.EVAL_SEEDS_0_0_8), max(seed_bands.EVAL_SEEDS_0_0_8)),
                (
                    min(seed_bands.RETIRED_EVAL_SEEDS_0_0_7),
                    max(seed_bands.RETIRED_EVAL_SEEDS_0_0_7),
                ),
            )
        ):
            raise ValueError("training seed range overlaps sealed seeds")
        validate_seed_splits((), selection_seeds, evaluation_seeds)
        for label, seeds in (("selection", selection_seeds), ("evaluation", evaluation_seeds)):
            if any(low <= _integer(seed) <= high for seed in seeds):
                raise ValueError(f"training seed range overlaps {label} seeds")
        if low <= 1200 and 1001 <= high:
            raise ValueError("training seed range overlaps reserved development seeds")
        return cls((low, high), master)

    def stream(self, worker_index: int = 0, worker_count: int = 1) -> EpisodeSeedStream:
        """Allocate a worker subsequence from the master permutation.

        Returns:
            The independent, disjoint worker stream.
        """
        return EpisodeSeedStream(self, worker_index, worker_count)


class EpisodeSeedStream:
    """Finite shuffled range; exhaustion refuses reuse instead of silently cycling."""

    def __init__(self, policy: TrainingSeedPolicy, worker_index: int, worker_count: int) -> None:
        """Partition a deterministic permutation independently of worker scheduling."""
        low, high = policy.seed_range
        if not 0 <= worker_index < worker_count <= high - low + 1:
            raise ValueError("invalid training seed worker partition")
        permutation = np.random.default_rng(policy.master_seed).permutation(high - low + 1)
        self._seeds = permutation[worker_index::worker_count] + low
        self._position = 0

    def next_seed(self) -> int:
        """Return a fresh episode seed or refuse an exhausted training budget."""
        if self._position == len(self._seeds):
            raise ValueError("training episode seed range exhausted; seed reuse refused")
        result = int(self._seeds[self._position])
        self._position += 1
        return result


def validate_seed_splits(
    training_seeds: Sequence[int],
    selection_seeds: Sequence[int],
    evaluation_seeds: Sequence[int] = (),
    *,
    validate_heldout: bool = True,
) -> None:
    """Reject explicit split leakage and sealed seeds without echoing restricted values."""
    splits = {
        "training": training_seeds,
        "selection": selection_seeds,
        "evaluation": evaluation_seeds,
    }
    for label, seeds in splits.items():
        if label != "training" and not validate_heldout:
            continue
        if any(_sealed(_integer(seed)) for seed in seeds):
            raise ValueError(f"{label} seeds overlap sealed seeds")
    for label, seeds in (("selection", selection_seeds), ("evaluation", evaluation_seeds)):
        if set(training_seeds).intersection(seeds):
            raise ValueError(f"training seeds overlap {label} seeds")
    if set(selection_seeds).intersection(evaluation_seeds):
        raise ValueError("selection seeds overlap development evaluation seeds")


def load_collection_seed_policy(path: Path) -> tuple[dict, TrainingSeedPolicy]:
    """Resolve a guarded predictive collection contract.

    Returns:
        Config bytes decoded as a mapping and the validated episode policy.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("collection training config must be a mapping")
    selection = data.get("selection_seeds", ())
    evaluation = data.get("evaluation_seeds", ())
    if not selection or not evaluation:
        raise ValueError("collection config requires fixed selection and evaluation seeds")
    if any(not 1001 <= _integer(seed) <= 1200 for seed in (*selection, *evaluation)):
        raise ValueError("collection held-out seeds must belong to the development band")
    policy = TrainingSeedPolicy.resolve(
        data.get("training_seed_range"),
        data.get("training_master_seed"),
        selection_seeds=selection,
        evaluation_seeds=evaluation,
    )
    return dict(data), policy
