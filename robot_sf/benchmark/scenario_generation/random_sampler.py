"""Deterministically sample source scenarios and episode seeds for generation runs."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampledEpisode:
    """One reproducible map-runner job selected by the Monte Carlo sampler."""

    sample_index: int
    source_scenario_name: str
    source_map: str
    episode_seed: int
    scenario: dict[str, Any]

    def manifest_record(self) -> dict[str, Any]:
        """Return all random choices needed to reproduce this sampled job."""

        return {
            "sample_index": self.sample_index,
            "source_scenario_name": self.source_scenario_name,
            "source_map": self.source_map,
            "episode_seed": self.episode_seed,
            "materialized_scenario_name": self.scenario["name"],
        }


def sample_episode_jobs(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    episode_budget: int,
    episode_seed_min: int = 1,
    episode_seed_max: int = 2_147_483_647,
) -> list[SampledEpisode]:
    """Sample scenario templates and unique runtime seeds with a local RNG.

    The source scenario's map routes remain the start/goal and pedestrian-route
    proposal space.  The sampled episode seed drives the existing map runner's
    route, point-of-interest, and population randomization.

    Returns:
        Materialized one-seed scenario jobs in deterministic sample order.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(episode_budget, int) or isinstance(episode_budget, bool):
        raise ValueError("episode_budget must be an integer")
    if episode_budget <= 0:
        raise ValueError("episode_budget must be > 0")
    if episode_seed_min < 0 or episode_seed_max < episode_seed_min:
        raise ValueError("episode seed range must be non-negative and ordered")

    candidates = [dict(scenario) for scenario in scenarios if _is_supported(scenario)]
    if not candidates:
        raise ValueError("source scenario set contains no supported scenarios")
    seed_capacity = episode_seed_max - episode_seed_min + 1
    if episode_budget > seed_capacity:
        raise ValueError("episode_budget exceeds the unique episode-seed range")

    rng = random.Random(seed)
    used_seeds: set[int] = set()
    sampled: list[SampledEpisode] = []
    for sample_index in range(episode_budget):
        source = candidates[rng.randrange(len(candidates))]
        source_name = _scenario_name(source)
        source_map = _source_map(source)
        episode_seed = rng.randint(episode_seed_min, episode_seed_max)
        while episode_seed in used_seeds:
            episode_seed = rng.randint(episode_seed_min, episode_seed_max)
        used_seeds.add(episode_seed)

        materialized = deepcopy(source)
        materialized["name"] = f"generated-sample-{sample_index:04d}-{source_name}"
        materialized["seeds"] = [episode_seed]
        metadata = dict(materialized.get("metadata") or {})
        metadata["scenario_generation"] = {
            "source": "auto_generated",
            "required_manual_review": True,
            "benchmark_evidence": False,
            "source_scenario_name": source_name,
            "sample_index": sample_index,
        }
        materialized["metadata"] = metadata
        sampled.append(
            SampledEpisode(
                sample_index=sample_index,
                source_scenario_name=source_name,
                source_map=source_map,
                episode_seed=episode_seed,
                scenario=materialized,
            )
        )
    return sampled


def _is_supported(scenario: Mapping[str, Any]) -> bool:
    metadata = scenario.get("metadata")
    return scenario.get("supported") is not False and not (
        isinstance(metadata, Mapping) and metadata.get("supported") is False
    )


def _scenario_name(scenario: Mapping[str, Any]) -> str:
    value = scenario.get("name") or scenario.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every source scenario must have a non-empty name or id")
    return value.strip()


def _source_map(scenario: Mapping[str, Any]) -> str:
    value = scenario.get("map_file") or scenario.get("map_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source scenario '{_scenario_name(scenario)}' has no map reference")
    value = value.strip()
    return Path(value).as_posix() if Path(value).is_absolute() else value


__all__ = ["SampledEpisode", "sample_episode_jobs"]
