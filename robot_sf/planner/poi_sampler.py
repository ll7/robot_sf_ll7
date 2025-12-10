"""POI sampling utilities for global planning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from robot_sf.common.types import Vec2D
    from robot_sf.nav.map_config import MapDefinition


class POISampler:
    """Utility for sampling POI-based waypoints."""

    def __init__(self, map_definition: MapDefinition, seed: int | None = None) -> None:
        """Initialize sampler with map POIs.

        Args:
            map_definition: Map containing at least one POI.
            seed: Optional RNG seed for reproducibility.

        Raises:
            ValueError: If the map has no POIs.
        """
        if not map_definition.poi_positions:
            raise ValueError("Cannot sample POIs from map without POIs")
        self._map = map_definition
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        count: int,
        strategy: Literal["random", "nearest", "farthest"] = "random",
        *,
        start: Vec2D | None = None,
    ) -> list[Vec2D]:
        """Sample POIs according to the requested strategy.

        Args:
            count: Number of POIs to sample (clamped to available POIs).
            strategy: Selection strategy ("random", "nearest", "farthest").
            start: Required for nearest/farthest strategies as distance reference.

        Returns:
            List of POI positions.

        Raises:
            ValueError: When an invalid strategy is provided or required params missing.
        """
        if count <= 0:
            return []

        available = min(count, len(self._map.poi_positions))
        if strategy == "random":
            indices = self._rng.choice(len(self._map.poi_positions), size=available, replace=False)
            return [self._map.poi_positions[i] for i in indices]

        if start is None:
            raise ValueError("start position is required for nearest/farthest strategies")

        distances = [
            (idx, (pos[0] - start[0]) ** 2 + (pos[1] - start[1]) ** 2)
            for idx, pos in enumerate(self._map.poi_positions)
        ]
        distances.sort(key=lambda t: t[1], reverse=(strategy == "farthest"))
        selected_indices = [idx for idx, _ in distances[:available]]
        return [self._map.poi_positions[i] for i in selected_indices]
