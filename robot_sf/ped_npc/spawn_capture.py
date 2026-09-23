"""Opt-in per-episode record of pedestrian spawn-sampler decisions.

This module is intentionally dependency-free (stdlib only) so sampler owners,
the simulator, and trace finalization can all share it without import cycles.

The capture object is created by the caller that wants diagnostics and threaded
through the sampler owners; every hook takes ``None`` (the default) to mean
"do not record", which preserves the historical default path exactly. Counters
stay zero and ``assigned_routes`` stays empty unless hooks run with an instance.
Use :meth:`SpawnSamplerCapture.to_mapping` for the JSON-serializable trace form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AssignedRoute:
    """One route-following group's route assignment for trace provenance."""

    group_index: int
    spawn_id: int
    goal_id: int
    source_path_id: str = ""
    source_label: str = ""
    initial_section: int = 0
    ped_offset: int = 0
    waypoints: tuple[tuple[float, float], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable assignment record."""
        return {
            "group_index": int(self.group_index),
            "spawn_id": int(self.spawn_id),
            "goal_id": int(self.goal_id),
            "source_path_id": str(self.source_path_id),
            "source_label": str(self.source_label),
            "initial_section": int(self.initial_section),
            "ped_offset": int(self.ped_offset),
            "waypoint_count": len(self.waypoints),
            "waypoints": [[float(x), float(y)] for x, y in self.waypoints],
        }


@dataclass
class SpawnSamplerCapture:
    """Mutable per-episode sampler-decision record.

    Sampler hooks increment counters and append assignment records in place;
    the owning episode converts the result with :meth:`to_mapping` once.
    """

    route_anchor_attempts: int = 0
    route_anchor_failures: int = 0
    obstacle_rejections: int = 0
    separation_rejections: int = 0
    accepted_samples: int = 0
    assigned_routes: list[AssignedRoute] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable capture payload."""
        return {
            "route_anchor_attempts": int(self.route_anchor_attempts),
            "route_anchor_failures": int(self.route_anchor_failures),
            "obstacle_rejections": int(self.obstacle_rejections),
            "separation_rejections": int(self.separation_rejections),
            "accepted_samples": int(self.accepted_samples),
            "assigned_routes": [record.to_mapping() for record in self.assigned_routes],
        }
