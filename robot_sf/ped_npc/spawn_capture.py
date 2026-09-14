"""Opt-in capture of pedestrian spawn-sampler decisions (issue #9312).

The spawn samplers in :mod:`robot_sf.ped_npc.ped_population` discard their
decisions: how many attempts a group needed, which candidates were rejected for
obstacle or minimum-separation reasons, which point was finally accepted, and
which route (not just which route template) a pedestrian was assigned. A
diagnostic consumer (#9051) needs those payloads per episode.

``SpawnSamplerCapture`` is an explicitly opt-in recording sink. Callers create
one and pass it into ``populate_simulation``; the samplers fill it in place, and
trace finalization renders it into the ``simulation-step-trace.v1`` reset block.
Nothing here changes sampling behavior or default captures: without a capture
object the samplers take exactly the code path they took before, and the reset
block keeps its explicit ``unavailable`` provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "spawn-sampler-capture.v1"
SOURCES = ("route", "crowded_zone", "synthesized", "single")


def _finite_point(point: Any) -> list[float] | None:
    """Return a JSON-safe finite ``[x, y]`` pair, or ``None`` for malformed input."""
    try:
        x, y = (float(point[0]), float(point[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return [x, y]


def _finite_waypoints(waypoints: Any) -> list[list[float]]:
    """Return JSON-safe finite waypoints in source order, skipping malformed rows.

    Accepts both point rows ``(x, y)`` and section rows ``((x1, y1), (x2, y2))``,
    flattening the latter into consecutive points without duplicating shared
    endpoints.
    """
    rendered: list[list[float]] = []
    if waypoints is None:
        return rendered
    for row in waypoints:
        point = _finite_point(row)
        if point is not None:
            rendered.append(point)
            continue
        try:
            start, end = row
        except (TypeError, ValueError):
            continue
        for endpoint in (start, end):
            endpoint_point = _finite_point(endpoint)
            if endpoint_point is not None and (not rendered or rendered[-1] != endpoint_point):
                rendered.append(endpoint_point)
    return rendered


@dataclass(slots=True)
class PedestrianSpawnRecord:
    """Per-pedestrian spawn decisions recorded by the samplers."""

    ped_id: int
    source: str
    spawn_point: list[float] | None = None
    attempts: int = 0
    obstacle_rejections: int = 0
    separation_checks: int = 0
    separation_rejections: int = 0
    route_index: int | None = None
    route_waypoints: list[list[float]] = field(default_factory=list)
    zone_index: int | None = None
    counters_recorded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable record."""
        return {
            "ped_id": int(self.ped_id),
            "source": self.source,
            "spawn_point": self.spawn_point,
            "counters_recorded": bool(self.counters_recorded),
            "attempts": int(self.attempts),
            "obstacle_rejections": int(self.obstacle_rejections),
            "separation_checks": int(self.separation_checks),
            "separation_rejections": int(self.separation_rejections),
            "route_index": self.route_index,
            "route_waypoints": [list(row) for row in self.route_waypoints],
            "zone_index": self.zone_index,
        }


@dataclass(slots=True)
class SpawnSamplerCapture:
    """Mutable, opt-in capture sink threaded through the spawn samplers.

    Records are keyed by ``(source, local ped id)`` because each spawner numbers
    its pedestrians from zero; the merged-state row index only exists after
    :func:`~robot_sf.ped_npc.ped_population.populate_simulation` concatenates the
    spawner arrays, at which point :meth:`apply_merged_offsets` re-keys them.
    """

    records: dict[tuple[str, int], PedestrianSpawnRecord] = field(default_factory=dict)
    route_offset: int | None = None
    single_offset: int | None = None
    offsets_applied: bool = False

    def record(self, ped_id: int, *, source: str) -> PedestrianSpawnRecord:
        """Return the record for one spawner-local pedestrian id, creating it when new."""
        key = (source, int(ped_id))
        existing = self.records.get(key)
        if existing is None:
            existing = PedestrianSpawnRecord(ped_id=int(ped_id), source=source)
            self.records[key] = existing
        return existing

    def record_spawn_point(self, ped_id: int, *, source: str, point: Any) -> None:
        """Record the accepted spawn point for one pedestrian."""
        finite = _finite_point(point)
        if finite is not None:
            self.record(ped_id, source=source).spawn_point = finite

    def record_attempts(
        self,
        ped_id: int,
        *,
        source: str,
        attempts: int = 0,
        obstacle_rejections: int = 0,
        separation_checks: int = 0,
        separation_rejections: int = 0,
    ) -> None:
        """Accumulate sampler attempt and rejection counters for one pedestrian."""
        record = self.record(ped_id, source=source)
        record.counters_recorded = True
        record.attempts += max(int(attempts), 0)
        record.obstacle_rejections += max(int(obstacle_rejections), 0)
        record.separation_checks += max(int(separation_checks), 0)
        record.separation_rejections += max(int(separation_rejections), 0)

    def record_route_assignment(
        self,
        ped_id: int,
        *,
        source: str,
        route_index: int,
        waypoints: Any = None,
    ) -> None:
        """Record the per-episode route identity (and waypoints) assigned to one pedestrian."""
        record = self.record(ped_id, source=source)
        record.route_index = int(route_index)
        if waypoints is not None:
            record.route_waypoints = _finite_waypoints(waypoints)

    def record_zone_assignment(
        self, ped_id: int, *, zone_id: int, source: str = "crowded_zone"
    ) -> None:
        """Record the crowded/synthetic zone index a pedestrian was placed in."""
        record = self.record(ped_id, source=source)
        record.zone_index = int(zone_id)

    def record_offsets(self, *, route_offset: int | None, single_offset: int | None) -> None:
        """Record the merged-state offsets so consumers can map records to state rows."""
        self.route_offset = None if route_offset is None else int(route_offset)
        self.single_offset = None if single_offset is None else int(single_offset)

    def apply_merged_offsets(self, *, route_offset: int, single_offset: int) -> None:
        """Re-key spawner-local ids onto merged-state row indices.

        Crowd records lead the merged state, route records follow them, and single
        pedestrians follow those, so both later groups shift by the recorded offsets.
        """
        self.record_offsets(route_offset=route_offset, single_offset=single_offset)
        for (_source, local_id), record in self.records.items():
            shift = 0
            if record.source == "route":
                shift = route_offset
            elif record.source == "single":
                shift = single_offset
            record.ped_id = int(local_id) + shift
        self.offsets_applied = True

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable capture payload."""
        by_source: dict[str, int] = {}
        totals = {
            "attempts": 0,
            "obstacle_rejections": 0,
            "separation_checks": 0,
            "separation_rejections": 0,
        }
        route_assignments = 0
        pedestrians: list[dict[str, Any]] = []
        for key in sorted(self.records, key=lambda item: (item[1], item[0])):
            record = self.records[key]
            by_source[record.source] = by_source.get(record.source, 0) + 1
            totals["attempts"] += record.attempts
            totals["obstacle_rejections"] += record.obstacle_rejections
            totals["separation_checks"] += record.separation_checks
            totals["separation_rejections"] += record.separation_rejections
            if record.route_index is not None:
                route_assignments += 1
            pedestrians.append(record.to_dict())
        pedestrians.sort(key=lambda item: (item["ped_id"], item["source"]))
        return {
            "schema_version": SCHEMA_VERSION,
            "counts": {
                "pedestrians": len(pedestrians),
                "by_source": by_source,
                "route_assignments": route_assignments,
                "offsets_applied": bool(self.offsets_applied),
                **totals,
            },
            "route_offset": self.route_offset,
            "single_offset": self.single_offset,
            "pedestrians": pedestrians,
        }


def reset_block_payload(capture: SpawnSamplerCapture | None) -> dict[str, Any] | None:
    """Return the capture payload for a trace reset block, or ``None`` when absent or empty."""
    if capture is None:
        return None
    payload = capture.to_dict()
    if not payload["pedestrians"]:
        return None
    return payload
