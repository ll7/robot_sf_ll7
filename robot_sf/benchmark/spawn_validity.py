"""Spawn-validity block for benchmark episode records (issue #9725).

Every episode record carries ``spawn_validity``: the reset clearance between the
robot and pedestrians and between the robot and the static map, plus any
route-end respawn that could not avoid the robot footprint. A row whose reset is
already in contact, or whose robot collides with a respawned pedestrian within one
second of that respawn, is marked ``invalid_run`` with reason ``spawn_overlap``
(unless the route was completed). The event ledger mirrors the
flag and aggregation excludes these rows from rates, because the outcome is a
simulator spawn defect rather than planner behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from robot_sf.nav.spawn_clearance import SPAWN_OVERLAP_INVALID_REASON

SPAWN_VALIDITY_SCHEMA_VERSION = "spawn_validity.v1"


#: A pedestrian collision counts as caused by a respawn only this soon after it (seconds).
RESPAWN_COLLISION_WINDOW_S = 1.0


def _respawn_collisions(
    respawn_events: Sequence[Mapping[str, Any]],
    collision_events: Sequence[Mapping[str, Any]],
    dt_seconds: float,
) -> list[dict[str, Any]]:
    """Match pedestrian collisions to respawn overlaps of the same pedestrian.

    A collision is attributed to a respawn when its partner is one of the respawned
    group's pedestrian rows and it happens within ``RESPAWN_COLLISION_WINDOW_S`` after
    the respawn step.

    Returns:
        One record per attributed collision.
    """
    matches: list[dict[str, Any]] = []
    if dt_seconds <= 0.0:
        return matches
    for collision in collision_events:
        if collision.get("collision_partner_type") != "pedestrian":
            continue
        try:
            raw_partners = collision.get("contact_partner_ids") or [
                collision.get("collision_partner_id")
            ]
            partners = {int(str(partner)) for partner in raw_partners}
            collision_time = float(collision.get("collision_time"))
        except (TypeError, ValueError):
            continue
        for event in respawn_events:
            rows = {int(row) for row in event.get("ped_rows", [])}
            respawn_time = float(event.get("step", 0)) * dt_seconds
            elapsed = collision_time - respawn_time
            in_window = -dt_seconds - 1.0e-9 <= elapsed <= RESPAWN_COLLISION_WINDOW_S + 1.0e-9
            hit = sorted(partners & rows)
            if hit and in_window:
                matches.append(
                    {
                        "group_id": event.get("group_id"),
                        "ped_row": hit[0],
                        "respawn_time_s": respawn_time,
                        "collision_time_s": collision_time,
                    }
                )
    return matches


def build_spawn_validity(
    reset_clearance: Mapping[str, Any] | None,
    respawn_overlap_events: Sequence[Mapping[str, Any]] | None,
    *,
    collision_events: Sequence[Mapping[str, Any]] | None = None,
    dt_seconds: float = 0.1,
    route_complete: bool = False,
    reset_clearance_error: str | None = None,
) -> dict[str, Any]:
    """Return the per-episode spawn-validity block.

    Args:
        reset_clearance: Output of ``reset_spawn_clearance`` right after reset, or
            ``None`` when it could not be measured.
        respawn_overlap_events: Route-end respawns that landed inside a robot footprint.
        collision_events: Typed collision events of the episode.
        dt_seconds: Simulation step length used to time respawn events.
        reset_clearance_error: Why the reset clearance could not be measured, if so.
        route_complete: Whether the robot completed its route. A completed route is
            never marked invalid: no collision was counted, so no rate is distorted,
            and the ledger keeps ``goal_reached`` and ``invalid_run`` exclusive.

    Returns:
        JSON-serializable spawn-validity block.
    """
    events = [dict(event) for event in respawn_overlap_events or []]
    reset_overlap = bool(reset_clearance.get("overlap")) if reset_clearance else False
    respawn_collisions = _respawn_collisions(events, collision_events or [], float(dt_seconds))
    invalid = (reset_overlap or bool(respawn_collisions)) and not route_complete
    return {
        "schema_version": SPAWN_VALIDITY_SCHEMA_VERSION,
        "reset_clearance": dict(reset_clearance) if reset_clearance else None,
        "reset_clearance_status": "available" if reset_clearance else "unavailable",
        "reset_clearance_error": None if reset_clearance else reset_clearance_error,
        "reset_overlap": reset_overlap,
        "respawn_overlap_events": events,
        "respawn_overlap_collisions": respawn_collisions,
        "invalid_run": invalid,
        "invalid_reason": SPAWN_OVERLAP_INVALID_REASON if invalid else None,
    }


def spawn_validity_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count spawn-validity outcomes over episode records for report metadata.

    Returns:
        Counts of records with a spawn block, spawn-overlap exclusions, and records
        whose reset clearance could not be measured (so their validity is unknown).
    """
    blocks = [r.get("spawn_validity") for r in records if isinstance(r, Mapping)]
    present = [b for b in blocks if isinstance(b, Mapping)]
    return {
        "records_with_spawn_validity": len(present),
        "spawn_overlap_excluded_count": sum(
            1 for r in records if isinstance(r, Mapping) and record_has_spawn_overlap(r)
        ),
        "reset_clearance_unavailable_count": sum(
            1 for b in present if b.get("reset_clearance_status") != "available"
        ),
    }


def record_has_spawn_overlap(record: Mapping[str, Any]) -> bool:
    """Return whether an episode record is invalid because of a spawn overlap."""
    block = record.get("spawn_validity")
    return (
        isinstance(block, Mapping)
        and block.get("invalid_run") is True
        and block.get("invalid_reason") == SPAWN_OVERLAP_INVALID_REASON
    )


__all__ = [
    "RESPAWN_COLLISION_WINDOW_S",
    "SPAWN_VALIDITY_SCHEMA_VERSION",
    "build_spawn_validity",
    "record_has_spawn_overlap",
    "spawn_validity_counts",
]
