"""Spawn-validity block for benchmark episode records (issue #9725).

Every episode record carries ``spawn_validity``: the reset clearance between the
robot and pedestrians and between the robot and the static map, plus any
route-end respawn that could not avoid the robot footprint. A row whose reset is
already in contact, or whose pedestrian collision follows such a respawn, is
marked ``invalid_run`` with reason ``spawn_overlap``. The event ledger mirrors the
flag and aggregation excludes these rows from rates, because the outcome is a
simulator spawn defect rather than planner behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from robot_sf.nav.spawn_clearance import SPAWN_OVERLAP_INVALID_REASON

SPAWN_VALIDITY_SCHEMA_VERSION = "spawn_validity.v1"


def build_spawn_validity(
    reset_clearance: Mapping[str, Any] | None,
    respawn_overlap_events: Sequence[Mapping[str, Any]] | None,
    *,
    ped_collision_seen: bool,
) -> dict[str, Any]:
    """Return the per-episode spawn-validity block.

    Args:
        reset_clearance: Output of ``reset_spawn_clearance`` right after reset, or
            ``None`` when it could not be measured.
        respawn_overlap_events: Route-end respawns that landed inside a robot footprint.
        ped_collision_seen: Whether the episode recorded a robot-pedestrian collision.

    Returns:
        JSON-serializable spawn-validity block.
    """
    events = [dict(event) for event in respawn_overlap_events or []]
    reset_overlap = bool(reset_clearance.get("overlap")) if reset_clearance else False
    respawn_collision = bool(events) and bool(ped_collision_seen)
    invalid = reset_overlap or respawn_collision
    return {
        "schema_version": SPAWN_VALIDITY_SCHEMA_VERSION,
        "reset_clearance": dict(reset_clearance) if reset_clearance else None,
        "reset_clearance_status": "available" if reset_clearance else "unavailable",
        "reset_overlap": reset_overlap,
        "respawn_overlap_events": events,
        "respawn_overlap_collision": respawn_collision,
        "invalid_run": invalid,
        "invalid_reason": SPAWN_OVERLAP_INVALID_REASON if invalid else None,
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
    "SPAWN_VALIDITY_SCHEMA_VERSION",
    "build_spawn_validity",
    "record_has_spawn_overlap",
]
