"""Unit tests for ``GlobalRoute``."""

from __future__ import annotations

import pytest

from robot_sf.nav.global_route import GlobalRoute

_ZONE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def test_global_route_rejects_negative_spawn_id() -> None:
    """Spawn IDs must be non-negative."""
    with pytest.raises(ValueError, match="Spawn id needs to be an integer >= 0!"):
        GlobalRoute(
            spawn_id=-1,
            goal_id=0,
            waypoints=[(0.0, 0.0), (1.0, 1.0)],
            spawn_zone=_ZONE,
            goal_zone=_ZONE,
        )


def test_global_route_rejects_negative_goal_id() -> None:
    """Goal IDs must be non-negative."""
    with pytest.raises(ValueError, match="Goal id needs to be an integer >= 0!"):
        GlobalRoute(
            spawn_id=0,
            goal_id=-1,
            waypoints=[(0.0, 0.0), (1.0, 1.0)],
            spawn_zone=_ZONE,
            goal_zone=_ZONE,
        )


def test_global_route_rejects_empty_waypoints() -> None:
    """Routes must contain at least one waypoint."""
    with pytest.raises(ValueError, match="contains no waypoints"):
        GlobalRoute(
            spawn_id=0,
            goal_id=0,
            waypoints=[],
            spawn_zone=_ZONE,
            goal_zone=_ZONE,
        )


def test_global_route_sections_lengths_offsets_and_total_length() -> None:
    """Section-derived properties should match Euclidean geometry."""
    route = GlobalRoute(
        spawn_id=0,
        goal_id=1,
        waypoints=[(0.0, 0.0), (3.0, 4.0), (6.0, 4.0)],
        spawn_zone=_ZONE,
        goal_zone=_ZONE,
    )

    assert route.sections == [((0.0, 0.0), (3.0, 4.0)), ((3.0, 4.0), (6.0, 4.0))]
    assert route.section_lengths == pytest.approx([5.0, 3.0])
    assert route.section_offsets == pytest.approx([0.0, 5.0])
    assert route.total_length == pytest.approx(8.0)


def test_global_route_single_waypoint_has_zero_length() -> None:
    """A one-waypoint route has no sections and zero total length."""
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.0, 2.0)],
        spawn_zone=_ZONE,
        goal_zone=_ZONE,
    )

    assert route.sections == []
    assert route.section_lengths == []
    assert route.section_offsets == []
    assert route.total_length == 0.0


def test_global_route_preserves_source_metadata() -> None:
    """Source SVG metadata should be stored for traceability."""
    route = GlobalRoute(
        spawn_id=1,
        goal_id=2,
        waypoints=[(0.0, 0.0), (1.0, 1.0)],
        spawn_zone=_ZONE,
        goal_zone=_ZONE,
        source_path_id="path42",
        source_label="ped_route",
    )

    assert route.source_path_id == "path42"
    assert route.source_label == "ped_route"
