"""Tests for social group metadata (issue #3972).

Cover the :class:`SocialGroupDefinition` dataclass validation, the
``parse_social_group_definitions`` scenario parser, and
``MapDefinition`` group validation (duplicate ids + member resolution).
"""

from __future__ import annotations

import pytest
from shapely.geometry import Point

from robot_sf.nav.map_config import (
    MapDefinition,
    SinglePedestrianDefinition,
    SocialGroupDefinition,
    parse_social_group_definitions,
)
from robot_sf.nav.obstacle import Obstacle


def _build_map(social_groups=None, single_pedestrians=None) -> MapDefinition:
    """Create a minimal map definition, optionally with groups/pedestrians."""
    width, height = 10.0, 10.0
    obstacles = [Obstacle([(0, 0), (10, 0), (10, 1), (0, 1)])]
    bounds = [
        (0, width, 0, 0),
        (0, width, height, height),
        (0, 0, 0, height),
        (width, width, 0, height),
    ]
    return MapDefinition(
        width,
        height,
        obstacles,
        [((1, 1), (2, 1), (2, 2))],
        [((3, 3), (4, 3), (4, 4))],
        [((8, 8), (9, 8), (9, 9))],
        bounds,
        [],
        [((6, 6), (7, 6), (7, 7))],
        [],
        [],
        single_pedestrians or [],
        social_groups=social_groups or [],
    )


def test_valid_circular_group_normalizes_fields():
    """A valid circular group normalizes casing and exposes a circular o-space."""
    group = SocialGroupDefinition(
        group_id="  conversation_a ",
        type="Conversation",
        members=["ped_a", "ped_b"],
        formation="Circular_Conversation",
        centroid=(5, 3),
        radius=1.2,
    )

    assert group.group_id == "conversation_a"
    assert group.type == "conversation"
    assert group.formation == "circular_conversation"
    assert group.centroid == (5.0, 3.0)
    # Circular proxy o-space contains the centroid and excludes a far point.
    o_space = group.o_space()
    assert o_space.contains(Point(5.0, 3.0))
    assert not o_space.contains(Point(50.0, 50.0))

    spec = group.as_spec()
    assert spec["group_id"] == "conversation_a"
    assert spec["members"] == ["ped_a", "ped_b"]
    assert spec["o_space_polygon"] is None
    assert spec["centroid"] == [5.0, 3.0]


def test_polygon_group_o_space_uses_polygon():
    """An explicit polygon is used for the o-space instead of the circular proxy."""
    group = SocialGroupDefinition(
        group_id="square_a",
        type="static_group",
        members=["ped_a"],
        formation="cluster",
        centroid=(1.0, 1.0),
        radius=0.5,
        o_space_polygon=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    )
    o_space = group.o_space()
    # A point outside the circular proxy (r=0.5) but inside the square is covered.
    assert o_space.contains(Point(1.8, 1.8))
    assert group.as_spec()["o_space_polygon"] == [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_id": ""},
        {"type": "  "},
        {"members": []},
        {"formation": ""},
        {"radius": 0.0},
        {"radius": -1.0},
        {"o_space_polygon": [(0.0, 0.0), (1.0, 1.0)]},  # < 3 points
    ],
)
def test_invalid_group_fields_raise(kwargs):
    """Invalid group fields fail fast with a ValueError."""
    base = {
        "group_id": "g",
        "type": "conversation",
        "members": ["ped_a"],
        "formation": "circular",
        "centroid": (1.0, 1.0),
        "radius": 1.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        SocialGroupDefinition(**base)


def test_parse_social_group_definitions_from_scenario_entries():
    """Scenario-style mappings parse into validated group definitions."""
    groups = parse_social_group_definitions(
        [
            {
                "group_id": "conversation_a",
                "type": "conversation",
                "members": ["ped_a", "ped_b"],
                "formation": "circular_conversation",
                "centroid": [5.0, 3.0],
                "radius": 1.2,
            }
        ]
    )
    assert len(groups) == 1
    assert isinstance(groups[0], SocialGroupDefinition)
    assert groups[0].members == ("ped_a", "ped_b")


def test_parse_social_group_definitions_missing_key_raises():
    """A missing required key surfaces a clear ValueError."""
    with pytest.raises(ValueError, match="missing required key"):
        parse_social_group_definitions([{"group_id": "g", "type": "conversation"}])


def test_parse_social_group_definitions_rejects_non_list():
    """A non-list payload fails fast."""
    with pytest.raises(ValueError, match="must be a list"):
        parse_social_group_definitions({"group_id": "g"})


def test_map_definition_rejects_duplicate_group_ids():
    """Two groups sharing an id fail map validation."""
    group_a = SocialGroupDefinition(
        group_id="dup",
        type="conversation",
        members=["ped_a"],
        formation="circular",
        centroid=(5.0, 5.0),
        radius=1.0,
    )
    group_b = SocialGroupDefinition(
        group_id="dup",
        type="queue",
        members=["ped_b"],
        formation="line",
        centroid=(2.0, 2.0),
        radius=1.0,
    )
    with pytest.raises(ValueError, match="Duplicate social group id"):
        _build_map(social_groups=[group_a, group_b])


def test_map_definition_rejects_unresolved_members():
    """A group member absent from declared single pedestrians fails fast."""
    peds = [SinglePedestrianDefinition(id="ped_a", start=(2.0, 2.0), goal=(4.0, 4.0))]
    group = SocialGroupDefinition(
        group_id="conversation_a",
        type="conversation",
        members=["ped_a", "ghost"],
        formation="circular",
        centroid=(5.0, 5.0),
        radius=1.0,
    )
    with pytest.raises(ValueError, match="unknown"):
        _build_map(social_groups=[group], single_pedestrians=peds)


def test_map_definition_accepts_valid_group_without_pedestrians():
    """Groups are allowed without declared single pedestrians (no member check)."""
    group = SocialGroupDefinition(
        group_id="conversation_a",
        type="conversation",
        members=["ped_a"],
        formation="circular",
        centroid=(5.0, 5.0),
        radius=1.0,
    )
    map_def = _build_map(social_groups=[group])
    assert map_def.social_groups[0].group_id == "conversation_a"
