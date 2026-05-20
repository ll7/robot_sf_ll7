"""Regression tests for strict MapDefinition construction invariants."""

from __future__ import annotations

import pytest

from robot_sf.nav.map_config import MapDefinition


def _valid_map_kwargs() -> dict:
    """Return minimal valid MapDefinition kwargs for invariant tests."""

    zone = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    return {
        "width": 10.0,
        "height": 10.0,
        "obstacles": [],
        "robot_spawn_zones": [zone],
        "ped_spawn_zones": [],
        "robot_goal_zones": [zone],
        "bounds": [
            (0.0, 10.0, 0.0, 0.0),
            (0.0, 10.0, 10.0, 10.0),
            (0.0, 0.0, 0.0, 10.0),
            (10.0, 10.0, 0.0, 10.0),
        ],
        "robot_routes": [],
        "ped_goal_zones": [],
        "ped_crowded_zones": [],
        "ped_routes": [],
    }


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("width", 0.0, "Map width and height"),
        ("width", -1.0, "Map width and height"),
        ("height", 0.0, "Map width and height"),
        ("height", -1.0, "Map width and height"),
    ],
)
def test_map_definition_rejects_non_positive_dimensions(
    field_name: str, value: float, message: str
) -> None:
    """Map dimensions are construction-time invariants, not logged degradations."""

    kwargs = _valid_map_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        MapDefinition(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("robot_spawn_zones", "Robot spawn zones"),
        ("robot_goal_zones", "Robot goal zones"),
    ],
)
def test_map_definition_rejects_missing_required_robot_zones(field_name: str, message: str) -> None:
    """Robot spawn and goal zones must fail fast when absent."""

    kwargs = _valid_map_kwargs()
    kwargs[field_name] = []

    with pytest.raises(ValueError, match=message):
        MapDefinition(**kwargs)


def test_map_definition_rejects_wrong_bounds_count() -> None:
    """A complete map boundary must contain exactly four bounds."""

    kwargs = _valid_map_kwargs()
    kwargs["bounds"] = kwargs["bounds"][:3]

    with pytest.raises(ValueError, match="Expected exactly 4 bounds"):
        MapDefinition(**kwargs)


def test_map_definition_rejects_malformed_pair_bounds() -> None:
    """Malformed pair-of-point bounds should not survive normalization."""

    kwargs = _valid_map_kwargs()
    kwargs["bounds"] = [
        ((0.0, 0.0), (10.0,)),  # Missing y coordinate in second endpoint.
        (0.0, 10.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 10.0),
        (10.0, 10.0, 0.0, 10.0),
    ]

    with pytest.raises(ValueError, match="Invalid bound"):
        MapDefinition(**kwargs)


@pytest.mark.parametrize(
    "bad_bound",
    [
        "skip-me",
        (0.0, 10.0, "bad-y", 0.0),
    ],
)
def test_map_definition_rejects_malformed_flat_bounds(bad_bound: object) -> None:
    """Malformed flat bounds should fail before they reach simulation helpers."""

    kwargs = _valid_map_kwargs()
    kwargs["bounds"] = [
        bad_bound,
        (0.0, 10.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 10.0),
        (10.0, 10.0, 0.0, 10.0),
    ]

    with pytest.raises(ValueError, match="Invalid bound"):
        MapDefinition(**kwargs)


def test_map_definition_normalizes_valid_pair_bounds() -> None:
    """Valid pair-of-point bounds should normalize into flat pysf-compatible segments."""

    kwargs = _valid_map_kwargs()
    kwargs["bounds"] = [
        ((0.0, 0.0), (10.0, 0.0)),
        ((0.0, 10.0), (10.0, 10.0)),
        ((0.0, 0.0), (0.0, 10.0)),
        ((10.0, 0.0), (10.0, 10.0)),
    ]

    map_def = MapDefinition(**kwargs)

    assert map_def.bounds == [
        (0.0, 10.0, 0.0, 0.0),
        (0.0, 10.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 10.0),
        (10.0, 10.0, 0.0, 10.0),
    ]
