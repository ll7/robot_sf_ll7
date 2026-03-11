"""Unit tests for scenario thumbnail ID resolution helpers."""

from robot_sf.benchmark.scenario_thumbnails import _resolve_unique_scenario_ids


def test_unique_ids_avoid_colliding_with_natural_suffixes() -> None:
    """A natural `__2` label must not collide with generated duplicate suffixes."""

    scenarios = [
        {"name": "base"},
        {"name": "base__2"},
        {"name": "base"},
    ]

    resolved = _resolve_unique_scenario_ids(scenarios)

    assert resolved == ["base", "base__2", "base__3"]
