"""Regression tests for ``force_population_size`` accounting in ``populate_simulation``.

Issue #4618 R4: when a scenario declares BOTH pedestrian routes and crowded zones, the
route spawner and the crowd spawner each independently honored ``force_population_size``,
so the merged population was ``2 * force_population_size`` (a silent double-count). The
forced size must instead be the exact TOTAL, split across the active spawners.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.ped_npc import ped_population
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation


@pytest.fixture
def _captured_spawner_sizes(monkeypatch):
    """Patch the two spawners/behaviors and capture the forced size each spawner sees.

    Returns a dict with ``crowd`` and ``route`` keys holding the
    ``force_population_size`` value passed to each spawner (``None`` when a spawner is
    skipped or receives no override).
    """
    captured: dict[str, int | None] = {"crowd": None, "route": None}

    def _fake_crowded(config, crowded_zones, *, obstacle_polygons=None):
        captured["crowd"] = config.force_population_size
        return np.zeros((0, 6)), [], {}

    def _fake_routes(config, routes, *, obstacle_polygons=None):
        captured["route"] = config.force_population_size
        return np.zeros((0, 6)), [], {}, {}

    monkeypatch.setattr(ped_population, "populate_crowded_zones", _fake_crowded)
    monkeypatch.setattr(ped_population, "populate_ped_routes", _fake_routes)
    # Replace behavior constructors with no-ops; the spawners return empty populations.
    monkeypatch.setattr(ped_population, "CrowdedZoneBehavior", lambda *a, **k: object())
    monkeypatch.setattr(ped_population, "FollowRouteBehavior", lambda *a, **k: object())
    return captured


def test_force_population_size_split_when_routes_and_zones_present(_captured_spawner_sizes):
    """Both spawners active: the forced total is split, summing to the requested size."""
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        force_population_size=10,
    )
    populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=[object()],  # sentinel routes; spawner is stubbed
        ped_crowded_zones=[object()],  # sentinel zones; spawner is stubbed
    )

    crowd_size = _captured_spawner_sizes["crowd"]
    route_size = _captured_spawner_sizes["route"]
    assert crowd_size is not None and route_size is not None
    # No double-count: the two shares reconstruct the exact requested total.
    assert crowd_size + route_size == 10
    # Odd remainder is assigned to routes (5 + 5 here, but the contract is the sum).
    assert route_size >= crowd_size


def test_force_population_size_odd_remainder_goes_to_routes(_captured_spawner_sizes):
    """An odd forced total assigns the extra pedestrian to the route spawner."""
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        force_population_size=7,
    )
    populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=[object()],
        ped_crowded_zones=[object()],
    )

    assert _captured_spawner_sizes["crowd"] == 3
    assert _captured_spawner_sizes["route"] == 4


def test_force_population_size_not_split_when_only_routes(_captured_spawner_sizes):
    """Only routes active: the route spawner receives the full forced size, unsplit."""
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        force_population_size=10,
    )
    populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=[object()],
        ped_crowded_zones=[],
    )

    assert _captured_spawner_sizes["route"] == 10
