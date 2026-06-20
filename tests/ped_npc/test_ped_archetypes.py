"""Tests for speed-based pedestrian behavior archetypes (issue #3230).

Covers the pure archetype helpers, the shipped registry config, the
``PedSpawnConfig`` validation, and end-to-end spawning through
``populate_ped_routes`` — including the invariant that the homogeneous default
(no composition) is unchanged.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from robot_sf.nav.navigation import get_prepared_obstacles
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.ped_npc.ped_archetypes import (
    allocate_archetype_counts,
    assign_archetype_speed_factors,
    load_archetypes,
    validate_composition,
)
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_ped_routes

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHETYPE_CONFIG = _REPO_ROOT / "configs" / "research" / "pedestrian_archetypes_v1.yaml"
_CLASSIC_CROSSING_SVG = _REPO_ROOT / "maps" / "svg_maps" / "classic_crossing.svg"


# --- registry loading -----------------------------------------------------


def test_load_archetypes_real_config() -> None:
    """The shipped registry exposes >=3 archetypes with the documented factors."""
    factors = load_archetypes(_ARCHETYPE_CONFIG)
    assert {"cautious", "standard", "hurried"} <= set(factors)
    assert len(factors) >= 3
    assert factors["standard"] == 1.0
    assert factors["cautious"] < 1.0 < factors["hurried"]


def test_load_archetypes_rejects_missing_factor(tmp_path: Path) -> None:
    """An archetype without desired_speed_factor is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("archetypes:\n  x:\n    note: no factor\n", encoding="utf-8")
    with pytest.raises(ValueError, match="desired_speed_factor"):
        load_archetypes(bad)


def test_load_archetypes_rejects_nonpositive_factor(tmp_path: Path) -> None:
    """A non-positive speed factor is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("archetypes:\n  x:\n    desired_speed_factor: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be > 0"):
        load_archetypes(bad)


# --- composition validation ----------------------------------------------


def test_validate_composition_accepts_valid() -> None:
    """A composition summing to one over known archetypes validates."""
    validate_composition({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 1.2})


@pytest.mark.parametrize(
    ("composition", "match"),
    [
        ({"a": 0.5, "b": 0.4}, "sum to 1.0"),
        ({"a": 0.0, "b": 1.0}, "must be > 0"),
        ({"missing": 1.0}, "unknown archetypes"),
        ({}, "non-empty"),
    ],
)
def test_validate_composition_rejects(composition: dict[str, float], match: str) -> None:
    """Invalid compositions raise with an explanatory message."""
    with pytest.raises(ValueError, match=match):
        validate_composition(composition, {"a": 1.0, "b": 1.0})


# --- count allocation and factor assignment -------------------------------


def test_allocate_counts_sum_to_n_exactly() -> None:
    """Largest-remainder allocation always sums to n."""
    comp = {"cautious": 0.34, "standard": 0.33, "hurried": 0.33}
    for n in (0, 1, 2, 7, 10, 101):
        counts = allocate_archetype_counts(n, comp)
        assert sum(counts.values()) == n
        assert all(c >= 0 for c in counts.values())


def test_allocate_counts_is_deterministic() -> None:
    """Allocation is deterministic for the same inputs."""
    comp = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert allocate_archetype_counts(13, comp) == allocate_archetype_counts(13, comp)


def test_assign_speed_factors_distribution_and_determinism() -> None:
    """Assigned factors match the allocated counts and are seed-deterministic."""
    comp = {"cautious": 0.5, "hurried": 0.5}
    factors_map = {"cautious": 0.7, "hurried": 1.4}
    n = 20
    out = assign_archetype_speed_factors(n, comp, factors_map, seed=7)
    assert out.shape == (n,)
    counts = allocate_archetype_counts(n, comp)
    value_counts = Counter(np.round(out, 6).tolist())
    assert value_counts[0.7] == counts["cautious"]
    assert value_counts[1.4] == counts["hurried"]
    # Deterministic for a fixed seed, and order varies with seed.
    assert np.array_equal(out, assign_archetype_speed_factors(n, comp, factors_map, seed=7))


def test_assign_speed_factors_empty_population() -> None:
    """Zero pedestrians yields an empty factor array."""
    out = assign_archetype_speed_factors(0, {"a": 1.0}, {"a": 1.0}, seed=1)
    assert out.shape == (0,)


# --- PedSpawnConfig validation -------------------------------------------


def test_spawn_config_requires_factors_with_composition() -> None:
    """Setting a composition without speed factors is rejected at construction."""
    with pytest.raises(ValueError, match="archetype_speed_factors must be provided"):
        PedSpawnConfig(
            peds_per_area_m2=0.05,
            max_group_members=3,
            archetype_composition={"standard": 1.0},
        )


def test_spawn_config_rejects_invalid_composition() -> None:
    """An invalid composition is rejected at construction."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        PedSpawnConfig(
            peds_per_area_m2=0.05,
            max_group_members=3,
            archetype_composition={"standard": 0.5},
            archetype_speed_factors={"standard": 1.0},
        )


def test_spawn_config_accepts_valid_archetypes() -> None:
    """A valid composition + factors constructs without error."""
    cfg = PedSpawnConfig(
        peds_per_area_m2=0.05,
        max_group_members=3,
        archetype_composition={"cautious": 0.5, "hurried": 0.5},
        archetype_speed_factors={"cautious": 0.7, "hurried": 1.4},
    )
    assert cfg.archetype_composition is not None


# --- end-to-end spawning --------------------------------------------------


def _crossing_routes():
    """Load classic_crossing pedestrian routes + obstacles, or skip if unavailable."""
    if not _CLASSIC_CROSSING_SVG.exists():
        pytest.skip("classic_crossing.svg not available")
    md = SvgMapConverter(str(_CLASSIC_CROSSING_SVG)).get_map_definition()
    if not md.ped_routes:
        pytest.skip("classic_crossing has no pedestrian routes")
    return md


def test_populate_homogeneous_default_unchanged() -> None:
    """Without a composition, every pedestrian keeps the base initial speed."""
    md = _crossing_routes()
    cfg = PedSpawnConfig(
        peds_per_area_m2=0.08,
        max_group_members=3,
        initial_speed=0.5,
        route_spawn_distribution="spread",
        route_spawn_seed=123,
    )
    ped_states, *_ = populate_ped_routes(cfg, md.ped_routes, get_prepared_obstacles(md))
    assert ped_states.shape[0] > 0
    speeds = np.linalg.norm(ped_states[:, 2:4], axis=1)
    assert np.allclose(speeds, 0.5)


def test_populate_with_archetypes_scales_per_pedestrian_speed() -> None:
    """A composition scales each pedestrian's speed by its archetype factor."""
    md = _crossing_routes()
    base_speed = 0.5
    composition = {"cautious": 0.34, "standard": 0.33, "hurried": 0.33}
    speed_factors = {"cautious": 0.7, "standard": 1.0, "hurried": 1.4}
    cfg = PedSpawnConfig(
        peds_per_area_m2=0.12,
        max_group_members=3,
        initial_speed=base_speed,
        route_spawn_distribution="spread",
        route_spawn_seed=123,
        archetype_composition=composition,
        archetype_speed_factors=speed_factors,
        archetype_seed=42,
    )
    ped_states, *_ = populate_ped_routes(cfg, md.ped_routes, get_prepared_obstacles(md))
    n = ped_states.shape[0]
    assert n > 0
    speeds = np.linalg.norm(ped_states[:, 2:4], axis=1)

    # Every speed corresponds to one archetype factor times the base speed.
    expected_speeds = {round(base_speed * f, 6) for f in speed_factors.values()}
    assert set(np.round(speeds, 6).tolist()) <= expected_speeds

    # The per-archetype counts match the deterministic allocation.
    counts = allocate_archetype_counts(n, composition)
    value_counts = Counter(np.round(speeds, 6).tolist())
    assert value_counts[round(base_speed * 0.7, 6)] == counts["cautious"]
    assert value_counts[round(base_speed * 1.4, 6)] == counts["hurried"]


def test_populate_with_archetypes_is_seed_deterministic() -> None:
    """A fixed archetype seed yields identical per-pedestrian speeds."""
    md = _crossing_routes()
    kwargs = {
        "peds_per_area_m2": 0.12,
        "max_group_members": 3,
        "initial_speed": 0.5,
        "route_spawn_distribution": "spread",
        "route_spawn_seed": 123,
        "archetype_composition": {"cautious": 0.5, "hurried": 0.5},
        "archetype_speed_factors": {"cautious": 0.7, "hurried": 1.4},
        "archetype_seed": 99,
    }
    a, *_ = populate_ped_routes(PedSpawnConfig(**kwargs), md.ped_routes, get_prepared_obstacles(md))
    b, *_ = populate_ped_routes(PedSpawnConfig(**kwargs), md.ped_routes, get_prepared_obstacles(md))
    assert np.array_equal(a[:, 2:4], b[:, 2:4])
