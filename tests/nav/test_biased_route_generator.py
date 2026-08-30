"""Deterministic route-condition generator tests (issue #8033)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from robot_sf.nav.biased_route_generator import (
    ROUTE_CONDITIONS,
    corridor_map,
    doorway_map,
    generate_route_conditions,
    route_condition_report,
)


def test_all_conditions_feasible_on_both_canonical_maps() -> None:
    for grid in (corridor_map(), doorway_map()):
        variants = generate_route_conditions(grid, (4, 1), (4, 13))
        assert set(variants) == set(ROUTE_CONDITIONS)
        assert all(path is not None for path in variants.values())


def test_generation_is_deterministic() -> None:
    grid = corridor_map()
    first = generate_route_conditions(grid, (4, 1), (4, 13))
    second = generate_route_conditions(grid, (4, 1), (4, 13))
    assert first == second


def test_left_and_right_variants_differ_and_are_mirror_sides() -> None:
    grid = corridor_map()
    variants = generate_route_conditions(grid, (4, 1), (4, 13))
    assert variants["left"] is not None and variants["right"] is not None
    assert variants["left"] != variants["right"]
    left_rows = {row for row, _col in variants["left"] if (row, _col) not in ((4, 1), (4, 13))}
    right_rows = {row for row, _col in variants["right"] if (row, _col) not in ((4, 1), (4, 13))}
    assert left_rows and right_rows
    # Left-hand positive convention: left of the start-to-goal axis is the
    # smaller-row side in this frame; shared endpoints are excluded above.
    assert max(left_rows) < min(right_rows)


def test_neutral_variant_stays_near_the_axis() -> None:
    grid = corridor_map()
    variants = generate_route_conditions(grid, (4, 1), (4, 13))
    neutral = variants["neutral"]
    assert neutral is not None
    rows = {row for row, _col in neutral}
    assert max(abs(row - 4) for row in rows) <= 1


def test_report_verifies_side_and_identity_on_both_maps() -> None:
    for grid in (corridor_map(), doorway_map()):
        report = route_condition_report(grid, (4, 1), (4, 13))
        assert report.status == "verified", report.as_dict()
        record = report.as_dict()
        assert record["schema"] == "route_condition_report.v1"
        for condition in ROUTE_CONDITIONS:
            entry = record["conditions"][condition]
            assert entry["status"] == "verified", entry
            assert entry["side"] == condition
            assert entry["identity"]


def test_doorway_conditions_have_distinct_identities() -> None:
    grid = doorway_map()
    report = route_condition_report(grid, (4, 1), (4, 13))
    conditions = report.conditions
    assert conditions["left"]["identity"] != conditions["right"]["identity"]
    assert conditions["left"]["identity"] is not None


def test_blocked_endpoints_fail_closed_unavailable() -> None:
    grid = corridor_map()
    grid[4, 1] = True
    variants = generate_route_conditions(grid, (4, 1), (4, 13))
    assert all(path is None for path in variants.values())
    report = route_condition_report(grid, (4, 1), (4, 13))
    assert report.status == "failed"
    assert all(entry["status"] == "unavailable" for entry in report.conditions.values())


def test_disconnected_goal_fails_closed_unavailable() -> None:
    grid = doorway_map()
    for opening in (1, 4, 7):
        grid[opening, 7] = True
    variants = generate_route_conditions(grid, (4, 1), (4, 13))
    assert all(path is None for path in variants.values())


def test_non_boolean_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D boolean"):
        generate_route_conditions(np.zeros((3, 3), dtype=int), (1, 1), (1, 2))


def test_condition_paths_are_contiguous_8_connected() -> None:
    grid = corridor_map()
    variants = generate_route_conditions(grid, (4, 1), (4, 13))
    for path in variants.values():
        assert path is not None
        for (r0, c0), (r1, c1) in pairwise(path):
            assert max(abs(r1 - r0), abs(c1 - c0)) == 1


def test_identity_is_stable_under_deterministic_replan() -> None:
    grid = corridor_map()
    report = route_condition_report(grid, (4, 1), (4, 13))
    for condition in ROUTE_CONDITIONS:
        assert report.conditions[condition]["identity"] is not None
