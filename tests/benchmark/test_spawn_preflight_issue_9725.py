"""Fast contract tests for the reset-only spawn-clearance preflight (issue #9725)."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark import spawn_preflight

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / spawn_preflight.DEFAULT_MATRIX


def test_parse_seeds_accepts_ranges_and_lists() -> None:
    """Seed specs expand ranges, deduplicate, and sort."""
    assert spawn_preflight._parse_seeds("111-113, 115,111,") == [111, 112, 113, 115]


def test_reset_only_diagnostic_reports_clear_cell_and_map_warnings() -> None:
    """The legacy reset diagnostic still exposes one selected cell and map warnings."""
    from argparse import Namespace

    report = spawn_preflight.run_preflight(
        Namespace(
            matrix=MATRIX,
            scenario=["classic_cross_trap_high"],
            seeds="111",
            workers=1,
            step_zero=True,
            dump_spawns=True,
        )
    )
    assert report["cell_count"] == 1
    assert report["overlap_count"] == 0
    assert report["step1_collision_count"] == 0
    assert report["relocated_cell_count"] == 1
    row = report["rows"][0]
    assert row["overlap"] is False
    assert row["robot_start"] and row["ped_positions"]
    kinds = {w["kind"] for w in report["map_warnings"]["classic_cross_trap_high"]}
    assert "ped_waypoint_on_robot_route_waypoint" in kinds


def test_preflight_rejects_unknown_scenario() -> None:
    """An unknown scenario name fails instead of silently checking nothing."""
    from argparse import Namespace

    with pytest.raises(SystemExit, match="unknown scenario"):
        spawn_preflight.run_preflight(
            Namespace(
                matrix=MATRIX,
                scenario=["no_such_scenario"],
                seeds="111",
                workers=1,
                step_zero=False,
                dump_spawns=False,
            )
        )
