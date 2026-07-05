"""Tests for SocNavBench traversible-to-SVG conversion."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.maps.verification.svg_inspection import inspect_svg
from robot_sf.nav.svg_map_parser import convert_map
from scripts.tools import convert_socnavbench_traversible_to_svg as converter

if TYPE_CHECKING:
    from pathlib import Path


def _write_traversible(path: Path, traversible: np.ndarray, *, resolution: float = 2.0) -> None:
    """Write a minimal SocNavBench-compatible traversible pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {"resolution": resolution, "traversible": traversible},
            handle,
            protocol=2,
        )


def test_fixture_traversible_converts_to_parser_valid_svg(tmp_path: Path) -> None:
    """A staged traversible fixture should produce Robot SF parser-facing map primitives."""

    pkl_path = tmp_path / "data.pkl"
    output_svg = tmp_path / "socnavbench_eth.svg"
    _write_traversible(
        pkl_path,
        np.array(
            [
                [False, False, False, False, False, False],
                [True, True, True, False, True, True],
                [False, False, True, True, True, False],
                [False, False, False, False, False, False],
            ],
            dtype=bool,
        ),
    )

    exit_code = converter.main(
        [
            "--input-pkl",
            str(pkl_path),
            "--output-svg",
            str(output_svg),
        ]
    )

    assert exit_code == 0
    map_def = convert_map(str(output_svg))
    assert map_def.obstacles
    assert map_def.robot_spawn_zones
    assert map_def.robot_goal_zones
    assert map_def.robot_routes
    assert map_def.ped_spawn_zones
    assert map_def.ped_goal_zones
    assert map_def.ped_routes

    inspection = inspect_svg(output_svg)
    assert inspection.capability_metadata.has_robot_runtime_routes is True
    assert inspection.capability_metadata.has_pedestrian_runtime_routes is True
    assert inspection.capability_metadata.has_explicit_robot_runtime_zones is True
    assert inspection.capability_metadata.has_explicit_pedestrian_runtime_zones is True


def test_dry_run_reports_ready_without_writing_svg(tmp_path: Path) -> None:
    """Dry-run mode validates conversion shape but leaves output untouched."""

    pkl_path = tmp_path / "data.pkl"
    output_svg = tmp_path / "socnavbench_eth.svg"
    report_json = tmp_path / "report.json"
    _write_traversible(pkl_path, np.ones((3, 4), dtype=bool))

    exit_code = converter.main(
        [
            "--input-pkl",
            str(pkl_path),
            "--output-svg",
            str(output_svg),
            "--report-json",
            str(report_json),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not output_svg.exists()
    assert '"status": "dry_run_ready"' in report_json.read_text(encoding="utf-8")


def test_missing_official_source_fails_closed_without_placeholder(tmp_path: Path) -> None:
    """Absent staged traversible should exit blocked and not write a placeholder SVG."""

    output_svg = tmp_path / "socnavbench_eth.svg"
    report_json = tmp_path / "blocked.json"

    exit_code = converter.main(
        [
            "--input-pkl",
            str(tmp_path / "missing" / "data.pkl"),
            "--output-svg",
            str(output_svg),
            "--report-json",
            str(report_json),
        ]
    )

    assert exit_code == converter.EXIT_BLOCKED
    assert not output_svg.exists()
    report_text = report_json.read_text(encoding="utf-8")
    assert '"status": "blocked_missing_traversible"' in report_text
    assert '"conversion_ready": false' in report_text


def test_invalid_traversible_payload_fails_closed(tmp_path: Path) -> None:
    """Malformed source pickles should fail closed instead of emitting a map."""

    pkl_path = tmp_path / "data.pkl"
    output_svg = tmp_path / "socnavbench_eth.svg"
    with pkl_path.open("wb") as handle:
        pickle.dump({"resolution": 2.0}, handle, protocol=2)

    exit_code = converter.main(
        [
            "--input-pkl",
            str(pkl_path),
            "--output-svg",
            str(output_svg),
        ]
    )

    assert exit_code == converter.EXIT_BLOCKED
    assert not output_svg.exists()
