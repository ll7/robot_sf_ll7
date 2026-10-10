"""Endpoint audits include both actors' actual radii and round tangency."""

from pathlib import Path

import pytest

from robot_sf.evidence.writers import write_json
from scripts.validation.check_scenario_archetype_geometry import inspect_release_zones


@pytest.mark.parametrize("lane_y, intersects", [(5.0, True), (5.9, True), (6.0, False)])
def test_robot_radius_detects_separated_centres_and_tangency(tmp_path, lane_y, intersects):
    manifest = tmp_path / "probe.yaml"
    write_json(
        manifest,
        {
            "scenarios": [
                {
                    "name": "radius_probe",
                    "map_file": str(
                        Path(
                            "maps/successor_svg_maps/issue_10063_francis2023_ped_overtaking_safe_spawn_v1.svg"
                        ).resolve()
                    ),
                    "simulation_config": {"ped_density": 0.0},
                    "single_pedestrians": [
                        {"id": "h1", "goal": None, "trajectory": [[2, lane_y], [8, lane_y]]}
                    ],
                }
            ]
        },
    )
    row = inspect_release_zones([manifest])[0]
    hits = [h for h in row["intersections"] if h["actor"] == "h1"]
    assert bool(hits) == intersects
    if not intersects:
        return
    hit = hits[0]
    assert hit["distance_m"] == pytest.approx(lane_y - 4.5)
    assert hit["evidence"]["robot_radius_m"] == 1.0
    assert hit["evidence"]["ped_radius_m"] == 0.4
