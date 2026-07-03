"""Tests for the Stanford Drone Dataset scenario importer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.tools import import_sdd_scenarios

if TYPE_CHECKING:
    from pathlib import Path


def _write_sdd_fixture(path: Path) -> None:
    """Write a tiny SDD-format annotation fixture."""
    path.write_text(
        "\n".join(
            [
                "1 0 0 10 10 0 0 0 0 Pedestrian",
                "1 5 0 15 10 5 0 0 0 Pedestrian",
                "1 10 0 20 10 10 0 0 0 Pedestrian",
                "1 15 0 25 10 15 0 0 0 Pedestrian",
                "2 100 100 110 110 0 0 0 0 Pedestrian",
                "2 100 105 110 115 5 0 0 0 Pedestrian",
                "2 100 110 110 120 10 0 0 0 Pedestrian",
                "2 100 115 110 125 15 0 0 0 Pedestrian",
                "3 0 0 10 10 0 1 0 0 Pedestrian",
                "4 0 0 10 10 0 0 0 0 Biker",
            ]
        ),
        encoding="utf-8",
    )


def _write_quoted_sdd_fixture(path: Path) -> None:
    """Write tiny SDD-format annotation fixture with quoted labels."""
    path.write_text(
        "\n".join(
            [
                '1 0 0 10 10 0 0 0 0 "Pedestrian"',
                '1 5 0 15 10 5 0 0 0 "Pedestrian"',
                '1 10 0 20 10 10 0 0 0 "Pedestrian"',
                '1 15 0 25 10 15 0 0 0 "Pedestrian"',
                '2 100 100 110 110 0 0 0 0 "Pedestrian"',
                '2 100 105 110 115 5 0 0 0 "Pedestrian"',
                '2 100 110 110 120 10 0 0 0 "Pedestrian"',
                '2 100 115 110 125 15 0 0 0 "Pedestrian"',
                '3 0 0 10 10 0 1 0 0 "Pedestrian"',
                '4 0 0 10 10 0 0 0 0 "Biker"',
            ]
        ),
        encoding="utf-8",
    )


def test_normalize_sdd_label_strips_one_matching_quote_pair() -> None:
    """SDD labels should strip only a matching wrapper quote pair."""
    assert import_sdd_scenarios.normalize_sdd_label('"Pedestrian"') == "Pedestrian"
    assert import_sdd_scenarios.normalize_sdd_label("'Pedestrian'") == "Pedestrian"
    assert import_sdd_scenarios.normalize_sdd_label(' "Pedestrian" ') == "Pedestrian"
    assert import_sdd_scenarios.normalize_sdd_label('"Pedestrian') == '"Pedestrian'


def test_load_sdd_points_accepts_user_label_for_quoted_rows(tmp_path: Path) -> None:
    """User-facing label names should match quoted SDD annotation rows."""
    annotations = tmp_path / "annotations.txt"
    _write_quoted_sdd_fixture(annotations)

    points = import_sdd_scenarios.load_sdd_points(annotations, label="Pedestrian")

    assert len(points) == 8
    assert {point.track_id for point in points} == {"1", "2"}
    assert {point.label for point in points} == {"Pedestrian"}


def test_load_sdd_points_accepts_legacy_quoted_label_argument(tmp_path: Path) -> None:
    """The prior quoted-label workaround remains accepted."""
    annotations = tmp_path / "annotations.txt"
    _write_quoted_sdd_fixture(annotations)

    points = import_sdd_scenarios.load_sdd_points(annotations, label='"Pedestrian"')

    assert len(points) == 8
    assert {point.label for point in points} == {"Pedestrian"}


def test_quoted_sdd_labels_are_normalized_in_import_metadata(tmp_path: Path) -> None:
    """Generated metadata should store normalized source labels."""
    annotations = tmp_path / "annotations.txt"
    _write_quoted_sdd_fixture(annotations)
    points = import_sdd_scenarios.load_sdd_points(annotations, label="Pedestrian")
    options = import_sdd_scenarios.ImportOptions(
        dataset_id="sdd_fixture",
        source_annotation=annotations,
        meters_per_pixel=0.1,
        frame_rate_hz=30.0,
        min_track_points=4,
        max_pedestrians=2,
        stride=2,
        max_waypoints=3,
        margin_m=1.0,
        y_flip_height_px=None,
    )

    map_payload, scenario_payload, provenance = import_sdd_scenarios.build_import_payload(
        points,
        options,
    )

    assert map_payload["single_pedestrians"][0]["metadata"]["source_label"] == "Pedestrian"
    assert (
        scenario_payload["scenarios"][0]["single_pedestrians"][0]["metadata"]["source_label"]
        == "Pedestrian"
    )
    assert provenance["pedestrians"][0]["source_label"] == "Pedestrian"
    assert provenance["pedestrians"][0]["source_raw_label"] == '"Pedestrian"'


def test_import_sdd_scenarios_writes_loadable_map_scenario_and_provenance(tmp_path: Path) -> None:
    """Importer output should be consumable by the existing scenario loader."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    points = import_sdd_scenarios.load_sdd_points(annotations, label="Pedestrian")
    options = import_sdd_scenarios.ImportOptions(
        dataset_id="sdd_fixture",
        source_annotation=annotations,
        meters_per_pixel=0.1,
        frame_rate_hz=30.0,
        min_track_points=4,
        max_pedestrians=2,
        stride=2,
        max_waypoints=3,
        margin_m=1.0,
        y_flip_height_px=None,
    )
    map_payload, scenario_payload, provenance = import_sdd_scenarios.build_import_payload(
        points,
        options,
    )
    paths = import_sdd_scenarios.write_import_outputs(
        out_dir=tmp_path,
        dataset_id="sdd_fixture",
        map_payload=map_payload,
        scenario_payload=scenario_payload,
        provenance=provenance,
    )

    scenarios = load_scenarios(paths["scenario"])
    assert len(scenarios) == 1
    assert scenarios[0]["metadata"]["dataset"] == "stanford_drone_dataset"
    assert scenarios[0]["metadata"]["license"] == import_sdd_scenarios.SDD_LICENSE

    config = build_robot_config_from_scenario(scenarios[0], scenario_path=paths["scenario"])
    map_def = next(iter(config.map_pool.map_defs.values()))

    assert len(map_def.single_pedestrians) == 2
    assert map_def.single_pedestrians[0].metadata["real_world_dataset"] == (
        "stanford_drone_dataset"
    )
    assert map_def.single_pedestrians[0].trajectory is not None
    assert json.loads(paths["provenance"].read_text(encoding="utf-8"))["dataset_id"] == (
        "sdd_fixture"
    )


def test_yaml_map_definitions_are_supported_by_scenario_loader(tmp_path: Path) -> None:
    """Generated YAML map definitions should load through scenario configs."""
    map_path = tmp_path / "map.yaml"
    map_path.write_text(
        yaml.safe_dump(
            {
                "x_margin": [0.0, 10.0],
                "y_margin": [0.0, 10.0],
                "obstacles": [],
                "robot_spawn_zones": [[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]],
                "robot_goal_zones": [[[8.0, 8.0], [9.0, 8.0], [9.0, 9.0]]],
                "ped_spawn_zones": [],
                "ped_goal_zones": [],
                "ped_crowded_zones": [],
                "robot_routes": [
                    {"spawn_id": 0, "goal_id": 0, "waypoints": [[1.0, 1.0], [9.0, 9.0]]}
                ],
                "ped_routes": [],
                "single_pedestrians": [
                    {
                        "id": "sdd_1",
                        "start": [3.0, 3.0],
                        "trajectory": [[4.0, 4.0], [5.0, 5.0]],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump({"scenarios": [{"name": "yaml_map", "map_file": "map.yaml"}]}),
        encoding="utf-8",
    )

    scenario = load_scenarios(scenario_path)[0]
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)

    assert config.map_id == "map"
    assert next(iter(config.map_pool.map_defs.values())).single_pedestrians[0].id == "sdd_1"
