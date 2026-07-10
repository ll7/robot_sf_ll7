"""Tests for the fail-closed OSM/GeoJSON segment-map converter."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.nav.geojson_map_builder import (
    geojson_to_map_definition,
    geojson_to_map_structure,
    write_segment_map,
)
from robot_sf.nav.geojson_map_provenance import validate_import_provenance
from robot_sf.training.scenario_loader import resolve_map_definition
from scripts.validation.check_geojson_import import main as check_geojson_import

FIXTURE = Path("tests/fixtures/maps/geojson/annotated_plaza.geojson")


def test_geojson_converter_emits_loadable_segment_map(tmp_path: Path) -> None:
    """An annotated GeoJSON plaza becomes a local-metre YAML map with obstacle segments."""
    output = write_segment_map(FIXTURE, tmp_path / "plaza.yaml")
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert data["x_margin"][1] > 40.0
    assert data["y_margin"][1] > 30.0
    assert data["obstacles"]
    assert len(data["robot_spawn_zones"]) == 1
    assert len(data["robot_goal_zones"]) == 1
    assert len(data["robot_routes"]) == 1

    map_def = resolve_map_definition(str(output), scenario_path=tmp_path / "scenario.yaml")
    assert map_def is not None
    assert map_def.obstacles_pysf
    assert len(map_def.robot_routes) == 2  # Runtime YAML loader adds the reverse route.


def test_geojson_converter_returns_runtime_map_definition() -> None:
    """The direct conversion API has the same route and obstacle contract as YAML loading."""
    map_def = geojson_to_map_definition(FIXTURE)

    assert map_def.width > 40.0
    assert map_def.height > 30.0
    assert len(map_def.robot_routes) == 2
    assert map_def.obstacles


def test_geojson_converter_requires_explicit_route_metadata(tmp_path: Path) -> None:
    """Public geometry without a declared route cannot silently become a scenario."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["features"] = data["features"][:-1]
    missing_route = tmp_path / "missing-route.geojson"
    missing_route.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="No robot_route features"):
        geojson_to_map_structure(missing_route)


def test_geojson_converter_requires_zone_metadata(tmp_path: Path) -> None:
    """A route alone is not a runnable scenario without named endpoint zones."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["features"] = [
        feature
        for feature in data["features"]
        if feature["properties"].get("robot_sf_role") != "robot_goal"
    ]
    missing_goal = tmp_path / "missing-goal.geojson"
    missing_goal.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="robot_sf_role=robot_goal"):
        geojson_to_map_structure(missing_goal)


def test_public_import_checker_requires_matching_checksum_and_loads_output(tmp_path: Path) -> None:
    """Public imports retain provenance and only pass after the derived map loads."""
    checksum = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    manifest = tmp_path / "provenance.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "robot_sf.geojson_import_provenance.v1",
                "classification": "exploratory_only",
                "source": {
                    "url": "https://www.openstreetmap.org/copyright",
                    "accessed_on": "2026-07-10",
                    "license": "ODbL-1.0",
                    "citation": "OpenStreetMap contributors",
                },
                "raw_input": {"sha256": checksum},
            }
        ),
        encoding="utf-8",
    )

    assert validate_import_provenance(manifest, FIXTURE)["classification"] == "exploratory_only"
    output = tmp_path / "plaza.yaml"
    assert check_geojson_import([str(FIXTURE), str(manifest), str(output)]) == 0
    assert output.is_file()

    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["raw_input"]["sha256"] = "0" * 64
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 does not match"):
        validate_import_provenance(manifest, FIXTURE)

    data["raw_input"]["sha256"] = checksum
    data["classification"] = "benchmark_ready"
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="classification must be 'exploratory_only'"):
        validate_import_provenance(manifest, FIXTURE)
