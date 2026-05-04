"""CARLA-free tests for the T0 neutral replay export contract."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import jsonschema
import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.robot.differential_drive import DifferentialDriveSettings


def _minimal_payload() -> dict:
    """Return a minimal payload that future CARLA replay can consume."""
    return {
        "schema_version": "carla-replay-export.v1",
        "mode": "neutral-export",
        "scenario": {
            "id": "unit_crossing",
            "source_config": "configs/scenarios/unit_crossing.yaml",
            "map_id": "unit_map",
            "certificate": {
                "schema_version": "scenario_cert.v1",
                "status": "passed",
                "source": "output/scenario_cert/unit_crossing.json",
            },
        },
        "robot": {
            "start": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "goal": {"x": 4.0, "y": 0.0, "theta": 0.0},
            "footprint": {"radius_m": 0.3},
            "kinematics": {"model": "unicycle", "max_speed_mps": 1.0},
        },
        "pedestrians": [
            {
                "id": "ped_0",
                "start": {"x": 2.0, "y": -1.0, "theta": 1.57},
                "route": [{"x": 2.0, "y": 1.0}],
                "timing": {"start_delay_s": 0.0},
            }
        ],
        "static_geometry": {
            "obstacles": [],
            "route_topology_ref": "maps/svg_maps/unit_map.svg",
        },
        "simulation": {
            "dt_s": 0.1,
            "horizon_s": 10.0,
            "termination": ["success", "collision", "timeout"],
        },
        "metrics": {
            "trajectory_fields": [
                "success",
                "collision",
                "min_distance",
                "ttc",
                "comfort",
                "jerk",
                "curvature",
                "intervention_rate",
            ]
        },
        "provenance": {
            "robot_sf_commit": "abc123",
            "created_by": "unit-test",
            "certificate_generator": "scenario_cert.v1",
        },
    }


def _zone(
    x: float, y: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return a tiny triangular zone around one point."""
    return ((x - 0.2, y - 0.2), (x + 0.2, y - 0.2), (x + 0.2, y + 0.2))


def _bounds(width: float, height: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return rectangular map bounds as line segments."""
    return [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]


def _programmatic_map() -> MapDefinition:
    """Return a small map with one robot route, one obstacle, and scripted pedestrians."""
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.0, 1.0), (5.0, 1.0)],
        spawn_zone=_zone(1.0, 1.0),
        goal_zone=_zone(5.0, 1.0),
        source_label="unit_route",
    )
    return MapDefinition(
        width=6.0,
        height=4.0,
        obstacles=[Obstacle([(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)])],
        robot_spawn_zones=[route.spawn_zone],
        ped_spawn_zones=[],
        robot_goal_zones=[route.goal_zone],
        bounds=_bounds(6.0, 4.0),
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[
            SinglePedestrianDefinition(
                id="ped_goal",
                start=(3.0, 0.5),
                goal=(3.0, 3.5),
                speed_m_s=1.1,
            ),
            SinglePedestrianDefinition(
                id="ped_route",
                start=(4.0, 0.5),
                trajectory=[(4.0, 2.0), (4.0, 3.5)],
            ),
        ],
    )


def test_bridge_import_does_not_require_carla() -> None:
    """The bridge package itself must remain importable without CARLA installed."""
    module = importlib.import_module("robot_sf_carla_bridge")

    assert module.EXPORT_SCHEMA_VERSION == "carla-replay-export.v1"


def test_minimal_t0_payload_validates_against_schema() -> None:
    """A minimal neutral export payload should validate without CARLA runtime."""
    from robot_sf_carla_bridge.export import load_export_schema, validate_export_payload

    payload = _minimal_payload()

    schema = load_export_schema()
    jsonschema.validate(payload, schema)
    validate_export_payload(payload)


def test_invalid_status_is_rejected_by_schema() -> None:
    """Status/mode fields should stay explicit instead of accepting fallback claims."""
    from robot_sf_carla_bridge.export import validate_export_payload

    payload = _minimal_payload()
    payload["mode"] = "fallback"

    with pytest.raises(jsonschema.ValidationError, match="fallback"):
        validate_export_payload(payload)


def test_export_payload_round_trip_writes_stable_json(tmp_path) -> None:
    """Export helpers should write deterministic JSON that validates after reload."""
    from robot_sf_carla_bridge.export import validate_export_payload, write_export_payload

    payload = _minimal_payload()
    output_path = tmp_path / "carla_export.json"

    write_export_payload(payload, output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == payload
    validate_export_payload(loaded)


def test_build_export_payload_from_typed_sections_validates() -> None:
    """Typed builder sections should produce the same schema-valid T0 payload shape."""
    from robot_sf_carla_bridge import (
        CertificateRef,
        PedestrianReplaySpec,
        Pose2D,
        RobotReplaySpec,
        ScenarioReplayRef,
        SimulationSpec,
        build_export_payload,
        validate_export_payload,
    )

    payload = build_export_payload(
        scenario=ScenarioReplayRef(
            scenario_id="unit_crossing",
            source_config="configs/scenarios/unit_crossing.yaml",
            map_id="unit_map",
            certificate=CertificateRef(status="passed", source="output/cert.json"),
        ),
        robot=RobotReplaySpec(
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Pose2D(4.0, 0.0, 0.0),
            radius_m=0.3,
            kinematics={"model": "unicycle", "max_speed_mps": 1.0},
        ),
        pedestrians=[
            PedestrianReplaySpec(
                ped_id="ped_0",
                start=Pose2D(2.0, -1.0, 1.57),
                route=[Pose2D(2.0, 1.0)],
                timing={"start_delay_s": 0.0},
            )
        ],
        static_geometry={"obstacles": [], "route_topology_ref": "maps/svg_maps/unit_map.svg"},
        simulation=SimulationSpec(dt_s=0.1, horizon_s=10.0),
        trajectory_fields=["success", "collision", "min_distance"],
        provenance={
            "robot_sf_commit": "abc123",
            "created_by": "unit-test",
            "certificate_generator": "scenario_cert.v1",
        },
    )

    validate_export_payload(payload)
    assert payload["scenario"]["certificate"]["schema_version"] == "scenario_cert.v1"
    assert payload["pedestrians"][0]["route"] == [{"x": 2.0, "y": 1.0}]


def test_builder_output_round_trip_writes_stable_json(tmp_path) -> None:
    """Builder output should round-trip through the public JSON helpers."""
    from robot_sf_carla_bridge import (
        CertificateRef,
        PedestrianReplaySpec,
        Pose2D,
        RobotReplaySpec,
        ScenarioReplayRef,
        SimulationSpec,
        build_export_payload,
        read_export_payload,
        write_export_payload,
    )

    payload = build_export_payload(
        scenario=ScenarioReplayRef(
            scenario_id="unit_crossing",
            source_config="configs/scenarios/unit_crossing.yaml",
            map_id="unit_map",
            certificate=CertificateRef(status="passed"),
        ),
        robot=RobotReplaySpec(
            start=Pose2D(0.0, 0.0),
            goal=Pose2D(4.0, 0.0),
            radius_m=0.3,
            kinematics={"model": "unicycle"},
        ),
        pedestrians=[
            PedestrianReplaySpec(
                ped_id="ped_0",
                start=Pose2D(2.0, -1.0),
                route=[Pose2D(2.0, 1.0)],
            )
        ],
        static_geometry={"obstacles": []},
        simulation=SimulationSpec(dt_s=0.1, horizon_s=10.0),
        trajectory_fields=["success"],
        provenance={
            "robot_sf_commit": "abc123",
            "created_by": "unit-test",
            "certificate_generator": "scenario_cert.v1",
        },
    )

    output_path = write_export_payload(payload, tmp_path / "builder_export.json")
    loaded = read_export_payload(output_path)

    assert loaded == payload


def test_read_export_payload_rejects_invalid_json_payload(tmp_path) -> None:
    """Read helper should validate JSON loaded from disk before returning it."""
    from robot_sf_carla_bridge import read_export_payload, write_export_payload

    output_path = write_export_payload(_minimal_payload(), tmp_path / "invalid_export.json")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    loaded["mode"] = "fallback"
    output_path.write_text(json.dumps(loaded), encoding="utf-8")

    with pytest.raises(jsonschema.ValidationError, match="fallback"):
        read_export_payload(output_path)


def test_build_export_payload_from_map_definition_validates() -> None:
    """Certified map definitions should convert to schema-valid neutral export payloads."""
    from robot_sf_carla_bridge import build_export_payload_from_map_definition

    payload = build_export_payload_from_map_definition(
        map_def=_programmatic_map(),
        certificate={"schema_version": "scenario_cert.v1", "classification": "valid"},
        scenario_id="unit_map_definition",
        source_config=Path("configs/scenarios/unit.yaml"),
        map_id="unit_map",
        robot_radius_m=0.35,
        robot_kinematics={"model": "differential_drive", "max_speed_mps": 1.0},
        dt_s=0.1,
        horizon_s=12.0,
        provenance={
            "robot_sf_commit": "abc123",
            "created_by": "unit-test",
            "certificate_generator": "scenario_cert.v1",
        },
    )

    assert payload["scenario"]["certificate"]["status"] == "valid"
    assert payload["robot"]["start"] == {"x": 1.0, "y": 1.0}
    assert payload["robot"]["goal"] == {"x": 5.0, "y": 1.0}
    assert payload["pedestrians"][0]["route"] == [{"x": 3.0, "y": 3.5}]
    assert payload["pedestrians"][0]["timing"]["speed_m_s"] == 1.1
    assert payload["pedestrians"][1]["route"] == [{"x": 4.0, "y": 2.0}, {"x": 4.0, "y": 3.5}]
    assert payload["static_geometry"]["obstacles"][0]["type"] == "polygon"


def test_build_export_payload_from_map_definition_rejects_excluded_certificate() -> None:
    """T0 export should fail closed when certification excluded the scenario."""
    from robot_sf_carla_bridge import build_export_payload_from_map_definition

    with pytest.raises(ValueError, match="excluded"):
        build_export_payload_from_map_definition(
            map_def=_programmatic_map(),
            certificate={
                "schema_version": "scenario_cert.v1",
                "classification": "invalid",
                "benchmark_eligibility": "excluded",
            },
            scenario_id="unit_map_definition",
            source_config="configs/scenarios/unit.yaml",
            map_id="unit_map",
            robot_radius_m=0.35,
            robot_kinematics={"model": "differential_drive"},
            dt_s=0.1,
            horizon_s=12.0,
            provenance={
                "robot_sf_commit": "abc123",
                "created_by": "unit-test",
                "certificate_generator": "scenario_cert.v1",
            },
        )


def test_build_export_payload_from_scenario_entry_validates(monkeypatch) -> None:
    """Scenario-loader entries should export through certification and map-definition adapter."""
    import robot_sf_carla_bridge.export as export_module
    from robot_sf_carla_bridge import build_export_payload_from_scenario_entry

    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"unit_map": _programmatic_map()}),
        map_id="unit_map",
        robot_config=DifferentialDriveSettings(radius=0.35, max_linear_speed=1.25),
    )
    monkeypatch.setattr(
        export_module,
        "_build_robot_config_for_scenario_entry",
        lambda scenario, scenario_path: config,
    )
    monkeypatch.setattr(
        export_module,
        "_certificate_payload_for_scenario_entry",
        lambda scenario, scenario_path: {
            "schema_version": "scenario_cert.v1",
            "classification": "valid",
            "source": str(scenario_path),
        },
    )

    payload = build_export_payload_from_scenario_entry(
        {"name": "unit_scenario", "map_id": "unit_map"},
        scenario_path=Path("configs/scenarios/unit.yaml"),
        provenance={
            "robot_sf_commit": "abc123",
            "created_by": "unit-test",
            "certificate_generator": "scenario_cert.v1",
        },
    )

    assert payload["scenario"]["id"] == "unit_scenario"
    assert payload["scenario"]["source_config"] == "configs/scenarios/unit.yaml"
    assert payload["scenario"]["map_id"] == "unit_map"
    assert payload["robot"]["footprint"]["radius_m"] == 0.35
    assert payload["robot"]["kinematics"]["model"] == "differential_drive"
    assert payload["robot"]["kinematics"]["max_speed_mps"] == 1.25
    assert payload["simulation"]["dt_s"] == 0.1


def test_build_export_payload_from_scenario_entry_rejects_excluded_certificate(monkeypatch) -> None:
    """Scenario-entry export should fail closed when certification excludes the scenario."""
    import robot_sf_carla_bridge.export as export_module
    from robot_sf_carla_bridge import build_export_payload_from_scenario_entry

    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"unit_map": _programmatic_map()})
    )
    monkeypatch.setattr(
        export_module,
        "_build_robot_config_for_scenario_entry",
        lambda scenario, scenario_path: config,
    )
    monkeypatch.setattr(
        export_module,
        "_certificate_payload_for_scenario_entry",
        lambda scenario, scenario_path: {
            "schema_version": "scenario_cert.v1",
            "classification": "invalid",
            "benchmark_eligibility": "excluded",
        },
    )

    with pytest.raises(ValueError, match="excluded"):
        build_export_payload_from_scenario_entry(
            {"name": "unit_scenario", "map_id": "unit_map"},
            scenario_path=Path("configs/scenarios/unit.yaml"),
            provenance={
                "robot_sf_commit": "abc123",
                "created_by": "unit-test",
                "certificate_generator": "scenario_cert.v1",
            },
        )


def test_build_export_payloads_from_scenario_file_preserves_manifest_order(
    tmp_path,
    monkeypatch,
) -> None:
    """Scenario files should export each loaded entry in deterministic order."""
    import robot_sf_carla_bridge.export as export_module
    from robot_sf_carla_bridge import build_export_payloads_from_scenario_file

    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "scenarios:\n"
        "  - name: first\n"
        "    map_id: unit_map\n"
        "  - name: second\n"
        "    map_id: unit_map\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        export_module,
        "_load_scenario_entries",
        lambda path: [
            {"name": "first", "map_id": "unit_map"},
            {"name": "second", "map_id": "unit_map"},
        ],
    )

    def fake_entry_export(scenario, *, scenario_path, provenance, **kwargs):
        payload = _minimal_payload()
        payload["scenario"]["id"] = scenario["name"]
        payload["scenario"]["source_config"] = Path(scenario_path).as_posix()
        payload["provenance"] = dict(provenance)
        return payload

    monkeypatch.setattr(
        export_module, "build_export_payload_from_scenario_entry", fake_entry_export
    )

    records = build_export_payloads_from_scenario_file(
        scenario_path,
        provenance={
            "robot_sf_commit": "abc123",
            "created_by": "unit-test",
            "certificate_generator": "scenario_cert.v1",
        },
    )

    assert [record["scenario_id"] for record in records] == ["first", "second"]
    assert records[0]["payload"]["scenario"]["source_config"] == scenario_path.as_posix()
    assert records[1]["payload"]["provenance"]["created_by"] == "unit-test"


def test_write_export_records_writes_payloads_and_manifest(tmp_path) -> None:
    """Export records should persist deterministic JSON files plus a small manifest."""
    from robot_sf_carla_bridge import read_export_payload, write_export_records

    first = _minimal_payload()
    first["scenario"]["id"] = "first scenario"
    second = _minimal_payload()
    second["scenario"]["id"] = "second/slash"

    manifest = write_export_records(
        [
            {"scenario_id": "first scenario", "payload": first},
            {"scenario_id": "second/slash", "payload": second},
        ],
        tmp_path / "exports",
    )

    assert manifest["schema_version"] == "carla-replay-export-manifest.v1"
    assert [entry["scenario_id"] for entry in manifest["exports"]] == [
        "first scenario",
        "second/slash",
    ]
    assert [entry["path"] for entry in manifest["exports"]] == [
        "first_scenario.json",
        "second_slash.json",
    ]
    assert read_export_payload(tmp_path / "exports" / "first_scenario.json")["scenario"]["id"] == (
        "first scenario"
    )
    assert (tmp_path / "exports" / "manifest.json").exists()


def test_read_export_manifest_round_trips_writer_output(tmp_path) -> None:
    """Manifest reader should validate the writer output without loading payload files."""
    from robot_sf_carla_bridge import read_export_manifest, write_export_records

    payload = _minimal_payload()
    manifest = write_export_records(
        [{"scenario_id": "unit scenario", "payload": payload}],
        tmp_path / "exports",
    )

    loaded = read_export_manifest(tmp_path / "exports" / "manifest.json")

    assert loaded == manifest
    assert loaded["exports"] == [{"scenario_id": "unit scenario", "path": "unit_scenario.json"}]


def test_read_export_manifest_rejects_invalid_manifest_shape(tmp_path) -> None:
    """Manifest reader should fail clearly for unsupported versions and malformed entries."""
    from robot_sf_carla_bridge import read_export_manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "carla-replay-export-manifest.v0",
                "exports": [{"scenario_id": "unit", "path": "unit.json"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest schema_version"):
        read_export_manifest(manifest_path)

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "carla-replay-export-manifest.v1",
                "exports": [{"scenario_id": "unit"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exports\\[0\\].path"):
        read_export_manifest(manifest_path)


def test_resolve_export_manifest_payload_paths_returns_local_batch_paths(tmp_path) -> None:
    """Manifest path resolver should join payload paths relative to the manifest file."""
    from robot_sf_carla_bridge import resolve_export_manifest_payload_paths, write_export_records

    payload = _minimal_payload()
    write_export_records(
        [{"scenario_id": "unit scenario", "payload": payload}],
        tmp_path / "exports",
    )

    records = resolve_export_manifest_payload_paths(tmp_path / "exports" / "manifest.json")

    assert records == [
        {
            "scenario_id": "unit scenario",
            "path": tmp_path / "exports" / "unit_scenario.json",
        }
    ]


def test_resolve_export_manifest_payload_paths_rejects_unsafe_paths(tmp_path) -> None:
    """Manifest payload paths should stay scoped to the manifest directory."""
    from robot_sf_carla_bridge import resolve_export_manifest_payload_paths

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "carla-replay-export-manifest.v1",
                "exports": [{"scenario_id": "absolute", "path": "/tmp/payload.json"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absolute"):
        resolve_export_manifest_payload_paths(manifest_path)

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "carla-replay-export-manifest.v1",
                "exports": [{"scenario_id": "escape", "path": "../payload.json"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent"):
        resolve_export_manifest_payload_paths(manifest_path)


def test_builder_invalid_radius_fails_schema_validation() -> None:
    """Builder validation should fail through the schema for invalid payload values."""
    from robot_sf_carla_bridge import (
        CertificateRef,
        PedestrianReplaySpec,
        Pose2D,
        RobotReplaySpec,
        ScenarioReplayRef,
        SimulationSpec,
        build_export_payload,
    )

    with pytest.raises(jsonschema.ValidationError, match="radius_m"):
        build_export_payload(
            scenario=ScenarioReplayRef(
                scenario_id="unit_crossing",
                source_config="configs/scenarios/unit_crossing.yaml",
                map_id="unit_map",
                certificate=CertificateRef(status="passed"),
            ),
            robot=RobotReplaySpec(
                start=Pose2D(0.0, 0.0),
                goal=Pose2D(4.0, 0.0),
                radius_m=0.0,
                kinematics={"model": "unicycle"},
            ),
            pedestrians=[
                PedestrianReplaySpec(
                    ped_id="ped_0",
                    start=Pose2D(2.0, -1.0),
                    route=[Pose2D(2.0, 1.0)],
                )
            ],
            static_geometry={"obstacles": []},
            simulation=SimulationSpec(dt_s=0.1, horizon_s=10.0),
            trajectory_fields=["success"],
            provenance={
                "robot_sf_commit": "abc123",
                "created_by": "unit-test",
                "certificate_generator": "scenario_cert.v1",
            },
        )


def test_missing_carla_reports_not_available(monkeypatch) -> None:
    """Missing CARLA should be a status object, not an import-time crash."""
    from robot_sf_carla_bridge.availability import check_carla_availability

    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "carla" else object()
    )

    status = check_carla_availability()

    assert status["status"] == "not-available"
    assert "carla" in status["reason"].lower()
