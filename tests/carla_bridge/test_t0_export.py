"""CARLA-free tests for the T0 neutral replay export contract."""

from __future__ import annotations

import importlib
import json

import jsonschema
import pytest


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
