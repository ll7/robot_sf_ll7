"""CARLA T1 oracle replay smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest


def _minimal_t0_payload(scenario_id: str = "unit_crossing") -> dict:
    """Return a schema-valid minimal T0 export payload."""

    from robot_sf_carla_bridge import (
        CertificateRef,
        PedestrianReplaySpec,
        Pose2D,
        RobotReplaySpec,
        ScenarioReplayRef,
        SimulationSpec,
        Waypoint2D,
        build_export_payload,
    )

    return build_export_payload(
        scenario=ScenarioReplayRef(
            scenario_id=scenario_id,
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
                route=[Waypoint2D(2.0, 1.0)],
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


def _write_manifest(tmp_path, records: list[dict]) -> str:
    """Write T0 export records and return the manifest path."""

    from robot_sf_carla_bridge import write_export_records

    output_dir = tmp_path / "exports"
    write_export_records(records, output_dir)
    return (output_dir / "manifest.json").as_posix()


def test_replay_t1_oracle_smoke_main_fails_closed_without_carla(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """T1 smoke CLI should validate inputs, then fail closed when CARLA is unavailable."""

    import robot_sf_carla_bridge.replay_smoke as replay_module
    from robot_sf_carla_bridge.availability import CarlaUnavailableError
    from robot_sf_carla_bridge.cli import replay_t1_oracle_smoke_main

    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )

    monkeypatch.setattr(
        replay_module,
        "require_carla",
        lambda: (_ for _ in ()).throw(
            CarlaUnavailableError(
                "CARLA Python API package 'carla' is not importable. Install CARLA and ensure "
                "its Python API is on PYTHONPATH before using CARLA replay entry points."
            )
        ),
    )

    exit_code = replay_t1_oracle_smoke_main(["--manifest", manifest_path, "--json"])

    assert exit_code == 1
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "not-available"
    assert status["dependency"] == "carla"
    assert "CARLA Python API package 'carla' is not importable" in status["reason"]
    assert "PYTHONPATH" in status["action"]


def test_build_t1_oracle_replay_smoke_setup_reports_setup_only_summary(
    tmp_path,
    monkeypatch,
) -> None:
    """With a CARLA module available, the runner should emit a setup-only oracle summary."""

    import robot_sf_carla_bridge.replay_smoke as replay_module
    from robot_sf_carla_bridge import build_t1_oracle_replay_smoke_setup

    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )
    monkeypatch.setattr(replay_module, "require_carla", lambda: ModuleType("carla"))

    summary = build_t1_oracle_replay_smoke_setup(manifest_path)

    assert summary["status"] == "oracle-replay"
    assert summary["stage"] == "setup-only"
    assert summary["selected_payload"]["scenario_id"] == "unit_crossing"
    assert summary["catalog"]["t0_export_payload_schema_version"] == "carla-replay-export.v1"
    assert summary["scenario"]["certificate_status"] == "passed"
    assert summary["pedestrian_count"] == 1
    assert summary["boundary"] == {
        "full_metrics_parity": False,
        "multi_map_replay": False,
        "long_running_benchmark": False,
        "note": "setup-only smoke; no CARLA benchmark readiness or parity claim",
    }


def test_t1_oracle_replay_smoke_selects_requested_scenario(tmp_path, monkeypatch) -> None:
    """Scenario selection should allow one manifest to hold multiple T0 exports."""

    import robot_sf_carla_bridge.replay_smoke as replay_module
    from robot_sf_carla_bridge import build_t1_oracle_replay_smoke_setup

    manifest_path = _write_manifest(
        tmp_path,
        [
            {"scenario_id": "first", "payload": _minimal_t0_payload("first")},
            {"scenario_id": "second", "payload": _minimal_t0_payload("second")},
        ],
    )
    monkeypatch.setattr(replay_module, "require_carla", lambda: ModuleType("carla"))

    summary = build_t1_oracle_replay_smoke_setup(manifest_path, scenario_id="second")

    assert summary["selected_payload"]["scenario_id"] == "second"
    assert summary["selected_payload"]["payload_index"] == 1
    assert summary["scenario"]["id"] == "second"


def test_select_t0_export_payload_reads_only_selected_payload(tmp_path, monkeypatch) -> None:
    """Selecting one scenario should not deserialize every payload in the manifest."""

    import robot_sf_carla_bridge.replay_smoke as replay_module

    manifest_path = _write_manifest(
        tmp_path,
        [
            {"scenario_id": "first", "payload": _minimal_t0_payload("first")},
            {"scenario_id": "second", "payload": _minimal_t0_payload("second")},
        ],
    )

    loaded_paths: list[Path] = []
    real_read_export_payload = replay_module.read_export_payload

    def _counting_read_export_payload(path: str | Path) -> dict:
        loaded_paths.append(Path(path))
        return real_read_export_payload(path)

    monkeypatch.setattr(replay_module, "read_export_payload", _counting_read_export_payload)

    record = replay_module.select_t0_export_payload(manifest_path, scenario_id="second")

    assert record["scenario_id"] == "second"
    assert loaded_paths == [Path(record["path"])]


def test_t1_oracle_replay_smoke_rejects_catalog_payload_version_mismatch(
    monkeypatch,
) -> None:
    """Payload validation should be tied to the current schema catalog contract."""

    import robot_sf_carla_bridge.replay_smoke as replay_module
    from robot_sf_carla_bridge import validate_t1_replay_catalog_payload

    monkeypatch.setattr(
        replay_module,
        "list_carla_bridge_schema_catalog",
        lambda: {
            "schema_version": "carla-bridge-schema-catalog.v1",
            "schemas": [
                {
                    "name": "t0_export_payload",
                    "loader": "load_export_schema",
                    "schema_version": "carla-replay-export.v999",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="does not match catalog version"):
        validate_t1_replay_catalog_payload(_minimal_t0_payload())
