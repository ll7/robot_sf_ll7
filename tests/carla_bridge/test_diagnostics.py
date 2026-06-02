"""Tests for conservative CARLA replay diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jsonschema

from robot_sf_carla_bridge.diagnostics import (
    build_carla_replay_diagnostics,
    load_carla_replay_diagnostics_schema,
    write_carla_replay_diagnostics_outputs,
)
from scripts.carla_bridge.diagnose_replay_semantics import main as diagnostics_cli_main


def _robot_sf_record() -> dict[str, object]:
    return {
        "metrics": {
            "success": True,
            "collision": False,
            "min_distance_m": 1.2,
        }
    }


def _carla_summary() -> dict[str, object]:
    return {
        "schema_version": "carla-t1-oracle-live-replay.v1",
        "status": "oracle-replay",
        "mode": "oracle-replay",
        "carla": {"map": "Town10HD_Opt", "server_version": "0.9.16"},
        "actors": {"static_obstacles": 1, "robot": 1, "pedestrians": 2},
        "boundary": {"static_geometry_replay": True},
        "coordinate_alignment": {"replay_mode": "native", "carla_map": "Town10HD_Opt"},
        "trajectory": {"steps_requested": 4, "steps_replayed": 4},
        "metrics": {
            "success": True,
            "collision": False,
            "min_distance_m": 1.0,
        },
    }


def _row_by_axis(rows: list[dict[str, object]], axis: str) -> dict[str, object]:
    return next(row for row in rows if row["axis"] == axis)


def test_build_carla_replay_diagnostics_classifies_available_surfaces() -> None:
    """Nominal fake summaries should expose available rows without live CARLA."""
    report = build_carla_replay_diagnostics(_robot_sf_record(), _carla_summary())

    jsonschema.validate(report, load_carla_replay_diagnostics_schema())
    assert report["schema_version"] == "carla-replay-diagnostics.v1"
    assert report["status"] == "available"
    assert "not simulator-equivalence" in report["interpretation_boundary"]
    assert (
        _row_by_axis(report["capability_matrix"], "summary_schema_version")["status"] == "available"
    )
    assert _row_by_axis(report["metric_fields"], "min_distance_m")["status"] == "available"
    assert (
        _row_by_axis(report["unsupported_semantics"], "sensor_perception_replay")["status"]
        == "unsupported"
    )


def test_build_carla_replay_diagnostics_fails_closed_for_missing_required_fields() -> None:
    """Required CARLA summary fields should become explicit not_available rows."""
    report = build_carla_replay_diagnostics(_robot_sf_record(), {"metrics": {"success": True}})

    required_rows = {
        row["axis"]: row for row in report["capability_matrix"] if row["axis"].startswith("summary")
    }
    assert required_rows["summary_schema_version"]["status"] == "not_available"
    assert _row_by_axis(report["capability_matrix"], "carla_map")["status"] == "not_available"
    assert _row_by_axis(report["capability_matrix"], "actor_summary")["reason"] == (
        "required CARLA summary field missing"
    )


def test_build_carla_replay_diagnostics_preserves_degraded_and_unsupported_semantics() -> None:
    """Degraded summaries and unsupported geometry should not become zero-valued success rows."""
    carla_summary = _carla_summary()
    carla_summary["status"] = "failed"
    carla_summary["unsupported"] = {
        "reason": "T0 payload contains unsupported static obstacle geometry",
        "unsupported_static_obstacle_count": 1,
    }

    report = build_carla_replay_diagnostics(_robot_sf_record(), carla_summary)

    assert report["status"] == "degraded"
    assert _row_by_axis(report["metric_fields"], "success")["status"] == "degraded"
    assert (
        _row_by_axis(report["capability_matrix"], "static_geometry_support")["status"]
        == "unsupported"
    )
    unsupported = _row_by_axis(
        report["unsupported_semantics"],
        "carla_summary.unsupported_static_obstacle_count",
    )
    assert unsupported["status"] == "unsupported"
    assert unsupported["carla_value"] == 1


def test_write_carla_replay_diagnostics_outputs_writes_markdown_and_csv(tmp_path: Path) -> None:
    """Diagnostics outputs should be artifact-pack friendly."""
    report = build_carla_replay_diagnostics(_robot_sf_record(), _carla_summary())

    outputs = write_carla_replay_diagnostics_outputs(report, tmp_path)

    assert set(outputs) == {"json", "markdown", "capability_matrix", "unsupported_semantics"}
    assert json.loads((tmp_path / "carla_replay_diagnostics.json").read_text()) == report
    markdown = (tmp_path / "carla_replay_diagnostics.md").read_text(encoding="utf-8")
    assert "CARLA Replay Diagnostics" in markdown
    with (tmp_path / "carla_capability_matrix.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["axis"] == "summary_schema_version"


def test_diagnostics_cli_writes_outputs(monkeypatch, tmp_path: Path, capsys) -> None:
    """CLI should load JSON inputs and write every diagnostics artifact."""
    robot_sf_path = tmp_path / "robot_sf.json"
    carla_path = tmp_path / "carla.json"
    output_dir = tmp_path / "diagnostics"
    robot_sf_path.write_text(json.dumps(_robot_sf_record()), encoding="utf-8")
    carla_path.write_text(json.dumps(_carla_summary()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_replay_semantics.py",
            "--robot-sf",
            str(robot_sf_path),
            "--carla",
            str(carla_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert diagnostics_cli_main() == 0

    outputs = json.loads(capsys.readouterr().out)
    assert Path(outputs["json"]).is_file()
    assert (output_dir / "unsupported_semantics.csv").is_file()
