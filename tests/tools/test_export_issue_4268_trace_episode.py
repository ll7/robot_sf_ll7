"""Tests for the issue #4268 trace episode evidence exporter."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_issue_4268_trace_episode.py"
SPEC = importlib.util.spec_from_file_location("export_issue_4268_trace_episode", SCRIPT_PATH)
assert SPEC is not None
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def _sample_record() -> dict:
    return {
        "scenario_id": "classic_doorway_medium",
        "seed": 141,
        "status": "collision",
        "termination_reason": "collision",
        "algorithm_metadata": {
            "algorithm": "simple_policy",
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "step": 0,
                        "time_s": 0.1,
                        "robot": {
                            "position": [0.0, 0.0],
                            "heading": 0.0,
                            "velocity": [3.0, 4.0],
                        },
                        "pedestrians": [
                            {"id": 2, "position": [5.0, 0.0]},
                            {"id": 4, "position": [1.0, 1.0]},
                        ],
                        "planner": {
                            "selected_action": {
                                "linear_velocity": 0.8,
                                "angular_velocity": -0.2,
                            },
                        },
                    },
                    {
                        "step": 1,
                        "time_s": 0.2,
                        "robot": {
                            "position": [1.0, 0.0],
                            "heading": 0.1,
                            "velocity": [1.0, 0.0],
                        },
                        "pedestrians": [{"id": 4, "position": [1.5, 0.0]}],
                        "planner": {
                            "selected_action": {
                                "linear_velocity": 0.7,
                                "angular_velocity": 0.0,
                            },
                        },
                    },
                ],
            },
        },
    }


def test_derive_trace_rows_exports_figure_ready_distance_series() -> None:
    """Derived rows should expose nearest-distance and action fields for plotting."""

    derived = exporter.derive_trace_rows(_sample_record())

    assert derived.summary == {
        "step_count": 2,
        "global_min_robot_ped_distance_m": 0.5,
        "global_min_distance_step": 1,
        "episode_status": "collision",
        "termination_reason": "collision",
        "scenario_id": "classic_doorway_medium",
        "seed": 141,
        "planner": "simple_policy",
    }
    assert derived.trace_rows[0]["executed_speed_m_s"] == 5.0
    assert derived.trace_rows[0]["commanded_linear_velocity_m_s"] == 0.8
    assert derived.trace_rows[0]["nearest_pedestrian_id"] == "4"
    assert derived.min_distance_rows == [
        {
            "step": 0,
            "time_s": 0.1,
            "min_robot_ped_distance_m": 2**0.5,
            "nearest_pedestrian_id": "4",
        },
        {
            "step": 1,
            "time_s": 0.2,
            "min_robot_ped_distance_m": 0.5,
            "nearest_pedestrian_id": "4",
        },
    ]


def test_csv_and_checksum_outputs_are_stable(tmp_path: Path) -> None:
    """CSV and SHA256SUMS outputs should be deterministic over the written files."""

    derived = exporter.derive_trace_rows(_sample_record())
    csv_path = tmp_path / "trace_timeseries.csv"

    exporter._write_csv(csv_path, derived.trace_rows)
    exporter._write_json(tmp_path / "metadata.json", {"scenario_id": "classic_doorway_medium"})
    exporter._write_sha256sums(tmp_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert handle.readline() == "# AI-GENERATED NEEDS-REVIEW\n"
        rows = list(csv.DictReader(handle))

    assert rows[0]["step"] == "0"
    assert rows[0]["min_robot_ped_distance_m"] == str(2**0.5)
    checksums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert "metadata.json" in checksums
    assert "trace_timeseries.csv" in checksums
    assert "SHA256SUMS" not in checksums


def test_min_distance_series_carries_center_center_convention(tmp_path: Path) -> None:
    """Issue #5141: min_distance_series.csv must declare its distance convention.

    min_robot_ped_distance_m is computed via math.dist(robot_xy, ped_position),
    i.e. center-to-center; the export must annotate the CSV so the value is not
    misread as surface clearance.
    """
    derived = exporter.derive_trace_rows(_sample_record())
    csv_path = tmp_path / "min_distance_series.csv"

    exporter._write_distance_csv(
        csv_path, derived.min_distance_rows, convention=exporter.DistanceConvention.CENTER_CENTER
    )

    content = csv_path.read_text(encoding="utf-8")
    assert "# distance_convention: center_center" in content
    # CSV header and rows are still intact below the convention line.
    with csv_path.open(newline="", encoding="utf-8") as handle:
        data_lines = [line for line in handle if not line.lstrip().startswith("#")]
    rows = list(csv.DictReader(data_lines))
    assert rows[0]["min_robot_ped_distance_m"] == str(2**0.5)
