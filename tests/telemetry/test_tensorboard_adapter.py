"""Tests for tensorboard_adapter JSONL parsing and scalar iteration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.telemetry.models import TelemetrySnapshot
from robot_sf.telemetry.tensorboard_adapter import (
    _iter_scalar_values,
    iter_telemetry_snapshots,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, lines: list[str]) -> None:
    """Write newline-separated lines as a JSONL fixture file."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_iter_telemetry_snapshots_valid_records(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        path,
        [
            json.dumps(
                {
                    "timestamp_ms": 1000,
                    "frame_idx": 1,
                    "status": "running",
                    "step_id": "step_1",
                    "steps_per_sec": 15.5,
                    "fps": 30.0,
                    "cpu_percent_process": 45.2,
                    "cpu_percent_system": 12.1,
                    "memory_rss_mb": 256.0,
                    "gpu_util_percent": 80.0,
                    "gpu_mem_used_mb": 1024.0,
                    "notes": "test snapshot",
                }
            ),
            json.dumps(
                {
                    "timestamp_ms": 2000,
                    "frame_idx": 2,
                    "status": "completed",
                }
            ),
        ],
    )
    snapshots = list(iter_telemetry_snapshots(path))
    assert len(snapshots) == 2

    s0 = snapshots[0]
    assert isinstance(s0, TelemetrySnapshot)
    assert s0.timestamp_ms == 1000
    assert s0.frame_idx == 1
    assert s0.status == "running"
    assert s0.step_id == "step_1"
    assert s0.steps_per_sec == 15.5
    assert s0.fps == 30.0
    assert s0.cpu_percent_process == 45.2
    assert s0.cpu_percent_system == 12.1
    assert s0.memory_rss_mb == 256.0
    assert s0.gpu_util_percent == 80.0
    assert s0.gpu_mem_used_mb == 1024.0
    assert s0.notes == "test snapshot"

    s1 = snapshots[1]
    assert s1.timestamp_ms == 2000
    assert s1.frame_idx == 2
    assert s1.status == "completed"
    assert s1.step_id is None
    assert s1.steps_per_sec is None
    assert s1.cpu_percent_process is None
    assert s1.notes is None


def test_iter_telemetry_snapshots_extra_fields_ignored(tmp_path):
    path = tmp_path / "extra.jsonl"
    _write_jsonl(
        path,
        [
            json.dumps(
                {
                    "timestamp_ms": 42,
                    "unknown_field": "ignored",
                    "extra": [1, 2, 3],
                }
            ),
        ],
    )
    snapshots = list(iter_telemetry_snapshots(path))
    assert len(snapshots) == 1
    assert snapshots[0].timestamp_ms == 42
    assert snapshots[0].frame_idx is None


def test_iter_telemetry_snapshots_blank_lines(tmp_path):
    path = tmp_path / "blank.jsonl"
    path.write_text('\n\n{"timestamp_ms": 1}\n\n', encoding="utf-8")
    snapshots = list(iter_telemetry_snapshots(path))
    assert len(snapshots) == 1
    assert snapshots[0].timestamp_ms == 1


def test_iter_telemetry_snapshots_non_dict_json(tmp_path):
    path = tmp_path / "non_dict.jsonl"
    _write_jsonl(
        path,
        [
            json.dumps({"timestamp_ms": 1}),
            json.dumps([1, 2, 3]),
            json.dumps("string"),
            json.dumps(42),
            json.dumps(None),
            json.dumps(True),
            json.dumps({"timestamp_ms": 2}),
        ],
    )
    snapshots = list(iter_telemetry_snapshots(path))
    assert len(snapshots) == 2
    assert snapshots[0].timestamp_ms == 1
    assert snapshots[1].timestamp_ms == 2


def test_iter_telemetry_snapshots_malformed_json(tmp_path):
    path = tmp_path / "malformed.jsonl"
    _write_jsonl(
        path,
        [
            json.dumps({"timestamp_ms": 1}),
            "this is not json",
            "{unquoted: true}",
            "",
            json.dumps({"timestamp_ms": 2}),
        ],
    )
    snapshots = list(iter_telemetry_snapshots(path))
    assert len(snapshots) == 2
    assert snapshots[0].timestamp_ms == 1
    assert snapshots[1].timestamp_ms == 2


def test_iter_telemetry_snapshots_nonexistent_file(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    snapshots = list(iter_telemetry_snapshots(path))
    assert snapshots == []


def test_iter_scalar_values_prefix_and_pair_format():
    snapshot = TelemetrySnapshot(
        timestamp_ms=1000,
        frame_idx=3,
        status="running",
        step_id="train_01",
        steps_per_sec=12.5,
        fps=28.0,
        cpu_percent_process=55.0,
        cpu_percent_system=10.0,
        memory_rss_mb=512.0,
        gpu_util_percent=95.0,
        gpu_mem_used_mb=2048.0,
    )
    pairs = dict(_iter_scalar_values(snapshot, prefix="test_run"))
    assert pairs == {
        "test_run/steps_per_sec": 12.5,
        "test_run/cpu_process_percent": 55.0,
        "test_run/cpu_system_percent": 10.0,
        "test_run/memory_rss_mb": 512.0,
        "test_run/gpu_util_percent": 95.0,
        "test_run/gpu_mem_used_mb": 2048.0,
    }


def test_iter_scalar_values_handles_none():
    snapshot = TelemetrySnapshot(
        timestamp_ms=1,
        steps_per_sec=None,
        cpu_percent_process=None,
        cpu_percent_system=None,
        memory_rss_mb=None,
        gpu_util_percent=None,
        gpu_mem_used_mb=None,
    )
    pairs = dict(_iter_scalar_values(snapshot, prefix="run"))
    assert pairs["run/steps_per_sec"] is None
    assert pairs["run/cpu_process_percent"] is None
    assert pairs["run/cpu_system_percent"] is None
    assert pairs["run/memory_rss_mb"] is None
    assert pairs["run/gpu_util_percent"] is None
    assert pairs["run/gpu_mem_used_mb"] is None
