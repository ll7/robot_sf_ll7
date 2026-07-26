"""Tests for tensorboard_adapter JSONL parsing and scalar iteration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.telemetry.models import TelemetrySnapshot
from robot_sf.telemetry.tensorboard_adapter import (
    TensorBoardAdapter,
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


def test_iter_telemetry_snapshots_missing_timestamp_is_none(tmp_path):
    path = tmp_path / "missing_timestamp.jsonl"
    _write_jsonl(path, [json.dumps({"frame_idx": 1})])

    snapshots = list(iter_telemetry_snapshots(path))
    assert len(snapshots) == 1
    assert snapshots[0].timestamp_ms is None
    assert snapshots[0].frame_idx == 1


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


# ---------------------------------------------------------------------------
# TensorBoardAdapter class-level contracts
#
# A deterministic fake SummaryWriter is injected via ``adapter._writer_cls`` so
# these tests never require torch/tensorboardX and never create real event files.
# ---------------------------------------------------------------------------


class _FakeSummaryWriter:
    """Minimal SummaryWriter double recording add_scalar/flush/close calls."""

    def __init__(self, log_dir: str | None = None, **_kwargs: object) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.flushed = 0
        self.closed = 0

    def add_scalar(self, tag: str, value: float, global_step: int | None = None) -> None:
        self.scalars.append((tag, value, global_step))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


def _adapter(
    log_dir: Path, *, available: bool = True, tag_prefix: str = "telemetry"
) -> TensorBoardAdapter:
    """Build an adapter whose backend is a deterministic fake writer class."""
    adapter = TensorBoardAdapter(log_dir=log_dir, tag_prefix=tag_prefix)
    adapter._writer_cls = _FakeSummaryWriter if available else None
    return adapter


def _full_snapshot(
    timestamp_ms: int,
    steps_per_sec: float,
    cpu_process: float,
    cpu_system: float,
    memory: float,
    gpu_util: float,
    gpu_mem: float,
) -> TelemetrySnapshot:
    """Snapshot with every scalar field populated."""
    return TelemetrySnapshot(
        timestamp_ms=timestamp_ms,
        steps_per_sec=steps_per_sec,
        cpu_percent_process=cpu_process,
        cpu_percent_system=cpu_system,
        memory_rss_mb=memory,
        gpu_util_percent=gpu_util,
        gpu_mem_used_mb=gpu_mem,
    )


def test_default_tag_prefix_is_telemetry(tmp_path: Path):
    adapter = TensorBoardAdapter(log_dir=tmp_path)
    assert adapter.tag_prefix == "telemetry"


def test_is_available_true_when_writer_cls_injected(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True)
    assert adapter.is_available is True


def test_is_available_false_when_writer_cls_none(tmp_path: Path):
    adapter = _adapter(tmp_path, available=False)
    assert adapter.is_available is False


def test_start_raises_runtime_error_when_unavailable(tmp_path: Path):
    log_dir = tmp_path / "events"
    adapter = _adapter(log_dir, available=False)
    with pytest.raises(RuntimeError):
        adapter.start()
    # failed start must not create the log dir, a writer, or advance samples
    assert not log_dir.exists()
    assert adapter._writer is None
    assert adapter._samples == 0


def test_start_creates_writer_and_log_dir(tmp_path: Path):
    log_dir = tmp_path / "events"
    adapter = _adapter(log_dir, available=True)
    assert not log_dir.exists()
    adapter.start()
    assert log_dir.is_dir()
    assert isinstance(adapter._writer, _FakeSummaryWriter)
    # the writer receives the adapter log_dir as a string
    assert adapter._writer.log_dir == str(log_dir)


def test_start_is_idempotent(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True)
    adapter.start()
    first = adapter._writer
    assert first is not None
    adapter.start()  # second start must be a no-op
    assert adapter._writer is first


def test_consume_snapshot_lazy_starts_writer(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True)
    assert adapter._writer is None
    adapter.consume_snapshot(_full_snapshot(1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    assert isinstance(adapter._writer, _FakeSummaryWriter)


def test_consume_snapshot_no_op_when_unavailable(tmp_path: Path):
    adapter = _adapter(tmp_path, available=False)
    adapter.consume_snapshot(_full_snapshot(1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    # unavailable consume must not lazy-start a writer or count a sample
    assert adapter._writer is None
    assert adapter._samples == 0


def test_consume_snapshot_emits_prefixed_tags_with_incrementing_global_step(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True, tag_prefix="run")
    snap1 = _full_snapshot(1, 10.0, 50.0, 12.0, 256.0, 80.0, 1024.0)
    snap2 = _full_snapshot(2, 20.0, 60.0, 14.0, 300.0, 90.0, 1100.0)

    adapter.consume_snapshot(snap1)
    adapter.consume_snapshot(snap2)

    assert adapter._samples == 2
    assert adapter._writer.scalars == [
        ("run/steps_per_sec", 10.0, 1),
        ("run/cpu_process_percent", 50.0, 1),
        ("run/cpu_system_percent", 12.0, 1),
        ("run/memory_rss_mb", 256.0, 1),
        ("run/gpu_util_percent", 80.0, 1),
        ("run/gpu_mem_used_mb", 1024.0, 1),
        ("run/steps_per_sec", 20.0, 2),
        ("run/cpu_process_percent", 60.0, 2),
        ("run/cpu_system_percent", 14.0, 2),
        ("run/memory_rss_mb", 300.0, 2),
        ("run/gpu_util_percent", 90.0, 2),
        ("run/gpu_mem_used_mb", 1100.0, 2),
    ]


def test_consume_snapshot_omits_none_scalars(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True)
    snap = TelemetrySnapshot(timestamp_ms=1, steps_per_sec=7.5)
    # all other scalar fields default to None and must be skipped

    adapter.consume_snapshot(snap)

    assert adapter._samples == 1
    assert adapter._writer.scalars == [("telemetry/steps_per_sec", 7.5, 1)]


def test_close_flushes_and_closes_writer(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True)
    adapter.start()
    writer = adapter._writer
    assert writer is not None

    adapter.close()

    assert writer.flushed == 1
    assert writer.closed == 1
    assert adapter._writer is None


def test_close_is_idempotent_when_writer_none(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True)
    assert adapter._writer is None
    adapter.close()  # must not raise on a never-started adapter
    assert adapter._writer is None


def test_close_is_idempotent_after_first_close(tmp_path: Path):
    adapter = _adapter(tmp_path, available=True)
    adapter.start()
    writer = adapter._writer
    adapter.close()
    adapter.close()  # second close must be a no-op

    assert writer is not None
    assert writer.flushed == 1
    assert writer.closed == 1
    assert adapter._writer is None


def test_mirror_file_counts_mixed_jsonl_and_guarantees_close(tmp_path: Path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        "\n".join(
            [
                "",  # blank line -> skipped
                json.dumps({"timestamp_ms": 1, "steps_per_sec": 1.0}),
                json.dumps({"timestamp_ms": 2, "steps_per_sec": 2.0}),
                "",  # blank line -> skipped
                json.dumps([1, 2, 3]),  # non-object JSON -> skipped
                json.dumps("a string"),  # non-object JSON -> skipped
                json.dumps(42),  # non-object JSON -> skipped
                json.dumps(True),  # non-object JSON -> skipped
                json.dumps(None),  # non-object JSON -> skipped
                "not valid json",  # malformed JSON -> skipped
                "{unquoted: true}",  # malformed JSON -> skipped
                json.dumps({"timestamp_ms": 3, "steps_per_sec": 3.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # lazy path: no explicit start() before mirroring
    adapter = _adapter(tmp_path / "events", available=True)

    count = adapter.mirror_file(path)

    assert count == 3
    assert adapter._samples == 3
    # mirror_file must always close the writer, even on the lazy-start path
    assert adapter._writer is None


def test_mirror_file_emits_scalars_and_closes_writer(tmp_path: Path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp_ms": 1, "steps_per_sec": 10.0, "memory_rss_mb": 256.0}),
                json.dumps({"timestamp_ms": 2, "steps_per_sec": 20.0, "memory_rss_mb": 300.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = _adapter(tmp_path / "events", available=True, tag_prefix="mirror")
    adapter.start()
    writer = adapter._writer
    assert writer is not None

    count = adapter.mirror_file(path)

    assert count == 2
    assert adapter._samples == 2
    # only non-None scalars are emitted, with incrementing global_step
    assert writer.scalars == [
        ("mirror/steps_per_sec", 10.0, 1),
        ("mirror/memory_rss_mb", 256.0, 1),
        ("mirror/steps_per_sec", 20.0, 2),
        ("mirror/memory_rss_mb", 300.0, 2),
    ]
    # close is guaranteed
    assert writer.closed == 1
    assert adapter._writer is None
