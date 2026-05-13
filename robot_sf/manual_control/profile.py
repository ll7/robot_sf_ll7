"""Profiling helpers for manual-control JSONL recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from robot_sf.manual_control.recording import load_manual_jsonl_records
from robot_sf.manual_control.replay import group_records_by_attempt

if TYPE_CHECKING:
    from robot_sf.manual_control.recording import ManualControlRecord


@dataclass(frozen=True)
class ManualRecordingProfile:
    """Small size/read-throughput profile for one manual-control JSONL recording."""

    source_path: str
    size_bytes: int
    line_count: int
    record_count: int
    training_sample_count: int
    attempt_count: int
    read_seconds: float

    @property
    def bytes_per_record(self) -> float:
        """Return average serialized bytes per loaded record.

        Returns
        -------
        float
            Average bytes per record, or zero for empty recordings.
        """
        if self.record_count == 0:
            return 0.0
        return self.size_bytes / self.record_count

    @property
    def records_per_second(self) -> float:
        """Return observed load throughput.

        Returns
        -------
        float
            Loaded records per second, or zero when the read timer reports no elapsed time.
        """
        if self.read_seconds <= 0:
            return 0.0
        return self.record_count / self.read_seconds

    def estimated_horizon500_bytes(self) -> int:
        """Estimate one horizon-500 attempt size from observed bytes per record.

        Returns
        -------
        int
            Estimated serialized bytes for 500 records.
        """
        return int(round(self.bytes_per_record * 500))

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-compatible profiling summary.

        Returns
        -------
        dict[str, object]
            Serializable profile summary.
        """
        return {
            "profile_schema": "manual_control_recording_profile_v1",
            "source_path": self.source_path,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "record_count": self.record_count,
            "training_sample_count": self.training_sample_count,
            "attempt_count": self.attempt_count,
            "read_seconds": self.read_seconds,
            "bytes_per_record": self.bytes_per_record,
            "records_per_second": self.records_per_second,
            "estimated_horizon500_bytes": self.estimated_horizon500_bytes(),
            "recommendation": "keep_jsonl_source_of_truth",
        }


def profile_manual_jsonl_recording(path: str | Path) -> ManualRecordingProfile:
    """Measure size and read-throughput metadata for a manual-control JSONL recording.

    Returns
    -------
    ManualRecordingProfile
        Compact profile used to decide whether JSONL remains acceptable.
    """
    input_path = Path(path)
    start = perf_counter()
    records = load_manual_jsonl_records(input_path)
    read_seconds = perf_counter() - start
    return _build_profile(input_path, records=records, read_seconds=read_seconds)


def _build_profile(
    path: Path,
    *,
    records: list[ManualControlRecord],
    read_seconds: float,
) -> ManualRecordingProfile:
    """Build a profile from loaded records and measured read time.

    Returns
    -------
    ManualRecordingProfile
        Compact recording profile.
    """
    text = path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    return ManualRecordingProfile(
        source_path=str(path),
        size_bytes=path.stat().st_size,
        line_count=line_count,
        record_count=len(records),
        training_sample_count=sum(1 for record in records if record.training_sample),
        attempt_count=len(group_records_by_attempt(records)),
        read_seconds=read_seconds,
    )
