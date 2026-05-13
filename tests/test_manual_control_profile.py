"""Tests for manual-control recording profiling."""

import json

import pytest

from robot_sf.manual_control.profile import profile_manual_jsonl_recording
from robot_sf.manual_control.recording import ManualControlRecord, ManualSessionMetadata
from robot_sf.manual_control.session import AttemptKey


def _record(step_idx: int, *, training_sample: bool = False) -> ManualControlRecord:
    """Build one minimal manual-control profile fixture record."""
    return ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=step_idx,
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        ),
        mapped_action=(0.5, 0.0) if training_sample else None,
        observation={"obs": [step_idx]} if training_sample else None,
        training_sample=training_sample,
    )


def test_profile_manual_jsonl_recording_reports_size_and_attempts(tmp_path):
    """Recording profiles should summarize size, attempts, and training samples."""
    path = tmp_path / "manual.jsonl"
    records = [_record(0, training_sample=True), _record(1)]
    path.write_text(
        "\n".join(json.dumps(record.to_json_dict()) for record in records) + "\n",
        encoding="utf-8",
    )

    profile = profile_manual_jsonl_recording(path)
    payload = profile.to_json_dict()

    assert profile.record_count == 2
    assert profile.training_sample_count == 1
    assert profile.attempt_count == 1
    assert payload["profile_schema"] == "manual_control_recording_profile_v1"
    assert payload["estimated_horizon500_bytes"] > 0
    assert payload["recommendation"] == "keep_jsonl_source_of_truth"


def test_profile_manual_jsonl_recording_rejects_directory_input(tmp_path) -> None:
    """Recording profiling should fail closed when the input path is not a file."""
    with pytest.raises(ValueError, match="not a file"):
        profile_manual_jsonl_recording(tmp_path)
