"""Tests for manual-control JSONL recording contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from robot_sf.manual_control.recording import (
    ManualControlRecord,
    ManualJsonlRecorder,
    ManualRewindMetadata,
    ManualSessionMetadata,
    load_manual_jsonl_records,
)
from robot_sf.manual_control.session import AttemptKey


def test_manual_control_record_serializes_schema_and_session_metadata():
    """Manual records should keep session and attempt metadata together."""
    session = ManualSessionMetadata(
        session_id="session-1",
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        policy_to_beat="best-policy",
        policy_to_beat_source="model/registry.yaml",
    )

    record = ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=2,
        step_idx=3,
        session=session,
        input_keys=["w"],
        mapped_action=(1.0, 0.0),
        observation={"positions": [[1.0, 2.0]]},
        training_sample=True,
    )

    payload = record.to_json_dict()

    assert payload["record_schema"] == "manual_control_v1"
    assert payload["scenario_id"] == "scenario-a"
    assert payload["seed"] == 7
    assert payload["attempt_id"] == 2
    assert payload["session"]["policy_to_beat_source"] == "model/registry.yaml"
    assert payload["training_sample"] is True


def test_manual_jsonl_recorder_appends_records(tmp_path):
    """ManualJsonlRecorder should append one JSON object per line."""
    path = tmp_path / "manual.jsonl"
    session = ManualSessionMetadata(
        session_id="session-1",
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
    )
    record = ManualControlRecord.for_attempt(
        event="pause",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=5,
        session=session,
        training_sample=False,
    )

    with ManualJsonlRecorder(path) as recorder:
        recorder.write(record)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [record.to_json_dict()]


def test_load_manual_jsonl_records_round_trips_records(tmp_path):
    """Manual JSONL records should load back into dataclass records."""
    path = tmp_path / "manual.jsonl"
    session = ManualSessionMetadata(
        session_id="session-1",
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
    )
    record = ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=5,
        session=session,
        input_keys=["w"],
        mapped_action=(1.0, 0.0),
        observation={"obs": [1]},
        training_sample=True,
    )
    path.write_text(json.dumps(record.to_json_dict()) + "\n", encoding="utf-8")

    loaded = load_manual_jsonl_records(path)

    assert loaded == [record]


def test_load_manual_jsonl_records_rejects_unsupported_schema(tmp_path):
    """Manual JSONL loading should fail closed for schema drift."""
    path = tmp_path / "manual.jsonl"
    path.write_text(json.dumps({"record_schema": "legacy"}) + "\n", encoding="utf-8")

    try:
        load_manual_jsonl_records(path)
    except ValueError as exc:
        assert "unsupported manual-control record schema" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ValueError")


def test_manual_control_record_serializes_numpy_values_and_paths() -> None:
    """Manual record payloads should convert NumPy values and paths before JSON dumps."""
    session = ManualSessionMetadata(
        session_id="session-1",
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
    )
    record = ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=2,
        step_idx=3,
        session=session,
        observation={
            "positions": np.array([[1.0, 2.0]], dtype=np.float32),
            "score": np.float32(0.5),
            "artifact": Path("output/manual/demo.jsonl"),
        },
        metrics={"snqi": np.float64(0.75)},
        training_sample=True,
    )

    payload = record.to_json_dict()
    encoded = json.loads(json.dumps(payload))

    assert encoded["observation"]["positions"] == [[1.0, 2.0]]
    assert encoded["observation"]["score"] == pytest.approx(0.5)
    assert encoded["observation"]["artifact"] == "output/manual/demo.jsonl"
    assert encoded["metrics"]["snqi"] == pytest.approx(0.75)


def test_manual_rewind_metadata_round_trips_in_recording_schema(tmp_path):
    """Rewind events should be explicit append-only recording records."""
    path = tmp_path / "manual.jsonl"
    session = ManualSessionMetadata(
        session_id="session-1",
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
    )
    record = ManualControlRecord.for_attempt(
        event="rewind",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=1,
        session=session,
        rewind=ManualRewindMetadata(
            strategy="replay_to_step_v1",
            from_step_idx=4,
            to_step_idx=1,
            invalidates_samples_after_step=1,
            reason="operator requested",
        ),
    )
    path.write_text(json.dumps(record.to_json_dict()) + "\n", encoding="utf-8")

    loaded = load_manual_jsonl_records(path)

    assert loaded[0].event == "rewind"
    assert loaded[0].rewind == record.rewind


def test_load_manual_jsonl_records_validates_view_mode_and_training_sample(tmp_path):
    """Manual JSONL loading should fail closed on invalid mode and sample types."""
    path = tmp_path / "manual.jsonl"
    record = ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=5,
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        ),
        training_sample=True,
    ).to_json_dict()

    record["session"]["view_mode"] = "panorama"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported manual-control view mode"):
        load_manual_jsonl_records(path)

    record["session"]["view_mode"] = "fixed_map"
    record["training_sample"] = "false"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="training_sample must be a boolean"):
        load_manual_jsonl_records(path)


def test_load_manual_jsonl_records_reports_line_number_for_invalid_payload(tmp_path):
    """Manual JSONL loading should wrap malformed payloads with their line number."""
    path = tmp_path / "manual.jsonl"
    path.write_text(
        json.dumps({"record_schema": "manual_control_v1", "session": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manual-control line 1"):
        load_manual_jsonl_records(path)
