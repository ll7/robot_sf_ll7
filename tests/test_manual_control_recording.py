"""Tests for manual-control JSONL recording contracts."""

import json

from robot_sf.manual_control.recording import (
    ManualControlRecord,
    ManualJsonlRecorder,
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
