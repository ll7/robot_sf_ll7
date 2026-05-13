"""Tests for the manual-control recording profile CLI."""

import json

from robot_sf.manual_control.recording import ManualControlRecord, ManualSessionMetadata
from robot_sf.manual_control.session import AttemptKey
from scripts.manual_control.profile_recording import main


def test_profile_recording_cli_writes_summary(monkeypatch, tmp_path, capsys):
    """CLI should write a compact JSON profile for a manual-control recording."""
    input_path = tmp_path / "manual.jsonl"
    output_path = tmp_path / "profile.json"
    record = ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=1,
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        ),
    )
    input_path.write_text(json.dumps(record.to_json_dict()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_recording.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["profile_schema"] == "manual_control_recording_profile_v1"
    assert payload["record_count"] == 1
    assert "profiled 1 manual-control records" in capsys.readouterr().out
