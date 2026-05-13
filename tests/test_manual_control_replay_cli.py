"""Tests for the manual-control replay export CLI."""

import json

from robot_sf.manual_control.recording import ManualControlRecord, ManualSessionMetadata
from robot_sf.manual_control.session import AttemptKey
from scripts.manual_control.replay_attempts import main


def test_replay_attempts_cli_writes_grouped_replay(monkeypatch, tmp_path, capsys):
    """CLI should export grouped replay JSON from a manual-control JSONL stream."""
    input_path = tmp_path / "manual.jsonl"
    output_path = tmp_path / "replay.json"
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
            "replay_attempts.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["attempts"][0]["scenario_id"] == "scenario-a"
    assert "wrote 1 manual-control attempt replays" in capsys.readouterr().out
