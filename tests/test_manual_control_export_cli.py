"""Tests for the manual-control BC export CLI."""

import json

from robot_sf.manual_control.recording import ManualControlRecord, ManualSessionMetadata
from robot_sf.manual_control.session import AttemptKey
from scripts.manual_control.export_bc_samples import main


def test_export_bc_samples_cli_writes_compact_jsonl(monkeypatch, tmp_path, capsys):
    """CLI should export only training-marked manual records."""
    input_path = tmp_path / "manual.jsonl"
    output_path = tmp_path / "samples.jsonl"
    session = ManualSessionMetadata(
        session_id="session-1",
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
    )
    training_record = ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=1,
        session=session,
        input_keys=["w"],
        mapped_action=(0.5, 0.0),
        observation={"obs": [1.0]},
        training_sample=True,
    )
    pause_record = ManualControlRecord.for_attempt(
        event="pause",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=0,
        step_idx=2,
        session=session,
        training_sample=False,
    )
    input_path.write_text(
        "\n".join(
            [
                json.dumps(pause_record.to_json_dict()),
                json.dumps(training_record.to_json_dict()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_bc_samples.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["sample_schema"] == "manual_control_bc_v1"
    assert rows[0]["action"] == [0.5, 0.0]
    assert "wrote 1 manual-control BC samples" in capsys.readouterr().out
