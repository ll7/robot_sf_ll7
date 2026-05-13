"""Tests for manual-control demonstration export helpers."""

import json

import pytest

from robot_sf.manual_control.export import (
    export_demonstration_samples,
    export_demonstration_samples_from_jsonl,
    write_demonstration_samples_jsonl,
)
from robot_sf.manual_control.recording import ManualControlRecord, ManualSessionMetadata
from robot_sf.manual_control.session import AttemptKey


def _record(
    *,
    event: str = "step",
    training_sample: bool,
    observation=None,
    mapped_action=None,
) -> ManualControlRecord:
    """Build a minimal manual-control record for export tests."""
    return ManualControlRecord.for_attempt(
        event=event,
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=1,
        step_idx=2,
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        ),
        input_keys=["w"],
        mapped_action=mapped_action,
        observation=observation,
        training_sample=training_sample,
    )


def test_export_demonstration_samples_filters_non_training_events():
    """Pause/countdown-style records should not become BC samples."""
    records = [
        _record(event="pause", training_sample=False),
        _record(
            training_sample=True,
            observation={"obs": [1.0]},
            mapped_action=(0.5, 0.0),
        ),
    ]

    samples = export_demonstration_samples(records)

    assert len(samples) == 1
    assert samples[0].to_json_dict() == {
        "sample_schema": "manual_control_bc_v1",
        "session_id": "session-1",
        "scenario_id": "scenario-a",
        "seed": 7,
        "attempt_id": 1,
        "step_idx": 2,
        "observation": {"obs": [1.0]},
        "action": [0.5, 0.0],
        "input_keys": ["w"],
    }


def test_export_demonstration_samples_requires_observation_and_action():
    """Training-marked records without aligned obs/action should fail closed."""
    records = [_record(training_sample=True, observation={"obs": [1.0]}, mapped_action=None)]

    with pytest.raises(ValueError, match="observation and mapped_action"):
        export_demonstration_samples(records)


def test_export_demonstration_samples_from_jsonl(tmp_path):
    """JSONL convenience exporter should load records and filter samples."""
    path = tmp_path / "manual.jsonl"
    training_record = _record(
        training_sample=True,
        observation={"obs": [1.0]},
        mapped_action=(0.5, 0.0),
    )
    pause_record = _record(event="pause", training_sample=False)
    path.write_text(
        "\n".join(
            [
                json.dumps(pause_record.to_json_dict()),
                json.dumps(training_record.to_json_dict()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = export_demonstration_samples_from_jsonl(path)

    assert len(samples) == 1
    assert samples[0].action == (0.5, 0.0)


def test_write_demonstration_samples_jsonl(tmp_path):
    """Compact BC samples should be writable as one JSON object per line."""
    samples = export_demonstration_samples(
        [
            _record(
                training_sample=True,
                observation={"obs": [1.0]},
                mapped_action=(0.5, 0.0),
            )
        ]
    )
    path = tmp_path / "samples.jsonl"

    written_path = write_demonstration_samples_jsonl(samples, path)

    assert written_path == path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [samples[0].to_json_dict()]
