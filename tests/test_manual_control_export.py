"""Tests for manual-control demonstration export helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from robot_sf.manual_control.export import (
    export_demonstration_samples,
    export_demonstration_samples_from_jsonl,
    write_demonstration_samples_jsonl,
)
from robot_sf.manual_control.recording import (
    ManualControlRecord,
    ManualRewindMetadata,
    ManualSessionMetadata,
)
from robot_sf.manual_control.session import AttemptKey


def _record(
    *,
    event: str = "step",
    training_sample: bool,
    scenario_id: str = "scenario-a",
    seed: int = 7,
    attempt_id: int = 1,
    step_idx: int = 2,
    observation=None,
    mapped_action=None,
) -> ManualControlRecord:
    """Build a minimal manual-control record for export tests."""
    return ManualControlRecord.for_attempt(
        event=event,
        attempt_key=AttemptKey(scenario_id, seed),
        attempt_id=attempt_id,
        step_idx=step_idx,
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
        "rewind_segment_id": 0,
        "source": {
            "record_schema": "manual_control_v1",
            "record_index": 1,
            "path": None,
        },
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
    assert samples[0].source_path == str(path)
    assert samples[0].source_record_index == 1


def test_write_demonstration_samples_jsonl(tmp_path):
    """Compact BC samples should be writable as one JSON object per line."""
    samples = export_demonstration_samples(
        [
            _record(
                training_sample=True,
                observation={
                    "obs": np.array([1.0], dtype=np.float32),
                    "artifact": Path("output/manual/demo.jsonl"),
                },
                mapped_action=(0.5, 0.0),
            )
        ]
    )
    path = tmp_path / "samples.jsonl"

    written_path = write_demonstration_samples_jsonl(samples, path)

    assert written_path == path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [samples[0].to_json_dict()]
    assert rows[0]["observation"] == {
        "obs": [1.0],
        "artifact": "output/manual/demo.jsonl",
    }


def test_export_demonstration_samples_excludes_samples_invalidated_by_rewind():
    """BC export should not train on samples from a discarded pre-rewind suffix."""
    step0 = _record(
        training_sample=True,
        step_idx=0,
        observation={"obs": [0.0]},
        mapped_action=(0.0, 0.0),
    )
    old_step1 = _record(
        training_sample=True,
        step_idx=1,
        observation={"obs": [1.0]},
        mapped_action=(1.0, 0.0),
    )
    old_step2 = _record(
        training_sample=True,
        step_idx=2,
        observation={"obs": [2.0]},
        mapped_action=(2.0, 0.0),
    )
    rewind = ManualControlRecord.for_attempt(
        event="rewind",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=1,
        step_idx=0,
        session=old_step2.session,
        rewind=ManualRewindMetadata(
            strategy="replay_to_step_v1",
            from_step_idx=2,
            to_step_idx=0,
            invalidates_samples_after_step=0,
            reason="test",
        ),
    )
    new_step1 = _record(
        training_sample=True,
        step_idx=1,
        observation={"obs": [10.0]},
        mapped_action=(0.5, 0.0),
    )

    samples = export_demonstration_samples([step0, old_step1, old_step2, rewind, new_step1])

    assert [
        (sample.step_idx, sample.observation, sample.rewind_segment_id) for sample in samples
    ] == [
        (0, {"obs": [0.0]}, 0),
        (1, {"obs": [10.0]}, 1),
    ]


def test_export_demonstration_samples_scopes_rewind_segment_ids_per_attempt():
    """Rewind segment numbering should restart for each independent attempt stream."""
    attempt_a_step0 = _record(
        training_sample=True,
        step_idx=0,
        observation={"obs": [0.0]},
        mapped_action=(0.0, 0.0),
    )
    attempt_a_rewind = ManualControlRecord.for_attempt(
        event="rewind",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=1,
        step_idx=0,
        session=attempt_a_step0.session,
        rewind=ManualRewindMetadata(
            strategy="replay_to_step_v1",
            from_step_idx=1,
            to_step_idx=0,
            invalidates_samples_after_step=0,
            reason="test",
        ),
    )
    attempt_b_step0 = _record(
        training_sample=True,
        scenario_id="scenario-b",
        seed=11,
        attempt_id=3,
        step_idx=0,
        observation={"obs": [5.0]},
        mapped_action=(1.0, 0.0),
    )

    samples = export_demonstration_samples([attempt_a_step0, attempt_a_rewind, attempt_b_step0])

    assert [(sample.scenario_id, sample.attempt_id, sample.rewind_segment_id) for sample in samples] == [
        ("scenario-a", 1, 0),
        ("scenario-b", 3, 0),
    ]
