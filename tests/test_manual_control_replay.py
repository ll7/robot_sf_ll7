"""Tests for manual-control replay helpers."""

from robot_sf.manual_control.recording import ManualControlRecord, ManualSessionMetadata
from robot_sf.manual_control.replay import group_records_by_attempt
from robot_sf.manual_control.session import AttemptKey


def _record(scenario_id: str, seed: int, attempt_id: int, step_idx: int) -> ManualControlRecord:
    """Build a minimal manual-control record."""
    return ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey(scenario_id, seed),
        attempt_id=attempt_id,
        step_idx=step_idx,
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        ),
    )


def test_group_records_by_attempt_orders_attempts_and_steps():
    """Replay grouping should preserve attempt identity and sort records by step."""
    records = [
        _record("scenario-b", 1, 0, 2),
        _record("scenario-a", 1, 1, 3),
        _record("scenario-a", 1, 1, 1),
    ]

    replays = group_records_by_attempt(records)

    assert [(replay.key.scenario_id, replay.key.seed, replay.attempt_id) for replay in replays] == [
        ("scenario-a", 1, 1),
        ("scenario-b", 1, 0),
    ]
    assert [record.step_idx for record in replays[0].records] == [1, 3]
