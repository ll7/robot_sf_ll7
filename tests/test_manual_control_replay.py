"""Tests for manual-control replay helpers."""

import json

from robot_sf.manual_control.recording import ManualControlRecord, ManualSessionMetadata
from robot_sf.manual_control.replay import (
    group_records_by_attempt,
    iter_replay_events,
    write_attempt_replay_json,
)
from robot_sf.manual_control.rewind import (
    compute_rewind_invalidated_record_indexes,
    plan_replay_to_step_rewind,
)
from robot_sf.manual_control.session import AttemptKey


def _record(
    scenario_id: str,
    seed: int,
    attempt_id: int,
    step_idx: int,
    *,
    event: str = "step",
) -> ManualControlRecord:
    """Build a minimal manual-control record."""
    return ManualControlRecord.for_attempt(
        event=event,
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


def test_group_records_by_attempt_preserves_file_order_within_same_step():
    """Same-step events should keep original stream order for deterministic replay."""
    records = [
        _record("scenario-a", 1, 0, 2, event="input"),
        _record("scenario-a", 1, 0, 2, event="step"),
        _record("scenario-a", 1, 0, 1, event="countdown"),
    ]

    replays = group_records_by_attempt(records)

    assert [record.event for record in replays[0].records] == ["countdown", "input", "step"]


def test_iter_replay_events_exposes_serializable_event_stream():
    """Replay events should keep session, event, action, and training markers."""
    record = ManualControlRecord.for_attempt(
        event="step",
        attempt_key=AttemptKey("scenario-a", 7),
        attempt_id=1,
        step_idx=2,
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        ),
        input_keys=["w"],
        mapped_action=(0.5, 0.0),
        metrics={"success": False},
        training_sample=True,
    )
    replay = group_records_by_attempt([record])[0]

    events = iter_replay_events(replay)

    assert events[0].to_json_dict()["session_id"] == "session-1"
    assert events[0].to_json_dict()["mapped_action"] == [0.5, 0.0]
    assert events[0].to_json_dict()["training_sample"] is True


def test_write_attempt_replay_json_writes_grouped_attempts(tmp_path):
    """Replay writer should produce a deterministic JSON artifact for inspection."""
    output_path = tmp_path / "replay.json"
    replays = group_records_by_attempt(
        [
            _record("scenario-b", 3, 0, 0),
            _record("scenario-a", 1, 0, 0),
        ]
    )

    written_path = write_attempt_replay_json(replays, output_path)

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["replay_schema"] == "manual_control_attempt_replay_v1"
    assert [attempt["scenario_id"] for attempt in payload["attempts"]] == [
        "scenario-a",
        "scenario-b",
    ]


def test_plan_replay_to_step_rewind_restores_prefix_and_builds_event():
    """Rewind planning should keep the replay prefix and append an explicit event."""
    records = [
        _record("scenario-a", 1, 0, 0),
        _record("scenario-a", 1, 0, 1),
        _record("scenario-a", 1, 0, 2),
    ]
    replay = group_records_by_attempt(records)[0]

    plan = plan_replay_to_step_rewind(replay, target_step_idx=1, reason="operator")
    rewind_record = plan.to_rewind_record()

    assert [record.step_idx for record in plan.restored_records] == [0, 1]
    assert plan.from_step_idx == 2
    assert plan.to_step_idx == 1
    assert rewind_record.event == "rewind"
    assert rewind_record.rewind is not None
    assert rewind_record.rewind.strategy == "replay_to_step_v1"
    assert rewind_record.rewind.reason == "operator"


def test_compute_rewind_invalidated_record_indexes_marks_discarded_suffix():
    """Later rewind records should invalidate prior training samples after the target step."""
    step0 = _record("scenario-a", 1, 0, 0)
    old_step1 = _record("scenario-a", 1, 0, 1)
    old_step2 = _record("scenario-a", 1, 0, 2)
    replay = group_records_by_attempt([step0, old_step1, old_step2])[0]
    plan = plan_replay_to_step_rewind(replay, target_step_idx=0)
    rewind_record = plan.to_rewind_record()
    stream = [
        step0.__class__(**{**step0.__dict__, "training_sample": True}),
        old_step1.__class__(**{**old_step1.__dict__, "training_sample": True}),
        old_step2.__class__(**{**old_step2.__dict__, "training_sample": True}),
        rewind_record,
    ]

    invalidated = compute_rewind_invalidated_record_indexes(stream)

    assert invalidated == frozenset({1, 2})
