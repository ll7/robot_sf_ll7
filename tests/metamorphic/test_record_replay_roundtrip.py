"""Record/replay metamorphism for the crowd-only environment."""

from __future__ import annotations

from tests.metamorphic.support import BASE_MAP, assert_trace_equal, read_recording, run_episode


def test_recorded_state_trace_round_trips_without_drift(tmp_path) -> None:
    """The compact JSONL state recording must replay the authoritative observations exactly."""
    recording_path = tmp_path / "crowd_episode.jsonl"
    recorded = run_episode(BASE_MAP, recording_path=recording_path)
    events, replayed = read_recording(
        recording_path,
        row_keys=tuple(pedestrian.id for pedestrian in BASE_MAP.single_pedestrians),
    )

    assert events == ["reset", "step", "step", "step"]
    assert_trace_equal(recorded, replayed)
