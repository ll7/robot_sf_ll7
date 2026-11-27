"""Tests for the development doorway witness."""

from __future__ import annotations

import pytest

from scripts.validation.run_issue_10520_passable_doorway import summarize_trace


def _frames(*, robot_x: float = 14.0, pedestrian_y: float = 5.0) -> list[dict]:
    """Build a head-on encounter where the pedestrian enters from the east.

    Returns:
        Three synchronized samples spanning the opening.
    """
    return [
        {
            "time_s": float(index),
            "robot": {"position": [x, 5.0]},
            "pedestrians": [{"id": "ped-0", "position": [ped_x, pedestrian_y]}],
            "collision": {},
        }
        for index, (x, ped_x) in enumerate([(8.0, 19.0), (robot_x, 15.8), (14.5, 14.5)])
    ]


@pytest.mark.parametrize(
    ("robot_x", "pedestrian_y", "encounter_time_s"),
    [(14.0, 5.0, 1.0), (9.0, 5.0, None), (14.0, 7.0, None)],
)
def test_encounter_requires_pedestrian_in_opening_during_approach(
    robot_x: float, pedestrian_y: float, encounter_time_s: float | None
) -> None:
    """An eventual crossing, or a position behind the wall, cannot prove an encounter."""
    result = summarize_trace(
        _frames(robot_x=robot_x, pedestrian_y=pedestrian_y)[:2],
        opening=(15.0, 16.0, 3.2, 6.8),
        dt=1.0,
        expected_actors=(True, True),
    )
    assert result["encounter_time_s"] == encounter_time_s
    assert result["pedestrian_exit_time_s"] is None


@pytest.mark.parametrize("damage", ["changed_actor", "missing_step", "missing_actor"])
def test_witness_rejects_bad_trace_alignment(damage: str) -> None:
    """A changed actor, dropped frame, or wrong population invalidates a trace."""
    frames = _frames()
    match damage:
        case "changed_actor":
            frames[1]["pedestrians"][0]["id"] = "replacement"
        case "missing_step":
            frames.pop(1)
        case "missing_actor":
            frames[1]["pedestrians"] = []
    with pytest.raises(ValueError, match="actor|time"):
        summarize_trace(
            frames, opening=(15.0, 16.0, 3.2, 6.8), dt=1.0, expected_actors=(True, True)
        )


@pytest.mark.parametrize("outcome", ["returned_robot", "returned_pedestrian"])
def test_returning_from_far_side_is_not_an_approach(outcome: str) -> None:
    """A later visit to the opening cannot replace the original approach."""
    frames = _frames(robot_x=14.0)
    if outcome == "returned_robot":
        frames[1]["robot"]["position"] = [17.0, 5.0]
        frames[2]["robot"]["position"] = [14.0, 5.0]
    else:
        frames[1]["pedestrians"][0]["position"] = [14.0, 5.0]
    frames[2]["pedestrians"][0]["position"] = [15.8, 5.0]
    row = summarize_trace(
        frames, opening=(15.0, 16.0, 3.2, 6.8), dt=1.0, expected_actors=(True, True)
    )
    assert row["encounter_time_s"] is None
