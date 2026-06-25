"""Tests for trace-level safety-predicate producers (issue #3483)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.safety_predicates import (
    LATE_EVASIVE_PREDICATE_SCHEMA,
    OSCILLATORY_PREDICATE_SCHEMA,
    late_evasive_predicate,
    oscillatory_control_predicate,
)

_DT = 0.1


def _oscillatory_trajectory() -> dict[str, object]:
    """A zig-zag-in-place trajectory: many heading flips, ~zero net progress."""
    positions = np.array(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]
    )
    headings = np.array([0.0, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    velocities = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    return {"positions": positions, "headings": headings, "linear_velocities": velocities}


def _straight_trajectory() -> dict[str, object]:
    """A smooth straight run: no heading flips, full progress."""
    positions = np.array([[float(i), 0.0] for i in range(7)])
    headings = np.zeros(7)
    velocities = np.ones(7)
    return {"positions": positions, "headings": headings, "linear_velocities": velocities}


def test_oscillatory_trajectory_is_flagged() -> None:
    """A zig-zag-in-place trajectory must classify as oscillation with auditable fields."""
    result = oscillatory_control_predicate(**_oscillatory_trajectory(), dt=_DT)

    assert result["oscillation"] is True
    fields = result["fields"]
    assert fields["heading_rate_sign_changes"] >= 4
    assert fields["progress_ratio"] == pytest.approx(0.0)
    assert fields["path_length_m"] == pytest.approx(3.0)


def test_straight_trajectory_is_not_flagged() -> None:
    """A smooth straight run must not classify as oscillation."""
    result = oscillatory_control_predicate(**_straight_trajectory(), dt=_DT)

    assert result["oscillation"] is False
    fields = result["fields"]
    assert fields["heading_rate_sign_changes"] == 0
    assert fields["progress_ratio"] == pytest.approx(1.0)


def test_record_is_schema_tagged_and_labeled_diagnostic() -> None:
    """The produced record must carry the versioned schema and diagnostic label."""
    result = oscillatory_control_predicate(**_straight_trajectory(), dt=_DT)

    assert result["schema_version"] == OSCILLATORY_PREDICATE_SCHEMA
    assert result["predicate"] == "oscillatory_control"
    assert result["evidence_kind"] == "diagnostic_proxy"
    assert result["thresholds"]["min_heading_rate_sign_changes"] == 4


def test_command_source_changes_are_counted() -> None:
    """Command-source changes must be counted when the optional signal is provided."""
    traj = _straight_trajectory()
    sources = ["planner", "planner", "fallback", "fallback", "planner", "planner", "planner"]
    result = oscillatory_control_predicate(**traj, dt=_DT, command_sources=sources)

    assert result["fields"]["command_source_changes"] == 2


def test_velocity_sign_changes_counted() -> None:
    """Linear-velocity sign changes must reflect forward/reverse oscillation."""
    result = oscillatory_control_predicate(**_oscillatory_trajectory(), dt=_DT)

    # velocities = [0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5] -> 6 sign changes
    assert result["fields"]["linear_velocity_sign_changes"] == 6


def test_thresholds_are_overridable() -> None:
    """A stricter heading-flip threshold must be able to declassify a borderline case."""
    traj = _oscillatory_trajectory()
    strict = oscillatory_control_predicate(**traj, dt=_DT, min_heading_rate_sign_changes=99)

    assert strict["oscillation"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kw: kw.update(dt=0.0),
        lambda kw: kw.update(headings=np.zeros(3)),  # length mismatch
    ],
)
def test_invalid_inputs_are_rejected(mutate) -> None:
    """Bad dt or mismatched signal lengths must fail closed."""
    kwargs = {**_straight_trajectory(), "dt": _DT}
    mutate(kwargs)
    with pytest.raises(ValueError):
        oscillatory_control_predicate(**kwargs)


def test_single_step_is_rejected() -> None:
    """A trajectory shorter than two steps cannot be evaluated."""
    with pytest.raises(ValueError):
        oscillatory_control_predicate(
            positions=np.array([[0.0, 0.0]]),
            headings=np.array([0.0]),
            linear_velocities=np.array([0.0]),
            dt=_DT,
        )


# --- late-evasive reaction predicate -----------------------------------------

_HAZARD_DISTANCES = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
_HAZARD_VISIBLE = np.array([False, True, True, True, True, True])


def test_late_evasive_when_no_clearance_action_taken() -> None:
    """Hazard visible but constant speed (no deceleration) ⇒ late evasive."""
    result = late_evasive_predicate(_HAZARD_DISTANCES, _HAZARD_VISIBLE, np.ones(6), dt=_DT)

    assert result["late_evasive"] is True
    fields = result["fields"]
    assert fields["first_hazard_visible_step"] == 1
    assert fields["first_clearance_restoring_action_step"] is None
    assert fields["minimum_distance_m"] == pytest.approx(0.5)


def test_prompt_deceleration_is_not_late() -> None:
    """A prompt deceleration right after visibility is not late."""
    speeds = np.array([1.0, 1.0, 0.5, 0.2, 0.1, 0.0])
    result = late_evasive_predicate(_HAZARD_DISTANCES, _HAZARD_VISIBLE, speeds, dt=_DT)

    assert result["late_evasive"] is False
    assert result["fields"]["first_clearance_restoring_action_step"] == 2
    assert result["fields"]["response_latency_s"] == pytest.approx(0.1)


def test_latency_threshold_can_flag_a_borderline_reaction() -> None:
    """A reaction within default latency becomes late under a stricter threshold."""
    speeds = np.array([1.0, 1.0, 0.5, 0.2, 0.1, 0.0])
    result = late_evasive_predicate(
        _HAZARD_DISTANCES, _HAZARD_VISIBLE, speeds, dt=_DT, max_response_latency_s=0.05
    )

    assert result["late_evasive"] is True


def test_no_visible_hazard_is_not_late_evasive() -> None:
    """With no hazard ever visible, the predicate must not fire."""
    result = late_evasive_predicate(_HAZARD_DISTANCES, np.zeros(6, dtype=bool), np.ones(6), dt=_DT)

    assert result["late_evasive"] is False
    assert result["fields"]["first_hazard_visible_step"] is None


def test_required_deceleration_uses_visibility_state() -> None:
    """Required deceleration must be v^2/(2 d) at first visibility."""
    result = late_evasive_predicate(_HAZARD_DISTANCES, _HAZARD_VISIBLE, np.full(6, 2.0), dt=_DT)

    # At step 1: v=2.0, d=4.0 -> 4/(2*4) = 0.5
    assert result["fields"]["required_deceleration_m_s2"] == pytest.approx(0.5)


def test_late_evasive_record_is_schema_tagged() -> None:
    """The late-evasive record must carry its versioned schema and diagnostic label."""
    result = late_evasive_predicate(_HAZARD_DISTANCES, _HAZARD_VISIBLE, np.ones(6), dt=_DT)

    assert result["schema_version"] == LATE_EVASIVE_PREDICATE_SCHEMA
    assert result["predicate"] == "late_evasive"
    assert result["evidence_kind"] == "diagnostic_proxy"


def test_late_evasive_rejects_bad_inputs() -> None:
    """Bad dt, mismatched lengths, and single-step inputs must fail closed."""
    with pytest.raises(ValueError):
        late_evasive_predicate(_HAZARD_DISTANCES, _HAZARD_VISIBLE, np.ones(6), dt=0.0)
    with pytest.raises(ValueError):
        late_evasive_predicate(_HAZARD_DISTANCES, _HAZARD_VISIBLE, np.ones(3), dt=_DT)
    with pytest.raises(ValueError):
        late_evasive_predicate(np.array([1.0]), np.array([True]), np.array([1.0]), dt=_DT)
