"""Tests for interaction-exposure helpers."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.interaction_exposure import (
    InteractionExposureError,
    compute_interaction_exposure_fields,
    not_derivable_interaction_exposure,
)


def test_interaction_exposure_computes_deterministic_denominators() -> None:
    """Tiny traces produce expected exposure and pre-clearance motion shares."""

    fields = compute_interaction_exposure_fields(
        robot_positions=[(0.0, 0.0), (0.5, 0.0), (2.0, 0.0), (3.0, 0.0)],
        pedestrian_positions=[
            [(0.2, 0.0)],
            [(0.8, 0.0)],
            [(10.0, 0.0)],
            [(10.0, 0.0)],
        ],
        dt=0.1,
        exposure_radius_m=1.0,
        low_exposure_success_threshold=0.75,
        success=True,
    )

    assert fields["interaction_exposure_status"] == "computed"
    assert fields["interaction_exposure_steps"] == 2
    assert fields["interaction_exposure_denominator_steps"] == 4
    assert fields["interaction_exposure_share"] == pytest.approx(0.5)
    assert fields["first_clearance_step"] == 2
    assert fields["robot_motion_steps_before_first_clearance"] == 2
    assert fields["robot_motion_denominator_steps_before_first_clearance"] == 2
    assert fields["robot_motion_share_before_first_clearance"] == pytest.approx(1.0)
    assert fields["low_exposure_success"] is True


def test_low_exposure_success_requires_success() -> None:
    """Low exposure alone is not a success diagnostic when the episode failed."""

    fields = compute_interaction_exposure_fields(
        robot_positions=[(0.0, 0.0), (5.0, 0.0)],
        pedestrian_positions=[[(10.0, 0.0)], [(10.0, 0.0)]],
        dt=0.1,
        exposure_radius_m=1.0,
        low_exposure_success_threshold=0.75,
        success=False,
    )

    assert fields["interaction_exposure_share"] == pytest.approx(0.0)
    assert fields["low_exposure_success"] is False


def test_not_derivable_interaction_exposure_does_not_write_zeroes() -> None:
    """Missing traces are explicit statuses, not zero-valued exposure rows."""

    fields = not_derivable_interaction_exposure("not_derivable_missing_trace")

    assert fields["interaction_exposure_status"] == "not_derivable_missing_trace"
    assert fields["interaction_exposure_share"] == ""
    assert fields["first_clearance_reason"] == "missing_trace"


def test_interaction_exposure_no_pedestrians_is_not_derivable_status() -> None:
    """Episodes with no pedestrians keep a status distinct from missing traces."""

    fields = compute_interaction_exposure_fields(
        robot_positions=[(0.0, 0.0), (1.0, 0.0)],
        pedestrian_positions=[[], []],
        dt=0.1,
        exposure_radius_m=1.0,
        low_exposure_success_threshold=0.75,
        success=True,
    )

    assert fields["interaction_exposure_status"] == "not_derivable_no_pedestrians"
    assert fields["first_clearance_reason"] == "no_pedestrians"
    assert fields["low_exposure_success"] is True


def test_interaction_exposure_never_cleared_counts_motion_until_end() -> None:
    """Always-exposed traces use the whole trajectory before-clearance denominator."""

    fields = compute_interaction_exposure_fields(
        robot_positions=[(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        pedestrian_positions=[[(0.2, 0.0)], [(0.2, 0.0)], [(1.2, 0.0)]],
        dt=0.1,
        exposure_radius_m=1.0,
        low_exposure_success_threshold=0.75,
        success=True,
    )

    assert fields["interaction_exposure_status"] == "computed"
    assert fields["first_clearance_step"] == ""
    assert fields["first_clearance_reason"] == "never_cleared"
    assert fields["robot_motion_steps_before_first_clearance"] == 1
    assert fields["robot_motion_denominator_steps_before_first_clearance"] == 2


def test_interaction_exposure_rejects_malformed_inputs() -> None:
    """Malformed trajectories fail loudly instead of producing sidecar values."""

    with pytest.raises(InteractionExposureError, match="equal length"):
        compute_interaction_exposure_fields(
            robot_positions=[(0.0, 0.0)],
            pedestrian_positions=[[], []],
            dt=0.1,
            exposure_radius_m=1.0,
            low_exposure_success_threshold=0.75,
            success=True,
        )
    with pytest.raises(InteractionExposureError, match="positive"):
        compute_interaction_exposure_fields(
            robot_positions=[(0.0, 0.0)],
            pedestrian_positions=[[]],
            dt=0.0,
            exposure_radius_m=1.0,
            low_exposure_success_threshold=0.75,
            success=True,
        )
