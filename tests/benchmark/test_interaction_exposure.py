"""Tests for retained interaction-exposure row helpers."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.interaction_exposure import (
    InteractionExposureError,
    compute_interaction_exposure_fields,
    not_derivable_interaction_exposure,
)


def test_compute_interaction_exposure_fields_records_first_clearance_and_motion() -> None:
    """Exposure fields are derived from retained robot and pedestrian positions."""

    fields = compute_interaction_exposure_fields(
        robot_positions=((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0)),
        pedestrian_positions=(
            ((0.25, 0.0),),
            ((0.75, 0.0),),
            ((5.0, 0.0),),
            ((5.5, 0.0),),
        ),
        success=True,
        exposure_radius_m=0.5,
        low_exposure_success_threshold=0.5,
    )

    assert fields["interaction_exposure_share"] == pytest.approx(0.5)
    assert fields["robot_motion_share_before_first_clearance"] == pytest.approx(1.0)
    assert fields["first_clearance_step"] == 2
    assert fields["low_exposure_success"] is True
    assert fields["interaction_exposure_status"] == "computed"


def test_compute_interaction_exposure_fields_records_never_cleared() -> None:
    """Never-cleared episodes retain an empty first-clearance marker."""

    fields = compute_interaction_exposure_fields(
        robot_positions=((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        pedestrian_positions=(((0.1, 0.0),), ((0.1, 0.0),), ((1.1, 0.0),)),
        success=True,
        exposure_radius_m=0.5,
        low_exposure_success_threshold=0.25,
    )

    assert fields["interaction_exposure_share"] == pytest.approx(1.0)
    assert fields["robot_motion_share_before_first_clearance"] == pytest.approx(0.5)
    assert fields["first_clearance_step"] == ""
    assert fields["low_exposure_success"] is False


def test_not_derivable_interaction_exposure_retains_required_empty_fields() -> None:
    """Rows without trace inputs still retain the required schema fields."""

    fields = not_derivable_interaction_exposure("missing_trace")

    assert fields["interaction_exposure_share"] == ""
    assert fields["robot_motion_share_before_first_clearance"] == ""
    assert fields["first_clearance_step"] == ""
    assert fields["low_exposure_success"] == ""
    assert fields["interaction_exposure_status"] == "not_derivable_missing_trace"


def test_compute_interaction_exposure_fields_rejects_mismatched_traces() -> None:
    """Malformed retained traces fail closed instead of computing partial fields."""

    with pytest.raises(InteractionExposureError, match="length mismatch"):
        compute_interaction_exposure_fields(
            robot_positions=((0.0, 0.0),),
            pedestrian_positions=(),
            success=False,
            exposure_radius_m=0.5,
            low_exposure_success_threshold=0.25,
        )
