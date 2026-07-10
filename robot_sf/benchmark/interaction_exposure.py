"""Interaction-exposure helpers for episode-row evidence sidecars."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
from robot_sf.errors import RobotSfError

INTERACTION_EXPOSURE_SCHEMA_VERSION = "interaction_exposure.v1"

# Canonical claim-bearing interaction-exposure fields. Downstream consumers (for example the
# issue #3813 sustained-flow revival gate) must import this tuple instead of re-declaring the
# field names, so the required-field contract has a single owner.
INTERACTION_EXPOSURE_REQUIRED_FIELDS: tuple[str, ...] = (
    "interaction_exposure_share",
    "robot_motion_share_before_first_clearance",
    "first_clearance_step",
    "low_exposure_success",
)

# Status string emitted when the fields were successfully computed from retained trajectories.
INTERACTION_EXPOSURE_COMPUTED_STATUS = "computed"

# Prefix marking a status where the fields were retained/attempted but could not be derived from
# the episode rows (for example a missing trace or no pedestrians present). Consumers use this to
# distinguish "field never retained" from "field retained but not derivable".
NOT_DERIVABLE_STATUS_PREFIX = "not_derivable_"


def is_not_derivable_status(status: object) -> bool:
    """Return whether ``status`` marks a retained-but-not-derivable exposure record."""

    return isinstance(status, str) and status.strip().startswith(NOT_DERIVABLE_STATUS_PREFIX)


class InteractionExposureError(RobotSfError, ValueError):
    """Raised when interaction-exposure inputs are malformed."""


def _xy(point: Sequence[float]) -> tuple[float, float]:
    if len(point) < 2:
        raise InteractionExposureError("positions must contain at least x and y")
    x = float(point[0])
    y = float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise InteractionExposureError("positions must be finite")
    return x, y


def _pedestrian_points(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.values()
    return [_xy(point) for point in value]


def _within_radius(
    robot_position: Sequence[float],
    pedestrians: Any,
    *,
    exposure_radius_m: float,
) -> bool:
    robot_x, robot_y = _xy(robot_position)
    for pedestrian_x, pedestrian_y in _pedestrian_points(pedestrians):
        if math.hypot(robot_x - pedestrian_x, robot_y - pedestrian_y) <= exposure_radius_m:
            return True
    return False


def _robot_moved(previous: Sequence[float], current: Sequence[float]) -> bool:
    previous_x, previous_y = _xy(previous)
    current_x, current_y = _xy(current)
    return math.hypot(current_x - previous_x, current_y - previous_y) > 1e-9


def not_derivable_interaction_exposure(reason: str) -> dict[str, Any]:
    """Return explicit non-zero-imputed interaction-exposure fields.

    Returns:
        An interaction-exposure record with unavailable numeric values left blank.
    """

    status = reason.strip() or "not_derivable_missing_trace"
    return {
        "interaction_exposure_schema_version": INTERACTION_EXPOSURE_SCHEMA_VERSION,
        "interaction_exposure_share": "",
        "robot_motion_share_before_first_clearance": "",
        "first_clearance_step": "",
        "low_exposure_success": "",
        "interaction_exposure_radius_m": "",
        "interaction_exposure_steps": "",
        "interaction_exposure_denominator_steps": "",
        "robot_motion_steps_before_first_clearance": "",
        "robot_motion_denominator_steps_before_first_clearance": "",
        "first_clearance_reason": "missing_trace",
        "interaction_exposure_status": status,
        "low_exposure_success_threshold": "",
    }


def compute_interaction_exposure_fields(
    *,
    robot_positions: Sequence[Sequence[float]],
    pedestrian_positions: Sequence[Any],
    dt: float,
    exposure_radius_m: float,
    low_exposure_success_threshold: float,
    success: bool,
) -> dict[str, Any]:
    """Compute deterministic interaction-exposure fields from retained trajectories.

    Returns:
        Interaction-exposure schema fields computed from per-step positions.
    """

    if dt <= 0 or not math.isfinite(dt):
        raise InteractionExposureError("dt must be positive and finite")
    if exposure_radius_m < 0 or not math.isfinite(exposure_radius_m):
        raise InteractionExposureError("exposure_radius_m must be non-negative and finite")
    if not 0 <= low_exposure_success_threshold <= 1:
        raise InteractionExposureError("low_exposure_success_threshold must be in [0, 1]")
    if len(robot_positions) != len(pedestrian_positions):
        raise InteractionExposureError("robot and pedestrian traces must have equal length")
    if not robot_positions:
        return not_derivable_interaction_exposure("not_derivable_missing_trace")

    denominator_steps = len(robot_positions)
    pedestrian_present = any(_pedestrian_points(frame) for frame in pedestrian_positions)
    if not pedestrian_present:
        return {
            "interaction_exposure_schema_version": INTERACTION_EXPOSURE_SCHEMA_VERSION,
            "interaction_exposure_share": 0.0,
            "robot_motion_share_before_first_clearance": 0.0,
            "first_clearance_step": "",
            "low_exposure_success": bool(success),
            "interaction_exposure_radius_m": exposure_radius_m,
            "interaction_exposure_steps": 0,
            "interaction_exposure_denominator_steps": denominator_steps,
            "robot_motion_steps_before_first_clearance": 0,
            "robot_motion_denominator_steps_before_first_clearance": 0,
            "first_clearance_reason": "no_pedestrians",
            "interaction_exposure_status": "not_derivable_no_pedestrians",
            "low_exposure_success_threshold": low_exposure_success_threshold,
        }

    exposed_flags = [
        _within_radius(robot, pedestrians, exposure_radius_m=exposure_radius_m)
        for robot, pedestrians in zip(robot_positions, pedestrian_positions, strict=True)
    ]
    exposure_steps = sum(exposed_flags)
    exposure_share = exposure_steps / denominator_steps

    first_clearance_step = next(
        (index for index, is_exposed in enumerate(exposed_flags) if not is_exposed),
        None,
    )
    if first_clearance_step is None:
        motion_denominator = max(0, denominator_steps - 1)
        motion_steps = sum(
            _robot_moved(previous, current) for previous, current in pairwise(robot_positions)
        )
        first_clearance_reason = "never_cleared"
    else:
        motion_denominator = max(0, first_clearance_step)
        motion_steps = sum(
            _robot_moved(previous, current)
            for previous, current in pairwise(robot_positions[: first_clearance_step + 1])
        )
        first_clearance_reason = "clearance_observed"

    motion_share = motion_steps / motion_denominator if motion_denominator else 0.0
    return {
        "interaction_exposure_schema_version": INTERACTION_EXPOSURE_SCHEMA_VERSION,
        "interaction_exposure_share": exposure_share,
        "robot_motion_share_before_first_clearance": motion_share,
        "first_clearance_step": "" if first_clearance_step is None else first_clearance_step,
        "low_exposure_success": bool(success and exposure_share < low_exposure_success_threshold),
        "interaction_exposure_radius_m": exposure_radius_m,
        "interaction_exposure_steps": exposure_steps,
        "interaction_exposure_denominator_steps": denominator_steps,
        "robot_motion_steps_before_first_clearance": motion_steps,
        "robot_motion_denominator_steps_before_first_clearance": motion_denominator,
        "first_clearance_reason": first_clearance_reason,
        "interaction_exposure_status": "computed",
        "low_exposure_success_threshold": low_exposure_success_threshold,
    }
