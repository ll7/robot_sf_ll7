"""Interaction-exposure helpers for retained episode-row evidence."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

INTERACTION_EXPOSURE_SCHEMA_VERSION = "interaction_exposure.v1"


class InteractionExposureError(ValueError):
    """Raised when interaction-exposure inputs are malformed."""


def not_derivable_interaction_exposure(reason: str) -> dict[str, Any]:
    """Return explicit non-imputed interaction-exposure fields."""

    return {
        "interaction_exposure_schema_version": INTERACTION_EXPOSURE_SCHEMA_VERSION,
        "interaction_exposure_share": "",
        "robot_motion_share_before_first_clearance": "",
        "first_clearance_step": "",
        "low_exposure_success": "",
        "interaction_exposure_status": f"not_derivable_{reason}",
    }


def compute_interaction_exposure_fields(
    *,
    robot_positions: Sequence[Sequence[float]],
    pedestrian_positions: Sequence[Any],
    success: bool,
    exposure_radius_m: float,
    low_exposure_success_threshold: float,
) -> dict[str, Any]:
    """Compute row-level interaction-exposure fields from retained positions.

    Returns:
        Required interaction-exposure row fields plus diagnostic provenance fields.
    """

    if exposure_radius_m <= 0.0 or not math.isfinite(exposure_radius_m):
        raise InteractionExposureError("exposure_radius_m must be positive and finite")
    if not 0.0 <= low_exposure_success_threshold <= 1.0:
        raise InteractionExposureError("low_exposure_success_threshold must be in [0, 1]")
    if len(robot_positions) != len(pedestrian_positions):
        raise InteractionExposureError("robot_positions and pedestrian_positions length mismatch")
    if not robot_positions:
        return not_derivable_interaction_exposure("empty_trace")

    denominator_steps = len(robot_positions)
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
    else:
        motion_denominator = max(0, first_clearance_step)
        motion_steps = sum(
            _robot_moved(previous, current)
            for previous, current in pairwise(robot_positions[: first_clearance_step + 1])
        )

    motion_share = motion_steps / motion_denominator if motion_denominator else 0.0
    return {
        "interaction_exposure_schema_version": INTERACTION_EXPOSURE_SCHEMA_VERSION,
        "interaction_exposure_share": exposure_share,
        "robot_motion_share_before_first_clearance": motion_share,
        "first_clearance_step": "" if first_clearance_step is None else first_clearance_step,
        "low_exposure_success": bool(success and exposure_share <= low_exposure_success_threshold),
        "interaction_exposure_status": "computed",
    }


def _within_radius(
    robot_position: Sequence[float],
    pedestrians: Any,
    *,
    exposure_radius_m: float,
) -> bool:
    robot_x, robot_y = _xy(robot_position)
    return any(
        math.hypot(robot_x - pedestrian_x, robot_y - pedestrian_y) <= exposure_radius_m
        for pedestrian_x, pedestrian_y in _pedestrian_points(pedestrians)
    )


def _pedestrian_points(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.values()
    return [_xy(point) for point in value]


def _robot_moved(previous: Sequence[float], current: Sequence[float]) -> bool:
    previous_x, previous_y = _xy(previous)
    current_x, current_y = _xy(current)
    return math.hypot(current_x - previous_x, current_y - previous_y) > 1e-9


def _xy(point: Sequence[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise InteractionExposureError("positions must contain x and y")
    x = float(point[0])
    y = float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise InteractionExposureError("positions must be finite")
    return x, y
