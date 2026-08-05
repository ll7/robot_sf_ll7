"""Phase and process summaries for worked-example traces."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PHASE_PROFILE_VERSION = "worked_example_phase_profile.v1"
REVERSAL_PROFILE_VERSION = "worked_example_reversal_profile.v1"


def summarize_stall(
    frames: Sequence[Mapping[str, Any]],
    *,
    speed_getter: Any,
    stall_speed_threshold_mps: float,
) -> dict[str, Any]:
    """Summarize sustained low-speed time.

    Returns:
        JSON-safe stall summary.
    """

    duration = duration_where(
        frames,
        lambda frame: (
            speed_getter(frame) is not None
            and (speed_getter(frame) or 0.0) < stall_speed_threshold_mps
        ),
    )
    return {
        "profile_version": PHASE_PROFILE_VERSION,
        "status": "available",
        "sustained_stall_duration_s": duration,
    }


def summarize_reversals(
    frames: Sequence[Mapping[str, Any]],
    *,
    speed_getter: Any,
    heading_delta_threshold_rad: float,
) -> dict[str, Any]:
    """Count route-velocity and heading reversals.

    Returns:
        JSON-safe reversal summary.
    """

    signs: list[int] = []
    heading_reversals = 0
    previous_heading: float | None = None
    for frame in frames:
        route = frame["route"]
        if route.get("status") == "available" and route.get("progress_rate_mps") is not None:
            rate = float(route["progress_rate_mps"])
            signs.append(1 if rate > 1e-6 else -1 if rate < -1e-6 else 0)
        heading = frame["world"]["robot"].get("heading") if frame["world"]["robot"] else None
        if isinstance(heading, int | float):
            if (
                previous_heading is not None
                and abs(float(heading) - previous_heading) > heading_delta_threshold_rad
            ):
                heading_reversals += 1
            previous_heading = float(heading)
        if speed_getter(frame) is None:
            continue
    velocity_reversals = sum(
        1 for left, right in pairwise(signs) if left != 0 and right not in (0, left)
    )
    return {
        "profile_version": REVERSAL_PROFILE_VERSION,
        "heading_reversal_count": heading_reversals,
        "velocity_reversal_count": velocity_reversals,
    }


def duration_where(frames: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    """Return left-sample step duration for frames matching ``predicate``.

    Returns:
        Duration in seconds.
    """

    duration = 0.0
    for left, right in pairwise(frames):
        if predicate(left):
            duration += float(right["time_s"]) - float(left["time_s"])
    return duration
