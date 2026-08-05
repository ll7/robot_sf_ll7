"""Phase and process summaries for worked-example traces."""

from __future__ import annotations

import math
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
    stall_min_duration_s: float,
) -> dict[str, Any]:
    """Summarize sustained low-speed time.

    Returns:
        JSON-safe stall summary.
    """

    runs = _low_speed_runs(
        frames,
        speed_getter=speed_getter,
        stall_speed_threshold_mps=stall_speed_threshold_mps,
    )
    qualifying_runs = [
        (start, end)
        for start, end in runs
        if _run_duration(frames, start, end) >= stall_min_duration_s
    ]
    missing_speed_count = sum(1 for frame in frames if speed_getter(frame) is None)
    missing_speed_blocks_duration = missing_speed_count > 0 and bool(runs)
    duration = (
        None
        if missing_speed_blocks_duration
        else sum(_run_duration(frames, start, end) for start, end in qualifying_runs)
    )
    onset = (
        None
        if missing_speed_blocks_duration
        else first_sustained_stall_frame(
            frames,
            speed_getter=speed_getter,
            stall_speed_threshold_mps=stall_speed_threshold_mps,
            stall_min_duration_s=stall_min_duration_s,
        )
    )
    return {
        "profile_version": PHASE_PROFILE_VERSION,
        "status": "unavailable" if missing_speed_blocks_duration else "available",
        "reason": "missing_speed_within_candidate_stall"
        if missing_speed_blocks_duration
        else "coverage_complete",
        "stall_min_duration_s": stall_min_duration_s,
        "sustained_stall_duration_s": duration,
        "speed_coverage": {
            "status": "complete" if missing_speed_count == 0 else "partial",
            "available_frame_count": len(frames) - missing_speed_count,
            "missing_frame_count": missing_speed_count,
        },
        "sustained_stall_onset_step": int(onset["step"]) if onset is not None else None,
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
                and abs(_wrapped_angle_delta(float(heading), previous_heading))
                > heading_delta_threshold_rad
            ):
                heading_reversals += 1
            previous_heading = float(heading)
        if speed_getter(frame) is None:
            continue
    velocity_reversals = _nonzero_sign_reversals(signs)
    return {
        "profile_version": REVERSAL_PROFILE_VERSION,
        "direction_semantics": "route_progress_rate_sign_and_wrapped_heading_delta",
        "heading_reversal_count": heading_reversals,
        "velocity_reversal_count": velocity_reversals,
    }


def first_sustained_stall_frame(
    frames: Sequence[Mapping[str, Any]],
    *,
    speed_getter: Any,
    stall_speed_threshold_mps: float,
    stall_min_duration_s: float,
) -> Mapping[str, Any] | None:
    """Return the first frame in a low-speed run that satisfies minimum duration.

    Returns:
        Frame mapping or ``None``.
    """

    for start, end in _low_speed_runs(
        frames,
        speed_getter=speed_getter,
        stall_speed_threshold_mps=stall_speed_threshold_mps,
    ):
        duration = float(frames[end]["time_s"]) - float(frames[start]["time_s"])
        if duration >= stall_min_duration_s:
            return frames[start]
    return None


def first_recovery_frame(
    frames: Sequence[Mapping[str, Any]],
    *,
    speed_getter: Any,
    stall_speed_threshold_mps: float,
    stall_min_duration_s: float,
    recovery_speed_threshold_mps: float,
) -> Mapping[str, Any] | None:
    """Return first recovery frame after a qualifying stall run.

    Returns:
        Frame mapping or ``None``.
    """

    onset = first_sustained_stall_frame(
        frames,
        speed_getter=speed_getter,
        stall_speed_threshold_mps=stall_speed_threshold_mps,
        stall_min_duration_s=stall_min_duration_s,
    )
    if onset is None:
        return None
    after_onset = False
    for frame in frames:
        if frame is onset:
            after_onset = True
        if not after_onset:
            continue
        speed = speed_getter(frame)
        if speed is not None and speed >= recovery_speed_threshold_mps:
            return frame
    return None


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


def _low_speed_runs(
    frames: Sequence[Mapping[str, Any]],
    *,
    speed_getter: Any,
    stall_speed_threshold_mps: float,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, frame in enumerate(frames):
        speed = speed_getter(frame)
        is_stalled = speed is not None and speed < stall_speed_threshold_mps
        if is_stalled and start is None:
            start = index
        if not is_stalled and start is not None:
            runs.append((start, max(start, index - 1)))
            start = None
    if start is not None:
        runs.append((start, len(frames) - 1))
    return runs


def _run_duration(frames: Sequence[Mapping[str, Any]], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    return float(frames[end]["time_s"]) - float(frames[start]["time_s"])


def _wrapped_angle_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def _nonzero_sign_reversals(signs: Sequence[int]) -> int:
    reversals = 0
    previous_nonzero: int | None = None
    for sign in signs:
        if sign == 0:
            continue
        if previous_nonzero is not None and sign != previous_nonzero:
            reversals += 1
        previous_nonzero = sign
    return reversals
