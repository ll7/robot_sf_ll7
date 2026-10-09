"""Opt-in footprint metrics; released metric names keep their historical meaning."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.constants import COMFORT_FORCE_THRESHOLD

if TYPE_CHECKING:
    from robot_sf.benchmark.metrics import EpisodeData

FOOTPRINT_SCHEMA = "robot-sf-footprint.v1"
FOOTPRINT_MARKER = "footprint_metric_schema_version"


def footprint_enabled(metadata: dict[str, Any] | None) -> bool:
    """Validate the optional definition marker.

    Returns:
        Whether footprint definitions are enabled.
    """
    marker = (metadata or {}).get(FOOTPRINT_MARKER)
    if marker is not None and marker != FOOTPRINT_SCHEMA:
        raise ValueError(f"unsupported footprint metric definition: {marker}")
    return marker == FOOTPRINT_SCHEMA


def require_radius(value: Any) -> float:
    """Return a finite nonnegative physical radius, rejecting missing or invalid data."""
    if isinstance(value, bool) or value is None:
        raise ValueError("footprint radius must be finite and nonnegative")
    radius = float(value)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("footprint radius must be finite and nonnegative")
    return radius


def trace_radii(trace: dict[str, Any]) -> tuple[float, float]:
    """Require physical radii on an opted-in diagnostic trace.

    Returns:
        Robot and pedestrian radii in metres.
    """
    return require_radius(trace.get("robot_radius_m")), require_radius(trace.get("ped_radius_m"))


def disc_contact_ttc(delta: np.ndarray, velocity: np.ndarray, radius: float) -> np.ndarray:
    """Solve |delta - velocity*t| = radius for first nonnegative contact.

    Already touching/overlapping pairs return zero, including stationary pairs.
    Misses, receding pairs and stationary separated pairs return infinity.
    Returns:
        Contact times, or infinity for noncontact pairs.

    Arrays end in XY; a finite row represents a present pair. Nonfinite padding
    is ignored. Velocity is robot minus pedestrian velocity.
    """
    radius = require_radius(radius)
    delta, velocity = np.asarray(delta, dtype=float), np.asarray(velocity, dtype=float)
    if delta.shape != velocity.shape or delta.shape[-1] != 2:
        raise ValueError("TTC positions and velocities must have matching XY shapes")
    a = np.sum(velocity**2, axis=-1)
    b = np.sum(delta * velocity, axis=-1)
    c = np.sum(delta**2, axis=-1) - radius**2
    finite = np.isfinite(delta).all(axis=-1) & np.isfinite(velocity).all(axis=-1)
    discriminant = b**2 - a * c
    out = np.full(a.shape, np.inf)
    out[finite & (c <= 0)] = 0.0
    hit = finite & (c > 0) & (a > 1e-18) & (b > 0) & (discriminant >= 0)
    # Stable small root; avoids cancellation when contact is close.
    out[hit] = c[hit] / (b[hit] + np.sqrt(discriminant[hit]))
    return out


def segment_clearance(positions: np.ndarray, segments: np.ndarray, radius: float) -> np.ndarray:
    """Minimum centre-to-segment distance minus radius for each sample.

    Segments have shape (M,2,2), including zero-length point segments. An empty
    wall set returns infinity. Unlike a point cloud, sparse long walls remain
    detectable between endpoints.

    Returns:
        Minimum signed wall gap per robot position.
    """
    radius = require_radius(radius)
    positions, segments = np.asarray(positions, dtype=float), np.asarray(segments, dtype=float)
    if segments.shape != (len(segments), 2, 2) or not np.isfinite(segments).all():
        raise ValueError("wall segments must be finite (M,2,2)")
    if not np.isfinite(positions).all():
        raise ValueError("robot positions must be finite")
    if len(segments) == 0:
        return np.full(len(positions), np.inf)
    start, direction = segments[:, 0], segments[:, 1] - segments[:, 0]
    squared_length = np.sum(direction**2, axis=-1)
    relative = positions[:, None, :] - start
    fraction = np.divide(
        np.sum(relative * direction, axis=-1),
        squared_length,
        out=np.zeros((len(positions), len(segments))),
        where=squared_length > 0,
    )
    projected = start + np.clip(fraction, 0, 1)[..., None] * direction
    return np.min(np.linalg.norm(positions[:, None, :] - projected, axis=-1), axis=1) - radius


def contact_ttc_matrix(data: EpisodeData) -> np.ndarray:
    """Contact TTC with backward pedestrian velocity differences.

    Returns:
        Per-frame, per-pedestrian contact times after the first sample.
    """
    if len(data.robot_pos) < 2:
        return np.empty((0, data.peds_pos.shape[1]))
    return disc_contact_ttc(
        data.peds_pos[1:] - data.robot_pos[1:, None, :],
        data.robot_vel[1:, None, :] - np.diff(data.peds_pos, axis=0) / data.dt,
        require_radius(data.robot_radius) + require_radius(data.ped_radius),
    )


def paired_force_data(data: EpisodeData) -> EpisodeData:
    """Build aligned pre-integration arrays from recorded force inputs, with presence.

    Total forces and input pedestrian positions must have the same cardinality
    in every frame. Padding denotes absence, never an invalid force sample.
    A missing or incomplete input series fails closed instead of shifting across
    respawns. Marginal force counts need no temporal shift; only geometry joins
    and presence denominators use this paired view.

    Returns:
        Episode view whose force positions are pre-integration inputs.
    """
    samples = data.robot_force_samples
    if not samples or len(samples) != len(data.robot_pos):
        raise ValueError("footprint metrics require complete pre-integration force inputs")
    counts = [len(sample["peds_pos"]) for sample in samples]
    width = max(counts, default=0)
    peds = np.full((len(samples), width, 2), np.nan)
    forces = np.full_like(peds, np.nan)
    present = np.zeros(peds.shape[:2], dtype=bool)
    robots = []
    for t, (sample, count) in enumerate(zip(samples, counts, strict=True)):
        positions = np.asarray(sample["peds_pos"], dtype=float).reshape(-1, 2)
        total = np.asarray(sample["total_forces"], dtype=float).reshape(-1, 2)
        robot = np.asarray(sample["robot_pos"], dtype=float)
        if positions.shape != total.shape or robot.shape != (2,):
            raise ValueError("force/input cardinality or robot pose differs")
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(total).all()
            or not np.isfinite(robot).all()
        ):
            raise ValueError("present force inputs must be finite")
        peds[t, :count], forces[t, :count], present[t, :count] = positions, total, True
        robots.append(robot)
    return replace(
        data,
        robot_pos=np.asarray(robots),
        peds_pos=peds,
        ped_forces=forces,
        robot_force_presence=present,
    )


def compute_footprint_metrics(
    data: EpisodeData, *, legacy_reference_length: float | None = None
) -> dict[str, Any]:
    """Compute the separately versioned, additive footprint block.

    Collision counts count contact samples (touching is contact). Personal-space
    violation is the fraction of frames with gap < 0.5m; pedestrian-free frames
    contribute zero, while a pedestrian-free episode is undefined. NaN TTC means
    no predicted contact. References are supplied by a map-aware producer.

    Returns:
        Separately versioned footprint metric block.
    """
    from robot_sf.benchmark.metrics import path_efficiency, space_compliance, time_to_goal  # noqa: PLC0415 - avoid metric module cycle

    if not np.isfinite(data.dt) or data.dt <= 0:
        raise ValueError("footprint metrics require finite positive dt")
    robot_radius, ped_radius = require_radius(data.robot_radius), require_radius(data.ped_radius)
    if data.obstacle_segments is None:
        if data.obstacles is not None and len(data.obstacles):
            raise ValueError("footprint wall collisions require segments, not sampled points")
        segments = np.empty((0, 2, 2))
    else:
        segments = data.obstacle_segments
    walls = segment_clearance(data.robot_pos, segments, robot_radius)
    if data.other_agents_pos is None or data.other_agents_pos.shape[1] == 0:
        agents = 0.0
    else:
        radii = np.asarray(data.other_agents_radii, dtype=float)
        if (
            radii.shape != (data.other_agents_pos.shape[1],)
            or not np.isfinite(radii).all()
            or np.any(radii < 0)
        ):
            raise ValueError("footprint metrics require one explicit radius per other agent")
        gaps = (
            np.linalg.norm(data.other_agents_pos - data.robot_pos[:, None], axis=-1)
            - robot_radius
            - radii
        )
        agents = float(np.count_nonzero(np.any(gaps <= 0, axis=1)))
    gaps = (
        np.linalg.norm(data.peds_pos - data.robot_pos[:, None], axis=-1) - robot_radius - ped_radius
    )
    present = np.isfinite(gaps)
    contacts = contact_ttc_matrix(data)
    finite_ttc = contacts[np.isfinite(contacts)]
    result: dict[str, Any] = {
        "schema_version": FOOTPRINT_SCHEMA,
        "legacy_comparison": {
            "shortest_path_len": legacy_reference_length,
            "space_compliance": space_compliance(data),
            "force_exceed_near_robot": float(
                np.count_nonzero(
                    (gaps < 0.5)
                    & (np.linalg.norm(data.ped_forces, axis=-1) > COMFORT_FORCE_THRESHOLD)
                )
            )
            if data.ped_forces.shape == data.peds_pos.shape
            else float("nan"),
        },
        "robot_radius_m": robot_radius,
        "ped_radius_m": ped_radius,
        "wall_collisions": float(np.count_nonzero(walls <= 0)),
        "agent_collisions": agents,
        "time_to_collision_min": float(finite_ttc.min()) if finite_ttc.size else float("nan"),
        "near_miss_ttc": {
            str(t): float(np.count_nonzero(np.any(contacts < t, axis=1))) for t in (1, 2, 3)
        },
        "space_compliance": float(np.mean(np.any(gaps < 0.5, axis=1)))
        if present.any()
        else float("nan"),
        "near_miss_gap_sensitivity": {
            str(t): float(np.count_nonzero(np.any((gaps > 0) & (gaps < t), axis=1)))
            for t in (0.1, 0.2, 0.3, 0.5)
        },
        "comfort_gap_sensitivity": {
            str(t): float(np.mean(np.any(gaps < t, axis=1))) if present.any() else float("nan")
            for t in (0.3, 0.5, 0.8, 1.2)
        },
    }
    reference = data.footprint_reference_length
    successful = data.reached_goal_step is not None and not data.collision_event
    reference = float("nan") if reference is None else reference
    result["shortest_path_len"] = reference
    result["reference_status"] = "available" if np.isfinite(reference) else "unavailable"
    result["path_efficiency"] = path_efficiency(data, reference, successful=successful)
    speed = data.footprint_max_speed
    ideal = reference / speed if speed is not None and speed > 0 else float("nan")
    result["time_to_goal_ideal_ratio"] = (
        time_to_goal(data) / ideal if successful and ideal > 0 else float("nan")
    )
    if data.robot_force_samples:
        paired = paired_force_data(data)
        presence = paired.robot_force_presence
        magnitudes = np.linalg.norm(paired.ped_forces, axis=-1)
        exceeded = (magnitudes > COMFORT_FORCE_THRESHOLD) & presence
        result["force_pairing"] = "pre-integration-inputs.v1"
        result["force_exceed_events"] = float(exceeded.sum())
        result["comfort_exposure"] = (
            float(exceeded.sum() / presence.sum()) if presence.any() else 0.0
        )
        force_gaps = (
            np.linalg.norm(paired.peds_pos - paired.robot_pos[:, None], axis=-1)
            - robot_radius
            - ped_radius
        )
        result["force_exceed_near_robot"] = float(np.count_nonzero(exceeded & (force_gaps < 0.5)))
    else:
        result["force_pairing"] = "unavailable"
        for key in ("force_exceed_events", "comfort_exposure", "force_exceed_near_robot"):
            result[key] = float("nan")
    return result


def snapshot_total_forces(forces: Any, positions: np.ndarray, *, required: bool) -> np.ndarray:
    """Copy total force buffers, rejecting missing/shape-invalid opted-in samples.

    Returns:
        Copied force array; the historical path retains its zero fallback.
    """
    if forces is None:
        if required:
            raise ValueError("footprint force pairing requires evaluated total forces")
        return np.zeros_like(positions, dtype=float)
    values = np.array(forces, dtype=float, copy=True)
    if values.shape != positions.shape:
        if required:
            raise ValueError("total force/input cardinality differs")
        return np.zeros_like(positions, dtype=float)
    return values
