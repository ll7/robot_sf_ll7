"""Bounded residual-control reactive adversary for runtime pedestrian stress testing.

This module delivers the maintainer pre-registered "first runtime implementation"
for issue #4360 (post-freeze half, item 2): an opt-in adversary that *perturbs*,
and does NOT replace, the nominal Social Force pedestrian behavior
(``pysocialforce.forces.SocialForce`` remains the base law). The adversary emits a
bounded residual acceleration at a fixed macro-action cadence (0.5 s by default)
that is added to the already-computed pedestrian forces, so the Social Force base
law is preserved and only perturbed.

Capability-only slice
---------------------
This is a capability-only slice. It makes no benchmark, planner-ranking, safety, or
paper-facing claim. It defines no new stress-case metric (issue item 4, which
requires a maintainer Domain-Aware Approval, is deferred). It implements no
CMA-ES/MCTS search-baseline adversary and no PPO/learned adversary; those are
later sequenced slices. The :class:`ResidualAdversaryPolicy` interface below is the
seam those future adversaries will plug into. The bundled
:class:`ScriptedPullResidualAdversaryPolicy` is a deterministic, bounded example
policy used for wiring and tests -- it is not the search or learned adversary.

Hard bounds enforced (all fail-closed on non-finite input)
----------------------------------------------------------
- speed (the residual may not push a pedestrian beyond its ``max_speed``)
- acceleration magnitude (``max_residual_accel_mps2``)
- jerk (rate of change of the residual acceleration, ``max_jerk_mps3``)
- heading change per macro-action (``max_heading_change_per_macro_rad``)
- route deviation (``max_route_deviation_m`` around an optional reference polyline)
- walkable-space projection (obstacle-margin push-out and bounds clamp)
- inter-agent separation (``min_separation_m``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, isfinite, sin
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from robot_sf.common.types import Line2D, RobotPose

EPSILON = 1e-9
"""Small positive tolerance guarding divide-by-zero in vectorized bound helpers."""

DEFAULT_MACRO_ACTION_DT_S = 0.5
"""Default macro-action cadence in seconds (the pre-registered 0.5 s value)."""

MIN_WALKABLE_MARGIN_M = 1e-3
"""Floor for the walkable-space projection margin so it is always strictly positive."""


def _require_finite(value: float, name: str, *, strict_positive: bool = False) -> None:
    """Raise ``ValueError`` when a scalar config value is non-finite or out of range."""
    if not isfinite(value):
        raise ValueError(f"{name} must be finite (got {value!r})")
    if strict_positive and value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value!r})")


def _validate_finite_array(array: np.ndarray, name: str) -> np.ndarray:
    """Return a float array, raising ``ValueError`` on non-finite entries."""
    coerced = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(coerced)):
        raise ValueError(f"{name} must be finite (got non-finite entries)")
    return coerced


@dataclass(frozen=True)
class ResidualAdversaryConfig:
    """Opt-in bounded residual-control adversary parameters.

    All bounds are enforced by :class:`BoundedResidualAdversary`. The adversary is
    strictly additive and off by default (``is_active=False``); enabling it perturbs
    the nominal Social Force base law without replacing it.

    Attributes
    ----------
    is_active:
        Opt-in master switch. ``False`` (default) leaves pedestrian behavior
        unchanged.
    macro_action_dt_s:
        Macro-action cadence in seconds. A new residual proposal is requested from
        the policy every ``round(macro_action_dt_s / dt_s)`` physics steps and held
        constant in between.
    max_residual_accel_mps2:
        Hard magnitude bound on the residual acceleration per pedestrian (m/s^2).
    max_jerk_mps3:
        Hard bound on the rate of change of the residual acceleration (m/s^3). The
        applied residual is moved toward the held proposal by at most
        ``max_jerk_mps3 * dt_s`` per physics step.
    max_speed_delta_mps:
        Upper bound on how much the residual may add to a pedestrian's speed in a
        single physics step (m/s). Bounds the speed-increasing component.
    max_heading_change_per_macro_rad:
        Maximum rotation of a pedestrian's velocity direction that the residual may
        cause over one macro-action window (radians).
    max_route_deviation_m:
        Maximum distance the residual may push a pedestrian away from its reference
        route polyline (m). Enforced only when a reference polyline is supplied.
    min_separation_m:
        Minimum inter-agent distance the residual must preserve between targeted
        pedestrians (m).
    target_ped_idx:
        Index or indices of pedestrians the adversary may perturb. Non-targeted
        pedestrians always receive a zero residual. ``-1`` targets all pedestrians.
    obstacle_projection_margin_m:
        Extra clearance beyond the pedestrian radius enforced by the walkable-space
        obstacle projection (m).
    seed:
        Optional deterministic seed for any randomized helper policy. The bundled
        scripted policy is deterministic and does not use it.
    """

    is_active: bool = False
    macro_action_dt_s: float = DEFAULT_MACRO_ACTION_DT_S
    max_residual_accel_mps2: float = 1.5
    max_jerk_mps3: float = 7.5
    max_speed_delta_mps: float = 0.5
    max_heading_change_per_macro_rad: float = 0.7853981633974483  # pi/4
    max_route_deviation_m: float = 1.5
    min_separation_m: float = 0.6
    target_ped_idx: int | list[int] = -1
    obstacle_projection_margin_m: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate finite, positive bounds and fail closed on malformed input."""
        _require_finite(self.macro_action_dt_s, "macro_action_dt_s", strict_positive=True)
        _require_finite(
            self.max_residual_accel_mps2, "max_residual_accel_mps2", strict_positive=True
        )
        _require_finite(self.max_jerk_mps3, "max_jerk_mps3", strict_positive=True)
        _require_finite(self.max_speed_delta_mps, "max_speed_delta_mps", strict_positive=True)
        _require_finite(
            self.max_heading_change_per_macro_rad,
            "max_heading_change_per_macro_rad",
            strict_positive=True,
        )
        _require_finite(self.max_route_deviation_m, "max_route_deviation_m", strict_positive=True)
        _require_finite(self.min_separation_m, "min_separation_m", strict_positive=True)
        _require_finite(
            self.obstacle_projection_margin_m, "obstacle_projection_margin_m", strict_positive=True
        )
        if not isinstance(self.target_ped_idx, int | list):
            raise TypeError("target_ped_idx must be an int or a list[int]")

    def resolve_target_mask(self, num_peds: int) -> np.ndarray:
        """Return a boolean ``(num_peds,)`` mask of targeted pedestrians.

        Out-of-range indices are dropped silently (they target no one), matching the
        existing :class:`~robot_sf.ped_npc.adversial_ped_force.AdversarialPedForce`
        convention. ``-1`` targets every pedestrian.
        """
        if num_peds < 0:
            raise ValueError("num_peds must be >= 0")
        mask = np.zeros(num_peds, dtype=bool)
        if num_peds == 0:
            return mask
        if isinstance(self.target_ped_idx, int):
            if self.target_ped_idx == -1:
                mask[:] = True
                return mask
            indices = [self.target_ped_idx]
        else:
            indices = list(self.target_ped_idx)
        for raw in indices:
            if not isinstance(raw, int):
                raise TypeError("target_ped_idx entries must be ints")
            if -num_peds <= raw < num_peds:
                mask[raw % num_peds] = True
        return mask


@dataclass(frozen=True)
class ResidualAdversaryObservation:
    """Per-step observation handed to a :class:`ResidualAdversaryPolicy`.

    The arrays are read-only snapshots of the pedestrian state at the macro-action
    boundary. Policies must not mutate them.
    """

    positions: np.ndarray
    velocities: np.ndarray
    max_speeds: np.ndarray
    target_ped_mask: np.ndarray
    robot_pose: RobotPose
    sim_time_s: float
    step_index: int
    macro_action_index: int


class ResidualAdversaryPolicy(Protocol):
    """Interface for a reactive residual adversary.

    A policy proposes an unbounded residual acceleration per pedestrian at each
    macro-action boundary. :class:`BoundedResidualAdversary` is responsible for
    enforcing every hard bound on the proposal, so a policy implementation may emit
    any finite ``(N, 2)`` array. The future CMA-ES/MCTS search-baseline adversary
    and any PPO/learned adversary will implement this interface; this slice ships
    only the deterministic :class:`ScriptedPullResidualAdversaryPolicy`.
    """

    def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
        """Return a finite ``(N, 2)`` proposed residual acceleration array."""
        ...


@dataclass
class ScriptedPullResidualAdversaryPolicy:
    """Deterministic example residual adversary that pulls targeted peds toward a point.

    The policy computes a point at ``pull_offset_m`` in front of the robot (along its
    heading) and proposes a residual acceleration of magnitude
    ``min(max_pull_accel_mps2, max_pull_accel_mps2)`` directed from each targeted
    pedestrian toward that point, scaled to ``max_pull_accel_mps2``. This is the
    bounded reactive baseline used for runtime wiring and tests.

    It is intentionally simple and deterministic. It is NOT the CMA-ES/MCTS
    search-baseline adversary and NOT a PPO/learned adversary; those are deferred
    slices that will implement :class:`ResidualAdversaryPolicy` separately.
    """

    max_pull_accel_mps2: float = 1.0
    pull_offset_m: float = 3.0

    def __post_init__(self) -> None:
        """Validate finite positive pull parameters."""
        _require_finite(self.max_pull_accel_mps2, "max_pull_accel_mps2", strict_positive=True)
        _require_finite(self.pull_offset_m, "pull_offset_m", strict_positive=True)

    def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
        """Return a finite ``(N, 2)`` pull-toward-robot proposal for targeted peds."""
        positions = _validate_finite_array(observation.positions, "observation.positions")
        num_peds = positions.shape[0]
        proposal = np.zeros((num_peds, 2), dtype=float)
        if num_peds == 0:
            return proposal
        if not np.any(observation.target_ped_mask):
            return proposal
        robot_pos = np.asarray(observation.robot_pose[0], dtype=float)
        if (not np.all(np.isfinite(robot_pos))) or not isfinite(float(observation.robot_pose[1])):
            raise ValueError("observation.robot_pose must be finite")
        heading = float(observation.robot_pose[1])
        target_point = robot_pos + self.pull_offset_m * np.array([cos(heading), sin(heading)])
        masked_positions = positions[observation.target_ped_mask]
        offsets = target_point - masked_positions
        norms = np.linalg.norm(offsets, axis=1)
        unit = np.zeros_like(offsets)
        nonzero = norms > EPSILON
        unit[nonzero] = offsets[nonzero] / norms[nonzero, None]
        scaled = unit * float(self.max_pull_accel_mps2)
        proposal[observation.target_ped_mask] = scaled
        return proposal


def clamp_magnitude(residual: np.ndarray, max_magnitude: float) -> np.ndarray:
    """Scale each row so its Euclidean norm does not exceed ``max_magnitude``.

    The direction is preserved; rows already within the bound are unchanged. Fails
    closed on non-finite input.

    Returns
    -------
    np.ndarray
        Row-wise magnitude-clamped residual with the same shape as ``residual``.
    """
    array = _validate_finite_array(residual, "residual")
    _require_finite(max_magnitude, "max_magnitude", strict_positive=True)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("residual must have shape (N, 2)")
    norms = np.linalg.norm(array, axis=1)
    factors = np.ones_like(norms, dtype=float)
    np.divide(max_magnitude, norms, out=factors, where=norms > max_magnitude)
    np.minimum(factors, 1.0, out=factors)
    return array * factors[:, None]


def rate_limit_jerk(
    proposed: np.ndarray, previous: np.ndarray, dt_s: float, max_jerk_mps3: float
) -> np.ndarray:
    """Move ``previous`` toward ``proposed`` by at most ``max_jerk_mps3 * dt_s`` per row.

    This bounds the time-derivative of the residual acceleration (jerk). The per-row
    step is capped in magnitude; direction follows the difference vector. Fails
    closed on non-finite input.

    Returns
    -------
    np.ndarray
        Jerk-rate-limited residual with the same shape as ``previous``.
    """
    proposed_array = _validate_finite_array(proposed, "proposed")
    previous_array = _validate_finite_array(previous, "previous")
    _require_finite(dt_s, "dt_s", strict_positive=True)
    _require_finite(max_jerk_mps3, "max_jerk_mps3", strict_positive=True)
    if proposed_array.shape != previous_array.shape:
        raise ValueError("proposed and previous must have the same shape")
    delta = proposed_array - previous_array
    max_step = float(max_jerk_mps3) * float(dt_s)
    return previous_array + clamp_magnitude(delta, max_step)


def bound_speed(
    residual: np.ndarray,
    velocities: np.ndarray,
    max_speeds: np.ndarray,
    dt_s: float,
    max_speed_delta_mps: float,
) -> np.ndarray:
    """Clip the residual so the resulting speed stays within bounds.

    Two speed constraints are combined into a single per-pedestrian cap:

    - The resulting speed ``|v + residual * dt_s|`` may not exceed ``max_speeds``.
    - The residual may not raise the speed by more than ``max_speed_delta_mps`` in one
      step, i.e. ``|v + residual * dt_s| <= |v| + max_speed_delta_mps``.

    Rows whose resulting speed already satisfies the tighter of the two caps are
    left unchanged; over-speed rows are scaled toward zero (direction preserved)
    until the resulting speed equals the cap. Fails closed on non-finite input.

    Returns
    -------
    np.ndarray
        Speed-bounded residual with the same shape as ``residual``.
    """
    residual_array = _validate_finite_array(residual, "residual")
    velocities_array = _validate_finite_array(velocities, "velocities")
    max_speeds_array = _validate_finite_array(max_speeds, "max_speeds")
    _require_finite(dt_s, "dt_s", strict_positive=True)
    _require_finite(max_speed_delta_mps, "max_speed_delta_mps", strict_positive=True)
    if residual_array.shape != velocities_array.shape:
        raise ValueError("residual and velocities must have the same shape")
    if max_speeds_array.shape != (residual_array.shape[0],):
        raise ValueError("max_speeds must have shape (N,) matching residual rows")
    if np.any(max_speeds_array < 0):
        raise ValueError("max_speeds must be >= 0")

    current_speeds = np.linalg.norm(velocities_array, axis=1)
    speed_caps = np.minimum(max_speeds_array, current_speeds + float(max_speed_delta_mps))
    delta_velocity = residual_array * float(dt_s)
    resulting_velocity = velocities_array + delta_velocity
    resulting_speed = np.linalg.norm(resulting_velocity, axis=1)
    over = resulting_speed > speed_caps + EPSILON
    if not np.any(over):
        return residual_array.copy()

    v = velocities_array[over]
    dv = delta_velocity[over]
    cap = speed_caps[over]
    # Solve |v + s * dv|^2 = cap^2 for the smaller root s in [0, 1].
    a_coeff = np.sum(dv * dv, axis=1)
    b_coeff = 2.0 * np.sum(v * dv, axis=1)
    c_coeff = np.sum(v * v, axis=1) - cap * cap
    safe_a = np.where(a_coeff > EPSILON, a_coeff, 1.0)
    discriminant = np.maximum(b_coeff * b_coeff - 4.0 * safe_a * c_coeff, 0.0)
    root = (-b_coeff - np.sqrt(discriminant)) / (2.0 * safe_a)
    scale_factor = np.clip(root, 0.0, 1.0)
    scaled = residual_array.copy()
    scaled[over] = residual_array[over] * scale_factor[:, None]
    return scaled


def bound_heading_change(
    residual: np.ndarray,
    velocities: np.ndarray,
    per_step_allowance_rad: float,
) -> np.ndarray:
    """Limit how far the residual may rotate the velocity direction in one step.

    The velocity change induced by the residual is decomposed into a perpendicular
    (turning) and parallel component. The perpendicular component is capped so the
    resulting angular change does not exceed ``per_step_allowance_rad``. Rows whose
    velocity is below :data:`EPSILON` are left unchanged (no defined heading). Fails
    closed on non-finite input.

    Returns
    -------
    np.ndarray
        Heading-change-bounded residual with the same shape as ``residual``.
    """
    residual_array = _validate_finite_array(residual, "residual")
    velocities_array = _validate_finite_array(velocities, "velocities")
    _require_finite(per_step_allowance_rad, "per_step_allowance_rad", strict_positive=True)
    if residual_array.shape != velocities_array.shape:
        raise ValueError("residual and velocities must have the same shape")

    speeds = np.linalg.norm(velocities_array, axis=1)
    moving = speeds > EPSILON
    result = residual_array.copy()
    if not np.any(moving):
        return result
    v = velocities_array[moving]
    r = residual_array[moving]
    speed = speeds[moving]
    velocity_dir = v / speed[:, None]
    parallel = np.sum(r * velocity_dir, axis=1)[:, None] * velocity_dir
    perpendicular = r - parallel
    # Angular change ~ |perpendicular| / speed for small angles; cap it.
    perpendicular_norm = np.linalg.norm(perpendicular, axis=1)
    max_perp = speed * float(per_step_allowance_rad)
    scale = np.ones_like(perpendicular_norm)
    np.divide(max_perp, perpendicular_norm, out=scale, where=perpendicular_norm > max_perp)
    np.minimum(scale, 1.0, out=scale)
    capped_perpendicular = perpendicular * scale[:, None]
    result[moving] = parallel + capped_perpendicular
    return result


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    """Return the minimum Euclidean distance from ``point`` to a polyline."""
    if polyline.shape[0] < 2:
        # A single-waypoint reference treats that point as the anchor.
        if polyline.shape[0] == 1:
            return float(np.linalg.norm(point - polyline[0]))
        return float("inf")
    starts = polyline[:-1]
    ends = polyline[1:]
    segment = ends - starts
    segment_len_sq = np.sum(segment * segment, axis=1)
    # Avoid divide-by-zero for degenerate zero-length segments.
    safe_len_sq = np.where(segment_len_sq > EPSILON, segment_len_sq, 1.0)
    to_point = point - starts
    t = np.sum(to_point * segment, axis=1) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)
    projection = starts + t[:, None] * segment
    distances = np.linalg.norm(point - projection, axis=1)
    return float(np.min(distances))


def bound_route_deviation(
    residual: np.ndarray,
    positions: np.ndarray,
    dt_s: float,
    route_polylines: list[np.ndarray] | None,
    target_indices: np.ndarray,
    max_route_deviation_m: float,
) -> np.ndarray:
    """Scale the residual so the would-be position stays within the route corridor.

    For each targeted pedestrian with an assigned reference polyline, the residual
    displacement ``residual * dt_s^2`` is scaled back if it would move the pedestrian
    beyond ``max_route_deviation_m`` of its polyline. Pedestrians without an assigned
    polyline, or when ``route_polylines`` is ``None``, are unaffected.

    Returns
    -------
    np.ndarray
        Route-deviation-bounded residual with the same shape as ``residual``.

    Fails closed on non-finite input.
    """
    residual_array = _validate_finite_array(residual, "residual")
    positions_array = _validate_finite_array(positions, "positions")
    _require_finite(dt_s, "dt_s", strict_positive=True)
    _require_finite(max_route_deviation_m, "max_route_deviation_m", strict_positive=True)
    if residual_array.shape != positions_array.shape:
        raise ValueError("residual and positions must have the same shape")
    if route_polylines is None or target_indices.size == 0:
        return residual_array.copy()
    result = residual_array.copy()
    dt_sq = float(dt_s) * float(dt_s)
    for local_slot, ped_idx in enumerate(target_indices):
        if local_slot >= len(route_polylines) or ped_idx < 0 or ped_idx >= positions_array.shape[0]:
            continue
        polyline = route_polylines[local_slot]
        polyline = _validate_finite_array(polyline, "route_polylines entry")
        if polyline.ndim != 2 or polyline.shape[1] != 2:
            raise ValueError("each route polyline must have shape (K, 2)")
        current_distance = _distance_to_polyline(positions_array[ped_idx], polyline)
        displacement = residual_array[ped_idx] * dt_sq
        candidate_position = positions_array[ped_idx] + displacement
        candidate_distance = _distance_to_polyline(candidate_position, polyline)
        if candidate_distance <= max_route_deviation_m:
            continue
        if current_distance >= max_route_deviation_m:
            # Already at or beyond the corridor: zero the outward residual.
            result[ped_idx] = 0.0
            continue
        # Binary-search-free linear scale: move along displacement until on the boundary.
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            test_position = positions_array[ped_idx] + mid * displacement
            test_distance = _distance_to_polyline(test_position, polyline)
            if test_distance > max_route_deviation_m:
                hi = mid
            else:
                lo = mid
        result[ped_idx] = residual_array[ped_idx] * lo
    return result


def _project_point_against_segment(
    point: np.ndarray, segment: np.ndarray, radius: float
) -> np.ndarray | None:
    """Return a corrected point pushed at least ``radius`` out of ``segment``.

    Returns ``None`` when the point is already at least ``radius`` away. The push-out
    direction is the shortest vector from the segment to the point.
    """
    start = segment[0]
    end = segment[1]
    seg = end - start
    seg_len_sq = float(np.sum(seg * seg))
    if seg_len_sq <= EPSILON:
        delta = point - start
        dist = float(np.linalg.norm(delta))
        if dist >= radius or dist <= EPSILON:
            return None
        return start + delta / dist * radius
    t = float(np.dot(point - start, seg)) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest = start + t * seg
    delta = point - closest
    dist = float(np.linalg.norm(delta))
    if dist >= radius or dist <= EPSILON:
        return None
    return closest + delta / dist * radius


def _normalize_obstacle_segments(
    obstacle_segments: np.ndarray | list[Line2D] | None,
) -> np.ndarray | None:
    """Return obstacle segments as an ``(S, 2, 2)`` float array, or ``None``.

    Accepts the ``(S, 4)`` flat layout, the ``(S, 2, 2)`` stacked layout, or an
    empty/``None`` input. Fails closed on non-finite or malformed segments.
    """
    if obstacle_segments is None:
        return None
    if len(obstacle_segments) == 0:
        return None
    segments = np.asarray(obstacle_segments, dtype=float)
    if segments.ndim == 3 and segments.shape[2] == 2:
        stacked = segments
    elif segments.ndim == 2 and segments.shape[1] == 4:
        stacked = segments.reshape(segments.shape[0], 2, 2)
    else:
        raise ValueError("obstacle_segments must have shape (S, 4) or (S, 2, 2) when provided")
    if not np.all(np.isfinite(stacked)):
        raise ValueError("obstacle_segments must be finite")
    return stacked


def _push_out_of_obstacles(
    positions: np.ndarray,
    candidate_positions: np.ndarray,
    displacement: np.ndarray,
    stacked_segments: np.ndarray | None,
    effective_radius: float,
) -> np.ndarray:
    """Return displacement updated so candidate positions clear every obstacle.

    For each pedestrian whose candidate position lies within ``effective_radius`` of
    a segment, the displacement is replaced by the vector to the nearest pushed-out
    point. Positions already clear of all obstacles keep their original displacement.
    """
    if stacked_segments is None:
        return displacement
    corrected = displacement.copy()
    for i in range(candidate_positions.shape[0]):
        best_position = None
        best_norm = float("inf")
        for segment in stacked_segments:
            pushed = _project_point_against_segment(
                candidate_positions[i], segment, effective_radius
            )
            if pushed is None:
                continue
            move_norm = float(np.linalg.norm(pushed - candidate_positions[i]))
            if move_norm < best_norm:
                best_norm = move_norm
                best_position = pushed
        if best_position is not None:
            corrected[i] = best_position - positions[i]
    return corrected


def _validate_bounds(
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
) -> tuple[float, float, float, float] | None:
    """Return finite ``(min_x, max_x, min_y, max_y)`` bounds, or ``None``."""
    if bounds is None:
        return None
    (min_x, max_x), (min_y, max_y) = bounds
    for value in (min_x, max_x, min_y, max_y):
        if not isfinite(value):
            raise ValueError("bounds entries must be finite")
    if max_x < min_x or max_y < min_y:
        raise ValueError("bounds must satisfy max >= min on each axis")
    return float(min_x), float(max_x), float(min_y), float(max_y)


def project_residual_displacement_walkable(
    positions: np.ndarray,
    residual_displacement: np.ndarray,
    obstacle_segments: np.ndarray | list[Line2D] | None,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    radius: float,
    margin_m: float,
) -> np.ndarray:
    """Return a corrected residual displacement that keeps positions walkable.

    Two projections are applied:

    - Obstacle push-out: if ``position + displacement`` lies within ``radius +
      margin_m`` of any obstacle segment, the displacement is replaced by the vector
      from the current position to the pushed-out point.
    - Bounds clamp: the candidate position is clamped inside ``bounds`` (with the
      same effective radius).

    Returns
    -------
    np.ndarray
        Walkable-projected displacement with the same shape as
        ``residual_displacement``.

    Fails closed on non-finite input.
    """
    positions_array = _validate_finite_array(positions, "positions")
    displacement_array = _validate_finite_array(residual_displacement, "residual_displacement")
    _require_finite(radius, "radius")
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if displacement_array.shape != positions_array.shape:
        raise ValueError("residual_displacement and positions must have the same shape")

    effective_radius = max(float(radius) + max(float(margin_m), MIN_WALKABLE_MARGIN_M), EPSILON)
    stacked_segments = _normalize_obstacle_segments(obstacle_segments)
    validated_bounds = _validate_bounds(bounds)

    candidate_positions = positions_array + displacement_array
    corrected = _push_out_of_obstacles(
        positions_array, candidate_positions, displacement_array, stacked_segments, effective_radius
    )
    if validated_bounds is not None:
        min_x, max_x, min_y, max_y = validated_bounds
        candidate_positions = positions_array + corrected
        candidate_positions[:, 0] = np.clip(
            candidate_positions[:, 0], min_x + effective_radius, max_x - effective_radius
        )
        candidate_positions[:, 1] = np.clip(
            candidate_positions[:, 1], min_y + effective_radius, max_y - effective_radius
        )
        corrected = candidate_positions - positions_array
    return corrected


def _pairwise_separation_scale(
    relative_position: np.ndarray,
    relative_displacement: np.ndarray,
    min_separation_m: float,
) -> float:
    """Return the largest safe displacement prefix for one pedestrian pair."""
    quadratic_a = float(np.dot(relative_displacement, relative_displacement))
    quadratic_b = 2.0 * float(np.dot(relative_position, relative_displacement))
    quadratic_c = float(np.dot(relative_position, relative_position)) - min_separation_m**2

    if quadratic_c < -EPSILON or (quadratic_c <= EPSILON and quadratic_b < 0.0):
        return 0.0
    if quadratic_c <= EPSILON or quadratic_a <= EPSILON:
        return 1.0
    discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
    if discriminant <= 0.0:
        return 1.0
    first_root = (-quadratic_b - float(np.sqrt(discriminant))) / (2.0 * quadratic_a)
    return first_root if 0.0 < first_root < 1.0 else 1.0


def enforce_inter_agent_separation(
    residual_displacement: np.ndarray,
    positions: np.ndarray,
    target_mask: np.ndarray,
    min_separation_m: float,
) -> np.ndarray:
    """Scale targeted displacements so agents keep ``min_separation_m`` apart.

    A single shared scale is applied to every targeted displacement.  This makes
    the all-target case atomic: every candidate pair is evaluated from the same
    positions rather than from stale, row-by-row candidates.  The scale is the
    largest prefix of the proposed displacement that preserves every pairwise
    separation. Non-targeted rows are never modified.

    Returns
    -------
    np.ndarray
        Separation-preserving displacement with the same shape as the input.

    Fails closed on non-finite input.
    """
    displacement_array = _validate_finite_array(residual_displacement, "residual_displacement")
    positions_array = _validate_finite_array(positions, "positions")
    _require_finite(min_separation_m, "min_separation_m", strict_positive=True)
    if displacement_array.shape != positions_array.shape:
        raise ValueError("residual_displacement and positions must have the same shape")
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape != (positions_array.shape[0],):
        raise ValueError("target_mask must have shape (N,) matching positions")
    targeted_indices = np.flatnonzero(target_mask)
    if targeted_indices.size == 0:
        return displacement_array.copy()

    # A single global scale preserves all pairwise constraints simultaneously.
    # For each pair, squared separation along that scale is a quadratic.  Starting
    # from alpha=0 (the current state), its first positive root is the largest safe
    # prefix for that pair; taking the minimum root is safe for every pair.
    shared_scale = 1.0
    num_peds = positions_array.shape[0]
    for i in targeted_indices:
        for j in range(num_peds):
            if j == i or (target_mask[j] and j < i):
                continue
            pair_scale = _pairwise_separation_scale(
                positions_array[i] - positions_array[j],
                displacement_array[i] - displacement_array[j],
                min_separation_m,
            )
            shared_scale = min(shared_scale, pair_scale)
        if shared_scale == 0.0:
            break

    result = displacement_array.copy()
    result[targeted_indices] *= shared_scale
    return result


@dataclass
class BoundedResidualAdversary:
    """Stateful bounded residual-control adversary applied each physics step.

    The adversary holds a macro-action proposal between cadence boundaries and
    enforces every hard bound on the per-step applied residual. It is strictly
    additive: callers add :meth:`step_residual` output to the nominal pedestrian
    forces, so the Social Force base law is preserved.

    The controller is independent of simulator objects so it is unit-testable with
    plain NumPy arrays. Route polylines, obstacle segments, and map bounds are
    optional; when omitted the corresponding bound degrades to a no-op while the
    kinematic bounds (speed / acceleration / jerk / heading) and inter-agent
    separation are still enforced.

    Attributes
    ----------
    config:
        Validated :class:`ResidualAdversaryConfig` with all hard bounds.
    policy:
        :class:`ResidualAdversaryPolicy` proposing the unbounded residual.
    dt_s:
        Physics timestep in seconds.
    num_peds:
        Number of pedestrians the adversary was sized for.
    route_polylines:
        Optional per-target-slot reference route polylines (shape ``(K, 2)`` each).
    obstacle_segments:
        Optional ``(S, 4)`` or ``(S, 2, 2)`` obstacle segments for walkable projection.
    bounds:
        Optional ``((min_x, max_x), (min_y, max_y))`` map bounds for walkable clamping.
    ped_radius:
        Pedestrian radius used for the walkable-space projection clearance.
    """

    config: ResidualAdversaryConfig
    policy: ResidualAdversaryPolicy
    dt_s: float
    num_peds: int
    route_polylines: list[np.ndarray] | None = None
    obstacle_segments: np.ndarray | list[Line2D] | None = None
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    ped_radius: float = 0.4
    _last_residual: np.ndarray = field(init=False, repr=False)
    _held_proposal: np.ndarray = field(init=False, repr=False)
    _step_index: int = field(init=False, default=0, repr=False)
    _macro_action_index: int = field(init=False, default=0, repr=False)
    _macro_steps: int = field(init=False, default=1, repr=False)
    _target_mask: np.ndarray = field(init=False, repr=False)
    _target_indices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate sizing and initialize the held-residual and cadence state."""
        _require_finite(self.dt_s, "dt_s", strict_positive=True)
        if not isinstance(self.num_peds, int) or self.num_peds < 0:
            raise ValueError("num_peds must be a non-negative int")
        _require_finite(self.ped_radius, "ped_radius")
        if self.ped_radius < 0:
            raise ValueError("ped_radius must be >= 0")
        self._last_residual = np.zeros((self.num_peds, 2), dtype=float)
        self._held_proposal = np.zeros((self.num_peds, 2), dtype=float)
        self._target_mask = self.config.resolve_target_mask(self.num_peds)
        self._target_indices = np.flatnonzero(self._target_mask)
        self._macro_steps = max(1, round(self.config.macro_action_dt_s / self.dt_s))

    @property
    def macro_action_steps(self) -> int:
        """Number of physics steps between macro-action proposal refreshes."""
        return self._macro_steps

    @property
    def step_index(self) -> int:
        """Number of physics steps processed so far."""
        return self._step_index

    @property
    def macro_action_index(self) -> int:
        """Number of macro-action proposals requested so far."""
        return self._macro_action_index

    @property
    def last_residual(self) -> np.ndarray:
        """Return a copy of the residual applied on the most recent step."""
        return self._last_residual.copy()

    def _apply_non_jerk_bounds(
        self,
        residual: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        max_speeds: np.ndarray,
    ) -> np.ndarray:
        """Apply every hard bound other than the stateful jerk-rate limit.

        This helper is deliberately reusable after jerk limiting: geometry and
        kinematic projections may otherwise be invalidated by a later rate limit.

        Returns
        -------
        np.ndarray
            Residual satisfying acceleration, kinematic, route, walkable-space,
            and separation bounds.
        """
        bounded = clamp_magnitude(residual, self.config.max_residual_accel_mps2)
        bounded = bound_speed(
            bounded,
            velocities,
            max_speeds,
            self.dt_s,
            self.config.max_speed_delta_mps,
        )
        per_step_allowance = self.config.max_heading_change_per_macro_rad / self._macro_steps
        bounded = bound_heading_change(bounded, velocities, per_step_allowance)
        bounded = bound_route_deviation(
            bounded,
            positions,
            self.dt_s,
            self.route_polylines,
            self._target_indices,
            self.config.max_route_deviation_m,
        )
        residual_displacement = bounded * (self.dt_s * self.dt_s)
        residual_displacement = project_residual_displacement_walkable(
            positions,
            residual_displacement,
            self.obstacle_segments,
            self.bounds,
            self.ped_radius,
            self.config.obstacle_projection_margin_m,
        )
        bounded = residual_displacement / (self.dt_s * self.dt_s)
        bounded = clamp_magnitude(bounded, self.config.max_residual_accel_mps2)
        residual_displacement = bounded * (self.dt_s * self.dt_s)
        residual_displacement = enforce_inter_agent_separation(
            residual_displacement,
            positions,
            self._target_mask,
            self.config.min_separation_m,
        )
        bounded = residual_displacement / (self.dt_s * self.dt_s)
        return clamp_magnitude(
            bounded * self._target_mask[:, None], self.config.max_residual_accel_mps2
        )

    def step_residual(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        max_speeds: np.ndarray,
        robot_pose: RobotPose,
    ) -> np.ndarray:
        """Return the bounded ``(N, 2)`` residual acceleration to add to ped forces.

        On each macro-action boundary (``step_index % macro_steps == 0``) a fresh
        proposal is requested from the policy and held. The applied residual then
        passes through the full bound pipeline. Fails closed on non-finite input.
        """
        positions_array = _validate_finite_array(positions, "positions")
        velocities_array = _validate_finite_array(velocities, "velocities")
        max_speeds_array = _validate_finite_array(max_speeds, "max_speeds")
        if positions_array.shape != (self.num_peds, 2):
            raise ValueError(
                f"positions must have shape ({self.num_peds}, 2), got {positions_array.shape}"
            )
        if velocities_array.shape != positions_array.shape:
            raise ValueError("velocities must match positions shape")
        if max_speeds_array.shape != (self.num_peds,):
            raise ValueError(f"max_speeds must have shape ({self.num_peds},)")

        if self.num_peds == 0:
            self._step_index += 1
            self._last_residual = np.zeros((0, 2), dtype=float)
            return self._last_residual.copy()

        sim_time_s = self._step_index * self.dt_s
        if self._step_index % self._macro_steps == 0:
            observation = ResidualAdversaryObservation(
                positions=positions_array,
                velocities=velocities_array,
                max_speeds=max_speeds_array,
                target_ped_mask=self._target_mask,
                robot_pose=robot_pose,
                sim_time_s=sim_time_s,
                step_index=self._step_index,
                macro_action_index=self._macro_action_index,
            )
            proposal = _validate_finite_array(
                self.policy.propose_residual(observation), "policy.propose_residual"
            )
            if proposal.shape != (self.num_peds, 2):
                raise ValueError(
                    "policy.propose_residual must return shape "
                    f"({self.num_peds}, 2), got {proposal.shape}"
                )
            self._held_proposal = proposal
            self._macro_action_index += 1

        # Zero out non-targeted rows before bounding so only targets are perturbed.
        bounded = self._held_proposal * self._target_mask[:, None]

        # 1. Jerk rate-limit toward the held proposal.
        bounded = rate_limit_jerk(
            bounded, self._last_residual, self.dt_s, self.config.max_jerk_mps3
        )
        bounded = self._apply_non_jerk_bounds(
            bounded, positions_array, velocities_array, max_speeds_array
        )
        # Reapplying jerk can reintroduce a kinematic or geometry violation after
        # the state changes.  Project it once more, then accept the result only if
        # it still lies in the jerk-limited set.  When those sets do not intersect,
        # emit no perturbation instead of weakening a hard safety bound.
        jerk_limited = rate_limit_jerk(
            bounded, self._last_residual, self.dt_s, self.config.max_jerk_mps3
        )
        bounded = self._apply_non_jerk_bounds(
            jerk_limited, positions_array, velocities_array, max_speeds_array
        )
        if not np.allclose(bounded, jerk_limited, rtol=0.0, atol=EPSILON):
            bounded = np.zeros_like(bounded)
        self._last_residual = bounded
        self._step_index += 1
        return bounded.copy()

    def reset(self) -> None:
        """Clear held state so the adversary restarts from a fresh macro-action."""
        self._last_residual = np.zeros((self.num_peds, 2), dtype=float)
        self._held_proposal = np.zeros((self.num_peds, 2), dtype=float)
        self._step_index = 0
        self._macro_action_index = 0


def build_default_residual_adversary(
    config: ResidualAdversaryConfig,
    dt_s: float,
    num_peds: int,
    *,
    route_polylines: list[np.ndarray] | None = None,
    obstacle_segments: np.ndarray | list[Line2D] | None = None,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ped_radius: float = 0.4,
    policy: ResidualAdversaryPolicy | None = None,
) -> BoundedResidualAdversary | None:
    """Construct a :class:`BoundedResidualAdversary` or return ``None`` when inactive.

    When ``config.is_active`` is ``False`` this returns ``None`` so callers can short
    circuit without allocating state. The default policy is
    :class:`ScriptedPullResidualAdversaryPolicy`.

    Returns
    -------
    BoundedResidualAdversary | None
        A ready-to-step adversary, or ``None`` when the config is inactive.
    """
    normalized = _normalize_residual_adversary_config(config)
    if not normalized.is_active:
        return None
    selected_policy = (
        policy
        if policy is not None
        else ScriptedPullResidualAdversaryPolicy(
            max_pull_accel_mps2=normalized.max_residual_accel_mps2
        )
    )
    return BoundedResidualAdversary(
        config=normalized,
        policy=selected_policy,
        dt_s=dt_s,
        num_peds=num_peds,
        route_polylines=route_polylines,
        obstacle_segments=obstacle_segments,
        bounds=bounds,
        ped_radius=ped_radius,
    )


def _normalize_residual_adversary_config(
    value: ResidualAdversaryConfig | dict,
) -> ResidualAdversaryConfig:
    """Return a validated :class:`ResidualAdversaryConfig` from a config or dict."""
    if isinstance(value, ResidualAdversaryConfig):
        return value
    if isinstance(value, dict):
        return ResidualAdversaryConfig(**value)
    raise ValueError("residual_adversary must be a ResidualAdversaryConfig or dict")


# Backward-compatibility alias matching the existing adversarial-ped-force naming.
ResidualAdversarySettings = ResidualAdversaryConfig


def residual_displacement_from_accel(accel: np.ndarray, dt_s: float) -> np.ndarray:
    """Return the first-order displacement contribution of a residual acceleration.

    The residual acceleration adds ``accel * dt_s`` to the velocity, which over one
    step adds ``accel * dt_s^2`` to the position to first order. Exposed for tests
    and downstream consumers that reason in displacement space.
    """
    array = _validate_finite_array(accel, "accel")
    _require_finite(dt_s, "dt_s", strict_positive=True)
    return array * (float(dt_s) * float(dt_s))


__all__ = [
    "DEFAULT_MACRO_ACTION_DT_S",
    "EPSILON",
    "BoundedResidualAdversary",
    "ResidualAdversaryConfig",
    "ResidualAdversaryObservation",
    "ResidualAdversaryPolicy",
    "ResidualAdversarySettings",
    "ScriptedPullResidualAdversaryPolicy",
    "bound_heading_change",
    "bound_route_deviation",
    "bound_speed",
    "build_default_residual_adversary",
    "clamp_magnitude",
    "enforce_inter_agent_separation",
    "project_residual_displacement_walkable",
    "rate_limit_jerk",
    "residual_displacement_from_accel",
]
