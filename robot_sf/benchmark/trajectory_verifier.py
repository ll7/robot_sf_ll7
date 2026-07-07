"""Experimental trajectory verifier for AMMV planner outputs (issue #4757).

Opt-in, test-time / offline-analysis verifier for candidate or executed robot
trajectories. Evaluates deterministic safety/comfort predicates against a robot
trajectory and pedestrian state and returns ``accept``, ``warn``, or
``fallback_brake`` without changing default planner behavior.

Conservative by design: missing or stale inputs fail closed to ``warn``; only
hard safety violations (clearance / TTC / braking feasibility) escalate to
``fallback_brake``. TTC is never fabricated from missing velocities.

Scope boundary: experimental diagnostic only, not a formal safety case, not
conformalized, not learned. Wiring into a planner control loop, release gate, or
benchmark scoring is explicitly out of scope for the first slice and must remain
opt-in. The implementation is pure and side-effect free so it can be unit-tested
and called from offline analysis utilities (see :func:`verify_episode_trace_window`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

TRAJECTORY_VERIFIER_SCHEMA = "trajectory_verifier.v1"
TRAJECTORY_VERIFIER_CLAIM_BOUNDARY = (
    "experimental test-time verification prototype; not a formal safety case; "
    "not conformalized; not learned; missing data fails closed to warn; "
    "default planner behavior unchanged"
)

DECISION_ACCEPT = "accept"
DECISION_WARN = "warn"
DECISION_FALLBACK_BRAKE = "fallback_brake"
_DECISION_RANK = {DECISION_ACCEPT: 0, DECISION_WARN: 1, DECISION_FALLBACK_BRAKE: 2}

PRED_CLEARANCE_HARD = "min_clearance_hard"
PRED_CLEARANCE_WARN = "min_clearance_warn"
PRED_TTC_HARD = "ttc_hard"
PRED_TTC_WARN = "ttc_warn"
PRED_BRAKING_INFEASIBLE = "braking_infeasible"
PRED_STALE_OR_MISSING_STATE = "stale_or_missing_state"
PRED_RECOVERY_SMOOTHNESS = "recovery_smoothness"

_HEADING_OSCILLATION_MIN_SPEED_MPS = 0.05
_HEADING_OSCILLATION_ANGLE_RAD = float(np.pi / 6.0)


@dataclass(frozen=True, slots=True)
class TrajectoryVerifierConfig:
    """Thresholds for the experimental trajectory verifier.

    Attributes:
        min_clearance_m: Hard minimum footprint clearance (center distance minus
            robot and pedestrian radii). Below this fires ``fallback_brake``.
        warn_clearance_m: Soft clearance threshold. Below this (but above the hard
            minimum) fires ``warn``.
        min_ttc_s: Hard minimum time-to-collision. Below this fires ``fallback_brake``.
        warn_ttc_s: Soft TTC threshold. Below this (but above the hard minimum) fires
            ``warn``.
        max_brake_deceleration_mps2: Proxy maximum braking deceleration for the
            braking-feasibility predicate. Stopping distance is ``v^2 / (2 * a)``.
        max_heading_oscillation_count: Maximum allowed heading sign / large-angle
            changes along the planned path before firing ``warn``.
        stale_prediction_max_age_s: Maximum tolerable prediction age. Above this
            fires ``warn`` via the stale-prediction predicate.
    """

    min_clearance_m: float = 0.25
    warn_clearance_m: float = 0.5
    min_ttc_s: float = 1.0
    warn_ttc_s: float = 1.5
    max_brake_deceleration_mps2: float = 2.5
    max_heading_oscillation_count: int = 3
    stale_prediction_max_age_s: float = 0.5

    def __post_init__(self) -> None:
        """Validate thresholds so the verifier cannot be silently misconfigured."""
        if self.min_clearance_m < 0.0:
            raise ValueError("TrajectoryVerifierConfig.min_clearance_m must be >= 0")
        if self.warn_clearance_m < self.min_clearance_m:
            raise ValueError("TrajectoryVerifierConfig.warn_clearance_m must be >= min_clearance_m")
        if self.min_ttc_s <= 0.0:
            raise ValueError("TrajectoryVerifierConfig.min_ttc_s must be > 0")
        if self.warn_ttc_s < self.min_ttc_s:
            raise ValueError("TrajectoryVerifierConfig.warn_ttc_s must be >= min_ttc_s")
        if self.max_brake_deceleration_mps2 <= 0.0:
            raise ValueError("TrajectoryVerifierConfig.max_brake_deceleration_mps2 must be > 0")
        if self.max_heading_oscillation_count < 0:
            raise ValueError("TrajectoryVerifierConfig.max_heading_oscillation_count must be >= 0")
        if self.stale_prediction_max_age_s <= 0.0:
            raise ValueError("TrajectoryVerifierConfig.stale_prediction_max_age_s must be > 0")


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """Outcome of :func:`verify_trajectory`.

    Attributes:
        decision: Aggregate decision, one of ``accept``, ``warn``, ``fallback_brake``.
        risk_score: Interpretable deterministic score in ``[0, 1]``. ``1.0`` when any
            hard predicate fired; otherwise the maximum soft-predicate contribution
            (each soft predicate contributes a value in ``[0, 0.5]``); ``0.0`` for a
            clean accept. See :func:`_aggregate_risk_score` for the decomposition.
        violated_predicates: Tuple of predicate identifiers that fired, in evaluation
            order. Empty for a clean accept.
        min_distance_m: Minimum center-to-center distance between the robot and any
            pedestrian over the trajectory, or ``None`` when there are no pedestrians.
        min_clearance_m: Minimum footprint clearance (``min_distance_m`` minus the sum
            of robot and pedestrian radii), or ``None`` when there are no pedestrians.
        min_ttc_s: Minimum time-to-collision proxy over the trajectory, or ``None`` when
            velocities are missing or no converging pair exists.
        braking_feasible: ``True`` when braking is feasible everywhere along the
            trajectory, ``False`` when the braking-infeasible predicate fired, ``None``
            when robot velocities are missing so braking feasibility is not evaluated.
        claim_boundary: Explicit experimental claim boundary string; always equal to
            :data:`TRAJECTORY_VERIFIER_CLAIM_BOUNDARY`.
    """

    decision: Literal["accept", "warn", "fallback_brake"]
    risk_score: float
    violated_predicates: tuple[str, ...]
    min_distance_m: float | None
    min_clearance_m: float | None
    min_ttc_s: float | None
    braking_feasible: bool | None
    claim_boundary: str = field(default=TRAJECTORY_VERIFIER_CLAIM_BOUNDARY)


def _require_finite(name: str, arr: np.ndarray) -> None:
    """Raise ``ValueError`` if ``arr`` holds any non-finite (NaN/inf) value.

    Non-finite inputs would silently defeat every threshold comparison — e.g.
    ``nan < min_clearance`` evaluates to ``False`` — and could cause an unsafe
    trajectory to be accepted. Reject them at the input boundary (fail closed).
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")


def _as_float2d(name: str, array: np.ndarray | None) -> np.ndarray | None:
    """Validate a (T, 2) trajectory array or return None for missing input.

    Returns:
        The validated float array of shape ``(T, 2)``, or ``None`` if ``array`` is
        ``None``.

    Raises:
        ValueError: If the array is not shape ``(T, 2)`` or has zero timesteps.
    """
    if array is None:
        return None
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (T, 2); got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one timestep; got {arr.shape}")
    _require_finite(name, arr)
    return arr


def _as_ped_positions(name: str, array: np.ndarray, expected_t: int) -> np.ndarray:
    """Validate pedestrian positions and broadcast static (N, 2) to (T, N, 2).

    Returns:
        Pedestrian positions of shape ``(expected_t, N, 2)``.

    Raises:
        ValueError: If the shape is not ``(T, N, 2)`` or ``(N, 2)``, the time dim does
            not match ``expected_t``, or there are no pedestrians.
    """
    arr = np.asarray(array, dtype=float)
    _require_finite(name, arr)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"{name} with shape (N, 2) must have last dim 2; got {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one pedestrian; got {arr.shape}")
        return np.broadcast_to(arr[None, :, :], (expected_t, arr.shape[0], 2)).copy()
    if arr.ndim == 3:
        if arr.shape[2] != 2:
            raise ValueError(f"{name} with shape (T, N, 2) must have last dim 2; got {arr.shape}")
        if arr.shape[0] != expected_t:
            raise ValueError(
                f"{name} time dim {arr.shape[0]} does not match robot_positions time dim "
                f"{expected_t}"
            )
        if arr.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one pedestrian; got {arr.shape}")
        return arr
    raise ValueError(
        f"{name} must have shape (T, N, 2) or (N, 2); got {arr.shape} (ndim={arr.ndim})"
    )


def _as_ped_velocities(
    name: str, array: np.ndarray | None, target_shape: tuple[int, int, int]
) -> np.ndarray | None:
    """Validate pedestrian velocities against the broadcast pedestrian-position shape.

    Accepts ``(T, N, 2)`` (time-varying, must match ``target_shape`` exactly) or
    ``(N, 2)`` (static velocities broadcast across the robot time dim ``T``).

    Returns:
        The validated pedestrian velocities of shape ``target_shape``, or ``None`` if
        ``array`` is ``None``.

    Raises:
        ValueError: If the shape does not match ``target_shape`` and is not a valid
            static ``(N, 2)`` array.
    """
    if array is None:
        return None
    arr = np.asarray(array, dtype=float)
    _require_finite(name, arr)
    if arr.ndim == 2:
        if arr.shape != (target_shape[1], 2):
            raise ValueError(
                f"{name} with shape (N, 2) must be {target_shape[1], 2}; got {arr.shape}"
            )
        return np.broadcast_to(arr[None, :, :], target_shape).copy()
    if arr.shape != target_shape:
        raise ValueError(
            f"{name} must have shape {target_shape} or (N, 2) matching pedestrian_positions; "
            f"got {arr.shape}"
        )
    return arr


def _pairwise_distances(robot_positions: np.ndarray, ped_positions: np.ndarray) -> np.ndarray:
    """Return per-timestep per-pedestrian center distances, shape (T, N)."""
    diff = ped_positions - robot_positions[:, None, :]
    return np.linalg.norm(diff, axis=-1)


def _ttc_proxy(
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
    sum_radii: float,
) -> np.ndarray:
    """Return per-timestep per-pedestrian TTC proxy in seconds, shape (T, N).

    Uses a simple constant-velocity closure model:
    ``ttc = -dot(d, v_rel) / |v_rel|^2`` when the pair is converging (``dot(d, v_rel) < 0``)
    and the relative speed is non-negligible. Already-touching pairs (center distance
    below ``sum_radii``) yield ``ttc = 0``. Separating or parallel pairs yield ``inf``.
    """
    diff = ped_positions - robot_positions[:, None, :]
    rel_vel = ped_velocities - robot_velocities[:, None, :]
    rel_speed_sq = np.sum(rel_vel * rel_vel, axis=-1)
    closing = np.sum(diff * rel_vel, axis=-1)
    dist = np.linalg.norm(diff, axis=-1)
    ttc = np.full_like(dist, np.inf)
    converging = (closing < 0.0) & (rel_speed_sq > 1.0e-12)
    ttc = np.where(converging, -closing / np.where(converging, rel_speed_sq, 1.0), ttc)
    ttc = np.where(dist <= sum_radii, 0.0, ttc)
    return ttc


def _braking_feasible(
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray | None,
    sum_radii: float,
    max_decel: float,
) -> bool:
    """Return True iff the robot can stop before colliding with any pedestrian ahead.

    For each timestep with non-negligible robot speed, compute the stopping distance
    ``d_stop = v^2 / (2 * a)`` and the longitudinal clearance to each pedestrian
    projected onto the robot heading. Because both bodies have physical size, a
    collision occurs when the center-to-center along-heading distance reaches
    ``sum_radii``; the available braking distance is therefore ``along - sum_radii``.
    If a pedestrian is ahead of the robot within the lateral footprint envelope
    (``|perpendicular offset| <= sum_radii``) and that available distance is less than
    ``d_stop``, braking is infeasible.
    """
    robot_speed = np.linalg.norm(robot_velocities, axis=-1)
    diff = ped_positions - robot_positions[:, None, :]
    speed_safe = np.where(robot_speed > 1.0e-9, robot_speed, 1.0)
    heading = robot_velocities / speed_safe[:, None]
    along = np.sum(diff * heading[:, None, :], axis=-1)
    lateral = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1) - along * along, 0.0))
    d_stop = (robot_speed**2) / (2.0 * max_decel)
    ahead = along > 0.0
    moving = robot_speed > 1.0e-9
    in_envelope = lateral <= sum_radii
    # Subtract the summed radii: the robot must stop before the footprints touch,
    # not before the centers coincide. Ignoring sum_radii would call braking
    # feasible even when the robot halts inside the pedestrian (fail-open).
    available = along - sum_radii
    cannot_stop = available < d_stop[:, None]
    infeasible = moving[:, None] & ahead & in_envelope & cannot_stop
    return not bool(np.any(infeasible))


def _count_heading_oscillations(
    robot_velocities: np.ndarray,
    min_speed_mps: float = _HEADING_OSCILLATION_MIN_SPEED_MPS,
    angle_rad: float = _HEADING_OSCILLATION_ANGLE_RAD,
) -> int:
    """Count large heading changes between consecutive moving timesteps.

    A heading change counts when both timesteps have speed above ``min_speed_mps`` and
    the absolute angular difference exceeds ``angle_rad``. This is a deterministic
    oscillation proxy; it does not infer intent.

    Returns:
        The number of large heading changes between consecutive moving timesteps.
    """
    speed = np.linalg.norm(robot_velocities, axis=-1)
    moving = speed > min_speed_mps
    if int(np.sum(moving)) < 2:
        return 0
    # Compare consecutive *moving* timesteps. Filtering to the moving subsequence
    # first (rather than masking adjacent-pair differences) means a heading change
    # spanning a stopped/slow timestep is still counted, matching the docstring.
    moving_velocities = robot_velocities[moving]
    angles = np.arctan2(moving_velocities[:, 1], moving_velocities[:, 0])
    d_angle = np.abs(np.diff(angles))
    d_angle = np.minimum(d_angle, 2.0 * np.pi - d_angle)
    return int(np.sum(d_angle > angle_rad))


def _aggregate_risk_score(
    hard_fired: bool,
    soft_risks: Mapping[str, float],
) -> float:
    """Combine predicate-level risks into one interpretable score in ``[0, 1]``.

    Decomposition (deterministic, not learned):

    - Any hard predicate firing => ``1.0`` (fallback_brake territory).
    - Otherwise the score is the maximum soft-predicate contribution, each in
      ``[0, 0.5]``:
        * clearance soft risk: linear ramp from 0 at ``warn_clearance`` to 0.5 at the
          hard ``min_clearance`` threshold.
        * ttc soft risk: linear ramp from 0 at ``warn_ttc`` to 0.5 at the hard
          ``min_ttc`` threshold.
        * stale / missing-state risk: fixed 0.3 (input-state uncertainty, not a
          measured motion violation).
        * recovery-smoothness / oscillation risk: ``min(count / (2 * max_count), 0.5)``.
    - Clean accept => ``0.0``.

    Returns:
        The aggregated risk score in ``[0, 1]``.
    """
    if hard_fired:
        return 1.0
    if not soft_risks:
        return 0.0
    return max(soft_risks.values())


def _ramp(value: float, warn: float, hard: float, at_warn: float, at_hard: float) -> float:
    """Linear ramp of ``value`` between ``warn`` and ``hard`` thresholds.

    Returns ``at_warn`` when ``value >= warn`` and ``at_hard`` when ``value <= hard``;
    interpolated linearly in between. ``hard`` must be the more-severe threshold
    (smaller for clearance/ttc).

    Returns:
        Interpolated value between ``at_warn`` and ``at_hard``.
    """
    if warn == hard:
        return at_hard if value <= hard else at_warn
    span = warn - hard
    t = (warn - value) / span
    t = max(0.0, min(1.0, t))
    return at_warn + t * (at_hard - at_warn)


# ---------------------------------------------------------------------------
# Predicate evaluation helpers — each returns (violated_list, hard_fired, soft_risks)
# ---------------------------------------------------------------------------


def _eval_clearance(
    min_clearance_m: float,
    cfg: TrajectoryVerifierConfig,
) -> tuple[list[str], bool, dict[str, float]]:
    """Evaluate minimum footprint clearance predicate.

    Returns:
        Tuple of (violated_predicates, hard_fired, soft_risks).
    """
    violated: list[str] = []
    hard_fired = False
    soft_risks: dict[str, float] = {}
    if min_clearance_m < cfg.min_clearance_m:
        violated.append(PRED_CLEARANCE_HARD)
        hard_fired = True
    elif min_clearance_m < cfg.warn_clearance_m:
        violated.append(PRED_CLEARANCE_WARN)
        soft_risks[PRED_CLEARANCE_WARN] = _ramp(
            min_clearance_m, cfg.warn_clearance_m, cfg.min_clearance_m, 0.0, 0.5
        )
    return violated, hard_fired, soft_risks


def _eval_ttc_and_braking(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    sum_radii: float,
    cfg: TrajectoryVerifierConfig,
) -> tuple[list[str], bool, dict[str, float], float | None, bool]:
    """Evaluate TTC and braking-feasibility predicates (requires velocity inputs).

    Returns:
        Tuple of (violated_predicates, hard_fired, soft_risks, min_ttc_s, braking_feasible).
    """
    violated: list[str] = []
    hard_fired = False
    soft_risks: dict[str, float] = {}

    ttc = _ttc_proxy(robot_pos, robot_vel, ped_pos, ped_vel, sum_radii)
    finite = ttc[np.isfinite(ttc)]
    min_ttc_s: float | None = float(np.min(finite)) if finite.size else None
    if min_ttc_s is not None:
        if min_ttc_s < cfg.min_ttc_s:
            violated.append(PRED_TTC_HARD)
            hard_fired = True
        elif min_ttc_s < cfg.warn_ttc_s:
            violated.append(PRED_TTC_WARN)
            soft_risks[PRED_TTC_WARN] = _ramp(min_ttc_s, cfg.warn_ttc_s, cfg.min_ttc_s, 0.0, 0.5)

    braking_ok = _braking_feasible(
        robot_pos, robot_vel, ped_pos, ped_vel, sum_radii, cfg.max_brake_deceleration_mps2
    )
    if not braking_ok:
        violated.append(PRED_BRAKING_INFEASIBLE)
        hard_fired = True

    return violated, hard_fired, soft_risks, min_ttc_s, braking_ok


def _eval_stale_prediction(
    missing_state: bool,
    prediction_age_s: float | None,
    cfg: TrajectoryVerifierConfig,
) -> tuple[list[str], dict[str, float]]:
    """Evaluate stale / missing-state predicate.

    Returns:
        Tuple of (violated_predicates, soft_risks).
    """
    violated: list[str] = []
    soft_risks: dict[str, float] = {}
    if missing_state:
        violated.append(PRED_STALE_OR_MISSING_STATE)
        soft_risks[PRED_STALE_OR_MISSING_STATE] = 0.3
    if prediction_age_s is not None and prediction_age_s > cfg.stale_prediction_max_age_s:
        if PRED_STALE_OR_MISSING_STATE not in violated:
            violated.append(PRED_STALE_OR_MISSING_STATE)
        soft_risks[PRED_STALE_OR_MISSING_STATE] = max(
            soft_risks.get(PRED_STALE_OR_MISSING_STATE, 0.0), 0.3
        )
    return violated, soft_risks


def _eval_oscillation(
    robot_vel: np.ndarray | None,
    cfg: TrajectoryVerifierConfig,
) -> tuple[list[str], dict[str, float]]:
    """Evaluate recovery-smoothness / oscillation-proxy predicate.

    Returns:
        Tuple of (violated_predicates, soft_risks).
    """
    violated: list[str] = []
    soft_risks: dict[str, float] = {}
    if robot_vel is None:
        return violated, soft_risks
    osc_count = _count_heading_oscillations(robot_vel)
    if osc_count > cfg.max_heading_oscillation_count:
        violated.append(PRED_RECOVERY_SMOOTHNESS)
        soft_risks[PRED_RECOVERY_SMOOTHNESS] = min(
            osc_count / (2.0 * max(cfg.max_heading_oscillation_count, 1)), 0.5
        )
    return violated, soft_risks


def _validate_inputs(
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray | None,
    pedestrian_positions: np.ndarray,
    pedestrian_velocities: np.ndarray | None,
    dt_s: float,
    robot_radius_m: float,
    pedestrian_radius_m: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Validate and coerce all input arrays, returning coerced forms.

    Returns:
        Tuple of (robot_pos, robot_vel, ped_pos, ped_vel) after shape validation.

    Raises:
        ValueError: If any argument is invalid.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be > 0; got {dt_s}")
    if robot_radius_m < 0.0:
        raise ValueError(f"robot_radius_m must be >= 0; got {robot_radius_m}")
    if pedestrian_radius_m < 0.0:
        raise ValueError(f"pedestrian_radius_m must be >= 0; got {pedestrian_radius_m}")

    robot_pos = _as_float2d("robot_positions", robot_positions)
    if robot_pos is None:
        raise ValueError("robot_positions is required and must have shape (T, 2)")
    robot_vel = _as_float2d("robot_velocities", robot_velocities)
    if robot_vel is not None and robot_vel.shape != robot_pos.shape:
        raise ValueError(
            f"robot_velocities shape {robot_vel.shape} must match robot_positions "
            f"shape {robot_pos.shape}"
        )
    ped_pos = _as_ped_positions("pedestrian_positions", pedestrian_positions, robot_pos.shape[0])
    ped_vel = _as_ped_velocities("pedestrian_velocities", pedestrian_velocities, ped_pos.shape)
    return robot_pos, robot_vel, ped_pos, ped_vel


def verify_trajectory(
    *,
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray | None,
    pedestrian_positions: np.ndarray,
    pedestrian_velocities: np.ndarray | None,
    dt_s: float,
    robot_radius_m: float,
    pedestrian_radius_m: float,
    config: TrajectoryVerifierConfig | None = None,
    prediction_age_s: float | None = None,
) -> VerifierResult:
    """Verify a robot trajectory against deterministic safety/comfort predicates.

    Args:
        robot_positions: Robot positions over time, shape ``(T, 2)``.
        robot_velocities: Robot velocities over time, shape ``(T, 2)``, or ``None`` if
            missing (the verifier fails closed to warn for missing state).
        pedestrian_positions: Pedestrian positions, shape ``(T, N, 2)`` (time-varying)
            or ``(N, 2)`` (static; broadcast to the robot time dim).
        pedestrian_velocities: Pedestrian velocities, shape matching the broadcast
            pedestrian positions, or ``None`` if missing.
        dt_s: Timestep duration in seconds. Must be positive.
        robot_radius_m: Robot footprint radius in meters. Must be non-negative.
        pedestrian_radius_m: Pedestrian footprint radius in meters. Must be non-negative.
        config: Verifier thresholds. Defaults to :class:`TrajectoryVerifierConfig`.
        prediction_age_s: Age of the pedestrian prediction at the start of the window.
            If greater than ``config.stale_prediction_max_age_s`` the verifier fires the
            stale-prediction warning. ``None`` means no age signal is available, which
            is treated as not-stale (the missing-velocity predicate handles missing
            motion state separately).

    Returns:
        A :class:`VerifierResult` with the aggregate decision, risk score, violated
        predicates, and core metrics.

    Raises:
        ValueError: If any array shape is invalid, ``dt_s`` is non-positive, or radii
            are negative.
    """
    robot_pos, robot_vel, ped_pos, ped_vel = _validate_inputs(
        robot_positions,
        robot_velocities,
        pedestrian_positions,
        pedestrian_velocities,
        dt_s,
        robot_radius_m,
        pedestrian_radius_m,
    )
    cfg = config if config is not None else TrajectoryVerifierConfig()
    sum_radii = robot_radius_m + pedestrian_radius_m
    distances = _pairwise_distances(robot_pos, ped_pos)
    min_distance_m = float(np.min(distances))
    min_clearance_m = min_distance_m - sum_radii

    violated: list[str] = []
    hard_fired = False
    soft_risks: dict[str, float] = {}

    # Predicate 1: minimum footprint clearance.
    v, h, s = _eval_clearance(min_clearance_m, cfg)
    violated.extend(v)
    hard_fired = hard_fired or h
    soft_risks.update(s)

    # Predicates 2 & 3: TTC and braking feasibility (requires velocity).
    min_ttc_s: float | None = None
    braking_feasible: bool | None = None
    missing_state = robot_vel is None or ped_vel is None
    if robot_vel is not None and ped_vel is not None:
        v, h, s, min_ttc_s, bf = _eval_ttc_and_braking(
            robot_pos, robot_vel, ped_pos, ped_vel, sum_radii, cfg
        )
        braking_feasible = bf
        violated.extend(v)
        hard_fired = hard_fired or h
        soft_risks.update(s)

    # Predicate 4: stale / missing state.
    v, s = _eval_stale_prediction(missing_state, prediction_age_s, cfg)
    violated.extend(v)
    soft_risks.update(s)

    # Predicate 5: recovery smoothness / oscillation proxy.
    v, s = _eval_oscillation(robot_vel, cfg)
    violated.extend(v)
    soft_risks.update(s)

    if hard_fired:
        decision: Literal["accept", "warn", "fallback_brake"] = DECISION_FALLBACK_BRAKE
    elif violated:
        decision = DECISION_WARN
    else:
        decision = DECISION_ACCEPT

    risk_score = _aggregate_risk_score(hard_fired, soft_risks)

    return VerifierResult(
        decision=decision,
        risk_score=risk_score,
        violated_predicates=tuple(violated),
        min_distance_m=min_distance_m,
        min_clearance_m=min_clearance_m,
        min_ttc_s=min_ttc_s,
        braking_feasible=braking_feasible,
    )


def verify_episode_trace_window(
    trace: Mapping[str, Any],
    *,
    start: int = 0,
    end: int | None = None,
    config: TrajectoryVerifierConfig | None = None,
    robot_radius_m: float = 0.3,
    pedestrian_radius_m: float = 0.3,
    dt_s: float | None = None,
) -> VerifierResult:
    """Opt-in helper to verify a window of an episode trace mapping.

    This is a thin adapter intended for offline analysis and tests. It does **not**
    alter planner commands and is not wired into any runner by default. The trace must
    expose the keys ``robot_positions``, ``pedestrian_positions``, and ``dt_s`` (or
    ``dt``). Optional keys: ``robot_velocities``, ``pedestrian_velocities``,
    ``prediction_age_s``.

    Args:
        trace: Mapping with the keys described above. ``robot_positions`` must have
            shape ``(T, 2)``; ``pedestrian_positions`` may be ``(T, N, 2)`` or
            ``(N, 2)``.
        start: Inclusive start index of the window.
        end: Exclusive end index of the window; defaults to the trajectory end.
        config: Verifier thresholds. Defaults to :class:`TrajectoryVerifierConfig`.
        robot_radius_m: Robot footprint radius in meters.
        pedestrian_radius_m: Pedestrian footprint radius in meters.
        dt_s: Timestep duration override; if ``None`` the trace ``dt_s`` (or ``dt``)
            key is used.

    Returns:
        The :class:`VerifierResult` for the requested window.

    Raises:
        ValueError: If required keys are missing, shapes are invalid, or the window is
            empty.
    """
    for key in ("robot_positions", "pedestrian_positions"):
        if key not in trace:
            raise ValueError(f"verify_episode_trace_window requires trace key {key!r}")
    resolved_dt = dt_s if dt_s is not None else trace.get("dt_s", trace.get("dt"))
    if resolved_dt is None:
        raise ValueError("verify_episode_trace_window requires dt_s (or trace key 'dt_s' / 'dt')")

    robot_positions = np.asarray(trace["robot_positions"], dtype=float)
    pedestrian_positions = np.asarray(trace["pedestrian_positions"], dtype=float)
    end_idx = end if end is not None else robot_positions.shape[0]
    if start < 0 or end_idx <= start:
        raise ValueError(
            f"trace window must satisfy 0 <= start < end; got start={start}, end={end_idx}"
        )
    robot_positions = robot_positions[start:end_idx]
    pedestrian_positions = (
        pedestrian_positions[start:end_idx]
        if pedestrian_positions.ndim == 3
        else pedestrian_positions
    )
    robot_velocities = trace.get("robot_velocities")
    if robot_velocities is not None:
        robot_velocities = np.asarray(robot_velocities, dtype=float)[start:end_idx]
    pedestrian_velocities = trace.get("pedestrian_velocities")
    if pedestrian_velocities is not None:
        pedestrian_velocities_arr = np.asarray(pedestrian_velocities, dtype=float)
        if pedestrian_velocities_arr.ndim == 3:
            pedestrian_velocities = pedestrian_velocities_arr[start:end_idx]
        else:
            pedestrian_velocities = pedestrian_velocities_arr
    prediction_age_s = trace.get("prediction_age_s")

    return verify_trajectory(
        robot_positions=robot_positions,
        robot_velocities=robot_velocities,
        pedestrian_positions=pedestrian_positions,
        pedestrian_velocities=pedestrian_velocities,
        dt_s=float(resolved_dt),
        robot_radius_m=robot_radius_m,
        pedestrian_radius_m=pedestrian_radius_m,
        config=config,
        prediction_age_s=(float(prediction_age_s) if prediction_age_s is not None else None),
    )


__all__ = [
    "DECISION_ACCEPT",
    "DECISION_FALLBACK_BRAKE",
    "DECISION_WARN",
    "PRED_BRAKING_INFEASIBLE",
    "PRED_CLEARANCE_HARD",
    "PRED_CLEARANCE_WARN",
    "PRED_RECOVERY_SMOOTHNESS",
    "PRED_STALE_OR_MISSING_STATE",
    "PRED_TTC_HARD",
    "PRED_TTC_WARN",
    "TRAJECTORY_VERIFIER_CLAIM_BOUNDARY",
    "TRAJECTORY_VERIFIER_SCHEMA",
    "TrajectoryVerifierConfig",
    "VerifierResult",
    "verify_episode_trace_window",
    "verify_trajectory",
]
