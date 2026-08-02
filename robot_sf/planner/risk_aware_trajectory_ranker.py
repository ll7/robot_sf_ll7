"""Experimental risk-aware trajectory ranking prototype (issue #6567).

Orchestrates the existing action-conditioned collision-risk estimator
(:mod:`robot_sf.research.collision_risk`) and the hard deterministic verifiers
(:mod:`robot_sf.benchmark.trajectory_verifier` and
:mod:`robot_sf.benchmark.actuator_feasibility`) into an opt-in, offline prototype
that generates a small set of deterministic motion primitives and ranks them by
decomposed score components.

The module deliberately does **not** reimplement any contact geometry, risk
math, or feasibility predicate. It reuses ``CandidateAction``,
``estimate_action_conditioned_risk``, and ``RiskEstimatorConfig`` verbatim and
treats the two deterministic verifiers as authoritative hard gates: a candidate
rejected by ``verify_trajectory`` (``fallback_brake``) or reported physically
infeasible by ``evaluate_actuator_feasibility`` is marked ineligible regardless
of its probabilistic collision risk.

.. admonition:: Claim boundary
   :class: note

   Smoke / diagnostic evidence only. The probabilistic component is an
   explicitly-declared constant-velocity model probability; it is **not** a
   calibrated real-world collision probability and calibration to the simulator
   distribution is successor work (see :data:`RANKER_CLAIM_BOUNDARY`). This
   prototype is experimental, disabled by default, and is not wired into
   ``map_runner`` or any planner control loop.

Peak-risk timing is emitted in the anchor-step plus window shape consumed by
:mod:`robot_sf.benchmark.critical_intervals` so the peak-risk timestep of a
ranked candidate can be inspected by the same critical-interval machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.benchmark.actuator_feasibility import (
    ActuatorLimitsConfig,
    evaluate_actuator_feasibility,
)
from robot_sf.benchmark.trajectory_verifier import (
    DECISION_FALLBACK_BRAKE,
    TrajectoryVerifierConfig,
    verify_trajectory,
)
from robot_sf.research.collision_risk import (
    ActionConditionedRiskEstimate,
    CandidateAction,
    RiskEstimatorConfig,
    estimate_action_conditioned_risk,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from robot_sf.nav.predictive_types import PedestrianState

RANKER_SCHEMA_VERSION = "risk_aware_trajectory_ranker.v1"

#: Explicit experimental / diagnostic claim boundary for this prototype.
RANKER_CLAIM_BOUNDARY = (
    "experimental risk-aware trajectory ranking prototype; smoke/diagnostic "
    "evidence only; not calibrated real-world collision probability; not a "
    "formal safety case; hard deterministic gates remain authoritative; default "
    "planner behavior unchanged; not wired into map_runner or any planner loop"
)

#: Finite clearance sentinel passed to the actuator-feasibility gate when no
#: pedestrian hazard is present (the estimator reports ``+inf`` clearance in that
#: case, which the actuator gate rejects as non-finite).
_NO_HAZARD_CLEARANCE_M = 1.0e3

#: Half-window (in horizon steps) around the peak-risk timestep, matching the
#: ``before_s`` / ``after_s`` anchor-window convention of ``critical_intervals``.
_DEFAULT_PEAK_WINDOW_HALF_STEPS = 3


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveGeneratorConfig:
    """Geometric configuration for the deterministic primitive generator.

    The generator is planner-agnostic: it builds smooth waypoint sequences from a
    start state toward a local goal, parameterized by a small set of terminal
    lateral offsets plus an optional low-displacement brake primitive. Horizon
    length and timestep are supplied explicitly so the generated ``CandidateAction``
    shapes match the action-conditioned risk estimator contract.

    Attributes:
        lateral_offsets_m: Terminal lateral offsets (metres) for the arc
            primitives. Each offset yields one distinct candidate; the sign
            selects left vs. right of the start-to-goal direction.
        cruise_speed_mps: Reference cruise speed (m/s) used to bound each
            primitive's sampled speed to what is reachable within the horizon.
        goal_reach_fraction: Fraction of the start-to-goal distance the primitives
            attempt to cover (clipped to the reachable arc length).
        include_brake_primitive: When True, append a low-displacement brake
            candidate distinct from the lateral arcs.
        brake_displacement_m: Forward displacement (m) of the brake primitive.
    """

    lateral_offsets_m: tuple[float, ...] = (-0.6, 0.0, 0.6)
    cruise_speed_mps: float = 1.0
    goal_reach_fraction: float = 1.0
    include_brake_primitive: bool = True
    brake_displacement_m: float = 0.2

    def __post_init__(self) -> None:
        """Validate generator parameters so malformed configs fail closed."""
        if not self.lateral_offsets_m:
            raise ValueError("PrimitiveGeneratorConfig.lateral_offsets_m must be non-empty")
        if any(not math.isfinite(value) for value in self.lateral_offsets_m):
            raise ValueError("PrimitiveGeneratorConfig.lateral_offsets_m must be finite")
        if not math.isfinite(self.cruise_speed_mps) or self.cruise_speed_mps <= 0.0:
            raise ValueError("PrimitiveGeneratorConfig.cruise_speed_mps must be finite and > 0")
        if not (0.0 < self.goal_reach_fraction <= 1.0):
            raise ValueError("PrimitiveGeneratorConfig.goal_reach_fraction must be in (0, 1]")
        if self.brake_displacement_m < 0.0 or not math.isfinite(self.brake_displacement_m):
            raise ValueError("PrimitiveGeneratorConfig.brake_displacement_m must be finite >= 0")


@dataclass(frozen=True)
class RankingWeights:
    """Non-negative weights for the (lower-is-better) composite ranking cost.

    The composite is used only to order eligible candidates; every component is
    reported separately on each :class:`CandidateRanking` and the composite is
    never the sole decision signal. ``w_time`` is kept for completeness even
    though travel time is constant across fixed-horizon candidates.

    Attributes:
        w_risk: Weight on the (uncalibrated) model collision probability.
        w_time: Weight on travel time in seconds.
        w_jerk: Weight on the integrated jerk comfort proxy.
        w_length: Weight on path length in metres.
        w_clearance: Weight on the deterministic clearance penalty.
        safe_clearance_m: Clearance (m) above which the clearance penalty is zero;
            the penalty ramps linearly to 1.0 at zero clearance and above 1.0 for
            footprint overlap.
    """

    w_risk: float = 1.0
    w_time: float = 0.0
    w_jerk: float = 0.1
    w_length: float = 0.05
    w_clearance: float = 1.0
    safe_clearance_m: float = 0.5

    def __post_init__(self) -> None:
        """Validate weights so the composite cannot be silently mis-scaled."""
        for name in ("w_risk", "w_time", "w_jerk", "w_length", "w_clearance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"RankingWeights.{name} must be finite and >= 0")
        if self.safe_clearance_m <= 0.0 or not math.isfinite(self.safe_clearance_m):
            raise ValueError("RankingWeights.safe_clearance_m must be finite and > 0")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreComponents:
    """Decomposed per-candidate score components (reported separately).

    Attributes:
        calibrated_collision_risk: Model joint contact probability reported by
            :func:`estimate_action_conditioned_risk`. Named for the issue's
            component list; see :attr:`calibration_applied`.
        travel_time_s: Candidate travel time in seconds (``horizon_steps * dt``).
        integrated_jerk: Integral of the jerk magnitude over time (m/s^2), a
            comfort proxy derived from the waypoint kinematics.
        path_length_m: Total arc length of the waypoint polyline in metres.
        clearance_penalty: Deterministic clearance penalty in ``[0, +inf)``; zero
            above :attr:`RankingWeights.safe_clearance_m`, ramping to 1.0 at zero
            clearance and above 1.0 for footprint overlap.
        min_clearance_m: Raw deterministic minimum footprint clearance (metres)
            from the risk estimate; ``+inf`` when no pedestrian hazard is present.
        calibration_applied: Always False in this prototype; the reported
            collision risk is a declared model probability, not calibrated to the
            simulator or real-world distribution.
    """

    calibrated_collision_risk: float
    travel_time_s: float
    integrated_jerk: float
    path_length_m: float
    clearance_penalty: float
    min_clearance_m: float
    calibration_applied: bool = False


@dataclass(frozen=True)
class PeakRiskTiming:
    """Peak-risk timestep in the anchor-step shape consumed by critical intervals.

    Mirrors the ``anchor_step`` / ``start_step`` / ``end_step`` shape used by
    :mod:`robot_sf.benchmark.critical_intervals` so a peak-risk timestep can be
    inspected as a critical-interval anchor. ``peak_step`` is the horizon step of
    maximum first-passage probability; ``peak_time_s`` is ``peak_step * dt``;
    ``window_start_step`` / ``window_end_step`` delimit a half-window around it.

    Attributes:
        peak_step: Horizon step index of maximum first-passage probability, or
            ``-1`` when the candidate has no contact mass within the horizon.
        peak_time_s: ``peak_step * dt`` in seconds (``0.0`` when ``peak_step < 0``).
        peak_actor_id: Identifier of the actor contributing the peak risk, or
            ``None`` when there are no actors or no peak.
        window_start_step: Inclusive start of the inspection window (clamped).
        window_end_step: Exclusive end of the inspection window (clamped).
        first_passage_distribution: Per-step first-passage probabilities consumed
            verbatim from the action-conditioned risk estimate.
    """

    peak_step: int
    peak_time_s: float
    peak_actor_id: int | None
    window_start_step: int
    window_end_step: int
    first_passage_distribution: tuple[float, ...]


@dataclass(frozen=True)
class HardGateResult:
    """Outcome of the two authoritative deterministic hard gates for one candidate.

    A candidate is eligible only when neither hard gate rejects it. A low
    probabilistic collision risk never overrides a failed hard gate.

    Attributes:
        eligible: True iff the verifier did not return ``fallback_brake`` and the
            actuator-feasibility report is physically feasible. This rejects both
            ``infeasible`` and ``geometry_only_clear``: the latter means the
            geometry is clear, but the maneuver still violates an actuator limit.
        verifier_decision: Decision returned by :func:`verify_trajectory`
            (``accept`` / ``warn`` / ``fallback_brake``), or ``"skipped_no_hazard"``
            when there were no pedestrians to verify against.
        actuator_verdict: Verdict returned by :func:`evaluate_actuator_feasibility`
            (``actuator_feasible`` / ``geometry_only_clear`` / ``infeasible``).
        violated_predicates: Predicate identifiers fired by the trajectory verifier.
        violated_limits: Actuator-limit identifiers fired by the feasibility gate.
        ineligibility_reason: Human-readable reason when the candidate is
            ineligible, else ``None``.
    """

    eligible: bool
    verifier_decision: str
    actuator_verdict: str
    violated_predicates: tuple[str, ...]
    violated_limits: tuple[str, ...]
    ineligibility_reason: str | None = None


@dataclass(frozen=True)
class CandidateProvenance:
    """Risk-estimator provenance carried through to the ranking record.

    Attributes:
        action_id: Stable identifier of the scored candidate action.
        estimator_id: Identifier of the reused risk estimator.
        forecast_model: Declared pedestrian forecast model identifier.
        geometry_version: Footprint / contact-geometry version identifier.
        config_hash: Stable hash of the estimator configuration.
        seed: Monte Carlo seed used by the estimator.
        risk_schema_version: Schema version of the reused risk estimate.
        ranker_schema_version: Always :data:`RANKER_SCHEMA_VERSION`.
        abstained: True when the estimator flagged the estimate as untrusted.
    """

    action_id: str
    estimator_id: str
    forecast_model: str
    geometry_version: str
    config_hash: str
    seed: int
    risk_schema_version: str
    ranker_schema_version: str = RANKER_SCHEMA_VERSION
    abstained: bool = False


@dataclass(frozen=True)
class CandidateRanking:
    """Full decomposed ranking record for one candidate action.

    Attributes:
        action_id: Stable identifier of the scored candidate action.
        rank: 1-based rank among eligible candidates (lower composite cost first);
            ``-1`` for ineligible candidates, which are listed after eligible ones.
        eligible: True when no hard gate rejected the candidate.
        composite_score: Weighted cost used only to order eligible candidates.
        components: Decomposed :class:`ScoreComponents`.
        peak_risk: :class:`PeakRiskTiming` for critical-interval inspection.
        hard_gate: :class:`HardGateResult` with the deterministic-gate outcomes.
        provenance: :class:`CandidateProvenance` linking back to the risk estimator.
        joint_contact_probability: Raw model joint contact probability (identical
            to ``components.calibrated_collision_risk`` in this prototype).
        estimate: Full reused :class:`ActionConditionedRiskEstimate` for traceability.
        claim_boundary: Always :data:`RANKER_CLAIM_BOUNDARY`.
    """

    action_id: str
    rank: int
    eligible: bool
    composite_score: float
    components: ScoreComponents
    peak_risk: PeakRiskTiming
    hard_gate: HardGateResult
    provenance: CandidateProvenance
    joint_contact_probability: float
    estimate: ActionConditionedRiskEstimate
    claim_boundary: str = RANKER_CLAIM_BOUNDARY


# ---------------------------------------------------------------------------
# Deterministic primitive generator
# ---------------------------------------------------------------------------


def _smoothstep(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the Hermite smoothstep ``3s^2 - 2s^3`` of ``values`` in ``[0, 1]``."""
    return values * values * (3.0 - 2.0 * values)


def _arc_primitive(
    action_id: str,
    start: NDArray[np.floating],
    unit: NDArray[np.floating],
    perp: NDArray[np.floating],
    *,
    progress_m: float,
    lateral_offset_m: float,
    horizon_steps: int,
) -> CandidateAction:
    """Build a smooth lateral-arc primitive waypoint sequence.

    The primitive advances monotonically along ``unit`` by ``progress_m`` while
    blending laterally along ``perp`` to ``lateral_offset_m`` via a smoothstep, so
    the start velocity is purely longitudinal and the path is dynamically smooth.

    Args:
        action_id: Stable identifier for the primitive.
        start: Start position ``(2,)``.
        unit: Unit direction along the start-to-goal vector ``(2,)``.
        perp: Unit perpendicular to ``unit`` ``(2,)``.
        progress_m: Forward arc progress in metres.
        lateral_offset_m: Terminal lateral offset in metres.
        horizon_steps: Number of horizon steps ``H`` (waypoints have ``H + 1`` rows).

    Returns:
        A finite :class:`CandidateAction` with shape ``(H + 1, 2)`` waypoints.
    """
    steps = np.arange(horizon_steps + 1, dtype=float)
    fraction = steps / horizon_steps
    blend = _smoothstep(fraction)
    longitudinal = (progress_m * fraction)[:, None] * unit[None, :]
    lateral = (lateral_offset_m * blend)[:, None] * perp[None, :]
    waypoints = start[None, :] + longitudinal + lateral
    return CandidateAction(action_id=action_id, waypoints=waypoints, representation="primitive")


def _limit_candidate_speed(
    candidate: CandidateAction, *, dt_s: float, cruise_speed_mps: float
) -> CandidateAction:
    """Scale a primitive's displacement when its sampled speed exceeds the limit.

    Lateral blending adds arc length beyond longitudinal progress. Scaling all
    waypoints around the first one preserves the primitive's shape while making
    its sampled waypoint speed feasible for the supplied horizon.

    Returns:
        The original candidate when it is within the cruise-speed limit, otherwise
        a speed-limited candidate with the same identifier and representation.
    """
    step_speeds = np.linalg.norm(np.diff(candidate.waypoints, axis=0), axis=1) / dt_s
    max_speed_mps = float(np.max(step_speeds))
    if max_speed_mps <= cruise_speed_mps:
        return candidate

    displacement_scale = cruise_speed_mps / max_speed_mps
    start = candidate.waypoints[0]
    return CandidateAction(
        action_id=candidate.action_id,
        waypoints=start + (candidate.waypoints - start) * displacement_scale,
        representation=candidate.representation,
    )


def generate_primitive_candidates(
    start_position: Sequence[float] | NDArray[np.floating],
    local_goal: Sequence[float] | NDArray[np.floating],
    *,
    horizon_steps: int,
    dt_s: float,
    config: PrimitiveGeneratorConfig | None = None,
) -> list[CandidateAction]:
    """Generate deterministic planner-agnostic motion primitives.

    Produces at least three finite :class:`CandidateAction` waypoint sequences of
    shape ``(H + 1, 2)`` from a start state and a local goal. Each primitive
    carries a stable ``action_id`` and all sampled states are finite.

    Args:
        start_position: Robot start position ``(2,)`` in metres.
        local_goal: Local goal position ``(2,)`` in metres.
        horizon_steps: Number of horizon steps ``H`` (must be positive).
        dt_s: Timestep in seconds (must be positive).
        config: Generator geometry; defaults to :class:`PrimitiveGeneratorConfig`.

    Returns:
        List of at least three finite candidate actions with stable action ids.

    Raises:
        ValueError: If ``horizon_steps`` or ``dt_s`` are invalid, or fewer than
            three primitives would be produced.
    """
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    cfg = config if config is not None else PrimitiveGeneratorConfig()

    start = np.asarray(start_position, dtype=float).reshape(2)
    goal = np.asarray(local_goal, dtype=float).reshape(2)
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(goal))):
        raise ValueError("start_position and local_goal must be finite")

    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance < 1.0e-9:
        unit = np.array([1.0, 0.0])
        distance = 0.0
    else:
        unit = delta / distance
    perp = np.array([-unit[1], unit[0]])

    reach_m = cfg.cruise_speed_mps * horizon_steps * dt_s
    progress_m = min(distance * cfg.goal_reach_fraction, reach_m)

    candidates: list[CandidateAction] = []
    for index, offset in enumerate(cfg.lateral_offsets_m):
        label = "straight" if offset == 0.0 else ("left" if offset > 0.0 else "right")
        action_id = f"primitive_{label}_{index}"
        candidates.append(
            _limit_candidate_speed(
                _arc_primitive(
                    action_id,
                    start,
                    unit,
                    perp,
                    progress_m=progress_m,
                    lateral_offset_m=offset,
                    horizon_steps=horizon_steps,
                ),
                dt_s=dt_s,
                cruise_speed_mps=cfg.cruise_speed_mps,
            )
        )

    if cfg.include_brake_primitive:
        brake_progress = min(cfg.brake_displacement_m, reach_m)
        candidates.append(
            _limit_candidate_speed(
                _arc_primitive(
                    "primitive_brake",
                    start,
                    unit,
                    perp,
                    progress_m=brake_progress,
                    lateral_offset_m=0.0,
                    horizon_steps=horizon_steps,
                ),
                dt_s=dt_s,
                cruise_speed_mps=cfg.cruise_speed_mps,
            )
        )

    if len(candidates) < 3:
        raise ValueError(
            "primitive generator must produce at least three candidates; "
            f"got {len(candidates)} (enlarge lateral_offsets_m or enable the brake primitive)"
        )

    for candidate in candidates:
        candidate.as_array(horizon_steps=horizon_steps)  # validates shape + finiteness
    return candidates


# ---------------------------------------------------------------------------
# Pure ranking helpers
# ---------------------------------------------------------------------------


def _waypoint_velocities(positions: NDArray[np.floating], dt_s: float) -> NDArray[np.floating]:
    """Return per-step velocities ``(T, 2)`` via central differences matching ``positions``."""
    return np.gradient(positions, dt_s, axis=0)


def _integrated_jerk(positions: NDArray[np.floating], dt_s: float) -> float:
    """Return the time-integrated jerk magnitude (m/s^2) of the waypoint polyline."""
    if positions.shape[0] < 2:
        return 0.0
    velocity = np.gradient(positions, dt_s, axis=0)
    acceleration = np.gradient(velocity, dt_s, axis=0)
    jerk = np.gradient(acceleration, dt_s, axis=0)
    return float(np.sum(np.linalg.norm(jerk, axis=1)) * dt_s)


def _path_length(positions: NDArray[np.floating]) -> float:
    """Return the total arc length (m) of the waypoint polyline."""
    if positions.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def _clearance_penalty(min_clearance_m: float, safe_clearance_m: float) -> float:
    """Return a linear clearance penalty that is zero above ``safe_clearance_m``."""
    if not math.isfinite(min_clearance_m):
        return 0.0
    if min_clearance_m >= safe_clearance_m:
        return 0.0
    return (safe_clearance_m - min_clearance_m) / safe_clearance_m


def _peak_risk_timing(
    estimate: ActionConditionedRiskEstimate,
    *,
    dt_s: float,
    horizon_steps: int,
    window_half_steps: int,
) -> PeakRiskTiming:
    """Build peak-risk timing from the estimate's first-passage distribution.

    Returns:
        Peak-risk timing in the anchor-step plus window shape consumed by
        ``critical_intervals``; ``peak_step == -1`` signals no contact mass.
    """
    first_passage = estimate.first_passage_distribution
    if not first_passage or max(first_passage) <= 0.0:
        return PeakRiskTiming(
            peak_step=-1,
            peak_time_s=0.0,
            peak_actor_id=None,
            window_start_step=0,
            window_end_step=0,
            first_passage_distribution=first_passage,
        )

    peak_step = int(np.argmax(first_passage))
    peak_time_s = peak_step * dt_s
    window_start = max(0, peak_step - window_half_steps)
    window_end = min(horizon_steps, peak_step + 1 + window_half_steps)

    peak_actor_id: int | None = None
    best_marginal = -1.0
    for contribution in estimate.per_actor:
        at_peak = contribution.first_contact_step_mode == peak_step
        weight = contribution.marginal_contact_probability * (2.0 if at_peak else 1.0)
        if weight > best_marginal:
            best_marginal = weight
            peak_actor_id = contribution.actor_id

    return PeakRiskTiming(
        peak_step=peak_step,
        peak_time_s=float(peak_time_s),
        peak_actor_id=peak_actor_id,
        window_start_step=window_start,
        window_end_step=window_end,
        first_passage_distribution=first_passage,
    )


def _score_components(
    estimate: ActionConditionedRiskEstimate,
    positions: NDArray[np.floating],
    *,
    dt_s: float,
    weights: RankingWeights,
) -> ScoreComponents:
    """Assemble the decomposed score components for one candidate.

    Returns:
        Decomposed :class:`ScoreComponents` (collision risk, travel time, jerk,
        path length, clearance penalty, raw clearance, calibration flag).
    """
    min_clearance_m = estimate.deterministic.min_clearance_m
    return ScoreComponents(
        calibrated_collision_risk=float(estimate.joint_contact_probability),
        travel_time_s=float(positions.shape[0] - 1) * dt_s,
        integrated_jerk=_integrated_jerk(positions, dt_s),
        path_length_m=_path_length(positions),
        clearance_penalty=_clearance_penalty(min_clearance_m, weights.safe_clearance_m),
        min_clearance_m=float(min_clearance_m),
        calibration_applied=False,
    )


def _composite_cost(components: ScoreComponents, weights: RankingWeights) -> float:
    """Return the weighted (lower-is-better) composite cost of the components."""
    return (
        weights.w_risk * components.calibrated_collision_risk
        + weights.w_time * components.travel_time_s
        + weights.w_jerk * components.integrated_jerk
        + weights.w_length * components.path_length_m
        + weights.w_clearance * components.clearance_penalty
    )


def _hard_gate(
    action: CandidateAction,
    pedestrians: Sequence[PedestrianState],
    *,
    robot_positions: NDArray[np.floating],
    robot_velocities: NDArray[np.floating],
    estimate: ActionConditionedRiskEstimate,
    risk_config: RiskEstimatorConfig,
    verifier_config: TrajectoryVerifierConfig | None,
    actuator_config: ActuatorLimitsConfig | None,
) -> HardGateResult:
    """Run the two authoritative deterministic hard gates for one candidate.

    Returns:
        :class:`HardGateResult` marking the candidate ineligible when either the
        trajectory verifier returns ``fallback_brake`` or the actuator-feasibility
        report is physically infeasible. A ``geometry_only_clear`` verdict is
        geometrically clear but physically infeasible, so it remains a hard-gate
        rejection rather than a candidate the ranker may select.
    """
    min_clearance_m = estimate.deterministic.min_clearance_m
    hazard_clearance_m = (
        _NO_HAZARD_CLEARANCE_M if not math.isfinite(min_clearance_m) else min_clearance_m
    )

    actuator_report = evaluate_actuator_feasibility(
        robot_positions=robot_positions,
        robot_velocities=robot_velocities,
        dt_s=risk_config.dt_s,
        hazard_clearance_m=hazard_clearance_m,
        config=actuator_config,
    )

    verifier_decision: str
    violated_predicates: tuple[str, ...]
    if pedestrians:
        verifier_result = verify_trajectory(
            robot_positions=robot_positions,
            robot_velocities=robot_velocities,
            pedestrian_positions=_pedestrian_positions(pedestrians, risk_config),
            pedestrian_velocities=_pedestrian_velocities(pedestrians, risk_config),
            dt_s=risk_config.dt_s,
            robot_radius_m=risk_config.robot_radius_m,
            pedestrian_radius_m=risk_config.pedestrian_radius_m,
            config=verifier_config,
        )
        verifier_decision = verifier_result.decision
        violated_predicates = verifier_result.violated_predicates
    else:
        verifier_decision = "skipped_no_hazard"
        violated_predicates = ()

    ineligible_reasons: list[str] = []
    if verifier_decision == DECISION_FALLBACK_BRAKE:
        ineligible_reasons.append("trajectory_verifier returned fallback_brake")
    if not actuator_report.physically_feasible:
        ineligible_reasons.append(
            f"actuator_feasibility reported physically infeasible ({actuator_report.verdict})"
        )

    eligible = not ineligible_reasons
    return HardGateResult(
        eligible=eligible,
        verifier_decision=verifier_decision,
        actuator_verdict=actuator_report.verdict,
        violated_predicates=violated_predicates,
        violated_limits=actuator_report.violated_limits,
        ineligibility_reason="; ".join(ineligible_reasons) if ineligible_reasons else None,
    )


def _pedestrian_positions(
    pedestrians: Sequence[PedestrianState], config: RiskEstimatorConfig
) -> NDArray[np.floating]:
    """Return the ``(N, 2)`` static pedestrian positions for the verifier."""
    return np.asarray([np.asarray(actor.position, dtype=float).reshape(2) for actor in pedestrians])


def _pedestrian_velocities(
    pedestrians: Sequence[PedestrianState], config: RiskEstimatorConfig
) -> NDArray[np.floating]:
    """Return the ``(N, 2)`` static pedestrian velocities for the verifier."""
    return np.asarray([np.asarray(actor.velocity, dtype=float).reshape(2) for actor in pedestrians])


def _provenance(action_id: str, estimate: ActionConditionedRiskEstimate) -> CandidateProvenance:
    """Lift the estimator provenance into the ranking provenance record.

    Returns:
        :class:`CandidateProvenance` carrying the reused estimator identifiers.
    """
    risk_provenance = estimate.provenance
    return CandidateProvenance(
        action_id=action_id,
        estimator_id=risk_provenance.estimator_id,
        forecast_model=risk_provenance.forecast_model,
        geometry_version=risk_provenance.geometry_version,
        config_hash=risk_provenance.config_hash,
        seed=risk_provenance.seed,
        risk_schema_version=risk_provenance.schema_version,
        abstained=estimate.uncertainty.abstained,
    )


# ---------------------------------------------------------------------------
# Public ranking entry point
# ---------------------------------------------------------------------------


def rank_trajectories(
    candidates: Sequence[CandidateAction],
    pedestrians: Sequence[PedestrianState],
    *,
    risk_config: RiskEstimatorConfig | None = None,
    weights: RankingWeights | None = None,
    verifier_config: TrajectoryVerifierConfig | None = None,
    actuator_config: ActuatorLimitsConfig | None = None,
    peak_window_half_steps: int = _DEFAULT_PEAK_WINDOW_HALF_STEPS,
) -> list[CandidateRanking]:
    """Rank candidate actions by decomposed score components under hard gates.

    A pure, deterministic function. For each candidate it reuses
    :func:`estimate_action_conditioned_risk` for the probabilistic component, runs
    the two authoritative deterministic hard gates
    (:func:`verify_trajectory` and :func:`evaluate_actuator_feasibility`), and
    reports every score component separately. A candidate rejected by either hard
    gate is marked ineligible regardless of its probabilistic collision risk.

    Args:
        candidates: Candidate actions to rank. Each must carry waypoints of shape
            ``(risk_config.horizon_steps + 1, 2)``.
        pedestrians: Actor states at the planning timestep (may be empty).
        risk_config: Estimator configuration; defaults to
            :class:`RiskEstimatorConfig`. Its horizon and timestep must match the
            candidate waypoint shapes and are reused by both hard gates.
        weights: Ranking weights; defaults to :class:`RankingWeights`.
        verifier_config: Trajectory-verifier thresholds; defaults to
            :class:`TrajectoryVerifierConfig`.
        actuator_config: Actuator-feasibility limits; defaults to
            :class:`ActuatorLimitsConfig`.
        peak_window_half_steps: Half-window (horizon steps) around the peak-risk
            timestep for critical-interval inspection.

    Returns:
        Rankings ordered with eligible candidates first (1-based rank by
        ascending composite cost), followed by ineligible candidates. Each record
        exposes every score component, eligibility, hard-gate outcomes, peak-risk
        timing, and full risk-estimator provenance.

    Raises:
        ValueError: If ``peak_window_half_steps`` is negative.
    """
    if peak_window_half_steps < 0:
        raise ValueError("peak_window_half_steps must be >= 0")
    risk_config = risk_config if risk_config is not None else RiskEstimatorConfig()
    weights = weights if weights is not None else RankingWeights()

    unranked: list[CandidateRanking] = []
    for action in candidates:
        robot_positions = action.as_array(horizon_steps=risk_config.horizon_steps)
        robot_velocities = _waypoint_velocities(robot_positions, risk_config.dt_s)
        estimate = estimate_action_conditioned_risk(
            action, pedestrians, risk_config, measure_latency=False
        )
        components = _score_components(
            estimate, robot_positions, dt_s=risk_config.dt_s, weights=weights
        )
        composite = _composite_cost(components, weights)
        peak_risk = _peak_risk_timing(
            estimate,
            dt_s=risk_config.dt_s,
            horizon_steps=risk_config.horizon_steps,
            window_half_steps=peak_window_half_steps,
        )
        hard_gate = _hard_gate(
            action,
            pedestrians,
            robot_positions=robot_positions,
            robot_velocities=robot_velocities,
            estimate=estimate,
            risk_config=risk_config,
            verifier_config=verifier_config,
            actuator_config=actuator_config,
        )
        unranked.append(
            CandidateRanking(
                action_id=action.action_id,
                rank=-1,
                eligible=hard_gate.eligible,
                composite_score=float(composite),
                components=components,
                peak_risk=peak_risk,
                hard_gate=hard_gate,
                provenance=_provenance(action.action_id, estimate),
                joint_contact_probability=float(estimate.joint_contact_probability),
                estimate=estimate,
            )
        )

    eligible = [record for record in unranked if record.eligible]
    ineligible = [record for record in unranked if not record.eligible]
    eligible.sort(key=lambda record: (record.composite_score, record.action_id))
    ineligible.sort(key=lambda record: (record.composite_score, record.action_id))

    ranked: list[CandidateRanking] = []
    for position, record in enumerate(eligible, start=1):
        ranked.append(_with_rank(record, position))
    ranked.extend(ineligible)
    return ranked


def _with_rank(record: CandidateRanking, rank: int) -> CandidateRanking:
    """Return a copy of ``record`` with the assigned eligible rank."""
    return CandidateRanking(
        action_id=record.action_id,
        rank=rank,
        eligible=record.eligible,
        composite_score=record.composite_score,
        components=record.components,
        peak_risk=record.peak_risk,
        hard_gate=record.hard_gate,
        provenance=record.provenance,
        joint_contact_probability=record.joint_contact_probability,
        estimate=record.estimate,
        claim_boundary=record.claim_boundary,
    )


__all__ = [
    "RANKER_CLAIM_BOUNDARY",
    "RANKER_SCHEMA_VERSION",
    "CandidateProvenance",
    "CandidateRanking",
    "HardGateResult",
    "PeakRiskTiming",
    "PrimitiveGeneratorConfig",
    "RankingWeights",
    "ScoreComponents",
    "generate_primitive_candidates",
    "rank_trajectories",
]
