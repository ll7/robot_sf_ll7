"""Constant-velocity baselines for the action-conditioned collision-risk API (issue #5444).

The public entry point is :func:`estimate_action_conditioned_risk`. It scores a
single candidate robot action (a deterministic robot trajectory over the horizon)
against a set of pedestrians described as :class:`~robot_sf.nav.predictive_types.PedestrianState`,
and returns a versioned :class:`~robot_sf.research.collision_risk.schema.ActionConditionedRiskEstimate`.

Forecast model (declared assumptions)
-------------------------------------
Each pedestrian is propagated with a **constant-velocity** mean. Uncertainty is
injected as a per-sample Gaussian perturbation of that constant velocity::

    v_k^(s) = v_k0 + eps_k^(s),   eps_k^(s) ~ N(0, sigma_v^2 I_2)

so per-sample position variance grows linearly with time (a constant-but-uncertain
velocity model). Cross-actor correlation ``rho`` is injected via a shared latent
component ``eps_k = sqrt(1 - rho) z_k + sqrt(rho) z_shared`` (per axis), which lets
the joint Monte Carlo estimate differ from the independence approximation. The
robot follows its candidate trajectory deterministically (the estimand conditions
on the action ``u``).

Geometry (exact footprint semantics)
------------------------------------
Robot and actors are modelled as discs. Within each timestep interval both the
robot and each actor move linearly, so the minimum footprint clearance over the
interval has a closed form (segment-to-segment minimum distance minus the summed
radii). This is exact for the piecewise-linear discretised motion and detects
grazing contacts that a step-endpoint-only test would miss.

Status: API + baseline fixture evidence. Not a calibrated benchmark risk claim.
Hard guards remain authoritative; no ``safe`` label is emitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.research.collision_risk.schema import (
    ActionConditionedRiskEstimate,
    DeterministicRiskFields,
    LatencySummary,
    PerActorContribution,
    RiskProvenance,
    UncertaintyState,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.nav.predictive_types import PedestrianState

ESTIMATOR_ID = "constant_velocity_mc.v1"
FORECAST_MODEL_ID = "constant_velocity_gaussian.v1"
GEOMETRY_VERSION = "disc_footprint_segment.v1"


class CollisionRiskInputError(ValueError):
    """Raised, fail-closed, when risk-estimation inputs are malformed."""


@dataclass(frozen=True)
class RiskEstimatorConfig:
    """Configuration for the constant-velocity collision-risk estimator.

    Attributes:
        horizon_steps: Number of horizon steps ``H`` (must be positive).
        dt_s: Timestep in seconds (must be positive).
        n_samples: Monte Carlo sample count.
        velocity_std_m_s: Per-axis standard deviation of the constant-velocity
            perturbation (declares the forecast noise scale).
        cross_actor_correlation: Correlation ``rho`` in ``[0, 1)`` of the velocity
            perturbation shared across actors.
        robot_radius_m: Robot disc radius.
        pedestrian_radius_m: Default actor disc radius when an actor supplies none.
        seed: Monte Carlo seed (fixed for reproducibility).
        deadline_ms: Control deadline used to classify latency online vs offline.
        max_pedestrian_speed_m_s: Speed above which an actor is flagged
            out-of-distribution for the constant-velocity model.
        ci95_abstain_halfwidth: Abstain when the 95% interval half-width on the
            joint estimate exceeds this value.
        min_samples_for_estimate: Abstain when fewer samples than this are used.
    """

    horizon_steps: int = 20
    dt_s: float = 0.1
    n_samples: int = 512
    velocity_std_m_s: float = 0.3
    cross_actor_correlation: float = 0.0
    robot_radius_m: float = 0.3
    pedestrian_radius_m: float = 0.3
    seed: int = 0
    deadline_ms: float = 100.0
    max_pedestrian_speed_m_s: float = 2.5
    ci95_abstain_halfwidth: float = 0.05
    min_samples_for_estimate: int = 64

    def __post_init__(self) -> None:
        """Validate configuration invariants, failing closed on bad values."""
        if self.horizon_steps <= 0:
            raise CollisionRiskInputError("horizon_steps must be positive")
        if self.dt_s <= 0.0:
            raise CollisionRiskInputError("dt_s must be positive")
        if self.n_samples <= 0:
            raise CollisionRiskInputError("n_samples must be positive")
        if self.velocity_std_m_s < 0.0:
            raise CollisionRiskInputError("velocity_std_m_s must be non-negative")
        if not (0.0 <= self.cross_actor_correlation < 1.0):
            raise CollisionRiskInputError("cross_actor_correlation must be in [0, 1)")
        if self.robot_radius_m < 0.0 or self.pedestrian_radius_m < 0.0:
            raise CollisionRiskInputError("radii must be non-negative")

    @property
    def horizon_s(self) -> float:
        """Horizon length in seconds."""
        return self.horizon_steps * self.dt_s

    def config_hash(self) -> str:
        """Return a stable short hash of the configuration for provenance."""
        payload = json.dumps(
            {
                "horizon_steps": self.horizon_steps,
                "dt_s": self.dt_s,
                "n_samples": self.n_samples,
                "velocity_std_m_s": self.velocity_std_m_s,
                "cross_actor_correlation": self.cross_actor_correlation,
                "robot_radius_m": self.robot_radius_m,
                "pedestrian_radius_m": self.pedestrian_radius_m,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CandidateAction:
    """Planner-agnostic candidate robot action over the horizon.

    The action is represented as the deterministic robot trajectory it induces:
    ``waypoints`` of shape ``(H + 1, 2)`` giving robot positions at steps
    ``t, t+1, ..., t+H``. This keeps the API agnostic to how a specific planner
    parameterises actions (velocity command, MPC plan, RL action) -- callers roll
    the action out once and hand over the resulting waypoints.

    Attributes:
        action_id: Stable identifier for the candidate action.
        waypoints: ``(H + 1, 2)`` robot positions over the horizon.
        representation: Free-form label of the source representation.
    """

    action_id: str
    waypoints: np.ndarray
    representation: str = "waypoints"

    def as_array(self, *, horizon_steps: int) -> np.ndarray:
        """Return validated ``(H + 1, 2)`` float waypoints, failing closed."""
        array = np.asarray(self.waypoints, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            raise CollisionRiskInputError("action waypoints must have shape (H + 1, 2)")
        if array.shape[0] != horizon_steps + 1:
            raise CollisionRiskInputError(
                f"action waypoints must have {horizon_steps + 1} rows for horizon "
                f"{horizon_steps}, got {array.shape[0]}"
            )
        if not np.all(np.isfinite(array)):
            raise CollisionRiskInputError("action waypoints must be finite")
        return array


def action_from_constant_velocity(
    action_id: str,
    start_position: Sequence[float] | np.ndarray,
    velocity: Sequence[float] | np.ndarray,
    *,
    horizon_steps: int,
    dt_s: float,
) -> CandidateAction:
    """Build a :class:`CandidateAction` from a constant robot velocity command.

    Args:
        action_id: Identifier for the action.
        start_position: Robot start position ``(2,)``.
        velocity: Constant robot velocity ``(2,)`` in m/s.
        horizon_steps: Number of horizon steps.
        dt_s: Timestep in seconds.

    Returns:
        Candidate action whose waypoints follow the constant velocity.
    """
    start = np.asarray(start_position, dtype=float).reshape(2)
    vel = np.asarray(velocity, dtype=float).reshape(2)
    steps = np.arange(horizon_steps + 1, dtype=float)[:, None]
    waypoints = start[None, :] + steps * dt_s * vel[None, :]
    return CandidateAction(
        action_id=action_id, waypoints=waypoints, representation="constant_velocity"
    )


def segment_min_distance(robot_xy: np.ndarray, actor_xy: np.ndarray) -> np.ndarray:
    """Closed-form minimum centre distance per horizon interval.

    This is the canonical contact geometry used by
    :func:`estimate_action_conditioned_risk`. It is public so downstream
    calibration, replay, and label-generation code can compute contact on the
    *identical* geometry the estimator uses (see :func:`pedestrian_arrays`);
    subtract the summed robot+actor radii to obtain footprint clearance, and a
    non-positive clearance is a contact.

    Both the robot and each actor move linearly within a timestep interval, so
    the relative displacement is affine in the sub-step parameter ``u in [0, 1]``
    and its minimum norm has a closed form.

    Args:
        robot_xy: ``(H + 1, 2)`` robot positions.
        actor_xy: ``(..., H + 1, 2)`` actor positions (any leading batch dims).

    Returns:
        ``(..., H)`` array of minimum centre distances per interval.
    """
    # Relative displacement actor - robot at interval start (w0) and end (w1).
    w0 = actor_xy[..., :-1, :] - robot_xy[:-1, :]
    w1 = actor_xy[..., 1:, :] - robot_xy[1:, :]
    dw = w1 - w0
    denom = np.sum(dw * dw, axis=-1)
    numer = -np.sum(w0 * dw, axis=-1)
    # Guard the degenerate (no relative motion) interval where denom == 0.
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    u_star = np.clip(np.where(denom > 0.0, numer / safe_denom, 0.0), 0.0, 1.0)
    closest = w0 + u_star[..., None] * dw
    return np.linalg.norm(closest, axis=-1)


def pedestrian_arrays(
    pedestrians: Sequence[PedestrianState],
    config: RiskEstimatorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract validated position/velocity/radius/id arrays from pedestrians.

    Public companion to :func:`segment_min_distance`: downstream calibration and
    label-generation code reuses this extractor so ground-truth contact labels
    are computed on the same actor state the estimator scores. Fails closed on
    non-2D or non-finite pedestrian state.

    Returns:
        Tuple ``(positions (K, 2), velocities (K, 2), radii (K,), ids (K,))``.
    """
    positions = []
    velocities = []
    radii = []
    ids = []
    for actor in pedestrians:
        pos = np.asarray(actor.position, dtype=float).reshape(-1)
        vel = np.asarray(actor.velocity, dtype=float).reshape(-1)
        if pos.shape != (2,) or vel.shape != (2,):
            raise CollisionRiskInputError("pedestrian position and velocity must be 2D")
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise CollisionRiskInputError("pedestrian position and velocity must be finite")
        positions.append(pos)
        velocities.append(vel)
        radii.append(
            float(
                getattr(actor, "radius", config.pedestrian_radius_m) or config.pedestrian_radius_m
            )
        )
        ids.append(int(actor.id))
    if positions:
        return (
            np.stack(positions),
            np.stack(velocities),
            np.asarray(radii, dtype=float),
            np.asarray(ids, dtype=int),
        )
    empty2 = np.zeros((0, 2), dtype=float)
    return empty2, empty2, np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)


def _nominal_positions(
    ped_pos: np.ndarray, ped_vel: np.ndarray, config: RiskEstimatorConfig
) -> np.ndarray:
    """Return the noise-free constant-velocity actor rollout ``(K, H + 1, 2)``."""
    steps = np.arange(config.horizon_steps + 1, dtype=float)
    return ped_pos[:, None, :] + steps[None, :, None] * config.dt_s * ped_vel[:, None, :]


def _sample_positions(
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    config: RiskEstimatorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample constant-velocity actor rollouts ``(S, K, H + 1, 2)``.

    Velocity perturbations carry a shared latent component so actors are
    correlated with coefficient ``config.cross_actor_correlation``.

    Returns:
        ``(S, K, H + 1, 2)`` sampled actor positions over the horizon.
    """
    n_actors = ped_pos.shape[0]
    sigma = config.velocity_std_m_s
    rho = config.cross_actor_correlation
    independent = rng.standard_normal((config.n_samples, n_actors, 2))
    shared = rng.standard_normal((config.n_samples, 1, 2))
    eps = sigma * (math.sqrt(1.0 - rho) * independent + math.sqrt(rho) * shared)
    sampled_vel = ped_vel[None, :, :] + eps  # (S, K, 2)
    steps = np.arange(config.horizon_steps + 1, dtype=float)
    # (S, K, H+1, 2)
    return (
        ped_pos[None, :, None, :]
        + steps[None, None, :, None] * config.dt_s * sampled_vel[:, :, None, :]
    )


def _deterministic_fields(
    robot_xy: np.ndarray,
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    radii_sum: np.ndarray,
    config: RiskEstimatorConfig,
) -> DeterministicRiskFields:
    """Compute the non-probabilistic deterministic warning fields.

    Returns:
        The deterministic clearance/TTC/VO/reachability warning fields.
    """
    n_actors = ped_pos.shape[0]
    if n_actors == 0:
        return DeterministicRiskFields(
            min_clearance_m=float("inf"),
            min_clearance_step=-1,
            ttc_s=float("inf"),
            contact_certain=False,
            first_contact_step=-1,
        )

    nominal = _nominal_positions(ped_pos, ped_vel, config)  # (K, H+1, 2)
    seg_dist = segment_min_distance(robot_xy, nominal)  # (K, H)
    clearance = seg_dist - radii_sum[:, None]  # (K, H)

    flat_index = int(np.argmin(clearance))
    min_actor, min_step = np.unravel_index(flat_index, clearance.shape)
    min_clearance = float(clearance[min_actor, min_step])

    contact = clearance <= 0.0
    contact_any_step = contact.any(axis=0)  # (H,)
    if bool(contact_any_step.any()):
        first_step = int(np.argmax(contact_any_step))
        ttc_s = first_step * config.dt_s
        contact_certain = True
    else:
        first_step = -1
        ttc_s = float("inf")
        contact_certain = False

    # Velocity-obstacle membership: does the robot's initial commanded velocity,
    # extrapolated at constant velocity, collide with each actor within horizon?
    robot_v0 = (robot_xy[1] - robot_xy[0]) / config.dt_s
    steps = np.arange(config.horizon_steps + 1, dtype=float)
    robot_cv = robot_xy[0][None, :] + steps[:, None] * config.dt_s * robot_v0[None, :]
    vo_dist = segment_min_distance(robot_cv, nominal)  # (K, H)
    vo_flags = tuple(bool(value) for value in (vo_dist - radii_sum[:, None] <= 0.0).any(axis=1))

    # Conservative reachable-set warning: actor within max robot reach over horizon.
    robot_seg_speed = np.linalg.norm(np.diff(robot_xy, axis=0), axis=1) / config.dt_s
    v_robot_max = float(robot_seg_speed.max()) if robot_seg_speed.size else 0.0
    reach = v_robot_max * config.horizon_s
    start_gap = np.linalg.norm(ped_pos - robot_xy[0][None, :], axis=1) - radii_sum
    reachable_flags = tuple(bool(value) for value in (start_gap <= reach))

    return DeterministicRiskFields(
        min_clearance_m=min_clearance,
        min_clearance_step=int(min_step),
        ttc_s=ttc_s,
        contact_certain=contact_certain,
        first_contact_step=first_step,
        velocity_obstacle_flags=vo_flags,
        reachable_actor_flags=reachable_flags,
    )


def _first_passage_and_hazard(
    first_contact_step: np.ndarray, horizon_steps: int, n_samples: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the first-passage distribution and discrete hazard per step.

    Args:
        first_contact_step: ``(S,)`` first contact step per sample, ``-1`` when
            the sample never contacts.
        horizon_steps: Number of horizon steps ``H``.
        n_samples: Number of Monte Carlo samples.

    Returns:
        Tuple ``(first_passage, hazard)`` each of length ``H``.
    """
    counts = np.zeros(horizon_steps, dtype=float)
    contacting = first_contact_step[first_contact_step >= 0]
    for step in contacting:
        counts[int(step)] += 1.0
    first_passage = counts / n_samples

    hazard = np.zeros(horizon_steps, dtype=float)
    survivors = 1.0
    for step in range(horizon_steps):
        if survivors > 1e-12:
            hazard[step] = first_passage[step] / survivors
        survivors -= first_passage[step]
    return tuple(float(value) for value in first_passage), tuple(float(value) for value in hazard)


def estimate_action_conditioned_risk(
    action: CandidateAction,
    pedestrians: Sequence[PedestrianState],
    config: RiskEstimatorConfig | None = None,
    *,
    measure_latency: bool = True,
) -> ActionConditionedRiskEstimate:
    """Estimate action-conditioned collision risk for one candidate action.

    Args:
        action: Candidate robot action (deterministic waypoints over the horizon).
        pedestrians: Actor states at time ``t`` (position, velocity, optional
            semantic context).
        config: Estimator configuration; defaults to :class:`RiskEstimatorConfig`.
        measure_latency: When True, time this call and attach a single-call
            :class:`~robot_sf.research.collision_risk.schema.LatencySummary`.

    Returns:
        A validated :class:`~robot_sf.research.collision_risk.schema.ActionConditionedRiskEstimate`.
    """
    config = config or RiskEstimatorConfig()
    start_ns = time.perf_counter_ns()

    robot_xy = action.as_array(horizon_steps=config.horizon_steps)
    ped_pos, ped_vel, radii, ids = pedestrian_arrays(pedestrians, config)
    radii_sum = radii + config.robot_radius_m

    deterministic = _deterministic_fields(robot_xy, ped_pos, ped_vel, radii_sum, config)

    n_actors = ped_pos.shape[0]
    horizon = config.horizon_steps
    if n_actors == 0:
        joint = 0.0
        per_actor: tuple[PerActorContribution, ...] = ()
        union_bound = 0.0
        independence = 0.0
        first_passage = tuple(0.0 for _ in range(horizon))
        hazard = tuple(0.0 for _ in range(horizon))
        ood_flags: tuple[bool, ...] = ()
    else:
        rng = np.random.default_rng(config.seed)
        sampled = _sample_positions(ped_pos, ped_vel, config, rng)  # (S, K, H+1, 2)
        seg_dist = segment_min_distance(robot_xy, sampled)  # (S, K, H)
        contact = seg_dist - radii_sum[None, :, None] <= 0.0  # (S, K, H)

        contact_per_actor = contact.any(axis=2)  # (S, K)
        contact_any = contact_per_actor.any(axis=1)  # (S,)
        joint = float(contact_any.mean())

        marginals = contact_per_actor.mean(axis=0)  # (K,)
        union_bound = float(min(1.0, marginals.sum()))
        independence = float(1.0 - np.prod(1.0 - marginals))

        # First contact step over any actor (H = "never" sentinel), then mask.
        step_contact = contact.any(axis=1)  # (S, H)
        has_contact = step_contact.any(axis=1)
        first_step = np.where(has_contact, np.argmax(step_contact, axis=1), -1)
        first_passage, hazard = _first_passage_and_hazard(first_step, horizon, config.n_samples)

        per_actor = _build_per_actor(contact, ids, marginals, config)
        speeds = np.linalg.norm(ped_vel, axis=1)
        ood_flags = tuple(bool(value) for value in (speeds > config.max_pedestrian_speed_m_s))

    uncertainty = _build_uncertainty(joint, config, ood_flags)
    provenance = RiskProvenance(
        estimator_id=ESTIMATOR_ID,
        forecast_model=FORECAST_MODEL_ID,
        geometry_version=GEOMETRY_VERSION,
        horizon_steps=horizon,
        horizon_s=config.horizon_s,
        dt_s=config.dt_s,
        action_id=action.action_id,
        action_representation=action.representation,
        config_hash=config.config_hash(),
        seed=config.seed,
    )

    latency = None
    if measure_latency:
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        classification = "online" if elapsed_ms <= config.deadline_ms else "offline_only"
        latency = LatencySummary(
            p50_ms=elapsed_ms,
            p95_ms=elapsed_ms,
            p99_ms=elapsed_ms,
            n_calls=1,
            deadline_ms=config.deadline_ms,
            deadline_misses=int(elapsed_ms > config.deadline_ms),
            classification=classification,
        )

    estimate = ActionConditionedRiskEstimate(
        joint_contact_probability=joint,
        no_contact_probability=1.0 - joint,
        per_actor=per_actor,
        union_bound_probability=union_bound,
        independence_approx_probability=independence,
        first_passage_distribution=first_passage,
        binned_hazard=hazard,
        deterministic=deterministic,
        uncertainty=uncertainty,
        provenance=provenance,
        latency=latency,
    )
    return estimate.validate()


def _build_per_actor(
    contact: np.ndarray,
    ids: np.ndarray,
    marginals: np.ndarray,
    config: RiskEstimatorConfig,
) -> tuple[PerActorContribution, ...]:
    """Build per-actor marginal contributions with first-contact-step modes.

    Returns:
        A tuple of per-actor marginal contributions.
    """
    contributions = []
    for actor_index, actor_id in enumerate(ids):
        actor_contact = contact[:, actor_index, :]  # (S, H)
        has_contact = actor_contact.any(axis=1)
        if bool(has_contact.any()):
            first_steps = np.argmax(actor_contact[has_contact], axis=1)
            mode_step = int(Counter(int(step) for step in first_steps).most_common(1)[0][0])
        else:
            mode_step = -1
        contributions.append(
            PerActorContribution(
                actor_id=int(actor_id),
                marginal_contact_probability=float(marginals[actor_index]),
                first_contact_step_mode=mode_step,
            )
        )
    return tuple(contributions)


def _build_uncertainty(
    joint: float, config: RiskEstimatorConfig, ood_flags: tuple[bool, ...]
) -> UncertaintyState:
    """Build the Monte Carlo / OOD / abstention state for the joint estimate.

    Returns:
        The uncertainty state including standard error, interval, and abstention.
    """
    std_error = math.sqrt(max(joint * (1.0 - joint), 0.0) / config.n_samples)
    ci_halfwidth = 1.96 * std_error
    reasons: list[str] = []
    if config.n_samples < config.min_samples_for_estimate:
        reasons.append(
            f"n_samples {config.n_samples} below minimum {config.min_samples_for_estimate}"
        )
    if ci_halfwidth > config.ci95_abstain_halfwidth:
        reasons.append(
            f"95% interval half-width {ci_halfwidth:.3f} exceeds "
            f"{config.ci95_abstain_halfwidth:.3f}"
        )
    if any(ood_flags):
        reasons.append("at least one actor speed is out of the constant-velocity model range")
    return UncertaintyState(
        n_samples=config.n_samples,
        mc_standard_error=std_error,
        ci95_halfwidth=ci_halfwidth,
        abstained=bool(reasons),
        abstention_reasons=tuple(reasons),
        ood_actor_flags=ood_flags,
    )


# Backward-compatible internal aliases. The leading-underscore names were the
# only handle downstream code had before issue #5468 promoted the contact
# geometry to public API; keep them as aliases so pre-existing imports of the
# private names keep resolving to the exact same objects.
_segment_min_distance = segment_min_distance
_pedestrian_arrays = pedestrian_arrays


__all__ = [
    "ESTIMATOR_ID",
    "FORECAST_MODEL_ID",
    "GEOMETRY_VERSION",
    "CandidateAction",
    "CollisionRiskInputError",
    "RiskEstimatorConfig",
    "action_from_constant_velocity",
    "estimate_action_conditioned_risk",
    "pedestrian_arrays",
    "segment_min_distance",
]
