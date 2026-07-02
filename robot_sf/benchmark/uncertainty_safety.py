"""Uncertainty-aware safety: conformal prediction buffers and intrusion metrics (issue #3974).

Collision-only evaluation is a sparse, late signal, and point predictions of pedestrian
futures ignore uncertainty. This module implements the *bounded first slice* of
uncertainty-aware safety: it turns pedestrian-prediction residuals into calibrated spatial
**buffers** around predicted futures, and measures how far a robot **intrudes** into those
buffers over a run.

Two conformal estimators are provided:

* :func:`split_conformal_radius` -- offline split-conformal quantile of a fixed calibration
  set. On exchangeable data its buffer marginally covers the true future with probability at
  least the target, which is the acceptance signal for "empirical coverage approaches the
  target on a calibration set".
* :func:`adaptive_conformal_buffers` -- online Adaptive Conformal Inference (ACI,
  Gibbs & Candes 2021) over a streaming residual sequence. It integrates the realized
  miscoverage error to keep long-run coverage near the target even under distribution shift,
  which is the SoNIC-style setting where residual statistics drift as the crowd changes.

The intrusion metrics (:func:`compute_intrusion_metrics`) compare a robot trajectory to
pedestrian current positions, point predictions, and uncertainty buffers, producing the
time-ratio and cumulative-depth fields named in the issue.

.. admonition:: Claim boundary
   :class: note

   These are **versioned modeling choices**, labeled diagnostic. Producing a conformal radius
   or an intrusion ratio is not evidence of a safety guarantee, prediction quality, or planning
   improvement; the residual sequences must be exchangeable/stationary for the marginal-coverage
   property to hold, and any benchmark claim requires separate reproducible evidence per the
   project's maintainer values. The runtime uncertainty-triggered fallback is a deliberate
   successor slice and is intentionally *not* implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.benchmark.finite_checks import require_finite_array, require_finite_scalar

if TYPE_CHECKING:
    from numpy.typing import NDArray

SPLIT_CONFORMAL_RADIUS_SCHEMA = "uncertainty_safety.split_conformal_radius.v1"
ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA = "uncertainty_safety.adaptive_conformal_buffers.v1"
INTRUSION_METRICS_SCHEMA = "uncertainty_safety.intrusion_metrics.v1"


def residual_magnitudes(
    predicted: NDArray[np.float64],
    actual: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Euclidean nonconformity scores between predicted and actual positions.

    Args:
        predicted: Predicted positions, shape ``(N, 2)``.
        actual: Realized positions, shape ``(N, 2)``.

    Returns:
        Per-row residual magnitude ``||predicted - actual||``, shape ``(N,)``.

    Raises:
        ValueError: If shapes disagree or are not ``(N, 2)``.
    """
    pred = require_finite_array("predicted", predicted)
    act = require_finite_array("actual", actual)
    if pred.shape != act.shape:
        raise ValueError("predicted and actual must share the same shape")
    if pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError("predicted and actual must have shape (N, 2)")
    return np.linalg.norm(pred - act, axis=-1)


def split_conformal_radius(
    calibration_residuals: NDArray[np.float64],
    *,
    coverage_target: float = 0.9,
) -> float:
    """Split-conformal buffer radius from a fixed calibration residual set.

    Uses the finite-sample-corrected empirical quantile: with ``n`` calibration scores the
    radius is the ``k``-th smallest score where ``k = ceil((n + 1) * coverage_target)``. On
    exchangeable data a future residual falls within this radius with probability at least
    ``coverage_target``. When ``k > n`` the target cannot be certified from the sample and the
    radius is ``+inf`` (the buffer must be unbounded to promise that coverage).

    Args:
        calibration_residuals: Nonconformity scores, shape ``(n,)``, ``n >= 1``.
        coverage_target: Desired marginal coverage in ``(0, 1)``.

    Returns:
        Buffer radius (metres), possibly ``math.inf``.

    Raises:
        ValueError: If ``coverage_target`` is out of range or the calibration set is empty.
    """
    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage_target must be between 0 and 1 (exclusive)")
    scores = require_finite_array("calibration_residuals", calibration_residuals)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("calibration_residuals must be a non-empty 1-D array")
    ordered = np.sort(scores)
    n = ordered.size
    k = math.ceil((n + 1) * coverage_target)
    if k > n:
        return math.inf
    return float(ordered[k - 1])


@dataclass(frozen=True)
class AdaptiveConformalConfig:
    """Configuration for online Adaptive Conformal Inference.

    Attributes:
        coverage_target: Desired long-run coverage ``1 - alpha`` in ``(0, 1)``.
        step_size: ACI learning rate ``gamma`` (> 0) for the miscoverage integrator. Larger
            values track drift faster but let coverage oscillate more around the target.
        window: Trailing residual-history length used to fit each radius. ``None`` uses all
            past residuals.
        min_history: Number of past residuals required before a buffer is emitted; earlier
            steps are skipped (the radius is undefined without calibration data).
    """

    coverage_target: float = 0.9
    step_size: float = 0.05
    window: int | None = None
    min_history: int = 1


@dataclass(frozen=True)
class AdaptiveConformalResult:
    """Result of :func:`adaptive_conformal_buffers`.

    Attributes:
        indices: Residual indices for which a buffer was emitted (``min_history`` onward).
        radii: Buffer radius used at each emitted index (may include ``inf``).
        covered: Whether the realized residual fell within the radius at each emitted index.
        alphas: Adaptive miscoverage level ``alpha_t`` in effect at each emitted index.
        empirical_coverage: Fraction of emitted steps that were covered.
        coverage_target: Echoed target for provenance.
        schema: Versioned schema identifier.
    """

    indices: NDArray[np.int64]
    radii: NDArray[np.float64]
    covered: NDArray[np.bool_]
    alphas: NDArray[np.float64]
    empirical_coverage: float
    coverage_target: float
    schema: str = ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA


def _adaptive_radius(window_scores: NDArray[np.float64], *, effective_target: float) -> float:
    """Radius from the trailing window at the current adaptive coverage level.

    ACI lets ``alpha_t`` leave ``[0, 1]``; the corresponding coverage level is clamped so the
    radius degrades gracefully to ``0`` (cover nothing) or ``+inf`` (cover everything).

    Returns:
        Buffer radius (metres) for the current step, possibly ``0.0`` or ``math.inf``.
    """
    if effective_target <= 0.0:
        return 0.0
    if effective_target >= 1.0:
        return math.inf
    return split_conformal_radius(window_scores, coverage_target=effective_target)


def adaptive_conformal_buffers(
    residuals: NDArray[np.float64],
    *,
    config: AdaptiveConformalConfig | None = None,
) -> AdaptiveConformalResult:
    """Online Adaptive Conformal Inference over a streaming residual sequence.

    At each step ``t`` (once ``min_history`` residuals have been seen) the radius is the
    ``(1 - alpha_t)`` split-conformal quantile of the trailing window. The realized residual is
    then compared to the radius and the miscoverage level is integrated,
    ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`` with ``err_t = 1`` when the residual
    exceeds the radius, driving the empirical coverage toward ``coverage_target``.

    Args:
        residuals: Temporally ordered nonconformity scores, shape ``(T,)``.
        config: ACI configuration; defaults to :class:`AdaptiveConformalConfig`.

    Returns:
        An :class:`AdaptiveConformalResult` with per-step radii, coverage flags, and the
        alpha trace.

    Raises:
        ValueError: If ``residuals`` is not 1-D, or the config values are invalid.
    """
    cfg = config or AdaptiveConformalConfig()
    if not 0.0 < cfg.coverage_target < 1.0:
        raise ValueError("coverage_target must be between 0 and 1 (exclusive)")
    if cfg.step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if cfg.min_history < 1:
        raise ValueError("min_history must be at least 1")
    if cfg.window is not None and cfg.window < 1:
        raise ValueError("window must be None or a positive integer")

    scores = require_finite_array("residuals", residuals)
    if scores.ndim != 1:
        raise ValueError("residuals must be a 1-D array")

    alpha_target = 1.0 - cfg.coverage_target
    alpha_t = alpha_target
    indices: list[int] = []
    radii: list[float] = []
    covered: list[bool] = []
    alphas: list[float] = []

    for t in range(cfg.min_history, scores.size):
        start = 0 if cfg.window is None else max(0, t - cfg.window)
        window_scores = scores[start:t]
        radius = _adaptive_radius(window_scores, effective_target=1.0 - alpha_t)
        is_covered = bool(scores[t] <= radius)
        err = 0.0 if is_covered else 1.0

        indices.append(t)
        radii.append(radius)
        covered.append(is_covered)
        alphas.append(alpha_t)

        alpha_t = alpha_t + cfg.step_size * (alpha_target - err)

    covered_arr = np.asarray(covered, dtype=bool)
    empirical = float(covered_arr.mean()) if covered_arr.size else 0.0
    return AdaptiveConformalResult(
        indices=np.asarray(indices, dtype=np.int64),
        radii=np.asarray(radii, dtype=np.float64),
        covered=covered_arr,
        alphas=np.asarray(alphas, dtype=np.float64),
        empirical_coverage=empirical,
        coverage_target=cfg.coverage_target,
    )


@dataclass(frozen=True)
class IntrusionMetrics:
    """Cumulative-intrusion safety metrics over a run (issue #3974).

    Time ratios are the fraction of timesteps at which *any* pedestrian's relevant region is
    intruded. Depths are measured against the uncertainty buffer (``base_radius`` plus the
    conformal radius) around the predicted future position, in metres, aggregated over the run.

    Attributes:
        current_position_intrusion_time_ratio: Steps where the robot is within ``base_radius``
            of a pedestrian's *current* position.
        predicted_trajectory_intrusion_time_ratio: Steps where the robot is within
            ``base_radius`` of a pedestrian's *predicted* position (point prediction).
        uncertainty_buffer_intrusion_time_ratio: Steps where the robot is within the
            uncertainty buffer around the predicted position.
        cumulative_intrusion_depth: Sum over steps of the deepest per-step penetration into any
            uncertainty buffer (metres).
        max_intrusion_depth: Largest single-step penetration into any uncertainty buffer.
        num_steps: Number of timesteps evaluated.
        schema: Versioned schema identifier.
    """

    current_position_intrusion_time_ratio: float
    predicted_trajectory_intrusion_time_ratio: float
    uncertainty_buffer_intrusion_time_ratio: float
    cumulative_intrusion_depth: float
    max_intrusion_depth: float
    num_steps: int
    schema: str = INTRUSION_METRICS_SCHEMA


def _broadcast_radii(
    conformal_radii: NDArray[np.float64] | float,
    *,
    num_steps: int,
    num_peds: int,
) -> NDArray[np.float64]:
    """Normalize conformal radii to a ``(T, P)`` array from scalar, ``(T,)``, or ``(T, P)``.

    Returns:
        Non-negative radii array with shape ``(num_steps, num_peds)``.
    """
    array = np.asarray(conformal_radii, dtype=np.float64)
    if array.ndim == 0:
        array = np.full((num_steps, num_peds), float(array))
    elif array.ndim == 1:
        if array.shape[0] != num_steps:
            raise ValueError("1-D conformal_radii must have length num_steps")
        array = np.repeat(array[:, None], num_peds, axis=1)
    elif array.ndim == 2:
        if array.shape != (num_steps, num_peds):
            raise ValueError("2-D conformal_radii must have shape (num_steps, num_peds)")
    else:
        raise ValueError("conformal_radii must be scalar, (T,), or (T, P)")
    if np.any(array < 0.0):
        raise ValueError("conformal_radii must be non-negative")
    return array


def compute_intrusion_metrics(
    robot_positions: NDArray[np.float64],
    pedestrian_positions: NDArray[np.float64],
    predicted_positions: NDArray[np.float64],
    conformal_radii: NDArray[np.float64] | float,
    *,
    base_radius: float,
) -> IntrusionMetrics:
    """Compute cumulative-intrusion safety metrics against conformal buffers.

    Args:
        robot_positions: Robot positions, shape ``(T, 2)``.
        pedestrian_positions: Pedestrian *current* positions, shape ``(T, P, 2)``.
        predicted_positions: Predicted pedestrian positions relevant at each step, shape
            ``(T, P, 2)``. The caller aligns the prediction horizon; this function is
            horizon-agnostic.
        conformal_radii: Uncertainty-buffer radius added to ``base_radius`` around each
            predicted position. Scalar, ``(T,)``, or ``(T, P)``; non-negative.
        base_radius: Personal-space radius (metres, > 0) that defines a bare intrusion.

    Returns:
        Populated :class:`IntrusionMetrics`.

    Raises:
        ValueError: On shape mismatches, non-positive ``base_radius``, or bad radii.
    """
    base = require_finite_scalar("base_radius", base_radius)
    if base <= 0.0:
        raise ValueError("base_radius must be positive")

    robot = require_finite_array("robot_positions", robot_positions)
    if robot.ndim != 2 or robot.shape[-1] != 2:
        raise ValueError("robot_positions must have shape (T, 2)")
    num_steps = robot.shape[0]

    peds = require_finite_array("pedestrian_positions", pedestrian_positions)
    preds = require_finite_array("predicted_positions", predicted_positions)
    for name, arr in (("pedestrian_positions", peds), ("predicted_positions", preds)):
        if arr.ndim != 3 or arr.shape[0] != num_steps or arr.shape[-1] != 2:
            raise ValueError(f"{name} must have shape (T, P, 2) matching robot_positions")
    if peds.shape[1] != preds.shape[1]:
        raise ValueError("pedestrian_positions and predicted_positions must share P")
    num_peds = peds.shape[1]

    if num_peds == 0:
        return IntrusionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, num_steps=num_steps)

    buffer_radii = _broadcast_radii(conformal_radii, num_steps=num_steps, num_peds=num_peds)

    robot_exp = robot[:, None, :]  # (T, 1, 2) broadcasts over pedestrians
    dist_actual = np.linalg.norm(robot_exp - peds, axis=-1)  # (T, P)
    dist_pred = np.linalg.norm(robot_exp - preds, axis=-1)  # (T, P)
    buffer = base + buffer_radii  # (T, P)

    current_intruded = dist_actual <= base  # (T, P)
    predicted_intruded = dist_pred <= base
    buffer_intruded = dist_pred <= buffer

    # Per-step depth is the deepest penetration into any pedestrian's uncertainty buffer.
    depth = np.clip(buffer - dist_pred, 0.0, None)  # (T, P)
    step_depth = depth.max(axis=1)  # (T,)

    return IntrusionMetrics(
        current_position_intrusion_time_ratio=float(current_intruded.any(axis=1).mean()),
        predicted_trajectory_intrusion_time_ratio=float(predicted_intruded.any(axis=1).mean()),
        uncertainty_buffer_intrusion_time_ratio=float(buffer_intruded.any(axis=1).mean()),
        cumulative_intrusion_depth=float(step_depth.sum()),
        max_intrusion_depth=float(step_depth.max()),
        num_steps=num_steps,
    )
