"""Opt-in diagnostic surface for a closing-speed / time-to-collision (TTC) aware near-miss.

Background
----------
The canonical near-miss metric in :mod:`robot_sf.benchmark.metrics` is purely
distance-based: it counts steps whose minimum robot-pedestrian clearance falls
in ``[0, D_NEAR)``. That static proximity test treats a slow drift to ``D_NEAR``
the same as a fast head-on approach that decelerates near ``D_NEAR``, and it
never registers converging-but-not-yet-close encounters with a high closing
speed / small TTC -- arguably the more dangerous ones. GitHub issue #3700
proposes a closing-speed / TTC-aware variant alongside the distance metric.

Status: diagnostic-only, opt-in, additive
-----------------------------------------
This module stages an *opt-in, additive, diagnostic* surface only. It does
**not**:

- replace or modify the canonical distance-based ``near_misses`` metric,
- wire anything into SNQI or any scoring / ranking path,
- assert a calibrated threshold or any safety result.

The TTC threshold exposed here is an explicit, **uncalibrated diagnostic
placeholder** (the choice of ``t_thr`` and the TTC-count vs. severity-weighting
variant is ``decision-required`` per issue #3700). Treat outputs as
diagnostic-only, never as benchmark evidence.

Fail-closed input contract
--------------------------
A TTC-aware near-miss needs per-step relative *positions and velocities*, which
in turn require a valid timestep ``dt`` and at least two frames (pedestrian
velocities are derived by finite difference). :func:`near_miss_ttc_input_readiness`
validates those timing/velocity inputs and reports, fail-closed, exactly which
requirement is missing. :func:`compute_ttc_near_miss_diagnostic` refuses to emit
numbers (raises :class:`NearMissTtcInputError`) when the inputs are not ready,
rather than silently returning zeros that would read as "no near misses".

TTC convention
--------------
The closing geometry mirrors the existing :func:`robot_sf.benchmark.metrics.time_to_collision_min`
metric so this diagnostic stays consistent with the repository's TTC definition:

- relative velocity ``v_rel = v_robot - v_ped``,
- a pair is *approaching* when ``dot(v_rel, d_vec) > 0`` with ``d_vec`` pointing
  from the robot to the pedestrian (centre-to-centre distance decreasing),
- ``TTC = ||d_vec|| / ||v_rel||`` for approaching pairs, ``+inf`` otherwise,
- closing speed is the component of ``v_rel`` along the line of approach,
  ``dot(v_rel, d_vec) / ||d_vec||`` (positive when approaching).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# Reuse the canonical pedestrian-velocity primitive so this diagnostic does not
# fork the finite-difference convention used by the benchmark metrics.
from robot_sf.benchmark.metrics import _compute_ped_velocities

if TYPE_CHECKING:
    from robot_sf.benchmark.metrics import EpisodeData

# Uncalibrated diagnostic placeholder (decision-required, issue #3700). This is
# NOT a benchmark-calibrated threshold; it only gives the diagnostic a concrete
# default so the surface can be inspected. Callers should pass an explicit
# ``t_thr`` once a calibrated value is chosen.
DIAGNOSTIC_TTC_THRESHOLD_S: float = 2.0

# Minimum relative speed (m/s) treated as "moving" when projecting closing speed
# / computing TTC. Matches the epsilon used by ``time_to_collision_min``.
_MIN_RELATIVE_SPEED: float = 1e-9

# Minimum centre-to-centre distance (m) below which the line-of-approach
# direction is numerically undefined; closing speed is reported as 0.0 there.
_MIN_APPROACH_DISTANCE: float = 1e-9


class NearMissTtcInputError(RuntimeError):
    """Raised when TTC near-miss inputs fail the fail-closed readiness contract.

    Carries the structured readiness report so callers can surface exactly which
    timing/velocity requirement was missing instead of a bare message.
    """

    def __init__(self, message: str, *, readiness: NearMissTtcReadiness | None = None) -> None:
        """Store the actionable message plus the structured readiness report."""
        super().__init__(message)
        self.readiness = readiness


@dataclass(frozen=True)
class NearMissTtcReadiness:
    """Structured fail-closed readiness report for TTC near-miss inputs.

    Attributes
    ----------
    ready : bool
        True only when every timing/velocity input required to compute a
        TTC-aware near-miss is present and well-formed.
    reasons : tuple[str, ...]
        Human-readable reasons the inputs are not ready; empty when ``ready``.
    n_steps : int
        Number of trajectory frames (``T``) detected, or 0 when undetectable.
    n_peds : int
        Number of pedestrians (``K``) detected, or 0 when undetectable.
    dt : float
        Timestep value inspected (may be non-finite/non-positive when invalid).
    """

    ready: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    n_steps: int = 0
    n_peds: int = 0
    dt: float = float("nan")


def _as_float_array(value: object) -> np.ndarray | None:
    """Return ``value`` as a float ndarray, or ``None`` when it cannot be coerced."""
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _check_dt(data: EpisodeData, reasons: list[str]) -> float:
    """Validate the ``dt`` timing field, appending reasons; return the value seen.

    Returns:
        The ``dt`` value inspected, or NaN when it is missing/not a number.
    """
    try:
        dt = float(data.dt)
    except (TypeError, ValueError):
        reasons.append("dt is missing or not a number")
        return float("nan")
    if not np.isfinite(dt):
        reasons.append(f"dt must be finite (got {dt!r})")
    elif dt <= 0.0:
        reasons.append(f"dt must be strictly positive (got {dt!r})")
    return dt


def _check_robot_pos(robot_pos: np.ndarray | None, reasons: list[str]) -> int:
    """Validate ``robot_pos`` shape/length, appending reasons; return frame count.

    Returns:
        Detected number of frames ``T``, or 0 when the shape is invalid.
    """
    if robot_pos is None or robot_pos.ndim != 2 or robot_pos.shape[1] != 2:
        shape = None if robot_pos is None else robot_pos.shape
        reasons.append(f"robot_pos must be a (T, 2) array (got shape {shape})")
        return 0
    n_steps = int(robot_pos.shape[0])
    if n_steps < 2:
        reasons.append(f"robot_pos needs >= 2 frames to derive velocities (got T={n_steps})")
    return n_steps


def _check_robot_vel(
    robot_vel: np.ndarray | None, robot_pos: np.ndarray | None, reasons: list[str]
) -> None:
    """Validate ``robot_vel`` shape and frame consistency with ``robot_pos``."""
    if robot_vel is None or robot_vel.ndim != 2 or robot_vel.shape[1] != 2:
        shape = None if robot_vel is None else robot_vel.shape
        reasons.append(f"robot_vel must be a (T, 2) array (got shape {shape})")
        return
    if robot_pos is not None and robot_pos.ndim == 2 and robot_vel.shape[0] != robot_pos.shape[0]:
        reasons.append(
            "robot_vel frame count must match robot_pos "
            f"(robot_vel T={robot_vel.shape[0]} vs robot_pos T={robot_pos.shape[0]})"
        )


def _check_peds_pos(peds_pos: np.ndarray | None, n_steps: int, reasons: list[str]) -> int:
    """Validate ``peds_pos`` shape and frame consistency; return pedestrian count.

    Returns:
        Detected number of pedestrians ``K``, or 0 when the shape is invalid.
    """
    if peds_pos is None or peds_pos.ndim != 3 or peds_pos.shape[2] != 2:
        shape = None if peds_pos is None else peds_pos.shape
        reasons.append(f"peds_pos must be a (T, K, 2) array (got shape {shape})")
        return 0
    if n_steps and peds_pos.shape[0] != n_steps:
        reasons.append(
            "peds_pos frame count must match robot_pos "
            f"(peds_pos T={peds_pos.shape[0]} vs robot_pos T={n_steps})"
        )
    return int(peds_pos.shape[1])


def near_miss_ttc_input_readiness(data: EpisodeData) -> NearMissTtcReadiness:
    """Validate, fail-closed, the inputs required for a TTC-aware near-miss.

    The TTC-aware near-miss needs per-step relative positions and velocities.
    This checks the timing/velocity fields that make that derivation valid:

    - ``dt`` is finite and strictly positive (needed to derive pedestrian
      velocity from positions and to interpret TTC in seconds),
    - ``robot_pos`` is a ``(T, 2)`` array with at least two frames,
    - ``robot_vel`` is present and shaped like ``robot_pos`` (``(T, 2)``),
    - ``peds_pos`` is a ``(T, K, 2)`` array whose frame count matches the robot.

    A pedestrian-free episode (``K == 0``) is still *ready*: the inputs are valid
    and the diagnostic simply has no pairs to evaluate. Readiness is about the
    timing/velocity contract, not about whether any near miss occurred.

    Returns
    -------
    NearMissTtcReadiness
        ``ready=True`` with empty ``reasons`` when all inputs are valid;
        otherwise ``ready=False`` listing every failed requirement.
    """
    reasons: list[str] = []

    dt = _check_dt(data, reasons)
    robot_pos = _as_float_array(getattr(data, "robot_pos", None))
    robot_vel = _as_float_array(getattr(data, "robot_vel", None))
    peds_pos = _as_float_array(getattr(data, "peds_pos", None))

    n_steps = _check_robot_pos(robot_pos, reasons)
    _check_robot_vel(robot_vel, robot_pos, reasons)
    n_peds = _check_peds_pos(peds_pos, n_steps, reasons)

    return NearMissTtcReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        n_steps=n_steps,
        n_peds=n_peds,
        dt=dt,
    )


def compute_ttc_near_miss_diagnostic(
    data: EpisodeData,
    *,
    t_thr: float = DIAGNOSTIC_TTC_THRESHOLD_S,
) -> dict[str, float | str]:
    """Compute the opt-in, diagnostic-only TTC-aware near-miss surface.

    This is an *additive diagnostic*: it never touches the canonical
    distance-based ``near_misses`` metric and is not consumed by SNQI or any
    scoring path. It fails closed -- raising :class:`NearMissTtcInputError` --
    when the timing/velocity inputs are not ready, so missing data can never be
    misread as "no near misses".

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container (positions, robot velocity, ``dt``).
    t_thr : float, optional
        TTC threshold in seconds. A step counts as a TTC near-miss when its
        minimum projected TTC over all pedestrians is below ``t_thr``. Defaults
        to the uncalibrated :data:`DIAGNOSTIC_TTC_THRESHOLD_S` placeholder;
        callers should pass an explicit value once calibrated (issue #3700).

    Returns
    -------
    dict
        Diagnostic surface under ``near_miss_ttc__*`` keys:

        - ``near_miss_ttc__status``: ``"ok"``, ``"no-pedestrians"`` or
          ``"no-approaching-pairs"``,
        - ``near_miss_ttc__threshold_s``: the ``t_thr`` used,
        - ``near_miss_ttc__count``: number of steps whose minimum TTC < ``t_thr``,
        - ``near_miss_ttc__min_ttc_s``: smallest projected TTC (NaN if none),
        - ``near_miss_ttc__max_closing_speed_mps``: largest closing speed over
          approaching pairs (NaN if none),
        - ``near_miss_ttc__approaching_steps``: steps with >= 1 approaching pair,
        - ``near_miss_ttc__n_steps``: trajectory frames evaluated.

    Raises
    ------
    NearMissTtcInputError
        If the readiness contract fails, or ``t_thr`` is not finite and positive.
    """
    readiness = near_miss_ttc_input_readiness(data)
    if not readiness.ready:
        raise NearMissTtcInputError(
            "TTC near-miss inputs are not ready: " + "; ".join(readiness.reasons),
            readiness=readiness,
        )

    try:
        t_thr_value = float(t_thr)
    except (TypeError, ValueError) as exc:
        raise NearMissTtcInputError(f"t_thr must be a number (got {t_thr!r})") from exc
    if not np.isfinite(t_thr_value) or t_thr_value <= 0.0:
        raise NearMissTtcInputError(
            f"t_thr must be finite and strictly positive (got {t_thr_value!r})"
        )

    n_steps = readiness.n_steps
    base_result: dict[str, float | str] = {
        "near_miss_ttc__threshold_s": t_thr_value,
        "near_miss_ttc__count": 0.0,
        "near_miss_ttc__min_ttc_s": float("nan"),
        "near_miss_ttc__max_closing_speed_mps": float("nan"),
        "near_miss_ttc__approaching_steps": 0.0,
        "near_miss_ttc__n_steps": float(n_steps),
    }

    # No pedestrians: inputs are valid but there are no pairs to evaluate.
    if readiness.n_peds == 0:
        base_result["near_miss_ttc__status"] = "no-pedestrians"
        return base_result

    peds_pos = np.asarray(data.peds_pos, dtype=float)
    robot_pos = np.asarray(data.robot_pos, dtype=float)
    robot_vel = np.asarray(data.robot_vel, dtype=float)
    dt = readiness.dt

    ped_vels = _compute_ped_velocities(peds_pos, dt)  # (T-1, K, 2)
    if ped_vels.shape[0] == 0:
        base_result["near_miss_ttc__status"] = "no-approaching-pairs"
        return base_result

    # Align robot arrays to the (T-1) finite-difference grid, matching the
    # convention in ``time_to_collision_min``.
    robot_vel_aligned = robot_vel[1:]
    robot_pos_aligned = robot_pos[1:]
    peds_pos_aligned = peds_pos[1:]

    v_rel = robot_vel_aligned[:, None, :] - ped_vels  # (T-1, K, 2)
    d_vec = peds_pos_aligned - robot_pos_aligned[:, None, :]  # robot -> ped

    # dot(v_rel, d_vec) > 0 => centre-to-centre distance decreasing (approaching).
    dot_product = np.einsum("ijk,ijk->ij", v_rel, d_vec)
    v_rel_mag = np.linalg.norm(v_rel, axis=2)
    d_mag = np.linalg.norm(d_vec, axis=2)

    approaching = dot_product > 0.0
    valid = approaching & (v_rel_mag > _MIN_RELATIVE_SPEED)

    # Projected TTC for approaching, moving pairs; +inf elsewhere so a per-step
    # min over pedestrians ignores diverging/static pairs.
    ttc_matrix = np.full_like(d_mag, np.inf)
    ttc_matrix[valid] = d_mag[valid] / v_rel_mag[valid]

    # Closing speed along the line of approach (severity proxy), only where the
    # approach direction is numerically defined.
    closing_speed = np.zeros_like(d_mag)
    speed_defined = valid & (d_mag > _MIN_APPROACH_DISTANCE)
    closing_speed[speed_defined] = dot_product[speed_defined] / d_mag[speed_defined]

    step_min_ttc = ttc_matrix.min(axis=1)  # (T-1,)
    ttc_near_miss_steps = int(np.count_nonzero(step_min_ttc < t_thr_value))
    approaching_steps = int(np.count_nonzero(approaching.any(axis=1)))

    finite_ttc = ttc_matrix[np.isfinite(ttc_matrix)]
    min_ttc = float(finite_ttc.min()) if finite_ttc.size else float("nan")
    max_closing = float(closing_speed.max()) if np.any(speed_defined) else float("nan")

    base_result["near_miss_ttc__count"] = float(ttc_near_miss_steps)
    base_result["near_miss_ttc__min_ttc_s"] = min_ttc
    base_result["near_miss_ttc__max_closing_speed_mps"] = max_closing
    base_result["near_miss_ttc__approaching_steps"] = float(approaching_steps)
    base_result["near_miss_ttc__status"] = "ok" if approaching_steps else "no-approaching-pairs"
    return base_result


__all__ = [
    "DIAGNOSTIC_TTC_THRESHOLD_S",
    "NearMissTtcInputError",
    "NearMissTtcReadiness",
    "compute_ttc_near_miss_diagnostic",
    "near_miss_ttc_input_readiness",
]
