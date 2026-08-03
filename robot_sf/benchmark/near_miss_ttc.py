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

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Reuse the canonical pedestrian-velocity primitive so this diagnostic does not
# fork the finite-difference convention used by the benchmark metrics.
from robot_sf.benchmark.metrics import _compute_ped_velocities

if TYPE_CHECKING:
    from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.errors import RobotSfError

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


class NearMissTtcInputError(RobotSfError, RuntimeError):
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


@dataclass(frozen=True)
class NearMissTtcDecisionPacket:
    """Read-only TTC near-miss diagnostic decision packet.

    The packet summarizes whether the existing opt-in diagnostic can be
    evaluated for a trajectory and what the result can and cannot support. It
    intentionally carries no threshold recommendation and does not replace the
    canonical distance-based near-miss metric.
    """

    issue: str
    evidence_status: str
    diagnostic_status: str
    available_inputs: tuple[str, ...]
    unsupported_cases: tuple[str, ...]
    cannot_claim: tuple[str, ...]
    readiness: NearMissTtcReadiness
    diagnostic: dict[str, float | str] = field(default_factory=dict)
    threshold_s: float = DIAGNOSTIC_TTC_THRESHOLD_S

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe packet representation."""
        return {
            "issue": self.issue,
            "evidence_status": self.evidence_status,
            "diagnostic_status": self.diagnostic_status,
            "available_inputs": list(self.available_inputs),
            "unsupported_cases": list(self.unsupported_cases),
            "cannot_claim": list(self.cannot_claim),
            "readiness": _json_safe_value(asdict(self.readiness)),
            "diagnostic": _json_safe_value(dict(self.diagnostic)),
            "threshold_s": _json_safe_value(self.threshold_s),
        }


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


def build_ttc_near_miss_decision_packet(
    data: EpisodeData,
    *,
    t_thr: float = DIAGNOSTIC_TTC_THRESHOLD_S,
    issue: str = "#3808",
) -> NearMissTtcDecisionPacket:
    """Build a read-only TTC near-miss diagnostic decision packet.

    The packet consumes the issue #3700 diagnostic surface without mutating the
    trajectory or canonical benchmark metrics. Unsupported inputs are reported
    fail-closed instead of converted to a zero near-miss result.

    Returns
    -------
    NearMissTtcDecisionPacket
        Structured diagnostic-only packet for review handoff.
    """

    readiness = near_miss_ttc_input_readiness(data)
    available_inputs = _describe_available_ttc_inputs(readiness)
    cannot_claim = (
        "no canonical near-miss metric replacement",
        "no calibrated TTC threshold or severity weighting choice",
        "no planner comparison, benchmark ranking, or paper/dissertation claim",
    )

    if not readiness.ready:
        return NearMissTtcDecisionPacket(
            issue=issue,
            evidence_status="diagnostic-only",
            diagnostic_status="unsupported-inputs",
            available_inputs=available_inputs,
            unsupported_cases=tuple(readiness.reasons),
            cannot_claim=cannot_claim,
            readiness=readiness,
            threshold_s=t_thr,
        )

    diagnostic = compute_ttc_near_miss_diagnostic(data, t_thr=t_thr)
    status = str(diagnostic["near_miss_ttc__status"])
    unsupported_cases = _describe_unsupported_ttc_cases(status)

    return NearMissTtcDecisionPacket(
        issue=issue,
        evidence_status="diagnostic-only",
        diagnostic_status=status,
        available_inputs=available_inputs,
        unsupported_cases=unsupported_cases,
        cannot_claim=cannot_claim,
        readiness=readiness,
        diagnostic=diagnostic,
        threshold_s=float(diagnostic["near_miss_ttc__threshold_s"]),
    )


def render_ttc_near_miss_decision_packet_markdown(packet: NearMissTtcDecisionPacket) -> str:
    """Render a compact Markdown decision packet for review or issue handoff.

    Returns
    -------
    str
        Markdown packet text.
    """

    lines = [
        "# TTC Near-Miss Diagnostic Decision Packet",
        "",
        f"- Issue: `{packet.issue}`",
        f"- Evidence status: `{packet.evidence_status}`",
        f"- Diagnostic status: `{packet.diagnostic_status}`",
        f"- Threshold inspected: `{packet.threshold_s}` seconds",
        "",
        "## Available Diagnostic Inputs",
        *_bullet_lines(packet.available_inputs),
        "",
        "## Unsupported Cases",
        *_bullet_lines(packet.unsupported_cases),
        "",
        "## Cannot Claim Before Canonical Metric Change",
        *_bullet_lines(packet.cannot_claim),
    ]
    if packet.diagnostic:
        lines.extend(["", "## Diagnostic Values"])
        lines.extend(
            f"- `{key}`: `{value}`"
            for key, value in sorted(packet.diagnostic.items(), key=lambda item: item[0])
        )
    return "\n".join(lines) + "\n"


def _describe_available_ttc_inputs(readiness: NearMissTtcReadiness) -> tuple[str, ...]:
    """Describe the timing and trajectory inputs visible to the packet.

    Returns
    -------
    tuple[str, ...]
        Human-readable input availability lines.
    """

    inputs = [
        f"dt inspected: {readiness.dt}",
        f"trajectory frames inspected: {readiness.n_steps}",
        f"pedestrians inspected: {readiness.n_peds}",
    ]
    if readiness.ready:
        inputs.append("robot position, robot velocity, and pedestrian position arrays are usable")
    return tuple(inputs)


def _describe_unsupported_ttc_cases(status: str) -> tuple[str, ...]:
    """Describe cases this diagnostic cannot support as TTC near-miss evidence.

    Returns
    -------
    tuple[str, ...]
        Human-readable unsupported-case lines.
    """

    cases = [
        "static distance-only proximity remains covered only by canonical near_misses",
        "trajectory-free aggregate metrics cannot be reinterpreted as TTC evidence",
    ]
    if status == "no-pedestrians":
        cases.append("no pedestrian pairs were available for TTC evaluation")
    elif status == "no-approaching-pairs":
        cases.append("opening or non-converging pairs do not support TTC near-miss counts")
    return tuple(cases)


def _bullet_lines(items: tuple[str, ...]) -> list[str]:
    """Format tuple content as Markdown bullets.

    Returns
    -------
    list[str]
        Markdown bullet lines.
    """

    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


@dataclass(frozen=True)
class NearMissThresholdProfile:
    """Versioned threshold profile identity for encounter-level near-miss aggregation.

    Attributes
    ----------
    profile_id : str
        Explicit threshold-profile identity.
    mode : str
        Qualification mode: 'distance', 'ttc', or 'combined'.
    distance_threshold_m : float | None
        Clearance threshold in meters for distance qualification.
    ttc_threshold_s : float | None
        TTC threshold in seconds for TTC qualification.
    contact_threshold_m : float
        Surface clearance below which body contact occurs (default 0.0 m).
    max_gap_steps : int
        Maximum allowed non-qualifying gap steps before starting a new encounter (default 0).
    clearance_type : str
        Clearance definition ('surface' or 'center', default 'surface').
    units : dict[str, str]
        Explicit unit mapping for reported metrics.
    """

    profile_id: str
    mode: str = "ttc"
    distance_threshold_m: float | None = None
    ttc_threshold_s: float | None = None
    contact_threshold_m: float = 0.0
    max_gap_steps: int = 0
    clearance_type: str = "surface"
    units: dict[str, str] = field(
        default_factory=lambda: {
            "time": "s",
            "distance": "m",
            "speed": "m/s",
            "clearance": "m",
        }
    )

    def __post_init__(self) -> None:
        """Validate threshold profile properties."""
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if self.mode not in ("distance", "ttc", "combined"):
            raise ValueError(
                f"mode must be one of ('distance', 'ttc', 'combined'), got {self.mode!r}"
            )
        if self.mode in ("distance", "combined"):
            if (
                self.distance_threshold_m is None
                or not np.isfinite(self.distance_threshold_m)
                or self.distance_threshold_m <= 0.0
            ):
                raise ValueError(
                    f"distance_threshold_m must be finite and positive for mode {self.mode!r}"
                )
        if self.mode in ("ttc", "combined"):
            if (
                self.ttc_threshold_s is None
                or not np.isfinite(self.ttc_threshold_s)
                or self.ttc_threshold_s <= 0.0
            ):
                raise ValueError(
                    f"ttc_threshold_s must be finite and positive for mode {self.mode!r}"
                )
        if not np.isfinite(self.contact_threshold_m):
            raise ValueError("contact_threshold_m must be finite")
        if not isinstance(self.max_gap_steps, int) or self.max_gap_steps < 0:
            raise ValueError("max_gap_steps must be a non-negative integer")

    @classmethod
    def ttc_v1(
        cls,
        ttc_threshold_s: float = DIAGNOSTIC_TTC_THRESHOLD_S,
        max_gap_steps: int = 0,
    ) -> NearMissThresholdProfile:
        """Factory for versioned TTC threshold profile.

        Returns
        -------
        NearMissThresholdProfile
            Threshold profile configured for TTC near-miss criteria.
        """
        return cls(
            profile_id="ttc_diagnostic_v1",
            mode="ttc",
            ttc_threshold_s=ttc_threshold_s,
            max_gap_steps=max_gap_steps,
        )

    @classmethod
    def distance_v1(
        cls,
        distance_threshold_m: float = 1.0,
        max_gap_steps: int = 0,
    ) -> NearMissThresholdProfile:
        """Factory for versioned distance threshold profile.

        Returns
        -------
        NearMissThresholdProfile
            Threshold profile configured for distance near-miss criteria.
        """
        return cls(
            profile_id="distance_d_near_v1",
            mode="distance",
            distance_threshold_m=distance_threshold_m,
            max_gap_steps=max_gap_steps,
        )

    @classmethod
    def combined_v1(
        cls,
        distance_threshold_m: float = 1.0,
        ttc_threshold_s: float = DIAGNOSTIC_TTC_THRESHOLD_S,
        max_gap_steps: int = 0,
    ) -> NearMissThresholdProfile:
        """Factory for versioned combined threshold profile.

        Returns
        -------
        NearMissThresholdProfile
            Threshold profile configured for combined distance and TTC criteria.
        """
        return cls(
            profile_id="combined_diagnostic_v1",
            mode="combined",
            distance_threshold_m=distance_threshold_m,
            ttc_threshold_s=ttc_threshold_s,
            max_gap_steps=max_gap_steps,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return _json_safe_value(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class NearMissEncounterRecord:
    """Versioned encounter record for a single actor's near-miss episode segment.

    Attributes
    ----------
    encounter_id : str
        Unique identifier for this encounter (e.g. 'enc_actor0_0').
    actor_id : int
        Pedestrian/actor index (0-indexed).
    start_step : int
        Starting timestep index.
    end_step : int
        Ending timestep index (inclusive).
    start_time_s : float
        Start time in seconds (start_step * dt).
    end_time_s : float
        End time in seconds (end_step * dt).
    duration_s : float
        Duration of the encounter in seconds.
    duration_steps : int
        Duration in number of timesteps.
    min_clearance_m : float
        Minimum surface clearance during the encounter in meters.
    min_ttc_s : float
        Minimum projected TTC during the encounter in seconds (NaN if unavailable).
    max_closing_speed_mps : float
        Maximum closing speed during the encounter in m/s (NaN if unavailable).
    pet_s : float
        Post-encounter / post-encroachment time in seconds (NaN if unavailable).
    contact_terminated : bool
        True if the encounter was terminated by body contact/collision.
    exposure_duration_s : float
        Valid exposure duration for the actor in seconds.
    threshold_profile_id : str
        Identifier of the threshold profile used for aggregation.
    units : dict[str, str]
        Field units dictionary.
    """

    encounter_id: str
    actor_id: int
    start_step: int
    end_step: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    duration_steps: int
    min_clearance_m: float
    min_ttc_s: float
    max_closing_speed_mps: float
    pet_s: float
    contact_terminated: bool
    exposure_duration_s: float
    threshold_profile_id: str
    units: dict[str, str] = field(
        default_factory=lambda: {
            "start_time_s": "s",
            "end_time_s": "s",
            "duration_s": "s",
            "min_clearance_m": "m",
            "min_ttc_s": "s",
            "max_closing_speed_mps": "m/s",
            "pet_s": "s",
            "exposure_duration_s": "s",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return _json_safe_value(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class NearMissEncounterSummary:
    """Summary diagnostic packet for encounter-level near-miss aggregation.

    Attributes
    ----------
    evidence_status : str
        Always 'diagnostic-only' per domain approval.
    status : str
        Diagnostic status ('ok', 'no-pedestrians', 'no-encounters', 'unsupported-inputs').
    total_encounters : int
        Total number of encounters detected across all actors.
    encounters_by_actor : dict[int, int]
        Number of encounters per actor index.
    records : tuple[NearMissEncounterRecord, ...]
        Tuple of encounter records.
    total_encounter_duration_s : float
        Sum of durations of all encounters in seconds.
    min_encounter_clearance_m : float
        Global minimum clearance across all encounters (NaN if no encounters).
    min_encounter_ttc_s : float
        Global minimum TTC across all encounters (NaN if no encounters/approaching pairs).
    max_encounter_closing_speed_mps : float
        Global maximum closing speed across all encounters (NaN if no encounters).
    contact_terminated_encounters : int
        Count of encounters terminated by contact.
    threshold_profile : NearMissThresholdProfile
        The threshold profile used.
    readiness : NearMissTtcReadiness
        Input readiness report.
    denominators : dict[str, float | int]
        Context denominators (e.g. n_steps, n_peds, total_exposure_s).
    unsupported_reasons : tuple[str, ...]
        Reasons if status is unsupported-inputs.
    """

    evidence_status: str
    status: str
    total_encounters: int
    encounters_by_actor: dict[int, int]
    records: tuple[NearMissEncounterRecord, ...]
    total_encounter_duration_s: float
    min_encounter_clearance_m: float
    min_encounter_ttc_s: float
    max_encounter_closing_speed_mps: float
    contact_terminated_encounters: int
    threshold_profile: NearMissThresholdProfile
    readiness: NearMissTtcReadiness
    denominators: dict[str, float | int]
    unsupported_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return _json_safe_value(asdict(self))  # type: ignore[return-value]


def _is_step_qualifying(
    *,
    clearance: float,
    ttc_val: float,
    profile: NearMissThresholdProfile,
) -> bool:
    """Check if a single timestep qualifies under the threshold profile.

    Returns
    -------
    bool
        True if qualifying, False otherwise.
    """
    if profile.mode == "distance":
        assert profile.distance_threshold_m is not None
        return clearance < profile.distance_threshold_m
    if profile.mode == "ttc":
        assert profile.ttc_threshold_s is not None
        return np.isfinite(ttc_val) and (ttc_val < profile.ttc_threshold_s)
    if profile.mode == "combined":
        dist_qual = (
            profile.distance_threshold_m is not None and clearance < profile.distance_threshold_m
        )
        ttc_qual = (
            profile.ttc_threshold_s is not None
            and np.isfinite(ttc_val)
            and ttc_val < profile.ttc_threshold_s
        )
        return dist_qual or ttc_qual
    return False


def _compute_trajectory_matrices(
    data: EpisodeData,
    readiness: NearMissTtcReadiness,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-step clearance, TTC, and closing speed matrices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Clearances (T, K), ttc_matrix (T, K), closing_speed_matrix (T, K).
    """
    n_steps = readiness.n_steps
    n_peds = readiness.n_peds
    dt = readiness.dt

    peds_pos = np.asarray(data.peds_pos, dtype=float)
    robot_pos = np.asarray(data.robot_pos, dtype=float)
    robot_vel = np.asarray(data.robot_vel, dtype=float)
    robot_radius = float(getattr(data, "robot_radius", 1.0))
    ped_radius = float(getattr(data, "ped_radius", 0.4))

    diffs = peds_pos - robot_pos[:, None, :]  # (T, K, 2)
    center_dists = np.linalg.norm(diffs, axis=2)  # (T, K)
    clearances = center_dists - (robot_radius + ped_radius)  # (T, K)

    ttc_matrix = np.full((n_steps, n_peds), np.inf)
    closing_speed_matrix = np.full((n_steps, n_peds), np.nan)

    ped_vels = _compute_ped_velocities(peds_pos, dt)  # (T-1, K, 2)
    if ped_vels.shape[0] > 0:
        robot_vel_aligned = robot_vel[1:]
        robot_pos_aligned = robot_pos[1:]
        peds_pos_aligned = peds_pos[1:]

        v_rel = robot_vel_aligned[:, None, :] - ped_vels
        d_vec = peds_pos_aligned - robot_pos_aligned[:, None, :]
        dot_product = np.einsum("ijk,ijk->ij", v_rel, d_vec)
        v_rel_mag = np.linalg.norm(v_rel, axis=2)
        d_mag = np.linalg.norm(d_vec, axis=2)

        approaching = dot_product > 0.0
        valid_ttc = approaching & (v_rel_mag > _MIN_RELATIVE_SPEED)

        ttc_sub = np.full_like(d_mag, np.inf)
        ttc_sub[valid_ttc] = d_mag[valid_ttc] / v_rel_mag[valid_ttc]
        ttc_matrix[1:] = ttc_sub

        closing_sub = np.full_like(d_mag, np.nan)
        speed_defined = valid_ttc & (d_mag > _MIN_APPROACH_DISTANCE)
        closing_sub[speed_defined] = dot_product[speed_defined] / d_mag[speed_defined]
        closing_speed_matrix[1:] = closing_sub

    return clearances, ttc_matrix, closing_speed_matrix


def _segment_actor_encounters(
    *,
    actor_id: int,
    n_steps: int,
    dt: float,
    clearances: np.ndarray,
    ttc_matrix: np.ndarray,
    closing_speed_matrix: np.ndarray,
    profile: NearMissThresholdProfile,
) -> list[NearMissEncounterRecord]:
    """Segment encounters for a single actor.

    Returns
    -------
    list[NearMissEncounterRecord]
        List of encounter records for actor_id.
    """
    records: list[NearMissEncounterRecord] = []
    encounter_idx = 0
    current_start: int | None = None
    last_qualifying: int | None = None
    gap_count = 0

    for t in range(n_steps):
        is_contact = clearances[t, actor_id] < profile.contact_threshold_m
        is_qualifying = (
            False
            if is_contact
            else _is_step_qualifying(
                clearance=clearances[t, actor_id],
                ttc_val=ttc_matrix[t, actor_id],
                profile=profile,
            )
        )

        if is_contact:
            start_step = current_start if current_start is not None else t
            records.append(
                _build_encounter_record(
                    actor_id=actor_id,
                    encounter_index=encounter_idx,
                    start_step=start_step,
                    end_step=t,
                    clearances=clearances[:, actor_id],
                    ttc_matrix=ttc_matrix[:, actor_id],
                    closing_speed_matrix=closing_speed_matrix[:, actor_id],
                    dt=dt,
                    contact_terminated=True,
                    profile_id=profile.profile_id,
                )
            )
            encounter_idx += 1
            current_start = None
            last_qualifying = None
            gap_count = 0
        elif is_qualifying:
            if current_start is None:
                current_start = t
            last_qualifying = t
            gap_count = 0
        elif current_start is not None:
            gap_count += 1
            if gap_count > profile.max_gap_steps:
                assert last_qualifying is not None
                records.append(
                    _build_encounter_record(
                        actor_id=actor_id,
                        encounter_index=encounter_idx,
                        start_step=current_start,
                        end_step=last_qualifying,
                        clearances=clearances[:, actor_id],
                        ttc_matrix=ttc_matrix[:, actor_id],
                        closing_speed_matrix=closing_speed_matrix[:, actor_id],
                        dt=dt,
                        contact_terminated=False,
                        profile_id=profile.profile_id,
                    )
                )
                encounter_idx += 1
                current_start = None
                last_qualifying = None
                gap_count = 0

    if current_start is not None and last_qualifying is not None:
        records.append(
            _build_encounter_record(
                actor_id=actor_id,
                encounter_index=encounter_idx,
                start_step=current_start,
                end_step=last_qualifying,
                clearances=clearances[:, actor_id],
                ttc_matrix=ttc_matrix[:, actor_id],
                closing_speed_matrix=closing_speed_matrix[:, actor_id],
                dt=dt,
                contact_terminated=False,
                profile_id=profile.profile_id,
            )
        )

    return records


def compute_near_miss_encounters(
    data: EpisodeData,
    profile: NearMissThresholdProfile | None = None,
) -> NearMissEncounterSummary:
    """Aggregate qualifying contiguous timesteps into versioned encounter records per actor.

    This implements encounter-level aggregation per issue #6709 domain-aware approval.
    It groups qualifying per-actor trace samples into versioned encounter records and
    reports their observed duration, minimum clearance, available TTC/closing-speed,
    contact termination, and provenance.

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container (positions, robot velocity, dt).
    profile : NearMissThresholdProfile, optional
        Explicit threshold profile. Defaults to `NearMissThresholdProfile.ttc_v1()`.

    Returns
    -------
    NearMissEncounterSummary
        Encounter aggregation summary packet.
    """
    if profile is None:
        profile = NearMissThresholdProfile.ttc_v1()

    readiness = near_miss_ttc_input_readiness(data)
    if not readiness.ready:
        return NearMissEncounterSummary(
            evidence_status="diagnostic-only",
            status="unsupported-inputs",
            total_encounters=0,
            encounters_by_actor={},
            records=(),
            total_encounter_duration_s=0.0,
            min_encounter_clearance_m=float("nan"),
            min_encounter_ttc_s=float("nan"),
            max_encounter_closing_speed_mps=float("nan"),
            contact_terminated_encounters=0,
            threshold_profile=profile,
            readiness=readiness,
            denominators={
                "n_steps": readiness.n_steps,
                "n_peds": readiness.n_peds,
                "total_exposure_s": 0.0,
            },
            unsupported_reasons=readiness.reasons,
        )

    n_steps = readiness.n_steps
    n_peds = readiness.n_peds
    dt = readiness.dt

    if n_peds == 0:
        return NearMissEncounterSummary(
            evidence_status="diagnostic-only",
            status="no-pedestrians",
            total_encounters=0,
            encounters_by_actor={},
            records=(),
            total_encounter_duration_s=0.0,
            min_encounter_clearance_m=float("nan"),
            min_encounter_ttc_s=float("nan"),
            max_encounter_closing_speed_mps=float("nan"),
            contact_terminated_encounters=0,
            threshold_profile=profile,
            readiness=readiness,
            denominators={
                "n_steps": n_steps,
                "n_peds": 0,
                "total_exposure_s": 0.0,
            },
        )

    clearances, ttc_matrix, closing_speed_matrix = _compute_trajectory_matrices(data, readiness)

    all_records: list[NearMissEncounterRecord] = []
    encounters_by_actor: dict[int, int] = dict.fromkeys(range(n_peds), 0)

    for k in range(n_peds):
        actor_records = _segment_actor_encounters(
            actor_id=k,
            n_steps=n_steps,
            dt=dt,
            clearances=clearances,
            ttc_matrix=ttc_matrix,
            closing_speed_matrix=closing_speed_matrix,
            profile=profile,
        )
        all_records.extend(actor_records)
        encounters_by_actor[k] = len(actor_records)

    records_tuple = tuple(all_records)
    total_encounters = len(records_tuple)
    total_duration = sum(r.duration_s for r in records_tuple)

    if total_encounters > 0:
        min_clearance = min(r.min_clearance_m for r in records_tuple)
        finite_ttcs = [r.min_ttc_s for r in records_tuple if np.isfinite(r.min_ttc_s)]
        min_ttc = min(finite_ttcs) if finite_ttcs else float("nan")
        finite_closings = [
            r.max_closing_speed_mps for r in records_tuple if np.isfinite(r.max_closing_speed_mps)
        ]
        max_closing = max(finite_closings) if finite_closings else float("nan")
        contact_terminated_count = sum(1 for r in records_tuple if r.contact_terminated)
        status = "ok"
    else:
        min_clearance = float("nan")
        min_ttc = float("nan")
        max_closing = float("nan")
        contact_terminated_count = 0
        status = "no-encounters"

    return NearMissEncounterSummary(
        evidence_status="diagnostic-only",
        status=status,
        total_encounters=total_encounters,
        encounters_by_actor=encounters_by_actor,
        records=records_tuple,
        total_encounter_duration_s=total_duration,
        min_encounter_clearance_m=min_clearance,
        min_encounter_ttc_s=min_ttc,
        max_encounter_closing_speed_mps=max_closing,
        contact_terminated_encounters=contact_terminated_count,
        threshold_profile=profile,
        readiness=readiness,
        denominators={
            "n_steps": n_steps,
            "n_peds": n_peds,
            "total_exposure_s": float(n_steps * dt * n_peds),
        },
    )


def _build_encounter_record(  # noqa: PLR0913
    *,
    actor_id: int,
    encounter_index: int,
    start_step: int,
    end_step: int,
    clearances: np.ndarray,
    ttc_matrix: np.ndarray,
    closing_speed_matrix: np.ndarray,
    dt: float,
    contact_terminated: bool,
    profile_id: str,
) -> NearMissEncounterRecord:
    duration_steps = end_step - start_step + 1
    duration_s = float(duration_steps * dt)
    start_time_s = float(start_step * dt)
    end_time_s = float(end_step * dt)

    c_slice = clearances[start_step : end_step + 1]
    min_clearance = float(np.min(c_slice)) if c_slice.size > 0 else float("nan")

    ttc_slice = ttc_matrix[start_step : end_step + 1]
    finite_ttc = ttc_slice[np.isfinite(ttc_slice)]
    min_ttc = float(np.min(finite_ttc)) if finite_ttc.size > 0 else float("nan")

    closing_slice = closing_speed_matrix[start_step : end_step + 1]
    finite_closing = closing_slice[np.isfinite(closing_slice)]
    max_closing = float(np.max(finite_closing)) if finite_closing.size > 0 else float("nan")

    return NearMissEncounterRecord(
        encounter_id=f"enc_actor{actor_id}_{encounter_index}",
        actor_id=actor_id,
        start_step=start_step,
        end_step=end_step,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        duration_s=duration_s,
        duration_steps=duration_steps,
        min_clearance_m=min_clearance,
        min_ttc_s=min_ttc,
        max_closing_speed_mps=max_closing,
        pet_s=float("nan"),
        contact_terminated=contact_terminated,
        exposure_duration_s=duration_s,
        threshold_profile_id=profile_id,
    )


def build_near_miss_encounter_decision_packet(
    data: EpisodeData,
    *,
    profile: NearMissThresholdProfile | None = None,
    issue: str = "#6709",
) -> dict[str, Any]:
    """Build a read-only encounter-level near-miss diagnostic decision packet.

    Returns
    -------
    dict[str, Any]
        Structured packet dict for review.
    """
    if profile is None:
        profile = NearMissThresholdProfile.ttc_v1()
    summary = compute_near_miss_encounters(data, profile=profile)
    return {
        "issue": issue,
        "evidence_status": summary.evidence_status,
        "diagnostic_status": summary.status,
        "threshold_profile": summary.threshold_profile.to_dict(),
        "summary": summary.to_dict(),
        "cannot_claim": (
            "no canonical near-miss metric replacement",
            "no calibrated encounter severity weighting",
            "no planner comparison, benchmark ranking, or paper/dissertation claim",
        ),
    }


def _json_safe_value(value: object) -> object:
    """Convert packet values to strict JSON-compatible primitives.

    Returns
    -------
    object
        Value with non-finite floats replaced by ``None``.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        return float_value if np.isfinite(float_value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


__all__ = [
    "DIAGNOSTIC_TTC_THRESHOLD_S",
    "NearMissEncounterRecord",
    "NearMissEncounterSummary",
    "NearMissThresholdProfile",
    "NearMissTtcDecisionPacket",
    "NearMissTtcInputError",
    "NearMissTtcReadiness",
    "build_near_miss_encounter_decision_packet",
    "build_ttc_near_miss_decision_packet",
    "compute_near_miss_encounters",
    "compute_ttc_near_miss_diagnostic",
    "near_miss_ttc_input_readiness",
    "render_ttc_near_miss_decision_packet_markdown",
]
