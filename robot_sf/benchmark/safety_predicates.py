"""Trace-level safety-predicate producers (issue #3483).

The thesis motivates several trace-level failure families but cannot report their
rates until in-sim *producers* emit auditable fields. This module implements those
producers as pure, versioned functions over a per-step trajectory, so the fields are
reproducible and fixture-backed instead of simulation-only assumptions.

This module provides three trace-level predicates: **oscillatory local-control**,
**late-evasive**, and **occlusion-triggered-near-miss**. The oscillatory boolean output
populates the existing ``SurrogateEvents.oscillation`` ledger slot; the detailed field
records are intended for the ledger ``surrogate_events`` block. Live emission wiring into
the in-sim runners is a deliberate follow-up.

.. note::

   Predicate definitions are **versioned modeling choices**, labeled diagnostic until
   real-world label validation (externally blocked, #3278). Thresholds are explicit and
   overridable; the producer always emits the raw fields so a different threshold can be
   applied downstream without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.finite_checks import require_finite_array, require_finite_scalar

if TYPE_CHECKING:
    from numpy.typing import NDArray

OSCILLATORY_PREDICATE_SCHEMA = "safety_predicate.oscillatory_control.v1"
LATE_EVASIVE_PREDICATE_SCHEMA = "safety_predicate.late_evasive.v2"
OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA = "safety_predicate.occlusion_near_miss.v1"


def _sign_changes(values: NDArray[np.float64]) -> int:
    """Count sign changes in a 1D sequence, ignoring zeros.

    Returns:
        int: Number of times the (nonzero) sign flips between successive entries.
    """
    signs = np.sign(values)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(nonzero) != 0))


def oscillatory_control_predicate(
    positions: NDArray[np.floating] | object,
    headings: NDArray[np.floating] | object,
    linear_velocities: NDArray[np.floating] | object,
    *,
    dt: float,
    command_sources: list[Any] | None = None,
    min_heading_rate_sign_changes: int = 4,
    max_progress_ratio: float = 0.5,
) -> dict[str, Any]:
    """Produce the oscillatory local-control predicate fields for one episode.

    Args:
        positions: ``(N, 2)`` robot positions (m).
        headings: ``(N,)`` robot headings (radians); unwrapped internally.
        linear_velocities: ``(N,)`` signed forward velocity (m/s).
        dt: Per-step timestep (s, > 0).
        command_sources: Optional ``(N,)`` per-step command-source labels. A ``None``
            label marks unavailable evidence for that step; transitions touching it
            are not counted.
        min_heading_rate_sign_changes: Classification threshold on heading-rate flips.
        max_progress_ratio: Classification threshold on net/path progress ratio.

    Returns:
        dict[str, Any]: Versioned record with the motivated fields and the diagnostic
        ``oscillation`` boolean (compatible with ``SurrogateEvents.oscillation``).
    """
    if not (dt > 0.0):
        raise ValueError(f"dt must be > 0, got {dt!r}")
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    head = np.asarray(headings, dtype=np.float64).reshape(-1)
    vel = np.asarray(linear_velocities, dtype=np.float64).reshape(-1)
    n = pos.shape[0]
    if not (head.shape[0] == n == vel.shape[0]):
        raise ValueError("positions, headings, and linear_velocities must share length N")
    if command_sources is not None and len(command_sources) != n:
        raise ValueError("command_sources must have length N when provided")
    if n < 2:
        raise ValueError("at least two steps are required")
    require_finite_array("positions", pos)
    require_finite_array("headings", head)
    require_finite_array("linear_velocities", vel)

    heading_rate = np.diff(np.unwrap(head)) / dt
    heading_rate_sign_changes = _sign_changes(heading_rate)
    linear_velocity_sign_changes = _sign_changes(vel)

    command_source_changes = 0
    if command_sources is not None:
        command_source_changes = int(
            sum(
                1
                for a, b in pairwise(command_sources)
                if a is not None and b is not None and a != b
            )
        )

    step_vectors = np.diff(pos, axis=0)
    path_length = float(np.sum(np.linalg.norm(step_vectors, axis=1)))
    net_progress = float(np.linalg.norm(pos[-1] - pos[0]))
    progress_ratio = 1.0 if path_length == 0.0 else net_progress / path_length

    # Jerk: third derivative of position via successive finite differences of velocity.
    if vel.size >= 3:
        accel = np.diff(vel) / dt
        jerk = np.diff(accel) / dt
        mean_abs_jerk = float(np.mean(np.abs(jerk)))
    else:
        mean_abs_jerk = 0.0

    oscillation = bool(
        heading_rate_sign_changes >= min_heading_rate_sign_changes
        and progress_ratio <= max_progress_ratio
    )

    return {
        "schema_version": OSCILLATORY_PREDICATE_SCHEMA,
        "predicate": "oscillatory_control",
        "evidence_kind": "diagnostic_proxy",
        "oscillation": oscillation,
        "fields": {
            "heading_rate_sign_changes": heading_rate_sign_changes,
            "linear_velocity_sign_changes": linear_velocity_sign_changes,
            "command_source_changes": command_source_changes,
            "net_progress_m": net_progress,
            "path_length_m": path_length,
            "progress_ratio": progress_ratio,
            "mean_abs_jerk": mean_abs_jerk,
            "n_steps": n,
        },
        "thresholds": {
            "min_heading_rate_sign_changes": min_heading_rate_sign_changes,
            "max_progress_ratio": max_progress_ratio,
        },
    }


def _first_true_index(flags: NDArray[np.bool_]) -> int | None:
    """Return the index of the first ``True`` flag, or ``None`` when there is none."""
    hits = np.flatnonzero(flags)
    return int(hits[0]) if hits.size else None


def late_evasive_predicate(
    hazard_distances: NDArray[np.floating] | object,
    hazard_visible: NDArray[np.bool_] | object,
    speeds: NDArray[np.floating] | object,
    *,
    dt: float,
    conflict_radius_m: float = 2.0,
    decel_threshold_m_s2: float = 0.5,
    max_response_latency_s: float = 1.0,
) -> dict[str, Any]:
    """Produce the late-evasive-reaction predicate fields for one episode.

    A late evasive reaction is a clearance-restoring action (here: the first significant
    deceleration) that comes too long after the hazard first becomes visible, or only
    after the robot has already entered the conflict zone.

    Args:
        hazard_distances: ``(N,)`` distance to the nearest hazard per step (m).
        hazard_visible: ``(N,)`` boolean — whether the hazard is visible per step.
        speeds: ``(N,)`` robot speed per step (m/s).
        dt: Per-step timestep (s, > 0).
        conflict_radius_m: Distance defining conflict-zone entry.
        decel_threshold_m_s2: Deceleration magnitude that counts as an evasive action.
        max_response_latency_s: Latency above which a reaction is "late".

    Returns:
        dict[str, Any]: Versioned record with the motivated fields and the diagnostic
        ``late_evasive`` boolean.
    """
    if not (dt > 0.0):
        raise ValueError(f"dt must be > 0, got {dt!r}")
    dist = np.asarray(hazard_distances, dtype=np.float64).reshape(-1)
    visible = np.asarray(hazard_visible, dtype=bool).reshape(-1)
    speed = np.asarray(speeds, dtype=np.float64).reshape(-1)
    n = dist.shape[0]
    if not (visible.shape[0] == n == speed.shape[0]):
        raise ValueError("hazard_distances, hazard_visible, and speeds must share length N")
    if n < 2:
        raise ValueError("at least two steps are required")
    require_finite_array("hazard_distances", dist)
    require_finite_array("speeds", speed)

    first_visible = _first_true_index(visible)
    conflict_entry = _first_true_index(dist <= conflict_radius_m)
    minimum_distance = float(np.min(dist))

    # Clearance-restoring action: first step whose deceleration exceeds the threshold,
    # at or after the hazard becomes visible.
    decel = -np.diff(speed) / dt  # positive when slowing down; index i is step i->i+1
    action_step: int | None = None
    if first_visible is not None:
        candidates = np.flatnonzero(decel >= decel_threshold_m_s2)
        candidates = candidates[candidates >= first_visible]
        action_step = int(candidates[0]) + 1 if candidates.size else None

    response_latency_s: float | None = None
    if first_visible is not None and action_step is not None:
        response_latency_s = float((action_step - first_visible) * dt)

    # Fail closed: a null response_latency_s must always carry a machine-readable reason so a
    # late-evasive event is never a silent empty. The dominant goal-planner case is
    # ``no_clearance_restoring_action`` — the robot never decelerated hard enough after the hazard
    # became visible, so there is no reaction to time even though ``late_evasive`` fires. See #5000.
    latency_unavailable_reason: str | None = None
    if response_latency_s is None:
        if first_visible is None:
            latency_unavailable_reason = "hazard_never_visible"
        else:
            latency_unavailable_reason = "no_clearance_restoring_action"

    # Required deceleration to stop within the visible distance, at first visibility.
    required_deceleration_m_s2 = 0.0
    if first_visible is not None:
        v0 = speed[first_visible]
        d0 = max(dist[first_visible], 1e-9)
        required_deceleration_m_s2 = float(v0 * v0 / (2.0 * d0))

    late_evasive = bool(
        first_visible is not None
        and (
            action_step is None
            or (response_latency_s is not None and response_latency_s > max_response_latency_s)
            or (conflict_entry is not None and action_step > conflict_entry)
        )
    )

    return {
        "schema_version": LATE_EVASIVE_PREDICATE_SCHEMA,
        "predicate": "late_evasive",
        "evidence_kind": "diagnostic_proxy",
        "late_evasive": late_evasive,
        "fields": {
            "first_hazard_visible_step": first_visible,
            "conflict_zone_entry_step": conflict_entry,
            "first_clearance_restoring_action_step": action_step,
            "minimum_distance_m": minimum_distance,
            "required_deceleration_m_s2": required_deceleration_m_s2,
            "response_latency_s": response_latency_s,
            "latency_unavailable_reason": latency_unavailable_reason,
            "n_steps": n,
        },
        "thresholds": {
            "conflict_radius_m": conflict_radius_m,
            "decel_threshold_m_s2": decel_threshold_m_s2,
            "max_response_latency_s": max_response_latency_s,
        },
    }


def _emergence_step(visible: NDArray[np.bool_]) -> int | None:
    """Return the first occluded->visible transition index, or ``None``."""
    for i in range(1, visible.shape[0]):
        if visible[i] and not visible[i - 1]:
            return i
    return None


@dataclass(frozen=True, slots=True)
class OcclusionNearMissParams:
    """Tunable thresholds for the occlusion-near-miss predicate.

    Attributes:
        near_miss_radius_m: Minimum-separation threshold defining a near miss.
        detection_confidence_threshold: Track confidence counting as a detection.
        decel_threshold_m_s2: Deceleration magnitude counting as a first response.
    """

    near_miss_radius_m: float = 0.5
    detection_confidence_threshold: float = 0.5
    decel_threshold_m_s2: float = 0.5


def occlusion_near_miss_predicate(  # noqa: PLR0913
    hazard_distances: NDArray[np.floating] | object,
    visible: NDArray[np.bool_] | object | None,
    track_confidence: NDArray[np.floating] | object | None,
    speeds: NDArray[np.floating] | object,
    *,
    dt: float,
    params: OcclusionNearMissParams | None = None,
    predicted_minimum_separation_m: float | None = None,
    visibility_evidence_status: str = "available",
    visibility_evidence_reason: str | None = None,
) -> dict[str, Any]:
    """Produce the occlusion-triggered-near-miss predicate fields for one episode.

    The predicate fires when a previously **occluded** actor emerges and a near miss
    follows — the failure family where the planner's separation buffer is too thin to
    absorb a late detection.

    Args:
        hazard_distances: ``(N,)`` distance to the actor per step (m).
        visible: ``(N,)`` boolean line-of-sight visibility per step.
        track_confidence: ``(N,)`` track-existence confidence per step in ``[0, 1]``.
        speeds: ``(N,)`` robot speed per step (m/s).
        dt: Per-step timestep (s, > 0).
        params: Tunable thresholds; defaults to :class:`OcclusionNearMissParams`.
        predicted_minimum_separation_m: Optional predicted min separation under occlusion.
        visibility_evidence_status: Evidence availability state for visibility inputs.
        visibility_evidence_reason: Optional reason for unavailable visibility evidence.

    Returns:
        dict[str, Any]: Versioned record with the motivated fields and the diagnostic
        ``occlusion_near_miss`` boolean.
    """
    if not (dt > 0.0):
        raise ValueError(f"dt must be > 0, got {dt!r}")
    params = params or OcclusionNearMissParams()
    dist = np.asarray(hazard_distances, dtype=np.float64).reshape(-1)
    speed = np.asarray(speeds, dtype=np.float64).reshape(-1)
    n = dist.shape[0]
    if speed.shape[0] != n:
        raise ValueError("hazard_distances and speeds must share length N")
    if n < 2:
        raise ValueError("at least two steps are required")
    require_finite_array("hazard_distances", dist)
    require_finite_array("speeds", speed)
    if predicted_minimum_separation_m is not None:
        predicted_minimum_separation_m = require_finite_scalar(
            "predicted_minimum_separation_m", predicted_minimum_separation_m
        )

    min_sep_step = int(np.argmin(dist))
    actual_minimum_separation = float(dist[min_sep_step])
    near_miss = actual_minimum_separation <= params.near_miss_radius_m

    missing_visibility = visible is None or track_confidence is None
    if missing_visibility:
        status = (
            "not_applicable" if visibility_evidence_status == "not_applicable" else "unavailable"
        )
        status_reason = visibility_evidence_reason or "missing_visibility_evidence"
        return {
            "schema_version": OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
            "predicate": "occlusion_near_miss",
            "evidence_kind": "diagnostic_proxy",
            "occlusion_near_miss": False,
            "status": status,
            "status_reason": status_reason,
            "fields": {
                "was_occluded_before_min": None,
                "emergence_step": None,
                "first_detection_step": None,
                "first_response_step": None,
                "min_separation_step": min_sep_step,
                "actual_minimum_separation_m": actual_minimum_separation,
                "predicted_minimum_separation_m": predicted_minimum_separation_m,
                "near_miss": near_miss,
                "visibility_evidence_status": status,
                "visibility_evidence_reason": status_reason,
                "n_steps": n,
            },
            "thresholds": {
                "near_miss_radius_m": params.near_miss_radius_m,
                "detection_confidence_threshold": params.detection_confidence_threshold,
                "decel_threshold_m_s2": params.decel_threshold_m_s2,
            },
        }

    vis = np.asarray(visible, dtype=bool).reshape(-1)
    conf = np.asarray(track_confidence, dtype=np.float64).reshape(-1)
    if not (vis.shape[0] == n == conf.shape[0]):
        raise ValueError("all per-step signals must share length N")
    require_finite_array("track_confidence", conf)

    # The actor was occluded at some point up to (and including) the closest approach.
    was_occluded_before_min = bool(np.any(~vis[: min_sep_step + 1]))
    emergence_step = _emergence_step(vis)

    detected = np.flatnonzero(conf >= params.detection_confidence_threshold)
    first_detection_step = int(detected[0]) if detected.size else None

    decel = -np.diff(speed) / dt
    first_response_step: int | None = None
    if first_detection_step is not None:
        candidates = np.flatnonzero(decel >= params.decel_threshold_m_s2)
        candidates = candidates[candidates >= first_detection_step]
        first_response_step = int(candidates[0]) + 1 if candidates.size else None

    occlusion_near_miss = bool(near_miss and was_occluded_before_min and emergence_step is not None)

    return {
        "schema_version": OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
        "predicate": "occlusion_near_miss",
        "evidence_kind": "diagnostic_proxy",
        "occlusion_near_miss": occlusion_near_miss,
        "status": "true" if occlusion_near_miss else "false",
        "status_reason": None,
        "fields": {
            "was_occluded_before_min": was_occluded_before_min,
            "emergence_step": emergence_step,
            "first_detection_step": first_detection_step,
            "first_response_step": first_response_step,
            "min_separation_step": min_sep_step,
            "actual_minimum_separation_m": actual_minimum_separation,
            "predicted_minimum_separation_m": predicted_minimum_separation_m,
            "near_miss": near_miss,
            "visibility_evidence_status": "available",
            "visibility_evidence_reason": visibility_evidence_reason,
            "n_steps": n,
        },
        "thresholds": {
            "near_miss_radius_m": params.near_miss_radius_m,
            "detection_confidence_threshold": params.detection_confidence_threshold,
            "decel_threshold_m_s2": params.decel_threshold_m_s2,
        },
    }


__all__ = [
    "LATE_EVASIVE_PREDICATE_SCHEMA",
    "OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA",
    "OSCILLATORY_PREDICATE_SCHEMA",
    "OcclusionNearMissParams",
    "late_evasive_predicate",
    "occlusion_near_miss_predicate",
    "oscillatory_control_predicate",
]
