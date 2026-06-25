"""Trace-level safety-predicate producers (issue #3483).

The thesis motivates several trace-level failure families but cannot report their
rates until in-sim *producers* emit auditable fields. This module implements those
producers as pure, versioned functions over a per-step trajectory, so the fields are
reproducible and fixture-backed instead of simulation-only assumptions.

This increment provides the **oscillatory local-control** predicate. Its boolean output
populates the existing ``SurrogateEvents.oscillation`` ledger slot; the detailed field
record is intended for the ledger ``surrogate_events`` block. The late-evasive and
occlusion-triggered-near-miss producers, and live emission wiring, are deliberate
follow-ups.

.. note::

   Predicate definitions are **versioned modeling choices**, labeled diagnostic until
   real-world label validation (externally blocked, #3278). Thresholds are explicit and
   overridable; the producer always emits the raw fields so a different threshold can be
   applied downstream without recomputation.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

OSCILLATORY_PREDICATE_SCHEMA = "safety_predicate.oscillatory_control.v1"


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
        command_sources: Optional ``(N,)`` per-step command-source labels.
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

    heading_rate = np.diff(np.unwrap(head)) / dt
    heading_rate_sign_changes = _sign_changes(heading_rate)
    linear_velocity_sign_changes = _sign_changes(vel)

    command_source_changes = 0
    if command_sources is not None:
        command_source_changes = int(sum(1 for a, b in pairwise(command_sources) if a != b))

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


__all__ = ["OSCILLATORY_PREDICATE_SCHEMA", "oscillatory_control_predicate"]
