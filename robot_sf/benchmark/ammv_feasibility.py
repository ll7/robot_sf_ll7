"""AMMV kinematic-feasibility and tip-over evaluation over a command sequence (issue #3466).

Some planner outputs that look acceptable under the current robot abstraction may violate a
three-wheeled Autonomous Micromobility Vehicle's lateral-stability (tip-over) or non-holonomic
limits, especially in dense pedestrian interactions. This module is the pure evaluator that flags
such commands over a per-step ``(velocity, turn_rate)`` sequence, **without** changing planner
behavior.

It reuses the benchmark-surface source of truth
``robot_sf.benchmark.metrics.evaluate_stability_margin`` (reconciled in #3587) for the tip-over
margin, so this lens and the existing ``rollover_min_stability_margin`` column stay consistent. The
non-holonomic curvature limit is an explicit **internal proxy**, not a hardware-AMMV claim.

Surfacing these fields in benchmark artifacts / evidence summaries (the artifact-pipeline wiring) is
a deliberate follow-up; this evaluator is pure and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.metrics import ROLLOVER_CRITICAL_EVENT, evaluate_stability_margin

if TYPE_CHECKING:
    from numpy.typing import NDArray

AMMV_FEASIBILITY_SCHEMA = "ammv_feasibility.v1"


@dataclass(frozen=True, slots=True)
class AmmvFeasibilityParams:
    """Internal-proxy AMMV geometry and kinematic limits (non-hardware).

    Stability geometry defaults match ``metrics.evaluate_stability_margin`` so the tip-over margin
    is identical to the benchmark-surface column. The curvature limit is a proxy non-holonomic bound.

    Attributes:
        track_width_m: ``t_w`` for the stability margin.
        wheelbase_m: ``L`` for the stability margin.
        cog_height_m: ``h_c`` for the stability margin.
        front_axle_to_cog_m: ``a`` for the stability margin.
        max_curvature_per_m: Proxy maximum path curvature ``|ω| / v`` while moving (1 / min radius).
        in_place_yaw_rate_max: Proxy maximum yaw rate permitted at near-zero speed.
        zero_speed_eps: Speed below which the in-place yaw limit applies instead of curvature.
    """

    track_width_m: float = 0.80
    wheelbase_m: float = 1.20
    cog_height_m: float = 0.60
    front_axle_to_cog_m: float = 0.50
    max_curvature_per_m: float = 1.0
    in_place_yaw_rate_max: float = 0.5
    zero_speed_eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate the proxy parameters."""
        for name in (
            "track_width_m",
            "wheelbase_m",
            "cog_height_m",
            "front_axle_to_cog_m",
            "max_curvature_per_m",
            "in_place_yaw_rate_max",
        ):
            if not (getattr(self, name) > 0.0):
                raise ValueError(f"AmmvFeasibilityParams.{name} must be > 0")


def _stability_margin(v: float, omega: float, params: AmmvFeasibilityParams) -> float:
    """Return the tip-over stability margin via the benchmark-surface source of truth."""
    return evaluate_stability_margin(
        v,
        omega,
        t_w=params.track_width_m,
        L=params.wheelbase_m,
        h_c=params.cog_height_m,
        a=params.front_axle_to_cog_m,
    )


def _curvature_feasible(v: float, omega: float, params: AmmvFeasibilityParams) -> bool:
    """Return whether one ``(v, ω)`` command respects the proxy non-holonomic limit."""
    if abs(v) <= params.zero_speed_eps:
        return abs(omega) <= params.in_place_yaw_rate_max
    return abs(omega) <= params.max_curvature_per_m * abs(v)


def evaluate_command_feasibility(
    velocities: NDArray[np.floating] | object,
    turn_rates: NDArray[np.floating] | object,
    params: AmmvFeasibilityParams | None = None,
) -> dict[str, Any]:
    """Evaluate AMMV tip-over and non-holonomic feasibility over a command sequence.

    Args:
        velocities: ``(N,)`` commanded forward speeds (m/s).
        turn_rates: ``(N,)`` commanded yaw rates (rad/s).
        params: Internal-proxy parameters; defaults to :class:`AmmvFeasibilityParams`.

    Returns:
        dict[str, Any]: Versioned telemetry — minimum stability margin, per-step tip-over and
        curvature violation counts, and the overall feasibility verdict.
    """
    params = params or AmmvFeasibilityParams()
    v = np.asarray(velocities, dtype=np.float64).reshape(-1)
    omega = np.asarray(turn_rates, dtype=np.float64).reshape(-1)
    if v.shape[0] != omega.shape[0]:
        raise ValueError("velocities and turn_rates must share length N")
    if v.shape[0] == 0:
        raise ValueError("at least one command is required")

    margins = [
        _stability_margin(float(vi), float(wi), params) for vi, wi in zip(v, omega, strict=True)
    ]
    tip_over_steps = [i for i, m in enumerate(margins) if m <= 0.0]
    curvature_violation_steps = [
        i
        for i, (vi, wi) in enumerate(zip(v, omega, strict=True))
        if not _curvature_feasible(vi, wi, params)
    ]
    min_margin = float(min(margins))
    feasible = not tip_over_steps and not curvature_violation_steps

    return {
        "schema_version": AMMV_FEASIBILITY_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "proxy_kind": "internal_non_hardware",
        "n_commands": int(v.shape[0]),
        "min_stability_margin": min_margin,
        "tip_over_violation": bool(tip_over_steps),
        "n_tip_over_steps": len(tip_over_steps),
        "rollover_event": ROLLOVER_CRITICAL_EVENT if tip_over_steps else "",
        "n_curvature_violations": len(curvature_violation_steps),
        "feasible": feasible,
    }


__all__ = [
    "AMMV_FEASIBILITY_SCHEMA",
    "AmmvFeasibilityParams",
    "evaluate_command_feasibility",
]
