"""Quantify how much pedestrian (non-)reactivity flatters planners (issue #3573).

A structural validity question for any pedestrian-interaction benchmark is whether the simulated
pedestrians **react to the robot** or merely follow predetermined motion. Non-reactive replay
(open-loop playback) lets a planner intrude and still accrue good downstream comfort/efficiency,
inflating apparent performance; fully reactive social-force pedestrians may over-yield. This module
is the pure **quantification layer** that turns a paired reactive-vs-replay ablation (run over
identical scenarios + seeds with common random numbers) into the issue's deliverable: per-planner
deltas attributable to reactivity, the replay inflation, and which planners' rank is
reactivity-sensitive.

The paired ablation runs (and the open-loop replay pedestrian mode) are deferred; this layer is
pure and side-effect free, mirroring the accepted decision layers in
``robot_sf/scenario_certification/failure_cause.py`` (#3484) and siblings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

REACTIVITY_ABLATION_SCHEMA = "reactivity_ablation.v1"

_CONTRAST_METRIC_FIELDS = (
    "reactive_collision_rate",
    "replay_collision_rate",
    "reactive_near_miss_rate",
    "replay_near_miss_rate",
    "reactive_min_separation_m",
    "replay_min_separation_m",
)


@dataclass(frozen=True, slots=True)
class ReactivityContrast:
    """Paired reactive-vs-replay safety results for one planner (fixed scenarios + seeds).

    Attributes:
        planner: Planner name.
        reactive_collision_rate: Collision rate with reactive social-force pedestrians.
        replay_collision_rate: Collision rate with non-reactive open-loop replay.
        reactive_near_miss_rate: Near-miss rate (reactive).
        replay_near_miss_rate: Near-miss rate (replay).
        reactive_min_separation_m: Mean minimum separation (reactive).
        replay_min_separation_m: Mean minimum separation (replay).
    """

    planner: str
    reactive_collision_rate: float
    replay_collision_rate: float
    reactive_near_miss_rate: float
    replay_near_miss_rate: float
    reactive_min_separation_m: float
    replay_min_separation_m: float


def reactivity_delta(contrast: ReactivityContrast) -> dict[str, Any]:
    """Quantify the per-planner safety change attributable to pedestrian reactivity.

    Deltas are ``reactive − replay``. A positive collision/near-miss delta means replay *under*
    reports the hazard (replay flatters the planner); a positive separation delta means replay
    overstates separation.

    Returns:
        dict[str, Any]: Per-metric deltas and a ``replay_flatters`` flag.
    """
    for field in _CONTRAST_METRIC_FIELDS:
        value = getattr(contrast, field)
        if not math.isfinite(value):
            raise ValueError(
                f"contrast.{field} for planner {contrast.planner!r} must be finite, got {value}"
            )
    collision_delta = contrast.reactive_collision_rate - contrast.replay_collision_rate
    near_miss_delta = contrast.reactive_near_miss_rate - contrast.replay_near_miss_rate
    separation_delta = contrast.reactive_min_separation_m - contrast.replay_min_separation_m
    return {
        "planner": contrast.planner,
        "collision_delta": collision_delta,
        "near_miss_delta": near_miss_delta,
        "min_separation_delta_m": separation_delta,
        # Replay flatters when it shows fewer collisions/near-misses or more separation.
        "replay_flatters": collision_delta > 0.0 or near_miss_delta > 0.0 or separation_delta < 0.0,
    }


def _rank_by_collision(rates: dict[str, float]) -> dict[str, int]:
    """Rank planners by ascending collision rate (1 = safest); ties broken by name.

    Returns:
        dict[str, int]: Planner name → 1-based rank.
    """
    ordered = sorted(rates.items(), key=lambda kv: (kv[1], kv[0]))
    return {planner: i + 1 for i, (planner, _) in enumerate(ordered)}


def assess_reactivity_ablation(contrasts: list[ReactivityContrast]) -> dict[str, Any]:
    """Summarize the reactive-vs-replay ablation across planners.

    Returns:
        dict[str, Any]: Versioned report with per-planner deltas, the mean replay collision/near-miss
        inflation, the planners flattered by replay, and the rank-reactivity-sensitive planners.
    """
    if not contrasts:
        raise ValueError("at least one planner contrast is required")
    deltas = [reactivity_delta(contrast) for contrast in contrasts]

    reactive_ranks = _rank_by_collision({c.planner: c.reactive_collision_rate for c in contrasts})
    replay_ranks = _rank_by_collision({c.planner: c.replay_collision_rate for c in contrasts})
    rank_sensitive = sorted(
        planner for planner in reactive_ranks if reactive_ranks[planner] != replay_ranks[planner]
    )

    n = len(deltas)
    return {
        "schema_version": REACTIVITY_ABLATION_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "n_planners": n,
        "per_planner": deltas,
        # Mean (reactive − replay): positive means replay systematically under-reports the hazard.
        "mean_replay_collision_inflation": sum(d["collision_delta"] for d in deltas) / n,
        "mean_replay_near_miss_inflation": sum(d["near_miss_delta"] for d in deltas) / n,
        "planners_flattered_by_replay": sorted(
            d["planner"] for d in deltas if d["replay_flatters"]
        ),
        "rank_reactivity_sensitive_planners": rank_sensitive,
        "ranking_is_reactivity_sensitive": bool(rank_sensitive),
    }


__all__ = [
    "REACTIVITY_ABLATION_SCHEMA",
    "ReactivityContrast",
    "assess_reactivity_ablation",
    "reactivity_delta",
]
