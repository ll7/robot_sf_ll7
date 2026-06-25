"""Constraints-first (non-compensatory) scoring layer over episode records (issue #3572).

The headline benchmark scoring is *compensatory*: safety, progress, comfort, and efficiency
are combined so a collision can be numerically offset by faster or smoother motion. For
safety-relevant navigation this is the wrong default. This module adds an **optional,
post-hoc** scoring layer over existing episode records that:

1. gates runs **lexicographically** (collision → near-miss severity → deadlock/timeout) so
   comfort/efficiency only rank among admissible runs;
2. reports each comfort/efficiency metric **twice** — unconditional and conditioned on safe
   success — and surfaces the survivorship-bias delta;
3. reports an **upper confidence bound** on collision/near-miss probability so low-N planners
   cannot appear collision-free; and
4. produces a **ranking-inversion** diagnostic between the compensatory composite and the
   constraints-first ordering.

It does **not** touch the simulator or the metric producers, keeps the existing composite
available for comparison, and claims no planner is "safe" — it only ranks under explicit
admissibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scipy.stats import beta

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CONSTRAINTS_FIRST_SCHEMA = "constraints_first_scoring.v1"


def collision_upper_confidence_bound(
    n_events: int, n_episodes: int, *, confidence: float = 0.95
) -> float:
    """Return the one-sided Clopper–Pearson upper bound on an event probability.

    With ``n_events`` collisions (or near misses) in ``n_episodes`` runs, this is the upper
    end of the one-sided ``confidence``-level exact binomial interval. At ``n_events == 0``
    it reproduces the familiar rule-of-three (``≈ 3/N`` at 95%), so a planner with zero
    observed collisions over few episodes is not reported as collision-free.

    Returns:
        float: Upper confidence bound in ``[0, 1]``.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be > 0")
    if not (0 <= n_events <= n_episodes):
        raise ValueError("n_events must satisfy 0 <= n_events <= n_episodes")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if n_events == n_episodes:
        return 1.0
    return float(beta.ppf(confidence, n_events + 1, n_episodes - n_events))


@dataclass(frozen=True, slots=True)
class AdmissibilityGates:
    """Lexicographic admissibility gates for a single episode.

    Attributes:
        max_collisions: Maximum collisions an admissible episode may have (default 0).
        max_near_miss_severity: Optional cap on near-miss severity; ``None`` disables it.
        forbid_timeout: Whether a timed-out episode is inadmissible.
        forbid_deadlock: Whether a deadlocked episode is inadmissible.
    """

    max_collisions: int = 0
    max_near_miss_severity: float | None = None
    forbid_timeout: bool = True
    forbid_deadlock: bool = True


def is_episode_admissible(
    episode: Mapping[str, Any],
    gates: AdmissibilityGates | None = None,
    *,
    collision_key: str = "collisions",
    near_miss_key: str = "near_miss_severity",
    timeout_key: str = "timeout",
    deadlock_key: str = "deadlock",
) -> bool:
    """Return whether an episode clears the lexicographic admissibility gates."""
    gates = gates or AdmissibilityGates()
    if _as_float(episode.get(collision_key), default=0.0) > gates.max_collisions:
        return False
    if gates.max_near_miss_severity is not None:
        severity = episode.get(near_miss_key)
        if severity is not None and _as_float(severity, default=0.0) > gates.max_near_miss_severity:
            return False
    if gates.forbid_timeout and bool(episode.get(timeout_key, False)):
        return False
    return not (gates.forbid_deadlock and bool(episode.get(deadlock_key, False)))


def survivorship_aware_metric(
    episodes: Sequence[Mapping[str, Any]],
    metric_key: str,
    *,
    safe_key: str = "safe_success",
) -> dict[str, Any]:
    """Report a metric unconditionally and conditioned on safe success, with the delta.

    Comfort/efficiency metrics computed only over successful episodes flatter planners that
    fail more often; the delta exposes that survivorship bias.

    Returns:
        dict[str, Any]: ``unconditional_mean``, ``conditioned_on_safe_success_mean``,
        ``survivorship_delta`` (conditioned − unconditional), and the two sample sizes.
    """
    all_values = [
        _as_float(e[metric_key], default=None) for e in episodes if e.get(metric_key) is not None
    ]
    safe_values = [
        _as_float(e[metric_key], default=None)
        for e in episodes
        if e.get(metric_key) is not None and bool(e.get(safe_key, False))
    ]
    unconditional = _mean(all_values)
    conditioned = _mean(safe_values)
    delta = None
    if unconditional is not None and conditioned is not None:
        delta = conditioned - unconditional
    return {
        "metric": metric_key,
        "unconditional_mean": unconditional,
        "conditioned_on_safe_success_mean": conditioned,
        "survivorship_delta": delta,
        "n_all": len(all_values),
        "n_safe_success": len(safe_values),
    }


def constraints_first_planner_summary(
    episodes: Sequence[Mapping[str, Any]],
    *,
    gates: AdmissibilityGates | None = None,
    comfort_key: str = "comfort",
    efficiency_key: str = "efficiency",
    collision_key: str = "collisions",
    safe_key: str = "safe_success",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize one planner under constraints-first scoring.

    Returns:
        dict[str, Any]: Versioned summary with admissibility rate, collision rate + upper
        confidence bound, and survivorship-aware comfort/efficiency.
    """
    gates = gates or AdmissibilityGates()
    n = len(episodes)
    if n == 0:
        raise ValueError("at least one episode is required")
    admissible_flags = [
        is_episode_admissible(e, gates, collision_key=collision_key) for e in episodes
    ]
    n_collision_episodes = sum(
        1 for e in episodes if _as_float(e.get(collision_key), default=0.0) > 0
    )
    return {
        "schema_version": CONSTRAINTS_FIRST_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "n_episodes": n,
        "admissible_rate": sum(admissible_flags) / n,
        "collision_rate": n_collision_episodes / n,
        "collision_upper_confidence_bound": collision_upper_confidence_bound(
            n_collision_episodes, n, confidence=confidence
        ),
        "comfort": survivorship_aware_metric(episodes, comfort_key, safe_key=safe_key),
        "efficiency": survivorship_aware_metric(episodes, efficiency_key, safe_key=safe_key),
    }


def ranking_inversion(
    compensatory_scores: Mapping[str, float],
    constraints_first_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Diagnose rank changes between the compensatory and constraints-first orderings.

    Both mappings are planner → score (higher is better). Returns each planner's rank under
    both orderings, the rank delta, and the planners whose rank changes — the empirical
    justification for non-compensatory evaluation.

    Returns:
        dict[str, Any]: Per-planner ranks/deltas and the list of inverted planners.
    """
    if set(compensatory_scores) != set(constraints_first_scores):
        raise ValueError("both rankings must cover the same planners")
    comp_rank = _ranks(compensatory_scores)
    cons_rank = _ranks(constraints_first_scores)
    per_planner = {
        name: {
            "compensatory_rank": comp_rank[name],
            "constraints_first_rank": cons_rank[name],
            "rank_delta": cons_rank[name] - comp_rank[name],
        }
        for name in compensatory_scores
    }
    inverted = sorted(name for name, r in per_planner.items() if r["rank_delta"] != 0)
    return {
        "schema_version": CONSTRAINTS_FIRST_SCHEMA,
        "per_planner": per_planner,
        "inverted_planners": inverted,
        "any_inversion": bool(inverted),
    }


def _ranks(scores: Mapping[str, float]) -> dict[str, int]:
    """Return 1-based ranks (1 = best); ties broken by planner name for determinism."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {name: i + 1 for i, (name, _) in enumerate(ordered)}


def _as_float(value: Any, *, default: float | None) -> float | None:
    """Coerce a value to float.

    Returns:
        float | None: The float value, or ``default`` on missing/non-numeric input.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float | None]) -> float | None:
    """Return the mean of the present values, or ``None`` when empty."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


__all__ = [
    "CONSTRAINTS_FIRST_SCHEMA",
    "AdmissibilityGates",
    "collision_upper_confidence_bound",
    "constraints_first_planner_summary",
    "is_episode_admissible",
    "ranking_inversion",
    "survivorship_aware_metric",
]
