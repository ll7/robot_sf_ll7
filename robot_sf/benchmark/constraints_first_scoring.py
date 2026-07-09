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

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scipy.stats import beta

from robot_sf.benchmark.identity.hash_utils import read_jsonl
from robot_sf.benchmark.metric_layers import LAYER_ORDER, METRIC_LAYER_SCHEMA_VERSION

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
        v
        for e in episodes
        if e.get(metric_key) is not None
        and (v := _as_float(e[metric_key], default=None)) is not None
    ]
    safe_values = [
        v
        for e in episodes
        if e.get(metric_key) is not None
        and bool(e.get(safe_key, False))
        and (v := _as_float(e[metric_key], default=None)) is not None
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


def build_constraints_first_report(
    planner_episodes: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    compensatory_scores: Mapping[str, float] | None = None,
    gates: AdmissibilityGates | None = None,
    comfort_key: str = "comfort",
    efficiency_key: str = "efficiency",
    collision_key: str = "collisions",
    safe_key: str = "safe_success",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Build the end-to-end constraints-first report over per-planner episode records.

    For each planner this emits the constraints-first summary (admissibility, collision UCB,
    survivorship-aware comfort/efficiency). When ``compensatory_scores`` is supplied, it also
    emits the ranking-inversion diagnostic using the **admissible rate** as the
    constraints-first ranking score — the empirical contrast between the soft composite and
    the constraints-first order.

    Returns:
        dict[str, Any]: Versioned report with ``per_planner`` summaries and an optional
        ``ranking_inversion`` block.
    """
    if not planner_episodes:
        raise ValueError("at least one planner is required")
    per_planner = {
        planner: constraints_first_planner_summary(
            episodes,
            gates=gates,
            comfort_key=comfort_key,
            efficiency_key=efficiency_key,
            collision_key=collision_key,
            safe_key=safe_key,
            confidence=confidence,
        )
        for planner, episodes in planner_episodes.items()
    }
    report: dict[str, Any] = {
        "schema_version": CONSTRAINTS_FIRST_SCHEMA,
        "metric_layer_schema_version": METRIC_LAYER_SCHEMA_VERSION,
        "metric_layer_order": list(LAYER_ORDER),
        "evidence_kind": "diagnostic_proxy",
        "per_planner": per_planner,
    }
    if compensatory_scores is not None:
        constraints_first_scores = {
            planner: summary["admissible_rate"] for planner, summary in per_planner.items()
        }
        report["ranking_inversion"] = ranking_inversion(
            compensatory_scores, constraints_first_scores
        )
    return report


def group_episodes_by_planner(
    records: Sequence[Mapping[str, Any]], *, planner_key: str = "planner"
) -> dict[str, list[dict[str, Any]]]:
    """Group flat episode records into per-planner lists by ``planner_key``.

    Returns:
        dict[str, list[dict[str, Any]]]: Planner name → its episode records.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        planner = record.get(planner_key)
        if planner is None:
            raise ValueError(f"episode record is missing planner key {planner_key!r}")
        grouped.setdefault(str(planner), []).append(dict(record))
    return grouped


def _build_parser() -> argparse.ArgumentParser:
    """Build the constraints-first report CLI parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Build a constraints-first scoring report.")
    parser.add_argument("--episodes", type=Path, required=True, help="Episode JSONL input.")
    parser.add_argument("--planner-key", default="planner", help="Record field naming the planner.")
    parser.add_argument(
        "--compensatory",
        type=Path,
        default=None,
        help="Optional JSON mapping planner -> compensatory composite score.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Report JSON output path.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence for the UCB.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Apply the constraints-first report to an existing episode JSONL file.

    Returns:
        int: ``0`` on success, ``2`` on a load/validation error.
    """
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        records = read_jsonl(args.episodes)
        planner_episodes = group_episodes_by_planner(records, planner_key=args.planner_key)
        compensatory = None
        if args.compensatory is not None:
            compensatory = json.loads(args.compensatory.read_text(encoding="utf-8"))
        report = build_constraints_first_report(
            planner_episodes, compensatory_scores=compensatory, confidence=args.confidence
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered + "\n")
    return 0


__all__ = [
    "CONSTRAINTS_FIRST_SCHEMA",
    "AdmissibilityGates",
    "build_constraints_first_report",
    "collision_upper_confidence_bound",
    "constraints_first_planner_summary",
    "group_episodes_by_planner",
    "is_episode_admissible",
    "main",
    "ranking_inversion",
    "survivorship_aware_metric",
]


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
