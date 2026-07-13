"""Reproducible seed-flip and held-out planner-inversion candidate mining.

This module mines *case candidates* from benchmark result rows for issue #5446:
scenario/seed/planner cells that show a **reproducible seed-dependent outcome
flip** or a **genuine held-out planner upset**. It is analysis tooling, not a
benchmark metric and not a planner-ranking claim: it selects *candidates worth
confirming*, records the uncertainty behind each, and explains every exclusion.

Design contract (issue #5446)
-----------------------------
The miner deliberately avoids the failure modes of a single opaque
"interestingness" score (see the figure-selection heuristic in
``robot_sf/benchmark/trace_exemplar_interest.py`` / ``exemplar_selection.py``,
which this module does **not** replace):

1. **Evidence gates first.** A row is eligible only with complete provenance
   (episode id, scenario, seed, config hash, pinned repo commit), *native*
   execution (never fallback/degraded/adapter), and typed collision semantics
   where declared. Ineligible rows are excluded with a recorded reason.
2. **Separate typed fields, never one hidden score.** Each candidate keeps
   ``seed_flip`` posterior/entropy/effective-denominator, held-out planner-skill
   gap, upset outcome, cross-planner disagreement entropy, and slots for the
   sibling-issue signals (oracle regret #5302, transfer #5303,
   quality-diversity descriptors #5308, multiplicity/hierarchy #5351). Missing
   signals are reported ``unavailable`` — they are never fabricated or folded
   into a composite.
3. **Non-circular strength.** Planner strength is estimated *leave-one-family
   (scenario)-out*, excluding the candidate cell's scenario, so a cell cannot
   both define and be judged by a planner's strength. Raw paired outcomes are
   retained.
4. **Pareto/diversity selection.** Final selection is the non-dominated frontier
   over the separate axes; weights are never hidden in a scalar.
5. **Fail closed.** Empty/short/ineligible evidence is reported as such. Three-
   seed cells are triage-only. Release 0.0.2 collision-derived fields are
   excluded while #5097 remains unresolved.

The output is a schema-versioned candidate manifest (dict) suitable for a
compact evidence file under ``docs/context/evidence/``. Confirmation *runs* are
explicitly out of scope: they belong to a separate exact-compute packet.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

# Reuse the canonical binary-outcome coercion and the Wilson interval already
# used by the flakiness audit / full-classic aggregation so this miner reports
# the *same* outcome and interval semantics as the rest of the benchmark stack.
from robot_sf.benchmark.full_classic.aggregation import _wilson_interval
from robot_sf.benchmark.grouping import (
    DEFAULT_REPORT_FALLBACK_GROUP_BY,
    DEFAULT_REPORT_GROUP_BY,
    get_nested,
    resolve_report_group_key,
)
from robot_sf.benchmark.scenario_flakiness import _binary_outcome

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = "seed_flip_inversion_candidates.v1"

#: Provenance fields every eligible row must carry (mirrors the benchmark
#: result-provenance row contract, ``benchmark_row_provenance.v1``).
REQUIRED_PROVENANCE_FIELDS = ("episode_id", "scenario_id", "seed", "config_hash", "repo_commit")

#: Only native execution counts as evidence. Fallback/degraded/adapter rows are
#: excluded (never presented as success evidence — AGENTS.md research discipline).
NATIVE_EXECUTION_MODE = "native"

#: Release 0.0.2 collision-derived fields are withdrawn while issue #5097 is
#: unresolved (exact per-event collision provenance unavailable). A candidate's
#: outcome metric must not be one of these for a 0.0.2-provenance row.
COLLISION_DERIVED_WITHDRAWN_FIELDS = frozenset(
    {"total_collision_count", "snqi_collision_term", "success_rate_collision_gate"}
)
WITHDRAWN_RELEASE_TAG = "0.0.2"

#: Distinct-seed thresholds. A cell needs >= 2 seeds to *flip*; cells at or below
#: the triage cap carry a ``triage_only`` flag (three-seed results are triage).
DEFAULT_MIN_SEEDS = 2
DEFAULT_TRIAGE_MAX_SEEDS = 3

#: Minimum other-scenario cells required before a leave-one-out planner strength
#: estimate is trusted; below this the strength (and any upset) is unavailable.
DEFAULT_MIN_HELDOUT_CELLS = 1

DEFAULT_CONF = 0.95

#: Sibling issues whose signals this miner *consumes* (links) rather than
#: reimplements. Absent inputs are reported unavailable, not fabricated.
EXTERNAL_SIGNAL_ISSUES = {
    "oracle_regret": "#5302",
    "transfer": "#5303",
    "quality_diversity": "#5308",
    "multiplicity": "#5351",
}

#: Candidate archetypes the miner supports (issue #5446 acceptance criteria).
ARCHETYPES = ("seed_flip", "planner_upset", "causal_divergence", "disagreement_recovery")


class SeedFlipMiningError(ValueError):
    """Raised when the miner cannot produce a defensible candidate manifest."""


def _binary_entropy(p: float) -> float:
    """Shannon entropy (bits) of a Bernoulli(p); 0 at the deterministic ends.

    Returns:
        The entropy in bits, in ``[0, 1]``.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _beta_posterior(ones: int, zeros: int) -> dict[str, float]:
    """Jeffreys-prior Beta posterior summary for a Bernoulli success count.

    Uses the ``Beta(0.5, 0.5)`` reference prior so the posterior stays proper for
    all-success or all-failure cells. Returns closed-form mean and variance; the
    reported credible *interval* is the Wilson score interval on the raw
    proportion (the benchmark stack's canonical interval), kept as a separate
    field so the interval method is explicit rather than implied.

    Returns:
        Mapping with ``mean``, ``variance``, ``alpha``, and ``beta``.
    """
    alpha = ones + 0.5
    beta = zeros + 0.5
    total = alpha + beta
    mean = alpha / total
    var = (alpha * beta) / (total * total * (total + 1.0))
    return {"mean": mean, "variance": var, "alpha": alpha, "beta": beta}


def _row_eligibility(
    row: dict[str, Any],
    *,
    outcome_metric: str,
    require_native: bool,
) -> str | None:
    """Return an exclusion reason for ``row`` or ``None`` when eligible.

    Fails closed: missing provenance, non-native execution, untyped collision
    semantics (when declared), and withdrawn release-0.0.2 collision-derived
    outcome metrics are all rejected with a concrete reason string.

    Returns:
        A concrete exclusion reason string, or ``None`` when the row is eligible.
    """
    for field in REQUIRED_PROVENANCE_FIELDS:
        value = get_nested(row, field)
        if value is None or (isinstance(value, str) and not value.strip()):
            return f"missing_provenance:{field}"

    if require_native:
        mode = get_nested(row, "execution_mode")
        if mode is None:
            return "missing_provenance:execution_mode"
        if str(mode) != NATIVE_EXECUTION_MODE:
            return f"non_native_execution:{mode}"

    # Typed collision semantics: only reject when the row *declares* an
    # incompatible value; absence is handled by the provenance gate above only
    # when callers require it. Here a declared non-typed value fails closed.
    collision_semantics = get_nested(row, "collision_semantics")
    if collision_semantics is not None and str(collision_semantics) != "typed":
        return f"untyped_collision_semantics:{collision_semantics}"

    # Release 0.0.2 collision-derived exclusion (issue #5097 unresolved).
    release = get_nested(row, "release")
    if (
        release is not None
        and str(release) == WITHDRAWN_RELEASE_TAG
        and outcome_metric in COLLISION_DERIVED_WITHDRAWN_FIELDS
    ):
        return f"withdrawn_collision_derived_field:{outcome_metric}@{WITHDRAWN_RELEASE_TAG}"
    if bool(get_nested(row, "collision_derived_withdrawn")):
        return "withdrawn_collision_derived_field:row_flagged"

    return None


def _cell_key(scenario_id: str, planner: str) -> str:
    """Return the ``scenario::planner`` cell key.

    Returns:
        The composite cell key string.
    """
    return f"{scenario_id}::{planner}"


def _external_status(
    external: dict[str, Any] | None,
    signal: str,
    key: str,
) -> dict[str, Any]:
    """Look up a sibling-issue signal for a candidate, or report it unavailable.

    ``external`` maps a signal name (e.g. ``"oracle_regret"``) to a mapping from
    candidate key to a value. Missing signals/keys are reported ``unavailable``
    with the owning issue link so the manifest never fabricates a value.

    Returns:
        A mapping with ``status`` (``consumed``/``unavailable``), the owning
        ``consumes_issue``, and ``value`` when the signal was consumed.
    """
    issue = EXTERNAL_SIGNAL_ISSUES[signal]
    if not external or signal not in external or external[signal] is None:
        return {"status": "unavailable", "consumes_issue": issue}
    table = external[signal]
    if not isinstance(table, dict) or key not in table:
        return {"status": "unavailable", "consumes_issue": issue}
    return {"status": "consumed", "consumes_issue": issue, "value": table[key]}


def _build_cells(
    rows: Sequence[dict[str, Any]],
    *,
    outcome_metric: str,
    group_by: str,
    fallback_group_by: str,
    seed_field: str,
    require_native: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Group eligible rows into ``(scenario, planner)`` cells keyed by seed.

    Each cell collapses within-seed repeats to a single representative vote
    (majority, ties -> success, matching the flakiness audit) so cross-seed flips
    are counted per distinct seed.

    Returns:
        ``(cells, exclusions)`` where ``cells`` maps each cell key to its
        aggregated stats and ``exclusions`` lists every dropped row with a reason.
    """
    exclusions: list[dict[str, Any]] = []
    # cell -> seed -> list[int]
    staged: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict[str, str]] = {}
    boundary_margin: dict[str, list[float]] = defaultdict(list)
    visualizability: dict[str, list[float]] = defaultdict(list)

    for idx, row in enumerate(rows):
        reason = _row_eligibility(row, outcome_metric=outcome_metric, require_native=require_native)
        scenario_id = get_nested(row, "scenario_id")
        planner = (
            resolve_report_group_key(
                row, group_by=group_by, fallback_group_by=fallback_group_by, missing="unknown"
            )
            or "unknown"
        )
        if reason is not None:
            exclusions.append(
                {
                    "row_index": idx,
                    "scenario_id": None if scenario_id is None else str(scenario_id),
                    "planner": planner,
                    "reason": reason,
                }
            )
            continue
        outcome = _binary_outcome(_extract_metric(row, outcome_metric))
        if outcome is None:
            exclusions.append(
                {
                    "row_index": idx,
                    "scenario_id": str(scenario_id),
                    "planner": planner,
                    "reason": f"uninterpretable_outcome:{outcome_metric}",
                }
            )
            continue
        key = _cell_key(str(scenario_id), planner)
        staged[key][str(get_nested(row, seed_field))].append(outcome)
        meta[key] = {"scenario_id": str(scenario_id), "planner": planner}
        margin = _extract_metric(row, "temporal_boundary_margin")
        if isinstance(margin, int | float) and not math.isnan(float(margin)):
            boundary_margin[key].append(float(margin))
        vis = _extract_metric(row, "visualizability")
        if isinstance(vis, int | float) and not math.isnan(float(vis)):
            visualizability[key].append(float(vis))

    cells: dict[str, dict[str, Any]] = {}
    for key, seed_map in staged.items():
        seed_votes: dict[str, int] = {}
        n_episodes = 0
        for seed, outcomes in seed_map.items():
            n_episodes += len(outcomes)
            seed_votes[seed] = 1 if (sum(outcomes) / len(outcomes)) >= 0.5 else 0
        ones = sum(seed_votes.values())
        n_seeds = len(seed_votes)
        cells[key] = {
            "cell_key": key,
            "scenario_id": meta[key]["scenario_id"],
            "planner": meta[key]["planner"],
            "n_seeds": n_seeds,
            "n_episodes": n_episodes,
            "ones": ones,
            "zeros": n_seeds - ones,
            "success_rate": ones / n_seeds if n_seeds else None,
            "seeds": dict(sorted(seed_votes.items())),
            "temporal_boundary_margin": (
                min(boundary_margin[key]) if boundary_margin.get(key) else None
            ),
            "visualizability": (
                sum(visualizability[key]) / len(visualizability[key])
                if visualizability.get(key)
                else None
            ),
        }
    return cells, exclusions


def _extract_metric(row: dict[str, Any], metric: str) -> Any:
    """Read ``metric`` from a row's ``metrics`` block or a flattened top-level path.

    Returns:
        The metric value, or ``None`` when absent.
    """
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and metric in metrics:
        return metrics[metric]
    return get_nested(row, metric)


def _heldout_strength(
    cells: dict[str, dict[str, Any]],
    planner: str,
    exclude_scenario: str,
    *,
    min_cells: int,
) -> dict[str, Any]:
    """Leave-one-scenario-out strength for ``planner`` excluding ``exclude_scenario``.

    Strength is the mean cell success rate over the planner's *other* scenarios,
    so the candidate cell never informs the strength used to judge it.

    Returns:
        A mapping with ``status`` (``ok``/``unavailable``), the ``strength`` mean
        when available, and the ``n_heldout_cells`` denominator.
    """
    rates = [
        c["success_rate"]
        for c in cells.values()
        if c["planner"] == planner
        and c["scenario_id"] != exclude_scenario
        and c["success_rate"] is not None
    ]
    if len(rates) < min_cells:
        return {"status": "unavailable", "n_heldout_cells": len(rates)}
    return {
        "status": "ok",
        "strength": sum(rates) / len(rates),
        "n_heldout_cells": len(rates),
    }


def _seed_flip_candidates(
    cells: dict[str, dict[str, Any]],
    *,
    conf: float,
    min_seeds: int,
    triage_max_seeds: int,
    external: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Emit a seed-flip candidate for every cell whose seeds actually flip.

    Returns:
        A list of seed-flip candidate records (one per flipping cell).
    """
    out: list[dict[str, Any]] = []
    for key in sorted(cells):
        cell = cells[key]
        n_seeds = cell["n_seeds"]
        ones, zeros = cell["ones"], cell["zeros"]
        if n_seeds < min_seeds or ones == 0 or zeros == 0:
            continue  # no flip: not a seed-flip candidate
        posterior = _beta_posterior(ones, zeros)
        low, high = _wilson_interval(cell["success_rate"], n_seeds, conf)
        entropy = _binary_entropy(cell["success_rate"])
        out.append(
            {
                "candidate_id": f"seed_flip::{key}",
                "archetype": "seed_flip",
                "scenario_id": cell["scenario_id"],
                "planner": cell["planner"],
                "triage_only": n_seeds <= triage_max_seeds,
                "seed_flip": {
                    "posterior_mean": posterior["mean"],
                    "posterior_variance": posterior["variance"],
                    "posterior_prior": "jeffreys_beta_0.5_0.5",
                    "interval_method": "wilson_score",
                    "interval": [low, high],
                    "entropy_bits": entropy,
                    "effective_denominator": n_seeds,
                    "raw_success_seeds": ones,
                    "raw_failure_seeds": zeros,
                },
                "heldout_planner_skill_gap": None,
                "upset_outcome": None,
                "cross_planner_disagreement_entropy": None,
                "oracle_regret": _external_status(external, "oracle_regret", key),
                "transfer": _external_status(external, "transfer", key),
                "quality_diversity": _external_status(external, "quality_diversity", key),
                "multiplicity": _external_status(external, "multiplicity", key),
                "temporal_boundary_margin": cell["temporal_boundary_margin"],
                "reproducibility": {"n_seeds": n_seeds, "raw_seed_outcomes": cell["seeds"]},
                "visualizability": cell["visualizability"],
            }
        )
    return out


def _evaluate_upset_pair(
    cells: dict[str, dict[str, Any]],
    scenario: str,
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    min_heldout_cells: int,
    external: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return an upset candidate for the ``(a, b)`` pair on ``scenario`` or ``None``.

    The held-out underdog is the planner with the lower leave-this-scenario-out
    strength; it upsets only when it also wins the cell. Both strengths must be
    trustworthy (enough held-out cells) or the pair is skipped.

    Returns:
        A planner-upset candidate record, or ``None`` when no upset is present or
        strength is unavailable.
    """
    strength_a = _heldout_strength(cells, a["planner"], scenario, min_cells=min_heldout_cells)
    strength_b = _heldout_strength(cells, b["planner"], scenario, min_cells=min_heldout_cells)
    if strength_a["status"] != "ok" or strength_b["status"] != "ok":
        return None
    if strength_a["strength"] == strength_b["strength"]:
        return None
    if strength_a["strength"] < strength_b["strength"]:
        weak, strong, weak_str, strong_str = a, b, strength_a, strength_b
    else:
        weak, strong, weak_str, strong_str = b, a, strength_b, strength_a
    if weak["success_rate"] <= strong["success_rate"]:
        return None  # underdog did not actually win the cell
    key = f"{scenario}::{weak['planner']}>{strong['planner']}"
    return {
        "candidate_id": f"planner_upset::{key}",
        "archetype": "planner_upset",
        "scenario_id": scenario,
        "planner": weak["planner"],
        "triage_only": min(weak["n_seeds"], strong["n_seeds"]) <= 3,
        "seed_flip": None,
        "heldout_planner_skill_gap": strong_str["strength"] - weak_str["strength"],
        "upset_outcome": {
            "underdog_planner": weak["planner"],
            "favorite_planner": strong["planner"],
            "underdog_heldout_strength": weak_str["strength"],
            "favorite_heldout_strength": strong_str["strength"],
            "underdog_heldout_cells": weak_str["n_heldout_cells"],
            "favorite_heldout_cells": strong_str["n_heldout_cells"],
            "underdog_cell_success": weak["success_rate"],
            "favorite_cell_success": strong["success_rate"],
            "outcome_gap": weak["success_rate"] - strong["success_rate"],
            "raw_paired_outcomes": {
                weak["planner"]: weak["seeds"],
                strong["planner"]: strong["seeds"],
            },
        },
        "cross_planner_disagreement_entropy": None,
        "oracle_regret": _external_status(external, "oracle_regret", scenario),
        "transfer": _external_status(external, "transfer", scenario),
        "quality_diversity": _external_status(external, "quality_diversity", scenario),
        "multiplicity": _external_status(external, "multiplicity", scenario),
        "temporal_boundary_margin": None,
        "reproducibility": {
            "underdog_n_seeds": weak["n_seeds"],
            "favorite_n_seeds": strong["n_seeds"],
        },
        "visualizability": weak["visualizability"],
    }


def _planner_upset_candidates(
    cells: dict[str, dict[str, Any]],
    *,
    min_heldout_cells: int,
    external: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Emit held-out planner-upset candidates.

    On a scenario, planner ``a`` *upsets* planner ``b`` when ``a`` is weaker on
    the held-out (leave-this-scenario-out) strength estimate yet wins the cell.
    Raw paired outcomes are retained and the strength that judges the upset never
    includes the candidate scenario.

    Returns:
        A list of planner-upset candidate records (one per detected upset pair).
    """
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells.values():
        if cell["success_rate"] is not None:
            by_scenario[cell["scenario_id"]].append(cell)

    out: list[dict[str, Any]] = []
    for scenario in sorted(by_scenario):
        scenario_cells = by_scenario[scenario]
        for a in scenario_cells:
            for b in scenario_cells:
                if a["planner"] >= b["planner"]:
                    continue  # unordered pair, evaluate each direction once
                candidate = _evaluate_upset_pair(
                    cells, scenario, a, b, min_heldout_cells=min_heldout_cells, external=external
                )
                if candidate is not None:
                    out.append(candidate)
    return out


def _disagreement_by_scenario(cells: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cross-planner outcome-disagreement entropy for each scenario.

    Binarizes each planner's cell (success_rate >= 0.5) and reports the Bernoulli
    entropy of the fraction of planners that succeed. High entropy == planners
    disagree on the scenario, the ``disagreement_recovery`` archetype signal.

    Returns:
        A mapping from scenario id to its disagreement summary (``n_planners``,
        ``fraction_success``, ``entropy_bits``).
    """
    by_scenario: dict[str, list[int]] = defaultdict(list)
    for cell in cells.values():
        if cell["success_rate"] is not None:
            by_scenario[cell["scenario_id"]].append(1 if cell["success_rate"] >= 0.5 else 0)
    out: dict[str, dict[str, Any]] = {}
    for scenario, votes in by_scenario.items():
        n = len(votes)
        frac = sum(votes) / n if n else 0.0
        out[scenario] = {
            "n_planners": n,
            "fraction_success": frac,
            "entropy_bits": _binary_entropy(frac),
        }
    return out


def _pareto_select(candidates: list[dict[str, Any]]) -> list[str]:
    """Non-dominated (Pareto) selection over the separate candidate axes.

    Axes (all maximized, missing -> 0): seed-flip entropy, |held-out skill gap|,
    cross-planner disagreement entropy, and effective denominator. A candidate is
    selected when no other candidate is >= on every axis and strictly greater on
    at least one. No axis is weighted into a scalar (issue #5446 contract).

    Returns:
        The candidate ids on the non-dominated frontier.
    """

    def axes(cand: dict[str, Any]) -> tuple[float, float, float, float]:
        flip = cand.get("seed_flip") or {}
        gap = cand.get("heldout_planner_skill_gap")
        return (
            float(flip.get("entropy_bits") or 0.0),
            abs(float(gap)) if gap is not None else 0.0,
            float(cand.get("cross_planner_disagreement_entropy") or 0.0),
            float(flip.get("effective_denominator") or 0),
        )

    scored = [(cand["candidate_id"], axes(cand)) for cand in candidates]
    selected: list[str] = []
    for cid, a in scored:
        dominated = False
        for other_cid, b in scored:
            if other_cid == cid:
                continue
            if all(b[i] >= a[i] for i in range(len(a))) and any(b[i] > a[i] for i in range(len(a))):
                dominated = True
                break
        if not dominated:
            selected.append(cid)
    return selected


def mine_seed_flip_inversion_candidates(  # noqa: PLR0913
    rows: Sequence[dict[str, Any]],
    *,
    outcome_metric: str = "success",
    group_by: str = DEFAULT_REPORT_GROUP_BY,
    fallback_group_by: str = DEFAULT_REPORT_FALLBACK_GROUP_BY,
    seed_field: str = "seed",
    conf: float = DEFAULT_CONF,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    triage_max_seeds: int = DEFAULT_TRIAGE_MAX_SEEDS,
    min_heldout_cells: int = DEFAULT_MIN_HELDOUT_CELLS,
    require_native: bool = True,
    external: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mine seed-flip and held-out planner-inversion candidates from result rows.

    Args:
        rows: Benchmark result rows (episode or per-seed summary). Each needs the
            provenance fields in :data:`REQUIRED_PROVENANCE_FIELDS`, a planner
            dimension resolvable via ``group_by``/``fallback_group_by``, and the
            binary ``outcome_metric`` (under ``metrics`` or a top-level path).
        outcome_metric: Binary outcome metric name (default ``success``).
        group_by: Primary planner-identifying dotted path.
        fallback_group_by: Fallback planner-identifying dotted path.
        seed_field: Dotted path to the seed field.
        conf: Confidence level for the Wilson interval.
        min_seeds: Minimum distinct seeds before a cell can flip.
        triage_max_seeds: Cells at/under this seed count are flagged triage-only.
        min_heldout_cells: Minimum leave-one-out cells for a trusted strength.
        require_native: Reject non-native execution rows when ``True``.
        external: Optional sibling-issue signal tables (``oracle_regret`` #5302,
            ``transfer`` #5303, ``quality_diversity`` #5308, ``multiplicity``
            #5351). Absent signals are reported ``unavailable``, never fabricated.

    Returns:
        A schema-versioned candidate manifest (see module docstring).

    Raises:
        SeedFlipMiningError: When ``rows`` is empty or no row survives the
            eligibility gates (fail closed rather than report an empty claim).
    """
    if not rows:
        raise SeedFlipMiningError(
            "seed-flip miner requires at least one result row; refusing to mine candidates "
            "with no evidence"
        )
    if not (0.0 < conf < 1.0):
        raise SeedFlipMiningError(f"conf must be in (0, 1), got {conf}")
    if min_seeds < 2:
        raise SeedFlipMiningError(f"min_seeds must be >= 2 to observe a flip, got {min_seeds}")

    cells, exclusions = _build_cells(
        rows,
        outcome_metric=outcome_metric,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        seed_field=seed_field,
        require_native=require_native,
    )
    if not cells:
        raise SeedFlipMiningError(
            "no eligible rows survived the evidence gates; refusing to mine candidates. "
            f"excluded {len(exclusions)} rows (see exclusion reasons)"
        )

    disagreement = _disagreement_by_scenario(cells)

    seed_flips = _seed_flip_candidates(
        cells,
        conf=conf,
        min_seeds=min_seeds,
        triage_max_seeds=triage_max_seeds,
        external=external,
    )
    upsets = _planner_upset_candidates(
        cells, min_heldout_cells=min_heldout_cells, external=external
    )

    # Attach cross-planner disagreement entropy to every candidate by scenario so
    # it is available for Pareto selection and the disagreement/recovery archetype.
    candidates = seed_flips + upsets
    for cand in candidates:
        entry = disagreement.get(cand["scenario_id"])
        if entry is not None:
            cand["cross_planner_disagreement_entropy"] = entry["entropy_bits"]

    selected_ids = set(_pareto_select(candidates))

    # Archetype availability: seed_flip / planner_upset are data-driven; causal
    # divergence needs a temporal boundary margin signal; disagreement/recovery
    # needs a scenario where planners genuinely disagree.
    has_boundary = any(c.get("temporal_boundary_margin") is not None for c in seed_flips)
    max_disagreement = max((d["entropy_bits"] for d in disagreement.values()), default=0.0)
    archetype_availability = {
        "seed_flip": {
            "available": bool(seed_flips),
            "n_candidates": len(seed_flips),
        },
        "planner_upset": {
            "available": bool(upsets),
            "n_candidates": len(upsets),
        },
        "causal_divergence": {
            "available": has_boundary,
            "reason": (
                "temporal_boundary_margin signal present"
                if has_boundary
                else "no temporal_boundary_margin signal in eligible rows"
            ),
        },
        "disagreement_recovery": {
            "available": max_disagreement > 0.0,
            "max_disagreement_entropy_bits": max_disagreement,
            "reason": (
                "planners disagree on at least one scenario"
                if max_disagreement > 0.0
                else "no cross-planner disagreement among eligible cells"
            ),
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": "#5446",
        "claim_boundary": (
            "Analysis tooling: seed-flip / held-out planner-inversion CASE CANDIDATES only. "
            "Not a benchmark metric, not a planner-ranking claim. Confirmation runs are a "
            "separate exact-compute packet. Candidates below the triage seed count or without "
            "held-out strength are flagged; they are proposals, not established effects."
        ),
        "params": {
            "outcome_metric": outcome_metric,
            "group_by": group_by,
            "fallback_group_by": fallback_group_by,
            "seed_field": seed_field,
            "conf": conf,
            "min_seeds": min_seeds,
            "triage_max_seeds": triage_max_seeds,
            "min_heldout_cells": min_heldout_cells,
            "require_native": require_native,
        },
        "external_signals": {
            name: {
                "consumes_issue": issue,
                "provided": bool(external and external.get(name) is not None),
            }
            for name, issue in EXTERNAL_SIGNAL_ISSUES.items()
        },
        "summary": {
            "n_rows": len(rows),
            "n_eligible_cells": len(cells),
            "n_excluded_rows": len(exclusions),
            "n_candidates": len(candidates),
            "n_seed_flip_candidates": len(seed_flips),
            "n_planner_upset_candidates": len(upsets),
            "n_selected": len(selected_ids),
        },
        "archetype_availability": archetype_availability,
        "cross_planner_disagreement": dict(sorted(disagreement.items())),
        # Every eligible candidate is recorded before diversity selection; the
        # ``selected`` flag marks the Pareto frontier (issue #5446 acceptance).
        "candidates": [
            {**cand, "selected": cand["candidate_id"] in selected_ids} for cand in candidates
        ],
        "exclusions": exclusions,
    }
