"""Scenario flakiness audit for the Social Navigation Benchmark.

This module measures two distinct forms of outcome instability from benchmark
episode records (see issue #4978):

1. **Exact-repeat determinism** — when the same ``(scenario, planner, seed)``
   cell is executed more than once, do the repeated runs produce the *same*
   binary outcome? If not, exact-repeat runs are non-deterministic, which is a
   reproducibility bug the audit doubles as a detector for.
2. **Per-cell outcome stability** — within a ``(scenario, planner)`` cell, how
   consistently do independent seeds agree on the binary outcome? A cell whose
   seeds flip between success and failure is *knife-edge*: campaign comparisons
   on it carry hidden variance that per-seed confidence intervals understate.

The audit is intentionally advisory and read-only: it emits a schema-versioned
report describing stability, and never mutates existing rankings, campaign
summaries, or metric semantics. Downstream tooling may *choose* to exclude or
down-weight flagged knife-edge cells, but that decision stays out of scope here.

The audit fails closed: an empty record set raises, cells with too few seeds are
reported as *not assessable* rather than silently counted as stable, and
determinism is reported as ``None`` (unknown) when no exact-repeat data exists
instead of being asserted without evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.grouping import (
    DEFAULT_REPORT_FALLBACK_GROUP_BY,
    DEFAULT_REPORT_GROUP_BY,
    get_nested,
    resolve_report_group_key,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = "scenario_flakiness.v1"

#: Default metric treated as the binary episode outcome. ``success`` is already
#: recorded as 0/1 in benchmark episode records.
DEFAULT_OUTCOME_METRIC = "success"

#: Default fraction of seeds that must agree with the majority outcome for a
#: cell to be considered stable. Cells below this are flagged ``knife_edge``.
DEFAULT_STABILITY_THRESHOLD = 0.8

#: Minimum distinct seeds required before a cell's stability can be assessed.
DEFAULT_MIN_SEEDS = 2

#: Cap on how many concrete examples are embedded in the report for each list so
#: the JSON stays compact on large campaigns.
_MAX_EXAMPLES = 25


def _binary_outcome(value: Any) -> int | None:
    """Coerce an outcome metric value into a binary 0/1 label.

    A ``success``-style metric is already 0/1, but records may carry booleans or
    floats (e.g. an averaged success rate). Values ``>= 0.5`` map to 1, finite
    values below map to 0, and non-numeric/missing values map to ``None`` so the
    caller can fail closed instead of guessing.

    Returns:
        ``1``, ``0``, or ``None`` when the value cannot be interpreted.
    """
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else None
    if isinstance(value, int | float):
        f = float(value)
        if math.isnan(f):
            return None
        return 1 if f >= 0.5 else 0
    return None


def _outcome_from_record(record: dict[str, Any], outcome_metric: str) -> int | None:
    """Extract the binary outcome for a record from its metrics block.

    Accepts both ``metrics.<outcome_metric>`` and a top-level ``<outcome_metric>``
    fallback so the audit works on flattened and nested record shapes.

    Returns:
        Binary outcome, or ``None`` when the outcome metric is absent/uninterpretable.
    """
    metrics = record.get("metrics")
    if isinstance(metrics, dict) and outcome_metric in metrics:
        return _binary_outcome(metrics[outcome_metric])
    # Fall back to a dotted/top-level path for already-flattened rows.
    return _binary_outcome(get_nested(record, outcome_metric))


def _seed_key(record: dict[str, Any], seed_field: str) -> str:
    """Return a stable string key for a record's seed (``"__missing__"`` if absent)."""
    value = get_nested(record, seed_field)
    if value is None:
        return "__missing__"
    return str(value)


def _cell_stability(seed_outcomes: dict[str, int]) -> tuple[float, float]:
    """Compute the success rate and stability score for a cell's seed votes.

    Args:
        seed_outcomes: Mapping of seed key to that seed's representative binary outcome.

    Returns:
        ``(success_rate, stability_score)`` where ``success_rate`` is the fraction
        of seeds with outcome 1 and ``stability_score`` is the fraction of seeds
        agreeing with the majority outcome (range ``[0.5, 1.0]``; 1.0 == unanimous).
    """
    n = len(seed_outcomes)
    ones = sum(1 for v in seed_outcomes.values() if v == 1)
    success_rate = ones / n
    stability_score = max(success_rate, 1.0 - success_rate)
    return success_rate, stability_score


def _ingest_records(
    records: Sequence[dict[str, Any]],
    *,
    outcome_metric: str,
    group_by: str,
    fallback_group_by: str,
    seed_field: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, list[int]]], int, int]:
    """Group episode outcomes into ``(scenario, planner)`` cells by seed.

    Returns:
        ``(cell_meta, cell_seed_outcomes, n_missing_outcome, n_missing_scenario)``
        where ``cell_seed_outcomes[cell][seed]`` lists that seed's binary outcomes.
    """
    cell_meta: dict[str, tuple[str, str]] = {}
    cell_seed_outcomes: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    n_missing_outcome = 0
    n_missing_scenario = 0

    for record in records:
        scenario_id = get_nested(record, "scenario_id")
        if scenario_id is None:
            n_missing_scenario += 1
            continue
        planner = (
            resolve_report_group_key(
                record,
                group_by=group_by,
                fallback_group_by=fallback_group_by,
                missing="unknown",
            )
            or "unknown"
        )
        outcome = _outcome_from_record(record, outcome_metric)
        if outcome is None:
            n_missing_outcome += 1
            continue
        cell_key = f"{scenario_id}::{planner}"
        cell_meta[cell_key] = (str(scenario_id), str(planner))
        cell_seed_outcomes[cell_key][_seed_key(record, seed_field)].append(outcome)

    return cell_meta, cell_seed_outcomes, n_missing_outcome, n_missing_scenario


def _audit_cell(
    cell_key: str,
    scenario_id: str,
    planner: str,
    seed_map: dict[str, list[int]],
    *,
    stability_threshold: float,
    min_seeds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    """Score a single cell's stability and detect within-seed non-determinism.

    Returns:
        ``(cell_report, determinism_examples, checked_repeat_groups,
        nondeterministic_repeat_groups)``.
    """
    seed_votes: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    nondeterministic_seeds = 0
    has_within_seed_repeats = False
    checked_repeat_groups = 0
    nondeterministic_repeat_groups = 0
    n_episodes = 0

    for seed_key, outcomes in seed_map.items():
        n_episodes += len(outcomes)
        if len(outcomes) >= 2:
            has_within_seed_repeats = True
            checked_repeat_groups += 1
            if len(set(outcomes)) > 1:
                nondeterministic_repeat_groups += 1
                nondeterministic_seeds += 1
                examples.append(
                    {
                        "scenario_id": scenario_id,
                        "planner": planner,
                        "seed": seed_key,
                        "n_repeats": len(outcomes),
                        "outcomes": list(outcomes),
                    }
                )
        # Representative vote for this seed: majority (ties -> 1, matching
        # `_binary_outcome`'s >= 0.5 rule) so within-seed repeats collapse to one
        # cross-seed vote.
        seed_votes[seed_key] = 1 if (sum(outcomes) / len(outcomes)) >= 0.5 else 0

    n_seeds = len(seed_votes)
    ones = sum(seed_votes.values())
    assessable = n_seeds >= min_seeds
    if assessable:
        success_rate, stability_score = _cell_stability(seed_votes)
        knife_edge = stability_score < stability_threshold
    else:
        success_rate = ones / n_seeds if n_seeds else None
        stability_score = None
        knife_edge = False

    cell_report = {
        "cell_key": cell_key,
        "scenario_id": scenario_id,
        "planner": planner,
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "stability_score": stability_score,
        "knife_edge": knife_edge,
        "assessable": assessable,
        "outcome_counts": {"1": ones, "0": n_seeds - ones},
        "has_within_seed_repeats": has_within_seed_repeats,
        "nondeterministic_seeds": nondeterministic_seeds,
    }
    return cell_report, examples, checked_repeat_groups, nondeterministic_repeat_groups


def compute_flakiness_audit(
    records: Sequence[dict[str, Any]],
    *,
    outcome_metric: str = DEFAULT_OUTCOME_METRIC,
    group_by: str = DEFAULT_REPORT_GROUP_BY,
    fallback_group_by: str = DEFAULT_REPORT_FALLBACK_GROUP_BY,
    seed_field: str = "seed",
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    min_seeds: int = DEFAULT_MIN_SEEDS,
) -> dict[str, Any]:
    """Audit scenario-level outcome flakiness from benchmark episode records.

    Builds ``(scenario, planner)`` cells and, within each cell, groups episodes
    by seed. It measures exact-repeat determinism (repeated runs of the same
    seed) and per-cell outcome stability (agreement across seeds).

    Args:
        records: Episode records with a ``scenario_id``, a planner-identifying
            field resolvable via ``group_by``/``fallback_group_by``, a seed, and
            the binary ``outcome_metric`` (typically under ``metrics``).
        outcome_metric: Metric name treated as the binary outcome.
        group_by: Primary dotted path identifying the planner dimension.
        fallback_group_by: Fallback dotted path when ``group_by`` is missing.
        seed_field: Dotted path to the seed field.
        stability_threshold: Majority-agreement fraction below which a cell is
            flagged ``knife_edge``.
        min_seeds: Minimum distinct seeds before a cell's stability is assessed.

    Returns:
        A schema-versioned audit report (see module docstring for structure).

    Raises:
        ValueError: When ``records`` is empty (an audit of nothing is not
            evidence) or when ``stability_threshold`` is outside ``(0, 1]``.
    """
    if not records:
        raise ValueError(
            "scenario flakiness audit requires at least one episode record; "
            "refusing to report stability with no evidence"
        )
    if not (0.0 < stability_threshold <= 1.0):
        raise ValueError(f"stability_threshold must be in (0, 1], got {stability_threshold}")

    cell_meta, cell_seed_outcomes, n_missing_outcome, n_missing_scenario = _ingest_records(
        records,
        outcome_metric=outcome_metric,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        seed_field=seed_field,
    )

    cells: list[dict[str, Any]] = []
    determinism_examples: list[dict[str, Any]] = []
    checked_repeat_groups = 0
    nondeterministic_repeat_groups = 0

    for cell_key in sorted(cell_seed_outcomes):
        scenario_id, planner = cell_meta[cell_key]
        cell_report, examples, checked, nondeterministic = _audit_cell(
            cell_key,
            scenario_id,
            planner,
            cell_seed_outcomes[cell_key],
            stability_threshold=stability_threshold,
            min_seeds=min_seeds,
        )
        cells.append(cell_report)
        checked_repeat_groups += checked
        nondeterministic_repeat_groups += nondeterministic
        for example in examples:
            if len(determinism_examples) < _MAX_EXAMPLES:
                determinism_examples.append(example)

    assessable_cells = [c for c in cells if c["assessable"]]
    knife_edge_cells = [c for c in assessable_cells if c["knife_edge"]]

    # Determinism verdict fails closed: unknown when no repeat data exists.
    if checked_repeat_groups == 0:
        is_deterministic: bool | None = None
    else:
        is_deterministic = nondeterministic_repeat_groups == 0

    return {
        "schema_version": SCHEMA_VERSION,
        "outcome_metric": outcome_metric,
        "group_by": group_by,
        "fallback_group_by": fallback_group_by,
        "stability_threshold": stability_threshold,
        "min_seeds": min_seeds,
        "exact_repeat": {
            "checked_repeat_groups": checked_repeat_groups,
            "nondeterministic_repeat_groups": nondeterministic_repeat_groups,
            "is_deterministic": is_deterministic,
            "examples": determinism_examples,
        },
        "summary": {
            "n_records": len(records),
            "n_cells": len(cells),
            "n_assessable_cells": len(assessable_cells),
            "n_knife_edge_cells": len(knife_edge_cells),
            "knife_edge_fraction": (
                len(knife_edge_cells) / len(assessable_cells) if assessable_cells else None
            ),
            "n_records_missing_outcome": n_missing_outcome,
            "n_records_missing_scenario": n_missing_scenario,
        },
        "cells": cells,
    }
