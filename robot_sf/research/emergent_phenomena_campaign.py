"""Multi-seed measured campaign over the emergent-phenomena demonstration.

Issue robot_sf_ll7#5149 follow-up (maintainer authorization 2026-08-11):
elevate the pinned single-seed exhibit
(``docs/context/evidence/issue_5149_emergent_phenomena_2026-07/``) to
*measured* face-validity evidence by running the same canonical scenarios
(bidirectional corridor, narrow doorway, high-density exit) at the same two
speed calibrations (released default ~0.65 m/s, literature-typical ~1.3 m/s)
across a pinned list of seeds, and aggregating the order parameters into
per-scenario/per-calibration statistics with verdict distributions.

This module holds the campaign runner and the aggregation logic; the thin
orchestrator that writes the archived evidence bundle (run records, summary,
provenance manifest, figures, SHA256SUMS) is
``scripts/validation/build_issue_5149_emergent_phenomena_campaign.py``.

Everything is deterministic given the seed list: seeds only affect pedestrian
placement (and the literature calibration's per-pedestrian speed draw), via
``numpy.random.default_rng(seed)`` inside the scenario builders.

Claim boundary: this remains diagnostic face-validity evidence for THIS
implementation at the pinned parameterizations -- multi-seed measurement, not
benchmark-matrix evidence and not paper-grade validation against real human
trajectory data (robot_sf_ll7#4975).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.research.emergent_phenomena import (
    LITERATURE_CALIBRATION,
    RELEASED_DEFAULT_CALIBRATION,
    default_scenario_set,
    derive_phenomenon_verdict,
    run_scenario,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pysocialforce.config import SimulatorConfig

    from robot_sf.research.emergent_phenomena import (
        ScenarioConfig,
        ScenarioResult,
        SpeedCalibration,
    )

__all__ = [
    "DEFAULT_CAMPAIGN_SEEDS",
    "VERDICT_SEVERITY",
    "aggregate_run_records",
    "result_to_run_record",
    "run_multiseed_campaign",
]

# Pinned campaign seed list. The first seed (5149) is the pinned seed of the
# 2026-07 single-seed exhibit, kept first for continuity with that anchor.
DEFAULT_CAMPAIGN_SEEDS: tuple[int, ...] = tuple(range(5149, 5159))

# Verdict labels ordered weakest-first; aggregation tie-breaks toward the
# weaker verdict so the campaign never overclaims on a split seed population.
VERDICT_SEVERITY: tuple[str, ...] = (
    "absent_or_negligible",
    "weak_partial",
    "clearly_present",
)


def result_to_run_record(result: ScenarioResult) -> dict[str, Any]:
    """Serialize one seeded scenario run into an episode-record-style dict.

    One record per scenario x calibration x seed; these are the rows written
    to the campaign bundle's ``runs.jsonl``.

    Args:
        result: A single :func:`run_scenario` result.

    Returns:
        JSON-serializable record including order parameters and the verdict.
    """
    ops = {k: float(v) for k, v in result.order_parameters.items()}
    return {
        "record_type": "emergent_phenomena_run.v1",
        "scenario": result.scenario.name,
        "calibration": result.calibration.name,
        "seed": int(result.scenario.seed),
        "desired_speed_mean_mps": float(result.calibration.desired_speed_mean),
        "desired_speed_std_mps": float(result.calibration.desired_speed_std),
        "realized_mean_max_speed_mps": float(result.max_speeds.mean()),
        "n_pedestrians": int(result.scenario.n_pedestrians),
        "n_steps": int(result.scenario.n_steps),
        "dt_secs": float(result.trajectory.dt),
        "duration_secs": float(result.scenario.n_steps * result.trajectory.dt),
        "scenario_extra": dict(result.scenario.extra),
        "order_parameters": ops,
        "phenomenon_verdict": derive_phenomenon_verdict(result.scenario.name, ops),
    }


def run_multiseed_campaign(
    seeds: Sequence[int] | None = None,
    scenarios: Sequence[ScenarioConfig] | None = None,
    calibrations: Sequence[SpeedCalibration] | None = None,
    sim_config: SimulatorConfig | None = None,
) -> list[ScenarioResult]:
    """Run every scenario x calibration x seed combination.

    Deterministic given the seed list; results are ordered scenario-major,
    then calibration, then seed (matching the loop nesting).

    Args:
        seeds: Seed list; defaults to :data:`DEFAULT_CAMPAIGN_SEEDS`.
        scenarios: Scenario set; defaults to the canonical demonstration set
            (each scenario's own seed field is replaced per campaign seed).
        calibrations: Speed calibrations; defaults to released + literature.
        sim_config: Optional simulator config; defaults to released defaults.

    Returns:
        Flat list of per-run :class:`ScenarioResult` objects.
    """
    if seeds is None:
        seeds = DEFAULT_CAMPAIGN_SEEDS
    if scenarios is None:
        scenarios = default_scenario_set()
    if calibrations is None:
        calibrations = [RELEASED_DEFAULT_CALIBRATION, LITERATURE_CALIBRATION]
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        for calibration in calibrations:
            for seed in seeds:
                seeded = replace(scenario, seed=int(seed))
                results.append(run_scenario(seeded, calibration, sim_config=sim_config))
    return results


def _order_parameter_stats(values: Sequence[float]) -> dict[str, float]:
    """Compute summary statistics for one order parameter across seeds.

    Returns:
        Dict with ``mean``, ``std`` (sample, 0.0 for n=1), ``min``, ``median``,
        ``max``.
    """
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _majority_verdict(verdict_counts: Counter[str]) -> str:
    """Pick the most common verdict, tie-breaking toward the weaker label.

    Returns:
        The majority verdict label.
    """

    def sort_key(item: tuple[str, int]) -> tuple[int, int]:
        label, count = item
        severity = (
            VERDICT_SEVERITY.index(label) if label in VERDICT_SEVERITY else len(VERDICT_SEVERITY)
        )
        return (-count, severity)

    return min(verdict_counts.items(), key=sort_key)[0]


def aggregate_run_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed run records into scenario x calibration statistics.

    Args:
        records: Run records as produced by :func:`result_to_run_record`.

    Returns:
        One aggregate dict per (scenario, calibration) group, sorted by group
        key, with order-parameter statistics, verdict counts, and the
        majority verdict (ties broken toward the weaker verdict).
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((rec["scenario"], rec["calibration"]), []).append(rec)

    aggregates: list[dict[str, Any]] = []
    for (scenario, calibration), recs in sorted(groups.items()):
        op_names = sorted({name for rec in recs for name in rec["order_parameters"]})
        op_stats = {
            name: _order_parameter_stats(
                [
                    float(rec["order_parameters"][name])
                    for rec in recs
                    if name in rec["order_parameters"]
                ]
            )
            for name in op_names
        }
        verdict_counts = Counter(rec["phenomenon_verdict"] for rec in recs)
        aggregates.append(
            {
                "scenario": scenario,
                "calibration": calibration,
                "n_seeds": len(recs),
                "seeds": sorted(int(rec["seed"]) for rec in recs),
                "order_parameter_stats": op_stats,
                "verdict_counts": dict(sorted(verdict_counts.items())),
                "majority_verdict": _majority_verdict(verdict_counts),
            }
        )
    return aggregates
