#!/usr/bin/env python3
"""Sweep stream_gap uncertainty-gate thresholds for safe agent-dropping (#3558).

#3471 found that dropping uncertain agents at the **current default** ``stream_gap``
uncertainty-gate thresholds increases unsafe commitment. That is a finding about the
defaults, not about gating in general. This script runs the missing **bounded threshold
sweep** the issue asks for: it reuses the #3471 episode harness
(``run_scenario_belief_episode_safety_issue_3471``) and rolls the ``uncertain_dropped``
mode once per swept gate setting, then feeds the per-setting safety aggregates and the
conservative-retention baseline into the merged pure decision layer
``robot_sf.planner.stream_gap_gate_calibration.calibrate_stream_gap_gate``.

The output is the issue deliverable: per-setting ``at_least_as_safe`` / ``less_safe``
classifications, the safe operating region (if any), a recommended setting, or the
``conservative_retention_dominates`` conclusion.

**Active axis / claim boundary.** The #3471 scenario degrades only the corridor agent's
*existence* confidence (to 0.2), so this sweep exercises the
``uncertainty_min_existence_probability`` gate axis; the other three thresholds are
recorded and passed through but are not exercised by this scenario's single-source
degradation. The gate drops an agent when ``existence < min_existence_probability``
(strict), so a threshold at or below 0.2 retains the degraded agent (behaviorally
conservative retention) while a threshold above 0.2 drops it.

This is **diagnostic-tier, controlled-scenario evidence** — the real stream_gap planner +
ScenarioBelief gate on one synthetic crossing scenario, not the full benchmark
environment, not paper-grade, and no trained-policy or traffic-realism claim. It does not
change the production default; that is a separate decision once a safe region is or is not
found.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.planner.stream_gap import StreamGapPlannerConfig
from robot_sf.planner.stream_gap_gate_calibration import (
    GateSettingResult,
    SafetyTolerance,
    calibrate_stream_gap_gate,
)
from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (
    EpisodeParams,
    run_episode,
)

SCHEMA_VERSION = "stream-gap-gate-threshold-sweep.v1"
ISSUE = 3558

#: Predeclared existence-probability gate grid. The degraded corridor agent sits at
#: existence 0.2, so ``<= 0.2`` retains it (safe) and ``> 0.2`` drops it (the #3471 finding).
DEFAULT_EXISTENCE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7)

#: Default values for the gate thresholds the scenario does not exercise (recorded for
#: a self-describing report).
_DEFAULTS = StreamGapPlannerConfig()
_PASSTHROUGH_THRESHOLDS = {
    "uncertainty_min_position_confidence": float(_DEFAULTS.uncertainty_min_position_confidence),
    "uncertainty_min_class_probability": float(_DEFAULTS.uncertainty_min_class_probability),
    "uncertainty_max_position_variance": float(_DEFAULTS.uncertainty_max_position_variance),
}


def _aggregate(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> GateSettingResult:
    """Aggregate per-seed episode rows for one gate setting into a ``GateSettingResult``.

    Safety axes (lower is safer except separation):

    * ``unsafe_commit_rate`` — fraction of episodes with any unsafe commitment;
    * ``collision_rate`` — fraction of episodes with a collision;
    * ``min_separation_m`` — worst (smallest) true surface separation across episodes
      (the conservative, fail-closed choice).

    Returns:
        GateSettingResult: The per-setting safety aggregates.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("cannot aggregate an empty set of episode rows")
    unsafe_commit_rate = sum(1 for row in rows if row["unsafe_commit_steps"] > 0) / n
    collision_rate = sum(1 for row in rows if row["collision"]) / n
    worst_min_separation = min(row["min_separation"] for row in rows)
    return GateSettingResult(
        thresholds=thresholds,
        unsafe_commit_rate=round(unsafe_commit_rate, 4),
        collision_rate=round(collision_rate, 4),
        min_separation_m=round(float(worst_min_separation), 4),
    )


def run_sweep(
    seeds: list[int],
    params: EpisodeParams,
    existence_grid: tuple[float, ...] = DEFAULT_EXISTENCE_GRID,
    tolerance: SafetyTolerance | None = None,
) -> dict[str, Any]:
    """Run the gate-threshold sweep and assemble the calibrated report.

    The baseline is the ``uncertain_retained`` mode (conservative retention, gate off).
    Each swept setting is the ``uncertain_dropped`` mode with gating on at the given
    existence threshold.

    Returns:
        dict[str, Any]: Versioned report with the swept settings, the baseline, and the
        :func:`calibrate_stream_gap_gate` decision.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    tolerance = tolerance or SafetyTolerance()

    baseline_rows = [run_episode("uncertain_retained", seed, params) for seed in seeds]
    baseline = _aggregate(baseline_rows, thresholds={})

    settings: list[GateSettingResult] = []
    for existence in existence_grid:
        thresholds = {
            "uncertainty_min_existence_probability": float(existence),
            **_PASSTHROUGH_THRESHOLDS,
        }
        override = {"uncertainty_min_existence_probability": float(existence)}
        rows = [
            run_episode("uncertain_dropped", seed, params, gate_thresholds=override)
            for seed in seeds
        ]
        settings.append(_aggregate(rows, thresholds=thresholds))

    calibration = calibrate_stream_gap_gate(settings, baseline, tolerance)
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "evidence_tier": "diagnostic",
        "claim_boundary": (
            "Bounded existence-probability gate-threshold sweep on the #3471 controlled "
            "crossing scenario with the real stream_gap planner + ScenarioBelief gate. Only "
            "the existence axis is exercised (the scenario degrades existence alone); not the "
            "full benchmark environment, not paper-grade, no trained-policy or traffic-realism "
            "claim. Does not change the production default."
        ),
        "active_gate_axis": "uncertainty_min_existence_probability",
        "degraded_existence": 0.2,
        "seeds": seeds,
        "seed_count": len(seeds),
        "existence_grid": list(existence_grid),
        "params": vars(params),
        "baseline_conservative_retention": {
            "unsafe_commit_rate": baseline.unsafe_commit_rate,
            "collision_rate": baseline.collision_rate,
            "min_separation_m": baseline.min_separation_m,
        },
        "calibration": calibration,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--existence-grid",
        type=float,
        nargs="+",
        default=None,
        help="Existence-probability thresholds to sweep (default: 0.1 0.2 0.3 0.5 0.7).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the sweep and emit the calibrated report."""
    args = parse_args(argv)
    from dataclasses import replace

    seeds = args.seeds if args.seeds is not None else list(range(101, 113))
    params = EpisodeParams()
    if args.max_steps is not None:
        params = replace(params, max_steps=args.max_steps)
    grid = tuple(args.existence_grid) if args.existence_grid is not None else DEFAULT_EXISTENCE_GRID

    report = run_sweep(seeds, params, existence_grid=grid)
    report["generated_at_utc"] = datetime.now(UTC).isoformat()

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
