#!/usr/bin/env python3
"""Export SNQI scalarization-sensitivity diagnostics from episode JSONL.

The outputs are analysis artifacts only. They expose weight-zero/sweep rank
reversals, term dominance, decision disagreement against a constraints-first
ordering, and a Pareto SVG without changing benchmark metrics or claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    DEFAULT_SWEEP_FACTORS,
    build_scalarization_sensitivity_report,
    load_baseline_mapping,
    load_jsonl,
    load_weight_mapping,
    write_diagnostic_artifacts,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="Episode JSONL input.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional SNQI weights JSON.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional SNQI baseline JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for artifacts.")
    parser.add_argument("--planner-key", default="planner_key", help="Dotted planner key.")
    parser.add_argument(
        "--fallback-planner-key", default="planner", help="Fallback dotted planner key."
    )
    parser.add_argument(
        "--sweep-factors",
        nargs="+",
        type=float,
        default=list(DEFAULT_SWEEP_FACTORS),
        help="Multipliers applied one SNQI weight at a time.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run scalarization-sensitivity export.

    Returns:
        Process exit code.
    """

    args = _parse_args(argv)
    report = build_scalarization_sensitivity_report(
        load_jsonl(args.episodes),
        weights=load_weight_mapping(args.weights),
        baseline=load_baseline_mapping(args.baseline),
        planner_key=args.planner_key,
        fallback_planner_key=args.fallback_planner_key,
        sweep_factors=args.sweep_factors,
    )
    artifacts = write_diagnostic_artifacts(report, args.output_dir)
    print(
        json.dumps(
            {
                "json": str(artifacts.json_path),
                "csv": str(artifacts.csv_path),
                "markdown": str(artifacts.markdown_path),
                "svg": str(artifacts.svg_path),
                "decision_disagreement_rate": report["summary"]["decision_disagreement_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
