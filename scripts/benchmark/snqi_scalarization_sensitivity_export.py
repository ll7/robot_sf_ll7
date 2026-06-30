#!/usr/bin/env python3
"""Fail-closed SNQI scalarization-sensitivity export for issue #3653.

The command runs the Social Navigation Quality Index (SNQI) readiness preflight
before writing diagnostic-only scalarization-sensitivity artifacts. It refuses
missing or malformed normalized inputs instead of producing report artifacts
from incomplete fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    SENSITIVITY_PREFLIGHT_READY,
    build_scalarization_sensitivity_report,
    classify_scalarization_sensitivity_inputs,
    input_file_provenance,
    load_baseline_mapping,
    load_jsonl,
    load_weight_mapping,
    write_diagnostic_artifacts,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path, help="Input episode JSONL.")
    parser.add_argument("--baseline", required=True, type=Path, help="SNQI baseline stats JSON.")
    parser.add_argument("--weights", required=True, type=Path, help="SNQI weights JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Artifact output directory.")
    parser.add_argument(
        "--planner-key",
        default="planner_key",
        help="Dotted planner key; falls back to --fallback-planner-key.",
    )
    parser.add_argument(
        "--fallback-planner-key",
        default="planner",
        help="Dotted fallback planner key; scenario_params.algo is tried last.",
    )
    parser.add_argument("--stem", default="snqi_scalarization_sensitivity")
    parser.add_argument(
        "--preflight-out",
        type=Path,
        help="Optional JSON path for the readiness preflight payload.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed preflight and export diagnostic artifacts when ready."""

    args = _build_parser().parse_args(argv)

    records = load_jsonl(args.episodes)
    baseline = load_baseline_mapping(args.baseline)
    weights = load_weight_mapping(args.weights)
    provenance = {
        "episodes": input_file_provenance(args.episodes),
        "baseline": input_file_provenance(args.baseline),
        "weights": input_file_provenance(args.weights),
    }

    preflight = classify_scalarization_sensitivity_inputs(
        records,
        weights=weights,
        baseline=baseline,
        planner_key=args.planner_key,
        fallback_planner_key=args.fallback_planner_key,
    )
    if args.preflight_out is not None:
        args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_out.write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    if preflight["status"] != SENSITIVITY_PREFLIGHT_READY:
        print(json.dumps(preflight, indent=2), file=sys.stderr)
        return 2

    report = build_scalarization_sensitivity_report(
        records,
        weights=weights,
        baseline=baseline,
        planner_key=args.planner_key,
        fallback_planner_key=args.fallback_planner_key,
        input_provenance=provenance,
    )
    artifacts = write_diagnostic_artifacts(report, args.output_dir, stem=args.stem)
    print(
        json.dumps(
            {
                "status": "exported",
                "claim_boundary": report["claim_boundary"],
                "artifacts": {
                    "json": str(artifacts.json_path),
                    "csv": str(artifacts.csv_path),
                    "markdown": str(artifacts.markdown_path),
                    "svg": str(artifacts.svg_path),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
