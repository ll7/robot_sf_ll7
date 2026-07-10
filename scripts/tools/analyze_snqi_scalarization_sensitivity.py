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
    SENSITIVITY_PREFLIGHT_READY,
    build_scalarization_sensitivity_report,
    classify_scalarization_sensitivity_inputs,
    input_file_provenance,
    load_admissible_weight_family_config,
    load_baseline_mapping,
    load_jsonl,
    load_weight_mapping,
    missing_input_preflight,
    write_diagnostic_artifacts,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="Episode JSONL input.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional SNQI weights JSON.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional SNQI baseline JSON.")
    parser.add_argument(
        "--weight-family-config",
        type=Path,
        default=None,
        help="Optional ex-ante admissible weight-family YAML; labels stress probes separately.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for artifacts. Required unless --preflight-only is set.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Classify input readiness without writing scalarization artifacts.",
    )
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
    parser.add_argument(
        "--application-packet",
        type=Path,
        default=None,
        help="Optional tracked packet proving the campaign application surface being gated.",
    )
    parser.add_argument(
        "--preflight-out",
        type=Path,
        default=None,
        help="Optional JSON path for the readiness preflight payload.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run scalarization-sensitivity export.

    Returns:
        Process exit code.
    """
    args = _parse_args(argv)
    if not args.episodes.exists():
        preflight = missing_input_preflight(
            args.episodes,
            input_kind="episodes",
            application_packet=args.application_packet,
        )
        if args.preflight_out is not None:
            args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_out.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        print(json.dumps(preflight, indent=2), file=sys.stderr)
        return 2

    records = load_jsonl(args.episodes)
    weights = load_weight_mapping(args.weights)
    baseline = load_baseline_mapping(args.baseline)
    weight_family = (
        load_admissible_weight_family_config(args.weight_family_config)
        if args.weight_family_config is not None
        else None
    )
    provenance = {
        "episodes": input_file_provenance(args.episodes),
        "weights": input_file_provenance(args.weights),
        "baseline": input_file_provenance(args.baseline),
        "weight_family_config": input_file_provenance(args.weight_family_config),
    }
    if args.application_packet is not None:
        provenance["application_packet"] = input_file_provenance(args.application_packet)

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
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["status"] == SENSITIVITY_PREFLIGHT_READY else 2

    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --preflight-only is set")

    if preflight["status"] != SENSITIVITY_PREFLIGHT_READY:
        print(json.dumps(preflight, indent=2))
        return 2

    report = build_scalarization_sensitivity_report(
        records,
        weights=weights,
        baseline=baseline,
        planner_key=args.planner_key,
        fallback_planner_key=args.fallback_planner_key,
        sweep_factors=args.sweep_factors,
        admissible_weight_family=weight_family,
        input_provenance=provenance,
    )
    artifacts = write_diagnostic_artifacts(report, args.output_dir)
    print(
        json.dumps(
            {
                "json": str(artifacts.json_path),
                "csv": str(artifacts.csv_path),
                "decision_disagreement_csv": str(artifacts.decision_disagreement_csv_path),
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
