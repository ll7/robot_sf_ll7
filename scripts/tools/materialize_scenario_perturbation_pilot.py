#!/usr/bin/env python3
"""Materialize preflight-eligible scenario perturbations into a local pilot matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.scenario_certification import materialize_perturbation_pilot_matrix


def _build_parser() -> argparse.ArgumentParser:
    """Build the perturbation pilot materialization parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Scenario perturbation manifest YAML.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for the generated scenario matrix and route overrides. "
            "Repository-local outputs must be under output/."
        ),
    )
    parser.add_argument(
        "--seed-limit",
        type=int,
        help="Optional positive cap on seeds per materialized variant for local smoke pilots.",
    )
    return parser


def main() -> int:
    """Run perturbation pilot materialization."""
    args = _build_parser().parse_args()
    result = materialize_perturbation_pilot_matrix(
        args.manifest,
        output_dir=args.output_dir,
        seed_limit=args.seed_limit,
    )
    print(
        json.dumps(
            {
                "schema_version": result.schema_version,
                "manifest_id": result.manifest_id,
                "scenario_matrix_path": result.scenario_matrix_path,
                "summary_path": result.summary_path,
                "included_variants": list(result.included_variants),
                "excluded_variants": list(result.excluded_variants),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
