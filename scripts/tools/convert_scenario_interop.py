#!/usr/bin/env python3
"""Dry-run Robot SF -> external-benchmark scenario converter CLI (issue #3285).

Reads a Robot SF scenario-matrix file (YAML or JSON; a single scenario dict or a
list of scenario dicts) and emits the deterministic, schema-validated interop
**intermediate representation (IR)** for each scenario, plus an explicit
unsupported-field report.

This is a *dry run*: it requires no external assets and produces no SocNavBench or
HuNavSim file. It makes no cross-benchmark validity claim. See
``robot_sf/benchmark/scenario_interop.py`` for the contract and claim boundary.

Examples:
    # Print IR + unsupported report for every scenario in a matrix
    uv run python scripts/tools/convert_scenario_interop.py \
        --matrix configs/baselines/example_matrix.yaml

    # Write one IR JSON file per scenario into a directory
    uv run python scripts/tools/convert_scenario_interop.py \
        --matrix configs/baselines/example_matrix.yaml --out-dir output/interop_ir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.scenario_interop import (
    SUPPORTED_TARGETS,
    build_target_compatibility_report,
    convert_scenario_to_ir,
    dump_ir,
)


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load scenario dicts from a YAML/JSON matrix file.

    Accepts a single scenario mapping, a top-level list, or a ``{"scenarios": [...]}``
    wrapper.

    Returns:
        Scenario dicts in file order.

    Raises:
        ValueError: When the file does not contain a usable scenario shape.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "scenarios" in raw:
        raw = raw["scenarios"]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
        return raw
    raise ValueError(
        f"{path}: expected a scenario mapping, a list of mappings, or a top-level 'scenarios' list"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the dry-run converter CLI.

    Returns:
        Process exit code: ``0`` when every scenario IR validated, ``1`` otherwise.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="Path to a Robot SF scenario-matrix YAML/JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to write one <scenario_id>.ir.json file per scenario.",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=SUPPORTED_TARGETS,
        default=[],
        help=(
            "Include fail-closed compatibility report for target exporter. "
            "May be passed more than once."
        ),
    )
    args = parser.parse_args(argv)

    scenarios = _load_scenarios(args.matrix)
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    summary: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = convert_scenario_to_ir(scenario, source_file=str(args.matrix))
        scenario_id = result.ir["provenance"]["source_scenario_id"]
        if not result.is_valid:
            exit_code = 1
        summary.append(
            {
                "scenario_id": scenario_id,
                "ir_valid": result.is_valid,
                "unsupported_field_count": len(result.unsupported_fields),
                "schema_errors": result.schema_errors,
                "target_compatibility": build_target_compatibility_report(
                    result.ir,
                    targets=args.target or SUPPORTED_TARGETS,
                ),
            }
        )
        if args.out_dir is not None:
            out_path = args.out_dir / f"{scenario_id}.ir.json"
            out_path.write_text(dump_ir(result.ir), encoding="utf-8")
        else:
            sys.stdout.write(dump_ir(result.ir))

    sys.stderr.write(json.dumps({"dry_run_summary": summary}, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
