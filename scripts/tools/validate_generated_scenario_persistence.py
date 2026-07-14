#!/usr/bin/env python3
"""Validate ``generated_scenario_persistence.v1`` records and compute promotion verdicts.

Usage:
    uv run python scripts/tools/validate_generated_scenario_persistence.py --help
    uv run python scripts/tools/validate_generated_scenario_persistence.py record.json
    uv run python scripts/tools/validate_generated_scenario_persistence.py --batch *.json

The tool validates a prebuilt record against the versioned schema and the
fail-closed promotion invariants, then prints the verdict.  It does not execute
replays or perturbations; those are produced by the gate writer.  This keeps the
validation step cheap and CPU-only so it can run in CI without a simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.scenario_generation.persistence_gate import (
    PERSISTENCE_SCHEMA_VERSION,
    ScenarioPersistenceValidationError,
    validate_persistence_record,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records",
        type=Path,
        nargs="*",
        help="One or more generated_scenario_persistence.v1 JSON records to validate.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat every positional path as part of one batch and exit non-zero on any failure.",
    )
    parser.add_argument(
        "--schema-version",
        action="store_true",
        help="Print the expected schema version and exit.",
    )
    return parser


def _validate_one(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_persistence_record(payload)
    return payload


def main() -> int:
    """Validate persistence records and report promotion verdicts."""

    args = _build_parser().parse_args()
    if args.schema_version:
        print(PERSISTENCE_SCHEMA_VERSION)
        return 0
    if not args.records:
        print("error: no record paths supplied", file=sys.stderr)
        return 2

    failures = 0
    for path in args.records:
        try:
            record = _validate_one(path)
        except (OSError, ValueError, ScenarioPersistenceValidationError) as exc:
            failures += 1
            print(f"INVALID {path}: {exc}", file=sys.stderr)
            continue
        verdict = record["promotion"]["verdict"]
        print(f"{verdict.upper()} {path} :: {record['promotion']['exclusion_reason']}")
        if verdict != "promote":
            failures += 1

    if args.batch:
        return 2 if failures else 0
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
