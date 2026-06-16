#!/usr/bin/env python3
"""Validate a ForecastBatch.v1 JSON artifact and emit a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.forecast_batch import validate_forecast_batch


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate a ForecastBatch.v1 JSON artifact against the typed contract."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the ForecastBatch.v1 JSON file to validate.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help=(
            "Perform only JSON Schema structural validation; skip the stricter "
            "dataclass contract checks (shape alignment, oracle gating, etc.)."
        ),
    )
    return parser


def _load_batch(path: Path) -> dict[str, Any]:
    """Load and minimally check the input JSON object."""
    if not path.exists():
        raise ValueError(f"input file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("forecast batch JSON must be an object")
    return data


def main(argv: list[str] | None = None) -> int:
    """Validate a ForecastBatch artifact and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        data = _load_batch(args.input)
        if args.schema_only:
            from robot_sf.benchmark.schemas.forecast_batch_schema import ForecastBatchSchema

            schema_path = (
                Path(__file__).resolve().parents[2]
                / "robot_sf"
                / "benchmark"
                / "schemas"
                / "forecast_batch.schema.v1.json"
            )
            schema = ForecastBatchSchema(schema_path)
            schema.validate_forecast_batch_data(data)
        else:
            validate_forecast_batch(data)
    except Exception as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        return 1

    print(
        json.dumps(
            {
                "status": "valid",
                "schema_version": data.get("schema_version"),
                "predictor_id": data.get("provenance", {}).get("predictor_id"),
                "scenario_id": data.get("provenance", {}).get("scenario_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
