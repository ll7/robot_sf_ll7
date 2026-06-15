#!/usr/bin/env python3
"""Record a tiny ForecastDataset.v1 artifact from simulation trace exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.forecast_dataset_recorder import (
    DEFAULT_FORECAST_DATASET_ID,
    record_forecast_dataset_from_trace_exports,
)


def main() -> None:
    """Run the forecast dataset recorder CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path, help="simulation_trace_export.v1 JSON files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_FORECAST_DATASET_ID)
    parser.add_argument(
        "--feature-schema-name",
        default="forecast_dataset_recorder_oracle_state_v1",
    )
    parser.add_argument("--horizon-s", type=float, action="append")
    args = parser.parse_args()

    result = record_forecast_dataset_from_trace_exports(
        args.trace,
        args.output_dir,
        dataset_id=args.dataset_id,
        feature_schema={
            "name": args.feature_schema_name,
            "features": ["position_m", "velocity_mps"],
        },
        horizons_s=args.horizon_s or [0.1],
    )
    print(
        json.dumps(
            {
                "dataset_path": str(result.dataset_path),
                "manifest_path": str(result.manifest_path),
                "example_count": result.manifest["example_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
