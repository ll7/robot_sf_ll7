#!/usr/bin/env python3
"""Build a ForecastConformalPilot.v1 artifact from ForecastBatch.v1 JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.forecast_conformal_pilot import (
    build_forecast_conformal_pilot_report,
    write_forecast_conformal_pilot_report,
)


def main() -> None:
    """Run the conformal pilot CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-batch", nargs="+", type=Path, required=True)
    parser.add_argument("--calibration-ground-truth", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluation-batch", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluation-ground-truth", nargs="+", type=Path, required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--coverage-target", type=float, default=0.9)
    args = parser.parse_args()

    try:
        calibration_cases = _cases_from_paths(
            args.calibration_batch,
            args.calibration_ground_truth,
            split_id="calibration",
        )
        evaluation_cases = _cases_from_paths(
            args.evaluation_batch,
            args.evaluation_ground_truth,
            split_id="heldout_evaluation",
        )
        report = build_forecast_conformal_pilot_report(
            calibration_cases,
            evaluation_cases,
            report_id=args.report_id,
            coverage_target=args.coverage_target,
        )
        paths = write_forecast_conformal_pilot_report(
            report,
            json_path=args.out_json,
            markdown_path=args.out_md,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "json_path": str(paths["json"]),
                "markdown_path": str(paths.get("markdown", "")),
                "pilot_rows": len(report["pilot_rows"]),
                "limitation_rows": len(report["limitation_rows"]),
                "decision": report["recommendation"]["decision"],
                "claim_status": report["recommendation"]["claim_status"],
            },
            sort_keys=True,
        )
    )


def _cases_from_paths(
    batch_paths: list[Path],
    truth_paths: list[Path],
    *,
    split_id: str,
) -> list[dict[str, object]]:
    if len(batch_paths) != len(truth_paths):
        raise ValueError(f"{split_id} batch and ground-truth counts must match")
    return [
        {
            "batch": _read_json(batch_path, kind=f"{split_id} batch"),
            "ground_truth": _read_json(truth_path, kind=f"{split_id} ground truth"),
            "split_id": split_id,
        }
        for batch_path, truth_path in zip(batch_paths, truth_paths, strict=True)
    ]


def _read_json(path: Path, *, kind: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {kind} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse {kind} {path}: {exc}") from exc


if __name__ == "__main__":
    main()
