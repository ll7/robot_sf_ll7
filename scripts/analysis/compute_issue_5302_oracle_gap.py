#!/usr/bin/env python3
"""CLI runner for the issue #5302 selection ceilings and hierarchical uncertainty analysis.

Loads complete six-arm benchmark rows, validates them against the frozen
pre-registration contract (configs/analysis/issue_5302_oracle_gap_packet.yaml),
computes selection ceilings, hierarchical bootstrap uncertainty, Pareto
dominance probability, normalized regret, and the claim gate status.

Fail-closed policy:
- Incomplete six-arm episodes, non-native rows, invalid row_status, duplicate keys,
  or split leakage block output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.issue_5302_oracle_gap import (
    DEFAULT_PACKET_PATH,
    OracleGapAnalysisError,
    run_full_oracle_gap_analysis,
    write_report_artifacts,
)
from scripts.validation.check_issue_5302_oracle_gap_packet import (
    load_packet,
    validate_packet,
)


def load_input_rows(path: Path) -> list[dict[str, Any]]:
    """Load benchmark rows from CSV or JSONL format."""
    if not path.is_file():
        raise FileNotFoundError(f"Input rows file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rows.append(json.loads(line_str))
        return rows
    elif suffix in (".csv", ".txt"):
        import csv

        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                converted_row: dict[str, Any] = dict(row)
                for num_key in (
                    "selection_score",
                    "collision_rate",
                    "severe_intrusion_rate",
                    "completion_rate",
                    "timeout_rate",
                    "tail_clearance",
                    "jerk",
                    "pedestrian_disturbance",
                    "compute_time_ms",
                ):
                    if num_key in converted_row and converted_row[num_key] is not None:
                        converted_row[num_key] = float(converted_row[num_key])
                if "seed" in converted_row and converted_row["seed"] is not None:
                    converted_row["seed"] = int(converted_row["seed"])
                rows.append(converted_row)
        return rows
    else:
        raise ValueError(f"Unsupported input rows format '{suffix}'; expected .csv or .jsonl")


def main(argv: list[str] | None = None) -> int:
    """Run issue #5302 oracle gap analysis on input benchmark rows."""
    parser = argparse.ArgumentParser(
        description="Compute frozen issue #5302 selection ceilings and hierarchical uncertainty."
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_PACKET_PATH,
        help="Path to issue #5302 oracle gap packet config (YAML).",
    )
    parser.add_argument(
        "--input-rows",
        type=Path,
        required=True,
        help="Path to CSV or JSONL benchmark rows containing complete 6-arm episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/benchmarks/issue_5302_oracle_gap"),
        help="Directory where reports will be written.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of hierarchical bootstrap samples (default 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5302,
        help="Random seed for hierarchical bootstrap (default 5302).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit JSON summary to stdout on completion.",
    )

    args = parser.parse_args(argv)

    # 1. Validate packet pre-registration contract
    try:
        packet_payload = load_packet(args.packet)
        validate_packet(packet_payload)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        err_msg = f"Packet validation failed: {exc}"
        if args.as_json:
            print(json.dumps({"status": "blocked", "error": err_msg}))
        else:
            print(f"ERROR: {err_msg}", file=sys.stderr)
        return 1

    # 2. Load input rows and execute analysis
    try:
        rows = load_input_rows(args.input_rows)
        results = run_full_oracle_gap_analysis(rows, n_bootstrap=args.n_bootstrap, seed=args.seed)
        written_files = write_report_artifacts(results, args.output_dir)
    except (OracleGapAnalysisError, FileNotFoundError, ValueError) as exc:
        err_msg = f"Oracle gap analysis fail-closed error: {exc}"
        if args.as_json:
            print(json.dumps({"status": "blocked", "error": err_msg}))
        else:
            print(f"ERROR: {err_msg}", file=sys.stderr)
        return 1

    summary_payload = {
        "status": "ok",
        "issue": 5302,
        "preflight": results["preflight"],
        "best_fixed_planner": results["best_fixed_planner"],
        "claim_gate": results["claim_gate"],
        "output_dir": str(args.output_dir),
        "written_files": [str(p) for p in written_files],
    }

    if args.as_json:
        print(json.dumps(summary_payload, indent=2))
    else:
        print(
            f"SUCCESS: Computed issue #5302 selection ceilings on {results['preflight']['total_episodes']} episodes."
        )
        print(f"Best fixed planner: {results['best_fixed_planner']}")
        print(f"Claim gate status: {results['claim_gate']['status']}")
        print(f"Reports written to: {args.output_dir}/reports/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
