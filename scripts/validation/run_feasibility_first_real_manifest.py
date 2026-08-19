#!/usr/bin/env python3
"""Run the bounded issue #7340 real-manifest feasibility-first diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.adversarial.feasibility_first_real import run_real_manifest_diagnostic

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_7340_feasibility_first_real_manifest_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output/issue_7340_real_manifest/report.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/issue_7340_real_manifest"


def build_parser() -> argparse.ArgumentParser:
    """Build the config-first real-manifest CLI."""
    parser = argparse.ArgumentParser(
        description="Run the bounded, diagnostic-only issue #7340 real-manifest probe."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic and print compact provenance and availability counts."""
    args = build_parser().parse_args(argv)
    try:
        report = run_real_manifest_diagnostic(
            args.config,
            output_path=args.output,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"real-manifest feasibility diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "schema_version": report["schema_version"],
                "evidence_tier": report["evidence_tier"],
                "total_candidates": report["feasibility"]["total_candidates"],
                "feasible_candidates": report["feasibility"]["feasible_candidates"],
                "rejected_candidates": report["feasibility"]["rejected_candidates"],
                "simulator_executed": report["governance"]["simulator_executed"],
                "benchmark_evidence": report["governance"]["benchmark_evidence"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
