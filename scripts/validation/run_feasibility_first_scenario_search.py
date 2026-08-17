#!/usr/bin/env python3
"""Run the bounded issue #7315 feasibility-first fixture diagnostic.

The command emits a versioned report showing candidate rejection accounting and
deterministic uniform-versus-risk-feature selection.  It does not execute the
simulator and is not benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.adversarial.feasibility_first import run_fixture_diagnostic

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_7315_feasibility_first_smoke.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output/issue_7315_feasibility_first/report.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the config-first diagnostic CLI."""
    parser = argparse.ArgumentParser(
        description="Run the fixture-only feasibility-first scenario search diagnostic."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic and print a compact provenance summary."""
    args = build_parser().parse_args(argv)
    try:
        report = run_fixture_diagnostic(args.config, output_path=args.output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"feasibility-first diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "schema_version": report["schema_version"],
                "evidence_tier": report["evidence_tier"],
                "feasible_candidates": report["feasibility"]["feasible_candidates"],
                "rejected_candidates": report["feasibility"]["rejected_candidates"],
                "simulator_executed": report["governance"]["simulator_executed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
