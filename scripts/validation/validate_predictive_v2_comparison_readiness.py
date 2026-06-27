#!/usr/bin/env python3
"""CLI for the predictive planner v2 same-seed comparison readiness preflight (#1490).

This is a thin wrapper around
``robot_sf.benchmark.predictive_v2_comparison_readiness``. It is read-only and
coordination-only: it never trains, evaluates, submits Slurm, or tunes planners.

Exit codes:
  0  ready    — metadata complete AND the blocked Slurm/maintainer gate is cleared.
  2  not ready — metadata complete but blocked (default), or prerequisite metadata incomplete.
  1  error     — the contract could not be loaded or parsed.

Default invocation (fail-closed, expected ``blocked``)::

    uv run python scripts/validation/validate_predictive_v2_comparison_readiness.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.predictive_v2_comparison_readiness import (
    DEFAULT_CONTRACT_PATH,
    PredictiveV2ComparisonReadinessError,
    validate_predictive_v2_comparison_readiness,
)
from robot_sf.common.artifact_paths import get_repository_root


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed readiness preflight for the predictive planner v2 same-seed "
            "comparison (#1490). Validates variant/conditioning/seed/provenance metadata "
            "and surfaces the blocked Slurm/maintainer-gate state. Does not run benchmarks."
        )
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Path to the predictive ego-features comparison contract YAML.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=get_repository_root(),
        help="Repository root used to resolve contract-relative provenance paths.",
    )
    parser.add_argument(
        "--coupling-gate",
        type=Path,
        default=None,
        help=(
            "Optional coupling-gate clearance artifact (JSON or Markdown) recording a "
            "recommendation of 'continue'. Required to clear the blocked Slurm gate."
        ),
    )
    parser.add_argument(
        "--revised-hypothesis-recorded",
        action="store_true",
        help=(
            "Explicit maintainer acknowledgement that a revised predictive-v2 hypothesis "
            "has been recorded. Required (with --coupling-gate) to clear the blocked gate."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON readiness report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the preflight and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    contract_path = args.contract if args.contract.is_absolute() else repo_root / args.contract

    try:
        report = validate_predictive_v2_comparison_readiness(
            contract_path=contract_path,
            repo_root=repo_root,
            coupling_gate_path=args.coupling_gate,
            revised_hypothesis_recorded=args.revised_hypothesis_recorded,
        )
    except PredictiveV2ComparisonReadinessError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"predictive-v2 same-seed comparison readiness: {report['status'].upper()}")
        for name, payload in report["stages"].items():
            print(f" - {name}: {payload['status']}")
            for message in payload["messages"]:
                print(f"   * {message}")

    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
