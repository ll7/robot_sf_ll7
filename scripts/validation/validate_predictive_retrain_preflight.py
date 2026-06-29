"""Validate a crossing-conflict predictive retraining config before SLURM launch.

Static, CPU-only preflight for the predictive training pipeline config consumed
by ``scripts/training/run_predictive_training_pipeline.py``. It does not collect
data, train, submit SLURM, or make any model-improvement claim. See
``robot_sf.training.predictive_retrain_preflight`` for the validation contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.predictive_retrain_preflight import (
    PredictiveRetrainPreflightError,
    build_retrain_decision_packet,
    validate_retrain_preflight,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate a predictive crossing-conflict retraining launch config."
    )
    parser.add_argument("--config", required=True, type=Path, help="Pipeline config YAML path.")
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON validation report.")
    parser.add_argument(
        "--decision-packet",
        action="store_true",
        help="Emit the no-submit training readiness decision packet.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a predictive retraining launch config; return a shell-friendly code."""
    args = build_arg_parser().parse_args(argv)
    if args.decision_packet:
        packet = build_retrain_decision_packet(args.config, repo_root=args.repo_root)
        print(json.dumps(packet, indent=2, sort_keys=True))
        return 0

    try:
        report = validate_retrain_preflight(args.config, repo_root=args.repo_root)
    except PredictiveRetrainPreflightError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"predictive retrain launch preflight valid: {report['run_id']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
