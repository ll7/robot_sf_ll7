#!/usr/bin/env python3
"""Check predictive retraining launch readiness without submitting compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.predictive_retraining_readiness import (
    DEFAULT_PACKET,
    PredictiveRetrainingReadinessError,
    evaluate_retraining_readiness,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(DEFAULT_PACKET))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--require-launch-ready",
        action="store_true",
        help="Return non-zero when the packet is valid but blocked from launch.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the readiness CLI and return a shell-friendly exit code."""

    args = _parse_args()
    try:
        report = evaluate_retraining_readiness(args.packet, repo_root=args.repo_root)
    except PredictiveRetrainingReadinessError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_launch_ready and not report["launch_ready"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
