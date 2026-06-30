#!/usr/bin/env python3
"""Diagnostic preflight: report whether AMV actuation calibration inputs are ready or blocked.

Issue #1559. This script does NOT calibrate, tune envelope values, or run any campaign. It inspects
a benchmark config's actuation profile and prints a fail-closed readiness report. It exits non-zero
when the calibration inputs are blocked (placeholder/pending provenance, missing fields, synthetic
profile, or conflation), so it can gate downstream calibrated work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.amv_calibration_readiness import (
    assess_amv_calibration_readiness_from_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/benchmarks/issue_1586_calibrated_actuation_profile_skeleton_v0.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Benchmark config containing the actuation profile (default: #1586 skeleton).",
    )
    parser.add_argument(
        "--profile-key",
        default="synthetic_actuation_profile",
        help="Top-level config key holding the actuation profile mapping.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit 0 even when the profile is blocked (diagnostic-only inspection).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point. Returns 1 when blocked unless --allow-blocked is set."""
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    readiness = assess_amv_calibration_readiness_from_config(
        config_path, profile_key=args.profile_key
    )

    report = {"config": args.config, **readiness.to_dict()}
    print(json.dumps(report, indent=2))

    if readiness.is_ready:
        print("\nAMV calibration inputs: READY")
        if not readiness.paper_facing_allowed:
            print("Note: ready for calibrated exploratory use only; paper-facing remains BLOCKED")
            print("(requires a hardware trace #2000 or official spec, not a proxy source).")
        return 0

    print("\nAMV calibration inputs: BLOCKED (fail-closed)")
    for reason in readiness.blocking_reasons:
        print(f"  - {reason}")
    return 0 if args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
