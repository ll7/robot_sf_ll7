#!/usr/bin/env python3
"""Validate the issue #6971 safety-wrapper paired-campaign preregistration.

The checker verifies the frozen protocol, source lineage, retained metric manifest,
inference contract, and cost estimate. It never runs benchmark episodes or submits
compute; a successful check is proposal-level readiness only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.issue_6971_safety_wrapper_preregistration import (
    DEFAULT_CONFIG,
    SafetyWrapperPreregistrationError,
    build_validation_report,
)


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed preregistration validator."""
    parser = argparse.ArgumentParser(
        description="Validate the issue #6971 safety-wrapper preregistration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the preregistration YAML (default: the tracked issue #6971 packet).",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON summary.")
    args = parser.parse_args(argv)

    try:
        report = build_validation_report(args.config)
    except (OSError, SafetyWrapperPreregistrationError, TypeError, ValueError) as exc:
        result = {
            "schema_version": "issue_6971_safety_wrapper_preregistration_validation.v1",
            "issue": 6971,
            "status": "blocked",
            "error": str(exc),
            "execution_authorized": False,
            "compute_submit_authorized": False,
        }
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"BLOCKED: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "OK: issue #6971 preregistration validated; "
            f"{report['planned_episode_count']} episodes are planned, not run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
