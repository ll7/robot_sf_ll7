#!/usr/bin/env python3
"""Fail-closed CLI gate for the issue #5355 prediction-MPC 2x2 factorial campaign.

This is the executable pre-submission gate that ops runs *before* any GPU/Slurm
submission. It is a thin, CPU-only wrapper around
``robot_sf.benchmark.prediction_mpc_factorial_preregistration.assess_campaign_readiness``:
it inspects only tracked config and evidence artifacts, never runs benchmark
episodes, and authorizes no submission by itself. The process exit code is the
gate contract: ``0`` only when every §6 criterion passes, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
    assess_campaign_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/research/prediction_mpc_factorial_v1.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Tracked factorial preregistration config to gate.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional override for the sha256 evidence registry JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the readiness report JSON.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the readiness report as JSON.")
    return parser.parse_args(argv)


def _format_report(report: dict) -> str:
    lines = [
        f"issue #{report['issue']} factorial campaign readiness: "
        f"{'READY' if report['ready'] else 'NOT READY'}",
        f"config: {report['config_path']}",
        "criteria:",
    ]
    for name, entry in report["criteria"].items():
        mark = "PASS" if entry["ready"] else "BLOCK"
        lines.append(f"  [{mark}] {name}: {entry['detail']}")
    if report["blockers"]:
        lines.append("blockers:")
        lines.extend(f"  - {blocker}" for blocker in report["blockers"])
    lines.append(f"claim_boundary: {report['claim_boundary']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed readiness gate.

    Returns:
        ``0`` when the campaign is ready for submission, ``1`` otherwise.
    """

    args = _parse_args(argv)
    report = assess_campaign_readiness(args.config, registry_path=args.registry)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))

    return 0 if report["ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
