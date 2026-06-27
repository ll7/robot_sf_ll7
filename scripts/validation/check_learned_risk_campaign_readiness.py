"""Check learned-risk model v1 Slurm campaign readiness (issue #1472, fail-closed).

This is the single campaign-level gate for #1472. It aggregates the launch-packet
validator and the durable trace-manifest validator into one decision so the
campaign can be launched only when *both* canonical owners report ready. It submits
nothing, trains nothing, and fetches nothing; a ready decision means the checked-in
contract is locally complete, not that a job has run.

Exit codes are distinct so callers can branch mechanically:

- ``0`` -- both gates pass: ``campaign_launch_ready``.
- ``2`` -- an input config file is missing/unreadable (cannot be evaluated).
- ``3`` -- a gate is blocked: ``campaign_blocked`` (fail-closed; never launch-ready).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.learned_risk_campaign_readiness import (
    CAMPAIGN_BLOCKED,
    DEFAULT_LAUNCH_PACKET,
    DEFAULT_TRACE_MANIFEST,
    LearnedRiskCampaignReadinessError,
    evaluate_campaign_readiness,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Check learned-risk model v1 Slurm campaign readiness (fail-closed)."
    )
    parser.add_argument(
        "--launch-packet",
        default=DEFAULT_LAUNCH_PACKET,
        type=Path,
        help="Learned-risk launch-packet YAML path.",
    )
    parser.add_argument(
        "--trace-manifest",
        default=DEFAULT_TRACE_MANIFEST,
        type=Path,
        help="Durable learned-risk trace-manifest YAML path.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON readiness report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate campaign readiness and return a decision-coded exit status."""
    args = build_arg_parser().parse_args(argv)
    try:
        report = evaluate_campaign_readiness(
            args.launch_packet,
            args.trace_manifest,
            repo_root=args.repo_root,
        )
    except LearnedRiskCampaignReadinessError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"learned-risk campaign decision: {report['campaign_state']}")
        for gate in report["gates"]:
            print(f"  - {gate['name']}: {gate['status']} ({gate['summary']})")
        for blocker in report["blockers"]:
            print(f"    blocker: {blocker}")
    return 3 if report["campaign_state"] == CAMPAIGN_BLOCKED else 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
