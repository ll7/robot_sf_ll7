"""Validate issue #1428 ORCA-residual behavior-cloning lineage packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.orca_residual_lineage_packet import (
    OrcaResidualLineagePacketError,
    validate_launch_packet,
    validate_smoke_nominal_gate,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Validate an ORCA-residual behavior-cloning lineage packet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/orca_residual/orca_residual_bc_issue_1428.yaml"),
        help="Path to lineage packet YAML.",
    )
    parser.add_argument(
        "--smoke-summary",
        type=Path,
        help="Optional completed smoke summary JSON to validate before nominal escalation.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON validation report.")
    return parser


def main() -> int:
    """Run packet validation."""
    args = build_parser().parse_args()
    try:
        report = validate_launch_packet(args.config)
        if args.smoke_summary is not None:
            summary = json.loads(args.smoke_summary.read_text(encoding="utf-8"))
            report["smoke_nominal_gate"] = validate_smoke_nominal_gate(summary)
    except OrcaResidualLineagePacketError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status={report['status']} campaign_id={report['campaign_id']}")
        if "smoke_nominal_gate" in report:
            print("smoke_nominal_gate=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
