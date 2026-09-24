#!/usr/bin/env python3
"""Compose three complete issue #6642 campaign arms for Gate 3 (#6643)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.radius_sweep_summary import (
    RadiusSweepSummaryError,
    compose_radius_sweep_summary,
    write_radius_sweep_summary,
)


def main(argv: list[str] | None = None) -> int:
    """Validate the campaign roots and write one deterministic sweep summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        action="append",
        type=Path,
        required=True,
        help="Complete camera-ready campaign root; pass exactly three times.",
    )
    parser.add_argument(
        "--gate1-canary-receipt",
        type=Path,
        required=True,
        help="Exact original passing Gate 1 receipt whose digest binds all three arms.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print a JSON result envelope.")
    args = parser.parse_args(argv)
    try:
        summary = compose_radius_sweep_summary(
            args.campaign_root, gate1_canary_receipt=args.gate1_canary_receipt
        )
        output = write_radius_sweep_summary(summary, args.output)
    except (OSError, RadiusSweepSummaryError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"ok": True, "output": str(output)}, indent=2, sort_keys=True))
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
