#!/usr/bin/env python3
"""Reconcile issue #6987 analysis-trace overhead receipts fail-closed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.analysis_trace_overhead_reconciliation import reconcile_receipts


def _parse_args() -> argparse.Namespace:
    """Parse receipt paths and optional output destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path, help="Measurement receipt JSON paths.")
    parser.add_argument("--output", type=Path, help="Optional reconciliation packet path.")
    return parser.parse_args()


def main() -> int:
    """Build and emit a diagnostic-only reconciliation packet."""

    args = _parse_args()
    packet = reconcile_receipts(args.receipts)
    rendered = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if packet["reconciliation"]["classification"] != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
