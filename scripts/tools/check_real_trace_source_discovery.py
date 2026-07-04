"""CLI checker for issue #3278 public-source discovery ledgers.

The checker validates metadata only. It does not download, copy, or inspect raw
external traces, and it makes no real-world validation claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.analysis_workbench.real_trace_source_discovery import (
    SOURCE_DISCOVERY_STATUS_READY,
    RealTraceSourceDiscoveryError,
    check_real_trace_source_discovery,
    load_real_trace_source_discovery,
)

DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_3278_real_trace_source_discovery_example.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="Path to real_trace_source_discovery.v1 ledger (JSON or YAML).",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless discovery_status is 'ready'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ledger check and print a JSON report."""
    args = _parse_args(argv)
    try:
        ledger = load_real_trace_source_discovery(args.ledger)
        report = check_real_trace_source_discovery(ledger, source=args.ledger)
    except RealTraceSourceDiscoveryError as exc:
        print(
            json.dumps(
                {
                    "status": "schema_error",
                    "source": exc.source,
                    "errors": list(exc.errors),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    print(report.to_json())
    if args.require_ready and report.discovery_status != SOURCE_DISCOVERY_STATUS_READY:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
