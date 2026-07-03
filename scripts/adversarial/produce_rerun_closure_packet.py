#!/usr/bin/env python3
"""Produce a consolidated issue #3275 rerun closure packet, fail-closed.

This CLI consolidates the accumulated failure-archive rerun readiness gates into
one durable closure packet for a *real* disjoint source/rerun archive pair. It
reads archive metadata only: it never runs benchmarks, proposal-model inference,
planner executions, SLURM jobs, or artifact publication, and it never
substitutes a synthetic fixture for a missing real archive.

Exit codes mirror the disposition so the packet doubles as a fail-closed gate:

- ``0``: ``ready_for_rerun`` — disjoint and certified; safe to run the
  independent-outcome proposal-vs-random rerun next.
- ``2``: ``diagnostic_only`` — inputs pass leakage/certification checks but the
  supplied rerun output is diagnostic-only, not held-out evidence.
- ``3``: ``fail_closed_blocked`` — leakage, missing certification/metadata,
  malformed input, or an absent archive blocks the rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.failure_archive_rerun_closure import (
    DIAGNOSTIC_ONLY_DISPOSITION,
    FAIL_CLOSED_BLOCKED,
    READY_FOR_RERUN,
    build_rerun_closure_packet,
)

_EXIT_CODES = {READY_FOR_RERUN: 0, DIAGNOSTIC_ONLY_DISPOSITION: 2, FAIL_CLOSED_BLOCKED: 3}


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Real failure archive used to fit/select proposal-model candidates.",
    )
    parser.add_argument(
        "--rerun-archive",
        type=Path,
        required=True,
        help="Real disjoint certified failure archive intended for the rerun slice.",
    )
    parser.add_argument(
        "--rerun-output",
        type=Path,
        default=None,
        help="Optional rerun report JSON; diagnostic-only markers cap the disposition.",
    )
    parser.add_argument(
        "--null-test-prerequisites",
        type=Path,
        default=None,
        help="Optional null-test prerequisite JSON required before held-out claims.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON closure packet.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the closure packet and return a fail-closed exit code."""

    args = _build_parser().parse_args(argv)
    packet = build_rerun_closure_packet(
        args.source_archive,
        args.rerun_archive,
        rerun_output=args.rerun_output,
        null_test_prerequisites=args.null_test_prerequisites,
    )
    rendered = json.dumps(packet.to_payload(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return _EXIT_CODES.get(packet.disposition, 3)


if __name__ == "__main__":
    sys.exit(main())
