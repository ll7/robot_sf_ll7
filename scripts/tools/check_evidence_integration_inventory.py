#!/usr/bin/env python3
"""List, report, and presence-check evidence-stream integration contracts.

Issue #3293 design-stage helper. The tool does not ingest real data, validate field
values, weight evidence, or make any safety/calibration claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.research.evidence_integration_inventory import (
    build_integration_report,
    check_stream_metadata,
    get_stream,
    list_streams,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--list",
        action="store_true",
        help="Print canonical evidence-stream inventory JSON.",
    )
    mode.add_argument(
        "--report",
        action="store_true",
        help="Print issue #3293 integration report JSON.",
    )
    mode.add_argument(
        "--check",
        metavar="STREAM_ID",
        help="Presence-check metadata record against stream's contract.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to synthetic metadata JSON record (required with --check).",
    )
    return parser


def _load_metadata(path: Path) -> dict[str, Any]:
    """Load a minimally synthetic metadata JSON record."""
    if not path.is_file():
        raise ValueError(f"metadata file not found or not a regular file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("metadata JSON must be an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run inventory CLI and return shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)

    if args.list:
        payload = {"streams": [spec.to_dict() for spec in list_streams()]}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.report:
        print(json.dumps(build_integration_report(), indent=2, sort_keys=True))
        return 0

    # --check mode
    if args.metadata is None:
        print(json.dumps({"status": "error", "error": "--check requires --metadata"}, indent=2))
        return 2
    try:
        get_stream(args.check)  # fail fast with a clear message on unknown stream
        metadata = _load_metadata(args.metadata)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    result = check_stream_metadata(args.check, metadata)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
