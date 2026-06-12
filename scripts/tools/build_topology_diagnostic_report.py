#!/usr/bin/env python3
"""Build a topology diagnostic report snapshot from JSONL/JSON trace files.

Reads one or more topology-hypothesis diagnostic traces and emits a compact
Markdown/JSON summary covering selected hypotheses, near-parity gate reasons,
reuse-penalty activations, route-progress deltas, top regressions, and top
unchanged cases.

This report is diagnostic/reviewability support, not planner-promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.analysis.topology_diagnostic_report import (
    build_report_payload,
    render_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="*",
        type=Path,
        help="JSON or JSONL topology diagnostic trace files.",
    )
    parser.add_argument(
        "--label",
        default="topology-diagnostic-snapshot",
        help="Label for the report header.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write the JSON report payload.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path to write the Markdown report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = parse_args(argv)
    if not args.traces:
        print("No trace files provided.", file=sys.stderr)
        return 1

    missing = [p for p in args.traces if not p.is_file()]
    if missing:
        for p in missing:
            print(f"File not found or not a regular file: {p}", file=sys.stderr)
        return 2

    payload = build_report_payload(args.traces, label=args.label)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        md_text = render_markdown(payload)
        args.output_md.write_text(md_text, encoding="utf-8")

    if args.output_json is None and args.output_md is None:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
