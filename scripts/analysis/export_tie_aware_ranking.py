#!/usr/bin/env python3
"""Export a validated benchmark comparison as a tie-aware partial order.

The input is a JSON mapping with ``metric`` and ``rows`` fields. The script
writes the versioned JSON contract and, optionally, its stable Markdown
summary. It does not read or alter campaign artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.benchmark.tie_aware_ranking import (
    build_tie_aware_ranking,
    render_tie_aware_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Read an input JSON document and write tie-aware outputs."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _load_payload(args.input)
        result = _build_result(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(render_tie_aware_summary(result), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        print(f"tie-aware ranking export failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON comparison input")
    parser.add_argument("--output", required=True, type=Path, help="JSON output path")
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional Markdown summary output path",
    )
    return parser


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("input JSON must contain a mapping")
    return payload


def _build_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    metric = payload.get("metric")
    rows = payload.get("rows")
    if not isinstance(metric, (str, Mapping)):
        raise ValueError("input metric must be a string or mapping")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("input rows must be an array")
    comparisons = payload.get("pairwise_comparisons", ())
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes)):
        raise ValueError("pairwise_comparisons must be an array")
    return build_tie_aware_ranking(
        rows,
        metric=metric,
        display_order=payload.get("display_order"),
        pairwise_comparisons=comparisons,
    )


if __name__ == "__main__":
    raise SystemExit(main())
