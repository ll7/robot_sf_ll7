#!/usr/bin/env python3
"""CLI for trace-exemplar episode interest scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.trace_exemplar_interest import (
    DEFAULT_WEIGHTS,
    score_bundles,
    write_report_json,
    write_report_markdown,
)


def main() -> int:
    """Run the trace-exemplar interest scoring command."""

    parser = argparse.ArgumentParser(
        description="Rank trace-exemplar episodes for figure-selection interest.",
    )
    parser.add_argument("bundle_roots", nargs="+", type=Path)
    parser.add_argument("--json", dest="json_out", type=Path)
    parser.add_argument("--md", dest="markdown_out", type=Path)
    parser.add_argument("--weights", action="append", default=[], metavar="KEY=VAL")
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    weights = _parse_weights(args.weights)
    report = score_bundles(args.bundle_roots, weights=weights or None)
    if args.json_out is not None:
        write_report_json(report, args.json_out, top_n=args.top)
    if args.markdown_out is not None:
        write_report_markdown(report, args.markdown_out, top_n=args.top)
    if args.json_out is None and args.markdown_out is None:
        for index, episode in enumerate(report.episodes[: args.top], start=1):
            print(
                f"{index}\t{episode.composite_score:.6f}\t{episode.episode_id}\t"
                f"{episode.planner}\t{episode.scenario_id}\t{episode.seed}\t"
                f"{episode.episode_status}"
            )
    return 0


def _parse_weights(raw_items: list[str]) -> dict[str, float]:
    """Parse repeated ``KEY=VAL`` weight overrides."""

    weights: dict[str, float] = {}
    for raw_item in raw_items:
        if "=" not in raw_item:
            raise SystemExit(f"--weights must be KEY=VAL, got: {raw_item}")
        key, raw_value = raw_item.split("=", 1)
        if key not in DEFAULT_WEIGHTS:
            raise SystemExit(f"unknown weight key: {key}")
        try:
            weights[key] = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"invalid weight value for {key}: {raw_value}") from exc
    return weights


if __name__ == "__main__":
    raise SystemExit(main())
