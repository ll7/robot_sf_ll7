"""CLI: annotate episode JSONL files with diagnostic social preference labels.

Reads episode records from a JSONL file (or stdin), applies threshold-band rules
from the social preference label config, and writes annotated results to an output
JSONL file plus a compact summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.social_preference_labels import (
    annotate_episodes_social_preferences,
    build_label_summary,
    load_social_preference_label_config,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate episode JSONL with diagnostic social preference labels."
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=str,
        default="-",
        help="Path to episode JSONL file (default: stdin).",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to social preference label YAML config.",
    )
    parser.add_argument(
        "--trace-fields",
        type=str,
        nargs="*",
        default=None,
        help="Explicit trace-field names for availability checks.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Path to write annotation JSONL (default: stdout).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write annotation JSONL (default: stdout).",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Path to write summary JSON with label counts and reasons.",
    )
    return parser.parse_args(argv)


def _read_episodes(path: str) -> list[dict[str, Any]]:
    """Read episode records from a JSONL file or stdin."""

    episodes: list[dict[str, Any]] = []
    source = sys.stdin if path == "-" else open(path, encoding="utf-8")
    with source as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed JSONL line {line_num}: {exc}",
                    file=sys.stderr,
                )
    return episodes


def main() -> None:
    """Entry point for the social preference label annotation CLI."""
    args = _parse_args()

    if args.schema is None:
        print("Error: --schema is required.", file=sys.stderr)
        sys.exit(1)

    schema = load_social_preference_label_config(Path(args.schema))

    episodes = _read_episodes(args.episodes_jsonl)
    if not episodes:
        print("Warning: no episodes to annotate.", file=sys.stderr)

    trace_fields = set(args.trace_fields) if args.trace_fields else None

    annotations = annotate_episodes_social_preferences(
        episodes,
        schema=schema,
        trace_fields=trace_fields,
    )

    output_jsonl = args.output_jsonl
    output_json = args.output_json
    summary_json = args.summary_json
    summary = build_label_summary(annotations)

    # Write annotation JSONL
    if output_jsonl and output_jsonl != "-":
        with open(output_jsonl, "w", encoding="utf-8") as fh:
            for ann in annotations:
                fh.write(json.dumps(ann, sort_keys=True) + "\n")
    elif output_json and output_json != "-":
        with open(output_json, "w", encoding="utf-8") as fh:
            for ann in annotations:
                fh.write(json.dumps(ann, sort_keys=True) + "\n")
    else:
        for ann in annotations:
            print(json.dumps(ann, sort_keys=True))

    # Write summary JSON
    if summary_json:
        with open(summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"Summary written to {summary_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
