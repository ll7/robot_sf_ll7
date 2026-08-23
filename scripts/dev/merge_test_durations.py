#!/usr/bin/env python3
"""Validate and merge the four pytest-split duration shard stores.

The CI aggregate job uploads one ``.test_durations`` store per fast-feedback
shard and merges them so a later run restores a complete balancing store. This
helper is the single importable implementation of the former inline workflow
program, preserving the exact fail-closed contract:

- exactly one store per expected shard name;
- numeric, finite, non-negative durations;
- no overlapping node ids across stores;
- deterministic sorted JSON output;
- failure on missing, duplicate, unexpected, or malformed shards.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXPECTED_SHARD_NAMES = tuple(f"pytest-durations-{index}" for index in range(1, 5))


def _validate_duration_store(path: Path) -> dict[str, float]:
    """Return the parsed durations for one shard store, failing on any violation."""
    try:
        durations = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid pytest duration store: {path}: {exc}") from exc
    if not isinstance(durations, dict):
        raise SystemExit(f"Invalid pytest duration store: {path}: expected a mapping")
    for nodeid, duration in durations.items():
        if (
            not isinstance(nodeid, str)
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise SystemExit(f"Invalid pytest duration store: {path}")
    return {str(nodeid): float(duration) for nodeid, duration in durations.items()}


def merge_duration_stores(artifact_dir: str | Path) -> dict[str, float]:
    """Merge the expected shard stores under *artifact_dir* into one mapping.

    Raises SystemExit on missing/unexpected shards, overlapping node ids, or
    malformed stores — matching the former workflow's fail-closed behavior.
    """
    artifact_path = Path(artifact_dir)
    files = sorted(artifact_path.glob("*/ .test_durations".replace(" ", "")))
    actual_names = {path.parent.name for path in files}
    expected_names = set(EXPECTED_SHARD_NAMES)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise SystemExit(
            "Expected exactly one pytest duration store from each of four shards; "
            f"missing={missing or 'none'} unexpected={unexpected or 'none'}."
        )

    merged: dict[str, float] = {}
    for path in files:
        durations = _validate_duration_store(path)
        overlap = set(merged).intersection(durations)
        if overlap:
            raise SystemExit(f"Overlapping pytest duration stores: {path}")
        merged.update(durations)
    return merged


def main(argv: list[str] | None = None) -> int:
    """Merge shard stores and write the deterministic aggregate to stdout or a file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default=".duration-artifacts",
        help="Directory containing the per-shard store subdirectories",
    )
    parser.add_argument(
        "--output",
        help="Write the merged JSON to this path (default: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        merged = merge_duration_stores(args.artifact_dir)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = json.dumps(merged, indent=4, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    print(f"Merged {len(EXPECTED_SHARD_NAMES)} shard stores with {len(merged)} test durations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
