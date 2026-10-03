#!/usr/bin/env python3
"""Validate and merge pytest-split measurements for scheduling only.

Strict callers require every expected shard. CI can opt into an incomplete union;
missing measurements fall back to pytest-split's estimate. Neither durations nor
this helper represent a test, coverage, or release verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

EXPECTED_SHARD_NAMES = tuple(f"pytest-durations-{index}" for index in range(1, 5))


def _validate_duration_store(path: Path) -> dict[str, float]:
    """Return the parsed durations for one shard store, failing on any violation."""
    try:
        durations = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid pytest duration store: {path}: {exc}") from exc
    if not isinstance(durations, dict) or not durations:
        raise SystemExit(f"Invalid pytest duration store: {path}: expected a nonempty mapping")
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


def merge_duration_stores(
    artifact_dir: str | Path, *, allow_partial: bool = False, shard_count: int = 4
) -> dict[str, float]:
    """Merge the expected shard stores under *artifact_dir* into one mapping.

    Missing shards are accepted only with explicit scheduling-only partial mode.
    Empty, unexpected, overlapping, and malformed stores always fail closed.
    """
    artifact_path = Path(artifact_dir)
    files = sorted(artifact_path.glob("*/ .test_durations".replace(" ", "")))
    actual_names = {path.parent.name for path in files}
    if shard_count < 1:
        raise SystemExit("Expected a positive shard count")
    expected_names = {f"pytest-durations-{index}" for index in range(1, shard_count + 1)}
    if (
        not actual_names
        or actual_names - expected_names
        or (not allow_partial and actual_names != expected_names)
    ):
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise SystemExit(
            f"Expected exactly one pytest duration store from each of {shard_count} shards; "
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
    """Freeze a cache or merge stores while preserving scheduling-only boundaries."""
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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing shards (scheduling hints only)",
    )
    parser.add_argument("--metadata-output", help="Write scheduling provenance alongside the cache")
    parser.add_argument("--snapshot-input", help="Freeze one restored cache for all matrix jobs")
    parser.add_argument(
        "--shard-count", type=int, default=4, help="Expected matrix size (default: 4)"
    )
    args = parser.parse_args(argv)
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")

    if args.snapshot_input:
        if not args.output:
            parser.error("--snapshot-input requires --output")
        source = Path(args.snapshot_input)
        try:
            durations = _validate_duration_store(source) if source.exists() else {}
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 1
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(durations, indent=4, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        return 0

    try:
        merged = merge_duration_stores(
            args.artifact_dir, allow_partial=args.allow_partial, shard_count=args.shard_count
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    available = sorted(
        path.parent.name for path in Path(args.artifact_dir).glob("*/.test_durations")
    )
    expected_names = {f"pytest-durations-{index}" for index in range(1, args.shard_count + 1)}
    if args.metadata_output:
        metadata = {
            "schema_version": 2,
            "purpose": "scheduling-only",
            "source_sha": os.environ.get(
                "DURATION_SOURCE_SHA", os.environ.get("GITHUB_SHA", "unknown")
            ),
            "producer_sha": os.environ.get("GITHUB_SHA", "unknown"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "unknown"),
            "cache_key": os.environ.get("DURATION_CACHE_KEY", "unknown"),
            "matrix_result": os.environ.get("FAST_FEEDBACK_RESULT", "unknown"),
            "available_shards": available,
            "missing_shards": sorted(expected_names - set(available)),
            "complete_matrix": len(available) == args.shard_count
            and os.environ.get("FAST_FEEDBACK_RESULT") == "success",
            "measurement_completeness": "partial_or_unverified",
            "test_count": len(merged),
        }
        if metadata["complete_matrix"]:
            metadata["measurement_completeness"] = "complete"
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    payload = json.dumps(merged, indent=4, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    print(f"Merged {len(available)} shard stores with {len(merged)} test durations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
