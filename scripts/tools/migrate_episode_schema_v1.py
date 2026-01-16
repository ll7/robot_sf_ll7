#!/usr/bin/env python3
"""Migrate episode JSONL records to include v1 schema defaults.

Usage:
  uv run python scripts/tools/migrate_episode_schema_v1.py \
    --input output/benchmarks/episodes.jsonl \
    --output output/benchmarks/episodes_v1.jsonl

This script adds missing fields that the v1 schema expects or documents:
- version: "v1" when missing
- scenario_params: {} when missing
- algorithm_metadata: {"status": "unknown"} when missing
- timestamps: inferred from created_at when possible
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path (will be overwritten)",
    )
    return parser.parse_args()


def _to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    if "version" not in updated:
        updated["version"] = "v1"
    if "scenario_params" not in updated:
        updated["scenario_params"] = {}
    if "algorithm_metadata" not in updated:
        updated["algorithm_metadata"] = {"status": "unknown"}
    if "timestamps" not in updated:
        created_at = updated.get("created_at")
        if isinstance(created_at, (int, float)):
            iso = _to_iso(float(created_at))
            updated["timestamps"] = {"start": iso, "end": iso}
    return updated


def main() -> int:
    """Run the schema v1 migration CLI.

    Returns:
        Exit code (0 for success).
    """
    args = _parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            migrated = _migrate_record(record)
            dst.write(json.dumps(migrated) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
