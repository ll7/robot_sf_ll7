#!/usr/bin/env python3
"""Export per-episode metric-semantics audit tables from benchmark JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.event_ledger import ensure_event_ledger
from robot_sf.benchmark.event_ledger_reconciliation import build_metric_semantics_table

EXPORT_SCHEMA_VERSION = "event_ledger_reconciliation_export.v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        required=True,
        type=Path,
        help="Input benchmark episode records JSONL.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output JSONL path for one reconciliation table per episode.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero after export when any episode ledger fails reconciliation.",
    )
    return parser.parse_args(argv)


def _read_episode_records(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL episode records from ``path``."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object episode record")
            records.append(payload)
    return records


def build_export_row(record: dict[str, Any]) -> dict[str, Any]:
    """Build one exported reconciliation row from an episode record."""

    ledger = ensure_event_ledger(record)
    table = build_metric_semantics_table(ledger)
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "episode_id": record.get("episode_id"),
        "scenario_id": ledger.get("scenario_id"),
        "seed": ledger.get("seed"),
        "planner": ledger.get("planner"),
        "software_commit": ledger.get("software_commit"),
        "event_ledger_schema_version": ledger.get("schema_version"),
        "reconciliation_table": table,
    }


def export_reconciliation_tables(
    records: list[dict[str, Any]],
    out_path: Path,
) -> tuple[int, int]:
    """Write reconciliation rows and return ``(episode_count, violation_count)``."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    violation_count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = build_export_row(record)
            table = row["reconciliation_table"]
            if not table["reconciles"]:
                violation_count += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(records), violation_count


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    records = _read_episode_records(args.episodes)
    episode_count, violation_count = export_reconciliation_tables(records, args.out)
    print(
        "wrote "
        f"{episode_count} event-ledger reconciliation rows to {args.out} "
        f"({violation_count} violation episodes)"
    )
    if args.fail_on_violations and violation_count:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
