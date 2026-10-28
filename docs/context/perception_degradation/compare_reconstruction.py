"""Compare one complete seed block with the archived compact projection."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

PACKET = Path(__file__).resolve().parent
PLANNERS = ("orca", "hybrid_v4_continuous_guard_on", "hybrid_v4_continuous_guard_off", "ppo")
CONDITIONS = ("oracle", "position_noise", "delay", "static_occlusion")
NONCOMPARABLE = {"git_hash", "source_path", "source_line", "source_line_sha256"}


def load_rows(path: Path, seed: int, *, selected_only: bool) -> list[dict[str, Any]]:
    """Load one seed; reject unexpected seeds in the replay input.

    Returns:
        Compact episode rows for the selected seed.
    """
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not selected_only and any(row["seed"] != seed for row in rows):
        raise ValueError("replay contains an unexpected seed")
    return [row for row in rows if row["seed"] == seed]


def index_rows(rows: list[dict[str, Any]], expected_commit: str) -> dict[tuple, dict[str, Any]]:
    """Check per-row commit attribution and unique keys.

    Returns:
        Rows indexed by seed, cell, planner and condition.
    """
    result: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (row["seed"], row["scenario_id"], row["planner_key"], row["condition"])
        if key in result:
            raise ValueError("duplicate episode key")
        if row["git_hash"] != expected_commit:
            raise ValueError("episode commit does not match its declared source")
        result[key] = row
    return result


def main() -> None:
    """Check coverage, schema and exact normalized row values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--archived-commit", required=True)
    parser.add_argument("--replay-commit", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cells = yaml.safe_load((PACKET / "contract.yaml").read_text())["cells"]
    expected = set(itertools.product([args.seed], cells, PLANNERS, CONDITIONS))
    archived = index_rows(
        load_rows(args.archived, args.seed, selected_only=True), args.archived_commit
    )
    replay = index_rows(load_rows(args.replay, args.seed, selected_only=False), args.replay_commit)
    if set(archived) != expected or set(replay) != expected:
        raise ValueError("archived or replay input lacks the complete distinct seed block")

    field_differences: dict[str, int] = {}
    normalized_matches = 0
    complete_matches = 0
    for key in sorted(expected):
        original, current = archived[key], replay[key]
        if original.keys() != current.keys():
            raise ValueError("archived and replay projection schemas differ")
        equal = True
        for field in original.keys() - NONCOMPARABLE:
            left = json.dumps(original[field], sort_keys=True, separators=(",", ":"))
            right = json.dumps(current[field], sort_keys=True, separators=(",", ":"))
            if left != right:
                field_differences[field] = field_differences.get(field, 0) + 1
                equal = False
        normalized_matches += int(equal)
        complete_matches += int(original == current)
    report = {
        "schema": "xperc.reconstruction-comparison.v1",
        "seed": args.seed,
        "expected_rows": len(expected),
        "archived_commit": args.archived_commit,
        "replay_commit": args.replay_commit,
        "normalized_matches": normalized_matches,
        "complete_row_matches": complete_matches,
        "excluded_fields": sorted(NONCOMPARABLE),
        "field_differences": field_differences,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if field_differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
