"""Compare one development seed's reconstructed episodes and raw reset receipts.

Run after extracting the rerun with the acquisition commit's analyzer. The raw
archive layout and rerun layout are both ``planner/chunk/observer``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

PLANNERS = ("hybrid_v4_continuous", "orca", "social_force", "ppo_release")
CELLS = ("crossing", "group_crossing", "head_on_corridor", "overtaking")
CONDITIONS = ("uniform", "mixed")
VARIABLE_EPISODE_FIELDS = frozenset({"git_hash", "wall_time_s"})
Key = tuple[str, str, str, int]


def _rows(path: Path, seed: int) -> tuple[list[str], dict[Key, dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = {}
        for row in reader:
            if int(row["seed"]) != seed:
                continue
            key = (row["planner"], row["cell"], row["condition"], seed)
            if key in rows:
                raise ValueError(f"duplicate episode key: {key}")
            rows[key] = row
    return fields, rows


def _receipts(root: Path, seed: int) -> dict[Key, dict[str, Any]]:
    receipts = {}
    for planner in PLANNERS:
        for path in (root / planner).glob(f"*/observer/*__s{seed}.receipt.json"):
            receipt = json.loads(path.read_text())
            key = (planner, receipt["cell"], receipt["condition"], int(receipt["seed"]))
            if key in receipts:
                raise ValueError(f"duplicate receipt key: {key}")
            receipts[key] = receipt
    return receipts


def main() -> None:
    """Compare every scientific CSV field and every reset receipt field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--rerun-episodes", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if 111 <= args.seed <= 140 or 10000 <= args.seed <= 99999:
        raise SystemExit("refusing reserved seed block")
    expected = {
        (p, c, condition, args.seed) for p in PLANNERS for c in CELLS for condition in CONDITIONS
    }
    fields, original = _rows(args.episodes, args.seed)
    other_fields, rerun = _rows(args.rerun_episodes, args.seed)
    before = _receipts(args.archive, args.seed)
    after = _receipts(args.rerun, args.seed)
    if fields != other_fields or any(
        set(values) != expected for values in (original, rerun, before, after)
    ):
        raise SystemExit("missing, duplicated, or unexpected episode/receipt keys or CSV columns")
    exact_fields = [field for field in fields if field not in VARIABLE_EPISODE_FIELDS]
    episode_differences: Counter[str] = Counter()
    receipt_differences: Counter[str] = Counter()
    for key in expected:
        for field in exact_fields:
            if original[key][field] != rerun[key][field]:
                episode_differences[field] += 1
        for field in before[key].keys() | after[key].keys():
            if (
                field not in before[key]
                or field not in after[key]
                or before[key][field] != after[key][field]
            ):
                receipt_differences[field] += 1
    result = {
        "seed": args.seed,
        "episode_pairs_compared": len(expected),
        "episode_exact_fields": exact_fields,
        "allowed_variable_episode_fields": sorted(VARIABLE_EPISODE_FIELDS),
        "receipt_exact_fields": sorted({field for item in before.values() for field in item}),
        "episode_field_difference_counts": dict(sorted(episode_differences.items())),
        "receipt_field_difference_counts": dict(sorted(receipt_differences.items())),
        "pass": not episode_differences and not receipt_differences,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
