"""Compare a one-seed replay against the archived projected evidence.

Raw-source hashes remain in both projections for custody, but a fresh run is
expected to produce different raw JSON timestamps, wall times and paths.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

CELLS = (
    "classic_doorway_medium",
    "classic_doorway_high",
    "classic_bottleneck_medium",
    "classic_bottleneck_high",
)
ARMS = ("none", "G1E1", "G0E1", "orca")
CONDITIONS = ("traverse", "outside")
KEY_FIELDS = ("arm", "cell", "condition", "seed")
COLUMNS = (
    "arm",
    "cell",
    "condition",
    "seed",
    "mode",
    "fallback_count",
    "fallback_or_degraded_marker",
    "guard_disabled",
    "status",
    "git_hash",
    "steps",
    "termination_reason",
    "parking",
    "success",
    "ped_contact",
    "wall_contact",
    "timeout",
    "n_peds",
    "passages",
    "first_passage_s",
    "nr_passers",
    "nr_passers_blocked",
    "mean_delay_censored_s",
    "throughput_loss",
    "min_robot_ped_dist_m",
    "t0_ped_match",
    "sim_config_match",
    "source_line_sha256",
)
RAW_HASH_FIELD = "source_line_sha256"


def load_seed(path: Path, seed: int, *, replay: bool) -> dict[tuple[str, ...], dict[str, str]]:
    """Load the contracted seed, rejecting schema drift, duplicates and extra replay seeds."""
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Unexpected projected schema: {path}")
        for row in reader:
            if row["seed"] != str(seed):
                if replay:
                    raise ValueError("Replay contains rows for another seed")
                continue
            key = tuple(row[field] for field in KEY_FIELDS)
            if key in rows:
                raise ValueError(f"Duplicate row key: {key}")
            rows[key] = row
    expected = {
        (arm, cell, condition, str(seed))
        for arm in ARMS
        for cell in CELLS
        for condition in CONDITIONS
    }
    if set(rows) != expected:
        raise ValueError(
            f"Wrong row set: {path}; missing={len(expected - set(rows))}, extra={len(set(rows) - expected)}"
        )
    return rows


def compare(archived: Path, replay: Path, seed: int) -> dict[str, object]:
    """Require exact equality of all projected fields except the raw custody hash."""
    if not (1121 <= seed <= 1130 or 1161 <= seed <= 1200):
        raise ValueError("Seed is not in the preregistered development roster")
    before = load_seed(archived, seed, replay=False)
    after = load_seed(replay, seed, replay=True)
    required_fields = tuple(field for field in COLUMNS if field != RAW_HASH_FIELD)
    differences = [
        {
            "key": key,
            "fields": [
                field for field in required_fields if before[key][field] != after[key][field]
            ],
        }
        for key in sorted(before)
        if any(before[key][field] != after[key][field] for field in required_fields)
    ]
    result: dict[str, object] = {
        "seed": seed,
        "expected_rows": len(before),
        "normalized_rows_equal": len(before) - len(differences),
        "full_rows_equal": sum(before[key] == after[key] for key in before),
        "raw_hashes_equal": sum(
            before[key][RAW_HASH_FIELD] == after[key][RAW_HASH_FIELD] for key in before
        ),
        "excluded_from_normalized_comparison": [RAW_HASH_FIELD],
        "differences": differences,
    }
    if differences:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    """Run the comparison and emit a compact, machine-readable receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.archived, args.replay, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
