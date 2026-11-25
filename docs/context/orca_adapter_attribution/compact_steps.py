"""Rebuild the packet's rounded ORCA-only step table from a full extraction."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

FIELDS = (
    "config",
    "cell",
    "seed",
    "step",
    "x",
    "y",
    "goal_x",
    "goal_y",
    "pref_final_vx",
    "pref_final_vy",
    "solver_vx",
    "solver_vy",
    "occ_penalty",
    "adapter_v",
    "adapter_w",
    "sel_a",
    "sel_b",
    "post_x",
    "post_y",
    "measured_speed",
    "door_gap_m",
    "n_obstacles",
    "stall_class",
)
IDENTIFIERS = {"config", "cell", "seed", "step", "n_obstacles", "stall_class"}


def main() -> int:
    """Write six-decimal values and a reproducible gzip stream."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        gzip.open(args.steps, "rt", encoding="utf-8", newline="") as source,
        args.output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("full step table lacks a required column")
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        count = 0
        for row in reader:
            if row["config"] not in {"orca_dd", "orca_holo"}:
                continue
            writer.writerow(
                {
                    field: (
                        row[field]
                        if field in IDENTIFIERS or row[field] == ""
                        else f"{float(row[field]):.6f}"
                    )
                    for field in FIELDS
                }
            )
            count += 1
    print(f"{count} ORCA steps -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
