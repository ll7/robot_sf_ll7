#!/usr/bin/env python3
"""Run the issue #3574 mean-matched heterogeneity ablation campaign.

Runs simulations for each row of the dry-run manifest, writes episode records,
and outputs a complete JSONL dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.heterogeneous_population_ablation_runner import run_manifest_row

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="output/issue_3574_mean_matched_harness/manifest.json",
        help="Path to the pre-run manifest JSON.",
    )
    parser.add_argument(
        "--output",
        default="output/issue_3574_mean_matched_harness/episode_records.jsonl",
        help="Path to write the resulting episode records JSONL.",
    )
    parser.add_argument(
        "--legacy-map",
        help=(
            "Explicit fallback map for legacy inline manifests that predate per-row map_file "
            "fields; matrix-derived rows always use their own map."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run all ablation scenarios."""
    args = parse_args()
    manifest_path = REPO_ROOT / args.manifest
    output_path = REPO_ROOT / args.output

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Build it first.")
        return 1

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_rows = manifest.get("manifest_rows", [])
    print(f"Loaded {len(manifest_rows)} manifest rows from {manifest_path}")

    # Build scenarios dynamically matching the manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    legacy_map_path = None if args.legacy_map is None else REPO_ROOT / args.legacy_map

    for idx, row in enumerate(manifest_rows):
        scenario_id = row["scenario_id"]
        planner = row["planner"]
        seed = int(row["seed"])
        arm = row["population_arm"]

        print(
            f"[{idx + 1}/{len(manifest_rows)}] Running scenario={scenario_id} arm={arm} planner={planner} seed={seed}"
        )

        try:
            # Assemble the runtime scenario and run the episode through the shared
            # harness helper so the emitted record carries the per-pedestrian control
            # trace the readiness gate requires (issue #5397).
            rec = run_manifest_row(
                row,
                map_path=legacy_map_path,
                scenario_path=manifest_path,
                horizon=600,
                dt=0.1,
            )
            records.append(rec)
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as e:
            # Narrow the handler (issue #4618 CI-B) to the runtime/config failures a
            # simulation run can plausibly raise; annotate with the failing arm/planner/seed
            # before re-raising so the traceback identifies the offending row.
            print(f"Exception during simulation (arm={arm}, planner={planner}, seed={seed}): {e}")
            raise

    # Write out JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully wrote {len(records)} episode records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
