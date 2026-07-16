#!/usr/bin/env python3
"""Run the issue #3574 mean-matched heterogeneity ablation campaign.

Runs simulations for each row of the dry-run manifest, writes episode records,
and outputs a complete JSONL dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TextIO

from robot_sf.benchmark.campaign_logging import (
    add_campaign_logging_argument,
    configure_campaign_logging,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FSYNC_EVERY = 10


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
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=DEFAULT_FSYNC_EVERY,
        help="Call fsync after this many completed episode records (default: 10).",
    )
    add_campaign_logging_argument(parser)
    return parser.parse_args()


def _run_manifest_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Import the heavy runtime only after campaign logging has been configured."""

    from robot_sf.benchmark.heterogeneous_population_ablation_runner import (
        run_manifest_row,
    )

    return run_manifest_row(*args, **kwargs)


def _append_episode_record(
    output_file: TextIO,
    record: dict[str, Any],
    *,
    completed_count: int,
    fsync_every: int,
) -> None:
    """Append one complete JSON line, flush it, and periodically sync it to disk."""

    output_file.write(json.dumps(record) + "\n")
    output_file.flush()
    if completed_count % fsync_every == 0:
        os.fsync(output_file.fileno())


def main() -> int:
    """Run all ablation scenarios."""
    args = parse_args()
    configure_campaign_logging(debug=args.debug)
    if args.fsync_every <= 0:
        raise ValueError("--fsync-every must be positive")
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

    legacy_map_path = None if args.legacy_map is None else REPO_ROOT / args.legacy_map
    completed_count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for idx, row in enumerate(manifest_rows):
            scenario_id = row["scenario_id"]
            planner = row["planner"]
            seed = int(row["seed"])
            arm = row["population_arm"]

            print(
                f"[{idx + 1}/{len(manifest_rows)}] Running scenario={scenario_id} "
                f"arm={arm} planner={planner} seed={seed}"
            )

            try:
                # Assemble the runtime scenario and run the episode through the shared
                # harness helper so the emitted record carries the per-pedestrian control
                # trace the readiness gate requires (issue #5397).
                record = _run_manifest_row(
                    row,
                    map_path=legacy_map_path,
                    scenario_path=manifest_path,
                    horizon=600,
                    dt=0.1,
                )
                completed_count += 1
                _append_episode_record(
                    output_file,
                    record,
                    completed_count=completed_count,
                    fsync_every=args.fsync_every,
                )
            except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                # Narrow the handler (issue #4618 CI-B) to the runtime/config failures a
                # simulation run can plausibly raise; annotate with the failing arm/planner/seed
                # before re-raising so the traceback identifies the offending row.
                print(
                    "Exception during simulation "
                    f"(arm={arm}, planner={planner}, seed={seed}): {exc}"
                )
                raise
        if completed_count and completed_count % args.fsync_every:
            os.fsync(output_file.fileno())

    print(f"Successfully wrote {completed_count} episode records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
