#!/usr/bin/env python3
"""Run the issue #6962 lane-formation sensitivity diagnostic.

The output is diagnostic-only and must not be interpreted as benchmark or paper
evidence.  Defaults are intentionally bounded; pass explicit axes for a wider
surface.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from robot_sf.evidence.writers import write_json, write_sha256sums
from robot_sf.research.emergent_phenomena import (
    LITERATURE_CALIBRATION,
    RELEASED_DEFAULT_CALIBRATION,
)
from robot_sf.research.lane_formation_sensitivity import (
    DEFAULT_HALF_WIDTHS_M,
    DEFAULT_LENGTHS_M,
    DEFAULT_PEDESTRIAN_COUNTS,
    DEFAULT_SEEDS,
    DEFAULT_STEPS,
    run_lane_formation_sensitivity,
)

ISSUE_REF = "robot_sf_ll7#6962"
DEFAULT_OUTPUT_DIR = Path("output/diagnostics/issue_6962_lane_formation_sensitivity")
CALIBRATIONS = {
    RELEASED_DEFAULT_CALIBRATION.name: RELEASED_DEFAULT_CALIBRATION,
    LITERATURE_CALIBRATION.name: LITERATURE_CALIBRATION,
}


def _csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=_csv_ints, default=list(DEFAULT_SEEDS))
    parser.add_argument("--lengths-m", type=_csv_floats, default=list(DEFAULT_LENGTHS_M))
    parser.add_argument("--half-widths-m", type=_csv_floats, default=list(DEFAULT_HALF_WIDTHS_M))
    parser.add_argument(
        "--pedestrian-counts", type=_csv_ints, default=list(DEFAULT_PEDESTRIAN_COUNTS)
    )
    parser.add_argument("--steps", type=_csv_ints, default=list(DEFAULT_STEPS))
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Optional pinned ISO-8601 timestamp for reproducible manifest bytes.",
    )
    parser.add_argument(
        "--calibrations",
        choices=sorted(CALIBRATIONS),
        nargs="+",
        default=sorted(CALIBRATIONS),
    )
    parser.add_argument("--lane-segregation-thresholds", type=_csv_floats, default=[0.15, 0.3, 0.5])
    parser.add_argument("--lane-purity-thresholds", type=_csv_floats, default=[0.4, 0.6, 0.8])
    return parser.parse_args()


def main() -> int:
    """Run the diagnostic CLI.

    Returns:
        Process exit code.
    """
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = run_lane_formation_sensitivity(
        seeds=args.seeds,
        lengths_m=args.lengths_m,
        half_widths_m=args.half_widths_m,
        pedestrian_counts=args.pedestrian_counts,
        steps=args.steps,
        calibrations=[CALIBRATIONS[name] for name in args.calibrations],
        thresholds={
            "lane_segregation_index": args.lane_segregation_thresholds,
            "lane_purity": args.lane_purity_thresholds,
        },
    )
    payload["manifest"] = {
        **payload["manifest"],
        "generated_at_utc": args.generated_at
        or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(),
        "generation_command": shlex.join(
            ["uv", "run", "python", *[str(argument) for argument in sys.argv]]
        ),
    }

    non_native_rows = payload["manifest"]["execution_policy"]["non_native_rows"]
    if non_native_rows:
        raise RuntimeError(f"Non-native diagnostic rows are not acceptable: {non_native_rows}")

    write_json(output_dir / "manifest.json", payload["manifest"])
    write_json(output_dir / "summary.json", {"summary": payload["summary"]})
    write_json(output_dir / "rows.json", {"rows": payload["rows"]})
    if output_dir.parts[:3] == ("docs", "context", "evidence"):
        write_sha256sums(output_dir)

    print(
        json.dumps(
            {
                "issue": ISSUE_REF,
                "status": "computed",
                "output_dir": str(output_dir),
                "rows": len(payload["rows"]),
                "summary_rows": len(payload["summary"]),
                "claim_boundary": payload["manifest"]["claim_boundary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
