#!/usr/bin/env python3
"""Run the issue #6969 preregistered Stage A parameter screen."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pysocialforce as pysf

from robot_sf.evidence.writers import write_json, write_sha256sums
from robot_sf.research.lane_formation_parameter_screen import (
    DEFAULT_PARAMETER_SCREEN_PROFILES,
    DEFAULT_REFERENCE_SEEDS,
    DEFAULT_SAMPLING_STRIDES,
    ReferenceProtocol,
    run_parameter_screen,
)

ISSUE_REF = "robot_sf_ll7#6969"
DEFAULT_OUTPUT_DIR = Path("output/diagnostics/issue_6969_parameter_screen")


def _csv_ints(value: str) -> list[int]:
    """Parse a comma-separated integer list."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=_csv_ints, default=list(DEFAULT_REFERENCE_SEEDS))
    parser.add_argument("--profiles", type=int, default=DEFAULT_PARAMETER_SCREEN_PROFILES)
    parser.add_argument("--profile-seed", type=int, default=6969)
    parser.add_argument(
        "--sampling-strides", type=_csv_ints, default=list(DEFAULT_SAMPLING_STRIDES)
    )
    parser.add_argument("--length-m", type=float, default=24.0)
    parser.add_argument("--half-width-m", type=float, default=2.5)
    parser.add_argument("--pedestrian-count", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--observation-steps", type=int, default=200)
    parser.add_argument("--recycle-margin-m", type=float, default=0.2)
    parser.add_argument("--lane-offset-m", type=float, default=0.85)
    parser.add_argument("--entry-y-span-m", type=float, default=1.2)
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Optional pinned ISO-8601 timestamp for reproducible manifest bytes.",
    )
    return parser.parse_args()


def main() -> int:
    """Run and serialize Stage A."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = ReferenceProtocol(
        length_m=args.length_m,
        half_width_m=args.half_width_m,
        n_pedestrians=args.pedestrian_count,
        warmup_steps=args.warmup_steps,
        observation_steps=args.observation_steps,
        recycle_margin_m=args.recycle_margin_m,
        lane_offset_m=args.lane_offset_m,
        entry_y_span_m=args.entry_y_span_m,
    )
    payload = run_parameter_screen(
        protocol=protocol,
        seeds=args.seeds,
        n_profiles=args.profiles,
        profile_seed=args.profile_seed,
        sampling_strides=args.sampling_strides,
    )
    generated_at = args.generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    payload["manifest"] = {
        **payload["manifest"],
        "generated_at_utc": generated_at,
        "git_commit": _git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pysocialforce": getattr(pysf, "__version__", "unknown"),
        },
        "generation_command": shlex.join(
            ["uv", "run", "python", *[str(argument) for argument in sys.argv]]
        ),
    }
    write_json(output_dir / "manifest.json", payload["manifest"])
    write_json(output_dir / "profiles.json", {"profiles": payload["profiles"]})
    write_json(output_dir / "summary.json", {"summary": payload["summary"]})
    write_json(output_dir / "rows.json", {"rows": payload["rows"]})
    write_sha256sums(output_dir)
    print(
        json.dumps(
            {
                "issue": ISSUE_REF,
                "stage": "A",
                "status": "computed",
                "output_dir": str(output_dir),
                "profiles": len(payload["profiles"]),
                "rows": len(payload["rows"]),
                "claim_boundary": payload["manifest"]["claim_boundary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
