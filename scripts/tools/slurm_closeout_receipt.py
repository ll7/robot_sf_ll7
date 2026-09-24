#!/usr/bin/env python3
"""CLI wrapper for the packaged Slurm closeout receipt validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.slurm_closeout_receipt import (
    SCHEMA_VERSION,
    validate_file,
    validate_payload,
)

__all__ = ["SCHEMA_VERSION", "validate_file", "validate_payload"]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--expected-campaign-id")
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--expected-job-id")
    args = parser.parse_args()
    problems = validate_file(
        args.receipt,
        expected_campaign_id=args.expected_campaign_id,
        expected_source_sha=args.expected_source_sha,
        expected_job_id=args.expected_job_id,
    )
    print(json.dumps({"status": "pass" if not problems else "blocked", "problems": problems}))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(_main())
