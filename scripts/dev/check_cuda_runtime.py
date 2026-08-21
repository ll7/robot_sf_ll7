#!/usr/bin/env python3
"""Classify the usable CUDA runtime and record a readiness receipt (issue #7712).

Torch CUDA builds report ``torch.cuda.is_available() == True`` even when the
NVML/driver path is unusable; the first real device operation then crashes with
an internal NVML assertion. The PR-readiness lane uses this probe to record the
environment class once, so CUDA-gated tests skip behind an explicit
``gpu_unavailable`` receipt instead of failing the lane as an ordinary code
regression.

Exit codes:
- 0: probe ran; classification written (any of usable/unavailable/unusable_nvml)
- 1: probe itself failed (module import error unrelated to classification)
- 2: receipt directory or receipt file could not be written
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from robot_sf.telemetry.gpu import classify_cuda_runtime


def _run_git(args: list[str]) -> str:
    """Return stdout for a git command, empty on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _receipt_path(receipt_dir: Path) -> Path:
    """Return the canonical readiness receipt path for the current branch."""
    branch = _run_git(["branch", "--show-current"])
    safe_branch = re.sub(r"[^A-Za-z0-9._-]+", "-", branch or "detached")
    return receipt_dir / f"cuda_runtime_{safe_branch}.json"


def main(argv: list[str] | None = None) -> int:
    """Run the CUDA runtime classification probe and write its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=Path("output/validation/pr_ready"),
        help="Directory for the gpu_unavailable receipt (default: output/validation/pr_ready)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the classification as a single JSON object to stdout",
    )
    args = parser.parse_args(argv)

    classification = classify_cuda_runtime()

    receipt = {
        "schema": "cuda_runtime_readiness.v1",
        "status": classification.status,
        "usable": classification.usable,
        "reason": classification.reason,
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "branch": _run_git(["branch", "--show-current"]) or "detached",
        "head_sha": _run_git(["rev-parse", "HEAD"]),
    }

    try:
        args.receipt_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create receipt directory {args.receipt_dir}: {exc}", file=sys.stderr)
        return 2
    receipt_path = _receipt_path(args.receipt_dir)
    try:
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"ERROR: cannot write receipt {receipt_path}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        # Keep stdout a single JSON document for automation; diagnostics go to stderr.
        print(json.dumps(receipt, sort_keys=True))
        print(f"Receipt written: {receipt_path}", file=sys.stderr)
    else:
        print(f"CUDA runtime classification: {classification.status}")
        print(f"Reason: {classification.reason}")
        if not classification.usable:
            print(
                "Readiness note: CUDA-gated tests will skip with a gpu_unavailable "
                "receipt; this is an environment classification, not GPU evidence."
            )
        print(f"Receipt written: {receipt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
