#!/usr/bin/env python3
"""Promote a reviewed PR after CI passes on its exact head.

The conditional label records a completed review, not merge permission. The
existing merge-ready carrier and merge gates remain authoritative.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.dev.gh_pr_label_rest import add_label, get_label_names, remove_label

CONDITIONAL_LABEL = "merge-if-ci-green"
READY_LABEL = "merge-ready"


def read_ci(number: int, *, repo: str, head_sha: str) -> dict[str, Any]:
    """Read the repository's required-check rollup for one immutable PR head."""
    command = [
        sys.executable,
        str(Path(__file__).with_name("check_pr_ci_status.py")),
        str(number),
        "--repo",
        repo,
        "--expected-head-sha",
        head_sha,
        "--json",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        data = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"status": "error", "error": f"CI read unavailable: {exc}"}
    if not isinstance(data, dict) or result.returncode not in {0, 1}:
        return {"status": "error", "error": "CI read returned an invalid result"}
    return data


def _clear_conditional(number: int, *, repo: str, head_sha: str, base_sha: str) -> dict[str, Any]:
    """Finish promotion with readback, preserving merge-ready on cleanup failure."""
    cleared = remove_label(
        number,
        CONDITIONAL_LABEL,
        repo=repo,
        target="pr",
        expected_head_sha=head_sha,
        expected_base_sha=base_sha,
    )
    if cleared.get("status") != "ok":
        return {
            "status": "partial",
            "reason": "merge_ready_applied_conditional_label_retained",
            "number": number,
            "cleanup": cleared,
        }
    labels = get_label_names(number, repo=repo)
    if labels.get("status") != "ok" or READY_LABEL not in labels.get("labels", []):
        return {
            "status": "error",
            "error": "merge-ready label could not be verified",
            "number": number,
        }
    return {"status": "promoted", "number": number, "head_sha": head_sha}


def promote(number: int, *, repo: str, head_sha: str, base_sha: str) -> dict[str, Any]:
    """Add merge-ready only after green CI, then clear the conditional label."""
    labels = get_label_names(number, repo=repo)
    if labels.get("status") != "ok":
        return labels
    if CONDITIONAL_LABEL not in labels["labels"]:
        return {"status": "skipped", "reason": "conditional_label_absent", "number": number}
    ci = read_ci(number, repo=repo, head_sha=head_sha)
    if ci.get("status") == "error":
        return ci
    if str(ci.get("head_sha") or "").lower() != head_sha.lower():
        return {"status": "skipped", "reason": "head_moved", "number": number}
    overall = str((ci.get("checks") or {}).get("overall") or "")
    if overall != "success":
        return {
            "status": "waiting" if overall == "pending" else "blocked",
            "ci": overall,
            "number": number,
        }
    # A label sweep is a live ruling; do not re-create readiness after removal.
    labels = get_label_names(number, repo=repo)
    if labels.get("status") != "ok":
        return labels
    if CONDITIONAL_LABEL not in labels["labels"]:
        return {"status": "skipped", "reason": "conditional_label_removed", "number": number}
    ready = add_label(
        number,
        READY_LABEL,
        repo=repo,
        target="pr",
        expected_head_sha=head_sha,
        expected_base_sha=base_sha,
    )
    if ready.get("status") != "ok":
        return ready
    return _clear_conditional(number, repo=repo, head_sha=head_sha, base_sha=base_sha)


def main(argv: list[str] | None = None) -> int:
    """Parse the exact-head promotion request and emit a compact result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int)
    parser.add_argument("--repo", default="ll7/robot_sf_ll7")
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--expected-base-sha", required=True)
    args = parser.parse_args(argv)
    if args.number < 1 or any(
        len(sha) != 40 or any(char not in "0123456789abcdefABCDEF" for char in sha)
        for sha in (args.expected_head_sha, args.expected_base_sha)
    ):
        parser.error("number must be positive and expected SHAs must be full 40-hex values")
    result = promote(
        args.number,
        repo=args.repo,
        head_sha=args.expected_head_sha,
        base_sha=args.expected_base_sha,
    )
    print(json.dumps(result, sort_keys=True))
    return (
        0
        if result["status"] == "promoted"
        else 2
        if result["status"] in {"waiting", "skipped"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
