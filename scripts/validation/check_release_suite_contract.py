#!/usr/bin/env python3
"""Check structural metadata completeness for a benchmark release suite manifest.

The checker can block on incomplete declarations, but it cannot freeze suites,
validate the referenced records, establish benchmark evidence, or publish a
release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.release_suite_contract import (
    ReleaseSuiteContractError,
    evaluate_release_suite_contract,
    load_release_suite_contract,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _render_text(report: dict[str, object]) -> str:
    """Render a compact human-readable report."""

    lines = [
        f"Release suite contract: {report['release_id']}",
        f"status: {report['status']}",
        f"claim boundary: {report['claim_boundary']}",
        (
            f"suites: {report['complete_suite_count']} complete, "
            f"{report['blocked_suite_count']} blocked"
        ),
    ]
    blockers = report["blockers"]
    if isinstance(blockers, list):
        lines.extend(f"blocker: {blocker}" for blocker in blockers)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker and return 0 for pass, 1 for blocked, or 2 for malformed."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    try:
        manifest = load_release_suite_contract(args.manifest)
        report = evaluate_release_suite_contract(manifest)
    except ReleaseSuiteContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
