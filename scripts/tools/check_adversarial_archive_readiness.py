#!/usr/bin/env python3
"""Fail-closed readiness check for a certified adversarial failure archive.

This is the up-front gate for issue #3275. Before the proposal-vs-random runner
(``scripts/adversarial/run_proposal_vs_random_issue_2921.py``) consumes a *real*
certified failure archive, the archive must satisfy structural prerequisites or
the downstream disjoint split, overlap provenance, candidate certification, and
null tests cannot be computed.

Unlike the runner — which degrades a missing/malformed archive to a synthetic
fixture for plumbing — this checker fails closed: it never fabricates entries and
never falls back to synthetic data. It prints a JSON readiness report and exits
non-zero when the archive is not ready, so it is safe to use as a gate.

It makes no benchmark or held-out-yield claim; it only reports whether the
archive *input* is usable for a later held-out comparison.

Example:
    uv run python scripts/tools/check_adversarial_archive_readiness.py \\
        --archive output/adversarial/certified_failure_archive.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.adversarial.disjoint_evaluation import assess_archive_file_readiness


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fail-closed readiness check for a certified failure archive (issue #3275)."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the certified failure archive JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON readiness report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Assess the archive and return 0 when ready, 1 otherwise (fail-closed)."""
    args = parse_args(argv)
    report = assess_archive_file_readiness(args.archive)
    report_str = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(report_str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_str + "\n", encoding="utf-8")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
