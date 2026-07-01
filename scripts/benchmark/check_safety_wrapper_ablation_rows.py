#!/usr/bin/env python3
"""Check issue #3501 safety-wrapper ablation rows before comparison.

The checker is intentionally opt-in and fail-closed. It verifies only that later
row artifacts are complete enough for a paired ``wrapper_off``/``wrapper_on``
comparison; it does not execute benchmarks or claim mitigation effectiveness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.safety_wrapper_ablation_manifest import (
    check_factorial_ablation_rows,
    load_safety_wrapper_ablation_rows,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        required=True,
        help="JSONL rows or JSON list to check for paired wrapper_off/wrapper_on completeness.",
    )
    parser.add_argument(
        "--out",
        help="Optional path for the JSON checker report. Defaults to stdout only.",
    )
    return parser.parse_args()


def _write_report(report: dict[str, object], out: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> int:
    """Run the ablation row checker."""
    args = parse_args()
    rows = load_safety_wrapper_ablation_rows(args.rows)
    report = check_factorial_ablation_rows(rows)
    _write_report(report, args.out)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
