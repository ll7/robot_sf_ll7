#!/usr/bin/env python3
"""Validate the issue #3079 Package B report matrix and claim boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_report import validate_package_b_report

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Generated Package B report JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the compact report-gate JSON payload.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one report and optionally write the gate payload."""
    args = parse_args(argv)
    gate = validate_package_b_report(args.report)
    payload = gate.to_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if gate.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
