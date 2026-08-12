#!/usr/bin/env python3
"""Render the reduced case-workbench publication figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.case_publication_figure import render_publication_figure


def main() -> int:
    """Parse arguments and render one figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["pdf", "svg", "png"], default="pdf")
    args = parser.parse_args()
    render_publication_figure(
        args.package,
        case_id=args.case_id,
        output=args.output,
        output_format=args.format,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
