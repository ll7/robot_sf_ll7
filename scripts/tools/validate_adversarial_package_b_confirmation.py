#!/usr/bin/env python3
"""Validate issue #3079 Package B confirmed-failure evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_confirmation import (
    validate_package_b_confirmation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Generated Package B comparison report JSON.")
    parser.add_argument(
        "confirmation",
        type=Path,
        help="Post-run Package B confirmation sidecar JSON.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Optional root used to resolve relative manifest and evidence paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the compact confirmation-gate JSON payload.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one confirmation sidecar and optionally write its gate payload."""
    args = parse_args(argv)
    gate = validate_package_b_confirmation(
        args.report,
        args.confirmation,
        artifact_root=args.artifact_root,
    )
    rendered = json.dumps(gate.to_payload(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if gate.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
