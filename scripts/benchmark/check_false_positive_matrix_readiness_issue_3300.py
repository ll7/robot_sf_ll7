#!/usr/bin/env python3
"""Fail-closed readiness check for issue #3300 stronger replay matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.false_positive_matrix_readiness import (
    STATUS_BLOCKED,
    check_false_positive_matrix_readiness,
)

BLOCKED_EXIT_CODE = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-config", type=Path, required=True)
    parser.add_argument("--perturbed-config", type=Path, required=True)
    parser.add_argument("--min-scenarios", type=int, default=2)
    parser.add_argument("--min-seeds", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the issue #3300 matrix readiness CLI."""

    args = _build_parser().parse_args(argv)
    readiness = check_false_positive_matrix_readiness(
        args.nominal_config,
        args.perturbed_config,
        min_scenarios=args.min_scenarios,
        min_seeds=args.min_seeds,
    )
    print(json.dumps(readiness.to_dict(), indent=2, sort_keys=True))
    if readiness.status == STATUS_BLOCKED:
        return BLOCKED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
