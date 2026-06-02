"""Generate hazard and ODD coverage rollups for benchmark campaign bundles."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from robot_sf.benchmark.hazard_odd_coverage import build_arg_parser, run_from_args

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the hazard/ODD coverage rollup CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_from_args(args)
    except Exception as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    sys.stdout.write(json.dumps(result["outputs"], indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
