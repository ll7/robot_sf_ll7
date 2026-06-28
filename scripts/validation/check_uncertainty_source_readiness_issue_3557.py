#!/usr/bin/env python3
"""Print issue #3557 uncertainty-source episode-run readiness inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.uncertainty_source_readiness import (  # noqa: E402
    build_uncertainty_source_readiness_inventory,
)


def main(argv: list[str] | None = None) -> int:
    """Run the readiness inventory CLI.

    Returns:
        int: ``1`` only when ``--fail-on-blocked`` is set and some source is blocked.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit 1 when any uncertainty source is not ready for episode-level runs.",
    )
    args = parser.parse_args(argv)

    report = build_uncertainty_source_readiness_inventory()
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(args.fail_on_blocked and report["blocked_sources"]))


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests when needed.
    raise SystemExit(main())
