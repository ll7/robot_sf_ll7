#!/usr/bin/env python3
"""Check that a generated replay review manifest covers its whole catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.scenario_generation.review_manifest import validate_review_manifest


def main() -> int:
    """Validate one generated replay review packet and print its compact summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_review_manifest(args.catalog, args.review_manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
