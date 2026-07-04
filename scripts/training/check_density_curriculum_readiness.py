"""Fail-closed readiness check for issue #4018 density-curriculum comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.density_curriculum_readiness import (
    evaluate_density_curriculum_readiness,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the readiness check and return a fail-closed process status."""
    args = _parse_args()
    readiness = evaluate_density_curriculum_readiness(args.manifest)
    payload = readiness.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if readiness.status == "ready_diagnostic_smoke" else 1


if __name__ == "__main__":
    raise SystemExit(main())
