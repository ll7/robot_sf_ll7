"""Compare Robot-SF and CARLA oracle replay metric JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.carla_bridge.parity import compare_oracle_replay_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-sf", required=True, type=Path, help="Robot-SF metrics JSON file.")
    parser.add_argument("--carla", required=True, type=Path, help="CARLA replay metrics JSON file.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output parity report JSON path."
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from a path."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def main() -> int:
    """Run the parity comparison CLI."""
    args = parse_args()
    report = compare_oracle_replay_metrics(_load_json(args.robot_sf), _load_json(args.carla))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote CARLA parity report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
