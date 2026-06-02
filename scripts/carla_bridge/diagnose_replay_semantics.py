"""Build conservative CARLA replay diagnostics from Robot-SF and CARLA JSON inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf_carla_bridge.diagnostics import (
    build_carla_replay_diagnostics,
    write_carla_replay_diagnostics_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-sf", required=True, type=Path, help="Robot-SF trace/episode JSON.")
    parser.add_argument("--carla", required=True, type=Path, help="CARLA live-replay summary JSON.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for diagnostics JSON, Markdown, and CSV outputs.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def main() -> int:
    """Run the diagnostics CLI."""
    args = parse_args()
    report = build_carla_replay_diagnostics(_load_json(args.robot_sf), _load_json(args.carla))
    outputs = write_carla_replay_diagnostics_outputs(report, args.output_dir)
    print(json.dumps(outputs, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
