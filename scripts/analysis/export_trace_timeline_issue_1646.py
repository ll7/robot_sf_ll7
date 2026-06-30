"""Compatibility CLI for exporting ``simulation_timeline.v1`` artifacts.

The reusable implementation lives in ``robot_sf.analysis_workbench.simulation_timeline``.
This issue-specific path remains as a stable command for existing issue #1646 validation notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.analysis_workbench.simulation_timeline import (
    DEFAULT_FIXTURE,
    SIMULATION_TIMELINE_SCHEMA_FILE,
    SIMULATION_TIMELINE_SCHEMA_VERSION,
    SimulationTimelineValidationError,
    build_simulation_timeline,
    load_simulation_timeline_schema,
    validate_simulation_timeline,
    write_simulation_timeline,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExportValidationError,
)

__all__ = [
    "DEFAULT_FIXTURE",
    "SIMULATION_TIMELINE_SCHEMA_FILE",
    "SIMULATION_TIMELINE_SCHEMA_VERSION",
    "SimulationTimelineValidationError",
    "build_simulation_timeline",
    "load_simulation_timeline_schema",
    "main",
    "validate_simulation_timeline",
    "write_simulation_timeline",
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for timeline export generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=(
            "Input simulation_trace_export.v1 JSON. Defaults to the tracked "
            "planner_sanity_open trace-export fixture."
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="Output timeline JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the conversion command."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output = write_simulation_timeline(args.input, args.out)
    except (
        OSError,
        json.JSONDecodeError,
        SimulationTraceExportValidationError,
        SimulationTimelineValidationError,
    ) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    print(f"wrote simulation timeline {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
