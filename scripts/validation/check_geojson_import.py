"""Validate public GeoJSON provenance, convert it, and verify the resulting map loads."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from robot_sf.nav.geojson_map_builder import write_segment_map
from robot_sf.nav.geojson_map_provenance import validate_import_provenance
from robot_sf.training.scenario_loader import resolve_map_definition


def build_parser() -> argparse.ArgumentParser:
    """Build the public GeoJSON import checker CLI.

    Returns:
        Configured command-line argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Public raw GeoJSON extract")
    parser.add_argument(
        "manifest", type=Path, help="Source URL, licence, citation, and raw checksum"
    )
    parser.add_argument(
        "output", type=Path, help="Derived Robot SF YAML map (not a raw source file)"
    )
    parser.add_argument("--line-buffer-m", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate and convert one public GeoJSON input.

    Returns:
        Zero after provenance validation and successful map loading.
    """
    args = build_parser().parse_args(argv)
    validate_import_provenance(args.manifest, args.input)
    output = write_segment_map(args.input, args.output, line_buffer_m=args.line_buffer_m)
    map_def = resolve_map_definition(str(output), scenario_path=output)
    if map_def is None:
        raise RuntimeError(f"Converted map did not load: {output}")
    logger.info("GeoJSON import check passed: {}", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
