"""Manual SVG map validation & bulk loading helper.

Run this file directly to:
1. Convert a single SVG (default: classic_bottleneck.svg)
2. Optionally bulk-validate every SVG under maps/svg_maps/
3. Provide a per-map summary of routes/zones and highlight invalid maps

Design notes:
 - This is intentionally a script (not part of automated test suite) to allow
   iterative map authoring without slowing CI.
 - Uses existing `convert_map` + `load_svg_maps` functions (lenient mode) and
   re-runs strict validation when requested.

Exit codes:
 - 0: All requested validations succeeded (or lenient skips only)
 - 1: Strict validation failed for one or more maps
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

from robot_sf.nav.svg_map_parser import convert_map, load_svg_maps

DEFAULT_SINGLE = "maps/svg_maps/classic_bottleneck.svg"
DEFAULT_DIR = "maps/svg_maps"


def _summarize_map(
    name: str, md
) -> None:  # md: MapDefinition (duck-typed here to avoid import churn)
    logger.info(
        "Map '{n}': robot_routes={rr} ped_routes={pr} robot_spawn={rs} robot_goal={rg} ped_spawn={ps} ped_goal={pg} obstacles={ob}",
        n=name,
        rr=len(getattr(md, "robot_routes", [])),
        pr=len(getattr(md, "ped_routes", [])),
        rs=len(getattr(md, "robot_spawn_zones", [])),
        rg=len(getattr(md, "robot_goal_zones", [])),
        ps=len(getattr(md, "ped_spawn_zones", [])),
        pg=len(getattr(md, "ped_goal_zones", [])),
        ob=len(getattr(md, "obstacles", [])),
    )


def _validate_single(path: str) -> bool:
    logger.info("Validating single SVG: {p}", p=path)
    try:
        md = convert_map(path)
        if md is None:
            logger.error("Conversion returned None (lenient failure).")
            return False
        _summarize_map(Path(path).stem, md)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Strict validation failed for {p}: {e}", p=path, e=exc)
        return False


def _bulk_validate(directory: str, strict: bool) -> tuple[list[str], list[str]]:
    logger.info("Bulk validating directory: {d} (strict={s})", d=directory, s=strict)
    valid: list[str] = []
    invalid: list[str] = []
    try:
        maps = load_svg_maps(directory, strict=False)  # always start lenient to inspect all
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list/load maps in directory {d}: {e}", d=directory, e=exc)
        return valid, [f"<dir_error>:{exc}"]

    for name, md in maps.items():
        _summarize_map(name, md)
        # Optional re-run for strict mode: we call convert_map again to surface exceptions
        if strict:
            svg_path = Path(directory) / f"{name}.svg"
            try:
                _ = convert_map(str(svg_path))  # will raise if structurally invalid under strict
                valid.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Strict re-validation failed: {n} ({e})", n=name, e=exc)
                invalid.append(name)
        else:
            valid.append(name)

    return valid, invalid


def _print_summary(header: str, collection: Iterable[str]) -> None:
    items = list(collection)
    if not items:
        logger.info("{h}: none", h=header)
        return
    logger.info("{h} ({n}): {items}", h=header, n=len(items), items=", ".join(sorted(items)))


@dataclass
class _Options:
    single: str | None
    all: bool
    directory: str
    strict: bool


def _parse_args(argv: list[str]) -> _Options:
    """Parse CLI arguments (kept minimal to avoid over‑engineering)."""
    parser = argparse.ArgumentParser(
        prog="svg_map_example",
        description=(
            "Manual SVG validation helper. Default behavior validates a single map. "
            "Use --all to bulk validate directory contents."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--single", metavar="FILE", help="Single SVG file to validate", default=None
    )
    parser.add_argument("--all", action="store_true", help="Validate all SVG maps in directory")
    parser.add_argument(
        "--dir",
        dest="directory",
        default=DEFAULT_DIR,
        metavar="DIR",
        help="Directory containing SVG maps",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Re-run strict structural validation for each map (bulk only)",
    )
    ns = parser.parse_args(argv)
    return _Options(single=ns.single, all=ns.all, directory=ns.directory, strict=ns.strict)


def main(argv: list[str]) -> int:
    opts = _parse_args(argv)
    strict = opts.strict or os.getenv("SVG_VALIDATE_STRICT") == "1"

    single_file = opts.single or (None if opts.all else DEFAULT_SINGLE)

    # Single file phase
    if single_file and not _validate_single(single_file):
        return 1

    # Bulk phase (optional)
    if not opts.all:
        return 0

    valid, invalid = _bulk_validate(opts.directory, strict=strict)
    _print_summary("Valid maps", valid)
    _print_summary("Invalid maps", invalid)

    if strict and invalid:
        return 1
    if not valid:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - manual script
    rc = main(sys.argv[1:])
    if rc != 0:
        logger.error("SVG validation script exiting with code {c}", c=rc)
    sys.exit(rc)
