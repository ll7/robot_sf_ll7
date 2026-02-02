#!/usr/bin/env python3
"""Generate a map registry YAML from SVG map files."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path("maps/svg_maps"),
        help="Root directory containing SVG maps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/registry.yaml"),
        help="Output path for the registry YAML.",
    )
    return parser


def _collect_svg_maps(map_root: Path) -> list[tuple[str, Path]]:
    """Collect SVG map files and assign map_ids based on filenames."""
    if not map_root.exists():
        raise FileNotFoundError(f"Map root not found: {map_root}")
    entries: list[tuple[str, Path]] = []
    seen: dict[str, Path] = {}
    for path in sorted(map_root.rglob("*.svg")):
        map_id = path.stem
        if map_id in seen:
            raise ValueError(
                f"Duplicate map_id '{map_id}' for {path} and {seen[map_id]}",
            )
        seen[map_id] = path
        entries.append((map_id, path))
    return entries


def _relative_or_absolute(path: Path, *, base: Path) -> str:
    """Return a path relative to base when possible; else absolute."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path.resolve())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    map_root = args.map_root
    output_path = args.output
    entries = _collect_svg_maps(map_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry_base = output_path.parent
    data = {
        "version": 1,
        "maps": [
            {"map_id": map_id, "path": _relative_or_absolute(path, base=registry_base)}
            for map_id, path in entries
        ],
    }
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    logger.info("Wrote {} map entries to {}", len(entries), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
