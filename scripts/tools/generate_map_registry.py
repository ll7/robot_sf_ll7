#!/usr/bin/env python3
"""Generate a capability-aware map registry YAML from SVG map files."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.maps.verification.svg_inspection import inspect_svg

CATALOG_SCHEMA = "robot_sf.map_catalog.v2"
PARSER_VERSION = "parser-capability-metadata.v1"
REQUIRED_CAPABILITY_KEYS = (
    "robot_runtime",
    "pedestrian_runtime",
    "route_only",
    "obstacle_source",
    "benchmark_candidate",
)


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the generated registry differs from the output file without writing it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show parser debug/info logs while scanning SVG files.",
    )
    return parser


def _configure_logging(*, verbose: bool) -> None:
    """Keep parser scan output concise by default."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "CRITICAL")


def _load_existing_registry(path: Path) -> dict[str, Any]:
    """Load an existing registry mapping if present."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Registry must contain a mapping: {path}")
    return data


def _existing_rows_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return existing registry map rows keyed by map id."""
    rows = registry.get("maps", [])
    if isinstance(rows, dict):
        rows = [{"map_id": map_id, "path": path} for map_id, path in rows.items()]
    if not isinstance(rows, list):
        return {}

    existing: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        map_id = row.get("map_id") or row.get("id")
        if isinstance(map_id, str):
            existing[map_id] = dict(row)
    return existing


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
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_capabilities(metadata: dict[str, Any]) -> dict[str, bool]:
    """Derive conservative default capability booleans from parser metadata."""
    robot_runtime = bool(
        metadata.get("has_robot_runtime_routes")
        and (
            metadata.get("has_explicit_robot_runtime_zones")
            or metadata.get("robot_route_only_mode")
        )
    )
    pedestrian_runtime = bool(
        metadata.get("has_pedestrian_runtime_routes")
        and (
            metadata.get("has_explicit_pedestrian_runtime_zones")
            or metadata.get("ped_route_only_mode")
        )
    )
    return {
        "robot_runtime": robot_runtime,
        "pedestrian_runtime": pedestrian_runtime,
        "route_only": bool(
            metadata.get("robot_route_only_mode") or metadata.get("ped_route_only_mode"),
        ),
        "obstacle_source": bool(metadata.get("has_obstacles")),
        "benchmark_candidate": False,
    }


def _default_profile(capabilities: dict[str, bool]) -> str:
    """Derive a default validation profile from capability booleans."""
    if capabilities.get("benchmark_candidate"):
        return "benchmark_candidate"
    if capabilities.get("robot_runtime"):
        return "robot_runtime"
    if capabilities.get("pedestrian_runtime"):
        return "pedestrian_runtime"
    if capabilities.get("route_only"):
        return "route_only"
    if capabilities.get("obstacle_source"):
        return "obstacle_only"
    return "template"


def _default_limitations(metadata: dict[str, Any], capabilities: dict[str, bool]) -> list[str]:
    """Build stable limitation labels from parser metadata."""
    limitations = list(metadata.get("parser_limitation_codes") or [])
    if capabilities.get("route_only"):
        limitations.append("route_derived_zones")
    if not metadata.get("has_robot_runtime_routes"):
        limitations.append("no_robot_routes")
    if not metadata.get("has_pedestrian_runtime_routes"):
        limitations.append("no_pedestrian_routes")
    return sorted(set(limitations))


def _build_row(
    *,
    map_id: str,
    path: Path,
    registry_base: Path,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one v2 registry row while preserving reviewed fields."""
    existing = existing or {}
    report = inspect_svg(path)
    metadata = asdict(report.capability_metadata)
    capabilities = dict(existing.get("capabilities") or _default_capabilities(metadata))
    profile = str(existing.get("profile") or _default_profile(capabilities))
    role = str(existing.get("role") or profile)
    limitations = list(existing.get("limitations") or _default_limitations(metadata, capabilities))

    return {
        "map_id": map_id,
        "path": _relative_or_absolute(path, base=registry_base),
        "source_sha256": sha256_file(path),
        "role": role,
        "capabilities": capabilities,
        "profile": profile,
        "parser_metadata": metadata,
        "limitations": limitations,
        "validation": existing.get("validation", {"status": "unchecked", "rule_ids": []}),
    }


def generate_registry(map_root: Path, output_path: Path) -> dict[str, Any]:
    """Generate a v2 registry payload from SVG files and any existing registry."""
    existing_registry = _load_existing_registry(output_path)
    existing_rows = _existing_rows_by_id(existing_registry)
    registry_base = output_path.parent
    entries = _collect_svg_maps(map_root)
    rows_by_id: dict[str, dict[str, Any]] = {}

    for map_id, path in entries:
        rows_by_id[map_id] = _build_row(
            map_id=map_id,
            path=path,
            registry_base=registry_base,
            existing=existing_rows.get(map_id),
        )

    for map_id, existing in existing_rows.items():
        if map_id in rows_by_id:
            continue
        existing_path = existing.get("path")
        if not isinstance(existing_path, str):
            continue
        resolved = Path(existing_path)
        if not resolved.is_absolute():
            resolved = (registry_base / resolved).resolve()
        if resolved.is_file():
            rows_by_id[map_id] = _build_row(
                map_id=map_id,
                path=resolved,
                registry_base=registry_base,
                existing=existing,
            )

    return {
        "version": 2,
        "schema": CATALOG_SCHEMA,
        "parser_version": PARSER_VERSION,
        "maps": [rows_by_id[map_id] for map_id in sorted(rows_by_id)],
    }


def _dump_registry(data: dict[str, Any]) -> str:
    """Serialize registry YAML deterministically."""
    return yaml.safe_dump(data, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose)
    data = generate_registry(args.map_root, args.output)
    rendered = _dump_registry(data)

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            print(f"ERROR: generated registry differs from {args.output}", file=sys.stderr)
            return 1
        print(f"Registry is up to date: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(data['maps'])} map entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
