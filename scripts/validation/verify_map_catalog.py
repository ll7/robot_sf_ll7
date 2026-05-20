#!/usr/bin/env python3
"""Verify the capability-aware SVG map catalog is synchronized."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.maps.verification.svg_inspection import inspect_svg
from scripts.tools.generate_map_registry import (
    CATALOG_SCHEMA,
    PARSER_VERSION,
    REQUIRED_CAPABILITY_KEYS,
    sha256_file,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("maps/registry.yaml"),
        help="Path to the map registry YAML.",
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=None,
        help="Root directory containing SVG maps; defaults to <registry-dir>/svg_maps.",
    )
    parser.add_argument(
        "--scope",
        choices=("all",),
        default="all",
        help="Validation scope. Only 'all' is currently supported.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show parser debug/info logs while checking SVG files.",
    )
    return parser


def _configure_logging(*, verbose: bool) -> None:
    """Keep parser validation output concise by default."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "CRITICAL")


def _load_registry(path: Path) -> dict[str, Any]:
    """Load a registry YAML mapping.

    Args:
        path: Registry YAML path.

    Returns:
        dict[str, Any]: Parsed registry payload.

    Raises:
        ValueError: If the registry does not contain a mapping.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Registry must contain a mapping: {path}")
    return data


def _resolve_catalog_path(registry_path: Path, value: Any) -> Path | None:
    """Resolve a catalog row path relative to the registry directory."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (registry_path.parent / path).resolve()


def _relative_to_registry(registry_path: Path, path: Path) -> str:
    """Return a normalized path relative to the registry directory when possible."""
    try:
        return path.resolve().relative_to(registry_path.parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_capabilities(
    *,
    map_id: str,
    capabilities: Any,
    metadata: dict[str, Any],
) -> list[str]:
    """Validate reviewed capability booleans against parser facts."""
    errors: list[str] = []
    if not isinstance(capabilities, dict):
        return [f"{map_id}: capabilities must be a mapping"]

    for key in REQUIRED_CAPABILITY_KEYS:
        value = capabilities.get(key)
        if not isinstance(value, bool):
            errors.append(f"{map_id}: capabilities.{key} must be a boolean")

    if capabilities.get("robot_runtime") and not _has_robot_runtime_support(metadata):
        errors.append(f"{map_id}: capabilities.robot_runtime contradicts parser metadata")
    if capabilities.get("pedestrian_runtime") and not _has_pedestrian_runtime_support(metadata):
        errors.append(f"{map_id}: capabilities.pedestrian_runtime contradicts parser metadata")
    if capabilities.get("route_only") and not _has_route_only_support(metadata):
        errors.append(f"{map_id}: capabilities.route_only contradicts parser metadata")
    if capabilities.get("obstacle_source") and not metadata.get("has_obstacles"):
        errors.append(f"{map_id}: capabilities.obstacle_source contradicts parser metadata")
    errors.extend(
        _validate_benchmark_capability(
            map_id=map_id,
            capabilities=capabilities,
            metadata=metadata,
        ),
    )

    return errors


def _has_robot_runtime_support(metadata: dict[str, Any]) -> bool:
    """Return whether parser facts support robot runtime use."""
    return bool(
        metadata.get("has_robot_runtime_routes")
        and (
            metadata.get("has_explicit_robot_runtime_zones")
            or metadata.get("robot_route_only_mode")
        ),
    )


def _has_pedestrian_runtime_support(metadata: dict[str, Any]) -> bool:
    """Return whether parser facts support pedestrian runtime use."""
    return bool(
        metadata.get("has_pedestrian_runtime_routes")
        and (
            metadata.get("has_explicit_pedestrian_runtime_zones")
            or metadata.get("ped_route_only_mode")
        ),
    )


def _has_route_only_support(metadata: dict[str, Any]) -> bool:
    """Return whether parser facts include route-only endpoint-derived zones."""
    return bool(metadata.get("robot_route_only_mode") or metadata.get("ped_route_only_mode"))


def _validate_benchmark_capability(
    *,
    map_id: str,
    capabilities: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    """Validate benchmark-specific capability constraints."""
    if not capabilities.get("benchmark_candidate"):
        return []
    errors: list[str] = []
    if not capabilities.get("robot_runtime"):
        errors.append(f"{map_id}: capabilities.benchmark_candidate requires robot_runtime")
    if metadata.get("parser_limitation_codes"):
        errors.append(f"{map_id}: capabilities.benchmark_candidate has parser limitations")
    return errors


def _validate_catalog_path(
    *,
    row_name: str,
    row_path: Any,
    registry_path: Path,
) -> tuple[Path | None, list[str]]:
    """Resolve and validate a portable catalog row path."""
    svg_path = _resolve_catalog_path(registry_path, row_path)
    if svg_path is None:
        return None, [f"{row_name}: path must be a non-empty string"]

    errors: list[str] = []
    if Path(row_path).is_absolute():
        errors.append(f"{row_name}: path must be relative to the registry directory")
    try:
        svg_path.resolve().relative_to(registry_path.parent.resolve())
    except ValueError:
        errors.append(f"{row_name}: path must stay under {registry_path.parent}")
    return svg_path, errors


def _validate_row(
    *,
    row: Any,
    index: int,
    registry_path: Path,
    seen_ids: set[str],
    registered_paths: set[str],
) -> list[str]:
    """Validate one registry row and record its map ID/path if valid enough."""
    errors: list[str] = []
    if not isinstance(row, dict):
        return [f"maps[{index}]: row must be a mapping"]

    map_id = row.get("map_id")
    row_name = map_id if isinstance(map_id, str) and map_id else f"maps[{index}]"
    if not isinstance(map_id, str) or not map_id:
        errors.append(f"{row_name}: map_id must be a non-empty string")
    elif map_id in seen_ids:
        errors.append(f"{row_name}: duplicate map_id")
    else:
        seen_ids.add(map_id)

    svg_path, path_errors = _validate_catalog_path(
        row_name=row_name,
        row_path=row.get("path"),
        registry_path=registry_path,
    )
    errors.extend(path_errors)
    if svg_path is None:
        return errors
    if not svg_path.is_file():
        errors.append(f"{row_name}: path does not exist: {row.get('path')}")
        return errors

    registered_paths.add(_relative_to_registry(registry_path, svg_path))

    expected_hash = sha256_file(svg_path)
    if row.get("source_sha256") != expected_hash:
        errors.append(f"{row_name}: stale source_sha256")

    fresh_metadata = asdict(inspect_svg(svg_path).capability_metadata)
    if row.get("parser_metadata") != fresh_metadata:
        errors.append(f"{row_name}: stale parser_metadata")
    if row.get("role") == "benchmark_candidate" and not row.get("capabilities", {}).get(
        "benchmark_candidate",
    ):
        errors.append(f"{row_name}: role benchmark_candidate requires benchmark capability")
    if row.get("profile") == "benchmark_candidate" and not row.get("capabilities", {}).get(
        "benchmark_candidate",
    ):
        errors.append(f"{row_name}: profile benchmark_candidate requires benchmark capability")
    errors.extend(
        _validate_capabilities(
            map_id=row_name,
            capabilities=row.get("capabilities"),
            metadata=fresh_metadata,
        ),
    )

    return errors


def _find_unregistered_svgs(
    *,
    registry_path: Path,
    map_root: Path,
    registered_paths: set[str],
) -> list[str]:
    """Return errors for SVG files that have no catalog row."""
    if not map_root.exists():
        return [f"map root does not exist: {map_root}"]

    errors: list[str] = []
    for svg_path in sorted(map_root.rglob("*.svg")):
        catalog_path = _relative_to_registry(registry_path, svg_path)
        if catalog_path not in registered_paths:
            errors.append(f"unregistered SVG map: {catalog_path}")
    return errors


def validate_registry(registry_path: Path, map_root: Path | None = None) -> list[str]:
    """Validate registry schema, file synchronization, and parser-derived facts.

    Args:
        registry_path: Path to the registry YAML.
        map_root: SVG root for unregistered-file detection.

    Returns:
        list[str]: Human-readable validation errors; empty when valid.
    """
    registry_path = registry_path.resolve()
    map_root = (map_root or registry_path.parent / "svg_maps").resolve()
    errors: list[str] = []

    try:
        registry = _load_registry(registry_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"failed to load registry: {exc}"]

    if registry.get("version") != 2:
        errors.append("registry.version must be 2")
    if registry.get("schema") != CATALOG_SCHEMA:
        errors.append(f"registry.schema must be {CATALOG_SCHEMA}")
    if registry.get("parser_version") != PARSER_VERSION:
        errors.append(f"registry.parser_version must be {PARSER_VERSION}")

    rows = registry.get("maps")
    if not isinstance(rows, list):
        return [*errors, "registry.maps must be a list"]

    seen_ids: set[str] = set()
    registered_paths: set[str] = set()

    for index, row in enumerate(rows):
        errors.extend(
            _validate_row(
                row=row,
                index=index,
                registry_path=registry_path,
                seen_ids=seen_ids,
                registered_paths=registered_paths,
            ),
        )
    errors.extend(
        _find_unregistered_svgs(
            registry_path=registry_path,
            map_root=map_root,
            registered_paths=registered_paths,
        ),
    )

    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose)
    del args.scope
    errors = validate_registry(args.registry, args.map_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Map catalog is synchronized: {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
