"""Map registry module for centralized map asset management.

This module provides a registry abstraction for loading and validating map assets,
ensuring all map access goes through a single canonical interface.

Key Functions
-------------
- build_registry: Scans canonical directories and builds map registry
- list_ids: Returns all available map IDs
- get: Retrieves map definition by ID
- validate_map_id: Validates map ID and provides helpful error messages

Design Principles
-----------------
- Single source of truth for map locations (maps/svg_maps/ and maps/metadata/)
- Lazy loading with module-level caching for performance
- Clear error messages listing available IDs on validation failures
- Backward compatibility with existing map IDs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition

# Module-level cache for registry
_CACHE: dict[str, tuple[Path, Path | None]] | None = None


def get_canonical_paths() -> tuple[Path, Path]:
    """Get canonical paths for SVG and metadata directories.
    
    Returns
    -------
    tuple[Path, Path]
        Tuple of (svg_dir, metadata_dir) paths
    """
    # Get the repository root (3 levels up from this file)
    repo_root = Path(__file__).parent.parent.parent
    svg_dir = repo_root / "maps" / "svg_maps"
    metadata_dir = repo_root / "maps" / "metadata"
    
    return svg_dir, metadata_dir


def build_registry() -> dict[str, tuple[Path, Path | None]]:
    """Build map registry by scanning canonical directories.
    
    Scans the canonical SVG and metadata directories to build a registry
    mapping map IDs to their asset paths. SVG files are considered primary;
    metadata JSON files are optional.
    
    Returns
    -------
    dict[str, tuple[Path, Path | None]]
        Registry mapping map_id -> (svg_path, json_path or None)
    """
    svg_dir, metadata_dir = get_canonical_paths()
    registry = {}
    
    # Ensure directories exist
    if not svg_dir.exists():
        logger.warning(f"SVG directory does not exist: {svg_dir}")
        return registry
    
    # Scan SVG files
    for svg_file in svg_dir.glob("*.svg"):
        map_id = svg_file.stem  # filename without extension
        
        # Look for corresponding metadata JSON
        json_path = None
        if metadata_dir.exists():
            json_file = metadata_dir / f"{map_id}.json"
            if json_file.exists():
                json_path = json_file
        
        registry[map_id] = (svg_file, json_path)
        logger.debug(f"Registered map '{map_id}': SVG={svg_file.name}, JSON={json_path.name if json_path else 'None'}")
    
    logger.info(f"Built registry with {len(registry)} maps")
    return registry


def get_registry() -> dict[str, tuple[Path, Path | None]]:
    """Get cached registry, building it if necessary.
    
    Returns
    -------
    dict[str, tuple[Path, Path | None]]
        Cached registry mapping map_id -> (svg_path, json_path or None)
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = build_registry()
    return _CACHE


def list_ids() -> list[str]:
    """List all available map IDs in alphabetical order.
    
    Returns
    -------
    list[str]
        Sorted list of map IDs
    """
    registry = get_registry()
    return sorted(registry.keys())


def validate_map_id(map_id: str) -> None:
    """Validate that a map ID exists in the registry.
    
    Parameters
    ----------
    map_id : str
        The map ID to validate
    
    Raises
    ------
    ValueError
        If map_id is not found in registry, with message listing available IDs
    
    Examples
    --------
    >>> validate_map_id("example_map")  # Passes if map exists
    >>> validate_map_id("invalid_map")  # Raises ValueError with available IDs
    """
    registry = get_registry()
    if map_id not in registry:
        available = list_ids()
        raise ValueError(
            f"Map ID '{map_id}' not found in registry. "
            f"Available map IDs ({len(available)}): {', '.join(available[:10])}"
            + (f", ... and {len(available) - 10} more" if len(available) > 10 else "")
        )


def get(map_id: str) -> tuple[Path, Path | None]:
    """Get map asset paths by ID.
    
    Parameters
    ----------
    map_id : str
        The map ID to retrieve
    
    Returns
    -------
    tuple[Path, Path | None]
        Tuple of (svg_path, json_path or None)
    
    Raises
    ------
    ValueError
        If map_id is not found in registry
    
    Examples
    --------
    >>> svg_path, json_path = get("example_map")
    >>> print(svg_path)
    /path/to/maps/svg_maps/example_map.svg
    """
    validate_map_id(map_id)
    registry = get_registry()
    return registry[map_id]


def clear_cache() -> None:
    """Clear the registry cache.
    
    Useful for testing or when map files are added/removed dynamically.
    """
    global _CACHE
    _CACHE = None
    logger.debug("Registry cache cleared")
