"""Map inventory management for loading and filtering SVG maps.

This module provides utilities to discover and enumerate SVG maps in the
repository, respecting metadata tags and CI enablement flags.
"""

from pathlib import Path
from datetime import datetime
from typing import Iterator

from loguru import logger

from robot_sf.maps.verification import MapRecord


DEFAULT_MAP_DIR = Path("maps/svg_maps")


def load_map_inventory(
    map_dir: Path | None = None,
    *,
    include_archived: bool = False,
) -> list[MapRecord]:
    """Load all SVG maps from the specified directory.
    
    Args:
        map_dir: Directory containing SVG maps (default: maps/svg_maps)
        include_archived: Whether to include archived/experimental maps
    
    Returns:
        List of MapRecord objects for discovered maps
    """
    if map_dir is None:
        map_dir = DEFAULT_MAP_DIR
    
    if not map_dir.exists():
        logger.warning(f"Map directory does not exist: {map_dir}")
        return []
    
    maps = []
    for svg_file in _iter_svg_files(map_dir):
        try:
            record = _create_map_record(svg_file)
            
            # Filter archived maps unless explicitly requested
            if not include_archived and "archived" in record.tags:
                logger.debug(f"Skipping archived map: {record.map_id}")
                continue
            
            maps.append(record)
        except Exception as e:
            logger.warning(f"Failed to load map {svg_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(maps)} maps from {map_dir}")
    return maps


def _iter_svg_files(map_dir: Path) -> Iterator[Path]:
    """Iterate over SVG files in the given directory.
    
    Args:
        map_dir: Directory to search for SVG files
        
    Yields:
        Path objects for each SVG file found
    """
    for file_path in map_dir.glob("*.svg"):
        if file_path.is_file():
            yield file_path


def _create_map_record(svg_file: Path) -> MapRecord:
    """Create a MapRecord from an SVG file.
    
    Args:
        svg_file: Path to the SVG file
        
    Returns:
        MapRecord with metadata extracted from the file
    """
    map_id = svg_file.stem
    
    # Extract tags from filename patterns
    tags = _extract_tags_from_filename(map_id)
    
    # Determine CI enablement (default True unless marked otherwise)
    ci_enabled = "ci_disabled" not in tags and "archived" not in tags
    
    # Get file modification time
    try:
        stat = svg_file.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime)
    except OSError:
        last_modified = datetime.now()
    
    # Placeholder metadata (will be enhanced in later phases)
    metadata = {
        "file_size_bytes": svg_file.stat().st_size if svg_file.exists() else 0,
    }
    
    return MapRecord(
        map_id=map_id,
        file_path=svg_file,
        tags=tags,
        ci_enabled=ci_enabled,
        metadata=metadata,
        last_modified=last_modified,
    )


def _extract_tags_from_filename(filename: str) -> set[str]:
    """Extract classification tags from filename patterns.
    
    Common patterns:
    - classic_* → 'classic' tag
    - debug_* → 'debug' tag
    - test_* → 'test' tag
    - *_archived → 'archived' tag
    
    Args:
        filename: Map filename (without extension)
        
    Returns:
        Set of extracted tags
    """
    tags = set()
    
    filename_lower = filename.lower()
    
    # Check for common prefixes
    if filename_lower.startswith("classic_"):
        tags.add("classic")
    if filename_lower.startswith("debug_"):
        tags.add("debug")
    if filename_lower.startswith("test_"):
        tags.add("test")
    
    # Check for archived marker
    if filename_lower.endswith("_archived") or "archived" in filename_lower:
        tags.add("archived")
    
    # Check for pedestrian-only maps
    if "ped_" in filename_lower or "pedestrian" in filename_lower:
        tags.add("pedestrian_focused")
    
    return tags


def get_ci_enabled_maps(maps: list[MapRecord]) -> list[MapRecord]:
    """Filter maps to only those enabled for CI.
    
    Args:
        maps: List of all maps
        
    Returns:
        Filtered list containing only CI-enabled maps
    """
    return [m for m in maps if m.ci_enabled]
