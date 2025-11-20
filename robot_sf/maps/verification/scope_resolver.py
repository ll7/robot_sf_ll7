"""Scope filtering for map verification workflows.

This module provides utilities to filter maps based on different scopes:
- 'all': All maps (excluding archived unless specified)
- 'ci': Only CI-enabled maps
- 'changed': Maps that have been modified in the current git working tree
"""

import subprocess
from pathlib import Path
from typing import Literal

from loguru import logger

from robot_sf.maps.verification import MapRecord
from robot_sf.maps.verification.map_inventory import (
    load_map_inventory,
    get_ci_enabled_maps,
    DEFAULT_MAP_DIR,
)


ScopeType = Literal["all", "ci", "changed"]


def resolve_scope(
    scope: ScopeType,
    map_dir: Path | None = None,
) -> list[MapRecord]:
    """Resolve a scope string to a list of maps to verify.
    
    Args:
        scope: One of 'all', 'ci', or 'changed'
        map_dir: Directory containing maps (default: maps/svg_maps)
        
    Returns:
        List of MapRecord objects matching the scope
        
    Raises:
        ValueError: If scope is not recognized
    """
    if map_dir is None:
        map_dir = DEFAULT_MAP_DIR
    
    logger.info(f"Resolving scope: {scope}")
    
    if scope == "all":
        maps = load_map_inventory(map_dir, include_archived=False)
    elif scope == "ci":
        all_maps = load_map_inventory(map_dir, include_archived=False)
        maps = get_ci_enabled_maps(all_maps)
    elif scope == "changed":
        all_maps = load_map_inventory(map_dir, include_archived=False)
        changed_files = _get_changed_svg_files(map_dir)
        maps = _filter_by_changed_files(all_maps, changed_files)
    else:
        raise ValueError(f"Unknown scope: {scope}")
    
    logger.info(f"Resolved {len(maps)} maps for scope '{scope}'")
    return maps


def _get_changed_svg_files(map_dir: Path) -> set[Path]:
    """Get list of SVG files that have been modified in git.
    
    Uses git status to find:
    - Modified files (M)
    - Added files (A)
    - Untracked files (??)
    
    Args:
        map_dir: Directory to check for changes
        
    Returns:
        Set of absolute paths to changed SVG files
    """
    try:
        # Run git status in porcelain mode for easy parsing
        result = subprocess.run(
            ["git", "status", "--porcelain", str(map_dir)],
            capture_output=True,
            text=True,
            check=True,
            cwd=map_dir.parent if map_dir.is_absolute() else Path.cwd(),
        )
        
        changed_files = set()
        for line in result.stdout.splitlines():
            # Porcelain format: XY filename
            # X = index status, Y = working tree status
            if len(line) < 3:
                continue
            
            status = line[:2]
            filename = line[3:].strip()
            
            # Filter for relevant statuses and SVG files
            if filename.endswith(".svg"):
                file_path = Path(filename)
                if not file_path.is_absolute():
                    file_path = (map_dir.parent / filename).resolve()
                changed_files.add(file_path)
        
        logger.debug(f"Found {len(changed_files)} changed SVG files")
        return changed_files
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get git status: {e}")
        logger.warning("Falling back to treating all maps as changed")
        return set()
    except FileNotFoundError:
        logger.warning("git command not found; cannot determine changed files")
        logger.warning("Falling back to treating all maps as changed")
        return set()


def _filter_by_changed_files(
    maps: list[MapRecord],
    changed_files: set[Path],
) -> list[MapRecord]:
    """Filter maps to only those whose files have changed.
    
    Args:
        maps: All maps to consider
        changed_files: Set of paths that have been modified
        
    Returns:
        Filtered list of maps
    """
    if not changed_files:
        # If we couldn't determine changed files, return all maps
        logger.warning("No changed files detected; returning all maps")
        return maps
    
    filtered = []
    for map_record in maps:
        # Resolve both paths to absolute for comparison
        map_path = map_record.file_path.resolve()
        
        if map_path in changed_files:
            filtered.append(map_record)
            logger.debug(f"Including changed map: {map_record.map_id}")
    
    return filtered
