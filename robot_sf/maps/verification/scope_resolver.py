"""Scope resolution for map verification filtering.

This module implements filtering logic to select which maps to verify
based on scope specifiers like 'all', 'ci', 'changed', or specific filenames.

Supported Scopes
----------------
- 'all': All maps in the repository
- 'ci': Only CI-enabled maps
- 'changed': Maps modified in git working tree (requires git)
- '<filename>.svg': Specific map by filename
- '<glob>': Glob pattern matching (e.g., 'classic_*.svg')
"""

import subprocess

from loguru import logger

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.maps.verification.map_inventory import MapInventory, MapRecord


class ScopeResolver:
    """Resolves scope specifiers to concrete map lists."""

    def __init__(self, inventory: MapInventory):
        """Initialize scope resolver.

        Parameters
        ----------
        inventory : MapInventory
            Map inventory to filter from
        """
        self.inventory = inventory
        self.repo_root = get_repository_root()

    def resolve(self, scope: str) -> list[MapRecord]:
        """Resolve a scope specifier to a list of maps.

        Parameters
        ----------
        scope : str
            Scope specifier ('all', 'ci', 'changed', filename, or glob)

        Returns
        -------
        list[MapRecord]
            Maps matching the scope

        Raises
        ------
        ValueError
            If scope is invalid or no maps match
        """
        scope = scope.strip().lower()

        if scope == "all":
            return self._resolve_all()
        elif scope == "ci":
            return self._resolve_ci()
        elif scope == "changed":
            return self._resolve_changed()
        elif scope.endswith(".svg"):
            return self._resolve_specific(scope)
        elif "*" in scope or "?" in scope:
            return self._resolve_glob(scope)
        else:
            # Try as exact map ID (without .svg)
            return self._resolve_specific(f"{scope}.svg")

    def _resolve_all(self) -> list[MapRecord]:
        """Resolve 'all' scope."""
        maps = self.inventory.get_all_maps()
        logger.info(f"Scope 'all': {len(maps)} maps")
        return maps

    def _resolve_ci(self) -> list[MapRecord]:
        """Resolve 'ci' scope."""
        maps = self.inventory.get_ci_enabled_maps()
        logger.info(f"Scope 'ci': {len(maps)} CI-enabled maps")
        return maps

    def _resolve_changed(self) -> list[MapRecord]:
        """Resolve 'changed' scope using git status.

        Returns
        -------
        list[MapRecord]
            Maps with uncommitted changes
        """
        try:
            # Get changed files from git
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )

            changed_files = result.stdout.strip().split("\n")
            changed_files = [f for f in changed_files if f]  # Remove empty

            # Filter for SVG maps
            maps_dir = self.repo_root / "maps" / "svg_maps"
            changed_maps = []

            for file_path in changed_files:
                full_path = (self.repo_root / file_path).resolve()

                # Check if it's in svg_maps and is an SVG
                if full_path.parent == maps_dir and full_path.suffix == ".svg":
                    map_id = full_path.stem
                    map_record = self.inventory.get_map_by_id(map_id)
                    if map_record:
                        changed_maps.append(map_record)

            logger.info(f"Scope 'changed': {len(changed_maps)} modified maps")

            if not changed_maps:
                logger.warning("No changed SVG maps found in working tree")

            return changed_maps

        except subprocess.CalledProcessError as e:
            # Git command failed (e.g., not a git repo or git error) - fall back to all maps
            logger.warning(f"Failed to query git for changed files: {e}")
            logger.info("Falling back to 'all' scope")
            return self._resolve_all()
        except FileNotFoundError:
            # Git executable not found on system - fall back to all maps
            logger.warning("git command not found; cannot resolve 'changed' scope")
            logger.info("Falling back to 'all' scope")
            return self._resolve_all()

    def _resolve_specific(self, filename: str) -> list[MapRecord]:
        """Resolve a specific filename.

        Parameters
        ----------
        filename : str
            SVG filename (with or without .svg extension)

        Returns
        -------
        list[MapRecord]
            Single map if found

        Raises
        ------
        ValueError
            If map not found
        """
        if not filename.endswith(".svg"):
            filename = f"{filename}.svg"

        map_id = filename[:-4]  # Remove .svg
        map_record = self.inventory.get_map_by_id(map_id)

        if not map_record:
            raise ValueError(f"Map not found: {filename}")

        logger.info(f"Scope specific: 1 map ({map_id})")
        return [map_record]

    def _resolve_glob(self, pattern: str) -> list[MapRecord]:
        """Resolve a glob pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern (e.g., 'classic_*.svg')

        Returns
        -------
        list[MapRecord]
            Maps matching the pattern

        Raises
        ------
        ValueError
            If no maps match
        """
        maps_dir = self.repo_root / "maps" / "svg_maps"
        matched_files = list(maps_dir.glob(pattern))

        matched_maps = []
        for svg_file in matched_files:
            map_id = svg_file.stem
            map_record = self.inventory.get_map_by_id(map_id)
            if map_record:
                matched_maps.append(map_record)

        if not matched_maps:
            raise ValueError(f"No maps match pattern: {pattern}")

        logger.info(f"Scope glob '{pattern}': {len(matched_maps)} maps")
        return matched_maps
