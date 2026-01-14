"""Map inventory loading and enumeration.

This module handles discovering and cataloging SVG maps from the repository.
It respects manifest metadata (when available) and provides filtering
capabilities.

Alignment with data-model.md
-----------------------------
Implements the MapRecord entity from specs/001-map-verification/data-model.md.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from robot_sf.common.artifact_paths import get_repository_root


@dataclass
class MapRecord:
    """Logical representation of an SVG map and its metadata.

    Attributes
    ----------
    map_id : str
        Unique identifier (filename without extension)
    file_path : Path
        Absolute path to the SVG file
    tags : set[str]
        Map tags (e.g., 'pedestrian_only', 'benchmark', 'classic')
    ci_enabled : bool
        Whether this map should be validated in CI
    metadata : dict
        Parsed metadata (spawn zones, goals, pedestrian flags)
    last_modified : datetime
        File modification timestamp
    """

    map_id: str
    file_path: Path
    tags: set[str] = field(default_factory=set)
    ci_enabled: bool = True
    metadata: dict = field(default_factory=dict)
    last_modified: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate invariants."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Map file not found: {self.file_path}")


class MapInventory:
    """Discovers and manages the repository's SVG map inventory."""

    def __init__(self, maps_root: Path | None = None):
        """Initialize map inventory.

        Parameters
        ----------
        maps_root : Path | None
            Root directory containing SVG maps.
            If None, uses repository_root/maps/svg_maps/
        """
        if maps_root is None:
            repo_root = get_repository_root()
            maps_root = repo_root / "maps" / "svg_maps"

        self.maps_root = Path(maps_root).resolve()

        if not self.maps_root.exists():
            logger.warning(f"Maps root does not exist: {self.maps_root}")
            self.maps_root.mkdir(parents=True, exist_ok=True)

        self._inventory: dict[str, MapRecord] = {}
        self._load_inventory()

    def _load_inventory(self) -> None:
        """Load all SVG maps from maps_root.

        Populates internal inventory mapping of map_id -> MapRecord.
        """
        if not self.maps_root.exists():
            logger.warning(f"Cannot load inventory: {self.maps_root} does not exist")
            return

        svg_files = sorted(self.maps_root.rglob("*.svg"))
        logger.info(f"Discovered {len(svg_files)} SVG files in {self.maps_root}")

        for svg_file in svg_files:
            map_id = svg_file.stem
            if map_id in self._inventory:
                logger.warning(
                    "Duplicate map id '{}' at {}; already registered at {}",
                    map_id,
                    svg_file,
                    self._inventory[map_id].file_path,
                )
                continue

            # Get file modification time
            try:
                stat = svg_file.stat()
                last_modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            except OSError:
                last_modified = datetime.now(tz=UTC)

            # Infer tags from filename patterns
            tags = self._infer_tags(map_id)

            # Determine CI enablement (all maps enabled by default)
            # Future: read from manifest or map metadata
            ci_enabled = True

            record = MapRecord(
                map_id=map_id,
                file_path=svg_file,
                tags=tags,
                ci_enabled=ci_enabled,
                last_modified=last_modified,
            )

            self._inventory[map_id] = record
            logger.debug(f"Loaded map: {map_id} (tags: {tags})")

    def _infer_tags(self, map_id: str) -> set[str]:
        """Infer tags from map filename patterns.

        Parameters
        ----------
        map_id : str
            Map identifier (filename without extension)

        Returns
        -------
        set[str]
            Inferred tags
        """
        tags = set()

        # Classic scenario patterns
        if map_id.startswith("classic_"):
            tags.add("classic")

        # Pedestrian-only indicators
        if "ped" in map_id.lower() and "only" in map_id.lower():
            tags.add("pedestrian_only")

        # Simple maps
        if "simple" in map_id.lower():
            tags.add("simple")

        return tags

    def get_all_maps(self) -> list[MapRecord]:
        """Get all maps in inventory.

        Returns
        -------
        list[MapRecord]
            All discovered maps
        """
        return list(self._inventory.values())

    def get_ci_enabled_maps(self) -> list[MapRecord]:
        """Get maps that should be verified in CI.

        Returns
        -------
        list[MapRecord]
            CI-enabled maps
        """
        return [m for m in self._inventory.values() if m.ci_enabled]

    def get_map_by_id(self, map_id: str) -> MapRecord | None:
        """Get a specific map by ID.

        Parameters
        ----------
        map_id : str
            Map identifier

        Returns
        -------
        MapRecord | None
            Map record if found, None otherwise
        """
        return self._inventory.get(map_id)

    def get_maps_by_tag(self, tag: str) -> list[MapRecord]:
        """Get all maps with a specific tag.

        Parameters
        ----------
        tag : str
            Tag to filter by

        Returns
        -------
        list[MapRecord]
            Maps with the specified tag
        """
        return [m for m in self._inventory.values() if tag in m.tags]

    def __len__(self) -> int:
        """Return number of maps in inventory."""
        return len(self._inventory)

    def __iter__(self):
        """Iterate over all maps.

        Returns:
            Iterator[MapRecord]: Iterator over all map records in the inventory.
        """
        return iter(self._inventory.values())
