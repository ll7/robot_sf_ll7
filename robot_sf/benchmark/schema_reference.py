"""
SchemaReference entity for runtime schema loading and caching.

This module provides the SchemaReference entity that manages runtime loading
of schemas from canonical locations with caching to improve performance.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from robot_sf.benchmark.schemas.episode_schema import EpisodeSchema

logger = logging.getLogger(__name__)


class SchemaReference:
    """
    Entity for runtime schema loading and caching.

    This class provides a mechanism to load schemas from canonical locations
    with caching to avoid repeated file I/O operations.
    """

    # Global cache for loaded schemas
    _schema_cache: Dict[str, EpisodeSchema] = {}

    def __init__(self, schema_path: str, version: str):
        """
        Initialize SchemaReference.

        Args:
            schema_path: Relative path from package root to schema file
            version: Expected schema version identifier

        Raises:
            ValueError: If schema_path or version is invalid
        """
        if not schema_path:
            raise ValueError("schema_path cannot be empty")
        if not version:
            raise ValueError("version cannot be empty")

        self.schema_path = schema_path
        self.version = version
        self._loaded_schema: Optional[EpisodeSchema] = None
        self._cache_key = f"{schema_path}:{version}"

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the global schema cache."""
        cls._schema_cache.clear()
        logger.debug("Schema cache cleared")

    def load_schema(self) -> EpisodeSchema:
        """
        Load the schema from the canonical location.

        Returns:
            Loaded EpisodeSchema instance

        Raises:
            FileNotFoundError: If schema file doesn't exist
            ValueError: If schema cannot be loaded or validated
        """
        # Check cache first
        if self._cache_key in self._schema_cache:
            logger.debug("Schema loaded from cache: %s", self._cache_key)
            return self._schema_cache[self._cache_key]

        # Resolve the full path
        full_path = self._resolve_schema_path()

        try:
            # Load the schema
            schema = EpisodeSchema(full_path)
            logger.debug("Schema loaded from file: %s", full_path)

            # Validate version matches expected
            if schema.version != self.version:
                raise ValueError(
                    f"Schema version mismatch: expected {self.version}, "
                    f"got {schema.version} in {full_path}"
                )

            # Cache the loaded schema
            self._schema_cache[self._cache_key] = schema
            self._loaded_schema = schema

            return schema

        except Exception as e:
            logger.error("Failed to load schema from %s: %s", full_path, e)
            raise

    def _resolve_schema_path(self) -> Path:
        """
        Resolve the schema path relative to the package root.

        Returns:
            Absolute path to the schema file

        Raises:
            FileNotFoundError: If resolved path doesn't exist
        """
        # Get the robot_sf package root
        import robot_sf

        package_root = Path(robot_sf.__file__).parent

        # Resolve the relative path
        full_path = package_root / self.schema_path

        if not full_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {full_path} "
                f"(resolved from package root + '{self.schema_path}')"
            )

        return full_path

    @property
    def loaded_schema(self) -> Optional[EpisodeSchema]:
        """Get the currently loaded schema, or None if not loaded."""
        return self._loaded_schema

    @property
    def is_loaded(self) -> bool:
        """Check if the schema has been loaded."""
        return self._loaded_schema is not None

    def validate_episode_data(self, episode_data: Dict[str, Any]) -> None:
        """
        Validate episode data against the loaded schema.

        Args:
            episode_data: Episode data to validate

        Raises:
            RuntimeError: If schema not loaded
            ValueError: If validation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Schema not loaded. Call load_schema() first.")

        self._loaded_schema.validate_episode_data(episode_data)

    def get_schema_property(self, property_name: str) -> Dict[str, Any]:
        """
        Get a property definition from the loaded schema.

        Args:
            property_name: Name of the property

        Returns:
            Property schema definition

        Raises:
            RuntimeError: If schema not loaded
            KeyError: If property doesn't exist
        """
        if not self.is_loaded:
            raise RuntimeError("Schema not loaded. Call load_schema() first.")

        return self._loaded_schema.get_property_schema(property_name)

    def __str__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"SchemaReference(path={self.schema_path}, version={self.version}, status={status})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"SchemaReference(schema_path='{self.schema_path}', version='{self.version}')"

    def __eq__(self, other: object) -> bool:
        """Check equality based on path and version."""
        if not isinstance(other, SchemaReference):
            return False
        return self.schema_path == other.schema_path and self.version == other.version

    def __hash__(self) -> int:
        """Hash based on path and version."""
        return hash((self.schema_path, self.version))
