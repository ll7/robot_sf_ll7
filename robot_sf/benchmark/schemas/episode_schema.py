"""
EpisodeSchema entity for consolidating episode schema definitions.

This module provides the EpisodeSchema entity that encapsulates the JSON schema
definition for episode metrics data structures, providing a programmatic interface
for schema validation and metadata extraction.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)


class EpisodeSchema:
    """
    Entity representing a JSON schema definition for episode metrics data.

    This class encapsulates the schema structure and provides methods for
    validation, metadata extraction, and schema evolution tracking.
    """

    def __init__(self, schema_path: Path):
        """
        Initialize EpisodeSchema from a JSON schema file.

        Args:
            schema_path: Path to the JSON schema file

        Raises:
            FileNotFoundError: If schema file doesn't exist
            json.JSONDecodeError: If schema file contains invalid JSON
            ValueError: If schema is not a valid JSON Schema
        """
        self.schema_path = schema_path
        self._schema_data: dict[str, Any] | None = None
        self._version: str | None = None

        # Load and validate schema on initialization
        self._load_schema()

    def _load_schema(self) -> None:
        """Load and validate the schema from file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        try:
            with open(self.schema_path, encoding="utf-8") as f:
                self._schema_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file {self.schema_path}: {e}")

        self._validate_schema_structure()
        self._extract_version()

    def _validate_schema_structure(self) -> None:
        """Validate that the loaded schema has required structure."""
        if not isinstance(self._schema_data, dict):
            raise ValueError("Schema must be a JSON object")

        required_fields = ["$schema", "$id", "title", "type", "properties", "required"]
        for field in required_fields:
            if field not in self._schema_data:
                raise ValueError(f"Schema missing required field: {field}")

        # Validate JSON Schema draft
        if self._schema_data["$schema"] != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError("Schema must use JSON Schema draft 2020-12")

        # Validate root type
        if self._schema_data["type"] != "object":
            raise ValueError("Schema root type must be 'object'")

        # Validate required episode fields
        required_props = ["episode_id", "scenario_id", "seed", "metrics"]
        if not all(prop in self._schema_data["required"] for prop in required_props):
            raise ValueError(f"Schema must require properties: {required_props}")

    def _extract_version(self) -> None:
        """Extract version information from schema."""
        # Try to extract from const version field
        assert self._schema_data is not None, "Schema data must be loaded"
        properties = self._schema_data.get("properties", {})
        version_prop = properties.get("version", {})

        if "const" in version_prop:
            self._version = version_prop["const"]
        elif "enum" in version_prop and len(version_prop["enum"]) == 1:
            self._version = version_prop["enum"][0]
        else:
            # Fallback: try to extract from title or $id
            title = self._schema_data.get("title", "")
            id_field = self._schema_data.get("$id", "")

            version_match = re.search(r"v(\d+)", title) or re.search(r"v(\d+)", id_field)
            if version_match:
                self._version = f"v{version_match.group(1)}"
            else:
                self._version = "unknown"

    @property
    def schema_data(self) -> dict[str, Any]:
        """Get the raw schema data."""
        if self._schema_data is None:
            raise RuntimeError("Schema not loaded")
        return self._schema_data

    @property
    def version(self) -> str:
        """Get the schema version."""
        if self._version is None:
            raise RuntimeError("Version not extracted")
        return self._version

    @property
    def title(self) -> str:
        """Get the schema title."""
        return self._schema_data["title"]

    @property
    def schema_id(self) -> str:
        """Get the schema $id."""
        return self._schema_data["$id"]

    @property
    def required_properties(self) -> list[str]:
        """Get the list of required properties."""
        return self._schema_data["required"]

    def get_property_schema(self, property_name: str) -> dict[str, Any]:
        """
        Get the schema definition for a specific property.

        Args:
            property_name: Name of the property

        Returns:
            Schema definition for the property

        Raises:
            KeyError: If property doesn't exist in schema
        """
        properties = self._schema_data.get("properties", {})
        if property_name not in properties:
            raise KeyError(f"Property '{property_name}' not defined in schema")
        return properties[property_name]

    def validate_episode_data(self, episode_data: dict[str, Any]) -> None:
        """
        Validate episode data against this schema.

        Args:
            episode_data: Episode data to validate

        Raises:
            ValueError: If data doesn't conform to schema
        """

        try:
            jsonschema.validate(instance=episode_data, schema=self._schema_data)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Episode data validation failed: {e.message}") from e

    def is_backward_compatible_with(self, other_schema: "EpisodeSchema") -> bool:
        """
        Check if this schema is backward compatible with another schema.

        Backward compatibility means this schema can validate data that
        conforms to the other schema.

        Args:
            other_schema: Schema to check compatibility against

        Returns:
            True if this schema is backward compatible with other_schema
        """
        # For now, implement basic version-based compatibility
        # In a full implementation, this would do structural analysis

        try:
            this_major = int(self.version.lstrip("v"))
            other_major = int(other_schema.version.lstrip("v"))
            return this_major >= other_major  # Newer major versions may break compatibility
        except (ValueError, AttributeError):
            # If version parsing fails, assume incompatible
            return False

    def __str__(self) -> str:
        """String representation of the schema.

        Returns:
            Human-readable string showing version and title.
        """
        return f"EpisodeSchema(version={self.version}, title={self.title})"

    def __repr__(self) -> str:
        """Detailed string representation.

        Returns:
            Constructor-style representation showing schema path and version.
        """
        return f"EpisodeSchema(schema_path={self.schema_path}, version={self.version})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on schema content.

        Returns:
            True if both schemas have identical content.
        """
        if not isinstance(other, EpisodeSchema):
            return False
        return self._schema_data == other._schema_data

    def __hash__(self) -> int:
        """Hash based on schema content.

        Returns:
            Hash value computed from JSON-serialized schema data.
        """
        return hash(json.dumps(self._schema_data, sort_keys=True))
