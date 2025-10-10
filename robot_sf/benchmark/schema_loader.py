"""
Schema loader module for runtime schema resolution.

This module provides functions for loading schemas from canonical locations
with caching and version validation.
"""

import logging
import re
from typing import Any

from .schema_reference import SchemaReference

logger = logging.getLogger(__name__)

# Default schema reference for episode schemas
DEFAULT_EPISODE_SCHEMA_REF = SchemaReference(
    schema_path="benchmark/schemas/episode.schema.v1.json",
    version="v1",
)


def _parse_schema_name(schema_name: str) -> tuple[str, str]:
    """
    Parse schema name and extract version.

    Args:
        schema_name: Schema filename (e.g., 'episode.schema.v1.json')

    Returns:
        Tuple of (schema_path, version_string)

    Raises:
        ValueError: If schema name format is invalid
    """
    # Validate schema name format
    if not re.match(r"^[a-zA-Z0-9_.-]+\.schema\.v[0-9]+\.json$", schema_name):
        raise ValueError(f"Invalid schema name format: {schema_name}")

    # Extract version from filename (e.g., "episode.schema.v1.json" -> "v1")
    version_match = re.search(r"\.schema\.v(\d+)\.json$", schema_name)
    if not version_match:
        raise ValueError(f"Cannot extract version from schema name: {schema_name}")
    version = f"v{version_match.group(1)}"

    # Construct canonical path
    schema_path = f"benchmark/schemas/{schema_name}"

    return schema_path, version


def load_schema(schema_name: str, validate_integrity: bool = True) -> dict[str, Any]:
    """
    Load a schema from the canonical location.

    Args:
        schema_name: Schema filename (e.g., 'episode.schema.v1.json')
        validate_integrity: Whether to validate schema is well-formed JSON Schema

    Returns:
        Loaded schema data as a dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema cannot be loaded or validated
    """
    schema_path, version = _parse_schema_name(schema_name)

    # Create schema reference
    schema_ref = SchemaReference(schema_path=schema_path, version=version)

    schema = schema_ref.load_schema()

    if validate_integrity:
        # Basic validation that it's a dict with expected structure
        if not isinstance(schema.schema_data, dict):
            raise ValueError(f"Schema is not a valid JSON object: {schema_name}")
        if "$schema" not in schema.schema_data:
            raise ValueError(f"Schema missing $schema field: {schema_name}")

    return schema.schema_data


def get_schema_version(schema_name: str | None = None) -> dict[str, int]:
    """
    Get the version of a schema.

    Args:
        schema_name: Schema filename (e.g., 'episode.schema.v1.json').
                    If None, uses the default episode schema.

    Returns:
        Dict with 'major', 'minor', 'patch' version numbers

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema cannot be loaded or version cannot be parsed
    """
    if schema_name is None:
        schema_name = "episode.schema.v1.json"

    _, version = _parse_schema_name(schema_name)

    # Extract major version number from version string (e.g., "v1" -> 1)
    try:
        major = int(version[1:])  # Remove 'v' prefix
        return {"major": major, "minor": 0, "patch": 0}
    except ValueError:
        raise ValueError(f"Invalid version number in schema name: {schema_name}")


def get_schema_version_string(schema_name: str | None = None) -> str:
    """
    Get the version of a schema as a formatted string.

    Args:
        schema_name: Schema filename (e.g., 'episode.schema.v1.json').
                    If None, uses the default episode schema.

    Returns:
        Version string in X.Y.Z format

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema cannot be loaded or version cannot be parsed
    """
    version_dict = get_schema_version(schema_name)
    return f"{version_dict['major']}.{version_dict['minor']}.{version_dict['patch']}"


def validate_episode_data(
    episode_data: dict[str, Any],
    schema_ref: SchemaReference | None = None,
) -> None:
    """
    Validate episode data against a schema.

    Args:
        episode_data: Episode data to validate
        schema_ref: SchemaReference to use. If None, uses the default episode schema.

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If validation fails
    """
    if schema_ref is None:
        schema_ref = DEFAULT_EPISODE_SCHEMA_REF

    # Ensure schema is loaded before validation
    schema_ref.load_schema()
    schema_ref.validate_episode_data(episode_data)


def clear_schema_cache() -> None:
    """Clear the global schema cache."""
    SchemaReference.clear_cache()
