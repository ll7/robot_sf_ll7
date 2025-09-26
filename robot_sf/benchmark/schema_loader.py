"""
Schema loader module for runtime schema resolution.

This module provides functions for loading schemas from canonical locations
with caching and version validation.
"""

import logging
from typing import Any, Dict, Optional

from .schema_reference import SchemaReference

logger = logging.getLogger(__name__)

# Default schema reference for episode schemas
DEFAULT_EPISODE_SCHEMA_REF = SchemaReference(
    schema_path="benchmark/schemas/episode.schema.v1.json", version="v1"
)


def load_schema(schema_ref: Optional[SchemaReference] = None) -> Dict[str, Any]:
    """
    Load a schema from the canonical location.

    Args:
        schema_ref: SchemaReference to load. If None, loads the default episode schema.

    Returns:
        Loaded schema data as a dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema cannot be loaded or validated
    """
    if schema_ref is None:
        schema_ref = DEFAULT_EPISODE_SCHEMA_REF

    schema = schema_ref.load_schema()
    return schema.schema_data


def get_schema_version(schema_ref: Optional[SchemaReference] = None) -> str:
    """
    Get the version of a schema.

    Args:
        schema_ref: SchemaReference to check. If None, checks the default episode schema.

    Returns:
        Schema version string

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema cannot be loaded or validated
    """
    if schema_ref is None:
        schema_ref = DEFAULT_EPISODE_SCHEMA_REF

    schema = schema_ref.load_schema()
    return schema.version


def validate_episode_data(
    episode_data: Dict[str, Any], schema_ref: Optional[SchemaReference] = None
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

    schema_ref.validate_episode_data(episode_data)


def clear_schema_cache() -> None:
    """Clear the global schema cache."""
    SchemaReference.clear_cache()
