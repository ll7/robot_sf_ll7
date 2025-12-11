"""JSON schema loading and validation utilities.

This module provides utilities for loading JSON schemas and validating data
against them, ensuring consistency with the contracts defined in the
specs/270-imitation-report/contracts/ directory.

Key Features:
    - Schema loading from contracts/ directory
    - Data validation against schemas
    - Clear error messages for validation failures

Usage:
    >>> from robot_sf.research.schema_loader import load_schema, validate_data
    >>> schema = load_schema("report_metadata.schema.v1.json")
    >>> validate_data(metadata_dict, schema)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from robot_sf.common.logging import get_logger
from robot_sf.research.exceptions import ValidationError

logger = get_logger(__name__)


def get_schema_path(schema_filename: str) -> Path:
    """Get the path to a schema file.

    Args:
        schema_filename: Schema filename (e.g., "report_metadata.schema.v1.json")

    Returns:
        Path to the schema file

    Raises:
        ValidationError: If schema file not found
    """
    # Check in robot_sf/benchmark/schemas/ (canonical location)
    schema_dir = Path(__file__).parent.parent / "benchmark" / "schemas"
    schema_path = schema_dir / schema_filename

    if not schema_path.exists():
        # Fallback to specs directory (development location)
        specs_dir = (
            Path(__file__).parent.parent.parent / "specs" / "270-imitation-report" / "contracts"
        )
        schema_path = specs_dir / schema_filename

    if not schema_path.exists():
        msg = f"Schema file not found: {schema_filename}"
        logger.error(msg, search_paths=[str(schema_dir), str(specs_dir)])
        raise ValidationError(msg)

    logger.debug("Located schema file", path=str(schema_path))
    return schema_path


def load_schema(schema_filename: str) -> dict[str, Any]:
    """Load a JSON schema from file.

    Args:
        schema_filename: Schema filename (e.g., "report_metadata.schema.v1.json")

    Returns:
        Parsed JSON schema as dictionary

    Raises:
        ValidationError: If schema file not found or invalid JSON
    """
    schema_path = get_schema_path(schema_filename)

    try:
        with schema_path.open() as f:
            schema = json.load(f)
        logger.debug("Loaded schema", filename=schema_filename)
        return schema
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in schema file: {schema_filename}"
        logger.exception(msg, error=str(e))
        raise ValidationError(msg) from e


def validate_data(data: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate data against a JSON schema.

    Args:
        data: Data dictionary to validate
        schema: JSON schema dictionary

    Raises:
        ValidationError: If validation fails with detailed error message
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
        logger.debug("Data validation passed")
    except jsonschema.ValidationError as e:
        msg = f"Schema validation failed: {e.message}"
        logger.exception(msg, path=list(e.path), schema_path=list(e.schema_path))
        raise ValidationError(msg) from e
    except jsonschema.SchemaError as e:
        msg = f"Invalid schema: {e.message}"
        logger.exception(msg)
        raise ValidationError(msg) from e


def validate_file(filepath: Path, schema_filename: str) -> None:
    """Validate a JSON file against a schema.

    Args:
        filepath: Path to JSON file to validate
        schema_filename: Schema filename to validate against

    Raises:
        ValidationError: If file not found, invalid JSON, or validation fails
    """
    if not filepath.exists():
        msg = f"File not found: {filepath}"
        logger.error(msg)
        raise ValidationError(msg)

    try:
        with filepath.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in file: {filepath}"
        logger.exception(msg, error=str(e))
        raise ValidationError(msg) from e

    schema = load_schema(schema_filename)
    validate_data(data, schema)

    logger.info("File validation passed", filepath=str(filepath), schema=schema_filename)
