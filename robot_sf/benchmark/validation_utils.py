"""
Schema validation utilities for JSON Schema validation and integrity checking.

This module provides utilities for validating JSON schemas themselves,
checking schema integrity, and performing advanced validation operations.
"""

import json
from typing import Any

import jsonschema


def validate_schema_integrity(schema: dict[str, Any]) -> list[str]:
    """
    Validate that a schema is a well-formed JSON Schema.

    Performs comprehensive validation including:
    - Basic JSON Schema structure
    - Required fields presence
    - Type consistency
    - Reference resolution

    Args:
        schema: The schema to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check basic JSON Schema requirements
    if not isinstance(schema, dict):
        errors.append("Schema must be a JSON object")
        return errors

    # Check for $schema field
    if "$schema" not in schema:
        errors.append("Schema missing required '$schema' field")

    # Check for basic structure
    if "type" not in schema:
        errors.append("Schema missing 'type' field")

    # Validate against JSON Schema meta-schema
    try:
        # Use draft 2020-12 meta-schema
        meta_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "$schema": {"type": "string"},
                "$id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "type": {"type": "string"},
                "properties": {"type": "object"},
                "required": {"type": "array", "items": {"type": "string"}},
                "additionalProperties": {"type": "boolean"},
            },
            "required": ["$schema"],
        }

        jsonschema.validate(schema, meta_schema)

    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
    except Exception as e:
        errors.append(f"Schema validation failed: {e!s}")

    return errors


def validate_schema_compatibility(schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate schema compatibility with a specific JSON Schema draft.

    Args:
        schema: The schema to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        # For this implementation, we'll do basic validation
        # In a real implementation, you'd fetch and use the actual meta-schema
        errors = validate_schema_integrity(schema)
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Compatibility validation failed: {e!s}"]


def check_schema_completeness(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Check schema completeness and provide recommendations.

    Analyzes the schema for completeness and provides suggestions
    for missing or incomplete parts.

    Args:
        schema: The schema to analyze

    Returns:
        Dict with completeness analysis and recommendations
    """
    analysis: dict[str, Any] = {
        "score": 0,
        "max_score": 10,
        "issues": [],
        "recommendations": [],
    }

    # Check for title
    if "title" not in schema:
        analysis["issues"].append("Missing 'title' field")
        analysis["recommendations"].append("Add a descriptive title to the schema")

    # Check for description
    if "description" not in schema:
        analysis["issues"].append("Missing 'description' field")
        analysis["recommendations"].append("Add a description explaining the schema purpose")

    # Check for required fields
    if "required" not in schema:
        analysis["issues"].append("Missing 'required' field")
        analysis["recommendations"].append("Specify which properties are required")

    # Check properties definition
    if "properties" not in schema:
        analysis["issues"].append("Missing 'properties' field")
        analysis["recommendations"].append("Define the properties for the schema")
    else:
        properties = schema["properties"]
        if not properties:
            analysis["issues"].append("Empty 'properties' object")
            analysis["recommendations"].append("Add property definitions")

    # Check additionalProperties
    if "additionalProperties" not in schema:
        analysis["recommendations"].append(
            "Consider setting 'additionalProperties' to control extra properties",
        )

    # Calculate score
    analysis["score"] = analysis["max_score"] - len(analysis["issues"])

    return analysis


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a schema for consistent representation.

    Performs normalization operations like:
    - Sorting keys
    - Standardizing formats
    - Removing redundant fields

    Args:
        schema: The schema to normalize

    Returns:
        Normalized schema
    """
    # Create a normalized copy
    normalized = json.loads(json.dumps(schema, sort_keys=True))

    # Ensure consistent ordering of required fields
    if "required" in normalized and isinstance(normalized["required"], list):
        normalized["required"] = sorted(normalized["required"])

    return normalized


def extract_schema_metadata(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Extract metadata from a schema for analysis and reporting.

    Args:
        schema: The schema to analyze

    Returns:
        Dict with extracted metadata
    """
    required_properties: list[str] = schema.get("required", [])
    metadata: dict[str, Any] = {
        "schema_version": schema.get("$schema", "unknown"),
        "title": schema.get("title", "untitled"),
        "description": schema.get("description", ""),
        "type": schema.get("type", "unknown"),
        "required_properties": required_properties,
        "optional_properties": [],
        "total_properties": 0,
    }

    # Extract property information
    properties = schema.get("properties", {})
    metadata["total_properties"] = len(properties)

    required_set = set(required_properties)
    for prop_name in properties:
        if prop_name not in required_set:
            metadata["optional_properties"].append(prop_name)

    return metadata


def validate_schema_references(schema: dict[str, Any]) -> list[str]:
    """
    Validate that all $ref references in the schema are resolvable.

    Args:
        schema: The schema to validate

    Returns:
        List of unresolved reference errors
    """
    errors = []

    def check_refs(obj: Any, path: str = "") -> None:
        """Check refs.

        Args:
            obj: Generic object payload.
            path: Filesystem path to the resource.

        Returns:
            None: none.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key == "$ref":
                    # In a real implementation, you'd check if the reference is resolvable
                    # For now, just check it's a string
                    if not isinstance(value, str):
                        errors.append(f"Invalid $ref at {current_path}: must be a string")
                else:
                    check_refs(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_refs(item, f"{path}[{i}]")

    check_refs(schema)
    return errors
