"""
Semantic versioning utilities for schema evolution.

This module provides utilities for detecting breaking changes between schema versions,
determining appropriate version bumps, and comparing schema structures.
"""

from typing import Any

from robot_sf.benchmark.schema_version import SchemaVersion


def detect_breaking_changes(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> list[str]:
    """
    Detect breaking changes between two schema versions.

    Analyzes schema structure to identify changes that would break backward compatibility:
    - Removed required properties
    - Changed property types to incompatible types
    - Removed enum values
    - Changed from optional to required properties

    Args:
        old_schema: The original schema
        new_schema: The new schema to compare against

    Returns:
        List of breaking change descriptions
    """
    breaking_changes = []

    # Check required properties
    old_required = set(old_schema.get("required", []))
    new_required = set(new_schema.get("required", []))

    # Properties that became required
    newly_required = new_required - old_required
    if newly_required:
        breaking_changes.append(f"Properties became required: {', '.join(newly_required)}")

    # Check properties structure
    old_properties = old_schema.get("properties", {})
    new_properties = new_schema.get("properties", {})

    for prop_name, old_prop_def in old_properties.items():
        if prop_name not in new_properties:
            # Property was removed
            if prop_name in old_required:
                breaking_changes.append(f"Required property removed: {prop_name}")
            else:
                # Optional property removed - not breaking for consumers
                pass
        else:
            # Property exists in both - check for type changes
            new_prop_def = new_properties[prop_name]
            type_change = _detect_property_type_change(old_prop_def, new_prop_def)
            if type_change:
                breaking_changes.append(f"Property type changed: {prop_name} - {type_change}")

            # Check enum constraints
            enum_change = _detect_enum_change(old_prop_def, new_prop_def)
            if enum_change:
                breaking_changes.append(f"Property enum changed: {prop_name} - {enum_change}")

    return breaking_changes


def _detect_property_type_change(
    old_prop: dict[str, Any],
    new_prop: dict[str, Any],
) -> str | None:
    """
    Detect if a property type change is breaking.

    Args:
        old_prop: Old property definition
        new_prop: New property definition

    Returns:
        Description of breaking change if any, None otherwise
    """
    old_type = old_prop.get("type")
    new_type = new_prop.get("type")

    if old_type != new_type:
        # Type changes are generally breaking
        return f"type changed from {old_type} to {new_type}"

    # For object types, check if required properties were removed
    if old_type == "object":
        old_required = set(old_prop.get("required", []))
        new_required = set(new_prop.get("required", []))
        removed_required = old_required - new_required
        if removed_required:
            return f"required object properties removed: {', '.join(removed_required)}"

    # For array types, check item type changes
    if old_type == "array":
        old_items = old_prop.get("items", {})
        new_items = new_prop.get("items", {})
        if old_items.get("type") != new_items.get("type"):
            return (
                f"array item type changed from {old_items.get('type')} to {new_items.get('type')}"
            )

    return None


def _detect_enum_change(old_prop: dict[str, Any], new_prop: dict[str, Any]) -> str | None:
    """
    Detect if enum constraints became more restrictive.

    Args:
        old_prop: Old property definition
        new_prop: New property definition

    Returns:
        Description of breaking change if any, None otherwise
    """
    old_enum = set(old_prop.get("enum", []))
    new_enum = set(new_prop.get("enum", []))

    if old_enum and new_enum:
        removed_values = old_enum - new_enum
        if removed_values:
            return f"enum values removed: {', '.join(str(v) for v in removed_values)}"

    return None


def determine_version_bump(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> str:
    """
    Determine what type of version bump is needed based on schema changes.

    Args:
        old_schema: The original schema
        new_schema: The new schema

    Returns:
        'major' for breaking changes, 'minor' for additions, 'patch' for fixes
    """
    breaking_changes = detect_breaking_changes(old_schema, new_schema)

    if breaking_changes:
        return "major"

    # Check for additions (minor version bump)
    if _has_additions(old_schema, new_schema):
        return "minor"

    # Default to patch for other changes (documentation, formatting, etc.)
    return "patch"


def _has_additions(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> bool:
    """
    Check if the new schema has additions compared to the old schema.

    Args:
        old_schema: The original schema
        new_schema: The new schema

    Returns:
        True if additions were detected
    """
    # Check for new optional properties
    old_property_names = set(old_schema.get("properties", {}))
    new_property_names = set(new_schema.get("properties", {}))
    added_property_names = new_property_names - old_property_names

    if added_property_names:
        return True

    # Check for new enum values in existing properties
    old_props_dict = old_schema.get("properties", {})
    new_props_dict = new_schema.get("properties", {})

    for prop_name in old_property_names:
        if prop_name in new_props_dict:
            old_enum = set(old_props_dict[prop_name].get("enum", []))
            new_enum = set(new_props_dict[prop_name].get("enum", []))
            if new_enum - old_enum:  # New enum values added
                return True

    return False


def compare_schema_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.

    Args:
        version1: First version string
        version2: Second version string

    Returns:
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    v1 = SchemaVersion.parse(version1)
    v2 = SchemaVersion.parse(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def get_latest_version(versions: list[str]) -> str:
    """
    Get the latest version from a list of version strings.

    Args:
        versions: List of version strings

    Returns:
        The latest version string
    """
    if not versions:
        raise ValueError("Version list cannot be empty")

    parsed_versions = [SchemaVersion.parse(v) for v in versions]
    latest = max(parsed_versions)
    return str(latest)


def validate_version_string(version: str) -> bool:
    """
    Validate that a string is a valid semantic version.

    Args:
        version: Version string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        SchemaVersion.parse(version)
        return True
    except ValueError:
        return False
