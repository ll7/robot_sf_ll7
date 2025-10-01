"""
Unit tests for semantic versioning utilities.

Tests the version_utils module functions for breaking change detection,
version bump determination, version comparison, and validation.
"""

import pytest

from robot_sf.benchmark.version_utils import (
    compare_schema_versions,
    detect_breaking_changes,
    determine_version_bump,
    get_latest_version,
    validate_version_string,
)


class TestDetectBreakingChanges:
    """Test detect_breaking_changes function."""

    def test_no_breaking_changes_for_identical_schemas(self):
        """Test that identical schemas have no breaking changes."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

        changes = detect_breaking_changes(schema, schema)
        assert changes == []

    def test_detects_removed_required_property(self):
        """Test detection of removed required property."""
        old_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        new_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        changes = detect_breaking_changes(old_schema, new_schema)
        assert len(changes) == 1
        assert "Required property removed: age" in changes[0]

    def test_detects_property_becoming_required(self):
        """Test detection of property becoming required."""
        old_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

        new_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        changes = detect_breaking_changes(old_schema, new_schema)
        assert len(changes) == 1
        assert "Properties became required: age" in changes[0]

    def test_detects_property_type_change(self):
        """Test detection of property type changes."""
        old_schema = {"type": "object", "properties": {"age": {"type": "integer"}}}

        new_schema = {"type": "object", "properties": {"age": {"type": "string"}}}

        changes = detect_breaking_changes(old_schema, new_schema)
        assert len(changes) == 1
        assert "Property type changed: age" in changes[0]
        assert "type changed from integer to string" in changes[0]

    def test_detects_enum_value_removal(self):
        """Test detection of removed enum values."""
        old_schema = {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active", "inactive", "pending"]}},
        }

        new_schema = {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active", "inactive"]}},
        }

        changes = detect_breaking_changes(old_schema, new_schema)
        assert len(changes) == 1
        assert "Property enum changed: status" in changes[0]

    def test_allows_optional_property_removal(self):
        """Test that removing optional properties is not breaking."""
        old_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

        new_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        changes = detect_breaking_changes(old_schema, new_schema)
        assert changes == []

    def test_detects_removed_required_object_properties(self):
        """Test detection of removed required properties in object types."""
        old_schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"host": {"type": "string"}, "port": {"type": "integer"}},
                    "required": ["host", "port"],
                },
            },
        }

        new_schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"host": {"type": "string"}},
                    "required": ["host"],
                },
            },
        }

        changes = detect_breaking_changes(old_schema, new_schema)
        assert len(changes) == 1
        assert "required object properties removed" in changes[0]


class TestDetermineVersionBump:
    """Test determine_version_bump function."""

    def test_major_bump_for_breaking_changes(self):
        """Test that breaking changes result in major version bump."""
        old_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        new_schema = {
            "type": "object",
            "properties": {"name": {"type": "integer"}},  # Type change = breaking
            "required": ["name"],
        }

        bump = determine_version_bump(old_schema, new_schema)
        assert bump == "major"

    def test_minor_bump_for_additions(self):
        """Test that additions result in minor version bump."""
        old_schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        new_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},  # New optional property
            },
        }

        bump = determine_version_bump(old_schema, new_schema)
        assert bump == "minor"

    def test_minor_bump_for_enum_additions(self):
        """Test that adding enum values results in minor version bump."""
        old_schema = {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active", "inactive"]}},
        }

        new_schema = {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active", "inactive", "pending"]}},
        }

        bump = determine_version_bump(old_schema, new_schema)
        assert bump == "minor"

    def test_patch_bump_for_other_changes(self):
        """Test that other changes result in patch version bump."""
        old_schema = {
            "type": "object",
            "title": "Old Title",
            "properties": {"name": {"type": "string"}},
        }

        new_schema = {
            "type": "object",
            "title": "New Title",  # Documentation change
            "properties": {"name": {"type": "string"}},
        }

        bump = determine_version_bump(old_schema, new_schema)
        assert bump == "patch"


class TestCompareSchemaVersions:
    """Test compare_schema_versions function."""

    def test_equal_versions(self):
        """Test comparison of equal versions."""
        result = compare_schema_versions("1.0.0", "1.0.0")
        assert result == 0

    def test_greater_version(self):
        """Test comparison where first version is greater."""
        result = compare_schema_versions("2.0.0", "1.5.0")
        assert result == 1

    def test_lesser_version(self):
        """Test comparison where first version is lesser."""
        result = compare_schema_versions("1.0.0", "1.1.0")
        assert result == -1

    def test_patch_version_comparison(self):
        """Test comparison of patch versions."""
        result = compare_schema_versions("1.0.5", "1.0.3")
        assert result == 1

    def test_pre_release_versions(self):
        """Test comparison with pre-release versions."""
        result = compare_schema_versions("1.0.0-alpha", "1.0.0-beta")
        assert result == -1  # alpha < beta

    @pytest.mark.parametrize(
        "version1,version2,expected",
        [
            ("1.0.0", "1.0.0", 0),
            ("1.1.0", "1.0.0", 1),
            ("1.0.0", "1.1.0", -1),
            ("1.0.1", "1.0.0", 1),
            ("2.0.0", "1.9.9", 1),
        ],
    )
    def test_version_comparison_parametrized(self, version1, version2, expected):
        """Test version comparison with multiple test cases."""
        result = compare_schema_versions(version1, version2)
        assert result == expected


class TestGetLatestVersion:
    """Test get_latest_version function."""

    def test_single_version(self):
        """Test getting latest from single version."""
        result = get_latest_version(["1.0.0"])
        assert result == "1.0.0"

    def test_multiple_versions(self):
        """Test getting latest from multiple versions."""
        versions = ["1.0.0", "1.1.0", "0.9.0", "1.0.5"]
        result = get_latest_version(versions)
        assert result == "1.1.0"

    def test_pre_release_versions(self):
        """Test getting latest with pre-release versions."""
        versions = ["1.0.0-alpha", "1.0.0-beta", "1.0.0"]
        result = get_latest_version(versions)
        assert result == "1.0.0"

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Version list cannot be empty"):
            get_latest_version([])

    def test_unsorted_versions(self):
        """Test getting latest from unsorted version list."""
        versions = ["0.5.0", "2.0.0", "1.0.0", "0.1.0"]
        result = get_latest_version(versions)
        assert result == "2.0.0"


class TestValidateVersionString:
    """Test validate_version_string function."""

    @pytest.mark.parametrize(
        "version,expected",
        [
            ("1.0.0", True),
            ("2.1.3", True),
            ("0.0.1", True),
            ("1.0.0-alpha", True),
            ("1.0.0-beta.1", True),
            ("1.0.0-rc.1+build.1", True),
            ("1.0.0-invalid", True),  # Valid prerelease identifier
            ("1", False),
            ("1.0", False),
            ("1.0.0.0", False),
            ("v1.0.0", False),
            ("", False),
            ("not-a-version", False),
        ],
    )
    def test_version_string_validation(self, version, expected):
        """Test validation of version strings."""
        result = validate_version_string(version)
        assert result == expected

    def test_valid_semantic_versions(self):
        """Test that valid semantic versions pass validation."""
        valid_versions = [
            "0.0.1",
            "1.0.0",
            "2.1.3",
            "1.0.0-alpha",
            "1.0.0-beta.2",
            "1.0.0-rc.1+build.123",
        ]

        for version in valid_versions:
            assert validate_version_string(version), f"Version {version} should be valid"

    def test_invalid_version_strings(self):
        """Test that invalid version strings fail validation."""
        invalid_versions = [
            "",
            "1",
            "1.0",
            "1.0.0.0",
            "v1.0.0",
            "not-a-version",
            "1..0.0",
            "1.0.0-",
            "1.0.0+",
        ]

        for version in invalid_versions:
            assert not validate_version_string(version), f"Version {version} should be invalid"
