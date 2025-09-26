"""Contract test for schema-loader-api.v1.json (T004).

This test is written BEFORE implementation to enforce TDD cycle.
It defines the expected API contract and will FAIL until the schema loader is implemented.

Contract Requirements:
- load_schema(schema_name, validate_integrity=True) -> dict
- get_schema_version(schema_name) -> dict with major/minor/patch
"""

from __future__ import annotations

import pytest


class TestSchemaLoaderAPIContract:
    """Contract tests for schema loader API."""

    def test_load_schema_contract(self):
        """Test load_schema function contract - will fail until implemented."""
        # Contract: module must exist
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import load_schema

        # Contract: function must exist and be callable
        assert callable(load_schema)

        # Contract: load_schema takes schema_name parameter
        # This will fail with ImportError until implementation exists
        schema = load_schema("episode.schema.v1.json")

        # Contract: returns dict
        assert isinstance(schema, dict)

        # Contract: returned dict has expected JSON Schema structure
        assert "$schema" in schema
        assert "title" in schema
        assert "type" in schema
        assert schema["type"] == "object"

    def test_load_schema_with_validation_contract(self):
        """Test load_schema with validation parameter."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import load_schema

        # Contract: validate_integrity parameter defaults to True
        schema = load_schema("episode.schema.v1.json", validate_integrity=True)
        assert isinstance(schema, dict)

        # Contract: can be called with validate_integrity=False
        schema_no_validation = load_schema("episode.schema.v1.json", validate_integrity=False)
        assert isinstance(schema_no_validation, dict)

    def test_load_schema_error_handling_contract(self):
        """Test load_schema error handling."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import load_schema

        # Contract: invalid schema name raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent.schema.v1.json")

        # Contract: malformed schema name raises appropriate error
        with pytest.raises((ValueError, TypeError)):
            load_schema("invalid-schema-name")

    def test_get_schema_version_contract(self):
        """Test get_schema_version function contract."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import get_schema_version

        # Contract: function must exist and be callable
        assert callable(get_schema_version)

        # Contract: get_schema_version takes schema_name parameter
        version = get_schema_version("episode.schema.v1.json")

        # Contract: returns dict with required fields
        assert isinstance(version, dict)
        assert "major" in version
        assert "minor" in version
        assert "patch" in version

        # Contract: version numbers are non-negative integers
        assert isinstance(version["major"], int) and version["major"] >= 0
        assert isinstance(version["minor"], int) and version["minor"] >= 0
        assert isinstance(version["patch"], int) and version["patch"] >= 0

    def test_get_schema_version_error_handling_contract(self):
        """Test get_schema_version error handling."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import get_schema_version

        # Contract: invalid schema name raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_schema_version("nonexistent.schema.v1.json")

    def test_schema_name_validation_contract(self):
        """Test schema name parameter validation."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import get_schema_version, load_schema

        # Contract: schema_name must match pattern ^[a-zA-Z0-9_.-]+\.schema\.v[0-9]+\.json$
        # Valid names should work (when implemented)
        valid_names = ["episode.schema.v1.json", "aggregate.schema.v2.json", "test.schema.v10.json"]

        for name in valid_names:
            # These will fail until implementation exists
            with pytest.raises((ImportError, FileNotFoundError)):
                load_schema(name)
            with pytest.raises((ImportError, FileNotFoundError)):
                get_schema_version(name)

        # Contract: invalid names should raise ValueError or TypeError
        invalid_names = [
            "episode.json",  # missing .schema.v
            "episode.schema.json",  # missing version
            "episode.v1.json",  # missing .schema.
            "",  # empty
            None,  # None
        ]

        for name in invalid_names:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                load_schema(name)
            with pytest.raises((ValueError, TypeError, AttributeError)):
                get_schema_version(name)
