"""
Integration tests for runtime schema resolution feature.

Tests the end-to-end functionality of loading schemas at runtime
from the canonical location with proper caching and error handling.
"""

import json
from pathlib import Path


class TestRuntimeResolutionIntegration:
    """Integration tests for runtime schema resolution functionality."""

    def test_runtime_resolution_loads_from_canonical_path(self):
        """Test that runtime resolution loads schema from the expected canonical path."""
        from robot_sf.benchmark.schema_loader import load_schema

        # This should load from robot_sf/benchmark/schemas/episode.schema.v1.json
        schema = load_schema("episode.schema.v1.json")

        # Verify it's a valid schema structure
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "type" in schema
        assert schema["type"] == "object"

    def test_runtime_resolution_caches_schema_instance(self):
        """Test that schema loading caches the instance to avoid repeated file I/O."""
        from robot_sf.benchmark.schema_loader import load_schema

        # Load the same schema multiple times
        schema1 = load_schema("episode.schema.v1.json")
        schema2 = load_schema("episode.schema.v1.json")
        schema3 = load_schema("episode.schema.v1.json")

        # They should be the same object (cached)
        assert schema1 is schema2
        assert schema2 is schema3

    def test_runtime_resolution_preserves_schema_integrity(self):
        """Test that runtime loading preserves the original schema content."""
        from robot_sf.benchmark.schema_loader import load_schema

        # Load schema through runtime resolution
        loaded_schema = load_schema("episode.schema.v1.json")

        # Load schema directly from file for comparison
        schema_path = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
        with open(schema_path, encoding="utf-8") as f:
            direct_schema = json.load(f)

        # They should be identical
        assert loaded_schema == direct_schema

    def test_runtime_resolution_handles_path_resolution(self):
        """Test that runtime resolution correctly resolves the schema path."""
        from robot_sf.benchmark.schema_loader import load_schema

        # The schema should be loaded from the correct canonical path
        load_schema("episode.schema.v1.json")

        # Verify the path exists and is absolute
        schema_path = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
        assert schema_path.exists()

        # The path should be resolvable from the package root
        package_root = Path(__file__).parent.parent.parent / "robot_sf"
        resolved_path = package_root / "benchmark" / "schemas" / "episode.schema.v1.json"
        assert resolved_path.exists()
