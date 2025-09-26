"""
Integration tests for schema consolidation feature.

Tests the end-to-end functionality of consolidating episode schema definitions
into a single canonical source of truth with runtime resolution.
"""

import json
from pathlib import Path

import pytest

# Import the modules that will be implemented
# pytest.importorskip("robot_sf.benchmark.schema_loader")
# pytest.importorskip("robot_sf.benchmark.schemas.episode_schema")


class TestSchemaConsolidationIntegration:
    """Integration tests for schema consolidation functionality."""

    def test_schema_can_be_loaded_from_canonical_location(self):
        """Test that schema can be loaded from the canonical location (FR-002)."""
        from robot_sf.benchmark.schema_loader import load_schema

        # Load schema from canonical location
        schema = load_schema("episode.schema.v1.json")

        # Verify it's a valid JSON schema
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "type" in schema
        assert schema["type"] == "object"

    def test_schema_contains_required_episode_fields(self):
        """Test that schema defines required episode fields."""
        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        # Check required fields are defined
        assert "required" in schema
        required_fields = schema["required"]
        assert "episode_id" in required_fields
        assert "scenario_id" in required_fields
        assert "seed" in required_fields
        assert "metrics" in required_fields

    def test_schema_validation_works_for_valid_episode_data(self):
        """Test that schema validation works for valid episode data (FR-005)."""
        import jsonschema

        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        # Create valid episode data
        valid_episode = {
            "version": "v1",
            "episode_id": "test-episode-001",
            "scenario_id": "test-scenario-001",
            "seed": 42,
            "metrics": {"total_reward": 100.5, "steps": 50, "success_rate": 1.0},
        }

        # Should not raise an exception
        jsonschema.validate(instance=valid_episode, schema=schema)

    def test_schema_validation_fails_for_invalid_episode_data(self):
        """Test that schema validation fails for invalid episode data (FR-005)."""
        import jsonschema

        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        # Create invalid episode data (missing required field)
        invalid_episode = {
            "version": "v1",
            "episode_id": "test-episode-001",
            "scenario_id": "test-scenario-001",
            # Missing seed
            "metrics": {"total_reward": 100.5, "steps": 50, "success_rate": 1.0},
        }

        # Should raise ValidationError
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=invalid_episode, schema=schema)

    def test_schema_version_can_be_extracted(self):
        """Test that schema version can be extracted (FR-007)."""
        from robot_sf.benchmark.schema_loader import get_schema_version_string

        version = get_schema_version_string()

        # Version should be a string
        assert isinstance(version, str)
        # Should be "1.0.0" for this schema (current implementation)
        assert version == "1.0.0"

    def test_backward_compatibility_maintained(self):
        """Test that backward compatibility is maintained (FR-003)."""
        import jsonschema

        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        # Test with old episode format (minimal required fields)
        old_format_episode = {
            "version": "v1",
            "episode_id": "old-episode-001",
            "scenario_id": "old-scenario-001",
            "seed": 123,
            "metrics": {"total_reward": 75.0, "steps": 30},
        }

        # Should still validate (backward compatibility)
        jsonschema.validate(instance=old_format_episode, schema=schema)

    def test_schema_file_exists_in_canonical_location(self):
        """Test that canonical schema file exists (FR-001)."""
        # Check that the canonical schema file exists
        canonical_path = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")

        assert canonical_path.exists(), f"Canonical schema file not found at {canonical_path}"

        # Verify it's valid JSON
        with open(canonical_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)

        assert isinstance(schema_data, dict)
        assert "$schema" in schema_data

    def test_schema_consistency_across_loads(self):
        """Test that schema loading is consistent across multiple calls."""
        from robot_sf.benchmark.schema_loader import load_schema

        # Load schema multiple times
        schema1 = load_schema("episode.schema.v1.json")
        schema2 = load_schema("episode.schema.v1.json")
        schema3 = load_schema("episode.schema.v1.json")

        # All should be identical
        assert schema1 == schema2 == schema3

        # Should be the same object (caching)
        assert schema1 is schema2 is schema3

    def test_schema_properties_are_properly_defined(self):
        """Test that all schema properties are properly defined."""
        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        assert "properties" in schema
        properties = schema["properties"]

        # Check key properties exist and are properly typed
        assert "episode_id" in properties
        assert properties["episode_id"]["type"] == "string"

        assert "scenario_id" in properties
        assert properties["scenario_id"]["type"] == "string"

        assert "seed" in properties
        assert properties["seed"]["type"] == "integer"

        assert "metrics" in properties
        assert properties["metrics"]["type"] == "object"
