"""
Unit tests for schema validation utilities.

Tests the validation_utils module functions for schema integrity checking,
compatibility validation, completeness analysis, and metadata extraction.
"""

from robot_sf.benchmark.validation_utils import (
    check_schema_completeness,
    extract_schema_metadata,
    normalize_schema,
    validate_schema_compatibility,
    validate_schema_integrity,
    validate_schema_references,
)


class TestValidateSchemaIntegrity:
    """Test validate_schema_integrity function."""

    def test_valid_schema_passes_integrity_check(self):
        """Test that a valid schema passes integrity validation."""
        valid_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "title": "Test Schema",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        errors = validate_schema_integrity(valid_schema)
        assert errors == []

    def test_invalid_schema_fails_integrity_check(self):
        """Test that an invalid schema fails integrity validation."""
        invalid_schema = "not a dict"  # type: ignore

        errors = validate_schema_integrity(invalid_schema)
        assert len(errors) > 0
        assert "Schema must be a JSON object" in errors[0]

    def test_schema_missing_required_fields(self):
        """Test schema missing required fields."""
        incomplete_schema = {
            "type": "object"
            # Missing $schema
        }

        errors = validate_schema_integrity(incomplete_schema)
        assert len(errors) > 0
        assert any("$schema" in error for error in errors)


class TestValidateSchemaCompatibility:
    """Test validate_schema_compatibility function."""

    def test_compatible_schema_passes(self):
        """Test that a compatible schema passes validation."""
        compatible_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        is_valid, errors = validate_schema_compatibility(compatible_schema)
        assert is_valid
        assert errors == []

    def test_incompatible_schema_fails(self):
        """Test that an incompatible schema fails validation."""
        incompatible_schema = "not a schema"  # type: ignore

        is_valid, errors = validate_schema_compatibility(incompatible_schema)
        assert not is_valid
        assert len(errors) > 0


class TestCheckSchemaCompleteness:
    """Test check_schema_completeness function."""

    def test_complete_schema_gets_high_score(self):
        """Test that a complete schema gets a high completeness score."""
        complete_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Complete Schema",
            "description": "A complete test schema",
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
            "additionalProperties": False,
        }

        analysis = check_schema_completeness(complete_schema)
        assert analysis["score"] >= 8  # Should be mostly complete
        assert len(analysis["issues"]) == 0

    def test_incomplete_schema_gets_low_score(self):
        """Test that an incomplete schema gets a low completeness score."""
        incomplete_schema = {
            "type": "object"
            # Missing title, description, properties, required, etc.
        }

        analysis = check_schema_completeness(incomplete_schema)
        assert analysis["score"] < 7  # Should have multiple issues
        assert len(analysis["issues"]) > 0
        assert len(analysis["recommendations"]) > 0

    def test_completeness_analysis_includes_recommendations(self):
        """Test that completeness analysis provides helpful recommendations."""
        minimal_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        }

        analysis = check_schema_completeness(minimal_schema)
        assert "recommendations" in analysis
        assert len(analysis["recommendations"]) > 0


class TestNormalizeSchema:
    """Test normalize_schema function."""

    def test_normalize_sorts_keys(self):
        """Test that normalize_schema sorts dictionary keys."""
        schema = {"zebra": "last", "alpha": "first", "beta": "middle"}

        normalized = normalize_schema(schema)
        keys = list(normalized.keys())
        assert keys == ["alpha", "beta", "zebra"]

    def test_normalize_sorts_required_fields(self):
        """Test that normalize_schema sorts required fields array."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["zebra", "alpha", "beta"],
        }

        normalized = normalize_schema(schema)
        assert normalized["required"] == ["alpha", "beta", "zebra"]

    def test_normalize_preserves_structure(self):
        """Test that normalization preserves schema structure."""
        original_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Test Schema",
            "type": "object",
            "properties": {"b_prop": {"type": "string"}, "a_prop": {"type": "integer"}},
            "required": ["b_prop", "a_prop"],
        }

        normalized = normalize_schema(original_schema)

        # Structure should be preserved
        assert normalized["$schema"] == original_schema["$schema"]
        assert normalized["type"] == original_schema["type"]
        assert set(normalized["required"]) == set(original_schema["required"])


class TestExtractSchemaMetadata:
    """Test extract_schema_metadata function."""

    def test_extract_metadata_from_complete_schema(self):
        """Test metadata extraction from a complete schema."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Test Schema",
            "description": "A test schema",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }

        metadata = extract_schema_metadata(schema)

        assert metadata["schema_version"] == "https://json-schema.org/draft/2020-12/schema"
        assert metadata["title"] == "Test Schema"
        assert metadata["description"] == "A test schema"
        assert metadata["type"] == "object"
        assert metadata["required_properties"] == ["name", "email"]
        assert metadata["optional_properties"] == ["age"]
        assert metadata["total_properties"] == 3

    def test_extract_metadata_from_minimal_schema(self):
        """Test metadata extraction from a minimal schema."""
        schema = {"type": "string"}

        metadata = extract_schema_metadata(schema)

        assert metadata["schema_version"] == "unknown"
        assert metadata["title"] == "untitled"
        assert metadata["type"] == "string"
        assert metadata["required_properties"] == []
        assert metadata["total_properties"] == 0


class TestValidateSchemaReferences:
    """Test validate_schema_references function."""

    def test_valid_references_pass(self):
        """Test that valid $ref references pass validation."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"name": {"$ref": "#/definitions/name"}},
            "definitions": {"name": {"type": "string"}},
        }

        errors = validate_schema_references(schema)
        # In this basic implementation, we only check $ref is a string
        assert len(errors) == 0

    def test_invalid_ref_type_fails(self):
        """Test that invalid $ref types fail validation."""
        schema = {
            "properties": {
                "name": {"$ref": 123}  # Should be string
            }
        }

        errors = validate_schema_references(schema)
        assert len(errors) > 0
        assert any("must be a string" in error for error in errors)

    def test_nested_references_checked(self):
        """Test that references in nested structures are checked."""
        schema = {
            "properties": {"items": {"type": "array", "items": {"$ref": "#/definitions/item"}}},
            "definitions": {"item": {"type": "object"}},
        }

        errors = validate_schema_references(schema)
        assert len(errors) == 0
