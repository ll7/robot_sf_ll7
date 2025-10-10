"""
Integration tests for schema version detection feature.

Tests the end-to-end functionality of detecting and extracting schema versions
from schema files and ensuring proper semantic versioning.
"""


class TestVersionDetectionIntegration:
    """Integration tests for schema version detection functionality."""

    def test_version_detection_extracts_from_schema_file(self):
        """Test that version detection extracts version from schema file."""
        from robot_sf.benchmark.schema_loader import get_schema_version

        # Get the version from the canonical schema
        version_dict = get_schema_version()

        # Should be a dict with version components
        assert isinstance(version_dict, dict)
        assert "major" in version_dict
        assert "minor" in version_dict
        assert "patch" in version_dict

        # Format as string for validation
        version = f"{version_dict['major']}.{version_dict['minor']}.{version_dict['patch']}"

        # Should match X.Y.Z format
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", version)

    def test_version_detection_matches_schema_content(self):
        """Test that detected version matches what's embedded in schema content."""
        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        # The version should be extractable from the schema content
        # Check if version appears in title or other metadata
        assert "v1" in schema.get("title", "")

    def test_version_detection_consistent_across_calls(self):
        """Test that version detection is consistent across multiple calls."""
        from robot_sf.benchmark.schema_loader import get_schema_version

        # Get version multiple times
        version1 = get_schema_version()
        version2 = get_schema_version()
        version3 = get_schema_version()

        # All should be identical
        assert version1 == version2 == version3

    def test_version_detection_handles_schema_evolution(self):
        """Test that version detection supports schema evolution patterns."""
        from robot_sf.benchmark.schema_loader import get_schema_version

        version_dict = get_schema_version()

        # Parse version components
        major, minor, patch = version_dict["major"], version_dict["minor"], version_dict["patch"]

        # Version should follow semantic versioning rules
        assert major >= 0
        assert minor >= 0
        assert patch >= 0

        # For this feature, we expect version 1.x.x initially
        assert major >= 1

    def test_version_detection_provides_meaningful_versions(self):
        """Test that version detection provides meaningful, incrementing versions."""
        from robot_sf.benchmark.schema_loader import get_schema_version

        version_dict = get_schema_version()

        # Format as string for comparison
        version = f"{version_dict['major']}.{version_dict['minor']}.{version_dict['patch']}"

        # Version should not be 0.0.0 (which would indicate unversioned)
        assert version != "0.0.0"

        # Should be a reasonable version for a stable schema
        major, minor = version_dict["major"], version_dict["minor"]
        assert major >= 1 or (major == 0 and minor > 0)  # Either stable (1+) or development (0.x+)

    def test_version_detection_integrates_with_validation(self):
        """Test that version detection integrates properly with schema validation."""
        import jsonschema

        from robot_sf.benchmark.schema_loader import get_schema_version, load_schema

        schema = load_schema("episode.schema.v1.json")
        version_dict = get_schema_version("episode.schema.v1.json")

        # Create a valid episode according to the schema
        valid_episode = {
            "version": "v1",
            "episode_id": f"test-episode-{version_dict['major']}.{version_dict['minor']}.{version_dict['patch']}",
            "scenario_id": f"test-scenario-{version_dict['major']}.{version_dict['minor']}.{version_dict['patch']}",
            "seed": 42,
            "metrics": {"total_reward": 100.5, "steps": 50, "success": 1},
        }

        # Should validate successfully
        jsonschema.validate(instance=valid_episode, schema=schema)

    def test_version_detection_supports_backward_compatibility(self):
        """Test that version detection supports backward compatibility checks."""
        from robot_sf.benchmark.schema_loader import get_schema_version

        version_dict = get_schema_version()

        # Parse version to check compatibility
        major, minor, patch = version_dict["major"], version_dict["minor"], version_dict["patch"]

        # For backward compatibility, we should be able to determine
        # if a given version is compatible with the current schema
        current_version_tuple = (major, minor, patch)

        # Test compatibility with itself (should always be compatible)
        assert self._is_version_compatible(current_version_tuple, current_version_tuple)

        # Test compatibility with older versions (within same major version)
        if minor > 0:
            older_minor = (major, minor - 1, patch)
            assert self._is_version_compatible(current_version_tuple, older_minor)

        # Test incompatibility with newer major versions
        newer_major = (major + 1, 0, 0)
        assert not self._is_version_compatible(current_version_tuple, newer_major)

    def _is_version_compatible(self, current_version, check_version):
        """Helper method to check version compatibility."""
        current_major, current_minor, current_patch = current_version
        check_major, check_minor, check_patch = check_version

        # Major version changes break compatibility
        if check_major != current_major:
            return False

        # Within same major version, changes are backward compatible
        # (assuming proper semantic versioning)
        return True
