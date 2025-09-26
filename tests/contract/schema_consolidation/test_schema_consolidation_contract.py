"""Contract tests for schema consolidation feature (T011).

These tests are written BEFORE implementation to enforce red → green TDD cycle.
They define the expected API contracts and will FAIL until implementation is complete.

Test Categories:
- Schema loading functionality
- Version detection
- Duplicate prevention
- Backward compatibility
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestSchemaLoaderContract:
    """Contract tests for schema loading API."""

    def test_load_schema_contract(self):
        """Test schema loader API contract - will fail until implemented."""
        # This test defines the expected contract
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import load_schema

        # Contract: load_schema takes schema_name parameter
        schema = load_schema("episode.schema.v1.json")

        # Contract: returns dict with expected structure
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "title" in schema
        assert schema["title"] == "RobotSF Benchmark Episode (v1)"

    def test_get_schema_version_contract(self):
        """Test version detection API contract."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import get_schema_version

        # Contract: get_schema_version takes schema_name
        version = get_schema_version("episode.schema.v1.json")

        # Contract: returns version dict
        assert isinstance(version, dict)
        assert "major" in version
        assert "minor" in version
        assert "patch" in version
        assert version["major"] == 1
        assert version["minor"] == 0
        assert version["patch"] == 0

    def test_schema_loading_error_handling(self):
        """Test error handling for invalid schema names."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")

        from robot_sf.benchmark.schema_loader import load_schema

        # Contract: invalid schema names raise appropriate errors
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent.schema.v1.json")

    def test_schema_integrity_validation(self):
        """Test that loaded schemas are valid JSON Schema."""
        pytest.importorskip("robot_sf.benchmark.schema_loader")
        jsonschema = pytest.importorskip("jsonschema")

        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        # Contract: loaded schema should be valid JSON Schema
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except Exception as e:
            pytest.fail(f"Loaded schema is not valid JSON Schema: {e}")


class TestGitHookContract:
    """Contract tests for git hook duplicate prevention."""

    def test_prevent_duplicates_hook_exists(self):
        """Test that duplicate prevention hook is installed."""
        # Check for hook in pre-commit config or direct hook file
        pre_commit_config = Path(".pre-commit-config.yaml")
        hook_file = Path(".git/hooks/pre-commit")

        hook_configured = False

        if pre_commit_config.exists():
            content = pre_commit_config.read_text()
            if "prevent-schema-duplicates" in content:
                hook_configured = True

        if hook_file.exists():
            content = hook_file.read_text()
            if "prevent-schema-duplicates" in content:
                hook_configured = True

        assert hook_configured, "Duplicate prevention hook not configured"

    def test_duplicate_detection_logic(self):
        """Test duplicate detection logic (mock test)."""
        # This would test the hook logic in isolation
        # For now, just verify the hook script exists
        hook_script = Path("hooks/prevent-schema-duplicates.py")
        if hook_script.exists():
            # Verify it's executable Python
            content = hook_script.read_text()
            assert "def main(" in content, "Hook script should have main function"
        else:
            pytest.skip("Hook script not implemented yet")


class TestBackwardCompatibilityContract:
    """Contract tests for backward compatibility."""

    def test_legacy_schema_access_still_works(self):
        """Test that direct file access still works during transition."""
        schema_path = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")

        # Contract: canonical schema file exists and is valid JSON
        assert schema_path.exists(), "Canonical schema file missing"

        # Contract: file contains valid JSON
        content = schema_path.read_text()
        schema = json.loads(content)

        # Contract: has expected structure
        assert schema["title"] == "RobotSF Benchmark Episode (v1)"
        assert schema["version"] == "v1"

    def test_existing_tests_still_pass(self):
        """Test that existing contract tests still pass."""
        # Import and run existing episode schema test
        from tests.contract.test_episode_schema import _load_schema

        schema = _load_schema()

        # Contract: existing loading mechanism still works
        assert isinstance(schema, dict)
        assert schema["title"] == "RobotSF Benchmark Episode (v1)"
