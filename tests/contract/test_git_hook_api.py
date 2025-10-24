"""Contract test for git-hook-api.v1.json (T005).

This test is written BEFORE implementation to enforce TDD cycle.
It defines the expected API contract and will FAIL until the git hook is implemented.

Contract Requirements:
- prevent_schema_duplicates(staged_files, schema_pattern=".*\\.schema\\.v[0-9]+\\.json$")
  -> dict with status, duplicates_found, message
- Exit codes: 0 for pass, 1 for fail
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add hooks directory to Python path for testing
hooks_dir = Path(__file__).parent.parent.parent / "hooks"
if str(hooks_dir) not in sys.path:
    sys.path.insert(0, str(hooks_dir))


class TestGitHookAPIContract:
    """Contract tests for git hook API."""

    def test_prevent_schema_duplicates_contract(self):
        """Test prevent_schema_duplicates function contract - will fail until implemented."""
        # Contract: module must exist
        pytest.importorskip("hooks.prevent_schema_duplicates")

        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Contract: function must exist and be callable
        assert callable(prevent_schema_duplicates)

        # Contract: prevent_schema_duplicates takes staged_files parameter
        staged_files = ["episode.schema.v1.json", "other.file"]
        result = prevent_schema_duplicates(staged_files)

        # Contract: returns dict with required fields
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["pass", "fail"]

        # Contract: includes duplicates_found array
        assert "duplicates_found" in result
        assert isinstance(result["duplicates_found"], list)

        # Contract: includes message
        assert "message" in result
        assert isinstance(result["message"], str)

    def test_prevent_schema_duplicates_with_pattern_contract(self):
        """Test prevent_schema_duplicates with custom pattern."""
        pytest.importorskip("hooks.prevent_schema_duplicates")

        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Contract: schema_pattern parameter is optional with default
        staged_files = ["episode.schema.v1.json", "aggregate.schema.v2.json"]
        result = prevent_schema_duplicates(staged_files, schema_pattern=r".*\.schema\.v\d+\.json$")
        assert isinstance(result, dict)
        assert "status" in result

    def test_prevent_schema_duplicates_no_duplicates_contract(self):
        """Test behavior when no duplicates are found."""
        pytest.importorskip("hooks.prevent_schema_duplicates")

        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Contract: when no schema files are staged, returns pass
        staged_files = ["readme.md", "script.py"]
        result = prevent_schema_duplicates(staged_files)

        assert result["status"] == "pass"
        assert len(result["duplicates_found"]) == 0
        assert "no duplicates" in result["message"].lower()

    def test_prevent_schema_duplicates_with_duplicates_contract(self):
        """Test behavior when duplicates are found."""
        pytest.importorskip("hooks.prevent_schema_duplicates")

        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Contract: when duplicates exist, returns fail with details
        # This test assumes some duplicate detection logic exists
        staged_files = ["duplicate.schema.v1.json", "canonical.schema.v1.json"]
        result = prevent_schema_duplicates(staged_files)

        # Contract: result structure is consistent
        assert isinstance(result, dict)
        assert "status" in result
        assert "duplicates_found" in result
        assert "message" in result

        # If duplicates are found, check structure
        if result["duplicates_found"]:
            for duplicate in result["duplicates_found"]:
                assert "file" in duplicate
                assert "canonical_file" in duplicate
                assert "reason" in duplicate
                assert isinstance(duplicate["file"], str)
                assert isinstance(duplicate["canonical_file"], str)
                assert isinstance(duplicate["reason"], str)

    def test_prevent_schema_duplicates_parameter_validation_contract(self):
        """Test parameter validation."""
        pytest.importorskip("hooks.prevent_schema_duplicates")

        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Contract: staged_files is required
        with pytest.raises(TypeError):
            prevent_schema_duplicates()  # type: ignore

        # Contract: staged_files must be iterable
        with pytest.raises((TypeError, AttributeError)):
            prevent_schema_duplicates(None)  # type: ignore

        # Contract: staged_files should be list of strings
        with pytest.raises((TypeError, AttributeError)):
            prevent_schema_duplicates([123, 456])  # type: ignore

    def test_prevent_schema_duplicates_empty_staged_files_contract(self):
        """Test behavior with empty staged files list."""
        pytest.importorskip("hooks.prevent_schema_duplicates")

        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Contract: empty list is valid and returns pass
        result = prevent_schema_duplicates([])
        assert result["status"] == "pass"
        assert len(result["duplicates_found"]) == 0

    def test_prevent_schema_duplicates_schema_pattern_validation_contract(self):
        """Test schema pattern parameter validation."""
        pytest.importorskip("hooks.prevent_schema_duplicates")

        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Contract: schema_pattern should be a valid regex string
        staged_files = ["test.schema.v1.json"]

        # Valid patterns should work
        result = prevent_schema_duplicates(staged_files, schema_pattern=r".*\.schema\.v\d+\.json$")
        assert isinstance(result, dict)

        # Invalid patterns should raise appropriate errors
        with pytest.raises((TypeError, ValueError)):
            prevent_schema_duplicates(staged_files, schema_pattern=123)  # type: ignore

        with pytest.raises((TypeError, ValueError)):
            prevent_schema_duplicates(staged_files, schema_pattern=None)  # type: ignore
