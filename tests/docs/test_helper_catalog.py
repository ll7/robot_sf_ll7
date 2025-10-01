"""Tests for the docs helper catalog (T005)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf.docs.helper_catalog import register_helper


def test_register_helper_updates_index():
    """Test that register_helper updates the docs index properly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock docs/README.md
        mock_readme_content = """# Documentation

This is the main documentation index.

## Other Section

Some other content.
"""
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        readme_path = docs_dir / "README.md"
        readme_path.write_text(mock_readme_content)

        # Patch the docs_readme_path to use our temp file
        with patch("robot_sf.docs.helper_catalog.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.open.side_effect = readme_path.open
            mock_path.return_value.resolve.return_value = readme_path

            register_helper("test_helper", "A test helper function", "path/to/docs")

            # Check that content was updated
            updated_content = readme_path.read_text()
            assert "## Helper Catalog" in updated_content
            assert "test_helper" in updated_content
            assert "A test helper function" in updated_content


def test_register_helper_missing_docs_index():
    """Test error handling when docs index doesn't exist."""
    with patch("robot_sf.docs.helper_catalog.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Documentation index not found"):
            register_helper("test_helper", "Test summary", "path/to/docs")


def test_register_helper_duplicate_entry():
    """Test that duplicate helper entries are not added."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_readme_content = """# documentation

## Helper Catalog

- **existing_helper**: Already documented helper
"""
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        readme_path = docs_dir / "README.md"
        readme_path.write_text(mock_readme_content)

        with patch("robot_sf.docs.helper_catalog.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.open.side_effect = readme_path.open
            mock_path.return_value.resolve.return_value = readme_path

            # Try to register the same helper again
            register_helper("existing_helper", "Updated summary", "new/path")

            # Content should remain unchanged
            updated_content = readme_path.read_text()
            assert updated_content.count("existing_helper") == 1
