"""
Integration tests for git hook duplicate prevention feature.

Tests the end-to-end functionality of preventing duplicate schema files
from being committed using git hooks.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest


class TestGitHookPreventionIntegration:
    """Integration tests for git hook duplicate prevention functionality."""

    def test_git_hook_prevents_duplicate_schema_commits(self):
        """Test that git hooks prevent committing duplicate schema files."""
        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Create a temporary git repository for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "test_repo"
            repo_path.mkdir()

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
            )

            # Create initial commit
            (repo_path / "README.md").write_text("# Test Repo")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)

            # Create a schema file
            schema_dir = repo_path / "schemas"
            schema_dir.mkdir()
            schema_file = schema_dir / "episode.schema.v1.json"
            schema_file.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            # Stage the schema file
            subprocess.run(
                ["git", "add", "schemas/episode.schema.v1.json"], cwd=repo_path, check=True
            )

            # Test that prevent_schema_duplicates works with staged files
            staged_files = ["schemas/episode.schema.v1.json"]
            prevent_schema_duplicates(staged_files)  # Should not raise

            # Now create a duplicate schema file
            duplicate_schema = repo_path / "duplicate.schema.v1.json"
            duplicate_schema.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            # Stage the duplicate
            subprocess.run(["git", "add", "duplicate.schema.v1.json"], cwd=repo_path, check=True)

            # Test that prevent_schema_duplicates detects the duplicate
            staged_files_with_duplicate = [
                "schemas/episode.schema.v1.json",
                "duplicate.schema.v1.json",
            ]

            with pytest.raises(ValueError, match="Duplicate schema content detected"):
                prevent_schema_duplicates(staged_files_with_duplicate)

    def test_git_hook_allows_different_schemas(self):
        """Test that git hooks allow committing different schema files."""
        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Create a temporary git repository for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "test_repo"
            repo_path.mkdir()

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
            )

            # Create initial commit
            (repo_path / "README.md").write_text("# Test Repo")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)

            # Create different schema files
            schema_dir = repo_path / "schemas"
            schema_dir.mkdir()

            schema1 = schema_dir / "episode.schema.v1.json"
            schema1.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "title": "Episode"}'
            )

            schema2 = schema_dir / "pedestrian.schema.v1.json"
            schema2.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "title": "Pedestrian"}'
            )

            # Stage both files
            subprocess.run(["git", "add", "schemas/"], cwd=repo_path, check=True)

            # Test that prevent_schema_duplicates allows different schemas
            staged_files = ["schemas/episode.schema.v1.json", "schemas/pedestrian.schema.v1.json"]
            prevent_schema_duplicates(staged_files)  # Should not raise

    def test_git_hook_detects_duplicates_by_content_not_name(self):
        """Test that git hooks detect duplicates by content, not filename."""
        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Create a temporary git repository for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "test_repo"
            repo_path.mkdir()

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
            )

            # Create initial commit
            (repo_path / "README.md").write_text("# Test Repo")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)

            # Create schema files with same content but different names
            schema_dir = repo_path / "schemas"
            schema_dir.mkdir()

            schema1 = schema_dir / "episode.schema.v1.json"
            schema1.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            schema2 = schema_dir / "copy.schema.v1.json"
            schema2.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            # Stage both files
            subprocess.run(["git", "add", "schemas/"], cwd=repo_path, check=True)

            # Test that prevent_schema_duplicates detects content duplicates
            staged_files = ["schemas/episode.schema.v1.json", "schemas/copy.schema.v1.json"]

            with pytest.raises(ValueError, match="Duplicate schema content detected"):
                prevent_schema_duplicates(staged_files)

    def test_git_hook_provides_helpful_error_messages(self):
        """Test that git hooks provide helpful error messages for duplicates."""
        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        # Create a temporary git repository for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "test_repo"
            repo_path.mkdir()

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
            )

            # Create initial commit
            (repo_path / "README.md").write_text("# Test Repo")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)

            # Create duplicate schema files
            schema1 = repo_path / "schema1.json"
            schema1.write_text('{"type": "object"}')

            schema2 = repo_path / "schema2.json"
            schema2.write_text('{"type": "object"}')

            # Stage both files
            subprocess.run(
                ["git", "add", "schema1.json", "schema2.json"], cwd=repo_path, check=True
            )

            # Test that error message includes filenames
            staged_files = ["schema1.json", "schema2.json"]

            with pytest.raises(ValueError) as exc_info:
                prevent_schema_duplicates(staged_files)

            error_msg = str(exc_info.value)
            assert "schema1.json" in error_msg
            assert "schema2.json" in error_msg
            assert "Duplicate schema content detected" in error_msg
