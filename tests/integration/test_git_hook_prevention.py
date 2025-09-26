"""
Integration tests for git hook duplicate prevention feature.

Tests the end-to-end functionality of preventing duplicate schema files
from being committed using git hooks.
"""

import subprocess
import tempfile
from pathlib import Path


class TestGitHookPreventionIntegration:
    """Integration tests for git hook duplicate prevention functionality."""

    def test_git_hook_prevents_duplicate_schema_commits(self, monkeypatch):
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

            # Change to the repo directory for the test
            monkeypatch.chdir(repo_path)

            # Create a schema file
            schema_dir = Path("schemas")
            schema_dir.mkdir()
            schema_file = schema_dir / "episode.schema.v1.json"
            schema_file.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            # Stage the schema file
            subprocess.run(["git", "add", "schemas/episode.schema.v1.json"], check=True)

            # Test that prevent_schema_duplicates works with staged files
            staged_files = ["schemas/episode.schema.v1.json"]
            result = prevent_schema_duplicates(staged_files, canonical_dir=schema_dir)
            assert result["status"] == "pass"  # Should pass since it's in canonical dir

            # Now create a duplicate schema file
            duplicate_schema = Path("duplicate.schema.v1.json")
            duplicate_schema.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            # Stage the duplicate
            subprocess.run(["git", "add", "duplicate.schema.v1.json"], check=True)

            # Test that prevent_schema_duplicates detects the duplicate
            staged_files_with_duplicate = [
                "schemas/episode.schema.v1.json",
                "duplicate.schema.v1.json",
            ]

            result = prevent_schema_duplicates(
                staged_files_with_duplicate, canonical_dir=schema_dir
            )
            assert result["status"] == "fail"
            assert "duplicate" in result["message"].lower()

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

    def test_git_hook_detects_duplicates_by_content_not_name(self, monkeypatch):
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

            # Change to the repo directory for the test
            monkeypatch.chdir(repo_path)

            # Create canonical schema
            schema_dir = Path("schemas")
            schema_dir.mkdir()
            canonical_schema = schema_dir / "episode.schema.v1.json"
            canonical_schema.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            # Commit the canonical schema
            subprocess.run(["git", "add", "schemas/"], check=True)
            subprocess.run(["git", "commit", "-m", "Add canonical schema"], check=True)

            # Create a duplicate schema file with different name
            duplicate_schema = Path("copy.schema.v1.json")
            duplicate_schema.write_text(
                '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
            )

            # Stage the duplicate
            subprocess.run(["git", "add", "copy.schema.v1.json"], check=True)

            # Test that prevent_schema_duplicates detects content duplicates
            staged_files = ["copy.schema.v1.json"]

            result = prevent_schema_duplicates(staged_files, canonical_dir=schema_dir)
            assert result["status"] == "fail"
            assert "duplicate" in result["message"].lower()

    def test_git_hook_provides_helpful_error_messages(self, monkeypatch):
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

            # Change to the repo directory for the test
            monkeypatch.chdir(repo_path)

            # Create canonical schema directory and file
            canonical_dir = Path("robot_sf/benchmark/schemas")
            canonical_dir.mkdir(parents=True)
            canonical_schema = canonical_dir / "test.schema.v1.json"
            canonical_schema.write_text('{"type": "object", "canonical": true}')

            # Commit the canonical schema
            subprocess.run(["git", "add", "robot_sf/"], check=True)
            subprocess.run(["git", "commit", "-m", "Add canonical schema"], check=True)

            # Create duplicate schema files (one matching canonical, one different)
            schema1 = Path("schema1.schema.v1.json")
            schema1.write_text('{"type": "object", "canonical": true}')  # duplicate

            schema2 = Path("schema2.schema.v1.json")
            schema2.write_text('{"type": "object", "canonical": true}')  # duplicate

            # Stage both files
            subprocess.run(
                ["git", "add", "schema1.schema.v1.json", "schema2.schema.v1.json"], check=True
            )

            # Test that error message includes filenames
            staged_files = ["schema1.schema.v1.json", "schema2.schema.v1.json"]

            result = prevent_schema_duplicates(staged_files, canonical_dir=canonical_dir)
            assert result["status"] == "fail"
            assert len(result["duplicates_found"]) == 2
            duplicate_files = [dup["file"] for dup in result["duplicates_found"]]
            assert "schema1.schema.v1.json" in duplicate_files
            assert "schema2.schema.v1.json" in duplicate_files
