"""Contract tests for schema consolidation feature (T011)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EPISODE_SCHEMA_PATH = ROOT / "robot_sf" / "benchmark" / "schemas" / "episode.schema.v1.json"
PRE_COMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"
PRE_COMMIT_HOOK = ROOT / ".git" / "hooks" / "pre-commit"

try:
    import jsonschema  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("jsonschema dependency required for contract tests") from e


class TestSchemaLoaderContract:
    """Contract tests for schema loading API."""

    def test_load_schema_contract(self):
        """The schema loader should return the canonical episode schema."""
        from robot_sf.benchmark.schema_loader import load_schema

        schema = load_schema("episode.schema.v1.json")

        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "title" in schema
        assert schema["title"] == "RobotSF Benchmark Episode (v1)"

    def test_get_schema_version_contract(self):
        """The schema loader should expose parsed semantic version metadata."""
        from robot_sf.benchmark.schema_loader import get_schema_version

        version = get_schema_version("episode.schema.v1.json")

        assert isinstance(version, dict)
        assert "major" in version
        assert "minor" in version
        assert "patch" in version
        assert version["major"] == 1
        assert version["minor"] == 0
        assert version["patch"] == 0

    def test_schema_loading_error_handling(self):
        """Invalid schema names should fail closed."""
        from robot_sf.benchmark.schema_loader import load_schema

        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent.schema.v1.json")

    def test_schema_integrity_validation(self):
        """Loaded schemas should be valid JSON Schema documents."""
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
        hook_configured = False

        if PRE_COMMIT_CONFIG.exists():
            content = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
            if "prevent-schema-duplicates" in content:
                hook_configured = True

        if PRE_COMMIT_HOOK.exists():
            content = PRE_COMMIT_HOOK.read_text(encoding="utf-8")
            if "prevent-schema-duplicates" in content:
                hook_configured = True

        assert hook_configured, "Duplicate prevention hook not configured"

    def test_duplicate_detection_logic(self, tmp_path):
        """The hook helper should reject staged schemas that duplicate canonical files."""
        from hooks.prevent_schema_duplicates import prevent_schema_duplicates

        canonical_dir = tmp_path / "canonical"
        canonical_dir.mkdir()
        canonical_schema = canonical_dir / "episode.schema.v1.json"
        canonical_schema.write_text(
            EPISODE_SCHEMA_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )

        duplicate_dir = tmp_path / "duplicate"
        duplicate_dir.mkdir()
        duplicate_schema = duplicate_dir / "episode.schema.v1.json"
        duplicate_schema.write_text(canonical_schema.read_text(encoding="utf-8"), encoding="utf-8")

        result = prevent_schema_duplicates(
            staged_files=[str(duplicate_schema)],
            canonical_dir=canonical_dir,
        )

        assert result["status"] == "fail"
        assert result["duplicates_found"] == [
            {
                "file": str(duplicate_schema),
                "canonical_file": str(canonical_schema),
                "reason": "Content hash matches existing canonical schema",
            }
        ]


class TestBackwardCompatibilityContract:
    """Contract tests for backward compatibility."""

    def test_legacy_schema_access_still_works(self):
        """Test that direct file access still works during transition."""
        # Contract: canonical schema file exists and is valid JSON
        assert EPISODE_SCHEMA_PATH.exists(), "Canonical schema file missing"

        # Contract: file contains valid JSON
        content = EPISODE_SCHEMA_PATH.read_text(encoding="utf-8")
        schema = json.loads(content)

        # Contract: has expected structure
        assert schema["title"] == "RobotSF Benchmark Episode (v1)"
        # Note: "version" is a property constraint, not a top-level schema field
        assert "version" in schema["properties"]
        assert schema["properties"]["version"]["const"] == "v1"

    def test_episode_schema_wrapper_reads_canonical_schema(self):
        """The legacy EpisodeSchema wrapper should still read the canonical file."""
        from robot_sf.benchmark.schemas.episode_schema import EpisodeSchema

        schema = EpisodeSchema(EPISODE_SCHEMA_PATH)

        assert schema.title == "RobotSF Benchmark Episode (v1)"
        assert schema.version == "v1"
        assert "episode_id" in schema.required_properties
