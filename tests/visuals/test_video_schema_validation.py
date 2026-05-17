"""Tests for video_artifacts.schema.json validity and enforcement.

Covers:
- Successful validation of a minimal valid manifest
- Failure on missing required field
- Failure when success status missing filename
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path("specs/127-enhance-benchmark-visual/contracts/video_artifacts.schema.json")


def load_schema():  # pragma: no cover - trivial IO helper
    """Load the video artifact schema from disk."""
    return json.loads(SCHEMA_PATH.read_text())


def validate(instance):
    """Validate instance against video schema."""
    schema = load_schema()
    jsonschema.validate(instance=instance, schema=schema)


def test_valid_minimal_manifest():
    """Empty artifact lists should satisfy the video schema."""
    validate({"artifacts": []})  # empty list allowed


def test_valid_success_entry():
    """Successful video entries should validate when filename is present."""
    validate(
        {
            "artifacts": [
                {
                    "episode_id": "ep1",
                    "renderer": "synthetic",
                    "status": "success",
                    "filename": "video_ep1.mp4",
                },
            ],
        },
    )


def test_missing_required_field():
    """Video entries should require an episode id."""
    with pytest.raises(jsonschema.ValidationError):
        validate({"artifacts": [{"renderer": "synthetic", "status": "skipped"}]})


def test_success_without_filename_fails():
    """Successful video entries should require a filename."""
    with pytest.raises(jsonschema.ValidationError):
        validate({"artifacts": [{"episode_id": "1", "renderer": "synthetic", "status": "success"}]})
