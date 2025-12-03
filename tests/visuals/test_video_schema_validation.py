"""Tests for video_artifacts.schema.json validity and enforcement.

Covers:
- Successful validation of a minimal valid manifest
- Failure on missing required field
- Failure when success status missing filename
Skips if jsonschema not installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCHEMA_PATH = Path("specs/127-enhance-benchmark-visual/contracts/video_artifacts.schema.json")

jsonschema_spec = importlib.util.find_spec("jsonschema")
pytestmark = pytest.mark.skipif(jsonschema_spec is None, reason="jsonschema not installed")

if jsonschema_spec:  # type: ignore
    import jsonschema  # type: ignore


def load_schema():  # pragma: no cover - trivial IO helper
    """TODO docstring. Document this function."""
    return json.loads(SCHEMA_PATH.read_text())


def validate(instance):
    """Validate instance against video schema."""
    schema = load_schema()
    jsonschema.validate(instance=instance, schema=schema)  # type: ignore


def test_valid_minimal_manifest():
    """TODO docstring. Document this function."""
    validate({"artifacts": []})  # empty list allowed


def test_valid_success_entry():
    """TODO docstring. Document this function."""
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
    # Expect schema validation failure due to missing required 'episode_id' field
    """TODO docstring. Document this function."""
    with pytest.raises(jsonschema.ValidationError):  # type: ignore[name-defined]
        validate({"artifacts": [{"renderer": "synthetic", "status": "skipped"}]})


def test_success_without_filename_fails():
    # Success status requires a filename according to schema
    """TODO docstring. Document this function."""
    with pytest.raises(jsonschema.ValidationError):  # type: ignore[name-defined]
        validate({"artifacts": [{"episode_id": "1", "renderer": "synthetic", "status": "success"}]})
