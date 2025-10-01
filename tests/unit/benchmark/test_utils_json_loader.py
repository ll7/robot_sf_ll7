"""Test the load_optional_json utility function."""

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.utils import load_optional_json


def test_load_optional_json_none_path():
    """Test that None path returns None."""
    result = load_optional_json(None)
    assert result is None


def test_load_optional_json_empty_string():
    """Test that empty string returns None."""
    result = load_optional_json("")
    assert result is None


def test_load_optional_json_nonexistent_file():
    """Test that nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_optional_json("/nonexistent/path/file.json")


def test_load_optional_json_valid_file():
    """Test loading a valid JSON file."""
    test_data = {"key": "value", "number": 42}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        f.flush()

        result = load_optional_json(f.name)

    # Clean up
    Path(f.name).unlink()

    assert result == test_data


def test_load_optional_json_invalid_json():
    """Test that invalid JSON raises JSONDecodeError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json content {")
        f.flush()

        with pytest.raises(json.JSONDecodeError):
            load_optional_json(f.name)

    # Clean up
    Path(f.name).unlink()


def test_load_optional_json_empty_dict():
    """Test loading an empty JSON object."""
    test_data = {}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        f.flush()

        result = load_optional_json(f.name)

    # Clean up
    Path(f.name).unlink()

    assert result == test_data
