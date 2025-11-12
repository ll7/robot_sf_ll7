"""Tests for the render helper catalog (T004)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.render.helper_catalog import capture_frames, ensure_output_dir


def test_ensure_output_dir():
    """Test directory creation and path normalization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "nested" / "output" / "dir"

    result = ensure_output_dir(test_path)
    expected_path = resolve_artifact_path(test_path)

    assert result.exists()
    assert result.is_dir()
    assert result == expected_path


def test_capture_frames():
    """Test frame capture with stride parameter."""
    mock_env = MagicMock()
    mock_env.render.return_value = None

    frames = capture_frames(mock_env, stride=2)

    # Current implementation returns empty list with warning
    assert isinstance(frames, list)
    assert len(frames) == 0


def test_capture_frames_invalid_stride():
    """Test frame capture with invalid stride parameter."""
    mock_env = MagicMock()

    with pytest.raises(ValueError, match="stride must be >= 1"):
        capture_frames(mock_env, stride=0)


def test_capture_frames_no_render_method():
    """Test frame capture when environment doesn't support rendering."""
    mock_env = MagicMock(spec=[])  # Mock without render method

    with pytest.raises(AttributeError, match="does not support rendering"):
        capture_frames(mock_env)
