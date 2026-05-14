"""Tests for the render helper catalog (T004)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.render.helper_catalog import (
    capture_frames,
    derive_recording_tags,
    deterministic_seed_from_name,
    ensure_output_dir,
)


def test_ensure_output_dir():
    """Test directory creation and path normalization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "nested" / "output" / "dir"

    result = ensure_output_dir(test_path)
    expected_path = resolve_artifact_path(test_path)

    assert result.exists()
    assert result.is_dir()
    assert result == expected_path


def test_ensure_output_dir_reraises_os_error(monkeypatch, tmp_path):
    """Verify directory creation errors are surfaced to callers."""

    def _raise_os_error(self, *args, **kwargs):
        raise OSError("no write access")

    monkeypatch.setattr(Path, "mkdir", _raise_os_error)

    with pytest.raises(OSError, match="no write access"):
        ensure_output_dir(tmp_path / "blocked")


def test_capture_frames_samples_buffered_frames_with_stride():
    """Verify buffered render frames are sampled without requiring a live GUI loop."""
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (0, 40, 80, 120)]
    mock_env = MagicMock()
    mock_env.sim_ui.frames = frames

    frames = capture_frames(mock_env, stride=2)

    assert len(frames) == 2
    assert np.array_equal(frames[0], np.full((2, 2, 3), 0, dtype=np.uint8))
    assert np.array_equal(frames[1], np.full((2, 2, 3), 80, dtype=np.uint8))


def test_capture_frames_uses_render_result_when_no_buffered_frames():
    """Verify direct render-array environments produce a real frame sample."""
    frame = np.full((2, 3, 3), 128, dtype=np.uint8)
    mock_env = MagicMock()
    mock_env.render.return_value = frame

    frames = capture_frames(mock_env)

    assert len(frames) == 1
    assert np.array_equal(frames[0], frame)


def test_capture_frames_converts_numeric_rgb_render_result():
    """Verify non-uint8 RGB render arrays are converted to byte frames."""
    mock_env = MagicMock()
    mock_env.render.return_value = np.array([[[0.0, 128.2, 300.0]]])

    frames = capture_frames(mock_env)

    assert frames[0].dtype == np.uint8
    assert frames[0].tolist() == [[[0, 128, 255]]]


def test_capture_frames_returns_empty_when_render_produces_no_rgb_data():
    """Verify render-capable inputs with no RGB data fail softly with no frames."""
    mock_env = MagicMock()
    mock_env.render.return_value = np.zeros((2, 2), dtype=np.uint8)

    assert capture_frames(mock_env) == []


def test_capture_frames_uses_buffer_populated_by_render_side_effect():
    """Verify render methods that append to sim_ui.frames are sampled after rendering."""
    frame = np.full((2, 2, 3), 33, dtype=np.uint8)
    mock_env = MagicMock()
    mock_env.sim_ui.frames = []

    def _render():
        mock_env.sim_ui.frames.append(frame)

    mock_env.render.side_effect = _render

    frames = capture_frames(mock_env)

    assert len(frames) == 1
    assert np.array_equal(frames[0], frame)


def test_capture_frames_reraises_render_errors():
    """Verify render failures are not hidden by the helper."""
    mock_env = MagicMock()
    mock_env.render.side_effect = RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        capture_frames(mock_env)


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


def test_derive_recording_tags_handles_full_and_partial_stems():
    """Verify recording tag derivation keeps stable defaults for missing segments."""
    assert derive_recording_tags("suite_scenario_algo.jsonl") == ("suite", "scenario", "algo")
    assert derive_recording_tags("suite_only") == ("suite", "only", "unknown")
    assert derive_recording_tags("") == ("converted", "legacy", "unknown")


def test_deterministic_seed_from_name_is_stable_31_bit_value():
    """Verify deterministic seeds are stable and constrained to non-negative 31-bit values."""
    seed = deterministic_seed_from_name("demo_recording.jsonl")

    assert seed == deterministic_seed_from_name("demo_recording.jsonl")
    assert 0 <= seed <= 0x7FFFFFFF
