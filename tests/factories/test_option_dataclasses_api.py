"""Tests for option dataclasses (``RenderOptions`` and ``RecordingOptions``).

Purpose:
    Ensure option dataclasses expose the expected fields with correct default
    values and that their ``validate()`` methods enforce invalid parameter
    constraints (e.g., non‑positive FPS overrides, non‑positive velocity scale,
    invalid max frame counts, or unsupported video formats).

Scope covered by this suite:
    * Field presence & default values
    * Validation error raising for invalid numeric bounds
    * Validation of recording path / codec expectations
    * Convenience constructor precedence in ``RecordingOptions.from_bool_and_path``

All tests are expected to pass with the current implementation. If new
fields or validation rules are introduced, update tests (additions should
generally be additive—avoid silently changing existing semantics without
adjusting this suite). The original provisional "failing test" scaffolding
phase (T005/T006) has been completed.
"""

from __future__ import annotations

import pytest

from robot_sf.gym_env.options import RecordingOptions, RenderOptions


def test_render_options_fields_exist():
    """Test render options fields exist.

    Returns:
        Any: Auto-generated placeholder description.
    """
    r = RenderOptions()
    assert r.enable_overlay is False
    assert r.max_fps_override is None
    assert r.ped_velocity_scale == 1.0
    assert r.headless_ok is True


def test_recording_options_fields_exist():
    """Test recording options fields exist.

    Returns:
        Any: Auto-generated placeholder description.
    """
    ro = RecordingOptions()
    assert ro.record is False
    assert ro.video_path is None
    assert ro.max_frames is None
    assert ro.codec == "libx264"
    assert ro.bitrate is None


@pytest.mark.parametrize("fps", [-5, 0])
def test_render_options_invalid_fps_raises(fps):
    """Test render options invalid fps raises.

    Args:
        fps: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    opt = RenderOptions(max_fps_override=fps)
    with pytest.raises(ValueError):
        opt.validate()


def test_render_options_invalid_scale():
    """Test render options invalid scale.

    Returns:
        Any: Auto-generated placeholder description.
    """
    opt = RenderOptions(ped_velocity_scale=0.0)
    with pytest.raises(ValueError):
        opt.validate()


@pytest.mark.parametrize("max_frames", [0, -1])
def test_recording_options_invalid_max_frames(max_frames):
    """Test recording options invalid max frames.

    Args:
        max_frames: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    opt = RecordingOptions(record=True, max_frames=max_frames)
    with pytest.raises(ValueError):
        opt.validate()


def test_recording_options_invalid_video_path():
    """Test recording options invalid video path.

    Returns:
        Any: Auto-generated placeholder description.
    """
    opt = RecordingOptions(record=True, video_path="output.mov")
    with pytest.raises(ValueError):
        opt.validate()


def test_recording_options_from_bool_and_path_precedence():
    """Test recording options from bool and path precedence.

    Returns:
        Any: Auto-generated placeholder description.
    """
    base = RecordingOptions(record=False, video_path=None)
    merged = RecordingOptions.from_bool_and_path(True, "test.mp4", base)
    assert merged.record is True
    assert merged.video_path == "test.mp4"
