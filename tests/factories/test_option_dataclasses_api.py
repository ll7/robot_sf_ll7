"""T005: Failing tests for option dataclasses prior to full implementation.

These tests enforce field presence and validation behavior to be added in T006.
Initially, validate() is a no-op, so tests expecting errors will fail.
"""

from __future__ import annotations

import pytest

from robot_sf.gym_env.options import RecordingOptions, RenderOptions


def test_render_options_fields_exist():
    r = RenderOptions()
    assert r.enable_overlay is False
    assert r.max_fps_override is None
    assert r.ped_velocity_scale == 1.0
    assert r.headless_ok is True


def test_recording_options_fields_exist():
    ro = RecordingOptions()
    assert ro.record is False
    assert ro.video_path is None
    assert ro.max_frames is None
    assert ro.codec == "libx264"
    assert ro.bitrate is None


@pytest.mark.parametrize("fps", [-5, 0])
def test_render_options_invalid_fps_raises(fps):
    opt = RenderOptions(max_fps_override=fps)
    with pytest.raises(ValueError):
        opt.validate()


def test_render_options_invalid_scale():
    opt = RenderOptions(ped_velocity_scale=0.0)
    with pytest.raises(ValueError):
        opt.validate()


@pytest.mark.parametrize("max_frames", [0, -1])
def test_recording_options_invalid_max_frames(max_frames):
    opt = RecordingOptions(record=True, max_frames=max_frames)
    with pytest.raises(ValueError):
        opt.validate()


def test_recording_options_invalid_video_path():
    opt = RecordingOptions(record=True, video_path="output.mov")
    with pytest.raises(ValueError):
        opt.validate()


def test_recording_options_from_bool_and_path_precedence():
    base = RecordingOptions(record=False, video_path=None)
    merged = RecordingOptions.from_bool_and_path(True, "test.mp4", base)
    assert merged.record is True
    assert merged.video_path == "test.mp4"
