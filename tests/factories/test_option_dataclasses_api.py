"""Tests for environment factory option dataclasses.

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

from robot_sf.gym_env.options import (
    JsonlRecordingOptions,
    RecordingOptions,
    RenderOptions,
    TelemetryOptions,
)
from robot_sf.telemetry import DEFAULT_TELEMETRY_METRICS


def test_render_options_fields_exist():
    """Render options expose the reviewed defaults."""
    r = RenderOptions()
    assert r.enable_overlay is False
    assert r.max_fps_override is None
    assert r.ped_velocity_scale == 1.0
    assert r.headless_ok is True


def test_recording_options_fields_exist():
    """Recording options expose the reviewed defaults."""
    ro = RecordingOptions()
    assert ro.record is False
    assert ro.video_path is None
    assert ro.max_frames is None
    assert ro.codec == "libx264"
    assert ro.bitrate is None


def test_jsonl_recording_options_fields_exist():
    """JSONL options expose stable metadata defaults."""
    opts = JsonlRecordingOptions()
    assert opts.enabled is False
    assert opts.recording_dir == "recordings"
    assert opts.suite_name == "robot_sim"
    assert opts.scenario_name == "default"
    assert opts.algorithm_name == "manual"
    assert opts.recording_seed is None


def test_telemetry_options_fields_exist():
    """Telemetry options mirror the robot config telemetry contract."""
    opts = TelemetryOptions()
    assert opts.enable_panel is False
    assert opts.record is False
    assert opts.metrics == list(DEFAULT_TELEMETRY_METRICS)
    assert opts.refresh_hz == 1.0
    assert opts.pane_layout == "vertical_split"
    assert opts.decimation == 1


@pytest.mark.parametrize("fps", [-5, 0])
def test_render_options_invalid_fps_raises(fps):
    """Non-positive FPS overrides are rejected."""
    opt = RenderOptions(max_fps_override=fps)
    with pytest.raises(ValueError):
        opt.validate()


def test_render_options_invalid_scale():
    """Non-positive pedestrian velocity scale is rejected."""
    opt = RenderOptions(ped_velocity_scale=0.0)
    with pytest.raises(ValueError):
        opt.validate()


@pytest.mark.parametrize("max_frames", [0, -1])
def test_recording_options_invalid_max_frames(max_frames):
    """Non-positive recording frame caps are rejected."""
    opt = RecordingOptions(record=True, max_frames=max_frames)
    with pytest.raises(ValueError):
        opt.validate()


def test_recording_options_invalid_video_path():
    """Recorded video paths must use the project MP4 container."""
    opt = RecordingOptions(record=True, video_path="output.mov")
    with pytest.raises(ValueError):
        opt.validate()


def test_recording_options_from_bool_and_path_precedence():
    """Convenience recording flags can upgrade a disabled options object."""
    base = RecordingOptions(record=False, video_path=None)
    merged = RecordingOptions.from_bool_and_path(True, "test.mp4", base)
    assert merged.record is True
    assert merged.video_path == "test.mp4"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"recording_dir": ""}, "recording_dir"),
        ({"suite_name": ""}, "suite_name"),
        ({"scenario_name": ""}, "scenario_name"),
        ({"algorithm_name": ""}, "algorithm_name"),
    ],
)
def test_jsonl_recording_options_validate_required_metadata(kwargs, match):
    """Blank JSONL metadata fields fail before recorder construction."""
    opts = JsonlRecordingOptions(**kwargs)
    with pytest.raises(ValueError, match=match):
        opts.validate()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"refresh_hz": 0.0}, "refresh_hz"),
        ({"decimation": 0}, "decimation"),
        ({"pane_layout": "floating"}, "pane_layout"),
    ],
)
def test_telemetry_options_validate_bounds(kwargs, match):
    """Telemetry options enforce the same bounds as config validation."""
    opts = TelemetryOptions(**kwargs)
    with pytest.raises(ValueError, match=match):
        opts.validate()


def test_telemetry_options_validate_normalizes_metrics():
    """Blank metric lists fall back to default telemetry metrics."""
    opts = TelemetryOptions(metrics=["", "speed", "  "])
    opts.validate()
    assert opts.metrics == ["speed"]

    empty = TelemetryOptions(metrics=[])
    empty.validate()
    assert empty.metrics == list(DEFAULT_TELEMETRY_METRICS)
