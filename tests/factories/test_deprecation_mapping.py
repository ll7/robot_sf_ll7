"""T008: Tests for deprecation mapping layer (_factory_compat).

Covers:
- Mapping of known legacy parameters.
- Strict mode error on unknown legacy kw.
- Permissive mode via env var ROBOT_SF_FACTORY_LEGACY=1.
- Warning message capture using Loguru sink.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from loguru import logger

from robot_sf.gym_env._factory_compat import (
    LEGACY_PERMISSIVE_ENV,
    UnknownLegacyParameterError,
    apply_legacy_kwargs,
)


@contextmanager
def capture_loguru(level: str = "WARNING"):
    """TODO docstring. Document this function.

    Args:
        level: TODO docstring.
    """
    messages: list[str] = []

    def _sink(msg):  # type: ignore[override]
        """TODO docstring. Document this function.

        Args:
            msg: TODO docstring.
        """
        if msg.record["level"].name == level:
            messages.append(msg.record["message"])

    sink_id = logger.add(_sink)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_mapping_known_legacy_parameters():
    """TODO docstring. Document this function."""
    raw = {"record_video": True, "fps": 30, "video_output_path": "out.mp4"}
    with capture_loguru() as warnings:
        normalized, emitted = apply_legacy_kwargs(raw, strict=True)
    # Flattened mapping keys appear
    assert normalized["recording_options.record"] is True
    assert normalized["render_options.max_fps_override"] == 30
    assert normalized["recording_options.video_path"] == "out.mp4"
    # All three warnings present
    assert len(emitted) == 3
    assert len(warnings) == 3


def test_strict_mode_unknown_legacy_raises():
    """TODO docstring. Document this function."""
    raw = {"record_video": True, "fps": 10, "video_output_path": "x.mp4", "videp_path": "typo.mp4"}
    # 'videp_path' should trigger unknown parameter in strict mode
    with pytest.raises(UnknownLegacyParameterError):
        apply_legacy_kwargs(raw, strict=True)


def test_permissive_mode_unknown_ignored(monkeypatch):
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
    """
    monkeypatch.setenv(LEGACY_PERMISSIVE_ENV, "1")
    raw = {"record_video": True, "fps": 10, "video_output_path": "x.mp4", "videp_path": "typo.mp4"}
    with capture_loguru() as warnings:
        normalized, emitted = apply_legacy_kwargs(raw, strict=True)  # strict True but env overrides
    # Known ones mapped
    assert normalized["recording_options.record"] is True
    # Unknown should be ignored, not present
    assert "videp_path" not in normalized
    # Warnings include at least mapping warnings and an unknown warning
    assert any("Unknown parameter 'videp_path'" in w for w in warnings)
    assert len(emitted) >= 3
    monkeypatch.delenv(LEGACY_PERMISSIVE_ENV, raising=False)


def test_strict_false_behaves_like_permissive():
    # Even without env var, strict=False allows unknown ignore
    """TODO docstring. Document this function."""
    raw = {"fps": 20, "unknown_leg": 5}
    with capture_loguru() as warnings:
        normalized, emitted = apply_legacy_kwargs(raw, strict=False)
    assert normalized["render_options.max_fps_override"] == 20
    assert any("Unknown parameter 'unknown_leg'" in w for w in warnings)
    assert len(emitted) >= 2
