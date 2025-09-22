"""Recording success test (T014 / FR-009).

Behavior under test (future implementation): When recording is enabled and dependencies
(`moviepy` + ffmpeg) are available, each completed episode should produce an MP4 file
in the configured OUTPUT_DIR with a deterministic naming pattern including scenario
name and seed.

Skip conditions: If moviepy import is not available (MOVIEPY_AVAILABLE False), the test
is skipped (graceful behavior covered by T007). If implementation not ready, the test
will fail due to missing output file—expected during TDD phase.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_recording_creates_mp4_when_enabled():  # noqa: D401
    mod = importlib.import_module("examples.classic_interactions_pygame")
    # Skip if moviepy not available
    if not getattr(mod, "MOVIEPY_AVAILABLE", False):  # type: ignore[attr-defined]
        pytest.skip("moviepy not available; skipping recording success test")

    # Patch constants
    original_dry = getattr(mod, "DRY_RUN", None)
    original_rec = getattr(mod, "ENABLE_RECORDING", None)
    original_out = getattr(mod, "OUTPUT_DIR", None)
    mod.DRY_RUN = False  # type: ignore
    mod.ENABLE_RECORDING = True  # type: ignore
    tmp_out = Path("results/test_recording_success")
    if tmp_out.exists():
        # Clean stale artifacts
        for p in tmp_out.glob("*.mp4"):
            try:
                p.unlink()
            except Exception:  # noqa: BLE001
                pass
    mod.OUTPUT_DIR = tmp_out  # type: ignore

    try:
        episodes = mod.run_demo()
    finally:
        if original_dry is not None:
            mod.DRY_RUN = original_dry  # type: ignore
        if original_rec is not None:
            mod.ENABLE_RECORDING = original_rec  # type: ignore
        if original_out is not None:
            mod.OUTPUT_DIR = original_out  # type: ignore

    assert episodes, (
        "Expected episodes for recording success test (TDD failing until implementation)."
    )
    # Expect at least one mp4 in tmp_out
    mp4_files = list(tmp_out.glob("*.mp4"))
    assert mp4_files, "Expected at least one MP4 recording file (TDD failing until implementation)."
