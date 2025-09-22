"""Recording skip graceful test (T007 / FR-008).

TDD: Expects run_demo(recording disabled by constant) to *not* raise even if MOVIEPY unavailable.
Later when recording enabled constant is toggled in implementation, additional logic will
assert a summary field `recorded` is False when dependency missing.
Currently will FAIL because episodes list is empty (depends on core implementation).
"""

from __future__ import annotations

import importlib


def test_recording_skip_graceful():  # noqa: D401
    mod = importlib.import_module("examples.classic_interactions_pygame")
    # Force non-dry path
    if hasattr(mod, "DRY_RUN"):
        original_dry = mod.DRY_RUN
        mod.DRY_RUN = False  # type: ignore
    else:
        original_dry = None

    if hasattr(mod, "ENABLE_RECORDING"):
        original_rec = mod.ENABLE_RECORDING
        mod.ENABLE_RECORDING = True  # type: ignore
    else:
        original_rec = None

    try:
        episodes = mod.run_demo()
    finally:
        if hasattr(mod, "DRY_RUN"):
            mod.DRY_RUN = original_dry  # type: ignore
        if hasattr(mod, "ENABLE_RECORDING"):
            mod.ENABLE_RECORDING = original_rec  # type: ignore

    assert episodes, "Expected episodes for recording skip test (TDD failing until implementation)."
    # At least ensure recorded flag is boolean (future assertion will check False if MOVIEPY missing)
    assert isinstance(episodes[0].get("recorded"), bool)
