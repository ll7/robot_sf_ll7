"""Model path failure test (T011 / FR-007).

Purpose (TDD): Assert that when the PPO model path constant points to a non-existent
file, calling run_demo() produces a clear, actionable error message. We *intentionally*
require a guidance phrase that is NOT yet implemented so this test FAILS until task
T021 adds improved messaging.

Expected future behavior (will fail now):
  - Raising FileNotFoundError (or RuntimeError wrapper) containing the missing path
  - Message includes a guidance hint phrase like 'download' or 'pre-trained PPO model'
    so that users know how to resolve it.

Current state: run_demo() will raise FileNotFoundError with a short message lacking
guidance, so the assertion for the guidance phrase will fail (desired TDD failure).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _demo_module():
    """TODO docstring. Document this function."""
    return importlib.import_module("examples.classic_interactions_pygame")


def test_model_path_missing_provides_actionable_message():
    """TODO docstring. Document this function."""
    mod = _demo_module()
    # Patch constants: ensure DRY_RUN disabled and model path points to definitely-missing file.
    original_dry = getattr(mod, "DRY_RUN", None)
    original_model_path = getattr(mod, "MODEL_PATH", None)
    mod.DRY_RUN = False  # type: ignore
    # Use a deeply nested, improbable path to avoid accidental existence from fixtures or prior artifacts.
    missing_path = Path(
        "model/__definitely_missing_do_not_create__/__this_model_file_does_not_exist__.zip",
    )
    assert not missing_path.exists(), (
        "Test invariant violated: missing model path unexpectedly exists (choose a different sentinel)."
    )
    mod.MODEL_PATH = missing_path  # type: ignore

    try:
        with pytest.raises((FileNotFoundError, RuntimeError)) as excinfo:
            mod.run_demo()
    finally:
        if original_dry is not None:
            mod.DRY_RUN = original_dry  # type: ignore
        if original_model_path is not None:
            mod.MODEL_PATH = original_model_path  # type: ignore

    msg = str(excinfo.value).lower()
    # TDD FUTURE ASSERTION: require guidance keywords (will currently FAIL because
    # implementation only reports simple missing file).
    assert "download" in msg or "pre-trained" in msg, (
        "Model missing error message lacks user guidance (expected 'download' or 'pre-trained' phrase)."
    )
