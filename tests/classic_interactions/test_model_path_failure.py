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


def test_default_model_path_uses_registry_after_legacy_cutover(monkeypatch, tmp_path):
    """The demo should hydrate the cut-over legacy checkpoint through the registry."""
    mod = _demo_module()
    resolved_path = tmp_path / "legacy_ppo_run_043.zip"
    resolved_path.touch()
    resolved_ids: list[str] = []
    loaded_paths: list[str] = []

    def fake_resolve_model_path(model_id: str, **_kwargs: object) -> Path:
        """Record the registry lookup and return a deterministic fixture path."""
        resolved_ids.append(model_id)
        return resolved_path

    def fake_load_trained_policy(path: str) -> object:
        """Record the hydrated checkpoint passed to the policy loader."""
        loaded_paths.append(path)
        return object()

    monkeypatch.setattr(mod, "MODEL_PATH", Path("model/run_043.zip"))
    monkeypatch.setattr(mod, "resolve_model_path", fake_resolve_model_path, raising=False)
    monkeypatch.setattr(mod, "load_trained_policy", fake_load_trained_policy)
    monkeypatch.setattr(mod, "_load_map_definition", lambda _map_file: None)
    monkeypatch.setattr(mod, "make_robot_env", lambda **_kwargs: object())
    monkeypatch.setattr(mod, "_run_episodes", lambda *_args, **_kwargs: [])

    mod.run_demo(dry_run=False, max_episodes=1, enable_recording=False)

    assert resolved_ids == ["legacy_ppo_run_043"]
    assert loaded_paths == [str(resolved_path)]
