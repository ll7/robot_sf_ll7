"""Regression tests for predictive success campaign artifact naming."""

from __future__ import annotations

from scripts.validation import run_predictive_success_campaign as campaign


def test_checkpoint_token_is_path_sensitive() -> None:
    """Checkpoint token should differ for same filename under different directories."""
    a = campaign._checkpoint_token("output/a/predictive_model.pt")
    b = campaign._checkpoint_token("output/b/predictive_model.pt")
    assert a != b
    assert a.startswith("predictive_model_")
    assert b.startswith("predictive_model_")
