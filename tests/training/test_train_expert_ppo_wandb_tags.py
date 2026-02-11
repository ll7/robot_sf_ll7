"""Tests for W&B tag normalization in expert PPO training."""

from __future__ import annotations

from scripts.training.train_expert_ppo import _normalize_wandb_tags


def test_normalize_wandb_tags_keeps_single_string_as_single_tag() -> None:
    """Ensure a single string tag stays atomic so W&B tags are not split by character."""
    assert _normalize_wandb_tags("baseline") == ["baseline"]


def test_normalize_wandb_tags_converts_sequence_items_to_strings() -> None:
    """Ensure mixed tag sequences serialize predictably for W&B metadata reproducibility."""
    assert _normalize_wandb_tags(["optuna", 7, "metric:snqi"]) == ["optuna", "7", "metric:snqi"]


def test_normalize_wandb_tags_decodes_bytes_and_ignores_unknown_scalars() -> None:
    """Ensure bytes are accepted while unsupported scalar types cleanly disable tag injection."""
    assert _normalize_wandb_tags(b"trial") == ["trial"]
    assert _normalize_wandb_tags(12345) is None
