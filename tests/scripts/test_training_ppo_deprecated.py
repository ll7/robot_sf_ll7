"""Tests for legacy training_ppo.py deprecation guard."""

from __future__ import annotations

from scripts import training_ppo


def test_training_ppo_main_returns_deprecation_code(capsys) -> None:
    """Legacy script should hard-fail with migration guidance."""
    code = training_ppo.main()
    captured = capsys.readouterr()
    assert code == 2
    assert "deprecated" in captured.err
    assert "train_expert_ppo.py" in captured.err
