"""Tests for scenario split helpers."""

from __future__ import annotations

import pytest

from robot_sf.training.scenario_split import split_scenarios


def test_split_scenarios_defaults_to_train() -> None:
    """Default missing split entries to train so configs remain backward compatible."""
    scenarios = [{"name": "A"}, {"name": "B", "split": "holdout"}]
    splits = split_scenarios(scenarios)
    assert [s["name"] for s in splits["train"]] == ["A"]
    assert [s["name"] for s in splits["holdout"]] == ["B"]


def test_split_scenarios_rejects_unknown_split() -> None:
    """Reject unknown split labels to prevent silent evaluation leakage."""
    scenarios = [{"name": "A", "split": "invalid"}]
    with pytest.raises(ValueError):
        split_scenarios(scenarios)
