"""Tests for issue #4850 multiplier rank-sensitivity report formatting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/generate_multiplier_sensitivity_report_4850.py"
    )
    spec = importlib.util.spec_from_file_location("issue4850_multiplier_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


issue4850 = _load_module()


def test_print_arm_summaries_uses_nested_rank_sensitivity_shape(capsys) -> None:
    """Nested arm results should render rankings, means, and pairwise probabilities."""
    arms = {
        "multiplier_0.1": {
            "observed_means": {"goal": 1.23456, "social_force": 0.5},
            "ranking": ["goal", "social_force"],
            "pairwise_probabilities": {"goal_beats_social_force": 0.875},
        }
    }

    issue4850._print_arm_summaries(arms)

    output = capsys.readouterr().out
    assert "Arms: multiplier_0.1" in output
    assert "Rank order: goal > social_force" in output
    assert "goal: 1.235" in output
    assert "P(goal beats social_force) = 0.875" in output
