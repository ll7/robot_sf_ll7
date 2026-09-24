"""Scenario selection test (T008 / FR-005, FR-006).

Asserts:
- Valid scenario name resolves and produces at least one episode (future state)
- Invalid scenario name raises ValueError listing available scenarios
Currently episodes assertion will FAIL (TDD) until implementation populates summaries.
"""

from __future__ import annotations

import importlib
import re

import pytest


def _mod():
    """Import and return the classic interactions Pygame demo module."""
    return importlib.import_module("examples.classic_interactions_pygame")


def test_valid_scenario_selection():
    """Verify the first matrix scenario produces at least one episode.

    Loads the classic scenario matrix, temporarily sets SCENARIO_NAME to the first
    scenario with DRY_RUN off, runs run_demo, and asserts a non-empty episode list.
    """
    mod = _mod()
    # Identify first scenario name via loader directly
    from robot_sf.benchmark.classic_interactions_loader import load_classic_matrix

    matrix = load_classic_matrix(str(mod.SCENARIO_MATRIX_PATH))
    first_name = matrix[0]["name"]

    # Temporarily set constant to first scenario
    original_name = mod.SCENARIO_NAME
    original_dry = mod.DRY_RUN
    mod.SCENARIO_NAME = first_name  # type: ignore
    mod.DRY_RUN = False  # type: ignore
    try:
        episodes = mod.run_demo()
    finally:
        mod.SCENARIO_NAME = original_name  # type: ignore
        mod.DRY_RUN = original_dry  # type: ignore
    assert episodes, "Expected episodes for valid scenario (TDD failing until implementation)."


def test_invalid_scenario_name_lists_available():
    """Verify an unknown scenario name raises ValueError listing available scenarios.

    Temporarily sets SCENARIO_NAME to a sentinel unknown name with DRY_RUN off and
    asserts run_demo raises ValueError whose message contains 'Available:'.
    """
    mod = _mod()
    original_name = mod.SCENARIO_NAME
    original_dry = mod.DRY_RUN
    mod.SCENARIO_NAME = "__not_a_real_scenario__"  # type: ignore
    mod.DRY_RUN = False  # type: ignore
    try:
        with pytest.raises(ValueError) as excinfo:
            mod.run_demo()
    finally:
        mod.SCENARIO_NAME = original_name  # type: ignore
        mod.DRY_RUN = original_dry  # type: ignore

    msg = str(excinfo.value)
    # Expect the message to contain 'Available:' list
    assert re.search(r"Available:", msg), f"Expected 'Available:' in error message, got: {msg}"
