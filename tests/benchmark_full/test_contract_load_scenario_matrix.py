"""Contract test T005 for `load_scenario_matrix`.

Expectation (from contracts):
  - Function raises FileNotFoundError when the YAML path does not exist.

Initial TDD state: Function is not yet implemented and will raise
NotImplementedError causing this test to FAIL. After implementation (T022)
the FileNotFoundError branch should satisfy this test.
"""

from __future__ import annotations

import uuid

import pytest

from robot_sf.benchmark.full_classic.planning import load_scenario_matrix


def test_load_scenario_matrix_invalid_path():
    """Invalid path should raise FileNotFoundError (contract)."""
    missing_path = f"/tmp/does_not_exist_{uuid.uuid4().hex}.yaml"
    with pytest.raises(FileNotFoundError):
        load_scenario_matrix(missing_path)
