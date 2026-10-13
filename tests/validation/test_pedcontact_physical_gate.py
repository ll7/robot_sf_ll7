"""Fail-closed physical counters on an otherwise passing source bank."""

from copy import deepcopy

import pytest

from robot_sf.research.pedestrian_acceptance import engineering_gate
from scripts.validation.calfit_preflight_10074 import ideal_gate_records


@pytest.mark.parametrize(
    "field", ["unresolved_count", "over_cap_samples", "wall_penetration_ped_steps"]
)
@pytest.mark.parametrize("value", [1, float("nan"), float("inf"), -1, True])
def test_recorded_physical_violation_cannot_pass_source_bank(field, value):
    rows = deepcopy(ideal_gate_records())
    assert engineering_gate(rows)["exit_code"] == 0
    if field == "wall_penetration_ped_steps":
        rows[0][field] = value
    else:
        rows[0]["step_runtime"] = [{field: value}]
    gate = engineering_gate(rows)
    assert gate["exit_code"] != 0
    assert gate["physical_violations"] or gate["measurement_missing"]


def test_solver_fallback_is_classified_as_diagnostic_evidence():
    rows = ideal_gate_records()
    rows[0]["step_runtime"] = [{"fallback_count": 1, "unresolved_count": 0, "over_cap_samples": 0}]
    gate = engineering_gate(rows)
    assert gate["exit_code"] != 0
    assert gate["solver_fallbacks"]
    assert gate["fallback_disposition"] == "diagnostic; qualification requires explicit disposition"
