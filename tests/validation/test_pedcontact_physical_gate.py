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


@pytest.mark.parametrize("field", ["unresolved_count", "over_cap_samples"])
def test_collector_refuses_recorded_runtime_violation(tmp_path, monkeypatch, field):
    from scripts.validation import pedcontact_10101 as collector

    cfg = collector.suite.load_config(collector.suite.DEFAULT_CONFIG)
    cfg["seeds"] = [1001]
    monkeypatch.setattr(collector.suite, "load_config", lambda _: cfg)
    monkeypatch.setattr(collector.search, "verify_run", lambda _: None)
    bank = [row for row in ideal_gate_records() if row["seed"] == 1001]
    for row in bank:
        row.update(
            estimator_status="documented_equivalent",
            radius_m=0.28,
            speed_m_s=1.29,
            passed=True,
            wall_penetration_ped_steps=0,
            step_runtime=[
                {
                    "steps": 1,
                    "step_time_s": 0.01,
                    "maximum_projection_passes": 1,
                    "fallback_count": 0,
                    "unresolved_count": 0,
                    "over_cap_samples": 0,
                    "maximum_speed_m_s": 1.0,
                }
            ],
        )
    for row in bank:
        row["pair_overlap"] = {
            split: {
                "pair_steps": 0,
                "below_2r_count": 0,
                "below_0_45_count": 0,
                "minimum_centre_distance_m": None,
            }
            for split in ("group", "non_group", "all")
        }
        if row["case"] == "V4":
            row.update(width_m=float(row["variant"]), all_crossed=False)
    bank[0]["step_runtime"][0][field] = 1
    for arm in ("off", "contact", "wall", "on"):
        directory = tmp_path / arm / "1001"
        directory.mkdir(parents=True)
        for index in range(len(bank)):
            (directory / f"case_{index}.json").touch()

    def read(path):
        if path.name == "identity.json":
            return {
                "seeds": [1001],
                "candidate": collector.default_point(path.parent.parent.name),
                "source_sha": "a" * 40,
            }
        return deepcopy(bank[int(path.stem.removeprefix("case_"))])

    monkeypatch.setattr(collector.search, "read_json", read)
    result = collector.collect(tmp_path, tmp_path / "comparison.json")
    assert result["arms"]["on"][field] == 1
    assert result["fit_admitted"] is False
