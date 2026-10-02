"""Author-ruling regressions at real gate and task-grid boundaries; no stepping."""

import pytest

from scripts.validation import pedestrian_validation_10074 as suite


def row(case="V1", variant="native", **fields):
    return {
        "case": case,
        "variant": variant,
        "seed": 1001,
        "radius_m": 0.28,
        "wall_penetration_m": 0.0,
        "pair_overlap": {"all": {"below_2r_count": 0}},
        "fitted_desired_speed_m_s": 1.29,
        "fitted_tau_s": 0.54,
        **fields,
    }


def test_target_valued_v1_can_pass_author_engineering_gate():
    result = suite.acceptance_gate([row()])
    assert result["exit_code"] == 0


def test_physics_is_a_hard_failure_even_with_a_missing_estimate():
    result = suite.acceptance_gate([row(fitted_desired_speed_m_s=None, wall_penetration_m=0.001)])
    assert result["exit_code"] == 3


def test_calfit_grid_uses_three_feasible_widths():
    config = suite.load_config(suite.DEFAULT_CONFIG)
    config["seeds"] = [1001]
    tasks = suite.protocol_tasks(config, 0.28, "radius", {"calfit": True})
    apertures = [t for t in tasks if t[0] == "V2"]
    assert len(apertures) == 3
    assert [float(t[2]) for t in apertures] == pytest.approx([0.61, 0.788, 0.966])


def test_every_gate_can_pass_when_test_fixture_supplies_missing_spreads():
    from scripts.validation.calfit_preflight_10074 import ideal_gate_records

    config = suite.load_config(suite.DEFAULT_CONFIG)
    config["seeds"] = [1001]
    # Synthetic spreads demonstrate reachability, not literature claims.
    config["V5"]["published_sd_m"] = 0.1
    config["V6"]["published_onset_sd_m"] = [0.1, 0.1, 0.1]
    result = suite.acceptance_gate(ideal_gate_records(), config=config, require_complete=True)
    assert result["exit_code"] == 0
    assert len(result["checks"]) == 15
    assert all(c["status"] == "PASS" for c in result["checks"])
    assert all(c["difference"] == pytest.approx(0) for c in result["checks"])


@pytest.mark.parametrize("estimate, status", [(1.1, "PASS"), (1.48, "PASS"), (1.481, "FAIL")])
def test_v1_author_tolerance_boundary_and_residual(estimate, status):
    result = suite.acceptance_gate([row(fitted_desired_speed_m_s=estimate)])
    check = result["checks"][0]
    assert check["status"] == status
    assert check["difference"] == pytest.approx(estimate - 1.29)
    assert check["tolerance_range"] == [1.10, 1.48]
