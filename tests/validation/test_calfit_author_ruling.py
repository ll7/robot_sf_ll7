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


def test_initial_overlap_is_hard_failure_even_if_later_samples_are_clear():
    import numpy as np

    from robot_sf.research.pedestrian_validation import pair_overlap

    positions = np.zeros((2, 60, 2))
    positions[:, :, 0] = np.arange(60)
    positions[0, 1, 0] = 0.3
    measured = pair_overlap(positions, 0.25)
    assert measured["all"]["initial_overlapping_pairs"] == 1
    assert measured["all"]["below_2r_count"] == 0
    result = suite.acceptance_gate(
        [
            {
                "case": "V3",
                "variant": "1.0",
                "seed": 1001,
                "radius_m": 0.25,
                "specific_flow_persons_m_s": 1.9,
                "wall_penetration_m": 0.0,
                "pair_overlap": measured,
            }
        ]
    )
    assert result["exit_code"] == 3


def test_original_shoulder_rotation_case_is_a_limitation_not_a_failed_gate():
    original = {
        "case": "V2",
        "variant": "0.9",
        "seed": 1001,
        "radius_m": 0.28,
        "aperture_shoulder_ratio": 0.9,
        "speed_drop_m_s": None,
        "passed": False,
        "wall_penetration_m": 0.2,
        "pair_overlap": {"all": {"below_2r_count": 0}},
    }
    result = suite.acceptance_gate([row(), original])
    assert result["exit_code"] == 0
    assert (
        result["excluded_measurements"][0]["reason"]
        == "not reproducible with rigid discs (shoulder rotation)"
    )
    assert len(result["checks"]) == 1


@pytest.mark.parametrize(
    "case,variant,field,target,bounds",
    [
        ("V5", "diagnostic", "lateral_cm_to_edge_m", 0.5, [0.4, 0.6]),
    ],
)
def test_unreported_distribution_sd_uses_author_twenty_percent_fallback(
    case, variant, field, target, bounds
):
    gate = suite.acceptance_gate([row(case, variant, **{field: target})])
    check = gate["checks"][0]
    assert gate["exit_code"] == 0
    assert check["tolerance_range"] == pytest.approx(bounds)
    assert "no reported SD" in check["rule"]
    assert "20%" in check["rule"]
