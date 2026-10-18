"""Calibrated outcome flags protect null cells without hiding absolute failures."""

import random

import pytest

from robot_sf.analysis_workbench.audit_detectors import DetectorError, detect


def _rows(counts, size=30):
    return [
        {
            "episode_id": f"p{planner}-{index}",
            "planner_id": f"p{planner}",
            "scenario_id": "calibration",
            "seed": 100000 + index,
            "outcome": {"label": "collision" if index < count else "success"},
        }
        for planner, count in enumerate(counts)
        for index in range(size)
    ]


def _relative_signal(rows, planner=0):
    return detect(
        "outcome_incidence",
        rows[planner * (len(rows) // len({row["planner_id"] for row in rows}))],
        cohort=rows,
        config={"absolute_failure_rate_floor": 1.0},
    )


def test_null_scenario_family_false_positive_rate_is_at_most_five_percent():
    """Independent Bernoulli(0.2) cells expose the old uncalibrated excess flag."""
    rng = random.Random(87031)
    flagged_families = 0
    for _ in range(100):
        rows = _rows([sum(rng.random() < 0.2 for _ in range(30)) for _ in range(5)])
        signals = [_relative_signal(rows, planner) for planner in range(5)]
        assert all(signal.status in {"clear", "flagged"} for signal in signals)
        flagged_families += any(signal.status == "flagged" for signal in signals)
    assert flagged_families <= 5, f"{flagged_families}/100 null scenario families flagged"


def test_small_exact_table_uses_external_controls_and_family_correction():
    """All four adverse observations fall in one of three equally sized cells."""
    signal = _relative_signal(_rows([4, 0, 0], size=4))
    assert signal.status == "flagged"
    # Conditional on four collisions among twelve rows, only one of C(12,4)=495
    # allocations puts all four in the target: a one-sided Fisher p of 1/495.
    assert signal.measured["incidence_p_values"]["collision"] == pytest.approx(1 / 495)
    assert signal.measured["peer_outcome_counts"]["collision"] == 0
    assert signal.measured["peer_sample_size"] == 8
    assert signal.threshold["comparison_count"] == 12  # four outcomes x three planners
    assert signal.threshold["bonferroni_alpha"] == pytest.approx(0.05 / 12)


def test_absolute_floor_remains_independent_of_calibrated_comparison():
    rows = _rows([1])
    signal = detect("outcome_incidence", rows[0], cohort=rows)
    assert signal.status == "flagged"
    assert signal.measured["absolute_adverse_outcome"] is True
    assert signal.measured["excess_incidence"] is False


@pytest.mark.parametrize("alpha", [0, -0.01, 0.051, True])
def test_invalid_family_alpha_is_rejected(alpha):
    rows = _rows([6, 6])
    signal = detect("outcome_incidence", rows[0], cohort=rows, config={"family_alpha": alpha})
    assert signal.status == "error"


def test_nonfinite_configuration_keeps_existing_strict_json_rejection():
    rows = _rows([6, 6])
    with pytest.raises(DetectorError, match="not strict JSON"):
        detect("outcome_incidence", rows[0], cohort=rows, config={"family_alpha": float("nan")})


def test_retired_fixed_excess_threshold_is_not_silently_ignored():
    rows = _rows([6, 6])
    signal = detect(
        "outcome_incidence", rows[0], cohort=rows, config={"excess_rate_threshold": 0.1}
    )
    assert signal.status == "error"
