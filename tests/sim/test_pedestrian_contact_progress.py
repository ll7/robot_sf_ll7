"""Live replays of the four independent dense-crowd progress refutations."""

import pytest

from scripts.validation import pedestrian_validation_10074 as suite


@pytest.mark.slow
@pytest.mark.parametrize("case,passes", [("V3", 3), ("V3", 10), ("V4", 3), ("V4", 10)])
def test_actual_validation_crowd_makes_progress(case, passes):
    """An admissible crowd must cross, rather than be held for all 160 seconds."""
    options = {
        "calfit": True,
        "speed_tier": "literature",
        "execution_cap_m_s": 3.0,
        "wall_profile": "body_edge_v3",
        "pedestrian_contact_rule": "projection_v2",
        "pedestrian_contact_passes": passes,
    }
    row = suite.run_task((case, 1001, "1.0" if case == "V3" else "2.4", 0.28, "radius", options))
    assert len(row["_positions"]) - 1 == 1600
    assert row["crossed"] > 0, (
        f"{case}, {passes} passes: zero crossings after 160 s; {row['step_runtime']}"
    )
    expected = 60 if case == "V3" else 350
    assert row["crossed"] >= 0.9 * expected, "contact lost comparable bottleneck progress"
    assert row["pair_overlap"]["all"]["minimum_centre_distance_m"] >= 0.56 - 1e-10
    assert row["wall_penetration_m"] <= 1e-10
    receipt = row["step_runtime"][0]
    assert receipt["fallback_rate"] < 0.01
    assert receipt["unresolved_count"] == receipt["over_cap_samples"] == 0
