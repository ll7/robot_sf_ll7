"""Parent-compatible production task probe of desired draw and independent cap."""

import pytest

from scripts.validation import pedestrian_validation_10074 as suite


@pytest.mark.parametrize("quantity", ["desired_draw", "execution_cap"])
def test_calfit_normal_draw_survives_a_lower_execution_cap(quantity):
    row = suite.run_task(
        (
            "V1",
            1001,
            "native",
            0.25,
            "radius",
            {"calfit": True, "speed_tier": "literature", "execution_cap_m_s": 0.2},
        )
    )
    # First positive N(1.29, .19) draw of dev1001, independently specified.
    if quantity == "desired_draw":
        assert row["desired_speeds_m_s"] == pytest.approx([1.467141262682885])
        assert row["desired_speeds_m_s"][0] > 0.2
    else:
        assert row["_speeds"].max() <= 0.2 + 1e-12
    assert row["_positions"][-1, 0, 0] > 0
