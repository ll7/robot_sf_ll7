"""Actual protocol input controls; no simulated episodes or production test seams."""

import numpy as np
import pytest

from scripts.validation import pedestrian_validation_10074 as suite


def test_overlapping_input_is_refused_before_simulator_construction(monkeypatch):
    config = suite.reused.candidate_config(suite.CANDIDATE, 1.3)
    config.scene_config.agent_radius = 0.25
    state = np.array([[0, 0, 0, 0, 10, 0, 0.5], [0.4, 0, 0, 0, 10, 0, 0.5]])

    def constructor(*args, **kwargs):
        raise AssertionError("simulator reached an inadmissible initial state")

    monkeypatch.setattr(suite.pysocialforce, "Simulator", constructor)
    with pytest.raises(ValueError, match="inadmissible initial state"):
        suite.protocol_simulate(state, [], config, 1)


@pytest.mark.parametrize("case,radius,n,spacing", [("V3", 0.25, 60, 0.52), ("V4", 0.30, 350, 0.62)])
def test_actual_holding_input_has_required_spacing(monkeypatch, case, radius, n, spacing):
    def check_input(state, segments, config, steps, **kwargs):
        assert len(state) == n
        d = np.linalg.norm(state[:, None, :2] - state[None, :, :2], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= spacing, f"{case} starts with minimum spacing {d.min():.6f} < {spacing}"
        return np.repeat(state[None, :, :2], 3, axis=0), np.zeros((2, n)), np.ones(n)

    monkeypatch.setattr(suite, "protocol_simulate", check_input)
    row = suite.run_task((case, 1001, ".8" if case == "V3" else "2.4", radius, "radius"))
    assert row["persons"] == n


def test_admission_checks_body_to_wall_clearance_without_stepping():
    from robot_sf.research.pedestrian_initial_state import initial_admissibility

    state = np.array([[0.1, 0, 0, 0, 10, 0, 0.5]])
    receipt = initial_admissibility(state, [(0, -1, 0, 1)], 0.25)
    assert receipt["initial_wall_overlap_count"] == 1
    assert receipt["minimum_wall_centre_clearance_m"] == pytest.approx(0.1)
    assert receipt["admissible"] is False


@pytest.mark.parametrize("radius,spacing", [(0.27, 0.56), (0.28, 0.58), (0.30, 0.62)])
def test_holding_population_survives_required_area_expansion(monkeypatch, radius, spacing):
    def check_input(state, segments, config, steps, **kwargs):
        d = np.linalg.norm(state[:, None, :2] - state[None, :, :2], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert len(state) == 60
        assert d.min() >= spacing
        return np.repeat(state[None, :, :2], 3, axis=0), np.zeros((2, 60)), np.ones(60)

    monkeypatch.setattr(suite, "protocol_simulate", check_input)
    row = suite.run_task(("V3", 1001, ".8", radius, "radius"))
    assert row["initial_admissibility"]["admissible"]
    assert row["holding_layout"]["new_density_persons_m2"] <= 3.3 + 1e-12
