"""Behavioral witnesses for opt-in wall variants in the force-query adapter."""

from math import exp

import numpy as np
import pytest
from pysocialforce import Simulator

from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


def _wall_wrapper(law, *, obstacles=()):
    """Build an obstacle-only scene without stepping a simulator or using seeds."""
    simulation = Simulator(state=np.empty((0, 7)), obstacles=list(obstacles))
    simulation.config.obstacle_force_config.law_version = law
    simulation.config.obstacle_force_config.factor = 2.5
    simulation.peds.agent_radius = 0.35
    return FastPysfWrapper(simulation)


@pytest.mark.parametrize(
    "law",
    ["body_edge_exponential_v3_physical_margin", "body_edge_exponential_v3_contact_stiff"],
)
@pytest.mark.parametrize("factor", [0.0, 2.5])
def test_wrapper_contact_metadata_binds_active_parameters(law, factor):
    """The wrapper reports physical margin and scaled contact terms, even when disabled."""
    wrapper = _wall_wrapper(law)
    wrapper.sim.config.obstacle_force_config.factor = factor

    metadata = wrapper.obstacle_force_law_metadata()
    expected = {
        "factor": factor,
        "agent_radius": 0.35,
        "distance_floor": 1e-5,
        "amplitude_unscaled": 0.3,
        "amplitude_m_s2": 0.3 * factor,
        "decay_m": 0.04,
        "range_m": 0.2,
        "contact_radius_margin_m": 0.05,
    }
    if law == "body_edge_exponential_v3_contact_stiff":
        expected.update(
            {
                "contact_bias_unscaled": 0.6,
                "contact_bias_m_s2": 0.6 * factor,
                "contact_stiffness_unscaled_per_m": 4.0,
                "contact_stiffness_m_s2_per_m": 4.0 * factor,
            }
        )
    assert metadata["parameters"] == expected
    assert metadata["geometry_convention"] == "nearest_finite_segment_surface"
    assert metadata["radius_convention"] == "physical_body_edge_clearance"
    assert metadata["law_version"] == law
    assert metadata["enabled"] is (factor != 0.0)
    assert metadata["applied"] is False


@pytest.mark.parametrize(
    ("law", "near_x", "clearance", "contact_term"),
    [
        ("body_edge_exponential_v3", 0.4, 0.05, 0.0),
        ("body_edge_exponential_v3_physical_margin", 0.4, 0.0, 0.0),
        ("body_edge_exponential_v3_contact_stiff", 0.38, -0.02, 0.68),
    ],
)
def test_scalar_wall_query_uses_only_nearest_finite_surface(law, near_x, clearance, contact_term):
    """An opposite wall must not be summed into the nearest-surface law."""
    wrapper = _wall_wrapper(
        law,
        obstacles=[(-0.42, -0.42, -1.0, 1.0), (near_x, near_x, -1.0, 1.0)],
    )
    # Independent values from the declared law: A=0.3, B=0.04, range=0.2,
    # factor=2.5; the stiff contact term is 0.6 + 4 * 0.02.
    strength = 2.5 * (0.3 * (exp(-max(0.0, clearance) / 0.04) - exp(-5.0)) + contact_term)

    force = wrapper.get_forces_at([0.0, 0.0])

    np.testing.assert_allclose(force, [-strength, 0.0], rtol=1e-12, atol=1e-12)
    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is False
    assert diagnostics["obstacle_force_law"]["applied"] is True


def test_scalar_wall_query_with_no_finite_surface_stays_unapplied(monkeypatch):
    """Nonfinite geometry must yield zero without claiming a wall force was applied."""
    wrapper = _wall_wrapper("body_edge_exponential_v3")
    monkeypatch.setattr(
        wrapper.sim,
        "get_raw_obstacles",
        lambda: np.array([[np.nan, np.nan, np.nan, np.nan, 1.0, 0.0]]),
    )

    np.testing.assert_array_equal(wrapper.get_forces_at([0.0, 0.0]), [0.0, 0.0])

    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is False
    assert diagnostics["obstacle_force_law"]["applied"] is False


@pytest.mark.parametrize("bad_coordinate", ["not-a-coordinate", object()])
def test_scalar_wall_query_records_malformed_geometry_fallback(monkeypatch, bad_coordinate):
    """Numeric conversion failure must be visible, not an unmarked zero contribution."""
    wrapper = _wall_wrapper("body_edge_exponential_v3")
    monkeypatch.setattr(
        wrapper.sim,
        "get_raw_obstacles",
        lambda: [[bad_coordinate, 0.0, 0.4, 1.0, -1.0, 0.0]],
    )

    np.testing.assert_array_equal(wrapper.get_forces_at([0.0, 0.0]), [0.0, 0.0])

    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is True
    assert diagnostics["fallback_reason"] == "obstacle_force_dropped"
    assert diagnostics["fallback_reasons"] == {"obstacle_force_dropped": 1}
    assert diagnostics["obstacle_force_law"]["applied"] is False
