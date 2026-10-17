"""Production integration contracts for the opt-in bounded contact rule."""

import numpy as np
import pysocialforce
import pytest
from pysocialforce.config import SimulatorConfig


def simulator(state, *, passes=3, walls=None):
    config = SimulatorConfig()
    config.scene_config.agent_radius = 0.28
    config.scene_config.enable_group = False
    # Assignment is parent-compatible: an ignored selector must fail numerically.
    config.pedestrian_contact_rule = "projection_v2"
    config.pedestrian_contact_passes = passes
    return pysocialforce.Simulator(np.array(state, dtype=float), obstacles=walls, config=config)


@pytest.mark.parametrize("passes", [3, 10])
def test_head_on_disc_endpoints_and_swept_motion_do_not_overlap(passes):
    sim = simulator([[0, 0, 1.3, 0, 10, 0, 0.5], [0.7, 0, -1.3, 0, -10, 0, 0.5]], passes=passes)
    old = sim.peds.pos().copy()
    sim.step()
    delta = sim.peds.pos()[0] - sim.peds.pos()[1]
    assert np.linalg.norm(delta) >= 0.56 - 1e-12, "contact selector was ignored"
    for alpha in np.linspace(0, 1, 31):
        position = old + alpha * (sim.peds.pos() - old)
        assert np.linalg.norm(position[0] - position[1]) >= 0.56 - 1e-12
    assert np.max(np.linalg.norm(sim.peds.vel(), axis=1)) <= 1.69 + 1e-12


def test_wall_projection_preserves_tangent_and_body_clearance():
    sim = simulator([[0, 0.4, 0.2, -1, 10, -10, 0.5]], walls=[(-10, 10, 0, 0)])
    sim.step(5)
    assert sim.peds.pos()[0, 1] >= 0.28 - 1e-12
    assert sim.peds.pos()[0, 0] > 0


def test_held_body_does_not_move_when_neighbor_contacts_it():
    sim = simulator([[0, 0, 0, 0, 0, 0, 0.5], [0.7, 0, -1.3, 0, -10, 0, 0.5]])
    sim.peds.contact_held_indices = {0}
    sim.step(4)
    np.testing.assert_array_equal(sim.peds.state[0, :4], np.zeros(4))
    assert np.linalg.norm(sim.peds.pos()[1] - sim.peds.pos()[0]) >= 0.56 - 1e-12


def test_capped_dense_component_cannot_stop_an_isolated_walker():
    state = [[i * 0.58, 0, (-1) ** i * 1.3, 0, (-1) ** i * 100, 0, 0.5] for i in range(8)]
    state.append([40, 40, 1.3, 0, 100, 40, 0.5])
    sim = simulator(state)
    sim.step(8)
    assert sim.peds.pos()[-1, 0] > 40.8
    assert 3 <= sim.contact_projection_passes <= 10
    xy = sim.peds.pos()[:-1]
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    assert distance[np.triu_indices(8, 1)].min() >= 0.56 - 1e-12


@pytest.mark.parametrize("passes", [0, 2, 11, True, 3.5])
def test_invalid_pass_budget_is_refused(passes):
    with pytest.raises(ValueError, match="contact passes must be an integer in 3..10"):
        simulator([[0, 0, 1, 0, 10, 0, 0.5]], passes=passes)


def test_inadmissible_start_is_refused_without_position_repair():
    sim = simulator([[0, 0, 1, 0, 10, 0, 0.5], [0.1, 0, -1, 0, -10, 0, 0.5]])
    before = sim.peds.state.copy()
    with pytest.raises(ValueError, match="contact requires nonoverlapping initial bodies"):
        sim.step()
    np.testing.assert_array_equal(sim.peds.state, before)


def test_config_roundtrip_and_scenario_binding_keep_contact_explicit():
    """Actual production construction must bind the opt-in solver, not only native tests."""
    from copy import deepcopy
    from dataclasses import replace

    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.sim.sim_config import SimulationSettings
    from robot_sf.training.scenario_loader import _apply_simulation_overrides
    from tests.sim.test_pedestrian_desired_speed import _build_simulator

    options = {"contact_rule": "projection_v2", "passes": 3}
    settings = SimulationSettings(
        pedestrian_physics=options,
        pedestrian_radius_m=0.28,
        obstacle_force_profile="body_edge_v3",
        pedestrian_seed=1001,
        groups=0.5,
        difficulty=0,
        ped_density_by_difficulty=[0.0],
        episode_step_limit=7,
        group_allocation_mode="exact_small_crowd_v1",
    )
    for copy in (replace(settings), deepcopy(settings), SimulationSettings(**settings.to_dict())):
        assert copy == settings
        assert copy.episode_step_limit == 7
        assert copy.group_allocation_mode == "exact_small_crowd_v1"
    sim = _build_simulator(settings)
    assert sim.pysf_sim.peds.velocity_projector.passes == 3
    assert sim.pysf_sim.config.obstacle_force_config.law_version == "body_edge_exponential_v3"
    assert "pedestrian_physics" not in SimulationSettings().to_dict()
    candidate = RobotSimulationConfig()
    _apply_simulation_overrides(candidate, {"pedestrian_physics": options})
    assert candidate.sim_config.pedestrian_physics == options


def test_captured_step_diagnostics_match_executed_contact_motion():
    """Post-cap traces must describe the actual contact-constrained update."""
    sim = simulator([[0, 0, 1.3, 0, 10, 0, 0.5], [0.7, 0, -1.3, 0, -10, 0, 0.5]])
    before = sim.peds.pos().copy()
    sim.peds.step(sim.compute_forces(), capture_diagnostics=True)
    receipt = sim.peds.last_step_diagnostics
    np.testing.assert_array_equal(receipt.applied_velocity, sim.peds.vel())
    np.testing.assert_allclose(sim.peds.pos() - before, 0.1 * receipt.position_velocity, atol=1e-15)


def test_touching_convoy_keeps_collective_and_tangential_motion():
    """Coupled normal resolution advances a convoy instead of halting its component."""
    sim = simulator(
        [
            [0, 0, 1.3, 0.2, 10, 0, 0.5],
            [0.560000001, 0, 0.8, 0.2, 10, 0, 0.5],
            [1.120000002, 0, 0.3, 0.2, 10, 0, 0.5],
        ]
    )
    sim.peds.max_speeds = np.full(3, 3.0)
    sim.peds.step(np.zeros((3, 2)))
    # Isotonic normal projection: all three x velocities share their mean .8.
    np.testing.assert_allclose(sim.peds.vel()[:, 0], 0.8, atol=2e-7)
    np.testing.assert_allclose(sim.peds.vel()[:, 1], 0.2, atol=1e-12)
    assert sim.contact_projection_fallback_count == 0


def test_coupled_resolution_does_not_apply_tangential_friction():
    """Coupled normal exclusion cannot erase different tangential velocities."""
    sim = simulator(
        [
            [0, 0, 1.3, 0.2, 10, 0, 0.5],
            [0.560000001, 0, 0.8, -0.3, 10, 0, 0.5],
            [1.120000002, 0, 0.3, 0.1, 10, 0, 0.5],
        ]
    )
    sim.peds.max_speeds = np.full(3, 3.0)
    sim.peds.step(np.zeros((3, 2)))
    np.testing.assert_allclose(sim.peds.vel()[:, 0], 0.8, atol=2e-7)
    np.testing.assert_array_equal(sim.peds.vel()[:, 1], [0.2, -0.3, 0.1])
    assert sim.contact_projection_fallback_count == 0


def test_coupled_normal_projection_matches_independent_convex_oracle():
    """Basis changes must keep the nearest feasible velocity, not merely a safe halt."""
    from scipy.optimize import LinearConstraint, minimize

    xy = np.array([[0, 0], [0.58, 0], [0.29, 0.58 * np.sqrt(3) / 2], [0.87, 0.58 * np.sqrt(3) / 2]])
    first, second = np.triu_indices(4, 1)
    delta = xy[first] - xy[second]
    distance = np.linalg.norm(delta, axis=1)
    normals = delta / distance[:, None]
    constraints = np.zeros((len(first), 8))
    for k, (i, j) in enumerate(zip(first, second, strict=True)):
        constraints[k, 2 * i : 2 * i + 2] = normals[k]
        constraints[k, 2 * j : 2 * j + 2] = -normals[k]
    lower = -(distance - 0.56 - 1e-8) / 0.1
    rng = np.random.default_rng(1001)
    for _ in range(20):
        velocity = rng.normal(size=(4, 2))
        velocity *= 1.3 / np.linalg.norm(velocity, axis=1)[:, None]
        target = velocity.ravel()
        # All four preferred speeds are 1.3: the global projection's norm is
        # <=2.6, so the independent 3 m/s individual caps cannot bind here.
        result = minimize(
            lambda v: 0.5 * np.dot(v - target, v - target),
            np.zeros(8),
            jac=lambda v: v - target,
            constraints=[LinearConstraint(constraints, lower, np.inf)],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 100},
        )
        assert result.success
        state = np.column_stack([xy, velocity, xy + 10, np.full(4, 0.5)])
        sim = simulator(state)
        sim.peds.max_speeds = np.full(4, 3.0)
        sim.peds.step(np.zeros((4, 2)))
        np.testing.assert_allclose(sim.peds.vel().ravel(), result.x, atol=1e-7)
        assert sim.contact_projection_fallback_count == 0


def test_wall_contact_transmits_pressure_into_free_tangential_motion():
    """A stationary wall cannot halt an approaching pair's sliding motion."""
    state = [[-0.28, 0, 1, 0.2, 10, 0, 0.5], [-0.8, -0.25, 1.3, 0.2, 10, 0, 0.5]]
    state.extend([[-10 - 1.5 * i, 10, 1.3, 0, 0, 10, 0.5] for i in range(6)])
    sim = simulator(state, walls=[(0, 0, -20, 20)])
    sim.peds.max_speeds = np.full(8, 3.0)
    sim.peds.step(np.zeros((8, 2)))
    assert sim.peds.vel()[0, 1] > 0.5, "wall reaction erased free sliding motion"
    assert sim.peds.vel()[1, 0] > 0.4
    assert sim.peds.pos()[0, 0] <= -0.28 + 1e-10
    assert np.linalg.norm(sim.peds.pos()[0] - sim.peds.pos()[1]) >= 0.56 - 1e-10
    assert sim.contact_projection_fallback_count == 0
