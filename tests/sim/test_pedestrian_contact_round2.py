"""Independent witnesses for the defects reported at ead48d9c (dev seeds only)."""

import numpy as np
import pytest
from pysocialforce import Simulator
from pysocialforce.config import SimulatorConfig
from pysocialforce.contact import bounded_wall_force, closest_point, project_step

from robot_sf.research.pedestrian_validation import turning_onset
from robot_sf.sim.sim_config import SimulationSettings
from tests.sim.test_pedestrian_contact import config, free_forces


def test_corner_vertex_does_not_hide_segment_constraint():
    """The exact dev1005 corner must converge rather than repeat 4096 passes."""
    walls = np.array([[21, 28, 19, 28, 0, -1], [19, 28, 19, 12, -1, 0]], float)
    start = np.array([[18.73257189, 27.77176083]])
    result, passes, converged = project_step(start, start, walls, 0.4, False, True, 4096)
    assert converged, f"corner solver exhausted {passes} passes"
    for wall in walls:
        assert np.linalg.norm(result[0] - closest_point(result[0], wall)) >= 0.4


def test_iteration_cap_records_fallback_and_continues(monkeypatch):
    """A real cap exhaustion must not abort the environment's episode."""
    from pysocialforce import contact

    monkeypatch.setattr(contact, "MAX_PASSES", 1)
    state = np.array([[x, 0, 0, 0, 10, 0, 0.5] for x in [0, 0.1, 0.2]], float)
    sim = Simulator(state, config=config(), make_forces=free_forces)
    sim.step()
    assert sim.contact_projection_fallback_count == 1
    assert np.isfinite(sim.peds.state).all()
    assert (
        min(
            np.linalg.norm(sim.peds.pos()[i] - sim.peds.pos()[j])
            for i in range(3)
            for j in range(i)
        )
        >= 0.56
    )


@pytest.mark.parametrize("wall", [False, True])
def test_position_repair_does_not_inject_separation_velocity(wall):
    """Stationary overlapping bodies must stay stationary after geometric repair."""
    state = np.array([[0, 0, 0, 0, 10, 0, 0.5], [0.05, 0, 0, 0, 10, 0, 0.5]], float)
    sim = Simulator(state, config=config(wall=wall), make_forces=free_forces)
    sim.peds.max_speeds[:] = 2.0
    sim.step()
    np.testing.assert_array_equal(sim.peds.vel(), np.zeros((2, 2)))
    assert np.max(np.linalg.norm(sim.peds.vel(), axis=1)) <= 2.0


def test_wall_force_is_continuous_across_aperture_medial_axis():
    """An epsilon centreline displacement cannot reverse a finite lateral force."""
    walls = np.array([[-3, 0.305, 3, 0.305, 0, -1], [-3, -0.305, 3, -0.305, 0, 1]], float)
    p = np.array([[0, -1e-6], [0, 0], [0, 1e-6]], float)
    f = bounded_wall_force(p, walls, 0.28, 3.0, 0.04, 0.20)
    assert np.linalg.norm(f[2] - f[0]) < 0.001
    np.testing.assert_allclose(f[1], [0, 0], atol=1e-12)
    # Check the complete optional force too: restoring the singular near field
    # must not undo the finite-normal continuity correction.
    cfg = config(wall=True)
    cfg.obstacle_force_config.factor = 0.003
    cfg.obstacle_force_config.threshold = 0.375
    cfg.obstacle_force_config.sigma = 0.0
    state = np.column_stack([p, np.zeros((3, 2)), np.tile([10, 0, 0.5], (3, 1))])
    sim = Simulator(state, obstacles=[(-3, 3, 0.305, 0.305), (-3, 3, -0.305, -0.305)], config=cfg)
    full = sim.forces[2]()
    assert np.linalg.norm(full[2] - full[0]) < 0.001


def test_contact_removes_closing_velocity_after_overrelaxed_position_repair():
    """A normal head-on impact cannot keep its closing velocity after push-out."""
    state = np.array([[-0.35, 0, 1, 0, 10, 0, 0.5], [0.35, 0, -1, 0, -10, 0, 0.5]], float)
    sim = Simulator(state, config=config(), make_forces=free_forces)
    sim.step()
    np.testing.assert_allclose(sim.peds.vel(), np.zeros((2, 2)), atol=1e-12)


@pytest.mark.parametrize("integration_path", ["native", "robot_benchmark"])
def test_normal_velocity_transfer_reapplies_each_body_speed_cap(integration_path):
    """A normal impulse can increase one body's speed despite dissipating pair energy."""
    state = np.array([[-0.3, 0, 0, 2, 10, 0, 0.5], [0.3, 0, -2, 0, -10, 0, 0.5]], float)
    np.random.seed(1001)
    sim = Simulator(state, config=config(), make_forces=free_forces)
    sim.peds.max_speeds[:] = 2.0
    if integration_path == "native":
        sim.step()
    else:
        from types import SimpleNamespace

        from robot_sf.sim.simulator import Simulator as RobotSimulator

        wrapper = RobotSimulator.__new__(RobotSimulator)
        wrapper.config = SimulationSettings()
        wrapper.pysf_sim = sim
        wrapper.pedestrian_model = "social_force"
        wrapper.pysf_state = SimpleNamespace(ped_velocities=sim.peds.vel())
        wrapper._step_pedestrians(np.zeros((2, 2)), [])
    assert np.max(np.linalg.norm(sim.peds.vel(), axis=1)) <= 2.0 + 1e-12
    assert np.linalg.norm(sim.peds.pos()[0] - sim.peds.pos()[1]) >= 0.56


@pytest.mark.parametrize("amplitude", [3.0, 6.0, 9.0])
def test_gap_wall_response_damps_instead_of_entering_a_lateral_cycle(amplitude):
    """Actual goal/wall forces must damp an admissible near-wall disturbance."""
    from scripts.validation.pedestrian_validation_10074 import protocol_simulate

    cfg = config(wall=True)
    cfg.obstacle_force_config.wall_contact_amplitude_m_s2 = amplitude
    state = np.array([[0, 0.024998, 1.29, 0, 1000, 0, 0.5]], float)
    positions, _, _ = protocol_simulate(
        state,
        [(-100, 0.305, 1000, 0.305), (-100, -0.305, 1000, -0.305)],
        cfg,
        300,
        speed_cap_m_s=2.0,
        desired_distribution=(1.29, 0.0),
        desired_seed=1001,
    )
    assert np.max(np.abs(positions[-100:, 0, 1])) < 1e-5, "persistent gap chatter"


@pytest.mark.parametrize("pair,wall", [(True, False), (False, True), (True, True)])
def test_unset_radius_is_not_changed_by_law_selection(pair, wall):
    """Selectors preserve the backend radius when no physical radius is supplied."""
    from tests.sim.test_pedestrian_desired_speed import _build_simulator

    np.random.seed(1001)
    settings = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.04],
        population_size=8,
        pedestrian_contact_rule="projection_v1" if pair else None,
        pedestrian_wall_rule="bounded_edge_v1" if wall else None,
    )
    sim = _build_simulator(settings)
    assert sim.pysf_sim.peds.agent_radius == SimulatorConfig().scene_config.agent_radius


def test_numerical_yaw_noise_does_not_define_turning_onset():
    """A nanometre lateral perturbation is not a physical avoidance manoeuvre."""
    t = np.arange(301) * 0.1
    straight = np.column_stack([t, np.zeros(len(t))])
    noisy = straight.copy()
    noisy[:, 1] = 1e-8 * np.sin(t)
    q = np.column_stack([30 - t, np.zeros(len(t))])
    assert turning_onset(noisy, q, t, [straight] * 5)["onset_m"] is None


def test_turning_onset_changes_with_actual_manoeuvre_location():
    """Identical straight baselines must not saturate every turn at the window edge."""
    t = np.arange(601) * 0.1
    baseline = np.column_stack([t, np.zeros(len(t))])
    q = np.column_stack([60 - t, np.zeros(len(t))])
    values = []
    for location in [19.0, 24.0]:
        path = baseline.copy()
        path[:, 1] = 0.5 * (1 + np.tanh((t - location) / 0.5))
        values.append(turning_onset(path, q, t, [baseline] * 5, analysis_window_m=25.0)["onset_m"])
    assert all(v is not None for v in values)
    assert values[0] - values[1] == pytest.approx(5.0, abs=0.2)
    assert values[0] > 3.0
