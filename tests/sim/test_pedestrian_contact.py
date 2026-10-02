"""Physical witnesses for opt-in PEDCONTACT, using real simulator integration."""

from types import SimpleNamespace

import numpy as np
import pytest
from pysocialforce import Simulator
from pysocialforce.config import SimulatorConfig

from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import Simulator as RobotSimulator


def config(*, contact=True, wall=False, dt=0.1):
    """Request optional laws through the public simulator configuration."""
    cfg = SimulatorConfig()
    cfg.scene_config.agent_radius = 0.28
    cfg.scene_config.dt_secs = dt
    cfg.scene_config.enable_group = False
    if contact:
        cfg.pedestrian_contact_rule = "projection_v1"
    if wall:
        cfg.obstacle_force_config.wall_contact_rule = "bounded_edge_v1"
    return cfg


def free_forces(sim, cfg):
    """Use the existing force factory interface to isolate actual contact handling."""
    return [lambda: np.zeros_like(sim.peds.pos())]


def test_overlapping_bodies_separate():
    """Two stationary .28m bodies separated .40m must cease overlapping."""
    state = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5], [0.4, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5]])
    sim = Simulator(state.copy(), config=config(), make_forces=free_forces)
    sim.step()
    assert np.linalg.norm(sim.peds.pos()[0] - sim.peds.pos()[1]) >= 0.56
    np.testing.assert_allclose(sim.peds.pos().mean(axis=0), [0.2, 0], atol=1e-12)


def test_separated_bodies_receive_no_contact_correction():
    """A separated pair must retain exact free integration bytes."""
    state = np.array([[0.0, 0.0, 0.1, 0.0, 10.0, 0.0, 0.5], [2.0, 0.0, 0.1, 0.0, 10.0, 0.0, 0.5]])
    off = Simulator(state.copy(), config=config(contact=False), make_forces=free_forces)
    on = Simulator(state.copy(), config=config(), make_forces=free_forces)
    off.step(20)
    on.step(20)
    np.testing.assert_array_equal(on.peds.state, off.peds.state)


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
def test_contact_stable_at_simulator_dt(dt):
    """Approaching discs remain finite and disjoint for 100 real steps."""
    state = np.array(
        [[-0.35, 0.0, 1.0, 0.0, 10.0, 0.0, 0.5], [0.35, 0.0, -1.0, 0.0, -10.0, 0.0, 0.5]]
    )
    sim = Simulator(state.copy(), config=config(dt=dt), make_forces=free_forces)
    for _ in range(100):
        sim.step()
        assert np.isfinite(sim.peds.state).all()
        assert np.linalg.norm(sim.peds.pos()[0] - sim.peds.pos()[1]) >= 0.56
    assert sim.peds.pos()[0, 0] < sim.peds.pos()[1, 0]


def test_wall_not_penetrated_even_when_step_crosses_it():
    """A 1m attempted step across a thin wall stops at the body edge."""
    state = np.array([[-0.4, 0.0, 10.0, 0.0, 10.0, 0.0, 0.5]])
    sim = Simulator(
        state.copy(),
        obstacles=[(0.0, 0.0, -2.0, 2.0)],
        config=config(wall=True),
        make_forces=free_forces,
    )
    sim.step(5)
    assert sim.peds.pos()[0, 0] <= -0.28


def test_passes_gap_one_and_half_body_widths():
    """A lone walker traverses a .84m aperture with a .56m diameter."""
    cfg = config(wall=True)
    cfg.obstacle_force_config.factor = 0.003
    cfg.obstacle_force_config.threshold = 0.375
    cfg.obstacle_force_config.sigma = 0
    state = np.array([[2.0, 0.01, 1.29, 0.0, 15.0, 0.0, 0.5]])
    walls = [
        (8.0, 8.0, 0.42, 3.0),
        (8.0, 8.0, -3.0, -0.42),
        (8.1, 8.1, 0.42, 3.0),
        (8.1, 8.1, -3.0, -0.42),
        (8.0, 8.1, 0.42, 0.42),
        (8.0, 8.1, -0.42, -0.42),
    ]
    sim = Simulator(state.copy(), obstacles=walls, config=cfg)
    sim.peds.assign_desired_speeds(np.array([1.29]))
    for _ in range(300):
        sim.step()
    assert sim.peds.pos()[0, 0] > 9.2, "physically feasible aperture stalled upstream"


def test_nonreactive_interferer_is_not_displaced():
    """CALFIT's prescribed walker stays prescribed while the subject yields fully."""
    cfg = config()
    cfg.contact_prescribed_indices = (1,)
    state = np.array(
        [[-0.35, 0.0, 1.0, 0.0, 10.0, 0.0, 0.5], [0.35, 0.0, -1.0, 0.0, -10.0, 0.0, 0.5]]
    )
    sim = Simulator(state.copy(), config=cfg, make_forces=free_forces)
    sim.step()
    np.testing.assert_allclose(sim.peds.pos()[1], [0.25, 0.0], atol=1e-14)
    assert np.linalg.norm(sim.peds.pos()[0] - sim.peds.pos()[1]) >= 0.56


def test_wall_tessellation_does_not_multiply_force():
    """Splitting the same straight wall leaves its physical force unchanged."""
    state = np.array([[0.0, 0.30, 0.0, 0.0, 10.0, 0.30, 0.5]])
    whole = Simulator(state.copy(), obstacles=[(-2.0, 2.0, 0.0, 0.0)], config=config(wall=True))
    split = Simulator(
        state.copy(),
        obstacles=[(-2.0, 0.0, 0.0, 0.0), (0.0, 2.0, 0.0, 0.0)],
        config=config(wall=True),
    )
    np.testing.assert_array_equal(whole.forces[2](), split.forces[2]())


def test_manifest_rejects_disagreement_with_live_wall_law():
    """A declared decay cannot differ from the actual installed force object's decay."""
    from pysocialforce.contact import validate_physics_manifest

    sim = Simulator(np.array([[0.0, 1.0, 0.0, 0.0, 10.0, 1.0, 0.5]]), config=config(wall=True))
    declared = sim.pedestrian_physics_metadata()
    assert declared["wall"]["decay_m"] == 0.04
    assert declared["integration"] == {"dt_s": 0.1, "integrator": "semi_implicit_euler"}
    validate_physics_manifest(sim, declared)
    declared["wall"]["decay_m"] = 0.08
    with pytest.raises(ValueError, match="differs from live"):
        validate_physics_manifest(sim, declared)


def test_runtime_wall_record_names_executed_law_and_parameters():
    """The #10084 runtime site must describe the bounded force actually evaluated."""
    import hashlib
    import json

    cfg = config(wall=True)
    cfg.obstacle_force_config.wall_contact_decay_m = 0.08
    sim = Simulator(
        np.array([[0.0, 0.30, 0.0, 0.0, 10.0, 0.30, 0.5]]),
        obstacles=[(-2.0, 2.0, 0.0, 0.0)],
        config=cfg,
    )
    before = sim.obstacle_force_law_metadata()
    assert before["applied"] is False
    sim.step()
    record = sim.obstacle_force_law_metadata()
    assert record["law_version"] == "bounded_edge_v1"
    assert record["resolution_mode"] == "explicit"
    assert record["applied"] is True
    assert record["radius_convention"] == "physical_body_edge_clearance"
    assert record["parameters"]["decay_m"] == 0.08
    assert record["parameters"]["agent_radius"] == 0.28
    assert (
        record["parameters_sha256"]
        == hashlib.sha256(
            json.dumps(record["parameters"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_robot_benchmark_manual_pedestrian_step_enforces_contact():
    """The benchmark's manual integration path must enforce the same body exclusion."""
    state = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5], [0.4, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5]])
    backend = Simulator(state.copy(), config=config(), make_forces=free_forces)
    wrapper = RobotSimulator.__new__(RobotSimulator)
    wrapper.config = SimulationSettings()
    wrapper.config.pedestrian_contact_rule = "projection_v1"
    wrapper.pysf_sim = backend
    wrapper.pedestrian_model = "social_force"
    wrapper.pysf_state = SimpleNamespace(ped_velocities=backend.peds.vel())
    wrapper._step_pedestrians(np.zeros((2, 2)), [])
    assert np.linalg.norm(backend.peds.pos()[0] - backend.peds.pos()[1]) >= 0.56


def test_optional_rules_survive_configuration_roundtrip():
    """Only explicit laws enter serialization and survive replacement."""
    from dataclasses import replace

    legacy = SimulationSettings()
    assert "pedestrian_contact_rule" not in legacy.to_dict()
    assert "pedestrian_wall_rule" not in legacy.to_dict()
    settings = SimulationSettings(
        pedestrian_contact_rule="projection_v1", pedestrian_wall_rule="bounded_edge_v1"
    )
    restored = SimulationSettings(**settings.to_dict())
    assert restored.pedestrian_contact_rule == "projection_v1"
    assert replace(settings).pedestrian_wall_rule == "bounded_edge_v1"
