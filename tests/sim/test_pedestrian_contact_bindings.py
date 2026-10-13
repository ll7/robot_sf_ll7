"""Contact must honor production model, physical geometry and oracle admission."""

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.prediction.oracle_transition_trace import ExactInverseReason
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators
from tests.sim.test_pedestrian_desired_speed import _minimal_map


def contact_pair(*, model="social_force_default", radius=0.28, trace=False):
    """Construct the real two-body dev1001 integration boundary."""
    definition = _minimal_map()
    definition.single_pedestrians = [
        SinglePedestrianDefinition(id="a", start=(7, 5), goal=(18, 5)),
        SinglePedestrianDefinition(id="b", start=(7.9, 5), goal=(1, 5)),
    ]
    settings = SimulationSettings(
        pedestrian_model=model,
        pedestrian_physics={"contact_rule": "projection_v2"},
        pedestrian_radius_m=radius,
        pedestrian_seed=1001,
        difficulty=0,
        ped_density_by_difficulty=[0.0],
        oracle_force_trace_enabled=trace,
    )
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": definition}), sim_config=settings
    )
    sim = init_simulators(
        config, definition, num_robots=1, random_start_pos=False, peds_have_obstacle_forces=True
    )[0]
    separation = 0.9 if radius is None else 0.7
    sim.pysf_sim.peds.state[:, :4] = [[7, 5, 1.3, 0], [7 + separation, 5, -1.3, 0]]
    sim.pysf_sim.peds.assign_desired_speeds(np.array([1.69, 1.69]))
    return sim


@pytest.mark.parametrize(
    "model",
    [
        "hsfm_total_force_v1",
        "hsfm_ttc_predictive_v1",
        "hsfm_zanlungo_collision_prediction_v1",
        "hsfm_anisotropic_fov_v1",
        "hsfm_alignment_torque_v1",
    ],
)
def test_separate_hsfm_integrators_refuse_contact_at_construction(model):
    """Do not accept a contact flag that a separate integrator ignores."""
    with pytest.raises(ValueError, match="contact projection does not support HSFM"):
        contact_pair(model=model)


def test_contact_without_shared_override_uses_physical_body_radius():
    """The physical 0.40 m discs need 0.80 m separation, not force-radius 0.70 m."""
    sim = contact_pair(radius=None)
    sim.step_once([(0.0, 0.0)])
    assert sim.config.ped_radius == pytest.approx(0.4)
    assert np.linalg.norm(sim.ped_pos[0] - sim.ped_pos[1]) >= 0.8
    assert sim.pysf_sim.peds.agent_radius == pytest.approx(0.35)


def test_contact_corrected_oracle_trace_refuses_exact_force_inversion():
    """An uncapped contact impulse is missing from the recorded force sum."""
    sim = contact_pair(trace=True)
    sim.step_once([(0.0, 0.0)])
    diagnostics = sim.last_step_diagnostics
    assert not diagnostics.cap_active.any()
    assert not np.array_equal(diagnostics.uncapped_velocity, diagnostics.applied_velocity)
    assert len(sim.last_oracle_transition_traces) == 2
    for trace in sim.last_oracle_transition_traces:
        assert not trace.exact_inverse_eligible
        assert ExactInverseReason.FORCE_STAGE_UNINSTRUMENTED in trace.exact_inverse_reasons


def test_uncorrected_contact_step_keeps_exact_inverse_eligibility():
    """Selecting contact alone must not disqualify unaffected free motion."""
    sim = contact_pair(trace=True)
    sim.pysf_sim.peds.state[1, 0] = 15
    sim.step_once([(0.0, 0.0)])
    assert all(trace.exact_inverse_eligible for trace in sim.last_oracle_transition_traces)
