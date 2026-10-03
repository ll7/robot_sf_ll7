"""Round 3 witnesses use real simulator paths and independently known bounds."""

import numpy as np
import pytest
from pysocialforce import Simulator, contact
from pysocialforce.forces import ObstacleForce

from robot_sf.research.pedestrian_acceptance import engineering_gate
from robot_sf.research.pedestrian_validation import turning_onset
from tests.sim.test_pedestrian_contact import config, free_forces


@pytest.mark.parametrize("unresolved", [False, True])
def test_capped_contact_retry_preserves_isolated_walker(monkeypatch, unresolved):
    """A successful jam repair must not stop a walker fifty metres away."""
    np.random.seed(1001)
    monkeypatch.setattr(contact, "MAX_PASSES", 1)
    state = np.array(
        [[x, 0, 0, 0, 10, 0, 0.5] for x in [0, 0.1, 0.2]] + [[50, 50, 1.3, 0, 100, 50, 0.5]]
    )
    cfg = config()
    if unresolved:
        cfg.contact_prescribed_indices = (0, 1)
    sim = Simulator(state, config=cfg, make_forces=free_forces)
    sim.peds.max_speeds[:] = 2
    sim.step()
    assert sim.contact_projection_fallback_count == 1
    assert sim.contact_projection_unresolved_count == int(unresolved)
    np.testing.assert_array_equal(sim.peds.vel()[-1], [1.3, 0])


def test_local_rollback_closes_new_contacts_without_stopping_isolated_walker(monkeypatch):
    """Rolling back a jam must include a neighbour newly overlapped by that rollback."""
    np.random.seed(1001)
    cfg = config()
    cfg.scene_config.agent_radius = 0.30
    previous = np.array(
        [
            [0, 0, 1.2, 0, 10, 0, 0.5],
            [0.61, 0, 0.4, 0, 10, 0, 0.5],
            [-0.61, 0, 1.1, 0, 10, 0, 0.5],
            [50, 50, 1.3, 0, 100, 50, 0.5],
        ]
    )
    attempted = np.array([[0.12, 0], [0.65, 0], [-0.50, 0], [50.13, 50]])
    sim = Simulator(previous.copy(), config=cfg, make_forces=free_forces)
    sim.peds.max_speeds[:] = 2
    sim.peds.state[:, :2] = attempted
    monkeypatch.setattr(contact, "project_step", lambda *args: (attempted.copy(), 1, False))
    contact.apply_contact_step(sim, previous)
    distances = np.linalg.norm(sim.peds.pos()[:3, None] - sim.peds.pos()[None, :3], axis=2)
    assert distances[np.triu_indices(3, 1)].min() >= 0.60
    np.testing.assert_array_equal(sim.peds.pos()[-1], attempted[-1])
    np.testing.assert_array_equal(sim.peds.vel()[-1], [1.3, 0])
    assert sim.contact_projection_fallback_count == 1
    assert sim.contact_projection_unresolved_count == 0


def test_turn_already_underway_at_source_window_is_right_censored():
    """A manoeuvre beginning before x=-3 has a lower bound, not a late exact onset."""
    t = np.arange(601) * 0.1
    baseline = np.column_stack([t, np.zeros(len(t))])
    path = baseline.copy()
    path[:, 1] = 0.03 * np.maximum(t - 20, 0) ** 2
    q = np.column_stack([60 - t, np.zeros(len(t))])
    result = turning_onset(path, q, t, [baseline] * 5)
    assert result["analysis_window_m"] == 3.0
    assert result["onset_right_censored"] is True
    assert result["onset_m"] == 3.0
    assert result["onset_threshold_rad_s"] == 0.05


@pytest.mark.parametrize("onset,expected", [(3.0, "PASS"), (2.5, "FAIL")])
def test_v6_lower_bound_has_no_upper_acceptance_limit(onset, expected):
    """A source-comparable earlier onset satisfies the author-ruled lower bound."""
    row = {
        "case": "V6",
        "variant": "1.78",
        "seed": 1001,
        "onset_m": onset,
        "onset_right_censored": onset == 3.0,
        "pair_overlap": {"all": {"below_2r_count": 0}},
    }
    check = engineering_gate([row])["checks"][0]
    assert check["status"] == expected
    assert check["comparison"] == "published_lower_bound"
    assert check["right_censored_n"] == int(onset == 3.0)


def test_optional_wall_correction_preserves_legacy_far_field():
    """At 1.38 m body clearance the optional wall law retains actual legacy forces."""
    np.random.seed(1001)
    state = np.array([[0, 1.66, 0, 0, 10, 1.66, 0.5]])
    legacy = Simulator(state.copy(), obstacles=[(-10, 10, 0, 0)], config=config(contact=False))
    fixed = Simulator(
        state.copy(), obstacles=[(-10, 10, 0, 0)], config=config(contact=False, wall=True)
    )
    off = ObstacleForce(legacy.config.obstacle_force_config, legacy)()
    on = ObstacleForce(fixed.config.obstacle_force_config, fixed)()
    assert np.linalg.norm(off) > 0.01
    np.testing.assert_allclose(on, off, rtol=0, atol=1e-12)


def test_partial_bottleneck_publishes_full_flow_bound_and_crossing_count(monkeypatch):
    """Two known crossings in ten seconds bound completion of the 60-person trial."""
    from scripts.validation import compare_obstacle_laws_10061 as flow

    def trace(state, *args, **kwargs):
        positions = np.repeat(state[None, :, :2], 101, axis=0)
        positions[:, :, 0] = -1
        positions[:, 0, 0] = np.arange(101) * 0.1 - 1
        positions[:, 1, 0] = np.arange(101) * 0.1 - 2
        return positions, np.zeros((100, 60))

    monkeypatch.setattr(flow.harness, "simulate", trace)
    result = flow.bottleneck(("legacy_refit", 0.003, 0.375), 1001, 1.0, False)
    assert result["crossed"] == 2
    assert result["flow_right_censored"] is True
    assert result["specific_flow_persons_m_s"] == pytest.approx(60 / 7.6)
    assert result["specific_flow_bound"] == "upper"


def test_suite_config_versions_the_effective_onset_and_keeps_history():
    """The opt-in source config must describe the estimator its runner executes."""
    from scripts.validation import pedestrian_validation_10074 as suite

    current = suite.load_config(suite.DEFAULT_CONFIG)
    assert current["V6"]["estimator_id"] == "fixed_yaw_source_window_lower_bound_v3"
    assert current["V6"]["analysis_window_m"] == 3.0
    assert current["V6"]["comparison"] == "published_lower_bound"
    old = suite.load_config(
        suite.DEFAULT_CONFIG.with_name("pedestrian_validation_0_0_9_history_v1.json")
    )
    assert "five no-interferer trials" in old["V6"]["onset_definition"]


def test_censored_flow_inside_range_cannot_establish_a_pass():
    """58/60 crossings allow either an accepted or slower full completion flow."""
    row = {
        "case": "V3",
        "variant": "1.0",
        "seed": 1001,
        "specific_flow_persons_m_s": 60 / (32 - 2),
        "flow_right_censored": True,
        "crossed": 58,
        "pair_overlap": {"all": {"below_2r_count": 0}},
    }
    result = engineering_gate([row])
    check = result["checks"][0]
    assert check["status"] == "CENSORED"
    assert check["identified_interval"] == [0.0, 2.0]
    assert result["producible_items"] == 0
