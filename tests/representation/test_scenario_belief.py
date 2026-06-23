"""ScenarioBelief MVP adapter and projection tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.common.issue_provenance import SCENARIO_BELIEF_DESIGN_PARENT_ISSUE
from robot_sf.gym_env.unified_config import ObservationVisibilitySettings, RobotSimulationConfig
from robot_sf.representation import (
    TrackedAgentMetadata,
    VisibilityState,
    compute_clear_tracking_metrics,
    compute_projection_diff,
    scenario_belief_from_simulator_oracle,
    scenario_belief_from_visibility_limited_simulator,
)
from robot_sf.representation.scenario_belief import DESIGN_PARENT_ISSUE


def _simulator_fixture() -> SimpleNamespace:
    """Return one deterministic simulator-like step for representation tests."""
    return SimpleNamespace(
        ped_pos=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.1, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=8.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )


def test_oracle_and_partial_adapters_share_schema_and_projection_keys() -> None:
    """Different input paths should keep the same belief and policy-observation contracts."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()

    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    assert oracle.schema_version == partial.schema_version == "scenario-belief.v1"
    assert oracle.policy_projection_keys() == partial.policy_projection_keys()
    assert set(oracle.to_socnav_struct().keys()) == set(partial.to_socnav_struct().keys())
    assert set(oracle.to_socnav_struct()["pedestrians"].keys()) == set(
        partial.to_socnav_struct()["pedestrians"].keys()
    )


def test_partial_adapter_marks_unseen_agents_without_changing_projection_schema() -> None:
    """Partial perception should differ via visibility/uncertainty, not key drift."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()

    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    oracle_obs = oracle.to_socnav_struct()
    partial_obs = partial.to_socnav_struct()
    np.testing.assert_allclose(oracle_obs["robot"]["speed"], np.array([0.1, 0.0], dtype=np.float32))
    assert oracle_obs["pedestrians"]["count"][0] == pytest.approx(2.0)
    assert partial_obs["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert (
        partial_obs["pedestrians"]["positions"].shape
        == oracle_obs["pedestrians"]["positions"].shape
    )

    unseen = [
        agent for agent in partial.agents if agent.visibility_state is VisibilityState.OUTSIDE_FOV
    ]
    assert len(unseen) == 1
    assert unseen[0].source.adapter == "visibility_limited_simulator"
    assert unseen[0].position.confidence < oracle.agents[1].position.confidence
    assert unseen[0].missing_fields == ("policy_position", "policy_velocity")


def test_debug_projection_is_deterministic_and_exposes_uncertainty() -> None:
    """Debug output should explain source, confidence, visibility, and missing-data differences."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(enabled=True, max_range_m=2.5)
    simulator = _simulator_fixture()

    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    debug_a = partial.to_debug_dict()
    debug_b = partial.to_debug_dict()
    assert debug_a == debug_b
    assert DESIGN_PARENT_ISSUE == SCENARIO_BELIEF_DESIGN_PARENT_ISSUE
    assert debug_a["design_parent_issue"] == SCENARIO_BELIEF_DESIGN_PARENT_ISSUE
    assert debug_a["source_summary"]["adapter"] == "visibility_limited_simulator"
    assert debug_a["agents"][1]["visibility_state"] == "out_of_range"
    assert (
        debug_a["agents"][1]["position"]["confidence"]
        < debug_a["agents"][0]["position"]["confidence"]
    )
    assert "policy_position" in debug_a["agents"][1]["missing_fields"]


def test_adapter_tolerates_missing_optional_simulator_fields() -> None:
    """The MVP adapter should fail soft for optional map and pedestrian velocity fields."""
    simulator = SimpleNamespace(
        ped_pos=np.array([], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((1.0, 2.0), 0.0),
                current_speed=None,
                robot_velocity_xy=np.array([0.2, 0.3], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    belief = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )

    obs = belief.to_socnav_struct()
    assert obs["map"]["size"].tolist() == pytest.approx([50.0, 50.0])
    assert obs["robot"]["speed"].tolist() == pytest.approx([0.0, 0.0])
    assert obs["robot"]["velocity_xy"].tolist() == pytest.approx([0.2, 0.3])
    assert obs["pedestrians"]["count"][0] == pytest.approx(0.0)


def test_adapter_falls_back_when_pedestrian_velocity_shape_mismatches() -> None:
    """Mismatched pedestrian velocities should not break the belief adapter."""
    simulator = _simulator_fixture()
    simulator.ped_vel = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    belief = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )

    obs = belief.to_socnav_struct()
    np.testing.assert_allclose(obs["pedestrians"]["velocities"], np.zeros((4, 2), dtype=np.float32))


def test_tracking_metadata_populated_for_non_visible_agents() -> None:
    """Non-visible agents should carry TrackedAgentMetadata diagnostics."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    visible_agents = [a for a in partial.agents if a.visibility_state is VisibilityState.VISIBLE]
    non_visible_agents = [
        a for a in partial.agents if a.visibility_state is not VisibilityState.VISIBLE
    ]

    for agent in visible_agents:
        assert agent.tracking is None, f"visible agent {agent.entity_id} should not have tracking"

    for agent in non_visible_agents:
        assert agent.tracking is not None, f"non-visible agent {agent.entity_id} missing tracking"
        assert agent.tracking.is_coasted is True
        assert agent.tracking.missed_detections == 3
        assert agent.tracking.track_age_s == pytest.approx(1.0)
        debug = agent.tracking.to_debug_dict()
        assert debug["track_id"] == f"track_{agent.entity_id}"
        assert debug["is_coasted"] is True


def test_diagnostic_summary_reports_visibility_counts() -> None:
    """diagnostic_summary should report per-visibility-state counts and metadata."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    oracle_summary = oracle.diagnostic_summary()
    assert oracle_summary["total_agents"] == 2
    assert oracle_summary["visible_count"] == 2
    assert oracle_summary["occluded_count"] == 0
    assert oracle_summary["agents_with_missing_data"] == 0
    assert oracle_summary["agents_not_observed_this_step"] == 0
    assert oracle_summary["coasted_agents"] == 0
    assert oracle_summary["adapter"] == "simulator_oracle"

    partial_summary = partial.diagnostic_summary()
    assert partial_summary["total_agents"] == 2
    assert partial_summary["visible_count"] == 1
    assert partial_summary["outside_fov_count"] == 1
    assert partial_summary["agents_with_missing_data"] == 1
    assert partial_summary["agents_not_observed_this_step"] == 1
    assert partial_summary["agents_with_tracking_meta"] == 1
    assert partial_summary["coasted_agents"] == 1
    assert partial_summary["adapter"] == "visibility_limited_simulator"


def test_compute_projection_diff_detects_oracle_vs_partial_differences() -> None:
    """compute_projection_diff should report per-agent and summary diffs."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()
    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    diff = compute_projection_diff(oracle, partial)

    assert diff.total_agents_oracle == 2
    assert diff.total_agents_partial == 2
    assert diff.visible_agents_oracle == 2
    assert diff.visible_agents_partial == 1
    assert diff.agent_count_match is True
    assert diff.policy_key_set_match is True
    assert diff.ego_position_diff == pytest.approx(0.0)

    agent_diffs_by_id = {d.entity_id: d for d in diff.agent_diffs}
    assert len(agent_diffs_by_id) == 2

    fov_agent = agent_diffs_by_id["ped_001"]
    assert fov_agent.visibility_oracle == "visible"
    assert fov_agent.visibility_partial == "outside_fov"
    assert fov_agent.in_policy_oracle is True
    assert fov_agent.in_policy_partial is False
    assert fov_agent.confidence_oracle > fov_agent.confidence_partial
    assert "policy_position" in fov_agent.missing_fields_partial


def test_clear_tracking_metrics_are_perfect_for_oracle_vs_oracle() -> None:
    """Zero-noise oracle comparisons should report perfect CLEAR diagnostics."""
    simulator = _simulator_fixture()
    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )

    metrics = compute_clear_tracking_metrics(oracle, oracle)

    assert metrics["schema_version"] == "clear-tracking-metrics.v1"
    assert metrics["ground_truth_count"] == 2
    assert metrics["detection_count"] == 2
    assert metrics["missed_detection_count"] == 0
    assert metrics["false_positive_count"] == 0
    assert metrics["id_switch_count"] == 0
    assert metrics["mota"] == pytest.approx(1.0)
    assert metrics["motp_m"] == pytest.approx(0.0)


def test_clear_tracking_metrics_penalize_visibility_limited_misses() -> None:
    """Visibility-limited beliefs should expose missed detections through MOTA."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()
    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    metrics = compute_clear_tracking_metrics(oracle, partial)

    assert metrics["ground_truth_count"] == 2
    assert metrics["detection_count"] == 1
    assert metrics["missed_detection_count"] == 1
    assert metrics["false_positive_count"] == 0
    assert metrics["mota"] == pytest.approx(0.5)
    assert metrics["motp_m"] == pytest.approx(0.0)


def test_clear_tracking_metrics_penalize_false_positive_without_truth() -> None:
    """False positives should not look like perfect tracking when no truth agents exist."""
    oracle = scenario_belief_from_simulator_oracle(
        _simulator_fixture(),
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )
    ground_truth = replace(oracle, agents=())
    observed = replace(oracle, agents=(oracle.agents[0],))

    metrics = compute_clear_tracking_metrics(ground_truth, observed)

    assert metrics["ground_truth_count"] == 0
    assert metrics["false_positive_count"] == 1
    assert metrics["mota"] == pytest.approx(0.0)
    assert np.isnan(metrics["motp_m"])


def test_clear_tracking_metrics_report_configured_centroid_noise() -> None:
    """Configured synthetic tracking noise should surface as MOTP degradation."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=360.0,
        tracking_noise_std_m=0.25,
    )
    simulator = _simulator_fixture()
    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    degraded = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    metrics = compute_clear_tracking_metrics(oracle, degraded)

    assert metrics["ground_truth_count"] == 2
    assert metrics["detection_count"] == 2
    assert metrics["missed_detection_count"] == 0
    assert metrics["mota"] == pytest.approx(1.0)
    assert metrics["motp_m"] == pytest.approx(0.25)
    assert degraded.agents[0].position.covariance_xy[0][0] == pytest.approx(0.25**2)
    assert env_config.observation_visibility.to_metadata()["tracking_noise_std_m"] == 0.25


def test_compute_projection_diff_empty_agents() -> None:
    """compute_projection_diff should handle empty agent tuples."""
    env_config = RobotSimulationConfig()
    empty_sim = SimpleNamespace(
        ped_pos=np.zeros((0, 2), dtype=np.float32),
        ped_vel=np.zeros((0, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=8.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    oracle = scenario_belief_from_simulator_oracle(
        empty_sim,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        empty_sim,
        env_config=env_config,
        max_pedestrians=4,
    )

    diff = compute_projection_diff(oracle, partial)
    assert diff.total_agents_oracle == 0
    assert diff.total_agents_partial == 0
    assert diff.visible_agents_oracle == 0
    assert diff.visible_agents_partial == 0
    assert len(diff.agent_diffs) == 0
    assert diff.agent_count_match is True


def test_tracked_agent_metadata_to_debug_dict_produces_deterministic_output() -> None:
    """TrackedAgentMetadata.to_debug_dict should be stable and JSON-ready."""
    meta = TrackedAgentMetadata(
        track_id="track_ped_001",
        detection_count=5,
        missed_detections=2,
        track_age_s=3.0,
        last_detection_s=2.5,
        is_coasted=False,
    )
    debug = meta.to_debug_dict()
    assert debug["track_id"] == "track_ped_001"
    assert debug["detection_count"] == 5
    assert debug["missed_detections"] == 2
    assert debug["track_age_s"] == 3.0
    assert debug["last_detection_s"] == 2.5
    assert debug["is_coasted"] is False
    # second call should match
    assert meta.to_debug_dict() == debug


def test_to_uncertainty_report_preserves_covariance_and_class_probabilities() -> None:
    """to_uncertainty_report should preserve uncertainty fields that to_socnav_struct drops."""
    env_config = RobotSimulationConfig()
    simulator = _simulator_fixture()
    simulator.robots[0].pose = ((0.0, 0.0), 4.0)
    belief = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    report = belief.to_uncertainty_report()
    legacy_obs = belief.to_socnav_struct()

    assert "agents" in report
    assert "ego" in report
    assert len(report["agents"]) == 2

    first_agent = report["agents"][0]
    assert "class_probabilities" in first_agent
    assert "position_covariance_xy" in first_agent
    assert "velocity_covariance_xy" in first_agent
    assert "position_confidence" in first_agent
    assert "velocity_confidence" in first_agent
    assert first_agent["class_probabilities"] == {"pedestrian": 0.98}
    assert first_agent["position_confidence"] == pytest.approx(0.98)

    assert "class_probabilities" in report["ego"]
    assert "position_covariance_xy" in report["ego"]
    assert "velocity_covariance_xy" in report["ego"]
    assert report["ego"]["class_probabilities"] == {"ego_robot": 1.0}
    assert report["ego"]["heading"] == pytest.approx(float(legacy_obs["robot"]["heading"][0]))

    assert "goals" in report
    assert "current_covariance_xy" in report["goals"][0]
    assert "next_covariance_xy" in report["goals"][0]


def test_to_socnav_struct_fails_closed_for_uncertainty_consumption() -> None:
    """to_socnav_struct output lacks uncertainty fields; consumers must not find them."""
    env_config = RobotSimulationConfig()
    simulator = _simulator_fixture()
    belief = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    obs = belief.to_socnav_struct()

    # Fail-closed: the legacy projection drops covariance and class-probability keys.
    with pytest.raises(KeyError):
        _ = obs["position_covariance_xy"]
    with pytest.raises(KeyError):
        _ = obs["covariance"]
    robot = obs["robot"]
    assert "covariance" not in robot
    assert "confidence" not in robot
    pedestrians = obs["pedestrians"]
    assert "covariance" not in pedestrians
    assert "class_probabilities" not in pedestrians
