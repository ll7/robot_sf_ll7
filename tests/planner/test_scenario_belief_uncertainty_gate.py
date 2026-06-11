"""Diagnostic acceptance test for issue #2606: ScenarioBelief uncertainty gate evaluation.

This test answers one bounded question: can ScenarioBelief.to_uncertainty_report() feed
one planner input/projection without silently dropping covariance, confidence,
class_probabilities, or existence metadata?

Claim boundary: diagnostic only. Does not prove benchmark improvement, safety improvement,
planner performance, perception calibration, or paper-facing result.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.scenario_belief_adapter import project_scenario_belief_for_planner
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, StreamGapPlannerConfig
from robot_sf.representation import Estimate2D, scenario_belief_from_simulator_oracle


def _blocking_belief():
    """Return a ScenarioBelief with one blocking pedestrian at known metadata values."""
    simulator = SimpleNamespace(
        ped_pos=np.array([[0.8, 0.0]], dtype=np.float32),
        ped_vel=np.array([[0.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([4.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=8.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    return scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )


def _belief_with_known_uncertainty():
    """Return a ScenarioBelief with explicit, distinguishable uncertainty metadata.

    The agent has:
    - existence_probability = 0.95 (high, above gate threshold)
    - position confidence = 0.92 (high)
    - class_probabilities = {"pedestrian": 0.98} (high)
    - position_covariance_xy = [[0.1, 0.0], [0.0, 0.1]] (low variance, below gate threshold)
    """
    belief = _blocking_belief()
    agent = belief.agents[0]
    known_agent = replace(
        agent,
        position=Estimate2D.point(
            agent.position.mean_xy,
            confidence=0.92,
            variance=0.1,
        ),
        velocity=Estimate2D.point(
            agent.velocity.mean_xy,
            confidence=0.88,
            variance=0.05,
            units="m/s",
            covariance_units="(m/s)^2",
        ),
        existence_probability=0.95,
        class_probabilities=(("pedestrian", 0.98),),
    )
    return replace(belief, agents=(known_agent,))


def _belief_with_low_uncertainty():
    """Return a ScenarioBelief with high uncertainty that should be gate-dropped.

    The agent has:
    - existence_probability = 0.2 (below gate threshold 0.5)
    - position confidence = 0.2 (below gate threshold 0.5)
    - class_probabilities = {"pedestrian": 0.2} (below gate threshold 0.5)
    - position_covariance_xy = [[4.0, 0.0], [0.0, 4.0]] (high variance, above gate threshold 1.0)
    """
    belief = _blocking_belief()
    agent = belief.agents[0]
    known_agent = replace(
        agent,
        position=Estimate2D.point(
            agent.position.mean_xy,
            confidence=0.2,
            variance=4.0,
        ),
        existence_probability=0.2,
        class_probabilities=(("pedestrian", 0.2),),
    )
    return replace(belief, agents=(known_agent,))


# --- Required fields in the uncertainty report agent row ---

_REPORT_AGENT_REQUIRED_FIELDS = (
    "entity_id",
    "class_probabilities",
    "position_covariance_xy",
    "velocity_covariance_xy",
    "position_confidence",
    "velocity_confidence",
    "existence_probability",
    "visibility_state",
)


class TestUncertaintyReportFieldPreservation:
    """Verify that to_uncertainty_report() preserves all required metadata fields."""

    def test_report_has_all_required_agent_fields(self) -> None:
        """Each agent row in the uncertainty report must carry all metadata fields."""
        belief = _belief_with_known_uncertainty()
        report = belief.to_uncertainty_report()

        assert len(report["agents"]) == 1
        agent_row = report["agents"][0]
        for field in _REPORT_AGENT_REQUIRED_FIELDS:
            assert field in agent_row, f"missing field: {field}"

    def test_report_preserves_class_probabilities(self) -> None:
        """Class probabilities must match the belief's source data."""
        belief = _belief_with_known_uncertainty()
        report = belief.to_uncertainty_report()
        agent_row = report["agents"][0]

        assert agent_row["class_probabilities"] == {"pedestrian": pytest.approx(0.98)}

    def test_report_preserves_covariance_shape(self) -> None:
        """Position and velocity covariance must be 2x2 matrices."""
        belief = _belief_with_known_uncertainty()
        report = belief.to_uncertainty_report()
        agent_row = report["agents"][0]

        pos_cov = np.asarray(agent_row["position_covariance_xy"])
        vel_cov = np.asarray(agent_row["velocity_covariance_xy"])
        assert pos_cov.shape == (2, 2)
        assert vel_cov.shape == (2, 2)

    def test_report_preserves_confidence_values(self) -> None:
        """Position and velocity confidence must match the belief source."""
        belief = _belief_with_known_uncertainty()
        report = belief.to_uncertainty_report()
        agent_row = report["agents"][0]

        assert agent_row["position_confidence"] == pytest.approx(0.92)
        assert agent_row["velocity_confidence"] == pytest.approx(0.88)

    def test_report_preserves_existence_probability(self) -> None:
        """Existence probability must match the belief source."""
        belief = _belief_with_known_uncertainty()
        report = belief.to_uncertainty_report()
        agent_row = report["agents"][0]

        assert agent_row["existence_probability"] == pytest.approx(0.95)

    def test_report_preserves_velocity_covariance(self) -> None:
        """Velocity covariance must survive the report even though stream_gap doesn't use it."""
        belief = _belief_with_known_uncertainty()
        report = belief.to_uncertainty_report()
        agent_row = report["agents"][0]

        vel_cov = np.asarray(agent_row["velocity_covariance_xy"])
        assert np.allclose(vel_cov, np.diag([0.05, 0.05]), atol=1e-4)


class TestAdapterFieldPreservation:
    """Verify that project_scenario_belief_for_planner preserves uncertainty fields."""

    def test_adapter_copies_all_agent_fields_to_sidecar(self) -> None:
        """The uncertainty sidecar injected by the adapter must contain all report fields."""
        belief = _belief_with_known_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        assert projection.compatibility["status"] == "compatible"
        sidecar = projection.observation["pedestrians"]["uncertainty"]
        assert len(sidecar) == 1
        agent_row = sidecar[0]

        for field in _REPORT_AGENT_REQUIRED_FIELDS:
            assert field in agent_row, f"sidecar missing field: {field}"

    def test_adapter_sidecar_matches_report_exactly(self) -> None:
        """Sidecar agent rows must be equal to the uncertainty report agents."""
        belief = _belief_with_known_uncertainty()
        report = belief.to_uncertainty_report()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        sidecar = projection.observation["pedestrians"]["uncertainty"]
        assert sidecar[0] == report["agents"][0]

    def test_adapter_preserves_covariance_in_sidecar(self) -> None:
        """Covariance matrices must survive the adapter projection."""
        belief = _belief_with_known_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        sidecar = projection.observation["pedestrians"]["uncertainty"]
        pos_cov = np.asarray(sidecar[0]["position_covariance_xy"])
        assert pos_cov.shape == (2, 2)
        assert np.allclose(pos_cov, np.diag([0.1, 0.1]), atol=1e-4)

    def test_adapter_preserves_class_probabilities_in_sidecar(self) -> None:
        """Class probabilities must survive the adapter projection."""
        belief = _belief_with_known_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        sidecar = projection.observation["pedestrians"]["uncertainty"]
        assert sidecar[0]["class_probabilities"]["pedestrian"] == pytest.approx(0.98)

    def test_adapter_preserves_existence_in_sidecar(self) -> None:
        """Existence probability must survive the adapter projection."""
        belief = _belief_with_known_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        sidecar = projection.observation["pedestrians"]["uncertainty"]
        assert sidecar[0]["existence_probability"] == pytest.approx(0.95)


class TestPlannerConsumesUncertaintyFields:
    """Verify that the stream_gap planner actually reads and acts on uncertainty fields."""

    def test_high_confidence_agent_is_kept_by_planner(self) -> None:
        """A high-confidence blocker should remain after gating (not dropped)."""
        belief = _belief_with_known_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        planner = StreamGapPlannerAdapter(
            StreamGapPlannerConfig(
                uncertainty_gating_enabled=True,
                uncertainty_min_existence_probability=0.5,
                uncertainty_min_position_confidence=0.5,
                uncertainty_min_class_probability=0.5,
                uncertainty_max_position_variance=1.0,
            )
        )
        v, _ = planner.plan(projection.observation)

        assert v == 0.0
        assert planner.last_uncertainty_gate["status"] == "applied"
        assert planner.last_uncertainty_gate["kept_count"] == 1
        assert planner.last_uncertainty_gate["dropped_count"] == 0

    def test_low_confidence_agent_is_dropped_by_planner(self) -> None:
        """A low-confidence blocker should be dropped when gating is enabled."""
        belief = _belief_with_low_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        planner = StreamGapPlannerAdapter(
            StreamGapPlannerConfig(
                uncertainty_gating_enabled=True,
                uncertainty_min_existence_probability=0.5,
                uncertainty_min_position_confidence=0.5,
                uncertainty_min_class_probability=0.5,
                uncertainty_max_position_variance=1.0,
            )
        )
        v, _ = planner.plan(projection.observation)

        assert v > 0.0
        assert planner.last_uncertainty_gate["status"] == "applied"
        assert planner.last_uncertainty_gate["dropped_count"] == 1
        reasons = set(planner.last_uncertainty_gate["dropped_reasons"][0]["reasons"])
        assert "existence_probability_below_threshold" in reasons
        assert "position_confidence_below_threshold" in reasons
        assert "class_probability_below_threshold" in reasons
        assert "position_variance_above_threshold" in reasons

    def test_planner_reads_all_four_gating_fields(self) -> None:
        """The planner must parse existence, confidence, class_prob, and variance from each row."""
        belief = _belief_with_known_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")
        sidecar = projection.observation["pedestrians"]["uncertainty"]
        agent_row = sidecar[0]

        planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(uncertainty_gating_enabled=True))
        metrics = planner._uncertainty_row_metrics(agent_row)

        assert metrics is not None
        existence, confidence, class_probability, variance = metrics
        assert existence == pytest.approx(0.95)
        assert confidence == pytest.approx(0.92)
        assert class_probability == pytest.approx(0.98)
        assert variance == pytest.approx(0.1)

    def test_planner_gate_disabled_preserves_all_rows(self) -> None:
        """When gating is disabled, all rows must survive regardless of uncertainty values."""
        belief = _belief_with_low_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(uncertainty_gating_enabled=False))
        v, _ = planner.plan(projection.observation)

        assert v == 0.0
        assert planner.last_uncertainty_gate["status"] == "disabled"
        assert planner.last_uncertainty_gate["kept_count"] == 1
        assert planner.last_uncertainty_gate["dropped_count"] == 0


class TestEndToEndGateDecision:
    """Verify that uncertainty fields can change the planner's commit/wait decision."""

    def test_uncertainty_gate_changes_planner_decision(self) -> None:
        """With the same geometry, gating must change v from 0 to >0 for a low-confidence blocker."""
        belief = _belief_with_low_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        deterministic = StreamGapPlannerAdapter(StreamGapPlannerConfig())
        v_det, _ = deterministic.plan(projection.observation)

        gated = StreamGapPlannerAdapter(
            StreamGapPlannerConfig(
                uncertainty_gating_enabled=True,
                uncertainty_min_existence_probability=0.5,
                uncertainty_min_position_confidence=0.5,
                uncertainty_min_class_probability=0.5,
                uncertainty_max_position_variance=1.0,
            )
        )
        v_gated, _ = gated.plan(projection.observation)

        assert v_det == 0.0
        assert v_gated > 0.0
        assert gated.last_uncertainty_gate["status"] == "applied"

    def test_high_confidence_uncertainty_does_not_change_decision(self) -> None:
        """A high-confidence blocker should keep the planner waiting even with gating enabled."""
        belief = _belief_with_known_uncertainty()
        projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

        deterministic = StreamGapPlannerAdapter(StreamGapPlannerConfig())
        v_det, _ = deterministic.plan(projection.observation)

        gated = StreamGapPlannerAdapter(
            StreamGapPlannerConfig(
                uncertainty_gating_enabled=True,
                uncertainty_min_existence_probability=0.5,
                uncertainty_min_position_confidence=0.5,
                uncertainty_min_class_probability=0.5,
                uncertainty_max_position_variance=1.0,
            )
        )
        v_gated, _ = gated.plan(projection.observation)

        assert v_det == 0.0
        assert v_gated == 0.0
