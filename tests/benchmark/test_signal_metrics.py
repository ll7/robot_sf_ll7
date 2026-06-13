"""Tests for the signal compliance metrics."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.signal_metrics import SignalEpisode, calculate_signal_metrics


class MockSignalEpisode(SignalEpisode):
    """A mock episode class for testing signal metrics."""

    def __init__(
        self,
        robot_pos: np.ndarray,
        peds_pos: np.ndarray,
        dt: float,
        episode_metadata: dict | None,
    ):
        """Initializes the mock episode."""
        self.robot_pos = robot_pos
        self.peds_pos = peds_pos
        self.dt = dt
        self.episode_metadata = episode_metadata


def test_calculate_signal_metrics_valid_observable():
    """
    Tests the signal metrics calculation for a valid, observable scenario where the robot
    violates a red light.
    """
    # Metadata for a signalized intersection
    episode_metadata = {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [
                {"state": "red", "duration": 5.0},
                {"state": "green", "duration": 5.0},
            ],
            "stop_line": [[10.0, 10.0], [10.0, -10.0]],
            "crosswalk_polygon": [
                [11.0, 10.0],
                [15.0, 10.0],
                [15.0, -10.0],
                [11.0, -10.0],
            ],
        }
    }

    # Robot trajectory: starts before the stop line, crosses it during red phase
    robot_pos = np.array(
        [
            [0.0, 0.0],  # t=0, before stop line, red
            [11.0, 0.0],  # t=1, crossed stop line, red
            [12.0, 0.0],  # t=2, in crosswalk, red
            [13.0, 0.0],  # t=3, in crosswalk, red
            [14.0, 0.0],  # t=4, in crosswalk, red
            [15.0, 0.0],  # t=5, start of green
        ]
    )

    peds_pos = np.zeros((6, 0, 2))  # No pedestrians in this test
    dt = 1.0

    episode = MockSignalEpisode(robot_pos, peds_pos, dt, episode_metadata)
    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_red_phase_violations"] == 1
    assert metrics["signal_stop_line_crossings_under_red"] == 1
    assert np.isclose(metrics["signal_min_distance_to_stop_line_before_crossing_m"], 10.0)
    assert np.isclose(metrics["signal_delay_after_green_onset_s"], 0.0)
    assert metrics["signal_pedestrian_conflict_during_legal_crossing_count"] == 0
    assert metrics["signal_unavailable_exclusion_count"] == 0
    assert metrics["signal_metrics_denominator"] == 1
    assert metrics["signal_metrics_evidence"] == {
        "state": "planner_observable",
        "exclusion_reason": "",
    }


def test_calculate_signal_metrics_uses_crosswalk_side_for_stop_line():
    """Stop-line crossing should work for non-x-axis approaches."""
    episode_metadata = {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [{"state": "red", "duration": 0.3}],
            "stop_line": [[-1.0, 0.0], [1.0, 0.0]],
            "crosswalk_polygon": [[-1.0, 1.0], [1.0, 1.0], [1.0, 3.0], [-1.0, 3.0]],
        }
    }
    episode = MockSignalEpisode(
        np.array([[0.0, -1.0], [0.0, 0.5], [0.0, 1.5]]),
        np.zeros((3, 0, 2)),
        0.1,
        episode_metadata,
    )

    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_stop_line_crossings_under_red"] == 1
    assert metrics["signal_red_phase_violations"] == 1


def test_calculate_signal_metrics_green_crossing_not_red_violation():
    """A robot already past the stop line after green onset should not be a red violation."""
    episode_metadata = {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [
                {"state": "red", "duration": 0.2},
                {"state": "green", "duration": 0.2},
            ],
            "stop_line": [[0.0, 1.0], [0.0, -1.0]],
            "crosswalk_polygon": [[1.0, 1.0], [2.0, 1.0], [2.0, -1.0], [1.0, -1.0]],
        }
    }
    episode = MockSignalEpisode(
        np.array([[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0]]),
        np.zeros((4, 0, 2)),
        0.1,
        episode_metadata,
    )

    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_stop_line_crossings_under_red"] == 0
    assert metrics["signal_red_phase_violations"] == 0
    assert np.isclose(metrics["signal_delay_after_green_onset_s"], 0.0)


def test_calculate_signal_metrics_no_signal_data():
    """
    Tests that when no signal data is available in the metadata, the metrics reflect
    an unavailable exclusion.
    """
    robot_pos = np.zeros((10, 2))
    peds_pos = np.zeros((10, 0, 2))
    dt = 0.1

    # Test with None metadata
    episode_none = MockSignalEpisode(robot_pos, peds_pos, dt, None)
    metrics_none = calculate_signal_metrics(episode_none)
    assert metrics_none["signal_unavailable_exclusion_count"] == 1
    assert metrics_none["signal_metrics_denominator"] == 0
    assert metrics_none["signal_metrics_evidence"] == {
        "state": "unavailable",
        "exclusion_reason": "signal_state_metadata_absent",
    }

    # Test with empty metadata
    episode_empty = MockSignalEpisode(robot_pos, peds_pos, dt, {})
    metrics_empty = calculate_signal_metrics(episode_empty)
    assert metrics_empty["signal_unavailable_exclusion_count"] == 1
    assert metrics_empty["signal_metrics_denominator"] == 0
    assert metrics_empty["signal_metrics_evidence"] == {
        "state": "unavailable",
        "exclusion_reason": "signal_state_metadata_absent",
    }

    # Test with empty signal_state
    episode_empty_signal_state = MockSignalEpisode(robot_pos, peds_pos, dt, {"signal_state": {}})
    metrics_empty_signal_state = calculate_signal_metrics(episode_empty_signal_state)
    assert metrics_empty_signal_state["signal_unavailable_exclusion_count"] == 1
    assert metrics_empty_signal_state["signal_metrics_denominator"] == 0
    assert metrics_empty_signal_state["signal_metrics_evidence"] == {
        "state": "unavailable",
        "exclusion_reason": "signal_state_metadata_absent",
    }


def test_calculate_signal_metrics_proxy_diagnostic():
    """
    Tests that for proxy/diagnostic planners, the metrics are marked as unavailable for
    benchmark evidence.
    """
    episode_metadata = {
        "signal_state": {
            "contract_state": "proxy_diagnostic",
            # other data doesn't matter for this test
        }
    }
    robot_pos = np.zeros((10, 2))
    peds_pos = np.zeros((10, 0, 2))
    dt = 0.1
    episode = MockSignalEpisode(robot_pos, peds_pos, dt, episode_metadata)
    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_unavailable_exclusion_count"] == 1
    assert metrics["signal_metrics_denominator"] == 0
    assert metrics["signal_metrics_evidence"] == {
        "state": "proxy_diagnostic",
        "exclusion_reason": "signal_state_not_benchmark_evidence",
    }


def test_calculate_signal_metrics_degraded_observable_without_evidence():
    """Planner-observable metadata without benchmark evidence remains excluded."""
    episode_metadata = {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": False,
            "timeline": [{"state": "green", "duration": 1.0}],
            "stop_line": [[0.0, 1.0], [0.0, -1.0]],
        }
    }
    episode = MockSignalEpisode(
        np.zeros((2, 2)),
        np.zeros((2, 0, 2)),
        1.0,
        episode_metadata,
    )

    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_metrics_denominator"] == 0
    assert metrics["signal_unavailable_exclusion_count"] == 1
    assert metrics["signal_metrics_evidence"] == {
        "state": "planner_observable",
        "exclusion_reason": "signal_state_not_benchmark_evidence",
    }


def test_calculate_signal_metrics_incomplete_observable_fields():
    """Observable evidence without required fields is preserved but excluded."""
    episode_metadata = {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [{"state": "green", "duration": 1.0}],
        }
    }
    episode = MockSignalEpisode(
        np.zeros((2, 2)),
        np.zeros((2, 0, 2)),
        1.0,
        episode_metadata,
    )

    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_metrics_denominator"] == 0
    assert metrics["signal_unavailable_exclusion_count"] == 1
    assert metrics["signal_metrics_evidence"] == {
        "state": "planner_observable",
        "exclusion_reason": "observable_signal_fields_incomplete",
    }


def test_calculate_signal_metrics_missing_crosswalk_is_incomplete():
    """Pedestrian-conflict metrics need a conflict-zone polygon."""
    episode_metadata = {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [{"state": "green", "duration": 1.0}],
            "stop_line": [[0.0, 1.0], [0.0, -1.0]],
        }
    }
    episode = MockSignalEpisode(
        np.zeros((2, 2)),
        np.zeros((2, 0, 2)),
        1.0,
        episode_metadata,
    )

    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_metrics_denominator"] == 0
    assert metrics["signal_metrics_evidence"]["exclusion_reason"] == (
        "observable_signal_fields_incomplete"
    )


def test_calculate_signal_metrics_with_pedestrian_conflict():
    """
    Tests the pedestrian conflict metric during a legal crossing.
    """
    episode_metadata = {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [
                {"state": "red", "duration": 2.0},
                {"state": "green", "duration": 8.0},
            ],
            "stop_line": [[0.0, 5.0], [0.0, -5.0]],
            "crosswalk_polygon": [[1.0, 5.0], [5.0, 5.0], [5.0, -5.0], [1.0, -5.0]],
        }
    }

    robot_pos = np.array(
        [
            [-1.0, 0.0],  # t=0, red
            [-0.5, 0.0],  # t=1, red
            [1.5, 0.0],  # t=2, green, robot enters crosswalk
            [2.5, 0.0],  # t=3, green, robot in crosswalk
        ]
    )

    peds_pos = np.zeros((4, 1, 2))
    peds_pos[3, 0] = [2.0, 0.0]  # Pedestrian appears close to robot at t=3

    dt = 1.0

    episode = MockSignalEpisode(robot_pos, peds_pos, dt, episode_metadata)
    metrics = calculate_signal_metrics(episode)

    assert metrics["signal_pedestrian_conflict_during_legal_crossing_count"] == 1
    assert metrics["signal_unavailable_exclusion_count"] == 0
    assert metrics["signal_metrics_denominator"] == 1
    assert metrics["signal_metrics_evidence"]["state"] == "planner_observable"
