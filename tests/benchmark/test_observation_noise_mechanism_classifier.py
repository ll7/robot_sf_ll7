"""Tests for the observation-noise mechanism-layer classifier."""

from __future__ import annotations

import json
import pathlib

from robot_sf.benchmark.observation_noise_mechanism_classifier import (
    MECHANISM_LABELS,
    classify_all_conditions,
    classify_mechanism,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13"
    / "summary.json"
)


def _load_envelope() -> dict:
    """Load the envelope evidence for test use."""
    with open(EVIDENCE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _noop_result() -> dict:
    """Build a minimal noop result for comparison."""
    return {
        "spec": {"is_noop": True, "position_noise_std_m": 0.0},
        "first_observed_step": 5,
        "response_delay_steps": 0,
        "closest_distance_m": 0.5,
        "missed_actor_observations_total": 0,
        "occluded_actor_observations_total": 0,
        "stop_yield_feasibility": {
            "stop_feasible_first_observed": True,
            "yield_feasible_first_observed": True,
        },
        "action_proxy_changes": {
            "linear_velocity_changed": True,
            "events": ["a", "b"],
            "velocity_range": [0.0, 0.6],
        },
    }


def _fixture_meta() -> dict:
    """Minimal fixture metadata."""
    return {"first_visible_step": 5, "trace_path": "x", "scenario_id": "y"}


class TestMechanismLabels:
    """Verify the allowed label set is stable."""

    def test_labels_are_strings(self) -> None:
        """All mechanism labels are strings."""
        assert all(isinstance(label, str) for label in MECHANISM_LABELS)

    def test_expected_labels_present(self) -> None:
        """Label set matches the expected mechanism vocabulary."""
        expected = {
            "observation_did_not_affect_selected_source",
            "observation_affected_source_but_not_command",
            "command_changed_but_trajectory_did_not",
            "delay_shifted_stop_timing",
            "occlusion_changed_first_actionable_frame",
            "noise_stayed_below_decision_threshold",
            "scenario_had_no_actionable_conflict",
            "stored_action_proxy_prevents_live_conclusion",
            "diagnostic_only",
            "inconclusive",
        }
        assert expected == MECHANISM_LABELS


class TestClassifyNoop:
    """Noop baseline gets diagnostic_only."""

    def test_noop_is_diagnostic_only(self) -> None:
        """Noop baseline receives diagnostic_only mechanism label."""
        result = {
            "spec": {"is_noop": True, "position_noise_std_m": 0.0},
            "first_observed_step": 5,
            "response_delay_steps": 0,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
        }
        m = classify_mechanism(result, _fixture_meta())
        assert m["label"] == "diagnostic_only"
        assert "Baseline" in m["rationale"] or "baseline" in m["rationale"].lower()


class TestClassifyDelay:
    """Delay conditions get delay_shifted_stop_timing."""

    def test_delay_shifts_stop_timing(self) -> None:
        """Two-step delay classifies as delay_shifted_stop_timing."""
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.0, "delay_steps": 2},
            "first_observed_step": 7,
            "response_delay_steps": 2,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
            "stop_yield_feasibility": {
                "stop_feasible_first_observed": False,
                "yield_feasible_first_observed": True,
            },
            "action_proxy_changes": {
                "linear_velocity_changed": True,
                "events": ["a"],
                "velocity_range": [0.0, 0.6],
            },
        }
        noop = _noop_result()
        m = classify_mechanism(result, _fixture_meta(), noop_result=noop)
        assert m["label"] == "delay_shifted_stop_timing"
        assert "2 step" in m["rationale"]

    def test_delay_detects_stop_change(self) -> None:
        """Delay rationale mentions stop/yield feasibility change."""
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.0, "delay_steps": 2},
            "first_observed_step": 7,
            "response_delay_steps": 2,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
            "stop_yield_feasibility": {
                "stop_feasible_first_observed": False,
                "yield_feasible_first_observed": True,
            },
            "action_proxy_changes": {
                "linear_velocity_changed": True,
                "events": ["a"],
                "velocity_range": [0.0, 0.6],
            },
        }
        noop = _noop_result()
        m = classify_mechanism(result, _fixture_meta(), noop_result=noop)
        assert "Stop/yield feasibility" in m["rationale"]


class TestClassifyOcclusion:
    """Occlusion/missed-detection conditions get occlusion_changed_first_actionable_frame."""

    def test_missed_detection(self) -> None:
        """Full missed detection classifies as occlusion_changed_first_actionable_frame."""
        result = {
            "spec": {
                "is_noop": False,
                "position_noise_std_m": 0.0,
                "missed_detection_probability": 1.0,
            },
            "first_observed_step": None,
            "response_delay_steps": None,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 11,
            "occluded_actor_observations_total": 0,
        }
        m = classify_mechanism(result, _fixture_meta())
        assert m["label"] == "occlusion_changed_first_actionable_frame"
        assert "missed" in m["rationale"].lower()

    def test_occlusion_mask(self) -> None:
        """Full occlusion mask classifies as occlusion_changed_first_actionable_frame."""
        result = {
            "spec": {
                "is_noop": False,
                "position_noise_std_m": 0.0,
                "has_occlusion_mask": True,
            },
            "first_observed_step": None,
            "response_delay_steps": None,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 11,
        }
        m = classify_mechanism(result, _fixture_meta())
        assert m["label"] == "occlusion_changed_first_actionable_frame"
        assert "occlusion" in m["rationale"].lower()

    def test_occlusion_priority_over_far_distance(self) -> None:
        """Complete occlusion remains the mechanism even when the closest distance is far."""
        result = {
            "spec": {
                "is_noop": False,
                "position_noise_std_m": 0.0,
                "has_occlusion_mask": True,
            },
            "first_observed_step": None,
            "response_delay_steps": None,
            "closest_distance_m": 3.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 11,
        }
        m = classify_mechanism(result, _fixture_meta())
        assert m["label"] == "occlusion_changed_first_actionable_frame"


class TestClassifyNoise:
    """Gaussian noise conditions classify by command change."""

    def test_small_noise_below_threshold(self) -> None:
        """Small Gaussian noise (std<=0.15) with unchanged command stays below threshold."""
        noop = _noop_result()
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.10},
            "first_observed_step": 5,
            "response_delay_steps": 0,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
            "action_proxy_changes": {
                "linear_velocity_changed": True,
                "events": ["a", "b"],
                "velocity_range": [0.0, 0.6],
            },
        }
        m = classify_mechanism(result, _fixture_meta(), noop_result=noop)
        assert m["label"] == "noise_stayed_below_decision_threshold"

    def test_larger_noise_affects_source(self) -> None:
        """Larger Gaussian noise (std>0.15) with unchanged command affects source."""
        noop = _noop_result()
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.30},
            "first_observed_step": 5,
            "response_delay_steps": 0,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
            "action_proxy_changes": {
                "linear_velocity_changed": True,
                "events": ["a", "b"],
                "velocity_range": [0.0, 0.6],
            },
        }
        m = classify_mechanism(result, _fixture_meta(), noop_result=noop)
        assert m["label"] == "observation_affected_source_but_not_command"

    def test_noise_command_changed_needs_live_replay(self) -> None:
        """Noise with changed command requires live replay for conclusion."""
        noop = _noop_result()
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.30},
            "first_observed_step": 5,
            "response_delay_steps": 0,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
            "action_proxy_changes": {
                "linear_velocity_changed": True,
                "events": ["a", "b", "c"],  # differs from noop
                "velocity_range": [0.0, 0.6],
            },
        }
        m = classify_mechanism(result, _fixture_meta(), noop_result=noop)
        assert m["label"] == "stored_action_proxy_prevents_live_conclusion"

    def test_noise_without_noop_needs_live_replay(self) -> None:
        """Noise without a noop comparison cannot support command-layer conclusions."""
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.30},
            "first_observed_step": 5,
            "response_delay_steps": 0,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
            "action_proxy_changes": {
                "linear_velocity_changed": True,
                "events": ["a", "b"],
                "velocity_range": [0.0, 0.6],
            },
        }
        m = classify_mechanism(result, _fixture_meta(), noop_result=None)
        assert m["label"] == "stored_action_proxy_prevents_live_conclusion"


class TestClassifyFarScenario:
    """Far-away scenarios get scenario_had_no_actionable_conflict."""

    def test_far_scenario(self) -> None:
        """Far-away scenario classifies as scenario_had_no_actionable_conflict."""
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.30},
            "first_observed_step": 5,
            "response_delay_steps": 0,
            "closest_distance_m": 3.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
        }
        m = classify_mechanism(result, _fixture_meta())
        assert m["label"] == "scenario_had_no_actionable_conflict"


class TestClassifyAllConditions:
    """classify_all_conditions attaches mechanism to every condition."""

    def test_all_conditions_get_mechanism(self) -> None:
        """Every condition receives a mechanism label from the allowed set."""
        envelope = _load_envelope()
        classified = classify_all_conditions(envelope["conditions"], _fixture_meta())
        for r in classified:
            assert "mechanism" in r
            assert "label" in r["mechanism"]
            assert r["mechanism"]["label"] in MECHANISM_LABELS

    def test_envelope_mechanism_distribution(self) -> None:
        """Run against the real envelope and verify expected labels."""
        envelope = _load_envelope()
        classified = classify_all_conditions(envelope["conditions"], _fixture_meta())
        labels = {r["condition"]: r["mechanism"]["label"] for r in classified}

        assert labels["noop"] == "diagnostic_only"
        assert labels["delay_only"] == "delay_shifted_stop_timing"
        assert labels["missed_detection_only"] == "occlusion_changed_first_actionable_frame"
        assert labels["occlusion_only"] == "occlusion_changed_first_actionable_frame"
        assert labels["combined"] == "occlusion_changed_first_actionable_frame"
        # low_noise: std=0.10 <= 0.15 threshold
        assert labels["low_noise"] == "noise_stayed_below_decision_threshold"
        # medium_noise: std=0.30 > 0.15 threshold
        assert labels["medium_noise"] == "observation_affected_source_but_not_command"


class TestInconclusiveFallback:
    """Conditions without active perturbation get inconclusive."""

    def test_no_perturbation_non_noop(self) -> None:
        """Non-noop condition with no active perturbation gets inconclusive."""
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.0},
            "first_observed_step": 5,
            "response_delay_steps": 0,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
            "action_proxy_changes": {
                "linear_velocity_changed": False,
                "events": [],
                "velocity_range": None,
            },
        }
        m = classify_mechanism(result, _fixture_meta())
        assert m["label"] == "inconclusive"

    def test_never_observed_no_signal(self) -> None:
        """Never observed with no signal gets inconclusive."""
        result = {
            "spec": {"is_noop": False, "position_noise_std_m": 0.0},
            "first_observed_step": None,
            "response_delay_steps": None,
            "closest_distance_m": 0.5,
            "missed_actor_observations_total": 0,
            "occluded_actor_observations_total": 0,
        }
        m = classify_mechanism(result, _fixture_meta())
        assert m["label"] == "inconclusive"
