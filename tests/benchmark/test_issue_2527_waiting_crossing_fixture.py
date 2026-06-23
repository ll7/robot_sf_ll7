"""Smoke tests for the issue #2527 waiting-then-crossing fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import (
    _episode_metadata_for_signal_metrics,
    _intent_conditioned_behavior_summary,
    _signal_state_for_metric_metadata,
    _signal_state_promotion_contract,
    _signal_state_proxy_wrapper,
    _single_pedestrian_intent_metadata,
    _trace_pedestrians,
)
from scripts.tools.scenario_authoring import validate_scenario_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "configs/scenarios/single/issue_2527_waiting_then_crossing.yaml"


def _scenario_payload() -> dict:
    """Load the single waiting-then-crossing scenario payload."""
    payload = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]
    assert len(scenarios) == 1
    return scenarios[0]


def test_issue_2527_fixture_validates_and_declares_claim_boundary() -> None:
    """The authored fixture should stay loadable and diagnostic-only."""
    report = validate_scenario_file(FIXTURE_PATH)
    scenario = _scenario_payload()
    intent = scenario["metadata"]["intent_conditioned_behavior"]
    signal_state = scenario["metadata"]["signal_state"]

    assert report.ok is True
    assert scenario["name"] == "issue_2527_waiting_then_crossing"
    assert scenario["metadata"]["authoring"]["benchmark_evidence"] is False
    assert intent["status"] == "diagnostic_metadata_only"
    assert "does not prove" in intent["claim_boundary"]
    assert signal_state["status"] == "proxy_diagnostic_only"
    assert signal_state["benchmark_evidence"] is False
    assert signal_state["planner_observable"] is False
    assert signal_state["phase_timeline"][0]["phase"] == "robot_green_pedestrian_dont_walk"
    assert signal_state["phase_timeline"][1]["phase"] == "pedestrian_walk_robot_red"


def test_issue_2527_intent_metadata_reaches_trace_and_summary_fields() -> None:
    """Authored waiting/crossing intent should be present in trace and summary metadata."""
    scenario = _scenario_payload()
    intent_metadata = _single_pedestrian_intent_metadata(scenario)

    assert len(intent_metadata) == 1
    assert intent_metadata[0]["pedestrian_id"] == "h1"
    assert intent_metadata[0]["intent_label"] == "waiting_then_crossing"
    assert intent_metadata[0]["intent_phases"] == ["waiting", "crossing"]
    assert intent_metadata[0]["behavior_parameters"]["wait_interval_s"] == [2.0]
    assert intent_metadata[0]["signal_state"]["signal_id"] == "issue_2564_wait_cross_proxy"

    waiting_frame = _trace_pedestrians(
        np.array([[14.0, 15.0]], dtype=float),
        np.array([[14.0, 15.0]], dtype=float),
        0.1,
        intent_metadata,
    )[0]
    crossing_frame = _trace_pedestrians(
        np.array([[14.0, 14.9]], dtype=float),
        np.array([[14.0, 15.0]], dtype=float),
        0.1,
        intent_metadata,
    )[0]
    summary = _intent_conditioned_behavior_summary(scenario, intent_metadata)

    assert waiting_frame["pedestrian_id"] == "h1"
    assert waiting_frame["intent_label"] == "waiting_then_crossing"
    assert waiting_frame["intent_phase"] == "waiting"
    assert waiting_frame["intent_source"] == "authored_scenario_metadata"
    assert "not data-grounded" in waiting_frame["claim_boundary"]
    assert waiting_frame["signal_state"]["phase"] == "robot_green_pedestrian_dont_walk"
    assert waiting_frame["signal_state"]["claim_boundary"] == "proxy_diagnostic"
    assert waiting_frame["signal_state"]["pedestrian_intent"] == {
        "intent_label": "waiting_then_crossing",
        "intent_phase": "waiting",
        "intent_source": "authored_scenario_metadata",
    }
    assert waiting_frame["signal_state"]["robot_stop_or_yield_expectation"] == "proceed_clear"
    assert {
        "signal_phase",
        "pedestrian_intent",
        "robot_stop_or_yield_expectation",
        "claim_boundary",
        "trace_fields_present",
    }.issubset(waiting_frame["signal_state"]["trace_fields_present"])
    assert waiting_frame["signal_state"]["pedestrian_right_of_way"] is False
    assert crossing_frame["intent_phase"] == "crossing"
    assert crossing_frame["signal_state"]["phase"] == "pedestrian_walk_robot_red"
    assert crossing_frame["signal_state"]["pedestrian_intent"]["intent_phase"] == "crossing"
    assert (
        crossing_frame["signal_state"]["robot_stop_or_yield_expectation"] == "yield_to_pedestrian"
    )
    assert crossing_frame["signal_state"]["pedestrian_right_of_way"] is True
    assert summary is not None
    assert summary["status"] == "diagnostic_metadata_only"
    assert summary["benchmark_evidence"] is False
    assert summary["trace_field_source"].endswith("pedestrians[]")
    assert summary["signal_state"]["status"] == "proxy_diagnostic_only"
    assert summary["signal_state"]["claim_boundary"] == "proxy_diagnostic"
    assert summary["signal_state"]["pedestrian_intent"] == {
        "intent_label": "waiting_then_crossing",
        "intent_phase": "waiting",
        "intent_source": "authored_scenario_metadata",
    }
    assert summary["signal_state"]["robot_stop_or_yield_expectation"] == "proceed_clear"
    assert "trace_fields_present" in summary["signal_state"]


def test_issue_2527_mixed_single_pedestrian_intent_metadata_stays_aligned() -> None:
    """Per-pedestrian opt-in should not drift onto a later single pedestrian."""
    scenario = _scenario_payload()
    first_ped = scenario["single_pedestrians"][0]
    scenario["metadata"].pop("intent_conditioned_behavior")
    scenario["single_pedestrians"].append(
        {
            "id": "h2",
            "goal": None,
            "trajectory": [[13.0, 17.5], [13.0, 4.0]],
        }
    )

    intent_metadata = _single_pedestrian_intent_metadata(scenario)
    frames = _trace_pedestrians(
        np.array([[14.0, 15.0], [13.0, 12.0]], dtype=float),
        np.array([[14.0, 15.0], [13.0, 12.0]], dtype=float),
        0.1,
        intent_metadata,
    )
    summary = _intent_conditioned_behavior_summary(scenario, intent_metadata)

    assert first_ped["id"] == "h1"
    assert len(intent_metadata) == 2
    assert frames[0]["pedestrian_id"] == "h1"
    assert "pedestrian_id" not in frames[1]
    assert summary is not None
    assert [ped["pedestrian_id"] for ped in summary["pedestrians"]] == ["h1"]


def test_signal_state_proxy_wrapper_exposes_bounded_diagnostic_fields() -> None:
    """The proxy wrapper must expose all five required fields with claim_boundary=proxy_diagnostic."""
    scenario = _scenario_payload()
    signal_state = scenario["metadata"]["signal_state"]

    waiting_proxy = _signal_state_proxy_wrapper(
        signal_state, "waiting", "waiting_then_crossing", "authored_scenario_metadata"
    )
    crossing_proxy = _signal_state_proxy_wrapper(
        signal_state, "crossing", "waiting_then_crossing", "authored_scenario_metadata"
    )

    assert waiting_proxy is not None
    assert crossing_proxy is not None

    for proxy in (waiting_proxy, crossing_proxy):
        assert "signal_phase" in proxy
        assert "pedestrian_intent" in proxy
        assert "robot_stop_or_yield_expectation" in proxy
        assert "trace_fields_present" in proxy
        assert proxy["claim_boundary"] == "proxy_diagnostic"
        assert proxy["contract_state"] == "proxy_diagnostic"
        assert proxy["planner_consumed_fields"] == []
        assert "phase" in proxy["recorded_only_fields"]
        assert "phase_remaining_s" in proxy["promotion_required_fields"]
        assert proxy["benchmark_evidence"] is False
        assert "do_not_count_as_signalized_benchmark_evidence" in proxy["fail_closed_reason"]
        assert "intent_label" in proxy["pedestrian_intent"]
        assert "intent_phase" in proxy["pedestrian_intent"]
        assert "intent_source" in proxy["pedestrian_intent"]

    assert waiting_proxy["signal_phase"] == "robot_green_pedestrian_dont_walk"
    assert waiting_proxy["robot_stop_or_yield_expectation"] == "proceed_clear"
    assert waiting_proxy["pedestrian_intent"]["intent_phase"] == "waiting"

    assert crossing_proxy["signal_phase"] == "pedestrian_walk_robot_red"
    assert crossing_proxy["robot_stop_or_yield_expectation"] == "yield_to_pedestrian"
    assert crossing_proxy["pedestrian_intent"]["intent_phase"] == "crossing"

    assert "signal_state" in waiting_proxy["trace_fields_present"]
    assert "pedestrian_intent" in waiting_proxy["trace_fields_present"]


def test_signal_state_contract_fails_closed_when_unavailable() -> None:
    """Absent signal semantics are unavailable and cannot enter benchmark denominators."""
    contract = _signal_state_promotion_contract(None)

    assert contract["contract_state"] == "unavailable"
    assert contract["planner_consumed_fields"] == []
    assert contract["recorded_only_fields"] == []
    assert "phase_remaining_s" in contract["promotion_required_fields"]
    assert contract["fail_closed_reason"] == "signal_state_metadata_absent"
    assert contract["benchmark_evidence"] is False


def test_proxy_signal_state_cannot_be_interpreted_as_planner_observable() -> None:
    """Proxy fields remain recorded-only even if a fixture accidentally sets visibility flags."""
    scenario = _scenario_payload()
    spoofed_proxy = dict(scenario["metadata"]["signal_state"])
    spoofed_proxy["planner_observable"] = True
    spoofed_proxy["benchmark_evidence"] = True
    spoofed_proxy["observation_mode"] = "planner_observable"

    contract = _signal_state_promotion_contract(spoofed_proxy)

    assert contract["contract_state"] == "proxy_diagnostic"
    assert contract["planner_consumed_fields"] == []
    assert "phase" in contract["recorded_only_fields"]
    assert "phase_remaining_s" in contract["promotion_required_fields"]
    assert contract["benchmark_evidence"] is False


def test_signal_state_string_false_flags_fail_closed() -> None:
    """String-like false values must not promote proxy metadata into benchmark evidence."""
    contract = _signal_state_promotion_contract(
        {
            "schema_version": "signal-state-observable.v1",
            "status": "planner_observable_signal_state",
            "observation_mode": "planner_observable",
            "planner_observable": "false",
            "benchmark_evidence": "0",
        }
    )

    assert contract["contract_state"] == "proxy_diagnostic"
    assert contract["benchmark_evidence"] is False

    proxy = _signal_state_proxy_wrapper(
        {
            "phase_timeline": [
                {
                    "phase": "pedestrian_walk_robot_red",
                    "intent_phase": "crossing",
                    "robot_right_of_way": "false",
                    "pedestrian_right_of_way": np.bool_(True),
                    "legality_state": "pedestrian_crossing_allowed",
                }
            ],
            "planner_observable": "false",
            "benchmark_evidence": "0",
        },
        "crossing",
        "waiting_then_crossing",
        "authored_scenario_metadata",
    )

    assert proxy is not None
    assert proxy["robot_right_of_way"] is False
    assert proxy["pedestrian_right_of_way"] is True
    assert proxy["planner_observable"] is False
    assert proxy["benchmark_evidence"] is False


def test_explicit_observable_signal_state_names_planner_consumed_fields() -> None:
    """Only the explicit observable schema/status can promote signal fields to planner inputs."""
    contract = _signal_state_promotion_contract(
        {
            "schema_version": "signal-state-observable.v1",
            "status": "planner_observable_signal_state",
            "observation_mode": "planner_observable",
            "planner_observable": True,
            "benchmark_evidence": True,
        }
    )

    assert contract["contract_state"] == "planner_observable"
    assert "phase_remaining_s" in contract["planner_consumed_fields"]
    assert contract["recorded_only_fields"] == []
    assert contract["promotion_required_fields"] == []
    assert contract["fail_closed_reason"] == ""
    assert contract["benchmark_evidence"] is True


def test_proxy_signal_state_metric_metadata_stays_denominator_excluded() -> None:
    """Metric metadata should preserve proxy exclusion even when trace metadata exists."""
    scenario = _scenario_payload()

    metric_metadata = _episode_metadata_for_signal_metrics(scenario)

    assert metric_metadata == {
        "signal_state": {
            "contract_state": "proxy_diagnostic",
            "benchmark_evidence": False,
        }
    }


def test_metric_metadata_carries_enabled_rollover_stability_config() -> None:
    """Map-runner episode metadata should carry opt-in TWV rollover instrumentation."""
    scenario = {"metadata": {"rollover_stability": {"enabled": True, "yaw_rate": 3.0}}}

    metric_metadata = _episode_metadata_for_signal_metrics(scenario)

    assert metric_metadata == {"rollover_stability": {"enabled": True, "yaw_rate": 3.0}}


def test_metric_metadata_omits_disabled_rollover_stability_config() -> None:
    """Disabled TWV rollover instrumentation must leave default metric rows unchanged."""
    scenario = {"metadata": {"rollover_stability": {"enabled": False, "yaw_rate": 3.0}}}

    metric_metadata = _episode_metadata_for_signal_metrics(scenario)

    assert metric_metadata is None


def test_metric_metadata_carries_enabled_clear_tracking_config() -> None:
    """Map-runner episode metadata should carry opt-in CLEAR tracking instrumentation."""
    config = {
        "enabled": True,
        "ground_truth_count": 2,
        "detection_count": 1,
        "missed_detection_count": 1,
        "false_positive_count": 0,
        "id_switch_count": 0,
        "mota": 0.5,
        "motp_m": 0.25,
        "motp_match_count": 1,
    }
    scenario = {"metadata": {"clear_tracking_uncertainty": config}}

    metric_metadata = _episode_metadata_for_signal_metrics(scenario)

    assert metric_metadata == {"clear_tracking_uncertainty": config}


def test_metric_metadata_omits_disabled_clear_tracking_config() -> None:
    """Disabled CLEAR tracking instrumentation must leave default metric rows unchanged."""
    scenario = {"metadata": {"clear_tracking_uncertainty": {"enabled": False, "mota": 0.5}}}

    metric_metadata = _episode_metadata_for_signal_metrics(scenario)

    assert metric_metadata is None


def test_observable_signal_state_metric_metadata_carries_required_fields() -> None:
    """Metric metadata should include only explicit observable benchmark signal fields."""
    signal_state = {
        "schema_version": "signal-state-observable.v1",
        "status": "planner_observable_signal_state",
        "observation_mode": "planner_observable",
        "planner_observable": True,
        "benchmark_evidence": True,
        "timeline": [{"state": "green", "duration": 1.0}],
        "stop_line": [[0.0, -1.0], [0.0, 1.0]],
        "crosswalk_polygon": [[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]],
    }

    metric_state = _signal_state_for_metric_metadata(signal_state)

    assert metric_state == {
        "contract_state": "planner_observable",
        "benchmark_evidence": True,
        "timeline": signal_state["timeline"],
        "stop_line": signal_state["stop_line"],
        "crosswalk_polygon": signal_state["crosswalk_polygon"],
    }


def test_intent_summary_handles_missing_or_empty_intent_phases() -> None:
    """Malformed phase metadata should fall back to unknown instead of crashing summaries."""
    scenario = _scenario_payload()
    signal_state = scenario["metadata"]["signal_state"]

    for malformed_phases in (None, []):
        summary = _intent_conditioned_behavior_summary(
            scenario,
            [
                {
                    "pedestrian_id": "h1",
                    "intent_label": "waiting_then_crossing",
                    "intent_phases": malformed_phases,
                    "intent_source": "authored_scenario_metadata",
                    "claim_boundary": "diagnostic_metadata_only",
                    "behavior_parameters": {},
                    "signal_state": signal_state,
                }
            ],
        )

        assert summary is not None
        assert summary["signal_state"]["pedestrian_intent"]["intent_phase"] == "unknown"
        assert summary["signal_state"]["signal_phase"] == "robot_green_pedestrian_dont_walk"
