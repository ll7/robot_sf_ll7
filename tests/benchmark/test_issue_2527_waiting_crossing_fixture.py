"""Smoke tests for the issue #2527 waiting-then-crossing fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import (
    _intent_conditioned_behavior_summary,
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
    assert waiting_frame["signal_state"]["pedestrian_right_of_way"] is False
    assert crossing_frame["intent_phase"] == "crossing"
    assert crossing_frame["signal_state"]["phase"] == "pedestrian_walk_robot_red"
    assert crossing_frame["signal_state"]["pedestrian_right_of_way"] is True
    assert summary is not None
    assert summary["status"] == "diagnostic_metadata_only"
    assert summary["benchmark_evidence"] is False
    assert summary["trace_field_source"].endswith("pedestrians[]")
    assert summary["signal_state"]["status"] == "proxy_diagnostic_only"
    assert summary["signal_state"]["trace_fields"] == [
        "pedestrians[].signal_state",
        "pedestrians[].intent_phase",
    ]


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
