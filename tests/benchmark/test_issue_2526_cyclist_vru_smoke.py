"""Smoke tests for the issue #2526 cyclist-like VRU fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner import (
    _cyclist_like_vru_summary,
    _single_pedestrian_vru_metadata,
    _trace_pedestrians,
)
from scripts.tools.scenario_authoring import validate_scenario_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "configs/scenarios/single/issue_2526_cyclist_vru_smoke.yaml"


def _scenario_payload() -> dict:
    """Load the single cyclist-like VRU scenario payload."""
    payload = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]
    assert len(scenarios) == 1
    return scenarios[0]


def test_issue_2526_fixture_validates_and_declares_claim_boundary() -> None:
    """The authored cyclist-like VRU fixture should stay loadable and diagnostic-only."""
    report = validate_scenario_file(FIXTURE_PATH)
    scenario = _scenario_payload()
    vru = scenario["metadata"]["cyclist_like_vru"]

    assert report.ok is True
    assert scenario["name"] == "issue_2526_cyclist_vru_smoke"
    assert scenario["metadata"]["authoring"]["benchmark_evidence"] is False
    assert vru["status"] == "diagnostic_metadata_only"
    assert "does not prove" in vru["claim_boundary"]


def test_issue_2526_vru_metadata_reaches_trace_and_summary_fields() -> None:
    """Authored VRU speed/acceleration metadata should reach trace and summary payloads."""
    scenario = _scenario_payload()
    vru_metadata = _single_pedestrian_vru_metadata(scenario)

    assert len(vru_metadata) == 1
    assert vru_metadata[0] is not None
    assert vru_metadata[0]["pedestrian_id"] == "h1"
    assert vru_metadata[0]["actor_type"] == "cyclist_like_vru"
    assert vru_metadata[0]["speed_m_s"] == pytest.approx(4.5)
    assert vru_metadata[0]["acceleration_m_s2"] == pytest.approx(1.2)

    frames = _trace_pedestrians(
        np.array([[10.0, 11.0]], dtype=float),
        np.array([[10.0, 11.45]], dtype=float),
        0.1,
        vru_metadata=vru_metadata,
        robot_position=np.array([10.0, 10.0], dtype=float),
        robot_velocity=np.array([0.0, 0.0], dtype=float),
    )
    summary = _cyclist_like_vru_summary(scenario, vru_metadata)

    assert frames[0]["pedestrian_id"] == "h1"
    assert frames[0]["actor_type"] == "cyclist_like_vru"
    diagnostics = frames[0]["cyclist_like_vru"]
    assert diagnostics["configured_speed_m_s"] == pytest.approx(4.5)
    assert diagnostics["acceleration_m_s2"] == pytest.approx(1.2)
    assert diagnostics["relative_closing_speed_m_s"] > 0.0
    assert diagnostics["time_to_conflict_zone_s"] == pytest.approx(1.0 / 4.5)
    assert diagnostics["clearance_m"] == pytest.approx(0.35)
    assert diagnostics["pass_overtake_state"] == "approaching_conflict_zone"
    assert summary is not None
    assert summary["status"] == "diagnostic_metadata_only"
    assert summary["benchmark_evidence"] is False
    assert summary["trace_field_source"].endswith("pedestrians[]")


def test_issue_2526_vru_pass_state_marks_separating_after_pass() -> None:
    """The diagnostic pass/overtake state should distinguish a separating VRU."""
    scenario = _scenario_payload()
    vru_metadata = _single_pedestrian_vru_metadata(scenario)

    frames = _trace_pedestrians(
        np.array([[10.0, 11.0]], dtype=float),
        np.array([[10.0, 10.55]], dtype=float),
        0.1,
        vru_metadata=vru_metadata,
        robot_position=np.array([10.0, 10.0], dtype=float),
        robot_velocity=np.array([0.0, 0.0], dtype=float),
    )

    diagnostics = frames[0]["cyclist_like_vru"]
    assert diagnostics["relative_closing_speed_m_s"] < 0.0
    assert diagnostics["time_to_conflict_zone_s"] is None
    assert diagnostics["pass_overtake_state"] == "separating_after_pass"


def test_issue_2526_vru_metadata_float_falls_back_for_malformed_defaults() -> None:
    """Malformed authored numeric metadata should not crash diagnostic extraction."""
    scenario = {
        "name": "malformed_vru_metadata",
        "single_pedestrians": [
            {
                "id": "h1",
                "speed_m_s": "not-a-number",
                "metadata": {
                    "cyclist_like_vru": {
                        "actor_type": "cyclist_like_vru",
                        "speed_m_s": "also-bad",
                        "acceleration_m_s2": "nan",
                        "actor_radius_m": "invalid",
                        "robot_radius_m": "invalid",
                    }
                },
            }
        ],
    }

    vru_metadata = _single_pedestrian_vru_metadata(scenario)

    assert len(vru_metadata) == 1
    assert vru_metadata[0] is not None
    assert vru_metadata[0]["speed_m_s"] == 0.0
    assert vru_metadata[0]["acceleration_m_s2"] == 0.0
    assert vru_metadata[0]["actor_radius_m"] == pytest.approx(0.35)
    assert vru_metadata[0]["robot_radius_m"] == pytest.approx(0.3)
