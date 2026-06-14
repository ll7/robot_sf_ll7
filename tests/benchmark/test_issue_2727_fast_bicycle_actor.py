"""Smoke tests for issue #2727 fast-bicycle actor fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner import (
    _fast_bicycle_actor_summary,
    _single_pedestrian_vru_metadata,
    _trace_pedestrians,
)
from robot_sf.benchmark.pedestrian_forecast import is_pedestrian_actor
from scripts.tools.scenario_authoring import validate_scenario_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "configs/scenarios/single/issue_2727_fast_bicycle_dynamic_actor.yaml"
EXPECTED_CASES = {
    "issue_2727_fast_bicycle_crossing": "crossing",
    "issue_2727_fast_bicycle_overtaking": "overtaking",
    "issue_2727_fast_bicycle_parallel_lane": "parallel_lane",
}


def _scenario_payloads() -> list[dict]:
    """Load the fast-bicycle fixture scenarios."""
    payload = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    return payload["scenarios"]


def test_issue_2727_fixture_validates_and_declares_three_cases() -> None:
    """The authored fast-bicycle fixture matrix should stay diagnostic-only."""
    report = validate_scenario_file(FIXTURE_PATH)
    scenarios = _scenario_payloads()

    assert report.ok is True
    assert {scenario["name"] for scenario in scenarios} == set(EXPECTED_CASES)
    for scenario in scenarios:
        case = EXPECTED_CASES[scenario["name"]]
        bicycle = scenario["metadata"]["fast_bicycle_actor"]
        assert bicycle["case"] == case
        assert bicycle["status"] == "diagnostic_metadata_only"
        assert bicycle["benchmark_evidence"] is False
        assert "does not prove" in bicycle["claim_boundary"]


@pytest.mark.parametrize("scenario", _scenario_payloads(), ids=lambda item: item["name"])
def test_issue_2727_bicycle_metadata_reaches_trace_payload(scenario: dict) -> None:
    """Fast-bicycle actor metadata should reach trace frames under a distinct key."""
    vru_metadata = _single_pedestrian_vru_metadata(scenario)

    assert len(vru_metadata) == 1
    assert vru_metadata[0] is not None
    assert vru_metadata[0]["actor_type"] == "bicycle"
    assert vru_metadata[0]["diagnostic_payload_key"] == "fast_bicycle_actor"
    assert is_pedestrian_actor(vru_metadata[0]["actor_type"]) is False

    frames = _trace_pedestrians(
        np.array([[10.0, 11.0]], dtype=float),
        np.array([[10.0, 11.6]], dtype=float),
        0.1,
        vru_metadata=vru_metadata,
        robot_position=np.array([10.0, 10.0], dtype=float),
        robot_velocity=np.array([0.0, 0.0], dtype=float),
    )
    summary = _fast_bicycle_actor_summary(scenario, vru_metadata)

    assert frames[0]["pedestrian_id"] == vru_metadata[0]["pedestrian_id"]
    assert frames[0]["actor_type"] == "bicycle"
    assert "cyclist_like_vru" not in frames[0]
    diagnostics = frames[0]["fast_bicycle_actor"]
    assert diagnostics["configured_speed_m_s"] == pytest.approx(vru_metadata[0]["speed_m_s"])
    assert diagnostics["acceleration_m_s2"] == pytest.approx(vru_metadata[0]["acceleration_m_s2"])
    assert diagnostics["relative_closing_speed_m_s"] > 0.0
    assert diagnostics["time_to_conflict_zone_s"] is not None
    assert diagnostics["clearance_m"] == pytest.approx(0.25)
    assert diagnostics["pass_overtake_state"] == "approaching_conflict_zone"
    assert summary is not None
    assert summary["schema_version"] == "fast-bicycle-actor-summary.v1"
    assert summary["status"] == "diagnostic_metadata_only"
    assert summary["benchmark_evidence"] is False
    assert summary["trace_field_source"].endswith("pedestrians[]")


def test_issue_2727_existing_cyclist_like_payload_key_is_preserved() -> None:
    """The new fast-bicycle key should not rename existing cyclist-like fixtures."""
    fixture = REPO_ROOT / "configs/scenarios/single/issue_2526_cyclist_vru_smoke.yaml"
    scenario = yaml.safe_load(fixture.read_text(encoding="utf-8"))["scenarios"][0]
    vru_metadata = _single_pedestrian_vru_metadata(scenario)

    frames = _trace_pedestrians(
        np.array([[10.0, 11.0]], dtype=float),
        np.array([[10.0, 11.45]], dtype=float),
        0.1,
        vru_metadata=vru_metadata,
        robot_position=np.array([10.0, 10.0], dtype=float),
        robot_velocity=np.array([0.0, 0.0], dtype=float),
    )

    assert vru_metadata[0] is not None
    assert vru_metadata[0]["diagnostic_payload_key"] == "cyclist_like_vru"
    assert frames[0]["actor_type"] == "cyclist_like_vru"
    assert "cyclist_like_vru" in frames[0]
    assert "fast_bicycle_actor" not in frames[0]
