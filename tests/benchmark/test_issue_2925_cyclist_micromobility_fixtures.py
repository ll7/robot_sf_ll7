"""Smoke tests for issue #2925 cyclist and fast-micromobility proxy fixtures."""

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
from scripts.tools.scenario_authoring import validate_scenario_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "configs/scenarios/single/issue_2925_cyclist_micromobility_actor_fixtures.yaml"
)
EXPECTED_INTERACTION_CLASSES = {
    "crossing",
    "overtaking",
    "same_direction",
    "opposite_direction",
    "occluded_emergence",
}
REQUIRED_METRICS = {
    "relative_closing_speed_m_s",
    "time_to_conflict_zone_s",
    "clearance_m",
    "pass_overtake_state",
    "interaction_class",
}


def _scenario_payloads() -> list[dict]:
    """Load the #2925 cyclist/micromobility proxy fixture scenarios."""
    payload = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    return payload["scenarios"]


def test_issue_2925_fixture_validates_and_covers_required_interactions() -> None:
    """The proxy fixture matrix should stay loadable, bounded, and diagnostic-only."""
    report = validate_scenario_file(FIXTURE_PATH)
    scenarios = _scenario_payloads()

    assert report.ok is True
    assert len(scenarios) == 5
    assert {
        scenario["metadata"]["cyclist_micromobility_actor"]["interaction_class"]
        for scenario in scenarios
    } == EXPECTED_INTERACTION_CLASSES
    for scenario in scenarios:
        actor = scenario["metadata"]["cyclist_micromobility_actor"]
        pedestrian_actor = scenario["single_pedestrians"][0]["metadata"]["fast_bicycle_actor"]
        assert actor["status"] == "diagnostic_metadata_only"
        assert actor["benchmark_evidence"] is False
        assert "not a full cyclist dynamics model" in actor["claim_boundary"]
        assert set(actor["required_metrics"]) == REQUIRED_METRICS
        assert 4.0 <= float(pedestrian_actor["speed_m_s"]) <= 6.5
        assert set(pedestrian_actor["diagnostic_metric_subset"]) == REQUIRED_METRICS


@pytest.mark.parametrize("scenario", _scenario_payloads(), ids=lambda item: item["name"])
def test_issue_2925_trace_payload_exposes_proxy_metrics_and_class(scenario: dict) -> None:
    """Cyclist proxy trace payloads should expose class, closing speed, TTC, and clearance."""
    vru_metadata = _single_pedestrian_vru_metadata(scenario)

    assert len(vru_metadata) == 1
    assert vru_metadata[0] is not None
    assert vru_metadata[0]["diagnostic_payload_key"] == "fast_bicycle_actor"
    assert vru_metadata[0]["actor_type"] == "bicycle"
    assert vru_metadata[0]["interaction_class"] in EXPECTED_INTERACTION_CLASSES
    assert set(vru_metadata[0]["diagnostic_metric_subset"]) == REQUIRED_METRICS

    frames = _trace_pedestrians(
        np.array([[10.0, 11.0]], dtype=float),
        np.array([[10.0, 11.6]], dtype=float),
        0.1,
        vru_metadata=vru_metadata,
        robot_position=np.array([10.0, 10.0], dtype=float),
        robot_velocity=np.array([0.0, 0.0], dtype=float),
    )
    summary = _fast_bicycle_actor_summary(scenario, vru_metadata)

    frame = frames[0]
    diagnostics = frame["fast_bicycle_actor"]
    assert frame["interaction_class"] == vru_metadata[0]["interaction_class"]
    assert diagnostics["configured_speed_m_s"] == pytest.approx(vru_metadata[0]["speed_m_s"])
    assert diagnostics["relative_closing_speed_m_s"] > 0.0
    assert diagnostics["time_to_conflict_zone_s"] is not None
    assert diagnostics["clearance_m"] == pytest.approx(0.25)
    assert diagnostics["pass_overtake_state"] == "approaching_conflict_zone"
    assert summary is not None
    assert summary["schema_version"] == "fast-bicycle-actor-summary.v1"
    assert summary["status"] == "diagnostic_metadata_only"
    assert summary["benchmark_evidence"] is False


def test_issue_2925_docs_avoid_cyclist_realism_claims() -> None:
    """Fixture claim boundaries should explicitly reject realism and benchmark claims."""
    scenarios = _scenario_payloads()

    for scenario in scenarios:
        actor = scenario["metadata"]["cyclist_micromobility_actor"]
        claim_boundary = actor["claim_boundary"].lower()
        assert "not a full cyclist dynamics model" in claim_boundary
        assert "calibrated cyclist realism" in claim_boundary
        assert "planner-ranking benchmark evidence" in claim_boundary
