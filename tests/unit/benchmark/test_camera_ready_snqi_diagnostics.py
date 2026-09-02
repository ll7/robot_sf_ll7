"""Fast-lane contracts for camera-ready SNQI diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import robot_sf.benchmark.camera_ready.campaign as camera_ready_campaign


def test_diagnostics_payload_preserves_non_failing_positioning() -> None:
    """The fast lane covers payload assembly and the non-advisory pass-through."""
    contract = SimpleNamespace(
        enabled=True,
        enforcement="warn",
        rank_alignment_warn_threshold=0.5,
        rank_alignment_fail_threshold=0.2,
        outcome_separation_warn_threshold=0.1,
        outcome_separation_fail_threshold=0.0,
        max_component_dominance_warn_threshold=0.8,
        max_component_dominance_fail_threshold=0.9,
    )
    cfg = SimpleNamespace(
        snqi_contract=contract,
        snqi_weights_path=None,
        snqi_baseline_path=None,
    )
    contract_eval = SimpleNamespace(
        status="pass",
        rank_alignment_spearman=0.9,
        outcome_separation=0.4,
        objective_score=0.7,
        dominant_component="progress",
        dominant_component_mean_abs=0.3,
    )
    positioning = {"recommendation": "retain_as_diagnostic"}
    positioning_results = {
        "calibration": {"weights": {"progress": 1.0}},
        "component_dominance": {},
        "component_correlations": {},
        "planner_ordering": [],
        "planner_ordering_basis": "stored_metrics.snqi",
        "weight_sensitivity": [],
        "positioning": positioning,
    }

    payload = camera_ready_campaign._build_snqi_diagnostics_payload(
        cfg,
        campaign_id="campaign",
        campaign_finished_at_utc="2026-09-02T00:00:00Z",
        contract_eval=contract_eval,
        positioning_results=positioning_results,
        configured_weights={"progress": 1.0},
        baseline_for_eval={},
        baseline_source="test",
        baseline_adjustments=0,
        weights_sha256="a" * 64,
        baseline_sha256="b" * 64,
    )

    assert payload["planner_ordering_basis"] == "stored_metrics.snqi"
    assert payload["positioning"] == positioning
    assert "release_claim_boundary" not in payload


def test_snqi_positioning_uses_complete_stored_field_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication diagnostics rank the same canonical SNQI field stored in episodes."""
    episodes = [
        {
            "planner_key": "legacy_winner",
            "kinematics": "differential_drive",
            "metrics": {"snqi": -0.8},
        },
        {
            "planner_key": "stored_field_winner",
            "kinematics": "differential_drive",
            "metrics": {"snqi": -0.1},
        },
    ]
    legacy_ordering = [
        {
            "planner_key": "legacy_winner",
            "kinematics": "differential_drive",
            "episode_count": 1,
            "mean_snqi": 0.9,
            "rank": 1,
        },
        {
            "planner_key": "stored_field_winner",
            "kinematics": "differential_drive",
            "episode_count": 1,
            "mean_snqi": 0.1,
            "rank": 2,
        },
    ]
    monkeypatch.setattr(
        camera_ready_campaign,
        "compute_planner_snqi_ordering",
        lambda *_args, **_kwargs: legacy_ordering,
    )
    monkeypatch.setattr(
        camera_ready_campaign,
        "calibrate_weights",
        lambda *_args, **_kwargs: {"weights": {}},
    )
    monkeypatch.setattr(
        camera_ready_campaign,
        "compute_component_dominance",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        camera_ready_campaign,
        "compute_component_correlations",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        camera_ready_campaign,
        "compute_weight_sensitivity",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        camera_ready_campaign,
        "build_positioning_recommendation",
        lambda *_args, **_kwargs: {},
    )

    result = camera_ready_campaign._compute_snqi_positioning(
        [],
        episodes,
        {},
        {},
        SimpleNamespace(snqi_contract=SimpleNamespace(calibration_seed=1, calibration_trials=1)),
    )

    assert result["planner_ordering_basis"] == "stored_metrics.snqi"
    assert [row["planner_key"] for row in result["planner_ordering"]] == [
        "stored_field_winner",
        "legacy_winner",
    ]


def test_failed_warn_boundary_overrides_operational_positioning() -> None:
    """Failed calibration cannot retain an operational-strengthening recommendation."""
    positioning = {
        "recommendation": "strengthen_as_operational_multi_objective_aggregation",
        "planner_ordering_informative": True,
        "caveats": [],
    }
    payload = {"positioning": positioning}

    result = camera_ready_campaign._apply_snqi_advisory_boundary(
        payload,
        positioning=positioning,
        contract_status="fail",
        contract_enforcement="warn",
    )

    assert result["positioning"]["recommendation"] == "retain_as_advisory_only_not_for_ranking"
    assert result["positioning"]["planner_ordering_informative"] is False
    assert result["release_claim_boundary"]["ranking_claims_admitted"] is False
