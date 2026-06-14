"""Tests for the mixed-scenario coverage matrix generator (issue #2766)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import generate_mixed_scenario_matrix as gen

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_LEDGER = {
    "schema_version": "dissertation_evidence_ledger.v2",
    "rows": [
        {
            "area": "topology_guidance",
            "artifact_status": "current",
            "evidence_tier": "diagnostic",
            "allowed_wording": "test wording",
            "caveat": "test caveat",
        },
        {
            "area": "signalized_behavior",
            "artifact_status": "current",
            "evidence_tier": "diagnostic",
            "allowed_wording": "test",
            "caveat": "test",
        },
        {
            "area": "prediction",
            "artifact_status": "current",
            "evidence_tier": "diagnostic",
            "allowed_wording": "test",
            "caveat": "test",
        },
    ],
    "stale_artifact_summary": [
        {
            "artifact_id": "tab_issue_1023_campaign_table",
            "state": "non-claimable",
            "reason": "Missing payload file",
        },
    ],
}

SIGNAL_SUMMARY = {
    "eligible_rows": [
        {
            "episode_id": "ep1",
            "row_type": "red_required_stop",
            "planner_observable": True,
            "signal_metrics_denominator": 1,
        },
        {
            "episode_id": "ep2",
            "row_type": "green_proceed",
            "planner_observable": True,
            "signal_metrics_denominator": 1,
        },
    ],
    "excluded_rows": [
        {
            "episode_id": "ep3",
            "row_type": "unavailable_no_claim",
            "planner_observable": False,
            "signal_metrics_denominator": 0,
        },
    ],
}

OBS_SUMMARY = {
    "summary": {
        "classifications": {
            "noop": "diagnostic_only",
            "low_noise": "diagnostic_only",
            "medium_noise": "diagnostic_only",
            "missed_detection_only": "scenario_too_weak",
            "occlusion_only": "scenario_too_weak",
            "delay_only": "robustness_evidence",
            "combined": "scenario_too_weak",
        }
    },
    "conditions": [
        {"condition": "noop", "first_observed_step": 5},
        {"condition": "delay_only", "first_observed_step": 7},
        {"condition": "missed_detection_only", "first_observed_step": None},
    ],
}

CV_FORECAST = {
    "results_by_trace": [
        {
            "family": "corridor_interaction",
            "status": "evaluated",
            "label": "default_social_force",
        },
        {
            "family": "bottleneck",
            "status": "limited_no_pedestrian_motion",
            "label": "minimal_fixture",
        },
    ],
}


def _sources(tmp_path: Path) -> gen.SourceInputs:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(MINIMAL_LEDGER), encoding="utf-8")
    signal_path = tmp_path / "signal.json"
    signal_path.write_text(json.dumps(SIGNAL_SUMMARY), encoding="utf-8")
    obs_path = tmp_path / "obs.json"
    obs_path.write_text(json.dumps(OBS_SUMMARY), encoding="utf-8")
    cv_path = tmp_path / "cv.json"
    cv_path.write_text(json.dumps(CV_FORECAST), encoding="utf-8")
    gap_path = tmp_path / "gap.json"
    gap_path.write_text("{}", encoding="utf-8")
    neg_dir = tmp_path / "neg"
    neg_dir.mkdir()
    return gen.SourceInputs(
        ledger=ledger_path,
        signal_summary=signal_path,
        obs_noise_summary=obs_path,
        cv_forecast=cv_path,
        gap_report=gap_path,
        negative_result_dir=neg_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_matrix_rows_count() -> None:
    """One row per canonical scenario slice."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    assert len(rows) == len(gen.SCENARIO_SLICES)


def test_build_matrix_rows_columns() -> None:
    """Every row has all evidence columns with status and reason."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    for row in rows:
        assert "scenario_slice" in row
        for col in gen.EVIDENCE_COLUMNS:
            assert col in row
            assert "status" in row[col]
            assert "reason" in row[col]


def test_topology_hard_slices() -> None:
    """Hard slices report diagnostic_only with horizon_exhausted."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    for scenario in ("t_intersection", "doorway", "bottleneck"):
        row = next(r for r in rows if r["scenario_slice"] == scenario)
        assert row["topology_reselection"]["status"] == "diagnostic_only"
        assert "horizon_exhausted" in row["topology_reselection"]["reason"]


def test_signalized_crossing_available() -> None:
    """Signalized crossing has available signal metrics with partial denominator."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "signalized_crossing")
    assert row["signal_compliance_metrics"]["status"] == "available"
    assert row["denominator_health"]["status"] == "partial"


def test_signalized_crossing_no_summary() -> None:
    """Missing signal summary degrades signalized crossing to unavailable."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=None,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "signalized_crossing")
    assert row["signal_compliance_metrics"]["status"] == "unavailable"
    assert row["denominator_health"]["status"] == "missing"


def test_observation_occluded_emergence() -> None:
    """Occluded emergence reports partial_robustness observation evidence."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "occluded_emergence")
    assert row["observation_perturbation"]["status"] == "partial_robustness"
    assert row["denominator_health"]["status"] == "partial"


def test_dense_pedestrian_observation() -> None:
    """Dense pedestrian has diagnostic_only observation evidence."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "dense_pedestrian")
    assert row["observation_perturbation"]["status"] == "diagnostic_only"


def test_prediction_cv_forecast_evaluated() -> None:
    """Corridor interaction CV forecast is diagnostic_only when evaluated."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "corridor_interaction")
    assert row["prediction_baseline"]["status"] == "diagnostic_only"


def test_prediction_cv_forecast_limited() -> None:
    """Bottleneck CV forecast is limited_no_motion when pedestrians have no velocity."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "bottleneck")
    assert row["prediction_baseline"]["status"] == "limited_no_motion"


def test_staleness_corridor() -> None:
    """Corridor interaction is stale due to missing issue-1023 payload files."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "corridor_interaction")
    assert row["stale_current_status"]["status"] == "stale"


def test_staleness_current_negative() -> None:
    """t_intersection and doorway are current_negative from NR-001 candidates."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
        negative_result_available=True,
    )
    for scenario in ("t_intersection", "doorway"):
        row = next(r for r in rows if r["scenario_slice"] == scenario)
        assert row["stale_current_status"]["status"] == "current_negative"


def test_staleness_missing_negative_result_source() -> None:
    """Hard-slice negative status is unavailable without source evidence."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
        negative_result_available=False,
    )

    row = next(r for r in rows if r["scenario_slice"] == "t_intersection")
    assert row["stale_current_status"]["status"] == "unavailable"
    assert "Negative-result candidate source is missing" in row["stale_current_status"]["reason"]


def test_claim_eligibility_not_eligible_when_blockers() -> None:
    """Corridor interaction is not_eligible when blockers are present."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    corridor = next(r for r in rows if r["scenario_slice"] == "corridor_interaction")
    assert corridor["claim_eligibility"]["status"] == "not_eligible"
    assert "Blockers:" in corridor["claim_eligibility"]["reason"]


def test_claim_eligibility_signalized_has_observation_blocker() -> None:
    """Signalized crossing is not_eligible because observation is unavailable."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    row = next(r for r in rows if r["scenario_slice"] == "signalized_crossing")
    assert row["claim_eligibility"]["status"] == "not_eligible"
    assert "observation: unavailable" in row["claim_eligibility"]["reason"]


def test_zero_denominator_blocks_claim_eligibility() -> None:
    """Zero denominator health is a claim blocker."""
    claim = gen._claim_cell(
        "signalized_crossing",
        topology={"status": "diagnostic_only", "reason": "test"},
        signal={"status": "diagnostic_only", "reason": "test"},
        prediction={"status": "diagnostic_only", "reason": "test"},
        observation={"status": "diagnostic_only", "reason": "test"},
        denominator={"status": "zero", "reason": "test"},
        stale={"status": "current", "reason": "test"},
    )

    assert claim["status"] == "not_eligible"
    assert "denominator: zero" in claim["reason"]


def test_missing_ledger_fails_closed() -> None:
    """Missing inputs still produce fail-closed rows with statuses."""
    rows = gen.build_matrix_rows(
        ledger=None,
        signal_summary=None,
        obs_summary=None,
        cv_forecast=None,
    )
    assert len(rows) == len(gen.SCENARIO_SLICES)
    for row in rows:
        for col in gen.EVIDENCE_COLUMNS:
            assert "status" in row[col]


def test_malformed_json_sources_fail_closed(tmp_path: Path) -> None:
    """Malformed, null, and non-object JSON inputs generate unavailable rows."""
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{oops", encoding="utf-8")
    signal = tmp_path / "signal.json"
    signal.write_text("[]", encoding="utf-8")
    obs = tmp_path / "obs.json"
    obs.write_text("null", encoding="utf-8")
    cv = tmp_path / "cv.json"
    cv.write_text('"not-object"', encoding="utf-8")
    sources = gen.SourceInputs(
        ledger=ledger,
        signal_summary=signal,
        obs_noise_summary=obs,
        cv_forecast=cv,
        gap_report=tmp_path / "missing_gap.json",
        negative_result_dir=tmp_path / "missing_negative",
    )
    md_path, json_path = gen.generate_matrix(sources=sources, output_dir=tmp_path / "out")

    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["scenario_count"] == len(gen.SCENARIO_SLICES)
    assert payload["status_counts"]["unavailable"] >= 1


def test_null_and_mixed_json_lists_fail_closed() -> None:
    """Null list fields and non-dict list items are ignored safely."""
    ledger = {
        "rows": None,
        "stale_artifact_summary": None,
    }
    signal_summary = {
        "eligible_rows": [None, "bad", {"signal_metrics_denominator": 1}],
        "excluded_rows": None,
    }
    obs_summary = {
        "summary": {"classifications": None},
        "conditions": [None, {"first_observed_step": 3}, "bad"],
    }
    cv_forecast = {
        "results_by_trace": [None, {"family": "corridor_interaction", "status": "evaluated"}]
    }

    rows = gen.build_matrix_rows(
        ledger=ledger,
        signal_summary=signal_summary,
        obs_summary=obs_summary,
        cv_forecast=cv_forecast,
    )

    signalized = next(r for r in rows if r["scenario_slice"] == "signalized_crossing")
    occluded = next(r for r in rows if r["scenario_slice"] == "occluded_emergence")
    assert signalized["denominator_health"]["status"] == "partial"
    assert occluded["denominator_health"]["status"] == "partial"


def test_build_markdown_contains_all_scenarios() -> None:
    """Markdown output lists every scenario and conservative rules."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    md = gen.build_markdown(rows, generated_at="2026-06-14T00:00:00+00:00")
    for scenario in gen.SCENARIO_SLICES:
        assert f"### {scenario}" in md
    assert "Conservative Rules Applied" in md


def test_build_summary_fields() -> None:
    """Summary output records schema, dimensions, counts, and boundary."""
    sources = gen.SourceInputs(
        ledger=Path("a"),
        signal_summary=Path("b"),
        obs_noise_summary=Path("c"),
        cv_forecast=Path("d"),
        gap_report=Path("e"),
        negative_result_dir=Path("f"),
    )
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    summary = gen.build_summary(
        generated_at="2026-06-14T00:00:00+00:00",
        sources=sources,
        rows=rows,
    )
    assert summary["schema_version"] == "mixed_scenario_matrix.v1"
    assert summary["issue"] == 2766
    assert summary["scenario_count"] == len(gen.SCENARIO_SLICES)
    assert summary["column_count"] == len(gen.EVIDENCE_COLUMNS)
    assert summary["total_cells"] == summary["scenario_count"] * summary["column_count"]
    assert "status_counts" in summary
    assert "claim_boundary" in summary


def test_diagnostic_only_rows_are_not_counted_eligible() -> None:
    """Summary eligible count only counts explicit eligible statuses."""
    sources = gen.SourceInputs(
        ledger=Path("a"),
        signal_summary=Path("b"),
        obs_noise_summary=Path("c"),
        cv_forecast=Path("d"),
        gap_report=Path("e"),
        negative_result_dir=Path("f"),
    )
    rows = [
        {
            "scenario_slice": "synthetic",
            "topology_reselection": {"status": "diagnostic_only", "reason": "test"},
            "signal_compliance_metrics": {"status": "diagnostic_only", "reason": "test"},
            "prediction_baseline": {"status": "diagnostic_only", "reason": "test"},
            "observation_perturbation": {"status": "diagnostic_only", "reason": "test"},
            "denominator_health": {"status": "not_applicable", "reason": "test"},
            "stale_current_status": {"status": "current", "reason": "test"},
            "claim_eligibility": {"status": "diagnostic_only", "reason": "test"},
        }
    ]

    summary = gen.build_summary(
        generated_at="2026-06-14T00:00:00+00:00",
        sources=sources,
        rows=rows,
    )

    assert summary["eligible_scenario_count"] == 0
    assert summary["not_eligible_scenario_count"] == 1


def test_generate_matrix_creates_files(tmp_path: Path) -> None:
    """Generator writes Markdown and JSON artifacts."""
    sources = _sources(tmp_path)
    out_dir = tmp_path / "output"
    md_path, json_path = gen.generate_matrix(sources=sources, output_dir=out_dir)
    assert md_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "mixed_scenario_matrix.v1"
    assert payload["scenario_count"] == len(gen.SCENARIO_SLICES)


def test_no_overclaims_in_all_cells() -> None:
    """Every cell must be conservative -- no positive benchmark/paper overclaims."""
    rows = gen.build_matrix_rows(
        ledger=MINIMAL_LEDGER,
        signal_summary=SIGNAL_SUMMARY,
        obs_summary=OBS_SUMMARY,
        cv_forecast=CV_FORECAST,
    )
    overclaim_phrases = [
        "is benchmark evidence",
        "is paper-facing",
        "proven by",
        "establishes benchmark",
        "validates benchmark",
    ]
    for row in rows:
        for col in gen.EVIDENCE_COLUMNS:
            cell = row[col]
            reason_lower = cell["reason"].lower()
            for phrase in overclaim_phrases:
                assert phrase not in reason_lower, (
                    f"Overclaim in {row['scenario_slice']}/{col}: {phrase}"
                )
