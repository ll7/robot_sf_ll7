"""Tests for the horizon x timestep denominator-health report."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/build_horizon_timestep_denominator_report.py"


def _load_script_module():
    """Load the denominator report script as a module for testing."""
    spec = importlib.util.spec_from_file_location(
        "build_horizon_timestep_denominator_report", _SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {_SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_horizon_timestep_denominator_report"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script_module()
HORIZON_LADDER_S = _mod.HORIZON_LADDER_S
DT_LADDER_S = _mod.DT_LADDER_S
TRACE_CANDIDATES = _mod.TRACE_CANDIDATES
MISSINGNESS_CATEGORIES = _mod.MISSINGNESS_CATEGORIES
COVERAGE_TARGET_FRACTION = _mod.COVERAGE_TARGET_FRACTION
build_denominator_report = _mod.build_denominator_report
_classify_missingness = _mod._classify_missingness
_compute_fixture_proposal = _mod._compute_fixture_proposal


def test_category_totals_sum_to_expected_matrix_size() -> None:
    """The category totals must sum to horizon_count * dt_count * trace_count."""
    report = build_denominator_report(parent_issue=2837, issue=2903)
    expected_total = len(HORIZON_LADDER_S) * len(DT_LADDER_S) * len(TRACE_CANDIDATES)
    assert report["expected_total_cells"] == expected_total
    assert report["category_totals_valid"] is True
    assert sum(report["category_totals"].values()) == expected_total


def test_missingness_categories_include_required_reasons() -> None:
    """The contract-required missingness categories are present."""
    required = {
        "trace_too_short",
        "no_pedestrian_motion",
        "metadata_missing",
        "actor_class_missing",
        "observation_tier_missing",
        "other_explicit_reason",
    }
    assert required.issubset(set(MISSINGNESS_CATEGORIES))


def test_forecast_defaults_unchanged_flag_is_true() -> None:
    """The report explicitly states forecast defaults are unchanged."""
    report = build_denominator_report(parent_issue=2837, issue=2903)
    assert report["forecast_defaults_unchanged"] is True


def test_spot_checks_cover_representative_reasons() -> None:
    """At least one missing cell is spot-checked per observed reason."""
    report = build_denominator_report(parent_issue=2837, issue=2903)
    # The current durable fixture set is expected to have trace-too-short and
    # no-pedestrian-motion cells.  If those change, this test documents it.
    observed_missing = {
        cat for cat, count in report["category_totals"].items() if cat != "evaluated" and count > 0
    }
    assert observed_missing.issubset(set(report["spot_checks"].keys()))
    for reason, cell in report["spot_checks"].items():
        assert cell["reason"] == reason
        assert cell["status"] != "evaluated"
        assert "detail" in cell


def test_matrix_coverage_rows_match_ladder() -> None:
    """The matrix coverage section contains one row per horizon x dt cell."""
    report = build_denominator_report(parent_issue=2837, issue=2903)
    expected_keys = {(h, d) for h in HORIZON_LADDER_S for d in DT_LADDER_S}
    actual_keys = {(row["horizon_s"], row["dt_s"]) for row in report["matrix_coverage"]}
    assert actual_keys == expected_keys
    for row in report["matrix_coverage"]:
        assert row["total_cells"] == len(TRACE_CANDIDATES)
        missing_sum = sum(row["missing_by_reason"].values())
        assert row["evaluated_cells"] + missing_sum == row["total_cells"]


def test_fixture_proposal_targets_at_least_ninety_percent() -> None:
    """The fixture proposal aims at the 90% coverage target."""
    report = build_denominator_report(parent_issue=2837, issue=2903)
    proposal = report["fixture_proposal"]
    assert proposal["coverage_target_fraction"] == pytest.approx(COVERAGE_TARGET_FRACTION)
    assert proposal["minimum_coverage_estimate"] >= COVERAGE_TARGET_FRACTION
    assert proposal["minimum_fixture_additions"] > 0
    assert "corridor_interaction/ammv_social_force" in proposal["short_fixtures"]
    assert "corridor_interaction/default_social_force" in proposal["short_fixtures"]
    assert all(
        "/" in fixture for fixture in proposal["minimum_fixture_changes"]["extend_short_fixtures"]
    )


def test_classify_missingness_maps_known_statuses() -> None:
    """Known ablation statuses map to the expected missingness reasons."""
    assert _classify_missingness({"status": "evaluated"}) == "evaluated"
    assert _classify_missingness({"status": "horizon_longer_than_trace"}) == "trace_too_short"
    assert _classify_missingness({"status": "insufficient_frames"}) == "trace_too_short"
    assert (
        _classify_missingness({"status": "limited_no_pedestrian_motion"}) == "no_pedestrian_motion"
    )
    assert _classify_missingness({"status": "trace_file_missing"}) == "metadata_missing"
    assert _classify_missingness({"status": "evaluation_error"}) == "other_explicit_reason"


def test_fixture_proposal_counts_are_consistent() -> None:
    """The proposal's current + estimated cells matches the estimated coverage."""
    report = build_denominator_report(parent_issue=2837, issue=2903)
    proposal = report["fixture_proposal"]
    total = report["expected_total_cells"]
    estimated_evaluated = (
        proposal["current_evaluated_cells"] + proposal["minimum_additional_cells_estimate"]
    )
    expected_fraction = estimated_evaluated / total
    assert proposal["minimum_coverage_estimate"] == pytest.approx(expected_fraction)


def test_per_family_missingness_sums_match_total() -> None:
    """Each per-family row sums to the number of cells for that family."""
    report = build_denominator_report(parent_issue=2837, issue=2903)
    for row in report["per_family_missingness"]:
        missing_sum = sum(row["missing_by_reason"].values())
        assert row["evaluated_cells"] + missing_sum == row["total_cells"]
