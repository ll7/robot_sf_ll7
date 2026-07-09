"""Focused tests for seed_distribution_report schema, adapters, and builder."""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.seed_distribution_report import (
    IntervalEstimate,
    MetricSummary,
    PerSeedValue,
    SeedDistributionDiagnostics,
    SurfaceProvenance,
    SurfaceRecord,
    _classify_diagnostics,
    adapt_rank_stability_report,
    adapt_seed_variability_report,
    build_seed_distribution_report,
    format_report_markdown,
    validate_report_schema_version,
)

SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION = "seed_distribution_report.v1"


# --- Fixtures ---


def _seed_variability_payload() -> dict:
    """Return a seed_variability_by_scenario.v1 payload with 2 surfaces."""
    return {
        "schema_version": "benchmark-seed-variability-by-scenario.v1",
        "campaign_id": "test-campaign",
        "row_count": 2,
        "rows": [
            {
                "scenario_id": "cross_trap_low",
                "planner_key": "orca",
                "algo": "orca",
                "planner_group": "core",
                "kinematics": "differential_drive",
                "benchmark_profile": "baseline-safe",
                "n": 3,
                "seed_count": 3,
                "episode_count": 6,
                "seed_list": [101, 102, 103],
                "per_seed": [
                    {"seed": 101, "episode_count": 2, "metrics": {"success": 1.0, "collisions": 0.0}},
                    {"seed": 102, "episode_count": 2, "metrics": {"success": 0.5, "collisions": 0.5}},
                    {"seed": 103, "episode_count": 2, "metrics": {"success": 1.0, "collisions": 0.0}},
                ],
                "summary": {
                    "success": {
                        "mean": 0.833,
                        "std": 0.236,
                        "cv": 0.283,
                        "count": 3.0,
                        "ci_low": 0.5,
                        "ci_high": 1.0,
                        "ci_half_width": 0.167,
                    },
                    "collisions": {
                        "mean": 0.167,
                        "std": 0.236,
                        "cv": 1.414,
                        "count": 3.0,
                        "ci_low": 0.0,
                        "ci_high": 0.5,
                        "ci_half_width": 0.167,
                    },
                },
                "provenance": {
                    "campaign_id": "test-campaign",
                    "config_hash": "cfg-hash",
                    "git_hash": "git-hash",
                    "seed_policy": {"mode": "fixed-list"},
                    "confidence": {
                        "method": "bootstrap",
                        "confidence": 0.95,
                        "bootstrap_samples": 64,
                    },
                },
            },
            {
                "scenario_id": "cross_trap_high",
                "planner_key": "sf_planner",
                "algo": "sf_planner",
                "planner_group": "baseline",
                "kinematics": "differential_drive",
                "benchmark_profile": "baseline-safe",
                "n": 1,
                "seed_count": 1,
                "episode_count": 1,
                "seed_list": [201],
                "per_seed": [
                    {
                        "seed": 201,
                        "episode_count": 1,
                        "metrics": {"success": 0.0, "collisions": 1.0},
                    },
                ],
                "summary": {
                    "success": {
                        "mean": 0.0,
                        "std": 0.0,
                        "cv": float("nan"),
                        "count": 1.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "ci_half_width": 0.0,
                    },
                    "collisions": {
                        "mean": 1.0,
                        "std": 0.0,
                        "cv": float("nan"),
                        "count": 1.0,
                        "ci_low": 1.0,
                        "ci_high": 1.0,
                        "ci_half_width": 0.0,
                    },
                },
                "provenance": {
                    "campaign_id": "test-campaign",
                    "config_hash": "cfg-hash",
                    "git_hash": "git-hash",
                    "seed_policy": {"mode": "fixed-list"},
                    "confidence": {
                        "method": "bootstrap",
                        "confidence": 0.95,
                        "bootstrap_samples": 64,
                    },
                },
            },
        ],
    }


def _rank_stability_payload() -> dict:
    """Return an issue_3216_headline_ci_rank_stability.v1 payload."""
    return {
        "schema_version": "issue_3216_headline_ci_rank_stability.v1",
        "classification": "diagnostic",
        "cells": [
            {
                "scenario_id": "cross_trap_low",
                "planner_key": "orca",
                "row_status": "native",
                "counted": True,
                "exclusion_reason": None,
                "seed_count": 3,
                "metrics": {
                    "success": {
                        "mean": 0.833,
                        "std": 0.236,
                        "cv": 0.283,
                        "count": 3.0,
                        "ci_low": 0.5,
                        "ci_high": 1.0,
                        "ci_half_width": 0.167,
                    },
                    "snqi": {
                        "mean": 0.15,
                        "std": 0.05,
                        "cv": 0.333,
                        "count": 3.0,
                        "ci_low": 0.08,
                        "ci_high": 0.22,
                        "ci_half_width": 0.07,
                    },
                },
            },
            {
                "scenario_id": "cross_trap_low",
                "planner_key": "sf_planner",
                "row_status": "fallback",
                "counted": False,
                "exclusion_reason": "fallback",
                "seed_count": 3,
                "metrics": {
                    "success": {
                        "mean": 0.5,
                        "std": 0.1,
                        "cv": 0.2,
                        "count": 3.0,
                        "ci_low": 0.3,
                        "ci_high": 0.7,
                        "ci_half_width": 0.2,
                    },
                },
            },
        ],
        "rank_stability": [
            {
                "scenario_id": "cross_trap_low",
                "rank_metric": "snqi",
                "rank_flip_rate": 0.4,
                "kendall_tau_mean": 0.8,
                "kendall_tau_min": 0.6,
                "top1_stable": False,
            },
        ],
    }


def _stable_surfaces() -> list[SurfaceRecord]:
    """Return 2 stable surfaces with narrow intervals."""
    return [
        SurfaceRecord(
            surface_id="cross_a__orca__success",
            metric="success",
            seed_count=20,
            episode_count=40,
            point_estimate=0.85,
            interval=IntervalEstimate("bootstrap", 0.80, 0.90, 0.95),
            metrics_summary={
                "success": MetricSummary(
                    0.85, 0.05, 0.059, 20.0, 0.80, 0.90, 0.05, "bootstrap", 0.95
                ),
            },
            diagnostics=SeedDistributionDiagnostics(False, None, False, False),
            provenance=SurfaceProvenance("test.json", "v1", "seed_variability"),
            scenario_id="cross_a",
            planner_id="orca",
            per_seed=[PerSeedValue(i, 0.85, 2) for i in range(20)],
        ),
        SurfaceRecord(
            surface_id="cross_b__sf_planner__success",
            metric="success",
            seed_count=20,
            episode_count=40,
            point_estimate=0.70,
            interval=IntervalEstimate("bootstrap", 0.65, 0.75, 0.95),
            metrics_summary={
                "success": MetricSummary(
                    0.70, 0.06, 0.086, 20.0, 0.65, 0.75, 0.05, "bootstrap", 0.95
                ),
            },
            diagnostics=SeedDistributionDiagnostics(False, None, False, False),
            provenance=SurfaceProvenance("test.json", "v1", "seed_variability"),
            scenario_id="cross_b",
            planner_id="sf_planner",
            per_seed=[PerSeedValue(i, 0.70, 2) for i in range(20)],
        ),
    ]


def _unstable_surface() -> SurfaceRecord:
    """Return a surface with wide CI and insufficient seeds."""
    return SurfaceRecord(
        surface_id="narrow__orca__success",
        metric="success",
        seed_count=2,
        episode_count=2,
        point_estimate=0.50,
        interval=IntervalEstimate("bootstrap", 0.20, 0.80, 0.95),
        metrics_summary={
            "success": MetricSummary(
                0.50, 0.30, 0.60, 2.0, 0.20, 0.80, 0.30, "bootstrap", 0.95
            ),
        },
        diagnostics=SeedDistributionDiagnostics(True, None, True, True),
        provenance=SurfaceProvenance("test.json", "v1", "seed_variability"),
        scenario_id="narrow",
        planner_id="orca",
        per_seed=[PerSeedValue(1, 0.8, 1), PerSeedValue(2, 0.2, 1)],
    )


def _insufficient_surface() -> SurfaceRecord:
    """Return a surface with a single seed (insufficient)."""
    return SurfaceRecord(
        surface_id="single__orca__success",
        metric="success",
        seed_count=1,
        episode_count=1,
        point_estimate=1.0,
        interval=IntervalEstimate("bootstrap", 1.0, 1.0, 0.95),
        metrics_summary={
            "success": MetricSummary(
                1.0, 0.0, float("nan"), 1.0, 1.0, 1.0, 0.0, "bootstrap", 0.95
            ),
        },
        diagnostics=SeedDistributionDiagnostics(True, None, False, True),
        provenance=SurfaceProvenance("test.json", "v1", "seed_variability"),
        scenario_id="single",
        planner_id="orca",
        per_seed=[PerSeedValue(99, 1.0, 1)],
    )


# --- Schema tests ---


def test_schema_version_constant() -> None:
    """Schema version constant matches expected value."""
    assert SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION == "seed_distribution_report.v1"


def test_validate_schema_version_accepts_v1() -> None:
    """Validator accepts the correct schema version."""
    payload = {"schema_version": "seed_distribution_report.v1"}
    validate_report_schema_version(payload)


def test_validate_schema_version_rejects_wrong_version() -> None:
    """Validator rejects incorrect schema versions."""
    payload = {"schema_version": "wrong_version"}
    with pytest.raises(ValueError, match="Unsupported schema version"):
        validate_report_schema_version(payload)


def test_validate_schema_version_rejects_missing() -> None:
    """Validator rejects payloads without schema_version."""
    with pytest.raises(ValueError, match="Unsupported schema version"):
        validate_report_schema_version({})


def test_report_to_dict_roundtrip() -> None:
    """Report to_dict preserves schema version and surface count."""
    surfaces = _stable_surfaces()
    report = build_seed_distribution_report(surfaces, campaign_root="/tmp/test")
    d = report.to_dict()
    assert d["schema_version"] == SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION
    assert len(d["surfaces"]) == 2
    assert d["source"]["campaign_root"] == "/tmp/test"
    validate_report_schema_version(d)


def test_report_to_json_roundtrip() -> None:
    """Report to_json produces valid JSON with correct schema version."""
    surfaces = _stable_surfaces()
    report = build_seed_distribution_report(surfaces)
    j = report.to_json()
    parsed = json.loads(j)
    assert parsed["schema_version"] == SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION
    validate_report_schema_version(parsed)


def test_surface_record_preserves_raw_counts() -> None:
    """Adapted seed-variability surfaces preserve raw counts for discrete metrics."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    assert surfaces[0].raw_counts is not None
    assert surfaces[0].raw_counts.denominator == 6


def test_surface_record_to_dict_structure() -> None:
    """SurfaceRecord to_dict includes all required keys."""
    surf = _stable_surfaces()[0]
    d = surf.to_dict()
    required_keys = {
        "surface_id",
        "scenario_id",
        "planner_id",
        "metric",
        "seed_count",
        "episode_count",
        "point_estimate",
        "interval",
        "metrics_summary",
        "diagnostics",
        "provenance",
        "per_seed",
    }
    assert required_keys.issubset(set(d.keys()))


def test_diagnostics_insufficient_seed() -> None:
    """Diagnostics flag insufficient seed count correctly."""
    diag = _classify_diagnostics(1, 0.05)
    assert diag.insufficient_seed_count is True
    assert diag.advisory_only is True


def test_diagnostics_sufficient_seed() -> None:
    """Diagnostics pass for sufficient seed count."""
    diag = _classify_diagnostics(10, 0.05)
    assert diag.insufficient_seed_count is False
    assert diag.advisory_only is False


def test_diagnostics_wide_interval() -> None:
    """Diagnostics flag wide intervals correctly."""
    diag = _classify_diagnostics(10, 0.30)
    assert diag.wide_interval is True


def test_diagnostics_narrow_interval() -> None:
    """Diagnostics pass for narrow intervals."""
    diag = _classify_diagnostics(10, 0.05)
    assert diag.wide_interval is False


def test_diagnostics_unstable_rank() -> None:
    """Diagnostics flag unstable rank correctly."""
    diag = _classify_diagnostics(10, 0.05, rank_flip_rate=0.5)
    assert diag.unstable_rank is True


def test_diagnostics_stable_rank() -> None:
    """Diagnostics pass for stable rank."""
    diag = _classify_diagnostics(10, 0.05, rank_flip_rate=0.1)
    assert diag.unstable_rank is False


def test_diagnostics_no_rank_data() -> None:
    """Diagnostics return None for unstable_rank when no rank data."""
    diag = _classify_diagnostics(10, 0.05)
    assert diag.unstable_rank is None


# --- Adapter tests ---


def test_adapt_seed_variability_extracts_surfaces() -> None:
    """Adapter extracts correct number of surfaces from seed variability payload."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    assert len(surfaces) == 2


def test_adapt_seed_variability_preserves_identity() -> None:
    """Adapter preserves scenario and planner identity fields."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    assert surfaces[0].scenario_id == "cross_trap_low"
    assert surfaces[0].planner_id == "orca"
    assert surfaces[0].seed_count == 3
    assert surfaces[0].metric == "success"


def test_adapt_seed_variability_preserves_per_seed() -> None:
    """Adapter preserves per-seed observation records."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    assert len(surfaces[0].per_seed) == 3
    assert surfaces[0].per_seed[0].seed == 101


def test_adapt_seed_variability_insufficient_flag() -> None:
    """Adapter sets diagnostics based on seed count and threshold."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    orca = surfaces[0]
    assert orca.diagnostics.insufficient_seed_count is False
    assert orca.seed_count == 3


def test_adapt_seed_variability_insufficient_when_below_threshold() -> None:
    """Adapter flags insufficient seeds when threshold is raised."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload, insufficient_seed_threshold=5)
    orca = surfaces[0]
    assert orca.diagnostics.insufficient_seed_count is True
    assert orca.diagnostics.advisory_only is True


def test_adapt_seed_variability_multiple_metrics() -> None:
    """Adapter can use alternative primary metrics."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload, primary_metric="collisions")
    assert surfaces[0].point_estimate == pytest.approx(0.167, abs=0.01)


def test_adapt_rank_stability_extracts_counted_cells() -> None:
    """Adapter extracts only counted cells from rank stability payload."""
    payload = _rank_stability_payload()
    surfaces = adapt_rank_stability_report(payload)
    assert len(surfaces) == 1


def test_adapt_rank_stability_skips_fallback() -> None:
    """Adapter skips fallback cells in rank stability payload."""
    payload = _rank_stability_payload()
    surfaces = adapt_rank_stability_report(payload)
    planner_ids = [s.planner_id for s in surfaces]
    assert "sf_planner" not in planner_ids


def test_adapt_rank_stability_preserves_metrics() -> None:
    """Adapter preserves multiple metrics from rank stability cells."""
    payload = _rank_stability_payload()
    surfaces = adapt_rank_stability_report(payload)
    assert "success" in surfaces[0].metrics_summary
    assert "snqi" in surfaces[0].metrics_summary


def test_adapt_rank_stability_unstable_rank_flag() -> None:
    """Adapter propagates rank instability from rank stability data."""
    payload = _rank_stability_payload()
    surfaces = adapt_rank_stability_report(payload)
    assert surfaces[0].diagnostics.unstable_rank is True


# --- Builder tests ---


def test_build_report_from_surfaces() -> None:
    """Builder produces a valid report from pre-adapted surfaces."""
    surfaces = _stable_surfaces()
    report = build_seed_distribution_report(surfaces, campaign_root="/tmp/test")
    assert report.schema_version == SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION
    assert len(report.surfaces) == 2
    assert report.generated_at_utc


def test_build_report_deterministic_json() -> None:
    """Builder produces deterministic JSON output (ignoring timestamps)."""
    surfaces = _stable_surfaces()
    r1 = build_seed_distribution_report(surfaces).to_json()
    r2 = build_seed_distribution_report(surfaces).to_json()
    p1 = json.loads(r1)
    p2 = json.loads(r2)
    p1["generated_at_utc"] = ""
    p2["generated_at_utc"] = ""
    assert json.dumps(p1, sort_keys=False) == json.dumps(p2, sort_keys=False)


def test_build_report_preserves_interpretation_boundary() -> None:
    """Builder includes interpretation boundary text."""
    surfaces = _stable_surfaces()
    report = build_seed_distribution_report(surfaces)
    assert (
        "not a claim" in report.interpretation_boundary.lower()
        or "not" in report.interpretation_boundary.lower()
    )


def test_build_report_commit_tracking() -> None:
    """Builder tracks commit hash in source provenance."""
    surfaces = _stable_surfaces()
    report = build_seed_distribution_report(surfaces, commit="abc123")
    assert report.source.commit == "abc123"


def test_build_report_report_paths() -> None:
    """Builder tracks input report paths in source provenance."""
    surfaces = _stable_surfaces()
    report = build_seed_distribution_report(
        surfaces, report_paths=["/a/b.json", "/a/c.json"]
    )
    assert len(report.source.report_paths) == 2


def test_build_report_mixed_stability() -> None:
    """Builder handles mixed stable, unstable, and insufficient surfaces."""
    surfaces = _stable_surfaces() + [_unstable_surface(), _insufficient_surface()]
    report = build_seed_distribution_report(surfaces)
    assert len(report.surfaces) == 4
    n_advisory = sum(1 for s in report.surfaces if s.diagnostics.advisory_only)
    assert n_advisory == 2


def test_format_markdown_includes_summary() -> None:
    """Markdown output includes summary section with counts."""
    surfaces = _stable_surfaces() + [_unstable_surface()]
    report = build_seed_distribution_report(surfaces)
    md = format_report_markdown(report)
    assert "Seed Distribution Report" in md
    assert "Stable surfaces" in md
    assert "insufficient" in md.lower() or "Insufficient" in md


def test_format_markdown_table_rows() -> None:
    """Markdown output includes surface table rows."""
    surfaces = _stable_surfaces()
    report = build_seed_distribution_report(surfaces)
    md = format_report_markdown(report)
    assert "cross_a__orca__success" in md
    assert "cross_b__sf_planner__success" in md


def test_raw_counts_for_discrete_metric() -> None:
    """Adapter computes raw counts for discrete metrics."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    assert surfaces[0].raw_counts is not None
    assert surfaces[0].raw_counts.denominator == 6


def test_interval_confidence_level_preserved() -> None:
    """Adapter preserves confidence level from input provenance."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    assert surfaces[0].interval.confidence_level == pytest.approx(0.95)


def test_provenance_adapter_name() -> None:
    """Adapter sets correct provenance adapter name."""
    payload = _seed_variability_payload()
    surfaces = adapt_seed_variability_report(payload)
    assert surfaces[0].provenance.adapter == "seed_variability"

    rs_payload = _rank_stability_payload()
    rs_surfaces = adapt_rank_stability_report(rs_payload)
    assert rs_surfaces[0].provenance.adapter == "rank_stability"
