"""Tests for seed_distribution_report v1 schema, adapters, and builder.

Fixture-backed tests cover stable, unstable, insufficient-seed, and
missing-artifact cases without requiring retained campaign roots or live
benchmark outputs. Fixtures mirror the *real* on-disk artifact shapes emitted by
the benchmark pipeline so they exercise the same parsing paths used on real
campaign folders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.seed_distribution_report import (
    SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION,
    _compute_diagnostics,
    _is_interval_wide,
    _width_is_wide,
    build_seed_distribution_report,
    render_markdown,
    validate_schema_version,
    write_report,
)

# A real evidence folder that ships both seed-variability and sufficiency
# artifacts under reports/. Used for an optional real-data smoke test; skipped
# when absent so the suite stays self-contained in stripped checkouts.
_REAL_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "context"
    / "evidence"
    / "issue_1484_broader_cross_kinematics_2026-05-28"
)


def _stat(
    mean: float,
    *,
    std: float = 0.0,
    ci_low: float | None = None,
    ci_high: float | None = None,
    count: float = 12.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical per-metric stat dict (matches seed_variance._stats_for_vals)."""
    ci_low = mean if ci_low is None else ci_low
    ci_high = mean if ci_high is None else ci_high
    payload: dict[str, Any] = {
        "mean": mean,
        "std": std,
        "cv": 0.0 if mean else float("nan"),
        "count": count,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": (ci_high - ci_low) / 2,
    }
    if extra:
        payload.update(extra)
    return payload


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def stable_seed_variability(tmp_path: Path) -> Path:
    """Synthetic seed_variability_by_scenario.json with stable multi-seed evidence."""
    orca_success_per_seed = [
        0.95,
        0.93,
        0.94,
        0.96,
        0.92,
        0.95,
        0.93,
        0.94,
        0.96,
        0.95,
        0.94,
        0.93,
    ]
    orca_collision_per_seed = [
        0.02,
        0.03,
        0.02,
        0.01,
        0.03,
        0.02,
        0.02,
        0.03,
        0.02,
        0.02,
        0.02,
        0.03,
    ]
    social_success_per_seed = [
        0.88,
        0.87,
        0.89,
        0.86,
        0.88,
        0.87,
        0.89,
        0.88,
        0.87,
        0.88,
        0.87,
        0.89,
    ]
    data = {
        "schema_version": "benchmark-seed-variability-by-scenario.v1",
        "campaign_id": "stable-fixture",
        "confidence": {
            "method": "bootstrap_mean_over_seed_means",
            "confidence": 0.95,
            "bootstrap_samples": 100,
            "bootstrap_seed": 123,
        },
        "row_count": 2,
        "rows": [
            {
                "scenario_id": "urban_crossing",
                "planner_key": "orca",
                "algo": "orca",
                "planner_group": "core",
                "kinematics": "differential_drive",
                "benchmark_profile": "baseline-safe",
                "n": 12,
                "seed_count": 12,
                "episode_count": 60,
                "seed_list": list(range(12)),
                "per_seed": [
                    {
                        "seed": i,
                        "episode_count": 5,
                        "metrics": {
                            "success": orca_success_per_seed[i],
                            "collisions": orca_collision_per_seed[i],
                        },
                    }
                    for i in range(12)
                ],
                "summary": {
                    "success": _stat(0.942, std=0.012, ci_low=0.935, ci_high=0.949),
                    "collisions": _stat(0.023, std=0.006, ci_low=0.020, ci_high=0.026),
                },
            },
            {
                "scenario_id": "urban_crossing",
                "planner_key": "social_force",
                "algo": "social_force",
                "planner_group": "core",
                "kinematics": "differential_drive",
                "benchmark_profile": "baseline-safe",
                "n": 12,
                "seed_count": 12,
                "episode_count": 60,
                "seed_list": list(range(12)),
                "per_seed": [
                    {
                        "seed": i,
                        "episode_count": 5,
                        "metrics": {"success": social_success_per_seed[i]},
                    }
                    for i in range(12)
                ],
                "summary": {
                    "success": _stat(0.877, std=0.008, ci_low=0.873, ci_high=0.881),
                },
            },
        ],
    }
    return _write_json(tmp_path / "seed_variability_by_scenario.json", data)


@pytest.fixture()
def unstable_seed_variability(tmp_path: Path) -> Path:
    """Synthetic seed_variability_by_scenario.json with wide CI + rank drift."""
    data = {
        "schema_version": "benchmark-seed-variability-by-scenario.v1",
        "confidence": {"method": "bootstrap_mean_over_seed_means", "confidence": 0.95},
        "rows": [
            {
                "scenario_id": "narrow_corridor",
                "planner_key": "orca",
                "algo": "orca",
                "seed_count": 15,
                "episode_count": 75,
                "per_seed": [
                    {"seed": i, "episode_count": 5, "metrics": {"success": v}}
                    for i, v in enumerate(
                        [
                            0.80,
                            0.40,
                            0.75,
                            0.45,
                            0.82,
                            0.38,
                            0.78,
                            0.42,
                            0.81,
                            0.39,
                            0.76,
                            0.44,
                            0.83,
                            0.37,
                            0.79,
                        ]
                    )
                ],
                "summary": {
                    # width 0.228 > 0.15 success threshold -> wide
                    "success": _stat(
                        0.615,
                        std=0.198,
                        ci_low=0.501,
                        ci_high=0.729,
                        extra={"rank_changed_across_seeds": True},
                    ),
                },
            }
        ],
    }
    return _write_json(tmp_path / "seed_variability_by_scenario.json", data)


@pytest.fixture()
def insufficient_seed_variability(tmp_path: Path) -> Path:
    """Synthetic seed_variability_by_scenario.json with single-seed insufficient data."""
    data = {
        "schema_version": "benchmark-seed-variability-by-scenario.v1",
        "confidence": {"method": "bootstrap_mean_over_seed_means", "confidence": 0.95},
        "rows": [
            {
                "scenario_id": "rare_event",
                "planner_key": "orca",
                "algo": "orca",
                "seed_count": 1,
                "episode_count": 5,
                "per_seed": [{"seed": 111, "episode_count": 5, "metrics": {"success": 0.60}}],
                "summary": {"success": _stat(0.60, std=0.0, ci_low=0.60, ci_high=0.60, count=1)},
            }
        ],
    }
    return _write_json(tmp_path / "seed_variability_by_scenario.json", data)


@pytest.fixture()
def sufficiency_data(tmp_path: Path) -> Path:
    """Synthetic statistical_sufficiency.json with real shape (ci_half_width, planner_key)."""
    data = {
        "schema_version": "benchmark-seed-statistical-sufficiency.v1",
        "campaign_id": "sufficiency-fixture",
        "confidence": {
            "method": "bootstrap_mean_over_seed_means",
            "confidence": 0.95,
            "bootstrap_samples": 300,
            "bootstrap_seed": 123,
        },
        "row_count": 1,
        "rows": [
            {
                "scenario_id": "urban_crossing",
                "planner_key": "orca",
                "kinematics": "differential_drive",
                "algo": "orca",
                "seed_count": 20,
                "episode_count": 100,
                "sufficiency_status": "reported",
                "metric_half_widths": {"snqi": 0.03},
                "metrics": {"snqi": {"n": 20.0, "ci_half_width": 0.03}},
            }
        ],
    }
    return _write_json(tmp_path / "statistical_sufficiency.json", data)


@pytest.fixture()
def ci_rank_stability_data(tmp_path: Path) -> Path:
    """Synthetic headline CI rank-stability report with the real flat-cells shape."""
    data = {
        "schema_version": "issue_3216_headline_ci_rank_stability.v1",
        "classification": "rank_stable",
        "cells": [
            {
                "scenario_id": "urban_crossing",
                "planner_key": "orca",
                "row_status": "ok",
                "counted": True,
                "exclusion_reason": None,
                "seed_count": 20,
                "metrics": {"success": _stat(0.94, ci_low=0.93, ci_high=0.95, count=20)},
            },
            {
                "scenario_id": "urban_crossing",
                "planner_key": "goal",
                "row_status": "excluded_insufficient_seeds",
                "counted": False,
                "exclusion_reason": "insufficient_seeds",
                "seed_count": 2,
                "metrics": {"success": _stat(0.70, ci_low=0.40, ci_high=1.00, count=2)},
            },
        ],
    }
    return _write_json(tmp_path / "result.json", data)


@pytest.fixture()
def empty_campaign(tmp_path: Path) -> Path:
    """Create an empty campaign directory with no supported artifacts."""
    return tmp_path / "empty_campaign"


class TestSchemaVersionValidation:
    """Test schema version validation (fail-closed behavior)."""

    def test_valid_schema_version_accepted(self) -> None:
        payload = {"schema_version": SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION}
        validate_schema_version(payload)

    def test_unsupported_schema_version_rejected(self) -> None:
        payload = {"schema_version": "seed_distribution_report.v2"}
        with pytest.raises(ValueError, match="unsupported seed distribution report schema version"):
            validate_schema_version(payload)

    def test_missing_schema_version_rejected(self) -> None:
        payload = {"data": "something"}
        with pytest.raises(ValueError, match="missing required field"):
            validate_schema_version(payload)

    def test_unknown_schema_version_rejected(self) -> None:
        payload = {"schema_version": "completely_unknown_schema.v99"}
        with pytest.raises(ValueError, match="unsupported"):
            validate_schema_version(payload)


class TestDiagnostics:
    """Test diagnostic flag computation."""

    def test_sufficient_seeds_no_insufficient_flag(self) -> None:
        diag = _compute_diagnostics(seed_count=20, unstable_rank=False, wide_interval=False)
        assert diag["insufficient_seed_count"] is False
        assert diag["unstable_rank"] is False
        assert diag["wide_interval"] is False
        assert diag["advisory_only"] is False

    def test_insufficient_seeds_detected(self) -> None:
        diag = _compute_diagnostics(seed_count=5, unstable_rank=False, wide_interval=False)
        assert diag["insufficient_seed_count"] is True

    def test_unstable_rank_detected(self) -> None:
        diag = _compute_diagnostics(seed_count=20, unstable_rank=True, wide_interval=False)
        assert diag["unstable_rank"] is True

    def test_wide_interval_detected(self) -> None:
        diag = _compute_diagnostics(seed_count=20, unstable_rank=False, wide_interval=True)
        assert diag["wide_interval"] is True

    def test_advisory_only_flag(self) -> None:
        diag = _compute_diagnostics(
            seed_count=20, unstable_rank=False, wide_interval=False, advisory_only=True
        )
        assert diag["advisory_only"] is True

    def test_unstable_rank_null_collapses_to_false(self) -> None:
        diag = _compute_diagnostics(seed_count=20, unstable_rank=None, wide_interval=False)
        assert diag["unstable_rank"] is False


class TestIntervalWidth:
    """Test interval width classification."""

    def test_success_metric_wide(self) -> None:
        assert _is_interval_wide(0.40, 0.80, "success") is True

    def test_success_metric_narrow(self) -> None:
        assert _is_interval_wide(0.93, 0.96, "success") is False

    def test_collision_metric_wide(self) -> None:
        assert _is_interval_wide(0.01, 0.15, "collision") is True

    def test_none_ci_not_wide(self) -> None:
        assert _is_interval_wide(None, None, "success") is False

    def test_unknown_metric_uses_default(self) -> None:
        assert _is_interval_wide(0.0, 0.50, "some_unknown_metric") is True
        assert _is_interval_wide(0.40, 0.50, "some_unknown_metric") is False

    def test_width_is_wide_from_half_width(self) -> None:
        # snqi threshold 0.25; width 0.30 -> wide
        assert _width_is_wide(0.30, "snqi") is True
        assert _width_is_wide(None, "snqi") is False


class TestStableMultiSeedReport:
    """Test stable multi-seed evidence with narrow intervals."""

    def test_stable_report_schema(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        assert report["schema_version"] == SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION

    def test_stable_report_surfaces(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        surfaces = report["surfaces"]
        assert len(surfaces) == 3

    def test_stable_report_point_estimates(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        success_surface = next(
            s for s in report["surfaces"] if s["metric"] == "success" and s["planner_id"] == "orca"
        )
        assert success_surface["point_estimate"] == pytest.approx(0.942)

    def test_stable_report_ci_preserved(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        success_surface = next(
            s for s in report["surfaces"] if s["metric"] == "success" and s["planner_id"] == "orca"
        )
        assert success_surface["interval"]["lower"] == pytest.approx(0.935)
        assert success_surface["interval"]["upper"] == pytest.approx(0.949)
        assert success_surface["interval"]["confidence_level"] == 0.95

    def test_stable_report_seed_distribution(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        success_surface = next(
            s for s in report["surfaces"] if s["metric"] == "success" and s["planner_id"] == "orca"
        )
        sd = success_surface["seed_distribution"]
        assert len(sd["values"]) == 12
        assert sd["mean"] == pytest.approx(0.942)
        assert sd["min"] <= sd["mean"] <= sd["max"]

    def test_stable_report_seed_count_preserved(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        for surface in report["surfaces"]:
            assert surface["seed_count"] >= 1

    def test_stable_report_diagnostics_clean(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        orca_success = next(
            s for s in report["surfaces"] if s["metric"] == "success" and s["planner_id"] == "orca"
        )
        diag = orca_success["diagnostics"]
        assert diag["insufficient_seed_count"] is False
        assert diag["unstable_rank"] is False
        assert diag["wide_interval"] is False
        assert diag["advisory_only"] is False

    def test_stable_report_deterministic_ordering(self, stable_seed_variability: Path) -> None:
        report_a = build_seed_distribution_report(stable_seed_variability.parent)
        report_b = build_seed_distribution_report(stable_seed_variability.parent)
        assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)


class TestUnstableEvidenceReport:
    """Test unstable evidence with wide intervals and rank drift."""

    def test_unstable_rank_detected(self, unstable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(unstable_seed_variability.parent)
        surface = report["surfaces"][0]
        assert surface["diagnostics"]["unstable_rank"] is True

    def test_unstable_wide_interval(self, unstable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(unstable_seed_variability.parent)
        surface = report["surfaces"][0]
        assert surface["diagnostics"]["wide_interval"] is True


class TestInsufficientDataReport:
    """Test single-seed or incomplete seed coverage."""

    def test_insufficient_seed_flag(self, insufficient_seed_variability: Path) -> None:
        report = build_seed_distribution_report(insufficient_seed_variability.parent)
        surface = report["surfaces"][0]
        assert surface["diagnostics"]["insufficient_seed_count"] is True

    def test_insufficient_seed_distribution_single_value(
        self, insufficient_seed_variability: Path
    ) -> None:
        report = build_seed_distribution_report(insufficient_seed_variability.parent)
        surface = report["surfaces"][0]
        assert len(surface["seed_distribution"]["values"]) == 1

    def test_insufficient_raw_counts_preserved(self, insufficient_seed_variability: Path) -> None:
        report = build_seed_distribution_report(insufficient_seed_variability.parent)
        surface = report["surfaces"][0]
        assert surface["seed_count"] == 1
        assert surface["episode_count"] == 5


class TestMissingOptionalArtifacts:
    """Test that missing optional artifacts produce advisory diagnostics."""

    def test_only_seed_variability_present(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        assert len(report["surfaces"]) == 3
        assert "seed_variability_by_scenario.json" in report["source"]["report_paths"]

    def test_only_sufficiency_present(self, sufficiency_data: Path) -> None:
        report = build_seed_distribution_report(sufficiency_data.parent)
        assert len(report["surfaces"]) == 1
        assert "statistical_sufficiency.json" in report["source"]["report_paths"]

    def test_only_ci_rank_stability_present(self, ci_rank_stability_data: Path) -> None:
        report = build_seed_distribution_report(ci_rank_stability_data.parent)
        # Two cells x one metric each.
        assert len(report["surfaces"]) == 2
        assert "result.json" in report["source"]["report_paths"]

    def test_sufficiency_preserves_half_width(self, sufficiency_data: Path) -> None:
        report = build_seed_distribution_report(sufficiency_data.parent)
        surface = report["surfaces"][0]
        interval = surface["interval"]
        assert interval is not None
        assert interval["half_width"] == pytest.approx(0.03)
        assert surface["point_estimate"] is None


class TestNoSupportedArtifacts:
    """Test behavior when no supported seed evidence is present."""

    def test_empty_campaign_raises(self, empty_campaign: Path) -> None:
        empty_campaign.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No supported seed-level artifacts"):
            build_seed_distribution_report(empty_campaign)

    def test_nonexistent_campaign_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            build_seed_distribution_report(tmp_path / "nonexistent")

    def test_unrelated_result_json_is_skipped(self, tmp_path: Path) -> None:
        # A result.json that is NOT an issue_3216 CI rank report must be ignored,
        # and with no other supported artifacts the builder must fail closed.
        _write_json(tmp_path / "result.json", {"schema_version": "something_else.v1", "cells": []})
        with pytest.raises(FileNotFoundError, match="No supported seed-level artifacts"):
            build_seed_distribution_report(tmp_path)


class TestArtifactDiscovery:
    """Test artifact discovery across root and reports/ subdirectory layouts."""

    def test_finds_artifact_in_reports_subdir(
        self, stable_seed_variability: Path, tmp_path: Path
    ) -> None:
        campaign = tmp_path / "campaign"
        reports = campaign / "reports"
        reports.mkdir(parents=True)
        target = reports / "seed_variability_by_scenario.json"
        target.write_text(stable_seed_variability.read_text(), encoding="utf-8")
        report = build_seed_distribution_report(campaign)
        assert len(report["surfaces"]) == 3
        assert report["source"]["campaign_root"] == str(campaign)


class TestMultipleAdaptersCombined:
    """Test combining multiple adapter inputs."""

    def test_all_three_adapters(
        self,
        stable_seed_variability: Path,
        sufficiency_data: Path,
        ci_rank_stability_data: Path,
    ) -> None:
        import shutil

        campaign = stable_seed_variability.parent
        dst_suff = campaign / "statistical_sufficiency.json"
        dst_ci = campaign / "result.json"
        if sufficiency_data.resolve() != dst_suff.resolve():
            shutil.copy2(sufficiency_data, dst_suff)
        if ci_rank_stability_data.resolve() != dst_ci.resolve():
            shutil.copy2(ci_rank_stability_data, dst_ci)

        report = build_seed_distribution_report(campaign)
        # 3 (seed_var) + 1 (sufficiency) + 2 (ci cells)
        assert len(report["surfaces"]) == 3 + 1 + 2
        assert len(report["source"]["report_paths"]) == 3

    def test_ci_rank_excluded_cell_is_advisory(self, ci_rank_stability_data: Path) -> None:
        report = build_seed_distribution_report(ci_rank_stability_data.parent)
        goal_surface = next(s for s in report["surfaces"] if s["planner_id"] == "goal")
        assert goal_surface["diagnostics"]["advisory_only"] is True
        assert goal_surface["diagnostics"]["insufficient_seed_count"] is True


class TestMarkdownRendering:
    """Test Markdown report generation."""

    def test_render_markdown_contains_header(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        md = render_markdown(report)
        assert "# Seed Distribution Report" in md

    def test_render_markdown_contains_schema_version(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        md = render_markdown(report)
        assert SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION in md

    def test_render_markdown_contains_diagnostics(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        md = render_markdown(report)
        assert "Diagnostics Summary" in md

    def test_render_markdown_contains_surfaces(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        md = render_markdown(report)
        assert "Surfaces" in md
        assert "urban_crossing" in md
        assert "orca" in md

    def test_render_markdown_contains_validity_caveat(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        md = render_markdown(report)
        assert "not imply simulator fidelity" in md

    def test_render_markdown_handles_half_width_interval(self, sufficiency_data: Path) -> None:
        report = build_seed_distribution_report(sufficiency_data.parent)
        md = render_markdown(report)
        assert "CI half-width" in md


class TestWriteReport:
    """Test JSON and Markdown file writing."""

    def test_write_json(self, stable_seed_variability: Path, tmp_path: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        out_json = tmp_path / "report.json"
        written = write_report(report, out_json=out_json)
        assert "json" in written
        assert written["json"].exists()

        with open(out_json) as f:
            loaded = json.load(f)
        assert loaded["schema_version"] == SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION

    def test_write_md(self, stable_seed_variability: Path, tmp_path: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        out_md = tmp_path / "report.md"
        written = write_report(report, out_md=out_md)
        assert "md" in written
        assert written["md"].exists()
        content = out_md.read_text()
        assert "Seed Distribution Report" in content

    def test_write_both(self, stable_seed_variability: Path, tmp_path: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        written = write_report(
            report,
            out_json=tmp_path / "r.json",
            out_md=tmp_path / "r.md",
        )
        assert "json" in written
        assert "md" in written

    def test_write_report_validates_schema(self, tmp_path: Path) -> None:
        bad_report = {"schema_version": "bad.v1"}
        with pytest.raises(ValueError, match="unsupported"):
            write_report(bad_report, out_json=tmp_path / "bad.json")


class TestProvenanceFields:
    """Test that provenance metadata is preserved."""

    def test_source_campaign_root(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        assert report["source"]["campaign_root"] == str(stable_seed_variability.parent)

    def test_source_report_paths(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        assert "seed_variability_by_scenario.json" in report["source"]["report_paths"]

    def test_source_generated_by(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(
            stable_seed_variability.parent, generated_by="test_tool"
        )
        assert report["source"]["generated_by"] == "test_tool"

    def test_surface_provenance(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        surface = report["surfaces"][0]
        assert "input_artifact" in surface["provenance"]
        assert surface["provenance"]["input_schema_version"] == (
            "benchmark-seed-variability-by-scenario.v1"
        )


class TestSchemaVersionConstant:
    """Test the schema version constant is set correctly."""

    def test_schema_version_string(self) -> None:
        assert SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION == "seed_distribution_report.v1"

    def test_schema_version_in_supported_set(self) -> None:
        from robot_sf.benchmark.seed_distribution_report import _SUPPORTED_SCHEMA_VERSIONS

        assert SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION in _SUPPORTED_SCHEMA_VERSIONS


@pytest.mark.skipif(not _REAL_EVIDENCE_DIR.exists(), reason="real evidence dir not present")
class TestRealDataSmoke:
    """Smoke-test the builder against real shipped evidence artifacts.

    These artifacts live under docs/context/evidence/ (read-only here) and are
    the same shapes the adapters must consume in production. Outputs are written
    to tmp_path only -- never under docs/context/evidence/.
    """

    def test_builds_from_real_campaign_folder(self, tmp_path: Path) -> None:
        report = build_seed_distribution_report(_REAL_EVIDENCE_DIR)
        assert report["schema_version"] == SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION
        assert len(report["surfaces"]) > 0
        # Real planner identities (planner_key) must survive normalization.
        planners = {s["planner_id"] for s in report["surfaces"] if s["planner_id"]}
        assert planners, "expected non-null planner_id values from real data"
        # At least one surface should carry interval evidence.
        assert any(s.get("interval") for s in report["surfaces"])
        # Round-trips through the writer cleanly.
        out = write_report(report, out_json=tmp_path / "real.json", out_md=tmp_path / "real.md")
        assert out["json"].exists() and out["md"].exists()
