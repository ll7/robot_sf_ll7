"""Tests for seed_distribution_report v1 schema, adapters, and builder.

Fixture-backed tests cover stable, unstable, insufficient-seed, and missing-artifact
cases without requiring retained campaign roots or live benchmark outputs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.seed_distribution_report import (
    SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION,
    _compute_diagnostics,
    _is_interval_wide,
    build_seed_distribution_report,
    render_markdown,
    validate_schema_version,
    write_report,
)


@pytest.fixture()
def stable_seed_variability(tmp_path: Path) -> Path:
    """Create a synthetic seed_variability.json with stable multi-seed evidence."""
    data = {
        "rows": [
            {
                "scenario_id": "urban_crossing",
                "planner_id": "orca",
                "seed_count": 20,
                "episode_count": 100,
                "metrics": {
                    "success": {
                        "per_seed_means": [
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
                            0.95,
                            0.94,
                            0.93,
                            0.96,
                            0.95,
                            0.94,
                            0.93,
                            0.95,
                        ],
                        "mean": 0.942,
                        "std": 0.012,
                        "ci_low": 0.935,
                        "ci_high": 0.949,
                        "confidence_level": 0.95,
                        "method": "bootstrap_mean_over_seed_means",
                        "seed_count": 20,
                        "episode_count": 100,
                    },
                    "collision": {
                        "per_seed_means": [
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
                            0.02,
                            0.02,
                            0.03,
                            0.02,
                            0.02,
                            0.03,
                            0.02,
                            0.02,
                        ],
                        "mean": 0.023,
                        "std": 0.006,
                        "ci_low": 0.020,
                        "ci_high": 0.026,
                        "confidence_level": 0.95,
                        "method": "bootstrap_mean_over_seed_means",
                        "seed_count": 20,
                        "episode_count": 100,
                    },
                },
            },
            {
                "scenario_id": "urban_crossing",
                "planner_id": "social_force",
                "seed_count": 20,
                "episode_count": 100,
                "metrics": {
                    "success": {
                        "per_seed_means": [
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
                            0.88,
                            0.87,
                            0.88,
                            0.87,
                            0.89,
                            0.88,
                            0.87,
                            0.88,
                        ],
                        "mean": 0.877,
                        "std": 0.008,
                        "ci_low": 0.873,
                        "ci_high": 0.881,
                        "confidence_level": 0.95,
                        "method": "bootstrap_mean_over_seed_means",
                        "seed_count": 20,
                        "episode_count": 100,
                    },
                },
            },
        ],
    }
    path = tmp_path / "seed_variability.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def unstable_seed_variability(tmp_path: Path) -> Path:
    """Create a synthetic seed_variability.json with unstable rank evidence."""
    data = {
        "rows": [
            {
                "scenario_id": "narrow_corridor",
                "planner_id": "orca",
                "seed_count": 15,
                "episode_count": 75,
                "metrics": {
                    "success": {
                        "per_seed_means": [
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
                        ],
                        "mean": 0.615,
                        "std": 0.198,
                        "ci_low": 0.501,
                        "ci_high": 0.729,
                        "confidence_level": 0.95,
                        "method": "bootstrap_mean_over_seed_means",
                        "seed_count": 15,
                        "episode_count": 75,
                        "rank_changed_across_seeds": True,
                    },
                },
            },
        ],
    }
    path = tmp_path / "seed_variability.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def insufficient_seed_variability(tmp_path: Path) -> Path:
    """Create a synthetic seed_variability.json with single-seed insufficient data."""
    data = {
        "rows": [
            {
                "scenario_id": "rare_event",
                "planner_id": "orca",
                "seed_count": 1,
                "episode_count": 5,
                "metrics": {
                    "success": {
                        "per_seed_means": [0.60],
                        "mean": 0.60,
                        "std": 0.0,
                        "ci_low": 0.60,
                        "ci_high": 0.60,
                        "confidence_level": 0.95,
                        "method": "bootstrap_mean_over_seed_means",
                        "seed_count": 1,
                        "episode_count": 5,
                    },
                },
            },
        ],
    }
    path = tmp_path / "seed_variability.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def sufficiency_data(tmp_path: Path) -> Path:
    """Create a synthetic statistical_sufficiency.json."""
    data = {
        "rows": [
            {
                "scenario_id": "urban_crossing",
                "planner_id": "orca",
                "seed_count": 20,
                "episode_count": 100,
                "sufficiency_status": "reported",
                "metrics": {
                    "snqi": {
                        "mean": 0.85,
                        "ci_low": 0.82,
                        "ci_high": 0.88,
                        "confidence_level": 0.95,
                        "seed_count": 20,
                        "episode_count": 100,
                    },
                },
            },
        ],
    }
    path = tmp_path / "statistical_sufficiency.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def ci_rank_stability_data(tmp_path: Path) -> Path:
    """Create a synthetic CI rank stability report."""
    data = {
        "schema_version": "issue_3216_headline_ci_rank_stability.v1",
        "planners": [
            {
                "planner_id": "orca",
                "scenarios": [
                    {
                        "scenario_id": "urban_crossing",
                        "metrics": {
                            "success": {
                                "mean": 0.94,
                                "ci_low": 0.93,
                                "ci_high": 0.95,
                                "confidence_level": 0.95,
                                "seed_count": 20,
                                "episode_count": 100,
                                "rank_unstable": False,
                            },
                        },
                    },
                ],
            },
        ],
    }
    path = tmp_path / "issue_3216_headline_ci_rank_stability.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


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
        assert success_surface["point_estimate"] == 0.942

    def test_stable_report_ci_preserved(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        success_surface = next(
            s for s in report["surfaces"] if s["metric"] == "success" and s["planner_id"] == "orca"
        )
        assert success_surface["interval"]["lower"] == 0.935
        assert success_surface["interval"]["upper"] == 0.949
        assert success_surface["interval"]["confidence_level"] == 0.95

    def test_stable_report_seed_distribution(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        success_surface = next(
            s for s in report["surfaces"] if s["metric"] == "success" and s["planner_id"] == "orca"
        )
        sd = success_surface["seed_distribution"]
        assert len(sd["values"]) == 20
        assert sd["mean"] == 0.942
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
    """Test unstable evidence with rank or scenario-winner drift."""

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
        assert "seed_variability.json" in report["source"]["report_paths"]

    def test_only_sufficiency_present(self, sufficiency_data: Path) -> None:
        report = build_seed_distribution_report(sufficiency_data.parent)
        assert len(report["surfaces"]) == 1
        assert "statistical_sufficiency.json" in report["source"]["report_paths"]

    def test_only_ci_rank_stability_present(self, ci_rank_stability_data: Path) -> None:
        report = build_seed_distribution_report(ci_rank_stability_data.parent)
        assert len(report["surfaces"]) == 1
        assert "issue_3216_headline_ci_rank_stability.json" in report["source"]["report_paths"]


class TestNoSupportedArtifacts:
    """Test behavior when no supported seed evidence is present."""

    def test_empty_campaign_raises(self, empty_campaign: Path) -> None:
        empty_campaign.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No supported seed-level artifacts"):
            build_seed_distribution_report(empty_campaign)

    def test_nonexistent_campaign_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            build_seed_distribution_report(tmp_path / "nonexistent")


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
        dst_ci = campaign / "issue_3216_headline_ci_rank_stability.json"
        if sufficiency_data.resolve() != dst_suff.resolve():
            shutil.copy2(sufficiency_data, dst_suff)
        if ci_rank_stability_data.resolve() != dst_ci.resolve():
            shutil.copy2(ci_rank_stability_data, dst_ci)

        report = build_seed_distribution_report(campaign)
        assert len(report["surfaces"]) == 3 + 1 + 1
        assert len(report["source"]["report_paths"]) == 3


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
        assert "seed_variability.json" in report["source"]["report_paths"]

    def test_source_generated_by(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(
            stable_seed_variability.parent, generated_by="test_tool"
        )
        assert report["source"]["generated_by"] == "test_tool"

    def test_surface_provenance(self, stable_seed_variability: Path) -> None:
        report = build_seed_distribution_report(stable_seed_variability.parent)
        surface = report["surfaces"][0]
        assert "input_artifact" in surface["provenance"]


class TestSchemaVersionConstant:
    """Test the schema version constant is set correctly."""

    def test_schema_version_string(self) -> None:
        assert SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION == "seed_distribution_report.v1"

    def test_schema_version_in_supported_set(self) -> None:
        from robot_sf.benchmark.seed_distribution_report import _SUPPORTED_SCHEMA_VERSIONS

        assert SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION in _SUPPORTED_SCHEMA_VERSIONS
