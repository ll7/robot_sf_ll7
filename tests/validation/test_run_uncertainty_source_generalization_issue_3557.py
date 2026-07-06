"""Tests issue #3557 uncertainty-source generalization report runner."""

from __future__ import annotations

import json

import pytest

from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (
    UNCERTAINTY_SOURCES,
    EpisodeParams,
    run_matrix,
)
from scripts.validation.run_uncertainty_source_generalization_issue_3557 import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    build_report,
    contrast_from_source_report,
    write_report_artifacts,
)

_FAST = EpisodeParams(max_steps=40)


def test_contrast_from_source_report_uses_source_and_dropped_minus_retained_deltas() -> None:
    """Source report converts into the merged issue #3557 contrast primitive."""
    report = run_matrix([101, 102], _FAST, uncertainty_source="existence_degradation")
    contrast = contrast_from_source_report(report)
    assert contrast.source == "existence_degradation"
    assert contrast.n_episodes == 2
    assert contrast.dropped_unsafe_commit_rate >= 0.0
    assert isinstance(contrast.min_separation_delta_m, float)


def test_build_report_runs_all_requested_sources() -> None:
    """Report includes one visible decision row per requested uncertainty source."""
    sources = ["existence_degradation", "visibility_occlusion", "covariance_inflation"]
    report = build_report(seeds=[101, 102], params=_FAST, sources=sources)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["issue"] == 3557
    assert "no full benchmark campaign" in report["claim_boundary"]
    assert [row["source"] for row in report["per_source"]] == sources
    assert {entry["source"] for entry in report["decision_layer"]["per_source"]} == set(sources)
    assert {row["condition_builder"] for row in report["per_source"]}


def test_unknown_source_fails_closed() -> None:
    """Unknown source names fail closed before producing a report."""
    with pytest.raises(ValueError, match="unknown uncertainty source"):
        build_report(seeds=[101], params=_FAST, sources=["unknown"])


def test_write_report_artifacts(tmp_path) -> None:
    """Writer emits README, summary, and per-source CSV with claim boundary."""
    report = build_report(
        seeds=[101],
        params=_FAST,
        sources=["existence_degradation", "class_probability"],
    )
    write_report_artifacts(report, tmp_path, "uv run python source-script")

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == SCHEMA_VERSION
    assert "source_reports" not in summary
    assert (
        (tmp_path / "per_source_decisions.csv")
        .read_text(encoding="utf-8")
        .startswith("source,condition_builder")
    )
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert CLAIM_BOUNDARY in readme
    assert "not a full benchmark campaign" in readme


def test_default_source_registry_matches_harness_sources() -> None:
    """Report runner and harness share the same source registry."""
    report = build_report(seeds=[101], params=_FAST, sources=list(UNCERTAINTY_SOURCES))
    assert report["sources"] == list(UNCERTAINTY_SOURCES)
