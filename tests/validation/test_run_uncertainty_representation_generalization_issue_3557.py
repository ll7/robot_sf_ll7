"""Tests issue #3557 uncertainty-representation generalization report runner."""

from __future__ import annotations

import json

import pytest

from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (
    EpisodeParams,
    run_matrix,
)
from scripts.validation.run_uncertainty_representation_generalization_issue_3557 import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    build_report,
    contrast_from_representation_report,
    write_report_artifacts,
)

_FAST = EpisodeParams(max_steps=40)


def test_contrast_from_representation_report_uses_dropped_minus_retained_deltas() -> None:
    """Representation report converts into the existing issue #3557 contrast primitive."""
    report = run_matrix([101, 102], _FAST, uncertainty_representation="belief_drop")

    contrast = contrast_from_representation_report(report)

    assert contrast.source == "belief_drop"
    assert contrast.n_episodes == 2
    assert contrast.dropped_unsafe_commit_rate >= contrast.retained_unsafe_commit_rate
    assert isinstance(contrast.min_separation_delta_m, float)


def test_build_report_runs_all_requested_representations() -> None:
    """Report includes one visible decision row per requested representation."""
    report = build_report(
        seeds=[101, 102],
        params=_FAST,
        representations=["belief_drop", "conformal_radius", "envelope_inflation"],
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["issue"] == 3557
    assert "no full benchmark campaign" in report["claim_boundary"]
    assert [row["representation"] for row in report["per_representation"]] == [
        "belief_drop",
        "conformal_radius",
        "envelope_inflation",
    ]
    assert report["generalization"] in {
        "generalizes",
        "source_specific",
        "does_not_generalize",
        "inconclusive",
    }
    assert {entry["source"] for entry in report["decision_layer"]["per_source"]} == {
        "belief_drop",
        "conformal_radius",
        "envelope_inflation",
    }


def test_build_report_rejects_unknown_representation() -> None:
    """Unknown representation names fail closed before any implicit fallback."""
    with pytest.raises(ValueError, match="unknown uncertainty representation"):
        build_report(seeds=[101], params=_FAST, representations=["belief_drop", "unknown"])


def test_write_report_artifacts(tmp_path) -> None:
    """Writer emits reviewable README, JSON summary, and CSV decision table."""
    report = build_report(
        seeds=[101],
        params=EpisodeParams(max_steps=20),
        representations=["belief_drop"],
    )

    write_report_artifacts(report, tmp_path, "uv run python script.py")

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["claim_boundary"] == CLAIM_BOUNDARY
    assert summary["campaign_promotion_state"]["new_blockers"] == []
    assert "full benchmark campaign" in summary["campaign_promotion_state"]["next_empirical_action"]
    assert (
        (tmp_path / "per_representation_decisions.csv")
        .read_text()
        .startswith("representation,harness_decision")
    )
    readme = (tmp_path / "README.md").read_text()
    assert "Issue #3557" in readme
    assert "not a full benchmark campaign" in readme
    integration_report = (tmp_path / "integration_report.md").read_text()
    assert "Remaining Blockers" in integration_report
    assert "No full benchmark campaign" in integration_report
