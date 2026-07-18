"""Tests for the issue #5034 → parent #4977 diagnostic-only classification.

Job 13516 completed the authorized 7,344-episode fixed-scope campaign. The
strict native-only boundary is still not satisfied for the three-planner
comparison because ORCA and hybrid rows are adapter-backed, so the durable
classification remains ``diagnostic-only``. These tests pin the promoted
full-scope bundle and prevent a partial or over-claimed replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "docs/context/evidence/issue_5034_control_action_latency_sweep"
SUMMARY_PATH = BUNDLE_DIR / "summary.json"
SNQI_ANALYSIS_PATH = BUNDLE_DIR / "snqi_analysis.json"

#: Permitted result classifications declared by issue #5034's Definition of Done
#: and `docs/benchmark_governance.md`.
PERMITTED_CLASSIFICATIONS = frozenset(
    {"nominal benchmark evidence", "diagnostic-only", "blocked", "not benchmark evidence"}
)


def _load_bundle_summary() -> dict:
    """Return the durable job-13516 latency-evidence summary packet."""
    assert SUMMARY_PATH.exists(), f"durable latency-evidence bundle missing: {SUMMARY_PATH}"
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_bundle_classifies_as_diagnostic_only() -> None:
    """The completed mixed native/adapter sweep remains diagnostic-only."""
    summary = _load_bundle_summary()
    assert summary["result_classification"] == "diagnostic-only"
    assert summary["result_classification"] in PERMITTED_CLASSIFICATIONS
    assert summary["evidence_tier"] == "targeted smoke"


def test_snqi_claim_boundary_keeps_adapter_rows_diagnostic() -> None:
    """The mixed-mode ranking must not read as native-only execution evidence."""
    analysis = json.loads(SNQI_ANALYSIS_PATH.read_text(encoding="utf-8"))
    boundary = analysis["claim_boundary"].casefold()
    assert "diagnostic" in boundary
    assert "native claims apply only to native rows" in boundary
    assert "adapter rows remain explicitly labeled diagnostics" in boundary
    assert "native execution only" not in boundary


def test_bundle_points_at_parent_issue_4977() -> None:
    """The evidence packet records its parent issue so the classification propagates."""
    summary = _load_bundle_summary()
    assert summary.get("parent_issue") == 4977
    assert summary.get("issue") == 5034


def test_bundle_is_full_latency_coverage_without_exclusions() -> None:
    """The diagnostic-only classification rests on complete eligible 0/1/3 coverage.

    A partial or fallback-contaminated run must not be promoted as the latency
    sweep, so the parent classification would be unsafe. The bundle must show
    full latency-step coverage and zero excluded rows before we classify #4977.
    """
    summary = _load_bundle_summary()
    coverage = summary["latency_coverage"]
    assert coverage["coverage_complete"] is True
    assert coverage["missing_latency_steps"] == []
    assert coverage["required_latency_steps"] == [0, 1, 3]
    assert coverage["observed_latency_steps"] == [0, 1, 3]
    assert summary["scope"]["latency_row_count"] == 1296
    assert summary["scope"]["result_row_count"] == 1296
    assert summary["scope"]["planners"] == [
        "baseline_social_force",
        "hybrid_rule_v0_minimal",
        "orca",
    ]
    assert summary["exclusions"]["excluded_row_count"] == 0
    fixed_scope = summary["fixed_scope_coverage"]
    assert fixed_scope["status"] == "verified"
    assert fixed_scope["scenario_count"] == 48
    assert fixed_scope["observed_seeds"] == [111, 112, 113]
    assert fixed_scope["observed_planner_groups"] == [
        "default_social_force",
        "hybrid_rule_v0_minimal",
        "orca",
    ]
    assert fixed_scope["observed_result_row_count"] == 1296
    assert fixed_scope["expected_row_count"] == 1296


@pytest.mark.parametrize("classification", sorted(PERMITTED_CLASSIFICATIONS))
def test_classification_values_are_governance_permitted(classification: str) -> None:
    """Every value this slice may publish to #4977 is in the governance set."""
    assert classification in PERMITTED_CLASSIFICATIONS
