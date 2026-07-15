"""Tests for the issue #5034 → parent #4977 diagnostic-only classification (DoD #4).

This closes the last open Definition-of-Done item of issue #5034: the parent
issue #4977 must be updated with the latency-sweep result classification. The
authorized native 7,344-episode fixed-scope campaign (the `nominal benchmark
evidence` tier) has not run, so the honest, fail-closed classification of the
CPU-completed sweep is `diagnostic-only`, which issue #5034 explicitly lists as a
valid result classification. These tests assert that the durable #5648 evidence
bundle already carries that classification and that it is a permitted value, so
the published parent record cannot over-claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "docs/context/evidence/issue_5034_control_action_latency_sweep"
SUMMARY_PATH = BUNDLE_DIR / "summary.json"

#: Permitted result classifications declared by issue #5034's Definition of Done
#: and `docs/benchmark_governance.md`.
PERMITTED_CLASSIFICATIONS = frozenset(
    {"nominal benchmark evidence", "diagnostic-only", "blocked", "not benchmark evidence"}
)


def _load_bundle_summary() -> dict:
    """Return the durable #5648 latency-evidence summary packet."""
    assert SUMMARY_PATH.exists(), f"durable latency-evidence bundle missing: {SUMMARY_PATH}"
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_bundle_classifies_as_diagnostic_only() -> None:
    """The completed CPU sweep is diagnostic-only, never nominal benchmark evidence."""
    summary = _load_bundle_summary()
    assert summary["result_classification"] == "diagnostic-only"
    assert summary["result_classification"] in PERMITTED_CLASSIFICATIONS
    assert summary["evidence_tier"] == "targeted smoke"


def test_bundle_points_at_parent_issue_4977() -> None:
    """The evidence packet records its parent issue so the classification propagates."""
    summary = _load_bundle_summary()
    assert summary.get("parent_issue") == 4977
    assert summary.get("issue") == 5034


def test_bundle_is_full_latency_coverage_without_exclusions() -> None:
    """The diagnostic-only classification rests on complete native 0/1/3 coverage.

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
    assert summary["scope"]["result_row_count"] == 36
    assert summary["exclusions"]["excluded_row_count"] == 0


@pytest.mark.parametrize("classification", sorted(PERMITTED_CLASSIFICATIONS))
def test_classification_values_are_governance_permitted(classification: str) -> None:
    """Every value this slice may publish to #4977 is in the governance set."""
    assert classification in PERMITTED_CLASSIFICATIONS
