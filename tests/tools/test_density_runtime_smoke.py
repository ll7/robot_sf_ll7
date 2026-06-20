"""Tests for the diagnostic density runtime smoke helper."""

from __future__ import annotations

from scripts.tools.density_runtime_smoke import (
    classify_failure_semantics,
    select_candidate_rows,
)


def test_select_candidate_rows_uses_lowest_novelty_when_no_redundant_rows() -> None:
    """A complete report can still lack literal merge/drop rows."""
    report = {
        "summary": {"redundant_count": 0},
        "scenario_rows": [
            {
                "scenario_id": "novel_a",
                "novelty_score": 0.9,
                "recommendation": "retain_or_investigate",
            },
            {
                "scenario_id": "novel_b",
                "novelty_score": 0.8,
                "recommendation": "retain_or_investigate",
            },
            {
                "scenario_id": "typical_a",
                "novelty_score": 0.1,
                "recommendation": "review",
            },
            {
                "scenario_id": "typical_b",
                "novelty_score": 0.2,
                "recommendation": "review",
            },
        ],
    }

    selected, summary = select_candidate_rows(report, top_k=2)

    assert summary["comparator_source"] == "lowest_novelty_fallback"
    assert summary["coverage_redundant_count"] == 0
    assert [row["scenario_id"] for row in selected] == [
        "novel_a",
        "novel_b",
        "typical_a",
        "typical_b",
    ]
    assert selected[-1]["selection_group"] == "redundant_comparator"


def test_select_candidate_rows_rejects_malformed_rows() -> None:
    """Malformed coverage payloads should fail with clear ValueErrors."""
    try:
        select_candidate_rows({"scenario_rows": "not-a-list"}, top_k=1)
    except ValueError as exc:
        assert "scenario_rows must be a list" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-list scenario_rows")

    try:
        select_candidate_rows(
            {
                "scenario_rows": [
                    {
                        "scenario_id": "novel_bad",
                        "novelty_score": "not-a-score",
                        "recommendation": "retain_or_investigate",
                    }
                ]
            },
            top_k=1,
        )
    except ValueError as exc:
        assert "invalid novelty_score" in str(exc)
    else:
        raise AssertionError("expected ValueError for malformed novelty_score")


def test_classify_failure_semantics_precedence() -> None:
    """Collision and success take precedence over diagnostic stall labels."""
    assert (
        classify_failure_semantics(
            success=False,
            collision=True,
            near_misses=10.0,
            progress_ratio=0.0,
            progress_stall_threshold=0.05,
        )
        == "collision"
    )
    assert (
        classify_failure_semantics(
            success=True,
            collision=False,
            near_misses=10.0,
            progress_ratio=0.0,
            progress_stall_threshold=0.05,
        )
        == "success"
    )
    assert (
        classify_failure_semantics(
            success=False,
            collision=False,
            near_misses=1.0,
            progress_ratio=0.0,
            progress_stall_threshold=0.05,
        )
        == "near-miss"
    )
    assert (
        classify_failure_semantics(
            success=False,
            collision=False,
            near_misses=0.0,
            progress_ratio=0.01,
            progress_stall_threshold=0.05,
        )
        == "progress_stall"
    )
    assert (
        classify_failure_semantics(
            success=False,
            collision=False,
            near_misses=0.0,
            progress_ratio=0.5,
            progress_stall_threshold=0.05,
        )
        == "horizon_exhausted"
    )
