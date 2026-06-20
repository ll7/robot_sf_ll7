"""Tests for the diagnostic density runtime smoke helper."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from scripts.tools import density_runtime_smoke
from scripts.tools.density_runtime_smoke import (
    classify_failure_semantics,
    run_density_smoke,
    select_candidate_rows,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_select_candidate_rows_tolerates_null_report_fields() -> None:
    """Nullable report fields should not fail score sorting or summary coercion."""
    report = {
        "summary": {"redundant_count": None},
        "scenario_rows": [
            {
                "scenario_id": "novel_null",
                "novelty_score": None,
                "recommendation": "retain_or_investigate",
            },
            {
                "scenario_id": "redundant_null",
                "novelty_score": None,
                "recommendation": "merge_or_drop",
            },
        ],
    }

    selected, summary = select_candidate_rows(report, top_k=1)

    assert summary["coverage_redundant_count"] == 0
    assert [row["scenario_id"] for row in selected] == ["novel_null", "redundant_null"]


def test_select_candidate_rows_rejects_null_scenario_rows() -> None:
    """Null scenario_rows should fail closed with the normal empty-report error."""
    with pytest.raises(ValueError, match="has no scenario_rows"):
        select_candidate_rows({"scenario_rows": None}, top_k=1)


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


def test_run_density_smoke_does_not_stringify_null_ids_or_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Null selected ids and runtime semantics should be explicit missing/unknown values."""
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        """
{
  "summary": {"redundant_count": null},
  "scenario_rows": [
    {
      "scenario_id": null,
      "novelty_score": 1.0,
      "recommendation": "retain_or_investigate"
    },
    {
      "scenario_id": "typical",
      "novelty_score": 0.0,
      "recommendation": "merge_or_drop"
    }
  ]
}
""",
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(
        density_runtime_smoke,
        "load_scenarios",
        lambda *_args, **_kwargs: [{"name": "typical"}],
    )
    monkeypatch.setattr(
        density_runtime_smoke,
        "run_smoke_episode",
        lambda *_args, **_kwargs: {
            "scenario_id": "typical",
            "status": "completed",
            "row_status": "diagnostic_only",
            "failure_semantics": None,
        },
    )

    payload = run_density_smoke(
        argparse.Namespace(
            coverage_json=coverage_json,
            matrix=matrix,
            top_k=1,
            seed=3200,
            horizon=1,
            dt=0.1,
            max_speed=1.0,
            progress_stall_threshold=0.05,
        )
    )

    assert payload["status"] == "blocked"
    assert payload["missing_scenarios"] == ["<missing scenario_id>"]
    assert "None" not in payload["missing_scenarios"]
    assert payload["failure_semantics_counts"] == {"unknown": 1}
    assert payload["selection_summary"]["coverage_redundant_count"] == 0


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
