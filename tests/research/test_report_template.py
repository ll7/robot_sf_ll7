"""Tests for research report template rendering."""

from __future__ import annotations

from pathlib import Path

from robot_sf.research.report_template import MarkdownReportRenderer


def test_render_report_includes_sections(tmp_path: Path) -> None:
    """Ensure report renderer writes expected sections to disk."""
    renderer = MarkdownReportRenderer(tmp_path)
    hypothesis = {
        "decision": "PASS",
        "measured_value": 55.0,
        "threshold_value": 40.0,
        "description": "BC improves sample efficiency.",
        "note": "All runs completed.",
    }
    metrics = [
        {
            "condition": "baseline",
            "metric_name": "success_rate",
            "mean": 0.4,
            "median": 0.4,
            "p95": 0.5,
            "ci_low": 0.3,
            "ci_high": 0.5,
        },
        {
            "condition": "pretrained",
            "metric_name": "success_rate",
            "mean": 0.6,
            "median": 0.6,
            "p95": 0.7,
            "ci_low": 0.5,
            "ci_high": 0.7,
            "effect_size": 0.9,
        },
    ]
    report_path = renderer.render(
        experiment_name="Test Report",
        hypothesis_result=hypothesis,
        aggregated_metrics=metrics,
        figures=[],
        metadata={"run_id": "run-123"},
        seed_status=[{"seed": 1, "baseline_status": "ok", "pretrained_status": "ok"}],
        completeness={"score": 50, "completed": 1, "expected": 2, "missing_seeds": [2]},
    )
    content = Path(report_path).read_text(encoding="utf-8")
    assert "## Abstract" in content
    assert "## Results" in content
    assert "## Seed Summary" in content


def test_stats_table_labels_effect_size(tmp_path: Path) -> None:
    """Check effect size labeling for statistical summary table."""
    renderer = MarkdownReportRenderer(tmp_path)
    table = renderer._render_stats_table(
        [
            {"metric_name": "snqi", "condition": "baseline", "effect_size": 0.9},
        ]
    )
    assert "large" in table
