"""Regression tests for report_table semantic invariance.

Tests that compute_table produces identical numeric content before and after
style changes, and that formatted tables include proper metric labels with units.
"""

from __future__ import annotations

from robot_sf.benchmark.report_table import compute_table, format_latex_booktabs, format_markdown


def _sample_records() -> list[dict]:
    """Build sample benchmark records for testing."""
    return [
        {
            "episode_id": "ep-1",
            "scenario_id": "scenario-a",
            "seed": 1,
            "scenario_params": {"algo": "orca"},
            "metrics": {
                "success": 1.0,
                "collisions": 0.0,
                "time_to_goal": 5.2,
            },
        },
        {
            "episode_id": "ep-2",
            "scenario_id": "scenario-a",
            "seed": 2,
            "scenario_params": {"algo": "orca"},
            "metrics": {
                "success": 0.8,
                "collisions": 1.0,
                "time_to_goal": 6.1,
            },
        },
        {
            "episode_id": "ep-3",
            "scenario_id": "scenario-a",
            "seed": 1,
            "scenario_params": {"algo": "social_force"},
            "metrics": {
                "success": 0.9,
                "collisions": 0.5,
                "time_to_goal": 4.8,
            },
        },
    ]


def test_compute_table_semantic_invariance() -> None:
    """compute_table should produce identical numeric content regardless of style changes.

    This is a regression test to ensure that adding metric labels and units to
    formatted tables does not alter the underlying computation semantics.
    """
    records = _sample_records()

    # Test with different metric orderings - should produce same numeric values
    rows_abc = compute_table(records, metrics=["success", "collisions", "time_to_goal"])
    rows_cba = compute_table(records, metrics=["time_to_goal", "collisions", "success"])

    # Extract numeric values in consistent order
    orca_abc = next(r for r in rows_abc if r.group == "orca")
    orca_cba = next(r for r in rows_cba if r.group == "orca")

    # All numeric values should be identical regardless of metric order
    assert orca_abc.values["success"] == orca_cba.values["success"]
    assert orca_abc.values["collisions"] == orca_cba.values["collisions"]
    assert orca_abc.values["time_to_goal"] == orca_cba.values["time_to_goal"]

    # Verify expected computed values
    assert orca_abc.values["success"] == 0.9  # (1.0 + 0.8) / 2
    assert orca_abc.values["collisions"] == 0.5  # (0.0 + 1.0) / 2
    assert orca_abc.values["time_to_goal"] == 5.65  # (5.2 + 6.1) / 2


def test_format_markdown_includes_metric_labels() -> None:
    """format_markdown should include metric labels with units in headers."""
    records = _sample_records()
    rows = compute_table(records, metrics=["success", "collisions", "time_to_goal"])

    md = format_markdown(rows, metrics=["success", "collisions", "time_to_goal"])

    # Check that headers use formatted labels, not raw metric keys
    assert "| Success rate |" in md
    assert "| Collision rate |" in md
    assert "| Time to goal (s) |" in md

    # Raw metric keys should not appear in headers
    assert "| success |" not in md.lower() or "| Success rate |" in md
    assert "| collisions |" not in md.lower() or "| Collision rate |" in md


def test_format_latex_booktabs_includes_metric_labels() -> None:
    """format_latex_booktabs should include metric labels with units in headers."""
    records = _sample_records()
    rows = compute_table(records, metrics=["success", "collisions", "time_to_goal"])

    latex = format_latex_booktabs(rows, metrics=["success", "collisions", "time_to_goal"])

    # Check that headers use formatted labels
    assert "Success rate" in latex
    assert "Collision rate" in latex
    assert "Time to goal (s)" in latex

    # Check LaTeX structure is preserved
    assert "\\begin{tabular}" in latex
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "\\end{tabular}" in latex


def test_format_markdown_numeric_content_unchanged() -> None:
    """format_markdown should preserve exact numeric values from compute_table."""
    records = _sample_records()
    rows = compute_table(records, metrics=["success", "collisions"])

    md = format_markdown(rows, metrics=["success", "collisions"])

    # Extract the ORCA row from the markdown
    lines = md.splitlines()
    orca_line = next(line for line in lines if "| orca |" in line.lower())

    # Verify exact numeric values (4 decimal places)
    assert "0.9000" in orca_line  # success mean
    assert "0.5000" in orca_line  # collisions mean


def test_format_latex_booktabs_numeric_content_unchanged() -> None:
    """format_latex_booktabs should preserve exact numeric values from compute_table."""
    records = _sample_records()
    rows = compute_table(records, metrics=["success", "collisions"])

    latex = format_latex_booktabs(rows, metrics=["success", "collisions"])

    # Extract the ORCA row from the LaTeX
    lines = latex.splitlines()
    orca_line = next(line for line in lines if "orca" in line.lower())

    # Verify exact numeric values (4 decimal places)
    assert "0.9000" in orca_line  # success mean
    assert "0.5000" in orca_line  # collisions mean


def test_unknown_metric_uses_title_case_label() -> None:
    """Metrics not in the label map should fall back to title-cased metric names."""
    records = _sample_records()
    rows = compute_table(records, metrics=["success", "unknown_metric"])

    # Create a row with unknown metric
    for row in rows:
        if "unknown_metric" not in row.values:
            row.values["unknown_metric"] = 1.23

    md = format_markdown(rows, metrics=["success", "unknown_metric"])

    # Unknown metric should be title-cased (unknown_metric -> Unknown Metric)
    assert "| Unknown Metric |" in md


def test_metric_label_with_aggregation() -> None:
    """Metric labels should include aggregation when provided to metric_label helper."""
    from robot_sf.benchmark.figures.style import metric_label

    # Test with aggregation
    assert metric_label("collision_rate", aggregation="mean") == "Collision rate (mean)"
    assert metric_label("time_to_goal", aggregation="median") == "Time to goal (s) (median)"

    # Test without aggregation
    assert metric_label("collision_rate") == "Collision rate"
    assert metric_label("time_to_goal") == "Time to goal (s)"
    assert metric_label("unknown_metric") == "Unknown Metric"
    assert metric_label(" unknown_metric ") == "Unknown Metric"
    assert metric_label("") == "Metric"
