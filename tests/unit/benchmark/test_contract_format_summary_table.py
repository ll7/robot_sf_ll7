"""Contract tests for ``format_summary_table`` markdown rendering."""

import pytest

from robot_sf.benchmark.utils import format_summary_table


def test_format_summary_table_empty_raises():
    """Verify an empty metrics mapping raises ValueError."""
    with pytest.raises(ValueError):
        format_summary_table({})


def test_format_summary_table_markdown():
    """Verify the rendered table contains the header row and metric values."""
    metrics = {"success_rate": 0.95, "collision_count": 2}
    out = format_summary_table(metrics)
    assert "| Metric | Value |" in out
    assert "success_rate" in out
    assert "0.95" in out or "0.95" in out
