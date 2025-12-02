"""Module test_contract_format_summary_table auto-generated docstring."""

import pytest

from robot_sf.benchmark.utils import format_summary_table


def test_format_summary_table_empty_raises():
    """Test format summary table empty raises.

    Returns:
        Any: Auto-generated placeholder description.
    """
    with pytest.raises(ValueError):
        format_summary_table({})


def test_format_summary_table_markdown():
    """Test format summary table markdown.

    Returns:
        Any: Auto-generated placeholder description.
    """
    metrics = {"success_rate": 0.95, "collision_count": 2}
    out = format_summary_table(metrics)
    assert "| Metric | Value |" in out
    assert "success_rate" in out
    assert "0.95" in out or "0.95" in out
