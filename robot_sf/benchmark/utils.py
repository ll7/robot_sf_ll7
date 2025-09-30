from __future__ import annotations

"""Small utility helpers for benchmark reporting.

Phase A helpers are intentionally small and pure. The public function
``format_summary_table`` follows the contract in
``specs/139-extract-reusable-helpers/contracts/visualization_helper_contract.md``:
- accepts a mapping ``dict[str, float]`` of metrics
- returns a Markdown table string
- raises ValueError on empty input
"""


def format_summary_table(metrics: dict[str, float]) -> str:
    """Format a metrics mapping as a simple Markdown table.

    Args:
        metrics: mapping of metric name to numeric value.

    Returns:
        A Markdown formatted table as a string.

    Raises:
        ValueError: if `metrics` is empty.
    """
    if not metrics:
        raise ValueError("metrics must not be empty")

    lines = ["| Metric | Value |", "|---|---:|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value} |")

    return "\n".join(lines)
