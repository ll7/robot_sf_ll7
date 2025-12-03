"""Baseline comparison table generator.

Compute per-group means for a set of metrics and format as Markdown/CSV/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

Record = dict[str, object]


def _get_dotted(d: dict[str, object], path: str, default=None):
    """Get dotted.

    Args:
        d: Dictionary of metric values.
        path: Filesystem path to the resource.
        default: Default fallback value.

    Returns:
        Any: Arbitrary value passed through unchanged.
    """
    cur: object = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:  # type: ignore[redundant-expr]
            return default
        cur = cur[part]  # type: ignore[index]
    return cur


def _group_key(rec: Record, group_by: str, fallback_group_by: str) -> str | None:
    """Group key.

    Args:
        rec: Single record dictionary.
        group_by: group by.
        fallback_group_by: fallback group by.

    Returns:
        str | None: Optional string value.
    """
    g = _get_dotted(rec, group_by)
    if g is None:
        g = _get_dotted(rec, fallback_group_by)
    return None if g is None else str(g)


@dataclass
class TableRow:
    """TableRow class."""

    group: str
    values: dict[str, float | None]


def compute_table(
    records: Iterable[Record],
    metrics: Sequence[str],
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
) -> list[TableRow]:
    """Compute table.

    Args:
        records: List of serialized records.
        metrics: Dictionary of computed metrics.
        group_by: group by.
        fallback_group_by: fallback group by.

    Returns:
        list[TableRow]: list of TableRow.
    """
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for rec in records:
        g = _group_key(rec, group_by, fallback_group_by)
        if g is None:
            continue
        m = _get_dotted(rec, "metrics", {}) or {}
        if not isinstance(m, dict):
            continue
        gs = sums.setdefault(g, {})
        gc = counts.setdefault(g, {})
        for name in metrics:
            val = m.get(name)
            if val is None:
                continue
            try:
                fv = float(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            gs[name] = gs.get(name, 0.0) + fv
            gc[name] = gc.get(name, 0) + 1
    rows: list[TableRow] = []
    for g, gs in sums.items():
        gc = counts.get(g, {})
        out: dict[str, float | None] = {}
        for name in metrics:
            c = gc.get(name, 0)
            out[name] = (gs.get(name, 0.0) / c) if c > 0 else None
        rows.append(TableRow(group=g, values=out))
    # Stable order by group name
    rows.sort(key=lambda r: r.group)
    return rows


def format_markdown(rows: Sequence[TableRow], metrics: Sequence[str]) -> str:
    """Format markdown.

    Args:
        rows: Row definitions for table output.
        metrics: Dictionary of computed metrics.

    Returns:
        str: String value.
    """
    headers = ["Group", *metrics]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        vals = [r.group]
        for name in metrics:
            v = r.values.get(name)
            vals.append("" if v is None else f"{v:.4f}")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def format_csv(rows: Sequence[TableRow], metrics: Sequence[str]) -> str:
    """Format csv.

    Args:
        rows: Row definitions for table output.
        metrics: Dictionary of computed metrics.

    Returns:
        str: String value.
    """
    import csv
    from io import StringIO

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Group", *metrics])
    for r in rows:
        writer.writerow(
            [
                r.group,
                *[("" if r.values.get(m) is None else f"{r.values[m]:.4f}") for m in metrics],
            ],
        )
    return buf.getvalue()


def to_json(rows: Sequence[TableRow]) -> list[dict[str, object]]:
    """To json.

    Args:
        rows: Row definitions for table output.

    Returns:
        list[dict[str, object]]: list of dict[str, object].
    """
    out: list[dict[str, object]] = []
    for r in rows:
        out.append({"group": r.group, **r.values})
    return out


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in text cells."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def format_latex_booktabs(rows: Sequence[TableRow], metrics: Sequence[str]) -> str:
    r"""Return a LaTeX table using booktabs with numeric columns right-aligned.

    Notes
    - Assumes the caller includes \usepackage{booktabs} in the preamble.
    - Column spec: l for Group, r for each numeric metric column.
    - Values are formatted to 4 decimals; missing values render as empty.
    """
    # Column alignment: one 'l' plus one 'r' per metric
    col_spec = "l" + ("r" * len(metrics))
    lines: list[str] = []
    lines.append("% Auto-generated by robot_sf.benchmark.report_table.format_latex_booktabs")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = ["Group", *metrics]
    header_escaped = [_latex_escape(h) for h in header]
    header_line = " & ".join(header_escaped)
    lines.append(header_line + " \\")
    lines.append("\\midrule")
    # Body
    for r in rows:
        row_vals: list[str] = [_latex_escape(r.group)]
        for name in metrics:
            v = r.values.get(name)
            row_vals.append("" if v is None else f"{float(v):.4f}")
        row_line = " & ".join(row_vals)
        lines.append(row_line + " \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"
