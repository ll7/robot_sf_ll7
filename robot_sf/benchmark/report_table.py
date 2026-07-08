"""Baseline comparison table generator.

Compute per-group means for a set of metrics and format as Markdown/CSV/JSON.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING

from robot_sf.benchmark.aggregate import (
    ensure_observation_track_policy,
    normalize_observation_track_mode,
    observation_track_group_label,
    resolve_benchmark_track,
)
from robot_sf.benchmark.figures.style import metric_label
from robot_sf.benchmark.grouping import resolve_report_group_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

Record = dict[str, object]


def _get_dotted(d: dict[str, object], path: str, default=None):
    """Get nested dict value via dotted path.

    Args:
        d: Dictionary to navigate.
        path: Dot-separated key path (e.g., "metrics.success").
        default: Value to return if path is not found.

    Returns:
        The value at the dotted path, or default if not found.
    """
    cur: object = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:  # type: ignore[redundant-expr]
            return default
        cur = cur[part]  # type: ignore[index]
    return cur


def _group_key(rec: Record, group_by: str, fallback_group_by: str) -> str | None:
    """Resolve the grouping key from a record with fallback.

    Returns:
        Group key string or None if missing.
    """
    return resolve_report_group_key(
        rec,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        missing="skip",
    )


@dataclass
class TableRow:
    """Row of aggregated metric means for a group."""

    group: str
    values: dict[str, float | None]
    benchmark_track: str | None = None


def compute_table(
    records: Iterable[Record],
    metrics: Sequence[str],
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    observation_track_mode: str = "strict",
) -> list[TableRow]:
    """Compute mean metrics per group.

    Returns:
        List of TableRow entries.
    """
    record_list = [dict(record) for record in records]
    observation_track_meta = ensure_observation_track_policy(
        record_list,
        observation_track_mode=observation_track_mode,
    )
    mode = normalize_observation_track_mode(str(observation_track_meta["mode"]))
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    tracks: dict[str, str | None] = {}
    for rec in record_list:
        g = _group_key(rec, group_by, fallback_group_by)
        if g is None:
            continue
        track = resolve_benchmark_track(rec)
        display_track = None if track == "unspecified" else track
        g = observation_track_group_label(rec, g, mode=mode)
        m = _get_dotted(rec, "metrics", {}) or {}
        if not isinstance(m, dict):
            continue
        tracks[g] = display_track
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
        rows.append(TableRow(group=g, values=out, benchmark_track=tracks.get(g)))
    # Stable order by group name
    rows.sort(key=lambda r: r.group)
    return rows


def format_markdown(rows: Sequence[TableRow], metrics: Sequence[str]) -> str:
    """Format table rows as a Markdown table.

    Returns:
        Markdown table string with metric labels and units.
    """
    include_track = any(row.benchmark_track for row in rows)
    # Use formatted metric labels with units for headers
    metric_headers = [metric_label(m) for m in metrics]
    headers = ["Benchmark Track", "Group", *metric_headers] if include_track else ["Group", *metric_headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        vals = [r.benchmark_track or "", r.group] if include_track else [r.group]
        for name in metrics:
            v = r.values.get(name)
            vals.append("" if v is None else f"{v:.4f}")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def format_csv(rows: Sequence[TableRow], metrics: Sequence[str]) -> str:
    """Format table rows as CSV.

    Returns:
        CSV string for the table.
    """
    buf = StringIO()
    writer = csv.writer(buf)
    include_track = any(row.benchmark_track for row in rows)
    writer.writerow(
        ["Benchmark Track", "Group", *metrics] if include_track else ["Group", *metrics]
    )
    for r in rows:
        prefix = [r.benchmark_track or "", r.group] if include_track else [r.group]
        writer.writerow(
            [
                *prefix,
                *[("" if r.values.get(m) is None else f"{r.values[m]:.4f}") for m in metrics],
            ],
        )
    return buf.getvalue()


def to_json(rows: Sequence[TableRow]) -> list[dict[str, object]]:
    """Return JSON-serializable list of dicts for the table rows.

    Args:
        rows: Table rows to serialize.

    Returns:
        List of dicts with group and metric values.
    """
    out: list[dict[str, object]] = []
    for r in rows:
        row: dict[str, object] = {"group": r.group, **r.values}
        if r.benchmark_track:
            row["benchmark_track"] = r.benchmark_track
        out.append(row)
    return out


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in text cells.

    Returns:
        Text with special LaTeX characters escaped.
    """
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
    - Metric headers include units from the shared metric labels mapping.

    Returns:
        LaTeX table source code using booktabs package.
    """
    # Column alignment: one 'l' plus one 'r' per metric
    include_track = any(row.benchmark_track for row in rows)
    col_spec = ("ll" if include_track else "l") + ("r" * len(metrics))
    lines: list[str] = []
    lines.append("% Auto-generated by robot_sf.benchmark.report_table.format_latex_booktabs")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    # Use formatted metric labels with units for headers
    metric_headers = [metric_label(m) for m in metrics]
    header = ["Benchmark Track", "Group", *metric_headers] if include_track else ["Group", *metric_headers]
    header_escaped = [_latex_escape(h) for h in header]
    header_line = " & ".join(header_escaped)
    lines.append(header_line + " \\")
    lines.append("\\midrule")
    # Body
    for r in rows:
        row_vals = (
            [_latex_escape(r.benchmark_track or ""), _latex_escape(r.group)]
            if include_track
            else [_latex_escape(r.group)]
        )
        for name in metrics:
            v = r.values.get(name)
            row_vals.append("" if v is None else f"{float(v):.4f}")
        row_line = " & ".join(row_vals)
        lines.append(row_line + " \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"
