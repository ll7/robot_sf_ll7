"""Baseline comparison table generator.

Compute per-group means for a set of metrics and format as Markdown/CSV/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

Record = Dict[str, object]


def _get_dotted(d: Dict[str, object], path: str, default=None):
    cur: object = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:  # type: ignore[redundant-expr]
            return default
        cur = cur[part]  # type: ignore[index]
    return cur


def _group_key(rec: Record, group_by: str, fallback_group_by: str) -> str | None:
    g = _get_dotted(rec, group_by)
    if g is None:
        g = _get_dotted(rec, fallback_group_by)
    return None if g is None else str(g)


@dataclass
class TableRow:
    group: str
    values: Dict[str, float | None]


def compute_table(
    records: Iterable[Record],
    metrics: Sequence[str],
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
) -> List[TableRow]:
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
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
            except Exception:
                continue
            gs[name] = gs.get(name, 0.0) + fv
            gc[name] = gc.get(name, 0) + 1
    rows: List[TableRow] = []
    for g, gs in sums.items():
        gc = counts.get(g, {})
        out: Dict[str, float | None] = {}
        for name in metrics:
            c = gc.get(name, 0)
            out[name] = (gs.get(name, 0.0) / c) if c > 0 else None
        rows.append(TableRow(group=g, values=out))
    # Stable order by group name
    rows.sort(key=lambda r: r.group)
    return rows


def format_markdown(rows: Sequence[TableRow], metrics: Sequence[str]) -> str:
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
    import csv
    from io import StringIO

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Group", *metrics])
    for r in rows:
        writer.writerow(
            [r.group, *[("" if r.values.get(m) is None else f"{r.values[m]:.4f}") for m in metrics]]
        )
    return buf.getvalue()


def to_json(rows: Sequence[TableRow]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        out.append({"group": r.group, **r.values})
    return out
