"""Ranking utilities for Social Navigation Benchmark.

Compute group rankings by a selected metric across episode records and output
as structured rows or formatted Markdown/CSV tables.

Programmatic contract
- Input: iterable of episode records (dicts) as produced by read_jsonl
- Grouping: group_by dotted key, fallback_group_by used when missing
- Metric: name under metrics.<metric>
- Output: list of rows: {group, mean, count}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _get_nested(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@dataclass
class RankingRow:
    group: str
    mean: float
    count: int


def compute_ranking(
    records: Iterable[Dict[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    metric: str = "collisions",
    ascending: bool = True,
    top: Optional[int] = None,
) -> List[RankingRow]:
    """Compute ranking by mean of metrics.<metric> per group.

    - Missing/non-numeric metric values are ignored.
    - Groups with no valid values are omitted.
    - Sorting is ascending by default (smaller-is-better). Use ascending=False for higher-is-better metrics.
    """
    by_group: Dict[str, List[float]] = {}
    for rec in records:
        gid = _get_nested(rec, group_by)
        if gid is None:
            gid = _get_nested(rec, fallback_group_by)
        if gid is None:
            continue
        val = _to_float((rec.get("metrics") or {}).get(metric))
        if val is None:
            continue
        by_group.setdefault(str(gid), []).append(val)

    rows: List[RankingRow] = []
    for gid, vals in by_group.items():
        if not vals:
            continue
        m = sum(vals) / float(len(vals))
        rows.append(RankingRow(group=gid, mean=m, count=len(vals)))

    rows.sort(key=lambda r: r.mean, reverse=not ascending)
    if top is not None:
        rows = rows[: int(top)]
    return rows


def format_markdown(rows: Sequence[RankingRow], metric: str) -> str:
    """Return a Markdown table for the ranking rows."""
    header = f"| Rank | Group | mean({metric}) | count |\n|---:|---|---:|---:|"
    lines = [header]
    for i, r in enumerate(rows, start=1):
        lines.append(f"| {i} | {r.group} | {r.mean:.6g} | {r.count} |")
    return "\n".join(lines) + "\n"


def format_csv(rows: Sequence[RankingRow], metric: str) -> str:
    """Return a CSV string for the ranking rows."""
    lines = ["rank,group,mean_" + metric + ",count"]
    for i, r in enumerate(rows, start=1):
        lines.append(f"{i},{r.group},{r.mean:.6g},{r.count}")
    return "\n".join(lines) + "\n"


__all__ = ["RankingRow", "compute_ranking", "format_markdown", "format_csv"]
