"""SNQI ablation utilities.

Compute the effect of removing individual SNQI components on group rankings.

Definitions
- Base ranking: mean SNQI per group using provided weights/baseline.
- Ablation i: set the i-th weight to 0 (others unchanged), recompute mean SNQI per
  group, derive new ranking, and measure each group's rank shift relative to base.

Outputs
- Programmatic structure with per-group base stats and per-weight deltas.
- Formatters for Markdown/CSV/JSON for quick reporting.

Assumptions
- Records are episode dicts with a `metrics` mapping (as produced by the benchmark).
- Weights and baseline follow the canonical SNQI package types.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES, compute_snqi


def _get_nested(d: Mapping[str, Any], dotted: str, default: Any | None = None) -> Any:
    """Get nested.

    Args:
        d: Dictionary of metric values.
        dotted: Matplotlib dotted-line style.
        default: Default fallback value.

    Returns:
        Any: Arbitrary value passed through unchanged.
    """
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _group_by(
    records: Iterable[Mapping[str, Any]],
    group_by: str,
    fallback: str,
) -> dict[str, list[Mapping[str, Any]]]:
    """Group by.

    Args:
        records: List of serialized records.
        group_by: group by.
        fallback: Fallback option when primary data is missing.

    Returns:
        dict[str, list[Mapping[str, Any]]]: mapping of str, list[Mapping[str, Any]].
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        gid = _get_nested(rec, group_by)
        if gid is None:
            gid = _get_nested(rec, fallback)
        if gid is None:
            gid = "unknown"
        groups.setdefault(str(gid), []).append(rec)
    return groups


def _mean(values: Sequence[float]) -> float:
    """Mean.

    Args:
        values: Collection of numeric values.

    Returns:
        float: Floating-point value.
    """
    return float(sum(values) / len(values)) if values else float("nan")


def _episode_snqi(
    rec: Mapping[str, Any],
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> float:
    """Episode snqi.

    Args:
        rec: Single record dictionary.
        weights: Weight dictionary.
        baseline: Baseline statistics bundle.

    Returns:
        float: Floating-point value.
    """
    return float(compute_snqi(rec.get("metrics", {}), weights, baseline))


def _compute_group_means(
    records: Iterable[Mapping[str, Any]],
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
    group_by: str,
    fallback_group_by: str,
) -> dict[str, float]:
    """Compute group means.

    Args:
        records: List of serialized records.
        weights: Weight dictionary.
        baseline: Baseline statistics bundle.
        group_by: group by.
        fallback_group_by: fallback group by.

    Returns:
        dict[str, float]: mapping of str, float.
    """
    groups = _group_by(records, group_by, fallback_group_by)
    means: dict[str, float] = {}
    for gid, rows in groups.items():
        vals: list[float] = []
        for rec in rows:
            try:
                vals.append(_episode_snqi(rec, weights, baseline))
            except Exception:
                continue
        if vals:
            means[gid] = _mean(vals)
    return means


def _ranking_from_means(
    means: Mapping[str, float],
    ascending: bool = False,
) -> list[tuple[str, float, int]]:
    """Ranking from means.

    Args:
        means: Mean metric values.
        ascending: Whether sorting should be ascending.

    Returns:
        list[tuple[str, float, int]]: list of tuple[str, float, int].
    """
    # We treat higher SNQI as better -> descending by default; ascending flag kept for symmetry.
    items = [(g, float(m), 0) for g, m in means.items()]
    items.sort(key=lambda t: t[1], reverse=not ascending)
    # Count not tracked here; include 0 placeholder for compatibility
    return items


@dataclass
class AblationRow:
    """AblationRow class."""

    group: str
    base_rank: int
    base_mean: float
    deltas: dict[str, float]  # weight_name -> rank_shift (ablate_rank - base_rank)


def compute_snqi_ablation(
    records: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    top: int | None = None,
) -> list[AblationRow]:
    """Compute per-group rank shifts for one-at-a-time SNQI ablations.

    Returns a list of AblationRow with base mean SNQI, base rank, and per-weight
    rank shifts (positive = moved down/worse if higher-is-better baseline).
    """
    base_means = _compute_group_means(records, weights, baseline, group_by, fallback_group_by)
    base_ranked = _ranking_from_means(base_means, ascending=False)
    base_positions: dict[str, int] = {g: i + 1 for i, (g, _m, _c) in enumerate(base_ranked)}

    # Initialize rows for all groups seen in base ranking
    rows_by_group: dict[str, AblationRow] = {
        g: AblationRow(group=g, base_rank=base_positions[g], base_mean=base_means[g], deltas={})
        for g in base_positions
    }

    for wname in WEIGHT_NAMES:
        # Skip components absent in provided weights to avoid surprising zeros
        if wname not in weights:
            continue
        ablated_weights: dict[str, float] = dict(weights)
        ablated_weights[wname] = 0.0
        means = _compute_group_means(
            records,
            ablated_weights,
            baseline,
            group_by,
            fallback_group_by,
        )
        ranked = _ranking_from_means(means, ascending=False)
        pos = {g: i + 1 for i, (g, _m, _c) in enumerate(ranked)}
        for g, row in rows_by_group.items():
            new_pos = pos.get(g, row.base_rank)  # if missing, assume unchanged position
            row.deltas[wname] = float(new_pos - row.base_rank)

    # Return rows in base ranking order
    ordered = sorted(rows_by_group.values(), key=lambda r: r.base_rank)
    if isinstance(top, int) and top > 0:
        ordered = ordered[:top]
    return ordered


def format_markdown(rows: Sequence[AblationRow]) -> str:
    """Format markdown.

    Args:
        rows: Row definitions for table output.

    Returns:
        str: String value.
    """
    headers = ["Rank", "Group", "base_mean", *list(WEIGHT_NAMES)]
    # Only include weights that appear in any row
    used_weights = [w for w in WEIGHT_NAMES if any(w in r.deltas for r in rows)]
    headers = ["Rank", "Group", "base_mean", *used_weights]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:", "---", ":---:"] + [":---:"] * len(used_weights)) + "|",
    ]
    for r in rows:
        cells = [
            str(r.base_rank),
            r.group,
            f"{r.base_mean:.6g}",
        ] + [f"{r.deltas.get(w, 0.0):+.0f}" for w in used_weights]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def format_csv(rows: Sequence[AblationRow]) -> str:
    """Format csv.

    Args:
        rows: Row definitions for table output.

    Returns:
        str: String value.
    """
    used_weights = [w for w in WEIGHT_NAMES if any(w in r.deltas for r in rows)]
    headers = ["rank", "group", "base_mean"] + [f"delta_{w}" for w in used_weights]
    lines = [",".join(headers)]
    for r in rows:
        cells = [str(r.base_rank), r.group, f"{r.base_mean:.6g}"] + [
            f"{r.deltas.get(w, 0.0):+.0f}" for w in used_weights
        ]
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def to_json(rows: Sequence[AblationRow]) -> list[dict[str, Any]]:
    """To json.

    Args:
        rows: Row definitions for table output.

    Returns:
        list[dict[str, Any]]: list of dict[str, Any].
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        item = {
            "group": r.group,
            "base_rank": r.base_rank,
            "base_mean": r.base_mean,
            "deltas": dict(r.deltas),
        }
        out.append(item)
    return out


def compute_ablation_summary(rows: Sequence[AblationRow]) -> dict[str, dict[str, float]]:
    """Compute per-weight summary statistics from ablation rows.

    Returns mapping weight_name -> {changed, mean_abs, max_abs, pos, neg, mean}.
    All values except counts are floats; counts are returned as floats for JSON uniformity.
    """
    # Gather all weight names that appear in any row
    weight_names = sorted({w for r in rows for w in r.deltas})
    summary: dict[str, dict[str, float]] = {}
    for w in weight_names:
        vals: list[float] = []
        pos = 0
        neg = 0
        for r in rows:
            v = float(r.deltas.get(w, 0.0))
            if v != 0.0:
                vals.append(v)
                if v > 0:
                    pos += 1
                elif v < 0:
                    neg += 1
        if vals:
            abs_vals = [abs(v) for v in vals]
            summary[w] = {
                "changed": float(len(vals)),
                "mean_abs": float(sum(abs_vals) / len(abs_vals)),
                "max_abs": float(max(abs_vals)),
                "pos": float(pos),
                "neg": float(neg),
                "mean": float(sum(vals) / len(vals)),
            }
        else:
            summary[w] = {
                "changed": 0.0,
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "mean": 0.0,
            }
    return summary


__all__ = [
    "AblationRow",
    "compute_ablation_summary",
    "compute_snqi_ablation",
    "format_csv",
    "format_markdown",
    "to_json",
]
