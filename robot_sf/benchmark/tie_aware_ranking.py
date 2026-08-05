"""Tie-aware partial-order export for validated benchmark comparisons.

This module is an additive analysis surface. It does not change metric values,
the existing fairness gate, the legacy ranking formatter, or any campaign
artifact. Callers must provide support metadata and must declare whether each
row is eligible for comparison. Rows that fail those gates remain explicit
``incomparable`` items.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

TIE_AWARE_RANKING_SCHEMA_VERSION = "tie_aware_ranking.v1"
RELATION_KINDS = frozenset({"strict_before", "exact_tie", "non_identifiable", "incomparable"})
_MISSING = object()


class TieAwareRankingError(ValueError):
    """Raised when a tie-aware ranking input cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class _Item:
    """Canonical internal representation of one comparison item."""

    key: str
    score: Decimal
    score_json: float
    uncertainty: dict[str, Any] | None
    uncertainty_low: Decimal | None
    uncertainty_high: Decimal | None
    support: dict[str, int] | None
    eligible: bool
    comparison_eligible: bool
    comparability_reason: str
    evidence: Any
    declared_display_order: int | float | str | None

    @property
    def has_uncertainty(self) -> bool:
        """Return whether both marginal uncertainty bounds are available."""

        return self.uncertainty_low is not None and self.uncertainty_high is not None


def build_tie_aware_ranking(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str | Mapping[str, Any],
    higher_is_better: bool | None = None,
    display_order: Sequence[str] | None = None,
    pairwise_comparisons: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a deterministic tie-aware partial-order export.

    Args:
        rows: Mappings with ``key``, finite ``score``, and support metadata.
            Support is either ``support: {n, N}`` or top-level ``n`` and
            ``N``. A missing or invalid denominator makes the item
            incomparable. Optional uncertainty is supplied as
            ``uncertainty: {low, high, source}`` or ``interval`` with the same
            fields.
        metric: Metric name or a mapping with ``id`` and
            ``higher_is_better``. A mapping may also declare ``unit``.
        higher_is_better: Direction for a string metric name.
        display_order: Optional caller-declared item layout. It affects output
            layout only and never affects a scientific relation.
        pairwise_comparisons: Optional approved directional comparisons. Each
            row must set ``approved: true``, ``relation: strict_before``, and
            identify ``better`` and ``worse`` (or use ``left`` and ``right``
            in that order). These may provide a validated paired comparison
            when marginal intervals are not sufficient.

    Returns:
        JSON-safe ``tie_aware_ranking.v1`` data.

    Raises:
        TieAwareRankingError: If a required identity, score, support value, or
            approved comparison is malformed.
    """
    metric_payload = _normalise_metric(metric, higher_is_better)
    items = [_normalise_item(row, index) for index, row in enumerate(rows)]
    _ensure_unique_keys(items)
    ordered_items = _order_items(items, display_order)
    overrides = _normalise_pairwise_comparisons(pairwise_comparisons, items)
    relations = _build_relations(ordered_items, overrides)
    groups, tie_groups, group_ids = _build_groups(ordered_items, relations)
    rank_ranges = _compute_rank_ranges(groups, relations, group_ids)
    output_items = [
        _item_payload(item, index, group_ids, rank_ranges)
        for index, item in enumerate(ordered_items, start=1)
    ]
    relation_payload = [dict(relation) for relation in relations]
    summary = _build_summary(ordered_items, relation_payload, tie_groups)
    return {
        "schema_version": TIE_AWARE_RANKING_SCHEMA_VERSION,
        "metric": metric_payload,
        "policy": _policy_payload(),
        "items": output_items,
        "relations": relation_payload,
        "tie_groups": tie_groups,
        "summary": summary,
    }


def render_tie_aware_summary(payload: Mapping[str, Any]) -> str:
    """Render a stable human-facing summary without assigning total ranks.

    Returns:
        Markdown summary with display order, rank ranges, and relation counts.
    """
    metric = payload["metric"]
    summary = payload["summary"]
    lines = [
        "# Tie-aware ranking summary",
        "",
        f"Metric: `{metric['id']}` ({_direction_label(metric)})",
        "Scientific rank is represented only as a rank range; display order is not a rank.",
        "",
        "| Display order | Item | Rank range | Score | Support | Comparability |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for item in payload["items"]:
        rank = _format_rank_range(item.get("rank_range"))
        score = _format_number(item["score"])
        support = _format_support(item.get("support"))
        status = item["comparability"]["status"]
        lines.append(
            f"| {item['display_order']} | {item['key']} | {rank} | {score} | {support} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Relation summary",
            "",
            f"Items: {summary['item_count']}; comparison-eligible: "
            f"{summary['comparison_eligible_item_count']}; "
            f"incomparable: {summary['incomparable_item_count']}",
            f"Relations: {summary['relation_count']} ({summary['relation_counts']})",
            f"Exact tie groups: {summary['exact_tie_group_count']}",
        ]
    )
    for group in payload["tie_groups"]:
        lines.append(f"- Exact tie: {', '.join(group['members'])}")
    for relation in payload["relations"]:
        if relation["relation"] == "strict_before":
            lines.append(
                f"- `{relation['better']}` before `{relation['worse']}` ({relation['reason']})"
            )
        elif relation["relation"] != "exact_tie":
            lines.append(
                f"- `{relation['left']}` / `{relation['right']}:` "
                f"{relation['relation']} ({relation['reason']})"
            )
    return "\n".join(lines) + "\n"


def _normalise_metric(
    metric: str | Mapping[str, Any], higher_is_better: bool | None
) -> dict[str, Any]:
    if isinstance(metric, Mapping):
        metric_id = metric.get("id", metric.get("metric_id"))
        direction = metric.get("higher_is_better", higher_is_better)
        result = {key: metric[key] for key in ("id", "unit", "description") if key in metric}
    else:
        metric_id = metric
        direction = higher_is_better
        result = {}
    if not isinstance(metric_id, str) or not metric_id.strip():
        raise TieAwareRankingError("metric id must be a non-empty string")
    if not isinstance(direction, bool):
        raise TieAwareRankingError("higher_is_better must be declared as a boolean")
    result["id"] = metric_id.strip()
    result["higher_is_better"] = direction
    result["desirability"] = "higher_is_better" if direction else "lower_is_better"
    return result


def _normalise_item(row: Mapping[str, Any], index: int) -> _Item:
    if not isinstance(row, Mapping):
        raise TieAwareRankingError(f"row {index} must be a mapping")
    key = row.get("key", row.get("group"))
    if not isinstance(key, str) or not key.strip():
        raise TieAwareRankingError(f"row {index} key must be a non-empty string")
    score = _decimal(row.get("score"), f"rows[{index}].score")
    score_json = _json_float(score, f"rows[{index}].score")
    uncertainty, low, high = _normalise_uncertainty(row, index)
    support, support_reason = _normalise_support(row, index)
    declared_eligible, eligible_reason = _declared_comparability(row, index)
    reason = eligible_reason or support_reason or _evidence_reason(row)
    return _Item(
        key=key.strip(),
        score=score,
        score_json=score_json,
        uncertainty=uncertainty,
        uncertainty_low=low,
        uncertainty_high=high,
        support=support,
        eligible=declared_eligible,
        comparison_eligible=reason is None,
        comparability_reason=reason or "caller_declared_comparable",
        evidence=row.get("evidence"),
        declared_display_order=row.get("display_order"),
    )


def _decimal(value: Any, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise TieAwareRankingError(f"{field} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise TieAwareRankingError(f"{field} must be a finite number") from exc
    if not result.is_finite():
        raise TieAwareRankingError(f"{field} must be a finite number")
    return result


def _json_float(value: Decimal, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TieAwareRankingError(f"{field} is outside JSON numeric range")
    return result


def _normalise_uncertainty(
    row: Mapping[str, Any], index: int
) -> tuple[dict[str, Any] | None, Decimal | None, Decimal | None]:
    raw = row.get("interval", _MISSING)
    source = None
    if raw is _MISSING:
        raw = row.get("uncertainty", _MISSING)
    if raw is _MISSING:
        return None, None, None
    if isinstance(raw, Mapping) and "interval" in raw:
        source = raw.get("source", raw.get("method"))
        raw = raw["interval"]
    elif isinstance(raw, Mapping):
        source = raw.get("source", raw.get("method"))
    if isinstance(raw, Mapping):
        low_value = raw.get("low")
        high_value = raw.get("high")
        source = source if source is not None else raw.get("source", raw.get("method"))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        low_value, high_value = raw
    else:
        raise TieAwareRankingError(f"rows[{index}].uncertainty must contain low and high")
    low = _decimal(low_value, f"rows[{index}].uncertainty.low")
    high = _decimal(high_value, f"rows[{index}].uncertainty.high")
    if low > high:
        raise TieAwareRankingError(f"rows[{index}].uncertainty low must not exceed high")
    source_value = None if source is None else str(source)
    return (
        {
            "low": _json_float(low, f"rows[{index}].uncertainty.low"),
            "high": _json_float(high, f"rows[{index}].uncertainty.high"),
            "source": source_value,
        },
        low,
        high,
    )


def _normalise_support(
    row: Mapping[str, Any], index: int
) -> tuple[dict[str, int] | None, str | None]:
    raw = row.get("support", _MISSING)
    if raw is _MISSING and ("n" in row or "N" in row):
        raw = {"n": row.get("n"), "N": row.get("N")}
    if raw is _MISSING or raw is None:
        return None, "missing_support"
    if not isinstance(raw, Mapping):
        return None, "invalid_support"
    n = raw.get("n", raw.get("numerator"))
    denominator = raw.get("N", raw.get("denominator"))
    if (
        isinstance(n, bool)
        or isinstance(denominator, bool)
        or not isinstance(n, int)
        or not isinstance(denominator, int)
        or n < 0
        or denominator <= 0
        or n > denominator
    ):
        return None, f"invalid_support_row_{index}"
    return {"n": n, "N": denominator}, None


def _declared_comparability(row: Mapping[str, Any], index: int) -> tuple[bool, str | None]:
    eligible = row.get("eligible", row.get("comparison_eligible", True))
    if not isinstance(eligible, bool):
        raise TieAwareRankingError(f"rows[{index}].eligible must be boolean")
    comparability = row.get("comparability")
    if isinstance(comparability, Mapping):
        status = str(comparability.get("status", "comparable")).strip().lower()
        if status not in {"comparable", "eligible", "incomparable", "excluded"}:
            raise TieAwareRankingError(f"rows[{index}].comparability.status is unsupported")
        if status in {"incomparable", "excluded"}:
            return eligible, str(comparability.get("reason", "incomparable"))
        if comparability.get("eligible") is False:
            return eligible, str(comparability.get("reason", "comparability_gate"))
    if not eligible:
        return False, str(row.get("eligibility_reason", "fairness_excluded"))
    return True, None


def _evidence_reason(row: Mapping[str, Any]) -> str | None:
    if row.get("evidence_valid") is False:
        return "invalid_evidence"
    evidence = row.get("evidence")
    if isinstance(evidence, Mapping):
        if evidence.get("valid") is False:
            return "invalid_evidence"
        status = str(evidence.get("status", "")).strip().lower()
        if status in {"missing", "invalid", "fallback", "degraded"}:
            return f"evidence_{status}"
    return None


def _ensure_unique_keys(items: Sequence[_Item]) -> None:
    keys = [item.key for item in items]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise TieAwareRankingError("duplicate item keys: " + ", ".join(duplicates))


def _order_items(items: Sequence[_Item], display_order: Sequence[str] | None) -> list[_Item]:
    if display_order is not None:
        requested = [str(key) for key in display_order]
        known = {item.key for item in items}
        if len(requested) != len(set(requested)) or set(requested) != known:
            raise TieAwareRankingError("display_order must contain every item key exactly once")
        by_key = {item.key: item for item in items}
        return [by_key[key] for key in requested]
    return sorted(
        items,
        key=lambda item: (
            _display_order_key(item.declared_display_order),
            item.key,
        ),
    )


def _display_order_key(value: int | float | str | None) -> tuple[int, Any]:
    if isinstance(value, bool):
        return 1, str(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return 0, float(value)
    return 1, "" if value is None else str(value)


def _normalise_pairwise_comparisons(
    comparisons: Sequence[Mapping[str, Any]], items: Sequence[_Item]
) -> dict[tuple[str, str], dict[str, Any]]:
    known = {item.key for item in items}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            raise TieAwareRankingError(f"pairwise comparison {index} must be a mapping")
        if comparison.get("approved") is not True:
            raise TieAwareRankingError(f"pairwise comparison {index} is not approved")
        if comparison.get("relation") != "strict_before":
            raise TieAwareRankingError(
                f"pairwise comparison {index} must use relation strict_before"
            )
        left = comparison.get("better", comparison.get("left"))
        right = comparison.get("worse", comparison.get("right"))
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            raise TieAwareRankingError(f"pairwise comparison {index} needs better and worse keys")
        if left not in known or right not in known:
            raise TieAwareRankingError(f"pairwise comparison {index} names an unknown key")
        pair = tuple(sorted((left, right)))
        value = {
            "better": left,
            "worse": right,
            "reason": str(comparison.get("reason", "approved_pairwise_comparison")),
        }
        if pair in result and result[pair] != value:
            raise TieAwareRankingError(f"conflicting pairwise comparison for {left} and {right}")
        result[pair] = value
    return result


def _build_relations(
    items: Sequence[_Item],
    overrides: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            relations.append(_compare_pair(left, right, overrides))
    return relations


def _compare_pair(
    left: _Item,
    right: _Item,
    overrides: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    pair = tuple(sorted((left.key, right.key)))
    if not left.comparison_eligible or not right.comparison_eligible:
        reason = (
            left.comparability_reason
            if not left.comparison_eligible
            else right.comparability_reason
        )
        return _relation(left, right, "incomparable", reason)
    if left.score == right.score:
        if not left.has_uncertainty and not right.has_uncertainty:
            return _relation(left, right, "exact_tie", "exact_score_equality")
        return _relation(left, right, "non_identifiable", "interval_contact_for_equal_scores")
    if pair in overrides:
        override = overrides[pair]
        return _relation(
            left,
            right,
            "strict_before",
            str(override["reason"]),
            better=str(override["better"]),
            worse=str(override["worse"]),
        )
    if not left.has_uncertainty and not right.has_uncertainty:
        return _relation(left, right, "non_identifiable", "no_approved_pairwise_comparison")
    if not left.has_uncertainty or not right.has_uncertainty:
        return _relation(left, right, "non_identifiable", "incomplete_uncertainty")
    if _intervals_overlap_or_contact(left, right):
        return _relation(left, right, "non_identifiable", "interval_overlap_or_contact")
    return _relation(
        left,
        right,
        "non_identifiable",
        "no_approved_pairwise_comparison_for_disjoint_marginal_intervals",
    )


def _intervals_overlap_or_contact(left: _Item, right: _Item) -> bool:
    assert left.uncertainty_low is not None
    assert left.uncertainty_high is not None
    assert right.uncertainty_low is not None
    assert right.uncertainty_high is not None
    return not (
        left.uncertainty_high < right.uncertainty_low
        or right.uncertainty_high < left.uncertainty_low
    )


def _relation(
    left: _Item,
    right: _Item,
    relation: str,
    reason: str,
    better: str | None = None,
    worse: str | None = None,
) -> dict[str, Any]:
    if relation not in RELATION_KINDS:
        raise TieAwareRankingError(f"unsupported relation kind: {relation}")
    result = {
        "left": left.key,
        "right": right.key,
        "relation": relation,
        "reason": reason,
    }
    if relation == "strict_before":
        result["better"] = better
        result["worse"] = worse
    return result


def _build_groups(
    items: Sequence[_Item], relations: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]], dict[str, str]]:
    eligible_keys = [item.key for item in items if item.comparison_eligible]
    parent = {key: key for key in eligible_keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for relation in relations:
        if relation["relation"] != "exact_tie":
            continue
        left, right = str(relation["left"]), str(relation["right"])
        if left in parent and right in parent:
            parent[find(right)] = find(left)
    components: dict[str, list[str]] = defaultdict(list)
    for key in eligible_keys:
        components[find(key)].append(key)
    ordered_components = sorted(
        (tuple(sorted(members)) for members in components.values()),
        key=lambda members: members,
    )
    groups: dict[str, tuple[str, ...]] = {}
    tie_groups: list[dict[str, Any]] = []
    group_ids: dict[str, str] = {}
    item_by_key = {item.key: item for item in items}
    tie_index = 1
    for members in ordered_components:
        if len(members) > 1:
            group_id = f"tie-{tie_index:03d}"
            tie_index += 1
            tie_groups.append(
                {
                    "id": group_id,
                    "members": list(members),
                    "score": item_by_key[members[0]].score_json,
                }
            )
        else:
            group_id = f"item:{members[0]}"
        groups[group_id] = members
        for key in members:
            group_ids[key] = group_id
    return groups, tie_groups, group_ids


def _compute_rank_ranges(
    groups: Mapping[str, tuple[str, ...]],
    relations: Sequence[Mapping[str, Any]],
    group_ids: Mapping[str, str],
) -> dict[str, list[int]]:
    adjacency = {group_id: set() for group_id in groups}
    for relation in relations:
        if relation["relation"] != "strict_before":
            continue
        better = group_ids.get(str(relation["better"]))
        worse = group_ids.get(str(relation["worse"]))
        if better is not None and worse is not None and better != worse:
            adjacency[better].add(worse)
    descendants = {group_id: _walk_descendants(group_id, adjacency) for group_id in groups}
    total = sum(len(members) for members in groups.values())
    ranges: dict[str, list[int]] = {}
    for group_id, members in groups.items():
        ancestors = {other for other, reachable in descendants.items() if group_id in reachable}
        descendant_count = sum(len(groups[other]) for other in descendants[group_id])
        ancestor_count = sum(len(groups[other]) for other in ancestors)
        minimum = 1 + ancestor_count
        maximum_start = total - len(members) - descendant_count + 1
        if minimum > maximum_start:
            raise TieAwareRankingError("strict relation graph is inconsistent")
        rank_range = [minimum, maximum_start + len(members) - 1]
        for key in members:
            ranges[key] = rank_range
    return ranges


def _walk_descendants(start: str, adjacency: Mapping[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    active: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise TieAwareRankingError("strict relation graph contains a cycle")
        active.add(node)
        for child in sorted(adjacency[node]):
            if child not in seen:
                visit(child)
        active.remove(node)
        seen.add(node)

    for child in sorted(adjacency[start]):
        visit(child)
    return seen


def _item_payload(
    item: _Item,
    display_order: int,
    group_ids: Mapping[str, str],
    rank_ranges: Mapping[str, list[int]],
) -> dict[str, Any]:
    tie_group_id = group_ids.get(item.key)
    if tie_group_id is not None and not tie_group_id.startswith("tie-"):
        tie_group_id = None
    return {
        "key": item.key,
        "score": item.score_json,
        "uncertainty": item.uncertainty,
        "support": item.support,
        "eligible": item.eligible,
        "comparison_eligible": item.comparison_eligible,
        "comparability": {
            "status": "comparable" if item.comparison_eligible else "incomparable",
            "reason": item.comparability_reason,
        },
        "evidence": item.evidence,
        "display_order": display_order,
        "tie_group_id": tie_group_id,
        "rank_range": rank_ranges.get(item.key),
    }


def _build_summary(
    items: Sequence[_Item],
    relations: Sequence[Mapping[str, Any]],
    tie_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = dict.fromkeys(sorted(RELATION_KINDS), 0)
    for relation in relations:
        counts[str(relation["relation"])] += 1
    comparison_eligible_count = sum(item.comparison_eligible for item in items)
    return {
        "item_count": len(items),
        "comparison_eligible_item_count": comparison_eligible_count,
        "incomparable_item_count": len(items) - comparison_eligible_count,
        "relation_count": len(relations),
        "relation_counts": counts,
        "exact_tie_group_count": len(tie_groups),
        "scientific_rank_is_not_total": True,
    }


def _policy_payload() -> dict[str, Any]:
    return {
        "exact_tie": {
            "rule": "finite_canonical_score_equality",
            "tolerance": 0.0,
            "rounded_display_values_are_not_ties": True,
        },
        "uncertainty": {
            "interval_overlap_or_contact": "non_identifiable",
            "incomplete_interval": "non_identifiable",
            "strict_order_requires_approved_pairwise_comparison": True,
            "disjoint_marginal_intervals_are_not_sufficient": True,
            "statistical_equivalence_is_not_inferred": True,
        },
        "partial_order": {
            "relation_kinds": sorted(RELATION_KINDS),
            "rank_ranges_use_only_strict_relations": True,
            "incomparable_rows_are_not_ranked": True,
        },
        "display_order": {
            "scientific_meaning": "layout_only",
            "catalog_order_is_not_a_rank": True,
        },
        "gate_compatibility": {
            "fairness_and_comparability_gates_remain_authoritative": True,
            "ineligible_or_invalid_rows_cannot_create_relations": True,
        },
    }


def _direction_label(metric: Mapping[str, Any]) -> str:
    return "higher is better" if metric["higher_is_better"] else "lower is better"


def _format_rank_range(value: Any) -> str:
    if not isinstance(value, Sequence) or len(value) != 2:
        return "—"
    return str(value[0]) if value[0] == value[1] else f"{value[0]}–{value[1]}"


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.12g}"
    return str(value)


def _format_support(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "—"
    return f"{value.get('n')}/{value.get('N')}"


__all__ = [
    "RELATION_KINDS",
    "TIE_AWARE_RANKING_SCHEMA_VERSION",
    "TieAwareRankingError",
    "build_tie_aware_ranking",
    "render_tie_aware_summary",
]
