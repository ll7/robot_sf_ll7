"""Tests for the additive tie-aware ranking contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.ranking import compute_ranking, format_csv, format_markdown
from robot_sf.benchmark.tie_aware_ranking import (
    build_tie_aware_ranking,
    render_tie_aware_summary,
)


def _row(key: str, score: float, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"key": key, "score": score, "support": {"n": 3, "N": 3}}
    row.update(overrides)
    return row


def _relation(payload: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    return next(
        relation
        for relation in payload["relations"]
        if relation["left"] == left and relation["right"] == right
    )


def test_exact_ties_and_rank_ranges_do_not_use_catalog_order() -> None:
    """Equal canonical scores form a tie group, independent of display order."""
    payload = build_tie_aware_ranking(
        [_row("zeta", 1), _row("alpha", 1), _row("omega", 2)],
        metric={"id": "loss", "higher_is_better": False},
        display_order=["alpha", "omega", "zeta"],
    )

    tie = _relation(payload, "alpha", "zeta")
    assert tie["relation"] == "exact_tie"
    assert payload["tie_groups"] == [{"id": "tie-001", "members": ["alpha", "zeta"], "score": 1.0}]
    assert [item["key"] for item in payload["items"]] == ["alpha", "omega", "zeta"]
    assert payload["items"][0]["rank_range"] == [1, 2]
    assert payload["items"][2]["rank_range"] == [1, 2]
    assert payload["items"][1]["rank_range"] == [3, 3]


def test_intervals_separate_strict_order_from_non_identifiability() -> None:
    """Disjoint intervals order items; overlap or contact stays non-identifiable."""
    payload = build_tie_aware_ranking(
        [
            _row("high", 0.9, uncertainty={"low": 0.8, "high": 0.95, "source": "seed"}),
            _row("middle", 0.6, uncertainty={"low": 0.55, "high": 0.65, "source": "seed"}),
            _row("low", 0.5, uncertainty={"low": 0.45, "high": 0.55, "source": "seed"}),
        ],
        metric={"id": "score", "higher_is_better": True},
    )

    high_middle = _relation(payload, "high", "middle")
    middle_low = _relation(payload, "low", "middle")
    high_low = _relation(payload, "high", "low")
    assert high_middle["relation"] == "strict_before"
    assert high_middle["better"] == "high"
    assert middle_low["relation"] == "non_identifiable"
    assert high_low["relation"] == "strict_before"
    assert high_low["reason"] == "disjoint_uncertainty_intervals"


def test_missing_support_and_excluded_rows_are_incomparable() -> None:
    """The exporter cannot create relations from failed comparison gates."""
    payload = build_tie_aware_ranking(
        [
            _row("valid", 1),
            {"key": "missing", "score": 2},
            _row("excluded", 3, eligible=False, eligibility_reason="fairness_excluded"),
        ],
        metric="loss",
        higher_is_better=False,
    )

    for relation in payload["relations"]:
        if "missing" in {relation["left"], relation["right"]}:
            assert relation["relation"] == "incomparable"
        if "excluded" in {relation["left"], relation["right"]}:
            assert relation["relation"] == "incomparable"
    missing_item = next(item for item in payload["items"] if item["key"] == "missing")
    assert missing_item["comparability"] == {
        "status": "incomparable",
        "reason": "missing_support",
    }
    assert missing_item["rank_range"] is None


def test_approved_pairwise_relation_can_resolve_overlapping_intervals() -> None:
    """Only an explicitly approved paired comparison can override interval ambiguity."""
    payload = build_tie_aware_ranking(
        [
            _row("a", 0.6, uncertainty={"low": 0.4, "high": 0.8, "source": "marginal"}),
            _row("b", 0.5, uncertainty={"low": 0.3, "high": 0.7, "source": "marginal"}),
        ],
        metric={"id": "score", "higher_is_better": True},
        pairwise_comparisons=[
            {
                "better": "a",
                "worse": "b",
                "relation": "strict_before",
                "approved": True,
                "reason": "paired_difference_interval_excludes_null",
            }
        ],
    )
    relation = _relation(payload, "a", "b")
    assert relation["relation"] == "strict_before"
    assert relation["reason"] == "paired_difference_interval_excludes_null"


def test_output_is_hash_stable_and_schema_valid() -> None:
    """Repeated export is byte-stable and validates against the public schema."""
    rows = [_row("b", 2), _row("a", 1)]
    first = build_tie_aware_ranking(rows, metric={"id": "loss", "higher_is_better": False})
    second = build_tie_aware_ranking(rows, metric={"id": "loss", "higher_is_better": False})
    first_bytes = json.dumps(first, indent=2, sort_keys=True).encode()
    second_bytes = json.dumps(second, indent=2, sort_keys=True).encode()
    assert hashlib.sha256(first_bytes).digest() == hashlib.sha256(second_bytes).digest()

    from pathlib import Path

    schema_path = Path(__file__).parents[2] / "robot_sf/benchmark/schemas/tie_aware_ranking.v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(first)) == []


def test_summary_preserves_partial_order_language() -> None:
    """The summary exposes ranges and does not present catalog order as rank."""
    payload = build_tie_aware_ranking(
        [_row("a", 1), _row("b", 1)],
        metric={"id": "loss", "higher_is_better": False},
    )
    summary = render_tie_aware_summary(payload)
    assert "rank range" in summary
    assert "display order is not a rank" in summary
    assert "Exact tie: a, b" in summary


def test_legacy_ranking_surfaces_remain_unchanged() -> None:
    """The additive exporter does not change the legacy ranking contract."""
    records = [
        {"scenario_params": {"algo": "a"}, "metrics": {"loss": 1}},
        {"scenario_params": {"algo": "b"}, "metrics": {"loss": 2}},
    ]
    rows = compute_ranking(records, metric="loss", ascending=True)
    assert format_markdown(rows, "loss") == (
        "| Rank | Group | mean(loss) | count |\n|---:|---|---:|---:|\n"
        "| 1 | a | 1 | 1 |\n| 2 | b | 2 | 1 |\n"
    )
    assert format_csv(rows, "loss") == "rank,group,mean_loss,count\n1,a,1,1\n2,b,2,1\n"
