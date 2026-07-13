"""Tests for quantitative SNQI Pareto SVG rendering."""

from __future__ import annotations

import html
import re
from typing import Any

from robot_sf.benchmark.snqi_scalarization_sensitivity import format_pareto_svg


def _make_report() -> dict[str, Any]:
    """Build a report with the same numeric range as the pinned evidence."""

    rows = [
        ("ppo", 0.206791, -0.048550, True),
        ("hybrid_rule_v3_fast_progress_static_escape", 0.287886, -0.079170, True),
        ("scenario_adaptive_hybrid_orca_v1", 0.274838, -0.080031, False),
        ("orca", 0.093697, -0.102191, False),
        ("socnav_sampling", -0.600579, -0.143172, False),
        ("prediction_planner", -0.348098, -0.180360, False),
        ("social_force", -0.582915, -0.207414, False),
        ("sacadrl", -0.820221, -0.238074, False),
        ("goal", -0.797935, -0.241916, False),
    ]
    return {
        "planner_rows": [
            {
                "planner": planner,
                "constraints_first_score": score,
                "snqi_mean": snqi,
                "pareto_front": pareto_front,
            }
            for planner, score, snqi, pareto_front in rows
        ]
    }


def _legend_entries(svg: str) -> list[tuple[int, str]]:
    matches = re.findall(r'class="legend-entry"[^>]*>(\d+)\. ([^<]+)</text>', svg)
    return [(int(index), html.unescape(name)) for index, name in matches]


def test_svg_renderer_draws_numeric_ticks_and_gridlines() -> None:
    """The renderer draws round numeric ticks and one gridline per tick."""
    svg = format_pareto_svg(_make_report())

    assert svg.startswith("<svg")
    assert svg.strip().endswith("</svg>")
    assert "Constraints-first score (higher is better)" in svg
    assert "SNQI mean (higher is better)" in svg
    for tick in ("-0.75", "-0.50", "-0.25", "0.00", "0.25"):
        assert f">{tick}</text>" in svg
    for tick in ("-0.25", "-0.20", "-0.15", "-0.10", "-0.05"):
        assert f">{tick}</text>" in svg
    assert svg.count('class="gridline x-gridline"') == 5
    assert svg.count('class="gridline y-gridline"') == 5
    assert svg.count('stroke="#E5E5E5"') == 10


def test_svg_renderer_numbers_every_point_and_lists_every_planner() -> None:
    """Every plotted point has an index and a matching full-name legend entry."""
    report = _make_report()
    svg = format_pareto_svg(report)
    entries = _legend_entries(svg)

    assert len(entries) == len(report["planner_rows"])
    assert [index for index, _ in entries] == list(range(1, len(entries) + 1))
    assert {name for _, name in entries} == {row["planner"] for row in report["planner_rows"]}
    assert svg.count('class="pareto-point ') == len(entries)
    assert svg.count('class="point-index ') == len(entries)


def test_svg_renderer_uses_deterministic_pareto_first_ranking() -> None:
    """Legend and point indices use Pareto-first then descending-score order."""
    svg = format_pareto_svg(_make_report())

    assert _legend_entries(svg) == [
        (1, "hybrid_rule_v3_fast_progress_static_escape"),
        (2, "ppo"),
        (3, "scenario_adaptive_hybrid_orca_v1"),
        (4, "orca"),
        (5, "prediction_planner"),
        (6, "social_force"),
        (7, "socnav_sampling"),
        (8, "goal"),
        (9, "sacadrl"),
    ]


def test_svg_renderer_highlights_only_pareto_front_points() -> None:
    """Only declared Pareto-front planners receive blue front markers."""
    svg = format_pareto_svg(_make_report())
    front_names = set(re.findall(r'class="pareto-point front" data-planner="([^"]+)"', svg))

    assert front_names == {"ppo", "hybrid_rule_v3_fast_progress_static_escape"}
    assert '<polyline points="' in svg
    assert 'stroke="#1f77b4"' in svg
