"""Tests for the issue #3216 headline CI + rank-stability report harness.

These cover: per-cell CI wiring (reuses canonical seed_variance), rank-stability
detection (a known rank flip vs a stable ordering, via fidelity_rank_stability),
fail-closed exclusion on degraded/missing cells, and the
blocked_until_run/diagnostic classification when the seed budget is insufficient.
"""

from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "build_headline_ci_rank_stability_report_issue_3216.py"
)
_spec = importlib.util.spec_from_file_location("headline_ci_3216", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass introspection can resolve the module.
sys.modules["headline_ci_3216"] = mod
_spec.loader.exec_module(mod)


def _cell_row(
    scenario: str,
    planner: str,
    metric_per_seed: dict[str, list[float]],
    *,
    row_status: str = "successful_evidence",
    execution_mode: str = "nominal",
) -> dict[str, Any]:
    """Build a synthetic headline row with per-seed metrics.

    Returns:
        A row in the ``per_seed`` list shape consumed by the harness.
    """
    seeds = max((len(v) for v in metric_per_seed.values()), default=0)
    per_seed = []
    for i in range(seeds):
        metrics = {
            metric: values[i] for metric, values in metric_per_seed.items() if i < len(values)
        }
        per_seed.append({"seed": 100 + i, "metrics": metrics})
    return {
        "scenario_family": scenario,
        "planner_key": planner,
        "row_status": row_status,
        "execution_mode": execution_mode,
        "per_seed": per_seed,
    }


def _report(rows: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    """Build a report with a small-seed-friendly ReportConfig.

    Returns:
        The report payload mapping.
    """
    defaults = {
        "metrics": ("snqi",),
        "rank_metric": "snqi",
        "higher_is_better": True,
        "bootstrap_samples": 100,
        "confidence": 0.95,
        "bootstrap_seed": 123,
        "resamples": 50,
        "rank_seed": 123,
    }
    defaults.update(overrides)
    config = mod.ReportConfig(**defaults)
    return mod.build_report(rows, config, campaign=None, rows_path="synthetic")


def test_per_cell_ci_reuses_canonical_seed_variance():
    """Per-cell CI is populated and brackets the mean using canonical stats."""
    rows = [
        _cell_row("merging", "orca", {"snqi": [0.5, 0.6, 0.7, 0.55, 0.65]}),
    ]
    cells = mod.build_cell_results(
        rows,
        metrics=["snqi"],
        bootstrap_samples=500,
        confidence=0.95,
        bootstrap_seed=123,
    )
    assert len(cells) == 1
    cell = cells[0]
    assert cell.counted is True
    stats = cell.metrics["snqi"]
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
    assert stats["count"] == 5.0


def test_stable_ordering_has_high_tau_and_no_flips():
    """A well-separated ordering is rank-stable under resampling."""
    rows = [
        _cell_row("merging", "best", {"snqi": [0.90, 0.91, 0.92, 0.93, 0.94]}),
        _cell_row("merging", "mid", {"snqi": [0.50, 0.51, 0.49, 0.50, 0.52]}),
        _cell_row("merging", "worst", {"snqi": [0.10, 0.11, 0.09, 0.12, 0.10]}),
    ]
    cells = mod.build_cell_results(
        rows, metrics=["snqi"], bootstrap_samples=0, confidence=0.95, bootstrap_seed=1
    )
    stability = mod.build_rank_stability(
        rows, cells, rank_metric="snqi", higher_is_better=True, resamples=200, rng_seed=7
    )
    assert len(stability) == 1
    entry = stability[0]
    assert entry.rank_identifiable is True
    assert entry.point_ranking == ["best", "mid", "worst"]
    assert entry.kendall_tau_mean is not None and entry.kendall_tau_mean > 0.9
    assert entry.top1_stable is True
    assert entry.rank_flip_rate is not None and entry.rank_flip_rate < 0.05


def test_overlapping_ordering_shows_rank_instability():
    """Two planners with heavily overlapping per-seed values flip under resamples."""
    rows = [
        _cell_row("bottleneck", "a", {"snqi": [0.50, 0.55, 0.45, 0.52, 0.48]}),
        _cell_row("bottleneck", "b", {"snqi": [0.51, 0.46, 0.54, 0.49, 0.53]}),
    ]
    cells = mod.build_cell_results(
        rows, metrics=["snqi"], bootstrap_samples=0, confidence=0.95, bootstrap_seed=1
    )
    stability = mod.build_rank_stability(
        rows, cells, rank_metric="snqi", higher_is_better=True, resamples=400, rng_seed=11
    )
    entry = stability[0]
    assert entry.rank_identifiable is True
    # Heavy overlap => the resampled ranking is NOT perfectly stable.
    assert entry.rank_flip_rate is not None and entry.rank_flip_rate > 0.0
    assert entry.top1_stable is False


def test_fail_closed_on_degraded_and_missing_cells():
    """Degraded execution_mode and non-success row_status cells are excluded."""
    rows = [
        _cell_row("merging", "good", {"snqi": [0.8, 0.7, 0.9, 0.75, 0.85]}),
        _cell_row(
            "merging",
            "degraded_planner",
            {"snqi": [0.4, 0.4, 0.4, 0.4, 0.4]},
            execution_mode="degraded",
        ),
        _cell_row(
            "merging",
            "fallback_planner",
            {"snqi": [0.3, 0.3, 0.3, 0.3, 0.3]},
            row_status="fallback",
        ),
    ]
    cells = mod.build_cell_results(
        rows, metrics=["snqi"], bootstrap_samples=0, confidence=0.95, bootstrap_seed=1
    )
    by_planner = {c.planner_key: c for c in cells}
    assert by_planner["good"].counted is True
    assert by_planner["degraded_planner"].counted is False
    assert "degraded" in by_planner["degraded_planner"].exclusion_reason
    assert by_planner["fallback_planner"].counted is False
    # Rank stability sees only one counted cell -> not identifiable.
    stability = mod.build_rank_stability(
        rows, cells, rank_metric="snqi", higher_is_better=True, resamples=10, rng_seed=1
    )
    assert stability[0].rank_identifiable is False


def test_classification_diagnostic_on_small_seed_budget():
    """S5-style per-cell budget classifies diagnostic, never paper_grade."""
    rows = [
        _cell_row("merging", "best", {"snqi": [0.9, 0.8, 0.85, 0.92, 0.88]}),
        _cell_row("merging", "worst", {"snqi": [0.2, 0.25, 0.18, 0.22, 0.19]}),
    ]
    report = _report(rows)
    assert report["classification"] == "diagnostic"
    assert report["classification"] != "paper_grade"
    assert "S20/S30" in report["classification_rationale"]


def test_classification_blocked_until_run_when_no_counted_cells():
    """All-degraded input classifies blocked_until_run."""
    rows = [
        _cell_row("merging", "a", {"snqi": [0.5, 0.5, 0.5]}, execution_mode="degraded"),
        _cell_row("merging", "b", {"snqi": [0.5, 0.5, 0.5]}, row_status="excluded"),
    ]
    report = _report(rows)
    assert report["classification"] == "blocked_until_run"


def test_paper_grade_seed_budget_still_blocked_until_slurm_run():
    """At S20+ per cell the harness still blocks paper-grade on the SLURM run."""
    big = [round(0.9 - 0.001 * i, 4) for i in range(20)]
    small = [round(0.2 + 0.001 * i, 4) for i in range(20)]
    rows = [
        _cell_row("merging", "best", {"snqi": big}),
        _cell_row("merging", "worst", {"snqi": small}),
    ]
    report = _report(rows)
    # Never self-certifies paper_grade; promotion requires the SLURM run.
    assert report["classification"] == "blocked_until_run"
    assert "S20/S30" in report["classification_rationale"]


def test_report_records_canonical_owners_and_git_head():
    """The report enumerates reused canonical owners for provenance."""
    rows = [_cell_row("merging", "a", {"snqi": [0.5, 0.6, 0.55]})]
    report = _report(rows)
    owners = report["canonical_owners_reused"]
    assert any("seed_variance" in o for o in owners)
    assert any("fidelity_rank_stability" in o for o in owners)
    assert any("canonical_table_export" in o for o in owners)


def test_markdown_renders_without_error():
    """Markdown rendering produces a non-empty summary with the classification."""
    rows = [
        _cell_row("merging", "best", {"snqi": [0.9, 0.8, 0.85]}),
        _cell_row("merging", "worst", {"snqi": [0.2, 0.25, 0.18]}),
    ]
    report = _report(rows)
    md = mod.render_markdown(report)
    assert "Headline 7x7 CI + Rank-Stability Report" in md
    assert report["classification"] in md


@pytest.mark.parametrize("shape", ["per_seed", "per_seed_metrics"])
def test_both_input_shapes_supported(shape: str):
    """Both per_seed-list and per_seed_metrics-mapping shapes are parsed."""
    if shape == "per_seed":
        row = _cell_row("merging", "a", {"snqi": [0.5, 0.6, 0.7]})
    else:
        row = {
            "scenario_family": "merging",
            "planner_key": "a",
            "row_status": "successful_evidence",
            "execution_mode": "nominal",
            "per_seed_metrics": {"snqi": [0.5, 0.6, 0.7]},
        }
    cells = mod.build_cell_results(
        [row], metrics=["snqi"], bootstrap_samples=0, confidence=0.95, bootstrap_seed=1
    )
    assert cells[0].metrics["snqi"]["count"] == 3.0


def test_adjacent_rank_claims_downgrade_overlapping_ci() -> None:
    """Adjacent planners with overlapping CIs are not strict headline claims."""

    rows = [
        _cell_row("bottleneck", "a", {"snqi": [0.50, 0.55, 0.45, 0.52, 0.48]}),
        _cell_row("bottleneck", "b", {"snqi": [0.51, 0.46, 0.54, 0.49, 0.53]}),
    ]

    report = _report(rows)

    assert len(report["adjacent_rank_claims"]) == 1
    claim = report["adjacent_rank_claims"][0]
    assert claim["decision"] == "not_statistically_distinguishable_budget"
    assert "overlap" in claim["rationale"]


def test_adjacent_rank_claims_mark_ci_separable() -> None:
    """Well-separated adjacent planner CIs are labeled separable."""

    rows = [
        _cell_row("merging", "best", {"snqi": [0.90, 0.91, 0.92, 0.93, 0.94]}),
        _cell_row("merging", "worst", {"snqi": [0.10, 0.11, 0.09, 0.12, 0.10]}),
    ]

    report = _report(rows)

    assert len(report["adjacent_rank_claims"]) == 1
    claim = report["adjacent_rank_claims"][0]
    assert claim["higher_rank_planner"] == "best"
    assert claim["lower_rank_planner"] == "worst"
    assert claim["decision"] == "ci_separable"


def test_main_supports_dry_run_and_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "dry_run_report"

    code = mod.main(
        [
            "--dry-run",
            "--output-dir",
            str(out_dir),
            "--bootstrap-samples",
            "0",
            "--rank-resamples",
            "1",
        ]
    )

    assert code == 0
    assert (out_dir / "result.json").exists()
    assert (out_dir / "report.md").exists()


def test_main_fail_on_decision_blocker_for_local_blocker_rows(tmp_path: Path) -> None:
    rows = [
        {
            "scenario_family": "merging",
            "planner_key": "a",
            "row_status": "fallback",
            "execution_mode": "degraded",
            "per_seed": [
                {"seed": 111, "metrics": {"snqi": 0.50}},
            ],
        },
        {
            "scenario_family": "merging",
            "planner_key": "b",
            "row_status": "fallback",
            "execution_mode": "degraded",
            "per_seed": [
                {"seed": 111, "metrics": {"snqi": 0.49}},
            ],
        },
    ]
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    code = mod.main(
        [
            "--rows",
            str(rows_path),
            "--output-dir",
            str(tmp_path / "blocker_report"),
            "--bootstrap-samples",
            "0",
            "--rank-resamples",
            "1",
            "--fail-on-decision-blocker",
        ]
    )

    assert code == 4
