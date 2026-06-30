"""Tests for the issue #3216 headline CI + rank-stability report harness.

These cover: per-cell CI wiring (reuses canonical seed_variance), rank-stability
detection (a known rank flip vs a stable ordering, via fidelity_rank_stability),
fail-closed exclusion on degraded/missing cells, and the
blocked_until_run/diagnostic classification when the seed budget is insufficient.
"""

from __future__ import annotations

import importlib.util
import json
import sys
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
    job_evidence = overrides.pop("job_evidence", None)
    defaults = {
        "metrics": ("snqi",),
        "rank_metric": "snqi",
        "rank_profile": "snqi_diagnostic",
        "higher_is_better": True,
        "bootstrap_samples": 100,
        "confidence": 0.95,
        "bootstrap_seed": 123,
        "resamples": 50,
        "rank_seed": 123,
    }
    defaults.update(overrides)
    config = mod.ReportConfig(**defaults)
    return mod.build_report(
        rows,
        config,
        campaign=None,
        rows_path="synthetic",
        job_evidence=job_evidence,
    )


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


def test_decision_packet_blocks_small_seed_manifest_table_update() -> None:
    """Small-seed rows are not ready for manuscript table review."""

    rows = [
        _cell_row("merging", "best", {"snqi": [0.90, 0.91, 0.92, 0.93, 0.94]}),
        _cell_row("merging", "worst", {"snqi": [0.10, 0.11, 0.09, 0.12, 0.10]}),
    ]

    packet = _report(rows)["decision_packet"]

    assert packet["manuscript_table_status"] == "blocked"
    assert packet["s30_decision_status"] == "needs_review"
    assert "missing_increased_seed_budget" in packet["manuscript_blockers"]
    assert "minimum_seed_count_below_s20" in packet["s30_reasons"]


def test_decision_packet_keeps_s30_review_open_for_adjacent_overlap() -> None:
    """S20 rows with adjacent CI overlap require claim downgrade or more data review."""

    rows = [
        _cell_row(
            "bottleneck",
            "hybrid_a",
            {"snqi": [0.50, 0.55, 0.45, 0.52, 0.48] * 4},
        ),
        _cell_row(
            "bottleneck",
            "hybrid_b",
            {"snqi": [0.51, 0.46, 0.54, 0.49, 0.53] * 4},
        ),
    ]

    packet = _report(rows)["decision_packet"]

    assert packet["manuscript_table_status"] == "ready_for_table_review_no_claim_promotion"
    assert packet["s30_decision_status"] == "needs_review"
    assert packet["adjacent_overlap_count"] == 1
    assert "adjacent_rank_ci_overlap_requires_claim_downgrade_or_more_data" in packet["s30_reasons"]


def test_decision_packet_can_clear_s30_by_local_preflight_only() -> None:
    """S20 separable/stable fixture records no local S30 blocker."""

    rows = [
        _cell_row(
            "merging",
            "best",
            {"snqi": [0.90, 0.91, 0.92, 0.93, 0.94] * 4},
        ),
        _cell_row(
            "merging",
            "worst",
            {"snqi": [0.10, 0.11, 0.09, 0.12, 0.10] * 4},
        ),
    ]

    packet = _report(rows)["decision_packet"]

    assert packet["manuscript_table_status"] == "ready_for_table_review_no_claim_promotion"
    assert packet["s30_decision_status"] == "not_required_by_local_preflight"
    assert packet["s30_reasons"] == []
    assert "no manuscript or paper claim is promoted" in packet["claim_boundary"]


def test_invalid_rank_metric_blocks_adjacent_claims_and_table_review() -> None:
    """Known SNQI contract invalidity blocks planner-order statements fail-closed."""

    rows = [
        _cell_row(
            "merging",
            "best",
            {"snqi": [0.90, 0.91, 0.92, 0.93, 0.94] * 4},
        ),
        _cell_row(
            "merging",
            "worst",
            {"snqi": [0.10, 0.11, 0.09, 0.12, 0.10] * 4},
        ),
    ]

    report = _report(rows, invalid_rank_metric_reason="SNQI contract status fail")

    claim = report["adjacent_rank_claims"][0]
    assert claim["decision"] == "blocked_invalid_metric"
    assert "SNQI contract status fail" in claim["rationale"]
    packet = report["decision_packet"]
    assert packet["manuscript_table_status"] == "blocked"
    assert packet["s30_decision_status"] == "needs_review"
    assert packet["invalid_metric_claim_count"] == 1
    assert "invalid_rank_metric_contract" in packet["manuscript_blockers"]
    assert "rank_metric_contract_invalid" in packet["s30_reasons"]


def test_dry_run_fixture_stays_diagnostic_and_fail_closed() -> None:
    """Built-in dry-run rows exercise CI/rank paths without paper-claim promotion."""

    rows = mod._dry_run_rows()
    report = _report(rows)

    assert report["classification"] == "diagnostic"
    assert report["inputs"]["excluded_cells"] == 1
    assert any(cell["planner_key"] == "excluded_degraded" for cell in report["cells"])
    degraded = next(cell for cell in report["cells"] if cell["planner_key"] == "excluded_degraded")
    assert degraded["counted"] is False
    assert "degraded" in degraded["exclusion_reason"]
    assert len(report["rank_stability"]) == 2
    assert any(
        claim["decision"] == "not_statistically_distinguishable_budget"
        for claim in report["adjacent_rank_claims"]
    )


def test_dry_run_cli_writes_report_without_rows_file(tmp_path) -> None:
    """CLI dry-run does not require a rows path or campaign artifact."""

    out_dir = tmp_path / "report"

    exit_code = mod.main(
        [
            "--dry-run",
            "--bootstrap-samples",
            "0",
            "--rank-resamples",
            "25",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    result = out_dir / "result.json"
    markdown = out_dir / "report.md"
    assert result.is_file()
    assert markdown.is_file()
    assert "builtin://issue3216-dry-run" in result.read_text(encoding="utf-8")
    assert "**Classification**: `diagnostic`" in markdown.read_text(encoding="utf-8")


def test_dry_run_cli_can_fail_closed_invalid_rank_metric(tmp_path) -> None:
    """CLI records invalid-rank-metric decision state in deterministic dry-run."""

    out_dir = tmp_path / "invalid_metric_report"

    exit_code = mod.main(
        [
            "--dry-run",
            "--bootstrap-samples",
            "0",
            "--rank-resamples",
            "25",
            "--invalid-rank-metric-reason",
            "SNQI normalization contract warning",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    result_text = (out_dir / "result.json").read_text(encoding="utf-8")
    markdown_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert '"decision": "blocked_invalid_metric"' in result_text
    assert '"invalid_metric_claim_count": 3' in result_text
    assert "SNQI normalization contract warning" in result_text
    assert "**Rank metric contract**: `invalid`" in markdown_text


def _job_13198_packet(tmp_path) -> Path:
    """Write a compact deterministic job-13198 evidence packet fixture."""

    packet = {
        "schema_version": "issue1554-slurm-evidence-packet.v1",
        "issue": 1554,
        "evidence": [
            {
                "job": 13198,
                "campaign": "2026-06-issue1554-s20-h500-split-mem180-run",
                "config": "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml",
                "role": "result_matrix",
                "slurm_state": "COMPLETED",
                "exit_code": "0:0",
                "public_commit": "12a188de7246aad3b9088ea76e6a25a20029f976",
                "artifact_summary": {
                    "matrix_rows": 9,
                    "planner_rows": 9,
                    "planner_row_status_counts": {"ok": 9},
                    "warnings": ["SNQI contract status=fail with snqi_contract.enforcement=warn."],
                },
                "limitations": [
                    "SNQI contract warning blocks paper-grade interpretation until analyzed."
                ],
            }
        ],
    }
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path


def test_job_13198_packet_auto_blocks_snqi_rank_claims(tmp_path) -> None:
    """Bound job evidence turns SNQI warning into fail-closed rank decisions."""

    packet_path = _job_13198_packet(tmp_path)
    rows = [
        _cell_row("merging", "best", {"snqi": [0.90, 0.91, 0.92, 0.93, 0.94] * 4}),
        _cell_row("merging", "worst", {"snqi": [0.10, 0.11, 0.09, 0.12, 0.10] * 4}),
    ]

    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    out_dir = tmp_path / "snqi_report"
    exit_code = mod.main(
        [
            "--rows",
            str(rows_path),
            "--evidence-packet",
            str(packet_path),
            "--job-id",
            "13198",
            "--bootstrap-samples",
            "0",
            "--rank-resamples",
            "25",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    packet = result["decision_packet"]
    assert result["job_evidence"]["job"] == 13198
    assert result["inputs"]["rank_metric"] == "snqi"
    assert packet["snqi_contract_warning"] is True
    assert packet["invalid_metric_claim_count"] == 1
    assert "rank_metric_contract_invalid" in packet["s30_reasons"]
    assert "**SNQI evidence status**: `diagnostic_only_contract_warning`" in (
        out_dir / "report.md"
    ).read_text(encoding="utf-8")


def test_constraints_first_profile_keeps_snqi_warning_diagnostic_only(tmp_path) -> None:
    """Constraints-first profile binds job warning without using SNQI as rank metric."""

    job_evidence = mod.load_job_evidence_packet(_job_13198_packet(tmp_path), job_id=13198)
    rows = [
        _cell_row(
            "merging",
            "best",
            {
                "success": [0.90, 0.91, 0.92, 0.93, 0.94] * 4,
                "collisions": [0.02, 0.01, 0.02, 0.01, 0.02] * 4,
                "near_misses": [0.04, 0.03, 0.04, 0.03, 0.04] * 4,
                "snqi": [0.90, 0.91, 0.92, 0.93, 0.94] * 4,
            },
        ),
        _cell_row(
            "merging",
            "worst",
            {
                "success": [0.40, 0.41, 0.39, 0.42, 0.40] * 4,
                "collisions": [0.30, 0.32, 0.31, 0.33, 0.30] * 4,
                "near_misses": [0.20, 0.22, 0.21, 0.23, 0.20] * 4,
                "snqi": [0.10, 0.11, 0.09, 0.12, 0.10] * 4,
            },
        ),
    ]

    report = _report(
        rows,
        metrics=("success", "collisions", "near_misses", "snqi"),
        rank_metric="success",
        rank_profile="constraints_first",
        job_evidence=job_evidence,
    )

    packet = report["decision_packet"]
    assert report["inputs"]["rank_profile"] == "constraints_first"
    assert packet["constraints_first_metric_gaps"] == []
    assert packet["snqi_contract_warning"] is True
    assert "snqi_contract_warning_diagnostic_only" in packet["s30_reasons"]
    assert "rank_metric_contract_invalid" not in packet["s30_reasons"]


def test_constraints_first_profile_blocks_missing_required_metrics(tmp_path) -> None:
    """Constraints-first profile cannot rank headline rows lacking safety metrics."""

    report = _report(
        [
            _cell_row("merging", "best", {"success": [0.9, 0.91, 0.92] * 7}),
            _cell_row("merging", "worst", {"success": [0.1, 0.11, 0.12] * 7}),
        ],
        metrics=("success",),
        rank_metric="success",
        rank_profile="constraints_first",
    )

    packet = report["decision_packet"]
    assert packet["constraints_first_metric_gaps"] == ["collisions", "near_misses"]
    assert "constraints_first_metrics_missing" in packet["manuscript_blockers"]
    assert "constraints_first_metric_gap" in packet["s30_reasons"]
