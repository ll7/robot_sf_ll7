"""Tests for the issue #3216 headline CI/rank-stability preflight."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_module() -> ModuleType:
    """Load the script module by path because scripts/benchmark is not a package."""
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py"
    )
    spec = importlib.util.spec_from_file_location("issue3216_headline_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


issue3216 = _load_module()
ReportConfig = issue3216.ReportConfig
build_report = issue3216.build_report
main = issue3216.main


def _row(
    scenario: str,
    planner: str,
    values: list[float],
    *,
    row_status: str = "successful_evidence",
    execution_mode: str = "nominal",
) -> dict:
    """Create one compact headline cell fixture with all constraints-first metrics."""
    return {
        "scenario_family": scenario,
        "planner_key": planner,
        "row_status": row_status,
        "execution_mode": execution_mode,
        "per_seed": [
            {
                "seed": 111 + index,
                "metrics": {
                    "success": value,
                    "collisions": 1.0 - value,
                    "near_misses": 0.0,
                    "snqi": value,
                },
            }
            for index, value in enumerate(values)
        ],
    }


def _stable_s20_rows() -> list[dict]:
    """Return deterministic S20 rows with separable adjacent ranks."""
    return [
        _row("crossing", "hybrid", [0.90] * 20),
        _row("crossing", "ppo", [0.70] * 20),
        _row("doorway", "hybrid", [0.88] * 20),
        _row("doorway", "ppo", [0.66] * 20),
    ]


def test_s20_preflight_stays_no_claim_but_allows_table_review() -> None:
    """S20-quality local statistics do not self-promote paper claims."""
    report = build_report(
        _stable_s20_rows(),
        ReportConfig(bootstrap_samples=32, resamples=16),
        campaign="issue3216_s20_fixture",
    )

    assert report["classification"] == "blocked_until_run"
    assert "claim-card review" in report["classification_rationale"]
    assert report["decision_packet"]["manuscript_table_status"] == (
        "ready_for_table_review_no_claim_promotion"
    )
    assert report["decision_packet"]["s30_decision_status"] == "not_required_by_local_preflight"
    assert {claim["decision"] for claim in report["adjacent_rank_claims"]} == {"ci_separable"}
    assert report["decision_packet"]["constraints_first_metric_gaps"] == []


def test_adjacent_ci_overlap_downgrades_strict_rank_claim() -> None:
    """Overlapping adjacent confidence intervals require a budget downgrade label."""
    rows = [
        _row("crossing", "hybrid", [0.76, 0.74] * 10),
        _row("crossing", "ppo", [0.75, 0.74] * 10),
    ]

    report = build_report(rows, ReportConfig(bootstrap_samples=64, resamples=24))

    assert report["adjacent_rank_claims"][0]["decision"] == (
        "not_statistically_distinguishable_budget"
    )
    assert (
        "adjacent_rank_ci_overlap_requires_claim_downgrade_or_more_data"
        in report["decision_packet"]["s30_reasons"]
    )


def test_invalid_rank_metric_blocks_metric_claims() -> None:
    """SNQI or other rank-metric contract warnings fail closed for rank statements."""
    report = build_report(
        _stable_s20_rows(),
        ReportConfig(
            bootstrap_samples=32,
            resamples=16,
            invalid_rank_metric_reason="SNQI contract warning in job 13198",
        ),
    )

    assert {claim["decision"] for claim in report["adjacent_rank_claims"]} == {
        "blocked_invalid_metric"
    }
    assert "invalid_rank_metric_contract" in report["decision_packet"]["manuscript_blockers"]
    assert "rank_metric_contract_invalid" in report["decision_packet"]["s30_reasons"]


def test_fail_closed_row_status_exclusion_blocks_packet() -> None:
    """Fallback/degraded rows are disclosed and block manuscript-table readiness."""
    rows = _stable_s20_rows()
    rows.append(_row("crossing", "fallback_adapter", [0.95] * 20, row_status="fallback"))

    report = build_report(rows, ReportConfig(bootstrap_samples=32, resamples=16))

    assert report["inputs"]["excluded_cells"] == 1
    assert report["excluded_cell_reasons"][0]["exclusion_reason"] == "row_status=fallback"
    assert "non_promotable_cells_present" in report["decision_packet"]["manuscript_blockers"]
    assert "resolve_or_disclose_excluded_cells" in report["decision_packet"]["s30_reasons"]


def test_cli_dry_run_writes_decision_packet_and_can_fail_on_blocker(tmp_path: Path) -> None:
    """Dry-run preflight is local-only and can fail closed for automation gates."""
    output_dir = tmp_path / "issue3216_dry_run"

    exit_code = main(
        [
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "32",
            "--rank-resamples",
            "16",
            "--fail-on-decision-blocker",
        ]
    )

    assert exit_code == 4
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["inputs"]["rows_path"] == "builtin://issue3216-dry-run"
    assert payload["decision_packet"]["claim_boundary"] == (
        "Decision packet is local preflight only; no manuscript or paper claim is promoted."
    )
    assert (output_dir / "report.md").is_file()
