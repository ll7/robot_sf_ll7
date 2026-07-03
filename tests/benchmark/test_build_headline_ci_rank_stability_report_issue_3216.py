"""Tests for the issue #3216 headline CI/rank-stability preflight."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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
_planner_keys_from_benchmark_config = issue3216._planner_keys_from_benchmark_config


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


def test_expected_headline_grid_missing_cell_blocks_packet() -> None:
    """A partial headline grid cannot pass even when present cells look stable."""
    rows = [
        _row("crossing", "hybrid", [0.90] * 20),
        _row("crossing", "ppo", [0.70] * 20),
        _row("doorway", "hybrid", [0.88] * 20),
        _row("extra_family", "hybrid", [0.89] * 20),
    ]

    report = build_report(
        rows,
        ReportConfig(
            bootstrap_samples=32,
            resamples=16,
            expected_scenarios=("crossing", "doorway"),
            expected_planners=("hybrid", "ppo"),
        ),
    )

    packet = report["decision_packet"]
    assert packet["manuscript_table_status"] == "blocked"
    assert packet["s30_decision_status"] == "blocked"
    assert "headline_grid_incomplete" in packet["manuscript_blockers"]
    assert "missing_expected_headline_cells" in packet["s30_reasons"]
    assert packet["grid_completeness"]["missing_cells"] == [
        {"scenario_id": "doorway", "planner_key": "ppo"}
    ]
    assert packet["grid_completeness"]["expected_cell_count"] == 4
    assert packet["grid_completeness"]["observed_cell_count"] == 4
    assert packet["grid_completeness"]["observed_expected_cell_count"] == 3
    assert packet["grid_completeness"]["unexpected_cells"] == [
        {"scenario_id": "extra_family", "planner_key": "hybrid"}
    ]


def test_expected_headline_planner_rows_block_missing_planner() -> None:
    """Expected planner-only coverage fails closed for aggregate headline rows."""
    rows = [
        _row("headline", "hybrid", [0.90] * 20),
        _row("headline", "orca", [0.70] * 20),
    ]

    report = build_report(
        rows,
        ReportConfig(
            bootstrap_samples=32,
            resamples=16,
            expected_planners=("hybrid", "orca", "ppo"),
        ),
    )

    packet = report["decision_packet"]
    assert packet["manuscript_table_status"] == "blocked"
    assert packet["s30_decision_status"] == "blocked"
    assert "headline_grid_incomplete" in packet["manuscript_blockers"]
    assert packet["grid_completeness"]["expected_scenarios"] == []
    assert packet["grid_completeness"]["expected_cell_count"] == 3
    assert packet["grid_completeness"]["observed_expected_cell_count"] == 2
    assert packet["grid_completeness"]["missing_cells"] == [
        {"scenario_id": "*", "planner_key": "ppo"}
    ]


def test_expected_headline_scenario_rows_block_missing_scenario() -> None:
    """Expected scenario-only coverage fails closed without planner labels."""
    rows = [
        _row("crossing", "headline", [0.90] * 20),
        _row("doorway", "headline", [0.70] * 20),
    ]

    report = build_report(
        rows,
        ReportConfig(
            bootstrap_samples=32,
            resamples=16,
            expected_scenarios=("crossing", "doorway", "bottleneck"),
        ),
    )

    packet = report["decision_packet"]
    assert packet["manuscript_table_status"] == "blocked"
    assert packet["s30_decision_status"] == "blocked"
    assert packet["grid_completeness"]["expected_planners"] == []
    assert packet["grid_completeness"]["expected_cell_count"] == 3
    assert packet["grid_completeness"]["observed_expected_cell_count"] == 2
    assert packet["grid_completeness"]["missing_cells"] == [
        {"scenario_id": "bottleneck", "planner_key": "*"}
    ]


def test_underpowered_missing_grid_remains_blocked_not_reviewable() -> None:
    """Hard coverage blockers outrank softer seed-budget review downgrades."""
    rows = [
        _row("crossing", "hybrid", [0.90] * 5),
        _row("crossing", "ppo", [0.70] * 5),
        _row("doorway", "hybrid", [0.88] * 5),
    ]

    report = build_report(
        rows,
        ReportConfig(
            bootstrap_samples=32,
            resamples=16,
            expected_scenarios=("crossing", "doorway"),
            expected_planners=("hybrid", "ppo"),
        ),
    )

    packet = report["decision_packet"]
    assert packet["manuscript_table_status"] == "blocked"
    assert packet["s30_decision_status"] == "blocked"
    assert "headline_grid_incomplete" in packet["manuscript_blockers"]
    assert "missing_increased_seed_budget" in packet["manuscript_blockers"]
    assert "missing_expected_headline_cells" in packet["s30_reasons"]
    assert "minimum_seed_count_below_s20" in packet["s30_reasons"]


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


def test_underpowered_adjacent_rank_statement_is_diagnostic_only() -> None:
    """Sub-S20 local rows never emit strict adjacent-rank separability claims."""
    rows = [
        _row("crossing", "hybrid", [0.90] * 10),
        _row("crossing", "ppo", [0.50] * 10),
    ]

    report = build_report(rows, ReportConfig(bootstrap_samples=32, resamples=16))

    assert report["adjacent_rank_claims"][0]["decision"] == "diagnostic_only"
    assert report["decision_packet"]["diagnostic_only_claim_count"] == 1
    assert (
        "seed_budget_below_paper_grade_diagnostic_only"
        in (report["decision_packet"]["s30_reasons"])
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


def test_cli_expected_planners_from_config_blocks_missing_planner(tmp_path: Path) -> None:
    """Config-derived headline planners become fail-closed expected rows."""
    rows_path = tmp_path / "rows.json"
    config_path = tmp_path / "benchmark.yaml"
    output_dir = tmp_path / "configured_planners"
    rows_path.write_text(
        json.dumps(
            [
                _row("headline", "hybrid", [0.90] * 20),
                _row("headline", "ppo", [0.70] * 20),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        """
name: issue3216-test
planners:
  - key: hybrid
  - key: ppo
  - key: orca
""".lstrip(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--rows",
            str(rows_path),
            "--expected-planners-from-config",
            str(config_path),
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
    grid = payload["decision_packet"]["grid_completeness"]
    assert grid["expected_planners"] == ["hybrid", "ppo", "orca"]
    assert grid["missing_cells"] == [{"scenario_id": "*", "planner_key": "orca"}]
    assert "headline_grid_incomplete" in payload["decision_packet"]["manuscript_blockers"]


def test_cli_expected_planners_from_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """Malformed benchmark YAML fails closed with a controlled error."""
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        _planner_keys_from_benchmark_config(config_path)


def test_campaign_wrapper_preflight_only_never_launches_campaign(tmp_path: Path) -> None:
    """The public #3216 wrapper has a local no-submit preflight path."""
    repo_root = Path(__file__).resolve().parents[2]
    report_dir = tmp_path / "wrapper_preflight"
    env = {**os.environ, "REPORT_DIR": str(report_dir)}

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark/run_issue3216_headline_campaign.sh",
            "--preflight-only",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "local preflight only" in result.stdout
    assert "campaign: increased-seed-budget" not in result.stdout
    payload = json.loads((report_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["inputs"]["rows_path"] == "builtin://issue3216-dry-run"
    assert payload["decision_packet"]["claim_boundary"] == (
        "Decision packet is local preflight only; no manuscript or paper claim is promoted."
    )
    grid = payload["decision_packet"]["grid_completeness"]
    assert "prediction_planner" in grid["expected_planners"]
    assert grid["expected_cell_count"] == 9
    assert grid["missing_cell_count"] > 0


def test_campaign_wrapper_preflight_honors_config_env_override(tmp_path: Path) -> None:
    """The documented CONFIG env override defines expected planner coverage."""
    repo_root = Path(__file__).resolve().parents[2]
    report_dir = tmp_path / "wrapper_preflight"
    config_path = tmp_path / "custom_benchmark.yaml"
    config_path.write_text(
        """
name: issue3216-custom-preflight
planners:
  - key: custom_config_only
""".lstrip(),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "CONFIG": str(config_path),
        "REPORT_DIR": str(report_dir),
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark/run_issue3216_headline_campaign.sh",
            "--preflight-only",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((report_dir / "result.json").read_text(encoding="utf-8"))
    grid = payload["decision_packet"]["grid_completeness"]
    assert grid["expected_planners"] == ["custom_config_only"]
    assert grid["missing_cells"] == [{"scenario_id": "*", "planner_key": "custom_config_only"}]


def test_campaign_recovery_writes_headline_rows_from_completed_artifacts(
    tmp_path: Path,
) -> None:
    """Completed campaign CSVs are enough to recover headline rows."""

    campaign = tmp_path / "campaign"
    reports = campaign / "reports"
    reports.mkdir(parents=True)
    (reports / "scenario_family_breakdown.csv").write_text(
        "planner_key,scenario_family,episodes\norca,bottleneck,2\norca,blind_corner,1\n",
        encoding="utf-8",
    )
    (reports / "campaign_table.csv").write_text(
        "planner_key,execution_mode,status\norca,native,ok\n",
        encoding="utf-8",
    )
    (reports / "seed_episode_rows.csv").write_text(
        "episode_id,scenario_id,planner_key,seed,success,collision,near_miss,snqi\n"
        "a,classic_realworld_double_bottleneck_high,orca,111,1,0,2,0.2\n"
        "b,classic_bottleneck_high,orca,111,0,1,4,0.4\n"
        "c,francis2023_blind_corner,orca,112,1,0,0,0.8\n",
        encoding="utf-8",
    )

    rows_path = issue3216._ensure_campaign_headline_rows(campaign)

    assert rows_path == reports / "headline_rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    by_family = {row["scenario_family"]: row for row in rows}
    assert sorted(by_family) == ["blind_corner", "bottleneck"]
    bottleneck_seed = by_family["bottleneck"]["per_seed"][0]
    assert bottleneck_seed["seed"] == 111
    assert bottleneck_seed["metrics"]["success"] == 0.5
    assert bottleneck_seed["metrics"]["collisions"] == 0.5
    assert bottleneck_seed["metrics"]["near_misses"] == 3.0
    assert bottleneck_seed["metrics"]["snqi"] == pytest.approx(0.3)
    assert by_family["bottleneck"]["execution_mode"] == "native"
    assert by_family["bottleneck"]["row_status"] == "ok"


def test_campaign_wrapper_runs_builder_before_requiring_headline_rows(
    tmp_path: Path,
) -> None:
    """The wrapper no longer searches for headline_rows before builder execution."""

    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "benchmarks"
    report_dir = tmp_path / "report"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "uv_calls.log"
    uv_stub = bin_dir / "uv"
    uv_stub.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$UV_STUB_LOG"
if [ "$1" != "run" ] || [ "$2" != "python" ]; then
  exit 99
fi
shift 2
case "${1:-}" in
  -)
    exec python -
    ;;
  scripts/tools/run_camera_ready_benchmark.py)
    shift
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --campaign-id) campaign_id="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$output_root/$campaign_id/reports"
    ;;
  scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py)
    shift
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --campaign) campaign="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    test -d "$campaign/reports"
    test ! -f "$campaign/reports/headline_rows.json"
    echo '[{"scenario_family":"fixture","planner_key":"orca","per_seed":[]}]' > "$campaign/reports/headline_rows.json"
    mkdir -p "$output_dir"
    echo '{"classification":"fixture"}' > "$output_dir/result.json"
    echo '# fixture' > "$output_dir/report.md"
    ;;
  *)
    exit 98
    ;;
esac
""",
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "OUTPUT_ROOT": str(output_root),
        "REPORT_DIR": str(report_dir),
        "UV_STUB_LOG": str(log_path),
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark/run_issue3216_headline_campaign.sh",
            "--campaign-id",
            "fixture_campaign",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "locate headline rows" not in result.stdout
    assert " rows=" in result.stdout
    calls = log_path.read_text(encoding="utf-8")
    assert "scripts/tools/run_camera_ready_benchmark.py" in calls
    assert "scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py" in calls
    assert f"--campaign {output_root / 'fixture_campaign'}" in calls
    assert (output_root / "fixture_campaign" / "reports" / "headline_rows.json").is_file()
    assert (report_dir / "result.json").is_file()
