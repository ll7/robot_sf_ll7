"""Tests for issue #3501 paired safety-wrapper factorial reports."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.safety_wrapper_ablation_manifest import (
    SAFETY_WRAPPER_CONFIG_FIELD,
    SAFETY_WRAPPER_MODE_DISABLED,
    SAFETY_WRAPPER_MODE_ENABLED,
    SAFETY_WRAPPER_MODE_FIELD,
    WRAPPER_OFF_ARM,
    WRAPPER_ON_ARM,
)
from robot_sf.benchmark.safety_wrapper_factorial_report import (
    SAFETY_WRAPPER_FACTORIAL_METRICS,
    build_safety_wrapper_factorial_report,
    write_safety_wrapper_factorial_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _row(
    wrapper_arm: str, *, seed: int, metrics: dict[str, float] | None = None
) -> dict[str, object]:
    enabled = wrapper_arm == WRAPPER_ON_ARM
    base_metrics = {
        "exact_collision_probability": 1.0,
        "near_miss_probability": 1.0,
        "min_predicted_separation_m": 0.2,
        "completion_probability": 0.0,
        "progress_at_timeout": 0.4,
        "false_positive_stop_rate": 0.0,
        "stop_yield_latency_s": 0.0,
    }
    if enabled:
        base_metrics.update(
            {
                "exact_collision_probability": 0.0,
                "near_miss_probability": 0.0,
                "min_predicted_separation_m": 0.8,
                "completion_probability": 1.0,
                "progress_at_timeout": 1.0,
                "false_positive_stop_rate": 0.1,
                "stop_yield_latency_s": 0.2,
            }
        )
    if metrics:
        base_metrics.update(metrics)
    return {
        "study_id": "issue_3501_safety_wrapper_ablation_v1",
        "planner": "orca",
        "wrapper_arm": wrapper_arm,
        SAFETY_WRAPPER_MODE_FIELD: (
            SAFETY_WRAPPER_MODE_ENABLED if enabled else SAFETY_WRAPPER_MODE_DISABLED
        ),
        SAFETY_WRAPPER_CONFIG_FIELD: (
            {
                "pedestrian_caution_radius_m": 2.0,
                "capped_speed_m_s": 0.5,
                "ttc_veto_threshold_s": 1.0,
                "clearance_veto_m": 0.3,
            }
            if enabled
            else None
        ),
        "scenario_id": "francis2023_blind_corner",
        "seed": seed,
        "software_commit": "abc1234",
        "event_ledger": {"schema_version": "EpisodeEventLedger.v1"},
        "metric_values": base_metrics,
        "wrapper_intervention_rate": 0.25 if enabled else 0.0,
    }


def _complete_rows() -> list[dict[str, object]]:
    return [
        _row(WRAPPER_OFF_ARM, seed=111),
        _row(WRAPPER_ON_ARM, seed=111),
        _row(WRAPPER_OFF_ARM, seed=112),
        _row(WRAPPER_ON_ARM, seed=112),
    ]


def test_report_computes_paired_effects_and_per_planner_means() -> None:
    """Complete off/on rows produce issue #3501 primary outcome deltas."""

    report = build_safety_wrapper_factorial_report(_complete_rows())

    assert report["status"] == "complete"
    assert report["benchmark_evidence"] is True
    assert report["pair_count"] == 2
    assert report["planner_count"] == 1
    assert report["row_check"]["complete"] is True
    assert len(report["paired_effects"]) == 2
    first = report["paired_effects"][0]
    assert first["exact_collision_probability_delta_on_minus_off"] == -1.0
    assert first["near_miss_probability_delta_on_minus_off"] == -1.0
    assert first["min_predicted_separation_m_delta_on_minus_off"] == pytest.approx(0.6)
    assert first["wrapper_intervention_rate_delta_on_minus_off"] == 0.25
    per_planner = report["per_planner_effects"][0]
    assert per_planner["planner"] == "orca"
    assert per_planner["pair_count"] == 2
    assert per_planner["completion_probability_mean_delta_on_minus_off"] == 1.0
    for metric in SAFETY_WRAPPER_FACTORIAL_METRICS:
        assert f"{metric}_mean_delta_on_minus_off" in per_planner


def test_report_blocks_incomplete_pairs_before_effect_claims() -> None:
    """The report builder reuses the fail-closed paired-row checker."""

    report = build_safety_wrapper_factorial_report([_row(WRAPPER_OFF_ARM, seed=111)])

    assert report["status"] == "blocked"
    assert report["reason"] == "ablation_rows_incomplete"
    assert report["benchmark_evidence"] is False
    assert report["row_check"]["complete"] is False
    assert report["per_planner_effects"] == []


def test_report_blocks_missing_primary_outcome_metric() -> None:
    """Complete row pairs still fail closed when issue #3501 outcomes are absent."""

    rows = _complete_rows()
    del rows[1]["metric_values"]["near_miss_probability"]  # type: ignore[index]
    report = build_safety_wrapper_factorial_report(rows)

    assert report["status"] == "blocked"
    assert report["reason"] == "metric_values_incomplete"
    assert report["benchmark_evidence"] is False
    assert report["missing_metrics"] == [
        {
            "pairing_key": {
                "planner": "orca",
                "scenario_id": "francis2023_blind_corner",
                "seed": 111,
            },
            "metrics": ["near_miss_probability"],
        }
    ]


def test_write_report_outputs_expected_artifacts(tmp_path: Path) -> None:
    """The writer emits the evidence packet files requested by issue #3501."""

    report = build_safety_wrapper_factorial_report(_complete_rows())
    paths = write_safety_wrapper_factorial_report(report, tmp_path)

    assert paths["summary"].name == "summary.json"
    assert paths["per_planner_effects"].name == "per_planner_effects.csv"
    assert paths["readme"].name == "README.md"
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["schema_version"] == "robot_sf.issue_3501_safety_wrapper_factorial_report.v1"
    assert "exact_collision_probability_mean_delta_on_minus_off" in paths[
        "per_planner_effects"
    ].read_text(encoding="utf-8")
    assert "Diagnostic paired safety-wrapper factorial report" in paths["readme"].read_text(
        encoding="utf-8"
    )


def test_cli_fails_closed_for_incomplete_rows(tmp_path: Path) -> None:
    """The CLI writes blockers and exits non-zero rather than overstating evidence."""

    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps([_row(WRAPPER_OFF_ARM, seed=111)]) + "\n", encoding="utf-8")
    out_dir = tmp_path / "report"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/build_issue3501_safety_wrapper_factorial_report.py",
            "--rows",
            str(rows_path),
            "--out",
            str(out_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "status=blocked" in completed.stdout
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"


def test_cli_writes_complete_report(tmp_path: Path) -> None:
    """The CLI builds the report artifacts for complete paired rows."""

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in _complete_rows()) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "report"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/build_issue3501_safety_wrapper_factorial_report.py",
            "--rows",
            str(rows_path),
            "--out",
            str(out_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=complete" in completed.stdout
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "per_planner_effects.csv").is_file()
    assert (out_dir / "README.md").is_file()
