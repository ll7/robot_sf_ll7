"""Tests for the issue #4195 h600 aggregation artifact builder."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from scripts.analysis.build_issue_4195_h600_aggregation_artifact import (
    build_artifact,
    extend_existing_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_fixture_reports(
    reports_dir: Path,
    *,
    scenario_matrix_hash: str = "matrix-a",
    planners: tuple[str, ...] = ("goal", "orca"),
    with_exposure_fields: bool = False,
) -> None:
    reports_dir.mkdir(parents=True)
    planner_rows = [
        {
            "planner_key": planner,
            "episodes": 4,
            "success_mean": "0.5000",
            "collisions_mean": "0.5000",
            "near_misses_mean": "1.5000",
            "comfort_exposure_mean": "0.2500",
            "snqi_mean": "0.1250",
        }
        for planner in planners
    ]
    runs = [
        {
            "planner": {"key": planner},
            "aggregates": {
                planner: {
                    "comfort_exposure": {
                        "mean": 0.25,
                        "mean_ci": [0.1, 0.4],
                    }
                }
            },
        }
        for planner in planners
    ]
    summary = {
        "campaign": {
            "campaign_id": reports_dir.name,
            "evidence_status": "valid",
            "benchmark_success": True,
            "scenario_matrix": "configs/scenarios/example.yaml",
            "scenario_matrix_hash": scenario_matrix_hash,
            "comparability_mapping_hash": "mapping-a",
            "created_at_utc": "2026-07-03T00:00:00Z",
            "git_hash": "abc123",
        },
        "planner_rows": planner_rows,
        "runs": runs,
    }
    (reports_dir / "campaign_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    with (reports_dir / "seed_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "episode_id",
            "scenario_id",
            "planner_key",
            "seed",
            "success",
            "collision",
            "near_miss",
            "snqi",
        ]
        if with_exposure_fields:
            fieldnames.extend(
                [
                    "interaction_exposure_share",
                    "robot_motion_share_before_first_clearance",
                    "first_clearance_step",
                    "low_exposure_success",
                    "interaction_exposure_source",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for planner in planners:
            rows = [
                {
                    "episode_id": f"{planner}-111-0",
                    "scenario_id": "a",
                    "planner_key": planner,
                    "seed": "111",
                    "success": "1.0",
                    "collision": "0.0",
                    "near_miss": "1.0",
                    "snqi": "0.2",
                },
                {
                    "episode_id": f"{planner}-111-1",
                    "scenario_id": "b",
                    "planner_key": planner,
                    "seed": "111",
                    "success": "0.0",
                    "collision": "1.0",
                    "near_miss": "3.0",
                    "snqi": "0.0",
                },
                {
                    "episode_id": f"{planner}-112-0",
                    "scenario_id": "a",
                    "planner_key": planner,
                    "seed": "112",
                    "success": "1.0",
                    "collision": "0.0",
                    "near_miss": "0.0",
                    "snqi": "0.4",
                },
                {
                    "episode_id": f"{planner}-112-1",
                    "scenario_id": "b",
                    "planner_key": planner,
                    "seed": "112",
                    "success": "0.0",
                    "collision": "1.0",
                    "near_miss": "2.0",
                    "snqi": "0.2",
                },
            ]
            if with_exposure_fields:
                for index, row in enumerate(rows):
                    row.update(
                        {
                            "interaction_exposure_share": "0.75" if index < 3 else "",
                            "robot_motion_share_before_first_clearance": "0.5" if index < 3 else "",
                            "first_clearance_step": "42" if index < 3 else "",
                            "low_exposure_success": "0.0" if index < 3 else "",
                            "interaction_exposure_source": "episode_metrics"
                            if index < 3
                            else "not_derivable_from_episode_record",
                        }
                    )
            writer.writerows(rows)


def _write_h500_fixture(reports_dir: Path) -> None:
    """Write compact h500 comparison fixture."""

    reports_dir.mkdir(parents=True)
    summary = {
        "campaign": {
            "campaign_id": "fixture-h500-s20",
            "scenario_matrix": "configs/scenarios/example.yaml",
            "scenario_matrix_hash": "h500-matrix",
        },
        "planner_rows": [
            {
                "planner_key": "goal",
                "success_mean": "0.25",
                "collisions_mean": "0.75",
                "near_misses_mean": "4.0",
                "snqi_mean": "-0.40",
            },
            {
                "planner_key": "orca",
                "success_mean": "0.75",
                "collisions_mean": "0.25",
                "near_misses_mean": "1.0",
                "snqi_mean": "-0.10",
            },
        ],
    }
    (reports_dir / "campaign_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_build_artifact_writes_tables_comparability_and_checksums(tmp_path: Path) -> None:
    """Fixture reports produce deterministic aggregation artifacts."""

    confirm_reports = tmp_path / "confirm" / "reports"
    extended_reports = tmp_path / "extended" / "reports"
    h500_reports = tmp_path / "h500" / "reports"
    _write_fixture_reports(confirm_reports)
    _write_fixture_reports(extended_reports, planners=("goal", "orca", "prediction_mpc"))
    _write_h500_fixture(h500_reports)
    output_dir = tmp_path / "evidence"

    result = build_artifact(
        confirm_reports=confirm_reports,
        extended_reports=extended_reports,
        h500_s20_reports=h500_reports,
        output_dir=output_dir,
        bootstrap_samples=100,
        confidence=0.95,
    )

    assert result["status"] == "ok"
    assert result["snqi_recalibration_status"] == "ok"
    assert result["horizon_sensitivity_status"] == "ok"
    assert result["interaction_exposure_status"] == "blocked_missing_required_fields"
    assert result["row_count"] == 25
    metric_rows = list(csv.DictReader((output_dir / "planner_metric_summary.csv").open()))
    goal_snqi = next(
        row
        for row in metric_rows
        if row["job_id"] == "13268" and row["planner_key"] == "goal" and row["metric"] == "snqi"
    )
    assert json.loads(goal_snqi["seed_values_json"]) == pytest.approx({"111": 0.1, "112": 0.3})
    assert goal_snqi["source"] == "seed_episode_rows"
    goal_comfort = next(
        row
        for row in metric_rows
        if row["job_id"] == "13268" and row["planner_key"] == "goal" and row["metric"] == "comfort"
    )
    assert goal_comfort["source"] == "campaign_summary_aggregate"
    assert goal_comfort["value_status"] == "no_seed_episode_column"
    assert goal_comfort["bootstrap_ci_low"] == "0.100000"
    assert goal_comfort["bootstrap_ci_high"] == "0.400000"

    comparability = json.loads((output_dir / "comparability_check.json").read_text())
    assert comparability["status"] == "pass"
    assert comparability["shared_planners"] == ["goal", "orca"]
    snqi_recalibration = json.loads((output_dir / "snqi_recalibration_bundle.json").read_text())
    assert snqi_recalibration["status"] == "ok"
    assert snqi_recalibration["h500_source"]["status"] == "available_s10_h500_not_s20"
    assert snqi_recalibration["decision_reversal_rows"]
    horizon = json.loads((output_dir / "horizon_sensitivity_report.json").read_text())
    assert horizon["status"] == "ok"
    assert {row["metric"] for row in horizon["comparison_rows"]} >= {"snqi", "success"}
    exposure = json.loads((output_dir / "interaction_exposure_diagnostics.json").read_text())
    assert exposure["status"] == "blocked_missing_required_fields"
    assert "interaction_exposure_share" in exposure["missing_required_fields"]
    assert exposure["backfill_policy"] == "derive_from_retained_episode_rows_only_no_imputation"
    assert exposure["runs"][0]["derivable_episode_rows"] == 0
    assert exposure["runs"][0]["not_derivable_episode_rows"] == 8
    assert (
        (output_dir / "README.md")
        .read_text(encoding="utf-8")
        .startswith("# Issue 4195 h600 Aggregation Artifact")
    )
    checksums = (output_dir / "SHA256SUMS").read_text(encoding="utf-8")
    assert "planner_metric_summary.csv" in checksums
    assert "snqi_recalibration_bundle.json" in checksums
    assert "horizon_sensitivity_report.json" in checksums
    assert "interaction_exposure_diagnostics.json" in checksums
    assert "source_manifest.json" in checksums

    extended_result = extend_existing_artifact(
        output_dir=output_dir,
        h500_s20_reports=h500_reports,
    )
    assert extended_result["snqi_recalibration_status"] == "ok"
    assert extended_result["interaction_exposure_status"] == "blocked_missing_required_fields"


def test_build_artifact_accepts_native_exposure_fields_without_imputation(tmp_path: Path) -> None:
    """Exposure diagnostics become ready only for rows carrying native exposure fields."""

    confirm_reports = tmp_path / "confirm" / "reports"
    extended_reports = tmp_path / "extended" / "reports"
    h500_reports = tmp_path / "h500" / "reports"
    _write_fixture_reports(confirm_reports, with_exposure_fields=True)
    _write_fixture_reports(extended_reports, planners=("goal", "orca"), with_exposure_fields=True)
    _write_h500_fixture(h500_reports)

    output_dir = tmp_path / "evidence"
    result = build_artifact(
        confirm_reports=confirm_reports,
        extended_reports=extended_reports,
        h500_s20_reports=h500_reports,
        output_dir=output_dir,
        bootstrap_samples=20,
        confidence=0.95,
    )

    assert result["interaction_exposure_status"] == "ready_for_episode_level_scan"
    exposure = json.loads((output_dir / "interaction_exposure_diagnostics.json").read_text())
    assert exposure["missing_required_fields"] == []
    assert exposure["runs"][0]["derivable_episode_rows"] == 6
    assert exposure["runs"][0]["not_derivable_episode_rows"] == 2
    assert exposure["runs"][0]["exposure_provenance_values"] == [
        "episode_metrics",
        "not_derivable_from_episode_record",
    ]


def _write_exposure_sidecar(
    path: Path,
    *,
    statuses: dict[str, str],
) -> None:
    """Write a minimal #4242 interaction-exposure sidecar keyed by job_id.

    ``statuses`` maps job_id -> row-level ``interaction_exposure_status``. A
    ``computed`` status also carries populated required values; other statuses
    leave the required values blank so no imputation is possible.
    """

    fieldnames = [
        "job_id",
        "run_label",
        "episode_id",
        "scenario_id",
        "planner_key",
        "seed",
        "interaction_exposure_share",
        "robot_motion_share_before_first_clearance",
        "first_clearance_step",
        "low_exposure_success",
        "interaction_exposure_source",
        "interaction_exposure_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for job_id, status in statuses.items():
            computed = status == "computed"
            writer.writerow(
                {
                    "job_id": job_id,
                    "run_label": "confirm" if job_id == "13268" else "extended_roster",
                    "episode_id": f"{job_id}-ep-0",
                    "scenario_id": "a",
                    "planner_key": "goal",
                    "seed": "111",
                    "interaction_exposure_share": "0.5" if computed else "",
                    "robot_motion_share_before_first_clearance": "0.5" if computed else "",
                    "first_clearance_step": "10" if computed else "",
                    "low_exposure_success": "0.0" if computed else "",
                    "interaction_exposure_source": "computed_from_retained_trace"
                    if computed
                    else "not_derivable_from_episode_record",
                    "interaction_exposure_status": status,
                }
            )


def _build_native_absent_artifact(tmp_path: Path) -> Path:
    """Build a native-absent aggregation artifact and return its output dir."""

    confirm_reports = tmp_path / "confirm" / "reports"
    extended_reports = tmp_path / "extended" / "reports"
    h500_reports = tmp_path / "h500" / "reports"
    _write_fixture_reports(confirm_reports)
    _write_fixture_reports(extended_reports, planners=("goal", "orca"))
    _write_h500_fixture(h500_reports)
    output_dir = tmp_path / "evidence"
    result = build_artifact(
        confirm_reports=confirm_reports,
        extended_reports=extended_reports,
        h500_s20_reports=h500_reports,
        output_dir=output_dir,
        bootstrap_samples=20,
        confidence=0.95,
    )
    assert result["interaction_exposure_status"] == "blocked_missing_required_fields"
    return output_dir


def _declare_sidecar_in_manifest(output_dir: Path, sidecar_name: str) -> None:
    """Add the #4242 sidecar declaration to the artifact's source manifest."""

    manifest_path = output_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["issue_4242_h600_sidecars"] = {
        "interaction_exposure_sidecar": sidecar_name,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_extend_consumes_declared_sidecar_all_not_derivable_without_imputation(
    tmp_path: Path,
) -> None:
    """A declared sidecar with only not-derivable rows stays fail-closed."""

    output_dir = _build_native_absent_artifact(tmp_path)
    _write_exposure_sidecar(
        output_dir / "h600_interaction_exposure_sidecar.csv",
        statuses={"13268": "not_derivable_missing_trace", "13273": "not_derivable_missing_trace"},
    )
    _declare_sidecar_in_manifest(output_dir, "h600_interaction_exposure_sidecar.csv")

    result = extend_existing_artifact(
        output_dir=output_dir,
        h500_s20_reports=tmp_path / "h500" / "reports",
    )

    assert result["interaction_exposure_status"] == "sidecar_all_not_derivable"
    exposure = json.loads((output_dir / "interaction_exposure_diagnostics.json").read_text())
    assert exposure["exposure_field_source"] == "sidecar_declared"
    assert exposure["declared_sidecar"] == (
        "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
        "h600_interaction_exposure_sidecar.csv"
    ) or exposure["declared_sidecar"].endswith("h600_interaction_exposure_sidecar.csv")
    confirm_run = next(row for row in exposure["runs"] if row["job_id"] == "13268")
    assert confirm_run["exposure_field_source"] == "sidecar_declared"
    assert confirm_run["sidecar_derivable_episode_rows"] == 0
    assert confirm_run["sidecar_not_derivable_episode_rows"] == 1
    assert confirm_run["sidecar_exposure_status_counts"] == {"not_derivable_missing_trace": 1}


def test_extend_consumes_declared_sidecar_with_computed_rows(tmp_path: Path) -> None:
    """A declared sidecar with a computed row unblocks the episode-level scan."""

    output_dir = _build_native_absent_artifact(tmp_path)
    _write_exposure_sidecar(
        output_dir / "h600_interaction_exposure_sidecar.csv",
        statuses={"13268": "computed", "13273": "not_derivable_missing_trace"},
    )
    _declare_sidecar_in_manifest(output_dir, "h600_interaction_exposure_sidecar.csv")

    result = extend_existing_artifact(
        output_dir=output_dir,
        h500_s20_reports=tmp_path / "h500" / "reports",
    )

    assert result["interaction_exposure_status"] == "sidecar_ready_for_episode_level_scan"
    exposure = json.loads((output_dir / "interaction_exposure_diagnostics.json").read_text())
    confirm_run = next(row for row in exposure["runs"] if row["job_id"] == "13268")
    assert confirm_run["status"] == "sidecar_ready_for_episode_level_scan"
    assert confirm_run["sidecar_derivable_episode_rows"] == 1
    extended_run = next(row for row in exposure["runs"] if row["job_id"] == "13273")
    assert extended_run["status"] == "sidecar_all_not_derivable"


def test_native_not_derivable_status_marker_is_not_counted_as_derivable(tmp_path: Path) -> None:
    """Native rows with populated values but not-derivable status are not imputed."""

    from scripts.analysis.build_issue_4195_h600_aggregation_artifact import (
        _classify_exposure_rows,
    )

    rows = [
        {
            "interaction_exposure_share": "0.5",
            "robot_motion_share_before_first_clearance": "0.5",
            "first_clearance_step": "10",
            "low_exposure_success": "0.0",
            "interaction_exposure_status": "computed",
            "interaction_exposure_source": "episode_metrics",
        },
        {
            "interaction_exposure_share": "0.9",
            "robot_motion_share_before_first_clearance": "0.9",
            "first_clearance_step": "3",
            "low_exposure_success": "0.0",
            "interaction_exposure_status": "not_derivable_missing_trace",
            "interaction_exposure_source": "not_derivable_from_episode_record",
        },
    ]
    derivable, not_derivable, status_counts, provenance = _classify_exposure_rows(rows)
    assert derivable == 1
    assert not_derivable == 1
    assert status_counts == {"computed": 1, "not_derivable_missing_trace": 1}
    assert provenance == ["episode_metrics", "not_derivable_from_episode_record"]


def test_build_artifact_fails_status_on_shared_hash_mismatch(tmp_path: Path) -> None:
    """Shared planners are not comparable when scenario matrix hashes differ."""

    confirm_reports = tmp_path / "confirm" / "reports"
    extended_reports = tmp_path / "extended" / "reports"
    h500_reports = tmp_path / "h500" / "reports"
    _write_fixture_reports(confirm_reports, scenario_matrix_hash="matrix-a")
    _write_fixture_reports(extended_reports, scenario_matrix_hash="matrix-b")
    _write_h500_fixture(h500_reports)

    result = build_artifact(
        confirm_reports=confirm_reports,
        extended_reports=extended_reports,
        h500_s20_reports=h500_reports,
        output_dir=tmp_path / "evidence",
        bootstrap_samples=20,
        confidence=0.95,
    )

    assert result["status"] == "comparability_failed"
    comparability = json.loads((tmp_path / "evidence" / "comparability_check.json").read_text())
    assert comparability["status"] == "fail"
    assert comparability["checks"]["scenario_matrix_hash_match"] is False
