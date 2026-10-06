"""Release cohort counterexamples through real JSONL/report consumers; no rollouts."""

import csv
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config_types import (
    CampaignConfig,
    PlannerSpec,
    SnqiContractConfig,
)
from robot_sf.benchmark.camera_ready._reporting import _build_breakdown_rows, _planner_report_row
from robot_sf.benchmark.camera_ready.campaign import (
    _CAMPAIGN_TABLE_HEADERS,
    _build_and_write_snqi_section,
    _write_table_artifacts,
)
from robot_sf.benchmark.seed_variance import (
    build_seed_variability_csv_rows,
    build_seed_variability_rows,
)
from robot_sf.benchmark.snqi.campaign_contract import collect_episodes_from_campaign_runs


def _records():
    return [
        {
            "scenario_id": "s",
            "algo": "goal",
            "seed": 1001,
            "metrics": {"success": 0, "snqi": -1.0, "collisions": 1},
        },
        {
            "scenario_id": "s",
            "algo": "goal",
            "seed": 1002,
            "metrics": {"success": 1, "snqi": 0.0, "collisions": 0},
            "algorithm_metadata": {"foresight_prediction": {"evidence_eligible": False}},
        },
    ]


def _entries(tmp_path, records):
    path = tmp_path / "episodes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return [{"planner": {"key": "goal", "algo": "goal"}, "episodes_path": str(path)}]


def test_snqi_release_diagnostics_filter_before_all_consumers(tmp_path):
    """Collector and actual written diagnostics admit only the eligible failure."""
    entries = _entries(tmp_path, _records())
    episodes = collect_episodes_from_campaign_runs(entries, repo_root=tmp_path)
    assert len(episodes) == 1
    cfg = CampaignConfig(
        name="cohort",
        scenario_matrix_path=tmp_path / "unused.yaml",
        planners=(),
        snqi_contract=SnqiContractConfig(calibration_trials=1),
    )
    result = _build_and_write_snqi_section(
        cfg,
        campaign_id="cohort",
        campaign_finished_at_utc="2026-09-30T00:00:00Z",
        reports_dir=tmp_path,
        planner_rows=[],
        run_entries=entries,
        snqi_weights=None,
        snqi_baseline=None,
        warnings=[],
    )
    payload = json.loads(result.snqi_diagnostics_json_path.read_text())
    assert payload["planner_ordering"][0]["episode_count"] == 1
    assert payload["planner_ordering"][0]["mean_snqi"] == -1.0
    assert payload["evidence_cohort"] == {
        "episodes_total": 2,
        "episodes_eligible": 1,
        "episodes_excluded": 1,
        "exclusion_reasons": {"foresight_ineligible": 1},
    }


def test_campaign_table_counts_match_metric_cohort_and_writers(tmp_path):
    """Published N matches means; CSV and Markdown retain total and excluded counts."""
    row = _planner_report_row(
        PlannerSpec(key="goal", algo="goal"),
        {},
        None,
        kinematics="differential_drive",
        records=_records(),
    )
    assert row["episodes"] == 1
    assert row["episodes_total"] == 2
    assert row["episodes_excluded"] == 1
    assert float(row["success_mean"]) == 0.0
    csv_path, md_path = _write_table_artifacts(
        tmp_path, "campaign_table", [row], headers=_CAMPAIGN_TABLE_HEADERS
    )
    with csv_path.open() as handle:
        written = next(csv.DictReader(handle))
    assert [written[key] for key in ("episodes", "episodes_total", "episodes_excluded")] == [
        "1",
        "2",
        "1",
    ]
    assert "episodes_total" in md_path.read_text()
    assert "episodes_excluded" in md_path.read_text()


def test_all_excluded_arms_remain_in_breakdown_and_seed_outputs(tmp_path):
    """An excluded arm stays visible with zero admitted N and no metric estimates."""
    rows = [_records()[1]]
    scenario, family = _build_breakdown_rows(_entries(tmp_path, rows))
    assert len(scenario) == len(family) == 1
    for row in (*scenario, *family):
        assert (row["episodes"], row["episodes_total"], row["episodes_excluded"]) == (0, 1, 1)
        assert row["success_mean"] == "nan"
    report = build_seed_variability_rows(
        rows,
        metrics=("success",),
        campaign_id="cohort",
        config_hash="fixture",
        git_hash="fixture",
        confidence_settings={"bootstrap_samples": 0},
    )
    assert len(report) == 1
    assert report[0]["episode_count"] == report[0]["seed_count"] == 0
    assert report[0]["episodes_total"] == report[0]["episodes_excluded"] == 1
    assert report[0]["summary"] == {}
    csv_rows = build_seed_variability_csv_rows(report, metrics=("success",))
    assert len(csv_rows) == 1
    assert csv_rows[0]["seed_episode_count"] == 0
    assert csv_rows[0]["seed_episodes_excluded"] == 1


def test_breakdown_skips_non_object_jsonl_lines(tmp_path):
    """Non-object JSONL must be ignored before invoking the mapping-only filter."""
    scenario, family = _build_breakdown_rows(_entries(tmp_path, [None, [], 7, *_records()]))
    assert len(scenario) == len(family) == 1
    assert scenario[0]["episodes"] == 1


def test_real_0_0_7_sample_preserves_count_and_means():
    """Immutable real release sample remains eligible without new provenance markers."""
    path = Path(__file__).parents[1] / "analysis/fixtures/issue_9668_0_0_7_goal_sample.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    row = _planner_report_row(
        PlannerSpec(key="goal", algo="goal"),
        {},
        None,
        kinematics="differential_drive",
        records=rows,
    )
    assert row["episodes"] == len(rows)
    assert row["episodes_total"] == len(rows)
    assert row["episodes_excluded"] == 0
    assert float(row["success_mean"]) == pytest.approx(
        sum(r["metrics"]["success"] for r in rows) / len(rows), abs=0.00005
    )
    assert float(row["snqi_mean"]) == pytest.approx(
        sum(r["metrics"]["snqi"] for r in rows) / len(rows), abs=0.00005
    )


def test_seed_variability_retains_excluded_only_seed():
    """Eligible seed statistics and excluded-only seed accounting stay distinct."""
    report = build_seed_variability_rows(
        _records(),
        metrics=("success",),
        campaign_id="cohort",
        config_hash="fixture",
        git_hash="fixture",
        confidence_settings={"bootstrap_samples": 0},
    )[0]
    assert report["episode_count"] == report["seed_count"] == 1
    assert report["seed_list"] == [1001]
    assert report["episodes_total"] == 2
    assert report["episodes_excluded"] == 1
    assert report["summary"]["success"]["mean"] == 0.0
    csv_rows = build_seed_variability_csv_rows([report], metrics=("success",))
    assert [
        (row["seed"], row["seed_episode_count"], row["seed_episodes_excluded"]) for row in csv_rows
    ] == [(1001, 1, 0), (1002, 0, 1)]


@pytest.mark.parametrize(
    "spawn",
    [
        {"invalid_run": True, "invalid_reason": "spawn_overlap"},
        {
            "schema_version": "spawn_validity.v2",
            "invalid_run": False,
            "invalid_reason": None,
            "reset_clearance_status": "unavailable",
        },
    ],
)
def test_collector_excludes_invalid_and_unmeasured_spawn(tmp_path, spawn):
    """The canonical spawn rule also gates legacy SNQI consumers, including forged flags."""
    bad = {**_records()[0], "spawn_validity": spawn, "seed": 1003}
    entries = _entries(tmp_path, [_records()[0], bad])
    assert len(collect_episodes_from_campaign_runs(entries, repo_root=tmp_path)) == 1
    metadata = {}
    collect_episodes_from_campaign_runs(entries, repo_root=tmp_path, cohort_metadata=metadata)
    assert metadata["episodes_excluded"] == 1
    assert metadata["exclusion_reasons"] == {"invalid_or_unmeasured_spawn": 1}


def test_comparator_detects_exclusion_count_drift():
    """Equal eligible means/N cannot hide a different run/exclusion cohort."""
    from scripts.tools.compare_camera_ready_campaigns import _planner_row_signature

    old = {"episodes": 1, "success_mean": 0.0}
    same = {**old, "episodes_total": 1, "episodes_excluded": 0}
    mixed = {**old, "episodes_total": 2, "episodes_excluded": 1}
    assert _planner_row_signature(old) != _planner_row_signature(mixed)
    assert _planner_row_signature(old) == _planner_row_signature(same)


def test_publication_payload_preserves_campaign_count_schema(tmp_path):
    """The real exporter ships all count columns and their schema without a whitelist loss."""
    from jsonschema import Draft202012Validator

    from robot_sf.benchmark.artifact_publication import export_publication_bundle
    from robot_sf.benchmark.camera_ready.campaign import _write_campaign_table_artifacts

    run_dir = tmp_path / "run"
    reports_dir = run_dir / "reports"
    row = _planner_report_row(
        PlannerSpec(key="goal", algo="goal"),
        {},
        None,
        kinematics="differential_drive",
        records=_records(),
    )
    cfg = CampaignConfig(name="cohort", scenario_matrix_path=tmp_path / "unused.yaml", planners=())
    _write_campaign_table_artifacts(cfg, reports_dir, [row])
    schema_path = reports_dir / "campaign_table.schema.json"
    assert schema_path.exists()
    Draft202012Validator(json.loads(schema_path.read_text())).validate(row)
    (reports_dir / "campaign_summary.json").write_text(json.dumps({"planner_rows": [row]}))
    result = export_publication_bundle(run_dir, tmp_path / "bundles", bundle_name="cohort")
    payload = result.bundle_dir / "payload/reports"
    assert (payload / "campaign_table.schema.json").read_bytes() == schema_path.read_bytes()
    assert (
        json.loads((payload / "campaign_summary.json").read_text())["planner_rows"][0][
            "episodes_excluded"
        ]
        == 1
    )
    with (payload / "campaign_table.csv").open() as handle:
        written = next(csv.DictReader(handle))
    assert [written[key] for key in ("episodes", "episodes_total", "episodes_excluded")] == [
        "1",
        "2",
        "1",
    ]


def test_column_selecting_readers_preserve_cohort_counts(tmp_path):
    """Derived difficulty, evidence packets and H600 exports retain both N diagnostics."""
    from robot_sf.benchmark.scenario_difficulty import _normalize_scenario_rows
    from scripts.analysis.build_issue_5138_h600_family_breakdown_export import (
        FAMILY_METRIC_PASSTHROUGH,
        RunSource,
        _norm_row,
    )
    from scripts.validation.extract_s20_s30_evidence_gap_packet import _read_campaign_table

    raw = {
        "planner_key": "goal",
        "algo": "goal",
        "scenario_id": "s",
        "scenario_family": "s",
        "episodes": "1",
        "episodes_total": "2",
        "episodes_excluded": "1",
        "success_mean": "0.0",
    }
    normalized, _ = _normalize_scenario_rows([raw], {}, {}, {})
    assert normalized[0]["episodes_total"] == 2
    assert normalized[0]["episodes_excluded"] == 1
    path = tmp_path / "campaign_table.csv"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw))
        writer.writeheader()
        writer.writerow(raw)
    packet = _read_campaign_table(path)[0]
    assert (packet["episodes_total"], packet["episodes_excluded"]) == ("2", "1")
    exported = _norm_row(
        raw,
        run=RunSource(job_id="fixture", run_label="fixture", reports_dir=tmp_path),
        matrix_hash="fixture",
        passthrough=FAMILY_METRIC_PASSTHROUGH,
    )
    assert (exported["episodes_total"], exported["episodes_excluded"]) == ("2", "1")
