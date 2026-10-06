"""Regression probes for defects found by the 0.0.8 rehearsal (dev seeds only)."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows
from robot_sf.benchmark.camera_ready import _artifacts, _reporting, campaign
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.planner.guarded_ppo import GuardedPPOAdapter
from scripts.tools import analyze_camera_ready_campaign, analyze_snqi_contract
from scripts.tools import run_camera_ready_benchmark as runner


def test_nested_guarded_diagnostics_are_not_fallback_counters():
    diagnostics = GuardedPPOAdapter().diagnostics()
    assert isinstance(diagnostics["fallback_diagnostics"], dict)
    assert runtime_fallback_or_degraded_marker(diagnostics) is None
    diagnostics["fallback_diagnostics"]["fallback_count"] = 1
    assert runtime_fallback_or_degraded_marker(diagnostics) == (
        "fallback_diagnostics.fallback_count",
        "1",
    )


@pytest.mark.parametrize(
    "version", ["v2_h600_s30_benchmark_data_2026_08", "v2_h600_s30_benchmark_data_template"]
)
def test_auditor_coverage_uses_manifest_arms(version):
    template = Path(f"configs/benchmarks/paper_experiment_matrix_{version}.yaml")
    roster = [
        p["key"] for p in yaml.safe_load(template.read_text())["planners"] if p.get("enabled", True)
    ]
    rows = [
        {
            "scenario_id": "smoke",
            "seed": 1001,
            "algo": arm,
            "_release_arm": arm,
            "steps": 30,
            "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
            "event_ledger": {"exact_events": {"invalid_run": False}},
            "metrics": {"success": 1},
        }
        for arm in roster
    ]
    report = analyze_release_rows(
        rows,
        source={"planner_ids": roster},
        preflight=[{"scenario_id": "smoke", "seed": 1001, "invalid_run": False}],
    )
    assert report["coverage"]["incomplete_cells"] == 0
    assert report["coverage"]["complete_cells"] == 1
    missing = analyze_release_rows(
        rows[:-1],
        source={"planner_ids": roster},
        preflight=[{"scenario_id": "smoke", "seed": 1001, "invalid_run": False}],
    )
    assert missing["coverage"]["incomplete_cells"] == 1


def test_breakdown_csv_metric_columns_have_values(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    metrics = {
        "success": 1,
        "collisions": 0,
        "ped_collision_count": 0,
        "obstacle_collision_count": 0,
        "total_collision_count": 0,
        "near_misses": 0,
        "time_to_goal_norm": 0.5,
        "path_efficiency": 0.75,
        "comfort_exposure": 0.25,
        "jerk_mean": 2.5,
        "snqi": 0.6,
    }
    episodes.write_text(
        json.dumps({"scenario_id": "smoke", "seed": 1001, "metrics": metrics}) + "\n"
    )
    scenarios, families = _reporting._build_breakdown_rows(
        [{"planner": {"key": "goal", "algo": "goal"}, "episodes_path": str(episodes)}]
    )
    for rows, headers in [
        (scenarios, campaign._SCENARIO_BREAKDOWN_HEADERS),
        (families, campaign._FAMILY_BREAKDOWN_HEADERS),
    ]:
        output, _ = _artifacts._write_table_artifacts(tmp_path, "breakdown", rows, headers=headers)
        row = next(csv.DictReader(output.open()))
        assert float(row["jerk_mean"]) == 2.5
        for metric, value in metrics.items():
            column = "jerk_mean" if metric == "jerk_mean" else f"{metric}_mean"
            assert float(row[column]) == value


def test_publication_bundle_follows_campaign_output_root(tmp_path):
    captured = []
    cfg = SimpleNamespace(
        export_publication_bundle=True,
        include_videos_in_publication=False,
        repository_url="https://example.org",
        release_tag="dev",
        doi="dev",
        overwrite_publication_bundle=False,
    )

    def export(root, destination, **kwargs):
        captured.append(destination)
        raise OSError("probe ends after recording destination")

    campaign._export_publication_bundle_section(
        cfg,
        campaign_id="smoke",
        campaign_root=tmp_path / "custom" / "smoke",
        campaign_summary={},
        warnings=[],
        skip_publication_bundle=False,
        snqi_hard_fail=False,
        benchmark_success=True,
        dependencies=SimpleNamespace(export_publication_bundle=export),
    )
    assert captured == [tmp_path / "custom" / "publication"]


def test_run_rejects_checkpoint_staging_option_before_loading_config(monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_campaign_config", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        runner, "run_campaign", lambda *a, **k: {"status": "benchmark_success", "exit_code": 0}
    )
    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--config",
                "unused.yaml",
                "--mode",
                "run",
                "--checkpoint-preflight-mode",
                "enforced_staged",
            ]
        )
    assert exc.value.code == 2
    assert "requires --mode preflight" in capsys.readouterr().err


def test_relocated_analyzer_prefers_local_bytes_and_preserves_provenance(tmp_path):
    root = tmp_path / "copied"
    local = root / "runs" / "goal" / "episodes.jsonl"
    local.parent.mkdir(parents=True)
    local.write_text('{"seed":1001}\n')
    recorded = "/another-host/campaign/runs/goal/episodes.jsonl"
    summary = {"episodes_path": recorded}
    resolved = analyze_camera_ready_campaign._resolve_safe_episodes_path(
        root, summary["episodes_path"]
    )
    assert resolved == local
    assert json.loads(resolved.read_text())["seed"] == 1001
    assert summary["episodes_path"] == recorded
    with pytest.raises(ValueError, match="Unsafe"):
        analyze_camera_ready_campaign._resolve_safe_episodes_path(root, "/host/runs/../../secret")


def test_snqi_analysis_does_not_change_campaign_reports(tmp_path, capsys):
    root = tmp_path / "campaign"
    reports = root / "reports"
    reports.mkdir(parents=True)
    episodes = root / "runs" / "goal" / "episodes.jsonl"
    episodes.parent.mkdir(parents=True)
    episodes.write_text(
        json.dumps(
            {
                "scenario_id": "smoke",
                "seed": 1001,
                "metrics": {"success": 1, "collisions": 0, "jerk_mean": 0.1},
            }
        )
        + "\n"
    )
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "planner_rows": [],
                "runs": [{"planner": {"key": "goal"}, "episodes_path": str(episodes)}],
            }
        )
    )
    (reports / "snqi_sensitivity.csv").write_text("original publication bytes\n")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert (
        analyze_snqi_contract.main(
            ["--campaign-root", str(root), "--trials", "2", "--seed", "1001"]
        )
        == 0
    )
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert after == before
    outputs = json.loads(capsys.readouterr().out)
    assert Path(outputs["snqi_sensitivity_csv"]).parent == tmp_path / "campaign_snqi_analysis"


@pytest.mark.parametrize(
    "option",
    [
        ["--checkpoint-preflight-mode", "metadata_only"],
        ["--checkpoint-preflight-mode=metadata_only"],
    ],
)
def test_run_accepts_explicit_metadata_only(monkeypatch, option):
    monkeypatch.setattr(runner, "load_campaign_config", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        runner, "run_campaign", lambda *a, **k: {"status": "benchmark_success", "exit_code": 0}
    )
    assert runner.main(["--config", "unused.yaml", "--mode", "run", *option]) == 0


@pytest.mark.parametrize("matching", [True, False])
def test_relocated_bytes_bind_sidecar_when_recorded_file_also_exists(tmp_path, matching):
    recorded = tmp_path / "producer" / "runs" / "goal" / "episodes.jsonl"
    recorded.parent.mkdir(parents=True)
    recorded.write_bytes(b'{"seed":1002}\n')
    root = tmp_path / "copied"
    local = root / "runs" / "goal" / "episodes.jsonl"
    local.parent.mkdir(parents=True)
    local.write_bytes(b'{"seed":1001}\n')
    expected = hashlib.sha256(local.read_bytes() if matching else recorded.read_bytes()).hexdigest()
    local.with_name(local.name + ".provenance.json").write_text(
        json.dumps(
            {
                "raw_artifacts": [
                    {"kind": "episodes_jsonl", "path": str(recorded), "sha256": expected}
                ]
            }
        )
    )
    if matching:
        assert (
            analyze_camera_ready_campaign._resolve_safe_episodes_path(root, str(recorded)) == local
        )
    else:
        with pytest.raises(ValueError, match="digest mismatch"):
            analyze_camera_ready_campaign._resolve_safe_episodes_path(root, str(recorded))
