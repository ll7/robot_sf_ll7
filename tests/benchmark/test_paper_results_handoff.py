"""Tests for paper Results handoff exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.paper_results_handoff import (
    PAPER_RESULTS_HANDOFF_SCHEMA_VERSION,
    build_paper_results_handoff_payload,
    export_paper_results_handoff,
)
from scripts.tools import paper_results_handoff


def _write(path: Path, payload: str) -> None:
    """Write UTF-8 fixture text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _episode(
    *,
    scenario_id: str,
    seed: int,
    success: float,
    collisions: float,
    near_misses: float,
    time_to_goal_norm: float,
    snqi: float,
) -> str:
    """Build one episode JSONL line."""
    return json.dumps(
        {
            "episode_id": f"{scenario_id}-{seed}",
            "scenario_id": scenario_id,
            "seed": seed,
            "metrics": {
                "success": success,
                "collisions": collisions,
                "near_misses": near_misses,
                "time_to_goal_norm": time_to_goal_norm,
                "snqi": snqi,
            },
        }
    )


def _make_publication_bundle(bundle_dir: Path) -> None:
    """Create a minimal publication-bundle fixture for handoff tests."""
    payload = bundle_dir / "payload"
    _write(
        bundle_dir / "publication_manifest.json",
        json.dumps(
            {
                "schema_version": "benchmark-publication-bundle.v2",
                "bundle_name": "paper_campaign_publication_bundle",
                "provenance": {"repo": {"commit": "git-from-manifest"}},
            }
        ),
    )
    _write(
        payload / "reports" / "campaign_table.csv",
        "\n".join(
            [
                "planner_key,algo,planner_group,kinematics,readiness_tier,"
                "readiness_status,preflight_status,status,episodes,success_mean,"
                "collisions_mean,near_misses_mean,time_to_goal_norm_mean,snqi_mean",
                "orca,orca,core,differential_drive,baseline-ready,native,ok,ok,2,"
                "0.5000,0.5000,1.0000,0.5000,-0.1000",
                "ppo,ppo,experimental,differential_drive,experimental,native,ok,ok,2,"
                "1.0000,0.0000,0.5000,0.3000,-0.2000",
            ]
        )
        + "\n",
    )
    _write(
        payload / "reports" / "seed_variability_by_scenario.json",
        json.dumps(
            {
                "campaign_id": "paper_campaign",
                "rows": [
                    {
                        "provenance": {
                            "campaign_id": "paper_campaign",
                            "config_hash": "cfg-123",
                            "git_hash": "git-123",
                            "seed_policy": {
                                "mode": "seed-set",
                                "seed_set": "eval",
                                "resolved_seeds": [111, 112],
                            },
                        }
                    }
                ],
            }
        ),
    )
    _write(
        payload / "runs" / "orca__differential_drive" / "episodes.jsonl",
        "\n".join(
            [
                _episode(
                    scenario_id="s1",
                    seed=111,
                    success=1.0,
                    collisions=0.0,
                    near_misses=2.0,
                    time_to_goal_norm=0.4,
                    snqi=-0.2,
                ),
                _episode(
                    scenario_id="s2",
                    seed=112,
                    success=0.0,
                    collisions=1.0,
                    near_misses=0.0,
                    time_to_goal_norm=0.6,
                    snqi=0.0,
                ),
            ]
        )
        + "\n",
    )
    _write(
        payload / "runs" / "ppo__differential_drive" / "episodes.jsonl",
        "\n".join(
            [
                _episode(
                    scenario_id="s1",
                    seed=111,
                    success=1.0,
                    collisions=0.0,
                    near_misses=1.0,
                    time_to_goal_norm=0.2,
                    snqi=-0.1,
                ),
                _episode(
                    scenario_id="s2",
                    seed=112,
                    success=1.0,
                    collisions=0.0,
                    near_misses=0.0,
                    time_to_goal_norm=0.4,
                    snqi=-0.3,
                ),
            ]
        )
        + "\n",
    )


def test_build_paper_results_handoff_payload_adds_interval_metadata(tmp_path: Path) -> None:
    """Builder should emit planner-summary rows with CI method metadata."""
    bundle_dir = tmp_path / "paper_campaign_publication_bundle"
    _make_publication_bundle(bundle_dir)

    payload = build_paper_results_handoff_payload(
        bundle_dir,
        confidence_settings={
            "method": "bootstrap_mean_over_seed_means",
            "confidence": 0.95,
            "bootstrap_samples": 0,
            "bootstrap_seed": 123,
        },
    )

    assert payload["schema_version"] == PAPER_RESULTS_HANDOFF_SCHEMA_VERSION
    assert payload["campaign_id"] == "paper_campaign"
    assert payload["row_count"] == 2
    assert payload["metrics"] == [
        "success",
        "collisions",
        "near_misses",
        "time_to_goal_norm",
        "snqi",
    ]
    orca = payload["rows"][0]
    assert orca["planner_key"] == "orca"
    assert orca["readiness_tier"] == "baseline-ready"
    assert orca["episode_count"] == 2
    assert orca["seed_count"] == 2
    assert orca["seed_list"] == [111, 112]
    assert orca["confidence_method"] == "bootstrap_mean_over_seed_means"
    assert orca["bootstrap_samples"] == 0
    assert orca["config_hash"] == "cfg-123"
    assert orca["git_hash"] == "git-123"
    assert orca["success_mean"] == pytest.approx(0.5)
    assert orca["success_ci_low"] == pytest.approx(0.5)
    assert orca["success_ci_high"] == pytest.approx(0.5)
    assert orca["success_source_table_mean"] == pytest.approx(0.5)
    assert orca["near_misses_mean"] == pytest.approx(1.0)


def test_export_paper_results_handoff_writes_json_and_csv(tmp_path: Path) -> None:
    """Exporter should write compact JSON and CSV handoff artifacts."""
    bundle_dir = tmp_path / "paper_campaign_publication_bundle"
    _make_publication_bundle(bundle_dir)
    output_dir = tmp_path / "handoff"

    result = export_paper_results_handoff(
        bundle_dir,
        output_dir=output_dir,
        confidence_settings={"bootstrap_samples": 0},
    )

    assert result.row_count == 2
    assert result.json_path.exists()
    assert result.csv_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PAPER_RESULTS_HANDOFF_SCHEMA_VERSION

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert "success_ci_low" in (reader.fieldnames or [])
        assert "confidence_method" in (reader.fieldnames or [])
        first = next(reader)
    assert first["planner_key"] == "orca"
    assert first["seed_list"] == "111,112"


def test_paper_results_handoff_cli_fails_closed_for_missing_episodes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should fail closed when the source has no episode JSONL inputs."""
    bundle_dir = tmp_path / "empty_publication_bundle"
    _write(bundle_dir / "publication_manifest.json", "{}")
    (bundle_dir / "payload" / "runs").mkdir(parents=True)

    exit_code = paper_results_handoff.main(["--source", str(bundle_dir)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "No episode JSONL files found" in captured.err


def test_paper_results_handoff_cli_writes_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI should emit a JSON summary for generated handoff paths."""
    bundle_dir = tmp_path / "paper_campaign_publication_bundle"
    _make_publication_bundle(bundle_dir)
    output_dir = tmp_path / "handoff"

    exit_code = paper_results_handoff.main(
        [
            "--source",
            str(bundle_dir),
            "--out-dir",
            str(output_dir),
            "--bootstrap-samples",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    summary = json.loads(captured.out)
    assert summary["row_count"] == 2
    assert Path(summary["json_path"]).exists()
    assert Path(summary["csv_path"]).exists()


def test_canonical_handoff_matches_frozen_campaign_table_when_available() -> None:
    """Local canonical artifact should round-trip planner means when present."""
    source = Path(
        "output/benchmarks/publication/"
        "paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle"
    )
    if not source.exists():
        pytest.skip("canonical March 18 publication bundle is not present locally")

    payload = build_paper_results_handoff_payload(
        source,
        confidence_settings={"bootstrap_samples": 0},
    )
    rows = {row["planner_key"]: row for row in payload["rows"]}

    assert (
        payload["campaign_id"]
        == "paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407"
    )
    assert rows["orca"]["success_mean"] == pytest.approx(
        rows["orca"]["success_source_table_mean"],
        abs=5e-5,
    )
    assert rows["ppo"]["collisions_mean"] == pytest.approx(
        rows["ppo"]["collisions_source_table_mean"],
        abs=5e-5,
    )
