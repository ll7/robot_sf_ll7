"""Tests for the Package A transfer-report renderer."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from scripts.tools.build_package_a_transfer_report import (
    _surface_families,
    _table_rows,
    main,
    render_report,
)
from scripts.tools.campaign_result_store import write_result_store

REPO_ROOT = Path(__file__).resolve().parents[2]
READINESS_MANIFEST = REPO_ROOT / "configs" / "benchmarks" / "issue_3078_package_a_readiness.yaml"
PARTITION_MANIFEST = (
    REPO_ROOT / "configs" / "benchmarks" / "issue_2128_heldout_family_transfer_partitions.yaml"
)
PARQUET_ENGINES = ("pyarrow", "fastparquet", "duckdb")


def _skip_without_parquet_engine() -> None:
    if not any(importlib.util.find_spec(engine) for engine in PARQUET_ENGINES):
        pytest.skip("requires parquet engine: pyarrow, fastparquet, duckdb")


def _seed_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "seed_sufficiency_analysis.v1",
                "headline_rank_stability_contract": {
                    "label": "diagnostic_only",
                    "promotion_allowed": False,
                },
                "planner_rank_stability": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _result_store(path: Path) -> Path:
    write_result_store(
        path,
        [
            {
                "run_id": "run-a",
                "episode_id": "run-a-001",
                "planner": "orca",
                "scenario_id": "blind-corner",
                "scenario_family": "francis2023_blind_corner",
                "seed": 111,
                "row_status": "native",
                "artifact_uri": "wandb://robot-sf/run-a/episodes/run-a-001.jsonl",
                "artifact_sha256": "a" * 64,
                "snqi": 0.5,
            },
            {
                "run_id": "run-a",
                "episode_id": "run-a-002",
                "planner": "orca",
                "scenario_id": "station-platform",
                "scenario_family": "classic_station_platform",
                "seed": 111,
                "row_status": "adapter",
                "artifact_uri": "wandb://robot-sf/run-a/episodes/run-a-002.jsonl",
                "artifact_sha256": "b" * 64,
                "snqi": 0.2,
            },
            {
                "run_id": "run-a",
                "episode_id": "run-a-003",
                "planner": "social_force",
                "scenario_id": "station-platform",
                "scenario_family": "classic_station_platform",
                "seed": 111,
                "row_status": "fallback",
                "artifact_uri": "wandb://robot-sf/run-a/episodes/run-a-003.jsonl",
                "artifact_sha256": "c" * 64,
                "snqi": 0.9,
            },
        ],
        study_id="issue-3078-package-a-fixture",
        command="uv run python scripts/tools/build_package_a_transfer_report.py ...",
    )
    return path


def test_renderer_writes_blocked_skeleton_without_campaign_inputs(tmp_path: Path) -> None:
    """Missing ordinary Package A evidence renders a blocked skeleton, not a claim."""
    output_dir = tmp_path / "report"
    summary = render_report(
        output_dir=output_dir,
        readiness_manifest=READINESS_MANIFEST,
        heldout_partition_manifest=PARTITION_MANIFEST,
        result_store=None,
        seed_analysis_report=None,
        repo_root=REPO_ROOT,
    )

    assert summary["classification"] == "blocked_pending_package_a_evidence"
    packet = json.loads((output_dir / "package_a_decision_packet.json").read_text())
    assert "no canonical campaign result store supplied" in packet["reasons"]
    assert (output_dir / "baseline_table.csv").read_text().splitlines() == [
        "surface,planner,scenario_family,episode_count,eligible_episode_count,row_status_counts,mean_snqi"
    ]
    claim_card = yaml.safe_load((output_dir / "claim_card.yaml").read_text())
    assert claim_card["claim_status"] == "not_reviewed"
    assert claim_card["non_promotable_row_status_counts"] == {}
    assert "checksums.sha256" in summary["files"]


def test_renderer_builds_transfer_tables_and_keeps_fallback_visible(tmp_path: Path) -> None:
    """Valid local evidence produces deterministic tables with non-promotable rows visible."""
    _skip_without_parquet_engine()

    output_dir = tmp_path / "report"
    result_store = _result_store(tmp_path / "result-store")
    seed_report = _seed_report(tmp_path / "seed_sufficiency_analysis.json")

    summary = render_report(
        output_dir=output_dir,
        readiness_manifest=READINESS_MANIFEST,
        heldout_partition_manifest=PARTITION_MANIFEST,
        result_store=result_store,
        seed_analysis_report=seed_report,
        repo_root=REPO_ROOT,
    )

    assert summary["classification"] == "diagnostic_review_ready"
    heldout_rows = list(csv.DictReader((output_dir / "heldout_family_table.csv").open()))
    assert any(row["row_status_counts"] == '{"fallback": 1}' for row in heldout_rows)
    transfer_rows = list(csv.DictReader((output_dir / "transfer_delta.csv").open()))
    assert transfer_rows == [
        {
            "planner": "orca",
            "benchmark_set_mean_snqi": "0.500000",
            "heldout_family_mean_snqi": "0.200000",
            "transfer_delta_snqi": "-0.300000",
            "claim_eligible": "false",
            "claim_boundary": "diagnostic_only_until_claim_card_review",
        }
    ]
    claim_card = yaml.safe_load((output_dir / "claim_card.yaml").read_text())
    assert claim_card["non_promotable_row_status_counts"] == {"fallback": 1}
    leakage_audit = (output_dir / "leakage_audit.md").read_text()
    assert "Benchmark-set labeling mode: `inferred_by_excluding_heldout_families`" in leakage_audit
    assert "benchmark-set families are not declared" in leakage_audit


def test_renderer_fails_closed_when_transfer_surface_families_undeclared(tmp_path: Path) -> None:
    """Partition manifests without any surface families cannot label transfer rows."""
    partition_manifest = tmp_path / "partitions.yaml"
    partition_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "test.transfer_partitions.v1",
                "benchmark_set_evaluation": {"description": "intentionally empty"},
                "heldout_family_evaluation": {"description": "intentionally empty"},
            }
        ),
        encoding="utf-8",
    )

    families = _surface_families(partition_manifest)
    with pytest.raises(ValueError, match="must declare benchmark_set_evaluation"):
        _table_rows(tmp_path / "unused-result-store", families)


def test_renderer_fails_closed_for_invalid_supplied_result_store(tmp_path: Path) -> None:
    """Supplying an invalid result store blocks rendering instead of hiding bad rows."""
    output_dir = tmp_path / "report"
    bad_store = tmp_path / "bad-store"
    bad_store.mkdir()

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--result-store",
            str(bad_store),
            "--heldout-partition-manifest",
            str(PARTITION_MANIFEST),
            "--readiness-manifest",
            str(READINESS_MANIFEST),
        ]
    )

    assert exit_code == 1
    packet = json.loads((output_dir / "package_a_decision_packet.json").read_text())
    assert packet["result_stores"][0]["ok"] is False
