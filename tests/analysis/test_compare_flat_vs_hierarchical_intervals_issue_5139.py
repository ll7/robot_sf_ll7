"""Focused tests for the retained-bundle interval comparison in issue #5139."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from scripts.analysis import compare_flat_vs_hierarchical_intervals_issue_5139 as comparison

if TYPE_CHECKING:
    from pathlib import Path


def _write_bundle(path: Path) -> None:
    fieldnames = [
        "episode_id",
        "scenario_id",
        "planner_key",
        "kinematics",
        "algo",
        "seed",
        "repeat_index",
        "success",
        "collision",
        "near_miss",
        "time_to_goal",
        "snqi",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for planner_offset, planner in enumerate(("goal", "orca")):
            for scenario_index in range(4):
                for seed in range(3):
                    collision = float((scenario_index + planner_offset) % 3 == 0)
                    writer.writerow(
                        {
                            "episode_id": f"{planner}-{scenario_index}-{seed}",
                            "scenario_id": f"scenario-{scenario_index}",
                            "planner_key": planner,
                            "kinematics": "differential_drive",
                            "algo": planner,
                            "seed": seed,
                            "repeat_index": 0,
                            "success": 1.0 - collision,
                            "collision": collision,
                            "near_miss": scenario_index + seed,
                            "time_to_goal": 10 + scenario_index * 3 + seed / 10,
                            "snqi": 0.1 * (scenario_index - seed),
                        }
                    )


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-camera-ready-campaign.v1",
                "campaign_id": "fixture-campaign",
                "config_hash": "fixture-config-hash",
                "git": {"commit": "0123456789abcdef"},
            }
        ),
        encoding="utf-8",
    )


def test_canonical_retained_bundle_has_expected_provenance_surface() -> None:
    """The named retained input must stay present and retain its expected shape."""
    records, provenance = comparison.load_retained_bundle(comparison.DEFAULT_RETAINED_BUNDLE)

    assert len(records) == provenance["row_count"] == 5760
    assert len(provenance["planner_keys"]) == 12
    assert len(provenance["scenario_ids"]) == 48
    assert provenance["seeds"] == list(range(111, 121))
    assert len(provenance["episode_table_sha256"]) == 64


def test_retained_comparison_is_deterministic_and_analysis_only(tmp_path: Path) -> None:
    """Fixed inputs and seeds must produce byte-identical, claim-bounded reports."""
    bundle = tmp_path / "seed_episode_rows.csv"
    manifest = tmp_path / "campaign_manifest.json"
    output_dir = tmp_path / "evidence"
    _write_bundle(bundle)
    _write_manifest(manifest)

    argv = [
        "--retained-bundle",
        str(bundle),
        "--retained-manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
    ]
    assert comparison.main(argv) == 0
    first_json = (output_dir / "retained_comparison_report.json").read_bytes()
    first_markdown = (output_dir / "retained_comparison_report.md").read_bytes()
    assert comparison.main(argv) == 0

    assert (output_dir / "retained_comparison_report.json").read_bytes() == first_json
    assert (output_dir / "retained_comparison_report.md").read_bytes() == first_markdown
    payload = json.loads(first_json)
    assert payload["schema_version"] == "issue_5139.retained_interval_comparison.v1"
    assert payload["evidence_status"] == "diagnostic-only (analysis-only)"
    assert payload["source_provenance"]["campaign_id"] == "fixture-campaign"
    assert payload["source_provenance"]["row_count"] == 24
    assert payload["grouping_contract"]["hierarchical_cluster"] == "scenario_id"
    assert payload["config"]["rate_interval_flat"] == "wilson"
    assert payload["config"]["rate_interval_hierarchical"] == "cluster_robust"
    assert {row["planner_key"] for row in payload["width_ratios"]} == {"goal", "orca"}
    assert "not benchmark-strength evidence" in first_markdown.decode()


def test_noncanonical_retained_bundle_requires_an_explicit_manifest(tmp_path: Path) -> None:
    """An arbitrary table cannot inherit provenance from the canonical campaign."""
    bundle = tmp_path / "seed_episode_rows.csv"
    _write_bundle(bundle)

    with pytest.raises(SystemExit, match="2"):
        comparison.main(["--retained-bundle", str(bundle)])


def test_retained_loader_fails_closed_on_missing_metric_column(tmp_path: Path) -> None:
    """A schema mismatch must stop analysis instead of silently dropping endpoints."""
    bundle = tmp_path / "seed_episode_rows.csv"
    bundle.write_text(
        "episode_id,scenario_id,planner_key,kinematics,seed,success,collision,near_miss,time_to_goal\n"
        "ep-1,scenario-1,goal,differential_drive,1,1,0,0,10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing columns: snqi"):
        comparison.load_retained_bundle(bundle)
