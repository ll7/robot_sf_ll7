"""Tests for the Package A pipeline orchestrator (issue #3078).

Tests exercise the synthetic-fixture pathway and the partition-manifest
integration without running real benchmark campaigns.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.tools.run_package_a_pipeline import (
    PACKAGE_A_CLASSIFICATION,
    SCHEMA_VERSION,
    _build_baseline_table,
    _build_claim_card,
    _build_transfer_delta,
    _safe_float,
    _write_transfer_delta_figure,
    run_pipeline,
)


def _write_json(path: Path, payload: dict) -> None:
    """Write an indented JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _campaign_fixture(
    root: Path,
    *,
    seed_count: int,
    planner_snqi: float,
    planner_success: float = 0.9,
) -> Path:
    """Create a minimal campaign fixture directory."""
    reports = root / "reports"
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    _write_json(
        reports / "seed_variability_by_scenario.json",
        {
            "schema_version": "benchmark-seed-variability-by-scenario.v1",
            "metrics": ["success", "collisions", "snqi"],
            "rows": [
                {
                    "scenario_id": "family_a_s1",
                    "scenario_family": "family_a",
                    "planner_key": "goal",
                    "kinematics": "differential_drive",
                    "seed_count": seed_count,
                    "summary": {
                        "success": {
                            "mean": planner_success,
                            "ci_low": planner_success - 0.1,
                            "ci_high": planner_success + 0.1,
                            "ci_half_width": 0.1,
                        },
                        "collisions": {
                            "mean": 0.05,
                            "ci_low": 0.0,
                            "ci_high": 0.1,
                            "ci_half_width": 0.05,
                        },
                        "snqi": {
                            "mean": planner_snqi,
                            "ci_low": planner_snqi - 0.1,
                            "ci_high": planner_snqi + 0.1,
                            "ci_half_width": 0.1,
                        },
                    },
                },
            ],
        },
    )

    episodes = []
    for seed_index in range(seed_count):
        episodes.append(
            {
                "episode_id": f"goal-{seed_index}",
                "scenario_id": "family_a_s1",
                "scenario_family": "family_a",
                "planner_key": "goal",
                "seed": 111 + seed_index,
                "success": "1",
                "collision": "0",
                "snqi": f"{planner_snqi:.4f}",
            }
        )
    fieldnames = [
        "episode_id",
        "scenario_id",
        "scenario_family",
        "planner_key",
        "seed",
        "success",
        "collision",
        "snqi",
    ]
    with (reports / "seed_episode_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episodes)

    _write_json(reports / "statistical_sufficiency.json", {"bootstrap": {"seed": 123}})
    return root


class TestSafeFloat:
    """Tests for the _safe_float utility."""

    def test_finite_value(self) -> None:
        assert _safe_float("0.7") == 0.7
        assert _safe_float(0.7) == 0.7
        assert _safe_float("1.0") == 1.0

    def test_non_finite(self) -> None:
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("-inf")) is None

    def test_invalid(self) -> None:
        assert _safe_float("not_a_number") is None
        assert _safe_float(None) is None
        assert _safe_float("") is None


class TestBaselineTable:
    """Tests for the _build_baseline_table utility."""

    def test_empty_campaign(self, tmp_path: Path) -> None:
        rows = _build_baseline_table([])
        assert rows == []

    def test_single_campaign(self, tmp_path: Path) -> None:
        campaign = _campaign_fixture(
            tmp_path / "s5",
            seed_count=5,
            planner_snqi=0.72,
            planner_success=0.9,
        )
        rows = _build_baseline_table([campaign])
        assert len(rows) == 1
        assert rows[0]["planner_key"] == "goal"
        assert rows[0]["snqi_mean"] == pytest.approx(0.72, abs=0.001)
        assert rows[0]["success_mean"] == pytest.approx(0.9, abs=0.001)
        assert rows[0]["campaign"] == "s5"
        assert rows[0]["snqi_ci_low"] is not None


class TestTransferDelta:
    """Tests for the _build_transfer_delta utility."""

    def test_identical_surfaces(self) -> None:
        benchmark = [{"planner_key": "goal", "snqi_mean": 0.7}]
        heldout = [{"planner_key": "goal", "snqi_mean": 0.7}]
        deltas = _build_transfer_delta(benchmark, heldout)
        assert len(deltas) == 1
        assert deltas[0]["transfer_direction"] == "identical"
        assert deltas[0]["transfer_delta_snqi"] == 0.0

    def test_positive_transfer(self) -> None:
        benchmark = [{"planner_key": "goal", "snqi_mean": 0.75}]
        heldout = [{"planner_key": "goal", "snqi_mean": 0.65}]
        deltas = _build_transfer_delta(benchmark, heldout)
        assert deltas[0]["transfer_direction"] == "positive_transfer"
        assert deltas[0]["transfer_delta_snqi"] == pytest.approx(0.1, abs=0.001)

    def test_missing_heldout(self) -> None:
        benchmark = [{"planner_key": "goal", "snqi_mean": 0.7}]
        heldout = []
        deltas = _build_transfer_delta(benchmark, heldout)
        assert deltas[0]["transfer_direction"] == "incomplete"
        assert deltas[0]["transfer_delta_snqi"] is None


class TestTransferDeltaFigure:
    """Tests for the _write_transfer_delta_figure utility."""

    def test_writes_png(self, tmp_path: Path) -> None:
        deltas = [
            {"planner_key": "goal", "transfer_delta_snqi": 0.1},
            {"planner_key": "orca", "transfer_delta_snqi": None},
        ]
        path = tmp_path / "fig.png"
        _write_transfer_delta_figure(path, deltas)
        assert path.is_file()
        assert path.stat().st_size > 0


class TestClaimCard:
    """Tests for the _build_claim_card utility."""

    def test_diagnostic_card(self, tmp_path: Path) -> None:
        card = _build_claim_card(
            classification="diagnostic",
            reasons=["synthetic fixture data"],
            seed_analysis_path=tmp_path / "seed.json",
            baseline_table_path=tmp_path / "baseline.csv",
            transfer_delta_path=tmp_path / "transfer.csv",
            figure_path=tmp_path / "fig.png",
            partition_manifests=[],
            output_dir=tmp_path,
        )
        assert card["classification"] == "diagnostic"
        assert card["promotion_allowed"] is False
        assert "synthetic fixture data" in card["reasons"]
        assert card["schema_version"] == "package_a_claim_card.v1"


class TestSyntheticPipeline:
    """Full pipeline tests with synthetic fixture data (no real campaigns)."""

    def test_synthetic_pipeline_produces_artifacts(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        payload = run_pipeline(
            out,
            benchmark_families=["classic_bottleneck", "classic_cross_trap"],
            heldout_families=["classic_station_platform"],
        )

        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["classification"] == "diagnostic"

        for artifact_key in (
            "seed_sufficiency_analysis",
            "baseline_table",
            "transfer_delta",
            "transfer_delta_figure",
            "claim_card",
            "summary_markdown",
        ):
            path = Path(payload["artifacts"][artifact_key])
            assert path.is_file(), f"Missing artifact: {artifact_key}"

    def test_synthetic_pipeline_classification_vocabulary(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        payload = run_pipeline(out)
        assert set(payload["classification_vocabulary"]) == set(PACKAGE_A_CLASSIFICATION)

    def test_baseline_table_has_expected_planners(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        run_pipeline(out)
        table_path = out / "baseline_table.csv"
        with table_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        planners = {r["planner_key"] for r in rows}
        assert planners == {"goal", "social_force", "orca"}

    def test_transfer_delta_computed_for_all_planners(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        run_pipeline(out)
        delta_path = out / "transfer_delta.csv"
        with delta_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        planners = {r["planner_key"] for r in rows}
        assert planners == {"goal", "social_force", "orca"}
        for row in rows:
            assert row["transfer_direction"] in (
                "positive_transfer",
                "negative_transfer",
                "identical",
                "incomplete",
            )

    def test_output_json_roundtrips(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        run_pipeline(out)
        payload_path = out / "pipeline_output.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["classification"] == "diagnostic"
        assert "synthetic_fixture_campaigns_used" in payload["reasons"]

    def test_claim_card_includes_all_artifacts(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        run_pipeline(out)
        card = json.loads((out / "claim_card.json").read_text(encoding="utf-8"))
        assert card["classification"] == "diagnostic"
        assert card["promotion_allowed"] is False
        assert card["artifacts"]["seed_sufficiency_analysis"] is not None
        assert card["artifacts"]["baseline_table"] is not None
        assert card["artifacts"]["transfer_delta_table"] is not None
        assert card["artifacts"]["transfer_delta_figure"] is not None

    def test_markdown_summary_contains_classification(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        run_pipeline(out)
        md = (out / "package_a_summary.md").read_text(encoding="utf-8")
        assert "diagnostic" in md
        assert "#3078" in md


class TestPartitionManifestIntegration:
    """Test with a real shipped partition manifest."""

    def test_with_valid_partition_manifest(self, tmp_path: Path) -> None:
        manifest = Path("configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml")
        if not manifest.exists():
            pytest.skip("partition manifest not available in test environment")

        out = tmp_path / "output"
        payload = run_pipeline(
            out,
            partition_manifest=manifest,
        )
        assert payload["classification"] == "diagnostic"
        card = json.loads((out / "claim_card.json").read_text(encoding="utf-8"))
        assert len(card.get("partition_manifests", [])) == 1


class TestRealCampaignPipeline:
    """Pipeline tests with real fixture campaigns (not synthetic fixtures)."""

    def test_with_real_campaign_roots(self, tmp_path: Path) -> None:
        c5 = _campaign_fixture(tmp_path / "s5", seed_count=5, planner_snqi=0.72)
        out = tmp_path / "output"
        payload = run_pipeline(
            out,
            campaign_roots=[c5],
            heldout_families=["heldout_a"],
        )
        assert payload["classification"] == "diagnostic"
        assert "synthetic_fixture_heldout_used" in payload["reasons"]

        seed_path = out / "seed_sufficiency" / "seed_sufficiency_analysis.json"
        assert seed_path.is_file()
        seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
        assert seed_payload["schema_version"] == "seed_sufficiency_analysis.v1"
