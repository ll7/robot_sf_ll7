"""Tests for the issue #2011 AMV actuation sensitivity sweep CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from scripts.tools import run_amv_actuation_sensitivity_sweep as sweep

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml"


def test_issue_2011_sweep_manifest_materializes_diagnostic_configs(tmp_path: Path) -> None:
    """The sweep manifest should create one concrete camera-ready config per variant."""
    exit_code = sweep.main(["--manifest", str(MANIFEST_PATH), "--output", str(tmp_path)])

    assert exit_code == 0
    resolved = json.loads((tmp_path / "resolved_sweep_manifest.json").read_text(encoding="utf-8"))
    assert resolved["claim_boundary"] == "diagnostic-only"
    assert len(resolved["variants"]) == 12
    assert resolved["variability_sampling"]["mode"] == "fixed-variants"
    assert resolved["sampled_parameter_summary"]["row_count"] == 0
    assert {(entry["field_group"], entry["level"]) for entry in resolved["variants"]} >= {
        ("longitudinal_proxy", "low"),
        ("longitudinal_proxy", "nominal"),
        ("longitudinal_proxy", "high"),
        ("update_rate_synthetic", "low"),
        ("update_rate_synthetic", "nominal"),
        ("update_rate_synthetic", "high"),
    }

    low_update = yaml.safe_load(
        (tmp_path / "generated_configs" / "update_rate_synthetic_low.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert low_update["synthetic_actuation_profile"]["update_mode"] == "2.5hz-hold"
    assert (
        low_update["synthetic_actuation_profile"]["variability_distribution"]["schema_version"]
        == "synthetic-actuation-variability-distribution.v1"
    )
    assert "variability_sample" not in low_update["synthetic_actuation_profile"]
    assert low_update["latency_stress_profile"]["planner_update_mode"] == "hold-last"
    assert low_update["latency_stress_profile"]["planner_update_period_steps"] == 4
    assert low_update["paper_facing"] is False
    assert low_update["scenario_candidates"] == [
        "classic_cross_trap_high",
        "francis2023_intersection_wait",
    ]
    assert low_update["seed_policy"] == {"mode": "fixed-list", "seeds": [111]}
    assert [planner["key"] for planner in low_update["planners"]] == ["goal", "social_force"]


def test_issue_3284_sweep_materializes_seeded_variability_samples(tmp_path: Path) -> None:
    """The opt-in variability sweep should produce deterministic sampled profile configs."""
    output_a = tmp_path / "sampled_a"
    output_b = tmp_path / "sampled_b"
    args = [
        "--manifest",
        str(MANIFEST_PATH),
        "--sampling-mode",
        "variability-sweep",
        "--sampling-seed",
        "17",
    ]

    assert sweep.main([*args, "--output", str(output_a)]) == 0
    assert sweep.main([*args, "--output", str(output_b)]) == 0

    resolved = json.loads((output_a / "resolved_sweep_manifest.json").read_text(encoding="utf-8"))
    assert resolved["variability_sampling"] == {
        "mode": "variability-sweep",
        "seed": 17,
        "sample_count": 3,
    }
    assert len(resolved["variants"]) == 3
    assert resolved["sampled_parameter_summary"]["row_count"] == 3

    first_profile = yaml.safe_load(
        (output_a / "generated_configs" / "variability_sample_000.yaml").read_text(encoding="utf-8")
    )["synthetic_actuation_profile"]
    repeated_profile = yaml.safe_load(
        (output_b / "generated_configs" / "variability_sample_000.yaml").read_text(encoding="utf-8")
    )["synthetic_actuation_profile"]
    assert first_profile == repeated_profile
    assert first_profile["variability_sample"]["sampling_seed"] == 17
    assert first_profile["variability_sample"]["sample_id"] == "sample-000"
    assert 2.425 <= first_profile["max_linear_accel_m_s2"] <= 4.625
    assert 3.210 <= first_profile["max_linear_decel_m_s2"] <= 3.842
    assert 0.8 <= first_profile["max_yaw_rate_rad_s"] <= 1.6
    assert 2.0 <= first_profile["max_angular_accel_rad_s2"] <= 6.0

    cfg = load_campaign_config(output_a / "generated_configs" / "variability_sample_000.yaml")
    assert cfg.synthetic_actuation_profile is not None
    assert cfg.synthetic_actuation_profile.variability_sample is not None
    assert cfg.synthetic_actuation_profile.variability_sample["sample_id"] == "sample-000"

    assert (output_a / "reports" / "sampled_parameter_summary.json").exists()
    assert (output_a / "reports" / "sampled_parameter_summary.csv").exists()
    md_text = (output_a / "reports" / "sampled_parameter_summary.md").read_text(encoding="utf-8")
    assert "diagnostic-only" in md_text
    assert "hardware-calibrated AMV evidence" in md_text


def test_issue_3284_aggregate_carries_sample_metadata(tmp_path: Path) -> None:
    """Aggregate outputs should preserve sampled-parameter summaries for sampled variants."""
    materialized_dir = tmp_path / "materialized"
    sweep.main(
        [
            "--manifest",
            str(MANIFEST_PATH),
            "--output",
            str(materialized_dir),
            "--sampling-mode",
            "variability-sweep",
            "--sampling-seed",
            "17",
        ]
    )
    resolved = json.loads(
        (materialized_dir / "resolved_sweep_manifest.json").read_text(encoding="utf-8")
    )
    entries = resolved["variants"]
    first_variant = entries[0]["variant_name"]
    roots = [_write_campaign_fixture(tmp_path, first_variant, 1.0, 0.0)]

    rows = sweep.aggregate_campaigns(
        output_dir=tmp_path / "aggregated",
        entries=entries,
        campaign_roots=roots,
    )

    assert rows[0]["sampling_mode"] == "variability-sweep"
    assert rows[0]["sample_id"] == "sample-000"
    assert "max_linear_accel_m_s2" in rows[0]["sampled_parameters"]
    summary = json.loads(
        (tmp_path / "aggregated" / "reports" / "effect_size_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["sampled_parameter_summary"]["row_count"] == 3
    assert "synthetic/provisional" in (
        tmp_path / "aggregated" / "reports" / "effect_size_summary.md"
    ).read_text(encoding="utf-8")


def test_issue_2011_sweep_aggregates_effect_sizes_from_campaign_roots(tmp_path: Path) -> None:
    """Fixture campaign roots should compile into effect-size tables and an SVG figure."""
    materialized_dir = tmp_path / "materialized"
    sweep.main(["--manifest", str(MANIFEST_PATH), "--output", str(materialized_dir)])
    entries = json.loads(
        (materialized_dir / "resolved_sweep_manifest.json").read_text(encoding="utf-8")
    )["variants"]

    roots = [
        _write_campaign_fixture(tmp_path, "longitudinal_proxy_nominal", 1.0, 0.0),
        _write_campaign_fixture(tmp_path, "longitudinal_proxy_low", 0.5, 1.0),
        _write_campaign_fixture(tmp_path, "longitudinal_proxy_high", 1.0, 0.0),
    ]
    rows = sweep.aggregate_campaigns(
        output_dir=tmp_path / "aggregated",
        entries=entries,
        campaign_roots=roots,
    )

    low_row = next(
        row
        for row in rows
        if row["variant_name"] == "longitudinal_proxy_low"
        and row["planner_key"] == "goal"
        and row["scenario_family"] == "crossing"
    )
    assert low_row["episodes"] == 2
    assert low_row["campaign_status"] == "success"
    assert low_row["campaign_benchmark_success"] is True
    assert low_row["campaign_exit_code"] == 0
    assert low_row["success_mean"] == "0.500000"
    assert low_row["success_delta_vs_nominal"] == "-0.500000"
    assert low_row["collisions_delta_vs_nominal"] == "1.000000"
    assert (tmp_path / "aggregated" / "reports" / "effect_size_summary.csv").exists()
    assert (tmp_path / "aggregated" / "figures" / "outcome_sensitivity.svg").exists()
    assert "diagnostic-only" in (
        tmp_path / "aggregated" / "reports" / "effect_size_summary.md"
    ).read_text(encoding="utf-8")
    assert "angular-acceleration" not in (
        tmp_path / "aggregated" / "reports" / "effect_size_summary.md"
    ).read_text(encoding="utf-8")


def test_issue_2011_sweep_labels_unavailable_campaign_roots(tmp_path: Path) -> None:
    """Unavailable campaign roots should remain visibly labeled in effect-size outputs."""
    materialized_dir = tmp_path / "materialized"
    sweep.main(["--manifest", str(MANIFEST_PATH), "--output", str(materialized_dir)])
    entries = json.loads(
        (materialized_dir / "resolved_sweep_manifest.json").read_text(encoding="utf-8")
    )["variants"]

    roots = [
        _write_campaign_fixture(
            tmp_path,
            "latency_synthetic_medium",
            0.0,
            1.0,
            status="accepted_unavailable_only",
            benchmark_success=False,
            exit_code=3,
            status_reason="latency metrics are provenance-only",
        )
    ]

    rows = sweep.aggregate_campaigns(
        output_dir=tmp_path / "aggregated",
        entries=entries,
        campaign_roots=roots,
    )

    assert rows[0]["campaign_status"] == "accepted_unavailable_only"
    assert rows[0]["campaign_benchmark_success"] is False
    assert rows[0]["campaign_exit_code"] == 3
    assert rows[0]["campaign_status_reason"] == "latency metrics are provenance-only"
    csv_text = (tmp_path / "aggregated" / "reports" / "effect_size_summary.csv").read_text(
        encoding="utf-8"
    )
    md_text = (tmp_path / "aggregated" / "reports" / "effect_size_summary.md").read_text(
        encoding="utf-8"
    )
    assert "campaign_status" in csv_text
    assert "accepted_unavailable_only" in csv_text
    assert "\r\n" not in csv_text
    assert "accepted_unavailable_only" in md_text


def test_issue_2011_sweep_aggregate_cli_labels_unavailable_campaign_roots(
    tmp_path: Path,
) -> None:
    """Aggregate CLI mode should preserve non-success campaign status in outputs."""
    root = _write_campaign_fixture(
        tmp_path,
        "latency_synthetic_medium",
        0.0,
        1.0,
        status="accepted_unavailable_only",
        benchmark_success=False,
        exit_code=3,
        status_reason="latency metrics are provenance-only",
    )

    exit_code = sweep.main(
        [
            "--manifest",
            str(MANIFEST_PATH),
            "--output",
            str(tmp_path / "aggregated"),
            "--mode",
            "aggregate",
            "--campaign-root",
            str(root),
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (tmp_path / "aggregated" / "reports" / "effect_size_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["rows"][0]["campaign_status"] == "accepted_unavailable_only"
    assert summary["rows"][0]["campaign_benchmark_success"] is False


def test_issue_2011_sweep_aggregate_requires_campaign_roots(tmp_path: Path) -> None:
    """Aggregate mode should fail closed when no campaign roots are supplied."""
    materialized_dir = tmp_path / "materialized"
    sweep.main(["--manifest", str(MANIFEST_PATH), "--output", str(materialized_dir)])
    entries = json.loads(
        (materialized_dir / "resolved_sweep_manifest.json").read_text(encoding="utf-8")
    )["variants"]

    with pytest.raises(ValueError, match="At least one campaign root"):
        sweep.aggregate_campaigns(
            output_dir=tmp_path / "aggregated",
            entries=entries,
            campaign_roots=[],
        )


def test_issue_2011_sweep_aggregate_rejects_unmatched_campaign_roots(tmp_path: Path) -> None:
    """Aggregate mode should not silently accept roots without matching summaries."""
    materialized_dir = tmp_path / "materialized"
    sweep.main(["--manifest", str(MANIFEST_PATH), "--output", str(materialized_dir)])
    entries = json.loads(
        (materialized_dir / "resolved_sweep_manifest.json").read_text(encoding="utf-8")
    )["variants"]
    empty_root = tmp_path / "empty_campaign"
    empty_root.mkdir()

    with pytest.raises(ValueError, match="No valid campaign summaries"):
        sweep.aggregate_campaigns(
            output_dir=tmp_path / "aggregated",
            entries=entries,
            campaign_roots=[empty_root],
        )


def _write_campaign_fixture(
    tmp_path: Path,
    variant_name: str,
    success: float,
    collisions: float,
    *,
    status: str = "success",
    benchmark_success: bool = True,
    exit_code: int = 0,
    status_reason: str = "",
) -> Path:
    """Create a minimal campaign root with summary and episode JSONL files."""
    root = tmp_path / variant_name
    reports = root / "reports"
    runs = root / "runs" / "goal"
    reports.mkdir(parents=True)
    runs.mkdir(parents=True)
    episodes_path = runs / "episodes.jsonl"
    episodes_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "scenario_id": "classic_cross_trap_high",
                    "scenario_params": {"scenario_family": "crossing"},
                    "metrics": {
                        "success": success,
                        "collisions": collisions,
                        "near_misses": collisions,
                        "time_to_goal_norm": 1.0,
                        "command_clip_fraction": 0.25,
                        "yaw_rate_saturation_fraction": 0.5,
                        "signed_braking_peak_m_s2": -1.5,
                    },
                }
            )
            for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": status,
                    "benchmark_success": benchmark_success,
                    "exit_code": exit_code,
                    "status_reason": status_reason,
                    "synthetic_actuation_profile": {"name": variant_name},
                },
                "runs": [
                    {
                        "planner": {"key": "goal", "algo": "goal"},
                        "episodes_path": str(episodes_path),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root
