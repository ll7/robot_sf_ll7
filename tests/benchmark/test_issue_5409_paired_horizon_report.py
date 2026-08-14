"""Tests for the fail-closed issue #5409 paired horizon handoff."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.benchmark.build_issue_5409_paired_horizon_report import analyze_pair

if TYPE_CHECKING:
    from pathlib import Path


PLANNERS = ("goal", "orca")
SCENARIOS = ("s1", "s2")
SEEDS = (1, 2)


def _write_json(path: Path, payload: dict) -> None:
    """Write a fixture JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _campaign_root(
    root: Path,
    *,
    role: str,
    execution_mode: str = "native",
    missing_key: tuple[str, str, int] | None = None,
) -> None:
    """Create a compact camera-ready campaign fixture."""
    horizon = 500 if role == "h500" else 600
    campaign_id = f"issue5409_horizon_ablation_{role}"
    _write_json(
        root / "campaign_manifest.json",
        {
            "schema_version": "benchmark-camera-ready-campaign.v1",
            "campaign_id": campaign_id,
            "scenario_matrix": "configs/scenarios/example.yaml",
            "scenario_matrix_hash": "fixture-hash",
            "seed_policy": {"resolved_seeds": list(SEEDS)},
            "git": {"commit": "a" * 40},
            "config_hash": f"config-{role}",
            "comparability_mapping_hash": "mapping-hash",
            "observation_noise_hash": "noise-hash",
        },
    )
    _write_json(
        root / "preflight" / "validate_config.json",
        {
            "scenario_count": len(SCENARIOS),
            "scenario_candidates": {"resolved": list(SCENARIOS)},
        },
    )
    _write_json(
        root / "preflight" / "checkpoint_staging.json",
        {"status": "ok", "submit_safe": True, "checked": 1, "resolved": 1},
    )
    _write_json(
        root / "reports" / "comparability_matrix.json",
        {"mapping_hash": "mapping-hash"},
    )
    _write_json(
        root / "reports" / "amv_coverage_summary.json",
        {"scenario_count": len(SCENARIOS), "status": "warn"},
    )
    _write_json(
        root / "reports" / "matrix_summary.json",
        {
            "rows": [
                {
                    "planner_key": planner,
                    "scenario_count": len(SCENARIOS),
                    "scenario_matrix_hash": "fixture-hash",
                    "resolved_seeds": list(SEEDS),
                    "horizon": horizon,
                    "config_hash": f"config-{role}",
                }
                for planner in PLANNERS
            ]
        },
    )
    _write_json(
        root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": campaign_id, "scenario_matrix_hash": "fixture-hash"},
            "runs": [
                {
                    "planner": {"key": planner},
                    "summary": {
                        "benchmark_availability": {
                            "execution_mode": execution_mode,
                            "readiness_status": execution_mode,
                            "availability_status": "available"
                            if execution_mode in {"native", "adapter"}
                            else "unavailable",
                            "benchmark_success": execution_mode in {"native", "adapter"},
                        },
                        "failed_jobs": 0,
                        "skipped_jobs": 0,
                    },
                }
                for planner in PLANNERS
            ],
        },
    )
    (root / "reports" / "campaign_table.csv").parent.mkdir(parents=True, exist_ok=True)
    (root / "reports" / "campaign_table.csv").write_text("planner_key\n", encoding="utf-8")

    for planner in PLANNERS:
        run_dir = root / "runs" / f"{planner}__differential_drive"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "summary.json",
            {
                "benchmark_availability": {
                    "execution_mode": execution_mode,
                    "readiness_status": execution_mode,
                    "availability_status": "available"
                    if execution_mode in {"native", "adapter"}
                    else "unavailable",
                    "benchmark_success": execution_mode in {"native", "adapter"},
                },
                "failed_jobs": 0,
                "skipped_jobs": 0,
                "algorithm_metadata_contract": {
                    "planner_kinematics": {"execution_mode": execution_mode}
                },
            },
        )
        rows: list[dict] = []
        for scenario in SCENARIOS:
            for seed in SEEDS:
                key = (planner, scenario, seed)
                if key == missing_key:
                    continue
                base = float(seed + (0 if scenario == "s1" else 1)) / 10.0
                rows.append(
                    {
                        "scenario_id": scenario,
                        "seed": seed,
                        "status": "success",
                        "metrics": {
                            "success": 1.0,
                            "collisions": 0.0,
                            "near_misses": base,
                            "time_to_goal_norm": base + 0.1,
                            "snqi": base + 0.2,
                        },
                        "scenario_params": {
                            "run_horizon": horizon,
                            "metadata": {"archetype": "fixture_family"},
                        },
                    }
                )
        (run_dir / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def _fixture_pair(tmp_path: Path, **kwargs: object) -> tuple[Path, Path, Path]:
    """Create both arms and an output directory."""
    h500 = tmp_path / "h500"
    h600 = tmp_path / "h600"
    output = tmp_path / "paired"
    _campaign_root(h500, role="h500", **kwargs)
    _campaign_root(h600, role="h600", **kwargs)
    return h500, h600, output


def _analyze(
    h500: Path,
    h600: Path,
    output: Path,
    *,
    expected_campaign_ids: tuple[str, str] | None = None,
) -> dict:
    """Run the fixture analysis with a small declared denominator."""
    return analyze_pair(
        h500,
        h600,
        output_dir=output,
        h500_config=None,
        h600_config=None,
        expected_planners=PLANNERS,
        expected_scenarios=SCENARIOS,
        expected_seeds=SEEDS,
        expected_rows_per_arm=8,
        expected_scenario_count=2,
        expected_scenario_matrix_hash="fixture-hash",
        expected_campaign_ids=expected_campaign_ids,
        validate_config_pair=False,
        bootstrap_samples=100,
    )


def test_valid_pair_emits_deltas_and_seed_uncertainty(tmp_path: Path) -> None:
    """A complete native pair produces all required numeric artifacts."""
    h500, h600, output = _fixture_pair(tmp_path)

    result = _analyze(h500, h600, output)

    assert result["status"] == "ready"
    completeness = json.loads((output / "matched_key_completeness.json").read_text())
    deltas = json.loads((output / "paired_horizon_deltas.json").read_text())
    uncertainty = json.loads((output / "paired_uncertainty_summary.json").read_text())
    assert completeness["benchmark_success_allowed"] is True
    assert len(deltas["rows"]) == 8
    assert len(deltas["planner_point_estimates"]) == 2
    assert len(uncertainty["planner_rows"]) == 2
    assert len(uncertainty["scenario_family_rows"]) == 2
    assert deltas["rows"][0]["metrics"]["near_misses"]["delta_h600_minus_h500"] == 0.0
    assert uncertainty["planner_rows"][0]["metrics"]["snqi"]["seed_count"] == 2


def test_reviewed_rerun_ids_and_enforced_staging_receipts_are_supported(
    tmp_path: Path,
) -> None:
    """A reviewed rerun suffix and current staged gate remain provenance-safe."""
    h500, h600, output = _fixture_pair(tmp_path)
    campaign_ids = (
        "issue5409_horizon_ablation_rerun1_h500_20260814",
        "issue5409_horizon_ablation_rerun1_h600_20260814",
    )
    for root, campaign_id in zip((h500, h600), campaign_ids, strict=True):
        manifest_path = root / "campaign_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["campaign_id"] = campaign_id
        _write_json(manifest_path, manifest)

        summary_path = root / "reports" / "campaign_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["campaign"]["campaign_id"] = campaign_id
        _write_json(summary_path, summary)

        checkpoint_path = root / "preflight" / "checkpoint_staging.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint.pop("status")
        checkpoint.update({"mode": "enforced_staged", "stage": True})
        _write_json(checkpoint_path, checkpoint)

    result = _analyze(h500, h600, output, expected_campaign_ids=campaign_ids)

    assert result["status"] == "ready"
    completeness = json.loads((output / "matched_key_completeness.json").read_text())
    assert completeness["expected"]["campaign_ids"] == {
        "h500": campaign_ids[0],
        "h600": campaign_ids[1],
    }


def test_missing_key_blocks_without_partial_numeric_output(tmp_path: Path) -> None:
    """A missing identity blocks both numeric artifacts."""
    h500, h600, output = _fixture_pair(
        tmp_path,
        missing_key=("orca", "s2", 2),
    )

    result = _analyze(h500, h600, output)

    assert result["status"] == "blocked"
    deltas = json.loads((output / "paired_horizon_deltas.json").read_text())
    uncertainty = json.loads((output / "paired_uncertainty_summary.json").read_text())
    assert deltas["rows"] == []
    assert uncertainty["planner_rows"] == []
    assert any("missing" in blocker for blocker in result["blockers"])


def test_fallback_execution_is_not_treated_as_success(tmp_path: Path) -> None:
    """Fallback/degraded execution remains fail-closed even with complete keys."""
    h500, h600, output = _fixture_pair(tmp_path)
    for summary_path in h600.glob("runs/*/summary.json"):
        payload = json.loads(summary_path.read_text())
        payload["benchmark_availability"] = {
            "execution_mode": "fallback",
            "readiness_status": "fallback",
            "availability_status": "unavailable",
            "benchmark_success": False,
        }
        payload["algorithm_metadata_contract"]["planner_kinematics"]["execution_mode"] = "fallback"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _analyze(h500, h600, output)

    assert result["status"] == "blocked"
    assert any("fallback" in blocker for blocker in result["blockers"])
