"""Tests for the diagnostic comparison report builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.training.compare_density_curriculum_results import main


def _manifest(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "density_curriculum_comparison.v1",
        "issue": "ll7/robot_sf_ll7#4018",
        "claim_boundary": "diagnostic harness only; no benchmark training-result claim",
        "dry_run": False,
        "curriculum": {
            "path": str(tmp_path / "curriculum.yaml"),
            "policy_id": "mock_curriculum_policy",
            "total_timesteps": 96,
            "density_curriculum_enabled": True,
        },
        "baseline": {
            "path": str(tmp_path / "baseline.yaml"),
            "policy_id": "mock_baseline_policy",
            "total_timesteps": 96,
            "density_curriculum_enabled": False,
        },
        "artifacts": {
            "curriculum_checkpoint": str(tmp_path / "mock_curriculum_policy.zip"),
            "baseline_checkpoint": str(tmp_path / "mock_baseline_policy.zip"),
        },
    }


def _policy_data() -> dict[str, object]:
    return {
        "metrics": {
            "success_rate": {"mean": 0.8},
            "collision_rate": {"mean": 0.1},
            "eval_episode_return": {"mean": 25.5},
            "timesteps_to_convergence": {"mean": 80.0},
            "total_timesteps_executed": {"mean": 96.0},
        }
    }


def _run_manifest_data() -> dict[str, object]:
    return {"wall_clock_hours": 0.25}


def test_comparison_report_builder_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report builder reads completed runs and writes MD and JSON reports."""
    # Write curriculum yaml
    curr_yaml = {
        "density_curriculum": {
            "enabled": True,
            "advance_rule": "timestep",
            "stages": [
                {"id": "sparse", "until_timesteps": 32, "density_m2": 0.04},
                {"id": "dense", "until_timesteps": None, "density_m2": 0.12},
            ],
        }
    }
    Path(tmp_path / "curriculum.yaml").write_text(yaml.safe_dump(curr_yaml), encoding="utf-8")
    Path(tmp_path / "baseline.yaml").write_text(
        "density_curriculum:\n  enabled: false\n", encoding="utf-8"
    )

    # Write manifest
    manifest_data = _manifest(tmp_path)
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Write mock policy JSONs beside mock checkpoints
    Path(tmp_path / "mock_curriculum_policy.json").write_text(
        json.dumps(_policy_data()), encoding="utf-8"
    )
    Path(tmp_path / "mock_baseline_policy.json").write_text(
        json.dumps(_policy_data()), encoding="utf-8"
    )

    # Mock the directory structure for latest runs manifest
    runs_dir = tmp_path / "output/benchmarks/ppo_imitation/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    Path(runs_dir / "mock_curriculum_policy_2026.json").write_text(
        json.dumps(_run_manifest_data()), encoding="utf-8"
    )
    Path(runs_dir / "mock_baseline_policy_2026.json").write_text(
        json.dumps(_run_manifest_data()), encoding="utf-8"
    )

    # Monkeypatch find_run_manifest to resolve against tmp_path
    from scripts.training import compare_density_curriculum_results

    def mock_find_run_manifest(policy_id: str) -> Path | None:
        candidates = list(runs_dir.glob(f"{policy_id}_*.json"))
        return candidates[0] if candidates else None

    monkeypatch.setattr(
        compare_density_curriculum_results, "find_run_manifest", mock_find_run_manifest
    )

    # Run comparison report builder
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_density_curriculum_results.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert main() == 0

    # Verify output reports
    report_md = tmp_path / "comparison_report.md"
    report_json = tmp_path / "comparison_report.json"

    assert report_md.exists()
    assert report_json.exists()

    md_text = report_md.read_text(encoding="utf-8")
    assert "mock_curriculum_policy" in md_text
    assert "mock_baseline_policy" in md_text
    assert "dense" in md_text

    json_data = json.loads(report_json.read_text(encoding="utf-8"))
    assert json_data["schema_version"] == "density_curriculum_comparison_report.v1"
    assert json_data["curriculum"]["final_stage_reached"] == "dense"
    assert json_data["baseline"]["final_stage_reached"] == "N/A (Disabled)"
