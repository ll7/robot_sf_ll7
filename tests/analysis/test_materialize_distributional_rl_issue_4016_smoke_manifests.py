"""Tests for issue #4016 smoke manifest materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis.compare_distributional_rl_issue_4016 import build_report_from_config
from scripts.analysis.materialize_distributional_rl_issue_4016_smoke_manifests import (
    materialize_smoke_manifests,
)
from scripts.training.train_distributional_rl import (
    load_distributional_rl_training_config,
    run_distributional_rl_training,
)


def _write_training_config(tmp_path: Path) -> Path:
    path = tmp_path / "qr_dqn.yaml"
    path.write_text(
        f"""
policy_id: qr_dqn_issue_4016_test
algorithm: qr_dqn
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: 12
seed: 4016
device: cpu
num_envs: 1
observation:
  synthetic_observation_dim: 4
action_lattice:
  linear_values: [0.0, 0.5]
  angular_values: [-0.5, 0.5]
  max_linear_speed: 0.5
  max_angular_speed: 0.5
critic:
  hidden_sizes: [8]
  num_quantiles: 4
  target_update_interval: 4
risk_selection:
  objective: cvar_lower
  alpha: 0.5
dqn:
  replay_size: 32
  batch_size: 2
  learning_starts: 2
  train_freq: 1
  gradient_steps: 1
output_dir: {tmp_path / "model"}
""",
        encoding="utf-8",
    )
    return path


def _write_comparison_config(tmp_path: Path, summary: dict[str, object]) -> Path:
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        f"""
schema_version: issue_4016.distributional_rl_risk_comparison.v1
issue: 4016
evidence_tier: diagnostic-only
claim_boundary: risk-selection diagnostic only; not benchmark evidence
mean_manifest: {summary["mean_manifest"]}
risk_manifest: {summary["risk_manifest"]}
output_json: {tmp_path / "comparison.json"}
output_markdown: {tmp_path / "comparison.md"}
""",
        encoding="utf-8",
    )
    return config_path


def test_materializer_writes_matched_mean_and_cvar_manifests(tmp_path: Path) -> None:
    """Real trainer output can be materialized into paired diagnostic manifests."""
    config = load_distributional_rl_training_config(_write_training_config(tmp_path))
    training = run_distributional_rl_training(config, dry_run=False)

    summary = materialize_smoke_manifests(
        training_manifest_path=training["manifest_path"],
        output_dir=tmp_path / "analysis",
        observation_count=4,
    )

    mean = json.loads(Path(str(summary["mean_manifest"])).read_text(encoding="utf-8"))
    risk = json.loads(Path(str(summary["risk_manifest"])).read_text(encoding="utf-8"))
    assert mean["risk_objective"] == "mean"
    assert risk["risk_objective"] == "cvar_lower"
    assert mean["checkpoint_path"] == risk["checkpoint_path"] == training["checkpoint_path"]
    assert mean["total_timesteps"] == risk["total_timesteps"] == 12
    assert mean["fallback_or_degraded"] is False
    assert risk["fallback_or_degraded"] is False
    assert set(mean["metrics"]) >= {
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "mean_min_clearance",
        "mean_path_efficiency",
    }


def test_materialized_manifests_feed_issue_4016_comparison(tmp_path: Path) -> None:
    """Materialized manifests satisfy the existing diagnostic comparison contract."""
    config = load_distributional_rl_training_config(_write_training_config(tmp_path))
    training = run_distributional_rl_training(config, dry_run=False)
    summary = materialize_smoke_manifests(
        training_manifest_path=training["manifest_path"],
        output_dir=tmp_path / "analysis",
        observation_count=4,
    )

    report = build_report_from_config(_write_comparison_config(tmp_path, summary))

    assert report["effect"]["comparison_status"] == "valid_diagnostic"
    assert report["matched_context"]["matched_checkpoint_path"] is True
    assert report["fallback_degraded_rows"]["excluded"] == 0
    assert report["claim_boundary"] == "risk-selection diagnostic only; not benchmark evidence"


def test_materializer_rejects_degraded_training_manifest(tmp_path: Path) -> None:
    """Fallback/degraded inputs stay excluded from evidence."""
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "policy_id": "qr_dqn_issue_4016_test",
                "algorithm": "qr_dqn",
                "evidence_tier": "smoke",
                "claim_boundary": "diagnostic distributional-RL smoke",
                "seed": 4016,
                "total_timesteps": 12,
                "train_steps": 1,
                "checkpoint_path": "missing.pt",
                "action_lattice_path": "missing.json",
                "fallback_or_degraded": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fallback/degraded"):
        materialize_smoke_manifests(
            training_manifest_path=manifest_path,
            output_dir=tmp_path / "analysis",
        )
