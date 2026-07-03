"""Artifact smoke tests for issue #4016 QR-DQN trainer."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.baselines.distributional_rl import DistributionalRLPlanner
from scripts.training.train_distributional_rl import (
    load_distributional_rl_training_config,
    run_distributional_rl_training,
)


def _write_config(tmp_path: Path, *, total_timesteps: int = 12) -> Path:
    path = tmp_path / "qr_dqn.yaml"
    path.write_text(
        f"""
policy_id: qr_dqn_test
algorithm: qr_dqn
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: {total_timesteps}
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
output_dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )
    return path


def test_distributional_rl_dry_run_writes_manifest_without_claim(tmp_path: Path) -> None:
    """Dry-run writes provenance and keeps the benchmark claim boundary explicit."""

    config = load_distributional_rl_training_config(_write_config(tmp_path))
    result = run_distributional_rl_training(config, dry_run=True)

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["train_steps"] == 0
    assert manifest["fallback_or_degraded"] is False
    assert "not benchmark or paper-grade evidence" in manifest["claim_boundary"]
    assert Path(result["checkpoint_path"]).exists()
    assert Path(result["resolved_config_path"]).exists()
    assert Path(result["action_lattice_path"]).exists()


def test_distributional_rl_cpu_smoke_trains_and_writes_trace(tmp_path: Path) -> None:
    """CPU smoke run updates the critic and records trace rows."""

    config = load_distributional_rl_training_config(_write_config(tmp_path, total_timesteps=8))
    result = run_distributional_rl_training(config, dry_run=False)

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["train_steps"] > 0
    assert manifest["dry_run"] is False
    assert manifest["final_loss"] is not None
    assert Path(result["training_trace_path"]).read_text(encoding="utf-8").strip()


def test_distributional_rl_smoke_checkpoint_loads_runtime_adapter(tmp_path: Path) -> None:
    """Trainer output is directly loadable by the map-runner runtime adapter."""

    config = load_distributional_rl_training_config(_write_config(tmp_path, total_timesteps=8))
    result = run_distributional_rl_training(config, dry_run=False)

    planner = DistributionalRLPlanner(
        {
            "checkpoint_path": result["checkpoint_path"],
            "risk_objective": "cvar_lower",
            "risk_alpha": 0.5,
        }
    )

    action = planner.step({"goal_position": [1.0, 0.0], "robot_position": [0.0, 0.0]})
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert set(action) == {"v", "omega"}
    assert manifest["checkpoint_path"] == result["checkpoint_path"]
    assert planner.get_metadata()["evidence_tier"] == "diagnostic-only"
    assert (
        planner.diagnostics()["last_decision"]["candidate_count"]
        == config.action_lattice.action_count
    )
