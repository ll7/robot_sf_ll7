"""Tests for DreamerV3 RLlib training config loading and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.training.train_dreamerv3_rllib import (
    _build_ray_init_kwargs,
    _resolve_auto_overrides,
    load_run_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, content: str) -> Path:
    """Write UTF-8 YAML content to disk for test setup."""
    path.write_text(content, encoding="utf-8")
    return path


def test_load_run_config_parses_expected_defaults(tmp_path: Path):
    """Minimal valid YAML should parse into a fully-typed run config."""
    config_path = _write_yaml(
        tmp_path / "dreamer.yaml",
        """
experiment:
  run_id: smoke
  output_root: output/dreamerv3
  train_iterations: 3
  checkpoint_every: 1
  seed: 11
  log_level: WARNING
env:
  flatten_observation: true
  flatten_keys: [drive_state, rays]
  normalize_actions: true
algorithm:
  framework: torch
""",
    )

    run_config = load_run_config(config_path)

    assert run_config.experiment.run_id == "smoke"
    assert run_config.experiment.train_iterations == 3
    assert run_config.experiment.checkpoint_every == 1
    assert run_config.env.flatten_keys == ("drive_state", "rays")
    assert run_config.algorithm.framework == "torch"
    assert run_config.experiment.output_root.is_absolute()


def test_load_run_config_rejects_unknown_root_keys(tmp_path: Path):
    """Unknown top-level config keys should fail fast."""
    config_path = _write_yaml(
        tmp_path / "invalid_root.yaml",
        """
experiment:
  run_id: smoke
unknown_root: 1
""",
    )

    with pytest.raises(ValueError, match="Unknown key\\(s\\) in 'root'"):
        load_run_config(config_path)


def test_load_run_config_rejects_unknown_env_keys(tmp_path: Path):
    """Unknown env keys should fail fast to keep configs reproducible."""
    config_path = _write_yaml(
        tmp_path / "invalid_env.yaml",
        """
experiment:
  run_id: smoke
env:
  flatten_observation: true
  unknown_env_key: true
""",
    )

    with pytest.raises(ValueError, match="Unknown key\\(s\\) in 'env'"):
        load_run_config(config_path)


def test_load_run_config_rejects_non_sequence_flatten_keys(tmp_path: Path):
    """flatten_keys must be list/tuple for deterministic ordering."""
    config_path = _write_yaml(
        tmp_path / "invalid_flatten.yaml",
        """
experiment:
  run_id: smoke
env:
  flatten_observation: true
  flatten_keys: drive_state
""",
    )

    with pytest.raises(ValueError, match="env.flatten_keys"):
        load_run_config(config_path)


def test_load_run_config_parses_wandb_tracking_settings(tmp_path: Path):
    """Tracking block should parse with explicit wandb settings."""
    config_path = _write_yaml(
        tmp_path / "wandb.yaml",
        """
experiment:
  run_id: smoke
tracking:
  wandb:
    enabled: true
    project: robot_sf_dreamerv3
    group: smoke
    mode: offline
    tags: [dreamerv3, test]
    dir: output/wandb
""",
    )

    run_config = load_run_config(config_path)

    assert run_config.tracking.wandb.enabled is True
    assert run_config.tracking.wandb.project == "robot_sf_dreamerv3"
    assert run_config.tracking.wandb.group == "smoke"
    assert run_config.tracking.wandb.mode == "offline"
    assert run_config.tracking.wandb.tags == ("dreamerv3", "test")
    assert run_config.tracking.wandb.directory.is_absolute()


def test_load_run_config_rejects_unknown_wandb_keys(tmp_path: Path):
    """Unknown tracking.wandb keys should fail fast."""
    config_path = _write_yaml(
        tmp_path / "invalid_wandb.yaml",
        """
experiment:
  run_id: smoke
tracking:
  wandb:
    enabled: true
    unexpected: true
""",
    )

    with pytest.raises(ValueError, match="Unknown key\\(s\\) in 'tracking.wandb'"):
        load_run_config(config_path)


def test_load_run_config_accepts_auto_resource_placeholders(tmp_path: Path):
    """Config should accept 'auto' placeholders for runtime resource detection."""
    config_path = _write_yaml(
        tmp_path / "auto_resources.yaml",
        """
experiment:
  run_id: smoke
ray:
  num_cpus: auto
  num_gpus: auto
algorithm:
  env_runners:
    num_env_runners: auto
  resources:
    num_gpus: auto
""",
    )

    run_config = load_run_config(config_path)

    assert run_config.ray.num_cpus == "auto"
    assert run_config.ray.num_gpus == "auto"
    assert run_config.algorithm.env_runners["num_env_runners"] == "auto"
    assert run_config.algorithm.resources["num_gpus"] == "auto"


def test_auto_resource_resolution_uses_slurm_env(monkeypatch, tmp_path: Path):
    """Auto placeholders should resolve against Slurm-provided CPU/GPU allocation."""
    config_path = _write_yaml(
        tmp_path / "auto_resolution.yaml",
        """
experiment:
  run_id: smoke
ray:
  num_cpus: auto
  num_gpus: auto
algorithm:
  env_runners:
    num_env_runners: auto
  resources:
    num_gpus: auto
""",
    )
    run_config = load_run_config(config_path)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "24")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "gpu:a30:1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    ray_kwargs = _build_ray_init_kwargs(run_config)
    env_runner_payload, resources_payload, capacity = _resolve_auto_overrides(run_config)

    assert ray_kwargs["num_cpus"] == 24
    assert ray_kwargs["num_gpus"] == 1
    assert env_runner_payload["num_env_runners"] == 20
    assert resources_payload["num_gpus"] == 1
    assert capacity is not None
