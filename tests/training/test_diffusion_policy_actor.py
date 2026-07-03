"""CPU smoke tests for issue #4010 diffusion-policy training artifacts."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.training.diffusion_policy import (
    SMOKE_MANIFEST_SCHEMA_VERSION,
    DiffusionPolicyTrainingSmokeConfig,
    load_smoke_config,
    run_training_smoke,
)


def _map_runner_obs() -> dict[str, object]:
    return {
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [2.0, 0.0],
            "heading": [0.0],
            "radius": [0.3],
        },
        "pedestrians": {
            "positions": [[0.7, 0.1], [1.1, -0.2]],
            "velocities": [[-0.1, 0.0], [0.0, 0.1]],
            "radii": [0.25, 0.25],
            "count": [2],
        },
        "dt": [0.1],
    }


def test_smoke_config_rejects_non_cpu_scale_values() -> None:
    """The smoke trainer validates the narrow first-slice action contract."""
    with pytest.raises(ValueError, match="training_steps"):
        DiffusionPolicyTrainingSmokeConfig(training_steps=0)
    with pytest.raises(ValueError, match="action_dim=2"):
        DiffusionPolicyTrainingSmokeConfig(action_dim=3)


def test_load_smoke_config_accepts_nested_yaml(tmp_path) -> None:
    """Tracked YAML config shape maps to the smoke-training dataclass."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 1\n"
        "  batch_size: 2\n",
        encoding="utf-8",
    )

    config = load_smoke_config(config_path)

    assert config.training_steps == 1
    assert config.batch_size == 2


def test_training_smoke_writes_checkpoint_normalizer_manifest(tmp_path) -> None:
    """CPU smoke writes the three artifacts required by the issue #4010 contract."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=3,
        max_pedestrians=2,
        artifact_prefix="unit_smoke",
    )

    artifacts = run_training_smoke(config, output_dir=tmp_path)

    assert artifacts.checkpoint_path.is_file()
    assert artifacts.normalizer_path.is_file()
    assert artifacts.manifest_path.is_file()
    normalizer = json.loads(artifacts.normalizer_path.read_text(encoding="utf-8"))
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert normalizer["schema_version"] == "diffusion_policy_normalizer.v1"
    assert normalizer["sample_count"] > 0
    assert manifest["schema_version"] == SMOKE_MANIFEST_SCHEMA_VERSION
    assert manifest["issue"] == 4010
    assert manifest["artifacts"]["checkpoint_path"] == artifacts.checkpoint_path.name
    assert manifest["artifacts"]["normalizer_path"] == artifacts.normalizer_path.name
    assert manifest["training"]["status"] == "completed"
    assert manifest["evidence_tier"] == "diagnostic-only"
    load_contract = manifest["artifacts"]["map_runner_load_contract"]
    assert load_contract["allow_untrained_smoke"] is False
    assert load_contract["requires_checkpoint_path"] is True
    assert load_contract["requires_normalizer_path"] is True
    assert (
        load_contract["algo_config_fragment"]["checkpoint_path"] == artifacts.checkpoint_path.name
    )
    assert (
        load_contract["algo_config_fragment"]["normalizer_path"] == artifacts.normalizer_path.name
    )


def test_training_script_entrypoint_writes_manifest_path(tmp_path) -> None:
    """Canonical script command writes smoke manifest for issue #4010."""
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 1\n"
        "  batch_size: 2\n"
        "  max_pedestrians: 2\n"
        "  artifact_prefix: cli_smoke\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/training/train_diffusion_policy.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    manifest_path = output_dir / "cli_smoke.manifest.json"
    assert payload == {"manifest_path": str(manifest_path)}
    assert manifest_path.exists()


def test_map_runner_loads_smoke_checkpoint_not_untrained_weights(tmp_path) -> None:
    """Map-runner can build diffusion policy from the trained smoke checkpoint."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=3,
        max_pedestrians=2,
        max_linear_speed=0.6,
        max_angular_speed=0.5,
        artifact_prefix="map_runner_smoke",
    )
    artifacts = run_training_smoke(config, output_dir=tmp_path)

    policy, meta = _build_policy(
        "diffusion_policy",
        {
            "checkpoint_path": str(artifacts.checkpoint_path),
            "normalizer_path": str(artifacts.normalizer_path),
            "deterministic": True,
            "seed": 4010,
            "max_pedestrians": config.max_pedestrians,
            "max_linear_speed": config.max_linear_speed,
            "max_angular_speed": config.max_angular_speed,
            "num_action_samples": config.num_action_samples,
            "denoising_steps": config.denoising_steps,
        },
        robot_kinematics="differential_drive",
    )

    command = policy(_map_runner_obs())
    stats = policy._planner_stats()

    assert 0.0 <= command[0] <= config.max_linear_speed
    assert -config.max_angular_speed <= command[1] <= config.max_angular_speed
    assert meta["diffusion_policy"]["allow_untrained_smoke"] is False
    assert meta["diffusion_policy"]["checkpoint_status"] == "checkpoint_loaded"
    assert meta["diffusion_policy"]["normalizer_status"] == "loaded"
    assert stats["diffusion_policy"]["allow_untrained_smoke"] is False
    assert stats["diffusion_policy"]["checkpoint_status"] == "checkpoint_loaded"
    assert stats["diffusion_policy"]["normalizer_status"] == "loaded"
    assert stats["diffusion_policy"]["evidence_tier"] == "diagnostic-only"
