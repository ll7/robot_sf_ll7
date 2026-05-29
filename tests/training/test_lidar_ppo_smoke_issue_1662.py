"""Contract tests for the issue #1662 fixed LiDAR PPO MLP smoke config."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_CONFIG = _REPO_ROOT / "configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml"
_LAUNCH_PACKET = (
    _REPO_ROOT / "configs/training/lidar/lidar_learned_policy_launch_packet_issue_1615.yaml"
)
_ELIGIBILITY_SPEC = _REPO_ROOT / "configs/training/lidar/lidar_ppo_mlp_eligibility_issue_1615.yaml"


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_issue_1662_smoke_config_materializes_ppo_lidar_mlp_gate() -> None:
    """The fixed smoke should exercise the intended issue #1615 MLP baseline."""
    config = _load_yaml(_SMOKE_CONFIG)
    packet = _load_yaml(_LAUNCH_PACKET)
    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in packet["candidate_baselines"]  # type: ignore[index]
    }
    candidate = candidates["ppo_lidar_mlp_gate_v1"]

    assert config["policy_id"] == "ppo_lidar_mlp_gate_v1_issue_1662_smoke"
    assert config["feature_extractor"] == candidate["feature_extractor"] == "mlp"
    assert config["feature_extractor_kwargs"] == {
        "ray_hidden_dims": [64, 32],
        "drive_hidden_dims": [16, 8],
        "dropout_rate": 0.1,
    }
    assert config["policy_net_arch"] == [64, 64]
    assert config["total_timesteps"] == 32000
    assert config["seeds"] == [123]
    assert config["randomize_seeds"] is False


def test_issue_1662_smoke_config_keeps_lidar_only_training_boundary() -> None:
    """The smoke should stay aligned with the LiDAR-only eligibility metadata."""
    config = _load_yaml(_SMOKE_CONFIG)
    eligibility = _load_yaml(_ELIGIBILITY_SPEC)

    assert config["scenario_config"] == "../../scenarios/classic_interactions.yaml"
    assert config["tracking"] == {"tensorboard": False, "wandb": {"enabled": False}}
    assert config["num_envs"] == 4
    assert config["worker_mode"] == "subproc"
    observation_fields = eligibility["observation_fields"]
    assert observation_fields["deployment_observable"] == [
        "DEFAULT_GYM drive_state",
        "DEFAULT_GYM rays",
    ]
    assert observation_fields["forbidden_evaluation_time"] == []
    assert "rollout rewards" in observation_fields["training_only"]
