"""Keep the prepared pilot on the released plain-PPO contract, without training."""

from pathlib import Path

import yaml

from robot_sf.models import get_registry_entry
from scripts.training.train_ppo import _load_expert_training_config_mapping

ROOT = Path(__file__).resolve().parents[2]


def test_pilot_inheritance_preserves_plain_ppo_contract():
    """Resolve inheritance to catch an accidental guarded/curriculum source or plant."""
    protocol = yaml.safe_load(
        (ROOT / "configs/training/ppo/ftplan/pilot_protocol_20261012.yaml").read_text()
    )
    source = protocol["source"]
    baseline = yaml.safe_load(
        (ROOT / "configs/baselines/ppo_release_robot_0_0_8_cpu.yaml").read_text()
    )
    assert source["model_id"] == baseline["model_id"]
    entry = get_registry_entry(source["model_id"])
    config = _load_expert_training_config_mapping(ROOT / protocol["training_config"])
    assert config["num_envs"] == source["num_envs"] == 18
    assert (
        config["feature_extractor_kwargs"]["grid_channels"]
        == source["cnn_channels"]
        == [32, 64, 64]
    )
    assert (
        config["env_factory_kwargs"]["reward_name"]
        == source["reward_name"]
        == "route_completion_v3"
    )
    assert not config["env_factory_kwargs"].get("reward_curriculum")
    assert source["configured_reward_curriculum"] is False
    plant = config["env_overrides"]
    assert plant["ppo_action_semantics"] == entry["action_semantics"] == "velocity_delta"
    assert (
        plant["robot_config"]["max_linear_speed"] == source["training_linear_speed_cap_m_s"] == 2.0
    )
    assert plant["robot_config"]["allow_backwards"] is source["training_reverse_enabled"] is False
    assert plant["sim_config"]["time_per_step_in_secs"] == 0.1
    for key in ("max_linear_accel", "max_linear_decel", "max_angular_accel"):
        assert plant["robot_config"][key] == 1.0
    assert plant.get("predictive_foresight_enabled", False) is False
    assert config["seeds"] == []
    assert config["training_seed_range"] == [100000, 199999]
    assert config["evaluation"]["evaluation_seeds"] == list(range(1031, 1061))
    assert config["evaluation"]["evaluation_seed_manifest"] is None
    assert config["total_timesteps"] == protocol["steps_per_seed"] == 3000000
