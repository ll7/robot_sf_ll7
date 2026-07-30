"""Regression tests for unknown-key rejection in PPO config loading.

Issue #6489: load_expert_training_config must reject unknown top-level
and nested config keys with a deterministic ValueError before any
environment, model, or training process is created.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.training.imitation_config import (
    _ALLOWED_PPO_HYPERPARAMS,
    _ALLOWED_TOP_LEVEL_KEYS,
    validate_expert_training_config_keys,
)
from scripts.training import train_ppo

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestUnknownTopLevelKey:
    """Unknown keys at the top level must be rejected deterministically."""

    def test_typo_unknown_top_level_key_raises_value_error(self) -> None:
        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
            },
            "unknwn_key": "should_fail",
        }
        with pytest.raises(ValueError, match="unsupported top-level keys") as exc:
            validate_expert_training_config_keys(data)
        msg = str(exc.value)
        assert "unknwn_key" in msg
        assert "unknown" not in msg  # difflib should suggest "unknwn"

    def test_random_unknown_key_is_rejected(self) -> None:
        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
            },
            "garbage_field": "xyz",
        }
        with pytest.raises(ValueError, match="unsupported top-level keys") as exc:
            validate_expert_training_config_keys(data)
        assert "garbage_field" in str(exc.value)

    def test_several_unknown_keys_all_reported(self) -> None:
        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
            },
            "bad_key_one": 1,
            "bad_key_two": 2,
        }
        with pytest.raises(ValueError, match="unsupported top-level keys") as exc:
            validate_expert_training_config_keys(data)
        msg = str(exc.value)
        assert "bad_key_one" in msg
        assert "bad_key_two" in msg


class TestUnknownNestedKey:
    """Unknown nested keys under documented structural sections must be rejected."""

    def test_unknown_convergence_key_raises(self) -> None:
        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
                "typo_rate": 0.99,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
            },
        }
        with pytest.raises(ValueError, match="convergence has unsupported keys") as exc:
            validate_expert_training_config_keys(data)
        msg = str(exc.value)
        assert "typo_rate" in msg
        assert "success_rate" in msg  # difflib suggestion

    def test_unknown_evaluation_key_raises(self) -> None:
        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
                "unknown_eval_field": True,
            },
        }
        with pytest.raises(ValueError, match="evaluation has unsupported keys") as exc:
            validate_expert_training_config_keys(data)
        assert "unknown_eval_field" in str(exc.value)

    def test_unknown_ppo_hyperparams_key_raises(self) -> None:
        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
            },
            "ppo_hyperparams": {
                "learning_rate": 3e-4,
                "batch_size": 128,
                "unknown_param": 0.5,
            },
        }
        with pytest.raises(ValueError, match="ppo_hyperparams has unsupported keys") as exc:
            validate_expert_training_config_keys(data)
        msg = str(exc.value)
        assert "unknown_param" in msg

    def test_known_ppo_hyperparams_are_accepted(self) -> None:
        """All keys in _ALLOWED_PPO_HYPERPARAMS must pass validation."""
        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
            },
            "ppo_hyperparams": dict.fromkeys(_ALLOWED_PPO_HYPERPARAMS, 0.0),
        }
        validate_expert_training_config_keys(data)  # should not raise


class TestValidConfigsPass:
    """Existing canonical PPO configs must still load without raising."""

    _CANONICAL_CONFIGS = [
        "expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml",
        "expert_ppo_issue_576_br06_v4_validation_120k.yaml",
        "expert_ppo_issue_576_br06_v2_sanity_500k_all_maps.yaml",
        "expert_ppo_issue_576_br06_v5_predictive_foresight.yaml",
        "expert_ppo_issue_576_br06_v6_predictive_foresight_success_aligned.yaml",
        "expert_ppo_issue_576_br06_v7_predictive_foresight_xl_ego_success_aligned.yaml",
        "expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml",
        "expert_ppo_issue_739_12m_baseline_retrain.yaml",
        "expert_ppo_issue_576_br06_v9_predictive_foresight_xl_ego_success_priority.yaml",
    ]

    @pytest.mark.parametrize("config_name", _CANONICAL_CONFIGS)
    def test_canonical_config_loads_without_validation_error(self, config_name: str) -> None:
        config_path = _REPO_ROOT / "configs/training/ppo" / config_name
        assert config_path.is_file(), f"Config missing: {config_path}"
        config = train_ppo.load_expert_training_config(config_path)
        assert config.policy_id is not None


class TestSideEffectFree:
    """Rejection must happen before any env/model/training process is created."""

    def test_unknown_key_at_load_boundary_prevents_env_creation(self, monkeypatch) -> None:
        """make_robot_env must never be called when an unknown key is present."""
        make_env_calls: list[object] = []

        def _track_make_env(**kwargs: object) -> object:
            make_env_calls.append(kwargs)
            raise RuntimeError("should never be reached")

        monkeypatch.setattr(train_ppo, "make_robot_env", _track_make_env)
        monkeypatch.setattr(train_ppo, "PPO", object)

        data: dict[str, object] = {
            "scenario_config": "/dev/null/scenarios.yaml",
            "total_timesteps": 1000,
            "policy_id": "test",
            "seeds": [1],
            "convergence": {
                "success_rate": 0.9,
                "collision_rate": 0.05,
                "plateau_window": 100,
            },
            "evaluation": {
                "evaluation_episodes": 5,
                "step_schedule": [{"every_steps": 500}],
            },
            "typo_key": "should_fail_before_env",
        }
        with pytest.raises(ValueError, match="unsupported top-level keys"):
            validate_expert_training_config_keys(data)
        assert len(make_env_calls) == 0, "make_robot_env was called despite unknown key"

    def test_valid_config_calls_env_creation_normally(self, monkeypatch, tmp_path) -> None:
        """Sanity check: a valid config path reaches env creation."""
        child_yaml = tmp_path / "valid_minimal.yaml"
        child_yaml.write_text(
            "\n".join(
                [
                    "scenario_config: /dev/null/scenarios.yaml",
                    "total_timesteps: 1000",
                    "policy_id: test",
                    "seeds: [1]",
                    "convergence:",
                    "  success_rate: 0.9",
                    "  collision_rate: 0.05",
                    "  plateau_window: 100",
                    "evaluation:",
                    "  evaluation_episodes: 5",
                    "  step_schedule:",
                    "    - every_steps: 500",
                ]
            ),
            encoding="utf-8",
        )
        config = train_ppo.load_expert_training_config(child_yaml)

        make_env_calls: list[object] = []

        def _track_make_env(**kwargs: object) -> object:
            make_env_calls.append(kwargs)
            raise RuntimeError("should not reach real env creation in this test")

        monkeypatch.setattr(train_ppo, "make_robot_env", _track_make_env)
        monkeypatch.setattr(train_ppo, "PPO", object)

        assert config is not None


class TestAllowListCoverage:
    """Ensure all known config keys are covered by the allow-list."""

    def test_all_canonical_keys_in_allow_list(self) -> None:
        """Every key appearing in canonical PPO YAML files must be in the top-level allow-list."""
        import yaml

        seen: set[str] = set()
        for config_name in TestValidConfigsPass._CANONICAL_CONFIGS:
            config_path = _REPO_ROOT / "configs/training/ppo" / config_name
            with config_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "base_config" in data:
                base_path = config_path.parent / str(data["base_config"])
                with base_path.open(encoding="utf-8") as bf:
                    base_data = yaml.safe_load(bf) or {}
                data = {**base_data, **{k: v for k, v in data.items() if k != "base_config"}}
            seen.update(data.keys())

        covered = seen - _ALLOWED_TOP_LEVEL_KEYS
        assert not covered, f"Keys not in allow-list: {sorted(covered)}"
