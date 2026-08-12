"""Config validation tests for the issue #4017 constrained-RL training entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.training import train_ppo
from scripts.training.train_constrained_rl import load_constrained_rl_config

CONFIG_DIR = Path("configs/training/ppo")
_ISSUE_4017_VARIANTS = (
    "issue_4017_constrained_smoke.yaml",
    "issue_4017_unconstrained_smoke.yaml",
)
_ISSUE_4017_BASE_NAME = "issue_4017_constrained_rl_smoke_base.yaml"
_ISSUE_4017_BASELINE_PATH = Path("tests/training/_baseline_issue_4017_constrained_rl_resolved.json")


def test_constrained_smoke_config_loads_constraint_specs() -> None:
    """The PPO-Lagrangian smoke config exposes the three initial safety budgets."""

    config = load_constrained_rl_config(CONFIG_DIR / "issue_4017_constrained_smoke.yaml")

    assert config.policy_id == "ppo_lagrangian_issue_4017_smoke"
    assert config.total_timesteps == 256
    assert config.device == "cpu"
    assert config.safety_constraints.enabled is True
    assert [spec.name for spec in config.safety_constraints.constraints] == [
        "collision_any",
        "near_miss",
        "comfort_exposure",
    ]


def test_unconstrained_smoke_config_matches_training_shape_without_constraints() -> None:
    """The matched baseline differs by policy id, output dir, and disabled constraints only."""

    constrained = load_constrained_rl_config(CONFIG_DIR / "issue_4017_constrained_smoke.yaml")
    baseline = load_constrained_rl_config(CONFIG_DIR / "issue_4017_unconstrained_smoke.yaml")

    assert baseline.safety_constraints.enabled is False
    assert baseline.safety_constraints.constraints == ()
    assert baseline.total_timesteps == constrained.total_timesteps
    assert baseline.seed == constrained.seed
    assert baseline.ppo_hyperparams == constrained.ppo_hyperparams
    assert baseline.env_overrides == constrained.env_overrides
    assert baseline.env_factory_kwargs == constrained.env_factory_kwargs


def _issue_4017_baseline() -> dict[str, object]:
    """Load and validate the frozen pre-inheritance resolved-config baseline."""

    assert _ISSUE_4017_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; capture it before changing issue-4017 configs"
    )
    baseline = json.loads(_ISSUE_4017_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    return baseline


def _issue_4017_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for an issue-4017 config."""

    resolved = train_ppo._load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_4017_VARIANTS)
def test_issue_4017_smoke_configs_preserve_prechange_resolved_values(variant: str) -> None:
    """Base inheritance must preserve both pre-refactor constrained-RL mappings."""

    config_path = (CONFIG_DIR / variant).resolve()
    baseline = _issue_4017_baseline()

    assert _issue_4017_fingerprint(config_path) == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )


def test_issue_4017_smoke_base_keeps_launch_identity_variant_local() -> None:
    """The shared base carries no policy, safety, or output identity."""

    base_path = (CONFIG_DIR / _ISSUE_4017_BASE_NAME).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "safety_constraints" not in base_yaml
    assert "output_dir" not in base_yaml

    for variant in _ISSUE_4017_VARIANTS:
        variant_yaml = yaml.safe_load((CONFIG_DIR / variant).read_text(encoding="utf-8"))
        assert variant_yaml["base_config"] == _ISSUE_4017_BASE_NAME
        assert "policy_id" in variant_yaml
        assert "safety_constraints" in variant_yaml
        assert "output_dir" in variant_yaml


def test_constrained_loader_supports_recursive_base_config_and_deep_merge(
    tmp_path: Path,
) -> None:
    """Constrained-RL loading follows the PPO resolver's recursive merge contract."""

    scenario_config = Path("configs/scenarios/sets/classic_cross_trap_subset.yaml").resolve()
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "policy_id": "base_policy",
                "algorithm": "ppo",
                "scenario_config": str(scenario_config),
                "total_timesteps": 32,
                "seed": 99,
                "env_overrides": {"base": True, "nested": {"base": True}},
                "ppo_hyperparams": {"learning_rate": 0.0001, "n_steps": 16},
                "safety_constraints": {
                    "schema_version": "constrained_rl.v1",
                    "enabled": False,
                    "method": "lagrangian_ppo",
                    "update_mode": "episode",
                    "constraints": [],
                },
                "tracking": {"enabled": False},
                "output_dir": "output/base",
            }
        ),
        encoding="utf-8",
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "base.yaml",
                "policy_id": "child_policy",
                "env_overrides": {"nested": {"child": True}},
                "ppo_hyperparams": {"n_epochs": 2},
                "safety_constraints": {
                    "enabled": True,
                    "constraints": [
                        {
                            "name": "collision_any",
                            "source_key": "collision_any",
                            "budget_per_episode": 0.0,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_constrained_rl_config(child_path)

    assert config.policy_id == "child_policy"
    assert config.total_timesteps == 32
    assert config.env_overrides == {"base": True, "nested": {"base": True, "child": True}}
    assert config.ppo_hyperparams == {"learning_rate": 0.0001, "n_steps": 16, "n_epochs": 2}
    assert config.safety_constraints.enabled is True
    assert config.safety_constraints.method == "lagrangian_ppo"
    assert [spec.name for spec in config.safety_constraints.constraints] == ["collision_any"]


def test_constrained_loader_missing_base_config_fails_closed(tmp_path: Path) -> None:
    """A missing constrained-RL base must fail before config validation or launch."""

    config_path = tmp_path / "missing_base.yaml"
    config_path.write_text("base_config: does_not_exist.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="base_config .* does not exist"):
        load_constrained_rl_config(config_path)


def test_constrained_loader_base_config_cycle_fails_closed(tmp_path: Path) -> None:
    """Recursive constrained-RL base cycles must fail closed."""

    (tmp_path / "a.yaml").write_text("base_config: b.yaml\npolicy_id: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("base_config: a.yaml\npolicy_id: b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="base_config cycle detected"):
        load_constrained_rl_config(tmp_path / "a.yaml")


def test_unsupported_constraint_source_fails_closed(tmp_path: Path) -> None:
    """Unsupported safety-cost sources fail before any training launch."""

    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
policy_id: bad_constraint_source
algorithm: ppo
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: 256
seed: 4017
num_envs: 1
device: cpu
safety_constraints:
  enabled: true
  constraints:
    - name: unknown
      source_key: unknown_metric
      budget_per_episode: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported safety-cost source"):
        load_constrained_rl_config(config_path)


def test_disabled_constraints_reject_accidental_constraint_list(tmp_path: Path) -> None:
    """Baseline configs must not silently carry inactive constraint definitions."""

    config_path = tmp_path / "disabled_with_constraints.yaml"
    config_path.write_text(
        """
policy_id: disabled_with_constraints
algorithm: ppo
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: 256
seed: 4017
num_envs: 1
device: cpu
safety_constraints:
  enabled: false
  constraints:
    - name: collision_any
      source_key: collision_any
      budget_per_episode: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Disabled safety_constraints"):
        load_constrained_rl_config(config_path)
