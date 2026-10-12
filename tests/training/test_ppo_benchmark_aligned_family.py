"""Config and runtime evaluation admission for #9628; no PPO learning."""

from copy import deepcopy
from dataclasses import field, fields, make_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner_policies import map_runner_actions
from robot_sf.training.snqi_utils import default_training_snqi_context
from scripts.training import train_ppo
from scripts.training import train_ppo_benchmark_aligned as aligned


@pytest.mark.parametrize(
    ("training", "selection", "reporting", "reason"),
    [
        ([100000, 199999], [1031], [1031], "selection and reporting seeds overlap"),
        ([1000, 199999], [1031], [1101], "training range overlaps"),
        ([100000, 199999], [1101], [1101], "selection and reporting seeds overlap"),
        ([100000, 199999], [111], [1101], "selection seeds must be exactly"),
        ([100000, 199999], list(range(1031, 1061)), [111], "reporting seeds must be exactly"),
        ([111, 140], [1031], [1101], "training range must be the approved"),
        ([100000, 199999], [True], [1101], "selection seeds must contain integers"),
        ([100000, 199999], [1031, 1031], [1101], "selection seeds must be unique"),
    ],
)
def test_reject_seed_overlap_and_forbidden_identities(training, selection, reporting, reason):
    """Invalid partitions fail before any simulator is constructed; errors expose no values."""
    with pytest.raises(ValueError, match=reason):
        aligned.validate_seed_partitions(training, selection, reporting)


def test_seed_guard_excludes_every_identity_outside_positive_allowlists():
    """The guard admits only the exact authorised pools, including full-band exclusion."""
    selection = list(range(1031, 1061))
    reporting = list(range(1101, 1131))
    bounds, selected, reported = aligned.validate_seed_partitions(
        [100000, 199999], selection, reporting
    )
    assert bounds == (100000, 199999)
    assert selected == tuple(selection)
    assert reported == tuple(reporting)
    # Reject other dev pools and training identities in either evaluation role.
    for invalid in (1001, 1200, 100000, 199999, -1):
        with pytest.raises(ValueError):
            aligned.validate_seed_partitions([100000, 199999], [invalid], reporting)
        with pytest.raises(ValueError):
            aligned.validate_seed_partitions([100000, 199999], selection, [invalid])


def test_family_effective_conditions_and_preflight_without_training(monkeypatch, capsys):
    """Validate actual inherited YAML and every scenario, with no sampler requirement for preflight."""
    monkeypatch.setattr(train_ppo, "run_expert_training", lambda *_a, **_k: pytest.fail("training"))
    assert aligned.main(["--preflight"]) == 0
    assert '"training_started": false' in capsys.readouterr().out
    data = aligned.validate_family()
    assert data["env_overrides"]["robot_config"]["max_linear_speed"] == 2.0
    assert data["total_timesteps"] == 10000000
    assert data["evaluation"]["evaluation_episodes"] == 1440
    assert data["env_overrides"]["ppo_action_semantics"] == "velocity_delta"
    original = train_ppo._load_expert_training_config_mapping(
        aligned.ROOT / "configs/training/ppo/ablations/"
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
    )
    assert data["env_factory_kwargs"] == original["env_factory_kwargs"]
    assert data["ppo_hyperparams"] == original["ppo_hyperparams"]
    assert data["feature_extractor_kwargs"] == original["feature_extractor_kwargs"]


def test_launch_refuses_missing_sampler_before_training(monkeypatch):
    """A passing condition preflight never authorises unseeded training on main."""
    monkeypatch.setattr(aligned, "sampler_interface_available", lambda: False)
    monkeypatch.setattr(train_ppo, "run_expert_training", lambda *_a, **_k: pytest.fail("training"))
    with pytest.raises(RuntimeError, match="integrate PR #10324"):
        aligned.main([])


def test_launch_consumes_existing_sampler_loader(monkeypatch):
    """The hook delegates to the trainer API with the declared range, master and pools."""
    stock = train_ppo.load_expert_training_config(aligned.FAMILY)
    # Model the typed interface added by #10324 while retaining the real stock
    # loader's resolved environment/evaluation configuration.
    extended_type = make_dataclass(
        "ExtendedExpertConfig",
        [
            ("training_seed_range", tuple[int, int] | None, field(default=None)),
            ("training_master_seed", int | None, field(default=None)),
            ("development_evaluation_seeds", tuple[int, ...], field(default=())),
        ],
        bases=(train_ppo.ExpertTrainingConfig,),
    )
    extended = extended_type(**{item.name: getattr(stock, item.name) for item in fields(stock)})
    monkeypatch.setattr(aligned, "sampler_interface_available", lambda: True)
    monkeypatch.setattr(train_ppo, "load_expert_training_config", lambda _path: extended)
    calls = []
    monkeypatch.setattr(train_ppo, "run_expert_training", lambda config, **kw: calls.append(config))
    assert aligned.main([]) == 0
    assert len(calls) == 1
    configured = calls[0]
    assert configured.training_seed_range == (100000, 199999)
    assert configured.training_master_seed == 20261012
    assert configured.development_evaluation_seeds == tuple(range(1101, 1131))
    assert configured.evaluation.evaluation_seeds == tuple(range(1031, 1061))
    assert configured.randomize_seeds is True
    assert configured.seeds == ()
    assert configured.scenario_sampling["episode_seed_pool"] is None
    assert stock.seeds == (100000,)  # Hook binding must not mutate the stock template.


def test_runner_reuses_training_acceleration_conversion(monkeypatch):
    """Both public action paths share the velocity-target helper rather than duplicate its math."""
    from robot_sf.gym_env import robot_env
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.robot.action_adapters import unicycle_velocity_target_to_acceleration

    assert (
        robot_env.unicycle_velocity_target_to_acceleration
        is unicycle_velocity_target_to_acceleration
    )
    calls = []

    def observed(target, speed, dt):
        calls.append(dt)
        return unicycle_velocity_target_to_acceleration(target, speed, dt)

    monkeypatch.setattr(
        map_runner_actions, "unicycle_velocity_target_to_acceleration", observed, raising=False
    )
    env = SimpleNamespace(
        simulator=SimpleNamespace(robots=[SimpleNamespace(current_speed=(0.6, 0.2))])
    )
    action = map_runner_actions.policy_command_to_env_action(
        env=env,
        config=RobotSimulationConfig(),
        command=(0.55, 0.16),
    )
    np.testing.assert_allclose(action, [-0.5, -0.4], atol=1e-12)
    assert calls == [0.1]


def test_all_periodic_and_final_evaluations_use_selection_only(monkeypatch, tmp_path):
    """Exercise the real trainer evaluator over 48×30; reporting pool never reaches its factory."""
    data = deepcopy(aligned.validate_family())
    # Main's supported evaluator does not need the unmerged training sampler fields.
    for key in ("training_seed_range", "training_master_seed", "development_evaluation_seeds"):
        data.pop(key)
    data.pop("base_config", None)
    data["seeds"] = [100000]
    data["randomize_seeds"] = False
    data["scenario_config"] = str(aligned.FAMILY.parent / data["scenario_config"])
    evaluation = data["evaluation"]
    evaluation["scenario_config"] = data["scenario_config"]
    evaluation["evaluation_seed_manifest"] = str(
        aligned.FAMILY.parent / evaluation["evaluation_seed_manifest"]
    )
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(yaml.safe_dump(data))
    config = train_ppo.load_expert_training_config(config_path)
    seen = []

    class OneStepEnv:
        state = SimpleNamespace(max_sim_steps=1)

        def reset(self):
            return np.zeros(2), {}

        def step(self, _action):
            return np.zeros(2), 0.0, True, False, {"meta": {"is_route_complete": model.phase == 1}}

        def close(self):
            pass

    def factory(**kwargs):
        seen.append(kwargs["seed"])
        return OneStepEnv()

    monkeypatch.setattr(train_ppo, "make_robot_env", factory)
    monkeypatch.setattr(train_ppo, "build_robot_config_from_scenario", lambda *_a, **_k: object())
    monkeypatch.setattr(train_ppo, "_apply_env_overrides", lambda *_a, **_k: None)

    class Model:
        phase = 0
        logger = SimpleNamespace(record=lambda *_a, **_k: None, dump=lambda *_a, **_k: None)

        def learn(self, **_kwargs):
            self.phase += 1

        def predict(self, *_args, **_kwargs):
            return np.zeros(2), None

        def save(self, path):
            Path(path).write_bytes(str(self.phase).encode())

    model = Model()
    scenarios = [{"name": f"cell-{index}"} for index in range(48)]
    _, periodic, best, _, _ = train_ppo._train_with_schedule(
        model,
        config=config,
        evaluation_ctx=SimpleNamespace(
            scenario_definitions=scenarios, scenario_config=config.scenario_config
        ),
        scenario_definitions=scenarios,
        scenario_id=None,
        hold_out_scenarios=(),
        eval_steps=(524288, 1048576),
        snqi_context=default_training_snqi_context(),
        wandb_run=None,
        wandb_callback=None,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert best.eval_step == 524288
    assert best.value == 1.0
    assert best.checkpoint_path.read_bytes() == b"1"
    _, final = train_ppo._evaluate_policy(
        model,
        config,
        scenario_definitions=scenarios,
        scenario_path=config.scenario_config,
        scenario_id=None,
        hold_out_scenarios=(),
        snqi_context=default_training_snqi_context(),
    )
    for records in (periodic[:1440], periodic[1440:], final):
        assert len(records) == 1440
        assert {(r["scenario_id"], r["seed"]) for r in records} == {
            (f"cell-{cell}", seed) for cell in range(48) for seed in range(1031, 1061)
        }
    assert set(seen) == set(range(1031, 1061))
    assert not set(seen).intersection(range(1101, 1131))


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        ("evaluation", "randomize_seeds", True, "fixed seeds"),
        ("evaluation", "evaluation_episodes", 70, "every scenario-seed pair"),
        ("evaluation", "full_policy_analysis_on_new_best", True, "auxiliary evaluation"),
        ("env_overrides", "ppo_action_semantics", "acceleration", "velocity-delta interface"),
    ],
)
def test_reject_family_condition_drift(monkeypatch, section, key, value, reason):
    """The launch boundary rejects changes that otherwise silently weaken selection or parity."""
    data = train_ppo._load_expert_training_config_mapping(aligned.FAMILY)
    data[section][key] = value
    stock = train_ppo.load_expert_training_config(aligned.FAMILY)
    monkeypatch.setattr(train_ppo, "load_expert_training_config", lambda _p: stock)
    monkeypatch.setattr(train_ppo, "_load_expert_training_config_mapping", lambda _p: data)
    with pytest.raises(ValueError, match=reason):
        aligned.validate_family()


def test_stock_trainer_loads_family_without_sampler_fields():
    """The canonical leaf must load on main without sanitising unsupported fields."""
    config = train_ppo.load_expert_training_config(aligned.FAMILY)
    assert config.evaluation.evaluation_seeds == tuple(range(1031, 1061))
    assert config.env_overrides["ppo_action_semantics"] == "velocity_delta"
