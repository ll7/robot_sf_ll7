"""Regression tests for strict expert PPO configuration loading (Issue #6489)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from robot_sf.training.imitation_config import (
    ExpertTrainingConfig,
    validate_expert_training_config_keys,
)
from scripts.training import train_ppo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _REPO_ROOT / "configs" / "training" / "ppo"
_MAPPING_FIELDS = {
    "convergence",
    "density_curriculum",
    "domain_randomization",
    "env_factory_kwargs",
    "env_overrides",
    "evaluation",
    "feature_extractor_kwargs",
    "multi_map_protocol",
    "ppo_hyperparams",
    "scenario_sampling",
    "tracking",
}
_FIELD_ALIASES = {
    "snqi_baseline_path": "snqi_baseline",
    "snqi_weights_path": "snqi_weights",
}


def _valid_config() -> dict[str, object]:
    """Return a minimal mapping accepted by the expert PPO loader."""
    return {
        "scenario_config": "/dev/null/scenarios.yaml",
        "total_timesteps": 1_000,
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
    }


def _validate(data: object) -> None:
    """Validate with the PPO launcher's canonical coercion keys."""
    validate_expert_training_config_keys(
        data,
        allowed_ppo_hyperparams=train_ppo._PPO_PARAM_COERCIONS,
    )


def test_top_level_typo_reports_full_path_and_deterministic_suggestion() -> None:
    data = _valid_config()
    data["policy_di"] = "typo"

    with pytest.raises(ValueError) as exc_info:
        _validate(data)

    assert str(exc_info.value) == (
        "Invalid ExpertTrainingConfig keys or sections:\n"
        "- config.policy_di: unsupported key; did you mean 'config.policy_id'?"
    )


def test_nested_typos_are_aggregated_in_full_path_order() -> None:
    data = _valid_config()
    convergence = data["convergence"]
    evaluation = data["evaluation"]
    assert isinstance(convergence, dict)
    assert isinstance(evaluation, dict)
    convergence["success_rte"] = 0.8
    schedule = evaluation["step_schedule"]
    assert isinstance(schedule, list)
    assert isinstance(schedule[0], dict)
    schedule[0]["every_step"] = 250
    data["ppo_hyperparams"] = {"learning_rte": 0.001}
    data["socnav_orca"] = {"time_horizn": 2.0}

    with pytest.raises(ValueError) as exc_info:
        _validate(data)

    assert str(exc_info.value) == "\n".join(
        [
            "Invalid ExpertTrainingConfig keys or sections:",
            "- config.convergence.success_rte: unsupported key; "
            "did you mean 'config.convergence.success_rate'?",
            "- config.evaluation.step_schedule[0].every_step: unsupported key; "
            "did you mean 'config.evaluation.step_schedule[0].every_steps'?",
            "- config.ppo_hyperparams.learning_rte: unsupported key; "
            "did you mean 'config.ppo_hyperparams.learning_rate'?",
            "- config.socnav_orca.time_horizn: unsupported key; "
            "did you mean 'config.socnav_orca.time_horizon'?",
        ]
    )


def test_unrelated_training_schema_keys_are_not_silently_accepted() -> None:
    data = _valid_config()
    data.update({"output_dir": "output/models", "safety_constraints": {}})

    with pytest.raises(ValueError) as exc_info:
        _validate(data)

    assert str(exc_info.value) == "\n".join(
        [
            "Invalid ExpertTrainingConfig keys or sections:",
            "- config.output_dir: unsupported key",
            "- config.safety_constraints: unsupported key",
        ]
    )


@pytest.mark.parametrize(
    ("section_name", "value", "type_name"),
    [
        ("convergence", [], "list"),
        ("density_curriculum", "enabled", "str"),
        ("evaluation", [], "list"),
        ("ppo_hyperparams", 3, "int"),
        ("recurrent_ppo_hyperparams", [], "list"),
        ("socnav_orca", None, "NoneType"),
        ("tracking", [], "list"),
    ],
)
def test_non_mapping_sections_fail_with_full_path(
    section_name: str,
    value: object,
    type_name: str,
) -> None:
    data = _valid_config()
    data[section_name] = value

    with pytest.raises(ValueError) as exc_info:
        _validate(data)

    assert str(exc_info.value) == (
        "Invalid ExpertTrainingConfig keys or sections:\n"
        f"- config.{section_name}: expected a mapping, got {type_name}"
    )


@pytest.mark.parametrize(
    ("step_schedule", "expected"),
    [
        (
            "every 500",
            "config.evaluation.step_schedule: expected a sequence of mappings, got str",
        ),
        (
            [500],
            "config.evaluation.step_schedule[0]: expected a mapping, got int",
        ),
    ],
)
def test_malformed_step_schedule_reports_exact_nested_path(
    step_schedule: object,
    expected: str,
) -> None:
    data = _valid_config()
    evaluation = data["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["step_schedule"] = step_schedule

    with pytest.raises(ValueError) as exc_info:
        _validate(data)

    assert str(exc_info.value) == (f"Invalid ExpertTrainingConfig keys or sections:\n- {expected}")


def test_every_expert_training_dataclass_field_has_a_yaml_key() -> None:
    data = _valid_config()
    for field_info in fields(ExpertTrainingConfig):
        yaml_key = _FIELD_ALIASES.get(field_info.name, field_info.name)
        if yaml_key in data:
            continue
        data[yaml_key] = {} if field_info.name in _MAPPING_FIELDS else None

    _validate(data)


def test_documented_compatibility_extensions_remain_valid() -> None:
    data = _valid_config()
    data.update(
        {
            "algorithm": "recurrent_ppo",
            "base_config": "base.yaml",
            "recurrent_policy": "MultiInputLstmPolicy",
            "recurrent_ppo_hyperparams": {"policy_kwargs": {}},
            "socnav_orca": {"neighbor_dist": 3.0, "time_horizon": 2.0},
        }
    )
    evaluation = data["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["full_policy_analysis_on_new_best"] = False
    evaluation["full_policy_analysis_videos"] = False

    _validate(data)


def test_all_runtime_ppo_coercion_keys_are_accepted() -> None:
    data = _valid_config()
    data["ppo_hyperparams"] = dict.fromkeys(train_ppo._PPO_PARAM_COERCIONS, 0)

    _validate(data)


def _canonical_expert_configs(config_root: Path = _CONFIG_ROOT) -> tuple[Path, ...]:
    """Find tracked configs intended for ExpertTrainingConfig or its recurrent extension.

    A tracked config is canonical when it is a full config on its own or an
    expert overlay that no other tracked config uses as a base. An incomplete
    referenced overlay is an intermediate node rather than a runnable leaf;
    its shared keys are validated transitively through the runnable variants
    that inherit them. A referenced overlay may still be independently
    loadable (as with the predictive sub-base after #6748), but it is not a
    leaf and remains covered through its descendants. Constrained-RL variants
    are owned by their dedicated loader and are not ExpertTrainingConfig
    leaves, even when they inherit an expert-shaped base.
    """
    required_keys = {
        "convergence",
        "evaluation",
        "policy_id",
        "scenario_config",
        "total_timesteps",
    }
    runnable_overlay_keys = {"env_factory_kwargs", "num_envs", "worker_mode"}
    tracked_paths = sorted(config_root.rglob("*.yaml"))
    # Resolve base_config references exactly like train_ppo does so chained
    # intermediate bases (configs used as a base by another tracked config)
    # are identifiable across subdirectories.
    base_reference_paths: set[Path] = set()
    for config_path in tracked_paths:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            continue
        base_raw = raw.get("base_config")
        if not isinstance(base_raw, str):
            continue
        base_path = Path(base_raw)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base_reference_paths.add(base_path.resolve())

    selected: list[Path] = []
    for config_path in tracked_paths:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            continue
        raw_keys = set(raw)
        is_full_config = required_keys <= raw_keys
        is_expert_overlay = "base_config" in raw_keys and "candidates" not in raw_keys
        is_constrained_rl_config = "safety_constraints" in raw_keys
        # A runnable overlay has its own execution controls.  The predictive
        # sub-base is independently loadable after #6748, but intentionally
        # lacks those controls and remains an intermediate shared overlay.
        is_runnable_expert_overlay = runnable_overlay_keys <= raw_keys
        is_intermediate_base = (
            is_expert_overlay
            and not is_full_config
            and not is_runnable_expert_overlay
            and config_path.resolve() in base_reference_paths
        )
        if not is_constrained_rl_config and (
            is_full_config or (is_expert_overlay and not is_intermediate_base)
        ):
            selected.append(config_path)
    return tuple(selected)


def test_all_tracked_canonical_expert_configs_load() -> None:
    config_paths = _canonical_expert_configs()
    # The issue-791 leader became a shared intermediate base in #6691, so the
    # runnable-leaf inventory is one smaller without dropping a config file.
    assert len(config_paths) == 135

    failures: list[str] = []
    for config_path in config_paths:
        try:
            train_ppo.load_expert_training_config(config_path)
        except Exception as exc:
            relative_path = config_path.relative_to(_REPO_ROOT)
            failures.append(f"{relative_path}: {type(exc).__name__}: {exc}")

    assert not failures, "\n".join(failures)


def test_constrained_rl_variants_are_not_expert_config_leaves() -> None:
    """Dedicated constrained-RL configs stay out of the expert inventory."""
    selected_relative = {str(path.relative_to(_REPO_ROOT)) for path in _canonical_expert_configs()}
    for config_name in (
        "issue_4017_constrained_smoke.yaml",
        "issue_4017_unconstrained_smoke.yaml",
    ):
        assert f"configs/training/ppo/{config_name}" not in selected_relative


def test_chained_intermediate_base_is_not_a_canonical_expert_leaf() -> None:
    """Intermediate bases stay out of the leaf gate, while runnable leaves stay in it.

    Issue #6680 introduced the issue_576_br06 predictive sub-base as an
    intermediate between the family base and the v5-v11 variants. Issue #6691
    likewise made the issue_791 all-scenarios leader an intermediate for the
    best-checkpoint config. Both hold shared keys covered transitively through
    their runnable descendants and are independently loadable, but neither is
    a canonical leaf.
    """
    config_paths = _canonical_expert_configs()
    selected_relative = {str(p.relative_to(_REPO_ROOT)) for p in config_paths}

    sub_base = _CONFIG_ROOT / "expert_ppo_issue_576_br06_predictive_sub_base.yaml"
    assert str(sub_base.relative_to(_REPO_ROOT)) not in selected_relative
    assert sub_base in _CONFIG_ROOT.rglob("expert_ppo_issue_576_br06_predictive_sub_base.yaml")
    issue_791_leader = (
        _CONFIG_ROOT / "ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml"
    )
    assert str(issue_791_leader.relative_to(_REPO_ROOT)) not in selected_relative
    v3 = _CONFIG_ROOT / "expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml"
    assert str(v3.relative_to(_REPO_ROOT)) in selected_relative
    train_ppo.load_expert_training_config(sub_base)
    for variant in _CONFIG_ROOT.rglob("expert_ppo_*_predictive*.yaml"):
        if variant == sub_base:
            continue
        assert str(variant.relative_to(_REPO_ROOT)) in selected_relative


def test_incomplete_intermediate_base_is_excluded_but_leaf_is_selected(
    tmp_path: Path,
) -> None:
    """Keep a non-runnable referenced overlay from re-entering the load gate."""
    config_root = tmp_path / "ppo"
    config_root.mkdir()
    (config_root / "base.yaml").write_text(
        yaml.safe_dump({"scenario_config": "scenarios.yaml"}),
        encoding="utf-8",
    )
    intermediate = config_root / "intermediate.yaml"
    intermediate.write_text(
        yaml.safe_dump({"base_config": "base.yaml", "tracking": {"enabled": True}}),
        encoding="utf-8",
    )
    leaf = config_root / "leaf.yaml"
    leaf.write_text(
        yaml.safe_dump(
            {
                "base_config": "intermediate.yaml",
                "policy_id": "leaf",
                "total_timesteps": 1,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 10,
                },
                "evaluation": {
                    "evaluation_episodes": 1,
                    "step_schedule": [{"every_steps": 1}],
                },
            }
        ),
        encoding="utf-8",
    )

    selected = set(_canonical_expert_configs(config_root))

    assert intermediate not in selected
    assert leaf in selected


def test_base_config_typo_survives_merge_and_is_rejected(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    child_config = tmp_path / "child.yaml"
    base_config.write_text(
        yaml.safe_dump({**_valid_config(), "total_timestep": 999}),
        encoding="utf-8",
    )
    child_config.write_text(
        yaml.safe_dump({"base_config": "base.yaml", "policy_id": "child"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"config\.total_timestep") as exc_info:
        train_ppo.load_expert_training_config(child_config)

    assert "did you mean 'config.total_timesteps'?" in str(exc_info.value)


def test_cli_rejects_invalid_config_before_process_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    log_path = tmp_path / "logs" / "train.log"
    config_path.write_text(
        yaml.safe_dump({**_valid_config(), "policy_di": "typo"}),
        encoding="utf-8",
    )
    training_calls: list[object] = []
    logger_calls: list[str] = []
    monkeypatch.setenv("LOGURU_LEVEL", "SENTINEL")
    monkeypatch.setattr(
        train_ppo,
        "run_expert_training",
        lambda *args, **kwargs: training_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(train_ppo.logger, "remove", lambda *args: logger_calls.append("remove"))
    monkeypatch.setattr(
        train_ppo.logger,
        "add",
        lambda *args, **kwargs: logger_calls.append("add"),
    )

    with pytest.raises(ValueError, match=r"config\.policy_di"):
        train_ppo.main(
            [
                "--config",
                str(config_path),
                "--log-file",
                str(log_path),
                "--log-level",
                "DEBUG",
            ]
        )

    assert training_calls == []
    assert logger_calls == []
    assert not log_path.parent.exists()
    assert train_ppo.os.environ["LOGURU_LEVEL"] == "SENTINEL"
