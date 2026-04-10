"""Train RLlib DreamerV3 on Robot SF drive_state+rays observations."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin

import numpy as np
import yaml
from loguru import logger

from robot_sf.common.hardware import detect_hardware_capacity, recommend_env_runners
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.training.rllib_env_wrappers import DEFAULT_FLATTEN_KEYS, wrap_for_dreamerv3
from robot_sf.training.runtime_helpers import append_jsonl_record, resolve_ray_runtime_env
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from robot_sf.training.scenario_sampling import ScenarioSampler, ScenarioSwitchingEnv

_LOG_LEVEL_CHOICES = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
_TOP_LEVEL_KEYS = {"experiment", "ray", "env", "algorithm", "tracking", "evaluation"}
_EXPERIMENT_KEYS = {
    "run_id",
    "output_root",
    "train_iterations",
    "checkpoint_every",
    "seed",
    "log_level",
}
_RAY_KEYS = {
    "address",
    "local_mode",
    "include_dashboard",
    "num_cpus",
    "num_gpus",
    "disable_uv_run_runtime_env",
    "runtime_env",
}
_RAY_RUNTIME_ENV_KEYS = {"working_dir", "excludes", "env_vars", "py_executable"}
_ENV_KEYS = {
    "flatten_observation",
    "flatten_keys",
    "normalize_actions",
    "factory_kwargs",
    "config",
    "scenario_matrix",
}
_SCENARIO_MATRIX_KEYS = {
    "path",
    "strategy",
    "switch_per_reset",
    "include_scenarios",
    "exclude_scenarios",
    "weights",
}
_ALGORITHM_KEYS = {"framework", "training", "env_runners", "resources", "learners", "api_stack"}
_TRACKING_KEYS = {"wandb"}
_EVALUATION_KEYS = {
    "enabled",
    "every_iterations",
    "evaluation_episodes",
    "output_subdir",
    "scenario_matrix",
}
_WANDB_KEYS = {
    "enabled",
    "project",
    "entity",
    "group",
    "job_type",
    "mode",
    "tags",
    "run_name",
    "dir",
}
_IGNORED_ENV_CONTEXT_KEYS = {"worker_index", "vector_index", "num_workers", "remote"}
_AUTO_TOKEN = "auto"
_RAY_ENABLE_UV_RUN_RUNTIME_ENV_ENV_VAR = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
_DEFAULT_API_STACK: dict[str, bool] = {
    "enable_rl_module_and_learner": True,
    "enable_env_runner_and_connector_v2": True,
}


@dataclass(slots=True)
class ExperimentSettings:
    """Top-level run-control settings for one DreamerV3 execution."""

    run_id: str
    output_root: Path
    train_iterations: int
    checkpoint_every: int
    seed: int
    log_level: str


@dataclass(slots=True)
class RaySettings:
    """Ray cluster initialization settings."""

    address: str | None
    local_mode: bool
    include_dashboard: bool
    num_cpus: int | str | None
    num_gpus: float | int | str | None
    disable_uv_run_runtime_env: bool
    runtime_env: dict[str, object]


@dataclass(slots=True)
class ScenarioMatrixSettings:
    """Scenario-matrix-driven DreamerV3 training or evaluation settings."""

    path: Path
    strategy: str
    switch_per_reset: bool
    include_scenarios: tuple[str, ...]
    exclude_scenarios: tuple[str, ...]
    weights: dict[str, float]


@dataclass(slots=True)
class EnvSettings:
    """Environment construction settings for RLlib workers."""

    flatten_observation: bool
    flatten_keys: tuple[str, ...] | None
    normalize_actions: bool
    factory_kwargs: dict[str, object]
    config_overrides: dict[str, object]
    scenario_matrix: ScenarioMatrixSettings | None


@dataclass(slots=True)
class AlgorithmSettings:
    """Algorithm-specific settings forwarded into DreamerV3Config."""

    framework: str
    training: dict[str, object]
    env_runners: dict[str, object]
    resources: dict[str, object]
    learners: dict[str, object]
    api_stack: dict[str, object] | None


@dataclass(slots=True)
class WandbSettings:
    """Weights & Biases tracking settings."""

    enabled: bool
    project: str
    entity: str | None
    group: str | None
    job_type: str
    mode: str
    tags: tuple[str, ...]
    run_name: str | None
    directory: Path


@dataclass(slots=True)
class TrackingSettings:
    """Tracking backend settings for the training run."""

    wandb: WandbSettings


@dataclass(slots=True)
class EvaluationSettings:
    """Optional periodic evaluation settings for Dreamer training runs."""

    enabled: bool
    every_iterations: int
    evaluation_episodes: int
    output_subdir: str
    scenario_matrix: ScenarioMatrixSettings | None


@dataclass(slots=True)
class DreamerRunConfig:
    """Complete validated training configuration."""

    config_path: Path
    experiment: ExperimentSettings
    ray: RaySettings
    env: EnvSettings
    algorithm: AlgorithmSettings
    tracking: TrackingSettings
    evaluation: EvaluationSettings


def _ensure_mapping(payload: object, *, field_name: str) -> dict[str, object]:
    """Validate that a payload section is a mapping."""
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"'{field_name}' must be a mapping.")
    return dict(payload)


def _ensure_optional_mapping(payload: object, *, field_name: str) -> dict[str, object] | None:
    """Validate optional mapping payloads while preserving None."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"'{field_name}' must be a mapping.")
    return dict(payload)


def _ensure_known_keys(payload: dict[str, object], *, field_name: str, allowed: set[str]) -> None:
    """Reject unknown keys to keep configuration reproducible."""
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{field_name}': {', '.join(unknown)}")


def _coerce_optional_int(
    value: object, *, field_name: str, allow_auto: bool = False
) -> int | str | None:
    """Coerce optional integer fields."""
    if value is None:
        return None
    if allow_auto and isinstance(value, str) and value.strip().lower() == _AUTO_TOKEN:
        return _AUTO_TOKEN
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer for '{field_name}', received {value!r}") from exc


def _coerce_optional_float(
    value: object, *, field_name: str, allow_auto: bool = False
) -> float | int | str | None:
    """Coerce optional numeric fields."""
    if value is None:
        return None
    if allow_auto and isinstance(value, str) and value.strip().lower() == _AUTO_TOKEN:
        return _AUTO_TOKEN
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected number for '{field_name}', received {value!r}") from exc


def _coerce_optional_text(value: object, *, field_name: str) -> str | None:
    """Coerce optional text fields."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"'{field_name}' cannot be empty when provided.")
    return text


def _resolve_config_path(
    raw_value: object, *, field_name: str, base_dir: Path | None = None
) -> Path:
    """Resolve a path-like config value to an absolute path."""
    text_value = str(raw_value).strip()
    if not text_value:
        raise ValueError(f"{field_name} cannot be empty.")
    path_value = Path(text_value)
    if not path_value.is_absolute():
        origin = Path.cwd() if base_dir is None else base_dir
        path_value = (origin / path_value).resolve()
    return path_value


def _parse_experiment_settings(payload: dict[str, object]) -> ExperimentSettings:
    """Parse and validate the experiment section."""
    experiment_raw = _ensure_mapping(payload.get("experiment"), field_name="experiment")
    _ensure_known_keys(experiment_raw, field_name="experiment", allowed=_EXPERIMENT_KEYS)

    run_id = str(experiment_raw.get("run_id", "dreamerv3_drive_state_rays")).strip()
    if not run_id:
        raise ValueError("experiment.run_id cannot be empty.")

    output_root = _resolve_config_path(
        experiment_raw.get("output_root", "output/dreamerv3"),
        field_name="experiment.output_root",
    )
    train_iterations = int(experiment_raw.get("train_iterations", 20))
    checkpoint_every = int(experiment_raw.get("checkpoint_every", 5))
    seed = int(experiment_raw.get("seed", 123))
    if train_iterations < 1:
        raise ValueError("experiment.train_iterations must be >= 1.")
    if checkpoint_every < 1:
        raise ValueError("experiment.checkpoint_every must be >= 1.")
    log_level = str(experiment_raw.get("log_level", "INFO")).upper()
    if log_level not in _LOG_LEVEL_CHOICES:
        allowed_levels = ", ".join(_LOG_LEVEL_CHOICES)
        raise ValueError(f"experiment.log_level must be one of: {allowed_levels}")
    return ExperimentSettings(
        run_id=run_id,
        output_root=output_root,
        train_iterations=train_iterations,
        checkpoint_every=checkpoint_every,
        seed=seed,
        log_level=log_level,
    )


def _parse_ray_settings(payload: dict[str, object]) -> RaySettings:
    """Parse and validate the ray section."""
    ray_raw = _ensure_mapping(payload.get("ray"), field_name="ray")
    _ensure_known_keys(ray_raw, field_name="ray", allowed=_RAY_KEYS)
    runtime_env_raw = _ensure_mapping(ray_raw.get("runtime_env"), field_name="ray.runtime_env")
    _ensure_known_keys(runtime_env_raw, field_name="ray.runtime_env", allowed=_RAY_RUNTIME_ENV_KEYS)
    runtime_env: dict[str, object] = {}
    working_dir = _coerce_optional_text(
        runtime_env_raw.get("working_dir"), field_name="ray.runtime_env.working_dir"
    )
    if working_dir is not None:
        runtime_env["working_dir"] = str(
            _resolve_config_path(working_dir, field_name="working_dir")
        )

    excludes_raw = runtime_env_raw.get("excludes")
    if excludes_raw is not None:
        if not isinstance(excludes_raw, list | tuple):
            raise ValueError("ray.runtime_env.excludes must be a list or tuple of strings.")
        runtime_env["excludes"] = [str(path) for path in excludes_raw]

    env_vars_raw = runtime_env_raw.get("env_vars")
    if env_vars_raw is not None:
        env_vars_mapping = _ensure_mapping(env_vars_raw, field_name="ray.runtime_env.env_vars")
        runtime_env["env_vars"] = {str(key): str(value) for key, value in env_vars_mapping.items()}

    py_executable = _coerce_optional_text(
        runtime_env_raw.get("py_executable"), field_name="ray.runtime_env.py_executable"
    )
    if py_executable is not None:
        runtime_env["py_executable"] = py_executable

    return RaySettings(
        address=None if ray_raw.get("address") in {None, ""} else str(ray_raw.get("address")),
        local_mode=bool(ray_raw.get("local_mode", False)),
        include_dashboard=bool(ray_raw.get("include_dashboard", False)),
        num_cpus=_coerce_optional_int(
            ray_raw.get("num_cpus"), field_name="ray.num_cpus", allow_auto=True
        ),
        num_gpus=_coerce_optional_float(
            ray_raw.get("num_gpus"), field_name="ray.num_gpus", allow_auto=True
        ),
        disable_uv_run_runtime_env=bool(ray_raw.get("disable_uv_run_runtime_env", True)),
        runtime_env=runtime_env,
    )


def _parse_scenario_matrix_settings(
    raw_payload: object,
    *,
    field_name: str,
    base_dir: Path | None = None,
) -> ScenarioMatrixSettings | None:
    """Parse optional scenario-matrix settings."""
    if raw_payload is None:
        return None
    scenario_raw = _ensure_mapping(raw_payload, field_name=field_name)
    _ensure_known_keys(scenario_raw, field_name=field_name, allowed=_SCENARIO_MATRIX_KEYS)
    path_raw = scenario_raw.get("path")
    if path_raw in {None, ""}:
        raise ValueError(f"{field_name}.path cannot be empty.")
    include_raw = scenario_raw.get("include_scenarios", ())
    exclude_raw = scenario_raw.get("exclude_scenarios", ())
    if not isinstance(include_raw, list | tuple):
        raise ValueError(f"{field_name}.include_scenarios must be a list or tuple.")
    if not isinstance(exclude_raw, list | tuple):
        raise ValueError(f"{field_name}.exclude_scenarios must be a list or tuple.")
    weights_raw = _ensure_mapping(scenario_raw.get("weights"), field_name=f"{field_name}.weights")
    return ScenarioMatrixSettings(
        path=_resolve_config_path(path_raw, field_name=f"{field_name}.path", base_dir=base_dir),
        strategy=str(scenario_raw.get("strategy", "random")).strip().lower(),
        switch_per_reset=bool(scenario_raw.get("switch_per_reset", True)),
        include_scenarios=tuple(str(name) for name in include_raw),
        exclude_scenarios=tuple(str(name) for name in exclude_raw),
        weights={str(key): float(value) for key, value in weights_raw.items()},
    )


def _parse_env_settings(payload: dict[str, object], *, base_dir: Path | None = None) -> EnvSettings:
    """Parse and validate the environment section."""
    env_raw = _ensure_mapping(payload.get("env"), field_name="env")
    _ensure_known_keys(env_raw, field_name="env", allowed=_ENV_KEYS)

    flatten_keys_raw = env_raw.get("flatten_keys", list(DEFAULT_FLATTEN_KEYS))
    if flatten_keys_raw is None:
        flatten_keys = None
    else:
        if not isinstance(flatten_keys_raw, list | tuple):
            raise ValueError("env.flatten_keys must be a list/tuple of strings or null.")
        flatten_keys = tuple(str(key) for key in flatten_keys_raw)

    return EnvSettings(
        flatten_observation=bool(env_raw.get("flatten_observation", True)),
        flatten_keys=flatten_keys,
        normalize_actions=bool(env_raw.get("normalize_actions", True)),
        factory_kwargs=_ensure_mapping(
            env_raw.get("factory_kwargs"), field_name="env.factory_kwargs"
        ),
        config_overrides=_ensure_mapping(env_raw.get("config"), field_name="env.config"),
        scenario_matrix=_parse_scenario_matrix_settings(
            env_raw.get("scenario_matrix"),
            field_name="env.scenario_matrix",
            base_dir=base_dir,
        ),
    )


def _parse_algorithm_settings(payload: dict[str, object]) -> AlgorithmSettings:
    """Parse and validate the algorithm section."""
    algorithm_raw = _ensure_mapping(payload.get("algorithm"), field_name="algorithm")
    _ensure_known_keys(algorithm_raw, field_name="algorithm", allowed=_ALGORITHM_KEYS)
    return AlgorithmSettings(
        framework=str(algorithm_raw.get("framework", "torch")),
        training=_ensure_mapping(algorithm_raw.get("training"), field_name="algorithm.training"),
        env_runners=_ensure_mapping(
            algorithm_raw.get("env_runners"), field_name="algorithm.env_runners"
        ),
        resources=_ensure_mapping(algorithm_raw.get("resources"), field_name="algorithm.resources"),
        learners=_ensure_mapping(algorithm_raw.get("learners"), field_name="algorithm.learners"),
        api_stack=_ensure_optional_mapping(
            algorithm_raw.get("api_stack"), field_name="algorithm.api_stack"
        ),
    )


def _parse_tracking_settings(payload: dict[str, object]) -> TrackingSettings:
    """Parse and validate tracking configuration."""
    tracking_raw = _ensure_mapping(payload.get("tracking"), field_name="tracking")
    _ensure_known_keys(tracking_raw, field_name="tracking", allowed=_TRACKING_KEYS)

    wandb_raw = _ensure_mapping(tracking_raw.get("wandb"), field_name="tracking.wandb")
    _ensure_known_keys(wandb_raw, field_name="tracking.wandb", allowed=_WANDB_KEYS)
    tags_raw = wandb_raw.get("tags", ())
    if not isinstance(tags_raw, list | tuple):
        raise ValueError("tracking.wandb.tags must be a list or tuple of strings.")

    wandb_dir = _resolve_config_path(
        wandb_raw.get("dir", "output/wandb"),
        field_name="tracking.wandb.dir",
    )
    return TrackingSettings(
        wandb=WandbSettings(
            enabled=bool(wandb_raw.get("enabled", False)),
            project=str(wandb_raw.get("project", "robot_sf_dreamerv3")),
            entity=_coerce_optional_text(
                wandb_raw.get("entity"), field_name="tracking.wandb.entity"
            ),
            group=_coerce_optional_text(wandb_raw.get("group"), field_name="tracking.wandb.group"),
            job_type=str(wandb_raw.get("job_type", "train")),
            mode=str(wandb_raw.get("mode", "offline")).strip().lower(),
            tags=tuple(str(tag) for tag in tags_raw),
            run_name=_coerce_optional_text(
                wandb_raw.get("run_name"), field_name="tracking.wandb.run_name"
            ),
            directory=wandb_dir,
        )
    )


def _parse_evaluation_settings(
    payload: dict[str, object], *, base_dir: Path | None = None
) -> EvaluationSettings:
    """Parse and validate the optional periodic evaluation section."""
    evaluation_raw = _ensure_mapping(payload.get("evaluation"), field_name="evaluation")
    _ensure_known_keys(evaluation_raw, field_name="evaluation", allowed=_EVALUATION_KEYS)
    enabled = bool(evaluation_raw.get("enabled", False))
    every_iterations = int(evaluation_raw.get("every_iterations", 0) or 0)
    evaluation_episodes = int(evaluation_raw.get("evaluation_episodes", 0) or 0)
    if enabled and every_iterations < 1:
        raise ValueError("evaluation.every_iterations must be >= 1 when evaluation is enabled.")
    if enabled and evaluation_episodes < 1:
        raise ValueError("evaluation.evaluation_episodes must be >= 1 when evaluation is enabled.")
    output_subdir = str(evaluation_raw.get("output_subdir", "evaluation")).strip()
    return EvaluationSettings(
        enabled=enabled,
        every_iterations=every_iterations,
        evaluation_episodes=evaluation_episodes,
        output_subdir=output_subdir or "evaluation",
        scenario_matrix=_parse_scenario_matrix_settings(
            evaluation_raw.get("scenario_matrix"),
            field_name="evaluation.scenario_matrix",
            base_dir=base_dir,
        ),
    )


def load_run_config(path: Path) -> DreamerRunConfig:
    """Load, validate, and normalize the DreamerV3 run configuration YAML."""
    resolved = path.resolve()
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("DreamerV3 config must be a mapping.")
    payload = dict(raw)
    _ensure_known_keys(payload, field_name="root", allowed=_TOP_LEVEL_KEYS)

    return DreamerRunConfig(
        config_path=resolved,
        experiment=_parse_experiment_settings(payload),
        ray=_parse_ray_settings(payload),
        env=_parse_env_settings(payload, base_dir=resolved.parent),
        algorithm=_parse_algorithm_settings(payload),
        tracking=_parse_tracking_settings(payload),
        evaluation=_parse_evaluation_settings(payload, base_dir=resolved.parent),
    )


def _apply_nested_overrides(  # noqa: C901
    config_obj: object,
    overrides: dict[str, object],
    *,
    context_name: str = "env.config",
) -> None:
    """Recursively apply dict overrides onto dataclass-like objects."""
    field_map = (
        {field.name: field for field in fields(config_obj)} if is_dataclass(config_obj) else {}
    )

    def _coerce_from_annotation(annotation: object, value: object) -> object:
        if annotation in {Any, object}:
            return value
        origin = get_origin(annotation)
        if origin is not None:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if origin is list and isinstance(value, list | tuple):
                item_annotation = args[0] if args else Any
                return [_coerce_from_annotation(item_annotation, item) for item in value]
            if len(args) == 1:
                return _coerce_from_annotation(args[0], value)
        if isinstance(annotation, type):
            if issubclass(annotation, Enum) and not isinstance(value, annotation):
                return annotation(value)
            if is_dataclass(annotation) and isinstance(value, dict):
                nested_obj = annotation()
                _apply_nested_overrides(
                    nested_obj,
                    dict(value),
                    context_name=context_name,
                )
                return nested_obj
        return value

    def _coerce_override_value(
        current_attr: object,
        value: object,
        *,
        field_annotation: object | None,
    ) -> object:
        if field_annotation is not None:
            coerced = _coerce_from_annotation(field_annotation, value)
            if coerced is not value:
                return coerced
        if isinstance(current_attr, Enum) and not isinstance(value, type(current_attr)):
            return type(current_attr)(value)
        if isinstance(current_attr, list) and current_attr:
            exemplar = current_attr[0]
            if isinstance(exemplar, Enum) and isinstance(value, list | tuple):
                enum_type = type(exemplar)
                return [item if isinstance(item, enum_type) else enum_type(item) for item in value]
        if current_attr is None and field_annotation is not None:
            return _coerce_from_annotation(field_annotation, value)
        return value

    for key, value in overrides.items():
        if not hasattr(config_obj, key):
            raise ValueError(f"Unknown {context_name} override field: '{key}'")
        current_attr = getattr(config_obj, key)
        field_annotation = field_map.get(key).type if key in field_map else None
        if isinstance(value, dict) and (
            hasattr(current_attr, "__dict__") or is_dataclass(current_attr)
        ):
            _apply_nested_overrides(current_attr, value, context_name=f"{context_name}.{key}")
        elif key == "grid_config" and isinstance(value, dict):
            setattr(config_obj, key, _coerce_from_annotation(GridConfig, value))
        else:
            setattr(
                config_obj,
                key,
                _coerce_override_value(
                    current_attr,
                    value,
                    field_annotation=field_annotation,
                ),
            )


def _create_env_template(env_settings: EnvSettings) -> RobotSimulationConfig:
    """Build the base RobotSimulationConfig used by all RLlib workers."""
    config = RobotSimulationConfig()
    _apply_nested_overrides(config, env_settings.config_overrides, context_name="env.config")
    # DreamerV3 configs in this launcher currently assume non-image observations.
    config.use_image_obs = False
    return config


def _make_env_creator(config: DreamerRunConfig) -> Any:
    """Create the RLlib-compatible environment factory closure."""
    template = _create_env_template(config.env)
    flatten_observation = config.env.flatten_observation
    flatten_keys = config.env.flatten_keys
    normalize_actions = config.env.normalize_actions
    factory_kwargs = dict(config.env.factory_kwargs)
    factory_kwargs.setdefault("debug", False)
    factory_kwargs.setdefault("recording_enabled", False)
    scenario_matrix = config.env.scenario_matrix
    loaded_scenarios = (
        load_scenarios(scenario_matrix.path, base_dir=scenario_matrix.path)
        if scenario_matrix is not None
        else None
    )

    def _creator(worker_env_config: dict[str, object] | None = None) -> Any:
        worker_payload = dict(worker_env_config or {})
        worker_overrides = {
            key: value
            for key, value in worker_payload.items()
            if key not in _IGNORED_ENV_CONTEXT_KEYS
        }
        worker_index = int(worker_payload.get("worker_index", 0) or 0)
        worker_seed = int(config.experiment.seed + worker_index)
        if scenario_matrix is None:
            worker_config = copy.deepcopy(template)
            _apply_nested_overrides(worker_config, worker_overrides, context_name="worker.config")
            env = make_robot_env(config=worker_config, **factory_kwargs)
        else:
            sampler = ScenarioSampler(
                scenarios=loaded_scenarios or (),
                include_scenarios=scenario_matrix.include_scenarios,
                exclude_scenarios=scenario_matrix.exclude_scenarios,
                weights=scenario_matrix.weights or None,
                seed=worker_seed,
                strategy=scenario_matrix.strategy,
            )

            def _config_builder(scenario: dict[str, object]) -> RobotSimulationConfig:
                scenario_config = build_robot_config_from_scenario(
                    scenario,
                    scenario_path=scenario_matrix.path,
                )
                _apply_nested_overrides(
                    scenario_config,
                    copy.deepcopy(config.env.config_overrides),
                    context_name="env.config",
                )
                _apply_nested_overrides(
                    scenario_config,
                    worker_overrides,
                    context_name="worker.config",
                )
                scenario_config.use_image_obs = False
                return scenario_config

            env = ScenarioSwitchingEnv(
                scenario_sampler=sampler,
                scenario_path=scenario_matrix.path,
                env_factory=make_robot_env,
                config_builder=_config_builder,
                env_factory_kwargs=factory_kwargs,
                suite_name="dreamerv3_rllib",
                algorithm_name=config.experiment.run_id,
                switch_per_reset=scenario_matrix.switch_per_reset,
                seed=worker_seed,
            )
        return wrap_for_dreamerv3(
            env,
            flatten_observation=flatten_observation,
            flatten_keys=flatten_keys,
            normalize_actions=normalize_actions,
        )

    return _creator


def _import_rllib():
    """Import Ray/RLlib symbols with a clear installation error."""
    try:
        import ray
        from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
        from ray.tune.registry import register_env
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError(
            "Ray RLlib is required. Install with `uv sync --extra rllib` and run via "
            "`uv run --extra rllib ...`."
        ) from exc
    return ray, DreamerV3Config, register_env


def _apply_config_method(
    config_obj: object, method_name: str, payload: dict[str, object]
) -> object:
    """Apply one optional AlgorithmConfig method when available."""
    if not payload:
        return config_obj
    method = getattr(config_obj, method_name, None)
    if callable(method):
        return method(**payload)
    logger.warning(
        "Skipping algorithm.{} payload because method is unavailable on {}. keys={}",
        method_name,
        type(config_obj).__name__,
        sorted(payload.keys()),
    )
    return config_obj


def _apply_env_runner_settings(config_obj: object, payload: dict[str, object]) -> object:
    """Apply env-runner/rollout settings across RLlib API variants."""
    if not payload:
        return config_obj
    env_runners = getattr(config_obj, "env_runners", None)
    if callable(env_runners):
        return env_runners(**payload)
    rollouts = getattr(config_obj, "rollouts", None)
    if callable(rollouts):
        return rollouts(**payload)
    logger.warning(
        "Skipping algorithm.env_runners payload because neither env_runners() nor rollouts() "
        "is available on {}. keys={}",
        type(config_obj).__name__,
        sorted(payload.keys()),
    )
    return config_obj


def _is_auto(value: object) -> bool:
    """Return True when value is the special 'auto' token."""
    return isinstance(value, str) and value.strip().lower() == _AUTO_TOKEN


def _resolve_auto_overrides(
    run_config: DreamerRunConfig,
    *,
    capacity: object | None = None,
) -> tuple[dict[str, object], dict[str, object], object | None]:
    """Resolve auto resource placeholders against detected local/slurm capacity."""
    env_runner_payload = dict(run_config.algorithm.env_runners)
    resources_payload = dict(run_config.algorithm.resources)
    use_auto = any(
        (
            _is_auto(run_config.ray.num_cpus),
            _is_auto(run_config.ray.num_gpus),
            _is_auto(env_runner_payload.get("num_env_runners")),
            _is_auto(resources_payload.get("num_gpus")),
        )
    )
    if not use_auto:
        return env_runner_payload, resources_payload, None

    if capacity is None:
        capacity = detect_hardware_capacity(reserve_cpu_cores=0, minimum_cpus=1)
    if _is_auto(env_runner_payload.get("num_env_runners")):
        env_runner_payload["num_env_runners"] = recommend_env_runners(capacity, cpu_headroom=4)
    if _is_auto(resources_payload.get("num_gpus")):
        resources_payload["num_gpus"] = capacity.visible_gpus

    logger.info(
        "Resolved auto resources: logical_cpus={} allocated_cpus={} usable_cpus={} "
        "allocated_gpus={} visible_gpus={} env_runners={} algo_num_gpus={}",
        capacity.logical_cpus,
        capacity.allocated_cpus,
        capacity.usable_cpus,
        capacity.allocated_gpus,
        capacity.visible_gpus,
        env_runner_payload.get("num_env_runners"),
        resources_payload.get("num_gpus"),
    )
    return env_runner_payload, resources_payload, capacity


def _build_algorithm_config(
    run_config: DreamerRunConfig,
    *,
    env_name: str,
    env_overrides: dict[str, object] | None = None,
    capacity: object | None = None,
) -> Any:
    """Construct DreamerV3Config from the validated YAML payload."""
    _, DreamerV3Config, _ = _import_rllib()
    cfg = DreamerV3Config()
    cfg = cfg.environment(
        env=env_name,
        env_config=dict(env_overrides or {}),
        disable_env_checking=False,
    )
    cfg = cfg.framework(run_config.algorithm.framework)
    env_runner_payload, resources_payload, _ = _resolve_auto_overrides(
        run_config,
        capacity=capacity,
    )
    api_stack_settings = (
        run_config.algorithm.api_stack
        if run_config.algorithm.api_stack is not None
        else dict(_DEFAULT_API_STACK)
    )
    cfg = _apply_config_method(cfg, "api_stack", api_stack_settings)
    cfg = _apply_env_runner_settings(cfg, env_runner_payload)
    cfg = _apply_config_method(cfg, "resources", resources_payload)
    cfg = _apply_config_method(cfg, "learners", run_config.algorithm.learners)
    cfg = _apply_config_method(cfg, "training", run_config.algorithm.training)
    debugging = getattr(cfg, "debugging", None)
    if callable(debugging):
        cfg = cfg.debugging(
            seed=run_config.experiment.seed, log_level=run_config.experiment.log_level
        )
    else:
        cfg.seed = run_config.experiment.seed
        cfg.log_level = run_config.experiment.log_level
    return cfg


def _extract_metric(result: dict[str, Any], *keys: str) -> float | int | None:
    """Extract a scalar metric from nested training result dictionaries."""
    for key in keys:
        value = result.get(key)
        if isinstance(value, int | float):
            return value
    for container_name in ("env_runners", "env_runner_results"):
        container = result.get(container_name)
        if isinstance(container, dict):
            for key in keys:
                value = container.get(key)
                if isinstance(value, int | float):
                    return value
    return None


def _extract_finite_metric(result: dict[str, Any], *keys: str) -> float | int | None:
    """Extract a scalar metric, but return None when the value is non-finite."""
    value = _extract_metric(result, *keys)
    return value if _is_finite_scalar(value) else None


def _is_finite_scalar(value: Any) -> bool:
    """Return True when value is an int/float and finite."""
    return isinstance(value, int | float) and math.isfinite(float(value))


def _find_nonfinite_scalars(
    value: Any,
    *,
    prefix: str = "",
    limit: int = 32,
) -> list[dict[str, object]]:
    """Collect paths for non-finite scalar values inside nested results."""
    findings: list[dict[str, object]] = []

    def _visit(current: Any, path: str) -> None:
        if len(findings) >= limit:
            return
        if isinstance(current, int | float):
            numeric = float(current)
            if not math.isfinite(numeric):
                findings.append(
                    {
                        "path": path or "<root>",
                        "value": str(current),
                    }
                )
            return
        if isinstance(current, dict):
            for key, nested in current.items():
                child_path = f"{path}.{key}" if path else str(key)
                _visit(nested, child_path)
                if len(findings) >= limit:
                    return
        elif isinstance(current, list | tuple):
            for index, nested in enumerate(current):
                child_path = f"{path}[{index}]" if path else f"[{index}]"
                _visit(nested, child_path)
                if len(findings) >= limit:
                    return

    _visit(value, prefix)
    return findings


def _build_nonfinite_diagnostics(
    result: dict[str, Any],
    *,
    iteration: int,
    reward_mean: float | int | None,
    timesteps_total: float | int | None,
) -> dict[str, object]:
    """Build a compact, JSON-safe diagnostic payload for non-finite training metrics."""
    interesting_paths = {
        "env_episode_return_mean": _extract_metric(result, "episode_return_mean", "episode_reward_mean"),
        "env_episode_len_mean": _extract_metric(result, "episode_len_mean", "episode_len_mean"),
        "num_env_steps_sampled_lifetime": _extract_metric(result, "num_env_steps_sampled_lifetime"),
        "num_env_steps_trained_lifetime": _extract_metric(result, "num_env_steps_trained_lifetime"),
        "learner_total_loss": (
            result.get("learners", {})
            .get("__all_modules__", {})
            .get("total_loss")
            if isinstance(result.get("learners"), dict)
            else None
        ),
        "learner_policy_total_loss": (
            result.get("learners", {})
            .get("default_policy", {})
            .get("total_loss")
            if isinstance(result.get("learners"), dict)
            else None
        ),
        "learner_world_model_loss": (
            result.get("learners", {})
            .get("default_policy", {})
            .get("world_model_loss")
            if isinstance(result.get("learners"), dict)
            else None
        ),
        "learner_actor_loss": (
            result.get("learners", {})
            .get("default_policy", {})
            .get("actor_loss")
            if isinstance(result.get("learners"), dict)
            else None
        ),
        "learner_critic_loss": (
            result.get("learners", {})
            .get("default_policy", {})
            .get("critic_loss")
            if isinstance(result.get("learners"), dict)
            else None
        ),
    }
    return {
        "iteration": iteration,
        "reward_mean": reward_mean,
        "timesteps_total": timesteps_total,
        "top_level_keys": sorted(result.keys()),
        "interesting_metrics": interesting_paths,
        "nonfinite_scalars": _find_nonfinite_scalars(result),
    }


def _save_checkpoint(algo: Any, checkpoint_dir: Path) -> str:
    """Persist one checkpoint and return its path as string."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_to_path = getattr(algo, "save_to_path", None)
    if callable(save_to_path):
        return str(save_to_path(str(checkpoint_dir)))
    save = getattr(algo, "save", None)
    if callable(save):
        return str(save(str(checkpoint_dir)))
    raise RuntimeError("Unable to save checkpoint: algorithm has neither save_to_path nor save.")


def _tree_to_torch_batch(value: Any, *, torch: Any, device: Any, add_time_dim: bool) -> Any:
    """Convert nested numpy/python structures into batched torch tensors."""
    if isinstance(value, dict):
        return {
            key: _tree_to_torch_batch(
                nested,
                torch=torch,
                device=device,
                add_time_dim=add_time_dim,
            )
            for key, nested in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _tree_to_torch_batch(nested, torch=torch, device=device, add_time_dim=add_time_dim)
            for nested in value
        )
    if isinstance(value, list):
        return [
            _tree_to_torch_batch(nested, torch=torch, device=device, add_time_dim=add_time_dim)
            for nested in value
        ]
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    tensor = tensor.unsqueeze(0)
    if add_time_dim:
        tensor = tensor.unsqueeze(1)
    return tensor


def _tree_to_numpy(value: Any) -> Any:
    """Detach nested torch structures into numpy/python values."""
    if isinstance(value, dict):
        return {key: _tree_to_numpy(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_numpy(nested) for nested in value)
    if isinstance(value, list):
        return [_tree_to_numpy(nested) for nested in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _tree_strip_batch_dim(value: Any) -> Any:
    """Remove the leading batch axis from nested inference outputs."""
    if isinstance(value, dict):
        return {key: _tree_strip_batch_dim(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_strip_batch_dim(nested) for nested in value)
    if isinstance(value, list):
        return [_tree_strip_batch_dim(nested) for nested in value]
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def _normalize_action_output(action: Any) -> Any:
    """Collapse batch/time dimensions from single-step action outputs."""
    if isinstance(action, tuple):
        action = action[0]
    if hasattr(action, "detach") or hasattr(action, "numpy"):
        action = _tree_to_numpy(action)
    if isinstance(action, np.ndarray):
        while action.ndim > 0 and action.shape[0] == 1:
            action = action[0]
        if action.ndim == 0:
            return action.item()
    return action


def _predict_action(
    algo: Any,
    observation: Any,
    *,
    policy_state: Any | None = None,
    is_first: bool = False,
) -> tuple[Any, Any | None]:
    """Compute one deterministic action across common RLlib Algorithm APIs."""
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        module = get_module()
        if module is not None:
            import torch
            from ray.rllib.core.columns import Columns

            try:
                device = next(module.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cpu")
            if policy_state is None:
                policy_state = module.get_initial_state()
            module_inputs = {
                Columns.OBS: _tree_to_torch_batch(
                    observation,
                    torch=torch,
                    device=device,
                    add_time_dim=True,
                ),
                Columns.STATE_IN: _tree_to_torch_batch(
                    policy_state,
                    torch=torch,
                    device=device,
                    add_time_dim=False,
                ),
                "is_first": torch.as_tensor([is_first], dtype=torch.float32, device=device),
            }
            module_outputs = module.forward_inference(module_inputs)
            action = _normalize_action_output(module_outputs[Columns.ACTIONS])
            next_state = _tree_strip_batch_dim(
                _tree_to_numpy(module_outputs.get(Columns.STATE_OUT, policy_state))
            )
            return action, next_state

    compute_single_action = getattr(algo, "compute_single_action", None)
    if callable(compute_single_action):
        try:
            action = compute_single_action(observation, explore=False)
        except NotImplementedError:
            logger.debug("algorithm.compute_single_action() is unsupported; falling back.")
        else:
            return _normalize_action_output(action), policy_state

    compute_actions = getattr(algo, "compute_actions", None)
    if not callable(compute_actions):
        raise RuntimeError(
            "Unable to evaluate DreamerV3 policy: algorithm exposes neither "
            "compute_single_action, get_module, nor compute_actions."
        )
    action = compute_actions([observation], explore=False)
    if isinstance(action, tuple):
        action = action[0]
    action = action[0]
    return _normalize_action_output(action), policy_state


def _is_better_reward(candidate: float | int | None, incumbent: float | int | None) -> bool:
    """Return whether a new scalar reward should replace the tracked best value."""
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return float(candidate) > float(incumbent)


def _episode_flags(info: dict[str, Any]) -> dict[str, bool]:
    """Extract benchmark-style outcome flags from Robot SF episode info."""
    meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
    return {
        "success": bool(meta.get("is_route_complete") or info.get("success", False)),
        "collision": bool(
            meta.get("is_pedestrian_collision")
            or meta.get("is_robot_collision")
            or meta.get("is_obstacle_collision")
            or info.get("collision", False)
        ),
        "pedestrian_collision": bool(meta.get("is_pedestrian_collision", False)),
        "robot_collision": bool(meta.get("is_robot_collision", False)),
        "obstacle_collision": bool(meta.get("is_obstacle_collision", False)),
        "timeout": bool(meta.get("is_timesteps_exceeded", False)),
    }


def _run_periodic_evaluation(
    algo: Any,
    run_config: DreamerRunConfig,
    *,
    iteration: int,
    evaluation_dir: Path,
) -> dict[str, object]:
    """Evaluate the in-memory policy on the configured scenario matrix."""
    matrix = run_config.evaluation.scenario_matrix or run_config.env.scenario_matrix
    eval_env_settings = copy.deepcopy(run_config.env)
    if matrix is not None:
        eval_env_settings.scenario_matrix = copy.deepcopy(matrix)
        eval_env_settings.scenario_matrix.strategy = "cycle"
        eval_env_settings.scenario_matrix.switch_per_reset = True
    eval_config = DreamerRunConfig(
        config_path=run_config.config_path,
        experiment=run_config.experiment,
        ray=run_config.ray,
        env=eval_env_settings,
        algorithm=run_config.algorithm,
        tracking=run_config.tracking,
        evaluation=run_config.evaluation,
    )
    env = _make_env_creator(eval_config)({"worker_index": 0})
    records: list[dict[str, object]] = []
    try:
        for episode in range(1, run_config.evaluation.evaluation_episodes + 1):
            observation, _ = env.reset(seed=run_config.experiment.seed + episode)
            done = False
            episode_return = 0.0
            steps = 0
            policy_state = None
            is_first = True
            last_info: dict[str, Any] = {}
            max_steps = int(getattr(getattr(env, "state", None), "max_sim_steps", 1000))
            while not done and steps < max_steps:
                action, policy_state = _predict_action(
                    algo,
                    observation,
                    policy_state=policy_state,
                    is_first=is_first,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                last_info = dict(info or {})
                done = bool(terminated or truncated)
                is_first = False
            flags = _episode_flags(last_info)
            records.append(
                {
                    "iteration": iteration,
                    "episode": episode,
                    "scenario_id": getattr(env, "scenario_id", None),
                    "return": episode_return,
                    "steps": steps,
                    **flags,
                }
            )
    finally:
        env.close()

    count = max(len(records), 1)
    summary = {
        "iteration": iteration,
        "episodes": len(records),
        "scenario_matrix": str(matrix.path) if matrix is not None else None,
        "success_rate": sum(bool(row["success"]) for row in records) / count,
        "collision_rate": sum(bool(row["collision"]) for row in records) / count,
        "timeout_rate": sum(bool(row["timeout"]) for row in records) / count,
        "evaluation_records_path": str(evaluation_dir / f"iteration_{iteration:06d}.jsonl"),
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    records_path = Path(str(summary["evaluation_records_path"]))
    for row in records:
        append_jsonl_record(records_path, row)
    _write_json(evaluation_dir / f"iteration_{iteration:06d}_summary.json", summary)
    return summary


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _serialize_for_wandb(value: Any) -> Any:
    """Recursively convert dataclass/path values into wandb-friendly primitives."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            key: _serialize_for_wandb(getattr(value, key))
            for key in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, dict):
        return {str(k): _serialize_for_wandb(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize_for_wandb(v) for v in value]
    return value


def _init_wandb_tracking(run_config: DreamerRunConfig, *, run_stamp: str):
    """Initialize Weights & Biases run if enabled in config."""
    wandb_cfg = run_config.tracking.wandb
    if not wandb_cfg.enabled:
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "W&B tracking is enabled but wandb is unavailable. Install dependencies with "
            "`uv sync` and retry."
        ) from exc

    wandb_cfg.directory.mkdir(parents=True, exist_ok=True)
    run_name = wandb_cfg.run_name or f"{run_config.experiment.run_id}_{run_stamp}"
    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group,
        job_type=wandb_cfg.job_type,
        name=run_name,
        mode=wandb_cfg.mode,
        tags=list(wandb_cfg.tags),
        dir=str(wandb_cfg.directory),
        config=_serialize_for_wandb(run_config),
        reinit="finish_previous",
    )


def _resolve_ray_resource_kwargs(
    run_config: DreamerRunConfig, *, capacity: object | None = None
) -> dict[str, object]:
    """Resolve optional Ray CPU/GPU kwargs including 'auto' placeholders."""
    resource_kwargs: dict[str, object] = {}
    needs_auto = _is_auto(run_config.ray.num_cpus) or _is_auto(run_config.ray.num_gpus)
    if needs_auto and capacity is None:
        capacity = detect_hardware_capacity(reserve_cpu_cores=0, minimum_cpus=1)
    if _is_auto(run_config.ray.num_cpus):
        if capacity is None:
            raise RuntimeError("Hardware capacity detection failed for auto CPU resolution.")
        resource_kwargs["num_cpus"] = capacity.usable_cpus
    elif run_config.ray.num_cpus is not None:
        resource_kwargs["num_cpus"] = run_config.ray.num_cpus
    if _is_auto(run_config.ray.num_gpus):
        if capacity is None:
            raise RuntimeError("Hardware capacity detection failed for auto GPU resolution.")
        resource_kwargs["num_gpus"] = capacity.visible_gpus
    elif run_config.ray.num_gpus is not None:
        resource_kwargs["num_gpus"] = run_config.ray.num_gpus
    return resource_kwargs


def _build_runtime_env_kwargs(run_config: DreamerRunConfig) -> dict[str, object]:
    """Build runtime_env payload with safe defaults for Dreamer worker startup."""
    return resolve_ray_runtime_env(
        runtime_env_base=run_config.ray.runtime_env,
        working_dir=Path(__file__).resolve().parents[2],
        py_executable=sys.executable,
        expected_virtual_env=os.environ.get("VIRTUAL_ENV"),
    )


def _build_ray_init_kwargs(
    run_config: DreamerRunConfig, *, capacity: object | None = None
) -> dict[str, object]:
    """Assemble kwargs passed into ray.init()."""
    ray_kwargs: dict[str, object] = {
        "address": run_config.ray.address,
        "local_mode": run_config.ray.local_mode,
        "include_dashboard": run_config.ray.include_dashboard,
        "ignore_reinit_error": True,
        "logging_level": getattr(logging, run_config.experiment.log_level, logging.INFO),
    }
    if run_config.experiment.log_level in {"WARNING", "ERROR", "CRITICAL"}:
        ray_kwargs["log_to_driver"] = False
    ray_kwargs.update(_resolve_ray_resource_kwargs(run_config, capacity=capacity))
    ray_kwargs["runtime_env"] = _build_runtime_env_kwargs(run_config)
    return ray_kwargs


def _should_detect_shared_capacity(run_config: DreamerRunConfig) -> bool:
    """Return True when any ray/algo setting requires hardware auto detection."""
    return any(
        (
            _is_auto(run_config.ray.num_cpus),
            _is_auto(run_config.ray.num_gpus),
            _is_auto(run_config.algorithm.env_runners.get("num_env_runners")),
            _is_auto(run_config.algorithm.resources.get("num_gpus")),
        )
    )


def _detect_shared_capacity(run_config: DreamerRunConfig) -> object | None:
    """Detect hardware capacity once when auto resource placeholders are used."""
    if not _should_detect_shared_capacity(run_config):
        return None
    return detect_hardware_capacity(reserve_cpu_cores=0, minimum_cpus=1)


def _build_algorithm_instance(algo_config: object) -> Any:
    """Build an Algorithm instance across RLlib API variants."""
    build_algo = getattr(algo_config, "build_algo", None)
    if callable(build_algo):
        return build_algo()
    return algo_config.build()


def _run_training_iterations(
    algo: Any,
    run_config: DreamerRunConfig,
    *,
    checkpoint_dir: Path,
    evaluation_dir: Path,
    result_log_path: Path,
    wandb_run: Any | None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Execute iterations, logging metrics and writing checkpoints."""
    history: list[dict[str, object]] = []
    best_checkpoint: dict[str, object] | None = None
    best_reward_mean: float | int | None = None
    diagnostics_dir = result_log_path.parent / "diagnostics"
    first_nonfinite_iteration: int | None = None
    for iteration in range(1, run_config.experiment.train_iterations + 1):
        result = dict(algo.train())
        reward_mean_raw = _extract_metric(result, "episode_return_mean", "episode_reward_mean")
        reward_mean = reward_mean_raw if _is_finite_scalar(reward_mean_raw) else None
        timesteps_total = _extract_metric(
            result,
            "num_env_steps_sampled_lifetime",
            "timesteps_total",
        )
        reward_mean_status = (
            "finite"
            if reward_mean is not None
            else "nonfinite"
            if reward_mean_raw is not None
            else "missing"
        )
        logger.info(
            "iter={} reward_mean={} reward_mean_raw={} reward_mean_status={} timesteps_total={}",
            iteration,
            reward_mean,
            reward_mean_raw,
            reward_mean_status,
            timesteps_total,
        )
        history.append(
            {
                "iteration": iteration,
                "reward_mean": reward_mean,
                "reward_mean_raw": reward_mean_raw,
                "reward_mean_status": reward_mean_status,
                "timesteps_total": timesteps_total,
            }
        )
        append_jsonl_record(
            result_log_path,
            {
                "ts_utc": datetime.now(UTC).isoformat(),
                "iteration": iteration,
                "reward_mean": reward_mean,
                "reward_mean_raw": reward_mean_raw,
                "reward_mean_status": reward_mean_status,
                "timesteps_total": timesteps_total,
            },
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "iteration": iteration,
                    "reward_mean": reward_mean,
                    "reward_mean_raw": reward_mean_raw,
                    "reward_mean_status_nonfinite": 1 if reward_mean_status == "nonfinite" else 0,
                    "reward_mean_status_missing": 1 if reward_mean_status == "missing" else 0,
                    "timesteps_total": timesteps_total,
                },
                step=iteration,
            )

        if reward_mean_raw is not None and not _is_finite_scalar(reward_mean_raw):
            diagnostics = _build_nonfinite_diagnostics(
                result,
                iteration=iteration,
                reward_mean=reward_mean_raw,
                timesteps_total=timesteps_total,
            )
            diagnostics_path = diagnostics_dir / f"iteration_{iteration:06d}_nonfinite.json"
            _write_json(diagnostics_path, diagnostics)
            history[-1]["nonfinite_diagnostics_path"] = str(diagnostics_path)
            history[-1]["nonfinite_diagnostics"] = diagnostics
            logger.warning(
                "non_finite_reward_mean iteration={} reward_mean={} diagnostics_path={}",
                iteration,
                reward_mean_raw,
                diagnostics_path,
            )
            if first_nonfinite_iteration is None:
                first_nonfinite_iteration = iteration
                if wandb_run is not None:
                    wandb_run.summary["first_nonfinite_reward_iteration"] = iteration
                    wandb_run.summary["first_nonfinite_reward_diagnostics_path"] = str(
                        diagnostics_path
                    )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "reward_mean_nonfinite": 1,
                    },
                    step=iteration,
                )

        if _is_better_reward(reward_mean, best_reward_mean):
            best_reward_mean = reward_mean
            best_path = _save_checkpoint(algo, checkpoint_dir / "best_reward")
            best_checkpoint = {
                "iteration": iteration,
                "reward_mean": reward_mean,
                "timesteps_total": timesteps_total,
                "path": best_path,
            }
            history[-1]["best_checkpoint"] = dict(best_checkpoint)
            logger.info(
                "best_reward_checkpoint_saved iteration={} reward_mean={} path={}",
                iteration,
                reward_mean,
                best_path,
            )
            if wandb_run is not None:
                wandb_run.summary["best_reward_mean"] = reward_mean
                wandb_run.summary["best_reward_iteration"] = iteration
                wandb_run.summary["best_reward_checkpoint_path"] = best_path

        is_periodic_ckpt = (iteration % run_config.experiment.checkpoint_every) == 0
        is_final_ckpt = iteration == run_config.experiment.train_iterations
        if is_periodic_ckpt or is_final_ckpt:
            ckpt_path = _save_checkpoint(algo, checkpoint_dir)
            logger.info("checkpoint_saved={}", ckpt_path)
        should_evaluate = run_config.evaluation.enabled and (
            iteration % run_config.evaluation.every_iterations == 0 or is_final_ckpt
        )
        if should_evaluate:
            eval_summary = _run_periodic_evaluation(
                algo,
                run_config,
                iteration=iteration,
                evaluation_dir=evaluation_dir,
            )
            history[-1]["evaluation"] = eval_summary
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/success_rate": eval_summary["success_rate"],
                        "eval/collision_rate": eval_summary["collision_rate"],
                        "eval/timeout_rate": eval_summary["timeout_rate"],
                    },
                    step=iteration,
                )
    return history, best_checkpoint


def run_training(run_config: DreamerRunConfig) -> int:
    """Execute the DreamerV3 training loop and persist run artifacts."""
    if run_config.ray.disable_uv_run_runtime_env:
        os.environ[_RAY_ENABLE_UV_RUN_RUNTIME_ENV_ENV_VAR] = "0"
    ray, _, register_env = _import_rllib()
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        run_config.experiment.output_root / f"{run_config.experiment.run_id}_{run_stamp}"
    ).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    algo = None
    wandb_run = None
    summary_path = run_dir / "run_summary.json"
    result_log_path = run_dir / "result.jsonl"
    evaluation_dir = run_dir / run_config.evaluation.output_subdir
    history: list[dict[str, object]] = []
    shared_capacity = _detect_shared_capacity(run_config)
    try:
        wandb_run = _init_wandb_tracking(run_config, run_stamp=run_stamp)
        env_name = f"robot_sf_dreamerv3_{run_config.experiment.run_id}"
        register_env(env_name, _make_env_creator(run_config))
        ray.init(**_build_ray_init_kwargs(run_config, capacity=shared_capacity))

        algo_config = _build_algorithm_config(
            run_config,
            env_name=env_name,
            capacity=shared_capacity,
        )
        algo = _build_algorithm_instance(algo_config)
        history, best_checkpoint = _run_training_iterations(
            algo,
            run_config,
            checkpoint_dir=checkpoint_dir,
            evaluation_dir=evaluation_dir,
            result_log_path=result_log_path,
            wandb_run=wandb_run,
        )

        _write_json(
            summary_path,
            {
                "run_id": run_config.experiment.run_id,
                "run_dir": str(run_dir),
                "train_iterations": run_config.experiment.train_iterations,
                "checkpoint_every": run_config.experiment.checkpoint_every,
                "seed": run_config.experiment.seed,
                "config_path": str(run_config.config_path),
                "result_log_path": str(result_log_path),
                "scenario_matrix": (
                    str(run_config.env.scenario_matrix.path)
                    if run_config.env.scenario_matrix is not None
                    else None
                ),
                "evaluation": {
                    "enabled": run_config.evaluation.enabled,
                    "scenario_matrix": (
                        str(run_config.evaluation.scenario_matrix.path)
                        if run_config.evaluation.scenario_matrix is not None
                        else None
                    ),
                    "output_dir": str(evaluation_dir),
                    "episodes": run_config.evaluation.evaluation_episodes,
                    "every_iterations": run_config.evaluation.every_iterations,
                },
                "best_checkpoint": best_checkpoint,
                "history": history,
            },
        )
        if wandb_run is not None:
            wandb_run.summary["run_summary_path"] = str(summary_path)
            wandb_run.summary["checkpoint_dir"] = str(checkpoint_dir)
            if best_checkpoint is not None:
                wandb_run.summary["best_reward_timesteps_total"] = best_checkpoint[
                    "timesteps_total"
                ]
        logger.info("DreamerV3 run summary written to {}", summary_path)
        return 0
    finally:
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        if ray.is_initialized():
            ray.shutdown()
        if wandb_run is not None:
            wandb_run.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train RLlib DreamerV3 on Robot SF drive_state+rays observations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/rllib_dreamerv3/drive_state_rays.yaml"),
        help="Path to DreamerV3 YAML config.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run-id override.",
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=None,
        help="Optional iteration override.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Optional checkpoint cadence override.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=_LOG_LEVEL_CHOICES,
        default=None,
        help="Optional console log level override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print resolved settings without Ray execution.",
    )
    return parser


def _apply_cli_overrides(
    run_config: DreamerRunConfig, args: argparse.Namespace
) -> DreamerRunConfig:
    """Apply explicit CLI overrides on top of loaded YAML settings."""
    experiment = copy.deepcopy(run_config.experiment)
    if args.run_id is not None:
        experiment.run_id = str(args.run_id)
    if args.train_iterations is not None:
        if int(args.train_iterations) < 1:
            raise ValueError("--train-iterations must be >= 1.")
        experiment.train_iterations = int(args.train_iterations)
    if args.checkpoint_every is not None:
        if int(args.checkpoint_every) < 1:
            raise ValueError("--checkpoint-every must be >= 1.")
        experiment.checkpoint_every = int(args.checkpoint_every)
    if args.log_level is not None:
        experiment.log_level = str(args.log_level).upper()
    return DreamerRunConfig(
        config_path=run_config.config_path,
        experiment=experiment,
        ray=run_config.ray,
        env=run_config.env,
        algorithm=run_config.algorithm,
        tracking=run_config.tracking,
        evaluation=run_config.evaluation,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for DreamerV3 RLlib training."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_config = _apply_cli_overrides(load_run_config(args.config), args)
    logger.remove()
    logger.add(sys.stderr, level=run_config.experiment.log_level)
    os.environ["LOGURU_LEVEL"] = run_config.experiment.log_level

    if args.dry_run:
        logger.info(
            "Dry-run config resolved: run_id={} iterations={} checkpoint_every={} output_root={}",
            run_config.experiment.run_id,
            run_config.experiment.train_iterations,
            run_config.experiment.checkpoint_every,
            run_config.experiment.output_root,
        )
        return 0

    return run_training(run_config)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
