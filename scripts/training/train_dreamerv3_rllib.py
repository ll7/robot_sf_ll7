"""Train RLlib DreamerV3 on Robot SF drive_state+rays observations."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.common.hardware import detect_hardware_capacity, recommend_env_runners
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.training.rllib_env_wrappers import DEFAULT_FLATTEN_KEYS, wrap_for_dreamerv3

_LOG_LEVEL_CHOICES = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
_TOP_LEVEL_KEYS = {"experiment", "ray", "env", "algorithm", "tracking"}
_EXPERIMENT_KEYS = {
    "run_id",
    "output_root",
    "train_iterations",
    "checkpoint_every",
    "seed",
    "log_level",
}
_RAY_KEYS = {"address", "local_mode", "include_dashboard", "num_cpus", "num_gpus"}
_ENV_KEYS = {"flatten_observation", "flatten_keys", "normalize_actions", "factory_kwargs", "config"}
_ALGORITHM_KEYS = {"framework", "training", "env_runners", "resources", "learners", "api_stack"}
_TRACKING_KEYS = {"wandb"}
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


@dataclass(slots=True)
class EnvSettings:
    """Environment construction settings for RLlib workers."""

    flatten_observation: bool
    flatten_keys: tuple[str, ...]
    normalize_actions: bool
    factory_kwargs: dict[str, object]
    config_overrides: dict[str, object]


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
class DreamerRunConfig:
    """Complete validated training configuration."""

    experiment: ExperimentSettings
    ray: RaySettings
    env: EnvSettings
    algorithm: AlgorithmSettings
    tracking: TrackingSettings


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


def _resolve_config_path(raw_value: object, *, field_name: str) -> Path:
    """Resolve a path-like config value to an absolute path."""
    text_value = str(raw_value).strip()
    if not text_value:
        raise ValueError(f"{field_name} cannot be empty.")
    path_value = Path(text_value)
    if not path_value.is_absolute():
        path_value = (Path.cwd() / path_value).resolve()
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
    )


def _parse_env_settings(payload: dict[str, object]) -> EnvSettings:
    """Parse and validate the environment section."""
    env_raw = _ensure_mapping(payload.get("env"), field_name="env")
    _ensure_known_keys(env_raw, field_name="env", allowed=_ENV_KEYS)

    flatten_keys_raw = env_raw.get("flatten_keys", list(DEFAULT_FLATTEN_KEYS))
    if not isinstance(flatten_keys_raw, list | tuple):
        raise ValueError("env.flatten_keys must be a list or tuple of strings.")

    return EnvSettings(
        flatten_observation=bool(env_raw.get("flatten_observation", True)),
        flatten_keys=tuple(str(key) for key in flatten_keys_raw),
        normalize_actions=bool(env_raw.get("normalize_actions", True)),
        factory_kwargs=_ensure_mapping(
            env_raw.get("factory_kwargs"), field_name="env.factory_kwargs"
        ),
        config_overrides=_ensure_mapping(env_raw.get("config"), field_name="env.config"),
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
        wandb_raw.get("dir", "output/wandb"), field_name="tracking.wandb.dir"
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
        experiment=_parse_experiment_settings(payload),
        ray=_parse_ray_settings(payload),
        env=_parse_env_settings(payload),
        algorithm=_parse_algorithm_settings(payload),
        tracking=_parse_tracking_settings(payload),
    )


def _apply_nested_overrides(
    config_obj: object,
    overrides: dict[str, object],
    *,
    context_name: str = "env.config",
) -> None:
    """Recursively apply dict overrides onto dataclass-like objects."""
    for key, value in overrides.items():
        if not hasattr(config_obj, key):
            raise ValueError(f"Unknown {context_name} override field: '{key}'")
        current_attr = getattr(config_obj, key)
        if isinstance(value, dict) and (
            hasattr(current_attr, "__dict__") or is_dataclass(current_attr)
        ):
            _apply_nested_overrides(current_attr, value, context_name=f"{context_name}.{key}")
        else:
            setattr(config_obj, key, value)


def _create_env_template(env_settings: EnvSettings) -> RobotSimulationConfig:
    """Build the base RobotSimulationConfig used by all RLlib workers."""
    config = RobotSimulationConfig()
    _apply_nested_overrides(config, env_settings.config_overrides, context_name="env.config")
    # Force vector-observation mode for the drive_state+rays DreamerV3 pipeline.
    config.use_image_obs = False
    config.include_grid_in_observation = False
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

    def _creator(worker_env_config: dict[str, object] | None = None) -> Any:
        worker_overrides = {
            key: value
            for key, value in dict(worker_env_config or {}).items()
            if key not in _IGNORED_ENV_CONTEXT_KEYS
        }
        worker_config = copy.deepcopy(template)
        _apply_nested_overrides(worker_config, worker_overrides, context_name="worker.config")
        env = make_robot_env(config=worker_config, **factory_kwargs)
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
    return config_obj


def _is_auto(value: object) -> bool:
    """Return True when value is the special 'auto' token."""
    return isinstance(value, str) and value.strip().lower() == _AUTO_TOKEN


def _resolve_auto_overrides(
    run_config: DreamerRunConfig,
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
    env_runner_payload, resources_payload, _ = _resolve_auto_overrides(run_config)
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


def _build_ray_init_kwargs(run_config: DreamerRunConfig) -> dict[str, object]:
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
    needs_auto = _is_auto(run_config.ray.num_cpus) or _is_auto(run_config.ray.num_gpus)
    capacity = detect_hardware_capacity(reserve_cpu_cores=0, minimum_cpus=1) if needs_auto else None
    if _is_auto(run_config.ray.num_cpus):
        if capacity is None:
            raise RuntimeError("Hardware capacity detection failed for auto CPU resolution.")
        ray_kwargs["num_cpus"] = capacity.usable_cpus
    elif run_config.ray.num_cpus is not None:
        ray_kwargs["num_cpus"] = run_config.ray.num_cpus
    if _is_auto(run_config.ray.num_gpus):
        if capacity is None:
            raise RuntimeError("Hardware capacity detection failed for auto GPU resolution.")
        ray_kwargs["num_gpus"] = capacity.visible_gpus
    elif run_config.ray.num_gpus is not None:
        ray_kwargs["num_gpus"] = run_config.ray.num_gpus
    return ray_kwargs


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
    wandb_run: Any | None,
) -> list[dict[str, object]]:
    """Execute iterations, logging metrics and writing checkpoints."""
    history: list[dict[str, object]] = []
    for iteration in range(1, run_config.experiment.train_iterations + 1):
        result = dict(algo.train())
        reward_mean = _extract_metric(result, "episode_return_mean", "episode_reward_mean")
        timesteps_total = _extract_metric(
            result,
            "num_env_steps_sampled_lifetime",
            "timesteps_total",
        )
        logger.info(
            "iter={} reward_mean={} timesteps_total={}",
            iteration,
            reward_mean,
            timesteps_total,
        )
        history.append(
            {
                "iteration": iteration,
                "reward_mean": reward_mean,
                "timesteps_total": timesteps_total,
            }
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "iteration": iteration,
                    "reward_mean": reward_mean,
                    "timesteps_total": timesteps_total,
                },
                step=iteration,
            )

        is_periodic_ckpt = (iteration % run_config.experiment.checkpoint_every) == 0
        is_final_ckpt = iteration == run_config.experiment.train_iterations
        if is_periodic_ckpt or is_final_ckpt:
            ckpt_path = _save_checkpoint(algo, checkpoint_dir)
            logger.info("checkpoint_saved={}", ckpt_path)
    return history


def run_training(run_config: DreamerRunConfig) -> int:
    """Execute the DreamerV3 training loop and persist run artifacts."""
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
    history: list[dict[str, object]] = []
    try:
        wandb_run = _init_wandb_tracking(run_config, run_stamp=run_stamp)
        env_name = f"robot_sf_dreamerv3_{run_config.experiment.run_id}"
        register_env(env_name, _make_env_creator(run_config))
        ray.init(**_build_ray_init_kwargs(run_config))

        algo_config = _build_algorithm_config(run_config, env_name=env_name)
        algo = _build_algorithm_instance(algo_config)
        history = _run_training_iterations(
            algo,
            run_config,
            checkpoint_dir=checkpoint_dir,
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
                "history": history,
            },
        )
        if wandb_run is not None:
            wandb_run.summary["run_summary_path"] = str(summary_path)
            wandb_run.summary["checkpoint_dir"] = str(checkpoint_dir)
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
        experiment=experiment,
        ray=run_config.ray,
        env=run_config.env,
        algorithm=run_config.algorithm,
        tracking=run_config.tracking,
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
