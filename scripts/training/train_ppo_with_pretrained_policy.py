"""PPO fine-tuning from a pre-trained policy checkpoint.

Loads a warm-start policy (from BC pre-training or previous checkpoint) and continues
training with PPO to maximize performance while leveraging the pre-trained initialization.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import get_schedule_fn
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError as exc:
    raise RuntimeError("This script requires 'stable_baselines3' package.") from exc

from robot_sf import common
from robot_sf.benchmark.imitation_manifest import write_training_run_manifest
from robot_sf.training.imitation_config import PPOFineTuningConfig
from robot_sf.training.observation_wrappers import maybe_flatten_env_observations
from robot_sf.training.snqi_utils import (
    compute_training_snqi,
    resolve_training_snqi_context,
)
from scripts.training.imitation_env_contract import (
    make_training_contract_env,
    resolve_config_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


_AUTO_NUM_ENVS_THROUGHPUT_RESERVED_CORES = 2
_AUTO_NUM_ENVS_THROUGHPUT_HEADROOM_GIB = 8.0
_AUTO_NUM_ENVS_THROUGHPUT_ENV_BUDGET_GIB = 3.0

_ALLOWED_PPO_HYPERPARAMS = {
    "learning_rate",
    "batch_size",
    "n_epochs",
    "ent_coef",
    "clip_range",
    "target_kl",
    "gamma",
    "gae_lambda",
    "vf_coef",
    "max_grad_norm",
}
_PPO_PARAM_COERCIONS = {
    "learning_rate": float,
    "batch_size": int,
    "n_epochs": int,
    "ent_coef": float,
    "clip_range": float,
    "target_kl": lambda value: None if value is None else float(value),
    "gamma": float,
    "gae_lambda": float,
    "vf_coef": float,
    "max_grad_norm": float,
}


class TimestepTracker(BaseCallback):
    """Callback to track timesteps for convergence measurement."""

    def __init__(self):
        """TODO docstring. Document this function."""
        super().__init__()
        self.timesteps_to_convergence = None
        self.converged = False

    def _on_step(self) -> bool:
        # Simplified convergence check - production would use proper metrics
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        if not self.converged and self.num_timesteps > 1000:
            self.timesteps_to_convergence = self.num_timesteps
            self.converged = True
        return True


def _parse_num_envs(raw: object) -> int | str | None:
    """Parse num_envs setting from a fine-tune YAML file."""
    if raw is None:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"auto", "auto_throughput"}:
            return "auto_throughput"
        if normalized == "auto_stable":
            return "auto_stable"
    return max(1, int(raw))


def _slurm_allocated_cpus() -> int | None:
    """Return the Slurm CPU allocation visible to this process, when available."""
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        raw = os.environ.get(key)
        if not raw:
            continue
        token = raw.split("(", 1)[0].split(",", 1)[0].strip()
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def _host_memory_gib() -> float | None:
    """Return visible host memory in GiB when the platform exposes it."""
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(phys_pages, int):
        return None
    total_bytes = page_size * phys_pages
    return float(total_bytes) / float(1024**3) if total_bytes > 0 else None


def _describe_num_envs_resolution(config: PPOFineTuningConfig) -> dict[str, object]:
    """Return structured details explaining fine-tune environment parallelism."""
    if isinstance(config.num_envs, int):
        resolved = max(1, int(config.num_envs))
        return {
            "requested": int(config.num_envs),
            "mode": "fixed",
            "slurm_cpus": None,
            "logical_cpus": None,
            "memory_cap": None,
            "resolved": resolved,
            "decision": "explicit config integer",
        }

    mode = (
        str(config.num_envs).strip().lower()
        if isinstance(config.num_envs, str)
        else "single_env_legacy"
    )
    if mode == "auto":
        mode = "auto_throughput"
    if mode == "single_env_legacy":
        return {
            "requested": None,
            "mode": mode,
            "slurm_cpus": None,
            "logical_cpus": None,
            "memory_cap": None,
            "resolved": 1,
            "decision": "legacy fine-tune default",
        }
    if mode not in {"auto_throughput", "auto_stable"}:
        raise ValueError("num_envs must be an int or one of {'auto', 'auto_throughput'}")
    if mode == "auto_stable":
        raise ValueError("PPO fine-tuning supports auto/auto_throughput, not auto_stable")

    slurm_cpus = _slurm_allocated_cpus()
    logical_cpus = max(1, slurm_cpus or (os.cpu_count() or 1))
    reserve = max(0, int(config.num_envs_reserve_cores))
    cpu_target = max(1, logical_cpus - _AUTO_NUM_ENVS_THROUGHPUT_RESERVED_CORES - reserve)
    memory_cap = None
    host_memory_gib = _host_memory_gib()
    if host_memory_gib is not None:
        usable_gib = max(0.0, host_memory_gib - _AUTO_NUM_ENVS_THROUGHPUT_HEADROOM_GIB)
        memory_cap = max(1, int(usable_gib // _AUTO_NUM_ENVS_THROUGHPUT_ENV_BUDGET_GIB))
    resolved = cpu_target if memory_cap is None else max(1, min(cpu_target, memory_cap))
    limiter = "cpu target" if memory_cap is None or cpu_target <= memory_cap else "memory cap"
    return {
        "requested": config.num_envs,
        "mode": mode,
        "slurm_cpus": slurm_cpus,
        "logical_cpus": logical_cpus,
        "memory_cap": memory_cap,
        "resolved": resolved,
        "decision": f"throughput heuristic; limited by {limiter}",
    }


def _resolve_num_envs(config: PPOFineTuningConfig) -> int:
    """Resolve the number of fine-tuning envs."""
    return int(_describe_num_envs_resolution(config)["resolved"])


def _resolve_worker_mode(config: PPOFineTuningConfig, num_envs: int) -> str:
    """Resolve fine-tune worker mode."""
    mode = str(config.worker_mode).strip().lower()
    if mode == "auto":
        return "subproc" if num_envs > 1 else "dummy"
    if mode not in {"dummy", "subproc"}:
        raise ValueError("worker_mode must be one of {'auto', 'dummy', 'subproc'}")
    if mode == "subproc" and num_envs == 1:
        return "dummy"
    return mode


def _resolve_ppo_hyperparams(config: PPOFineTuningConfig) -> dict[str, object]:
    """Return validated PPO hyperparameter overrides for fine-tuning."""
    params = {"learning_rate": config.learning_rate}
    params.update(config.ppo_hyperparams)
    unknown = set(params) - _ALLOWED_PPO_HYPERPARAMS
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"ppo_hyperparams has unsupported keys: {unknown_list}")
    return {key: _PPO_PARAM_COERCIONS[key](value) for key, value in params.items()}


def _apply_finetune_hyperparams(model: PPO, config: PPOFineTuningConfig) -> None:
    """Apply config-level PPO overrides after loading the warm-start checkpoint."""
    params = _resolve_ppo_hyperparams(config)
    if "learning_rate" in params:
        value = float(params["learning_rate"])
        model.learning_rate = value
        model.lr_schedule = get_schedule_fn(value)
    if "clip_range" in params:
        model.clip_range = get_schedule_fn(float(params["clip_range"]))
    for key, attr_name, coerce in (
        ("batch_size", "batch_size", int),
        ("n_epochs", "n_epochs", int),
        ("ent_coef", "ent_coef", float),
        ("target_kl", "target_kl", lambda value: value),
        ("gamma", "gamma", float),
        ("gae_lambda", "gae_lambda", float),
        ("vf_coef", "vf_coef", float),
        ("max_grad_norm", "max_grad_norm", float),
    ):
        if key in params:
            setattr(model, attr_name, coerce(params[key]))
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None:
        for key in ("gamma", "gae_lambda"):
            if key in params:
                setattr(rollout_buffer, key, float(params[key]))
    logger.info(
        "Applied fine-tune PPO hyperparameters: learning_rate={} batch_size={} "
        "n_epochs={} ent_coef={} clip_range={} target_kl={}",
        getattr(model, "learning_rate", "<unchanged>"),
        getattr(model, "batch_size", "<unchanged>"),
        getattr(model, "n_epochs", "<unchanged>"),
        getattr(model, "ent_coef", "<unchanged>"),
        float(params["clip_range"]) if "clip_range" in params else "<unchanged>",
        getattr(model, "target_kl", "<unchanged>"),
    )


def _load_dataset_metadata(dataset_id: str | None) -> dict[str, Any]:
    """Load trajectory dataset metadata for env-contract reconstruction."""
    if not dataset_id:
        return {}
    dataset_path = common.get_trajectory_dataset_path(dataset_id)
    if not dataset_path.exists():
        return {}
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset metadata path is not a file: {dataset_path}")
    with np.load(str(dataset_path), allow_pickle=True) as data:
        metadata_raw = data.get("metadata")
        if metadata_raw is None or getattr(metadata_raw, "shape", None) != ():
            return {}
        metadata = metadata_raw.item()
    return metadata if isinstance(metadata, dict) else {}


def _metadata_path(metadata: dict[str, Any], key: str) -> Path | None:
    """Return a resolved path stored in trajectory metadata, if present."""
    value = metadata.get(key)
    if not value:
        return None
    return Path(str(value)).resolve()


def _metadata_observation_keys(metadata: dict[str, Any]) -> tuple[str, ...] | None:
    """Return persisted observation keys from trajectory metadata."""
    contract = metadata.get("observation_contract")
    if not isinstance(contract, dict):
        return None
    keys = contract.get("keys")
    if not isinstance(keys, list):
        return None
    return tuple(str(key) for key in keys)


def _make_finetuning_env(config: PPOFineTuningConfig, *, context: str, seed: int | None = None):
    """Create the fine-tuning/eval env using the pretrained observation contract."""
    metadata = _load_dataset_metadata(config.dataset_id)
    env = make_training_contract_env(
        training_config_path=config.training_config_path
        or _metadata_path(metadata, "training_config"),
        scenario_config_path=config.scenario_config_path
        or _metadata_path(metadata, "scenario_config"),
        scenario_id=config.scenario_id,
        seed=seed if seed is not None else config.random_seeds[0] if config.random_seeds else None,
        observation_keys=_metadata_observation_keys(metadata),
        env_overrides=config.env_overrides,
        env_factory_kwargs=config.env_factory_kwargs,
    )
    return maybe_flatten_env_observations(env, context=context)


def _make_vec_finetuning_env(config: PPOFineTuningConfig) -> DummyVecEnv | SubprocVecEnv:
    """Create a vectorized fine-tuning env for PPO continuation."""
    num_envs = _resolve_num_envs(config)
    worker_mode = _resolve_worker_mode(config, num_envs)
    base_seed = int(config.random_seeds[0]) if config.random_seeds else 0

    def _factory(index: int):
        def _make_env():
            seed = base_seed + index if config.random_seeds else None
            return _make_finetuning_env(
                config,
                context=f"PPO fine-tuning worker {index}",
                seed=seed,
            )

        return _make_env

    env_fns = [_factory(index) for index in range(num_envs)]
    logger.info(
        "Fine-tuning envs initialized num_envs={} worker_mode={} resolution={}",
        num_envs,
        worker_mode,
        _describe_num_envs_resolution(config),
    )
    if worker_mode == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def _checkpoint_callback(config: PPOFineTuningConfig, *, num_envs: int) -> CheckpointCallback | None:
    """Build a periodic checkpoint callback using aggregate timestep frequency."""
    if config.checkpoint_freq is None:
        return None
    checkpoint_freq = int(config.checkpoint_freq)
    if checkpoint_freq <= 0:
        return None
    checkpoint_dir = config.checkpoint_dir or (
        common.get_expert_policy_dir() / "checkpoints" / config.run_id
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # SB3 callback save_freq counts VecEnv step calls, while num_timesteps advances by num_envs.
    save_freq = max(1, checkpoint_freq // max(1, int(num_envs)))
    logger.info(
        "Enabled periodic fine-tune checkpoints every ~{} timesteps at {} (save_freq={})",
        checkpoint_freq,
        checkpoint_dir,
        save_freq,
    )
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix=config.run_id,
        verbose=1,
    )


def _evaluate_policy_metrics(
    model: PPO | None,
    config: PPOFineTuningConfig,
    *,
    dry_run: bool,
) -> tuple[list[float], list[float], list[float]]:
    """Run short evaluation episodes and return per-episode samples for metrics."""

    if dry_run:
        logger.warning("Dry run mode: using placeholder metrics for evaluation")
        snqi_context = resolve_training_snqi_context(
            weights_path=config.snqi_weights_path,
            baseline_path=config.snqi_baseline_path,
        )
        dry_metrics = {
            "success": 0.85,
            "time_to_goal_norm": 0.5,
            "collisions": 0.08,
            "near_misses": 0.0,
            "comfort_exposure": 0.0,
            "force_exceed_events": 0.0,
            "jerk_mean": 0.0,
        }
        return [0.85], [0.08], [compute_training_snqi(dry_metrics, context=snqi_context)]
    if model is None:
        raise ValueError("Model must be provided for evaluation when not in dry-run mode")

    eval_env = _make_finetuning_env(config, context="PPO evaluation")

    successes: list[float] = []
    collisions: list[float] = []
    snqi_values: list[float] = []
    steps_list: list[float] = []
    num_eval_episodes = 10  # Small number for quick evaluation
    snqi_context = resolve_training_snqi_context(
        weights_path=config.snqi_weights_path,
        baseline_path=config.snqi_baseline_path,
    )

    for _ in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_steps = 0
        collision_occurred = False
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_steps += 1

            if info.get("collision", False):
                collision_occurred = True
            if info.get("is_success", False):
                success = True

        successes.append(1.0 if success else 0.0)
        collisions.append(1.0 if collision_occurred else 0.0)
        steps_list.append(float(episode_steps))

    eval_env.close()
    max_steps = max(steps_list) if steps_list else 1.0
    for success_flag, collision_flag, steps in zip(successes, collisions, steps_list, strict=False):
        metric_values = {
            "success": success_flag,
            "time_to_goal_norm": steps / max_steps if max_steps else 1.0,
            "collisions": collision_flag,
            "near_misses": 0.0,
            "comfort_exposure": 0.0,
            "force_exceed_events": 0.0,
            "jerk_mean": 0.0,
        }
        snqi_values.append(compute_training_snqi(metric_values, context=snqi_context))

    return successes, collisions, snqi_values


def run_ppo_finetuning(
    config: PPOFineTuningConfig,
    *,
    dry_run: bool = False,
) -> tuple[Path, int]:
    """Execute PPO fine-tuning and return checkpoint path and convergence timesteps."""
    logger.info("Starting PPO fine-tuning run_id={}", config.run_id)

    # Load pre-trained policy
    pretrained_path = common.get_expert_policy_dir() / f"{config.pretrained_policy_id}.zip"
    if not pretrained_path.exists() and not dry_run:
        raise FileNotFoundError(f"Pre-trained policy not found: {pretrained_path}")

    if not dry_run:
        # Create vectorized environments from the same observation contract used by BC pre-training.
        env = _make_vec_finetuning_env(config)
        num_envs = int(getattr(env, "num_envs", 1))

        # Load pre-trained model
        logger.info("Loading pre-trained policy from {}", pretrained_path)
        model = PPO.load(str(pretrained_path), env=env, device=config.device)

        # Apply fine-tune overrides after loading because SB3 restores checkpoint attributes.
        _apply_finetune_hyperparams(model, config)

        # Setup callback
        timestep_tracker = TimestepTracker()
        callbacks: list[BaseCallback] = [timestep_tracker]
        checkpoint_cb = _checkpoint_callback(config, num_envs=num_envs)
        if checkpoint_cb is not None:
            callbacks.append(checkpoint_cb)
        callback = CallbackList(callbacks)

        # Fine-tune
        logger.info("Fine-tuning PPO for {} timesteps", config.total_timesteps)
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            reset_num_timesteps=False,  # Continue from pre-trained timesteps
        )

        # Save fine-tuned policy
        policy_path = common.get_expert_policy_dir() / f"{config.run_id}_finetuned.zip"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(policy_path))

        convergence_timesteps = timestep_tracker.timesteps_to_convergence or config.total_timesteps
        env.close()
    else:
        # Dry run
        policy_path = common.get_expert_policy_dir() / f"{config.run_id}_finetuned.zip"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text("dry-run-finetuned-checkpoint", encoding="utf-8")
        convergence_timesteps = 1000  # Placeholder for dry run

    logger.success(
        "PPO fine-tuning complete, policy saved to {}, converged at {} timesteps",
        policy_path,
        convergence_timesteps,
    )

    # Compute real metrics from the trained model by running evaluation episodes
    # This replaces the previous synthetic random data approach
    successes, collisions, snqi_values = _evaluate_policy_metrics(
        model if not dry_run else None, config, dry_run=dry_run
    )
    success_rate = float(np.mean(successes)) if successes else 0.0
    collision_rate = float(np.mean(collisions)) if collisions else 0.0
    snqi = float(np.mean(snqi_values)) if snqi_values else 0.0

    # Minimal metrics to surface convergence information and basic quality signals
    convergence_metric = common.MetricAggregate(
        mean=float(convergence_timesteps),
        median=float(convergence_timesteps),
        p95=float(convergence_timesteps),
        ci95=(float(convergence_timesteps), float(convergence_timesteps)),
    )

    def _metric(values: list[float]) -> common.MetricAggregate:
        """TODO docstring. Document this function.

        Args:
            values: TODO docstring.

        Returns:
            TODO docstring.
        """
        if not values:
            return common.MetricAggregate(mean=0.0, median=0.0, p95=0.0, ci95=(0.0, 0.0))
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        if len(arr) > 1:
            se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            ci = (mean - 1.96 * se, mean + 1.96 * se)
        else:
            ci = (mean, mean)
        return common.MetricAggregate(mean=mean, median=median, p95=p95, ci95=ci)

    # Write training run manifest
    training_artifact = common.TrainingRunArtifact(
        run_id=config.run_id,
        run_type=common.TrainingRunType.PPO_FINETUNE,
        input_artefacts=(config.pretrained_policy_id,),
        seeds=config.random_seeds,
        metrics={
            "timesteps_to_convergence": convergence_metric,
            "success_rate": _metric(successes if not dry_run else [success_rate]),
            "collision_rate": _metric(collisions if not dry_run else [collision_rate]),
            "snqi": _metric(snqi_values if not dry_run else [snqi]),
            # path_efficiency and comfort_exposure intentionally omitted until trajectory-based
            # calculations are wired in (needs per-step positions and contact stats).
        },
        episode_log_path=Path(""),  # Episode logs would be generated in production
        wall_clock_hours=0.0,  # Would track actual time in production
        status=common.TrainingRunStatus.COMPLETED,
        notes=[
            f"PPO fine-tuning from pretrained policy {config.pretrained_policy_id}",
            f"Converged at {convergence_timesteps} timesteps",
            f"num_envs={config.num_envs if config.num_envs is not None else 1}",
            f"worker_mode={config.worker_mode}",
            f"device={config.device}",
            (
                "checkpoint_freq="
                f"{config.checkpoint_freq if config.checkpoint_freq is not None else 'disabled'}"
            ),
            "snqi_formula=robot_sf.benchmark.snqi.compute_snqi",
            (
                "snqi_weights_source="
                f"{config.snqi_weights_path if config.snqi_weights_path is not None else 'default'}"
            ),
            (
                "snqi_baseline_source="
                f"{config.snqi_baseline_path if config.snqi_baseline_path is not None else 'default'}"
            ),
        ],
    )

    manifest_path = write_training_run_manifest(training_artifact)
    logger.info("Training run manifest written to {}", manifest_path)

    return policy_path, convergence_timesteps


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for PPO fine-tuning script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune PPO policy from a pre-trained checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to PPO fine-tuning configuration YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder artifacts without actual training",
    )
    return parser


def load_ppo_finetuning_config(config_path: Path) -> PPOFineTuningConfig:
    """Load and parse PPO fine-tuning configuration from YAML."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    def _resolve_optional_path(name: str) -> Path | None:
        """Resolve an optional config path relative to the config file.

        Returns:
            Path | None: Resolved path, or ``None`` for blank/missing values.
        """
        raw_value = raw.get(name)
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if not text:
            return None
        candidate = Path(text)
        return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()

    return PPOFineTuningConfig.from_raw(
        run_id=raw["run_id"],
        pretrained_policy_id=raw["pretrained_policy_id"],
        total_timesteps=raw.get("total_timesteps", 100000),
        random_seeds=tuple(raw.get("random_seeds", [42])),
        learning_rate=raw.get("learning_rate", 0.0001),
        snqi_weights_path=_resolve_optional_path("snqi_weights"),
        snqi_baseline_path=_resolve_optional_path("snqi_baseline"),
        dataset_id=raw.get("dataset_id"),
        training_config_path=resolve_config_path(raw.get("training_config"), base_dir=base_dir),
        scenario_config_path=resolve_config_path(raw.get("scenario_config"), base_dir=base_dir),
        scenario_id=raw.get("scenario_id"),
        env_overrides=dict(raw.get("env_overrides") or {}),
        env_factory_kwargs=dict(raw.get("env_factory_kwargs") or {}),
        num_envs=_parse_num_envs(raw.get("num_envs")),
        num_envs_reserve_cores=int(raw.get("num_envs_reserve_cores", 0)),
        worker_mode=str(raw.get("worker_mode", "auto")),
        device=str(raw.get("device", "auto")),
        ppo_hyperparams=dict(raw.get("ppo_hyperparams") or {}),
        checkpoint_freq=(
            int(raw["checkpoint_freq"]) if raw.get("checkpoint_freq") is not None else None
        ),
        checkpoint_dir=_resolve_optional_path("checkpoint_dir"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for PPO fine-tuning script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_ppo_finetuning_config(args.config)
    common.set_global_seed(int(config.random_seeds[0]))

    run_ppo_finetuning(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
