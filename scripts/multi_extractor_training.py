"""Compare multiple feature extractors with reproducible summaries.

The script orchestrates short PPO training runs for the configured feature
extractors, captures hardware metadata, and emits both JSON and Markdown
summaries describing outcomes. It defaults to a macOS-friendly single-threaded
execution path while enabling multi-process vectorization via configuration.

Test mode is available through the ``ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE``
environment variable; when set, the script runs dramatically shorter smoke
loops to keep CI fast while still exercising the end-to-end workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore
from loguru import logger

from robot_sf.gym_env import environment_factory
from robot_sf.training import (
    ENV_TMP_OVERRIDE,
    ExtractorConfigurationProfile,
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
    collect_hardware_profile,
    make_extractor_directory,
    make_run_directory,
    summarize_metric,
    write_summary_artifacts,
)
from robot_sf.training.multi_extractor_analysis import (
    convergence_timestep,
    generate_figures,
    load_eval_history,
    sample_efficiency_ratio,
)

if TYPE_CHECKING:
    from robot_sf.feature_extractors.config import FeatureExtractorConfig

TEST_MODE_ENV = "ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE"
DEFAULT_CONFIG = Path("configs/scenarios/multi_extractor_default.yaml")

if sys.platform == "darwin":  # Align with macOS spawn requirements
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


@dataclass
class RunSettings:
    """Execution parameters shared across extractors."""

    run_label: str = "multi-extractor"
    worker_mode: str = "single-thread"
    num_envs: int = 1
    total_timesteps: int = 32_000
    eval_freq: int = 4_000
    save_freq: int = 16_000
    n_eval_episodes: int = 5
    device: str = "cpu"
    seed: int | None = None
    output_root: str | None = None
    notes: list[str] = field(default_factory=list)
    baseline_extractor: str | None = None
    convergence_fraction: float = 0.95
    training_diagnostics: bool = False
    diagnostics_start_timestep: int = 0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> RunSettings:
        """Build run settings from the YAML configuration payload.

        Args:
            payload: Parsed scenario configuration mapping.

        Returns:
            Normalized run settings with validation applied to user-provided overrides.
        """
        options = payload.get("run", {})
        settings = cls()
        settings.run_label = str(options.get("run_id", settings.run_label))
        settings.worker_mode = str(options.get("worker_mode", settings.worker_mode))
        settings.num_envs = int(options.get("num_envs", settings.num_envs))
        settings.total_timesteps = int(options.get("total_timesteps", settings.total_timesteps))
        settings.eval_freq = int(options.get("eval_freq", settings.eval_freq))
        settings.save_freq = int(options.get("save_freq", settings.save_freq))
        settings.n_eval_episodes = int(options.get("n_eval_episodes", settings.n_eval_episodes))
        settings.device = str(options.get("device", settings.device)).lower()
        settings.seed = options.get("seed", settings.seed)
        settings.output_root = options.get("output_root", settings.output_root)
        settings.baseline_extractor = options.get("baseline_extractor", settings.baseline_extractor)
        settings.convergence_fraction = float(
            options.get("convergence_fraction", settings.convergence_fraction)
        )
        settings.training_diagnostics = bool(
            options.get("training_diagnostics", settings.training_diagnostics)
        )
        settings.diagnostics_start_timestep = int(
            options.get("diagnostics_start_timestep", settings.diagnostics_start_timestep)
        )

        if settings.worker_mode not in {"single-thread", "vectorized"}:
            raise ValueError(f"Unknown worker_mode: {settings.worker_mode}")
        if settings.worker_mode == "vectorized" and settings.num_envs < 2:
            logger.warning(
                "vectorized worker_mode is being used with num_envs < 2; "
                "this is unusual but allowed for testing or consistency.",
            )
        if settings.device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device requested: {settings.device}")
        if not (0.0 < settings.convergence_fraction <= 1.0):
            raise ValueError("convergence_fraction must be in (0, 1].")
        if settings.diagnostics_start_timestep < 0:
            raise ValueError("diagnostics_start_timestep must be non-negative.")
        return settings

    def effective_timesteps(self, *, test_mode: bool) -> int:
        """Return the effective training steps for normal or test mode.

        Args:
            test_mode: Whether to apply the short-run cap and floor.

        Returns:
            Configured steps normally; in test mode, steps capped at 128 and floored at 8.
        """
        if not test_mode:
            return self.total_timesteps
        return max(8, min(self.total_timesteps, 128))

    def effective_eval_freq(self, *, test_mode: bool) -> int:
        """Return the effective evaluation frequency for normal or test mode.

        Args:
            test_mode: Whether to apply the short-run cap and floor.

        Returns:
            Configured frequency normally; in test mode, frequency capped at 32 and floored at 1.
        """
        if not test_mode:
            return self.eval_freq
        return max(1, min(self.eval_freq, 32))

    def effective_save_freq(self, *, test_mode: bool) -> int:
        """Return the effective checkpoint frequency for normal or test mode.

        Args:
            test_mode: Whether to apply the short-run cap and floor.

        Returns:
            Configured frequency normally; in test mode, frequency capped at 64 and floored at 4.
        """
        if not test_mode:
            return self.save_freq
        return max(4, min(self.save_freq, 64))

    def worker_count(self) -> int:
        """Return the worker count used for hardware metadata collection.

        Returns:
            Configured environment count in vectorized mode; otherwise, one worker.
        """
        return self.num_envs if self.worker_mode == "vectorized" else 1


@dataclass
class RunContext:
    """Container for derived runtime values."""

    run_id: str
    timestamp: str
    created_at: str
    run_dir: Path
    hardware_profile: HardwareProfile
    settings: RunSettings
    test_mode: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a multi-extractor training run.

    Args:
        argv: Optional argument sequence; defaults to the process arguments when None.

    Returns:
        Parsed namespace containing configuration, run ID, output-root, and verbosity options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML configuration describing the run.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the run identifier used in the timestamped output directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the base output directory (defaults to ./tmp/multi_extractor_training).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging detail for debugging.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_logging(verbose: bool) -> None:
    """Configure Loguru output and verbosity for the training script.

    Args:
        verbose: Whether to enable DEBUG logging; otherwise, use INFO.
    """
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def _ensure_spawn_start_method() -> None:
    """Ensure macOS uses the multiprocessing ``spawn`` start method.

    On non-macOS platforms, the current multiprocessing start method is unchanged.
    """
    if sys.platform != "darwin":
        return
    current = mp.get_start_method(allow_none=True)
    if current != "spawn":
        mp.set_start_method("spawn", force=True)


def _get_vec_env_config(
    *, worker_mode: str, num_envs: int
) -> tuple[Any, dict[str, Any] | None, int]:
    """Get vectorized environment configuration based on worker mode and platform."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if worker_mode == "vectorized":
        vec_env_cls = SubprocVecEnv
        vec_kwargs: dict[str, Any] | None = {"start_method": "spawn"}
        n_envs = num_envs
    else:
        vec_env_cls = DummyVecEnv
        vec_kwargs = None
        n_envs = 1

    return vec_env_cls, vec_kwargs, n_envs


def load_configuration(
    config_path: Path,
) -> tuple[RunSettings, list[ExtractorConfigurationProfile]]:
    """Load run settings and extractor profiles from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Normalized run settings and extractor profiles sorted by priority.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration root or extractor definitions are invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")

    settings = RunSettings.from_mapping(payload)

    extractors_raw = payload.get("extractors", [])
    if not extractors_raw:
        raise ValueError("No extractors defined in configuration")

    profiles: list[ExtractorConfigurationProfile] = []
    for item in extractors_raw:
        name = item.get("name")
        if not name:
            raise ValueError("Extractor entries require a 'name'")
        profile = ExtractorConfigurationProfile(
            name=name,
            description=item.get("description"),
            parameters=item.get("parameters"),
            expected_resources=item.get("expected_resources", "cpu"),
            priority=item.get("priority"),
            preset=item.get("preset"),
        )
        profiles.append(profile)

    profiles.sort(key=lambda profile: profile.priority if profile.priority is not None else 0)
    return settings, profiles


def should_use_test_mode(environment: dict[str, str]) -> bool:
    """Return whether the environment enables the short test mode.

    Args:
        environment: Environment-variable mapping to inspect for ``TEST_MODE_ENV``.

    Returns:
        True for ``1``, ``true``, ``yes``, or ``on`` (case-insensitive); otherwise False.
    """
    value = environment.get(TEST_MODE_ENV)
    if not value:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def resolve_run_context(
    *,
    settings: RunSettings,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> RunContext:
    """Create the derived runtime context for a training run.

    Args:
        settings: Parsed and validated run settings.
        args: Parsed CLI arguments, including optional run ID and output-root overrides.
        environment: Environment variables used for output-root resolution and test-mode detection.

    Returns:
        Context containing timestamps, output directory, hardware profile, settings, and test mode.
    """
    run_id = args.run_id or settings.run_label
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    created_at = now.isoformat()

    env_override = dict(environment)
    if args.output_root:
        env_override[ENV_TMP_OVERRIDE] = str(args.output_root)
    elif settings.output_root:
        env_override[ENV_TMP_OVERRIDE] = str(settings.output_root)

    run_dir = make_run_directory(run_id, env=env_override, timestamp=timestamp).resolve()
    test_mode = should_use_test_mode(environment)
    hardware_profile = collect_hardware_profile(
        worker_count=settings.worker_count(),
        skip_gpu=test_mode,
    )

    return RunContext(
        run_id=run_id,
        timestamp=timestamp,
        created_at=created_at,
        run_dir=run_dir,
        hardware_profile=hardware_profile,
        settings=settings,
        test_mode=test_mode,
    )


def _resolve_feature_config(profile: ExtractorConfigurationProfile) -> FeatureExtractorConfig:
    """Build the feature-extractor configuration for a profile.

    Args:
        profile: Extractor profile naming a preset and optional parameter overrides.

    Returns:
        Configuration created from the selected preset and merged parameters.

    Raises:
        ValueError: If the selected preset is not defined on ``FeatureExtractorPresets``.
    """
    from robot_sf.feature_extractors.config import FeatureExtractorPresets

    preset_name = profile.preset or profile.name
    try:
        factory = getattr(FeatureExtractorPresets, preset_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown feature extractor preset: {preset_name}") from exc

    config = factory()
    config.params.update(profile.merged_parameters())
    return config


def _make_ppo_model(
    *,
    train_env: Any,
    tensorboard_log: Path,
    policy_kwargs: dict[str, Any],
    device: str,
    diagnostics_path: Path | None,
    diagnostics_start_timestep: int,
) -> Any:
    """Create a normal PPO model or the diagnostics-enabled PPO variant."""
    from stable_baselines3 import PPO

    if diagnostics_path is None:
        return PPO(
            "MultiInputPolicy",
            train_env,
            tensorboard_log=str(tensorboard_log),
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
        )

    from robot_sf.training.ppo_diagnostics import DiagnosticPPO

    return DiagnosticPPO(
        "MultiInputPolicy",
        train_env,
        tensorboard_log=str(tensorboard_log),
        diagnostics_path=diagnostics_path,
        diagnostics_start_timestep=diagnostics_start_timestep,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
    )


def _gpu_available() -> bool:
    """Return whether PyTorch reports CUDA availability.

    Returns:
        True when ``torch.cuda.is_available()`` succeeds and reports availability; otherwise False.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is an install requirement
        return False
    return bool(torch.cuda.is_available())


def _skip_record(
    *,
    profile: ExtractorConfigurationProfile,
    context: RunContext,
    artifacts: dict[str, str],
    reason: str,
) -> ExtractorRunRecord:
    """Create a skipped extractor result with no training metrics.

    Args:
        profile: Extractor profile that was not run.
        context: Resolved run context supplying hardware and worker settings.
        artifacts: Artifact paths already allocated for the extractor.
        reason: Explanation for skipping the extractor.

    Returns:
        Run record marked ``"skipped"`` with zero duration and training steps.
    """
    now = datetime.now(UTC).isoformat()
    return ExtractorRunRecord(
        config_name=profile.name,
        status="skipped",
        start_time=now,
        end_time=now,
        duration_seconds=0.0,
        hardware_profile=context.hardware_profile,
        worker_mode=context.settings.worker_mode,
        training_steps=0,
        metrics={},
        artifacts=artifacts,
        reason=reason,
    )


def _simulate_extractor_run(
    *,
    profile: ExtractorConfigurationProfile,
    context: RunContext,
    extractor_dir: Path,
    artifacts: dict[str, str],
) -> ExtractorRunRecord:
    """Record a simulated successful test-mode run without launching training.

    Args:
        profile: Extractor profile being simulated.
        context: Resolved run context supplying output and run metadata.
        extractor_dir: Per-extractor output directory for the metrics file.
        artifacts: Mutable artifact map to augment with the relative metrics path.

    Returns:
        Successful run record with zero-valued metrics and test-mode training steps.
    """
    logger.debug("Simulating extractor run", extractor=profile.name)
    start_time = datetime.now(UTC).isoformat()
    metrics = {
        "best_mean_reward": 0.0,
        "last_mean_reward": 0.0,
        "total_parameters": 0.0,
        "trainable_parameters": 0.0,
    }
    metrics_path = extractor_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    artifacts["metrics"] = str(metrics_path.relative_to(context.run_dir))

    return ExtractorRunRecord(
        config_name=profile.name,
        status="success",
        start_time=start_time,
        end_time=datetime.now(UTC).isoformat(),
        duration_seconds=0.0,
        hardware_profile=context.hardware_profile,
        worker_mode=context.settings.worker_mode,
        training_steps=context.settings.effective_timesteps(test_mode=True),
        metrics=metrics,
        artifacts=artifacts,
        reason=None,
    )


def _run_sb3_training(
    *,
    profile: ExtractorConfigurationProfile,
    context: RunContext,
    extractor_dir: Path,
    artifacts: dict[str, str],
    start_time_iso: str,
    start_wall: float,
) -> ExtractorRunRecord:
    """Train one feature extractor with SB3 PPO and collect run artifacts.

    Args:
        profile: Extractor profile being trained.
        context: Resolved run context with hardware, settings, and output paths.
        extractor_dir: Directory where model, eval, checkpoint, and metric artifacts are written.
        artifacts: Mutable artifact map carried into the run record.
        start_time_iso: ISO timestamp captured before the run started.
        start_wall: Monotonic wall-clock timestamp captured before the run started.

    Returns:
        Extractor run record with status, metrics, artifacts, and failure reason when applicable.
    """
    try:
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv

        config = _resolve_feature_config(profile)

        total_timesteps = context.settings.effective_timesteps(test_mode=context.test_mode)
        eval_freq = context.settings.effective_eval_freq(test_mode=context.test_mode)
        save_freq = context.settings.effective_save_freq(test_mode=context.test_mode)

        def env_factory() -> Any:
            """Create a robot environment using the configured seed.

            Returns:
                Robot environment initialized through ``environment_factory``.
            """
            return environment_factory.make_robot_env(seed=context.settings.seed)

        _ensure_spawn_start_method()

        vec_env_cls, vec_kwargs, n_envs = _get_vec_env_config(
            worker_mode=context.settings.worker_mode,
            num_envs=context.settings.num_envs,
        )

        train_env = make_vec_env(
            env_factory,
            n_envs=n_envs,
            seed=context.settings.seed,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_kwargs,
        )

        eval_env = make_vec_env(
            env_factory,
            n_envs=1,
            seed=context.settings.seed,
            vec_env_cls=DummyVecEnv,
        )

        diagnostics_path = (
            extractor_dir / "training_diagnostics.jsonl"
            if context.settings.training_diagnostics
            else None
        )

        model = _make_ppo_model(
            train_env=train_env,
            tensorboard_log=extractor_dir / "tensorboard",
            policy_kwargs=config.get_policy_kwargs(),
            device=context.settings.device,
            diagnostics_path=diagnostics_path,
            diagnostics_start_timestep=context.settings.diagnostics_start_timestep,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(extractor_dir / "best_model"),
            log_path=str(extractor_dir / "eval_logs"),
            eval_freq=max(1, eval_freq // n_envs),
            n_eval_episodes=context.settings.n_eval_episodes,
            deterministic=True,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, save_freq // n_envs),
            save_path=str(extractor_dir / "checkpoints"),
            name_prefix=f"ppo_{profile.name}",
        )

        callbacks = CallbackList([eval_callback, checkpoint_callback])
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)

        model_path = extractor_dir / "final_model"
        model.save(model_path)
        final_model_zip = model_path.with_suffix(".zip")
        if final_model_zip.exists():
            artifacts["final_model"] = str(final_model_zip.relative_to(context.run_dir))

        best_model_zip = extractor_dir / "best_model" / "best_model.zip"
        if best_model_zip.exists():
            artifacts["best_model"] = str(best_model_zip.relative_to(context.run_dir))

        checkpoints_dir = extractor_dir / "checkpoints"
        if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
            artifacts["checkpoints"] = str(checkpoints_dir.relative_to(context.run_dir))

        if diagnostics_path is not None and diagnostics_path.exists():
            artifacts["training_diagnostics"] = str(diagnostics_path.relative_to(context.run_dir))

        total_parameters = sum(p.numel() for p in model.policy.parameters())
        trainable_parameters = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

        metrics = {
            "best_mean_reward": float(getattr(eval_callback, "best_mean_reward", 0.0) or 0.0),
            "last_mean_reward": float(getattr(eval_callback, "last_mean_reward", 0.0) or 0.0),
            "total_timesteps": float(total_timesteps),
            "total_parameters": float(total_parameters),
            "trainable_parameters": float(trainable_parameters),
        }
        metrics_path = extractor_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        artifacts["metrics"] = str(metrics_path.relative_to(context.run_dir))

        status = "success"
        reason = None
    except Exception as exc:  # pragma: no cover - surfaced in integration tests
        logger.exception("Extractor failed", name=profile.name)
        metrics = {}
        status = "failed"
        reason = str(exc)
        metrics_path = extractor_dir / "metrics.json"
    finally:
        if "train_env" in locals():
            train_env.close()
        if "eval_env" in locals():
            eval_env.close()

    end_time_iso = datetime.now(UTC).isoformat()
    duration = time.perf_counter() - start_wall

    if status != "success" and not metrics_path.exists():
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        artifacts.setdefault("metrics", str(metrics_path.relative_to(context.run_dir)))

    return ExtractorRunRecord(
        config_name=profile.name,
        status=status,
        start_time=start_time_iso,
        end_time=end_time_iso,
        duration_seconds=duration,
        hardware_profile=context.hardware_profile,
        worker_mode=context.settings.worker_mode,
        training_steps=int(metrics.get("total_timesteps", 0)),
        metrics={k: v for k, v in metrics.items() if k != "total_timesteps"},
        artifacts=artifacts,
        reason=reason,
    )


def _determine_skip_reason(
    profile: ExtractorConfigurationProfile, context: RunContext
) -> str | None:
    """Determine whether an extractor lacks its required CUDA resources.

    Args:
        profile: Extractor profile with expected resource requirements.
        context: Resolved run context containing device settings.

    Returns:
        Skip explanation when CUDA is requested or required but unavailable; otherwise None.
    """
    settings = context.settings
    gpu_available = _gpu_available()

    # Check GPU availability for CUDA requirements regardless of test mode
    if settings.device == "cuda" and not gpu_available:
        return "CUDA device requested but torch.cuda.is_available() is False"
    if profile.expected_resources == "gpu" and not gpu_available:
        return "Extractor expects GPU resources but CUDA is unavailable"

    # In test mode, allow CUDA extractors to run if CUDA is available
    # This ensures GPU-enabled CI can exercise the full workflow
    return None


def train_extractor(
    *,
    profile: ExtractorConfigurationProfile,
    context: RunContext,
) -> ExtractorRunRecord:
    """Run one configured extractor and return its result record.

    Args:
        profile: Extractor profile to train or simulate.
        context: Resolved run context controlling mode, resources, paths, and settings.

    Returns:
        Result record for a skipped, simulated, or Stable-Baselines3 training attempt.
    """
    settings = context.settings
    extractor_dir = make_extractor_directory(context.run_dir, profile.name)
    artifacts: dict[str, str] = {"extractor_dir": str(extractor_dir.relative_to(context.run_dir))}

    logger.info("Starting extractor", name=profile.name, worker_mode=settings.worker_mode)

    skip_reason = _determine_skip_reason(profile, context)
    if skip_reason:
        logger.warning(skip_reason, extractor=profile.name)
        return _skip_record(
            profile=profile, context=context, artifacts=artifacts, reason=skip_reason
        )

    if context.test_mode:
        return _simulate_extractor_run(
            profile=profile,
            context=context,
            extractor_dir=extractor_dir,
            artifacts=artifacts,
        )

    start_wall = time.perf_counter()
    start_time_iso = datetime.now(UTC).isoformat()

    return _run_sb3_training(
        profile=profile,
        context=context,
        extractor_dir=extractor_dir,
        artifacts=artifacts,
        start_time_iso=start_time_iso,
        start_wall=start_wall,
    )


def _load_histories(
    records: list[ExtractorRunRecord],
    context: RunContext,
) -> dict[str, tuple[object | None, Path]]:
    """Load evaluation histories for extractor records.

    Args:
        records: Extractor records whose artifact directories identify evaluation logs.
        context: Run context providing the root output directory.

    Returns:
        Mapping from extractor name to its history (or None) and extractor directory.
    """
    history_map: dict[str, tuple[object | None, Path]] = {}
    for record in records:
        rel = record.artifacts.get("extractor_dir")
        extractor_dir = context.run_dir / rel if rel else context.run_dir
        history_map[record.config_name] = (load_eval_history(extractor_dir), extractor_dir)
    return history_map


def _enrich_records_with_analysis(
    records: list[ExtractorRunRecord],
    context: RunContext,
) -> tuple[float, float]:
    """Enrich extractor records with analysis metrics and generated figures.

    Args:
        records: Extractor records to update in place.
        context: Run context supplying baseline selection, thresholds, and output paths.

    Returns:
        Baseline target reward and baseline convergence timestep used for comparison.
    """
    if not records:
        return 0.0, 0.0
    history_map = _load_histories(records, context)
    baseline_name = context.settings.baseline_extractor or (
        records[0].config_name if records else ""
    )
    baseline_record = next((r for r in records if r.config_name == baseline_name), records[0])
    baseline_history, _ = history_map.get(baseline_record.config_name, (None, context.run_dir))
    total_timesteps = context.settings.effective_timesteps(test_mode=context.test_mode)

    baseline_best = float(baseline_record.metrics.get("best_mean_reward", 0.0) or 0.0)
    baseline_target = baseline_best
    baseline_conv = convergence_timestep(
        baseline_history,
        target_reward=baseline_best * context.settings.convergence_fraction,
        default_timesteps=float(total_timesteps),
    )

    for record in records:
        history, extractor_dir = history_map.get(record.config_name, (None, context.run_dir))
        best_reward = float(record.metrics.get("best_mean_reward", 0.0) or 0.0)
        conv_target = (
            best_reward * context.settings.convergence_fraction if best_reward > 0 else 0.0
        )
        conv_ts = convergence_timestep(
            history,
            target_reward=conv_target,
            default_timesteps=float(total_timesteps),
        )
        sample_ts = convergence_timestep(
            history,
            target_reward=baseline_target,
            default_timesteps=float(total_timesteps),
        )
        sample_ratio = sample_efficiency_ratio(
            baseline_timestep=baseline_conv if baseline_conv else float(total_timesteps),
            candidate_timestep=sample_ts,
        )
        record.metrics.update(
            {
                "convergence_timestep": float(conv_ts),
                "sample_efficiency_timestep": float(sample_ts),
                "sample_efficiency_ratio": float(sample_ratio),
                "baseline_target_reward": float(baseline_target),
                "delta_best_reward_vs_baseline": float(best_reward - baseline_target),
            }
        )
        figure_paths = generate_figures(history, extractor_dir / "figures", record.config_name)
        for key, path in figure_paths.items():
            record.artifacts[key] = str(path.relative_to(context.run_dir))

    return baseline_target, baseline_conv


def compute_aggregate_metrics(
    records: list[ExtractorRunRecord],
    *,
    baseline_target: float,
    baseline_convergence: float,
) -> dict[str, float]:
    """Compute aggregate metrics across extractor run records.

    Args:
        records: Extractor run records to summarize.
        baseline_target: Baseline best reward used as the sample-efficiency target.
        baseline_convergence: Baseline convergence timestep included in the output.

    Returns:
        Mapping of run counts, reward/time summaries, convergence statistics, and baseline values.
    """
    total_time = sum(record.duration_seconds or 0.0 for record in records)
    best_rewards = [
        float(v)
        for record in records
        if record.metrics and isinstance(v := record.metrics.get("best_mean_reward"), (int, float))
    ]
    completed = sum(1 for record in records if record.status == "success")
    skipped = sum(1 for record in records if record.status == "skipped")
    failed = sum(1 for record in records if record.status == "failed")

    best_reward = max(best_rewards) if best_rewards else 0.0
    convergence_stats = summarize_metric(
        record.metrics.get("convergence_timestep", math.nan) for record in records
    )
    sample_eff_stats = summarize_metric(
        record.metrics.get("sample_efficiency_ratio", math.nan) for record in records
    )

    return {
        "completed_runs": float(completed),
        "skipped_runs": float(skipped),
        "failed_runs": float(failed),
        "best_mean_reward": float(best_reward or 0.0),
        "total_wall_time": float(total_time),
        "avg_convergence_timestep": convergence_stats["mean"],
        "median_convergence_timestep": convergence_stats["median"],
        "avg_sample_efficiency_ratio": sample_eff_stats["mean"],
        "median_sample_efficiency_ratio": sample_eff_stats["median"],
        "baseline_target_reward": float(baseline_target or 0.0),
        "baseline_convergence_timestep": float(baseline_convergence or 0.0),
    }


def build_summary(context: RunContext, records: list[ExtractorRunRecord]) -> TrainingRunSummary:
    """Build the summary for a multi-extractor training run.

    Args:
        context: Resolved run context with metadata and output directory.
        records: Per-extractor result records to analyze and include.

    Returns:
        Training summary containing hardware, extractor results, aggregate metrics, and notes.
    """
    notes = [record.reason for record in records if record.reason]
    baseline_target, baseline_convergence = _enrich_records_with_analysis(records, context)
    aggregate = compute_aggregate_metrics(
        records,
        baseline_target=baseline_target,
        baseline_convergence=baseline_convergence,
    )

    return TrainingRunSummary(
        run_id=context.run_dir.name,
        created_at=context.created_at,
        output_root=str(context.run_dir),
        hardware_overview=[context.hardware_profile],
        extractor_results=records,
        aggregate_metrics=aggregate,
        notes=notes or None,
    )


def write_legacy_results(
    context: RunContext, records: list[ExtractorRunRecord], *, destination: Path
) -> None:
    """Emit a compatibility JSON matching the historical `complete_results.json`."""

    metadata = {
        "run_dir": str(context.run_dir),
        "run_id": context.run_dir.name,
        "created_at": context.created_at,
        "worker_mode": context.settings.worker_mode,
        "num_envs": context.settings.num_envs,
        "total_timesteps_per_extractor": context.settings.effective_timesteps(
            test_mode=context.test_mode
        ),
        "n_eval_episodes": context.settings.n_eval_episodes,
        "device": context.settings.device,
        "test_mode": context.test_mode,
        "notes": [record.reason for record in records if record.reason],
        "total_time": sum(record.duration_seconds or 0.0 for record in records),
    }

    results_block: dict[str, Any] = {}
    for record in records:
        entry = {
            "completed": record.status == "success",
            "status": record.status,
            "training_time": record.duration_seconds,
            "total_timesteps": record.training_steps,
            "best_reward": record.metrics.get("best_mean_reward"),
            "final_reward": record.metrics.get("last_mean_reward"),
            "total_parameters": record.metrics.get("total_parameters"),
            "trainable_parameters": record.metrics.get("trainable_parameters"),
            "artifacts": record.artifacts,
            "reason": record.reason,
        }
        results_block[record.config_name] = entry

    payload = {
        "comparison_metadata": metadata,
        "results": results_block,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the multi-extractor training CLI and write summary artifacts.

    Args:
        argv: Optional command-line arguments; defaults to process arguments when None.

    Returns:
        Zero when configuration, training, analysis, and artifact writing complete.
    """
    args = parse_args(argv)
    _configure_logging(verbose=args.verbose)

    settings, profiles = load_configuration(Path(args.config))
    context = resolve_run_context(settings=settings, args=args, environment=dict(os.environ))

    logger.info(
        "Multi-extractor run prepared",
        run_id=context.run_id,
        timestamp=context.timestamp,
        output=str(context.run_dir),
        worker_mode=settings.worker_mode,
        num_envs=settings.num_envs,
        test_mode=context.test_mode,
    )

    records: list[ExtractorRunRecord] = []
    for profile in profiles:
        record = train_extractor(profile=profile, context=context)
        records.append(record)

    summary = build_summary(context, records)
    artifacts = write_summary_artifacts(summary=summary, destination=context.run_dir)

    # Maintain compatibility with historical automation expecting complete_results.json
    write_legacy_results(context, records, destination=context.run_dir / "complete_results.json")
    write_legacy_results(
        context,
        records,
        destination=context.run_dir.parent / "complete_results.json",
    )

    logger.success(
        "Summary artifacts written",
        json=str(artifacts["json"]),
        markdown=str(artifacts["markdown"]),
    )
    logger.info(
        "Aggregate metrics",
        metrics=summary.aggregate_metrics,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
