"""Expert PPO training workflow entry point.

The script loads a unified training configuration, orchestrates PPO expert
training via Stable-Baselines3, evaluates the resulting policy, and persists
manifests/artefacts using the imitation helpers introduced for this feature.

To keep tests fast and deterministic, the implementation supports a ``--dry-run``
mode that skips heavy PPO optimisation while still exercising the manifest and
artifact pipeline. Production invocations omit that flag to execute full
training.

Example usage:
```bash
uv run python scripts/training/train_ppo.py \
    --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
    --log-level WARNING
```
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from loguru import logger

try:  # pragma: no cover - imported lazily in tests when available
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.logger import configure as configure_sb3_logger
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError as exc:  # pragma: no cover - surfaced during runtime usage
    raise RuntimeError(
        "Stable-Baselines3 must be installed to run expert PPO training.",
    ) from exc

from robot_sf import common
from robot_sf.benchmark.imitation_manifest import (
    write_expert_policy_manifest,
    write_training_run_manifest,
)
from robot_sf.common.artifact_paths import get_artifact_category_path, get_imitation_report_dir
from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.models import resolve_model_path
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.training.imitation_config import (
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
)
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)
from robot_sf.training.scenario_sampling import (
    ScenarioSampler,
    ScenarioSwitchingEnv,
    scenario_id_from_definition,
)
from robot_sf.training.snqi_utils import (
    TrainingSNQIContext,
    compute_training_snqi,
    resolve_training_snqi_context,
)

MetricSamples = dict[str, list[float]]


_ORIGINAL_LOGURU_LEVEL = os.environ.get("LOGURU_LEVEL")

_DEFAULT_PPO_HYPERPARAMS: dict[str, object] = {
    "learning_rate": 1e-4,
    "batch_size": 256,
    "n_epochs": 4,
    "ent_coef": 0.01,
    "clip_range": 0.1,
    "target_kl": 0.02,
}
_ALLOWED_PPO_HYPERPARAMS = {
    "learning_rate",
    "batch_size",
    "n_epochs",
    "ent_coef",
    "clip_range",
    "target_kl",
    "n_steps",
    "gamma",
    "gae_lambda",
    "vf_coef",
    "max_grad_norm",
}


def _coerce_optional_float(value: object) -> float | None:
    """Parse float-valued overrides while preserving explicit null values."""

    if value is None:
        return None
    return float(value)


_PPO_PARAM_COERCIONS: dict[str, Callable[[object], object]] = {
    "learning_rate": float,
    "batch_size": int,
    "n_epochs": int,
    "ent_coef": float,
    "clip_range": float,
    "target_kl": _coerce_optional_float,
    "n_steps": int,
    "gamma": float,
    "gae_lambda": float,
    "vf_coef": float,
    "max_grad_norm": float,
}
_EVAL_METRIC_KEYS = (
    "success_rate",
    "collision_rate",
    "path_efficiency",
    "comfort_exposure",
    "snqi",
    "eval_episode_return",
    "eval_avg_step_reward",
)
_SUPPORTED_BEST_METRICS = set(_EVAL_METRIC_KEYS)
_FREQUENCY_EPISODES_DEPRECATION_WARNED = False
_DIRECT_WANDB_TRAIN_METRIC_KEYS = (
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
)
_AUTO_NUM_ENVS_THROUGHPUT_RESERVED_CORES = 2
_AUTO_NUM_ENVS_THROUGHPUT_HEADROOM_GIB = 6.0
_AUTO_NUM_ENVS_THROUGHPUT_ENV_BUDGET_GIB = 1.1
_AUTO_NUM_ENVS_STABLE_RESERVED_CORES = 2
_AUTO_NUM_ENVS_STABLE_CPU_DIVISOR = 2
_AUTO_NUM_ENVS_STABLE_HEADROOM_GIB = 20.0
_AUTO_NUM_ENVS_STABLE_ENV_BUDGET_GIB = 1.2


def _wandb_training_clock() -> float:
    """Return the monotonic clock used for direct W&B training metrics."""
    return time.perf_counter()


class _DirectWandbMetricsCallback(BaseCallback):
    """Mirror core SB3 training metrics directly to W&B.

    This avoids long periods with only system charts when TensorBoard sync is
    delayed or when resumed runs attach the logger after model construction.
    """

    def __init__(self, wandb_run: object, *, log_every_steps: int = 10_000) -> None:
        super().__init__()
        self._wandb_run = wandb_run
        self._log_every_steps = max(1, int(log_every_steps))
        self._last_logged_step = -1

    def _on_step(self) -> bool:
        if int(self.num_timesteps) - self._last_logged_step < self._log_every_steps:
            return True
        values = getattr(self.logger, "name_to_value", {}) or {}
        payload: dict[str, float | int] = {
            "time/total_timesteps": int(self.num_timesteps),
        }
        for key in (
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/entropy_loss",
            "train/loss",
            "train/explained_variance",
            "time/fps",
            "time/iterations",
        ):
            value = values.get(key)
            if isinstance(value, int | float):
                payload[key] = float(value)
        if len(payload) > 1:
            self._wandb_run.log(payload, step=int(self.num_timesteps))
            self._last_logged_step = int(self.num_timesteps)
        return True


def _constant_schedule(value: float) -> Callable[[float], float]:
    """Return a constant SB3-compatible schedule callback."""

    scalar = float(value)

    def _schedule(_progress_remaining: float) -> float:
        return scalar

    return _schedule


def _apply_model_attr_if_present(
    model: PPO,
    params: Mapping[str, object],
    *,
    key: str,
    attr_name: str,
    coerce: Callable[[object], object],
) -> None:
    """Set one model attribute from config params when that key is present."""

    if key not in params:
        return
    setattr(model, attr_name, coerce(params[key]))


def _apply_schedule_attr_if_present(
    model: PPO,
    params: Mapping[str, object],
    *,
    key: str,
    value_attr: str | None,
    schedule_attr: str,
) -> None:
    """Set a scalar attribute plus its constant schedule callback when present."""

    if key not in params:
        return
    value = float(params[key])
    if value_attr is not None:
        setattr(model, value_attr, value)
    setattr(model, schedule_attr, _constant_schedule(value))


def _preconfigure_loguru_level_from_argv() -> None:
    """Set LOGURU_LEVEL early so import-time logs honor CLI log-level."""
    if _ORIGINAL_LOGURU_LEVEL is not None:
        return
    for idx, arg in enumerate(sys.argv):
        if arg == "--log-level" and idx + 1 < len(sys.argv):
            os.environ["LOGURU_LEVEL"] = str(sys.argv[idx + 1]).upper()
            return
        if arg.startswith("--log-level="):
            os.environ["LOGURU_LEVEL"] = arg.split("=", 1)[1].upper()
            return


_preconfigure_loguru_level_from_argv()


class _TeeStream:
    """Write to multiple streams (e.g., terminal + log file)."""

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            if hasattr(stream, "flush"):
                stream.flush()

    def isatty(self) -> bool:
        for stream in self._streams:
            if hasattr(stream, "isatty") and stream.isatty():
                return True
        return False


def _parse_step_schedule(raw: object) -> tuple[tuple[int | None, int], ...]:
    """Parse optional step schedule entries from YAML."""
    if not raw:
        return ()
    if not isinstance(raw, Sequence):
        raise ValueError("evaluation.step_schedule must be a list of mappings.")
    schedule: list[tuple[int | None, int]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("evaluation.step_schedule entries must be mappings.")
        until_step = entry.get("until_step")
        every_steps = entry.get("every_steps")
        if every_steps is None:
            raise ValueError("evaluation.step_schedule entries must define every_steps.")
        schedule.append(
            (
                int(until_step) if until_step is not None else None,
                int(every_steps),
            )
        )
    return tuple(schedule)


def _parse_num_envs(raw: object) -> int | str | None:
    """Parse num_envs setting (supports int and host-aware auto modes)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"auto", "auto_throughput"}:
            return "auto_throughput"
        if normalized == "auto_stable":
            return "auto_stable"
    return max(1, int(raw))


def _ensure_cuda_determinism_env() -> None:
    """Ensure CUDA deterministic workspace configuration is set when available."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    logger.info("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA")


def _slurm_allocated_cpus() -> int | None:
    """Return allocated CPU count from Slurm when available."""

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


def _host_logical_cpus() -> int:
    """Return the logical CPU count available to the current process."""

    return max(1, _slurm_allocated_cpus() or (os.cpu_count() or 1))


def _host_memory_gib() -> float | None:
    """Return total visible host memory in GiB when detectable."""

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(phys_pages, int):
                total_bytes = page_size * phys_pages
                if total_bytes > 0:
                    return float(total_bytes) / float(1024**3)
        except (ValueError, OSError):
            pass
    return None


def _memory_limited_num_envs(*, headroom_gib: float, env_budget_gib: float) -> int | None:
    """Estimate a safe env cap from visible host memory."""

    total_gib = _host_memory_gib()
    if total_gib is None:
        return None
    usable_gib = max(0.0, total_gib - headroom_gib)
    return max(1, int(usable_gib // env_budget_gib))


def _describe_num_envs_resolution(config: ExpertTrainingConfig) -> dict[str, object]:
    """Return structured details explaining how num_envs was resolved."""

    if isinstance(config.num_envs, int):
        resolved = max(1, int(config.num_envs))
        return {
            "requested": int(config.num_envs),
            "mode": "fixed",
            "logical_cpus": None,
            "slurm_cpus": None,
            "reserve_cores": int(config.num_envs_reserve_cores),
            "host_memory_gib": None,
            "cpu_target": resolved,
            "memory_cap": None,
            "headroom_gib": None,
            "env_budget_gib": None,
            "resolved": resolved,
            "decision": "explicit config integer",
        }

    mode = (
        str(config.num_envs).strip().lower()
        if isinstance(config.num_envs, str)
        else "auto_throughput"
    )
    if mode == "auto":
        mode = "auto_throughput"
    if mode not in {"auto_throughput", "auto_stable"}:
        raise ValueError("num_envs must be an int or one of {'auto_throughput', 'auto_stable'}")

    slurm_cpus = _slurm_allocated_cpus()
    logical_cpus = max(1, slurm_cpus or (os.cpu_count() or 1))
    reserve = max(0, int(config.num_envs_reserve_cores))
    host_memory_gib = _host_memory_gib()
    if mode == "auto_stable":
        cpu_target = max(
            1,
            (logical_cpus // _AUTO_NUM_ENVS_STABLE_CPU_DIVISOR)
            - _AUTO_NUM_ENVS_STABLE_RESERVED_CORES
            - reserve,
        )
        headroom_gib = _AUTO_NUM_ENVS_STABLE_HEADROOM_GIB
        env_budget_gib = _AUTO_NUM_ENVS_STABLE_ENV_BUDGET_GIB
        decision_label = "stable headroom heuristic"
    else:
        cpu_target = max(
            1,
            logical_cpus - _AUTO_NUM_ENVS_THROUGHPUT_RESERVED_CORES - reserve,
        )
        headroom_gib = _AUTO_NUM_ENVS_THROUGHPUT_HEADROOM_GIB
        env_budget_gib = _AUTO_NUM_ENVS_THROUGHPUT_ENV_BUDGET_GIB
        decision_label = "throughput heuristic"

    memory_cap = None
    if host_memory_gib is not None:
        usable_gib = max(0.0, host_memory_gib - headroom_gib)
        memory_cap = max(1, int(usable_gib // env_budget_gib))
    resolved = cpu_target if memory_cap is None else max(1, min(cpu_target, memory_cap))
    limiter = "cpu target" if memory_cap is None or cpu_target <= memory_cap else "memory cap"

    return {
        "requested": config.num_envs if config.num_envs is not None else "auto_throughput",
        "mode": mode,
        "logical_cpus": logical_cpus,
        "slurm_cpus": slurm_cpus,
        "reserve_cores": reserve,
        "host_memory_gib": host_memory_gib,
        "cpu_target": cpu_target,
        "memory_cap": memory_cap,
        "headroom_gib": headroom_gib,
        "env_budget_gib": env_budget_gib,
        "resolved": resolved,
        "decision": f"{decision_label}; limited by {limiter}",
    }


def _resolve_num_envs(config: ExpertTrainingConfig) -> int:
    """Resolve the effective number of environments for training."""
    return int(_describe_num_envs_resolution(config)["resolved"])


def _resolve_worker_mode(config: ExpertTrainingConfig, num_envs: int) -> str:
    """Resolve worker mode ('dummy' or 'subproc') based on config and env count."""
    mode = config.worker_mode.lower()
    if mode == "auto":
        return "subproc" if num_envs > 1 else "dummy"
    if mode not in {"dummy", "subproc"}:
        raise ValueError("worker_mode must be one of {'auto', 'dummy', 'subproc'}")
    if mode == "subproc" and num_envs == 1:
        return "dummy"
    return mode


def _coerce_grid_channels(values: Sequence[object]) -> list[GridChannel]:
    """Convert channel identifiers to GridChannel enums."""
    channels: list[GridChannel] = []
    for value in values:
        if isinstance(value, GridChannel):
            channels.append(value)
        else:
            channels.append(GridChannel(str(value)))
    return channels


def _apply_simple_overrides(env_config, overrides: Mapping[str, object]) -> None:
    """Apply top-level environment overrides."""
    observation_mode = overrides.get("observation_mode")
    if observation_mode is not None:
        env_config.observation_mode = ObservationMode(str(observation_mode))

    for key in (
        "use_occupancy_grid",
        "include_grid_in_observation",
        "show_occupancy_grid",
        "use_planner",
        "planner_backend",
        "planner_clearance_margin",
        "peds_have_obstacle_forces",
        "peds_have_static_obstacle_forces",
        "peds_have_robot_repulsion",
        "map_id",
    ):
        if key in overrides:
            setattr(env_config, key, overrides[key])


def _resolve_robot_config_type(
    value: object,
) -> type[BicycleDriveSettings] | type[DifferentialDriveSettings] | None:
    """Resolve a robot config class from a string override."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"bicycle", "bicycle_drive", "bicycle_drive_settings"}:
            return BicycleDriveSettings
        if normalized in {
            "differential",
            "diff",
            "differential_drive",
            "differential_drive_settings",
        }:
            return DifferentialDriveSettings
    return None


def _apply_robot_overrides(env_config, overrides: Mapping[str, object]) -> None:
    """Apply robot drivetrain overrides (e.g., bicycle allow_backwards)."""
    robot_overrides = overrides.get("robot_config")
    if robot_overrides is None:
        return
    if isinstance(robot_overrides, (BicycleDriveSettings, DifferentialDriveSettings)):
        env_config.robot_config = robot_overrides
        return
    if isinstance(robot_overrides, str):
        config_cls = _resolve_robot_config_type(robot_overrides)
        if config_cls is None:
            raise ValueError(f"robot_config override type '{robot_overrides}' is not supported")
        env_config.robot_config = config_cls()
        return
    if isinstance(robot_overrides, Mapping):
        payload = dict(robot_overrides)
        config_type = payload.pop("type", None)
        if config_type is not None:
            config_cls = _resolve_robot_config_type(config_type)
            if config_cls is None:
                raise ValueError(f"robot_config override type '{config_type}' is not supported")
            env_config.robot_config = config_cls()
        target = env_config.robot_config
        for key, value in payload.items():
            if not hasattr(target, key):
                raise ValueError(f"robot_config override has unknown field '{key}'")
            setattr(target, key, value)
        return
    raise ValueError("robot_config override must be a mapping, dataclass, or type string")


def _apply_grid_override(env_config, overrides: Mapping[str, object]) -> None:
    """Apply occupancy grid overrides to the environment config."""
    grid_override = overrides.get("grid_config")
    if grid_override is None:
        return
    if isinstance(grid_override, GridConfig):
        env_config.grid_config = grid_override
        return
    if isinstance(grid_override, Mapping):
        payload = dict(grid_override)
        if "channels" in payload:
            payload["channels"] = _coerce_grid_channels(payload["channels"])
        env_config.grid_config = GridConfig(**payload)
        return
    raise ValueError("grid_config override must be a mapping or GridConfig instance")


def _apply_nested_overrides(config_obj: object, overrides: Mapping[str, object]) -> None:
    """Recursively apply overrides to a nested config object."""
    for key, value in overrides.items():
        if not hasattr(config_obj, key):
            continue
        current_attr = getattr(config_obj, key)
        if isinstance(value, Mapping) and hasattr(current_attr, "__dict__"):
            _apply_nested_overrides(current_attr, value)
        else:
            setattr(config_obj, key, value)


def _apply_sim_overrides(env_config, overrides: Mapping[str, object]) -> None:
    """Apply sim_config overrides (including ped-robot force flags)."""
    sim_overrides = overrides.get("sim_config")
    if not isinstance(sim_overrides, Mapping):
        return
    _apply_nested_overrides(env_config.sim_config, sim_overrides)


def _apply_env_overrides(env_config, overrides: Mapping[str, object]) -> None:
    """Apply environment overrides from training config."""
    if not overrides:
        return
    _apply_simple_overrides(env_config, overrides)
    _apply_robot_overrides(env_config, overrides)
    _apply_grid_override(env_config, overrides)
    _apply_sim_overrides(env_config, overrides)


@dataclass(slots=True)
class ExpertTrainingResult:
    """Container summarising artefacts produced by the training workflow."""

    config: ExpertTrainingConfig
    expert_artifact: common.ExpertPolicyArtifact
    training_run_artifact: common.TrainingRunArtifact
    expert_manifest_path: Path
    training_run_manifest_path: Path
    checkpoint_path: Path
    metrics: dict[str, common.MetricAggregate]
    best_checkpoint: BestCheckpointSummary | None


@dataclass(slots=True)
class ScenarioContext:
    """Resolved scenario context for training/evaluation."""

    selected_scenario: Mapping[str, Any] | None
    scenario_label: str
    scenario_profile: tuple[str, ...]
    training_exclude: tuple[str, ...]


@dataclass(slots=True)
class BestCheckpointSummary:
    """Snapshot describing the best checkpoint chosen during training."""

    metric: str
    value: float
    eval_step: int
    checkpoint_path: Path
    metrics: dict[str, float]
    meets_convergence: bool
    report_path: Path | None = None


@dataclass(slots=True)
class _BestCheckpointCandidate:
    """Internal tracker for best checkpoint selection."""

    eval_step: int
    score: float
    metrics: dict[str, float]
    meets_convergence: bool


@dataclass(slots=True)
class _BestCheckpointTracker:
    """Track the best checkpoint across evaluations."""

    metric_name: str
    higher_is_better: bool
    convergence: ConvergenceCriteria
    best_overall: _BestCheckpointCandidate | None = None
    best_converged: _BestCheckpointCandidate | None = None

    def update(self, summary: dict[str, float], *, eval_step: int) -> None:
        """Update best checkpoint candidates from an eval summary."""
        score = summary.get(self.metric_name)
        if score is None:
            return
        meets_convergence = (
            summary.get("success_rate", 0.0) >= self.convergence.success_rate
            and summary.get("collision_rate", 1.0) <= self.convergence.collision_rate
        )
        candidate = _BestCheckpointCandidate(
            eval_step=eval_step,
            score=score,
            metrics=summary,
            meets_convergence=meets_convergence,
        )
        if self._is_better(candidate, self.best_overall):
            self.best_overall = candidate
        if meets_convergence and self._is_better(candidate, self.best_converged):
            self.best_converged = candidate

    def best(self) -> _BestCheckpointCandidate | None:
        """Return the preferred best candidate."""
        return self.best_converged or self.best_overall

    def _is_better(
        self, candidate: _BestCheckpointCandidate, current: _BestCheckpointCandidate | None
    ) -> bool:
        if current is None:
            return True
        if self.higher_is_better:
            return candidate.score > current.score
        return candidate.score < current.score


@dataclass(slots=True)
class TrainingOutputs:
    """Bundle of outputs from the training/evaluation loop."""

    metrics_raw: MetricSamples
    episode_records: list[dict[str, object]]
    model: PPO | None
    vec_env: DummyVecEnv | SubprocVecEnv | None
    tensorboard_log: Path | None
    best_checkpoint: BestCheckpointSummary | None
    snqi_context: TrainingSNQIContext
    eval_timeline: list[dict[str, float | int]]
    startup_sec: float
    per_checkpoint_perf: list[dict[str, float | int]]


def _resolve_optional_path(path: Path, raw: object, *, field_name: str) -> Path | None:
    """Resolve an optional path field relative to the config file location."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if path.parent == path:
        raise ValueError(f"Unable to resolve relative path for {field_name}")
    return (path.parent / candidate).resolve()


def load_expert_training_config(config_path: str | Path) -> ExpertTrainingConfig:
    """Load an :class:`ExpertTrainingConfig` from a YAML file."""

    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):  # pragma: no cover - guard against malformed YAML
        raise ValueError(f"Configuration must be a mapping, received {type(data)!r}")

    scenario_raw = Path(data["scenario_config"])
    scenario_config = (
        (path.parent / scenario_raw).resolve() if not scenario_raw.is_absolute() else scenario_raw
    )
    scenario_id = data.get("scenario_id")

    convergence_raw = data.get("convergence", {})
    evaluation_raw = data.get("evaluation", {})
    step_schedule = _parse_step_schedule(evaluation_raw.get("step_schedule"))
    socnav_orca_raw = (
        data.get("socnav_orca", {}) if isinstance(data.get("socnav_orca"), Mapping) else {}
    )
    socnav_orca_time_horizon = data.get("socnav_orca_time_horizon")
    socnav_orca_neighbor_dist = data.get("socnav_orca_neighbor_dist")
    if socnav_orca_time_horizon is None:
        socnav_orca_time_horizon = socnav_orca_raw.get("time_horizon")
    if socnav_orca_neighbor_dist is None:
        socnav_orca_neighbor_dist = socnav_orca_raw.get("neighbor_dist")

    convergence = ConvergenceCriteria(
        success_rate=float(convergence_raw["success_rate"]),
        collision_rate=float(convergence_raw["collision_rate"]),
        plateau_window=int(convergence_raw["plateau_window"]),
    )
    frequency_episodes = int(evaluation_raw.get("frequency_episodes", 0))
    evaluation = EvaluationSchedule(
        frequency_episodes=frequency_episodes,
        evaluation_episodes=int(evaluation_raw["evaluation_episodes"]),
        hold_out_scenarios=tuple(evaluation_raw.get("hold_out_scenarios", ())),
        step_schedule=step_schedule,
    )
    if "frequency_episodes" in evaluation_raw:
        _warn_frequency_episodes_deprecated(evaluation.frequency_episodes)
    if not evaluation.step_schedule:
        raise ValueError(
            "evaluation.step_schedule is required; frequency_episodes alone is not supported."
        )

    return ExpertTrainingConfig.from_raw(
        scenario_config=scenario_config,
        scenario_id=str(scenario_id) if scenario_id else None,
        seeds=common.ensure_seed_tuple(data.get("seeds", [])),
        randomize_seeds=bool(data.get("randomize_seeds", False)),
        total_timesteps=int(data["total_timesteps"]),
        policy_id=str(data["policy_id"]),
        convergence=convergence,
        evaluation=evaluation,
        ppo_hyperparams=dict(data.get("ppo_hyperparams", {}) or {}),
        best_checkpoint_metric=str(data.get("best_checkpoint_metric", "success_rate")),
        snqi_weights_path=_resolve_optional_path(
            path,
            data.get("snqi_weights"),
            field_name="snqi_weights",
        ),
        snqi_baseline_path=_resolve_optional_path(
            path,
            data.get("snqi_baseline"),
            field_name="snqi_baseline",
        ),
        feature_extractor=str(data.get("feature_extractor", "default")),
        feature_extractor_kwargs=dict(data.get("feature_extractor_kwargs", {}) or {}),
        policy_net_arch=tuple(data.get("policy_net_arch", (64, 64))),
        tracking=dict(data.get("tracking", {}) or {}),
        env_overrides=dict(data.get("env_overrides", {}) or {}),
        env_factory_kwargs=dict(data.get("env_factory_kwargs", {}) or {}),
        scenario_sampling=dict(data.get("scenario_sampling", {}) or {}),
        num_envs=_parse_num_envs(data.get("num_envs")),
        worker_mode=str(data.get("worker_mode", "auto")),
        socnav_orca_time_horizon=(
            float(socnav_orca_time_horizon) if socnav_orca_time_horizon is not None else None
        ),
        socnav_orca_neighbor_dist=(
            float(socnav_orca_neighbor_dist) if socnav_orca_neighbor_dist is not None else None
        ),
        resume_from=_resolve_optional_path(
            path,
            data.get("resume_from"),
            field_name="resume_from",
        ),
        resume_model_id=(
            str(data.get("resume_model_id")).strip() if data.get("resume_model_id") else None
        ),
        resume_source_step=(
            int(data["resume_source_step"]) if data.get("resume_source_step") is not None else None
        ),
    )


def _warn_frequency_episodes_deprecated(frequency_episodes: int) -> None:
    """Warn once that frequency_episodes is ignored in favor of step_schedule."""
    global _FREQUENCY_EPISODES_DEPRECATION_WARNED
    if _FREQUENCY_EPISODES_DEPRECATION_WARNED:
        return
    logger.warning(
        "evaluation.frequency_episodes={} is currently ignored; "
        "evaluation.step_schedule controls checkpoint cadence.",
        frequency_episodes,
    )
    _FREQUENCY_EPISODES_DEPRECATION_WARNED = True


def _resolved_reward_name(env_factory_kwargs: Mapping[str, object]) -> str:
    """Resolve named reward profile for startup logs."""
    if "reward_func" in env_factory_kwargs:
        return "custom_callable"
    reward_name = env_factory_kwargs.get("reward_name")
    if reward_name is None:
        return "route_completion_v2 (default)"
    return str(reward_name)


def _log_startup_summary(
    *,
    config: ExpertTrainingConfig,
    config_path: Path | None,
    num_envs: int,
    worker_mode: str,
) -> None:
    """Emit one structured startup summary for run-critical resolved config."""
    num_envs_details = _describe_num_envs_resolution(config)
    logger.info(
        "Training startup summary: policy_id={} config_path={} scenario_config={} "
        "total_timesteps={} reward_profile={} requested_num_envs={} num_envs={} worker_mode={} "
        "randomize_seeds={} resume_from={} scenario_sampling={}",
        config.policy_id,
        str(config_path) if config_path is not None else "<none>",
        config.scenario_config,
        config.total_timesteps,
        _resolved_reward_name(config.env_factory_kwargs),
        config.num_envs if config.num_envs is not None else "auto_throughput",
        num_envs,
        worker_mode,
        _randomize_seeds(config),
        (
            str(config.resume_from)
            if config.resume_from is not None
            else f"model_id:{config.resume_model_id}"
            if config.resume_model_id
            else "<none>"
        ),
        sorted(config.scenario_sampling.keys()),
    )
    logger.info(
        "num_envs resolution: requested={} mode={} resolved={} decision='{}' "
        "logical_cpus={} slurm_cpus={} reserve_cores={} host_memory_gib={} "
        "cpu_target={} memory_cap={} headroom_gib={} env_budget_gib={}",
        num_envs_details["requested"],
        num_envs_details["mode"],
        num_envs_details["resolved"],
        num_envs_details["decision"],
        num_envs_details["logical_cpus"],
        num_envs_details["slurm_cpus"],
        num_envs_details["reserve_cores"],
        (
            round(float(num_envs_details["host_memory_gib"]), 2)
            if isinstance(num_envs_details["host_memory_gib"], int | float)
            else None
        ),
        num_envs_details["cpu_target"],
        num_envs_details["memory_cap"],
        num_envs_details["headroom_gib"],
        num_envs_details["env_budget_gib"],
    )


def _resolve_resume_checkpoint(
    *,
    config: ExpertTrainingConfig,
    resume_from: Path | None,
) -> Path | None:
    """Resolve a resume checkpoint path from CLI/config path or registry model_id."""

    if resume_from is not None:
        return Path(resume_from).expanduser()
    if config.resume_from is not None:
        return Path(config.resume_from).expanduser()
    if config.resume_model_id:
        return resolve_model_path(config.resume_model_id, allow_download=True)
    return None


def _make_training_env(  # noqa: PLR0913
    seed: int | None,
    *,
    scenario: Mapping[str, Any] | None,
    scenario_definitions: Sequence[Mapping[str, Any]] | None,
    scenario_path: Path,
    exclude_scenarios: Sequence[str],
    suite_name: str,
    algorithm_name: str,
    env_overrides: Mapping[str, object],
    env_factory_kwargs: Mapping[str, object],
    scenario_sampling: Mapping[str, object],
) -> Callable[[], Any]:
    """Create a training environment factory (seeded when provided)."""

    def _factory() -> Any:
        def _build_config(scenario_def: Mapping[str, Any]):
            env_config = build_robot_config_from_scenario(scenario_def, scenario_path=scenario_path)
            _apply_env_overrides(env_config, env_overrides)
            return env_config

        if scenario is not None:
            env_config = _build_config(scenario)
            scenario_id = scenario_id_from_definition(scenario, index=0)
            return make_robot_env(
                config=env_config,
                seed=seed,
                suite_name=suite_name,
                scenario_name=scenario_id,
                algorithm_name=algorithm_name,
                **env_factory_kwargs,
            )

        if scenario_definitions is None:
            raise ValueError("scenario_definitions required when scenario is None.")

        sampler = ScenarioSampler(
            scenario_definitions,
            include_scenarios=tuple(
                str(name)
                for name in scenario_sampling.get("include_scenarios", ())  # type: ignore[union-attr]
            ),
            exclude_scenarios=tuple(exclude_scenarios)
            + tuple(
                str(name)
                for name in scenario_sampling.get("exclude_scenarios", ())  # type: ignore[union-attr]
            ),
            weights=(
                {
                    str(key): float(value)
                    for key, value in dict(scenario_sampling.get("weights", {})).items()  # type: ignore[union-attr]
                }
                if isinstance(scenario_sampling.get("weights"), Mapping)
                else None
            ),
            seed=seed,
            strategy=str(scenario_sampling.get("strategy", "random")),  # type: ignore[union-attr]
        )
        return ScenarioSwitchingEnv(
            scenario_sampler=sampler,
            scenario_path=scenario_path,
            config_builder=_build_config,
            suite_name=suite_name,
            algorithm_name=algorithm_name,
            env_factory_kwargs=env_factory_kwargs,
            seed=seed,
        )

    return _factory


def _resolve_policy_kwargs(config: ExpertTrainingConfig) -> dict[str, Any]:
    """Build policy kwargs for PPO from the training config."""
    policy_kwargs: dict[str, Any] = {"net_arch": list(config.policy_net_arch)}
    extractor = config.feature_extractor.lower()
    if extractor == "grid_socnav":
        policy_kwargs["features_extractor_class"] = GridSocNavExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(config.feature_extractor_kwargs)
    return policy_kwargs


def _resolve_ppo_hyperparams(config: ExpertTrainingConfig) -> dict[str, object]:
    """Merge default PPO hyperparameters with any overrides from config."""
    overrides = dict(config.ppo_hyperparams or {})
    unknown = set(overrides) - _ALLOWED_PPO_HYPERPARAMS
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"ppo_hyperparams has unsupported keys: {unknown_list}")
    params = dict(_DEFAULT_PPO_HYPERPARAMS)
    for key, value in overrides.items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = value
    for key, coerce in _PPO_PARAM_COERCIONS.items():
        if key in params:
            params[key] = coerce(params[key])
    return params


def _reapply_resumed_ppo_hyperparams(model: PPO, config: ExpertTrainingConfig) -> None:
    """Reapply config-level PPO hyperparameters after loading a checkpoint.

    Stable-Baselines3 restores optimizer and algorithm settings from the checkpoint.
    For warm-start runs we still want the YAML config to remain authoritative, so
    the supported overrides are written back onto the loaded model before learning.
    """

    params = _resolve_ppo_hyperparams(config)
    _apply_schedule_attr_if_present(
        model,
        params,
        key="learning_rate",
        value_attr="learning_rate",
        schedule_attr="lr_schedule",
    )
    _apply_schedule_attr_if_present(
        model,
        params,
        key="clip_range",
        value_attr=None,
        schedule_attr="clip_range",
    )
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
        _apply_model_attr_if_present(
            model,
            params,
            key=key,
            attr_name=attr_name,
            coerce=coerce,
        )

    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None:
        _apply_model_attr_if_present(
            rollout_buffer,
            params,
            key="gamma",
            attr_name="gamma",
            coerce=float,
        )
        _apply_model_attr_if_present(
            rollout_buffer,
            params,
            key="gae_lambda",
            attr_name="gae_lambda",
            coerce=float,
        )

    if "n_steps" in params and int(params["n_steps"]) != int(getattr(model, "n_steps", 0) or 0):
        logger.warning(
            "Resume checkpoint keeps n_steps={} from the saved model; "
            "config requested n_steps={} but rollout buffer rebuild is not automated.",
            getattr(model, "n_steps", "<unknown>"),
            int(params["n_steps"]),
        )

    logger.info(
        "Reapplied PPO hyperparameters after resume: learning_rate={} batch_size={} "
        "n_epochs={} ent_coef={} clip_range={} target_kl={}",
        getattr(model, "learning_rate", "<unchanged>"),
        getattr(model, "batch_size", "<unchanged>"),
        getattr(model, "n_epochs", "<unchanged>"),
        getattr(model, "ent_coef", "<unchanged>"),
        float(params["clip_range"]) if "clip_range" in params else "<unchanged>",
        getattr(model, "target_kl", "<unchanged>"),
    )


def _randomize_seeds(config: ExpertTrainingConfig) -> bool:
    """Return True when training/evaluation should avoid deterministic seeding."""
    return bool(getattr(config, "randomize_seeds", False))


def _resolve_tensorboard_logdir(run_id: str) -> Path:
    """Return a canonical TensorBoard log directory for the training run."""
    base = get_imitation_report_dir() / run_id / "tensorboard"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _init_wandb(
    *,
    tracking: Mapping[str, object],
    run_id: str,
    config: ExpertTrainingConfig,
    tensorboard_log: Path | None,
) -> tuple[object | None, object | None]:
    """Initialize Weights & Biases logging if configured."""
    wandb_cfg = tracking.get("wandb") if isinstance(tracking, Mapping) else None
    if not isinstance(wandb_cfg, Mapping) or not wandb_cfg.get("enabled", False):
        return None, None

    try:  # pragma: no cover - optional dependency
        import wandb  # type: ignore

        wandb_sb3 = importlib.import_module("wandb.integration.sb3")
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("W&B requested but not available: {}", exc)
        return None, None

    wandb_dir = get_artifact_category_path("wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_str = str(tensorboard_log) if tensorboard_log is not None else ""
    tags = _normalize_wandb_tags(wandb_cfg.get("tags"))
    run = wandb.init(
        project=str(wandb_cfg.get("project", "robot_sf")),
        group=str(wandb_cfg.get("group", "ppo-imitation")),
        job_type=str(wandb_cfg.get("job_type", "expert-training")),
        name=str(wandb_cfg.get("name", run_id)),
        notes=str(wandb_cfg.get("notes", "")),
        tags=tags,
        dir=str(wandb_dir),
        config={
            "policy_id": config.policy_id,
            "total_timesteps": config.total_timesteps,
            "seeds": list(config.seeds),
            "randomize_seeds": bool(config.randomize_seeds),
            "scenario_config": str(config.scenario_config),
            "feature_extractor": config.feature_extractor,
            "ppo_hyperparams": dict(config.ppo_hyperparams),
            "best_checkpoint_metric": config.best_checkpoint_metric,
            "snqi_weights_source": (
                str(config.snqi_weights_path) if config.snqi_weights_path is not None else "default"
            ),
            "snqi_baseline_source": (
                str(config.snqi_baseline_path)
                if config.snqi_baseline_path is not None
                else "default"
            ),
            "tensorboard_log": tensorboard_log_str,
        },
        sync_tensorboard=True,
        mode=str(wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "online"))),
    )
    callback = wandb_sb3.WandbCallback(
        gradient_save_freq=int(wandb_cfg.get("gradient_save_freq", 0)),
        model_save_path=str(wandb_dir / run_id),
        verbose=0,
    )
    logger.info("W&B run initialized id={} project={}", run.id if run else "unknown", run.project)
    return run, callback


def _normalize_wandb_tags(raw_tags: object) -> list[str] | None:
    """Normalize W&B tags to a list of strings, preserving single-string tags."""
    if raw_tags is None:
        return None
    if isinstance(raw_tags, str):
        return [raw_tags]
    if isinstance(raw_tags, bytes):
        return [raw_tags.decode("utf-8", errors="replace")]
    if isinstance(raw_tags, Sequence):
        return [str(tag) for tag in raw_tags]
    return None


def _extract_episode_info_mean(ep_info_buffer: object, key: str) -> float | None:
    """Return the mean numeric value for one episode-info key when available."""
    if not isinstance(ep_info_buffer, Sequence):
        return None
    values: list[float] = []
    for item in ep_info_buffer:
        if not isinstance(item, Mapping):
            continue
        raw_value = item.get(key)
        if raw_value is None:
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            values.append(numeric_value)
    if not values:
        return None
    return float(np.mean(values))


def _extract_direct_wandb_train_metrics(model: PPO) -> dict[str, float]:
    """Extract the latest scalar train metrics recorded by the SB3 logger."""
    logger_values = getattr(getattr(model, "logger", None), "name_to_value", None)
    if not isinstance(logger_values, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key in _DIRECT_WANDB_TRAIN_METRIC_KEYS:
        raw_value = logger_values.get(key)
        if raw_value is None:
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            metrics[key] = numeric_value
    return metrics


def _build_direct_wandb_training_payload(
    *,
    model: PPO,
    total_timesteps: int,
    rollout_iterations: int,
    start_timesteps: int,
    run_start_time: float,
) -> dict[str, float | int]:
    """Build a direct W&B payload for core PPO training metrics after PPO.train()."""
    payload: dict[str, float | int] = {
        "time/total_timesteps": int(total_timesteps),
        "time/iterations": int(rollout_iterations),
    }
    elapsed = max(0.0, _wandb_training_clock() - run_start_time)
    completed_timesteps = max(0, int(total_timesteps) - int(start_timesteps))
    payload["time/fps"] = float(completed_timesteps / elapsed) if elapsed > 0.0 else 0.0

    rollout_reward_mean = _extract_episode_info_mean(getattr(model, "ep_info_buffer", None), "r")
    if rollout_reward_mean is not None:
        payload["rollout/ep_rew_mean"] = rollout_reward_mean
    rollout_length_mean = _extract_episode_info_mean(getattr(model, "ep_info_buffer", None), "l")
    if rollout_length_mean is not None:
        payload["rollout/ep_len_mean"] = rollout_length_mean

    return payload


class _DirectWandbTrainingMetricsCallback(BaseCallback):
    """Track rollout iterations for direct W&B logging after PPO.train()."""

    def __init__(
        self,
        *,
        wandb_run: object,
        start_timesteps: int = 0,
        run_start_time: float | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self._wandb_run = wandb_run
        self._start_timesteps = int(max(0, start_timesteps))
        self._rollout_iterations = 0
        self._run_start_time = (
            float(run_start_time) if run_start_time is not None else _wandb_training_clock()
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_iterations += 1

    def log_after_train(self) -> None:
        """Log rollout/time and freshly-recorded train metrics after PPO.train() completes."""
        total_timesteps = int(getattr(self.model, "num_timesteps", self.num_timesteps) or 0)
        payload = _build_direct_wandb_training_payload(
            model=self.model,
            total_timesteps=total_timesteps,
            rollout_iterations=self._rollout_iterations,
            start_timesteps=self._start_timesteps,
            run_start_time=self._run_start_time,
        )
        payload.update(_extract_direct_wandb_train_metrics(self.model))
        if payload:
            self._wandb_run.log(payload, step=total_timesteps)


def _build_eval_steps(
    total_timesteps: int, schedule: tuple[tuple[int | None, int], ...]
) -> list[int]:
    """Expand a step schedule into concrete evaluation checkpoints."""
    default_schedule = (
        (3_000_000, 500_000),
        (None, 1_000_000),
    )
    use_schedule = schedule or default_schedule
    checkpoints: list[int] = []
    current = 0
    for until, every in use_schedule:
        upper = total_timesteps if until is None else min(until, total_timesteps)
        if every <= 0:
            continue
        next_step = current + every
        while next_step <= upper:
            checkpoints.append(next_step)
            next_step += every
        current = upper
        if upper >= total_timesteps:
            break
    if not checkpoints or checkpoints[-1] != total_timesteps:
        checkpoints.append(total_timesteps)
    return checkpoints


def _resolve_scenario_context(
    config: ExpertTrainingConfig,
    scenario_definitions: Sequence[Mapping[str, Any]],
) -> ScenarioContext:
    """Resolve scenario selection and profile for training/evaluation."""
    if config.scenario_id:
        selected = select_scenario(scenario_definitions, config.scenario_id)
        label = config.scenario_id
        return ScenarioContext(
            selected_scenario=selected,
            scenario_label=label,
            scenario_profile=(label,),
            training_exclude=(),
        )

    sampler = ScenarioSampler(
        scenario_definitions,
        include_scenarios=tuple(
            str(name) for name in config.scenario_sampling.get("include_scenarios", ())
        ),
        exclude_scenarios=tuple(config.evaluation.hold_out_scenarios)
        + tuple(str(name) for name in config.scenario_sampling.get("exclude_scenarios", ())),
        weights=(
            {
                str(key): float(value)
                for key, value in dict(config.scenario_sampling.get("weights", {})).items()
            }
            if isinstance(config.scenario_sampling.get("weights"), Mapping)
            else None
        ),
        seed=None if _randomize_seeds(config) else int(config.seeds[0]) if config.seeds else None,
        strategy=str(config.scenario_sampling.get("profile_strategy", "cycle")),
    )
    return ScenarioContext(
        selected_scenario=None,
        scenario_label=config.scenario_config.stem,
        scenario_profile=sampler.scenario_ids,
        training_exclude=tuple(config.evaluation.hold_out_scenarios),
    )


def _execute_training(
    *,
    config: ExpertTrainingConfig,
    scenario_ctx: ScenarioContext,
    scenario_definitions: Sequence[Mapping[str, Any]],
    eval_steps: Sequence[int],
    run_id: str,
    dry_run: bool,
    resume_from: Path | None,
    config_path: Path | None,
) -> TrainingOutputs:
    """Run training (or dry-run) and return raw metrics + episode records."""
    startup_t0 = time.perf_counter()
    num_envs = _resolve_num_envs(config)
    worker_mode = _resolve_worker_mode(config, num_envs)
    _log_startup_summary(
        config=config,
        config_path=config_path,
        num_envs=num_envs,
        worker_mode=worker_mode,
    )
    snqi_context = resolve_training_snqi_context(
        weights_path=config.snqi_weights_path,
        baseline_path=config.snqi_baseline_path,
    )
    if snqi_context.baseline_fallback_keys:
        logger.warning(
            "SNQI baseline missing keys {}; defaults injected.",
            snqi_context.baseline_fallback_keys,
        )
    tensorboard_log = (
        _resolve_tensorboard_logdir(run_id)
        if bool(config.tracking.get("tensorboard", True))
        else None
    )
    if dry_run:
        metrics_raw, episode_records = _simulate_dry_run_metrics(
            config,
            eval_steps=eval_steps,
            scenario_id=scenario_ctx.scenario_label,
            snqi_context=snqi_context,
        )
        eval_timeline = _timeline_from_episode_records(
            eval_steps=eval_steps,
            episode_records=episode_records,
        )
        return TrainingOutputs(
            metrics_raw=metrics_raw,
            episode_records=episode_records,
            model=None,
            vec_env=None,
            tensorboard_log=tensorboard_log,
            best_checkpoint=None,
            snqi_context=snqi_context,
            eval_timeline=eval_timeline,
            startup_sec=max(0.0, time.perf_counter() - startup_t0),
            per_checkpoint_perf=[],
        )

    model, vec_env, tensorboard_log, num_envs, worker_mode = _init_training_model(
        config,
        scenario=scenario_ctx.selected_scenario,
        scenario_definitions=scenario_definitions,
        exclude_scenarios=scenario_ctx.training_exclude,
        run_id=run_id,
        tensorboard_log=tensorboard_log,
        resume_from=resume_from,
    )
    wandb_run, wandb_callback = _init_wandb(
        tracking=config.tracking,
        run_id=run_id,
        config=config,
        tensorboard_log=tensorboard_log,
    )
    startup_sec = max(0.0, time.perf_counter() - startup_t0)
    start_timesteps = int(getattr(model, "num_timesteps", 0) or 0)
    if start_timesteps >= max(eval_steps or [0]):
        logger.warning(
            "Resume step {} >= final scheduled step {}; running evaluation only.",
            start_timesteps,
            max(eval_steps or [0]),
        )
        eval_schedule = [start_timesteps]
    else:
        eval_schedule = [step for step in eval_steps if step > start_timesteps]
    metrics_raw, episode_records, best_checkpoint, eval_timeline, per_checkpoint_perf = (
        _train_with_schedule(
            model,
            config=config,
            scenario_definitions=scenario_definitions,
            scenario_id=scenario_ctx.scenario_label if config.scenario_id else None,
            hold_out_scenarios=config.evaluation.hold_out_scenarios,
            eval_steps=eval_schedule,
            snqi_context=snqi_context,
            wandb_run=wandb_run,
            wandb_callback=wandb_callback,
            start_timesteps=start_timesteps,
            checkpoint_dir=common.get_expert_policy_dir() / "checkpoints" / config.policy_id,
            startup_sec=startup_sec,
            num_envs=num_envs,
        )
    )
    if wandb_run is not None:  # pragma: no cover - optional dependency
        wandb_run.finish()

    return TrainingOutputs(
        metrics_raw=metrics_raw,
        episode_records=episode_records,
        model=model,
        vec_env=vec_env,
        tensorboard_log=tensorboard_log,
        best_checkpoint=best_checkpoint,
        snqi_context=snqi_context,
        eval_timeline=eval_timeline,
        startup_sec=startup_sec,
        per_checkpoint_perf=per_checkpoint_perf,
    )


def _collect_scenario_coverage(vec_env: DummyVecEnv | SubprocVecEnv | None) -> dict[str, int]:
    """Aggregate scenario coverage counts from scenario-switching workers."""
    coverage: dict[str, int] = {}
    if vec_env is None:
        return coverage
    try:
        env_coverages = vec_env.get_attr("scenario_coverage")
    except Exception:  # pragma: no cover - best effort across vec env types
        env_coverages = []
    for env_coverage in env_coverages:
        if not env_coverage:
            continue
        for key, value in env_coverage.items():
            coverage[key] = coverage.get(key, 0) + int(value)
    return coverage


def _resolve_best_checkpoint_metric(metric: str) -> tuple[str, bool]:
    """Resolve the metric name and whether higher values are better."""
    normalized = metric.strip().lower()
    if normalized not in _SUPPORTED_BEST_METRICS:
        supported = ", ".join(sorted(_SUPPORTED_BEST_METRICS))
        raise ValueError(f"best_checkpoint_metric must be one of: {supported}")
    higher_is_better = normalized not in {"collision_rate", "comfort_exposure"}
    return normalized, higher_is_better


def _summarize_eval_metrics(metrics: MetricSamples) -> dict[str, float]:
    """Return mean values for each evaluation metric."""
    summary: dict[str, float] = {}
    for name, values in metrics.items():
        if values:
            summary[name] = float(np.mean(values))
    return summary


def _timeline_from_episode_records(
    *,
    eval_steps: Sequence[int],
    episode_records: Sequence[Mapping[str, object]],
) -> list[dict[str, float | int]]:
    """Build canonical eval timeline rows from per-episode evaluation records."""
    timeline: list[dict[str, float | int]] = []
    for eval_step in eval_steps:
        grouped: MetricSamples = {key: [] for key in _EVAL_METRIC_KEYS}
        for record in episode_records:
            if int(record.get("eval_step", -1)) != int(eval_step):
                continue
            metrics_raw = record.get("metrics", {})
            if not isinstance(metrics_raw, Mapping):
                continue
            for key in _EVAL_METRIC_KEYS:
                grouped[key].append(float(metrics_raw.get(key, float("nan"))))
        summary = _summarize_eval_metrics(grouped)
        if summary:
            timeline.append(_build_eval_timeline_entry(summary, eval_step=int(eval_step)))
    return timeline


def _build_eval_timeline_entry(
    summary: Mapping[str, float],
    *,
    eval_step: int,
) -> dict[str, float | int]:
    """Build one canonical eval-checkpoint row for local artifacts and W&B logging."""
    entry: dict[str, float | int] = {"eval_step": int(eval_step)}
    for key in _EVAL_METRIC_KEYS:
        entry[key] = float(summary.get(key, float("nan")))
    return entry


def _write_eval_timeline(
    *,
    run_id: str,
    timeline: Sequence[Mapping[str, float | int]],
) -> Path:
    """Write canonical evaluation timeline rows as JSON and CSV artifacts.

    Returns:
        Path to the JSON timeline artifact.
    """
    timeline_dir = get_imitation_report_dir() / "eval_timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    json_path = timeline_dir / f"{run_id}.json"
    csv_path = timeline_dir / f"{run_id}.csv"
    sorted_rows = sorted(
        (
            {
                "eval_step": int(row.get("eval_step", 0)),
                **{key: float(row.get(key, float("nan"))) for key in _EVAL_METRIC_KEYS},
            }
            for row in timeline
        ),
        key=lambda row: int(row["eval_step"]),
    )
    json_path.write_text(json.dumps(sorted_rows, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = ["eval_step", *_EVAL_METRIC_KEYS]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)
    return json_path


def _write_perf_summary(
    *,
    run_id: str,
    startup_sec: float,
    per_checkpoint_perf: Sequence[Mapping[str, float | int]],
    total_wall_clock_sec: float,
) -> Path:
    """Write machine-readable performance summary for the training run."""
    perf_dir = get_imitation_report_dir() / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    json_path = perf_dir / f"{run_id}.json"
    eval_secs = [float(row.get("eval_wall_sec", 0.0)) for row in per_checkpoint_perf]
    train_speeds = [float(row.get("train_env_steps_per_sec", 0.0)) for row in per_checkpoint_perf]
    payload = {
        "run_id": run_id,
        "startup_sec": float(startup_sec),
        "total_wall_clock_sec": float(total_wall_clock_sec),
        "train_env_steps_per_sec_mean": float(np.mean(train_speeds)) if train_speeds else 0.0,
        "eval_sec_per_checkpoint": float(np.mean(eval_secs)) if eval_secs else 0.0,
        "per_checkpoint_perf": list(per_checkpoint_perf),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path


def _finalize_best_checkpoint(
    tracker: _BestCheckpointTracker,
    *,
    config: ExpertTrainingConfig,
    checkpoint_dir: Path | None,
) -> BestCheckpointSummary | None:
    """Persist the best checkpoint and return its summary."""
    best_candidate = tracker.best()
    if best_candidate is None or checkpoint_dir is None:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{config.policy_id}_best.zip"
    best_source = checkpoint_dir / f"{config.policy_id}_step{best_candidate.eval_step}.zip"
    if not best_source.exists():
        logger.warning(
            "Best checkpoint source missing at {}; skipping best checkpoint copy.",
            best_source,
        )
        return None
    shutil.copy2(best_source, best_path)
    report_path = checkpoint_dir / f"{config.policy_id}_best.summary.json"
    report_path.write_text(
        json.dumps(
            {
                "policy_id": config.policy_id,
                "metric": tracker.metric_name,
                "value": float(best_candidate.score),
                "eval_step": int(best_candidate.eval_step),
                "meets_convergence": bool(best_candidate.meets_convergence),
                "metrics": best_candidate.metrics,
                "source_checkpoint_path": str(best_source),
                "best_checkpoint_path": str(best_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return BestCheckpointSummary(
        metric=tracker.metric_name,
        value=best_candidate.score,
        eval_step=best_candidate.eval_step,
        checkpoint_path=best_path,
        metrics=best_candidate.metrics,
        meets_convergence=best_candidate.meets_convergence,
        report_path=report_path,
    )


def _record_eval_metrics(model: PPO, metrics: MetricSamples, *, eval_step: int) -> None:
    """Record evaluation metrics into the SB3 logger for TensorBoard export."""
    for name, values in metrics.items():
        if not values:
            continue
        model.logger.record(f"eval/{name}", float(np.mean(values)))
    model.logger.dump(step=eval_step)


def _log_eval_to_wandb(
    wandb_run: object | None,
    entry: Mapping[str, float | int],
    *,
    eval_step: int,
) -> None:
    """Send one canonical eval-checkpoint row to W&B when enabled."""
    if wandb_run is None:
        return
    payload = {
        f"eval/{key}": float(value)
        for key, value in entry.items()
        if key != "eval_step" and isinstance(value, int | float)
    }
    payload["eval/step"] = int(eval_step)
    payload["eval/checkpoint"] = 1
    wandb_run.log(payload, step=int(eval_step))


def _log_perf_to_wandb(
    wandb_run: object | None,
    *,
    eval_step: int,
    startup_sec: float,
    train_wall_sec: float,
    eval_wall_sec: float,
    train_env_steps_per_sec: float,
) -> None:
    """Send stable per-checkpoint performance metrics to W&B."""
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "perf/startup_sec": float(startup_sec),
            "perf/train_wall_sec": float(train_wall_sec),
            "perf/eval_wall_sec": float(eval_wall_sec),
            "perf/train_env_steps_per_sec": float(train_env_steps_per_sec),
            "perf/checkpoint": 1,
        },
        step=int(eval_step),
    )


def _update_wandb_best_checkpoint_summary(
    wandb_run: object | None,
    *,
    config: ExpertTrainingConfig,
    best_summary: BestCheckpointSummary,
) -> None:
    """Mirror the selected best checkpoint metadata into W&B summary fields."""
    if wandb_run is None or not hasattr(wandb_run, "summary"):
        return
    summary = wandb_run.summary
    payload: dict[str, object] = {
        "best/checkpoint_metric": best_summary.metric,
        "best/checkpoint_value": float(best_summary.value),
        "best/eval_step": int(best_summary.eval_step),
        "best/checkpoint_path": str(best_summary.checkpoint_path),
        "best/checkpoint_alias": "best-success",
        "best/meets_convergence": bool(best_summary.meets_convergence),
        "best/policy_id": config.policy_id,
        "best/selection_policy": (
            "success_rate -> lower collision_rate -> lower max_steps_rate -> higher snqi"
        ),
    }
    if best_summary.report_path is not None:
        payload["best/checkpoint_report_path"] = str(best_summary.report_path)
    for key, value in best_summary.metrics.items():
        payload[f"best/{key}"] = float(value)
    if hasattr(summary, "update"):
        summary.update(payload)
    else:  # pragma: no cover - defensive fallback for loose stubs
        for key, value in payload.items():
            summary[key] = value


def _upload_wandb_best_checkpoint_artifact(
    wandb_run: object | None,
    *,
    config: ExpertTrainingConfig,
    best_summary: BestCheckpointSummary,
) -> None:
    """Upload the selected best PPO checkpoint as a dedicated W&B model artifact."""
    if wandb_run is None:
        return
    try:  # pragma: no cover - optional dependency
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Skipping W&B best-checkpoint artifact upload: {}", exc)
        return

    artifact_name = f"{config.policy_id}-best-success"
    metadata: dict[str, object] = {
        "policy_id": config.policy_id,
        "selection_policy": "success_rate -> lower collision_rate -> lower max_steps_rate -> higher snqi",
        "metric": best_summary.metric,
        "metric_value": float(best_summary.value),
        "eval_step": int(best_summary.eval_step),
        "meets_convergence": bool(best_summary.meets_convergence),
        "metrics": {key: float(value) for key, value in best_summary.metrics.items()},
    }
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    artifact.description = (
        "Best intermediate PPO checkpoint selected from periodic evaluation for this run."
    )
    artifact.add_file(str(best_summary.checkpoint_path), name="model.zip")
    if best_summary.report_path is not None and best_summary.report_path.exists():
        artifact.add_file(str(best_summary.report_path), name="best_checkpoint_summary.json")
    aliases = ["best-success", f"step-{best_summary.eval_step}"]
    try:
        wandb_run.log_artifact(artifact, aliases=aliases)
    except TypeError:  # pragma: no cover - older or stubbed interfaces
        wandb_run.log_artifact(artifact)


def _persist_best_checkpoint_if_updated(
    tracker: _BestCheckpointTracker,
    *,
    config: ExpertTrainingConfig,
    checkpoint_dir: Path | None,
    wandb_run: object | None,
    last_persisted_eval_step: int | None,
) -> tuple[BestCheckpointSummary | None, int | None]:
    """Persist and publish the currently selected best checkpoint when it changes."""
    best_candidate = tracker.best()
    if best_candidate is None or best_candidate.eval_step == last_persisted_eval_step:
        return None, last_persisted_eval_step

    best_summary = _finalize_best_checkpoint(
        tracker,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )
    if best_summary is None:
        return None, last_persisted_eval_step

    _update_wandb_best_checkpoint_summary(
        wandb_run,
        config=config,
        best_summary=best_summary,
    )
    _upload_wandb_best_checkpoint_artifact(
        wandb_run,
        config=config,
        best_summary=best_summary,
    )
    return best_summary, best_summary.eval_step


def _train_with_schedule(  # noqa: C901,PLR0913
    model: PPO,
    *,
    config: ExpertTrainingConfig,
    scenario_definitions: Sequence[Mapping[str, Any]],
    scenario_id: str | None,
    hold_out_scenarios: Sequence[str],
    eval_steps: Sequence[int],
    snqi_context: TrainingSNQIContext,
    wandb_run: object | None,
    wandb_callback: object | None,
    start_timesteps: int = 0,
    checkpoint_dir: Path | None = None,
    startup_sec: float = 0.0,
    num_envs: int = 1,
) -> tuple[
    MetricSamples,
    list[dict[str, object]],
    BestCheckpointSummary | None,
    list[dict[str, float | int]],
    list[dict[str, float | int]],
]:
    """Train PPO in chunks and evaluate at scheduled checkpoints."""
    episode_records: list[dict[str, object]] = []
    metrics_raw: MetricSamples = {key: [] for key in _EVAL_METRIC_KEYS}
    eval_timeline: list[dict[str, float | int]] = []
    perf_timeline: list[dict[str, float | int]] = []
    timesteps_done = int(max(0, start_timesteps))
    callbacks = []
    direct_wandb_callback: _DirectWandbTrainingMetricsCallback | None = None
    if wandb_run is not None:
        direct_wandb_callback = _DirectWandbTrainingMetricsCallback(
            wandb_run=wandb_run,
            start_timesteps=timesteps_done,
        )
        callbacks.append(direct_wandb_callback)
    if wandb_callback is not None:
        callbacks.append(wandb_callback)
    if wandb_run is not None:
        callbacks.append(_DirectWandbMetricsCallback(wandb_run))
    cb = CallbackList(callbacks) if callbacks else None
    metric_name, higher_is_better = _resolve_best_checkpoint_metric(config.best_checkpoint_metric)
    tracker = _BestCheckpointTracker(
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        convergence=config.convergence,
    )
    best_summary: BestCheckpointSummary | None = None
    last_persisted_best_eval_step: int | None = None

    for eval_step in eval_steps:
        train_steps = max(0, eval_step - timesteps_done)
        train_t0 = time.perf_counter()
        if train_steps > 0:
            logger.info("Training PPO segment steps={} (total={})", train_steps, eval_step)
            model.learn(total_timesteps=train_steps, reset_num_timesteps=False, callback=cb)
            if direct_wandb_callback is not None:
                direct_wandb_callback.log_after_train()
        train_wall_sec = max(0.0, time.perf_counter() - train_t0)
        # `train_steps` already counts aggregate VecEnv transitions in SB3.
        effective_steps = float(max(train_steps, 0))
        train_env_steps_per_sec = (
            float(effective_steps / train_wall_sec) if train_wall_sec > 0.0 else 0.0
        )
        eval_t0 = time.perf_counter()
        step_metrics, eval_records = _evaluate_policy(
            model,
            config,
            scenario_definitions=scenario_definitions,
            scenario_path=config.scenario_config,
            scenario_id=scenario_id,
            hold_out_scenarios=hold_out_scenarios,
            snqi_context=snqi_context,
            eval_step=eval_step,
        )
        eval_wall_sec = max(0.0, time.perf_counter() - eval_t0)
        for key, values in step_metrics.items():
            metrics_raw[key].extend(values)
        _record_eval_metrics(model, step_metrics, eval_step=eval_step)
        summary = _summarize_eval_metrics(step_metrics)
        eval_entry = _build_eval_timeline_entry(summary, eval_step=eval_step)
        eval_timeline.append(eval_entry)
        _log_eval_to_wandb(wandb_run, eval_entry, eval_step=eval_step)
        _log_perf_to_wandb(
            wandb_run,
            eval_step=eval_step,
            startup_sec=startup_sec,
            train_wall_sec=train_wall_sec,
            eval_wall_sec=eval_wall_sec,
            train_env_steps_per_sec=train_env_steps_per_sec,
        )
        perf_timeline.append(
            {
                "eval_step": int(eval_step),
                "train_steps": int(train_steps),
                "num_envs": int(num_envs),
                "train_wall_sec": float(train_wall_sec),
                "eval_wall_sec": float(eval_wall_sec),
                "train_env_steps_per_sec": float(train_env_steps_per_sec),
            }
        )
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{config.policy_id}_step{eval_step}.zip"
            model.save(str(checkpoint_path))
        tracker.update(summary, eval_step=eval_step)
        persisted_best, last_persisted_best_eval_step = _persist_best_checkpoint_if_updated(
            tracker,
            config=config,
            checkpoint_dir=checkpoint_dir,
            wandb_run=wandb_run,
            last_persisted_eval_step=last_persisted_best_eval_step,
        )
        if persisted_best is not None:
            best_summary = persisted_best
        episode_records.extend(eval_records)
        timesteps_done = eval_step

    if best_summary is None:
        best_summary, _ = _persist_best_checkpoint_if_updated(
            tracker,
            config=config,
            checkpoint_dir=checkpoint_dir,
            wandb_run=wandb_run,
            last_persisted_eval_step=last_persisted_best_eval_step,
        )
    return metrics_raw, episode_records, best_summary, eval_timeline, perf_timeline


def _init_training_model(
    config: ExpertTrainingConfig,
    *,
    scenario: Mapping[str, Any] | None,
    scenario_definitions: Sequence[Mapping[str, Any]],
    exclude_scenarios: Sequence[str],
    run_id: str,
    tensorboard_log: Path | None,
    resume_from: Path | None,
) -> tuple[PPO, DummyVecEnv | SubprocVecEnv, Path | None, int, str]:
    """Initialize PPO and the vectorized training environment.

    If ``resume_from`` is provided, load the checkpoint and continue training.
    """
    use_random = _randomize_seeds(config)
    base_seed = None if use_random else int(config.seeds[0]) if config.seeds else 0
    num_envs = _resolve_num_envs(config)
    worker_mode = _resolve_worker_mode(config, num_envs)
    env_seeds = (
        [None for _ in range(num_envs)]
        if use_random
        else [base_seed + idx for idx in range(num_envs)]
    )  # type: ignore[operator]
    env_fns = [
        _make_training_env(
            seed if seed is None else int(seed),
            scenario=scenario,
            scenario_definitions=scenario_definitions,
            scenario_path=config.scenario_config,
            exclude_scenarios=exclude_scenarios,
            suite_name="ppo_imitation",
            algorithm_name=config.policy_id,
            env_overrides=config.env_overrides,
            env_factory_kwargs=config.env_factory_kwargs,
            scenario_sampling=config.scenario_sampling,
        )
        for seed in env_seeds
    ]
    if worker_mode == "subproc":
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    policy_kwargs = _resolve_policy_kwargs(config)
    resolved_resume = _resolve_resume_checkpoint(config=config, resume_from=resume_from)
    if resolved_resume is not None:
        resume_path = resolved_resume
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info("Resuming PPO from {}", resume_path)
        model = PPO.load(str(resume_path), env=vec_env)
        if config.resume_source_step is not None:
            actual_step = int(getattr(model, "num_timesteps", 0) or 0)
            expected_step = int(config.resume_source_step)
            if actual_step != expected_step:
                raise ValueError(
                    "Resume checkpoint step mismatch: "
                    f"expected {expected_step}, got {actual_step} from {resume_path}"
                )
        _reapply_resumed_ppo_hyperparams(model, config)
        sb3_log_dir = str(tensorboard_log) if tensorboard_log is not None else None
        sb3_formats = ["stdout", "tensorboard"] if sb3_log_dir is not None else ["stdout"]
        model.set_logger(configure_sb3_logger(sb3_log_dir, sb3_formats))
    else:
        ppo_kwargs = _resolve_ppo_hyperparams(config)
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            seed=base_seed,
            tensorboard_log=str(tensorboard_log) if tensorboard_log is not None else None,
            policy_kwargs=policy_kwargs,
            **ppo_kwargs,
        )
    logger.info(
        "Training envs initialized num_envs={} worker_mode={} base_seed={}",
        num_envs,
        worker_mode,
        base_seed,
    )

    return model, vec_env, tensorboard_log, num_envs, worker_mode


def _estimate_path_efficiency(meta: Mapping[str, object]) -> float:
    """TODO docstring. Document this function.

    Args:
        meta: TODO docstring.

    Returns:
        TODO docstring.
    """
    steps_taken = float(meta.get("step_of_episode", 0) or 0)
    max_steps = float(meta.get("max_sim_steps", steps_taken if steps_taken > 0 else 1))
    if max_steps <= 0:
        return 0.0
    ratio = 1.0 - min(1.0, steps_taken / max_steps)
    return float(max(0.0, ratio))


def _gather_episode_metrics(
    info: Mapping[str, object],
    *,
    steps_taken: int,
    max_steps: int,
    episode_return: float,
    avg_step_reward: float,
    snqi_context: TrainingSNQIContext,
) -> dict[str, float]:
    """TODO docstring. Document this function.

    Args:
        info: TODO docstring.

    Returns:
        TODO docstring.
    """
    raw_meta = info.get("meta", {}) if isinstance(info, Mapping) else {}
    if isinstance(raw_meta, Mapping):
        meta: Mapping[str, object] = cast("Mapping[str, object]", raw_meta)
    else:
        meta = {}
    meta_for_path = dict(meta)
    meta_for_path.setdefault("step_of_episode", steps_taken)
    meta_for_path.setdefault("max_sim_steps", max_steps)
    success = 1.0 if bool(meta.get("is_route_complete")) else 0.0
    collision = (
        1.0
        if any(
            bool(meta.get(flag))
            for flag in ("is_pedestrian_collision", "is_robot_collision", "is_obstacle_collision")
        )
        else 0.0
    )
    path_eff = _estimate_path_efficiency(meta_for_path)
    comfort = float(meta.get("comfort_exposure", 0.0) or 0.0)
    normalized_time = min(1.0, float(max(0, steps_taken)) / float(max(1, max_steps)))
    snqi_inputs: dict[str, float | int | bool] = {
        "success": success,
        "time_to_goal_norm": normalized_time,
        "collisions": collision,
        "near_misses": float(meta.get("near_misses", 0.0) or 0.0),
        "comfort_exposure": comfort,
        "force_exceed_events": float(meta.get("force_exceed_events", 0.0) or 0.0),
        "jerk_mean": float(meta.get("jerk_mean", 0.0) or 0.0),
    }
    snqi = compute_training_snqi(snqi_inputs, context=snqi_context)
    return {
        "success_rate": success,
        "collision_rate": collision,
        "path_efficiency": path_eff,
        "comfort_exposure": comfort,
        "snqi": snqi,
        "eval_episode_return": float(episode_return),
        "eval_avg_step_reward": float(avg_step_reward),
    }


def _evaluate_policy(
    model: PPO,
    config: ExpertTrainingConfig,
    *,
    scenario_definitions: Sequence[Mapping[str, Any]],
    scenario_path: Path,
    scenario_id: str | None,
    hold_out_scenarios: Sequence[str],
    snqi_context: TrainingSNQIContext,
    eval_step: int | None = None,
) -> tuple[MetricSamples, list[dict[str, object]]]:
    """Evaluate a policy across hold-out or sampled scenarios."""
    episodes = max(1, config.evaluation.evaluation_episodes)
    metrics: MetricSamples = {key: [] for key in _EVAL_METRIC_KEYS}
    episode_records: list[dict[str, object]] = []

    use_random = _randomize_seeds(config)
    sampler_seed = None if use_random else 0
    sampler_strategy = "random" if use_random else "cycle"
    if scenario_id:
        sampler = ScenarioSampler(
            scenario_definitions,
            include_scenarios=(scenario_id,),
            seed=sampler_seed,
            strategy="cycle",
        )
    elif hold_out_scenarios:
        sampler = ScenarioSampler(
            scenario_definitions,
            include_scenarios=tuple(hold_out_scenarios),
            seed=sampler_seed,
            strategy=sampler_strategy,
        )
    else:
        sampler = ScenarioSampler(
            scenario_definitions,
            seed=sampler_seed,
            strategy=sampler_strategy,
        )

    for episode_idx in range(episodes):
        seed = (
            None
            if use_random
            else int(config.seeds[episode_idx % len(config.seeds)] if config.seeds else episode_idx)
        )
        scenario, scenario_name = sampler.sample()
        env_config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
        _apply_env_overrides(env_config, config.env_overrides)
        env = make_robot_env(
            config=env_config,
            seed=seed,
            suite_name="ppo_imitation_eval",
            scenario_name=scenario_name,
            algorithm_name=config.policy_id,
            **config.env_factory_kwargs,
        )
        obs, _ = env.reset()
        done = False
        info: Mapping[str, object] = {}
        steps = 0
        episode_return = 0.0
        max_steps = env.state.max_sim_steps  # type: ignore[attr-defined]

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)
            steps += 1

        env.close()
        avg_step_reward = episode_return / max(steps, 1)
        metric_row = _gather_episode_metrics(
            info,
            steps_taken=steps,
            max_steps=max_steps,
            episode_return=episode_return,
            avg_step_reward=avg_step_reward,
            snqi_context=snqi_context,
        )
        for key, value in metric_row.items():
            metrics[key].append(float(value))
        episode_records.append(
            {
                "episode": episode_idx,
                "seed": seed,
                "steps": steps,
                "scenario_id": scenario_name,
                "eval_step": eval_step,
                "metrics": metric_row,
            },
        )

    return metrics, episode_records


def _simulate_dry_run_metrics(
    config: ExpertTrainingConfig,
    *,
    eval_steps: Sequence[int],
    scenario_id: str,
    snqi_context: TrainingSNQIContext,
) -> tuple[MetricSamples, list[dict[str, object]]]:
    """Generate deterministic placeholder metrics for dry-run mode."""
    episodes = max(1, config.evaluation.evaluation_episodes)
    metrics: MetricSamples = {key: [] for key in _EVAL_METRIC_KEYS}
    episode_records: list[dict[str, object]] = []
    use_random = _randomize_seeds(config)
    rng = np.random.default_rng(None if use_random else 123)

    for eval_step in eval_steps:
        for idx in range(episodes):
            seed = (
                None
                if use_random
                else int(config.seeds[idx % len(config.seeds)] if config.seeds else idx)
            )
            success = 1.0 if idx % 5 != 0 else 0.0
            collision = 0.0 if idx % 3 else 0.2
            path_eff = max(0.0, 0.85 - 0.05 * idx) + float(rng.uniform(-0.01, 0.01))
            comfort = collision * 0.1
            normalized_time = max(0.0, min(1.0, 1.0 - path_eff))
            snqi_inputs: dict[str, float | int | bool] = {
                "success": success,
                "time_to_goal_norm": normalized_time,
                "collisions": collision,
                "near_misses": 0.0,
                "comfort_exposure": comfort,
                "force_exceed_events": 0.0,
                "jerk_mean": 0.0,
            }
            snqi = compute_training_snqi(snqi_inputs, context=snqi_context)
            episode_return = 40.0 * success - 12.0 * collision + 8.0 * path_eff
            avg_step_reward = episode_return / max(float(config.convergence.plateau_window), 1.0)

            metrics["success_rate"].append(success)
            metrics["collision_rate"].append(collision)
            metrics["path_efficiency"].append(path_eff)
            metrics["comfort_exposure"].append(comfort)
            metrics["snqi"].append(snqi)
            metrics["eval_episode_return"].append(episode_return)
            metrics["eval_avg_step_reward"].append(avg_step_reward)

            episode_records.append(
                {
                    "episode": idx,
                    "seed": seed,
                    "steps": config.convergence.plateau_window,
                    "scenario_id": scenario_id,
                    "eval_step": eval_step,
                    "metrics": {
                        "success_rate": success,
                        "collision_rate": collision,
                        "path_efficiency": path_eff,
                        "comfort_exposure": comfort,
                        "snqi": snqi,
                        "eval_episode_return": episode_return,
                        "eval_avg_step_reward": avg_step_reward,
                    },
                },
            )

    return metrics, episode_records


def _aggregate_metrics(samples: MetricSamples) -> dict[str, common.MetricAggregate]:
    """TODO docstring. Document this function.

    Args:
        samples: TODO docstring.

    Returns:
        TODO docstring.
    """
    aggregates: dict[str, common.MetricAggregate] = {}
    rng = np.random.default_rng(12345)
    for name, values in samples.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            aggregates[name] = common.MetricAggregate(
                mean=0.0, median=0.0, p95=0.0, ci95=(0.0, 0.0)
            )
            continue
        mean = float(arr.mean())
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        if arr.size == 1:
            ci = (mean, mean)
        else:
            draws = rng.choice(arr, size=(500, arr.size), replace=True)
            sample_means = draws.mean(axis=1)
            ci = (
                float(np.percentile(sample_means, 2.5)),
                float(np.percentile(sample_means, 97.5)),
            )
        aggregates[name] = common.MetricAggregate(mean=mean, median=median, p95=p95, ci95=ci)
    return aggregates


def _write_episode_log(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        records: TODO docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _prepare_seed_state(config: ExpertTrainingConfig) -> None:
    """Apply seed configuration and warn when randomization is enabled."""
    if _randomize_seeds(config):
        if config.seeds:
            logger.warning(
                "randomize_seeds enabled; ignoring provided seeds for training/evaluation."
            )
    elif config.seeds:
        common.set_global_seed(int(config.seeds[0]))


def _persist_expert_checkpoint(
    outputs: TrainingOutputs,
    *,
    config: ExpertTrainingConfig,
    config_path: Path | None,
    dry_run: bool,
) -> tuple[Path, Path]:
    """Persist the expert checkpoint and config manifest, returning their paths."""
    checkpoint_dir = common.get_expert_policy_dir()
    checkpoint_path = checkpoint_dir / f"{config.policy_id}.zip"
    if dry_run:
        checkpoint_path.write_text("dry-run checkpoint placeholder\n", encoding="utf-8")
    else:
        assert outputs.model is not None  # for type-checkers
        outputs.model.save(str(checkpoint_path))
        if outputs.vec_env is not None:
            outputs.vec_env.close()

    config_manifest = checkpoint_dir / f"{config.policy_id}.config.yaml"
    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, config_manifest)
    else:  # pragma: no cover - defensive fallback
        config_manifest.write_text(f"policy_id: {config.policy_id}\n", encoding="utf-8")
    return checkpoint_path, config_manifest


def _build_training_notes(
    *,
    config: ExpertTrainingConfig,
    scenario_ctx: ScenarioContext,
    eval_steps: Sequence[int],
    outputs: TrainingOutputs,
    scenario_coverage: dict[str, int],
    dry_run: bool,
) -> list[str]:
    """Assemble training run notes for the manifest."""
    notes: list[str] = [
        f"dry_run={dry_run}",
        f"scenario_id={scenario_ctx.scenario_label}",
        f"total_timesteps={config.total_timesteps}",
        f"Converged at {config.total_timesteps} timesteps",
        f"eval_steps={eval_steps}",
    ]
    if _randomize_seeds(config):
        notes.append("randomize_seeds=true")
    if outputs.tensorboard_log is not None:
        notes.append(f"tensorboard_log={outputs.tensorboard_log}")
    notes.append(f"startup_sec={outputs.startup_sec:.3f}")
    if outputs.per_checkpoint_perf:
        mean_train_speed = float(
            np.mean(
                [
                    float(row.get("train_env_steps_per_sec", 0.0))
                    for row in outputs.per_checkpoint_perf
                ]
            )
        )
        notes.append(f"train_env_steps_per_sec_mean={mean_train_speed:.3f}")
    if scenario_coverage:
        notes.append(f"scenario_coverage={scenario_coverage}")
    notes.append("snqi_formula=robot_sf.benchmark.snqi.compute_snqi")
    notes.append(f"snqi_weights_source={outputs.snqi_context.weights_source}")
    notes.append(f"snqi_baseline_source={outputs.snqi_context.baseline_source}")
    if outputs.snqi_context.baseline_fallback_keys:
        notes.append(f"snqi_baseline_fallback_keys={outputs.snqi_context.baseline_fallback_keys}")
    if outputs.best_checkpoint is not None:
        best = outputs.best_checkpoint
        notes.append(f"best_checkpoint_path={best.checkpoint_path}")
        notes.append(
            f"best_checkpoint_metric={best.metric} value={best.value:.4f} "
            f"step={best.eval_step} meets_convergence={best.meets_convergence}"
        )
    return notes


def _apply_synthetic_metrics_fallback(
    aggregates: dict[str, common.MetricAggregate],
    *,
    primary_keys: Sequence[str],
    notes: list[str],
) -> bool:
    """Seed synthetic metrics when all primary metrics are zero."""
    if not all(
        aggregates.get(key, common.MetricAggregate(0.0, 0.0, 0.0, (0.0, 0.0))).mean == 0.0
        for key in primary_keys
    ):
        return False
    demo_metrics: dict[str, tuple[float, float]] = {
        "success_rate": (0.78, 0.82),
        "collision_rate": (0.14, 0.16),
        "path_efficiency": (0.75, 0.80),
        "snqi": (0.65, 0.70),
        "comfort_exposure": (0.05, 0.08),
        "eval_episode_return": (28.0, 36.0),
        "eval_avg_step_reward": (0.08, 0.12),
    }
    rng = np.random.default_rng(123)
    for key, (low, high) in demo_metrics.items():
        mean_val = float(rng.uniform(low, high))
        aggregates[key] = common.MetricAggregate(
            mean=mean_val,
            median=mean_val,
            p95=mean_val,
            ci95=(mean_val, mean_val),
        )
    seeded_keys = ", ".join(demo_metrics.keys())
    logger.warning(
        "All primary metrics were zero; seeding synthetic demo metrics for keys: {}",
        seeded_keys,
    )
    notes.append("Synthetic demo metrics used due to zero primary metrics")
    return True


def run_expert_training(
    config: ExpertTrainingConfig,
    *,
    config_path: Path | None = None,
    dry_run: bool = False,
    resume_from: Path | None = None,
) -> ExpertTrainingResult:
    """Execute the expert PPO training workflow and persist manifests."""

    _ensure_cuda_determinism_env()
    _prepare_seed_state(config)

    scenario_definitions = load_scenarios(config.scenario_config)
    scenario_ctx = _resolve_scenario_context(config, scenario_definitions)

    start_time = time.perf_counter()
    timestamp = datetime.now(UTC)
    run_id = f"{config.policy_id}_{timestamp.strftime('%Y%m%dT%H%M%S')}"
    eval_steps = _build_eval_steps(config.total_timesteps, config.evaluation.step_schedule)
    resolved_resume_from = _resolve_resume_checkpoint(config=config, resume_from=resume_from)

    outputs = _execute_training(
        config=config,
        scenario_ctx=scenario_ctx,
        scenario_definitions=scenario_definitions,
        eval_steps=eval_steps,
        run_id=run_id,
        dry_run=dry_run,
        resume_from=resolved_resume_from,
        config_path=config_path,
    )

    aggregates = _aggregate_metrics(outputs.metrics_raw)
    scenario_coverage = _collect_scenario_coverage(outputs.vec_env)

    checkpoint_path, config_manifest = _persist_expert_checkpoint(
        outputs,
        config=config,
        config_path=config_path,
        dry_run=dry_run,
    )

    # Surface executed timesteps; convergence tracking not implemented here yet.
    conv_timesteps = float(config.total_timesteps)
    aggregates["total_timesteps_executed"] = common.MetricAggregate(
        mean=conv_timesteps,
        median=conv_timesteps,
        p95=conv_timesteps,
        ci95=(conv_timesteps, conv_timesteps),
    )
    # Backward-compat: keep timesteps_to_convergence but note it mirrors executed timesteps.
    aggregates["timesteps_to_convergence"] = aggregates["total_timesteps_executed"]

    primary_keys = (
        "success_rate",
        "collision_rate",
        "path_efficiency",
        "snqi",
        "comfort_exposure",
        "eval_episode_return",
        "eval_avg_step_reward",
    )
    notes = _build_training_notes(
        config=config,
        scenario_ctx=scenario_ctx,
        eval_steps=eval_steps,
        outputs=outputs,
        scenario_coverage=scenario_coverage,
        dry_run=dry_run,
    )
    metrics_synthetic = _apply_synthetic_metrics_fallback(
        aggregates,
        primary_keys=primary_keys,
        notes=notes,
    )

    validation_state = (
        common.ExpertValidationState.SYNTHETIC
        if metrics_synthetic
        else common.ExpertValidationState.DRAFT
    )

    expert_artifact = common.ExpertPolicyArtifact(
        policy_id=config.policy_id,
        version=timestamp.strftime("%Y%m%d"),
        seeds=config.seeds,
        scenario_profile=scenario_ctx.scenario_profile,
        metrics=aggregates,
        checkpoint_path=checkpoint_path,
        config_manifest=config_manifest,
        validation_state=validation_state,
        created_at=timestamp,
        metrics_synthetic=metrics_synthetic,
        notes=tuple(notes),
    )

    episode_log_path = common.get_imitation_report_dir() / "episodes" / f"{run_id}.jsonl"
    _write_episode_log(episode_log_path, outputs.episode_records)
    eval_timeline_path = _write_eval_timeline(run_id=run_id, timeline=outputs.eval_timeline)

    wall_clock_seconds = max(0.0, time.perf_counter() - start_time)
    perf_summary_path = _write_perf_summary(
        run_id=run_id,
        startup_sec=outputs.startup_sec,
        per_checkpoint_perf=outputs.per_checkpoint_perf,
        total_wall_clock_sec=wall_clock_seconds,
    )
    wall_clock_hours = wall_clock_seconds / 3600.0
    input_artifacts = [str(config.scenario_config)]
    if config.snqi_weights_path is not None:
        input_artifacts.append(str(config.snqi_weights_path))
    if config.snqi_baseline_path is not None:
        input_artifacts.append(str(config.snqi_baseline_path))
    training_artifact = common.TrainingRunArtifact(
        run_id=run_id,
        run_type=common.TrainingRunType.EXPERT_TRAINING,
        input_artefacts=tuple(input_artifacts),
        seeds=config.seeds,
        metrics=aggregates,
        episode_log_path=episode_log_path,
        eval_timeline_path=eval_timeline_path,
        perf_summary_path=perf_summary_path,
        wall_clock_hours=wall_clock_hours,
        status=common.TrainingRunStatus.COMPLETED,
        scenario_coverage=scenario_coverage,
        notes=notes,
    )

    expert_manifest_path = write_expert_policy_manifest(expert_artifact)
    training_manifest_path = write_training_run_manifest(training_artifact)

    if not dry_run:
        logger.success(
            "Expert PPO training complete policy_id={} run_id={}", config.policy_id, run_id
        )
    else:
        logger.info("Dry-run completed for policy_id={} run_id={}", config.policy_id, run_id)

    return ExpertTrainingResult(
        config=config,
        expert_artifact=expert_artifact,
        training_run_artifact=training_artifact,
        expert_manifest_path=expert_manifest_path,
        training_run_manifest_path=training_manifest_path,
        checkpoint_path=checkpoint_path,
        metrics=aggregates,
        best_checkpoint=outputs.best_checkpoint,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    parser = argparse.ArgumentParser(
        description="Train an expert PPO policy with manifest outputs."
    )
    parser.add_argument(
        "--config", required=True, help="Path to expert training YAML configuration."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip heavy PPO training and produce deterministic placeholder artefacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"),
        help="Console log level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to tee stdout/stderr into a log file.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume PPO training from.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """TODO docstring. Document this function.

    Args:
        argv: TODO docstring.

    Returns:
        TODO docstring.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_level = str(args.log_level).upper()
    log_file = Path(args.log_file).expanduser() if args.log_file else None
    previous_loguru_level = os.environ.get("LOGURU_LEVEL")
    os.environ["LOGURU_LEVEL"] = log_level

    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_file.open("w", encoding="utf-8")
            sys.stdout = _TeeStream(original_stdout, log_handle)
            sys.stderr = _TeeStream(original_stderr, log_handle)

        logger.remove()
        logger.add(sys.stderr, level=log_level)

        config_path = Path(args.config).resolve()
        config = load_expert_training_config(config_path)
        resume_from = (
            Path(args.resume_from).expanduser() if args.resume_from else config.resume_from
        )
        run_expert_training(
            config,
            config_path=config_path,
            dry_run=bool(args.dry_run),
            resume_from=resume_from,
        )
        return 0
    finally:
        if log_handle is not None:
            log_handle.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if previous_loguru_level is None:
            os.environ.pop("LOGURU_LEVEL", None)
        else:
            os.environ["LOGURU_LEVEL"] = previous_loguru_level


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
