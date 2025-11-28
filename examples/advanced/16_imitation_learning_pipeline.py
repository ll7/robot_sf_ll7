#!/usr/bin/env python3
"""Imitation Learning Pipeline - End-to-End Example.

This example demonstrates the complete imitation learning workflow by calling
the training scripts in sequence:

1. Train an expert PPO policy (or use existing)
2. Collect expert trajectories
3. Pre-train a new policy via behavioral cloning
4. Fine-tune with PPO
5. Compare baseline vs pre-trained performance

**Purpose**: Show complete workflow for accelerating PPO training with expert demonstrations

**Prerequisites**:
- uv sync --all-extras
- Sufficient disk space for models and trajectories (~500MB)
- 30-60 minutes runtime for full pipeline (less in demo mode)

**Usage**:
    # Run full pipeline (expert training takes longest - 30-60 min)
    uv run python examples/advanced/16_imitation_learning_pipeline.py

    # Skip expert training if you already have a trained policy
    uv run python examples/advanced/16_imitation_learning_pipeline.py --skip-expert --policy-id ppo_expert_v1

    # Quick demo mode (reduced episodes/timesteps)
    uv run python examples/advanced/16_imitation_learning_pipeline.py --demo-mode

**Note**: expert_ppo.yaml drives expert training. BC and PPO fine-tuning configs are
generated automatically under output/tmp for each run, so no manual edits are needed.

**Output**:
- Timestamped run root: output/benchmarks/<timestamp>/
- Expert policy: <run-root>/expert_policies/
- Trajectories: <run-root>/expert_trajectories/
- Pre-trained policy: <run-root>/expert_policies/
- Comparison report: <run-root>/ppo_imitation/comparisons/

**Related**:
- Full documentation: docs/imitation_learning_pipeline.md
- Quickstart guide: specs/001-ppo-imitation-pretrain/quickstart.md
- Individual scripts: scripts/training/train_expert_ppo.py, collect_expert_trajectories.py, etc.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.benchmark.imitation_manifest import get_training_run_manifest_path
from robot_sf.common.artifact_paths import get_imitation_report_dir
from robot_sf.sim.registry import select_best_backend
from robot_sf.telemetry import (
    ManifestWriter,
    PipelineStepDefinition,
    ProgressTracker,
    RecommendationEngine,
    RunTrackerConfig,
    TelemetrySampler,
    generate_run_id,
)
from robot_sf.telemetry.models import PipelineRunRecord, PipelineRunStatus, serialize_many
from robot_sf.training.imitation_config import build_imitation_pipeline_steps

PIPELINE_CONFIG_DIR = Path("output/tmp/imitation_pipeline")
RUN_ROOT: Path | None = None


def _ensure_run_root() -> Path:
    """Ensure a timestamped artifact root under output/benchmarks and export env override."""

    global RUN_ROOT, PIPELINE_CONFIG_DIR
    if RUN_ROOT is not None:
        return RUN_ROOT
    if os.environ.get("ROBOT_SF_ARTIFACT_ROOT"):
        RUN_ROOT = Path(os.environ["ROBOT_SF_ARTIFACT_ROOT"]).expanduser()
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = Path("output/benchmarks") / timestamp
        os.environ["ROBOT_SF_ARTIFACT_ROOT"] = str(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_CONFIG_DIR = RUN_ROOT / "tmp" / "imitation_pipeline"
    return RUN_ROOT


def _load_training_manifest(run_id: str) -> dict[str, Any]:
    """Best-effort load of a training run manifest; returns {} if missing or invalid."""

    path = get_training_run_manifest_path(run_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - defensive fallback
        return {}


def _build_comparison_summary(
    group_id: str, baseline_run_id: str, pretrained_run_id: str
) -> dict[str, Any]:
    """Assemble a tracker-friendly summary from the comparison report."""

    comparison_path = get_imitation_report_dir() / "comparisons" / f"{group_id}_comparison.json"
    if not comparison_path.exists():
        return {}

    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return {}

    summary: dict[str, Any] = {
        "baseline_run_id": baseline_run_id,
        "pretrained_run_id": pretrained_run_id,
        "comparison_path": str(comparison_path),
        "comparison": comparison,
    }

    timesteps = comparison.get("timesteps_to_convergence") or {}
    if "baseline" in timesteps:
        summary["baseline_timesteps"] = [timesteps["baseline"]]
    if "pretrained" in timesteps:
        summary["pretrained_timesteps"] = [timesteps["pretrained"]]

    seeds: list[int] = []
    for run_id in (baseline_run_id, pretrained_run_id):
        manifest = _load_training_manifest(run_id)
        if isinstance(manifest.get("seeds"), list):
            try:
                seeds.extend(int(s) for s in manifest["seeds"])
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Could not convert some seeds to int in manifest",
                    f"for run_id={run_id}: {manifest.get('seeds')!r} ({e})",
                )
    if seeds:
        summary["seeds"] = sorted({int(s) for s in seeds})
    return summary


@dataclass(slots=True)
class TrackerContext:
    """Holds run-tracking handles for the pipeline."""

    run_id: str
    writer: ManifestWriter
    tracker: ProgressTracker
    created_at: datetime
    enabled_steps: tuple[str, ...]
    initiator: str
    scenario_config: Path
    telemetry_sampler: TelemetrySampler | None = None
    recommendation_engine: RecommendationEngine | None = None
    telemetry_samples: int = 0


def _run_command(cmd: list[str], step_name: str, env: dict[str, str] | None = None) -> int:
    """Run a subprocess command and log results.

    Args:
        cmd: Command and arguments to run
        step_name: Human-readable step name for logging

    Returns:
        Exit code (0 for success)
    """
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)

    if result.returncode != 0:
        logger.error(f"{step_name} failed with exit code {result.returncode}")
        return result.returncode

    logger.success(f"{step_name} completed successfully")
    return 0


def _load_policy_id_from_config(config_path: Path) -> str:
    """Read the policy_id from the expert training YAML config."""

    data = _load_yaml_config(config_path)
    policy_id = data.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(
            "expert_ppo.yaml must define a top-level 'policy_id' key to identify the checkpoint.",
        )

    return policy_id


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config and require a mapping."""

    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a mapping at the top level.")
    return data


def _write_pipeline_config(filename: str, payload: dict[str, Any]) -> Path:
    """Write a temporary YAML config under output/tmp for scripted steps."""

    PIPELINE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = PIPELINE_CONFIG_DIR / filename
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    logger.debug("Wrote pipeline config {}", config_path)
    return config_path


def _prepare_bc_config(dataset_id: str, bc_policy_id: str, demo_mode: bool) -> Path:
    """Create a BC config tailored to the current pipeline run."""

    bc_epochs = 5 if demo_mode else 20
    payload = {
        "run_id": f"bc_pretrain_{dataset_id}",
        "dataset_id": dataset_id,
        "policy_output_id": bc_policy_id,
        "bc_epochs": bc_epochs,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "random_seeds": [42, 43, 44],
    }
    return _write_pipeline_config("bc_pretrain.yaml", payload)


def _prepare_ppo_config(
    bc_policy_id: str,
    finetuned_policy_id: str,
    total_timesteps: int,
    demo_mode: bool,
) -> Path:
    """Create a PPO fine-tuning config tailored to the current pipeline run."""

    payload = {
        "run_id": f"ppo_finetune_{finetuned_policy_id}",
        "pretrained_policy_id": bc_policy_id,
        "total_timesteps": total_timesteps,
        "learning_rate": 0.0001 if not demo_mode else 0.0003,
        "random_seeds": [42, 43, 44],
    }
    return _write_pipeline_config("ppo_finetune.yaml", payload)


def _override_expert_config_policy_id(config_path: Path, policy_id: str) -> Path:
    """Write a temp expert config with an overridden policy_id for this run."""

    data = _load_yaml_config(config_path)
    data["policy_id"] = policy_id
    scenario_cfg = data.get("scenario_config")
    if isinstance(scenario_cfg, str):
        # Normalize to absolute path so downstream copies do not break relative references
        data["scenario_config"] = str((config_path.parent / scenario_cfg).resolve())
    filename = f"{config_path.stem}__policy_override.yaml"
    logger.info("Applying policy_id override -> {}", policy_id)
    return _write_pipeline_config(filename, data)


def _run_manifest_dir() -> Path:
    return get_training_run_manifest_path("placeholder").parent


def _discover_latest_training_run_id(prefix: str) -> str | None:
    """Return the newest training run identifier with the given prefix."""

    base_dir = _run_manifest_dir()
    if not base_dir.exists():
        return None
    candidates: list[tuple[float, str]] = []
    pattern = f"{prefix}*.json"
    for path in base_dir.glob(pattern):
        try:
            stat = path.stat()
        except OSError:  # pragma: no cover - filesystem races
            continue
        candidates.append((stat.st_mtime, path.stem))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_tracker_destination(hint: str | None) -> tuple[RunTrackerConfig, str]:
    config = RunTrackerConfig()
    run_id = generate_run_id("pipeline")
    if not hint:
        return config, run_id
    candidate = Path(hint).expanduser()
    parts = candidate.parts
    if "run-tracker" in parts:
        index = parts.index("run-tracker")
        tracker_root = Path(*parts[: index + 1]) if index >= 0 else None
        run_segment = Path(*parts[index + 1 :]) if index + 1 < len(parts) else None
        if tracker_root is not None and run_segment and run_segment.name:
            artifact_root = (
                tracker_root.parent if tracker_root.parent != Path(".") else config.artifact_root
            )
            return RunTrackerConfig(artifact_root=artifact_root), run_segment.name
    if candidate.parent != Path(".") or candidate.is_absolute():
        artifact_root = candidate.parent if candidate.parent != Path(".") else config.artifact_root
        return RunTrackerConfig(artifact_root=artifact_root), candidate.name or run_id
    return config, candidate.name or run_id


def _create_tracker_context(
    step_definitions: list[PipelineStepDefinition],
    *,
    tracker_output: str | None,
    initiator: str,
    scenario_config: Path,
    telemetry_interval: float | None,
) -> TrackerContext:
    config, run_id = _resolve_tracker_destination(tracker_output)
    writer = ManifestWriter(config, run_id)
    tracker = ProgressTracker(step_definitions, writer=writer, log_fn=logger.info)
    context = TrackerContext(
        run_id=run_id,
        writer=writer,
        tracker=tracker,
        created_at=datetime.now(UTC),
        enabled_steps=tuple(step.step_id for step in step_definitions),
        initiator=initiator,
        scenario_config=scenario_config,
    )
    _append_run_snapshot(context, PipelineRunStatus.RUNNING)
    tracker.enable_failure_guard(
        heartbeat=lambda status: _append_run_snapshot(context, status),
    )
    _start_tracker_telemetry(context, interval_seconds=telemetry_interval)
    return context


def _start_tracker_telemetry(
    context: TrackerContext,
    *,
    interval_seconds: float | None,
) -> None:
    if interval_seconds is None or interval_seconds <= 0:
        return
    engine = RecommendationEngine()
    sampler = TelemetrySampler(
        context.writer,
        progress_tracker=context.tracker,
        started_at=context.created_at,
        interval_seconds=max(interval_seconds, 0.5),
    )
    sampler.add_consumer(engine.observe_snapshot)
    sampler.start()
    context.telemetry_sampler = sampler
    context.recommendation_engine = engine


def _stop_tracker_telemetry(context: TrackerContext) -> None:
    sampler = context.telemetry_sampler
    if sampler is None:
        return
    sampler.stop()
    context.telemetry_samples = sampler.samples_written
    context.telemetry_sampler = None


def _build_tracker_summary(
    context: TrackerContext,
    base_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = dict(base_summary or {})
    telemetry_summary = _telemetry_summary(context)
    if telemetry_summary:
        summary.setdefault("telemetry", telemetry_summary)
    if context.telemetry_samples:
        summary.setdefault("telemetry_samples", context.telemetry_samples)
    return summary


def _telemetry_summary(context: TrackerContext) -> dict[str, Any]:
    engine = context.recommendation_engine
    if engine is None:
        return {}
    summary = engine.summary()
    if context.telemetry_samples and "telemetry_samples" not in summary:
        summary["telemetry_samples"] = context.telemetry_samples
    return summary


def _collect_recommendations(context: TrackerContext) -> list[dict[str, Any]]:
    engine = context.recommendation_engine
    if engine is None:
        return []
    return serialize_many(engine.generate_recommendations())


def _append_run_snapshot(
    context: TrackerContext,
    status: PipelineRunStatus,
    *,
    summary: dict[str, Any] | None = None,
) -> None:
    record = PipelineRunRecord(
        run_id=context.run_id,
        created_at=context.created_at,
        status=status,
        enabled_steps=context.enabled_steps,
        artifact_dir=context.writer.run_directory,
        initiator=context.initiator,
        scenario_config_path=context.scenario_config,
        completed_at=datetime.now(UTC) if status != PipelineRunStatus.RUNNING else None,
        summary=summary or {},
        steps=context.tracker.clone_entries(),
    )
    context.writer.append_run_record(record)


def _finalize_tracker(
    context: TrackerContext | None,
    status: PipelineRunStatus,
    *,
    summary: dict[str, Any] | None = None,
) -> None:
    if context is None:
        return
    context.tracker.disable_failure_guard()
    _stop_tracker_telemetry(context)
    summary_payload = _build_tracker_summary(context, summary)
    recommendations = _collect_recommendations(context)
    if recommendations:
        context.writer.append_recommendations(recommendations)
        summary_payload.setdefault("recommendation_count", len(recommendations))
    _append_run_snapshot(context, status, summary=summary_payload)


def _run_tracked_step(
    tracker_ctx: TrackerContext | None,
    tracked_steps: set[str],
    step_id: str,
    func: Callable[[], Any],
) -> Any:
    is_tracked = tracker_ctx is not None and step_id in tracked_steps
    if is_tracked:
        assert tracker_ctx is not None  # type narrowing for static analysis
        tracker_ctx.tracker.start_step(step_id)
    try:
        result = func()
    except Exception as exc:  # pragma: no cover - pipeline invoked externally
        if is_tracked:
            assert tracker_ctx is not None
            tracker_ctx.tracker.fail_step(step_id, reason=str(exc))
        raise
    if is_tracked:
        assert tracker_ctx is not None
        if isinstance(result, int) and result != 0:
            tracker_ctx.tracker.fail_step(step_id, reason=f"exit code {result}")
        else:
            tracker_ctx.tracker.complete_step(step_id)
    return result


def _run_tracker_smoke(
    context: TrackerContext | None, step_definitions: list[PipelineStepDefinition]
) -> None:
    if context is None:
        raise RuntimeError("Tracker smoke execution requires --enable-tracker")
    for definition in step_definitions:
        context.tracker.start_step(definition.step_id)
        time.sleep(0.05)
        context.tracker.complete_step(definition.step_id)
    _finalize_tracker(context, PipelineRunStatus.COMPLETED, summary={"mode": "tracker-smoke"})


def main():  # noqa: C901 - Sequential workflow orchestration; complexity is intentional
    """Run complete imitation learning pipeline by calling training scripts."""
    _ensure_run_root()
    parser = argparse.ArgumentParser(description="Imitation Learning Pipeline - End-to-End Example")
    parser.add_argument(
        "--skip-expert",
        action="store_true",
        help="Skip expert training (use existing policy)",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="ppo_expert_demo",
        help="Expert policy ID to use/create",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="trajectories_demo",
        help="Trajectory dataset ID to create",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Quick demo mode (reduced timesteps/episodes)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"),
        help="Console log level (use DEBUG to see resolved config dumps)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Preferred simulation backend (auto-selects fastest if omitted)",
    )
    parser.add_argument(
        "--enable-tracker",
        action="store_true",
        help="Enable progress tracking and telemetry output",
    )
    parser.add_argument(
        "--tracker-output",
        type=str,
        help="Optional run identifier or output path for tracker artifacts",
    )
    parser.add_argument(
        "--tracker-telemetry-interval",
        type=float,
        default=1.0,
        help="Telemetry sampling interval in seconds when the tracker is enabled",
    )
    parser.add_argument(
        "--tracker-smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    tracker_smoke_env = _env_flag("ROBOT_SF_TRACKER_SMOKE")
    args.tracker_smoke = args.tracker_smoke or tracker_smoke_env
    tracker_requested = (
        args.enable_tracker or _env_flag("ROBOT_SF_ENABLE_PROGRESS_TRACKER") or args.tracker_smoke
    )
    tracker_output = args.tracker_output or os.environ.get("ROBOT_SF_TRACKER_OUTPUT")

    try:
        chosen_backend = select_best_backend(args.backend)
    except RuntimeError as err:
        logger.error(f"Unable to choose backend: {err}")
        return 1

    inherited_env = os.environ.copy()
    inherited_env["ROBOT_SF_BACKEND"] = chosen_backend

    dataset_id = args.dataset_id
    baseline_run_id: str | None = None
    pretrained_run_id: str | None = None
    comparison_group_id: str | None = None
    comparison_summary: dict[str, Any] = {}
    comparison_candidates = (
        Path("scripts/tools/compare_training_runs.py"),
        Path("scripts/training/compare_training_runs.py"),
    )
    for candidate in comparison_candidates:
        if candidate.exists():
            comparison_script = candidate
            break
    else:  # pragma: no cover - only hit when neither path exists locally
        comparison_script = comparison_candidates[0]

    logger.info("=" * 70)
    logger.info("IMITATION LEARNING PIPELINE - END-TO-END EXAMPLE")
    logger.info("=" * 70)

    # Configuration file path (only expert_ppo.yaml exists; others use CLI args)
    expert_config = Path("configs/training/ppo_imitation/expert_ppo.yaml")

    # Verify expert config exists
    if not expert_config.exists():
        logger.error(f"Configuration file not found: {expert_config}")
        logger.info("Please ensure you're running from repository root")
        return 1

    configured_policy_id = _load_policy_id_from_config(expert_config)
    expert_config_path = expert_config

    if args.skip_expert:
        expert_policy_id = args.policy_id
    else:
        expert_policy_id = configured_policy_id
        if args.policy_id and args.policy_id != configured_policy_id:
            expert_config_path = _override_expert_config_policy_id(expert_config, args.policy_id)
            expert_policy_id = args.policy_id
            logger.info(
                "Applying policy_id override (--policy-id): using {} with policy_id={}",
                expert_config_path,
                expert_policy_id,
            )

    step_definitions = build_imitation_pipeline_steps(
        skip_expert=args.skip_expert,
        include_comparison=comparison_script.exists(),
    )
    tracked_step_ids = {step.step_id for step in step_definitions}
    if "compare_runs" not in tracked_step_ids and not comparison_script.exists():
        logger.warning("Comparison script not found: {}", comparison_script)
        logger.info("Skipping comparison step")
    tracker_context: TrackerContext | None = None
    if tracker_requested:
        tracker_context = _create_tracker_context(
            step_definitions,
            tracker_output=tracker_output,
            initiator=" ".join(sys.argv),
            scenario_config=expert_config_path,
            telemetry_interval=args.tracker_telemetry_interval,
        )
        logger.info(
            "Run tracker enabled (run_id={} dir={})",
            tracker_context.run_id,
            tracker_context.writer.run_directory,
        )

    if args.tracker_smoke:
        _run_tracker_smoke(tracker_context, step_definitions)
        return 0

    def fail_and_return(step_id: str, exit_code: int) -> int:
        _finalize_tracker(
            tracker_context,
            PipelineRunStatus.FAILED,
            summary={"failed_step": step_id, "exit_code": exit_code},
        )
        return exit_code

    bc_policy_id = f"bc_{expert_policy_id}"
    finetuned_policy_id = f"finetuned_{expert_policy_id}"
    pretrained_run_id = f"ppo_finetune_{finetuned_policy_id}"

    logger.info(f"Expert policy ID: {expert_policy_id}")
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Demo mode: {args.demo_mode}")
    logger.info(f"Simulation backend: {chosen_backend}")
    logger.info("")

    try:
        # Step 1: Train expert policy (or use existing)
        if args.skip_expert:
            logger.info("=" * 70)
            logger.info("STEP 1: Using Existing Expert Policy")
            logger.info("=" * 70)
            logger.info(f"Policy ID: {expert_policy_id}")

            from robot_sf import common

            policy_path = common.get_expert_policy_dir() / f"{expert_policy_id}.zip"
            if not policy_path.exists():
                logger.error(f"Expert policy not found: {policy_path}")
                logger.info("Run without --skip-expert to train a new policy")
                return fail_and_return("train_expert", 1)
            logger.info(f"Found policy: {policy_path}")
            baseline_run_id = _discover_latest_training_run_id(f"{expert_policy_id}_")
        elif "train_expert" in tracked_step_ids:
            logger.info("=" * 70)
            logger.info("STEP 1: Training Expert PPO Policy")
            logger.info("=" * 70)

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/training/train_expert_ppo.py",
                "--config",
                str(expert_config_path),
            ]
            if args.demo_mode:
                cmd.append("--dry-run")

            exit_code = _run_tracked_step(
                tracker_context,
                tracked_step_ids,
                "train_expert",
                lambda: _run_command(cmd, "Expert training", env=inherited_env),
            )
            if exit_code != 0:
                return fail_and_return("train_expert", exit_code)
            baseline_run_id = _discover_latest_training_run_id(f"{expert_policy_id}_")
        else:
            baseline_run_id = _discover_latest_training_run_id(f"{expert_policy_id}_")

        if not baseline_run_id:
            logger.warning(
                "Failed to locate training run manifest for policy_id={}; comparison output may be unavailable.",
                expert_policy_id,
            )

        # Step 2: Collect trajectories
        if "collect_trajectories" in tracked_step_ids:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 2: Collecting Expert Trajectories")
            logger.info("=" * 70)

            num_episodes = 20 if args.demo_mode else 100
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/training/collect_expert_trajectories.py",
                "--dataset-id",
                dataset_id,
                "--policy-id",
                expert_policy_id,
                "--episodes",
                str(num_episodes),
            ]

            exit_code = _run_tracked_step(
                tracker_context,
                tracked_step_ids,
                "collect_trajectories",
                lambda: _run_command(cmd, "Trajectory collection", env=inherited_env),
            )
            if exit_code != 0:
                return fail_and_return("collect_trajectories", exit_code)

        # Step 3: BC pre-training
        if "bc_pretrain" in tracked_step_ids:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 3: Behavioral Cloning Pre-training")
            logger.info("=" * 70)

            bc_config_path = _prepare_bc_config(dataset_id, bc_policy_id, args.demo_mode)
            logger.info("Using BC config: {}", bc_config_path)
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/training/pretrain_from_expert.py",
                "--config",
                str(bc_config_path),
            ]

            exit_code = _run_tracked_step(
                tracker_context,
                tracked_step_ids,
                "bc_pretrain",
                lambda: _run_command(cmd, "BC pre-training", env=inherited_env),
            )
            if exit_code != 0:
                return fail_and_return("bc_pretrain", exit_code)

        # Step 4: PPO fine-tuning
        if "ppo_finetune" in tracked_step_ids:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 4: PPO Fine-tuning")
            logger.info("=" * 70)

            timesteps = 30000 if args.demo_mode else 200000
            ppo_config_path = _prepare_ppo_config(
                bc_policy_id,
                finetuned_policy_id,
                timesteps,
                args.demo_mode,
            )
            logger.info("Using PPO fine-tune config: {}", ppo_config_path)
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/training/train_ppo_with_pretrained_policy.py",
                "--config",
                str(ppo_config_path),
            ]
            if args.demo_mode:
                cmd.append("--dry-run")

            exit_code = _run_tracked_step(
                tracker_context,
                tracked_step_ids,
                "ppo_finetune",
                lambda: _run_command(cmd, "PPO fine-tuning", env=inherited_env),
            )
            if exit_code != 0:
                return fail_and_return("ppo_finetune", exit_code)

        # Step 5: Generate comparison (optional - script may not exist yet)
        if "compare_runs" in tracked_step_ids:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 5: Performance Comparison")
            logger.info("=" * 70)

            if comparison_script.exists():
                if not baseline_run_id:
                    logger.warning(
                        "Baseline training run id unavailable; skipping automatic comparison. Run scripts/tools/compare_training_runs.py manually."
                    )
                else:
                    comparison_group_id = (
                        f"{expert_policy_id}_vs_{finetuned_policy_id}_"
                        f"{datetime.now(UTC).strftime('%Y%m%d%H%M')}"
                    )
                    effective_pretrained_run_id = (
                        pretrained_run_id or f"ppo_finetune_{finetuned_policy_id}"
                    )
                    cmd = [
                        "uv",
                        "run",
                        "python",
                        str(comparison_script),
                        "--group",
                        comparison_group_id,
                        "--baseline",
                        baseline_run_id,
                        "--pretrained",
                        effective_pretrained_run_id,
                    ]
                    exit_code = _run_tracked_step(
                        tracker_context,
                        tracked_step_ids,
                        "compare_runs",
                        lambda: _run_command(cmd, "Comparison generation", env=inherited_env),
                    )
                    if exit_code != 0:
                        return fail_and_return("compare_runs", exit_code)
                    comparison_summary = _build_comparison_summary(
                        comparison_group_id, baseline_run_id, effective_pretrained_run_id
                    )
            else:
                logger.warning(f"Comparison script not found: {comparison_script}")
                logger.info("Skipping comparison step")

        # Success summary
        logger.info("")
        logger.info("=" * 70)
        logger.success("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Artifacts created:")
        logger.info(f"  - Expert policy: {expert_policy_id}")
        logger.info(f"  - Trajectory dataset: {dataset_id}")
        logger.info(f"  - BC pre-trained policy: {bc_policy_id}")
        logger.info(f"  - Fine-tuned policy: {finetuned_policy_id}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  - View artifacts: output/benchmarks/expert_policies/")
        logger.info("  - View trajectories: output/benchmarks/expert_trajectories/")
        logger.info("  - Full docs: docs/imitation_learning_pipeline.md")
        logger.info("  - Quickstart: specs/001-ppo-imitation-pretrain/quickstart.md")

        tracker_summary = {
            "completed_steps": len(tracked_step_ids),
            "demo_mode": args.demo_mode,
            "skip_expert": args.skip_expert,
        }
        tracker_summary.update(comparison_summary)
        _finalize_tracker(
            tracker_context,
            PipelineRunStatus.COMPLETED,
            summary=tracker_summary,
        )

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        _finalize_tracker(
            tracker_context,
            PipelineRunStatus.FAILED,
            summary={"error": e.__class__.__name__},
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
