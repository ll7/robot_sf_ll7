#!/usr/bin/env python3
"""Run the issue #4013 paired diagnostic model-based planning comparison.

This is the end-to-end RUN that was deferred across the issue #4013 scaffolding PRs
(#4160 smoke lane, #4474 comparator preflight, #4587 report contract, #4629 trainer,
#4644 checkpoint-backed action selection). It composes the existing canonical owners
rather than re-deriving them:

1. Train the short-horizon predictor checkpoint (CPU, git-ignored ``output/``) via
   ``scripts/training/train_learned_short_horizon_predictor_issue_4013.py`` when the
   checkpoint is missing, so the model-based arm loads real learned weights
   (``evidence_tier=checkpoint_loaded``) with no fallback.
2. Run the three benchmark arms through ``robot_sf.benchmark.map_runner.run_map_batch``
   and write per-role episode JSONL:
     - ``learned_prediction_mpc``  (learned short-horizon prediction + MPC)
     - ``cv_prediction_mpc``       (constant-velocity prediction + MPC)
     - ``model_free_baseline``     (deterministic goal-seeking; no model, no prediction)
3. Build the diagnostic comparison report via
   ``scripts/analysis/compare_model_based_planning_issue_4013.build_report_from_config``.

Claim boundary: diagnostic-only, single scenario / single seed smoke. This proves the
model-based selection path runs end-to-end and can be compared, paired by scenario/seed,
against a constant-velocity predictor and a model-free baseline. It is NOT benchmark,
navigation-quality, or paper-facing evidence; it is not a large generative world model.
The paired count is intentionally small; scaling seeds/scenarios is a separate campaign.

Usage (from repo root, in the project venv)::

    uv run python scripts/benchmark/run_issue_4013_model_based_comparison.py

Exit code is ``0`` when the comparison report reaches ``diagnostic_ready`` with no
blockers, otherwise non-zero.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.planner.learned_short_horizon_trainer import (
    ShortHorizonTrainerConfig,
    train_short_horizon_predictor,
)

# Canonical benchmark episode schema, as every other map-runner entry point uses.
DEFAULT_SCHEMA = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
DEFAULT_ANALYSIS_CONFIG = Path("configs/analysis/issue_4013_model_based_planning_comparison.yaml")
DEFAULT_TRAINER_CONFIG = Path(
    "configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml"
)
# The checkpoint algo config that the model-based arm loads; its ``checkpoint_path``
# is the artifact the trainer must produce before the model-based arm runs.
CHECKPOINT_ALGO_CONFIG = Path("configs/algos/learned_prediction_mpc_issue_4013_checkpoint.yaml")

# Map each required comparison role to the benchmark run config that produces it.
ROLE_TO_BENCHMARK_CONFIG: dict[str, Path] = {
    "learned_prediction_mpc": Path(
        "configs/benchmarks/issue_4013_model_based_checkpoint_smoke.yaml"
    ),
    "cv_prediction_mpc": Path("configs/benchmarks/issue_4013_model_free_smoke.yaml"),
    "model_free_baseline": Path("configs/benchmarks/issue_4013_model_free_baseline_smoke.yaml"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--trainer-config", type=Path, default=DEFAULT_TRAINER_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Fail closed if the checkpoint is missing instead of training it.",
    )
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return payload


def _fixed_list_seeds(config_payload: dict[str, Any], *, path: Path) -> list[int]:
    seed_policy = config_payload.get("seed_policy") or {}
    if not isinstance(seed_policy, dict) or seed_policy.get("mode") != "fixed-list":
        raise ValueError(f"{path} must declare seed_policy.mode=fixed-list.")
    seeds = seed_policy.get("seeds") or []
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path} must declare a non-empty fixed seed list.")
    return [int(seed) for seed in seeds]


def _scenario_payload_with_config_seeds(
    config_payload: dict[str, Any], *, path: Path
) -> list[dict[str, Any]]:
    scenario_matrix = Path(str(config_payload["scenario_matrix"]))
    matrix_payload = _load_yaml(scenario_matrix)
    scenarios = matrix_payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{scenario_matrix} must declare a non-empty scenarios list.")
    seeds = _fixed_list_seeds(config_payload, path=path)
    resolved: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError(f"{scenario_matrix} contains a non-mapping scenario entry.")
        scenario_copy = dict(scenario)
        scenario_copy["seeds"] = list(seeds)
        resolved.append(scenario_copy)
    return resolved


def _planner_spec(config_payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    planners = config_payload.get("planners")
    if not isinstance(planners, list) or len(planners) != 1 or not isinstance(planners[0], dict):
        raise ValueError(f"{path} must declare exactly one planner mapping.")
    return planners[0]


def _checkpoint_path_from_algo_config(algo_config: Path) -> Path:
    payload = _load_yaml(algo_config)
    checkpoint = payload.get("checkpoint_path")
    if not checkpoint:
        raise ValueError(f"{algo_config} must declare checkpoint_path for the model-based arm.")
    return Path(str(checkpoint))


def ensure_checkpoint(*, trainer_config: Path, skip_train: bool) -> dict[str, Any]:
    """Ensure the model-based checkpoint exists, training it (CPU) when missing.

    Returns:
        dict: A note describing whether the checkpoint was reused or freshly trained.
    """
    checkpoint = _checkpoint_path_from_algo_config(CHECKPOINT_ALGO_CONFIG)
    if checkpoint.exists():
        return {"checkpoint": str(checkpoint), "action": "reused_existing"}
    if skip_train:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} missing and --skip-train set. Run the trainer first: "
            "scripts/training/train_learned_short_horizon_predictor_issue_4013.py"
        )
    payload = _load_yaml(trainer_config)
    result = train_short_horizon_predictor(ShortHorizonTrainerConfig(**payload))
    trained = Path(result.checkpoint_path)
    if not trained.exists():
        raise FileNotFoundError(f"Trainer did not produce a checkpoint at {trained}.")
    return {
        "checkpoint": str(trained),
        "action": "trained",
        "final_loss": result.final_loss,
        "loss_reduction": result.loss_reduction,
    }


def _resolve_episode_path(config_path: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("analysis config run entry is missing episodes_jsonl.")
    path = Path(str(value))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def run_arm(
    *,
    role: str,
    benchmark_config: Path,
    episodes_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    """Run a single benchmark arm and write its episode JSONL for the report role."""
    config_payload = _load_yaml(benchmark_config)
    planner = _planner_spec(config_payload, path=benchmark_config)
    scenarios = _scenario_payload_with_config_seeds(config_payload, path=benchmark_config)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.unlink(missing_ok=True)
    algo_config = planner.get("algo_config")
    summary = run_map_batch(
        scenarios,
        episodes_path,
        schema_path,
        scenario_path=config_payload["scenario_matrix"],
        horizon=int(config_payload["horizon"]),
        dt=float(config_payload["dt"]),
        record_forces=bool(config_payload.get("record_forces", True)),
        algo=str(planner["algo"]),
        algo_config_path=str(algo_config) if algo_config else None,
        benchmark_profile=str(planner.get("benchmark_profile", "diagnostic-only")),
        workers=int(config_payload.get("workers", 1)),
        resume=False,
    )
    rows = sum(1 for _ in episodes_path.open(encoding="utf-8")) if episodes_path.exists() else 0
    return {
        "role": role,
        "benchmark_config": str(benchmark_config),
        "episodes_path": str(episodes_path),
        "written_rows": rows,
        "failed_jobs": int(summary.get("failed_jobs", 0)),
        "failures": summary.get("failures", []),
    }


def _load_report_builder() -> Any:
    """Load ``build_report_from_config`` from the sibling analysis script.

    The analysis module lives under ``scripts/analysis/`` (not an importable package),
    so it is loaded by file path to reuse its canonical report contract rather than
    re-deriving the comparison logic here.
    """
    module_path = Path("scripts/analysis/compare_model_based_planning_issue_4013.py")
    spec = importlib.util.spec_from_file_location(
        "compare_model_based_planning_issue_4013", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load report builder from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_report_from_config


def main(argv: list[str] | None = None) -> int:
    """Train (if needed), run the three arms, and build the comparison report."""
    args = parse_args(argv)
    analysis_config = _load_yaml(args.analysis_config)
    runs = analysis_config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{args.analysis_config} must declare a non-empty runs list.")

    checkpoint_note = ensure_checkpoint(
        trainer_config=args.trainer_config, skip_train=args.skip_train
    )

    arm_results: list[dict[str, Any]] = []
    for run in runs:
        role = str(run.get("role", ""))
        benchmark_config = ROLE_TO_BENCHMARK_CONFIG.get(role)
        if benchmark_config is None:
            raise ValueError(
                f"No benchmark config mapped for analysis role {role!r}. "
                f"Known roles: {sorted(ROLE_TO_BENCHMARK_CONFIG)}"
            )
        episodes_path = _resolve_episode_path(args.analysis_config, run.get("episodes_jsonl"))
        arm_results.append(
            run_arm(
                role=role,
                benchmark_config=benchmark_config,
                episodes_path=episodes_path,
                schema_path=args.schema,
            )
        )

    build_report_from_config = _load_report_builder()
    report = build_report_from_config(args.analysis_config)

    print(
        json.dumps(
            {
                "checkpoint": checkpoint_note,
                "arms": arm_results,
                "report_status": report["status"],
                "paired_seed_count": report["paired_seed_count"],
                "blockers": report["blockers"],
                "report_json": str(
                    _resolve_episode_path(args.analysis_config, analysis_config.get("output_json"))
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "diagnostic_ready" and not report["blockers"] else 1


if __name__ == "__main__":
    sys.exit(main())
