"""Evaluate PPO or Dreamer checkpoints with a standardized metrics pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.policy_checkpoint_evaluator import (
    build_eval_timeline_entry,
    evaluate_policy_episodes,
    write_episode_records,
    write_summary,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from robot_sf.training.scenario_sampling import ScenarioSampler
from robot_sf.training.snqi_utils import resolve_training_snqi_context


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "dreamer"), required=True)
    parser.add_argument("--config", type=Path, required=True, help="Training config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where summary.json and episodes.jsonl will be written.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for evaluation episode count.",
    )
    parser.add_argument(
        "--eval-step",
        type=int,
        default=None,
        help="Optional evaluation step/iteration label stored in output artifacts.",
    )
    return parser


def _evaluate_ppo_checkpoint(
    *,
    config_path: Path,
    checkpoint_path: Path,
    episodes_override: int | None,
    eval_step: int | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    from stable_baselines3 import PPO

    from scripts.training.train_ppo import _apply_env_overrides, load_expert_training_config

    config = load_expert_training_config(config_path)
    episodes = int(episodes_override or config.evaluation.evaluation_episodes)
    scenario_definitions = load_scenarios(
        config.scenario_config, base_dir=config.scenario_config.parent
    )
    snqi_context = resolve_training_snqi_context(
        weights_path=config.snqi_weights_path,
        baseline_path=config.snqi_baseline_path,
    )
    model = PPO.load(str(checkpoint_path))

    if config.scenario_id:
        sampler = ScenarioSampler(
            scenario_definitions,
            include_scenarios=(config.scenario_id,),
            seed=None if config.randomize_seeds else 0,
            strategy="cycle",
        )
    elif config.evaluation.hold_out_scenarios:
        sampler = ScenarioSampler(
            scenario_definitions,
            include_scenarios=tuple(config.evaluation.hold_out_scenarios),
            seed=None if config.randomize_seeds else 0,
            strategy="random" if config.randomize_seeds else "cycle",
        )
    else:
        sampler = ScenarioSampler(
            scenario_definitions,
            seed=None if config.randomize_seeds else 0,
            strategy="random" if config.randomize_seeds else "cycle",
        )

    def _make_env(_episode_idx: int, seed: int | None):
        scenario, scenario_name = sampler.sample()
        env_config = build_robot_config_from_scenario(
            scenario,
            scenario_path=config.scenario_config,
        )
        _apply_env_overrides(env_config, config.env_overrides)
        env = make_robot_env(
            config=env_config,
            seed=seed,
            suite_name="shared_checkpoint_eval",
            scenario_name=scenario_name,
            algorithm_name=config.policy_id,
            **config.env_factory_kwargs,
        )
        return env, scenario_name

    result = evaluate_policy_episodes(
        episodes=episodes,
        make_env=_make_env,
        action_fn=lambda obs: model.predict(obs, deterministic=True)[0],
        eval_step=eval_step,
        base_seed=int(config.seeds[0]) if config.seeds else 0,
        randomize_seeds=bool(config.randomize_seeds),
        snqi_context=snqi_context,
    )
    return result.summary, result.episode_records


def _evaluate_dreamer_checkpoint(
    *,
    config_path: Path,
    checkpoint_path: Path,
    episodes_override: int | None,
    eval_step: int | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    import scripts.training.train_dreamerv3_rllib as dreamer_train

    run_config = dreamer_train.load_run_config(config_path)
    episodes = int(episodes_override or run_config.evaluation.evaluation_episodes or 30)
    snqi_context = resolve_training_snqi_context(
        weights_path=run_config.evaluation.snqi_weights_path,
        baseline_path=run_config.evaluation.snqi_baseline_path,
    )
    shared_capacity = dreamer_train._detect_shared_capacity(run_config)
    should_shutdown = False
    if not ray.is_initialized():
        ray.init(**dreamer_train._build_ray_init_kwargs(run_config, capacity=shared_capacity))
        should_shutdown = True
    algo = Algorithm.from_checkpoint(str(checkpoint_path))
    try:
        env_factory = dreamer_train._create_dreamer_eval_env_factory(run_config)
        result = evaluate_policy_episodes(
            episodes=episodes,
            make_env=env_factory,
            action_fn=lambda obs: algo.compute_single_action(obs, explore=False),
            eval_step=eval_step,
            base_seed=int(run_config.experiment.seed),
            randomize_seeds=bool(run_config.env.config_overrides.get("randomize_seeds", False)),
            snqi_context=snqi_context,
        )
    finally:
        stop = getattr(algo, "stop", None)
        if callable(stop):
            stop()
        if should_shutdown and ray.is_initialized():
            ray.shutdown()
    return result.summary, result.episode_records


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (checkpoint_path.parent / "shared_eval").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.algorithm == "ppo":
        summary, episode_records = _evaluate_ppo_checkpoint(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            episodes_override=args.episodes,
            eval_step=args.eval_step,
        )
    else:
        summary, episode_records = _evaluate_dreamer_checkpoint(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            episodes_override=args.episodes,
            eval_step=args.eval_step,
        )

    timeline_row = build_eval_timeline_entry(summary, eval_step=int(args.eval_step or 0))
    payload: dict[str, Any] = {
        "algorithm": args.algorithm,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "timeline_row": timeline_row,
        **summary,
    }
    summary_path = output_dir / "summary.json"
    episodes_path = output_dir / "episodes.jsonl"
    write_summary(summary_path, payload)
    write_episode_records(episodes_path, episode_records)
    logger.info("shared evaluation written summary={} episodes={}", summary_path, episodes_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
