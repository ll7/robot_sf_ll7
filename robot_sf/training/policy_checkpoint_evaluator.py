"""Shared checkpoint evaluation helpers for PPO and Dreamer policies."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.benchmark.termination_reason import (
    build_outcome_payload,
    resolve_termination_reason,
    status_from_termination_reason,
)
from robot_sf.training.snqi_utils import TrainingSNQIContext, default_training_snqi_context

MetricSamples = dict[str, list[float]]

EVAL_METRIC_KEYS = (
    "success_rate",
    "collision_rate",
    "path_efficiency",
    "comfort_exposure",
    "snqi",
    "eval_episode_return",
    "eval_avg_step_reward",
)


@dataclass(slots=True)
class PolicyEvaluationResult:
    """Standardized evaluation payload for policy checkpoints."""

    summary: dict[str, object]
    episode_records: list[dict[str, object]]


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean for a finite list of values."""
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _estimate_path_efficiency(meta: Mapping[str, object]) -> float:
    """Estimate path efficiency from episode step count relative to horizon."""
    steps_taken = float(meta.get("step_of_episode", 0) or 0)
    max_steps = float(meta.get("max_sim_steps", steps_taken if steps_taken > 0 else 1))
    if max_steps <= 0:
        return 0.0
    ratio = 1.0 - min(1.0, steps_taken / max_steps)
    return float(max(0.0, ratio))


def _collision_flag(meta: Mapping[str, object]) -> bool:
    """Return whether any collision flag is present in the episode metadata."""
    return bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_robot_collision")
        or meta.get("is_obstacle_collision")
    )


def _gather_episode_metrics(
    info: Mapping[str, object],
    *,
    steps_taken: int,
    max_steps: int,
    episode_return: float,
    avg_step_reward: float,
    snqi_context: TrainingSNQIContext,
) -> dict[str, float]:
    """Build standardized evaluation metrics for one completed episode."""
    raw_meta = info.get("meta", {}) if isinstance(info, Mapping) else {}
    meta = raw_meta if isinstance(raw_meta, Mapping) else {}
    meta_for_path = dict(meta)
    meta_for_path.setdefault("step_of_episode", steps_taken)
    meta_for_path.setdefault("max_sim_steps", max_steps)
    success = 1.0 if bool(meta.get("is_route_complete")) else 0.0
    collision = 1.0 if _collision_flag(meta) else 0.0
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
    from robot_sf.training.snqi_utils import compute_training_snqi

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


def summarize_eval_metrics(metrics: MetricSamples) -> dict[str, float]:
    """Reduce per-episode metric samples to scalar means."""
    return {key: _mean(values) for key, values in metrics.items()}


def build_eval_timeline_entry(
    summary: Mapping[str, object],
    *,
    eval_step: int,
) -> dict[str, object]:
    """Flatten an evaluation summary into one timeline row."""
    metric_means = summary.get("metric_means", {})
    if not isinstance(metric_means, Mapping):
        metric_means = {}
    termination_rates = summary.get("termination_reason_rates", {})
    if not isinstance(termination_rates, Mapping):
        termination_rates = {}

    row: dict[str, object] = {
        "eval_step": int(eval_step),
        "episodes": int(summary.get("episodes", 0) or 0),
        "episode_len_mean": float(summary.get("episode_len_mean", 0.0) or 0.0),
        "eval_wall_sec": float(summary.get("eval_wall_sec", 0.0) or 0.0),
    }
    for key in EVAL_METRIC_KEYS:
        row[key] = float(metric_means.get(key, 0.0) or 0.0)
    for reason, rate in sorted(termination_rates.items()):
        row[f"termination_reason_{reason}_rate"] = float(rate or 0.0)
    return row


def write_episode_records(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Write episode-level evaluation records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True))
            handle.write("\n")


def write_summary(path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON evaluation summary with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def coerce_policy_action(action_output: object) -> object:
    """Normalize framework-specific action outputs to the raw env action."""
    if isinstance(action_output, tuple) and len(action_output) in {2, 3}:
        return action_output[0]
    return action_output


def evaluate_policy_episodes(
    *,
    episodes: int,
    make_env: Callable[[int, int | None], tuple[Any, str | None]],
    action_fn: Callable[[object], object],
    eval_step: int | None = None,
    base_seed: int = 0,
    randomize_seeds: bool = False,
    snqi_context: TrainingSNQIContext | None = None,
) -> PolicyEvaluationResult:
    """Run standardized evaluation episodes against a policy action function."""
    snqi_context = snqi_context or default_training_snqi_context()
    metrics: MetricSamples = {key: [] for key in EVAL_METRIC_KEYS}
    termination_counts: Counter[str] = Counter()
    episode_lengths: list[float] = []
    episode_records: list[dict[str, object]] = []
    eval_started = time.perf_counter()

    for episode_idx in range(max(1, int(episodes))):
        seed = None if randomize_seeds else int(base_seed + episode_idx)
        env, scenario_name = make_env(episode_idx, seed)
        terminated = False
        truncated = False
        info: Mapping[str, object] = {}
        steps = 0
        episode_return = 0.0
        try:
            obs, _ = env.reset()
            done = False
            max_steps = int(getattr(env.state, "max_sim_steps", 1))  # type: ignore[attr-defined]

            while not done and steps < max_steps:
                action = coerce_policy_action(action_fn(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                done = bool(terminated or truncated)
                steps += 1
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        raw_meta = info.get("meta", {}) if isinstance(info, Mapping) else {}
        meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        success = bool(meta.get("is_route_complete"))
        collision = _collision_flag(meta)
        timeout_event = bool(meta.get("is_timesteps_exceeded")) or (
            steps >= max_steps and not bool(terminated or truncated)
        )
        termination_reason = resolve_termination_reason(
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=success,
            collision=collision,
            reached_max_steps=timeout_event,
            had_error=False,
        )
        termination_counts[termination_reason] += 1
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
        episode_lengths.append(float(steps))
        outcome = build_outcome_payload(
            route_complete=success,
            collision=collision,
            timeout=timeout_event,
        )
        episode_records.append(
            {
                "episode": episode_idx,
                "seed": seed,
                "steps": int(steps),
                "scenario_id": scenario_name,
                "eval_step": eval_step,
                "termination_reason": termination_reason,
                "status": status_from_termination_reason(termination_reason),
                "outcome": outcome,
                "metrics": metric_row,
            }
        )

    eval_wall_sec = max(0.0, time.perf_counter() - eval_started)
    summary = {
        "episodes": len(episode_records),
        "eval_step": eval_step,
        "metric_means": summarize_eval_metrics(metrics),
        "termination_reason_counts": dict(sorted(termination_counts.items())),
        "termination_reason_rates": {
            reason: float(count / max(1, len(episode_records)))
            for reason, count in sorted(termination_counts.items())
        },
        "episode_len_mean": _mean(episode_lengths),
        "eval_wall_sec": float(eval_wall_sec),
    }
    logger.info(
        "evaluation_complete eval_step={} episodes={} success_rate={:.3f} collision_rate={:.3f} timeout_rate={:.3f}",
        eval_step,
        len(episode_records),
        float((summary["metric_means"] or {}).get("success_rate", 0.0)),
        float((summary["metric_means"] or {}).get("collision_rate", 0.0)),
        float((summary["termination_reason_rates"] or {}).get("max_steps", 0.0))
        + float((summary["termination_reason_rates"] or {}).get("truncated", 0.0)),
    )
    return PolicyEvaluationResult(summary=summary, episode_records=episode_records)
