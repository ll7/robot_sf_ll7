"""Collect RLlib SingleAgentEpisode datasets for Dreamer world-model pretraining."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import msgpack
    import msgpack_numpy as mnp
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise RuntimeError(
        "Ray RLlib plus msgpack_numpy are required for Dreamer episode export."
    ) from exc

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover - optional runtime dependency
    PPO = None  # type: ignore[assignment]

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.models import resolve_model_path
from scripts.training.train_dreamerv3_rllib import _make_env_creator, load_run_config


def _default_output_dir(dataset_id: str) -> Path:
    """Return the default dataset directory under the canonical artifact root."""
    target = get_artifact_category_path("benchmarks") / "dreamer_world_model" / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_teacher(
    *,
    teacher_mode: str,
    teacher_model_id: str | None,
    teacher_checkpoint: Path | None,
    deterministic: bool,
) -> tuple[Any | None, dict[str, Any]]:
    """Construct the teacher policy and record metadata for the dataset manifest."""
    metadata = {
        "teacher_mode": teacher_mode,
        "teacher_model_id": teacher_model_id,
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint is not None else None,
        "deterministic": bool(deterministic),
    }
    if teacher_mode in {"random", "zero"}:
        return None, metadata
    if PPO is None:
        raise RuntimeError("stable_baselines3 is required for --teacher-mode ppo.")

    resolved_checkpoint = (
        resolve_model_path(teacher_model_id)
        if teacher_model_id is not None
        else teacher_checkpoint.resolve()
        if teacher_checkpoint is not None
        else None
    )
    if resolved_checkpoint is None:
        raise ValueError("A PPO teacher requires --teacher-model-id or --teacher-checkpoint.")
    metadata["teacher_checkpoint"] = str(resolved_checkpoint)
    logger.info("Loading PPO teacher from {}", resolved_checkpoint)
    return PPO.load(str(resolved_checkpoint)), metadata


def _predict_action(
    teacher: Any | None,
    obs: Any,
    action_space: Any,
    *,
    teacher_mode: str,
    deterministic: bool,
) -> np.ndarray:
    """Produce one action from the requested teacher mode."""
    if teacher_mode == "zero":
        return np.zeros(action_space.shape, dtype=np.float32)
    if teacher_mode == "random":
        return np.asarray(action_space.sample(), dtype=np.float32)
    action, _ = teacher.predict(obs, deterministic=deterministic)
    return np.asarray(action, dtype=np.float32)


def _serialize_episode(episode: SingleAgentEpisode) -> bytes:
    """Convert one finalized RLlib episode into msgpack bytes."""
    episode.to_numpy()
    return msgpack.packb(episode.get_state(), default=mnp.encode, use_bin_type=True)


def _quality_gate(
    *,
    lengths: list[int],
    returns: list[float],
    action_vectors: list[np.ndarray],
    minimum_episodes: int,
    minimum_mean_length: float,
    minimum_action_std: float,
) -> dict[str, Any]:
    """Compute simple dataset health checks and fail fast on degenerate exports."""
    if len(lengths) < minimum_episodes:
        raise RuntimeError(
            f"Collected only {len(lengths)} episodes, expected at least {minimum_episodes}."
        )
    mean_length = float(np.mean(lengths)) if lengths else 0.0
    if mean_length < minimum_mean_length:
        raise RuntimeError(
            f"Mean episode length {mean_length:.2f} is below required {minimum_mean_length:.2f}."
        )
    action_std = 0.0
    if action_vectors:
        flat_actions = np.concatenate(action_vectors, axis=0)
        if not np.isfinite(flat_actions).all():
            raise RuntimeError("Collected actions contain non-finite values.")
        action_std = float(np.std(flat_actions))
        if action_std < minimum_action_std:
            raise RuntimeError(
                f"Action standard deviation {action_std:.6f} is below required "
                f"{minimum_action_std:.6f}; dataset is too degenerate."
            )
    if returns and not np.isfinite(np.asarray(returns, dtype=np.float64)).all():
        raise RuntimeError("Collected returns contain non-finite values.")
    return {
        "episode_count": len(lengths),
        "episode_length_mean": mean_length,
        "episode_length_min": int(min(lengths)) if lengths else 0,
        "episode_length_max": int(max(lengths)) if lengths else 0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_min": float(np.min(returns)) if returns else 0.0,
        "return_max": float(np.max(returns)) if returns else 0.0,
        "action_std": action_std,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Collect RLlib SingleAgentEpisode datasets for Dreamer pretraining."
    )
    parser.add_argument(
        "--dreamer-config",
        type=Path,
        required=True,
        help="Dreamer YAML config whose wrapped env contract should be recorded.",
    )
    parser.add_argument("--dataset-id", type=str, required=True, help="Dataset identifier.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override (defaults to output/benchmarks/...).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of expert episodes to collect.",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=("ppo", "random", "zero"),
        default="ppo",
        help="Teacher source for actions.",
    )
    parser.add_argument(
        "--teacher-model-id",
        type=str,
        default=None,
        help="Optional model registry id for PPO teacher lookup.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit PPO checkpoint path.",
    )
    parser.add_argument(
        "--stochastic-teacher",
        action="store_true",
        help="Use stochastic PPO teacher actions instead of deterministic inference.",
    )
    parser.add_argument(
        "--minimum-episodes",
        type=int,
        default=50,
        help="Fail if fewer than this many episodes are collected.",
    )
    parser.add_argument(
        "--minimum-mean-length",
        type=float,
        default=20.0,
        help="Fail if the dataset mean episode length falls below this value.",
    )
    parser.add_argument(
        "--minimum-action-std",
        type=float,
        default=0.02,
        help="Fail if action variance is too small across the dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    run_config = load_run_config(args.dreamer_config)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_dir(args.dataset_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher, teacher_meta = _load_teacher(
        teacher_mode=args.teacher_mode,
        teacher_model_id=args.teacher_model_id,
        teacher_checkpoint=args.teacher_checkpoint,
        deterministic=not args.stochastic_teacher,
    )
    env_creator = _make_env_creator(run_config)

    rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    returns: list[float] = []
    action_vectors: list[np.ndarray] = []
    scenario_ids: list[str] = []

    for episode_idx in range(args.episodes):
        env = env_creator({"worker_index": episode_idx})
        try:
            obs, _infos = env.reset()
            episode = SingleAgentEpisode(
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            episode.add_env_reset(obs, {})
            episode_return = 0.0
            max_steps = int(getattr(getattr(env, "state", None), "max_sim_steps", 1000))
            steps = 0
            done = False
            actions_for_episode: list[np.ndarray] = []
            while not done and steps < max_steps:
                action = _predict_action(
                    teacher,
                    obs,
                    env.action_space,
                    teacher_mode=args.teacher_mode,
                    deterministic=not args.stochastic_teacher,
                )
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode.add_env_step(
                    observation=next_obs,
                    action=action,
                    reward=float(reward),
                    infos={},
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                obs = next_obs
                episode_return += float(reward)
                done = bool(terminated or truncated)
                actions_for_episode.append(action.reshape(1, -1))
                scenario_id = info.get("scenario_id") if isinstance(info, dict) else None
                if scenario_id:
                    scenario_ids.append(str(scenario_id))
                steps += 1
            if not episode.is_done:
                episode.is_truncated = True

            serialized = _serialize_episode(episode)
            rows.append(
                {
                    "item": serialized,
                    "episode_index": int(episode_idx),
                    "episode_len": len(episode),
                    "episode_return": float(episode_return),
                }
            )
            lengths.append(len(episode))
            returns.append(float(episode_return))
            if actions_for_episode:
                action_vectors.append(np.concatenate(actions_for_episode, axis=0))
            logger.info(
                "Collected episode {}/{} len={} return={:.3f}",
                episode_idx + 1,
                args.episodes,
                len(episode),
                episode_return,
            )
        finally:
            env.close()

    gate_summary = _quality_gate(
        lengths=lengths,
        returns=returns,
        action_vectors=action_vectors,
        minimum_episodes=int(args.minimum_episodes),
        minimum_mean_length=float(args.minimum_mean_length),
        minimum_action_std=float(args.minimum_action_std),
    )

    table = pa.table(
        {
            "item": [row["item"] for row in rows],
            "episode_index": [row["episode_index"] for row in rows],
            "episode_len": [row["episode_len"] for row in rows],
            "episode_return": [row["episode_return"] for row in rows],
        }
    )
    pq.write_table(table, output_dir / "episodes.parquet")

    manifest = {
        "dataset_id": args.dataset_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dreamer_config_path": str(args.dreamer_config.resolve()),
        "output_dir": str(output_dir),
        "teacher": teacher_meta,
        "quality_gate": gate_summary,
        "unique_scenarios": sorted(set(scenario_ids)),
        "episodes_requested": int(args.episodes),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.success(
        "Wrote Dreamer world-model dataset to {} (manifest={})", output_dir, manifest_path
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
