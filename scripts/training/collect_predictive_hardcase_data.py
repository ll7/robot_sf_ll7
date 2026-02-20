#!/usr/bin/env python3
"""Collect hard-case trajectory supervision data for predictive planner training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import _build_env_config
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios


@dataclass
class Frame:
    """Single rollout frame for training sample extraction."""

    robot_pos: np.ndarray
    robot_heading: float
    ped_positions_world: np.ndarray
    ped_velocities_world: np.ndarray
    ped_count: int


def _goal_policy(obs: dict, max_speed: float) -> np.ndarray:
    """Simple goal-seeking policy used during data collection rollouts."""
    robot = obs.get("robot", {})
    goal = obs.get("goal", {})
    pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=np.float32)[:2]
    heading = float(np.asarray(robot.get("heading", [0.0]), dtype=np.float32).reshape(-1)[0])
    tgt = np.asarray(goal.get("current", [0.0, 0.0]), dtype=np.float32)[:2]

    vec = tgt - pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    desired_heading = float(np.arctan2(vec[1], vec[0]))
    heading_error = float((desired_heading - heading + np.pi) % (2 * np.pi) - np.pi)
    omega = np.clip(heading_error, -1.0, 1.0)
    v = np.clip(dist, 0.0, max_speed * max(0.0, 1.0 - abs(heading_error) / np.pi))
    return np.array([v, omega], dtype=np.float32)


def _extract_frame(obs: dict, max_agents: int) -> Frame:
    """Convert observation payload to a compact frame container."""
    robot = obs.get("robot", {})
    peds = obs.get("pedestrians", {})
    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=np.float32)[:2]
    robot_heading = float(np.asarray(robot.get("heading", [0.0]), dtype=np.float32).reshape(-1)[0])

    ped_positions = np.asarray(peds.get("positions", []), dtype=np.float32)
    ped_velocities = np.asarray(peds.get("velocities", []), dtype=np.float32)
    ped_count = int(np.asarray(peds.get("count", [0]), dtype=np.float32).reshape(-1)[0])

    if ped_positions.ndim == 1:
        ped_positions = (
            ped_positions.reshape(-1, 2)
            if ped_positions.size % 2 == 0
            else np.zeros((0, 2), dtype=np.float32)
        )
    if ped_velocities.ndim == 1:
        ped_velocities = (
            ped_velocities.reshape(-1, 2)
            if ped_velocities.size % 2 == 0
            else np.zeros((0, 2), dtype=np.float32)
        )

    ped_count = max(0, min(ped_count, ped_positions.shape[0], max_agents))
    ped_positions = ped_positions[:ped_count]
    if ped_velocities.shape[0] < ped_count:
        ped_velocities = np.pad(ped_velocities, ((0, ped_count - ped_velocities.shape[0]), (0, 0)))
    ped_velocities = ped_velocities[:ped_count]

    return Frame(
        robot_pos=robot_pos,
        robot_heading=robot_heading,
        ped_positions_world=ped_positions,
        ped_velocities_world=ped_velocities,
        ped_count=ped_count,
    )


def _world_to_ego(
    points_world: np.ndarray, robot_pos: np.ndarray, robot_heading: float
) -> np.ndarray:
    """Transform world-frame points into the robot frame."""
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rel = points_world - robot_pos.reshape(1, 2)
    cos_h = float(np.cos(robot_heading))
    sin_h = float(np.sin(robot_heading))
    x_ego = cos_h * rel[:, 0] + sin_h * rel[:, 1]
    y_ego = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
    return np.stack([x_ego, y_ego], axis=1).astype(np.float32)


def _frames_to_samples(
    frames: list[Frame],
    *,
    max_agents: int,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create supervised samples from temporal frame sequences."""
    if len(frames) <= horizon_steps:
        return (
            np.zeros((0, max_agents, 4), dtype=np.float32),
            np.zeros((0, max_agents, horizon_steps, 2), dtype=np.float32),
            np.zeros((0, max_agents), dtype=np.float32),
        )

    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for t in range(0, len(frames) - horizon_steps):
        frame_t = frames[t]
        c = min(frame_t.ped_count, max_agents)
        state = np.zeros((max_agents, 4), dtype=np.float32)
        target = np.zeros((max_agents, horizon_steps, 2), dtype=np.float32)
        mask = np.zeros((max_agents,), dtype=np.float32)

        if c > 0:
            pos_rel = _world_to_ego(
                frame_t.ped_positions_world[:c],
                frame_t.robot_pos,
                frame_t.robot_heading,
            )
            state[:c, 0:2] = pos_rel
            state[:c, 2:4] = _vel_world_to_ego(
                frame_t.ped_velocities_world[:c],
                frame_t.robot_heading,
            )
            mask[:c] = 1.0

            for k in range(1, horizon_steps + 1):
                frame_k = frames[t + k]
                ck = min(frame_k.ped_count, c)
                if ck <= 0:
                    continue
                matches = _nearest_match_indices(
                    frame_t.ped_positions_world[:c],
                    frame_k.ped_positions_world[:ck],
                )
                if not matches:
                    continue
                for src_idx, tgt_idx in matches.items():
                    tgt_rel = _world_to_ego(
                        frame_k.ped_positions_world[tgt_idx : tgt_idx + 1],
                        frame_t.robot_pos,
                        frame_t.robot_heading,
                    )[0]
                    target[src_idx, k - 1, :] = tgt_rel

        states.append(state)
        targets.append(target)
        masks.append(mask)

    return np.stack(states), np.stack(targets), np.stack(masks)


def _load_seed_manifest(path: Path) -> dict[str, list[int]]:
    """Load scenario->seed map from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Seed manifest must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key)] = [int(v) for v in value]
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI args for hard-case data collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-matrix", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--max-agents", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=8)
    parser.add_argument("--max-speed", type=float, default=1.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/tmp/predictive_planner/datasets/predictive_rollouts_hardcase_v1.npz"),
    )
    return parser.parse_args()


def main() -> int:
    """Collect hard-case trajectories and persist ``.npz`` + metadata sidecar."""
    args = parse_args()
    scenarios = load_scenarios(args.scenario_matrix)
    seed_manifest = _load_seed_manifest(args.seed_manifest)

    selected: list[tuple[dict, int]] = []
    for scenario in scenarios:
        scenario_id = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        for seed in seed_manifest.get(scenario_id, []):
            selected.append((scenario, int(seed)))

    if not selected:
        raise RuntimeError(
            "No hard-case scenarios selected. Check scenario matrix and seed manifest."
        )

    all_states: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    for scenario, seed in selected:
        config = _build_env_config(scenario, scenario_path=args.scenario_matrix)
        env = make_robot_env(config=config, seed=int(seed), debug=False)
        try:
            obs, _ = env.reset(seed=int(seed))
            episode_frames: list[Frame] = []
            for _step in range(int(args.max_steps)):
                episode_frames.append(_extract_frame(obs, int(args.max_agents)))
                action = _goal_policy(obs, max_speed=float(args.max_speed))
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break

            states, targets, masks = _frames_to_samples(
                episode_frames,
                max_agents=int(args.max_agents),
                horizon_steps=int(args.horizon_steps),
            )
            if states.shape[0] > 0:
                all_states.append(states)
                all_targets.append(targets)
                all_masks.append(masks)
        finally:
            env.close()

    if not all_states:
        raise RuntimeError("No hard-case samples were collected.")

    states_cat = np.concatenate(all_states, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)
    masks_cat = np.concatenate(all_masks, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        state=states_cat,
        target=targets_cat,
        mask=masks_cat,
    )

    summary = {
        "scenario_matrix": str(args.scenario_matrix),
        "seed_manifest": str(args.seed_manifest),
        "num_scenarios": len(selected),
        "num_samples": int(states_cat.shape[0]),
        "max_agents": int(states_cat.shape[1]),
        "horizon_steps": int(targets_cat.shape[2]),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    def _vel_world_to_ego(vectors_world: np.ndarray, robot_heading: float) -> np.ndarray:
        """Rotate world-frame vectors to ego frame (no translation)."""
        if vectors_world.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        vx_ego = cos_h * vectors_world[:, 0] + sin_h * vectors_world[:, 1]
        vy_ego = -sin_h * vectors_world[:, 0] + cos_h * vectors_world[:, 1]
        return np.stack([vx_ego, vy_ego], axis=1).astype(np.float32)

    def _nearest_match_indices(
        source_positions: np.ndarray,
        target_positions: np.ndarray,
        *,
        max_match_distance: float = 1.5,
    ) -> dict[int, int]:
        """Match source pedestrians to target pedestrians by nearest neighbor."""
        if source_positions.size == 0 or target_positions.size == 0:
            return {}
        matches: dict[int, int] = {}
        taken_targets: set[int] = set()
        for src_idx in range(source_positions.shape[0]):
            distances = np.linalg.norm(target_positions - source_positions[src_idx], axis=1)
            sorted_target = np.argsort(distances)
            for tgt_idx in sorted_target:
                tgt_i = int(tgt_idx)
                if tgt_i in taken_targets:
                    continue
                if float(distances[tgt_i]) > max_match_distance:
                    break
                matches[src_idx] = tgt_i
                taken_targets.add(tgt_i)
                break
        return matches
