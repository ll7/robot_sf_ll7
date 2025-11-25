"""Minimal fixture generator for research report tests (T013).

Creates lightweight tracker-like manifest JSON files for baseline and
pretrained policies. Each manifest contains per-seed episode metrics so
aggregation and orchestration logic can run without the full training
pipeline.

Exported helpers:
    create_minimal_manifest() – single policy/seed manifest
    create_seed_set()         – multi-seed manifest set

Manifest schema (subset):
    {
        "run_id": str,
        "seed": int,
        "policy_type": "baseline" | "pretrained",
        "episodes": [ {"episode_id": int, "timesteps": int, "success": bool, "collision": bool } ],
        "metrics": {"success_rate": float, "collision_rate": float, "avg_timesteps": float}
    }
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GeneratedManifest:
    path: Path
    run_id: str
    seed: int
    policy_type: str


def _now_run_id(prefix: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}"


def create_minimal_manifest(
    run_id: str | None,
    seed: int,
    policy_type: str,
    output_dir: str | Path,
    episode_count: int = 3,
) -> Path:
    if not isinstance(episode_count, int) or episode_count <= 0:
        raise ValueError("episode_count must be a positive integer")
    if policy_type not in {"baseline", "pretrained"}:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    run_id = run_id or _now_run_id(f"{policy_type}_seed{seed}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{run_id}.manifest.json"

    rng = random.Random(seed)
    episodes: list[dict] = []
    successes = 0
    collisions = 0
    total_timesteps = 0
    for ep_id in range(episode_count):
        timesteps = rng.randint(80, 120)
        success = rng.random() > 0.2
        collision = (not success) and rng.random() < 0.3
        episodes.append(
            {
                "episode_id": ep_id,
                "timesteps": timesteps,
                "success": success,
                "collision": collision,
            }
        )
        successes += int(success)
        collisions += int(collision)
        total_timesteps += timesteps

    metrics = {
        "success_rate": successes / episode_count,
        "collision_rate": collisions / episode_count,
        "avg_timesteps": total_timesteps / episode_count,
    }

    payload = {
        "run_id": run_id,
        "seed": seed,
        "policy_type": policy_type,
        "episodes": episodes,
        "metrics": metrics,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return manifest_path


def create_seed_set(
    policy_type: str,
    seeds: list[int],
    output_dir: str | Path,
    episode_count: int = 3,
) -> list[Path]:
    paths: list[Path] = []
    for seed in seeds:
        paths.append(
            create_minimal_manifest(
                run_id=None,
                seed=seed,
                policy_type=policy_type,
                output_dir=output_dir,
                episode_count=episode_count,
            )
        )
    return paths


__all__ = ["GeneratedManifest", "create_minimal_manifest", "create_seed_set"]
