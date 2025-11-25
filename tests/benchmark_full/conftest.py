"""Shared pytest fixtures for Full Classic Interaction Benchmark tests.

Implements scaffolding for task T003.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class BenchmarkConfig:  # lightweight test double; replaced by real implementation later
    output_root: str
    scenario_matrix_path: str
    fast_stub: bool = True
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    master_seed: int = 123
    initial_episodes: int = 1  # keep smoke-fast for real execution
    batch_size: int = 1
    max_episodes: int = 2
    workers: int = 1
    smoke: bool = True
    capture_replay: bool = True
    disable_videos: bool = False
    max_videos: int = 1
    force_continue: bool = False
    snqi_weights_path: str | None = None
    algo: str = "ppo"
    horizon_override: int | None = None
    effect_size_reference_density: str = "low"
    video_renderer: str = "auto"
    video_fps: int = 10
    smoke_horizon_cap: int = 40


@pytest.fixture()
def temp_results_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="classic_bench_") as d:
        yield Path(d)


@pytest.fixture()
def config_factory(temp_results_dir: Path):
    """Return factory producing BenchmarkConfig test doubles."""

    def _factory(**overrides):
        base = BenchmarkConfig(
            output_root=str(temp_results_dir),
            scenario_matrix_path="configs/scenarios/classic_interactions.yaml",
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    return _factory


@pytest.fixture()
def synthetic_episode_record():
    def _make(**overrides):
        rec = {
            "episode_id": overrides.get("episode_id", "ep-1"),
            "scenario_id": overrides.get("scenario_id", "scenario_a"),
            "seed": overrides.get("seed", 1),
            "archetype": overrides.get("archetype", "crossing"),
            "density": overrides.get("density", "low"),
            "status": overrides.get("status", "success"),
            "metrics": overrides.get(
                "metrics",
                {
                    "collision_rate": 0.0,
                    "success_rate": 1.0,
                    "time_to_goal": 12.3,
                    "path_efficiency": 0.93,
                    "average_speed": 1.1,
                    "snqi": 0.75,
                },
            ),
            "steps": overrides.get("steps", 120),
            "wall_time_sec": overrides.get("wall_time_sec", 0.5),
            "algo": overrides.get("algo", "ppo"),
            "created_at": overrides.get("created_at", 0.0),
        }
        return rec

    return _make
