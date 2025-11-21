"""Test fixture generator for minimal tracker manifests.

This module generates minimal tracker manifests for testing the research
reporting pipeline without requiring full imitation learning runs.

Generated Fixtures:
    - Baseline policy manifest (3 seeds)
    - Pretrained policy manifest (3 seeds)
    - Ablation variant manifests

Usage:
    >>> from tests.fixtures.minimal_manifests.generator import generate_minimal_manifests
    >>> manifests = generate_minimal_manifests(output_dir=Path("output/test"))
    >>> manifests["baseline"]  # Path to baseline manifest
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def generate_baseline_manifest(
    seeds: list[int],
    output_dir: Path,
) -> Path:
    """Generate a minimal baseline policy manifest.

    Args:
        seeds: Random seeds to include
        output_dir: Directory to write manifest

    Returns:
        Path to generated manifest file
    """
    manifest_data = {
        "run_id": "test_baseline_001",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "experiment_name": "Test Baseline",
        "status": "completed",
        "steps": [
            {
                "name": "train_baseline",
                "status": "completed",
                "duration_seconds": 120.0,
            }
        ],
        "seeds": seeds,
        "metrics": {},
    }

    # Generate per-seed metrics
    for seed in seeds:
        np.random.seed(seed)
        manifest_data["metrics"][str(seed)] = {
            "success_rate": float(np.random.uniform(0.60, 0.75)),
            "collision_rate": float(np.random.uniform(0.15, 0.25)),
            "timesteps_to_convergence": int(np.random.randint(800000, 1200000)),
            "final_reward_mean": float(np.random.uniform(0.50, 0.65)),
            "run_duration_seconds": float(np.random.uniform(100, 150)),
        }

    output_path = output_dir / "baseline_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(manifest_data, f, indent=2)

    return output_path


def generate_pretrained_manifest(
    seeds: list[int],
    output_dir: Path,
    improvement_factor: float = 0.45,
) -> Path:
    """Generate a minimal pretrained policy manifest.

    Args:
        seeds: Random seeds to include
        output_dir: Directory to write manifest
        improvement_factor: Improvement over baseline (0.45 = 45% reduction)

    Returns:
        Path to generated manifest file
    """
    manifest_data = {
        "run_id": "test_pretrained_001",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "experiment_name": "Test Pretrained",
        "status": "completed",
        "steps": [
            {
                "name": "collect_trajectories",
                "status": "completed",
                "duration_seconds": 60.0,
            },
            {
                "name": "bc_pretrain",
                "status": "completed",
                "duration_seconds": 30.0,
            },
            {
                "name": "ppo_finetune",
                "status": "completed",
                "duration_seconds": 80.0,
            },
        ],
        "seeds": seeds,
        "metrics": {},
    }

    # Generate per-seed metrics with improvement
    for seed in seeds:
        np.random.seed(seed + 1000)  # Offset seed for different values
        baseline_timesteps = int(np.random.randint(800000, 1200000))
        improved_timesteps = int(baseline_timesteps * (1 - improvement_factor))

        manifest_data["metrics"][str(seed)] = {
            "success_rate": float(np.random.uniform(0.75, 0.85)),
            "collision_rate": float(np.random.uniform(0.08, 0.15)),
            "timesteps_to_convergence": improved_timesteps,
            "final_reward_mean": float(np.random.uniform(0.70, 0.80)),
            "run_duration_seconds": float(np.random.uniform(80, 120)),
        }

    output_path = output_dir / "pretrained_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(manifest_data, f, indent=2)

    return output_path


def generate_ablation_manifests(
    seeds: list[int],
    output_dir: Path,
    bc_epochs: list[int] | None = None,
    dataset_sizes: list[int] | None = None,
) -> dict[str, Path]:
    """Generate ablation study manifests.

    Args:
        seeds: Random seeds to include
        output_dir: Directory to write manifests
        bc_epochs: BC epoch values to test (defaults to [5, 10, 20])
        dataset_sizes: Dataset size values to test (defaults to [100, 200, 300])

    Returns:
        Dictionary mapping variant_id to manifest path
    """
    if bc_epochs is None:
        bc_epochs = [5, 10, 20]
    if dataset_sizes is None:
        dataset_sizes = [100, 200, 300]

    manifests = {}

    for bc_epoch in bc_epochs:
        for dataset_size in dataset_sizes:
            variant_id = f"bc{bc_epoch}_ds{dataset_size}"

            # Improvement scales with BC epochs and dataset size
            base_improvement = 0.30
            epoch_bonus = (bc_epoch - 5) * 0.03  # 3% per additional 5 epochs
            dataset_bonus = (dataset_size - 100) * 0.0005  # 0.05% per additional 100 episodes
            total_improvement = min(base_improvement + epoch_bonus + dataset_bonus, 0.55)

            manifest_data = {
                "run_id": f"test_ablation_{variant_id}",
                "created_at": datetime.now(tz=UTC).isoformat(),
                "experiment_name": f"Test Ablation {variant_id}",
                "variant_id": variant_id,
                "status": "completed",
                "config": {
                    "bc_epochs": bc_epoch,
                    "dataset_size": dataset_size,
                },
                "seeds": seeds,
                "metrics": {},
            }

            for seed in seeds:
                np.random.seed(seed + hash(variant_id) % 10000)
                baseline_timesteps = int(np.random.randint(800000, 1200000))
                improved_timesteps = int(baseline_timesteps * (1 - total_improvement))

                manifest_data["metrics"][str(seed)] = {
                    "success_rate": float(np.random.uniform(0.70, 0.85)),
                    "collision_rate": float(np.random.uniform(0.10, 0.18)),
                    "timesteps_to_convergence": improved_timesteps,
                    "final_reward_mean": float(np.random.uniform(0.65, 0.78)),
                    "run_duration_seconds": float(np.random.uniform(90, 130)),
                }

            output_path = output_dir / f"ablation_{variant_id}_manifest.json"
            with output_path.open("w") as f:
                json.dump(manifest_data, f, indent=2)

            manifests[variant_id] = output_path

    return manifests


def generate_minimal_manifests(
    output_dir: Path,
    seeds: list[int] | None = None,
    include_ablation: bool = False,
) -> dict[str, Path]:
    """Generate complete set of minimal test manifests.

    Args:
        output_dir: Directory to write manifests
        seeds: Random seeds (defaults to [42, 43, 44])
        include_ablation: Whether to generate ablation manifests

    Returns:
        Dictionary mapping manifest type to path
    """
    if seeds is None:
        seeds = [42, 43, 44]

    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "baseline": generate_baseline_manifest(seeds, output_dir),
        "pretrained": generate_pretrained_manifest(seeds, output_dir),
    }

    if include_ablation:
        ablation_manifests = generate_ablation_manifests(seeds, output_dir)
        manifests.update(ablation_manifests)

    return manifests
