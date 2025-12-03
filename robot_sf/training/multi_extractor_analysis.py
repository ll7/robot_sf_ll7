"""Analysis helpers for multi-extractor training runs.

Loads evaluation histories from Stable-Baselines3 ``evaluations.npz`` files,
computes convergence/sample-efficiency metrics, and generates lightweight
figures for summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

try:  # Set non-interactive backend for headless environments when plotting
    import matplotlib

    matplotlib.use("Agg")
except (ImportError, RuntimeError):  # pragma: no cover - plotting already guarded
    pass

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass
class EvalHistory:
    """EvalHistory class."""

    timesteps: list[float]
    mean_rewards: list[float]


def load_eval_history(extractor_dir: Path) -> EvalHistory | None:
    """Load evaluation timesteps and mean rewards from evaluations.npz if present."""

    path = extractor_dir / "eval_logs" / "evaluations.npz"
    if not path.exists():
        return None

    try:
        data = np.load(path)
        timesteps_raw = np.asarray(data.get("timesteps", []))
        results_raw = np.asarray(data.get("results", []))
        timesteps = [float(t) for t in timesteps_raw.reshape(-1).tolist()]
        if results_raw.size == 0:
            return None
        rewards_flat = results_raw.reshape(results_raw.shape[0], -1)
        mean_rewards = [float(r) for r in rewards_flat.mean(axis=1).tolist()]
        return EvalHistory(timesteps=timesteps, mean_rewards=mean_rewards)
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to load eval history from %s: %s", path, exc)
        return None


def convergence_timestep(
    history: EvalHistory | None,
    *,
    target_reward: float,
    default_timesteps: float,
) -> float:
    """Return earliest timestep meeting target_reward, else default_timesteps."""

    if history is None or not history.timesteps or target_reward <= 0:
        return float(default_timesteps)
    for t, r in zip(history.timesteps, history.mean_rewards, strict=False):
        if r >= target_reward:
            return float(t)
    return float(default_timesteps)


def sample_efficiency_ratio(
    *,
    baseline_timestep: float,
    candidate_timestep: float,
) -> float:
    """Compute baseline/candidate timestep ratio (higher = more sample efficient)."""

    if candidate_timestep <= 0:
        return 0.0
    return float(baseline_timestep) / float(candidate_timestep)


def generate_figures(
    history: EvalHistory | None,
    out_dir: Path,
    extractor_name: str,
) -> dict[str, Path]:
    """Generate learning curve and reward distribution figures for an extractor."""

    if history is None or not history.timesteps:
        return {}
    try:
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        learning_curve = out_dir / f"{extractor_name}_learning_curve.png"
        reward_hist = out_dir / f"{extractor_name}_reward_distribution.png"

        plt.figure(figsize=(6, 4))
        plt.plot(history.timesteps, history.mean_rewards, marker="o")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean eval reward")
        plt.title(f"Learning curve: {extractor_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(learning_curve, dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        bin_count = min(20, max(5, int(len(history.mean_rewards) / 2)))
        plt.hist(history.mean_rewards, bins=bin_count, alpha=0.8)
        plt.xlabel("Mean eval reward")
        plt.ylabel("Frequency")
        plt.title(f"Reward distribution: {extractor_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reward_hist, dpi=150)
        plt.close()

        return {"learning_curve": learning_curve, "reward_distribution": reward_hist}
    except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
        logger.warning("Failed to generate figures for %s: %s", extractor_name, exc)
        return {}


def summarize_metric(values: Iterable[float]) -> dict[str, float]:
    """Summarize metric.

    Args:
        values: Collection of numeric values.

    Returns:
        dict[str, float]: mapping of str, float.
    """
    items = [v for v in values if math.isfinite(v)]
    if not items:
        return {"mean": 0.0, "median": 0.0}
    items_sorted = sorted(items)
    mid = len(items_sorted) // 2
    if len(items_sorted) % 2 == 0:
        median = (items_sorted[mid - 1] + items_sorted[mid]) / 2.0
    else:
        median = items_sorted[mid]
    mean_val = sum(items_sorted) / len(items_sorted)
    return {"mean": float(mean_val), "median": float(median)}
