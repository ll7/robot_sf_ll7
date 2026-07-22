"""Action bin accounting for continuous acceleration and angular velocity surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_action_bin_accounting(
    actions: np.ndarray | list[np.ndarray],
    *,
    num_acc_bins: int = 10,
    num_ang_vel_bins: int = 10,
    acc_range: tuple[float, float] = (-3.0, 3.0),
    ang_vel_range: tuple[float, float] = (-3.14159, 3.14159),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute 2D action bin counts, inverse-frequency weights, and balance metrics.

    Args:
        actions: 2D array or list of 2D action vectors (acc, ang_vel) of shape (N, 2).
        num_acc_bins: Number of bins along acceleration axis.
        num_ang_vel_bins: Number of bins along angular velocity axis.
        acc_range: (min, max) bounds for acceleration.
        ang_vel_range: (min, max) bounds for angular velocity.

    Returns:
        Tuple of (weights_array, balance_summary_dict).
        weights_array is a 1D float32 array of shape (N,) with mean=1.0.
        balance_summary_dict contains bin counts, active bins, and balance metrics.
    """
    if isinstance(actions, list):
        if not actions:
            raw_actions = np.zeros((0, 2), dtype=np.float32)
        else:
            raw_actions = np.vstack([np.asarray(a, dtype=np.float32) for a in actions])
    else:
        raw_actions = np.asarray(actions, dtype=np.float32)

    total_transitions = len(raw_actions)
    if total_transitions == 0 or raw_actions.ndim != 2 or raw_actions.shape[1] < 2:
        empty_weights = np.zeros(0, dtype=np.float32)
        summary = {
            "total_transitions": 0,
            "num_bins": num_acc_bins * num_ang_vel_bins,
            "active_bins": 0,
            "max_bin_count": 0,
            "min_active_bin_count": 0,
            "imbalance_ratio": 1.0,
            "acc_bins": num_acc_bins,
            "ang_vel_bins": num_ang_vel_bins,
            "acc_range": list(acc_range),
            "ang_vel_range": list(ang_vel_range),
            "bin_counts": {},
        }
        return empty_weights, summary

    acc_min, acc_max = acc_range
    ang_min, ang_max = ang_vel_range

    acc_vals = raw_actions[:, 0]
    ang_vals = raw_actions[:, 1]

    # Clip values to bounds before binning
    acc_norm = np.clip((acc_vals - acc_min) / (acc_max - acc_min + 1e-8), 0.0, 0.999999)
    ang_norm = np.clip((ang_vals - ang_min) / (ang_max - ang_min + 1e-8), 0.0, 0.999999)

    acc_indices = (acc_norm * num_acc_bins).astype(int)
    ang_indices = (ang_norm * num_ang_vel_bins).astype(int)

    bin_flat_indices = acc_indices * num_ang_vel_bins + ang_indices
    total_bins = num_acc_bins * num_ang_vel_bins

    bin_counts = np.bincount(bin_flat_indices, minlength=total_bins)

    # Inverse frequency weighting
    non_zero = bin_counts > 0
    raw_weights_per_bin = np.zeros(total_bins, dtype=np.float64)
    raw_weights_per_bin[non_zero] = 1.0 / bin_counts[non_zero]

    raw_transition_weights = raw_weights_per_bin[bin_flat_indices]

    # Normalize weights so that mean weight == 1.0
    weight_sum = np.sum(raw_transition_weights)
    if weight_sum > 0:
        norm_weights = (raw_transition_weights / weight_sum) * total_transitions
    else:
        norm_weights = np.ones(total_transitions, dtype=np.float64)

    weights_array = norm_weights.astype(np.float32)

    active_bin_counts = bin_counts[non_zero]
    max_count = int(np.max(bin_counts)) if len(bin_counts) > 0 else 0
    min_active_count = int(np.min(active_bin_counts)) if len(active_bin_counts) > 0 else 0
    imbalance_ratio = float(max_count / max(1, min_active_count))

    bin_counts_dict: dict[str, int] = {}
    for i in range(num_acc_bins):
        for j in range(num_ang_vel_bins):
            idx = i * num_ang_vel_bins + j
            cnt = int(bin_counts[idx])
            if cnt > 0:
                bin_counts_dict[f"acc_{i}_ang_{j}"] = cnt

    summary = {
        "total_transitions": total_transitions,
        "num_bins": total_bins,
        "active_bins": int(np.sum(non_zero)),
        "max_bin_count": max_count,
        "min_active_bin_count": min_active_count,
        "imbalance_ratio": imbalance_ratio,
        "acc_bins": num_acc_bins,
        "ang_vel_bins": num_ang_vel_bins,
        "acc_range": list(acc_range),
        "ang_vel_range": list(ang_vel_range),
        "bin_counts": bin_counts_dict,
    }

    return weights_array, summary


__all__ = ["compute_action_bin_accounting"]
